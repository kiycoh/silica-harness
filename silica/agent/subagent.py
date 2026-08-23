# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""BoundedSubAgent — a small, tightly-bounded worker that runs on the worker model.

A bounded sub-agent consumes one WorkItem at a time and dispatches it to the
capability registered under ``item.kind``. Each capability (see
``silica/capabilities/``) is a self-contained ``run(item, config) -> dict``
function that writes only through its CapabilityBounds + the commit_ops micro-gate.
The sub-agent runs on the *worker* model (role="worker"), concurrently with the
Injector.

``BoundedSubAgent`` itself is just the dispatch seam: it owns the worker config,
catches capability errors so a single item never crashes the pool, and returns a
status dict. Adding a behaviour means adding a capability module + one registry
line — never editing this file.
"""
from __future__ import annotations

import logging
from typing import Any

from silica.config import CONFIG
from silica.capabilities import CAPABILITIES, Capability
from silica.kernel.workqueue import WorkItem

logger = logging.getLogger(__name__)


def _item_provenance(item: Any) -> tuple[str, str, str] | None:
    """(source basename, content sha, ledger run id) of the work item.

    dedup carries the inbox path, expand the op's source_basename. The run id
    is the FSM's own `progress.run_id` — the key CLEANUP stamps and the key the
    dangling-link sweep and check_renucleate read back. It is NOT the undo
    journal's run id, which is a separate uuid4 keyspace.
    """
    import os

    ctx = getattr(item, "context", None) or {}
    sha = ctx.get("content_hash") or ""
    run_id = ctx.get("run_id") or ""
    src = ctx.get("inbox_file") or (ctx.get("op") or {}).get("source_basename") or ""
    if not src:
        # A dedup FAMILY batch: `inbox_file` is in _BATCH_CONCEPT_KEYS, so
        # batch_dedup_items moved it into the per-concept entries and the shared
        # context has no source left. One batch is always one file — the sha it
        # carries is fsm._current_content_hash — so the first entry answers for all.
        src = next((c.get("inbox_file") for c in (ctx.get("concepts") or [])
                    if isinstance(c, dict) and c.get("inbox_file")), "")
    return (os.path.basename(src), sha, run_id) if src and sha else None


def consume(wq: Any, agent: "BoundedSubAgent", stop: Any = None,
            undo_run: Any = None, parent_span: str | None = None,
            run_id: str | None = None) -> None:
    """One consumer thread: claim → handle → complete until the queue closes.

    THE consumer loop — shared by the Coordinator's in-run pool and the ad-hoc
    ``run_subagent_batch`` path, so cancel semantics, bookkeeping, and bus
    events live in exactly one place.  Blocks at OS level on ``wq.claim()`` —
    no polling; the sentinel injected by ``wq.close()`` cascades through all
    consumers so they all wake and exit cleanly.  ``stop`` (optional Event) is
    checked before each item so a producer crash or user cancel causes pending
    items to be marked cancelled rather than dispatched.
    """
    from silica.agent.bus import BUS
    from silica.agent.commit import _current_provenance, _current_undo_run
    from silica.agent.events import WorkCancelledEvent
    from silica.agent import narration as _narr_mod

    # Thread contexts are born empty: rebind the run/parent anchors the
    # submitting thread held, so children attribute to the run span.
    if run_id is not None:
        _narr_mod.set_run(run_id)
    if parent_span is not None:
        _narr_mod.set_parent(parent_span)

    while True:
        item = wq.claim()               # blocks; no timeout, no polling
        if item is None:
            return                      # sentinel received — queue fully drained
        if (stop is not None and stop.is_set()) or item.cancel_token.is_set():
            wq.complete(item, "cancelled", {"status": "cancelled", "reason": "cancel_token"})
            BUS.publish(
                "work/cancelled",
                WorkCancelledEvent(item.id, item.kind, "pre_handle"),
            )
            _narr_mod.NARRATOR.narrate(
                "subagent", "cancelled", f"{item.kind} cancelled at pre_handle",
                {"kind": item.kind, "phase": "pre_handle"}, id=item.id)
            continue
        # The subagent span (spec §4): this worker thread starts with an empty
        # context, so attach makes every inner beat (calls, writes) carry
        # item.id as parent — the attribution that keeps parallel appends
        # unambiguous.
        _narr_mod.NARRATOR.span_open(
            "subagent", item.id, f"{item.kind} {item.target_path}".strip(),
            {"kind": item.kind}, attach=True)
        tok = _current_provenance.set(_item_provenance(item))
        run_tok = (_current_undo_run.set(undo_run())
                   if undo_run is not None else None)
        try:
            res = agent.handle(item)
        except BaseException:
            _narr_mod.NARRATOR.span_close("subagent", item.id, "failed",
                                          f"{item.kind} crashed", {"kind": item.kind})
            raise
        finally:
            _current_provenance.reset(tok)
            if run_tok is not None:
                _current_undo_run.reset(run_tok)
        wq.complete(item, res.get("status", "done"), res)
        status = res.get("status", "done")
        mapped = {"error": "failed", "cancelled": "cancelled"}.get(status, "done")
        # `done` carries the queue's richer verdict (no_merge/skipped/…) in the
        # summary: the axis stays three-valued, the truth stays visible.
        _narr_mod.NARRATOR.narrate(
            "work", mapped, f"{item.kind} {status}",
            {"kind": item.kind, "status": status}, id=f"wk-{item.id}")
        _narr_mod.NARRATOR.span_close(
            "subagent", item.id, mapped,
            f"{item.kind} {status}", {"kind": item.kind, "status": status})


def run_subagent_batch(
    items: list[WorkItem],
    config: Any = CONFIG,
    *,
    max_workers: int | None = None,
    cancel_token: Any = None,
) -> dict[str, Any]:
    """Run a batch of WorkItems through leashed sub-agents in parallel.

    Used by the ad-hoc /dedup, /refine, /enrich commands and silica_delegate
    (out of the inject pipeline).  A pre-closed WorkQueue drained by the shared
    ``consume`` loop — the exact engine the Coordinator runs in-pipeline — so
    both paths get identical cancel/bookkeeping semantics.  BoundedSubAgent is
    stateless beyond its config, so one instance is safely shared across
    threads; commit_ops serialises same-note writes via path_lease.
    """
    from concurrent.futures import ThreadPoolExecutor

    from silica.kernel.workqueue import WorkQueue

    if not items:
        return {"items": 0, "summary": {}, "results": []}

    if cancel_token is not None:
        for it in items:
            it.cancel_token = cancel_token

    mw = max(1, int(max_workers or getattr(config, "subagent_max_concurrent", 3) or 3))
    agent = BoundedSubAgent(config)

    wq = WorkQueue()
    for it in items:
        wq.enqueue(it)
    wq.close()

    # One undo-journal run for the whole batch so /revert can undo the entire
    # /refine, /enrich or /dedup. Pool workers don't inherit contextvars, so
    # each worker sets the same run id at entry (see commit._current_undo_run).
    from silica.agent.commit import _current_undo_run
    from silica.kernel.write.undo_journal import get_undo_journal
    undo_run_id = get_undo_journal().start_run(
        source=f"{items[0].kind}-batch",
        vault=getattr(config, "vault_path", None) or None,
    )

    def _worker():
        _current_undo_run.set(undo_run_id)
        return consume(wq, agent, cancel_token)

    with ThreadPoolExecutor(max_workers=mw, thread_name_prefix="subagent") as ex:
        futures = [ex.submit(_worker) for _ in range(mw)]
    for f in futures:
        exc = f.exception()
        if exc:
            logger.warning("sub-agent consumer crashed: %s", exc)

    return {
        "items": len(items),
        "summary": wq.summary(),
        "results": [
            {"target": it.target_path, **(it.result or {"status": it.status})}
            for it in items
        ],
    }


class BoundedSubAgent:
    """Dispatches a WorkItem to the capability registered under its kind."""

    def __init__(
        self,
        config: Any = CONFIG,
        capabilities: dict[str, Capability] | None = None,
    ):
        self.config = config
        # Injected registry defaults to the global one, so production is
        # unchanged while tests supply a fake registry without mutating state.
        self._capabilities = capabilities if capabilities is not None else CAPABILITIES

    def handle(self, item: WorkItem) -> dict[str, Any]:
        res = self._run_one(item)
        # A capability may propose follow-ups (e.g. dedup's mechanical spoke →
        # refine, ADR-0001): ``followup`` (single) or ``followups`` (one per
        # member of a batch item). Dispatching here keeps capabilities peers
        # (P9) and works on both consume() paths even after the run queue
        # closed. One hop only: a follow-up's own follow-up is never dispatched.
        if not isinstance(res, dict):
            return res
        single = res.get("followup")
        proposed = res.get("followups") or ([single] if isinstance(single, dict) else [])
        dispatched = []
        for followup in proposed:
            if not (isinstance(followup, dict) and followup.get("kind") in self._capabilities):
                continue
            fu_item = WorkItem(
                kind=followup["kind"],
                target_path=followup.get("target_path", item.target_path),
                context=followup.get("context", {}) or {},
                reason=f"followup:{item.kind}",
                cancel_token=item.cancel_token,
            )
            fu_res = self._run_one(fu_item)
            dispatched.append({**followup, "status": fu_res.get("status", "done")})
        if dispatched:
            if "followups" in res:
                res["followups"] = dispatched
            else:
                res["followup"] = dispatched[0]
        return res

    def _run_one(self, item: WorkItem) -> dict[str, Any]:
        run = self._capabilities.get(item.kind)
        if run is None:
            return {"status": "skipped", "reason": f"no capability for kind '{item.kind}'"}
        try:
            return run(item, self.config)
        except Exception as e:  # never let a sub-agent crash the pool
            logger.warning("Capability '%s' error on item %s: %s", item.kind, item.id, e)
            return {"status": "error", "error": str(e)}
