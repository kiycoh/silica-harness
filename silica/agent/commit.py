# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""commit_ops — the reusable validate→snapshot→write→lint micro-gate.

Lets bounded sub-agents commit their proposed ops through the *same*
deterministic safety machinery the
main pipeline uses: structural validation, an inverse-op snapshot, the bulk
write, a lint gate, and automatic rollback on failure.

Two guarantees layered on top for sub-agents:
  * CapabilityBounds filters ops to the permitted envelope *before* anything is written;
  * every touched note is held under `path_lease` for the whole snapshot→write→lint
    window, so a concurrent writer (another sub-agent, or a lease-aware Injector
    write) cannot interleave on the same note.
"""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import ExitStack, suppress
from typing import Any, Callable

import orjson

from silica.kernel.write.ops import Op, OpType
from silica.kernel.write.ops_io import load_ops
from silica.kernel.recall.paths import silica_tmp_dir

logger = logging.getLogger(__name__)

# Ambient undo-journal run for a subagent batch. run_subagent_batch opens one
# run and sets this in every pool worker, so each commit_ops within the batch
# records its inverses under the SAME run — /revert then undoes the whole
# /refine, /enrich or /dedup, not just the last note. Default None: callers
# outside a batch (interactive tools) journal nothing, unchanged.
import contextvars
_current_undo_run: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "silica_undo_run", default=None
)

# Ambient PROVENANCE-ledger run for writes that happen outside CLEANUP but
# still belong to a run — the boundary anneal, which fires inside the FSM's
# `finally`. Distinct from _current_undo_run: that one is the undo journal's
# uuid4, this one is `fsm.progress.run_id`, the key every ledger reader matches
# on. Unset for a standalone tool call, which records under "anneal".
_current_ledger_run: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "silica_ledger_run", default=None
)

# (source_basename, content_sha[, ledger_run_id]) of the work item a worker
# thread is handling. commit_ops has no manifest and no CLEANUP behind it, so
# this is the only way a sub-agent's note reaches the provenance ledger. Unset
# for ad-hoc commits, which have no source to attribute to. The run id is
# optional so a 2-tuple (older callers, tests) still works.
_current_provenance: contextvars.ContextVar[tuple[str, ...] | None] = contextvars.ContextVar(
    "silica_current_provenance", default=None
)


def _journal_inverses(run_id: str, inverses: list[dict]) -> None:
    """Record a committed batch's inverses so /revert can undo subagent writes.

    Best-effort: a failed hash only weakens the /revert "modified since" guard
    for that note, never blocks the commit that already landed.
    """
    import hashlib
    from silica.driver import DRIVER
    from silica.kernel.write.ops import InverseOp
    from silica.kernel.write.undo_journal import get_undo_journal

    journal = get_undo_journal()
    for raw in inverses:
        try:
            inv = InverseOp(**raw)
        except Exception:
            continue
        try:
            post = DRIVER.read_note(inv.path).content
            post_hash = hashlib.sha256((post or "").encode("utf-8")).hexdigest()
        except Exception:
            post_hash = None
        journal.record(run_id, inv, post_hash)


def _write_ops_tmp(ops: list[Op]) -> str:
    path = str(silica_tmp_dir() / f"{uuid.uuid4().hex}.json")
    payload = [op.model_dump() for op in ops]
    with open(path, "wb") as f:
        f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return path


def commit_derived(rel: str, content: str) -> dict[str, Any]:
    """Commit ONE machine-derived note: lease → snapshot → write → lint → rollback.

    Derived artifacts (code wiki) have their ground truth outside the vault
    (the deterministic digest), so the nucleate concept validators do not apply:
    the hub fallback would inject a junk hub note for the target dir, and the
    anti-info-loss bounds would reject a legitimately shrunk regen. The
    transactional guarantees stay: prior content is captured before the write
    and restored when the OFM lint fails. Creates or overwrites as needed —
    the single write path for first writes and regens alike.

    Returns {"status": committed | rolled_back | error, "reason": ...}.
    """
    from silica.driver import DRIVER
    from silica.kernel.write.templates import ensure_system_floor
    from silica.kernel.workqueue import path_lease
    from silica.tools.composed import silica_lint

    with path_lease(rel):
        try:
            prior: str | None = DRIVER.read_note(rel).content
        except Exception:
            prior = None
        content = ensure_system_floor(content, prior=prior)
        try:
            if prior is None:
                DRIVER.create(rel, content)
            else:
                DRIVER.overwrite(rel, content)
        except Exception as e:
            return {"status": "error", "reason": str(e)}
        lr = silica_lint(rel, op_type="overwrite")
        if not lr.get("success", True):
            reason = "; ".join(str(e) for e in lr.get("errors") or []) or "lint failed"
            try:
                if prior is None:
                    DRIVER.delete(rel)
                else:
                    DRIVER.overwrite(rel, prior)
            except Exception as e:
                logger.error("commit_derived rollback failed for %s: %s", rel, e)
                return {"status": "error", "reason": f"{reason}; rollback failed: {e}"}
            return {"status": "rolled_back", "reason": reason}
    return {"status": "committed"}


def _append_provenance(run_id: str | None, touched: list[tuple[str, str, str]]) -> None:
    """Record this worker's notes against the source it was dispatched for.

    Only the worker lane needs this: the FSM's own writes reach the ledger
    through CLEANUP's manifest. Best-effort — the notes are already on disk.
    """
    prov = _current_provenance.get()
    if not prov:
        return
    source, sha, *tail = prov
    if not (source and sha):
        return
    # No `.md`: the ledger stores note refs bare, and CLEANUP writes them that way.
    # `overwrite` is in the set because it is the ONLY op four sub-agent profiles
    # in agent/bounds.py may emit (and what all three dedup merges emit), so
    # leaving it out silently unrecorded the majority of this lane's writes —
    # byte for byte the defect this function exists to close.
    try:
        from silica.kernel.write.provenance import append_record, is_deriving_op

        notes = sorted({p.removesuffix(".md") for p, op_type, _ in touched
                        if is_deriving_op(op_type)})
        if not notes:
            return
        # The ledger's key is the FSM's own run id, which the item carries: every
        # reader (Coordinator._sweep_dangling_links, CLEANUP's own writer,
        # check_renucleate) matches on `fsm.progress.run_id`. `run_id` here is
        # the UNDO-journal id — a uuid4 from journal.start_run, a different
        # keyspace — so records written under it matched nothing and the sweep
        # still missed exactly the notes this attribution exists to reach.
        ledger_run = (tail[0] if tail else "") or run_id or "worker"
        append_record(source, sha, ledger_run, notes)
    except Exception as exc:
        logger.debug("commit_ops: provenance append failed (non-fatal): %s", exc)


def commit_ops(
    ops: list[Op],
    *,
    target_dir: str = "",
    hub: str | None = None,
    bounds: Any | None = None,
    read_note: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Commit `ops` through leash → validate → snapshot → write → lint (+rollback).

    Returns a result dict with a `status` of:
      committed | rolled_back | no_ops | error
    plus `committed` (count), `txn_id`, and `rejected_by_bounds` (ops the bounds dropped).
    """
    from silica.tools.composed import silica_validate_ops, silica_bulk_write, silica_lint
    from silica.tools.wrapped import silica_snapshot, silica_restore
    from silica.kernel.workqueue import path_lease

    rejected_by_bounds: list[dict] = []
    if bounds is not None:
        ops, rejected_by_bounds = bounds.enforce(ops, read_note=read_note)

    actionable = [o for o in ops if o.op != OpType.skip]
    if not actionable:
        return {"status": "no_ops", "committed": 0, "rejected_by_bounds": rejected_by_bounds}

    ops_path = _write_ops_tmp(ops)
    try:
        # Validate (C4: overwrites ops_path with the validated, coerced ops).
        vres = silica_validate_ops(ops_path, payload_paths=[], target_dir=target_dir, hub=hub or "")
        if "error" in vres:
            return {"status": "error", "error": vres["error"], "rejected_by_bounds": rejected_by_bounds}
        if vres.get("validated_count", 0) == 0:
            return {"status": "no_ops", "committed": 0, "rejected_by_bounds": rejected_by_bounds, "validate": vres}

        touched = [
            (p, op.op.value if op.op else "", op.hub or "")
            for op in load_ops(ops_path)
            if (p := op.touched_ref()) and op.op is not OpType.skip
        ]
        lease_paths = sorted({p for p, _, _ in touched})

        # The write-gate lane (spec §4): what is about to touch the vault and
        # why, narrated from the material this function already computed.
        from silica.agent import narration as _narr_mod
        import uuid as _uuid
        _wid = f"w-{_uuid.uuid4().hex[:8]}"
        _narr_mod.NARRATOR.narrate(
            "write", "proposed",
            f"write {len(touched)} note(s): " + ", ".join(p for p, _, _ in touched[:3])
            + ("…" if len(touched) > 3 else ""),
            {"touched": touched, "rejected_by_bounds": len(rejected_by_bounds)},
            id=_wid)

        with ExitStack() as stack:
            # Deterministic lock ordering (sorted) avoids deadlock between sub-agents.
            for p in lease_paths:
                stack.enter_context(path_lease(p))

            sres = silica_snapshot(ops_path)
            if "error" in sres:
                return {"status": "error", "error": sres["error"], "rejected_by_bounds": rejected_by_bounds}
            txn_id = sres["txn_id"]
            inverses = sres.get("inverses", [])

            def _rollback(reason: dict) -> dict:
                try:
                    silica_restore(txn_id=txn_id, inverses=inverses)
                except Exception as e:
                    logger.error("commit_ops rollback failed: %s", e)
                    reason["rollback_error"] = str(e)
                reason.update({"status": "rolled_back", "committed": 0, "txn_id": txn_id,
                               "rejected_by_bounds": rejected_by_bounds})
                _narr_mod.NARRATOR.narrate(
                    "write", "rolled_back",
                    f"rolled back {len(touched)} note(s): "
                    + str(reason.get("error") or reason.get("lint_failures") or "")[:80],
                    {"txn_id": txn_id, "reason": {k: v for k, v in reason.items()
                                                  if k in ("error", "lint_failures")}},
                    id=_wid)
                return reason

            wres = silica_bulk_write(ops_path)
            if "error" in wres:
                return _rollback({"error": wres["error"]})
            if wres.get("successful", 0) == 0 and wres.get("total", 0) > 0:
                return _rollback({"error": "all write ops failed"})

            lint_failures: list[dict] = []
            for p, op_type, h in touched:
                lr = silica_lint(p, op_type=op_type, hub=h)
                if not lr.get("success", True):
                    lint_failures.append({"path": p, "errors": lr.get("errors")})
            if lint_failures:
                return _rollback({"lint_failures": lint_failures})

        # Committed cleanly — journal the inverses if inside a batch run so the
        # whole pass is revertable via /revert.
        undo_run_id = _current_undo_run.get()
        if undo_run_id:
            _journal_inverses(undo_run_id, inverses)
        _append_provenance(undo_run_id, touched)

        _narr_mod.NARRATOR.narrate(
            "write", "committed",
            f"committed {wres.get('successful', len(lease_paths))} note(s), txn {txn_id[:8]}",
            {"txn_id": txn_id, "committed": wres.get("successful", len(lease_paths))},
            id=_wid)
        return {
            "status": "committed",
            "committed": wres.get("successful", len(lease_paths)),
            "txn_id": txn_id,
            "rejected_by_bounds": rejected_by_bounds,
        }
    finally:
        # Staging file is scoped to this call; nothing references it after return.
        # Unlink here — commit_ops runs outside the FSM's _cleanup_tmp lifecycle.
        with suppress(OSError):
            os.unlink(ops_path)
