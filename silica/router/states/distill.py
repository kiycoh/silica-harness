# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Injector distillation states: DELEGATE, SANITIZE, VALIDATE.

Handler bodies for InjectorFSM, extracted from orchestrator.py: each function
takes the FSM instance and mutates its context/state exactly as the former
method did. Patchable collaborators (DRIVER, CONFIG, tools, load_ops, time)
are resolved through the orchestrator module namespace (orch.X) so tests that
patch silica.router.orchestrator.* keep working.
"""
from __future__ import annotations

import logging
import os
import re
import hashlib
import json
import time
import typing
from functools import partial
from typing import Any, TYPE_CHECKING

from silica.router import orchestrator as orch
from silica.kernel.outline import parse_outline, run_outliner, vault_outline
from silica.kernel.prep_delegation import payload_inbox_text, run_distiller

if TYPE_CHECKING:
    from silica.router.orchestrator import InjectorFSM

logger = logging.getLogger(__name__)

# Emitted by the C3 title gate in validate_operations (band 2).
_NEAR_TITLE_RE = re.compile(
    r"near_title candidate='([^']*)' path='([^']*)'(?: ratio=([0-9.]+))?"
)

# Frontmatter `date:` of a source document. Two-step (block, then key inside
# it) so a `date:` line in the body can never match; the YYYY-MM-DD prefix is
# enough, quoted or datetime values included.
_FM_BLOCK_RE = re.compile(r"\A---\r?\n(.*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.S)
_FM_DATE_RE = re.compile(r"^date:\s*['\"]?(\d{4}-\d{2}-\d{2})", re.M)


def _doc_date(fsm: "InjectorFSM", idx: int) -> str:
    """Frontmatter `date:` of the chunk's source document, or "".

    A dated document (journal page, meeting note) anchors the distiller's
    relative-date resolution to the day it was written; callers fall back to
    the run's start date. Best-effort: unreadable source returns "".
    """
    try:
        fi = fsm._chunk_flat_to_fi_ci.get(idx, (None, None))[0]
        chunk = fsm._file_chunks.get(fi) if fi is not None else None
        src = (chunk or {}).get("source_file") or fsm.inbox_file
        text = orch.DRIVER.read_note(src).content or ""
        fm = _FM_BLOCK_RE.match(text)
        m = _FM_DATE_RE.search(fm.group(1)) if fm else None
        return m.group(1) if m else ""
    except Exception:
        return ""


def _enqueue_near_title_dedups(fsm: "InjectorFSM", rejected_raw: list, *, landed: bool = False) -> None:
    """Fuzzy-band title flags become live dedup WorkItems (C3 reuses C2).

    `landed`: the op is a note on disk (soft gate, write.py _settle_flagged),
    so the judge gets `loser_path` and a duplicate verdict has something to
    mark superseded. Without it the pair is judged as an incoming concept
    (legacy rejected shape). Best-effort: no queue (ad-hoc validate) → no-op.
    """
    wq = getattr(fsm, "work_queue", None)
    if wq is None:
        return
    from silica.kernel.workqueue import WorkItem, batch_dedup_items

    items: list[WorkItem] = []
    for r in rejected_raw:
        if not isinstance(r, dict):
            continue
        m = _NEAR_TITLE_RE.search(r.get("reason", "") or "")
        if not m:
            continue
        op = r.get("op", {}) or {}
        # The C3 title gate already computed the fuzzy ratio (validate.py band 2);
        # carry it as the dedup title_score so SILICA_DEDUP_GATE judges these
        # near-identical-title pairs on their real similarity instead of a
        # degenerate 0.0 (which would force every one of them to "distinct").
        try:
            title_score = float(m.group(3)) if m.group(3) else 0.0
        except (TypeError, ValueError):
            title_score = 0.0
        items.append(WorkItem(
            kind="dedup",
            target_path=m.group(2),
            context={
                "concept": op.get("heading", ""),
                "excerpt": op.get("snippet", ""),
                "candidate": m.group(1),
                "score": 0.0,
                "title_score": title_score,
                # _current_source_file, NOT inbox_file: the latter is pinned to
                # inbox_files[0] for the whole run, so on a folder nucleation
                # every worker's notes were filed under file 0 paired with
                # another file's sha — two sha values for one source, which
                # check_renucleate reads as "modified" and drifted_notes reads
                # as derived from a superseded version.
                "inbox_file": fsm._current_source_file,
                "hub": fsm.hub,
                "content_hash": fsm._current_content_hash,
                # The provenance ledger is keyed on the FSM's own run id — what
                # CLEANUP stamps and what the dangling-link sweep reads back.
                "run_id": getattr(getattr(fsm, "progress", None), "run_id", "") or "",
                "target_dir": fsm.target_dir,
                **({"loser_path": op.get("path", "")} if landed else {}),
            },
            reason=r.get("reason", "near_title"),
        ))
    # Same family collapse COLLISION applies: N rejections against one candidate
    # note become ONE judge call, instead of N calls each re-sending that note's
    # 8000-char body. Singletons pass through batch_dedup_items untouched.
    for wi in batch_dedup_items(items):
        try:
            wq.enqueue(wi)
        except Exception as _qe:
            logger.debug("VALIDATE: failed to enqueue near_title dedup item: %s", _qe)


def _enqueue_short_snippet_expands(fsm: "InjectorFSM", rejected_raw: list) -> None:
    """«snippet too short» rejections become live expand WorkItems.

    The op is already parked in the deferred bundle by _defer_ops; the expand
    worker re-prompts the LLM with the concept's inbox excerpt (max 2 attempts)
    and commits through the same gate — retry stays the exception. Called only
    for PARTIAL rejections: an all-rejected chunk re-delegates via the steer
    arc, and racing both would author the same notes twice. Best-effort: no
    queue (ad-hoc validate) → no-op.
    """
    wq = getattr(fsm, "work_queue", None)
    if wq is None:
        return
    from silica.kernel.workqueue import WorkItem

    excerpts: dict[str, str] = {}
    try:
        chunk = fsm._chunks[fsm._current_chunk_idx]
        for batch in chunk.get("batches", []):
            for c in batch.get("concepts", []):
                if isinstance(c, dict) and c.get("name"):
                    excerpts[c["name"]] = c.get("inbox_excerpt", "") or ""
    except Exception as _ce:
        logger.debug("EXPAND: excerpt lookup failed (items will skip): %s", _ce)

    for r in rejected_raw:
        if not isinstance(r, dict):
            continue
        reason = r.get("reason", "") or ""
        if not reason.startswith("snippet too short"):
            continue
        op = r.get("op", {}) or {}
        try:
            wq.enqueue(WorkItem(
                kind="expand",
                target_path=op.get("path", ""),
                context={
                    "op": op,
                    "excerpt": excerpts.get(op.get("heading", ""), ""),
                    "reason": reason,
                    "inbox_file": fsm._current_source_file,  # see the dedup item above
                    "hub": fsm.hub,
                    "content_hash": fsm._current_content_hash,
                    "run_id": getattr(getattr(fsm, "progress", None), "run_id", "") or "",
                    "target_dir": fsm.target_dir,
                },
                reason=reason,
            ))
        except Exception as _qe:
            logger.debug("VALIDATE: failed to enqueue expand item: %s", _qe)


def _steer_retryable(fsm: "InjectorFSM", rejected_raw: list, idx: int) -> list:
    """Rejections eligible for the partial steer arc.

    Excluded: failure classes with a specialized in-run recovery lane
    (near-title → dedup judge, short snippet → expand worker), collision-routed
    ops (not distiller-authored — steering the distiller about them is noise),
    and entries with no payload heading to rebuild a retry payload from.
    """
    collision_ops = fsm.context.get(f"chunk_{idx}_collision_ops", []) or []
    collision_keys = {
        (o.get("heading"), o.get("path")) for o in collision_ops if isinstance(o, dict)
    }
    out = []
    for r in rejected_raw:
        if not isinstance(r, dict):
            continue
        reason = r.get("reason", "") or ""
        if _NEAR_TITLE_RE.search(reason) or reason.startswith("snippet too short"):
            continue
        op = r.get("op") if isinstance(r.get("op"), dict) else None
        if not op or not op.get("heading"):
            continue
        if (op.get("heading"), op.get("path")) in collision_keys:
            continue
        out.append(r)
    return out


def _filter_chunk_to_concepts(chunk: dict, names: set) -> dict | None:
    """Copy of `chunk` with batches filtered to the named concepts (retry payload).

    Returns None when nothing matches — the caller falls back to the normal
    partial-success path instead of steering.
    """
    names = {n for n in names if n}
    if not names or not isinstance(chunk, dict):
        return None
    batches = []
    for batch in chunk.get("batches", []):
        kept = [c for c in batch.get("concepts", [])
                if isinstance(c, dict) and c.get("name") in names]
        if kept:
            batches.append({**batch, "concepts": kept})
    if not batches:
        return None
    return {**chunk, "batches": batches}


def _vocabulary_hubs(vault_ctx: dict) -> list[str]:
    """Hub note names fit to be shown as the vault's preferred terminology.

    The substrate prints these under "reuse these instead of coining synonyms",
    so what is in the list is what the distiller is told to name things after.
    Inbox hubs are dropped: staging clusters are hubbed by whatever the source
    happened to be, and on a vault of converted PDFs that meant 27 of 102 hub
    names were chapter slugs (`01-an-introduction-to-support-vector-machines`).
    Telling the model to reuse those is worse than telling it nothing.
    """
    from silica.kernel.recall.paths import is_inbox_path

    names = []
    for v in vault_ctx.values():
        hub = v.get("hub") if isinstance(v, dict) else None
        if not hub or not v.get("is_hub") or is_inbox_path(hub):
            continue
        names.append(hub.rsplit("/", 1)[-1])
    return sorted(set(names))


def _inject_graph_ctx(chunk: dict, vault_ctx: dict) -> dict:
    """Return a shallow-enriched copy of chunk with graph_context added to concepts.

    For every concept that has a vault_collision, adds a graph_context field:
        {"cluster_id": int, "hub": str|None, "is_hub": bool}
    Concepts without a vault_collision get graph_context=null.
    The original chunk dict is never mutated.
    """
    if not vault_ctx:
        return chunk

    enriched_batches = []
    for batch in chunk.get("batches", []):
        enriched_concepts = []
        for concept in batch.get("concepts", []):
            if not isinstance(concept, dict):
                enriched_concepts.append(concept)
                continue
            collision = concept.get("vault_collision") or {}
            cpath = collision.get("path", "").removesuffix(".md") if collision else ""
            gctx = vault_ctx.get(cpath)
            graph_context = (
                {
                    "cluster_id": gctx["cluster_id"],
                    "hub": gctx["hub"],
                    "is_hub": gctx["is_hub"],
                }
                if gctx and cpath
                else None
            )
            enriched_concepts.append({**concept, "graph_context": graph_context})
        enriched_batches.append({**batch, "concepts": enriched_concepts})
    return {**chunk, "batches": enriched_batches}


def _chunk_concept_count(chunk: dict) -> int:
    """Concepts remaining in a chunk (0 = nothing to distill).

    Production chunks carry ``batches[].concepts`` (kernel/partition.py). A
    chunk with no ``batches`` key but a top-level ``concepts`` list (legacy
    shape) is counted directly so it is never misread as empty.
    """
    if "batches" not in chunk:
        return len(chunk.get("concepts", []))
    return sum(len(b.get("concepts", [])) for b in chunk.get("batches", []))


def _file_profile(fsm: "InjectorFSM", idx: int) -> str | None:
    """Distill profile for chunk ``idx``: run-level arg > per-file PAYLOAD pin.

    The run-level arg (/promote, --profile) short-circuits the pin for every
    file; the pin is what the form ladder resolved at PAYLOAD
    (docs/specs/nucleation-forms.md). Both read sites (prompt and gate) go
    through here so they can never disagree on the lens.
    """
    run_level = getattr(fsm, "distill_profile", None)
    if run_level:
        return run_level
    fi = fsm._chunk_flat_to_fi_ci.get(idx, (fsm._current_file_idx, 0))[0]
    return fsm.context.get(f"file_{fi}_profile")


def _distill_inputs(fsm: "InjectorFSM", idx: int) -> dict[str, typing.Any]:
    """Snapshot the full run_distiller kwargs for chunk ``idx``.

    Called at dispatch time: everything the prompt needs is captured here, so a
    prefetched call is immune to later FSM mutation. steer_context is always
    None — steer retries never go through the prefetcher.
    """
    chunk = fsm._chunks[idx]

    ledger_digest: str | None = None
    try:
        ledger_digest = fsm.progress.digest(manifest=fsm.manifest)
    except Exception:
        pass

    vault_ctx = fsm.context.get("vault_graph_ctx", {})
    enriched_chunk = _inject_graph_ctx(chunk, vault_ctx) if vault_ctx else chunk

    substrate: str | None = None
    try:
        from silica.kernel.recall.run_substrate import build_substrate
        substrate = build_substrate(
            enriched_chunk,
            manifest_titles=fsm.manifest.titles(),
            cleared_parents=fsm.context.get("run_cleared_parents"),
            hub_names=_vocabulary_hubs(vault_ctx),
        )
    except Exception as _sub_e:
        logger.debug("DELEGATE: substrate build failed (non-fatal): %s", _sub_e)

    fi = fsm._chunk_flat_to_fi_ci.get(idx, (fsm._current_file_idx, 0))[0]
    return dict(
        payload=enriched_chunk,
        target=fsm.target_dir,
        hub=fsm.hub,
        ledger_digest=ledger_digest,
        steer_context=None,
        substrate=substrate,
        session_date=_doc_date(fsm, idx) or fsm.progress.started_at[:10],
        language=fsm.context.get(f"file_{fi}_language"),
        profile=_file_profile(fsm, idx),
    )


def _prefetch_ahead(fsm: "InjectorFSM", idx: int) -> None:
    """Dispatch distill calls for chunks [idx, idx+k) of the flat list.

    COLLISION for lookahead chunks runs here, early, on the main thread (the
    pass is main-thread-only by design); only the run_distiller network call
    goes to the pool. Spec: staleness ≤ k-1 chunks; the window may cross ONE
    file boundary (warm_next_file preps the next file when its chunks are not
    attached yet — same ≤ k-1 staleness class as within-file lookahead, which
    the 2026-07-18 k=1-vs-k=3 A/B bounded), never a second one.
    """
    k = int(getattr(orch.CONFIG, "distill_concurrency", 1) or 1)
    if k <= 1:
        return
    if fsm._prefetcher is None:
        from silica.router.prefetch import DistillPrefetcher
        fsm._prefetcher = DistillPrefetcher(max_workers=k)
    prefetcher = fsm._prefetcher

    from silica.router.states.collision import collision_pass
    from silica.router.states.setup import warm_next_file

    fi_cur = fsm._chunk_flat_to_fi_ci.get(idx, (fsm._current_file_idx, 0))[0]
    allowed_next: int | None = None  # the one file boundary the window may cross
    warmed = False
    j = idx
    while j < idx + k:
        if j >= len(fsm._chunks):
            # Window ran past the attached chunks: prep the next file early so
            # its first distill is in flight while this file's tail is written.
            if warmed or not warm_next_file(fsm):
                break
            warmed = True
            continue  # re-check j against the grown flat list
        fi_j = fsm._chunk_flat_to_fi_ci.get(j, (fi_cur, 0))[0]
        if fi_j != fi_cur:
            if allowed_next is None:
                allowed_next = fi_j
            elif fi_j != allowed_next:
                break  # never cross a second file boundary
        if fsm.context.get(f"chunk_{j}_steer_context"):
            # Steer/residue chunks run inline by design: handle_delegate
            # discards any prefetched future for them, so dispatching one
            # burns a full distill call for nothing.
            j += 1
            continue
        if j in prefetcher:
            j += 1
            continue
        if j > idx and not fsm.context.get(f"chunk_{j}_collision_done"):
            try:
                collision_pass(fsm, j)
                fsm.context[f"chunk_{j}_collision_done"] = True
            except Exception as _ce:
                logger.warning("prefetch: collision_pass(%d) failed (%s) — chunk stays sequential", j, _ce, exc_info=True)
                j += 1
                continue
        if _chunk_concept_count(fsm._chunks[j]) == 0:
            j += 1
            continue  # emptied by collision/novelty; the inline guard finishes it free
        # Content-addressed idempotency: never dispatch a chunk a prior run
        # already completed (same key derivation as handle_delegate).
        j_hash = fsm.context.get(f"chunk_{j}_input_hash") or hashlib.sha256(
            json.dumps(fsm._chunks[j], sort_keys=True).encode()
        ).hexdigest()
        try:
            done = fsm.progress.is_checkpoint_done(fsm._chunk_task_id("validate", j), j_hash)
        except Exception:
            done = None
        if done:
            j += 1
            continue
        kwargs = _distill_inputs(fsm, j)
        # partial, not `lambda kw=kwargs:` — the default-argument trick existed
        # to capture this turn's kwargs before j moves on, and partial does that
        # by binding them, without inventing a parameter the callee never takes.
        prefetcher.submit(j, partial(run_distiller, **kwargs))
        j += 1


def _run_outline_chunk(fsm: "InjectorFSM", idx: int, chunk: dict,
                       retry_payload: dict | None, steer_context: str | None) -> dict:
    """DELEGATE for the outline lane: whole source in, distiller-shaped ops out.

    Side effect on purpose: the chunk's concept list is replaced with the
    titles the model named, so VALIDATE's heading whitelist and patch-path
    check keep working unchanged. A steer retry reuses the stored outline and
    regenerates only the rejected bodies.
    """
    fi = fsm._chunk_flat_to_fi_ci.get(idx, (fsm._current_file_idx, 0))[0]
    inbox_file = (chunk.get("batches") or [{}])[0].get("inbox_file", "") or fsm._current_source_file()
    only: set[str] | None = None
    prior = None
    if retry_payload is not None:
        saved = fsm.context.get(f"chunk_{idx}_outline")
        prior = parse_outline(saved) if saved else None
        if prior is not None:
            only = {c.get("name") for b in retry_payload.get("batches", [])
                    for c in b.get("concepts", []) if isinstance(c, dict) and c.get("name")}
    try:
        rows = vault_outline(fsm.target_dir, exclude_titles={fsm.hub or ""})
    except Exception as _ve:  # a blind stage B loses cross edges, never the file
        logger.warning("DELEGATE outline: vault outline unavailable (%s) — no cross edges this chunk", _ve)
        rows = []
    from silica.router.states.finalize import file_language
    source_text = chunk.get("source_text", "")
    language = file_language(fsm, fi, source_text)
    res = run_outliner(
        source_text=source_text,
        source_basename=os.path.basename(inbox_file),
        target=fsm.target_dir, hub=fsm.hub,
        language=language,
        vault_outline=rows, outline=prior, only_titles=only, steer_context=steer_context,
    )
    fsm._chunks[idx] = {**chunk, "batches": [{"inbox_file": inbox_file, "concepts": res["concepts"]}]}
    if only is None:
        fsm.context[f"chunk_{idx}_outline"] = res["outline"]
    if fsm.warning_ledger is not None:
        for gap in res.get("gaps", []):
            fsm.warning_ledger.add(inbox_file, "outline_gap", f"no idea and no skip for section '{gap}'")
    return {"updates": res["updates"], "ephemerals": res.get("ephemerals", [])}


def handle_delegate(fsm: "InjectorFSM") -> None:
    fsm._get_chunks_from_context_if_empty()

    if not fsm._chunks or fsm._current_chunk_idx >= len(fsm._chunks):
        raise RuntimeError("No chunks available for iterative processing.")

    current_chunk = fsm._chunks[fsm._current_chunk_idx]
    idx = fsm._current_chunk_idx

    # Partial-steer retry: VALIDATE parked a payload filtered to the rejected
    # concepts. Consume it in place of the full chunk.
    retry_payload = fsm.context.pop(f"chunk_{idx}_retry_payload", None)
    if retry_payload is not None:
        current_chunk = retry_payload

    import json as _json
    if retry_payload is None:
        # Content-addressed idempotency (Phase 2): if this chunk was already
        # processed in a prior run with identical input, skip DELEGATE→SANITIZE→VALIDATE
        # and reuse the persisted knowledge-block ops file.
        #
        # Use the pre-COLLISION hash stored by _handle_collision so the key is
        # based on the original source input, not the vault-state-dependent
        # post-COLLISION chunk (which changes when resumed after a partial run).
        chunk_hash = fsm.context.get(f"chunk_{idx}_input_hash") or hashlib.sha256(
            _json.dumps(current_chunk, sort_keys=True).encode()
        ).hexdigest()
        saved_ops_path = fsm.progress.is_checkpoint_done(fsm._chunk_task_id("validate"), chunk_hash)
        if saved_ops_path and os.path.exists(saved_ops_path):
            logger.info(
                "DELEGATE chunk %d: content-addressed hit (hash=%s…) — skipping to SNAPSHOT",
                idx,
                chunk_hash[:8],
            )
            fsm._chunk_ctx["ops_path"] = saved_ops_path  # per-chunk ns: every consumer reads _chunk_ctx
            fsm.state = orch.InjectorState.SNAPSHOT
            return
    else:
        # Steer retry is mid-chunk work: never content-addressed-skip, and keep
        # the first pass's hash so the checkpoint stays keyed to the source input.
        chunk_hash = fsm.context.get(f"chunk_{idx}_hash", "") or hashlib.sha256(
            _json.dumps(current_chunk, sort_keys=True).encode()
        ).hexdigest()

    _prefetch_ahead(fsm, idx)

    logger.info(f"--- DISTILLING BATCH {idx + 1}/{len(fsm._chunks)} ---")
    fsm._progress_note(fsm._chunk_task_id("distill"), "distill", "running")

    # Phase 6: pass steering correction if VALIDATE sent us back here.
    # Tier 2 cascade: the rejection IS the calibrated uncertainty signal, so
    # every steer retry escalates to the escalation model.
    steer_context: str | None = fsm.context.get(f"chunk_{idx}_steer_context")
    if steer_context:
        logger.info("DELEGATE chunk %d: re-attempt with steering correction (escalated)", idx)
        fsm.context["escalations"] = fsm.context.get("escalations", 0) + 1

    kwargs = _distill_inputs(fsm, idx)
    kwargs["escalate"] = bool(steer_context)
    if retry_payload is not None or steer_context:
        # Steer retries always run inline with the retry payload + feedback;
        # the prefetcher refuses re-submission of a popped idx by design.
        kwargs["payload"] = (
            _inject_graph_ctx(current_chunk, fsm.context.get("vault_graph_ctx", {}))
            if fsm.context.get("vault_graph_ctx") else current_chunk
        )
        kwargs["steer_context"] = steer_context

    try:
        _t0 = time.monotonic()
        chunk_result: dict[str, Any]
        if current_chunk.get("lane") == "outline":
            # Before the zero-concept guard on purpose: an outline chunk has
            # no concepts until the model names them.
            chunk_result = _run_outline_chunk(fsm, idx, current_chunk, retry_payload, steer_context)
        elif _chunk_concept_count(current_chunk) == 0:
            # Empty chunk: the novelty gate emptied the file, or COLLISION
            # routed every concept away. Nothing to distill; synthesize an
            # empty result so VALIDATE still merges collision ops and its
            # no-actionable-ops path finishes the chunk. (Also fixes the
            # latent waste: a COLLISION-emptied chunk used to burn a call.)
            logger.info("DELEGATE chunk %d: zero concepts, skipping distiller call", idx)
            if fsm._prefetcher is not None:
                fsm._prefetcher.pop(idx)  # discard any stale prefetched future
            chunk_result = {"updates": []}
        else:
            fut = fsm._prefetcher.pop(idx) if fsm._prefetcher is not None else None
            if fut is not None and kwargs["steer_context"] is None:
                try:
                    chunk_result = fut.result()
                except Exception as _pe:
                    logger.warning("DELEGATE: prefetched distill for chunk %d failed (%s) — retrying inline", idx, _pe, exc_info=True)
                    chunk_result = run_distiller(**kwargs)
            else:
                chunk_result = run_distiller(**kwargs)
        # Wall-clock the chunk actually cost the run (wait time for prefetched
        # calls) — the A/B report's per-chunk latency source. Mirrored into the
        # persisted ledger (inputs) so a finished run can be diagnosed without
        # checkpoint-mtime archaeology; saved by the "done" progress note below.
        _secs = round(time.monotonic() - _t0, 2)
        fsm.context.setdefault("distill_secs", {})[idx] = _secs
        fsm.progress.inputs.setdefault("distill_secs", {})[fsm._chunk_task_id("distill")] = _secs
        if "error" in chunk_result:
            fsm._progress_note(fsm._chunk_task_id("distill"), "distill", "failed", error=chunk_result["error"])
            raise RuntimeError(f"Distiller error on batch {idx}: {chunk_result['error']}")

        # Episodic lane: route personal/ephemeral facts to the short-term
        # store. Never fails the ingest (capture_from_distill swallows).
        if getattr(fsm, "episodic_capture", True):
            from silica.kernel.recall.episodic import capture_from_distill
            capture_from_distill(
                chunk_result,
                run_id=fsm.progress.run_id,
                # Deliberately the ingest day, not the doc date: episodic TTL
                # keys off `seen`, and a backdated doc would expire on arrival.
                # seen_override is the one bench exception (LoCoMo e2e leg):
                # sessions carry historical dates temporal questions depend on.
                seen=fsm.seen_override or fsm.progress.started_at[:10],
            )

        distiller_path = fsm._make_tmp(chunk_result)
        fsm._chunk_ctx["distiller_output_path"] = distiller_path
        # Store chunk hash for knowledge-block write at VALIDATE
        fsm.context[f"chunk_{idx}_hash"] = chunk_hash
        fsm._progress_note(fsm._chunk_task_id("distill"), "distill", "done", output_ref=distiller_path)
        fsm._transition_success()

    except Exception as e:
        raise RuntimeError(f"Critical failure delegating batch {idx}: {e}")


def handle_sanitize(fsm: "InjectorFSM") -> None:
    with orch.phase(fsm, fsm._chunk_task_id("sanitize"), "sanitize"):
        # The chunk's own inbox text anchors verbatim-body escape repair:
        # doublings the source contains survive, everything else collapses.
        # The full chunk is a superset of any steer-retry payload — a
        # superset only errs toward preserving, the safe direction.
        source = payload_inbox_text(fsm._chunks[fsm._current_chunk_idx])
        res = orch.silica_sanitize(fsm._chunk_ctx["distiller_output_path"],
                                   verbatim_source=source or None)
        if "error" in res:
            fsm._progress_note(fsm._chunk_task_id("sanitize"), "sanitize", "failed", error=res["error"])
            raise RuntimeError(f"Sanitize failed: {res['error']}")
        fsm._chunk_ctx["sanitized"] = res


def handle_validate(fsm: "InjectorFSM") -> None:
    idx = fsm._current_chunk_idx
    fsm._progress_note(fsm._chunk_task_id("validate"), "validate", "running")
    sanitized = fsm._chunk_ctx["sanitized"]["parsed"]
    ops_raw = sanitized.get("updates", sanitized) if isinstance(sanitized, dict) else sanitized
    if not isinstance(ops_raw, list):
        ops_raw = [ops_raw]

    # Merge collision-routed patch ops (Phase 5): prepend so they go through
    # the same validate→snapshot→write path as distiller-generated ops.
    collision_ops = fsm.context.get(f"chunk_{idx}_collision_ops", [])
    if collision_ops:
        ops_raw = list(collision_ops) + list(ops_raw)

    # Partial-steer retry: validated ops from the previous attempt re-enter the
    # gate ahead of the retry output — re-validation is deterministic, and the
    # path-dedup keeps the richer op if the model re-emitted one anyway.
    carry_ops = fsm.context.pop(f"chunk_{idx}_carry_ops", None)
    if carry_ops:
        ops_raw = list(carry_ops) + list(ops_raw)

    # Cohesion pass: inject sibling cross-references into write ops' related[]
    # before validation so the links land in the written frontmatter.
    # Scope: same-chunk siblings only (cross-chunk handled by AUTOLINK/BACKLINK).
    from silica.kernel.text.cohesion import cohesion_pass
    ops_raw = cohesion_pass(ops_raw)

    ops_path = fsm._make_tmp(ops_raw)

    fsm._get_chunks_from_context_if_empty()

    payload_paths: list[str] = []
    if fsm._chunks and fsm._current_chunk_idx < len(fsm._chunks):
        payload_paths.append(fsm._make_tmp(fsm._chunks[fsm._current_chunk_idx]))
    else:
        # Fallback to general payload if _chunks is not populated
        payload_data = fsm.context.get("payload", {})
        if "chunks" in payload_data:
            for chunk in payload_data["chunks"]:
                payload_paths.append(fsm._make_tmp(chunk))
        elif "payload" in payload_data:
            payload_paths.append(fsm._make_tmp(payload_data["payload"]))

    res = orch.silica_validate_ops(
        ops_path,
        payload_paths=payload_paths,
        target_dir=fsm.target_dir,
        hub=fsm.hub,
        # Same profile as the prompt: an extractive run judged by the default
        # floor would reject its own (correct) verbatim output.
        profile=_file_profile(fsm, fsm._current_chunk_idx),
    )

    if "error" in res:
        raise RuntimeError(f"Validate failed: {res['error']}")

    fsm.context["validate"] = res

    # Span-grounding warnings (warn-only, never a rejection) → run ledger,
    # persisted as <run_dir>/warnings.json alongside the orphan warnings.
    # Doubles as calibration data for the gate's thresholds.
    if fsm.warning_ledger is not None:
        for u in res.get("ungrounded", []):
            fsm.warning_ledger.add(
                u.get("path") or "",
                "ungrounded_span",
                f"{u.get('heading', '')}: " + " | ".join(s[:60] for s in u.get("spans", [])),
            )

    max_rate = fsm._get_recipe_gate("rejection_rate_max", 0.10)

    if orch.CONFIG.verbose:
        total_ops = res.get("validated_count", 0) + res.get("rejected_count", 0)
        logger.info(
            "[DEBUG VALIDATE Gate]: Success: %s | Total evaluated ops: %d | Validated (accepted): %d | Rejected: %d | Rejection Rate: %.1f%% (Max Allowed: %.1f%%)",
            res.get("success"),
            total_ops,
            res.get("validated_count", 0),
            res.get("rejected_count", 0),
            res.get("rejection_rate", 0) * 100,
            max_rate * 100,
        )
        for r in res.get("rejected_ops", []):
            op = r.get("op", {}) if isinstance(r, dict) else {}
            label = op.get("path") or op.get("heading") or "?"
            logger.info(
                "VALIDATE reject: [%s] %s — %s",
                op.get("type", "?"), label, r.get("reason", "(no reason given)"),
            )

    if res.get("validated_count", 0) == 0 and res.get("rejected_count", 0) == 0:
        logger.info("VALIDATE: no actionable ops (all skip) — short-circuit to CLEANUP")
        # Provisional per-chunk verdict; CLEANUP overrides it to Success if any
        # chunk in the run had actionable ops (run_had_ops), fixing the sticky
        # run-global that mislabelled multi-chunk runs (A24).
        fsm.context["final_status"] = "no_ops"
        fsm._chunk_ctx["ops_path"] = ops_path
        fsm._progress_note(fsm._chunk_task_id("validate"), "validate", "done")
        fsm.state = orch.InjectorState.CLEANUP
        return

    # Run-global: any chunk with actionable ops makes the run not-no_ops, so a
    # later all-skip chunk can't drag a committing run back to no_ops (A24).
    if res.get("validated_count", 0) > 0:
        fsm.context["run_had_ops"] = True

    # Persist rejected ops to the deferred store so the model can retry them
    # later without re-running the expensive RECON → DELEGATE cycle.
    rejected_raw = res.get("rejected_ops", [])
    if rejected_raw:
        deferred_ops = [
            r.get("op", r) if isinstance(r, dict) and "op" in r else r
            for r in rejected_raw
        ]
        rejection_reasons = {
            (r.get("op", {}).get("path") or r.get("op", {}).get("heading") or "?"): r.get("reason", "")
            for r in rejected_raw if isinstance(r, dict)
        }
        if fsm._defer_ops(deferred_ops, rejection_reasons, phase="VALIDATE"):
            logger.warning(
                "VALIDATE: %d op(s) rejected and saved to deferred store (hash=%s…). "
                "Use silica_deferred_retry to attempt them later.",
                len(rejected_raw),
                fsm._current_content_hash[:8],
            )

    # Accumulate cleared parent references across chunks.
    # These are prospective links (parent notes not yet in vault) that the
    # distiller can anticipate in subsequent chunks or future runs.
    cleared = res.get("cleared_parents", [])
    if cleared:
        fsm.context.setdefault("run_cleared_parents", []).extend(cleared)
        logger.debug("VALIDATE: %d parent reference(s) cleared to hub fallback (tracked as forward refs)", len(cleared))

    # Unresolved inline wikilinks are kept verbatim as dangling forward-refs (no
    # rejection); the tool result lists them, nothing downstream reads the list.
    cleared_links = res.get("cleared_links", [])
    if cleared_links:
        logger.debug("VALIDATE: %d unresolved wikilink(s) kept as forward refs", len(cleared_links))

    rejection_rate = res.get("rejection_rate", 0)
    if rejection_rate >= max_rate:
        logger.warning(
            "VALIDATE: rejection rate %.1f%% exceeds threshold %.1f%% — continuing with %d validated op(s).",
            rejection_rate * 100,
            max_rate * 100,
            res.get("validated_count", 0),
        )

    from silica.kernel.prep_delegation import render_steer_feedback

    steer_attempts = fsm.context.get(f"chunk_{idx}_steer_attempts", 0)
    # One in-flight steer by default: rejected ops are parked in the deferred
    # store, and the second recovery pass now happens at the boundary via
    # silica_anneal (batched, off the critical path). Raise the gate in a
    # recipe if a lane wants the historical two in-flight retries.
    _max_steer = fsm._get_recipe_gate("max_steer_attempts", 1)

    # Abort only when no validated ops remain — partial success is fine.
    if res.get("validated_count", 0) == 0:
        # Phase 6 steering arc: re-delegate with per-op rejection feedback.
        if steer_attempts < _max_steer:
            steer_attempts += 1
            fsm.context[f"chunk_{idx}_steer_attempts"] = steer_attempts
            fsm.context[f"chunk_{idx}_steer_context"] = render_steer_feedback(
                res.get("rejected_ops", []),
                attempt=steer_attempts,
                max_attempts=_max_steer,
                partial=False,
                ungrounded=res.get("ungrounded", []),
            )
            logger.warning(
                "VALIDATE: steer attempt %d/%d for chunk %d — re-delegating with correction.",
                steer_attempts, _max_steer, idx,
            )
            fsm._progress_note(fsm._chunk_task_id("validate"), "validate", "running",
                                error=f"steer {steer_attempts}/{_max_steer}")
            try:
                fsm.progress.set_status(  # type: ignore[union-attr]
                    fsm._chunk_task_id("distill"), "running",
                    error=f"steer attempt {steer_attempts}"
                )
            except Exception:
                pass
            fsm.state = orch.InjectorState.DELEGATE
            return
        # Exhausted steering budget → defer and short-circuit.
        logger.warning("VALIDATE: steer budget exhausted (%d/%d) — deferring chunk %d.", steer_attempts, _max_steer, idx)
        fsm._chunk_ctx["abort_reason"] = "All ops rejected — nothing to write"
        # Provisional; CLEANUP overrides to Success if the run had ops elsewhere (A24).
        fsm.context["final_status"] = "no_ops"
        fsm._chunk_ctx["ops_path"] = ops_path
        fsm._progress_note(fsm._chunk_task_id("validate"), "validate", "done")
        fsm.state = orch.InjectorState.CLEANUP
        return

    # Partial rejection: steer ONLY the rejected concepts back through the
    # distiller with per-op feedback, carrying the validated ops forward for
    # the merge on the next VALIDATE pass. Rejections owned by a specialized
    # in-run lane (dedup judge, expand worker) are excluded — racing two
    # recovery paths would author the same notes twice.
    retryable = _steer_retryable(fsm, rejected_raw, idx) if rejected_raw else []
    if retryable and steer_attempts < _max_steer:
        retry_payload = _filter_chunk_to_concepts(
            fsm._chunks[fsm._current_chunk_idx] if fsm._chunks else {},
            {r["op"].get("heading", "") for r in retryable},
        )
        if retry_payload is not None:
            steer_attempts += 1
            fsm.context[f"chunk_{idx}_steer_attempts"] = steer_attempts
            fsm.context[f"chunk_{idx}_steer_context"] = render_steer_feedback(
                retryable,
                attempt=steer_attempts,
                max_attempts=_max_steer,
                accepted=res.get("validated_ops", []),
                partial=True,
                ungrounded=res.get("ungrounded", []),
            )
            fsm.context[f"chunk_{idx}_retry_payload"] = retry_payload
            fsm.context.setdefault(f"chunk_{idx}_carry_ops", []).extend(
                res.get("validated_ops", [])
            )
            logger.warning(
                "VALIDATE: partial steer attempt %d/%d for chunk %d — re-delegating %d rejected op(s), carrying %d validated.",
                steer_attempts, _max_steer, idx, len(retryable), res.get("validated_count", 0),
            )
            fsm._progress_note(fsm._chunk_task_id("validate"), "validate", "running",
                                error=f"steer {steer_attempts}/{_max_steer} (partial)")
            fsm.state = orch.InjectorState.DELEGATE
            return

    # Knowledge-block consolidation (Phase 2): persist the validated ops to
    # a stable path in the run directory so they survive tmp cleanup and
    # enable content-addressed skip on re-runs.
    chunk_hash = fsm.context.get(f"chunk_{idx}_hash", "")
    kb_path: str = ops_path  # fallback to tmp if save fails
    if chunk_hash:
        try:
            kb_path = fsm._save_knowledge_block(idx, ops_path)
        except Exception as _kb_e:
            logger.debug("Knowledge-block save failed (non-fatal): %s", _kb_e)

    fsm._chunk_ctx["ops_path"] = kb_path
    fsm._progress_note(
        # Same id the resume check reads (handle_delegate / _prefetch_ahead call
        # is_checkpoint_done on _chunk_task_id("validate")): a checkpoint filed
        # under any other id is invisible, and the real f{fi}_c{ci}_validate task
        # — set to "running" at the top of this handler — would never leave it.
        fsm._chunk_task_id("validate"),
        "validate",
        "done",
        output_ref=kb_path,
        content_hash=chunk_hash or None,
    )
    fsm._transition_success()
