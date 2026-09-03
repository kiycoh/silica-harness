# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Injector run-setup states: RECON, PAYLOAD, SALIENCE.

Handler bodies for InjectorFSM, extracted from orchestrator.py: each function
takes the FSM instance and mutates its context/state exactly as the former
method did. Patchable collaborators (DRIVER, CONFIG, tools, load_ops, time)
are resolved through the orchestrator module namespace (orch.X) so tests that
patch silica.router.orchestrator.* keep working.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from silica.kernel.forms import read_source_text
from silica.kernel.outline import lane_for
from silica.router import orchestrator as orch

if TYPE_CHECKING:
    from silica.router.orchestrator import InjectorFSM

logger = logging.getLogger(__name__)


def handle_recon(fsm: "InjectorFSM") -> None:
    """Concept recon for the CURRENT file only (per-file pipeline).

    The FSM loops RECON→…→WRITE per file: file 0 reaches its first write
    after one file's worth of embedding, not the whole inbox's. Cross-file
    coherence is carried by the substrate refreshed after each write, not by
    an up-front all-files pass.
    """
    fi = fsm._current_file_idx
    inbox_file = fsm.inbox_files[fi]
    with orch.phase(fsm, "recon", "recon"):
        # Warmed early by the distill prefetch window (warm_next_file): reuse
        # the recon it already ran instead of paying the pass twice.
        warm = fsm.context.pop(f"warm_recon_{fi}", None)
        res = warm if warm is not None else _recon_for_lane(fsm, fi, inbox_file)
        if "error" in res:
            fsm._progress_note("recon", "recon", "failed", error=res["error"])
            raise RuntimeError(f"Recon failed for {inbox_file}: {res['error']}")
        # Accumulated across files — context["recon"] stays a list for uniformity
        fsm.context.setdefault("recon", []).append(res)

        # Surface any deferred ops from a previous run of this file
        content_hash = fsm._file_content_hashes[fi] if fi < len(fsm._file_content_hashes) else ""
        if content_hash:
            from silica.kernel.recall.deferred import get_deferred_store
            bundle = get_deferred_store().get(content_hash)
            if bundle:
                rejected_count = len(bundle.get("rejected_ops", []))
                logger.info(
                    "RECON: %d deferred op(s) from a previous run of '%s' are waiting. "
                    "Call silica_deferred_retry('%s') to attempt them.",
                    rejected_count, inbox_file, content_hash[:8],
                )
                notice = {
                    "inbox_file": inbox_file,
                    "content_hash": content_hash,
                    "rejected_count": rejected_count,
                }
                existing = fsm.context.get("deferred")
                if existing is None:
                    fsm.context["deferred"] = notice
                elif isinstance(existing, list):
                    existing.append(notice)
                else:
                    fsm.context["deferred"] = [existing, notice]


def _recon_for_lane(fsm: "InjectorFSM", fi: int, inbox_file: str) -> dict:
    """The miner's report, or the outline lane's stand-in for it.

    The lane is decided here, not at PAYLOAD, because recon is the stage the
    outline lane has nothing to do in: the model names the units at DELEGATE
    (kernel/outline.py), so paying keyphrase extraction first would only feed
    a whitelist DELEGATE overwrites. The profile pin moves up with it (it is
    idempotent) since the form decides the lane.
    """
    try:
        _pin_file_profile(fsm, fi, inbox_file)
    except Exception as _form_e:
        # Same non-fatal contract the pin has at PAYLOAD: an unreadable or
        # unsniffable source distills under the vault fallback profile.
        logger.debug("RECON: profile pin skipped (non-fatal): %s", _form_e, exc_info=True)
    # The lane follows an explicit verdict only: the run-level profile
    # (/promote, --profile) or a form stamped on the file. A sniffed form
    # picks the lens, never the lane — the sniffer named the same lecture
    # `transcript`, `default` and `clip` on three consecutive runs
    # (2026-09-02), and the third would have sent it down the keyphrase
    # pipeline silently.
    run_level = getattr(fsm, "distill_profile", None)
    stamped = fsm.context.get(f"file_{fi}_form_origin") == "stamp"
    profile = run_level or (fsm.context.get(f"file_{fi}_profile") if stamped else None)
    lane = lane_for(profile)
    fsm.context[f"file_{fi}_lane"] = lane
    if lane != "outline":
        return orch.silica_recon(inbox_file)
    return {"success": True, "outline_lane": True, "file": inbox_file,
            "source_text": read_source_text(inbox_file)}


def _within_cluster_tol(cached_sig, sig: list[int]) -> bool:
    """Reuse cached clusters while the graph drifted < ~2% (or 50 nodes / 100 edges)."""
    if not cached_sig or len(cached_sig) != 2:
        return False
    cn, ce = cached_sig
    n, e = sig
    return abs(n - cn) <= max(50, n // 50) and abs(e - ce) <= max(100, e // 50)


def build_vault_graph_ctx() -> dict[str, dict]:
    """Compute per-note graph context (cluster/hub) from the current vault state.

    Returns a dict keyed by vault-relative path without .md extension:
        {"cluster_id": int, "hub": str|None, "is_hub": bool}
    Empty dict on any failure — all consumers treat missing context as a no-op.
    Uses the cheap structural report (no analytics): consumers read only
    cluster/hub, never PageRank.

    Scaling E: Louvain (~3.1s at 10k) is the per-run cost here. Clusters drift
    slowly, so the resulting ctx is cached keyed by a graph signature (node/edge
    counts) and reused while the graph drifted < ~2% — recomputed only when it
    has grown enough to matter. Accepts bounded staleness: a few recently-added
    notes read as cluster -1 (which consumers treat as "no cluster") until the
    next recompute — fine for routing context.
    """
    try:
        from silica.kernel.recall.graph_export import (
            build_graph_data,
            ctx_from_report,
            load_cluster_ctx,
            save_cluster_ctx,
        )
        from silica.kernel.report.graph_report import compute_report
        _t = orch.time.monotonic()

        nodes, edges = build_graph_data(folder="")  # cheap snapshot (no Louvain)
        sig = [
            sum(1 for n in nodes if n.get("type") != "ghost"),
            sum(1 for e in edges if e.get("type") == "EXTRACTED"),
        ]
        cached = load_cluster_ctx()
        if cached and _within_cluster_tol(cached.get("sig"), sig):
            ctx = cached.get("ctx") or {}
            logger.info(
                "PAYLOAD: vault graph context reused from cache — %d nodes (%.2fs, Louvain skipped)",
                len(ctx), orch.time.monotonic() - _t,
            )
            return ctx

        report = compute_report(_nodes_edges_override=(nodes, edges))  # Louvain on miss
        ctx = ctx_from_report(report)
        save_cluster_ctx(sig, ctx)
        logger.info(
            "PAYLOAD: vault graph context built — %d nodes, %d clusters (%.2fs)",
            len(ctx), len(report.clusters), orch.time.monotonic() - _t,
        )
        return ctx
    except Exception as _e:
        logger.info("PAYLOAD: vault graph context unavailable (%s) — graph features disabled", _e)
        return {}


def novelty_gate(fsm: "InjectorFSM", raw_payload: dict) -> tuple[dict, int]:
    """SAGE-style capture-side novelty gate (Tier 2 cost).

    A concept whose TITLE cosine to an existing note's title is >=
    CONFIG.novelty_tau leaves the payload BEFORE chunking, so chunk count
    (= distiller calls) falls with
    them. They are never dropped: each goes to the deferred store and, when a
    work queue is running, to the concurrent ternary dedup judge (duplicate /
    distinct / contradicts), which authors the patch when warranted.

    tau unset/0 = off (payload returned untouched). Best-effort: embedder or
    store trouble keeps concepts in the payload, same contract as COLLISION.
    Returns (filtered_payload, diverted_count).
    """
    tau = float(getattr(orch.CONFIG, "novelty_tau", 0.0) or 0.0)
    if tau <= 0.0:
        return raw_payload, 0

    from silica.agent.providers import get_embedder_or_none
    try:
        from silica.kernel.recall.embed import get_store
        store = get_store()
    except Exception as _e:
        logger.debug("NOVELTY: embed store unavailable (%s); gate skipped", _e)
        return raw_payload, 0
    if len(store) == 0:
        return raw_payload, 0
    embedder = get_embedder_or_none(orch.CONFIG, "NOVELTY", level="debug")
    if embedder is None:
        return raw_payload, 0

    from silica.kernel.recall.embed import _note_title_text
    from silica.kernel.recall.paths import is_inbox_path
    from silica.router.states.collision import _names_agree

    def _name_of(c) -> str:
        return c.get("name", "") if isinstance(c, dict) else str(c)

    # Order parameter: TITLE-vs-title cosine (like-vs-like). A short concept
    # name is embedded as a title and scored against stored title vectors,
    # never against full note bodies — the body signal was measured not to
    # separate captured from novel concepts (their cosine distributions
    # overlap; docs/architecture/silica-x-chemistry.md IV.3).
    names: list[str] = []
    for batch in raw_payload.get("batches", []):
        for c in batch.get("concepts", []):
            n = _name_of(c)
            if n.strip():
                names.append(n)
    uniq = list(dict.fromkeys(names))
    if not uniq:
        return raw_payload, 0
    try:
        vecs = embedder.embed([_note_title_text(n) for n in uniq])
        if len(vecs) != len(uniq):
            return raw_payload, 0
    except Exception as _e:
        logger.debug("NOVELTY: batch embed failed (%s); gate skipped", _e)
        return raw_payload, 0
    vec_by_name = dict(zip(uniq, vecs))

    kept_batches: list[dict] = []
    diverted: list[dict] = []
    for batch in raw_payload.get("batches", []):
        inbox_file = batch.get("inbox_file", fsm.inbox_file)
        kept: list = []
        for c in batch.get("concepts", []):
            name = _name_of(c)
            vec = vec_by_name.get(name)
            if not name or vec is None:
                kept.append(c)
                continue
            try:
                hits = store.title_cosine_top_k(vec, k=5)
            except Exception as _se:
                logger.debug("NOVELTY: title lookup failed for '%s': %s", name, _se)
                kept.append(c)
                continue
            hits = [h for h in hits if not is_inbox_path(h["path"])]
            best = hits[0] if hits else None
            # Cosine alone is a soup on dense taxonomic vaults: near-synonym and
            # negation-differing titles score ~0.97 (probe: nearest-distinct
            # p99=0.978). Require the same lexical name agreement COLLISION uses,
            # which rejects negation pairs ("context-free" vs "non context-free")
            # the embedding cannot tell apart.
            if (best is None or best["score"] < tau
                    or not _names_agree(name, best["name"])):
                kept.append(c)
                continue
            logger.info(
                "NOVELTY: '%s' ~ '%s' (title score=%.3f >= tau=%.2f); diverted",
                name, best["path"], best["score"], tau,
            )
            diverted.append({
                "concept": c,
                "inbox_file": inbox_file,
                "top_match": {"path": best["path"], "name": best["name"],
                              "score": best["score"]},
                "score": best["score"],
            })
        if kept:
            kept_batches.append({**batch, "concepts": kept})

    if diverted:
        from silica.router.states.collision import _deferred_op_dict
        fsm._defer_ops(
            [_deferred_op_dict(fsm, d, "novelty_gate") for d in diverted],
            {
                (d["concept"].get("name", str(i)) if isinstance(d["concept"], dict) else str(i)):
                f"novelty_gate score={d['score']:.3f}"
                for i, d in enumerate(diverted)
            },
            phase="NOVELTY",
        )
        if fsm.work_queue is not None:
            from silica.kernel.workqueue import WorkItem
            for d in diverted:
                c = d["concept"]
                match = d["top_match"]
                if not match.get("path"):
                    continue
                try:
                    fsm.work_queue.enqueue(WorkItem(
                        kind="dedup",
                        target_path=match["path"],
                        context={
                            "concept": c.get("name", "") if isinstance(c, dict) else str(c),
                            "excerpt": c.get("inbox_excerpt", "") if isinstance(c, dict) else "",
                            "candidate": match.get("name", match["path"]),
                            "score": d["score"],
                            "inbox_file": d["inbox_file"],
                            "hub": fsm.hub,
                            "content_hash": fsm._current_content_hash,
                            "target_dir": fsm.target_dir,
                        },
                        reason=f"novelty_gate score={d['score']:.3f}",
                    ))
                except Exception as _qe:
                    logger.debug("NOVELTY: failed to enqueue dedup item: %s", _qe)
        logger.info("NOVELTY: %d concept(s) diverted to the dedup lane pre-chunk",
                    len(diverted))

    return {**raw_payload, "batches": kept_batches}, len(diverted)


def _pin_file_profile(fsm: "InjectorFSM", fi: int, inbox_file: str) -> None:
    """Resolve the file's source form and pin the lens for all its chunks.

    The ladder lives in kernel.forms (stamp > sniff > fallback); a run-level
    profile (--profile, /promote) short-circuits the whole resolution, so no
    sniff call is ever paid under it. A `draft` verdict reaching the FSM means
    a direct tool call bypassed the dispatch filing path: it distills under
    the vault fallback, with the verdict kept visible in context and the log.
    """
    if getattr(fsm, "distill_profile", None) or f"file_{fi}_form" in fsm.context:
        return
    import silica.kernel.forms as forms

    text = forms.read_source_text(inbox_file)
    res = forms.resolve(text)
    profile = res.profile
    if res.form == "draft":
        from silica.kernel.prep_delegation import active_distill_profile

        profile = active_distill_profile()
        logger.info(
            "PAYLOAD: %s classified draft (%s) inside the FSM — filing happens "
            "at dispatch; distilling under the %r fallback.",
            inbox_file, res.origin, profile,
        )
    if profile:
        fsm.context[f"file_{fi}_profile"] = profile
    fsm.context[f"file_{fi}_form"] = res.form
    fsm.context[f"file_{fi}_form_origin"] = res.origin
    logger.info("PAYLOAD: %s profile=%s (%s)", inbox_file, profile or "default", res.origin)


def _assemble_file_chunks(fsm: "InjectorFSM", recon_cur: dict) -> tuple[dict, list[dict]]:
    """Recon result → (payload tool result, this file's chunk list).

    The pure assembly core of PAYLOAD, shared with warm_next_file. Raises
    RuntimeError when the payload tool errors; mutates no per-file FSM state.
    """
    if recon_cur.get("outline_lane"):
        # One chunk per file, the whole text, no concepts: COLLISION, SALIENCE
        # and the prefetcher all key off the concept list and stay inert;
        # DELEGATE fills the list with the titles the model names.
        chunk = {"schema_version": 1, "lane": "outline",
                 "batches": [{"inbox_file": recon_cur.get("file", ""), "concepts": []}],
                 "source_text": recon_cur.get("source_text", "")}
        return {"chunks": [chunk]}, [chunk]
    recon_path = fsm._make_tmp([recon_cur])
    phase_conf = fsm._get_recipe_phase("payload")
    max_concepts = phase_conf.get("partition_if_over", 200)
    max_bytes = int(os.getenv("DISTILLER_CHUNK_MAX_BYTES", str(30 * 1024)))
    res = orch.silica_payload(recon_path, max_concepts=max_concepts, max_bytes=max_bytes)
    if "error" in res:
        raise RuntimeError(f"Payload failed: {res['error']}")

    # Re-partition this file's payload (§3.6); fall back to the legacy
    # flat-chunk path when batch structure is absent (e.g. tests).
    from silica.kernel.partition import partition_by_file

    raw_payload: dict | None = None
    if "chunks" in res and res["chunks"]:
        all_batches: list[dict] = []
        for chunk in res["chunks"]:
            all_batches.extend(chunk.get("batches", []))
        if all_batches:
            raw_payload = {
                "schema_version": res["chunks"][0].get("schema_version", 1),
                "batches": all_batches,
            }
    elif "payload" in res:
        raw_payload = res["payload"]

    # Tier 2 novelty gate: divert already-captured concepts to the dedup lane
    # BEFORE chunking so chunk count (= distiller calls) falls with them.
    diverted_all = False
    if raw_payload is not None:
        raw_payload, _n_diverted = novelty_gate(fsm, raw_payload)
        diverted_all = _n_diverted > 0 and not any(
            b.get("concepts") for b in raw_payload.get("batches", [])
        )

    new_chunks: list[dict] = []
    if raw_payload and max_concepts > 0:
        # Single-file recon → normally a single group; collect all defensively.
        for fg in partition_by_file(raw_payload, max_concepts) or []:
            new_chunks.extend(fg.get("chunks", []))

    if not new_chunks and diverted_all:
        # Every concept diverted: one empty chunk carries the file through the
        # normal pipeline (DELEGATE skips the LLM, VALIDATE short-circuits).
        # Falling through would resurrect the unfiltered fallback chunks.
        new_chunks = [{"schema_version": (raw_payload or {}).get("schema_version", 1),
                       "batches": []}]

    if not new_chunks:
        # Fallback: all chunks of this payload belong to the current file.
        new_chunks = res.get("chunks", [])
        if not new_chunks and "payload" in res:
            new_chunks = [res["payload"]]
        if not new_chunks:
            new_chunks = [res]

    return res, new_chunks


def _attach_file_chunks(fsm: "InjectorFSM", fi: int, inbox_file: str,
                        new_chunks: list[dict]) -> int:
    """Attach a file's chunks: flat list + index, pins, facts, tasks.

    The bookkeeping half of PAYLOAD, shared with warm_next_file — which is
    why it never touches _current_chunk_idx (a warmed file is attached while
    the previous file is still being processed). Returns the first flat idx.
    """
    # Append this file's chunk group; flat indices continue after prior files'
    start_flat = len(fsm._chunks)
    fsm._file_chunks[fi] = {"source_file": inbox_file, "chunks": new_chunks}
    for ci, chunk in enumerate(new_chunks):
        fsm._chunks.append(chunk)
        fsm._chunk_flat_to_fi_ci[start_flat + ci] = (fi, ci)

    # Cache-stable prompt: pin the distiller LANGUAGE once per file so the
    # rendered template prefix is byte-identical across this file's chunks
    # (per-chunk detection can flap between chunks and bust the prefix cache).
    try:
        from silica.kernel.text import language as lang_mod
        from silica.kernel.prep_delegation import _payload_sample_text
        from silica.kernel.vault_manifest import get_active_manifest
        if not get_active_manifest().conventions.language:
            sample = ""
            for _chunk in new_chunks:
                sample = _payload_sample_text(_chunk) or _chunk.get("source_text", "")
                if sample:
                    break
            fsm.context[f"file_{fi}_language"] = lang_mod.display_name(
                lang_mod.detect(sample[:4000])
            )
    except Exception as _lang_e:
        logger.debug("PAYLOAD: language pin skipped (non-fatal): %s", _lang_e, exc_info=True)

    # Same seam, same cache-stability rationale as the language pin: resolve
    # the file's source form once and pin the lens for every chunk of the file
    # (docs/specs/nucleation-forms.md).
    try:
        _pin_file_profile(fsm, fi, inbox_file)
    except Exception as _form_e:
        logger.debug("PAYLOAD: profile pin skipped (non-fatal): %s", _form_e, exc_info=True)

    # Residue verification, stage 1: decompose the source into atomic facts
    # in the background — it depends only on the source text, so it rides the
    # file's whole distillation; the last chunk's WRITE picks it up. After
    # the profile pin on purpose: the dispatch skips draft forms.
    try:
        from silica.router.states import finalize as _fz
        _fz.maybe_dispatch_residue_decompose(fsm, fi, inbox_file)
    except Exception as _rde:
        logger.debug("PAYLOAD: residue decompose dispatch skipped (non-fatal): %s", _rde)

    # Accumulate facts["sources"] with per-file concept + chunk counts.
    # Replace-if-present: a warm-attached file that was detached for a residue
    # round re-attaches without doubling its entry.
    n_concepts = sum(
        len(b.get("concepts", []))
        for chunk in new_chunks
        for b in chunk.get("batches", [])
    )
    sources = fsm.progress.inputs.setdefault("sources", [])
    entry = {"inbox_file": inbox_file, "concepts": n_concepts, "chunks": len(new_chunks)}
    for i, s in enumerate(sources):
        if s.get("inbox_file") == inbox_file:
            sources[i] = entry
            break
    else:
        sources.append(entry)

    # Register per-chunk tasks with f{fi}_c{ci}_{cap} IDs and intra-file deps.
    # Existing ids are kept (resume re-runs PAYLOAD; warm may have registered).
    caps = ("collision", "distill", "sanitize", "validate", "snapshot", "write", "hub_update", "autolink", "backlink", "lint", "cleanup")
    existing = {t.id for t in fsm.progress.tasks}
    prev_in_file = "payload"
    for ci in range(len(new_chunks)):
        for cap in caps:
            tid = f"f{fi}_c{ci}_{cap}"
            if tid not in existing:
                fsm.progress.add_task(cap, task_id=tid, depends_on=[prev_in_file])
            prev_in_file = tid
    try:
        fsm.progress.save()
    except Exception as _e:
        logger.debug("progress save error (suppressed): %s", _e)

    return start_flat


def handle_payload(fsm: "InjectorFSM") -> None:
    """Payload assembly for the CURRENT file only (per-file pipeline).

    Appends this file's chunks to the flat chunk list and registers its
    progress tasks; earlier files' chunks are already written by the time
    this runs again for the next file. A file warm_next_file already attached
    fast-paths: only the chunk cursor is positioned.
    """
    fi = fsm._current_file_idx
    inbox_file = fsm.inbox_files[fi] if fi < len(fsm.inbox_files) else fsm.inbox_file
    fsm._progress_note("payload", "payload", "running")

    if fsm._file_chunks.get(fi, {}).get("chunks"):
        # Warmed early by the distill prefetch window: chunks, pins and tasks
        # are attached — position the cursor where the normal path would have.
        fsm.context["payload"] = fsm.context.pop(
            f"warm_payload_{fi}", fsm.context.get("payload"))
        fsm._current_chunk_idx = min(
            flat for flat, (f, _c) in fsm._chunk_flat_to_fi_ci.items() if f == fi
        )
        fsm._progress_note("payload", "payload", "done")
        if "vault_graph_ctx" not in fsm.context:
            fsm.context["vault_graph_ctx"] = build_vault_graph_ctx()
        fsm._transition_success()
        return

    # Current file's recon only — appended last by RECON
    recon_cur = fsm.context["recon"][-1]

    # The novelty gate runs inside _assemble_file_chunks and defers the concepts
    # it diverts through _defer_ops, which attributes the bundle to whatever file
    # the CHUNK CURSOR points at. Until this file's chunks are attached the cursor
    # still sits on the previous file's last chunk, so file N's diversions would be
    # filed under file N-1's content hash and source path. Park the cursor on the
    # flat index this file's first chunk is about to take (the same value
    # _attach_file_chunks computes and returns) and claim the file's identity now:
    # a populated `chunks` list, not the key's presence, is what marks a file as
    # attached, and _attach_file_chunks replaces the whole entry a few lines below.
    fsm._current_chunk_idx = len(fsm._chunks)
    fsm._chunk_flat_to_fi_ci[fsm._current_chunk_idx] = (fi, 0)
    fsm._file_chunks[fi] = {"source_file": inbox_file, "chunks": []}
    # Same attribution reason: the stale entry here is the PREVIOUS file's payload,
    # and _defer_ops falls back to it when the cursor has no chunk yet.
    fsm.context.pop("payload", None)

    try:
        res, new_chunks = _assemble_file_chunks(fsm, recon_cur)
    except RuntimeError as _pe:
        fsm._progress_note("payload", "payload", "failed", error=str(_pe))
        raise
    fsm.context["payload"] = res
    start_flat = _attach_file_chunks(fsm, fi, inbox_file, new_chunks)
    fsm._current_chunk_idx = start_flat

    fsm._progress_note("payload", "payload", "done")
    logger.info(
        "File %d/%d '%s': %d chunk(s) queued (flat %d–%d).",
        fi + 1, len(fsm.inbox_files), inbox_file,
        len(new_chunks), start_flat, len(fsm._chunks) - 1,
    )

    # Build vault graph context (cluster/hub/pagerank) once per run — reused
    # across files (consumers accept bounded staleness). Consumed by COLLISION,
    # DELEGATE (distiller enrichment), AUTOLINK, and HUB_UPDATE.
    if "vault_graph_ctx" not in fsm.context:
        fsm.context["vault_graph_ctx"] = build_vault_graph_ctx()

    fsm._transition_success()


def warm_next_file(fsm: "InjectorFSM") -> bool:
    """Prep the next uncommitted file early so the distill prefetch window
    can cross the file boundary (measured 40-65s of inline stall per file on
    the 2026-08-16 library batches: the first distill of every file ran with
    nothing in flight).

    Runs recon → payload assembly → salience on the main thread (~5s, all
    mechanical/local) and attaches the chunks; RECON/PAYLOAD/SALIENCE then
    fast-path through their warm guards at their normal position, keeping
    every per-file side effect (notices, ledger, archive) on its usual
    schedule. Best-effort: any failure leaves the file to its own states.
    Returns True when chunks were attached.
    """
    try:
        if float(getattr(orch.CONFIG, "novelty_tau", 0) or 0) > 0:
            # The gate diverts concepts to the deferred store; warming would
            # divert ahead of the file's own turn. Stand down.
            return False
        fi = fsm._next_uncommitted_file_idx(fsm._current_file_idx + 1)
        if fi >= len(fsm.inbox_files) or fsm._file_chunks.get(fi, {}).get("chunks"):
            return False
        inbox_file = fsm.inbox_files[fi]
        res = _recon_for_lane(fsm, fi, inbox_file)
        if "error" in res:
            return False
        payload_res, new_chunks = _assemble_file_chunks(fsm, res)
        fsm.context[f"warm_recon_{fi}"] = res
        fsm.context[f"warm_payload_{fi}"] = payload_res
        _attach_file_chunks(fsm, fi, inbox_file, new_chunks)
        dropped = _salience_filter(fsm, new_chunks)
        fsm.context[f"file_{fi}_salience_done"] = True
        logger.info(
            "WARM: file %d/%d '%s' prepped early — %d chunk(s) attached%s",
            fi + 1, len(fsm.inbox_files), inbox_file, len(new_chunks),
            f", {dropped} dropped by salience" if dropped else "",
        )
        return bool(new_chunks)
    except Exception as _we:
        logger.warning("WARM: next-file prep failed (%s) — staying sequential",
                       _we, exc_info=True)
        return False


def handle_salience(fsm: "InjectorFSM") -> None:
    """Thematic salience gate — Phase 2.05, current file's chunks only.

    Drops concepts whose embedding is too far from the document's thematic
    centroid.  Best-effort: any failure (embedder down, empty index) is
    logged and chunks pass unchanged.  Runs once per file (per-file
    pipeline); _on_pipeline_end restarts chunks from COLLISION, which is
    correct.  A file warm_next_file already filtered fast-paths.
    """
    if fsm.context.pop(f"file_{fsm._current_file_idx}_salience_done", None):
        fsm._transition_success()
        return

    fsm._get_chunks_from_context_if_empty()
    cur_fi = fsm._current_file_idx
    current_chunks = [
        chunk for flat_idx, chunk in enumerate(fsm._chunks)
        if fsm._chunk_flat_to_fi_ci.get(flat_idx, (0, 0))[0] == cur_fi
    ] or fsm._chunks  # fallback: no fi map (legacy/test paths) → all chunks

    dropped = _salience_filter(fsm, current_chunks)
    if dropped is not None:
        fsm.context["salience_dropped"] = dropped
        if dropped:
            logger.info("SALIENCE: %d concept(s) below thematic threshold removed", dropped)
    fsm._transition_success()


def _salience_filter(fsm: "InjectorFSM", chunks: list[dict]) -> int | None:
    """Drop off-theme concepts from ``chunks`` in place.

    The SALIENCE core, shared with warm_next_file. Returns the dropped count,
    or None when no embedder is available (gate skipped, chunks untouched).
    """
    τ_theme = getattr(orch.CONFIG, "sim_threshold_theme", 0.35)
    from silica.agent.providers import get_embedder_or_none
    from silica.kernel.recall.embed import document_theme_vector, _cosine
    from silica.kernel.text.text import clean_body
    embedder = get_embedder_or_none(orch.CONFIG, "SALIENCE")
    if embedder is None:
        return None

    theme_cache: dict[str, list[float]] = {}
    dropped = 0

    for chunk in chunks:
        for batch in chunk.get("batches", []):
            inbox_file = batch.get("inbox_file", fsm.inbox_file)
            if inbox_file not in theme_cache:
                try:
                    # Same cleaned body as RECON's keyphrase pass → the theme
                    # vector is a cache hit in embed._theme_cache, no re-embed.
                    body = clean_body(orch.DRIVER.read_note(inbox_file).content, fences=True)
                except Exception:
                    body = ""
                theme_cache[inbox_file] = document_theme_vector(embedder, body)
            theme = theme_cache[inbox_file]
            if not theme:
                continue

            concepts = batch.get("concepts", [])
            texts = [
                (c.get("name", "") + "\n" + c.get("inbox_excerpt", "")) if isinstance(c, dict) else str(c)
                for c in concepts
            ]
            if not texts:
                continue
            try:
                vecs = embedder.embed(texts)
            except Exception as _e:
                logger.debug("SALIENCE: embed failed (%s) — keeping batch", _e)
                continue

            kept = []
            for c, v in zip(concepts, vecs):
                score = _cosine(v, theme)
                name = c.get("name", "") if isinstance(c, dict) else str(c)
                if score < τ_theme:
                    logger.info(
                        "SALIENCE: drop '%s' (score=%.3f < τ_theme=%.2f)", name, score, τ_theme
                    )
                    dropped += 1
                else:
                    kept.append(c)
            batch["concepts"] = kept

    return dropped
