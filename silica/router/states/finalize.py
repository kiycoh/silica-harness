# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Injector terminal states: LINT (gate), CLEANUP, ROLLBACK.

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

from silica.router import orchestrator as orch

if TYPE_CHECKING:
    from silica.router.orchestrator import InjectorFSM

logger = logging.getLogger(__name__)


from silica.kernel.write.ops import OpType


def handle_lint(fsm: "InjectorFSM") -> None:
    fsm._progress_note(fsm._chunk_task_id("lint"), "lint", "running")
    try:
        ops = orch.load_ops(fsm._chunk_ctx["ops_path"])
    except Exception as e:
        raise RuntimeError(f"LINT: failed to read ops: {e}")

    touched = [
        (op.touched_ref(), op.op.value if op.op else "", op.hub or "")
        for op in ops
        if op.touched_ref() and op.op not in (OpType.delete, OpType.skip)
    ]

    # Per-op lint runs diff-aware (only violations a patch INTRODUCES block it);
    # this gate re-checks after autolink/backlink/hub mutations and must apply
    # the same tolerance, or a pre-existing issue in a patched user note aborts
    # the whole chunk (run 880b9aa9: f1_c0/f1_c1 died on an inherited
    # "frontmatter 'AI' missing" the WRITE phase had correctly waved through).
    baselines = fsm._chunk_ctx.get("lint_baseline", {})
    for path, op_type, hub in touched:
        res = orch.silica_lint(path, op_type=op_type or "", hub=hub or "")
        new_errors = [
            e for e in res.get("errors", []) if e not in set(baselines.get(path, []))
        ]
        if orch.CONFIG.verbose:
            logger.info(
                "[DEBUG LINT Gate]: File: %s | Type: %s | Hub: %s | Success: %s | Errors: %s",
                path,
                op_type,
                hub,
                res["success"],
                res.get("errors", []),
            )
        if not res["success"] and new_errors:
            fsm._chunk_ctx["abort_reason"] = f"Lint failed for {path}: {new_errors}"
            fsm._progress_note(fsm._chunk_task_id("lint"), "lint", "failed", error=fsm._chunk_ctx["abort_reason"])
            fsm.state = orch.InjectorState.ROLLBACK
            return

    # S3.2: Run graph-diff check
    regression_rule = fsm._get_recipe_gate("graph_regression", "forbid_new_orphans")
    if regression_rule != "allow":
        if fsm._pre_graph is None:
            fsm._chunk_ctx["abort_reason"] = "Graph regression gate failed: pre-write snapshot is missing"
            fsm._progress_note(fsm._chunk_task_id("lint"), "lint", "failed", error=fsm._chunk_ctx["abort_reason"])
            fsm.state = orch.InjectorState.ROLLBACK
            return
        try:
            from silica.driver.base import NoteRef
            snapshot_domain_dicts = fsm._chunk_ctx.get("snapshot_domain", [])
            if snapshot_domain_dicts:
                snapshot_domain = [NoteRef(**d) for d in snapshot_domain_dicts]
            else:
                # Fallback to touched refs if snapshot_domain is missing
                snapshot_domain = []
                for op in ops:
                    path = op.touched_ref()
                    if path:
                        name = os.path.splitext(os.path.basename(path))[0]
                        snapshot_domain.append(NoteRef(name=name, path=path))
            
            post_graph = orch.DRIVER.graph_snapshot(snapshot_domain)
            from silica.kernel.graph_diff import check_graph_regression
            
            created_paths = fsm._txn.created_paths if fsm._txn else []
            # Fold only chunks appended since the last LINT into the run-scoped stem
            # union (B8: was an O(chunks × concepts) rescan on every LINT). A chunk
            # collision-collapses after it is first folded, but its pre-collapse stems
            # only widen the allowed set — never manufacture a false regression.
            try:
                from silica.kernel.write.templates import slugify
                chunks = getattr(fsm, "_chunks", [])
                for chunk in chunks[fsm._run_concept_stems_n:]:
                    for batch in chunk.get("batches", []):
                        for concept in batch.get("concepts", []):
                            name = concept.get("name")
                            if name:
                                stem = os.path.splitext(os.path.basename(name))[0].lower()
                                fsm._run_concept_stems.add(stem)
                                fsm._run_concept_stems.add(slugify(stem))
                fsm._run_concept_stems_n = len(chunks)
            except Exception as _ce:
                logger.debug("Failed to extract run concept stems for graph check: %s", _ce)

            deferred_stems = set(fsm._chunk_ctx.get("deferred_stems", []))
            deferred_stems |= fsm._run_concept_stems

            # Exempt every path a phase of THIS chunk mutated, from the
            # rollback transaction (the single source of truth: SNAPSHOT
            # seeds it, every mutating phase appends). ops alone missed the
            # HUB_UPDATE/AUTOLINK/BACKLINK edits, so a pre-existing dangling
            # link in the hub's own planned-topic list read as "introduced"
            # and rolled back a healthy chunk (3 of the 6 rollbacks, run
            # 262e6847). validate deliberately keeps unresolved wikilinks in
            # run-authored content as forward-refs, so rule 2's business is
            # only notes this chunk never touched.
            patched_paths = frozenset(
                {p for op in ops
                 if op.op in (OpType.patch, OpType.overwrite) and (p := op.touched_ref())}
                | txn_touched_paths(fsm._txn)
            )

            success, errors = check_graph_regression(
                fsm._pre_graph, post_graph, created_paths, frozenset(deferred_stems),
                patched_paths,
            )

            if orch.CONFIG.verbose:
                logger.info(
                    "[DEBUG Graph Regression Gate]: Pre-write graph size: %d nodes | Post-write graph size: %d nodes | Rule: %s | Result: %s",
                    len(fsm._pre_graph.link_counts) if fsm._pre_graph and fsm._pre_graph.link_counts else 0,
                    len(post_graph.link_counts) if post_graph and post_graph.link_counts else 0,
                    regression_rule,
                    "PASSED" if success else f"FAILED: {errors}"
                )

            if not success:
                orphan_errors = [e for e in errors if e.startswith("Unplanned orphans")]
                drift_errors = [e for e in errors if e.startswith("Backlink drift")]
                blocking_errors = [
                    e for e in errors
                    if not e.startswith("Unplanned orphans") and not e.startswith("Backlink drift")
                ]
                if drift_errors:
                    logger.warning(
                        "[Graph Regression Gate]: Backlink drift (non-blocking): %s",
                        "; ".join(drift_errors),
                    )
                if orphan_errors:
                    logger.warning(
                        "[Graph Regression Gate]: Orphan warning (non-blocking): %s",
                        "; ".join(orphan_errors),
                    )
                    # Record run-created notes that ended this chunk orphaned.
                    # Acted on (if still orphaned) at end of run, not now —
                    # AUTOLINK/BACKLINK or a later chunk may yet connect them.
                    if fsm.warning_ledger is not None:
                        try:
                            from silica.kernel.graph_diff import normalize_ref
                            post_orphan_keys = {normalize_ref(r) for r in post_graph.orphans}
                            detail = "; ".join(orphan_errors)
                            for op in ops:
                                p = op.touched_ref()
                                if not p or op.op != OpType.write:
                                    continue
                                name = os.path.splitext(os.path.basename(p))[0]
                                if normalize_ref(NoteRef(name=name, path=p)) in post_orphan_keys:
                                    fsm.warning_ledger.add(p, "orphan", detail)
                        except Exception as _we:
                            logger.debug("orphan warning record failed (non-fatal): %s", _we)
                if blocking_errors:
                    reason = f"Graph regression gate failed: {'; '.join(blocking_errors)}"
                    logger.warning("[Graph Regression Gate]: Blocking errors (triggering rollback): %s", "; ".join(blocking_errors))
                    fsm._chunk_ctx["abort_reason"] = reason
                    fsm._progress_note(fsm._chunk_task_id("lint"), "lint", "failed", error=reason)
                    fsm.state = orch.InjectorState.ROLLBACK
                    return
        except Exception as e:
            logger.error("Failed to perform graph-diff check: %s", e)
            fsm._chunk_ctx["abort_reason"] = f"Graph regression gate error during check: {e}"
            fsm._progress_note(fsm._chunk_task_id("lint"), "lint", "failed", error=fsm._chunk_ctx["abort_reason"])
            fsm.state = orch.InjectorState.ROLLBACK
            return

    fsm._progress_note(fsm._chunk_task_id("lint"), "lint", "done")
    fsm._transition_success()


def _log_nucleate_completion(fsm: "InjectorFSM", fi: int, source_file: str) -> None:
    """Append one line to the vault's human journal (log.md).

    Pure projection of state WRITE/VALIDATE already recorded — the manifest
    (new/patch counts) and the deferred store (deferred count) — onto the
    log.md line shape. No new computation. Idempotent per (run_id, source
    file): a multi-file run shares one run_id and fires this once per file,
    so each file needs its own line, while a resume of the same run must not
    duplicate any (dedup_key). Best-effort and must never block CLEANUP.

    Called on every file conclusion — the success path (last chunk's CLEANUP)
    AND the failure path (last chunk rolled back, so CLEANUP never runs for it).
    A file whose earlier chunks committed real notes must still be accounted
    even when its last chunk fails; the `_files_logged` guard keeps the two
    entry points from double-recording.
    """
    try:
        from silica.kernel.recall.run_log import append_log_line, format_nucleate_event

        basename = os.path.basename(source_file)
        logged = getattr(fsm, "_files_logged", None)
        if logged is None:
            logged = fsm._files_logged = set()
        if fi in logged:
            return
        logged.add(fi)
        new_count = sum(
            1 for e in fsm.manifest.entries
            if e.source_basename == basename and e.op == "write"
        )
        patch_count = sum(
            1 for e in fsm.manifest.entries
            if e.source_basename == basename and e.op == "patch"
        )

        deferred_count = 0
        extractive_rejected = 0
        content_hashes = getattr(fsm, "_file_content_hashes", [])
        if fi < len(content_hashes):
            try:
                from silica.kernel.recall.deferred import get_deferred_store
                bundle = get_deferred_store().get(content_hashes[fi])
                if bundle:
                    deferred_count = len(bundle.get("rejected_ops", []))
                    extractive_rejected = sum(
                        1 for _op in bundle.get("rejected_ops", [])
                        if isinstance(_op, dict)
                        and str(_op.get("rejection_reason", "")).startswith("extractive:")
                    )
            except Exception as _de:
                logger.debug("CLEANUP: deferred count lookup failed (non-fatal): %s", _de)

        # Stash the grounded per-file outcome before any log I/O: this is what
        # the agent-facing tool result reports as actually written. Guarded so
        # a context-less fsm stub can't sink the log append below.
        ctx = getattr(fsm, "context", None)
        if isinstance(ctx, dict):
            entry = {"file": basename, "new": new_count, "patch": patch_count,
                     "deferred": deferred_count}
            if extractive_rejected:
                # W1 (survey-provenance spec): a claim the extractive span
                # gate dropped is declared in the run report, never silently
                # folded into the aggregate "deferred".
                entry["extractive_rejected"] = extractive_rejected
            declared = ctx.get("declared_residue", {}).get(basename)
            if declared:
                # nucleation-forms spec: the declared residue is part of the
                # run report, so a dropped fact is never invisible.
                entry["residue"] = len(declared)
            prior = (ctx.get("renucleated") or {}).get(basename)
            if prior:
                # The source was nucleated before at another version: the
                # report says how many notes of that version stay beside these.
                entry["renucleated_prior_notes"] = prior
            ctx.setdefault("files_summary", []).append(entry)

        event = format_nucleate_event(basename, new_count, patch_count, deferred_count)
        append_log_line(event, fsm.progress.run_id, dedup_key=f"`{basename}`")
    except Exception as exc:
        logger.debug("CLEANUP: log.md append skipped (non-fatal): %s", exc)


def _record_provenance(fsm: "InjectorFSM", fi: int, source_file: str) -> None:
    """Append one `<vault>/provenance.json` record (spec-hermes-coherence §3).

    Sibling projection to _log_nucleate_completion, at the same CLEANUP point:
    reuses fsm._file_content_hashes[fi] — the sha256 already computed once
    per file at RUN start (silica.router.orchestrator.InjectorFSM.run), the
    same value the /nucleate pre-check will later compare against. Recomputing
    it here would fail anyway: by CLEANUP the source file has already been
    archived (moved) out of its original inbox path. `notes` is the
    projection of this run's validated write/patch ops for this source,
    already recorded in fsm.manifest.entries — no new computation. Records
    even when notes is empty: a version change with zero touched notes still
    means every note derived from the prior version is now stale. Best-
    effort and must never block CLEANUP.
    """
    try:
        from silica.kernel.write.provenance import append_record, is_deriving_op

        basename = os.path.basename(source_file)
        content_hashes = getattr(fsm, "_file_content_hashes", [])
        sha256 = content_hashes[fi] if fi < len(content_hashes) else ""
        if not sha256:
            return

        notes = sorted({
            e.path for e in fsm.manifest.entries
            if e.source_basename == basename and is_deriving_op(e.op)
        })

        append_record(basename, sha256, fsm.progress.run_id, notes)
    except Exception as exc:
        logger.debug("CLEANUP: provenance append skipped (non-fatal): %s", exc)


from silica.kernel.recall.paths import SOURCES_MARKER as _SOURCES_MARKER


def _write_source_leaf(fsm: "InjectorFSM", source_file: str) -> None:
    """Verbatim source leaf + `## Sources` links (spec-harness-promotion §2).

    Runs at CLEANUP, before archiving — the one point where the source text
    and the produced notes are both in hand. A leaf is written when the run
    asked for it (`keep_sources`, the /nucleate --keep-sources flag) or when
    it is a conversation capture (`seen_override` set: the source is
    ephemeral, otherwise lost). Independently of who wrote the leaf — this
    run or web_research beside its own inbox note — every note this source
    produced (manifest write/patch entries, same derivation as provenance)
    gains a `## Sources` block linking the leaf. Idempotent on the link, so
    a re-ingest never double-links. Inverses ride fsm._run_inverses, so the
    undo journal covers leaf and block like any other write of the run.
    Best-effort and must never block CLEANUP.
    """
    try:
        from silica.kernel.write import frontmatter
        from silica.kernel.write.contested import append_before_superseded
        from silica.kernel.write.ops import InverseOp, InverseOpKind
        from silica.kernel.recall.paths import SOURCES_DIR
        from silica.kernel.vault_manifest import in_write_dir
        from silica.kernel.write.provenance import (
            attribute_lines,
            footnote_label,
            is_deriving_op,
            source_event_date,
        )

        basename = os.path.basename(source_file)
        leaf_rel = f"{in_write_dir(SOURCES_DIR)}/{basename}"
        leaf_stem = basename.removesuffix(".md")

        try:
            orch.DRIVER.read_note(leaf_rel)
            leaf_exists = True
        except Exception:
            leaf_exists = False

        wants_leaf = getattr(fsm, "keep_sources", False) or bool(
            getattr(fsm, "seen_override", None)
        )
        if wants_leaf and not leaf_exists:
            source_text = orch.DRIVER.read_note(source_file).content or ""
            # Single frontmatter block: keep the source BODY verbatim; the date
            # is the source's event clock, resolved by the same function that
            # stamps it on every claim (kernel/provenance) so leaf and claim
            # can never disagree. An undated source leaves the leaf undated:
            # `date: <today>` here would be read back by _incoming_side as the
            # incoming clock of every future contest this leaf feeds.
            _data, _raw, body = frontmatter.split(source_text)
            date = source_event_date(source_text, getattr(fsm, "seen_override", None))
            # Carry `source_file:` (stamped by convert on every converted inbox
            # note) into the leaf: this block otherwise replaces the source's
            # frontmatter wholesale, and the leaf is the only copy left once
            # the inbox note is archived.
            src_file = _data.get("source_file") if isinstance(_data, dict) else None
            carry = ""
            if src_file:
                quoted = str(src_file).replace("\\", "\\\\").replace('"', '\\"')
                carry = f'source_file: "{quoted}"\n'
            date_line = f"date: {date}\n" if date else ""
            leaf = f"---\n{date_line}source_id: {basename}\n{carry}---\n\n{body.lstrip()}"
            orch.DRIVER.create(leaf_rel, leaf)
            fsm._run_inverses.append(
                (leaf_rel, InverseOp(kind=InverseOpKind.delete_created, path=leaf_rel), None)
            )
            leaf_exists = True

        if not leaf_exists:
            return

        notes = sorted({
            e.path for e in fsm.manifest.entries
            if e.source_basename == basename and is_deriving_op(e.op)
        })
        # Keyed per-claim attribution (OKF §5.1): the leaf body is the only copy
        # of what this source actually said, and here is the one place it sits
        # beside the notes it produced. Lines that are verbatim from it get the
        # leaf's own id as a footnote label, so a note fed by three transcripts
        # says which line came from which instead of just naming all three.
        leaf_body = ""
        try:
            _d, _r, leaf_body = frontmatter.split(orch.DRIVER.read_note(leaf_rel).content or "")
        except Exception:
            leaf_body = ""
        label = footnote_label(basename)

        for rel in notes:
            note_path = f"{rel}.md"  # manifest paths carry no .md
            try:
                prior = orch.DRIVER.read_note(note_path).content or ""
            except Exception:
                continue
            # Idempotency is about the BLOCK, not the link anywhere in the file.
            # The distiller names the source basename as a `related:` /
            # `parent note:` wikilink long before the leaf exists, so a
            # whole-file grep read those notes as "already linked": they never
            # got the block, and reliability_tier — which looks for the marker,
            # not the link — filed them as distilled. 10 of 30 notes on one
            # paper (2026-08-18).
            already = prior.split(_SOURCES_MARKER, 1)[1] if _SOURCES_MARKER in prior else ""
            if f"[[{leaf_stem}]]" in already:
                continue  # this leaf is in the block already — re-ingest no-op
            attributed = attribute_lines(prior, leaf_body, label)
            # The definition rides the Sources block and only when a line was
            # actually marked, so a note never carries a dangling `[^id]`.
            note = f"[^{label}]: [[{leaf_stem}]]\n" if attributed != prior else ""
            # New block, or one more link appended below an existing block.
            # Routed through append_before_superseded: blocks used to land at
            # EOF by construction, which stopped being safe once a note can
            # end with a `## Superseded` section.
            head = f"\n{_SOURCES_MARKER}\n" if _SOURCES_MARKER not in prior else ""
            orch.DRIVER.overwrite(
                note_path,
                append_before_superseded(attributed, f"{head}[[{leaf_stem}]]\n{note}"),
            )
            fsm._run_inverses.append(
                (note_path,
                 InverseOp(kind=InverseOpKind.restore_version, path=note_path,
                           prior_content=prior),
                 None)
            )
    except Exception as exc:
        logger.debug("CLEANUP: source leaf skipped (non-fatal): %s", exc)


def _residue_pool(fsm: "InjectorFSM"):
    pool = getattr(fsm, "_residue_executor", None)
    if pool is None:
        from concurrent.futures import ThreadPoolExecutor
        # 3 workers: one decompose in flight (possibly a warmed next file's
        # too) plus parallel judge batches at the file boundary.
        pool = fsm._residue_executor = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="residue")
    return pool


def txn_touched_paths(txn) -> set[str]:
    """Every path the chunk's transaction recorded a mutation for, any phase.

    Reads Txn.inverses (InverseOp .path / .to_path), tolerating the serialized
    dict form the persisted snapshot fallback carries.
    """
    out: set[str] = set()
    for inv in getattr(txn, "inverses", None) or []:
        for key in ("path", "to_path"):
            val = inv.get(key) if isinstance(inv, dict) else getattr(inv, key, None)
            if val:
                out.add(val)
    return out


def maybe_dispatch_residue_decompose(fsm: "InjectorFSM", fi: int,
                                     inbox_file: str) -> None:
    """PAYLOAD-attach seam: start decomposing the source into atomic facts.

    Decompose is the verification's long pole and depends only on the source
    text, so it rides the file's entire distillation for free; the last
    chunk's WRITE picks the result up (maybe_dispatch_residue_check).
    Best-effort: never raises."""
    try:
        if fsm.context.get(f"file_{fi}_form") == "draft":
            return
        futs = getattr(fsm, "_residue_decompose", None)
        if futs is None:
            futs = fsm._residue_decompose = {}
        if fi in futs:
            return
        from silica.driver import DRIVER
        source = DRIVER.read_note(inbox_file).content or ""
        if not source.strip():
            return
        from silica.kernel import residue as _residue
        futs[fi] = _residue_pool(fsm).submit(_residue.decompose_facts, source)
    except Exception as exc:
        logger.debug("PAYLOAD: residue decompose dispatch skipped (non-fatal): %s", exc)


def maybe_dispatch_residue_check(fsm: "InjectorFSM") -> None:
    """WRITE-time seam: turn the decomposed facts into judge futures.

    On a file's last chunk, once its notes are on disk: gather per-fact
    evidence on the MAIN thread (reads snapshotted, so the background judge
    calls never race autolink/backlink edits of the same notes), then submit
    the judge batches to the pool; _residue_gate assembles the verdicts.
    Mirrors the gate's own refusals (mid-file chunk, draft form). Failed
    files are verified like any other — see _residue_gate.
    When decompose is missing or still running, the gate simply runs
    the whole verification inline. Best-effort: never raises."""
    try:
        fi, ci = fsm._chunk_flat_to_fi_ci.get(
            fsm._current_chunk_idx, (0, fsm._current_chunk_idx))
        group = fsm._file_chunks.get(fi, {})
        if ci + 1 < len(group.get("chunks", [])):
            return
        if fsm.context.get(f"file_{fi}_form") == "draft":
            return
        fut = getattr(fsm, "_residue_decompose", {}).get(fi)
        if fut is None or not fut.done():
            return
        facts = fut.result()
        _empty = {"missing": [], "total": 0, "judged": 0, "failures": 0}
        if facts is None:
            fsm._residue_ready = (fi, {**_empty, "skipped": "decompose failed"})
            return
        if not facts:
            fsm._residue_ready = (fi, _empty)
            return
        from silica.agent.providers import get_embedder_or_none
        from silica.kernel.recall.embed import get_store
        from silica.kernel import residue as _residue
        embedder = get_embedder_or_none(orch.CONFIG, "RESIDUE")
        if embedder is None:
            fsm._residue_ready = (fi, {**_empty, "skipped": "no embedder"})
            return
        from silica.driver import DRIVER
        inbox_file = group.get("source_file", fsm.inbox_file)
        source = DRIVER.read_note(inbox_file).content or ""
        total = len(facts)
        # Same theme filter SALIENCE applies to concepts: off-theme claims are
        # content the pipeline deliberately drops, never "missing" residue.
        facts, vecs, off_theme = _residue.filter_on_theme(
            facts, source, embedder=embedder,
            theme_tau=getattr(orch.CONFIG, "sim_threshold_theme", 0.35))
        fsm.context[f"file_{fi}_residue_meta"] = {
            "total": total, "off_theme": off_theme}
        if not facts:
            fsm._residue_ready = (fi, {**_empty, "total": total,
                                       "off_theme": off_theme})
            return
        evidence = _residue.gather_evidence(
            facts, embedder=embedder, store=get_store(),
            read_body=lambda p: DRIVER.read_note(p).content or "", vecs=vecs)
        if evidence is None:
            fsm._residue_ready = (fi, {**_empty, "total": total,
                                       "off_theme": off_theme,
                                       "skipped": "evidence failed"})
            return
        pool = _residue_pool(fsm)
        B = _residue._JUDGE_BATCH
        futures = [pool.submit(_residue.judge_covered,
                               facts[i:i + B], evidence[i:i + B])
                   for i in range(0, len(facts), B)]
        fsm._residue_future = (fi, facts, futures)
    except Exception as exc:
        logger.debug("WRITE: residue pre-dispatch skipped (non-fatal): %s", exc)


def _verify_now(fsm: "InjectorFSM", fi: int, inbox_file: str) -> dict:
    """The whole verification, synchronously: decompose (reusing a finished
    early dispatch when present) → theme filter → evidence → judge. The slow
    path — callers run it on the residue pool, never on the boundary."""
    from silica.agent.providers import get_embedder_or_none
    from silica.kernel.recall.embed import get_store
    from silica.kernel import residue as _residue
    from silica.driver import DRIVER

    source = DRIVER.read_note(inbox_file).content or ""
    dfut = getattr(fsm, "_residue_decompose", {}).get(fi)
    pre = dfut.result() if dfut is not None and dfut.done() else None
    return _residue.verify_missing(
        source,
        embedder=get_embedder_or_none(orch.CONFIG, "RESIDUE"),
        store=get_store(),
        read_body=lambda p: DRIVER.read_note(p).content or "",
        facts=pre,
        theme_tau=getattr(orch.CONFIG, "sim_threshold_theme", 0.35),
    )


def residue_facts(fsm: "InjectorFSM", fi: int, inbox_file: str) -> list[str]:
    """The file's VERIFIED-missing facts (verification-based residue).

    Consumes the WRITE-time judge futures (or a ready degrade marker) when
    present for this file; otherwise runs the whole verification inline.
    Stashes verification stats in context[file_{fi}_residue_stats] for the
    gate's instrument. Best-effort and fail-open: any failure degrades to []
    (a false "missing" declaration is the disease the 2026-08-16 ROI audit
    killed) so CLEANUP never blocks on it."""
    try:
        ready = getattr(fsm, "_residue_ready", None)
        pending = getattr(fsm, "_residue_future", None)
        if ready is not None and ready[0] == fi:
            fsm._residue_ready = None
            res = ready[1]
        elif pending is not None and pending[0] == fi:
            fsm._residue_future = None
            _fi, facts, futures = pending
            meta = fsm.context.pop(f"file_{fi}_residue_meta", {})
            verdicts = [v for f in futures for v in f.result()]
            res = {
                "missing": [fa for fa, v in zip(facts, verdicts) if v is False],
                "total": meta.get("total", len(facts)),
                "judged": sum(1 for v in verdicts if v is not None),
                "failures": sum(1 for v in verdicts if v is None),
                "off_theme": meta.get("off_theme", 0),
            }
        else:
            res = _verify_now(fsm, fi, inbox_file)
        fsm.context[f"file_{fi}_residue_stats"] = {
            k: v for k, v in res.items() if k != "missing"}
        return res["missing"]
    except Exception as exc:
        logger.debug("CLEANUP: residue verification failed (non-fatal): %s", exc)
        return []


def _residue_gate(
    fsm: "InjectorFSM", fi: int, inbox_file: str, file_has_failure: bool
) -> None:
    """Declare the file's verified-missing facts at its last-chunk CLEANUP.

    Verification-based (kernel/residue.py): declares only facts the narrow
    judge confirmed absent, to the run report, log.md and the deferred store
    (sticky residue_facts) — never silently dropped, and never a re-distill
    round: the round was refuted by the 2026-08-16 ROI audit (it re-added
    already-present facts at 60-170s each; the old open-enumeration check
    read 1.4-4% of the notes it judged). Draft files skip (body intact by
    construction). Files with failed chunks are verified like any other: a
    rolled-back chunk's facts are genuinely absent and the deferred store is
    their ONLY recovery channel — rollback does not re-queue ops, and
    provenance marks the file nucleated so a folder re-run skips it. The old
    silent skip lost exactly that recovery (run 262e6847: the 6 files without
    verification were the 6 with a failed chunk). put_residue_facts keys the
    bundle by content hash, so a deliberate full re-nucleate stays idempotent.
    """
    if file_has_failure:
        logger.info(
            "CLEANUP: %s had failed chunk(s) — residue verification proceeds "
            "(missing facts route to the deferred store).",
            os.path.basename(inbox_file),
        )
    if fsm.context.get(f"file_{fi}_form") == "draft":
        return
    # The round is gone, so nothing forces this gate to wait for the judge:
    # anything not already resolved is parked and declared later (next flush
    # or run end) — report, log and bundle are order-insensitive in-run.
    pending = getattr(fsm, "_residue_future", None)
    ready = getattr(fsm, "_residue_ready", None)
    lst = getattr(fsm, "_residue_pending", None)
    if lst is None:
        lst = fsm._residue_pending = []
    if (pending is not None and pending[0] == fi
            and not all(f.done() for f in pending[2])):
        # Judge futures still in flight: park them.
        fsm._residue_future = None
        meta = fsm.context.pop(f"file_{fi}_residue_meta", {})
        lst.append(("verdicts", fi, inbox_file, pending[1], pending[2], meta))
        logger.info(
            "CLEANUP: residue verification for %s still running — "
            "declaration deferred to the next flush.",
            os.path.basename(inbox_file),
        )
        return
    if (ready is not None and ready[0] == fi) or (
            pending is not None and pending[0] == fi):
        # Resolved (degrade marker or finished futures): declare now, cheap.
        from time import monotonic as _mono
        _t0 = _mono()
        facts = residue_facts(fsm, fi, inbox_file)
        stats = fsm.context.pop(f"file_{fi}_residue_stats", {})
        _declare_residue(fsm, fi, inbox_file, facts, stats,
                         round(_mono() - _t0, 2))
        return
    # No dispatch ran (decompose missing or still running at WRITE): the
    # whole verification goes to the pool — this fallback used to run inline
    # and block the boundary for its full duration (355s observed, run
    # d8b4d4c1). Pool-thread note reads accept bounded staleness from
    # concurrent autolink edits: link markup, not fact content.
    lst.append(("result", fi, inbox_file,
                _residue_pool(fsm).submit(_verify_now, fsm, fi, inbox_file)))
    logger.info(
        "CLEANUP: residue verification for %s dispatched late — "
        "declaration deferred to the next flush.",
        os.path.basename(inbox_file),
    )


def _declare_residue(fsm: "InjectorFSM", fi: int, inbox_file: str,
                     facts: list[str], stats: dict, secs: float) -> None:
    """Record the instrument and, when facts are missing, declare them:
    run report (capped at 12), log.md line, deferred-store bundle (full,
    sticky residue_facts field)."""
    try:
        fsm.progress.inputs.setdefault("residue_secs", {})[f"f{fi}"] = secs
        # Verification instrument, uncensored (the old cap right-censored the
        # ROI metric): the full missing list plus total/judged/failures.
        fsm.progress.inputs.setdefault("residue", {})[f"f{fi}"] = {
            "missing": facts, **stats}
    except Exception:
        pass
    if not facts:
        return
    basename = os.path.basename(inbox_file)
    # Report/log stay bounded; the deferred bundle carries the full list.
    fsm.context.setdefault("declared_residue", {})[basename] = facts[:12]
    logger.warning(
        "CLEANUP: %d verified-missing fact(s) in %s declared.",
        len(facts), basename,
    )
    try:
        from silica.kernel.recall.deferred import get_deferred_store
        hashes = getattr(fsm, "_file_content_hashes", []) or []
        content_hash = (hashes[fi] if fi < len(hashes) else
                        getattr(fsm, "_current_content_hash", ""))
        if content_hash:
            get_deferred_store().put_residue_facts(
                content_hash, inbox_file, fsm.target_dir, fsm.hub, facts)
    except Exception as exc:
        logger.debug("CLEANUP: residue defer failed (non-fatal): %s", exc)
    try:
        from silica.kernel.recall.run_log import append_log_line

        append_log_line(
            f"residue `{basename}` → {len(facts)} fact(s) declared uncovered",
            fsm.progress.run_id, dedup_key=f"residue `{basename}`",
        )
    except Exception as exc:
        logger.debug("CLEANUP: residue log line failed (non-fatal): %s", exc)


def flush_residue_pending(fsm: "InjectorFSM", wait: bool) -> None:
    """Declare parked verifications whose judge futures completed.

    wait=True (run end) resolves every parked entry, bounded by the real
    judge call durations; wait=False declares only what already finished.
    A failed future degrades that file to a no-declaration record, never to
    a false "missing" (same fail-open contract as the whole lane)."""
    lst = getattr(fsm, "_residue_pending", None)
    if not lst:
        return
    from time import monotonic as _mono
    keep: list = []
    declared = 0
    for entry in lst:
        kind = entry[0]
        futures = entry[4] if kind == "verdicts" else [entry[3]]
        if not wait and not all(f.done() for f in futures):
            keep.append(entry)
            continue
        _t0 = _mono()
        if kind == "result":
            _k, fi, inbox_file, fut = entry
            try:
                res = fut.result()
            except Exception as exc:
                logger.warning("residue: deferred verification for f%d failed (%s)",
                               fi, exc)
                res = {"missing": [], "total": 0, "judged": 0, "failures": 0,
                       "off_theme": 0, "skipped": "verification failed"}
            stats = {k: v for k, v in res.items() if k != "missing"}
            _declare_residue(fsm, fi, inbox_file, res["missing"], stats,
                             round(_mono() - _t0, 2))
            declared += 1
            continue
        _k, fi, inbox_file, facts, futures, meta = entry
        try:
            verdicts = [v for f in futures for v in f.result()]
        except Exception as exc:
            logger.warning("residue: deferred verification for f%d failed (%s)",
                           fi, exc)
            _declare_residue(fsm, fi, inbox_file, [], {
                "total": meta.get("total", len(facts)), "judged": 0,
                "failures": len(facts),
                "off_theme": meta.get("off_theme", 0),
            }, round(_mono() - _t0, 2))
            declared += 1
            continue
        missing = [fa for fa, v in zip(facts, verdicts) if v is False]
        stats = {
            "total": meta.get("total", len(facts)),
            "judged": sum(1 for v in verdicts if v is not None),
            "failures": sum(1 for v in verdicts if v is None),
            "off_theme": meta.get("off_theme", 0),
        }
        _declare_residue(fsm, fi, inbox_file, missing, stats,
                         round(_mono() - _t0, 2))
        declared += 1
    fsm._residue_pending = keep
    if declared:
        # A flush can run past the last progress note of the run (the DONE
        # hooks): without an explicit save the instrument recorded above
        # would never reach the ledger on disk (lost for f3, run d8b4d4c1).
        try:
            fsm.progress.save()
        except Exception as exc:
            logger.debug("residue flush save failed (non-fatal): %s", exc)


def handle_cleanup(fsm: "InjectorFSM") -> None:
    from silica.tools.wrapped import silica_cleanup

    fsm._get_chunks_from_context_if_empty()
    fi, ci = fsm._chunk_flat_to_fi_ci.get(fsm._current_chunk_idx, (0, fsm._current_chunk_idx))
    with orch.phase(fsm, fsm._chunk_task_id("cleanup"), "cleanup"):
        # Always write ledger for this chunk's ops (per chunk)
        fsm._write_ledger_for_file(fi, "committed")

        # Archive the physical file only on the last chunk of its file group
        file_group = fsm._file_chunks.get(fi, {})
        n_chunks_in_file = len(file_group.get("chunks", []))
        is_last_chunk_of_file = (ci + 1 >= n_chunks_in_file)

        fi_prefix = f"f{fi}_"
        file_has_failure = any(
            t.status == "failed" for t in fsm.progress.tasks
            if t.id.startswith(fi_prefix)
        )
        inbox_file_for_fi = file_group.get("source_file", fsm.inbox_file)

        # Residue declaration (nucleation-forms spec, verification-based since
        # 2026-08-17): hooked HERE, not post-validate, because the all-skip
        # short-circuit jumps straight to CLEANUP and this is the one seam
        # both paths share. Declares verified-missing facts (report + log +
        # deferred store); archiving proceeds in the same pass — there is no
        # residue round anymore.
        if is_last_chunk_of_file:
            _residue_gate(fsm, fi, inbox_file_for_fi, file_has_failure)

        if is_last_chunk_of_file:
            # Log the per-file outcome regardless of failures: a file that
            # committed notes in earlier chunks still deserves its log.md line
            # and files_summary entry. Only ARCHIVING is gated on no-failure
            # (a partial source stays in the inbox for retry).
            _log_nucleate_completion(fsm, fi, inbox_file_for_fi)
            if not file_has_failure:
                # Leaf before archive: the source text must still be readable
                # at its inbox path. On failure the source stays in the inbox
                # for retry, so nothing is lost by skipping the leaf here.
                _write_source_leaf(fsm, inbox_file_for_fi)
                res = silica_cleanup(inbox_file_for_fi)
                if "error" in res:
                    fsm.context["cleanup_warning"] = res["error"]
                # Title-index run cache: the archived source moved out of its
                # indexed path — drop its ref so AUTOLINK can't link to a stale
                # inbox title.
                _cached_refs = getattr(fsm, "_run_title_refs", None)
                if _cached_refs is not None:
                    _src_abs = os.path.abspath(inbox_file_for_fi)
                    _cached_refs[:] = [
                        r for r in _cached_refs
                        if not getattr(r, "path", None) or os.path.abspath(r.path) != _src_abs
                    ]
            else:
                logger.info(
                    "File %d (%s) had chunk failures — not archiving.",
                    fi, file_group.get("source_file", "?"),
                )
            # Provenance covers whatever DID commit: a partial file's validated
            # write/patch ops are real derived notes, and both session attribution
            # (eval session_recall) and re-ingest idempotence (note_authored_by)
            # must see them even while the source stays in inbox for retry.
            _record_provenance(fsm, fi, inbox_file_for_fi)
        else:
            logger.info(
                "Chunk f%d_c%d done. Archiving deferred until last chunk of file %d.",
                fi, ci, fi,
            )

        # Grounded outcome evidence: committed-chunk count lets the terminal
        # verdict (here and in ROLLBACK) tell "partial" from "failed".
        fsm.context["committed_chunks"] = fsm.context.get("committed_chunks", 0) + 1

        # Run-level verdict, recomputed each chunk's CLEANUP (last write wins).
        # no_ops is a whole-run property — it holds only when NO chunk had actionable
        # ops. A later chunk that had ops lifts a prior all-skip chunk's provisional
        # no_ops rather than staying stuck on it, in either order (A24).
        if fsm.context.get("has_partial_failure"):
            fsm.context["final_status"] = "partial"
        elif fsm.context.get("run_had_ops"):
            fsm.context["final_status"] = "Success"
        else:
            fsm.context["final_status"] = "no_ops"

        # Persist this run's inverses for /revert, with final content hash.
        if fsm._undo_run_id and fsm._run_inverses:
            import hashlib
            from silica.kernel.write.ops import InverseOpKind
            from silica.kernel.write.undo_journal import get_undo_journal
            journal = get_undo_journal()
            for path, inv, _ in fsm._run_inverses:
                try:
                    post = orch.DRIVER.read_note(path).content
                    post_hash = hashlib.sha256((post or "").encode("utf-8")).hexdigest()
                except Exception:
                    post_hash = None
                    if inv.kind != InverseOpKind.recreate_deleted:
                        # Note should exist after this write; without its hash the
                        # /revert "modified since inject" guard can't protect it.
                        logger.warning(
                            "finalize: could not hash %s post-write; /revert guard "
                            "disabled for it", path)
                journal.record(fsm._undo_run_id, inv, post_hash)
            fsm._run_inverses.clear()


def handle_rollback(fsm: "InjectorFSM") -> None:
    fsm._progress_note("rollback", "rollback", "running")
    snapshot_res = fsm._chunk_ctx.get("snapshot", {})
    # fsm._txn.inverses is the single source of truth for rollback (C3 /
    # ADR-0009): SNAPSHOT seeds it and every phase that mutates a pre-existing
    # note appends to it. Fall back to the persisted snapshot dict only when
    # no live transaction exists (defensive — both share the per-chunk lifetime).
    if fsm._txn is not None:
        txn_id = fsm._txn.id
        inverses = fsm._txn.inverses_serialized
    else:
        txn_id = snapshot_res.get("txn_id")
        inverses = snapshot_res.get("inverses", [])

    if txn_id and inverses:
        from silica.tools.wrapped import silica_restore
        try:
            res = silica_restore(txn_id=txn_id, inverses=inverses)
            if not res.get("success", False):
                err_msg = "; ".join(res.get("errors", []))
                logger.error("Rollback partially failed: %s", err_msg)
                fsm.context["rollback_error"] = err_msg
            else:
                logger.info("Rollback complete for txn %s", txn_id)
        except Exception as e:
            logger.error("Rollback failed: %s", e)
            fsm.context["rollback_error"] = str(e)
        fsm._write_ledger_rollback(txn_id)

    # Clean up the embedding index for notes that were created and then rolled back
    # to prevent stale phantom entries that would bias future candidate searches.
    created_paths: list[str] = []
    if fsm._txn is not None and fsm._txn.created_paths:
        created_paths = list(fsm._txn.created_paths)
    elif snapshot_res.get("created_paths"):
        created_paths = list(snapshot_res["created_paths"])
    if created_paths:
        try:
            from silica.kernel.recall.embed import get_store
            store = get_store()
            for cp in created_paths:
                store.delete(cp.removesuffix(".md"))
            store.save()
        except Exception as _ee:
            logger.debug("ROLLBACK: embed index cleanup failed (non-fatal): %s", _ee)

    fsm._progress_note("rollback", "rollback", "done")
    # Contain the failure at chunk level instead of aborting the whole run
    fsm._contain_chunk_failure()
