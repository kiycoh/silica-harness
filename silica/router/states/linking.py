# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Injector linking states: AUTOLINK, BACKLINK.

Handler bodies for InjectorFSM, extracted from orchestrator.py: each function
takes the FSM instance and mutates its context/state exactly as the former
method did. Patchable collaborators (DRIVER, CONFIG, tools, load_ops, time)
are resolved through the orchestrator module namespace (orch.X) so tests that
patch silica.router.orchestrator.* keep working.
"""
from __future__ import annotations

import logging
import os
import typing
from typing import TYPE_CHECKING

from silica.router import orchestrator as orch

if TYPE_CHECKING:
    from silica.router.orchestrator import InjectorFSM

logger = logging.getLogger(__name__)


from silica.kernel.write.ops import OpType


def _run_title_refs(fsm: "InjectorFSM") -> list[typing.Any]:
    """One full-vault scan per run; WRITE appends this run's new notes.

    The index itself (build_title_index) is recomputed from these refs per
    use — it's pure CPU over ~10k strings (ms), while list_files() is the
    per-chunk disk scan this cache removes. ponytail: no invalidation beyond
    append/remove — the ingest path never renames vault notes mid-run.
    """
    if fsm._run_title_refs is None:
        fsm._run_title_refs = list(orch.DRIVER.list_files())
    return fsm._run_title_refs


def _backlink_neighbourhood(new_titles: list[str], touched_abs: set[str]) -> list[str]:
    """Pre-existing notes whose body mentions any of `new_titles`.

    NOT a mention index (deleted 2026-08-23, it had no reader left): this phase
    used to ask `DRIVER.mentions_of`, whose postings were keyed by the titles in
    the trie when each body was last indexed, while `_patch_index` re-scans only
    the note just written. A title coined moments ago therefore had no postings
    against pre-existing bodies, the neighbourhood came out empty, and the phase
    was inert for exactly the case it exists for. A per-write index can never
    answer for a title that did not exist when the bodies were indexed; the
    sweep below is the primitive that can.

    `search_context_batch` is one vault sweep for every new title, over bodies
    the driver already caches. It is a substring match where the mention index
    is word-boundary anchored, i.e. looser — harmless, because this is only the
    prefilter: `backlink_pass` runs the real `autolink()` on each candidate, and
    that decides what gets wrapped. Source leaves are search-invisible, so they
    stay out of the neighbourhood as before.

    ponytail: one cached-body sweep per chunk. If BACKLINK ever shows up as slow
    on a large vault, cache the lowercased bodies alongside the body cache.
    """
    if not new_titles:
        return []
    try:
        hits_by_title = orch.DRIVER.search_context_batch(list(new_titles))
    except Exception as e:
        logger.debug("BACKLINK: neighbourhood sweep failed (phase abstains): %s", e)
        return []

    neighbourhood: list[str] = []
    seen_norm: set[str] = set()
    for title in new_titles:
        for hit in hits_by_title.get(title, []):
            path = hit.ref.path
            norm = os.path.abspath(path)
            if norm in seen_norm or norm in touched_abs:
                continue
            seen_norm.add(norm)
            neighbourhood.append(path)
    return neighbourhood


def _relevance_candidates(title_index: list[str]):
    """path -> the titles worth linking from it, by note-vector cosine.

    Returns a callable so the store, the threshold and the title lookup are
    resolved once per chunk rather than once per note. Falls back to the whole
    index — the historical behaviour — whenever the gate cannot answer: threshold
    off, no embed index, or this note has no stored vector yet (a note written
    moments ago). Never narrows on a guess.
    """
    from silica.config import CONFIG
    from silica.kernel.recall.relatedness import neighbours_above

    floor = float(getattr(CONFIG, "autolink_min_sim", 0.0) or 0.0)
    known = set(title_index)

    def candidates(path: str) -> list[str]:
        near = neighbours_above(path, floor)
        if near is None:
            return title_index          # gate unavailable — never narrow on a guess
        # `[]` is a real answer (nothing close enough), so it is passed through:
        # falling back to the full index there would restore the very noise the
        # gate exists to remove.
        return [n for n in near if n in known]

    return candidates


def handle_autolink(fsm: "InjectorFSM") -> None:
    """Best-effort wikilink injection into touched notes (Phase 4).

    Runs autolink on every note written by this chunk.  Failures are
    non-fatal: they are logged and the FSM continues to LINT.  This is
    intentional — autolink only ADDs links; it can never break a valid note.
    """
    fsm._progress_note(fsm._chunk_task_id("autolink"), "autolink", "running")

    try:
        from silica.kernel.link.autolink import build_title_index

        ops = orch.load_ops(fsm._chunk_ctx["ops_path"])
        touched_paths = [
            ref
            for op in ops
            if (ref := op.touched_ref()) and op.op not in (OpType.delete, OpType.skip)
        ]

        if not touched_paths:
            fsm._progress_note(fsm._chunk_task_id("autolink"), "autolink", "done")
            fsm._transition_success()
            return

        all_refs = _run_title_refs(fsm)
        title_index = build_title_index(all_refs)

        # Candidates come from embedding relevance, not from graph structure.
        # The same-cluster narrowing that used to sit here had never run once —
        # `vault_graph_ctx` keys carry `.md` and the lookup stripped it, so every
        # note read as cluster -1 and fell through to the full index (0 hits in
        # 717). Fixing the keys made it engage, and the first measurement said
        # don't: Louvain clusters this vault by structure, not topic, so the ML
        # concepts one lecture wants are spread over three clusters while its hub
        # sits in one — narrowing there dropped 7 good links to drop 1 bad one.
        _relevant = _relevance_candidates(title_index)

        total_added = 0
        for path in touched_paths:
            try:
                added = orch.DRIVER.autolink_note(
                    path,
                    candidates=_relevant(path),
                    title_index=title_index,
                )
                if added:
                    total_added += len(added)
                    logger.info("AUTOLINK: %s — added %d link(s): %s", path, len(added), added)
            except Exception as _ae:
                logger.debug("AUTOLINK: skipped '%s' (non-fatal): %s", path, _ae)

        logger.info("AUTOLINK: finished — %d link(s) added across %d note(s)", total_added, len(touched_paths))
        fsm.context["yield_links"] = fsm.context.get("yield_links", 0) + total_added
    except Exception as e:
        # AUTOLINK is best-effort: log and continue to LINT
        logger.warning("AUTOLINK: phase failed (non-fatal): %s", e)

    fsm._progress_note(fsm._chunk_task_id("autolink"), "autolink", "done")
    fsm._transition_success()


def handle_backlink(fsm: "InjectorFSM") -> None:
    """Best-effort reverse link injection into pre-existing neighbouring notes (Phase 4.5).

    For each newly-written note (write ops, excluding the hub auto-creation),
    scans pre-existing notes that textually mention the new title and wraps
    those mentions as wikilinks.  Extends snapshot_domain and registers
    rollback inverses for any modified note so ROLLBACK and LINT graph-diff
    both cover the backlinks.
    """
    fsm._progress_note(fsm._chunk_task_id("backlink"), "backlink", "running")

    try:
        from silica.kernel.link.autolink import backlink_pass, build_title_index

        ops = orch.load_ops(fsm._chunk_ctx["ops_path"])

        hub_name_lower = (fsm.hub or "").strip("[]").lower()
        new_titles: list[str] = []
        for op in ops:
            if op.op != OpType.write:
                continue
            path = op.touched_ref()
            if not path:
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem.lower() != hub_name_lower:
                new_titles.append(stem)

        if not new_titles:
            fsm._progress_note(fsm._chunk_task_id("backlink"), "backlink", "done")
            fsm._transition_success()
            return

        touched_paths_abs: set[str] = {
            os.path.abspath(p)
            for op in ops
            for p in (op.touched_ref(),)
            if p is not None
        }

        neighbourhood = _backlink_neighbourhood(new_titles, touched_paths_abs)

        if not neighbourhood:
            fsm._progress_note(fsm._chunk_task_id("backlink"), "backlink", "done")
            fsm._transition_success()
            return

        # Pre-read prior content before backlink_pass writes, for rollback inverses
        prior_contents: dict[str, str] = {}
        for path in neighbourhood:
            try:
                prior_contents[path] = orch.DRIVER.read_note(path).content or ""
            except Exception:
                pass

        all_refs = _run_title_refs(fsm)
        title_index = build_title_index(all_refs)
        added_map = backlink_pass(new_titles, title_index=title_index, neighbourhood=neighbourhood)

        if added_map and fsm._txn is not None:
            from silica.kernel.write.ops import InverseOp, InverseOpKind
            existing_snapshot_paths = {d["path"] for d in fsm._chunk_ctx.get("snapshot_domain", [])}
            for path_modified in added_map:
                if path_modified not in existing_snapshot_paths:
                    stem = os.path.splitext(os.path.basename(path_modified))[0]
                    fsm._chunk_ctx.setdefault("snapshot_domain", []).append(
                        {"name": stem, "path": path_modified}
                    )
                    existing_snapshot_paths.add(path_modified)
                if path_modified in prior_contents:
                    inverse = InverseOp(
                        kind=InverseOpKind.restore_version,
                        path=path_modified,
                        prior_content=prior_contents[path_modified],
                    )
                    fsm._txn.inverses.append(inverse)

        total_links = sum(len(v) for v in added_map.values())
        logger.info(
            "BACKLINK: %d link(s) added to %d pre-existing note(s)", total_links, len(added_map)
        )
        fsm.context["yield_links"] = fsm.context.get("yield_links", 0) + total_links
    except Exception as e:
        logger.warning("BACKLINK: phase failed (non-fatal): %s", e)

    fsm._progress_note(fsm._chunk_task_id("backlink"), "backlink", "done")
    fsm._transition_success()
