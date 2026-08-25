# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Static adapter registry — ADR-0014.

A list, not a plugin system (the ADR's scope line): N sources = N entries,
edited here. Dispatch is first-match over matches(); `enabled` (from the
vault manifest) filters which adapters participate for the current vault.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from silica.sources.base import SourceAdapter
from silica.sources.code import CODE
from silica.sources.notebook import NOTEBOOK
from silica.sources.prose import PROSE

logger = logging.getLogger(__name__)

ALL_ADAPTERS: tuple[SourceAdapter, ...] = (PROSE, CODE, NOTEBOOK)


def enabled_adapters(enabled: Sequence[str] | None = None) -> list[SourceAdapter]:
    """Adapters participating in dispatch. enabled=None → all registered."""
    if enabled is None:
        return list(ALL_ADAPTERS)
    known = {a.name for a in ALL_ADAPTERS}
    for name in enabled:
        if name not in known:
            logger.warning("vault manifest lists unknown source %r — ignored", name)
    return [a for a in ALL_ADAPTERS if a.name in enabled]


def supported_nucleate_extensions() -> list[str]:
    """File extensions every nucleate lane accepts — the GUI picker's `accept` set.

    Union of the prose/notebook/converter lanes and all tree-sitter code
    languages. Lives here (the dispatch hub), not per-adapter: CODE matches by
    language, not an extension list, so there's nothing to enumerate on the
    adapter.
    """
    from silica.kernel.code.codeast import BARE_LANGUAGES, EXTENSION_MAP
    from silica.sources.convert import DOC_EXTS
    from silica.sources.prose import _EXTS as PROSE_EXTS

    # bare languages (toml/html/css) are graph-only presence: no nucleate lane
    code_exts = (e for e, lang in EXTENSION_MAP.items() if lang not in BARE_LANGUAGES)
    return sorted({".ipynb", *DOC_EXTS, *PROSE_EXTS, *code_exts})


def folder_rel(target: str) -> str | None:
    """`target` as a repo-relative folder path, or None when it is not one.

    "" means the repo root. Split out of `expand_folder` because callers also
    need the folder itself, not just its contents: it is what a code note's
    destination folder is named after, and re-deriving it from the expanded
    files would land on their deepest common parent, not the folder asked for.
    """
    from pathlib import Path

    from silica.config import CONFIG
    from silica.kernel.recall.paths import repo_root_for

    root = repo_root_for(CONFIG.vault_path) if CONFIG.vault_path else None
    if root is None:
        return None
    p = Path(target)
    try:
        rel = (p if p.is_absolute() else root / p).resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return None  # escapes the repo
    if not (root / rel).is_dir():
        return None
    return "" if str(rel) == "." else rel.as_posix()


def expand_folder(target: str, enabled: Sequence[str] | None = None) -> list[str]:
    """Repo-relative ingestible files under a folder argument, sorted.

    `supported_files` is the repo's own git-backed census, so ignored and
    vendored trees never enter and no walk is hand-rolled here. Returns [] when
    `target` names no directory inside the code-lane repo — a folder of vault
    notes, a typo, or a non-code-lane vault all land there, and the caller keeps
    whatever it already did for an unresolved argument.
    """
    from silica.config import CONFIG
    from silica.kernel.code.codegraph import supported_files
    from silica.kernel.recall.paths import repo_root_for

    rel = folder_rel(target)
    if rel is None:
        return []
    root = repo_root_for(CONFIG.vault_path)
    if root is None:  # not a git checkout ⇒ the census has nothing to walk
        return []
    prefix = f"{rel}/" if rel else ""
    return [
        f for f in supported_files(root)
        if f.startswith(prefix) and adapter_for(f, enabled) is not None
    ]


def adapter_for(target: str, enabled: Sequence[str] | None = None) -> SourceAdapter | None:
    for adapter in enabled_adapters(enabled):
        if adapter.matches(target):
            return adapter
    return None


def _record_inverse(run_id: str, path: str, prior: str | None) -> None:
    """Journal one terminal-lane write so `/revert` can undo it.

    Mirrors the FSM's C3 strategy (silica/tools/wrapped.py): a note that did not
    exist is undone by deleting it, one that did by restoring the body read just
    before the write. The post-hash comes from reading the note back, not from
    the stub text, so it matches what `revert_run`'s modified-since guard
    recomputes after any normalisation the backend applied on the way in.
    """
    from silica.driver import DRIVER
    from silica.kernel.write.ops import InverseOp, InverseOpKind
    from silica.kernel.write.undo_journal import _content_hash, get_undo_journal

    inv = (
        InverseOp(kind=InverseOpKind.restore_version, path=path, prior_content=prior)
        if prior is not None
        else InverseOp(kind=InverseOpKind.delete_created, path=path)
    )
    try:
        post_hash = _content_hash(DRIVER.read_note(path).content)
    except Exception:
        # Without the hash /revert loses its "modified since" guard but still
        # undoes the write — the journal entry is worth more than the guard.
        post_hash = None
        logger.warning("could not hash %s post-write; /revert guard disabled for it", path)
    get_undo_journal().record(run_id, inv, post_hash)


def stage(
    adapter: SourceAdapter, target: str, run_root: str = "", run_id: str | None = None
) -> dict:
    """read → to_stub → write, for terminal-lane stubs; status dict out.

    Distill-lane stubs are NOT written here — the Injector FSM owns that
    lane (ADR-0013); the caller forwards the target to the agent instead.

    `run_root` is the folder argument this target was expanded from, when there
    was one. It rides in `meta` rather than the Protocol signature: only the code
    lane names its destination after it, and one more adapter method for one
    adapter is the bloat ADR-0014 exists to refuse.

    `run_id` opts this write into the caller's undo run. The terminal lane skips
    the FSM by design (ADR-0013), and the FSM is where journalling lived — so
    until a caller opens a run, these were the one writes `/revert` could not
    see. The run is the caller's because it spans the whole batch, not one file.
    """
    from silica.driver import DRIVER

    try:
        item = adapter.read(target)
        if run_root:
            item.meta["nucleate_root"] = run_root
        stub = adapter.to_stub(item)
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    if stub.lane != "terminal":
        return {"status": "distill", "target": target}
    prior: str | None = None
    if run_id:
        try:  # read before the write: refreshing a stub must restore, not delete
            prior = DRIVER.read_note(stub.note_path).content
        except Exception:
            prior = None
    # This write lands in the vault but NOT in the indexes. The distill lane
    # crosses `_refresh_cooccurrence_for_ops` (router/states/write.py:302); the
    # terminal lane calls the driver bare, and `sync.sweep()` only reconciles
    # stores that already exist ("cold indexes stay cold", recall/sync.py:28).
    # ponytail: no hook here because nothing had ever nucleated code — measured
    # 2026-08-25, silica/kernel/recall held 21 code files and 0 notes, so the
    # missing hop cost nothing yet. It starts costing on the first batch:
    # nucleate one folder of silica/, and if those notes do not answer
    # silica_semantic_search without a manual embed refresh, route this write
    # through the same refresh seam the FSM uses. Re-check by 2026-11-25, with
    # the two pieces measured beside it: `silica_files` returns two uncorrelated
    # counts (notes, code) while report/graph_report/code_signals._coverage_from
    # already computes real coverage, and codeast has NO guard for the
    # tree-sitter SIGSEGV of 2026-08-19 — its `except Exception` cannot catch a
    # native crash, and a batch over silica/ is what would trip it.
    DRIVER.upsert(stub.note_path, stub.body)  # re-ingesting the same target refreshes the stub
    if run_id:
        _record_inverse(run_id, stub.note_path, prior)
    return {"status": "ok", "note_path": stub.note_path, "meta": dict(item.meta)}
