# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Obsidian FS Backend — L0 implementation via direct filesystem access.

From SILICA.md §3 L0:
  Headless fallback and oracle for non-regression testing. Directly reads the
  filesystem and builds an in-memory graph index.

Note:
  This backend is independent of the Obsidian app, making it suitable for CI
  and headless cron jobs. It manages its own graph index which is refreshed
  as needed.
"""
from __future__ import annotations

import functools
import logging
import os
import threading
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any
import networkx as nx
from silica.kernel.link.ast import ADR_REF_RE, NON_MD_EXTENSIONS, extract_refs_typed, resolve_relative

from silica.driver.base import (
    GraphIndexMixin,
    GraphSnapshot,
    Heading,
    Hit,
    Link,
    NoteContent,
    NoteRef,
    Txn,
)
from silica.kernel.write import frontmatter as fm
from silica.kernel.write import session_changes
from silica.kernel.link import ofm
from silica.kernel.recall.graph_export import is_vault_artifact
from silica.kernel.recall.paths import (
    atomic_write_bytes,
    contain_in_vault,
    ignore_matcher,
    is_source_leaf,
)
from silica.kernel.write.notetype import stamp_type
logger = logging.getLogger(__name__)

# Per-note Hit ceiling for content search. The tool layer renders at most 3
# snippets per note and ranks notes by hit count; 20 saturates that ranking
# while bounding a pathological query at 20·N Hits instead of lines·N.
_MAX_HITS_PER_NOTE = 20

# Body-cache LRU bound, in notes. Sized so every realistic vault fits whole
# (zero evictions, zero cost: measured neutral at 1.2k and 10k with cap
# above N) while a pathological one is bounded at ~100-200 MB instead of
# unbounded. Below N the LRU pays the sequential-scan pathology (measured
# +45% on a 10k double scan at cap 4096), which is the deliberate trade:
# the cap exists against OOM, not for speed. The bench
# (scripts/bench_scale_levers.py) reads and flips this seam.
_BODY_CACHE_CAP = 16384

# Debounce on the out-of-band roster re-check (_roster_drifted). `_ensure_index`
# runs on every read op and several land in one agent turn, so the whole-tree
# directory walk needs the same brake the index sweep uses for the same reason
# (kernel/recall/sync.py `_MIN_INTERVAL`): a roster that just rescanned has
# nothing new to see. Measured 2026-08-22 on a 709-note / 111-folder vault:
# the scan is 1.1 ms against the 1086 ms rebuild it decides whether to run,
# so the brake is against call frequency, not against the walk. Reopen if a
# scan ever passes ~200 ms (roughly 20k folders).
_ROSTER_RECHECK_INTERVAL = 2.0

# Buffered embed-vector deletes per npz save. A move/delete sweep on a big
# vault was paying one whole-index serialization per note; buffering trades
# that for a bounded staleness window: another process sees the phantom
# vector until the threshold or exit flush lands. In-process reads stay
# exact (the store's memory is updated per op).
_EMBED_FLUSH_EVERY = 32


# Deferred embed-delete flush state (see _drop_embed_vector).
_embed_deletes_pending = 0
_embed_flush_registered = False


def _flush_pending_embed_deletes() -> None:
    """Persist buffered embed deletes; safe to call with nothing pending."""
    global _embed_deletes_pending
    if _embed_deletes_pending <= 0:
        return
    try:
        from silica.kernel.recall.embed import get_store
        get_store().save()
        _embed_deletes_pending = 0
    except Exception as exc:
        logger.debug("deferred embed flush failed (non-fatal): %s", exc)


def _register_embed_flush() -> None:
    global _embed_flush_registered
    if not _embed_flush_registered:
        import atexit
        atexit.register(_flush_pending_embed_deletes)
        _embed_flush_registered = True


def _locked(method):
    """Serialize this method against index mutation.

    The MCP server dispatches every request on its own thread, so a write's
    `_patch_index` can interleave with a search iterating `self._notes` or
    `self._graph` — "dictionary changed size during iteration" on a normal
    concurrent session. One reentrant lock over mutators and the readers that
    iterate; reentrant because readers call `_ensure_index` which may rebuild.
    # ponytail: one global lock, not per-structure — split it only if MCP
    # profiling ever shows read contention.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._index_lock:
            return method(self, *args, **kwargs)
    return wrapper


class ObsidianFSBackend(GraphIndexMixin):
    """ObsidianDriver implementation using direct filesystem access."""

    def __init__(self, vault_path: str):
        if not vault_path:
            raise ValueError("FS backend requires a valid vault_path")
        self.vault_path = Path(vault_path).resolve()
        self._index_lock = threading.RLock()

        # In-memory index
        self._notes: dict[str, NoteRef] = {}          # path -> NoteRef
        self._notes_by_name: dict[str, list[NoteRef]] = {}  # lower_name -> list of NoteRefs
        self._graph = nx.DiGraph()
        self._unresolved_links: set[tuple[str, str]] = set() # (source_path, raw_target)
        self._alias_pairs: dict[str, list[str]] = {}         # path -> frontmatter aliases
        # Note↔file reference maps. Two keyspaces, never merged: embeds are
        # vault-relative (resolved against the census below), `documents:`
        # entries are repo-relative (validated at write by codedocs). Kept out
        # of self._graph on purpose — orphans/snapshot/graph-gate semantics
        # are note↔note and must not shift because a note gained a figure.
        self._assets_lower: dict[str, str] = {}              # rel_path.lower() -> rel_path
        self._assets_by_name: dict[str, list[str]] = {}      # basename.lower() -> [rel_path]
        self._asset_edges: dict[str, list[str]] = {}         # note path -> resolved assets
        self._asset_notes: dict[str, list[str]] = {}         # asset path -> [note path]
        self._unresolved_assets: dict[str, list[str]] = {}   # note path -> raw targets
        self._documents_edges: dict[str, list[str]] = {}     # note path -> documents: paths
        self._documents_notes: dict[str, list[str]] = {}     # documents: path -> [note path]
        self._needs_reindex: bool = True
        self._dirty_paths: set[str] = set()           # paths patched since last full rebuild
        # LRU-bounded at _BODY_CACHE_CAP entries (a note is a few KB, so the
        # cap is tens of MB worst case): a vault larger than the cap trades
        # re-reads for a bounded footprint instead of growing to every body
        # in RAM. Plain dict as the LRU: hits reinsert, eviction pops the
        # oldest key (insertion order).
        self._body_cache: dict[str, tuple[float, str]] = {}  # abs-path str -> (mtime, content)
        # {directory -> mtime} as of the last full rebuild. POSIX bumps a
        # directory's mtime when a file inside it is created, deleted or
        # renamed, and NOT when one is edited in place — which is exactly the
        # half `_patch_index` and the mtime-keyed body cache cannot cover. One
        # stat per FOLDER, never per note.
        self._dir_stamps: dict[str, float] = {}
        self._roster_checked: float = 0.0  # monotonic, see _ROSTER_RECHECK_INTERVAL

    def _path_of(self, ref: NoteRef | str) -> str | None:
        if isinstance(ref, NoteRef):
            return ref.path
        if ref in self._notes:
            return ref
        # A bare name, a vault-relative path, with or without ".md": the link
        # resolver already handles all four shapes. Trusting a ".md" suffix as
        # a graph key instead made links("Statistica.md") return [] for a note
        # actually stored at Matematica/Statistica/Statistica.md.
        resolved = self._resolve_target(ref)
        return resolved.path if resolved else None

    # ------------------------------------------------------------------
    # Indexing (in-memory graph)
    # ------------------------------------------------------------------

    @_locked
    def _ensure_index(self):
        if not self._needs_reindex and self._roster_drifted():
            self._needs_reindex = True
        if self._needs_reindex:
            self._rebuild_index()

    def _scan_dir_stamps(self) -> dict[str, float]:
        """{directory -> mtime} over the same tree `_rebuild_index` walks.

        The pruning MUST mirror the rebuild's: a vault adopted as-is can be a
        repo root, where `.git/` alone rewrites directory mtimes constantly and
        would put the roster in a permanent rebuild loop over trees the index
        does not contain anyway. Directories only — os.walk classifies entries
        from the dirent type, so no note is stat'd here.
        """
        skip = ignore_matcher(self.vault_path)
        stamps: dict[str, float] = {}
        for root, dirs, _files in os.walk(self.vault_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and not skip(d)]
            try:
                stamps[root] = os.stat(root).st_mtime
            except OSError:
                continue  # vanished mid-walk: absent here reads as drift, correctly
        return stamps

    def _roster_drifted(self) -> bool:
        """True when a note appeared or disappeared out-of-band (Obsidian, rm,
        git checkout, a sync client) since the last full rebuild.

        Content edits are invisible here by design: `_read_cached` is
        mtime-keyed and the derived indexes re-embed by content signature, so
        the roster only has to notice appearances and disappearances. Any
        difference counts, not just a newer one — a restored backup moves an
        mtime backwards, same as the index sweep's stamps.

        Failures answer False: this runs inside every read, and a rebuild
        skipped now is recovered by the next call.
        """
        now = time.monotonic()
        if now - self._roster_checked < _ROSTER_RECHECK_INTERVAL:
            return False
        self._roster_checked = now
        try:
            return self._scan_dir_stamps() != self._dir_stamps
        except OSError:
            return False

    def _restamp_dirs(self, rel_path: str) -> None:
        """Carry the directory stamps forward past a write this backend made.

        Silica's own write bumps the folder mtime like anyone else's, so
        without this every `_patch_index` would read as drift on the next
        `_ensure_index` and force the full rebuild that `_patch_index` exists
        to avoid. Only already-stamped ancestors are touched: writing into a
        brand-new folder (or an ignored tree) must not invent a key
        `_scan_dir_stamps` will never produce, or the comparison would differ
        forever. That case pays one rebuild, which repopulates the stamps.

        ponytail: an out-of-band create landing in the same folder between the
        write and this stat is absorbed and stays invisible until that folder
        changes again. Microseconds wide; close it with a per-file roster if a
        report ever traces a missing note to it.
        """
        full = self.vault_path / rel_path
        for d in (full.parent, *full.parent.parents):
            key = str(d)
            if key not in self._dir_stamps:
                break
            try:
                self._dir_stamps[key] = os.stat(d).st_mtime
            except OSError:
                del self._dir_stamps[key]
            if d == self.vault_path:
                break

    _ensure_graph = _ensure_index  # mixin hook (tests call _ensure_index directly)

    @_locked
    def _resolve_target(self, target: str, source_path: str = "") -> NoteRef | None:
        """Resolve a link target to an existing NoteRef or None if unresolved.

        Obsidian link resolution rules:
        1. If target starts with '#' or '^', it is an internal link. It resolves to the
           source file itself.
        2. If target contains '/', it is a path link. We check if target (or target + '.md')
           matches the end of the path of any existing note (with a leading slash or exact match).
        3. If target does not contain '/', it is a name link. We check if target matches
           the name of any note in the vault. If multiple exist, we prioritize the one in
           the same directory as source_path, then by shortest path.
        """
        if target.startswith('#') or target.startswith('^'):
            if source_path:
                return self._notes.get(source_path)
            return None

        # `./`/`../` are meaningful only relative to the SOURCE note; the suffix
        # matcher below can never see through them, so every such markdown link
        # died as a ghost node until 2026-08-25.
        normalized = resolve_relative(target, source_path)
        if normalized is None:
            return None  # escapes the vault root
        target = normalized

        target_no_ext = target.removesuffix(".md")
        if "/" in target:
            p1 = target_no_ext + ".md"
            p1_norm = os.path.normcase(p1.replace("\\", "/").strip("/")).lower()
            
            # Try exact match first
            for path, ref in self._notes.items():
                path_norm = os.path.normcase(path.replace("\\", "/").strip("/")).lower()
                if path_norm == p1_norm:
                    return ref
                    
            # Try matching end of path (suffix matching with /)
            suffix = "/" + p1_norm
            candidates = []
            for path, ref in self._notes.items():
                path_norm = os.path.normcase(path.replace("\\", "/").strip("/")).lower()
                if path_norm.endswith(suffix):
                    candidates.append(ref)
            if candidates:
                # Prioritize same directory as source_path if available
                if source_path and "/" in source_path:
                    source_dir = source_path.rsplit("/", 1)[0]
                    same_dir_candidates = [c for c in candidates if c.path.startswith(source_dir + "/")]
                    if same_dir_candidates:
                        sorted_same = sorted(same_dir_candidates, key=lambda r: (r.path.count("/"), r.path.lower()))
                        return sorted_same[0]
                sorted_candidates = sorted(candidates, key=lambda r: (r.path.count("/"), r.path.lower()))
                return sorted_candidates[0]
            return None
        else:
            refs = self._notes_by_name.get(target_no_ext.lower(), [])
            if not refs and (m := ADR_REF_RE.fullmatch(target_no_ext)):
                # `ADR-0003` names the record filed as `0003-*`; an adr folder
                # wins over any other note that happens to start with the
                # number (a dated note, a numbered lecture).
                refs = sorted(
                    (r for name, rs in self._notes_by_name.items()
                     if name.startswith(m.group(1) + "-") for r in rs),
                    key=lambda r: ("/adr/" not in f"/{r.path.lower()}", r.path.count("/"), r.path.lower()),
                )[:1]
            if not refs:
                return None
            if len(refs) == 1:
                return refs[0]
                
            # Prioritize the one in the same directory as source_path
            if source_path and "/" in source_path:
                source_dir = source_path.rsplit("/", 1)[0]
                same_dir_refs = [r for r in refs if r.path.startswith(source_dir + "/")]
                if same_dir_refs:
                    sorted_same = sorted(same_dir_refs, key=lambda r: (r.path.count("/"), r.path.lower()))
                    return sorted_same[0]
                    
            # Prioritize the one with the shortest vault-relative path
            sorted_refs = sorted(refs, key=lambda r: (r.path.count("/"), r.path.lower()))
            return sorted_refs[0]

    def _resolve_asset(self, target: str, source_path: str = "") -> str | None:
        """Resolve a file reference to a censused vault asset, Obsidian-style.

        Mirrors `_resolve_target`'s rules on the asset census instead of
        generalizing it: that resolver is entangled with NoteRef and the .md
        suffix, and the shared part is three lines of tie-breaking.
        # ponytail: linear suffix scan over the census — index it per-suffix
        # only if a vault ever censuses >10k assets.
        """
        t = resolve_relative(target, source_path)
        if t is None:
            return None  # escapes the vault root: no censused file can match
        t = t.strip().replace("\\", "/").lstrip("/")
        if not t:
            return None
        low = t.lower()
        hit = self._assets_lower.get(low)
        if hit:
            return hit
        shortest = lambda paths: sorted(paths, key=lambda p: (p.count("/"), p.lower()))[0]
        if "/" in t:
            cands = [p for lp, p in self._assets_lower.items() if lp.endswith("/" + low)]
            return shortest(cands) if cands else None
        cands = self._assets_by_name.get(low) or []
        if not cands:
            return None
        if "/" in source_path:
            src_dir = source_path.rsplit("/", 1)[0] + "/"
            same = [p for p in cands if p.rsplit("/", 1)[0] + "/" == src_dir]
            if same:
                return shortest(same)
        return shortest(cands)

    def _index_file_refs(self, rel_path: str, content: str, file_targets: list[str],
                         mentions: Sequence[str] = ()) -> None:
        """Record one note's file references: resolved embeds into the forward
        and inverse maps, misses into `_unresolved_assets` (counted, never
        silently dropped), and `documents:` frontmatter into its own
        repo-relative keyspace. Backtick mentions resolve into the same edge
        maps but their misses are dropped: an embed that misses is a broken
        note, an inline cite that misses is prose."""
        for ft in file_targets:
            hit = self._resolve_asset(ft, source_path=rel_path)
            if hit:
                bucket = self._asset_edges.setdefault(rel_path, [])
                if hit not in bucket:
                    bucket.append(hit)
                    self._asset_notes.setdefault(hit, []).append(rel_path)
            else:
                misses = self._unresolved_assets.setdefault(rel_path, [])
                if ft not in misses:
                    misses.append(ft)
        for mt in mentions:
            hit = self._resolve_asset(mt, source_path=rel_path)
            if hit:
                bucket = self._asset_edges.setdefault(rel_path, [])
                if hit not in bucket:
                    bucket.append(hit)
                    self._asset_notes.setdefault(hit, []).append(rel_path)
        docs = fm.documents_in(content)
        if docs:
            self._documents_edges[rel_path] = docs
            for d in docs:
                notes = self._documents_notes.setdefault(d, [])
                if rel_path not in notes:
                    notes.append(rel_path)

    def _drop_file_refs(self, rel_path: str) -> None:
        """Remove one note from every file-reference map (delete or re-upsert)."""
        for asset in self._asset_edges.pop(rel_path, []):
            notes = self._asset_notes.get(asset)
            if notes and rel_path in notes:
                notes.remove(rel_path)
                if not notes:
                    del self._asset_notes[asset]
        self._unresolved_assets.pop(rel_path, None)
        for d in self._documents_edges.pop(rel_path, []):
            notes = self._documents_notes.get(d)
            if notes and rel_path in notes:
                notes.remove(rel_path)
                if not notes:
                    del self._documents_notes[d]

    @_locked
    def _rebuild_index(self):
        logger.debug("Rebuilding FS graph index...")
        self._notes.clear()
        self._notes_by_name.clear()
        self._graph.clear()
        self._unresolved_links.clear()
        self._alias_pairs.clear()
        self._assets_lower.clear()
        self._assets_by_name.clear()
        self._asset_edges.clear()
        self._asset_notes.clear()
        self._unresolved_assets.clear()
        self._documents_edges.clear()
        self._documents_notes.clear()

        # Stamped BEFORE the file pass, deliberately: a note created between
        # the two walks is missed by this pass but its folder mtime is already
        # newer than what we stored, so the next check reads it as drift. The
        # reverse order would stamp the change we just failed to see and hide
        # the note for good. A stale stamp costs one rebuild; a fresh one
        # costs a note.
        self._dir_stamps = self._scan_dir_stamps()
        self._roster_checked = time.monotonic()

        files_to_process = []
        skip = ignore_matcher(self.vault_path)

        # Pass 1: Find all markdown files and populate self._notes and self._graph nodes
        for root, dirs, files in os.walk(self.vault_path):
            # Skip hidden folders, plus vendored/build trees and whatever
            # `.silicaignore` adds: a vault adopted as-is can be a repo root,
            # where node_modules/ alone would flood the note graph with
            # thousands of third-party READMEs.
            dirs[:] = [d for d in dirs if not d.startswith(".") and not skip(d)]

            for file in files:
                if not file.endswith(".md"):
                    # Asset census, same walk for free (the dirent is already
                    # in hand). Extension-filtered so a repo-root vault's
                    # source tree never floods it; freshness rides the same
                    # dir-mtime roster as notes — a new file bumps its
                    # folder's mtime, which reads as drift and rebuilds.
                    low = file.lower()
                    if low.endswith(NON_MD_EXTENSIONS):
                        rel_asset = (Path(root) / file).relative_to(self.vault_path).as_posix()
                        self._assets_lower[rel_asset.lower()] = rel_asset
                        self._assets_by_name.setdefault(low, []).append(rel_asset)
                    continue

                path = Path(root) / file
                rel_path_file = path.relative_to(self.vault_path).as_posix()

                # Silica's own generated vault-root files (log.md, GRAPH_REPORT.md)
                # are tooling output, not knowledge notes. Keeping them out of the
                # index here excludes them from every metric that reads it —
                # list_files (embed + cooccurrence builds) and graph_data
                # (mindmap) — in one place.
                if is_vault_artifact(rel_path_file):
                    continue

                name = file[:-3]
                ref = NoteRef(name=name, path=rel_path_file)
                self._notes[rel_path_file] = ref
                self._graph.add_node(rel_path_file, ref=ref)
                
                name_lower = name.lower()
                if name_lower not in self._notes_by_name:
                    self._notes_by_name[name_lower] = []
                self._notes_by_name[name_lower].append(ref)
                
                files_to_process.append((rel_path_file, path))

        # Pass 2: parse and resolve links. The mention index that used to be
        # built here (title trie walked over every body) was deleted 2026-08-23:
        # its only reader, `mentions_of`, had no caller left once BACKLINK moved
        # to `search_context_batch` (a fresh title has no postings by construction).
        for rel_path_file, path in files_to_process:
            try:
                content = path.read_text(encoding="utf-8")
                typed, file_targets, mentions = extract_refs_typed(content)
                for target, scaffold in typed.items():
                    ref = self._resolve_target(target, source_path=rel_path_file)
                    if ref:
                        self._add_link_edge(rel_path_file, ref.path, scaffold=scaffold)
                    else:
                        self._unresolved_links.add((rel_path_file, target))
                self._index_file_refs(rel_path_file, content, file_targets, mentions)

                # Frontmatter aliases — harvested here because this pass already
                # holds every body in hand; a separate props_of() sweep would
                # re-read the whole vault to learn the same thing.
                if (al := fm.aliases_of(content)):
                    self._alias_pairs[rel_path_file] = al
            except Exception as e:
                logger.warning("Failed to index %s: %s", rel_path_file, e)

        self._needs_reindex = False
        self._dirty_paths.clear()
        logger.debug("Indexed %d notes", len(self._notes))

    @_locked
    def _patch_index(self, rel_path: str, content: str | None) -> None:
        """Incrementally update the graph index for a single changed path.

        If content is None the note was deleted — remove it from the index.
        Call this instead of setting _needs_reindex = True for single-file writes.
        """
        # Vault artifacts are never indexed (_rebuild_index drops them), so a
        # write to one must degrade to a removal — otherwise it strands an entry
        # the next rebuild drops. The inbox is indexed like any other folder:
        # staging notes are source material, and the ops that must not target
        # them are gated by `is_inbox_path`, not by their absence from the index.
        if content is not None and is_vault_artifact(rel_path):
            content = None

        # Every caller has already written or unlinked the file, so the folder
        # mtime this reads is the post-write one. Here rather than at the two
        # exits below: the deletion path returns early.
        self._restamp_dirs(rel_path)

        # --- remove stale data for this path ---
        if rel_path in self._graph:
            self._graph.remove_edges_from(list(self._graph.out_edges(rel_path)))
        self._unresolved_links = {(s, t) for s, t in self._unresolved_links if s != rel_path}
        self._alias_pairs.pop(rel_path, None)

        if content is None:
            # deletion path
            if rel_path in self._graph:
                self._graph.remove_node(rel_path)
            old_ref = self._notes.pop(rel_path, None)
            if old_ref:
                name_lower = old_ref.name.lower()
                if name_lower in self._notes_by_name:
                    self._notes_by_name[name_lower] = [
                        r for r in self._notes_by_name[name_lower] if r.path != rel_path
                    ]
            self._dirty_paths.discard(rel_path)
            self._drop_file_refs(rel_path)
            return

        # --- upsert node ---
        name = rel_path.rsplit("/", 1)[-1].removesuffix(".md")
        ref = NoteRef(name=name, path=rel_path)
        self._notes[rel_path] = ref
        self._graph.add_node(rel_path, ref=ref)
        name_lower = name.lower()
        if name_lower not in self._notes_by_name:
            self._notes_by_name[name_lower] = []
        if ref not in self._notes_by_name[name_lower]:
            self._notes_by_name[name_lower].append(ref)

        # --- rebuild edges for this path ---
        typed, file_targets, mentions = extract_refs_typed(content)
        for target, scaffold in typed.items():
            target_ref = self._resolve_target(target, source_path=rel_path)
            if target_ref:
                self._add_link_edge(rel_path, target_ref.path, scaffold=scaffold)
            else:
                self._unresolved_links.add((rel_path, target))
        self._drop_file_refs(rel_path)   # replace, never accumulate, on re-upsert
        self._index_file_refs(rel_path, content, file_targets, mentions)

        if (al := fm.aliases_of(content)):
            self._alias_pairs[rel_path] = al

        self._dirty_paths.add(rel_path)

    def _resolve_path(self, ref: NoteRef | str) -> Path:
        """Resolve a NoteRef or name to a full filesystem path."""
        self._ensure_index()
        
        if isinstance(ref, NoteRef) and ref.path:
            p = Path(ref.path)
            if p.is_absolute():
                return p
            return self.vault_path / ref.path
            
        name = ref if isinstance(ref, str) else ref.name
        
        # Strip .md if passed in string
        if name.endswith(".md"):
            name = name[:-3]
        
        # Look up in index
        matched = self._notes_by_name.get(name.lower(), [])
        if matched:
            return self.vault_path / matched[0].path
            
        # Check if the name/ref is actually a path pointing directly to an
        # existing FILE (CLI direct-path reads). is_file, never exists: a bare
        # name colliding with a cwd DIRECTORY must not escape the vault
        # (post-mortem 2026-07-19: hub "memory" resolved to the repo's
        # ./memory/ dir and read_note crashed with IsADirectoryError).
        p = Path(name + ".md")
        if p.is_file():
            return p.resolve()
        p = Path(name)
        if p.is_file():
            return p.resolve()
            
        # Fallback for new files not yet in index
        return self.vault_path / f"{name}.md"

    def _read_cached(self, full: Path) -> str:
        """Body of a file, served from an mtime-keyed in-memory cache.

        Backend writes invalidate their own path explicitly; this mtime check is
        the secondary guard for edits made outside the backend.
        """
        key = str(full)
        try:
            mtime = full.stat().st_mtime
        except OSError:
            self._body_cache.pop(key, None)
            raise
        hit = self._body_cache.pop(key, None)
        if hit is not None and hit[0] == mtime:
            self._body_cache[key] = hit  # reinsert: most recently used
            return hit[1]
        self._body_cache[key] = (mtime, content := full.read_text(encoding="utf-8"))
        while len(self._body_cache) > _BODY_CACHE_CAP:
            del self._body_cache[next(iter(self._body_cache))]
        return content

    def _invalidate_body(self, rel_path: str) -> None:
        """Drop the cached body for a vault-relative path (write just landed)."""
        self._body_cache.pop(str(self.vault_path / rel_path), None)

    # ------------------------------------------------------------------
    # Discovery / Read
    # ------------------------------------------------------------------

    @_locked
    def search_names(self, query: str) -> list[NoteRef]:
        """Search vault note names matching query.

        sources/ leaves stay in the index (wikilink + read_note resolution)
        but are retrieval-invisible: excluded here and from every list/search
        surface, reachable only via an explicit `## Sources` link."""
        self._ensure_index()
        query = query.lower()
        results = []
        for ref in self._notes.values():
            if is_source_leaf(ref.path):
                continue
            if query in ref.name.lower():
                results.append(ref)
        return results

    @_locked
    def search_context(self, query: str) -> list[Hit]:
        """Search vault content with line-level context snippets.

        Two bounds keep a broad query ("e" produced 14535 Hits) from
        materializing the whole vault: a whole-body substring pre-check, so a
        non-matching note costs one scan and zero per-line allocations, and a
        per-note hit cap on the MATERIALIZED Hits. Density ranking does not
        ride on the capped list: every Hit carries `note_matches`, the note's
        true occurrence count, so two notes both capped at the limit still
        rank by how much they actually match.
        """
        self._ensure_index()
        query_lower = query.lower()
        results = []

        for name, ref in self._notes.items():
            if is_source_leaf(ref.path):  # leaves are search-invisible
                continue
            path = self.vault_path / ref.path
            try:
                content = self._read_cached(path)
                lower = content.lower()  # lower() never adds/removes newlines
                if query_lower not in lower:
                    continue
                matches = lower.count(query_lower)  # true density, one C pass
                lines = content.splitlines()
                note_hits = 0
                for i, line_lower in enumerate(lower.splitlines()):
                    if query_lower in line_lower:
                        results.append(Hit(
                            ref=ref,
                            line=i + 1,
                            snippet=lines[i].strip(),
                            note_matches=matches,
                        ))
                        note_hits += 1
                        if note_hits >= _MAX_HITS_PER_NOTE:
                            break
            except Exception:
                continue

        return results

    @_locked
    def search_context_batch(self, queries: list[str]) -> dict[str, list[Hit]]:
        """Batch of search_context: one vault sweep instead of one per query.

        Reads and lowercases each body once, then scans every query against it,
        so the output is byte-for-byte identical to
        ``{q: self.search_context(q) for q in queries}`` (same Hit ordering:
        notes in ``self._notes`` iteration order, then ascending line number).
        """
        self._ensure_index()
        if not queries:
            return {}

        # Dedupe (first-seen order): search_context(q) is called once per
        # distinct q in the reference impl, so a repeated query string must
        # not append its hits twice here.
        uniq = list(dict.fromkeys(queries))
        results: dict[str, list[Hit]] = {q: [] for q in uniq}
        queries_lower = [(q, q.lower()) for q in uniq]

        for ref in self._notes.values():
            if is_source_leaf(ref.path):  # parity with search_context
                continue
            path = self.vault_path / ref.path
            try:
                content = self._read_cached(path)
                lower = content.lower()
                # Same bounds as search_context (parity by construction):
                # body-level pre-check per query, per-note cap per query.
                live = [(q, ql) for q, ql in queries_lower if ql in lower]
                if not live:
                    continue
                lines = content.splitlines()
                lines_lower = lower.splitlines()
                for q, q_lower in live:
                    matches = lower.count(q_lower)
                    note_hits = 0
                    for i, line_lower in enumerate(lines_lower):
                        if q_lower in line_lower:
                            results[q].append(Hit(
                                ref=ref,
                                line=i + 1,
                                snippet=lines[i].strip(),
                                note_matches=matches,
                            ))
                            note_hits += 1
                            if note_hits >= _MAX_HITS_PER_NOTE:
                                break
            except Exception:
                continue

        return results

    def read_note(self, ref: NoteRef | str) -> NoteContent:
        """Read a note's full content by name or ref."""
        path = self._resolve_path(ref)
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")

        content = self._read_cached(path)
        name = ref if isinstance(ref, str) else ref.name
        
        try:
            rel_path = path.relative_to(self.vault_path).as_posix()
        except ValueError:
            # Fallback for external files outside the vault
            rel_path = path.resolve().as_posix()
            
        return NoteContent(
            ref=NoteRef(name=name, path=rel_path),
            content=content,
        )

    def mtime_of(self, ref: NoteRef | str) -> float | None:
        """Last-modified epoch seconds of a note, or None if it can't be stat'd.

        Recency proxy for the report's attention signal. Returns None (abstain)
        rather than raising when the file is absent — a new/unresolved ref has
        no recency to report.
        """
        try:
            return self._resolve_path(ref).stat().st_mtime
        except (OSError, RuntimeError):
            return None

    def alias_index(self) -> list[tuple[str, list[str]]]:
        """(title, aliases) pairs for every note declaring frontmatter aliases.

        Feeds autolink.build_alias_map. Served from the graph index, so it costs
        no reads of its own.
        """
        self._ensure_index()
        return [
            (ref.name, self._alias_pairs[path])
            for path, ref in self._notes.items()
            if path in self._alias_pairs
        ]

    def props_of(self, ref: NoteRef | str) -> dict:
        """Read frontmatter properties."""
        try:
            nc = self.read_note(ref)
            data, _, _ = fm.split(nc.content)
            return data or {}
        except RuntimeError:
            return {}

    def outline(self, ref: NoteRef | str) -> list[Heading]:
        """Get the heading tree of a note."""
        try:
            nc = self.read_note(ref)
            raw_headings = ofm.parse_headings(nc.content)
            return [
                Heading(
                    level=h["level"],
                    text=h["text"],
                    position=h["pos"]
                ) for h in raw_headings
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    @_locked
    def graph_data(self) -> tuple[dict, set, Any]:
        """Locked override of the mixin's accessor, returning copies.

        The mixin hands out live references to `_notes`/`_unresolved_links`/
        `_graph`; consumers (graph_export, mindmap) iterate them long after
        the lock would have been released, while an MCP write thread runs
        `_patch_index` — the same "dictionary changed size during iteration"
        the method-level lock exists to close. Shallow copies decouple the
        iteration from the index; NoteRefs stay shared, which readers treat
        as immutable.
        """
        self._ensure_index()
        return dict(self._notes), set(self._unresolved_links), self._graph.copy()

    @_locked
    def links(self, ref: NoteRef | str) -> list[NoteRef]:
        """Outgoing links from a note."""
        self._ensure_index()
        path = self._path_of(ref)
        if not path:
            return []
        
        # Resolved outgoing links from graph
        results = []
        if path in self._graph:
            results.extend([self._node_ref(t) for t in self._graph.successors(path)])
        
        # Unresolved outgoing links
        for s, t in self._unresolved_links:
            if s == path:
                t_name = t.rsplit("/", 1)[-1].removesuffix(".md")
                results.append(NoteRef(name=t_name, path=f"{t_name}.md"))
                
        return results

    @_locked
    def backlinks(self, ref: NoteRef | str) -> list[NoteRef]:
        """Incoming links to a note."""
        self._ensure_index()
        path = self._path_of(ref)
        if not path or path not in self._graph:
            return []
        return [self._node_ref(s) for s in self._graph.predecessors(path)]

    @_locked
    def orphans(self) -> list[NoteRef]:
        """Notes with no incoming links."""
        self._ensure_index()
        return [self._graph.nodes[n]["ref"] for n, d in self._graph.in_degree() if d == 0]

    @_locked
    def unresolved(self) -> list[Link]:
        """Unresolved wikilinks in the vault."""
        self._ensure_index()
        results = []
        for s, t in self._unresolved_links:
            results.append(Link(source=self._node_ref(s), target=t.removesuffix(".md")))
        return results

    @_locked
    def file_refs_of(self, ref: NoteRef | str) -> dict:
        """Files one note references: resolved embeds (vault-relative), its
        `documents:` entries (repo-relative), and unresolved raw targets."""
        self._ensure_index()
        path = self._path_of(ref)
        if not path:
            return {"embeds": [], "documents": [], "unresolved": []}
        return {
            "embeds": list(self._asset_edges.get(path, [])),
            "documents": list(self._documents_edges.get(path, [])),
            "unresolved": list(self._unresolved_assets.get(path, [])),
        }

    @_locked
    def file_backlinks(self, path: str) -> dict:
        """Notes referencing a file: by body embed and by `documents:`.

        The embed leg resolves `path` like a link target (basename included);
        the documents leg matches the repo-relative entry exactly, plus a
        basename-suffix convenience — the two keyspaces are looked up side by
        side, never merged into one map (a vault-relative and a repo-relative
        path can name the same file with different strings).
        """
        self._ensure_index()
        t = (path or "").strip().replace("\\", "/").lstrip("/")
        embeds: list[str] = []
        if t:
            resolved = self._resolve_asset(t)
            if resolved:
                embeds = list(self._asset_notes.get(resolved, []))
        docs = list(self._documents_notes.get(t, []))
        if t and not docs:
            suffix = "/" + t.lower()
            for d, notes in self._documents_notes.items():
                if d.lower().endswith(suffix):
                    docs.extend(notes)
        return {"embeds": embeds, "documents": sorted(dict.fromkeys(docs))}

    @_locked
    def graph_snapshot(self, refs: list[NoteRef] | None = None) -> GraphSnapshot:
        """Graph snapshot for non-regression gating.

        If refs is provided, performs an incremental snapshot covering only
        the touched notes and their 1-hop neighborhood.
        """
        self._ensure_index()
        if refs is None:
            link_counts = {}
            for path, ref in self._notes.items():
                resolved_count = self._graph.out_degree(path) if path in self._graph else 0
                unresolved_count = sum(1 for s, t in self._unresolved_links if s == path)
                # Key by canonical path (no .md) — unique even with duplicate basenames.
                # graph_diff.normalize_path() strips .md and lowercases, so path-keyed
                # snapshots compare identically to name-keyed ones in the diff.
                key = path.removesuffix(".md")
                link_counts[key] = resolved_count + unresolved_count

            backlink_counts = {
                path.removesuffix(".md"): d
                for path, d in self._graph.in_degree()
            }
            return GraphSnapshot(
                orphans=self.orphans(),
                unresolved=self.unresolved(),
                link_counts=link_counts,
                backlink_counts=backlink_counts
            )

        # Incremental snapshot
        neighborhood = set()
        for r in refs:
            if r.path:
                neighborhood.add(r.path)
                # Add outgoing
                if r.path in self._graph:
                    for t in self._graph.successors(r.path):
                        neighborhood.add(t)
                # Add incoming
                if r.path in self._graph:
                    for s in self._graph.predecessors(r.path):
                        neighborhood.add(s)

        link_counts = {}
        backlink_counts = {}
        for path in neighborhood:
            note = self._notes.get(path)
            if note:
                resolved_count = self._graph.out_degree(path) if path in self._graph else 0
                unresolved_count = sum(1 for s, t in self._unresolved_links if s == path)
                key = path.removesuffix(".md")
                link_counts[key] = resolved_count + unresolved_count
                backlink_counts[key] = self._graph.in_degree(path) if path in self._graph else 0

        # Filter orphans & unresolved to neighborhood incrementally
        orphans = [
            self._notes[path] for path in neighborhood
            if path in self._notes and (path not in self._graph or self._graph.in_degree(path) == 0)
        ]
        unresolved = []
        for path in neighborhood:
            if path in self._notes:
                source_ref = self._notes[path]
                for s, t in self._unresolved_links:
                    if s == path:
                        unresolved.append(Link(source=source_ref, target=t.removesuffix(".md")))

        return GraphSnapshot(
            orphans=orphans,
            unresolved=unresolved,
            link_counts=link_counts,
            backlink_counts=backlink_counts
        )

    # ------------------------------------------------------------------
    # Write (graph-safe)
    # ------------------------------------------------------------------

    def create(self, path: str, content: str) -> NoteRef:
        """Create a new note at the given vault-relative path."""
        rel_path = contain_in_vault(path, self.vault_path)

        full_path = self.vault_path / rel_path
        # Base contract: "Raises if file exists" (base.py). Unconditional
        # write_text here used to clobber a note created between validate-time
        # path_exists() and execute-time create() — silent data loss, no
        # 3-way merge (write ops carry no base_content). WS backend already
        # raises (Obsidian vault.create throws on existing). Refresh callers
        # go through upsert() instead.
        if full_path.exists():
            raise FileExistsError(f"Note already exists: {rel_path}")
        full_path.parent.mkdir(parents=True, exist_ok=True)

        content = stamp_type(rel_path, content)   # OKF §4.1 `type`, if absent
        session_changes.touched(rel_path, None)  # no baseline: the note is new
        # Notes are irreplaceable and this backend keeps no version history
        # (snapshot_versions returns an empty Txn), so a truncating write_text
        # has nothing to roll back to when it dies mid-write.
        atomic_write_bytes(full_path, content.encode("utf-8"))
        self._invalidate_body(rel_path)
        name = rel_path.rsplit("/", 1)[-1].removesuffix(".md")
        if self._needs_reindex:
            self._rebuild_index()
        else:
            self._patch_index(rel_path, content)
        return NoteRef(name=name, path=rel_path)

    def overwrite(self, path: str, content: str) -> NoteRef:
        """Overwrite an existing note in-place.

        The FS backend does this as a direct write — history is not tracked
        in FS mode, so overwrite and patch rollback via versions is a no-op
        (see restore()). For write-op rollback, created_paths is used instead.
        """
        return self._overwrite_raw(path, content, stamp=True)

    def _overwrite_raw(self, path: str, content: str, stamp: bool = False) -> NoteRef:
        """The write itself. `stamp=False` writes the bytes as given — for
        callers replaying content that already exists (a rollback, or the WS
        stub emulating the Obsidian plugin, which is a verbatim pipe and does
        no stamping of its own).
        """
        rel_path = contain_in_vault(path, self.vault_path)

        full_path = self.vault_path / rel_path
        if not full_path.exists():
            raise RuntimeError(f"Cannot overwrite non-existent file: {path}")

        if stamp:
            content = stamp_type(rel_path, content)   # OKF §4.1 `type`, if absent
        session_changes.touched(rel_path, self._read_cached(full_path))
        atomic_write_bytes(full_path, content.encode("utf-8"))
        self._invalidate_body(rel_path)
        name = rel_path.rsplit("/", 1)[-1].removesuffix(".md")
        if self._needs_reindex:
            self._rebuild_index()
        else:
            self._patch_index(rel_path, content)
        return NoteRef(name=name, path=rel_path)

    def autolink_note(
        self,
        path: str,
        candidates: list[str] | None = None,
        title_index: list[str] | None = None,
    ) -> list[str]:
        """FS backend: pure-Python kernel autolink + direct overwrite.

        `title_index`, when given, is used as-is (caller-built, e.g. LINKING's
        one-per-chunk index) instead of rebuilding via build_title_index(
        self.list_files()) on every call.
        """
        import os
        from silica.kernel.link.autolink import autolink, build_alias_map, build_title_index
        nc = self.read_note(path)
        body = nc.content or ""
        if not body.strip():
            return []
        if title_index is None:
            title_index = build_title_index(self.list_files())
        self_title = os.path.splitext(os.path.basename(path))[0]
        aliases = build_alias_map(self.alias_index(), title_index)
        new_body, added = autolink(
            body, title_index, candidates=candidates, self_title=self_title, aliases=aliases
        )
        if added:
            self.overwrite(path, new_body)
        return added

    def append(self, ref: NoteRef | str, content: str) -> None:
        """Append content to an existing note."""
        path = self._resolve_path(ref)
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")

        # Same boundary as create/overwrite: `ref` is caller-supplied and
        # `_resolve_path` happily joins it onto the vault, so a note that is a
        # symlink out of the vault reads as vault-relative while the append
        # lands on the target. relative_to() alone only catches the absolute
        # case — it compares strings and never resolves the link.
        rel_path_str = contain_in_vault(str(path), self.vault_path)
        session_changes.touched(rel_path_str, self._read_cached(path))
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)

        self._invalidate_body(rel_path_str)
        if self._needs_reindex:
            self._rebuild_index()
        else:
            full_content = self._read_cached(path)
            self._patch_index(rel_path_str, full_content)

    def set_prop(self, ref: NoteRef | str, name: str, value: Any, type_: str = "text") -> None:
        """Set a frontmatter property on a note."""
        path = self._resolve_path(ref)
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")

        # Contained before the read, for the reason append() gives: the write
        # below resolves symlinks (atomic_write_bytes writes THROUGH them), so
        # an unresolved boundary check here would rewrite a file outside.
        rel_path_str = contain_in_vault(str(path), self.vault_path)
        content = path.read_text(encoding="utf-8")
        data, delim, body = fm.split(content)

        if data is None:
            data = {}

        data[name] = value

        new_content = fm.dump(data, body)
        session_changes.touched(rel_path_str, content)
        atomic_write_bytes(path, new_content.encode("utf-8"))
        self._body_cache.pop(str(path), None)

    @_locked
    def move(self, ref: NoteRef | str, to: str) -> None:
        """Move/rename a note, rewriting incoming wikilinks in all referrers.

        Mirrors Obsidian's "automatically update internal links" behaviour:

        - Resolved referrers (predecessors in the graph) have their link text
          rewritten via the pure kernel ``rewrite_links`` function.
        - Ambiguity guard: if the old basename is shared by multiple notes and
          the referrer's name-based resolution points elsewhere, only path-based
          links in that referrer are rewritten (``rewrite_name_links=False``).
        - After the physical rename, the in-memory index is updated
          incrementally for the moved note and every rewritten referrer.
        - Unresolved-promotion sweep: raw targets that were previously
          unresolvable but now resolve to the new path are promoted to resolved
          graph edges via ``_patch_index``.
        """
        from silica.kernel.link.rename import rewrite_links

        # Step 1: guarantee a fresh index before reading graph state
        self._ensure_index()

        src = self._resolve_path(ref)
        if not src.exists():
            raise RuntimeError(f"File not found: {src}")

        # Step 2: vault-relative paths. Both ends go through the containment
        # choke point: `ref` may be a NoteRef carrying an absolute path, and
        # `to` is caller-supplied, so neither is trusted to stay in the vault.
        old_rel = contain_in_vault(str(src), self.vault_path)
        new_rel = contain_in_vault(to, self.vault_path)
        old_basename = old_rel.rsplit("/", 1)[-1].removesuffix(".md")

        # Step 3: collect referrers BEFORE moving so graph is still accurate
        referrers: list[str] = list(self._graph.predecessors(old_rel)) if old_rel in self._graph else []

        # Ambiguity guard: check whether the old basename is shared by
        # multiple notes (i.e. name-based resolution could be ambiguous).
        basename_is_unique = len(self._notes_by_name.get(old_basename.lower(), [])) <= 1

        # Step 4: physical filesystem move
        dst = self.vault_path / new_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        # The row follows the file: baseline first (in case this is the note's
        # first touch this session), then move it onto the new key.
        session_changes.touched(old_rel, self._read_cached(src))
        session_changes.renamed(old_rel, new_rel)
        src.rename(dst)
        # The ledger follows the file the way the session row does: it keys
        # notes by path, and a moved note the ledger still knows under its old
        # name is re-appended into on the next nucleate of its source.
        # Never raises; a failure is logged with what it costs.
        from silica.kernel.write.provenance import rename_note

        rename_note(old_rel, new_rel, vault_path=str(self.vault_path))
        self._invalidate_body(old_rel)
        self._invalidate_body(new_rel)
        # The old key's vector now points at a gone path; drop it or a rename makes
        # the note appear twice in candidates (stale old key + fresh new key). The
        # new path re-embeds lazily on the next build_index (A13).
        self._drop_embed_vector(old_rel)

        # move() is the only multi-file write in this backend.  Everything after
        # the physical rename (referrer disk-writes, index patches, unresolved
        # sweep) is wrapped so that ANY failure sets _needs_reindex=True before
        # re-raising.  This forces a clean full rebuild on the next
        # _ensure_index(), matching the backend's existing fallback convention
        # (see restore()).  The alternative — leaving the in-memory index in a
        # torn state — would silently corrupt subsequent operations in the same
        # session (e.g. the rest of an /organize batch).
        try:
            # Step 5a: rewrite link text on disk for referrers that need it.
            # Also collect updated content for every referrer so we can re-patch
            # the index after step 6 (which deletes the old node and breaks edges).
            referrer_updates: list[tuple[str, str]] = []  # (rel_path, content_to_index)

            for referrer_rel in referrers:
                referrer_path = self.vault_path / referrer_rel
                if not referrer_path.exists():
                    continue
                referrer_content = referrer_path.read_text(encoding="utf-8")

                # The referrers are the third write target of a move, and the
                # only one that is discovered rather than passed in — so it is
                # the one the containment at step 2 does not cover. A referrer
                # that is a symlink out of the vault would be rewritten THROUGH
                # the link, editing a file the boundary does not own. Withhold
                # only the disk write: its edges are still re-indexed below, so
                # the graph stays correct while the foreign file keeps the old
                # link text.
                try:
                    contain_in_vault(referrer_rel, self.vault_path)
                except ValueError:
                    logger.warning(
                        "move: referrer %s leaves the vault — its links to %s "
                        "were left unrewritten", referrer_rel, old_rel)
                    referrer_updates.append((referrer_rel, referrer_content))
                    continue

                # Determine whether name-based rewrites are safe for this referrer
                if basename_is_unique:
                    allow_name = True
                else:
                    # Resolve where [[old_basename]] points from this referrer's
                    # perspective — only allow name-based rewrite if it resolves
                    # to the moved note (not some other same-named note).
                    resolved = self._resolve_target(old_basename, source_path=referrer_rel)
                    allow_name = resolved is not None and resolved.path == old_rel

                new_content, n = rewrite_links(
                    referrer_content, old_rel, new_rel,
                    rewrite_name_links=allow_name,
                )
                if n > 0:
                    # Write directly — avoids re-entrant overwrite() logic
                    session_changes.touched(referrer_rel, referrer_content)
                    atomic_write_bytes(referrer_path, new_content.encode("utf-8"))
                    self._invalidate_body(referrer_rel)
                    referrer_updates.append((referrer_rel, new_content))
                else:
                    # Even if content is unchanged, we must re-patch after the old
                    # node is removed (step 6) so that name-based edges that still
                    # resolve correctly are re-established in the graph.
                    referrer_updates.append((referrer_rel, referrer_content))

            # Step 6: patch index for the moved note itself first, so that when
            # referrer edges are rebuilt in step 5b, _resolve_target() can already
            # see the new path.
            moved_content = dst.read_text(encoding="utf-8")
            self._patch_index(old_rel, None)
            self._patch_index(new_rel, moved_content)

            # Step 5b: re-index every referrer now that new_rel is registered.
            # This rebuilds their outgoing edges (including name-based links that
            # now resolve to new_rel) without requiring any file content change.
            for referrer_rel, content in referrer_updates:
                self._patch_index(referrer_rel, content)

            # Step 7: unresolved-promotion sweep — targets that were previously
            # unresolvable may now resolve because the new name/path matches them.
            # Collect affected sources first, then patch (avoid mutating while iterating).
            sources_to_promote: list[tuple[str, str]] = []
            # Snapshot, not the live set: @_locked keeps another MCP thread's
            # _patch_index out, but this loop must survive any future in-loop
            # patch too — a set mutated while iterated raises.
            for source, target in list(self._unresolved_links):
                resolved = self._resolve_target(target, source_path=source)
                if resolved is not None and resolved.path == new_rel:
                    sources_to_promote.append((source, target))
            for source, _target in sources_to_promote:
                promote_path = self.vault_path / source
                if promote_path.exists():
                    promote_content = promote_path.read_text(encoding="utf-8")
                    self._patch_index(source, promote_content)

        except Exception:
            # Force a full rebuild on next _ensure_index() so no torn state
            # persists into subsequent operations. A torn move can leave any
            # number of referrer bodies rewritten on disk but uninvalidated
            # here, so drop the whole cache rather than track partial state.
            self._needs_reindex = True
            self._body_cache.clear()
            raise

    def _drop_embed_vector(self, rel_path: str) -> None:
        """Remove a note's embedding vector when it is deleted/renamed, so
        cosine_top_k stops returning it as a phantom candidate before the next
        full /embed rebuild (audit A13). Best-effort: retrieval quality, never fatal.

        The in-memory store updates per op (this process never sees the
        phantom); the npz save is buffered every _EMBED_FLUSH_EVERY deletes
        plus once at exit, because each save serializes the whole index and a
        bulk /organize was paying that per note.
        """
        global _embed_deletes_pending
        try:
            from silica.kernel.recall.embed import get_store
            store = get_store()
            key = rel_path.removesuffix(".md")
            if store.get_vec(key) is not None:  # skip non-embedding vaults / unindexed notes
                store.delete(key)
                _embed_deletes_pending += 1
                if _EMBED_FLUSH_EVERY is None or _embed_deletes_pending >= _EMBED_FLUSH_EVERY:
                    store.save()
                    _embed_deletes_pending = 0
                else:
                    _register_embed_flush()
        except Exception as exc:
            logger.debug("embed vector cleanup failed for %s (non-fatal): %s", rel_path, exc)

    def delete(self, ref: NoteRef | str) -> None:
        """Delete a note from the vault."""
        path = self._resolve_path(ref)
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")

        rel_path_str = path.relative_to(self.vault_path).as_posix()
        session_changes.touched(rel_path_str, self._read_cached(path))
        path.unlink()
        self._invalidate_body(rel_path_str)
        if self._needs_reindex:
            self._rebuild_index()
        else:
            self._patch_index(rel_path_str, None)
        self._drop_embed_vector(rel_path_str)

    # ------------------------------------------------------------------
    # Advanced
    # ------------------------------------------------------------------

    @_locked
    def list_files(self, folder: str = "") -> list[NoteRef]:
        """List all markdown files, optionally filtered by folder.

        sources/ leaves are excluded: list_files feeds every derived index
        (embed, co-occurrence, lexical, autolink title index, vault map), and
        the one-rule exclusion here keeps leaves out of all of them at once."""
        self._ensure_index()

        results = []
        for ref in self._notes.values():
            if is_source_leaf(ref.path):
                continue
            if not folder or ref.path.startswith(folder):
                results.append(ref)

        return results

    def list_inbox_files(self) -> list[NoteRef]:
        """List all files in the inbox directory."""
        from silica.kernel.vault_manifest import active_inbox_dir
        inbox = active_inbox_dir()
        if not inbox:
            return []
        inbox_path = self.vault_path / inbox
        if not inbox_path.exists() or not inbox_path.is_dir():
            return []
        results = []
        for root, dirs, files in os.walk(inbox_path):
            # The inbox holds files awaiting conversion (PDFs etc.), not just
            # notes — list everything except dotfiles (.trash, .DS_Store...).
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                if file.startswith("."):
                    continue
                full_p = Path(root) / file
                try:
                    rel_p = full_p.relative_to(self.vault_path).as_posix()
                except ValueError:
                    rel_p = full_p.resolve().as_posix()
                # Non-md names keep the extension (removesuffix is a no-op).
                name = file.removesuffix(".md")
                results.append(NoteRef(name=name, path=rel_p))
        return results

    # ------------------------------------------------------------------
    # Transactionality
    # ------------------------------------------------------------------

    def snapshot_versions(self, refs: list[NoteRef]) -> Txn:
        """Snapshot current versions for later rollback.

        The FS backend does not track version history, so `versions` is always
        empty. Rollback of patch ops is a no-op in FS mode. Rollback of write
        ops works via `created_paths` (delete the created notes).
        """
        txn_id = f"txn_fs_{int(time.time())}"
        return Txn(id=txn_id, refs=refs)

    def restore(self, txn: Txn) -> None:
        """Rollback a transaction.

        - created_paths: deletes newly-created notes to undo write ops.
        """
        for path in txn.created_paths:
            try:
                full_path = self.vault_path / path
                if full_path.exists():
                    full_path.unlink()
                    self._invalidate_body(path)
                    logger.info("Rolled back created note: %s", path)
                    if self._needs_reindex:
                        pass  # full rebuild will happen on next _ensure_index
                    else:
                        self._patch_index(path, None)
            except Exception as e:
                logger.error("Failed to delete created note %s during rollback: %s", path, e)
