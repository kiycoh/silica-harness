# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Invocation-time index sweep — out-of-band freshness for the derived indexes.

The write path keeps every derived index fresh for Silica's OWN writes
(write.py's freshness hooks). This module covers the other half: notes
created, edited, or deleted out-of-band (Obsidian, ``rm``, ``git checkout``,
a sync client), whether or not a Silica process was running at the time.
Detected at invocation — never by a watcher (charter: no resident processes).

The vault side of that is not this module's: the sweep enumerates through
``DRIVER.list_files("")``, so a note it never hears about is one it can never
index. Creates and deletes reach that roster because the fs backend re-checks
folder mtimes (``fs_backend._roster_drifted``); everything below assumes it.
Nor is the index side: a second Silica process writing the same index files
is absorbed by the store singletons themselves (``paths.DiskSynced``), so
what this sweep saves is always a merge, never an overwrite of their work.

Mechanism, git-index style: a sidecar stamp file maps note path → last-seen
file mtime. A note whose current mtime DIFFERS (any difference, not just
newer — a restored backup moves mtime backwards) is a candidate: its body is
read once and handed to the per-store refreshers, all idempotent (embed
dedups by content signature, cooccur/lexical upserts are replacement
semantics). The mtime is only the pre-filter; the embed content signature is
the correctness backstop for lying mtimes.

Cold indexes stay cold: the sweep maintains stores that exist, it never
builds one — /embed, /cooccur and /lexical own their first build. The
personal-memory lane (ADR-0019) is out of scope: its vault is not behind
DRIVER, and its stores are refreshed by their own capture path.

ponytail: mtime-only stamps — a byte-different rewrite landing on the
identical st_mtime float escapes until the next edit; add file size to the
stamp if that ever bites.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import orjson

from silica.config import CONFIG
from silica.kernel.recall.paths import atomic_write_bytes, disk_stamp, index_dir

logger = logging.getLogger(__name__)

# Mass-drift ceiling: embedding costs API calls, so a sweep with more changed
# notes than this warns and defers to an explicit /embed rather than stalling
# a query on an implicit whole-vault rebuild. Changed-but-unembedded notes
# stay unstamped, so they are retried every sweep until /embed clears the
# backlog — that repeat work is the price of never stamping a stale vector.
_RECONCILE_CAP = 500
# A live vault view missing more than this fraction of a populated store
# smells like a misconfig/partial read, not mass deletion — refuse to prune.
# ponytail: a flat ratio; swap for a delete-journal if false skips ever bite.
_PRUNE_RATIO_CEILING = 0.5
# Several facade_retrieve calls can land in one agent turn; a sweep that just
# ran has nothing new to see.
_MIN_INTERVAL = 2.0
_last_sweep = 0.0


@dataclass
class SweepStats:
    """What one sweep did. `skipped` = the sweep abstained before scanning."""
    embedded: int = 0   # notes re/embedded into the embed store
    refreshed: int = 0  # notes re-contributed to cooccur/lexical
    pruned: int = 0     # index entries dropped (deleted out-of-band), all stores
    stamped: int = 0    # stamp entries written/updated this sweep
    skipped: bool = False


def vault_version() -> str:
    """Short digest that changes when anything a derived VIEW renders from did.

    The read-only twin of `sweep`: it reports that the vault moved, it never
    rebuilds anything. The GUI's explore surfaces (graph, map, folders/areas/
    read) are expensive enough that they are built once and cached, and the
    only thing that used to invalidate them was a chat turn in that same
    browser tab — so an Obsidian edit, or a `silica nucleate` in a terminal,
    left them drawing a vault that no longer existed. Polling this is what
    tells them otherwise.

    Both halves of what those views read, because either alone lies: the note
    roster with its mtimes (titles, links, folders, the whole structural
    graph) and the three derived index files (the semantic overlay and the
    communities drawn from them). Reading the roster through the driver is
    also what refreshes it — `list_files` re-checks for out-of-band creates
    and deletes (fs_backend._roster_drifted), so the poll doubles as the thing
    that keeps the answer true.

    Measured 4.8 ms over 709 notes: cheaper than the 304 that /graph's ETag
    can only produce by building the whole graph, Louvain pass included, which
    is the reason a content ETag cannot be the poll.

    Never raises, and answers "" when it cannot read — a poll that threw would
    stop, and a caller cannot tell a broken vault from an unchanged one, so
    the safe reading of "" is "nothing to do".
    """
    from silica.driver import DRIVER
    from silica.kernel.recall.cooccurrence import _index_path as _cooccur_index
    from silica.kernel.recall.embed import _index_path as _embed_index
    from silica.kernel.recall.lexical import _index_path as _lexical_index

    h = hashlib.blake2b(digest_size=8)
    try:
        mtime_of = getattr(DRIVER, "mtime_of", None)
        # Sorted: list_files walks the filesystem, whose order is not stable
        # across rebuilds, and an order-sensitive digest would report a change
        # on every poll.
        for ref in sorted(DRIVER.list_files(""), key=lambda r: r.path or r.name):
            h.update((ref.path or ref.name).encode("utf-8"))
            if mtime_of is not None:  # ws backend has none: roster-only digest
                h.update(str(mtime_of(ref)).encode("ascii"))
        # Each store's own `_index_path`, not a list of names spelled again
        # here: a renamed index file would otherwise stop being watched and
        # nothing would say so.
        for path in (_embed_index(), _cooccur_index(), _lexical_index()):
            h.update(str(disk_stamp(path)).encode("ascii"))
    except Exception as e:
        logger.debug("vault version unavailable (%s)", e)
        return ""
    return h.hexdigest()


def _stamps_path() -> Path:
    return index_dir() / "sync_stamps.json"


def _load_stamps() -> dict[str, float]:
    try:
        return orjson.loads(_stamps_path().read_bytes())
    except Exception:
        return {}


def _safe_to_prune(orphaned: set[str], live: set[str], store_len: int) -> bool:
    """Guard against pruning on a bogus live view (wrong vault path, partial
    fs read): an empty view, or one missing more than half a populated store,
    is a misconfig — never mass deletion."""
    if not orphaned or not live:
        return False
    return len(orphaned) <= max(20, _PRUNE_RATIO_CEILING * store_len)


def _prune_orphans(store, live: set[str], delete, label: str) -> int:
    """Drop index entries whose note was deleted out-of-band (Obsidian, ``rm``)."""
    have = set(store.paths())
    orphaned = have - live
    if not orphaned:
        return 0
    if not _safe_to_prune(orphaned, live, len(have)):
        logger.warning(
            "%s index has %d/%d entries absent from vault — run /%s to reconcile; "
            "skipping auto-prune (stale/partial view)",
            label, len(orphaned), len(have), label,
        )
        return 0
    for p in orphaned:
        delete(p)
    logger.info("%s sweep pruned %d note(s) deleted out-of-band", label, len(orphaned))
    return len(orphaned)


def sweep(*, force: bool = False) -> SweepStats:
    """Reconcile every seeded index with the vault on disk. Never raises.

    Called from the read path (facade_retrieve, before the indexes are read)
    and from the injector at run start. ``force`` bypasses the debounce, not
    the ``SILICA_INDEX_SWEEP`` config gate.
    """
    global _last_sweep
    stats = SweepStats()
    if not getattr(CONFIG, "index_sweep", True):
        stats.skipped = True
        return stats
    now = time.monotonic()
    if not force and now - _last_sweep < _MIN_INTERVAL:
        stats.skipped = True
        return stats
    _last_sweep = now
    try:
        return _sweep(stats)
    except Exception as e:
        logger.debug("index sweep skipped (%s)", e)
        stats.skipped = True
        return stats


def _sweep(stats: SweepStats) -> SweepStats:
    from silica.driver import DRIVER
    from silica.kernel.recall.embed import get_store

    # A driver without mtime_of (ws backend) has no cheap change signal — the
    # bridge's own write path keeps the indexes fresh while it is connected.
    mtime_of = getattr(DRIVER, "mtime_of", None)
    if mtime_of is None:
        stats.skipped = True
        return stats

    embed_store = get_store()
    embed_seeded = len(embed_store) > 0
    cooccur_store = None
    try:
        from silica.kernel.recall.cooccurrence import get_cooccur_store
        cs = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
        if cs.paths():
            cooccur_store = cs
    except Exception:
        cooccur_store = None
    lexical_store = None
    try:
        if (index_dir() / "lexical.json").is_file():
            from silica.kernel.recall.lexical import get_lexical_store
            lexical_store = get_lexical_store()
    except Exception:
        lexical_store = None

    if not embed_seeded and cooccur_store is None and lexical_store is None:
        stats.skipped = True  # cold vault — explicit builds own the first index
        return stats

    # One vault scan: live paths + their current mtimes. A note that cannot be
    # stat'd (deleted mid-sweep) is treated as absent.
    live: dict[str, tuple[str, float]] = {}  # idx_path -> (name, mtime)
    for ref in DRIVER.list_files(""):
        idx_path = (ref.path or ref.name).removesuffix(".md")
        if not idx_path:
            continue
        mt = mtime_of(ref)
        if mt is None:
            continue
        live[idx_path] = (ref.name or Path(idx_path).name, mt)
    live_set = set(live)

    # PRUNE (free) — all three stores, each behind the partial-view guard.
    if embed_seeded:
        n_pruned = _prune_orphans(embed_store, live_set, embed_store.delete, "embed")
        if n_pruned:
            embed_store.save()
            stats.pruned += n_pruned
    if cooccur_store is not None:
        n_pruned = _prune_orphans(cooccur_store, live_set, cooccur_store.delete_note, "cooccur")
        if n_pruned:
            cooccur_store.save()
            stats.pruned += n_pruned
    if lexical_store is not None:
        n_pruned = _prune_orphans(lexical_store, live_set, lexical_store.remove, "lexical")
        if n_pruned:
            lexical_store.save()
            stats.pruned += n_pruned

    # Candidates: stamp missing (new note, or entry written before stamping
    # existed) or stamp != current mtime (out-of-band edit, restored backup).
    stamps = _load_stamps()
    candidates = [p for p, (_n, mt) in live.items() if stamps.get(p) != mt]
    # Carry every live stamp forward, then overwrite the ones that settle below.
    # A candidate that fails to settle keeps its OLD stamp instead of losing it:
    # dropping it would make the next sweep read it as a first sighting, and
    # first sightings are baseline for the deterministic legs — the note would
    # never be re-contributed again. Dead paths fall out (not in `live`).
    new_stamps = {p: stamps[p] for p in live if p in stamps}

    if not candidates:
        if new_stamps != stamps:  # dead paths dropped
            atomic_write_bytes(_stamps_path(), orjson.dumps(new_stamps))
        return stats

    bodies: list[tuple[str, str, str]] = []  # (idx_path, name, raw body)
    for p in candidates:
        try:
            bodies.append((p, live[p][0], DRIVER.read_note(p + ".md").content or ""))
        except Exception:
            continue  # unreadable now — unstamped, retried next sweep

    # Embed leg: split candidates into changed vs touch-only by content
    # signature (raw body is fine — the signature strips media itself, so it
    # matches what the write path stored). Without a seeded embed store there
    # is no signature baseline: every candidate counts as changed.
    changed = bodies
    embed_ok = True
    if embed_seeded:
        try:
            from silica.agent.providers import get_embedder
            from silica.kernel.recall.embed import _embed_signature, build_index
            embedder = get_embedder(CONFIG)
            model = getattr(embedder, "model", "")
            changed = [
                (p, n, b) for p, n, b in bodies
                if embed_store.get_content_hash(p)
                != _embed_signature(n, b, folder=p.rsplit("/", 1)[0] if "/" in p else "", model=model)
            ]
            if len(changed) > _RECONCILE_CAP:
                logger.warning(
                    "index sweep found %d changed notes > cap %d — run /embed to rebuild",
                    len(changed), _RECONCILE_CAP,
                )
                embed_ok = False
            elif changed:
                build_index(embedder, changed, store=embed_store, save=False)
                embed_store.save()
                stats.embedded = len(changed)
        except Exception as e:
            logger.debug("sweep embed leg skipped (%s)", e)
            embed_ok = False

    # First-sweep baseline. A note with NO stamp carries no change evidence:
    # only the embed leg can rule on it (content signature). The deterministic
    # legs have no per-note signature, so for them a note already in the store
    # IS the baseline, not a change — refreshing every one of them would
    # rebuild the whole index on the first query after upgrade (3.5s at 758
    # notes, embedder-free) purely to catch pre-upgrade edits that the reconcile
    # this replaced never caught either. Notes ABSENT from a store are still
    # contributed: that ADD leg is what the old cooccur reconcile lacked.
    unstamped = {p for p in candidates if p not in stamps}

    def _pending(changed_notes, known: set[str]) -> list[tuple[str, str, str]]:
        return [(p, n, b) for p, n, b in changed_notes
                if p not in unstamped or p not in known]

    # Deterministic legs: re-contribute changed notes. Replacement semantics,
    # so re-processing never inflates weights.
    if changed and cooccur_store is not None:
        try:
            from silica.kernel.recall.cooccurrence import build_index as cooccur_build
            pending = _pending(changed, set(cooccur_store.paths()))
            if pending:
                cooccur_build(pending, store=cooccur_store,
                              lang=CONFIG.cooccurrence_lang, force=True, save=False)
                cooccur_store.save()
                stats.refreshed += len(pending)
        except Exception as e:
            logger.debug("sweep cooccur leg skipped (%s)", e)
    if changed and lexical_store is not None:
        try:
            pending = _pending(changed, set(lexical_store.paths()))
            for p, n, b in pending:
                lexical_store.upsert(p, n, b)
            if pending:
                lexical_store.save()
                stats.refreshed += len(pending)
        except Exception as e:
            logger.debug("sweep lexical leg skipped (%s)", e)

    # Stamp what settled. Touch-only candidates always stamp; changed ones
    # only once the embed leg took them (or there is no embed leg) — an
    # unstamped path is exactly "retry me next sweep".
    changed_paths = {p for p, _n, _b in changed}
    for p, _n, _b in bodies:
        if p in changed_paths and embed_seeded and not embed_ok:
            continue
        new_stamps[p] = live[p][1]
        stats.stamped += 1
    if new_stamps != stamps:
        atomic_write_bytes(_stamps_path(), orjson.dumps(new_stamps))
    return stats
