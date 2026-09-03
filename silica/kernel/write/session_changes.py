# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""What changed in the vault — the ledger behind the GUI's Changes list, the
REPL's ``/changes`` and the MCP ``silica_changes`` tool.

Two layers over the same event, a note's FIRST touch by this process:

* **In memory, keyed by vault**: vault path -> note path -> how the note stood
  before this process touched it. The *after* side is deliberately not stored;
  it is read off disk when a diff is asked for. So a note written five times in
  one run still yields one honest row, an edit you made yourself in Obsidian
  shows up in the same diff, and an /undo that puts the bytes back makes the
  row disappear on its own. The vault is in the key because a ``/vault`` switch
  used to read the after side of the OLD vault's rows in the NEW vault: every
  note the previous session had touched came back as "deleted".
* **On disk, per vault** (``<index namespace>/changes.db``): the same baseline
  rows stamped with the writing session, so a second process on the same vault
  — another MCP client, a running pipeline — can read what changed since it
  last looked (``history``). Per-vault state follows the vault (ADR-0014): no
  vault column, and no central file every process would contend on.

The driver is the only writer here. That is the whole reason this is one file and
not a reporting duty spread across the tool surface: interactive patch, bulk
nucleation, a move's link rewrites and delete all reach disk through the same
four methods, so they all land in the list without any of them knowing it exists.

``rows()`` lives here rather than in the web server that first grew it, because
the REPL needs the same tally and a CLI that imported the server would drag
FastAPI in to count two integers.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ponytail: baseline bodies live in memory, oldest dropped past the cap. A note is
# a few KB, so 200 of them is single-digit MB; the on-disk ledger below is the
# longer view for anyone who needs more than this process's own writes.
MAX_TRACKED = 200

# ponytail: on-disk rows per vault, oldest dropped. A row carries one note body,
# so 2000 rows is tens of MB at the very worst; raise it, or move the bodies to
# the checkpoints DB, when a vault's feed needs more than ~2000 first-touches back.
HISTORY_CAP = 2000

# One id per process: the MCP server is one process per client and the REPL one
# per terminal, so "process" is what a session is here. Not the narration sid —
# that exists only once a chat has started, and the driver writes before and
# without one. ponytail: a host that hands over its own session id (Claude Code
# does not, over stdio) could replace this.
SESSION = f"{int(time.time())}-{os.getpid()}"

_lock = threading.Lock()


@dataclass(frozen=True)
class Baseline:
    """How a note stood before this session. ``before`` is None when the note did
    not exist yet; ``origin`` is the path it was moved from, if it was moved."""

    before: str | None
    origin: str | None = None


# vault key -> (note path -> Baseline); each inner dict insertion-ordered, oldest first
_baselines: dict[str, dict[str, Baseline]] = {}


def _vault_key() -> str:
    """The active vault as the ledger keys it: resolved path, '' when unset."""
    from silica.config import CONFIG

    raw = (getattr(CONFIG, "vault_path", "") or "").strip()
    return str(Path(raw).resolve()) if raw else ""


def touched(path: str, prior: str | None) -> None:
    """Record the state a note was in before this session's FIRST write to it.

    Later writes to the same note are no-ops: the baseline is what the diff is
    measured against, and it must not move under the reader.
    """
    vault = _vault_key()
    with _lock:
        ledger = _baselines.setdefault(vault, {})
        if path in ledger:
            return
        ledger[path] = Baseline(before=prior)
        while len(ledger) > MAX_TRACKED:
            ledger.pop(next(iter(ledger)))
    _persist_touch(vault, path, prior)


def touched_from_disk(path: str) -> None:
    """Same, for a backend that writes through someone else.

    The ws backend hands the write to the Obsidian plugin and keeps no copy of
    the bytes, but the plugin has the same folder open that CONFIG points at —
    so the baseline is one open() away and costs no round-trip. Call it BEFORE
    the write, or the baseline is the result instead of the starting point.
    """
    vault = _vault_key()
    with _lock:
        if path in _baselines.get(vault, {}):
            return  # already tracked — don't pay for a read that would be dropped
    from silica.config import CONFIG

    try:
        prior = (Path(CONFIG.vault_path) / path).read_text(encoding="utf-8")
    except OSError:
        prior = None  # not there yet: this write is a create
    touched(path, prior)


def renamed(old: str, new: str) -> None:
    """Follow a moved note, so it keeps one row instead of splitting into a
    phantom pair (a deletion at the old path, a creation at the new one)."""
    vault = _vault_key()
    with _lock:
        ledger = _baselines.get(vault)
        base = ledger.pop(old, None) if ledger else None
        if base is None:
            return
        ledger[new] = Baseline(before=base.before, origin=base.origin or old)
    _persist_rename(vault, old, new)


def snapshot() -> dict[str, Baseline]:
    """This process's baselines for the active vault."""
    vault = _vault_key()
    with _lock:
        return dict(_baselines.get(vault, {}))


def clear() -> None:
    """Drop this process's ledger for every vault and close the ledger
    connections (test isolation). The on-disk history is left alone: it is the
    other sessions' view as much as ours."""
    with _lock:
        _baselines.clear()
    with _db_lock:
        for conn in _conns.values():
            try:
                conn.close()
            except sqlite3.Error:
                pass  # a connection already broken has nothing left to release
        _conns.clear()


# ---------------------------------------------------------------------------
# On-disk ledger, one per vault
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS changes (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    ts      REAL NOT NULL,
    path    TEXT NOT NULL,
    before  TEXT,
    origin  TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_changes_session_path ON changes(session, path);
"""


def _db_path(vault: str) -> Path | None:
    if not vault:
        return None  # no vault configured: nothing another session could look up
    from silica.kernel.recall.paths import index_dir_for

    return index_dir_for(vault) / "changes.db"


# One connection per ledger file for the life of the process, shared across
# threads under _db_lock. Measured 2026-09-01: open-write-close per touch cost
# 20.7 ms, because closing the last connection checkpoints the WAL and fsyncs;
# a kept connection leaves checkpointing to sqlite's own 1000-page cadence and
# the same touch costs well under a millisecond. Separate processes still
# serialise through the WAL and the busy timeout; no app-level lock between them.
_conns: dict[str, sqlite3.Connection] = {}
_db_lock = threading.Lock()


def _connect(p: Path) -> sqlite3.Connection:
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=5.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    # NORMAL under WAL: a commit no longer fsyncs; a power cut can lose the
    # last rows of a feed, never corrupt it, and the notes themselves are
    # already on disk — full sync would pay one fsync per note in a bulk run.
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    return conn


@contextmanager
def _db(vault: str):
    """The vault's ledger connection, opened on first use; None without a vault.
    Holds _db_lock for the body, so a statement pair (delete + update) is one
    unit against the other threads of this process."""
    p = _db_path(vault)
    if p is None:
        yield None
        return
    with _db_lock:
        conn = _conns.get(str(p))
        if conn is None:
            conn = _connect(p)
            _conns[str(p)] = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def _persist_touch(vault: str, path: str, prior: str | None) -> None:
    try:
        with _db(vault) as conn:
            if conn is None:
                return
            conn.execute(
                "INSERT OR IGNORE INTO changes (session, ts, path, before) VALUES (?, ?, ?, ?)",
                (SESSION, time.time(), path, prior))
            conn.execute(
                "DELETE FROM changes WHERE id <= "
                "(SELECT id FROM changes ORDER BY id DESC LIMIT 1 OFFSET ?)",
                (HISTORY_CAP,))
    except (sqlite3.Error, OSError) as e:
        # The note write already landed; this row is only the feed other
        # sessions read. Losing it costs them a stale "what changed"; raising
        # would fail a write that succeeded. So log and go on.
        logger.warning("change ledger: could not record %s (%s)", path, e)


def _persist_rename(vault: str, old: str, new: str) -> None:
    try:
        with _db(vault) as conn:
            if conn is None:
                return
            # Same shape as the in-memory move: the row at `new` (if this
            # session had one) is replaced by the followed row.
            conn.execute("DELETE FROM changes WHERE session = ? AND path = ?", (SESSION, new))
            conn.execute(
                "UPDATE changes SET path = ?, origin = COALESCE(origin, ?) "
                "WHERE session = ? AND path = ?", (new, old, SESSION, old))
    except (sqlite3.Error, OSError) as e:
        # Same reasoning as _persist_touch: the move happened, the feed is late.
        logger.warning("change ledger: could not follow move %s -> %s (%s)", old, new, e)


def history(since: float | None = None, limit: int = 200) -> list[dict]:
    """What ANY session changed in the active vault, oldest first: ``rows()``'s
    columns plus ``session``, ``ts`` (epoch seconds) and ``mine``.

    ``since`` drops rows first touched at or before that epoch; ``limit`` keeps
    the newest. Diffs are measured exactly as in ``rows()``, baseline against
    the file as it stands NOW, so a note one session created and another
    deleted reads as nothing happened: the ledger states the vault, not the past.
    """
    vault = _vault_key()
    if not vault:
        return []
    try:
        with _db(vault) as conn:
            if conn is None:
                return []
            got = conn.execute(
                "SELECT session, ts, path, before, origin FROM changes "
                "WHERE ts > ? ORDER BY id DESC LIMIT ?",
                (since or 0.0, max(int(limit), 0))).fetchall()
    except (sqlite3.Error, OSError) as e:
        # An unreadable ledger is an empty feed, not a failed tool call: the
        # caller's own rows() still stand, and doctor is where a broken index
        # namespace gets reported.
        logger.warning("change ledger: unreadable for %s (%s)", vault, e)
        return []
    out: list[dict] = []
    for session, ts, path, before, origin in reversed(got):
        row = _row(path, Baseline(before=before, origin=origin))
        if row is not None:
            out.append({**row, "session": session, "ts": ts, "mine": session == SESSION})
    return out


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def tally(before: str, after: str) -> tuple[int, int]:
    """Lines added and removed — the same opcodes a unified diff walks."""
    import difflib

    added = removed = 0
    sm = difflib.SequenceMatcher(None, before.splitlines(), after.splitlines(), autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, removed


def kind(before: str | None, after: str | None, origin: str | None, changed: bool) -> str:
    if before is None:
        return "created"
    if after is None:
        return "deleted"
    return "moved" if origin and not changed else "modified"


def current_text(rel: str) -> str | None:
    """The note's bytes as they are now, or None if it is no longer there."""
    from silica.config import CONFIG

    try:
        return (Path(CONFIG.vault_path) / rel).read_text(encoding="utf-8")
    except OSError:
        return None


def _row(path: str, base: Baseline) -> dict | None:
    after = current_text(path)
    if base.before is None and after is None:
        return None  # created and then rolled back: nothing happened
    added, removed = tally(base.before or "", after or "")
    if not (added or removed or base.origin):
        return None  # written with the same bytes it already had
    return {
        "path": path,
        "kind": kind(base.before, after, base.origin, bool(added or removed)),
        "added": added,
        "removed": removed,
        "from": base.origin,
    }


def rows() -> list[dict]:
    """Every note this session changed in the active vault, oldest first.

    The *after* side is read off disk here and never remembered, so the list is
    not a claim about the past: it is the difference between the baseline and the
    file as it stands right now, and an ``/undo`` that puts the bytes back empties
    a row on its own instead of waiting for someone to remove it.
    """
    out: list[dict] = []
    for path, base in snapshot().items():
        row = _row(path, base)
        if row is not None:
            out.append(row)
    return out
