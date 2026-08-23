# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

# silica/kernel/write/undo_journal.py
from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from silica.kernel.write.ops import InverseOp, InverseOpKind

logger = logging.getLogger(__name__)

_DEFAULT_JOURNAL_PATH = Path.home() / ".silica" / "undo_journal.db"


class UndoJournalStore:
    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else _DEFAULT_JOURNAL_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # WAL + per-thread connections; sqlite's busy_timeout serialises
        # writers, so no app-level lock. A thread's conn is closed only by GC when
        # the thread dies — fine for the GUI's small to_thread pool.
        self._local = threading.local()
        try:
            self._init_schema()
        except sqlite3.DatabaseError as e:
            # A corrupt journal must not brick startup or the /revert of future
            # runs. Quarantine it and start fresh; the durable backstop for older
            # history is git (SILICA_GIT_COMMIT=auto), not this file.
            logger.warning(
                "undo journal at %s is corrupt (%s); quarantining and starting fresh",
                self._path, e,
            )
            conn = getattr(self._local, "conn", None)
            if conn is not None:
                conn.close()
                self._local.conn = None
            # paths.quarantine: timestamped, never clobbered — the flat
            # ".corrupt" rename here overwrote the previous preserved copy on
            # the second corruption, and doctor's *.corrupt.* glob never saw it.
            from silica.kernel.recall.paths import quarantine
            quarantine(self._path)
            for suffix in ("-wal", "-shm"):
                # a stale WAL sidecar must not be replayed into the fresh db
                Path(str(self._path) + suffix).unlink(missing_ok=True)
            self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._path))
            try:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
            except sqlite3.DatabaseError:
                conn.close()
                raise
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        self._conn().executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id        TEXT PRIMARY KEY,
                source        TEXT,
                vault         TEXT,
                started_at    REAL NOT NULL,
                reverted_at   REAL,
                ledger_run_id TEXT
            );
            CREATE TABLE IF NOT EXISTS inverses (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id        TEXT NOT NULL,
                path          TEXT NOT NULL,
                kind          TEXT NOT NULL,
                version       INTEGER,
                prior_content TEXT,
                post_hash     TEXT,
                to_path       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_inverses_run ON inverses(run_id);
            """
        )
        # Migration: pre-scoping DBs lack `vault`. Legacy rows stay NULL, so a
        # vault-filtered last_active_run() never surfaces them — foreign/stale
        # runs from a deleted or reorganised vault retire themselves.
        conn = self._conn()
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
        if "vault" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN vault TEXT")
        # Migration: pre-move_back DBs lack `to_path` (only restore_version /
        # delete_created / recreate_deleted were ever journalled). Legacy rows
        # stay NULL — those kinds don't use it.
        inv_cols = {r["name"] for r in conn.execute("PRAGMA table_info(inverses)")}
        if "to_path" not in inv_cols:
            conn.execute("ALTER TABLE inverses ADD COLUMN to_path TEXT")
        # Migration: pre-source-revert DBs lack `ledger_run_id`, the FSM's own
        # run id that the provenance ledger keys on (ADR-0028). Legacy rows stay NULL, so
        # /revert --source reports them as not in the journal instead of
        # guessing a join by file name; /revert <run-id> still reaches them.
        if "ledger_run_id" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN ledger_run_id TEXT")
        conn.commit()

    def start_run(self, source: str | None = None, vault: str | None = None,
                  ledger_run_id: str | None = None) -> str:
        run_id = uuid.uuid4().hex
        conn = self._conn()
        conn.execute(
            "INSERT INTO runs (run_id, source, vault, started_at, ledger_run_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, source, vault, time.time(), ledger_run_id),
        )
        conn.commit()
        return run_id

    def runs_for_ledger(self, ledger_run_ids: list[str],
                        vault: str | None = None) -> list[str]:
        """Un-reverted journal runs opened under these ledger run ids, newest
        first: a later run's inverses restore what an earlier one wrote, so
        that is the only order in which replaying both is sound."""
        ids = [i for i in ledger_run_ids if i]
        if not ids:
            return []
        query = ("SELECT run_id FROM runs WHERE reverted_at IS NULL AND ledger_run_id IN (%s)"
                 % ",".join("?" * len(ids)))
        params: list[str] = list(ids)
        if vault is not None:
            query += " AND vault = ?"
            params.append(vault)
        query += " ORDER BY started_at DESC, rowid DESC"
        return [r["run_id"] for r in self._conn().execute(query, params).fetchall()]

    def record(self, run_id: str, inverse: InverseOp, post_hash: str | None) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO inverses (run_id, path, kind, version, prior_content, post_hash, to_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, inverse.path, inverse.kind.value, inverse.version,
             inverse.prior_content, post_hash, inverse.to_path),
        )
        conn.commit()

    def last_active_run(self, vault: str | None = None) -> str | None:
        """Most recent un-reverted run that has inverses.

        When `vault` is given, only runs stamped with that vault are eligible —
        so /revert never walks back into another vault's (or a deleted vault's)
        history. `vault=None` keeps the unscoped behaviour (tests, legacy calls).
        """
        query = (
            "SELECT r.run_id FROM runs r WHERE r.reverted_at IS NULL "
            "AND EXISTS (SELECT 1 FROM inverses i WHERE i.run_id = r.run_id)"
        )
        params: list[str] = []
        if vault is not None:
            query += " AND r.vault = ?"
            params.append(vault)
        query += " ORDER BY r.started_at DESC, r.rowid DESC LIMIT 1"
        row = self._conn().execute(query, params).fetchone()
        return row["run_id"] if row else None

    def inverses_for(self, run_id: str) -> list[tuple[InverseOp, str | None]]:
        rows = self._conn().execute(
            "SELECT path, kind, version, prior_content, post_hash, to_path "
            "FROM inverses WHERE run_id = ? ORDER BY id DESC",
            (run_id,),
        ).fetchall()
        out: list[tuple[InverseOp, str | None]] = []
        for r in rows:
            inv = InverseOp(
                kind=InverseOpKind(r["kind"]), path=r["path"],
                version=r["version"], prior_content=r["prior_content"],
                to_path=r["to_path"],
            )
            out.append((inv, r["post_hash"]))
        return out

    def run_info(self, run_id: str) -> dict | None:
        """`{source, vault, started_at}` for a run, or None when unknown.

        /revert shows this next to the id: the journal's run ids live in a
        different id-space than the progress ledger's (log.md), so a bare id
        never tells the user WHAT they are about to revert.
        """
        row = self._conn().execute(
            "SELECT source, vault, started_at, ledger_run_id FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return {"source": row["source"], "vault": row["vault"],
                "started_at": row["started_at"], "ledger_run_id": row["ledger_run_id"]}

    def refresh_post_hashes(self, run_id: str) -> int:
        """Re-hash every journalled path from the vault's CURRENT content.

        FINALIZE records post-write hashes, but the coordinator's end-of-run
        passes (dangling-link sweep, orphan repairs) edit the run's notes AFTER
        that — so /revert's "modified since inject" guard refused the run's own
        writes. Called once when the run is truly over: the state on disk at
        that moment is the state the guard should protect. Returns the number
        of rows updated.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT DISTINCT path FROM inverses WHERE run_id = ?", (run_id,)
        ).fetchall()
        updated = 0
        for r in rows:
            path = r["path"]
            try:
                current = DRIVER.read_note(path).content
                new_hash: str | None = _content_hash(current)
            except Exception:
                new_hash = None  # note absent — the stale guard handles it
            conn.execute(
                "UPDATE inverses SET post_hash = ? WHERE run_id = ? AND path = ?",
                (new_hash, run_id, path),
            )
            updated += 1
        conn.commit()
        return updated

    def mark_reverted(self, run_id: str) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE runs SET reverted_at = ? WHERE run_id = ?", (time.time(), run_id)
        )
        conn.commit()


_store: UndoJournalStore | None = None


def get_undo_journal(path: Path | str | None = None) -> UndoJournalStore:
    global _store
    if _store is None:
        _store = UndoJournalStore(path)
    return _store


import hashlib as _hashlib

from silica.driver import DRIVER


def _content_hash(text: str | None) -> str:
    return _hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def revert_run(run_id: str, *, store: UndoJournalStore | None = None,
               only_paths: set[str] | None = None) -> dict:
    """Replay a run's inverses LIFO, refusing notes modified since the inject.

    Version guard: re-read each note, hash it. If recorded post_hash exists and
    the current hash differs (note edited since inject), skip it — don't clobber
    newer work. Mark the run reverted when done.

    `only_paths` scopes the replay to those notes (any spelling; folded through
    provenance.note_key): the rest are skipped as outside the scope and the run
    stays open, since what was not replayed is still undoable. A run scoped to
    its whole footprint closes as an unscoped one does.
    """
    from silica.kernel.write.provenance import note_key
    from silica.tools.wrapped import silica_restore

    store = store or get_undo_journal()
    entries = store.inverses_for(run_id)  # LIFO
    scope = {note_key(p) for p in only_paths} if only_paths is not None else None
    reverted: list[str] = []
    skipped: list[dict] = []
    stale: list[dict] = []
    errors: list[dict] = []
    applied_paths: set[str] = set()
    out_of_scope = 0

    for inv, post_hash in entries:
        if scope is not None and note_key(inv.path) not in scope:
            skipped.append({"path": inv.path, "reason": "outside the requested scope"})
            out_of_scope += 1
            continue
        try:
            current = DRIVER.read_note(inv.path).content
            cur_hash: str | None = _content_hash(current)
        except Exception:
            cur_hash = None  # note absent

        # Stale (B): the target note no longer exists in this vault, so there is
        # nothing to restore or delete — the journal describes a vault that was
        # reorganised or replaced. Report it honestly instead of counting an
        # empty overwrite as an error or an absent delete as a revert.
        # (recreate_deleted is exempt: an absent note is its expected precondition.)
        if cur_hash is None and inv.kind in (
            InverseOpKind.restore_version, InverseOpKind.delete_created
        ):
            stale.append({"path": inv.path, "reason": "note absent (vault changed)"})
            continue

        # The guard belongs to the NEWEST inverse per path only: two chunks
        # patching one note both carry the FINAL content's hash, so the older
        # link of the chain can never match once the newer one applied. After
        # the newest passes, the older ones are the chain's own history.
        if (
            inv.path not in applied_paths
            and post_hash is not None and cur_hash is not None
            and cur_hash != post_hash
        ):
            skipped.append({"path": inv.path, "reason": "modified since inject"})
            continue

        try:
            res = silica_restore(txn_id=run_id, inverses=[inv.model_dump()])
            if res["errors"]:
                # silica_restore swallows per-op failures into its return value;
                # route them to errors instead of miscounting as reverted.
                errors.append({"path": inv.path, "error": "; ".join(res["errors"])})
            else:
                reverted.append(inv.path)
                applied_paths.add(inv.path)
        except Exception as e:
            errors.append({"path": inv.path, "error": str(e)})

    # A partial revert can delete notes that surviving (skipped) notes still
    # link to — the run's own hub kept [[links]] to notes the revert removed.
    # Sweep only the survivors this run wrote: they are journalled paths, so
    # the edit stays inside the run's own footprint.
    if reverted and skipped:
        try:
            from silica.kernel.link.sweep import sweep_dangling_links

            sweep_dangling_links(sorted({s["path"] for s in skipped}))
        except Exception as e:
            logger.debug("revert: survivor link sweep failed (non-fatal): %s", e)

    if not out_of_scope:
        store.mark_reverted(run_id)
    return {"run_id": run_id, "reverted": reverted, "skipped": skipped,
            "stale": stale, "errors": errors}


def revert_source(source: str, *, vault: str | None,
                  store: UndoJournalStore | None = None) -> dict:
    """Undo every journalled run that derived notes from `source` (ADR-0028).

    The join is the ledger's run id, which the journal records at start_run:
    the ledger says which runs touched the source and which notes each of
    them attributed to it, the journal holds the inverses and the guard.
    Newest run first, each scoped to its own ledger notes, so a hub another
    source also patched is left to the guard rather than rewound past the
    other source's work. Ledger runs with no active journal row (legacy rows
    from before the join, a pruned journal, a run already reverted by id) are
    reported under `unrevertable` with their note count, never acted on: the
    ledger knows the paths but not the inverses, and deleting on a guess is
    the one thing a revert must never do.
    """
    from silica.kernel.write.provenance import read_records

    store = store or get_undo_journal()
    notes_by_ledger_run: dict[str, set[str]] = {}
    for r in read_records(source, vault_path=vault):
        rid = r.get("run_id")
        if rid:
            notes_by_ledger_run.setdefault(rid, set()).update(r.get("notes") or [])
    if not notes_by_ledger_run:
        return {"source": source, "runs": [], "unrevertable": []}

    runs: list[dict] = []
    covered: set[str] = set()
    for run_id in store.runs_for_ledger(list(notes_by_ledger_run), vault=vault):
        ledger_id = (store.run_info(run_id) or {}).get("ledger_run_id") or ""
        res = revert_run(run_id, store=store,
                         only_paths=notes_by_ledger_run.get(ledger_id, set()))
        res["ledger_run_id"] = ledger_id
        runs.append(res)
        covered.add(ledger_id)
    unrevertable = [{"run_id": rid, "notes": len(notes)}
                    for rid, notes in notes_by_ledger_run.items() if rid not in covered]
    return {"source": source, "runs": runs, "unrevertable": unrevertable}
