# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`/revert --source <file>`: every run that derived notes from one source.

The undo journal can revert (it holds the inverses and the modified-since
guard) but indexes runs by its own uuid and names only the first file of a
multi-file run. The provenance ledger indexes by source and by the FSM's run
id, which the journal never recorded, so the two could not be joined. The
journal now carries the ledger's run id; a source reverts through the join,
newest run first, touching only the notes the ledger attributes to it, and
what the journal does not hold is reported rather than guessed at.
"""
from __future__ import annotations

import hashlib
import sqlite3

from silica.kernel.write.ops import InverseOp, InverseOpKind
from silica.kernel.write.provenance import append_record
from silica.kernel.write.undo_journal import UndoJournalStore, revert_run, revert_source


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestTheJournalKnowsTheLedgerRun:
    def test_start_run_records_the_ledger_run_id(self, tmp_path):
        store = UndoJournalStore(tmp_path / "j.db")
        run = store.start_run("inbox/a.md", vault="/v", ledger_run_id="fsm-1")
        assert store.runs_for_ledger(["fsm-1"], vault="/v") == [run]

    def test_a_legacy_journal_gains_the_column(self, tmp_path):
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, source TEXT, vault TEXT, "
            "started_at REAL NOT NULL, reverted_at REAL);"
            "CREATE TABLE inverses (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, "
            "path TEXT NOT NULL, kind TEXT NOT NULL, version INTEGER, prior_content TEXT, "
            "post_hash TEXT, to_path TEXT);"
            "INSERT INTO runs VALUES ('old', 'inbox/a.md', '/v', 1.0, NULL);"
        )
        conn.commit()
        conn.close()

        store = UndoJournalStore(db)

        assert store.runs_for_ledger(["anything"], vault="/v") == []
        assert store.run_info("old")["source"] == "inbox/a.md"

    def test_runs_for_ledger_is_newest_first_and_skips_reverted(self, tmp_path):
        store = UndoJournalStore(tmp_path / "j.db")
        first = store.start_run("a.md", vault="/v", ledger_run_id="fsm-1")
        second = store.start_run("a.md", vault="/v", ledger_run_id="fsm-2")
        other_vault = store.start_run("a.md", vault="/w", ledger_run_id="fsm-3")
        store.mark_reverted(first)

        assert store.runs_for_ledger(["fsm-1", "fsm-2", "fsm-3"], vault="/v") == [second]
        assert other_vault not in store.runs_for_ledger(["fsm-3"], vault="/v")


class TestRevertRunCanBeScoped:
    def test_only_paths_leaves_the_other_notes_alone(self, tmp_vault, tmp_path):
        ada = tmp_vault.note("People/Ada.md", "PATCHED ada")
        grace = tmp_vault.note("People/Grace.md", "PATCHED grace")
        store = UndoJournalStore(tmp_path / "j.db")
        run_id = store.start_run("inbox/meeting.md")
        store.record(run_id, InverseOp(kind=InverseOpKind.restore_version, path=ada,
                                       prior_content="ORIGINAL ada"), post_hash=_h("PATCHED ada"))
        store.record(run_id, InverseOp(kind=InverseOpKind.restore_version, path=grace,
                                       prior_content="ORIGINAL grace"), post_hash=_h("PATCHED grace"))

        result = revert_run(run_id, store=store, only_paths={"People/Ada"})

        assert tmp_vault.read(ada) == "ORIGINAL ada"
        assert tmp_vault.read(grace) == "PATCHED grace"
        assert result["reverted"] == [ada]
        assert any(s["path"] == grace and "outside" in s["reason"] for s in result["skipped"])
        # A scoped revert leaves the run open: the rest of it is still undoable.
        assert store.last_active_run() == run_id


class TestRevertSource:
    def test_reverts_the_notes_the_ledger_attributes_newest_run_first(self, tmp_vault, tmp_path):
        from silica.config import CONFIG

        a = tmp_vault.note("Concepts/A.md", "v2 of A")
        b = tmp_vault.note("Concepts/B.md", "B from run 1")
        hub = tmp_vault.note("Hubs/Hub.md", "hub patched by other")
        store = UndoJournalStore(tmp_path / "j.db")
        vault = CONFIG.vault_path

        r1 = store.start_run("inbox/lec.md", vault=vault, ledger_run_id="fsm-1")
        store.record(r1, InverseOp(kind=InverseOpKind.delete_created, path=a), post_hash=_h("v1 of A"))
        store.record(r1, InverseOp(kind=InverseOpKind.delete_created, path=b), post_hash=_h("B from run 1"))
        append_record("lec.md", "sha-v1", "fsm-1", ["Concepts/A", "Concepts/B"])

        r2 = store.start_run("inbox/lec.md", vault=vault, ledger_run_id="fsm-2")
        store.record(r2, InverseOp(kind=InverseOpKind.restore_version, path=a,
                                   prior_content="v1 of A"), post_hash=_h("v2 of A"))
        store.record(r2, InverseOp(kind=InverseOpKind.restore_version, path=hub,
                                   prior_content="hub before"), post_hash=_h("hub patched by lec"))
        append_record("lec.md", "sha-v2", "fsm-2", ["Concepts/A", "Hubs/Hub"])

        res = revert_source("lec.md", vault=vault, store=store)

        assert [r["run_id"] for r in res["runs"]] == [r2, r1]
        import os
        assert not os.path.exists(a)          # r2 restored v1, then r1 deleted the creation
        assert not os.path.exists(b)
        assert tmp_vault.read(hub) == "hub patched by other"   # modified since: guarded
        assert res["unrevertable"] == []

    def test_ledger_runs_the_journal_does_not_hold_are_reported_not_guessed(self, tmp_vault, tmp_path):
        from silica.config import CONFIG

        tmp_vault.note("Concepts/A.md", "A")
        store = UndoJournalStore(tmp_path / "j.db")
        append_record("lec.md", "sha-v1", "fsm-legacy", ["Concepts/A"])

        res = revert_source("lec.md", vault=CONFIG.vault_path, store=store)

        assert res["runs"] == []
        assert res["unrevertable"] == [{"run_id": "fsm-legacy", "notes": 1}]
        assert tmp_vault.read(str(__import__("pathlib").Path(CONFIG.vault_path) / "Concepts/A.md")) == "A"

    def test_an_unknown_source_reverts_nothing(self, tmp_vault, tmp_path):
        from silica.config import CONFIG

        store = UndoJournalStore(tmp_path / "j.db")
        res = revert_source("never.md", vault=CONFIG.vault_path, store=store)
        assert res == {"source": "never.md", "runs": [], "unrevertable": []}


class TestTheReplCommand:
    def test_revert_source_flag_dispatches_by_source(self, monkeypatch, capsys):
        from silica.cli import _handle_direct_shortcut

        calls = {}

        def fake(source, **kw):
            calls["source"] = source
            return {"source": source, "runs": [{"run_id": "r2", "reverted": ["a"], "skipped": [],
                                                "stale": [], "errors": []}],
                    "unrevertable": [{"run_id": "fsm-legacy", "notes": 3}]}

        monkeypatch.setattr("silica.kernel.write.undo_journal.revert_source", fake)

        assert _handle_direct_shortcut("/revert --source lec.md", []) is True
        assert calls["source"] == "lec.md"
        out = capsys.readouterr().out
        assert "1 run(s)" in out and "3 note(s)" in out
