# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""The change ledger is keyed by vault and outlives the process.

session_changes used to be one process-wide dict with no vault in the key:
after a /vault switch the after side was read from the NEW vault at the OLD
relative paths, so every note the previous vault's session had touched came
back as "deleted" in the new one. And a second process on the same vault (a
second MCP client, a running pipeline) could not see what the first changed.
"""
from __future__ import annotations

import time

import pytest

from silica.config import CONFIG
from silica.kernel.write import session_changes


@pytest.fixture(autouse=True)
def _clean_ledger():
    session_changes.clear()
    yield
    session_changes.clear()


def _vault(tmp_path, name: str):
    v = tmp_path / name
    v.mkdir()
    return v


def _note(vault, rel: str, body: str) -> None:
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_rows_follow_the_active_vault_across_a_switch(tmp_path, monkeypatch):
    a, b = _vault(tmp_path, "a"), _vault(tmp_path, "b")
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _note(a, "n.md", "one\ntwo\n")
    session_changes.touched("n.md", "one\n")
    assert [r["path"] for r in session_changes.rows()] == ["n.md"]

    monkeypatch.setattr(CONFIG, "vault_path", str(b))
    # b was never touched. Before the vault key, the after side of a/n.md was
    # read at b/n.md, absent, and the row came back as "deleted".
    assert session_changes.rows() == []

    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    assert [r["path"] for r in session_changes.rows()] == ["n.md"]


def test_history_outlives_the_process_ledger(tmp_path, monkeypatch):
    a = _vault(tmp_path, "a")
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _note(a, "n.md", "one\ntwo\n")
    session_changes.touched("n.md", "one\n")

    session_changes.clear()  # what a fresh process starts from
    assert session_changes.rows() == []

    hist = session_changes.history()
    assert [(r["path"], r["kind"], r["added"], r["removed"]) for r in hist] == [
        ("n.md", "modified", 1, 0)]
    assert hist[0]["session"] == session_changes.SESSION
    assert hist[0]["mine"] is True


def test_history_tells_another_session_apart(tmp_path, monkeypatch):
    a = _vault(tmp_path, "a")
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _note(a, "x.md", "x")
    monkeypatch.setattr(session_changes, "SESSION", "other-1")
    session_changes.touched("x.md", None)

    monkeypatch.setattr(session_changes, "SESSION", "me-2")
    session_changes.clear()
    _note(a, "y.md", "y")
    session_changes.touched("y.md", None)

    by = {r["path"]: r for r in session_changes.history()}
    assert by["x.md"]["session"] == "other-1" and by["x.md"]["mine"] is False
    assert by["y.md"]["session"] == "me-2" and by["y.md"]["mine"] is True


def test_history_since_cuts_by_time(tmp_path, monkeypatch):
    a = _vault(tmp_path, "a")
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _note(a, "old.md", "o")
    session_changes.touched("old.md", None)
    ts_old = session_changes.history()[0]["ts"]
    time.sleep(0.01)
    _note(a, "new.md", "n")
    session_changes.touched("new.md", None)

    assert [r["path"] for r in session_changes.history(since=ts_old + 0.001)] == ["new.md"]
    assert {r["path"] for r in session_changes.history()} == {"old.md", "new.md"}


def test_history_keeps_one_row_for_a_move(tmp_path, monkeypatch):
    a = _vault(tmp_path, "a")
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _note(a, "Notes/Draft.md", "same bytes\n")
    session_changes.touched("Inbox/Draft.md", "same bytes\n")
    session_changes.renamed("Inbox/Draft.md", "Notes/Draft.md")

    session_changes.clear()
    hist = session_changes.history()
    assert [(r["path"], r["kind"], r["from"]) for r in hist] == [
        ("Notes/Draft.md", "moved", "Inbox/Draft.md")]


def test_history_is_scoped_to_the_vault(tmp_path, monkeypatch):
    a, b = _vault(tmp_path, "a"), _vault(tmp_path, "b")
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _note(a, "n.md", "n")
    session_changes.touched("n.md", None)

    monkeypatch.setattr(CONFIG, "vault_path", str(b))
    assert session_changes.history() == []


def test_an_unwritable_ledger_never_blocks_the_write(tmp_path, monkeypatch):
    import silica.kernel.recall.paths as paths_mod

    a = _vault(tmp_path, "a")
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    blocker = tmp_path / "home-is-a-file"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr(paths_mod, "_SILICA_HOME", blocker)  # the index dir cannot be created

    _note(a, "n.md", "n")
    session_changes.touched("n.md", None)  # must not raise: the note write already landed
    assert [r["path"] for r in session_changes.rows()] == ["n.md"]
    assert session_changes.history() == []


def test_tool_vault_scope_reads_the_persisted_ledger(tmp_path, monkeypatch):
    from silica.tools.atomic import silica_changes

    a = _vault(tmp_path, "a")
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _note(a, "n.md", "n")
    session_changes.touched("n.md", None)
    session_changes.clear()

    assert silica_changes()["total"] == 0  # this process: nothing (the default is unchanged)
    out = silica_changes(scope="vault")
    assert out["scope"] == "vault" and out["total"] == 1
    assert out["changes"][0]["path"] == "n.md" and out["changes"][0]["mine"] is True
