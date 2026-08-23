"""The signal the explore surfaces poll to learn the vault moved under them.

`vault_version` has to be true in both directions and the second is the one
that bites: a digest that changes on its own turns a 15-second poll into a
rebuild of a 6 MB graph document every 15 seconds, forever. So the stability
tests here are not padding — they are the ones a future "just add the note
count" would fail.

The behaviour on the browser side (offer on the graph, redraw on the cheap
surfaces) is pinned in test_gui_web.py's static-asset checks; this file owns
the server half.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from silica.kernel.recall import sync


@pytest.fixture
def vault(tmp_vault, monkeypatch):
    """Roster re-check debounce off: the driver brakes its folder scan to once
    every 2s (fs_backend._ROSTER_RECHECK_INTERVAL, its own test), which a
    15-second poll never notices and a test writing two notes in a row does."""
    from silica.driver import fs_backend

    monkeypatch.setattr(fs_backend, "_ROSTER_RECHECK_INTERVAL", 0.0)
    tmp_vault.note("A.md", "alpha [[B]]\n")
    tmp_vault.note("sub/B.md", "beta\n")
    return tmp_vault


def _index(name: str) -> Path:
    """The path the STORE resolves, not index_file(name): conftest isolates
    each store by monkeypatching its own `_index_path`, and a test that spelt
    the name again would write somewhere nothing reads."""
    from silica.kernel.recall import cooccurrence, embed, lexical

    return {"embeddings": embed, "cooccurrence": cooccurrence,
            "lexical": lexical}[name]._index_path()


# --- it must not move on its own --------------------------------------------

def test_repeated_calls_agree(vault):
    """The poll's whole premise. A digest over an unsorted `list_files`, or one
    that folded in a clock, would pass every other test here and still rebuild
    the graph on every tick."""
    assert sync.vault_version() == sync.vault_version() == sync.vault_version()


def test_a_read_does_not_move_it(vault):
    """Reading a note updates atime, never mtime; the version must not care."""
    from silica.driver import DRIVER

    before = sync.vault_version()
    DRIVER.read_note("A")
    assert sync.vault_version() == before


# --- it must move when the views would draw differently ----------------------

def test_a_new_note_moves_it(vault):
    before = sync.vault_version()
    vault.note("C.md", "charlie\n")
    assert sync.vault_version() != before


def test_an_edited_body_moves_it(vault, tmp_path):
    """The roster is unchanged here — only a body is. It still has to move: the
    edit can add or drop a wikilink, which is an edge in the graph."""
    before = sync.vault_version()
    vault.write(str(tmp_path / "vault" / "A.md"), "alpha [[B]] and now [[C]]\n")
    assert sync.vault_version() != before


def test_a_deleted_note_moves_it(vault, tmp_path):
    before = sync.vault_version()
    (tmp_path / "vault" / "sub" / "B.md").unlink()
    assert sync.vault_version() != before


@pytest.mark.parametrize("name", ["embeddings", "cooccurrence", "lexical"])
def test_another_process_writing_an_index_moves_it(vault, name):
    """The semantic overlay and the communities are drawn from these files, and
    the writer is usually a DIFFERENT process (a terminal `silica nucleate`),
    so no in-process signal reports them."""
    before = sync.vault_version()
    path = _index(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"notes": {}}')
    assert sync.vault_version() != before, f"{name} index is not watched"


# --- it must never take the poll down ----------------------------------------

def test_a_broken_driver_answers_empty(vault, monkeypatch):
    """A poll that raises stops polling, and the caller cannot tell a broken
    vault from an unchanged one — so "" has to mean "nothing to say"."""
    from types import SimpleNamespace

    monkeypatch.setattr("silica.driver.DRIVER", SimpleNamespace(
        list_files=lambda folder="": (_ for _ in ()).throw(OSError("vault gone")),
    ))
    assert sync.vault_version() == ""


def test_a_driver_without_mtimes_still_answers(vault, monkeypatch):
    """The ws backend (the Obsidian plugin bridge) has no `mtime_of`. It gets a
    roster-only digest rather than nothing, the same abstention the sweep makes."""
    from types import SimpleNamespace

    from silica.driver.base import NoteRef

    monkeypatch.setattr("silica.driver.DRIVER", SimpleNamespace(
        list_files=lambda folder="": [NoteRef("A", "A.md")],
    ))
    first = sync.vault_version()
    assert first
    assert sync.vault_version() == first


# --- and it must reach the browser -------------------------------------------

def test_endpoint_serves_the_version(vault):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from silica.ui.web import server

    with TestClient(server.app) as client:
        body = client.get("/vault_version").json()
    assert body["version"] == sync.vault_version()


def test_endpoint_does_not_rebuild_indexes(vault, monkeypatch):
    """A version is a report, not a repair. If the poll swept it would spend
    embedder calls every 15 seconds for as long as a browser tab is open."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from silica.ui.web import server

    monkeypatch.setattr(sync, "sweep", lambda **kw: pytest.fail("the poll swept"))
    with TestClient(server.app) as client:
        assert client.get("/vault_version").status_code == 200
