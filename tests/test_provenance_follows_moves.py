# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""A moved note keeps its provenance.

The ledger keys notes by bare path and move() rewrote every wikilink but not
the ledger, so after a rename the patch executor's idempotency check answered
False and the next nucleate of the source re-appended into the note it had
already written; the drift report kept naming a path that no longer existed.
"""
from __future__ import annotations

from silica.kernel.write.provenance import append_record, note_authored_by


def test_fs_move_rewrites_the_ledger(tmp_vault):
    from silica.config import CONFIG
    from silica.driver.fs_backend import ObsidianFSBackend

    tmp_vault.note("Concepts/A.md", "---\ntitle: A\n---\n\nbody\n")
    append_record("lec.md", "sha1", "r1", ["Concepts/A"])
    backend = ObsidianFSBackend(CONFIG.vault_path)
    backend._ensure_index()

    backend.move("Concepts/A.md", "Archive/A.md")

    assert note_authored_by("Archive/A.md", "lec.md")
    assert not note_authored_by("Concepts/A.md", "lec.md")


def test_ws_move_rewrites_the_ledger(tmp_vault, monkeypatch):
    """The plugin backend moves over RPC; the ledger lives on disk either way."""
    from silica.driver import ws_backend

    append_record("lec.md", "sha1", "r1", ["Concepts/A"])
    backend = ws_backend.ObsidianWSBackend.__new__(ws_backend.ObsidianWSBackend)
    monkeypatch.setattr(backend, "_path_arg", lambda ref: str(ref), raising=False)
    monkeypatch.setattr(backend, "_rpc", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ws_backend.session_changes, "touched_from_disk", lambda p: None)
    monkeypatch.setattr(ws_backend.session_changes, "renamed", lambda a, b: None)

    backend.move("Concepts/A.md", "Archive/A.md")

    assert note_authored_by("Archive/A.md", "lec.md")
