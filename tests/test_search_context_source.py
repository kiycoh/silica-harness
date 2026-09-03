# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""silica_search_context falls through to the repo's source files.

The driver indexes markdown only, so in a codebase vault an exact-string probe
for a symbol answered `{"hits": [], "notes_matched": 0}` while grep found it
in seven lines (measured 2026-09-03 on `follow_superseded`). An agent that
follows the skill ("exact strings: search_context") concluded the symbol did
not exist. The reply now scans tracked source too and says what it scanned.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from silica.tools import atomic


def _repo(tmp_path: Path, monkeypatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "pkg.py").write_text("def follow_superseded_xyz(path):\n    return path\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg.py"], cwd=repo, check=True)
    from silica.config import CONFIG
    from silica.kernel.recall.paths import clear_repo_root_cache

    monkeypatch.setattr(CONFIG, "vault_path", str(repo))
    clear_repo_root_cache()
    monkeypatch.setattr(atomic.DRIVER, "search_context", lambda q: [], raising=False)
    return repo


def test_source_hits_fill_in_when_notes_are_silent(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    out = atomic.silica_search_context("follow_superseded_xyz")
    assert out["notes_matched"] == 0
    assert out["scanned"] == ["notes", "source"]
    assert out["hits"] == [{"kind": "source", "path": "pkg.py", "line": 1,
                            "snippet": "def follow_superseded_xyz(path):"}]


def test_source_scan_reports_a_true_absence(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    out = atomic.silica_search_context("no_such_token_anywhere")
    assert out["hits"] == []
    assert out["scanned"] == ["notes", "source"]


def test_no_repo_keeps_the_markdown_only_reply(monkeypatch):
    monkeypatch.setattr(atomic.DRIVER, "search_context", lambda q: [], raising=False)
    out = atomic.silica_search_context("anything")
    assert out == {"hits": [], "notes_matched": 0}
