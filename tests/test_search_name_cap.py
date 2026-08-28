# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""silica_search ranks name matches and caps them.

Substring-over-names is unbounded like the body grep: "e" measured 575 paths /
41k chars on a 719-note vault, and a name lookup answering with 575 names has
answered nothing. Exact match first, then prefix, then substring.
"""
from silica.driver.base import NoteRef
from silica.tools import atomic


def _patch(monkeypatch, names):
    refs = [NoteRef(name=n, path=f"Folder/{n}.md") for n in names]
    monkeypatch.setattr(atomic.DRIVER, "search_names", lambda q: refs, raising=False)


def test_exact_then_prefix_then_substring_then_shortest(monkeypatch):
    _patch(monkeypatch, ["Modello OSI e livelli", "OSI reference model", "OSI", "Storia di OSI"])
    out = atomic.silica_search("osi")
    assert [p.split("/")[-1][:-3] for p in out["paths"]] == [
        "OSI",                      # exact
        "OSI reference model",      # prefix
        "Storia di OSI",            # substring, shorter name
        "Modello OSI e livelli",
    ]
    assert out["matched"] == 4
    assert "truncated" not in out


def test_caps_and_says_so(monkeypatch):
    _patch(monkeypatch, [f"Note {i:03d}" for i in range(120)])
    out = atomic.silica_search("note")
    assert len(out["paths"]) == atomic._SEARCH_CAP
    assert "120 notes matched" in out["truncated"]


def test_no_match_returns_an_empty_answer_that_names_the_other_two_tools(monkeypatch):
    """Empty, but not silent: a bare 0 is what the caller read as "not in the
    vault" when the term was only ever in the bodies (2026-08-27)."""
    _patch(monkeypatch, [])
    out = atomic.silica_search("nothing")
    assert out["paths"] == [] and out["matched"] == 0
    assert "silica_search_context" in out["hint"]
    assert "silica_semantic_search" in out["hint"]
