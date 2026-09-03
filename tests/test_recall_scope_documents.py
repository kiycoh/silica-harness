# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Recall scoped to a folder, and the code a note documents in its header.

Measured 2026-09-03 on this repo: 333 market-research notes (35% of the
vault) took 6 of 15 recall slots on a question about THIS repo's write gate,
by co-occurrence of "gate/merge/candidate"; `memory=false` cannot exclude
them because they are in the active vault. And the note that did answer
documents `validate.py`, but nothing in the reply said so: the `documents:`
binding existed only to compute `stale`.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _results(paths):
    from silica.kernel.recall.relatedness import RelatedNote
    return [RelatedNote(path=p, name=p.rsplit("/", 1)[-1], score=1.0 - i / 100,
                        evidence=["cooccur:w1"]) for i, p in enumerate(paths)]


MIXED = ["docs/research/market/a", "docs/adr/0003-x", "docs/research/market/b",
         "docs/adr/0030-y", "silica/prompts/rubric", "docs/adr/0031-z"]


def test_perceive_folder_keeps_only_that_subtree_and_overfetches(monkeypatch):
    from silica.kernel.recall import perception

    captured: dict = {}

    def fake_retrieve(query, *, k, **kw):
        captured["k"] = k
        return _results(MIXED), None

    monkeypatch.setattr(perception, "facade_retrieve", fake_retrieve)
    monkeypatch.setattr(perception, "_read_dated_body",
                        lambda path, origin="vault": ("", None, f"body of {path}", []))
    p = perception.perceive("q", now="2026-09-03", k=2, with_facts=False, folder="docs/adr")
    assert [b.path for b in p.blocks] == ["docs/adr/0003-x", "docs/adr/0030-y"]
    assert captured["k"] >= 6  # the cut to k happens after the filter


def test_recall_tool_forwards_folder(monkeypatch):
    from silica.kernel.recall.perception import Perception

    captured: dict = {}

    def fake_perceive(query, **kwargs):
        captured.update(kwargs)
        return Perception(query=query)

    monkeypatch.setattr("silica.kernel.recall.perception.perceive", fake_perceive)
    from silica.tools.graph import silica_recall

    silica_recall("q", k=5, memory=False, folder="docs/adr")
    assert captured.get("folder") == "docs/adr"


def test_semantic_search_folder_filters_results(monkeypatch):
    monkeypatch.setattr("silica.kernel.recall.perception.facade_retrieve",
                        lambda *a, **k: (_results(MIXED), None))
    monkeypatch.setattr("silica.agent.providers.get_reranker", lambda *_a, **_k: None)
    from silica.tools.graph import silica_semantic_search

    out = silica_semantic_search("q", k=5, memory=False, folder="docs/adr")
    assert [r["path"] for r in out["results"]] == ["docs/adr/0003-x", "docs/adr/0030-y", "docs/adr/0031-z"]


# --- documents: in the header ---------------------------------------------

def _bind(vault: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import silica.driver
    from silica.config import CONFIG

    vault.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(silica.driver, "_driver", None)


def test_read_dated_body_returns_the_documented_files(tmp_path, monkeypatch):
    _bind(tmp_path / "v", monkeypatch)
    from silica.driver import DRIVER
    from silica.kernel.recall.perception import _read_dated_body

    DRIVER.create("docs/gate.md",
                  '---\ndate: "2026-01-01"\ndocuments:\n  - silica/kernel/write/validate.py\n---\n\nbody\n')
    date, contested, body, documents = _read_dated_body("docs/gate")
    assert (date, contested, body.strip()) == ("2026-01-01", None, "body")
    assert documents == ["silica/kernel/write/validate.py"]


def test_render_header_names_the_documented_files():
    from silica.kernel.recall.perception import NoteBlock, Perception

    b = NoteBlock(path="docs/gate", date="", evidence="embed:0.7", body="x", excerpt="x",
                  documents=["silica/kernel/write/validate.py", "silica/kernel/write/contested.py"])
    ctx = Perception(query="q", blocks=[b]).render()
    assert ("[#1 | docs/gate | embed:0.7 | documents: silica/kernel/write/validate.py, "
            "silica/kernel/write/contested.py]") in ctx
