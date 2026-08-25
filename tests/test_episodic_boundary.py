# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""The episodic store stays out of vault retrieval — as a contract, not a habit.

Silica's own sessions become facts, never notes, for one load-bearing reason:
a distilled answer indexed as a vault note would be retrieved in the next
session and reinforce itself. That echo channel is closed today by
construction — no retrieval leg reads the store. These two tests make it a
contract, so a future refactor that feeds the store to a leg "for
completeness" fails the suite instead of reopening it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SILICA_ROOT = Path(__file__).resolve().parent.parent / "silica"

_EPISODIC_IMPORT_RE = re.compile(
    r"from silica\.kernel\.recall\.episodic import|"
    r"import silica\.kernel\.recall\.episodic\b|"
    r"from silica\.kernel\.recall import [^\n]*\bepisodic\b"
)

# The store is out of scope — it IS the implementation.
STORE = "kernel/recall/episodic.py"

# module (relative to silica/) → why it may reach the episodic store
ALLOWED = {
    "kernel/recall/perception.py":  "the answer seam: facts reach the model only in the Personal-memory block",
    "kernel/progress.py":           "the digest: nucleation candidates are suggested, never written",
    "router/states/distill.py":     "the capture seam: ephemerals of a note-path run",
    "cli.py":                       "the drain seam: own-session envelopes become facts",
    # Not in the spec's enumerated list, but deliberate and content-free: only
    # live KEYS travel (key_vocabulary), so capture snaps to the established
    # vocabulary instead of coining synonyms (ADR-0021). No fact text, no
    # retrieval leg.
    "kernel/recall/run_substrate.py": "distiller key vocabulary (keys only, no fact text)",
    # G2 orientation (2026-08-25): the vault map's one usage-derived line is
    # "Recurring facts (by runs)" — fact KEYS and run counts only, no fact
    # text, and the map is not a retrieval leg, so the echo channel this
    # contract closes stays closed.
    "kernel/recall/vault_map.py":   "orientation salience (keys + run counts, no fact text)",
}


def test_episodic_imports_are_allowlisted():
    offenders = []
    for path in SILICA_ROOT.rglob("*.py"):
        rel = path.relative_to(SILICA_ROOT).as_posix()
        if rel in ALLOWED or rel == STORE:
            continue
        if _EPISODIC_IMPORT_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert not offenders, (
        f"kernel.episodic imported outside the allowlist: {offenders}. Machine "
        "memory reaches the model through the Personal-memory block and enters "
        "the vault only by promotion — extend ALLOWED here with a justification "
        "if that is genuinely what the new site does."
    )


def test_allowlist_has_no_stale_entries():
    stale = [
        rel for rel in ALLOWED
        if not (SILICA_ROOT / rel).exists()
        or not _EPISODIC_IMPORT_RE.search((SILICA_ROOT / rel).read_text(encoding="utf-8"))
    ]
    assert not stale, f"Allowlist entries no longer import the store: {stale}."


def _bind(vault: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import silica.driver
    import silica.kernel.recall.cooccurrence as cooc_mod
    import silica.kernel.recall.embed as embed_mod
    from silica.config import CONFIG

    vault.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(CONFIG, "memory_vault", str(vault))
    monkeypatch.setattr(silica.driver, "_driver", None)
    embed_mod.clear()
    cooc_mod.clear()


def test_facts_never_surface_as_retrieval_hits(tmp_path, monkeypatch):
    """A populated store must not add a single leg hit — only the fact block."""
    from silica.driver import DRIVER
    from silica.kernel.recall import episodic
    from silica.kernel.recall.episodic import EpisodicStore
    from silica.kernel.recall.perception import perceive
    from silica.tools.graph import silica_cooccurrence_refresh

    _bind(tmp_path / "v", monkeypatch)
    monkeypatch.setattr(episodic, "store_path", lambda: tmp_path / "episodic.json")
    DRIVER.create("Concepts/Rerank.md",
                  '---\ndate: "2026-01-01"\n---\n\nthe reranker fuses three legs\n')
    silica_cooccurrence_refresh(force=True)

    store = EpisodicStore()
    store.capture([{"key": "silica.reranker", "text": "the reranker is jina v3"}],
                  run_id="silica-s1-end", seen="2026-08-01")
    store.save()

    p = perceive("what reranker do I use?", now="2026-08-02",
                 use_embedder=False, use_rerank=False)

    assert [b.path for b in p.blocks] == ["Concepts/Rerank"]
    assert all("jina v3" not in b.excerpt for b in p.blocks)
    # The one channel machine memory has to the model, and it is labelled.
    assert "jina v3" in p.facts_block
    assert p.facts_block.startswith("Personal memory:")
    rendered = p.render()
    assert rendered.count("jina v3") == 1
