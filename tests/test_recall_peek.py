# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""silica_recall(vault=...): read another vault without leaving this one.

The memory lane already answers from a vault that is not the active one
(ADR-0019); a peek is that lane pointed at a vault the caller names. The
session's vault never moves, so the write target, the change ledger and the
undo scope stay where the work is.
"""
from __future__ import annotations

from silica.config import CONFIG
from silica.kernel.recall.embed import EmbedStore
from silica.kernel.recall.paths import index_dir_for
from silica.tools import graph


def _adopted(tmp_path, name: str):
    v = tmp_path / name
    v.mkdir()
    (v / "vault.yaml").write_text("write_dir: ''\n", encoding="utf-8")
    return v


def _embedded(vault, notes) -> None:
    d = index_dir_for(str(vault))
    d.mkdir(parents=True, exist_ok=True)
    es = EmbedStore(path=d / "embeddings.json")
    for path, name, vec, body in notes:
        es.upsert(path, name, vec)
        f = vault / (path + ".md")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    es.save()


def _fake_embedder(monkeypatch, vec) -> None:
    import silica.agent.providers as providers

    class _E:
        def embed(self, texts):
            return [list(vec) for _ in texts]

    monkeypatch.setattr(providers, "get_embedder", lambda cfg: _E())
    monkeypatch.setattr(providers, "get_reranker", lambda cfg: None)


def _session(tmp_path, monkeypatch):
    import silica.driver

    active = _adopted(tmp_path, "active")
    monkeypatch.setattr(CONFIG, "vault_path", str(active))
    monkeypatch.setattr(CONFIG, "memory_vault", str(tmp_path / "no-such-memory"))
    monkeypatch.setattr(silica.driver, "_driver", None)
    return active


def test_peek_reads_the_other_vault_and_leaves_the_session_where_it_was(tmp_path, monkeypatch):
    active = _session(tmp_path, monkeypatch)
    stats = _adopted(tmp_path, "stats")
    _embedded(stats, [("Notes/Heteroscedasticity", "Heteroscedasticity", [1.0, 0.0],
                       "Residual variance grows with the fitted value.")])
    _fake_embedder(monkeypatch, [1.0, 0.0])

    out = graph.silica_recall("residual variance", vault=str(stats))

    assert out["vault"] == str(stats.resolve())
    assert out["notes"] == ["Notes/Heteroscedasticity"]
    assert "Residual variance grows" in out["context"]
    assert out["partial"] == []  # a foreign note cannot be re-read with silica_read_note
    assert CONFIG.vault_path == str(active)


def test_peek_hits_carry_the_vault_as_origin(tmp_path, monkeypatch):
    # Body readers dispatch on origin: "vault" would read the ACTIVE vault's
    # note of the same relative path (or nothing) and the peek would lie.
    from silica.kernel.recall.perception import perceive

    _session(tmp_path, monkeypatch)
    stats = _adopted(tmp_path, "stats")
    _embedded(stats, [("N", "N", [1.0, 0.0], "body")])
    _fake_embedder(monkeypatch, [1.0, 0.0])
    p = perceive("q", now="2026-09-01", vault=str(stats))
    assert [b.origin for b in p.blocks] == [str(stats.resolve())]


def test_peek_rejects_a_folder_silica_never_adopted(tmp_path, monkeypatch):
    _session(tmp_path, monkeypatch)
    out = graph.silica_recall("q", vault=str(tmp_path / "nowhere"))
    assert "error" in out and out["notes"] == []


def test_peek_of_a_cold_vault_says_cold_not_empty(tmp_path, monkeypatch):
    _session(tmp_path, monkeypatch)
    cold = _adopted(tmp_path, "cold")
    _fake_embedder(monkeypatch, [1.0, 0.0])
    out = graph.silica_recall("q", vault=str(cold))
    assert out["notes"] == [] and out["coverage"] == "cold" and out["hint"]


def test_peek_of_the_active_vault_is_plain_recall(tmp_path, monkeypatch):
    active = _session(tmp_path, monkeypatch)
    _embedded(active, [("A", "A", [1.0, 0.0], "alpha body")])
    _fake_embedder(monkeypatch, [1.0, 0.0])
    plain = graph.silica_recall("alpha")
    peek = graph.silica_recall("alpha", vault=str(active))
    assert "vault" not in peek and peek["notes"] == plain["notes"] == ["A"]


def test_peek_accepts_the_memory_vault_without_a_manifest(tmp_path, monkeypatch):
    # The memory vault is configured, not adopted: ~/.silica/vault carries no
    # vault.yaml, yet silica_vaults lists it, so the peek must take it too.
    active = _session(tmp_path, monkeypatch)
    mem = tmp_path / "mem"
    mem.mkdir()
    monkeypatch.setattr(CONFIG, "memory_vault", str(mem))
    _embedded(mem, [("Fact", "Fact", [1.0, 0.0], "remembered body")])
    _fake_embedder(monkeypatch, [1.0, 0.0])

    out = graph.silica_recall("remembered", vault=str(mem))
    assert "error" not in out
    assert out["vault"] == str(mem.resolve()) and out["notes"] == ["Fact"]
    assert CONFIG.vault_path == str(active)


# ---------------------------------------------------------------------------
# Reading a peeked note: silica_read_note(name, vault=) — the same wikilink
# resolution as the active vault, through a read-only backend on that folder.
# ---------------------------------------------------------------------------

def test_read_note_opens_a_note_of_another_vault_by_name(tmp_path, monkeypatch):
    from silica.tools.atomic import silica_read_note

    active = _session(tmp_path, monkeypatch)
    stats = _adopted(tmp_path, "stats")
    (stats / "Notes").mkdir()
    (stats / "Notes/Heteroscedasticity.md").write_text(
        "# Heteroscedasticity\n\nResidual variance grows.\n", encoding="utf-8")

    got = silica_read_note("Heteroscedasticity", vault=str(stats))  # bare name, no path

    assert "Residual variance grows." in got
    first = got.splitlines()[0]
    assert first.startswith("> [read-only]") and str(stats.resolve()) in first
    assert CONFIG.vault_path == str(active)


def test_read_note_accepts_the_relative_path_recall_returned(tmp_path, monkeypatch):
    from silica.tools.atomic import silica_read_note

    _session(tmp_path, monkeypatch)
    stats = _adopted(tmp_path, "stats")
    (stats / "Notes").mkdir()
    (stats / "Notes/Long.md").write_text("tail line\n", encoding="utf-8")
    assert "tail line" in silica_read_note("Notes/Long", vault=str(stats))


def test_read_note_without_vault_never_reaches_a_foreign_note(tmp_path, monkeypatch):
    import pytest

    from silica.tools.atomic import silica_read_note

    _session(tmp_path, monkeypatch)
    stats = _adopted(tmp_path, "stats")
    (stats / "Only.md").write_text("foreign\n", encoding="utf-8")
    with pytest.raises(Exception):
        silica_read_note("Only")


def test_read_note_rejects_a_folder_silica_never_adopted(tmp_path, monkeypatch):
    import pytest

    from silica.tools.atomic import silica_read_note

    _session(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not a Silica vault"):
        silica_read_note("x", vault=str(tmp_path / "nowhere"))


def test_peek_partial_names_the_notes_you_can_reread_with_the_same_vault(tmp_path, monkeypatch):
    from silica.tools.atomic import silica_read_note

    _session(tmp_path, monkeypatch)
    stats = _adopted(tmp_path, "stats")
    body = "intro\n" + ("a filler line about nothing in particular\n" * 300) \
        + "Residual variance grows with the fitted value.\n"
    _embedded(stats, [("Notes/Long", "Long", [1.0, 0.0], body)])
    _fake_embedder(monkeypatch, [1.0, 0.0])

    out = graph.silica_recall("residual variance", vault=str(stats))
    assert out["partial"] == ["Notes/Long"]  # a slice was delivered: re-read is worth it
    full = silica_read_note(out["partial"][0], vault=out["vault"])
    assert full.rstrip().endswith("Residual variance grows with the fitted value.")


def test_recall_names_the_memory_vault_so_its_notes_can_be_reread(tmp_path, monkeypatch):
    from silica.tools.atomic import silica_read_note

    _session(tmp_path, monkeypatch)
    mem = tmp_path / "mem"
    mem.mkdir()
    monkeypatch.setattr(CONFIG, "memory_vault", str(mem))
    _embedded(mem, [("Fact", "Fact", [1.0, 0.0], "remembered body\n")])
    _fake_embedder(monkeypatch, [1.0, 0.0])

    out = graph.silica_recall("remembered")  # plain call: the memory lane brings Fact in
    assert out["memory"] == ["Fact"] and out["memory_vault"] == str(mem.resolve())
    assert "remembered body" in silica_read_note("Fact", vault=out["memory_vault"])
