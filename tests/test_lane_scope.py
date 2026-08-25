# SPDX-License-Identifier: AGPL-3.0-or-later

"""Per-call lane scope on the retrieval interface (ADR-0032).

Measured 2026-08-25 on the live MCP server: with SILICA_MEMORY_VAULT pointing
at an unrelated vault, a repo-scoped recall handed 4 of 6 slots to that vault,
silica_related returned its notes at #1, and silica_recall named notes that
silica_read_note then denied (the payload dropped the origin marker ADR-0019
requires every consumer to respect). Nothing at the tool interface could say
"this vault only". These tests pin the scope parameter, the honest tie-break,
and the origin invariant.
"""
from __future__ import annotations

import types

import pytest

from silica.kernel.recall.cooccurrence import CooccurStore, build_contribution
from silica.kernel.recall.embed import EmbedStore
from silica.kernel.recall.relatedness import _fuse


# ---------------------------------------------------------------------------
# _fuse tie-break
# ---------------------------------------------------------------------------

def test_fuse_ties_break_vault_first_never_by_nul_prefix():
    # Both lanes propose their rank-1 with the identical RRF term (1/61). The
    # memory key namespace is NUL-prefixed, which sorts before every real path,
    # so before ADR-0032 the guest lane won every tie by an encoding accident.
    out = _fuse(
        None,
        [("docs/adr/0030-routing", 5.0)],
        k=5,
        mem_cooc_rank=[("Inbox/SVM-chapter", 5.0)],
    )
    assert [r.origin for r in out] == ["vault", "memory"]
    assert out[0].path == "docs/adr/0030-routing"


# ---------------------------------------------------------------------------
# memory=False on the three retrieval tools
# ---------------------------------------------------------------------------

def _embed_store(tmp_path) -> EmbedStore:
    es = EmbedStore(path=tmp_path / "e.json")
    es.upsert("A", "A note", [1.0, 0.0])
    es.upsert("B", "B note", [0.9, 0.1])
    return es


def _cooc_store(tmp_path) -> CooccurStore:
    st = CooccurStore(path=tmp_path / "c.json", lang="english")
    st.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    st.upsert_note("B", build_contribution("B", "beta gamma delta"))
    return st


def _fake_driver(names: dict[str, str]):
    def read_note(note: str):
        key = note.removesuffix(".md")
        if key in names or note in names:
            path = names.get(note, names.get(key, ""))
            return types.SimpleNamespace(
                ref=types.SimpleNamespace(path=path), content=f"# {key}\nbody of {key}"
            )
        raise KeyError(note)
    return types.SimpleNamespace(read_note=read_note)


class _LaneSpy:
    def __init__(self):
        self.called = False

    def __call__(self):
        self.called = True
        return None, None


@pytest.fixture
def wired(tmp_path, monkeypatch):
    es, st = _embed_store(tmp_path), _cooc_store(tmp_path)
    monkeypatch.setattr("silica.kernel.recall.embed.get_store", lambda: es)
    monkeypatch.setattr("silica.kernel.recall.cooccurrence.get_cooccur_store", lambda **_: st)
    monkeypatch.setattr("silica.driver.DRIVER", _fake_driver({"Alpha": "A", "A": "A", "B": "B"}))
    monkeypatch.setattr("silica.agent.providers.get_reranker", lambda *_a, **_k: None)
    spy = _LaneSpy()
    monkeypatch.setattr("silica.kernel.recall.memory_lane.memory_stores", spy)
    return spy


def test_related_memory_false_never_touches_the_lane(wired):
    from silica.tools.graph import silica_related

    out = silica_related("Alpha", k=5, memory=False)
    assert out["results"]          # the vault legs still answer
    assert wired.called is False


def test_related_default_keeps_the_lane_on(wired):
    # ADR-0019's default survives: scope is per call, never a behavior flip.
    from silica.tools.graph import silica_related

    silica_related("Alpha", k=5)
    assert wired.called is True


def test_facade_retrieve_use_memory_false_skips_the_lane(wired, monkeypatch):
    from silica.kernel.recall.perception import facade_retrieve

    results, _vec = facade_retrieve("beta", k=5, use_embedder=False, use_rerank=False,
                                    use_memory=False)
    assert results is not None
    assert wired.called is False


def test_semantic_search_forwards_memory_scope(monkeypatch):
    captured: dict = {}

    def fake_retrieve(query, *, k, **kwargs):
        captured.update(kwargs)
        return [], None

    monkeypatch.setattr("silica.kernel.recall.perception.facade_retrieve", fake_retrieve)
    from silica.tools.graph import silica_semantic_search

    silica_semantic_search("q", k=3, memory=False)
    assert captured.get("use_memory") is False


def test_perceive_use_memory_false_skips_episodic_facts(wired, monkeypatch):
    # memory=False means "this vault only": the episodic store homes in the
    # memory vault (no abstain rule of its own), so the facts block is the same
    # foreign lane through a second door and must go dark with it.
    from silica.kernel.recall import perception

    called = {"facts": False}

    def spy_facts(*a, **k):
        called["facts"] = True

    monkeypatch.setattr(perception, "_recall_facts", spy_facts)
    perception.perceive("q", now="2026-08-25", paths=[], use_memory=False)
    assert called["facts"] is False
    perception.perceive("q", now="2026-08-25", paths=[])
    assert called["facts"] is True


# ---------------------------------------------------------------------------
# recall payload carries the origin invariant
# ---------------------------------------------------------------------------

def test_recall_payload_marks_memory_notes_and_keeps_them_out_of_partial(monkeypatch):
    # The doctor's exact failure story: recall names a note, read_note denies
    # it. `memory` lists the foreign-lane paths; `partial` must not invite a
    # re-read that cannot succeed in the active vault.
    from silica.kernel.recall.perception import NoteBlock, Perception

    blocks = [
        NoteBlock(path="Informatica/SVM", date="", evidence="memory:embed:0.37",
                  body="long body " * 50, excerpt="long body", origin="memory"),
        NoteBlock(path="docs/adr/0030-routing", date="", evidence="cooccur:w9",
                  body="short", excerpt="short"),
    ]
    canned = Perception(query="q", blocks=blocks)
    monkeypatch.setattr("silica.kernel.recall.perception.perceive",
                        lambda *a, **k: canned)
    from silica.tools.graph import silica_recall

    out = silica_recall("q", k=5)
    assert out["memory"] == ["Informatica/SVM"]
    assert "Informatica/SVM" not in out["partial"]
    assert "docs/adr/0030-routing" in out["notes"]


def test_recall_forwards_memory_scope(monkeypatch):
    from silica.kernel.recall.perception import Perception

    captured: dict = {}

    def fake_perceive(query, **kwargs):
        captured.update(kwargs)
        return Perception(query=query)

    monkeypatch.setattr("silica.kernel.recall.perception.perceive", fake_perceive)
    from silica.tools.graph import silica_recall

    silica_recall("q", k=5, memory=False)
    assert captured.get("use_memory") is False


# ---------------------------------------------------------------------------
# the active vault's inbox filter must not judge foreign paths
# ---------------------------------------------------------------------------

def test_semantic_search_inbox_filter_spares_memory_origin(monkeypatch):
    # is_inbox_path answers for the ACTIVE vault's staging roots; a memory-lane
    # note whose vault happens to use the same folder name is not this vault's
    # staging and must not be silently dropped.
    from silica.config import CONFIG
    from silica.kernel.recall.relatedness import RelatedNote

    monkeypatch.setattr(CONFIG, "inbox_dir", "Inbox")
    results = [
        RelatedNote(path="Inbox/foreign-chapter", name="foreign-chapter",
                    score=0.03, evidence=["memory:embed:0.40"], origin="memory"),
        RelatedNote(path="Inbox/staged-here", name="staged-here",
                    score=0.03, evidence=["embed:0.39"]),
        RelatedNote(path="docs/x", name="x", score=0.02, evidence=["embed:0.30"]),
    ]
    monkeypatch.setattr("silica.kernel.recall.perception.facade_retrieve",
                        lambda *a, **k: (results, None))
    monkeypatch.setattr("silica.agent.providers.get_reranker", lambda *_a, **_k: None)
    from silica.tools.graph import silica_semantic_search

    out = silica_semantic_search("q", k=5)
    paths = [r["path"] for r in out["results"]]
    assert "Inbox/foreign-chapter" in paths
    assert "Inbox/staged-here" not in paths
    assert "docs/x" in paths
