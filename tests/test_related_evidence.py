# SPDX-License-Identifier: AGPL-3.0-or-later

"""silica_related returns evidence, not fusion residue (ADR-0032).

Measured 2026-08-25: six scores inside [0.0159, 0.0164] — the raw RRF term
1/(60+rank) — while the real cosine the facade had already computed sat on
RelatedNote.embed_score and was dropped at the payload seam. A caller ranking
reads by that number is reading noise, and an absent leg (this vault's embed
index was empty) is indistinguishable from full agreement. Same contract as
silica_semantic_search: reranked flag, cross-encoder score when reranked,
structural per-leg signals, legs named.
"""
from __future__ import annotations

import types

import pytest

from silica.kernel.recall.cooccurrence import CooccurStore, build_contribution
from silica.kernel.recall.embed import EmbedStore


def _embed_store(tmp_path) -> EmbedStore:
    es = EmbedStore(path=tmp_path / "e.json")
    es.upsert("A", "A note", [1.0, 0.0])
    es.upsert("B", "B note", [0.9, 0.1])
    es.upsert("C", "C note", [0.7, 0.3])
    return es


def _cooc_store(tmp_path) -> CooccurStore:
    st = CooccurStore(path=tmp_path / "c.json", lang="english")
    st.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    st.upsert_note("B", build_contribution("B", "beta gamma delta"))
    st.upsert_note("C", build_contribution("C", "zeta eta theta"))
    return st


def _fake_driver():
    def read_note(note: str):
        key = note.removesuffix(".md")
        return types.SimpleNamespace(
            ref=types.SimpleNamespace(path=key),
            content=f"# {key} note\nwords about {key} and more prose to read",
        )
    return types.SimpleNamespace(read_note=read_note)


@pytest.fixture
def wired(tmp_path, monkeypatch):
    es, st = _embed_store(tmp_path), _cooc_store(tmp_path)
    monkeypatch.setattr("silica.kernel.recall.embed.get_store", lambda: es)
    monkeypatch.setattr("silica.kernel.recall.cooccurrence.get_cooccur_store", lambda **_: st)
    monkeypatch.setattr("silica.driver.DRIVER", _fake_driver())
    monkeypatch.setattr("silica.kernel.recall.memory_lane.memory_stores",
                        lambda: (None, None))
    return es, st


class _Reranker:
    """Scores the LAST candidate highest so a reorder is observable."""

    def __init__(self):
        self.calls = 0

    def scores(self, query, docs):
        self.calls += 1
        n = len(docs)
        return [0.1 + 0.8 * (i / max(n - 1, 1)) for i in range(n)]


def test_reranked_scores_replace_rrf_residue(wired, monkeypatch):
    rr = _Reranker()
    monkeypatch.setattr("silica.agent.providers.get_reranker", lambda *_a, **_k: rr)
    from silica.tools.graph import silica_related

    out = silica_related("A", k=5)
    assert out["reranked"] is True
    assert rr.calls == 1
    scores = [r["score"] for r in out["results"]]
    # Cross-encoder relevance, descending — not the flat 1/(60+rank) band.
    assert scores == sorted(scores, reverse=True)
    assert scores[0] >= 0.5
    assert all(s > 0.05 for s in scores)  # nothing left on the RRF scale


def test_without_a_reranker_the_flag_says_so(wired, monkeypatch):
    monkeypatch.setattr("silica.agent.providers.get_reranker", lambda *_a, **_k: None)
    from silica.tools.graph import silica_related

    out = silica_related("A", k=5)
    assert out["reranked"] is False
    assert all(r["score"] < 0.1 for r in out["results"])  # first-stage fusion scale


def test_payload_carries_structural_leg_signals(wired, monkeypatch):
    # embed_score / cooccur_weight already live on RelatedNote; the payload must
    # stop dropping them — callers threshold structurally, not by parsing
    # evidence strings.
    monkeypatch.setattr("silica.agent.providers.get_reranker", lambda *_a, **_k: None)
    from silica.tools.graph import silica_related

    out = silica_related("A", k=5)
    by_path = {r["path"]: r for r in out["results"]}
    assert "embed" in by_path["B"] and by_path["B"]["embed"] > 0.9
    assert "cooccur" in by_path["B"]


def test_legs_are_named_so_an_absent_leg_is_visible(tmp_path, monkeypatch):
    # The 2026-08-25 failure: embed index empty, every result cooccur-only, and
    # nothing in the payload said the strongest leg never ran.
    st = _cooc_store(tmp_path)
    monkeypatch.setattr("silica.kernel.recall.embed.get_store",
                        lambda: EmbedStore(path=tmp_path / "empty.json"))
    monkeypatch.setattr("silica.kernel.recall.cooccurrence.get_cooccur_store", lambda **_: st)
    monkeypatch.setattr("silica.driver.DRIVER", _fake_driver())
    monkeypatch.setattr("silica.kernel.recall.memory_lane.memory_stores",
                        lambda: (None, None))
    monkeypatch.setattr("silica.agent.providers.get_reranker", lambda *_a, **_k: None)
    from silica.tools.graph import silica_related

    out = silica_related("A", k=5)
    assert out["legs"] == {"embed": False, "cooccur": True, "memory": False}


def test_doctor_names_a_cold_embed_index(tmp_path, monkeypatch):
    # cooccur populated + embed empty is the asymmetry that silently halves the
    # facade: someone indexed this vault and never embedded it. The doctor is
    # the one surface whose job is saying that out loud.
    st = _cooc_store(tmp_path)
    monkeypatch.setattr("silica.kernel.recall.embed.get_store",
                        lambda: EmbedStore(path=tmp_path / "empty.json"))
    monkeypatch.setattr("silica.kernel.recall.cooccurrence.get_cooccur_store", lambda **_: st)
    from silica.config import CONFIG
    from silica.onboarding.checks import check_recall_indexes

    r = check_recall_indexes(CONFIG)
    assert r.status == "warn"
    assert "embed" in r.detail.lower()
    assert "silica_embed_refresh" in r.hint or "embed" in r.hint


def test_doctor_recall_indexes_ok_when_both_stores_answer(tmp_path, monkeypatch):
    es, st = _embed_store(tmp_path), _cooc_store(tmp_path)
    monkeypatch.setattr("silica.kernel.recall.embed.get_store", lambda: es)
    monkeypatch.setattr("silica.kernel.recall.cooccurrence.get_cooccur_store", lambda **_: st)
    from silica.config import CONFIG
    from silica.onboarding.checks import check_recall_indexes

    assert check_recall_indexes(CONFIG).status == "ok"
