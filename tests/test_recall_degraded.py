# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""silica_recall must say when the reranker did not rank.

Measured 2026-09-03 on this repo with :1235 down: every reply carried
`reranked: false` only on semantic_search, and recall carried nothing at all,
so a caller read a raw-fusion ordering as the ranked one. The bench gap is
42/42 reranked vs 30/42 cosine, which is not a nuance to bury.
"""
from __future__ import annotations


def _fake_retrieve(reranked: bool):
    def fake(query, *, k, rerank_stats=None, **kwargs):
        if rerank_stats is not None:
            rerank_stats["reranked"] = reranked
        return [], None
    return fake


def test_perceive_forwards_rerank_stats(monkeypatch):
    monkeypatch.setattr("silica.kernel.recall.perception.facade_retrieve", _fake_retrieve(True))
    from silica.kernel.recall.perception import perceive

    rr: dict = {}
    perceive("q", now="2026-09-03", k=3, rerank_stats=rr)
    assert rr == {"reranked": True}


def test_recall_names_the_degraded_leg(monkeypatch):
    monkeypatch.setattr("silica.kernel.recall.perception.facade_retrieve", _fake_retrieve(False))
    from silica.tools.graph import silica_recall

    out = silica_recall("q", k=3, memory=False)
    assert out["degraded"] == ["rerank"]
    assert "SILICA_RERANK_SERVE_CMD" in out["hint"]


def test_recall_stays_silent_when_reranked(monkeypatch):
    monkeypatch.setattr("silica.kernel.recall.perception.facade_retrieve", _fake_retrieve(True))
    from silica.tools.graph import silica_recall

    out = silica_recall("q", k=3, memory=False)
    assert "degraded" not in out
