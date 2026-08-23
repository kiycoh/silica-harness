# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Unit tests for the fusion probe (silica.kernel.link.health) + reranker A/B.

Magnitudes are corpus properties; these pin the mechanics on synthetic vaults:
cheap-tier recovery via the cooccur legs, clean embed abstention, and the test
that keeps the probe honest — a text-dissimilar pair recovered ONLY when the
embed leg is live, proving the probe exercises the fusion rather than just the
lexical leg.
"""
from __future__ import annotations

import math

from silica.kernel.recall.cooccurrence import CooccurStore, build_contribution
from silica.kernel.link import health
from evals.golden import probe_fusion


def _store(tmp_path, notes: dict[str, str]) -> CooccurStore:
    st = CooccurStore(path=tmp_path / "idx" / "c.json", lang="english")
    for key, body in notes.items():
        st.upsert_note(key, build_contribution(key, body))
    return st


class _FakeEmbedStore:
    """The slice of the EmbedStore contract the facade touches: len, get_vec,
    cosine_top_k. Real cosine so rankings are honest."""

    def __init__(self, vecs: dict[str, list[float]]):
        self._vecs = vecs

    def __len__(self) -> int:
        return len(self._vecs)

    def get_vec(self, path: str):
        return self._vecs.get(path)

    def cosine_top_k(self, query_vec, k=5, exclude=None):
        exclude = exclude or set()

        def cos(a, b):
            den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
            return sum(x * y for x, y in zip(a, b)) / den if den else 0.0

        scored = [
            {"path": p, "name": p.rsplit("/", 1)[-1], "score": round(cos(query_vec, v), 4)}
            for p, v in self._vecs.items()
            if p not in exclude
        ]
        scored.sort(key=lambda d: (-d["score"], d["path"]))
        return scored[:k]


def test_empty_store_is_zeros(tmp_path):
    st = CooccurStore(path=tmp_path / "idx" / "c.json", lang="english")
    res = health.fusion_probe(tmp_path, st)
    assert res["pairs_evaluated"] == 0 and res["recall_at_10"] == 0.0
    assert res["legs"] == ""


def test_recovers_text_similar_pair_cheap_tier(tmp_path):
    (tmp_path / "A.md").write_text("quick sort compares array elements\n[[B]]")
    (tmp_path / "B.md").write_text("quick sort swaps array elements")
    st = _store(tmp_path, {
        "A": "quick sort compares array elements",
        "B": "quick sort swaps array elements",
    })
    res = health.fusion_probe(tmp_path, st)
    assert res["pairs_evaluated"] == 1
    assert res["recall_at_10"] == 1.0
    assert res["mrr"] > 0.0
    assert res["legs"] == "cooccur"
    assert res["embed_coverage"] == 0.0  # no embed store passed


def test_embed_only_pair_needs_the_embed_leg(tmp_path):
    # Zero stem overlap: every lexical leg abstains for this pair, so recovery
    # can only come from the embed leg flowing through the real RRF fusion.
    (tmp_path / "A.md").write_text("alpha beta gamma\n[[B]]")
    (tmp_path / "B.md").write_text("delta epsilon zeta")
    st = _store(tmp_path, {"A": "alpha beta gamma", "B": "delta epsilon zeta"})

    cheap = health.fusion_probe(tmp_path, st)
    assert cheap["pairs_evaluated"] == 1
    assert cheap["recall_at_10"] == 0.0

    fake = _FakeEmbedStore({"A": [1.0, 0.0], "B": [0.9, 0.1]})
    full = health.fusion_probe(tmp_path, st, embed_store=fake)
    assert full["recall_at_10"] == 1.0
    assert full["mrr"] == 1.0  # rank 1 from at least one direction
    assert full["legs"] == "embed+cooccur"
    assert full["embed_coverage"] == 1.0


def _point_driver_at(tmp_path):
    """link_query reads through the DRIVER — aim it at the synthetic vault."""
    import silica.driver
    from silica.config import CONFIG

    CONFIG.vault_path = str(tmp_path)
    silica.driver._driver = None


class _FakeReranker:
    """scores() by document content — deterministic, no HTTP. None = abstain."""

    def __init__(self, favour: str | None):
        self._favour = favour

    def scores(self, query, documents):
        if self._favour is None:
            return None
        return [1.0 if self._favour in d else 0.0 for d in documents]


def test_rerank_ab_reorders_within_the_fused_topk(tmp_path):
    # Reorder-only (retrieval-gates spec 2a): membership belongs to the first
    # stage, so recall@k is identical across arms by construction; the
    # reranker's win is ORDERING — it promotes the true counterpart above the
    # embed-favoured decoy inside top-k, visible in mrr and pairs_won.
    (tmp_path / "A.md").write_text("alpha alpha alpha\n[[B]]")
    (tmp_path / "B.md").write_text("epsilon epsilon epsilon")
    (tmp_path / "X.md").write_text("zeta zeta zeta")
    _point_driver_at(tmp_path)
    st = _store(tmp_path, {
        "A": "alpha alpha alpha", "B": "epsilon epsilon epsilon", "X": "zeta zeta zeta",
    })
    fake_embed = _FakeEmbedStore({
        "A": [1.0, 0.0], "B": [0.9, 0.436], "X": [0.995, 0.1],
    })
    res = probe_fusion.run_rerank_ab(
        tmp_path, st, embed_store=fake_embed,
        reranker=_FakeReranker(favour="epsilon"), k=2,
    )
    assert res["pairs_evaluated"] == 1
    assert res["base_recall"] == res["rerank_recall"] == 1.0  # membership invariant
    assert res["rerank_mrr"] > res["base_mrr"]                # B promoted 2 -> 1
    assert res["pairs_won"] == 1 and res["pairs_lost"] == 0
    assert res["empty_docs"] == 0         # every synthetic note is readable


def test_rerank_ab_abstaining_reranker_is_a_no_op(tmp_path):
    (tmp_path / "A.md").write_text("quick sort compares array elements\n[[B]]")
    (tmp_path / "B.md").write_text("quick sort swaps array elements")
    _point_driver_at(tmp_path)
    st = _store(tmp_path, {
        "A": "quick sort compares array elements",
        "B": "quick sort swaps array elements",
    })
    res = probe_fusion.run_rerank_ab(tmp_path, st, reranker=_FakeReranker(favour=None))
    assert res["base_recall"] == res["rerank_recall"] == 1.0
    assert res["base_mrr"] == res["rerank_mrr"]
    assert res["pairs_won"] == 0 and res["pairs_lost"] == 0


def test_empty_embed_store_abstains_cleanly(tmp_path):
    (tmp_path / "A.md").write_text("quick sort compares array elements\n[[B]]")
    (tmp_path / "B.md").write_text("quick sort swaps array elements")
    st = _store(tmp_path, {
        "A": "quick sort compares array elements",
        "B": "quick sort swaps array elements",
    })
    res = health.fusion_probe(tmp_path, st, embed_store=_FakeEmbedStore({}))
    assert res["recall_at_10"] == 1.0  # cooccur legs still carry the pair
    assert res["legs"] == "cooccur"  # empty store counts as absent
