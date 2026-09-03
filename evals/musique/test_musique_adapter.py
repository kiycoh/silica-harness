# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""End-to-end check for the MuSiQue adapter, fully offline.

Drives load → index (cooccur leg only, no embedder) → probe on a synthetic
MuSiQue-shaped corpus, and asserts the recall/rank bookkeeping. The gold
passage shares a rare token with the question; distractors do not, so a
correct fused ranking must place gold in the top-k.
"""
from __future__ import annotations


from evals.musique import runner


def _corpus():
    # p0 is the supporting passage (shares "zorblax"); p1/p2 are distractors.
    return [
        {"title": "Zorblax Tower", "text": "The zorblax tower stands in the old quarter."},
        {"title": "River Trade", "text": "Merchants shipped grain along the river."},
        {"title": "Mountain Pass", "text": "The pass connects two alpine valleys."},
    ]


def _questions():
    return [{
        "id": "2hop__test_1",
        "question": "Where does the zorblax tower stand?",
        "paragraphs": [
            {"title": "Zorblax Tower", "text": "The zorblax tower stands in the old quarter.",
             "is_supporting": True},
            {"title": "River Trade", "text": "Merchants shipped grain along the river.",
             "is_supporting": False},
        ],
    }]


def test_load_index_probe_roundtrip(tmp_path, monkeypatch):
    import silica.kernel.recall.embed as embed_mod

    # Share one resolved embed index path between refresh and probe (unused here
    # since use_embedder=False, but keeps get_store offline and deterministic).
    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")

    vault = tmp_path / "vault"
    vault.mkdir()
    runner.bind_vault(vault)

    corpus = _corpus()
    summary = runner.load_corpus(corpus)
    assert summary["committed"] == 3
    assert not summary["failures"]
    # Verbatim note carries the passage_id and raw text.
    note = (vault / "corpus" / "p00000.md").read_text()
    assert "passage_id: 0" in note and "zorblax tower stands" in note

    # Rerun is idempotent: nothing re-committed.
    assert runner.load_corpus(corpus)["skipped"] == 3

    runner.build_indexes(embed=False, force=True)

    doc = runner.probe(_questions(), corpus, k=10, use_embedder=False)
    m = doc["metrics"]
    assert m["questions_evaluated"] == 1
    assert m["unmappable_gold"] == 0
    assert doc["config"]["legs"] == "cooccur"
    # Gold (p0) shares the rare token; it must be retrieved and rank first.
    row = doc["questions"][0]
    assert 0 in row["top"]
    assert m["recall_at_10"] == 1.0
    assert row["first_gold_rank"] == 1
    assert m["mrr"] == 1.0
    # Per-hop breakdown keys off the id prefix.
    assert doc["metrics"]["per_hop"]["2hop"]["n"] == 1


def test_unmappable_gold_is_counted(tmp_path):
    runner.bind_vault(tmp_path / "vault")
    (tmp_path / "vault").mkdir(exist_ok=True)
    corpus = _corpus()
    q = [{
        "id": "q1",
        "question": "irrelevant",
        "paragraphs": [
            {"title": "Ghost", "text": "not in the corpus at all", "is_supporting": True},
        ],
    }]
    doc = runner.probe(q, corpus, k=10, use_embedder=False)
    # Gold maps to nothing in the corpus → question skipped, miss surfaced.
    assert doc["metrics"]["unmappable_gold"] == 1
    assert doc["metrics"]["questions_skipped"] == 1
    assert doc["metrics"]["questions_evaluated"] == 0


def test_pid_roundtrip():
    assert runner._pid_of("corpus/p00042") == 42
    assert runner._pid_of("corpus/p00042.md") == 42
    assert runner._pid_of("Concepts/Something") is None
    assert runner._rel(42) == "corpus/p00042.md"


def test_recall_and_rank_helpers():
    gold = {1, 4}
    ranked = [7, 1, 9, 4]
    assert runner._recall(gold, ranked, 2) == 0.5
    assert runner._recall(gold, ranked, 4) == 1.0
    assert runner._first_gold_rank(gold, ranked) == 2
    assert runner._first_gold_rank({99}, ranked) is None
