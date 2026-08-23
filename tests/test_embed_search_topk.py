"""Task 3.1 — EmbedStore._search: heapq.nlargest instead of a full sort.

Pure speedup: heapq.nlargest(k, results) is documented-equivalent to
sorted(results, reverse=True)[:k], so output must stay byte-identical. These
tests pin that equivalence (several k values, an exact score tie with
path-descending tie-break, a k > n case, and an exclude case) plus a whitebox
check that the optimization is actually wired in via a module-level
`import heapq`.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from silica.kernel.recall.embed import EmbedStore


def _store(tmp_path):
    """Five notes with hand-picked, unit-length vectors so cosine scores
    against query [1.0, 0.0] are exact and predictable:
        n_high  -> 1.0
        n_tie_b -> 0.8   (exact tie with n_tie_a; 'n_tie_b' > 'n_tie_a' so it
        n_tie_a -> 0.8    sorts first under path-descending tie-break)
        n_mid   -> 0.6
        n_low   -> 0.0
    """
    s = EmbedStore(path=tmp_path / "embeddings.json")
    s.upsert("n_high", "High", [1.0, 0.0])
    s.upsert("n_mid", "Mid", [0.6, 0.8])
    s.upsert("n_low", "Low", [0.0, 1.0])
    s.upsert("n_tie_a", "TieA", [0.8, 0.6])
    s.upsert("n_tie_b", "TieB", [0.8, 0.6])
    return s


def _reference(store, query, k, exclude=None):
    """Independent full-sort reference — the algorithm _search used before
    this task, re-derived here rather than imported, so it can't drift in
    lockstep with a broken implementation."""
    exclude = exclude or set()
    store._ensure_matrix()
    scores = {p: 0.0 for p in store._notes}
    q = np.asarray(query, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    if q_norm and store._mat is not None and store._mat.size and store._mat_dim == q.shape[0]:
        sims = store._mat @ (q / q_norm)
        for path, sim in zip(store._mat_paths, sims.tolist()):
            scores[path] = sim
    results = [(sc, p) for p, sc in scores.items() if p not in exclude]
    results.sort(reverse=True)
    return [
        {"path": path, "name": store._notes[path]["name"], "score": round(float(sc), 4)}
        for sc, path in results[:k]
    ]


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_cosine_top_k_matches_full_sort_reference(tmp_path, k):
    store = _store(tmp_path)
    query = [1.0, 0.0]
    assert store.cosine_top_k(query, k=k) == _reference(store, query, k)


def test_tie_break_is_path_descending(tmp_path):
    store = _store(tmp_path)
    results = store.cosine_top_k([1.0, 0.0], k=5)
    assert [r["path"] for r in results] == [
        "n_high", "n_tie_b", "n_tie_a", "n_mid", "n_low",
    ]
    assert results[1]["score"] == results[2]["score"] == pytest.approx(0.8)


def test_k_greater_than_total_notes(tmp_path):
    store = _store(tmp_path)
    query = [1.0, 0.0]
    results = store.cosine_top_k(query, k=10)  # only 5 notes exist
    assert len(results) == 5
    assert results == _reference(store, query, 10)


def test_exclude_removes_top_hit(tmp_path):
    store = _store(tmp_path)
    query = [1.0, 0.0]
    results = store.cosine_top_k(query, k=5, exclude={"n_high"})
    paths = [r["path"] for r in results]
    assert "n_high" not in paths
    assert paths[0] == "n_tie_b"
    assert results == _reference(store, query, 5, exclude={"n_high"})


def test_empty_matrix_all_zero_scores(tmp_path):
    store = _store(tmp_path)
    query = [0.0, 0.0]  # zero query → every note falls back to its 0.0 default
    results = store.cosine_top_k(query, k=5)
    assert all(r["score"] == 0.0 for r in results)
    assert results == _reference(store, query, 5)


def test_search_uses_heapq_nlargest(tmp_path):
    """Whitebox: pins that _search is actually implemented via
    heapq.nlargest (module-level `import heapq`), not just output-compatible
    with it by coincidence."""
    store = _store(tmp_path)
    from silica.kernel.recall import embed as embed_mod

    with patch.object(embed_mod.heapq, "nlargest", wraps=embed_mod.heapq.nlargest) as spy:
        store.cosine_top_k([1.0, 0.0], k=3)
    assert spy.called


def test_batch_with_k_at_or_above_store_size_never_returns_the_query_itself(tmp_path):
    """`cosine_top_k_batch` promises the per-note result; with k >= N the
    self row masked at -inf used to ride along as the last candidate."""
    from silica.kernel.recall.embed import EmbedStore
    store = EmbedStore(path=tmp_path / "e.json")
    store.upsert("A", "A", [1.0, 0.0, 0.0])
    store.upsert("B", "B", [0.9, 0.1, 0.0])
    store.upsert("C", "C", [0.0, 1.0, 0.0])
    batch = store.cosine_top_k_batch(["A"], k=25)["A"]
    single = store.cosine_top_k(store.get_vec("A"), k=25, exclude={"A"})
    assert [c["path"] for c in batch] == [c["path"] for c in single] == ["B", "C"]
    assert all(c["score"] > float("-inf") for c in batch)
