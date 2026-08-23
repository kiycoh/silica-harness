# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Pure math of CORRELATE (ADR-0013, ADR-0029): top-k stem selection + Jaccard.

These functions are the metric the spec's admission gate measured
(top-30 by raw count, Jaccard, tau=0.25). They know nothing about the store —
they operate on plain {stem: count} maps and frozensets.
"""
from __future__ import annotations

import pytest

from silica.kernel.link.correlate import compute_edges, jaccard, topk_set


def _nodes(**counts: int) -> dict:
    """A synthetic contribution with exactly these {stem: count} nodes.

    Bypasses the tokenizer so a fixture controls the top-k sets precisely.
    """
    return {"nodes": {s: {"label": s, "count": c} for s, c in counts.items()}, "edges": []}


def test_topk_set_keeps_highest_counts() -> None:
    nodes = {"quick": 10, "sort": 8, "boilerplate": 1}
    assert topk_set(nodes, k=2) == frozenset({"quick", "sort"})


def test_topk_set_tie_break_is_lexicographic() -> None:
    # same count: the lexicographically smaller stem wins the cut, so it is
    # deterministic across runs and machines (no dict-order leakage).
    nodes = {"beta": 5, "alpha": 5, "gamma": 5}
    assert topk_set(nodes, k=2) == frozenset({"alpha", "beta"})


def test_topk_set_returns_all_when_fewer_than_k() -> None:
    nodes = {"quick": 3, "sort": 2}
    assert topk_set(nodes, k=30) == frozenset({"quick", "sort"})


def test_topk_set_of_empty_is_empty() -> None:
    assert topk_set({}) == frozenset()


def test_jaccard_is_intersection_over_union() -> None:
    a = frozenset({"quick", "sort", "array"})
    b = frozenset({"quick", "sort", "tree"})
    # |{quick, sort}| / |{quick, sort, array, tree}| = 2 / 4
    assert jaccard(a, b) == 0.5


def test_jaccard_of_empty_sets_is_zero_not_a_crash() -> None:
    # a note with no stems must not blow up the metric with a ZeroDivisionError.
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_jaccard_disjoint_is_zero_identical_is_one() -> None:
    a = frozenset({"quick", "sort"})
    assert jaccard(a, frozenset({"tree", "heap"})) == 0.0
    assert jaccard(a, a) == 1.0


def _topk(**counts: int) -> dict[str, int]:
    return dict(counts)


def test_compute_edges_creates_edge_for_overlapping_notes() -> None:
    # top-k(a)={quick,sort,array,tree}, top-k(b)={quick,sort,array,heap}
    # jaccard = |3 shared| / |5 union| = 0.6 >= tau
    adj = compute_edges({"a": _topk(quick=1, sort=1, array=1, tree=1),
                         "b": _topk(quick=1, sort=1, array=1, heap=1)})
    assert adj == {"a": {"b": pytest.approx(0.6)}, "b": {"a": pytest.approx(0.6)}}


def test_compute_edges_no_edge_below_tau() -> None:
    adj = compute_edges({"a": _topk(quick=1, sort=1, array=1),
                         "b": _topk(bread=1, flour=1, yeast=1)})
    assert adj == {}


def test_compute_edges_is_a_pure_function_of_its_input() -> None:
    nodes = {"a": _topk(quick=1, sort=1, array=1, tree=1),
             "b": _topk(quick=1, sort=1, array=1, heap=1),
             "c": _topk(bread=1, flour=1, yeast=1, water=1),
             "d": _topk(bread=1, flour=1, yeast=1, salt=1)}
    assert compute_edges(nodes) == compute_edges(dict(nodes))
    nodes["a"] = _topk(xxx=1, yyy=1)   # a diverged: only its edges change
    adj = compute_edges(nodes)
    assert "a" not in adj and "b" not in adj
    assert adj["c"] == {"d": pytest.approx(0.6)}
