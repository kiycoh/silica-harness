# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""The arithmetic of probe_fusion_function, offline.

What must hold: the probe's RRF is the facade's RRF (same constant, same
cut), the convex combination normalises each leg between its theoretical
infimum and the query's best score and treats a note the cooccur leg never
scored as a zero, and the held-out loop never scores the cell it picked on
the half it picked it on.
"""
from __future__ import annotations

import pytest

from evals.probe_fusion_function import ALPHAS, cc, cc_scores, held_out, rrf


def test_rrf_matches_the_facade_constant_and_pool_cut():
    from silica.kernel.recall.relatedness import RRF_K

    embed = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    cooc = [("b", 5.0), ("c", 4.0), ("d", 1.0)]
    assert rrf([embed, cooc], pool=None, k=10) == ["b", "c", "a", "d"]
    # pool=1 sees only the leaders of each list; a note past the cut gets no term
    assert rrf([embed, cooc], pool=1, k=10) == ["a", "b"]
    # an abstaining leg contributes nothing and does not break the fusion
    assert rrf([embed, None], pool=None, k=2) == ["a", "b"]
    assert rrf([[("x", 1.0)]], pool=None, k=1) == ["x"]
    assert RRF_K == 60  # the probe imports the facade's constant; a drift here is a drift there


def test_cc_tmm_normalises_from_the_theoretical_infimum_and_scores_absent_cooccur_as_zero():
    embed = [("a", 0.9), ("b", 0.5)]      # cosine: infimum -1, best 0.9 -> span 1.9
    cooc = [("b", 4.0), ("c", 2.0)]       # overlap: infimum 0, best 4.0
    s = cc_scores(embed, cooc, alpha=0.75)
    assert s["a"] == pytest.approx(0.75 * (0.9 + 1.0) / 1.9)              # embed leader, no overlap
    assert s["b"] == pytest.approx(0.75 * (0.5 + 1.0) / 1.9 + 0.25 * 1.0)  # overlap leader
    assert s["c"] == pytest.approx(0.25 * 0.5)                            # cooccur only
    assert cc(embed, cooc, alpha=0.75, k=10) == ["b", "a", "c"]


def test_cc_with_one_leg_is_that_leg_in_order():
    embed = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
    assert cc(embed, None, alpha=0.7, k=2) == ["a", "b"]
    cooc = [("z", 9.0), ("y", 1.0)]
    assert cc(None, cooc, alpha=0.7, k=5) == ["z", "y"]
    assert cc(None, None, alpha=0.7, k=5) == []


def test_cc_alpha_one_is_the_embed_ranking_and_alpha_zero_the_cooccur_one():
    embed = [("a", 0.9), ("b", 0.5), ("c", 0.4)]
    cooc = [("c", 4.0), ("b", 2.0)]
    assert cc(embed, cooc, alpha=1.0, k=3) == ["a", "b", "c"]
    assert cc(embed, cooc, alpha=0.0, k=3)[:2] == ["c", "b"]


def test_alpha_grid_brackets_the_paper_range_on_both_sides():
    assert min(ALPHAS) < 0.6 and max(ALPHAS) > 0.8 and 0.8 in ALPHAS


def test_held_out_scores_the_picked_cell_on_the_other_half_only():
    pairs = [(f"q{i}", f"t{i}") for i in range(8)]
    # "good" recovers every pair, "bad" none; the reference recovers half
    good = {q: [t] for q, t in pairs} | {t: [q] for q, t in pairs}
    bad = {q: [] for q, t in pairs} | {t: [] for q, t in pairs}
    reference = {q: ([t] if i % 2 == 0 else []) for i, (q, t) in enumerate(pairs)}
    reference |= {t: [] for _q, t in pairs}
    res = held_out(pairs, {"good": good, "bad": bad}, reference=reference, splits=5, seed=1)
    assert res["picked"] == {"good": 5}
    assert res["heldout_recall"] == 1.0
    assert res["heldout_delta_mrr"] > 0
    # the delta is paired on the held-out half, so it is the reference's miss rate there
    assert 0.0 < res["heldout_delta_recall"] <= 1.0
    assert res["splits"] == 5


def test_held_out_is_deterministic_for_a_seed():
    pairs = [(f"q{i}", f"t{i}") for i in range(6)]
    cells = {"x": {q: [t] for q, t in pairs} | {t: [] for _q, t in pairs}}
    ref = {q: [] for q, _t in pairs} | {t: [] for _q, t in pairs}
    a = held_out(pairs, cells, reference=ref, splits=3, seed=7)
    b = held_out(pairs, cells, reference=ref, splits=3, seed=7)
    assert a == b
