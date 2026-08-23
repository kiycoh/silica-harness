# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""The arithmetic of probe_collision_risk_coverage, offline and without a
model: the policy evaluation routes through the real `route_concept`, the
frontier keeps only Pareto points, the AURC integrates risk over coverage
from the all-defer origin, and the source sample spreads over domains.
"""
from __future__ import annotations

from evals.probe_collision_risk_coverage import (
    Incoming, aurc, evaluate, frontier, routed, sample_sources,
)


def _rec(key: str, title: str, target: str | None, score: float, *, dup: str | None = None) -> Incoming:
    rec = Incoming(key=key, title=title, excerpt="", dups=frozenset({dup} if dup else ()),
                   exclude=frozenset(), synthetic=dup is not None)
    rec.scalars["s"] = (target, score)
    return rec


def test_evaluate_counts_false_merges_leaks_and_abstentions_through_the_real_router():
    records = [
        # distinct note, names agree, cosine over tau_high -> mechanical patch = FALSE MERGE
        _rec("a", "Entropia", "x/Entropia", 0.95),
        # distinct note in the band -> deferred, no mistake
        _rec("b", "Gradiente", "x/Discesa del gradiente", 0.80),
        # distinct note under tau_low -> keep, correct
        _rec("c", "Kernel", "x/Olio", 0.40),
        # duplicate under tau_low -> keep = LEAK
        _rec("d", "Maximum likelihood", "x/Stimatore", 0.50, dup="x/Stimatore"),
        # duplicate in the band -> deferred
        _rec("e", "ML estimator", "x/Stimatore", 0.80, dup="x/Stimatore"),
        # cold path: no candidate at all -> keep; with a duplicate that is a leak too
        _rec("f", "Orfano", None, 0.0, dup="x/Qualcosa"),
    ]
    res = evaluate(routed(records, "s", set()), tau_high=0.85, tau_low=0.75)
    assert res["false_merges"] == 1 and res["leaks"] == 2 and res["deferred"] == 2
    assert res["auto"] == 4 and res["coverage"] == round(4 / 6, 4) and res["risk"] == 0.75


def test_evaluate_patch_into_the_duplicate_is_not_an_error():
    rec = _rec("a", "Entropia", "x/Entropia", 0.95, dup="x/Entropia")
    res = evaluate(routed([rec], "s", set()), tau_high=0.85, tau_low=0.75)
    assert res["false_merges"] == 0 and res["leaks"] == 0 and res["auto"] == 1 and res["risk"] == 0.0


def test_evaluate_hub_target_uses_the_lowered_threshold():
    rec = _rec("a", "Entropia", "x/Entropia", 0.80)
    assert evaluate(routed([rec], "s", set()), tau_high=0.85, tau_low=0.75)["deferred"] == 1
    assert evaluate(routed([rec], "s", {"x/Entropia"}), tau_high=0.85, tau_low=0.75)["false_merges"] == 1


def test_frontier_keeps_only_pareto_points_sorted_by_coverage():
    pts = [
        {"coverage": 0.5, "risk": 0.02}, {"coverage": 0.5, "risk": 0.01},   # same coverage: lower risk wins
        {"coverage": 0.9, "risk": 0.05},
        {"coverage": 0.7, "risk": 0.06},                                    # dominated by (0.9, 0.05)
        {"coverage": 0.2, "risk": 0.0},
    ]
    front = frontier(pts)
    assert [(p["coverage"], p["risk"]) for p in front] == [(0.2, 0.0), (0.5, 0.01), (0.9, 0.05)]


def test_aurc_integrates_from_the_all_defer_origin():
    assert aurc([]) == 0.0
    # one point at full coverage and risk r: a triangle of area r/2
    assert aurc([{"coverage": 1.0, "risk": 0.1}]) == 0.05
    # a flat zero-risk frontier has no area
    assert aurc([{"coverage": 0.5, "risk": 0.0}, {"coverage": 1.0, "risk": 0.0}]) == 0.0


def test_sample_sources_round_robins_over_domains_and_skips_short_bodies():
    keys = [f"A/n{i}" for i in range(10)] + [f"B/m{i}" for i in range(2)] + ["C/short"]
    bodies = {k: "x" * 500 for k in keys}
    bodies["C/short"] = "x" * 10
    picked = sample_sources(keys, bodies, n=6, seed=3)
    assert len(picked) == 6
    assert sum(1 for k in picked if k.startswith("B/")) == 2   # the small domain is exhausted, not skipped
    assert "C/short" not in picked
    assert picked == sample_sources(keys, bodies, n=6, seed=3)


def test_judge_scoring_single_arm_cannot_be_right_when_the_winner_is_not_the_source():
    from evals.probe_collision_risk_coverage import score_single

    dup = _rec("s", "ML estimator", "x/Altro", 0.8, dup="x/Stimatore")
    assert not score_single(dup, "x/Altro", "duplicate")     # merges into the wrong note
    assert not score_single(dup, "x/Altro", "distinct")      # writes a second note
    assert score_single(dup, "x/Stimatore.md", "duplicate")  # the handed candidate is the source
    distinct = _rec("d", "Kernel", "x/Olio", 0.8)
    assert score_single(distinct, "x/Olio", "distinct")
    assert score_single(distinct, "x/Olio", "contradicts")
    assert not score_single(distinct, "x/Olio", "duplicate")


def test_judge_scoring_slate_needs_the_right_pick_and_treats_none_as_distinct():
    from evals.probe_collision_risk_coverage import score_slate

    slate = ("x/Altro.md", "x/Stimatore.md", "x/Terzo.md")
    dup = _rec("s", "ML estimator", None, 0.0, dup="x/Stimatore")
    assert score_slate(dup, slate, "duplicate", 2)
    assert not score_slate(dup, slate, "duplicate", 1)
    assert not score_slate(dup, slate, "duplicate", 0)
    assert not score_slate(dup, slate, "distinct", 0)
    assert not score_slate(dup, slate, "duplicate", 9)   # an index off the slate is no pick
    distinct = _rec("d", "Kernel", None, 0.0)
    assert score_slate(distinct, slate, "distinct", 0)
    assert not score_slate(distinct, slate, "duplicate", 1)
