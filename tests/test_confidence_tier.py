"""Confidence → tier: the tier follows an edge's provenance, not a flat default.

Vocabulary ported from Graphify (MIT): EXTRACTED / INFERRED / AMBIGUOUS. Applied
to the autolink candidates the relatedness facade proposes (ADR-0029): a shared
concept is textual evidence (INFERRED → propose); an associative-only pair needs
a human (AMBIGUOUS → escalate). Never EXTRACTED: >2 hops by construction.
"""
from __future__ import annotations

from silica.kernel.report.graph_report import AutolinkCandidate
from silica.kernel.report.graph_report.compute import _empty_report
from silica.kernel.analyst_plan import (
    build_task_plan,
    classify_autolink,
)





def test_classify_autolink_inferred_when_concepts_shared() -> None:
    # A directly shared concept is textual evidence the two notes cover the same thing.
    cand = AutolinkCandidate(source="A", target="B", weight=3.0, shared=["neural network"])
    assert classify_autolink(cand) == "INFERRED"


def test_classify_autolink_ambiguous_when_associative_only() -> None:
    # No shared concept → related only through transitive expansion → needs a human.
    cand = AutolinkCandidate(source="A", target="B", weight=3.0, shared=[])
    assert classify_autolink(cand) == "AMBIGUOUS"


def test_evidenced_autolink_candidate_is_proposed() -> None:
    # Embedder-free leg: a co-occurrence autolink with shared concepts enters the
    # plan as a propose (INFERRED), even with no embedding missing-links present.
    r = _empty_report()
    r.autolink_candidates = [AutolinkCandidate(source="A", target="B", weight=3.0, shared=["x"])]
    plan = build_task_plan(r)

    propose_sources = [p for c in plan.propose for p in c.payload.get("note_paths", [])]
    assert "A" in propose_sources
    assert any(
        c.confidence == "INFERRED"
        for c in plan.propose if "A" in c.payload.get("note_paths", [])
    )


def test_associative_autolink_candidate_is_escalated() -> None:
    # An associative-only pair (no shared concept) is flagged for human review.
    r = _empty_report()
    r.autolink_candidates = [AutolinkCandidate(source="A", target="B", weight=3.0, shared=[])]
    plan = build_task_plan(r)

    assert any(c.confidence == "AMBIGUOUS" for c in plan.escalate)
    # AMBIGUOUS is review-only: it must not auto-link.
    auto_sources = [p for c in plan.auto for p in c.payload.get("note_paths", [])]
    assert "A" not in auto_sources
