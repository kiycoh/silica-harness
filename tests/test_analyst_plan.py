"""Tests for silica/kernel/analyst_plan.py.

Verifies three-tier confidence logic, §3.2-bis irreversible-capability guard,
and AnalystPlan structure.
"""
from __future__ import annotations

import pytest

from silica.kernel.report.graph_report.compute import _empty_report
from silica.kernel.report.graph_report import (
    AutolinkCandidate,
    BridgeStat,
    ClusterStat,
    NodeStat,
    VaultReport,
)
from silica.kernel.analyst_plan import (
    AnalystPlan,
    TaskCandidate,
    _CLUSTER_SIZE_THRESHOLD,
    _DANGLING_REFS_THRESHOLD,
    _IRREVERSIBLE,
    build_task_plan,
)


def _node(nid: str, label: str = "", cluster: int = 0, degree: int = 5) -> NodeStat:
    return NodeStat(id=nid, label=label or nid, cluster=cluster, out_degree=degree // 2, in_degree=degree // 2, degree=degree)


def _cluster(cid: int, size: int, hub: str, members: list[str]) -> ClusterStat:
    return ClusterStat(cluster_id=cid, size=size, hub=hub, members=members, cohesion=0.5)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_build_task_plan_returns_analyst_plan():
    r = _empty_report()
    plan = build_task_plan(r)
    assert isinstance(plan, AnalystPlan)
    assert isinstance(plan.auto, list)
    assert isinstance(plan.propose, list)
    assert isinstance(plan.escalate, list)
    assert len(plan.checkpoints) == 3


def test_empty_vault_produces_empty_plan():
    r = _empty_report()
    plan = build_task_plan(r)
    assert plan.auto == []
    assert plan.propose == []
    assert plan.escalate == []


# ---------------------------------------------------------------------------
# Orphans → auto / propose
# ---------------------------------------------------------------------------

def test_orphan_with_matching_title_goes_auto():
    """An orphan whose stem matches another existing note's title → auto."""
    r = _empty_report()
    # Orphan "Concepts/Alpha"; another god-node "Hub/Alpha" has the same stem "alpha"
    r.god_nodes = [
        _node("Hub/Alpha",   label="Alpha",   degree=6),
        _node("Concepts/AlphaOrphan", label="AlphaOrphan", degree=1),
    ]
    # "AlphaOrphan" stem = "alphaorphan" which contains "alpha" (from "Hub/Alpha" stem)
    r.orphans = ["Concepts/AlphaOrphan"]
    r.totals["orphans"] = 1
    plan = build_task_plan(r)
    auto_caps = [c.capability_name for c in plan.auto]
    assert "silica_backlink" in auto_caps
    assert all(c.tier == "auto" for c in plan.auto if c.capability_name == "silica_backlink")


def test_orphan_without_match_goes_propose():
    """An orphan with no matching title → propose."""
    r = _empty_report()
    r.orphans = ["Notes/Zeta999"]
    r.totals["orphans"] = 1
    plan = build_task_plan(r)
    propose_caps = [c.capability_name for c in plan.propose]
    assert "silica_backlink" in propose_caps
    # Nothing in auto for this case (no matching title)
    auto_titles = []
    for c in plan.auto:
        auto_titles.extend(c.payload.get("new_titles", []))
    assert "Zeta999" not in auto_titles


def test_orphan_repair_injects_into_peers_not_into_the_orphan():
    """The plan must BACKLINK an orphan, never autolink it.

    An orphan is a note with in-degree 0, so scanning its own body for mentions
    of other titles gives it out-degree and leaves it orphaned. The link has to
    go the other way: into its cluster peers, naming the orphan.
    """
    r = _empty_report()
    r.orphans = ["Concepts/Lonely"]
    r.totals["orphans"] = 1
    r.clusters = [
        _cluster(0, size=3, hub="Concepts/Hub",
                 members=["Concepts/Lonely", "Concepts/Hub", "Concepts/Peer"]),
    ]
    plan = build_task_plan(r)
    orphan_tasks = [c for c in plan.auto + plan.propose if c.capability_name == "silica_backlink"]
    assert len(orphan_tasks) == 1
    payload = orphan_tasks[0].payload
    assert payload["new_titles"] == ["Lonely"], "the orphan is what gets linked TO"
    assert "Concepts/Hub" in payload["neighbourhood"]
    assert "Concepts/Peer" in payload["neighbourhood"]
    # And nothing anywhere in the plan autolinks the orphan itself.
    assert all(
        "Concepts/Lonely" not in c.payload.get("note_paths", [])
        for c in plan.auto + plan.propose
    )


def test_orphan_without_a_cluster_gets_an_empty_neighbourhood():
    """No cluster (-1) means no peers to scan — a no-op, never a full-vault sweep."""
    r = _empty_report()
    r.orphans = ["Notes/Nowhere"]
    r.totals["orphans"] = 1
    plan = build_task_plan(r)
    tasks = [c for c in plan.auto + plan.propose if c.capability_name == "silica_backlink"]
    assert tasks and tasks[0].payload["neighbourhood"] == []





# ---------------------------------------------------------------------------
# Oversized clusters → propose
# ---------------------------------------------------------------------------

def test_oversized_cluster_goes_propose():
    big_size = _CLUSTER_SIZE_THRESHOLD + 5
    hub = "Concepts/BigHub"
    members = [f"Note/{i}" for i in range(big_size)]
    r = _empty_report()
    r.clusters = [_cluster(0, big_size, hub, members)]
    r.totals["clusters"] = 1
    plan = build_task_plan(r)
    propose_caps = [c.capability_name for c in plan.propose]
    assert "silica_graph_explain" in propose_caps


def test_small_cluster_not_in_propose():
    r = _empty_report()
    r.clusters = [_cluster(0, 5, "X", ["X", "Y", "Z", "W", "V"])]
    r.totals["clusters"] = 1
    plan = build_task_plan(r)
    explain_tasks = [c for c in plan.propose if c.capability_name == "silica_graph_explain"]
    assert explain_tasks == []


# ---------------------------------------------------------------------------
# Dangling wikilinks → escalate
# ---------------------------------------------------------------------------

def test_recurring_dangling_goes_escalate():
    r = _empty_report()
    r.dangling = [{"target": "MissingNote", "refs": _DANGLING_REFS_THRESHOLD}]
    plan = build_task_plan(r)
    assert len(plan.escalate) >= 1
    reasons = [c.reason for c in plan.escalate]
    assert any("MissingNote" in reason for reason in reasons)


def test_single_dangling_not_escalated():
    """Dangling with refs < threshold should NOT appear in escalate."""
    r = _empty_report()
    r.dangling = [{"target": "Rare", "refs": _DANGLING_REFS_THRESHOLD - 1}]
    plan = build_task_plan(r)
    rare_tasks = [c for c in plan.escalate if "Rare" in c.reason]
    assert rare_tasks == []


# ---------------------------------------------------------------------------
# §3.2-bis: irreversible capabilities never in auto
# ---------------------------------------------------------------------------

def test_no_irreversible_capability_in_auto():
    r = _empty_report()
    r.orphans = ["Notes/SomeOrphan"]
    r.autolink_candidates = [AutolinkCandidate(source="A", target="B", weight=0.9, shared=["x"])]
    r.clusters = [_cluster(0, _CLUSTER_SIZE_THRESHOLD + 10, "H", ["H"] * (_CLUSTER_SIZE_THRESHOLD + 10))]
    r.dangling = [{"target": "X", "refs": 5}]
    plan = build_task_plan(r)
    for candidate in plan.auto:
        assert candidate.capability_name not in _IRREVERSIBLE, (
            f"Irreversible capability '{candidate.capability_name}' found in plan.auto"
        )


def test_all_auto_tiers_correct():
    r = _empty_report()
    r.god_nodes = [_node("Notes/Match", label="Match", degree=3)]
    r.orphans = ["Notes/Match"]
    plan = build_task_plan(r)
    for c in plan.auto:
        assert c.tier == "auto"
    for c in plan.propose:
        assert c.tier == "propose"
    for c in plan.escalate:
        assert c.tier == "escalate"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

