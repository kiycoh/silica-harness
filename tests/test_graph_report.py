"""Tests for silica/kernel/report/graph_report.py.

Uses a synthetic deterministic graph (2 Louvain clusters connected by one bridge
edge, one orphan note) without touching a live driver or Obsidian.
"""
from __future__ import annotations

import dataclasses

import pytest

from silica.kernel.report.graph_report.compute import _empty_report
from silica.kernel.report.graph_report import (
    BridgeStat,
    ClusterStat,
    MissingLink,
    NodeStat,
    VaultReport,
    compute_report,
    to_digest,
    to_facts,
    to_markdown,
    write_report,
)


# ---------------------------------------------------------------------------
# Synthetic graph fixture
#
# Layout:
#   Cluster 0: A ↔ B ↔ C   (triangle-ish)
#   Cluster 1: D ↔ E
#   Bridge:    C → D        (cross-cluster, single shared neighbour: none)
#   Orphan:    F             (no incoming links)
#
# Nodes: A, B, C (cluster 0), D, E (cluster 1), F (orphan, no cluster)
# EXTRACTED edges: A↔B, B↔C, A↔C, D↔E, C→D
# Ghost/AMBIGUOUS: F → __unresolved__Ghost
# ---------------------------------------------------------------------------

def _make_node(nid: str, label: str, group: int, note_type: str = "note") -> dict:
    return {"id": nid, "label": label, "group": group, "type": note_type}


def _make_edge(eid: str, src: str, dst: str, edge_type: str = "EXTRACTED") -> dict:
    return {"id": eid, "from": src, "to": dst, "type": edge_type}


@pytest.fixture()
def synthetic_graph():
    """Return (nodes, edges) for the synthetic test vault."""
    nodes = [
        _make_node("A", "Alpha",   group=0),
        _make_node("B", "Beta",    group=0),
        _make_node("C", "Gamma",   group=0),
        _make_node("D", "Delta",   group=1),
        _make_node("E", "Epsilon", group=1),
        _make_node("F", "Phi",     group=-1),  # orphan, no cluster
        # Ghost node for the unresolved link from F
        {"id": "__unresolved__Ghost", "label": "Ghost", "group": -1, "type": "ghost"},
    ]
    edges = [
        _make_edge("e0", "A", "B"),
        _make_edge("e1", "B", "C"),
        _make_edge("e2", "A", "C"),
        _make_edge("e3", "D", "E"),
        _make_edge("e4", "C", "D"),  # cross-cluster bridge
        _make_edge("e5", "F", "__unresolved__Ghost", "AMBIGUOUS"),
    ]
    return nodes, edges


@pytest.fixture()
def report(synthetic_graph):
    nodes, edges = synthetic_graph
    # Full report: these tests assert god_nodes/bridges/cohesion, which are the
    # analytics signals the on-demand /graph and /report commands consume.
    return compute_report(_nodes_edges_override=(nodes, edges), analytics=True)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_report_is_vault_report(report):
    assert isinstance(report, VaultReport)


def test_totals(report):
    t = report.totals
    assert t["notes"] == 6
    assert t["links"] == 5        # 5 EXTRACTED edges
    assert t["dangling_links"] == 1   # 1 AMBIGUOUS edge
    assert t["orphans"] >= 1      # F has no incoming links; D has 1 (from C)


def test_orphan_F_present(report):
    assert "F" in report.orphans


def test_god_nodes_sorted_by_degree(report):
    # The highest-degree node should appear first
    assert len(report.god_nodes) > 0
    degrees = [n.degree for n in report.god_nodes]
    assert degrees == sorted(degrees, reverse=True)


def test_god_nodes_no_ghost(report):
    """Ghost nodes must never appear in god_nodes."""
    for n in report.god_nodes:
        assert not n.id.startswith("__unresolved__")


def test_bridges_detected(report):
    """C→D is a cross-cluster bridge; the report must contain at least one bridge."""
    assert len(report.bridges) >= 1
    bridge_pairs = {(b.source, b.target) for b in report.bridges} | \
                   {(b.target, b.source) for b in report.bridges}
    assert ("C", "D") in bridge_pairs or ("D", "C") in bridge_pairs


def test_bridges_different_clusters(report):
    for b in report.bridges:
        assert b.source_cluster != b.target_cluster


def test_clusters_present(report):
    assert len(report.clusters) >= 1


def test_dangling_ghost_aggregated(report):
    """Ghost link from F should appear in dangling as target='Ghost', refs=1."""
    targets = {d["target"]: d["refs"] for d in report.dangling}
    assert "Ghost" in targets
    assert targets["Ghost"] == 1


# ---------------------------------------------------------------------------
# Empty vault / no edges degrades gracefully
# ---------------------------------------------------------------------------

def test_empty_vault_no_exception():
    nodes = [_make_node("X", "X", group=-1)]
    edges = []
    r = compute_report(_nodes_edges_override=(nodes, edges))
    assert isinstance(r, VaultReport)
    assert r.totals["notes"] == 1
    assert r.totals["links"] == 0
    # A single isolated node has degree=0 but still appears in god_nodes
    assert len(r.god_nodes) <= 1
    assert r.bridges == []
    assert r.clusters == []


def test_empty_report_helper():
    r = _empty_report("some/folder")
    assert r.scope == "some/folder"
    assert all(v == 0 for v in r.totals.values())


# ---------------------------------------------------------------------------
# to_facts
# ---------------------------------------------------------------------------

def test_to_facts_keys(report):
    facts = to_facts(report)
    assert set(facts.keys()) == {"scope", "totals", "god_nodes", "top_bridges", "orphan_count", "dangling_top"}


def test_to_facts_god_nodes_are_ids(report):
    facts = to_facts(report)
    # Each entry should be a string (node id)
    for gn in facts["god_nodes"]:
        assert isinstance(gn, str)


def test_to_facts_dangling_top_capped(report):
    facts = to_facts(report)
    assert len(facts["dangling_top"]) <= 5


# ---------------------------------------------------------------------------
# to_digest
# ---------------------------------------------------------------------------

def test_to_digest_non_empty(report):
    digest = to_digest(report)
    assert len(digest) > 0
    assert "VAULT AUDIT" in digest


def test_to_digest_empty_vault():
    r = _empty_report()
    digest = to_digest(r)
    assert "VAULT AUDIT" in digest
    assert "notes=0" in digest


def test_to_digest_contains_orphan(report):
    digest = to_digest(report)
    assert "ORPHANS" in digest
    assert "Phi" in digest or "F" in digest  # F's label is "Phi"


# ---------------------------------------------------------------------------
# to_markdown
# ---------------------------------------------------------------------------

def test_to_markdown_sections(report):
    md = to_markdown(report)
    assert "## Totals" in md
    assert "## God Nodes" in md
    assert "## Clusters" in md
    assert "## Orphans" in md
    assert "## Dangling Links" in md
    assert "## Surprising Cross-Cluster" in md


def test_to_markdown_no_proposed_section_when_empty(report):
    """Missing links section should be absent when missing_links is empty."""
    assert not report.missing_links
    md = to_markdown(report)
    assert "Proposed Missing Links" not in md


def test_to_markdown_proposed_section_when_present():
    r = _empty_report()
    r.missing_links = [MissingLink(source="X", target="Y", cosine=0.91)]
    md = to_markdown(r)
    assert "Proposed Missing Links" in md


def test_to_markdown_folds_long_lists_into_callouts():
    """Long lists fold into collapsed OFM callouts (`[!kind]-`); the health/tip
    summaries stay open. Wikilinks must survive inside the `>`-quoted callout."""
    r = _empty_report()
    r.totals = {"notes": 3, "orphans": 2, "dangling_links": 1, "lean_notes": 1}
    r.clusters = [ClusterStat(cluster_id=0, hub="Hub", size=2, cohesion=0.5, members=["A", "B"])]
    r.orphans = ["A", "B"]
    r.dangling = [{"target": "X", "refs": 2}]
    r.lean_notes = ["L"]
    md = to_markdown(r)
    assert "> [!warning] Health" in md          # health summary — open, not folded
    assert "> [!tip] Suggestions" in md         # fixes summary — open
    assert "> [!abstract]- [[Hub]]" in md       # cluster — folded
    assert "> [!warning]- 2 orphans" in md      # orphans — folded
    assert "> [!bug]- 1 broken links" in md     # dangling — folded
    assert "> - [[A]]" in md                    # wikilink bullet survives inside callout


def test_is_vault_artifact_matches_root_only():
    from silica.kernel.recall.graph_export import is_vault_artifact
    assert is_vault_artifact("GRAPH_REPORT.md")
    assert is_vault_artifact("log")
    assert not is_vault_artifact("Concepts/log.md")   # a real note in a subfolder
    assert not is_vault_artifact("Statistica.md")


def test_build_graph_data_excludes_vault_artifacts(tmp_vault):
    """GRAPH_REPORT.md/log.md are Silica's own output — they must stay out of the
    graph, or the report's own `[[...]]` would zero the orphan count next run."""
    from silica.kernel.recall.graph_export import build_graph_data

    tmp_vault.note("Real.md", "# Real\nNo links here.\n")
    tmp_vault.note("GRAPH_REPORT.md", "# Report\n[[Real]]\n")   # report links Real
    tmp_vault.note("log.md", "# Log\n[[Real]]\n")

    nodes, edges = build_graph_data()
    ids = {n["id"] for n in nodes}
    assert "GRAPH_REPORT.md" not in ids and "log.md" not in ids
    assert "Real.md" in ids
    # Real is linked ONLY by the artifacts -> excluded, it stays an orphan
    assert not any(e["to"] == "Real.md" for e in edges)


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------

def test_write_report_creates_files(tmp_path, report):
    out = str(tmp_path / "GRAPH_REPORT.md")
    result = write_report(report, out)
    assert "path_md" in result
    assert "path_json" in result
    import os
    assert os.path.exists(result["path_md"])
    assert os.path.exists(result["path_json"])


def test_write_report_json_deserializable(tmp_path, report):
    import orjson
    out = str(tmp_path / "GRAPH_REPORT.md")
    result = write_report(report, out)
    data = orjson.loads(open(result["path_json"], "rb").read())
    assert "totals" in data
    assert "god_nodes" in data


def test_write_report_json_handles_int_keyed_by_tier(tmp_path, report):
    """TemporalStat.by_tier is int-keyed; orjson rejects those without OPT_NON_STR_KEYS.

    The synthetic fixture builds from a nodes/edges override, so it never runs
    the on-disk triage that populates by_tier — set it explicitly.
    """
    import orjson
    report.temporal.by_tier = {3: 2, 1: 1}
    out = str(tmp_path / "GRAPH_REPORT.md")
    result = write_report(report, out)
    data = orjson.loads(open(result["path_json"], "rb").read())
    assert data["temporal"]["by_tier"] == {"3": 2, "1": 1}


# ---------------------------------------------------------------------------
# Determinism: same input → same output
# ---------------------------------------------------------------------------

def test_to_facts_byte_stable(synthetic_graph):
    """to_facts on identical input must produce identical dicts."""
    import orjson
    nodes, edges = synthetic_graph
    r1 = compute_report(_nodes_edges_override=(nodes, edges))
    r2 = compute_report(_nodes_edges_override=(nodes, edges))
    # generated_at will differ — compare only structural fields
    f1 = to_facts(r1)
    f2 = to_facts(r2)
    f1.pop("totals", None)  # totals are deterministic, but keep the check focused
    f2.pop("totals", None)
    assert orjson.dumps(f1, option=orjson.OPT_SORT_KEYS) == orjson.dumps(f2, option=orjson.OPT_SORT_KEYS)


# ---------------------------------------------------------------------------
# _compute_missing_links — common_neighbors structural boost (paper #2)
# ---------------------------------------------------------------------------

def test_missing_links_common_neighbors_boosts_ranking(monkeypatch):
    """Two candidates with equal cosine rank by shared-neighbor count.

    Paper (Marwitz et al. 2026) Baseline uses sum_i A^2_u,i (2-length path
    count) as a core feature. Silica equivalent: candidates sharing more
    common neighbors with the source are likelier to form a real link and
    must rank strictly higher than equally-similar but structurally-isolated
    candidates.
    """
    import networkx as nx
    from silica.kernel.report import graph_report as gr

    # S reaches A through two shared neighbors (X, Y) and B through one (Z).
    # Both A and B sit at shortest-path distance 2 from S (so both clear the
    # d_prev > 1 gate), and both will be returned with identical cosine.
    G = nx.Graph()
    G.add_edges_from([
        ("S", "X"), ("S", "Y"), ("S", "Z"),
        ("A", "X"), ("A", "Y"),
        ("B", "Z"),
    ])

    class _Store:
        def sync_from_disk(self):
            return False  # no file behind a fake (paths.DiskSynced)

        def __len__(self):
            return 6

        def get_vec(self, p):
            return [1.0, 0.0] if p == "S" else None

        def get_ts(self, p):
            return 0.0

        def cosine_top_k(self, vec, k=10, exclude=None):
            return [
                {"path": "A", "score": 0.90},
                {"path": "B", "score": 0.90},
            ]

    monkeypatch.setattr("silica.kernel.recall.embed.EmbedStore", _Store)
    monkeypatch.setattr("silica.agent.providers.get_embedder", lambda cfg: object())

    report = VaultReport(
        generated_at="x", scope="", totals={},
        god_nodes=[NodeStat(id="S", label="S", cluster=0,
                            out_degree=3, in_degree=0, degree=3)],
        bridges=[], orphans=[], dangling=[], clusters=[],
    )

    from silica.kernel.report.graph_report.embed_signals import _compute_missing_links
    links = _compute_missing_links(report, G, tau=0.5, k=10)
    by_target = {l.target: l for l in links}

    assert "A" in by_target and "B" in by_target
    # A shares 2 neighbors with S, B shares 1 → A must score strictly higher.
    assert by_target["A"].cosine > by_target["B"].cosine
    # …and the result list is ordered accordingly.
    pairs = [(l.source, l.target) for l in links]
    assert pairs.index(("S", "A")) < pairs.index(("S", "B"))


def test_missing_links_bridge_the_md_keyspace(monkeypatch):
    """Graph node ids carry '.md'; embed-store keys do not. _compute_missing_links
    must cross that boundary in both directions.

    Regression: it did not, so `tgt not in G_und` was true for every candidate and
    the section returned [] on every real vault. The other fixtures here use bare
    ids ("S"/"A"/"B") where the two keyspaces happen to coincide, which is exactly
    why the defect survived. This one uses the production shape.
    """
    import networkx as nx

    # S -- X -- A: A sits at distance 2 from S, so it clears the d_prev > 1 gate.
    G = nx.Graph()
    G.add_edges_from([("S.md", "X.md"), ("A.md", "X.md")])

    class _Store:
        def sync_from_disk(self):
            return False  # no file behind a fake (paths.DiskSynced)

        def __len__(self):
            return 3

        def get_vec(self, p):
            # Only ever answers to the STRIPPED key: proves the source side is bridged.
            return [1.0, 0.0] if p == "S" else None

        def get_ts(self, p):
            assert not p.endswith(".md"), "timestamps are keyed in the store keyspace"
            return 0.0

        def cosine_top_k(self, vec, k=10, exclude=None):
            assert exclude == {"S"}, "self-exclusion must use the store key"
            return [{"path": "A", "score": 0.90}]        # stripped, as the real store returns

    monkeypatch.setattr("silica.kernel.recall.embed.EmbedStore", _Store)
    monkeypatch.setattr("silica.agent.providers.get_embedder", lambda cfg: object())

    report = VaultReport(
        generated_at="x", scope="", totals={},
        god_nodes=[NodeStat(id="S.md", label="S", cluster=0,
                            out_degree=1, in_degree=0, degree=1)],
        bridges=[], orphans=[], dangling=[], clusters=[],
    )

    from silica.kernel.report.graph_report.embed_signals import _compute_missing_links
    links = _compute_missing_links(report, G, tau=0.5, k=10)

    # The proposal survives, and both ends are reported as graph node ids.
    assert [(l.source, l.target) for l in links] == [("S.md", "A.md")]


def test_duplicate_pairs_split_confirmed_vs_borderline(monkeypatch):
    """≥ τ_high → confirmed (merge candidate); τ_low..τ_high → borderline; ≤ τ_low dropped."""
    from silica.kernel.report import graph_report as gr

    nn = {  # each note's single nearest neighbour: (target, cosine)
        "a": ("b", 0.92),  # ≥ 0.85  → confirmed
        "c": ("d", 0.80),  # 0.75..0.85 → borderline
        "e": ("f", 0.50),  # ≤ 0.75  → dropped
    }

    class _Store:
        def sync_from_disk(self):
            return False  # no file behind a fake (paths.DiskSynced)

        def __len__(self): return len(nn)
        def paths(self): return list(nn)
        def get_vec(self, p): return [p] if p in nn else None
        def cosine_top_k(self, vec, k=1, exclude=None):
            tgt, score = nn[vec[0]]
            return [{"path": tgt, "score": score}]
        # The dedup leg ranks the whole scope in one call now.
        def cosine_top_k_batch(self, keys, k=1, *, exclude_self=True, block=256):
            return {p: self.cosine_top_k(self.get_vec(p), k, {p}) for p in keys if p in nn}

    monkeypatch.setattr("silica.kernel.recall.embed.EmbedStore", _Store)

    report = VaultReport(
        generated_at="x", scope="", totals={},
        god_nodes=[], bridges=[], orphans=[], dangling=[], clusters=[],
    )
    from silica.kernel.report.graph_report.embed_signals import _compute_duplicate_pairs
    borderline, confirmed = _compute_duplicate_pairs(report)

    assert [(d.source, d.target) for d in confirmed] == [("a", "b")]
    assert [(d.source, d.target) for d in borderline] == [("c", "d")]


# ---------------------------------------------------------------------------
# Contested notes (spec-hermes-coherence §1): analytics triage surfaces
# contested frontmatter so contradictions never silently harden into fact.
# ---------------------------------------------------------------------------

CONTESTED_NOTE = """---
AI: true
tags:
  - farmacologia
last modified: 2026, 07, 02
related:
  - "[[B]]"
contested: true
contradictions:
  - "source: appunti.md"
---

# A

corpo con [[B]]
"""

PLAIN_NOTE = """---
AI: true
tags:
  - t
last modified: 2026, 07, 02
related:
  - "[[A]]"
---

# B

corpo
"""


def test_contested_notes_surface_in_analytics_report(tmp_vault):
    tmp_vault.note("A.md", CONTESTED_NOTE)
    tmp_vault.note("B.md", PLAIN_NOTE)
    nodes = [_make_node("A", "A", group=0), _make_node("B", "B", group=0)]
    edges = [_make_edge("e0", "A", "B")]

    r = compute_report(_nodes_edges_override=(nodes, edges), analytics=True)

    assert [c.path for c in r.contested] == ["A"]
    assert r.contested[0].refs == ["source: appunti.md"]
    assert r.totals["contested"] == 1


def test_contested_skipped_without_analytics(tmp_vault):
    tmp_vault.note("A.md", CONTESTED_NOTE)
    nodes = [_make_node("A", "A", group=0)]
    r = compute_report(_nodes_edges_override=(nodes, []))
    assert r.contested == []


def test_contested_section_rendered(tmp_vault):
    tmp_vault.note("A.md", CONTESTED_NOTE)
    tmp_vault.note("B.md", PLAIN_NOTE)
    nodes = [_make_node("A", "A", group=0), _make_node("B", "B", group=0)]
    edges = [_make_edge("e0", "A", "B")]
    r = compute_report(_nodes_edges_override=(nodes, edges), analytics=True)

    md = to_markdown(r)
    assert "Contested" in md and "appunti.md" in md
    digest = to_digest(r)
    assert "contested" in digest.lower()


# ---------------------------------------------------------------------------
# Source drift (spec-hermes-coherence §3): note<->source drift via sha256
# provenance records. Embedder-free, pure read of .silica/provenance.json —
# no note bodies touched, unlike Contested above.
# ---------------------------------------------------------------------------

def test_source_drift_acceptance_v2_touching_half_drifts_the_other_half(tmp_vault):
    """Nucleate v1 (A,B) -> modify source -> re-nucleate v2 (A only) -> graph_report
    lists B as drifted from lezione-03.md."""
    from silica.kernel.write.provenance import append_record

    append_record("lezione-03.md", "sha-v1", "run1", ["A", "B"])
    append_record("lezione-03.md", "sha-v2", "run2", ["A"])

    nodes = [_make_node("A", "A", group=0), _make_node("B", "B", group=0)]
    r = compute_report(_nodes_edges_override=(nodes, []), analytics=True)

    assert [(d.note, d.source) for d in r.source_drift] == [("B", "lezione-03.md")]
    assert r.totals["source_drift"] == 1


def test_source_drift_empty_without_provenance_file(tmp_vault):
    """No .silica/provenance.json -> no drift, nothing fails (additive)."""
    nodes = [_make_node("A", "A", group=0)]
    r = compute_report(_nodes_edges_override=(nodes, []), analytics=True)
    assert r.source_drift == []
    assert r.totals["source_drift"] == 0


def test_source_drift_skipped_without_analytics(tmp_vault):
    from silica.kernel.write.provenance import append_record

    append_record("a.md", "sha1", "run1", ["A", "B"])
    append_record("a.md", "sha2", "run2", ["A"])

    nodes = [_make_node("A", "A", group=0), _make_node("B", "B", group=0)]
    r = compute_report(_nodes_edges_override=(nodes, []))
    assert r.source_drift == []


def test_source_drift_section_rendered(tmp_vault):
    from silica.kernel.write.provenance import append_record

    append_record("lezione-03.md", "sha-v1", "run1", ["A", "B"])
    append_record("lezione-03.md", "sha-v2", "run2", ["A"])

    nodes = [_make_node("A", "A", group=0), _make_node("B", "B", group=0)]
    r = compute_report(_nodes_edges_override=(nodes, []), analytics=True)

    md = to_markdown(r)
    assert "Source Drift" in md
    assert "lezione-03.md" in md
    assert "[[B]]" in md
    digest = to_digest(r)
    assert "drift" in digest.lower()


def test_source_drift_no_section_when_empty(tmp_vault):
    nodes = [_make_node("A", "A", group=0)]
    r = compute_report(_nodes_edges_override=(nodes, []), analytics=True)
    md = to_markdown(r)
    assert "Source Drift" not in md


def test_source_drift_matches_despite_md_suffix_on_node_ids(tmp_vault):
    """Regression: real vault node ids carry `.md` (driver index keys) while
    provenance notes are recorded WITHOUT it (RunManifestEntry.path strips
    the suffix). The id-form mismatch must not swallow the intersection —
    drift has to surface even when the graph node id is `Concepts/A.md` and
    the provenance note is `Concepts/A`."""
    from silica.kernel.write.provenance import append_record

    append_record("lezione-03.md", "sha-v1", "run1", ["Concepts/A", "Concepts/B"])
    append_record("lezione-03.md", "sha-v2", "run2", ["Concepts/A"])

    nodes = [
        _make_node("Concepts/A.md", "A", group=0),
        _make_node("Concepts/B.md", "B", group=0),
    ]
    r = compute_report(_nodes_edges_override=(nodes, []), analytics=True)

    assert [(d.note, d.source) for d in r.source_drift] == [("Concepts/B", "lezione-03.md")]
    assert r.totals["source_drift"] == 1


def test_to_digest_analytics_signals():
    # Tier A: shape/bet/coh/gaps/missing-hubs/integration reach the digest
    # (previously markdown-only, invisible to the agent).
    from silica.kernel.report.graph_report.models import (
        ClusterStat,
        IntegrationDeficit,
        MissingHub,
        NodeStat,
        StructuralGap,
        VaultReport,
    )

    r = VaultReport(
        generated_at="t", scope="", totals={},
        god_nodes=[NodeStat(id="a", label="A", cluster=0, out_degree=2,
                            in_degree=3, degree=5, betweenness=0.42)],
        bridges=[], orphans=[], dangling=[],
        clusters=[ClusterStat(cluster_id=0, size=3, hub="Hub/H.md",
                              members=["a"], cohesion=0.5)],
        structural_gaps=[StructuralGap(cluster_a=0, cluster_b=1, hub_a="Hub/H.md",
                                       hub_b="Other/O.md", inter_edges=0, gap_score=9.0)],
        missing_hubs=[MissingHub(concept="statistics", centrality=12.5)],
        integration_deficits=[IntegrationDeficit(path="Concepts/Rich", concepts=14,
                                                 degree=1, score=7.0)],
        discourse_state="Focused",
    )
    d = to_digest(r)
    assert "shape=Focused" in d
    assert "bet=0.42" in d
    assert "coh=0.5" in d
    assert "GAPS  H↮O(links=0)" in d
    assert "MISSING HUBS  statistics(cent=12.5)" in d
    assert "INTEGRATION DEFICIT  Rich(concepts=14,deg=1)" in d


def test_betweenness_map_populated_and_bridge_ranks_high(report):
    # Tier B: betweenness for ALL nodes (mirrors pagerank_map), not just hubs.
    # C is the sole cross-cluster bridge (C→D) → highest betweenness; the orphan
    # F (unresolved link only, no EXTRACTED edge) sits at zero.
    bm = report.betweenness_map
    assert set(bm) == {"A", "B", "C", "D", "E", "F"}
    assert bm["C"] == max(bm.values()) and bm["C"] > 0
    assert bm["F"] == 0.0


def test_betweenness_map_zero_without_analytics(synthetic_graph):
    nodes, edges = synthetic_graph
    r = compute_report(_nodes_edges_override=(nodes, edges), analytics=False)
    # Structural core skips betweenness entirely — map present but all-zero,
    # exactly like pagerank_map on this path.
    assert set(r.betweenness_map) == {"A", "B", "C", "D", "E", "F"}
    assert all(v == 0.0 for v in r.betweenness_map.values())


def test_fragmentation_components_counted(report):
    # {A,B,C,D,E} form one island; F (its only edge is unresolved) is the
    # second. AMBIGUOUS edges never stitch components together.
    assert report.totals["components"] == 2


def test_fragmentation_surfaces_in_health(report):
    assert "2 disconnected islands" in to_markdown(report)


def test_single_component_stays_out_of_health():
    nodes = [_make_node(n, n, group=0) for n in "AB"]
    edges = [_make_edge("e0", "A", "B")]
    r = compute_report(_nodes_edges_override=(nodes, edges))
    assert r.totals["components"] == 1
    assert "disconnected islands" not in to_markdown(r)


# ---------------------------------------------------------------------------
# Staging is not knowledge
#
# Opening the metrics panel on a 46-note vault showed E(vault) = +13.00, of
# which +12.00 was the orphan term — and all 12 orphans were Silica's own
# staging files: the converted chapters waiting in silica/Inbox/ and the
# consumed sources archived in silica/done/. The headline health number was
# measuring the inbox.
# ---------------------------------------------------------------------------

def _staging_graph():
    nodes = [
        _make_node("Concepts/Uriel.md", "Uriel", group=0),
        _make_node("Concepts/Angels.md", "Angels", group=0),
        _make_node("silica/Inbox/Enoch/03-chapter-3.md", "03-chapter-3", group=-1),
        _make_node("silica/done/01-enoch.md", "01-enoch", group=-1),
    ]
    edges = [_make_edge("e0", "Concepts/Uriel.md", "Concepts/Angels.md")]
    return nodes, edges


def test_inbox_and_done_files_are_not_orphans(monkeypatch):
    import silica.kernel.vault_manifest as vm

    monkeypatch.setattr(vm, "active_inbox_dir", lambda: "silica/Inbox")
    monkeypatch.setattr(vm, "active_write_dir", lambda: "silica")
    nodes, edges = _staging_graph()

    r = compute_report(_nodes_edges_override=(nodes, edges))

    assert r.orphans == ["Concepts/Uriel.md"]
    assert r.totals["orphans"] == 1


def test_edge_graph_is_the_same_graph_whatever_order_the_edges_arrive_in():
    """Sorting the NODES only fixed where Louvain starts. Each local move then
    scans `G[u]`, whose order is edge insertion order, and build_graph_data
    hands back a list whose order varies per process (dict/set iteration under
    hash randomisation). Measured 2026-08-22 on a 709-note vault: three runs of
    an unchanged vault gave 24, 24 and 23 areas, with the largest community at
    74, 74 and 96.

    Everything that persists a partition rides on this: cluster_id in
    clusters_ctx.json, the colour a community gets, E(vault), and the report
    history's `areas` delta, which otherwise reports movement nobody made.
    """
    import random

    from silica.kernel.recall.graph_export import edge_graph

    nodes = [{"id": f"n{i}.md", "type": "note"} for i in range(20)]
    edges = [{"from": f"n{i}.md", "to": f"n{(i * 7 + 3) % 20}.md", "type": "EXTRACTED"}
             for i in range(20)]

    def adjacency(order):
        G = edge_graph(nodes, order)
        return [(n, list(G[n])) for n in G]

    baseline = adjacency(edges)
    rng = random.Random(0)
    for _ in range(5):
        shuffled = edges[:]
        rng.shuffle(shuffled)
        assert adjacency(shuffled) == baseline
    # and the node order still cannot leak in either
    assert adjacency(edges) == adjacency(list(reversed(edges)))
