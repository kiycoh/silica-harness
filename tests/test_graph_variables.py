# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Seven inter-note variables (docs/superpowers/specs/2026-08-22-graph-variables-design.md).

Pure kernel functions first (silica.kernel.recall.signals), then the seams
that consume them: report wiring, relatedness legs, curator, learner, tools.
"""
from __future__ import annotations

import math

import networkx as nx
import pytest

from silica.kernel.recall import signals


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _path_graph() -> nx.Graph:
    """A - B - C - D  plus hub H linked to A, B, C (so A,C share B and H)."""
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("H", "A"), ("H", "B"), ("H", "C")])
    return G


# ---------------------------------------------------------------------------
# V1 Adamic-Adar
# ---------------------------------------------------------------------------

def test_adamic_adar_ranks_non_adjacent_by_rare_common_neighbours():
    G = _path_graph()
    ranking = signals.adamic_adar_ranking(G, "A", k=5)
    assert ranking is not None
    paths = [p for p, _ in ranking]
    # A and C share B (deg 3) and H (deg 3): the only distance-2 candidate
    # besides D (no common neighbour) -> C first, B/H excluded (adjacent).
    assert paths[0] == "C"
    assert "B" not in paths and "H" not in paths and "A" not in paths
    score = dict(ranking)["C"]
    assert math.isclose(score, 2 / math.log(3), rel_tol=1e-9)


def test_adamic_adar_abstains_for_isolated_or_unknown_node():
    G = _path_graph()
    G.add_node("Z")
    assert signals.adamic_adar_ranking(G, "Z", k=5) is None
    assert signals.adamic_adar_ranking(G, "nope", k=5) is None


def test_adamic_adar_ignores_degree_one_common_neighbour_without_crashing():
    # log(1) == 0: a common neighbour of degree 1 cannot exist (it would have
    # to touch both endpoints), but a degree-2 neighbour is the minimum and
    # must not divide by zero.
    G = nx.Graph([("A", "M"), ("M", "B")])
    ranking = signals.adamic_adar_ranking(G, "A", k=5)
    assert ranking == [("B", pytest.approx(1 / math.log(2)))]


def test_structural_links_lists_every_distance_two_pair_once():
    G = _path_graph()
    links = signals.structural_links(G, top_k=10)
    pairs = {(u, v) for u, v, _s, _c in links}
    assert ("A", "C") in pairs
    assert all(u < v for u, v in pairs)
    assert all(not G.has_edge(u, v) for u, v in pairs)
    top = links[0]
    assert top[0:2] == ("A", "C") and set(top[3]) == {"B", "H"}


# ---------------------------------------------------------------------------
# V2 RefD
# ---------------------------------------------------------------------------

def test_refd_points_from_prerequisite_to_dependent():
    # Notes related to "backprop" (r1, r2) link to "gradient"; nothing related
    # to "gradient" links to "backprop" -> gradient is a prerequisite of backprop.
    links = {"r1": {"gradient"}, "r2": {"gradient"}, "gradient": set(), "backprop": set(), "g1": set()}
    related = {"backprop": [("r1", 1.0), ("r2", 1.0)], "gradient": [("g1", 1.0)]}
    edges = signals.refd_edges(links, related, theta=0.1)
    assert edges == [("gradient", "backprop", 1.0)]


def test_refd_is_antisymmetric_and_thresholded():
    links = {"r1": {"a"}, "q1": {"b"}, "a": set(), "b": set()}
    related = {"b": [("r1", 1.0)], "a": [("q1", 1.0)]}
    # a<-b evidence and b<-a evidence cancel: RefD == 0 -> no edge
    assert signals.refd_edges(links, related, theta=0.1) == []


def test_refd_weights_related_notes_by_their_score():
    links = {"r1": {"a"}, "r2": set(), "a": set(), "b": set()}
    related = {"b": [("r1", 3.0), ("r2", 1.0)], "a": []}
    edges = signals.refd_edges(links, related, theta=0.1)
    assert edges == [("a", "b", pytest.approx(0.75))]


def test_refd_abstains_below_the_related_floor_and_damps_hub_citers():
    links = {"idx": {"a", "b", "c", "d"}, "r1": {"a"}, "r2": set(), "r3": set()}
    related = {"x": [("idx", 1.0)], "y": [("r1", 1.0), ("r2", 1.0), ("r3", 1.0)]}
    # x's only related note is an index page: with a floor of 2 its side abstains
    assert signals.refd_edges(links, related, theta=0.1, min_related=2) == [
        ("a", "y", pytest.approx(1 / 3)),
    ]
    # without the floor, the index page testifies for each of its four links,
    # damped by 1/log2(1+4) so none reaches the weight a single citation has
    edges = signals.refd_edges(links, related, theta=0.1, min_related=1)
    assert ("a", "x", pytest.approx(1 / math.log2(5))) in edges
    assert all(e[2] <= 1 / math.log2(5) + 1e-9 for e in edges if e[1] == "x")


# ---------------------------------------------------------------------------
# V3 coupling
# ---------------------------------------------------------------------------

def test_coupling_weights_shared_transactions_by_their_size():
    small = {"a", "b"}
    big = {"a", "b", "c", "d"}
    adj = signals.coupling_adjacency([small, big])
    expected = 1 / math.log(1 + 2) + 1 / math.log(1 + 4)
    assert adj["a"]["b"] == pytest.approx(expected)
    assert adj["b"]["a"] == pytest.approx(expected)
    assert adj["c"]["d"] == pytest.approx(1 / math.log(5))
    assert "c" not in adj["a"] or adj["a"]["c"] == pytest.approx(1 / math.log(5))


def test_coupling_drops_transactions_over_the_cap_and_reports_them():
    huge = {f"n{i}" for i in range(40)}
    adj, dropped = signals.coupling_adjacency([huge, {"x", "y"}], cap=30, report_dropped=True)
    assert dropped == 1
    assert "n1" not in adj and adj["x"]["y"] > 0


def test_coupling_ranking_abstains_without_row():
    adj = signals.coupling_adjacency([{"a", "b"}])
    assert signals.coupling_ranking(adj, "zzz", k=5) is None
    assert signals.coupling_ranking(adj, "a", k=5) == [("b", pytest.approx(1 / math.log(3)))]


# ---------------------------------------------------------------------------
# V4 load-bearing
# ---------------------------------------------------------------------------

def test_load_bearing_finds_articulation_points_and_coreness():
    # Two triangles joined by the path c - X - d: three cut vertices, and X is
    # the one with the most paths through it and the fewest links.
    G = nx.Graph([("a", "b"), ("b", "c"), ("a", "c"), ("c", "X"), ("X", "d"),
                  ("d", "e"), ("e", "f"), ("d", "f")])
    bet = nx.betweenness_centrality(G)
    deg = dict(G.degree())
    G.add_edge("X", "X")  # a self-link must not break core_number
    core, articulation, surprise = signals.load_bearing(G, betweenness=bet, degree=deg)
    assert articulation == {"c", "X", "d"}
    assert core["a"] == 2 and core["X"] == 2
    assert surprise["X"] == 1.0 == max(surprise.values())
    assert -1.0 <= min(surprise.values())


def test_load_bearing_surprise_is_zero_when_rank_orders_agree():
    G = nx.path_graph(["p", "q", "r"])
    bet = nx.betweenness_centrality(G)
    deg = dict(G.degree())
    _core, _art, surprise = signals.load_bearing(G, betweenness=bet, degree=deg)
    assert surprise == {"p": 0.0, "q": 0.0, "r": 0.0}


# ---------------------------------------------------------------------------
# V5 dissonance
# ---------------------------------------------------------------------------

def test_dissonance_is_the_share_of_neighbours_in_another_zone():
    G = nx.Graph([("n", "a"), ("n", "b"), ("n", "c"), ("n", "u")])
    zone = {"n": 1, "a": 2, "b": 2, "c": 1}  # u has no zone: ignored
    d = signals.dissonance(G, zone)
    assert d["n"] == pytest.approx(2 / 3)
    assert d["a"] == 1.0 and d["c"] == 0.0
    assert "u" not in d


# ---------------------------------------------------------------------------
# V6 burst
# ---------------------------------------------------------------------------

def test_burst_flags_a_stem_concentrated_in_the_recent_window():
    day = 86400.0
    created = {f"old{i}": 0.0 for i in range(20)}
    created.update({f"new{i}": 100 * day for i in range(5)})
    stems = {p: {"base": 1} for p in created}
    for i in range(5):
        stems[f"new{i}"]["novel"] = 2   # only recent notes carry it
    stems["old0"]["novel"] = 1
    out = signals.burst(created, stems, window_days=14, min_recent=3, z_min=2.0)
    assert [s for s, *_ in out] == ["novel"]
    stem, z, n_recent, n_all = out[0]
    assert n_recent == 5 and n_all == 6 and z > 2


def test_burst_abstains_when_everything_is_recent_or_nothing_is():
    day = 86400.0
    created = {f"n{i}": i * day for i in range(5)}  # all inside 14 days
    stems = {p: {"x": 1} for p in created}
    assert signals.burst(created, stems) == []
    assert signals.burst({}, {}) == []


# ---------------------------------------------------------------------------
# V7 entropy
# ---------------------------------------------------------------------------

def test_entropy_bits_of_uniform_and_degenerate_distributions():
    assert signals.entropy_bits({"a": 1, "b": 1, "c": 1, "d": 1}) == pytest.approx(2.0)
    assert signals.entropy_bits({"a": 7}) == 0.0
    assert signals.entropy_bits({}) == 0.0


def test_sprawling_requires_both_breadth_and_flatness():
    stems = {f"focused{i}": {"a": 9, "b": 1} for i in range(10)}
    stems["broad"] = {f"s{j}": 1 for j in range(40)}          # many stems, flat
    stems["long_focused"] = {"a": 90, **{f"t{j}": 1 for j in range(39)}}  # many stems, peaked
    rows = signals.sprawling(stems, pct=0.9)
    assert [r[0] for r in rows] == ["broad"]


# ===========================================================================
# Report wiring
# ===========================================================================

from silica.kernel.recall.cooccurrence import CooccurStore, build_contribution
from silica.kernel.report.graph_report import (
    VaultReport,
    compute_report,
    to_digest,
    to_markdown,
)


def _node(nid, label, group, note_type="note"):
    return {"id": nid, "label": label, "group": group, "type": note_type}


def _edge(eid, src, dst, edge_type="EXTRACTED"):
    return {"id": eid, "from": src, "to": dst, "type": edge_type}


@pytest.fixture()
def graph_nodes_edges():
    """A-B-C triangle, D-E, bridge C->D, F orphan: C and D are cut vertices."""
    nodes = [
        _node("A", "Alpha", 0), _node("B", "Beta", 0), _node("C", "Gamma", 0),
        _node("D", "Delta", 1), _node("E", "Epsilon", 1), _node("F", "Phi", -1),
        {"id": "__unresolved__Ghost", "label": "Ghost", "group": -1, "type": "ghost"},
    ]
    edges = [
        _edge("e0", "A", "B"), _edge("e1", "B", "C"), _edge("e2", "A", "C"),
        _edge("e3", "D", "E"), _edge("e4", "C", "D"),
        _edge("e5", "F", "__unresolved__Ghost", "AMBIGUOUS"),
    ]
    return nodes, edges


@pytest.fixture()
def cooccur_store(tmp_path) -> CooccurStore:
    st = CooccurStore(path=tmp_path / "c.json", lang="english")
    st.upsert_note("A", build_contribution("A", "gradient descent gradient descent"))
    st.upsert_note("B", build_contribution("B", "backprop uses gradient descent"))
    st.upsert_note("C", build_contribution("C", "backprop chain rule"))
    st.upsert_note("D", build_contribution("D", "sailing boat harbour"))
    st.upsert_note("E", build_contribution("E", "sailing boat regatta"))
    st.upsert_note("F", build_contribution("F", "isolated topic"))
    return st


def test_core_map_is_populated_without_analytics(graph_nodes_edges):
    r = compute_report(_nodes_edges_override=graph_nodes_edges)
    assert set(r.core_map) == {"A", "B", "C", "D", "E", "F"}
    assert r.core_map["A"] == 2 and r.core_map["F"] == 0
    assert r.articulation == ["C", "D"]
    assert r.load_bearing == []          # analytics-only rows


def test_load_bearing_rows_with_analytics(graph_nodes_edges):
    r = compute_report(_nodes_edges_override=graph_nodes_edges, analytics=True)
    rows = {lb.path: lb for lb in r.load_bearing}
    assert {"C", "D"} <= set(rows)
    assert rows["C"].articulation and rows["D"].articulation
    assert rows["C"].coreness == 2 and rows["D"].coreness == 1
    assert r.load_bearing[0].articulation   # cut vertices rank first
    assert r.totals["load_bearing"] == len(r.load_bearing)


def test_structural_links_are_unlinked_distance_two_pairs(graph_nodes_edges):
    r = compute_report(_nodes_edges_override=graph_nodes_edges, analytics=True)
    pairs = {(s.source, s.target): s for s in r.structural_links}
    assert {("A", "D"), ("B", "D"), ("C", "E")} <= set(pairs)
    assert pairs[("B", "D")].common == ["C"]
    assert all(s.shared == [] for s in r.structural_links)   # no store: no labels
    assert r.totals["structural_links"] == len(r.structural_links)


def test_coupling_from_transactions_override(graph_nodes_edges):
    r = compute_report(
        _nodes_edges_override=graph_nodes_edges, analytics=True,
        _transactions_override=[{"A", "E"}, {"A", "B", "E"}, {"F"}],
    )
    assert r.coupling_map["A"]["E"] > r.coupling_map["A"]["B"]
    coupled = {(c.source, c.target): c for c in r.coupled_pairs}
    assert ("A", "E") in coupled             # unlinked: proposed
    assert ("A", "B") not in coupled         # wikilinked already
    assert ("B", "E") in coupled
    assert r.totals["coupled_pairs"] == 2


def test_cooccur_variables_annotate_shared_and_emit_prerequisites(graph_nodes_edges, cooccur_store, monkeypatch):
    from silica.kernel.report.graph_report import cooccur_delta as cd
    monkeypatch.setattr(cd, "_REFD_MIN_RELATED", 1)
    r = compute_report(
        _nodes_edges_override=graph_nodes_edges, analytics=True,
        with_cooccurrence=True, _cooccur_store_override=cooccur_store,
        _transactions_override=[{"D", "E"}, {"A", "E"}],
    )
    coupled = {(c.source, c.target): c for c in r.coupled_pairs}
    assert ("A", "E") in coupled and coupled[("A", "E")].shared == []
    # every prerequisite edge is mirrored in prereq_map, store keyspace
    for e in r.prerequisites:
        assert e.prereq in r.prereq_map[e.dependent]
        assert e.refd >= 0.1
    assert r.sprawling == []                 # six notes: under the population floor
    assert r.totals["prerequisites"] == len(r.prerequisites)


def test_prerequisite_direction_on_synthetic_citations(graph_nodes_edges, cooccur_store, monkeypatch):
    # R(A) = {B} (they share "gradient descent") and B links C (B's only
    # out-link, so no damping), while nothing related to C links A:
    # RefD(C -> A) = 1, so C is a prerequisite of A. Nothing may be a
    # prerequisite of itself. The related-set floor is lowered: six notes
    # cannot have five related each.
    from silica.kernel.report.graph_report import cooccur_delta as cd
    monkeypatch.setattr(cd, "_REFD_MIN_RELATED", 1)
    r = compute_report(
        _nodes_edges_override=graph_nodes_edges, analytics=True,
        with_cooccurrence=True, _cooccur_store_override=cooccur_store,
    )
    assert all(e.prereq != e.dependent for e in r.prerequisites)
    assert any(e.prereq == "C" and e.dependent == "A" and e.refd == 1.0 for e in r.prerequisites)


def test_render_shows_the_new_sections(graph_nodes_edges, cooccur_store, monkeypatch):
    from silica.kernel.report.graph_report import cooccur_delta as cd
    monkeypatch.setattr(cd, "_REFD_MIN_RELATED", 1)
    r = compute_report(
        _nodes_edges_override=graph_nodes_edges, analytics=True,
        with_cooccurrence=True, _cooccur_store_override=cooccur_store,
        _transactions_override=[{"A", "E"}],
    )
    md = to_markdown(r)
    for heading in ("## Load-Bearing Notes", "## Predicted Links", "## Coupled Notes",
                    "## Prerequisite Chains"):
        assert heading in md, heading
    digest = to_digest(r)
    assert "LOAD-BEARING" in digest and "PREDICTED" in digest and "COUPLED" in digest


def test_dissonance_from_a_real_embed_store(graph_nodes_edges, tmp_path, monkeypatch):
    import silica.kernel.recall.embed as embed_mod
    from silica.kernel.recall.embed import EmbedStore, get_store

    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")
    embed_mod.clear()
    store = get_store()
    vecs = {"A": [1.0, 0.0], "B": [0.9, 0.1], "C": [0.8, 0.2],
            "D": [0.0, 1.0], "E": [0.1, 0.9], "F": [0.05, 0.95]}
    for k, v in vecs.items():
        store.upsert(k, k, v, content_hash="h" + k)
    nodes, edges = graph_nodes_edges
    # F reads like D/E but links only into the A/B/C area: the misfiled case.
    edges = edges + [_edge("e6", "F", "A"), _edge("e7", "F", "B")]
    r = compute_report(
        _nodes_edges_override=(nodes, edges), analytics=True, with_embeddings=True,
        _dissonance_knn_k=1,
    )
    assert r.dissonance_map["F"] == 1.0
    assert r.dissonance_map["A"] < 0.5
    assert [m.path for m in r.misfiled] == ["F"]
    assert r.misfiled[0].degree == 2
    assert r.totals["misfiled"] == 1


def test_cowrite_transactions_are_scoped_to_the_vault_and_capped(tmp_path, monkeypatch):
    import orjson
    from silica.kernel.report import cowrite

    runs = tmp_path / "runs"
    monkeypatch.setattr(cowrite, "runs_root", lambda: runs)

    def run(rid, vault, paths):
        d = runs / rid
        d.mkdir(parents=True)
        (d / "ledger.json").write_bytes(orjson.dumps({"run_id": rid, "vault": vault}))
        (d / "manifest.json").write_bytes(orjson.dumps({
            "run_id": rid,
            "entries": [{"title": p, "path": p, "parent": None, "cluster_id": -1,
                         "source_basename": "s", "op": "write"} for p in paths],
        }))

    run("r1", "/v/x", ["a", "b", "c"])
    run("r2", "/v/y", ["a", "q"])
    run("r3", "/v/x", [f"n{i}" for i in range(40)])
    run("r4", "/v/x", ["solo"])
    txns, dropped = cowrite.cowrite_transactions("/v/x", cap=30)
    assert txns == [{"a", "b", "c"}]
    assert dropped == 1


# ===========================================================================
# Relatedness legs (V1 structural, V3 coupling)
# ===========================================================================

from silica.kernel.recall.relatedness import related_notes







# ===========================================================================
# Curator
# ===========================================================================

from silica.kernel.recall.curator import compose_curation_plan
from silica.kernel.report.graph_report import (
    CoupledPair,
    LoadBearingNote,
    MisfiledNote,
    SprawlingNote,
    StructuralLink,
)


def _report(**overrides) -> VaultReport:
    base = dict(generated_at="", scope="", totals={}, god_nodes=[], bridges=[],
                orphans=[], dangling=[], clusters=[])
    base.update(overrides)
    return VaultReport(**base)


def test_curator_autolinks_only_corroborated_structural_and_coupled_pairs():
    plan = compose_curation_plan(_report(
        structural_links=[
            StructuralLink("A.md", "B.md", 1.4, ["H.md"], shared=["gradient"]),
            StructuralLink("A.md", "C.md", 1.4, ["H.md"], shared=[]),
        ],
        coupled_pairs=[
            CoupledPair("D.md", "E.md", 0.9, shared=["sailing"]),
            CoupledPair("D.md", "F.md", 0.9, shared=[]),
        ],
    ))
    auto = {(i.target, i.partner): i for i in plan.by_kind("autolink")}
    assert set(auto) == {("A.md", "B.md"), ("D.md", "E.md")}
    assert auto[("A.md", "B.md")].reason.startswith("structural")
    assert auto[("D.md", "E.md")].reason.startswith("coupling")


def test_curator_keeps_load_bearing_notes_out_of_refine():
    plan = compose_curation_plan(_report(
        reformat_notes=["hub.md", "leaf.md"],
        articulation=["hub.md"],
    ))
    assert [i.target for i in plan.by_kind("refine")] == ["leaf.md"]
    assert [i.target for i in plan.vetoed] == ["hub.md"]
    assert "load-bearing" in plan.vetoed[0].reason
    assert plan.counts().get("refine") == 1


def test_curator_leaves_misfiled_and_sprawling_as_report_rows():
    # Judged 2026-08-22: neither list had precision over random notes, so the
    # composer must not turn them into work (ADR-0027).
    plan = compose_curation_plan(_report(
        misfiled=[MisfiledNote("x.md", 3, 1.0)],
        sprawling=[SprawlingNote("y", 42, 5.1, 0.95)],
    ))
    assert plan.is_empty()


# ===========================================================================
# Learner
# ===========================================================================

from silica.kernel.report import learner


def _notes(*paths, ai=True):
    return {p: {"created": 0.0, "ai": ai} for p in paths}


def test_review_queue_puts_ready_notes_before_blocked_ones():
    notes = _notes("a/intro.md", "a/advanced.md", "a/other.md")
    prereqs = {"a/advanced": ["a/intro"]}
    rows = learner.review_queue(
        limit=3, now_ts=1.0, _notes_override=notes, _entries_override=[],
        _prereqs_override=prereqs, _store=None,
    )
    paths = [r["path"] for r in rows]
    assert paths.index("a/intro.md") < paths.index("a/advanced.md")
    blocked = next(r for r in rows if r["path"] == "a/advanced.md")
    assert blocked["prereqs"] == ["a/intro"] and blocked["ready"] is False
    assert next(r for r in rows if r["path"] == "a/intro.md")["ready"] is True


def test_review_queue_target_mode_is_in_prerequisite_order():
    notes = _notes("a/c.md", "a/b.md", "a/a.md")
    prereqs = {"a/c": ["a/b"], "a/b": ["a/a"]}
    rows = learner.review_queue(
        target="a/", now_ts=1.0, _notes_override=notes, _entries_override=[],
        _prereqs_override=prereqs, _store=None,
    )
    assert [r["path"] for r in rows] == ["a/a.md", "a/b.md", "a/c.md"]


def test_review_queue_a_known_prerequisite_unblocks():
    now = 100.0 * 86400
    notes = _notes("a/intro.md", "a/advanced.md")
    entries = [{"ts": "2026-01-01T00:00:00+00:00", "path": "a/intro.md", "correct": True}]
    import datetime as dt
    ts = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).timestamp()
    rows = learner.review_queue(
        limit=2, now_ts=ts + 1, _notes_override=notes, _entries_override=entries,
        _prereqs_override={"a/advanced": ["a/intro"]}, _store=None,
    )
    adv = next(r for r in rows if r["path"] == "a/advanced.md")
    assert adv["ready"] is True


# ===========================================================================
# Tools
# ===========================================================================

def test_graph_explain_exposes_the_structural_role_and_reading_order(monkeypatch, graph_nodes_edges):
    import silica.kernel.report.graph_report as gr
    from silica.tools import atomic

    base = compute_report(_nodes_edges_override=graph_nodes_edges, analytics=True)
    base.prereq_map = {"C": ["A"], "D": ["C"]}
    base.dissonance_map = {"C": 0.25}
    monkeypatch.setattr(gr, "compute_report", lambda **kw: base)
    out = atomic.silica_graph_explain("C")
    d = out["diagnosis"]
    assert d["is_articulation"] is True and d["coreness"] == 2
    assert d["prerequisites"] == ["A"] and d["unlocks"] == ["D"]
    assert d["dissonance"] == 0.25
    assert isinstance(d["surprise"], float)


def test_review_queue_tool_rows_carry_prerequisites(monkeypatch):
    from silica.tools import atomic

    rows = [{"path": "a/x.md", "R": None, "why": "unexplored", "misses": 0, "attempts": 0,
             "ai": True, "prereqs": ["a/y"], "ready": False}]
    monkeypatch.setattr(learner, "review_queue", lambda **kw: rows)
    out = atomic.silica_review_queue(limit=1)
    assert out[0]["prereqs"] == ["a/y"] and out[0]["ready"] is False


# --- G5 semantic shift (structure.semantic_shift, replaces V7 flatness) ---

def _bind_vault(vault, monkeypatch):
    import silica.driver
    from silica.config import CONFIG

    vault.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(silica.driver, "_driver", None)


class _MarkerEmbedder:
    """text -> axis vector by topic marker: TOPIC_A/TOPIC_B orthogonal, so a
    note mixing both has MPD ~1 and a single-topic note has MPD ~0."""

    def embed(self, texts):
        return [[1.0, 0.0] if "TOPIC_A" in t else [0.0, 1.0] for t in texts]


def test_paragraphs_filter_short_and_cap():
    from silica.kernel.report.structure import _paragraphs
    body = "\n\n".join(["x" * 250] * 12 + ["short"] * 5)
    got = _paragraphs(body)
    assert len(got) == 8            # cap
    assert all(len(p) >= 200 for p in got)


def test_semantic_shift_separates_mixed_note_from_atomic(tmp_path, monkeypatch):
    import silica.agent.providers as providers
    from silica.kernel.report import structure

    _bind_vault(tmp_path / "v", monkeypatch)
    para_a = ("TOPIC_A " + "alpha " * 60).strip()
    para_b = ("TOPIC_B " + "beta " * 60).strip()
    (tmp_path / "v" / "mixed.md").write_text(
        "\n\n".join([para_a, para_b, para_a, para_b]), encoding="utf-8")
    (tmp_path / "v" / "atomic.md").write_text(
        "\n\n".join([para_a, para_a, para_a, para_a]), encoding="utf-8")
    monkeypatch.setattr(providers, "get_embedder", lambda cfg: _MarkerEmbedder())

    rows = structure._build_shift(k=5)
    by = {r["path"]: r for r in rows}
    assert by["mixed"]["mpd"] > by["atomic"]["mpd"]
    assert rows[0]["path"] == "mixed"          # worklist ranks the diluted note first
    assert by["atomic"]["mpd"] == 0.0          # identical vectors: no dilution


def test_semantic_shift_absent_when_embedder_down(tmp_path, monkeypatch):
    import silica.agent.providers as providers
    from silica.kernel.report import structure

    _bind_vault(tmp_path / "v", monkeypatch)
    (tmp_path / "v" / "a.md").write_text("\n\n".join(["x" * 250] * 4), encoding="utf-8")

    def _boom(cfg):
        raise RuntimeError("embedder down")

    monkeypatch.setattr(providers, "get_embedder", _boom)
    structure._shift_memo.clear()
    assert structure.semantic_shift() == []    # absent, never a fake zero row
