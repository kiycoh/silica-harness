"""Two partitions over the same notes, never mixed (ADR-0023).

The structural one is Louvain on the wikilinks and owns node["group"]; the
semantic one is Louvain on the embedding k-NN and owns node["sgroup"]. They are
not nested on a real vault, so the tests below check the seam that keeps them
apart — distinct fields, distinct colour key, distinct persisted file — and the
id inheritance that keeps the semantic colours still across a recompute (the
partition is a global function of every vector: one note can lawfully reshuffle
it, and without inheritance the whole view would change colour).
"""
from __future__ import annotations

import orjson
import pytest

from silica.kernel.recall import graph_export as ge


def _nodes(*ids):
    return [{"id": i, "type": "note"} for i in ids]


def _clique(*ids, type="EXTRACTED"):
    return [{"from": a, "to": b, "type": type}
            for i, a in enumerate(ids) for b in ids[i + 1:]]


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """Point the semantic snapshot at a temp file (never the real vault index)."""
    p = tmp_path / "semantic_partition.json"
    monkeypatch.setattr(ge, "semantic_snapshot_path", lambda: p)
    monkeypatch.setattr("silica.kernel.recall.paths.vault_epoch", lambda: "test-epoch")
    return p


# --- The two partitions are independent -------------------------------------

def test_partitions_split_the_same_notes_differently(snapshot):
    """Same nodes, two edge layers that disagree — each field follows its own."""
    ids = ["a.md", "b.md", "x.md", "y.md"]
    nodes = _nodes(*ids)
    # Wikilinks pair a-b and x-y; the k-NN pairs a-x and b-y instead.
    wiki = _clique("a.md", "b.md") + _clique("x.md", "y.md")
    knn = _clique("a.md", "x.md", type="SIMILAR") + _clique("b.md", "y.md", type="SIMILAR")

    ge.detect_communities(nodes, wiki)
    ge.detect_semantic_partition(nodes, knn)

    group = {n["id"]: n["group"] for n in nodes}
    sgroup = {n["id"]: n["sgroup"] for n in nodes}
    assert group["a.md"] == group["b.md"] != group["x.md"] == group["y.md"]
    assert sgroup["a.md"] == sgroup["x.md"] != sgroup["b.md"] == sgroup["y.md"]


def test_semantic_partition_never_touches_the_structural_field(snapshot):
    nodes = _nodes("a.md", "b.md")
    ge.detect_communities(nodes, _clique("a.md", "b.md"))
    before = [n["group"] for n in nodes]
    colors = [n["color"] for n in nodes]

    ge.detect_semantic_partition(nodes, _clique("a.md", "b.md", type="SIMILAR"))

    assert [n["group"] for n in nodes] == before
    assert [n["color"] for n in nodes] == colors


def test_structural_detection_ignores_similar_edges():
    """The community stays a wikilink partition even when k-NN edges are present."""
    nodes = _nodes("a.md", "b.md")
    assert ge.detect_communities(nodes, _clique("a.md", "b.md", type="SIMILAR")) == []
    assert all(n.get("group", -1) == -1 for n in nodes)


def test_edge_graph_selects_the_named_layer():
    nodes = _nodes("a.md", "b.md")
    edges = _clique("a.md", "b.md", type="SIMILAR")
    assert ge.edge_graph(nodes, edges).number_of_edges() == 0
    assert ge.edge_graph(nodes, edges, edge_type="SIMILAR").number_of_edges() == 1


def test_no_embed_index_means_absent_not_disguised(snapshot):
    """No SIMILAR edges: zones are empty and every sgroup reads -1."""
    nodes = _nodes("a.md", "b.md")
    assert ge.detect_semantic_partition(nodes, _clique("a.md", "b.md")) == []
    assert all(n["sgroup"] == -1 for n in nodes)
    assert not snapshot.exists()


def test_the_two_partitions_share_no_colour_key():
    """Same id, two partitions, two colours — a swatch names one layer only."""
    assert all(ge._zone_color(i) != ge._community_color(i) for i in range(32))


def test_id_spaces_live_in_separate_files():
    assert ge.semantic_snapshot_path() != ge.cluster_ctx_path()


# --- Inheritance: the ids survive a recompute --------------------------------

def test_cluster_keeps_its_id_when_it_gains_a_member(snapshot):
    """Snapshot with one cluster, recompute with one more member: same id."""
    snapshot.write_bytes(orjson.dumps(
        {"next_id": 7, "clusters": {"6": ["a.md", "b.md"]}}))

    nodes = _nodes("a.md", "b.md", "c.md")
    zones = ge.detect_semantic_partition(
        nodes, _clique("a.md", "b.md", "c.md", type="SIMILAR"))

    assert [z.id for z in zones] == [6]
    assert {n["sgroup"] for n in nodes} == {6}


def test_a_cluster_with_no_predecessor_takes_an_unused_id(snapshot):
    """Fresh clusters never recycle an id, not even one nothing holds now."""
    snapshot.write_bytes(orjson.dumps(
        {"next_id": 9, "clusters": {"3": ["a.md", "b.md"]}}))

    nodes = _nodes("a.md", "b.md", "x.md", "y.md")
    zones = ge.detect_semantic_partition(
        nodes,
        _clique("a.md", "b.md", type="SIMILAR") + _clique("x.md", "y.md", type="SIMILAR"),
    )

    ids = sorted(z.id for z in zones)
    assert ids == [3, 9]          # 3 inherited, 9 fresh — never 0, 1 or 4
    assert orjson.loads(snapshot.read_bytes())["next_id"] == 10


def test_inheritance_follows_the_largest_overlap_not_the_size_order():
    """The biggest new cluster does not get first pick of the ids."""
    prev = {0: {"a.md", "b.md"}, 1: {"x.md"}}
    partition = [{"x.md", "y.md", "z.md"}, {"a.md"}]   # size-descending
    ids, next_id = ge.inherit_ids(partition, prev, 2)
    assert ids == [1, 0] and next_id == 2


def test_a_disappearing_cluster_does_not_free_its_id(snapshot):
    """Two generations: the id of a cluster that vanished is not handed on."""
    nodes = _nodes("a.md", "b.md", "x.md", "y.md")
    edges = (_clique("a.md", "b.md", type="SIMILAR")
             + _clique("x.md", "y.md", type="SIMILAR"))
    first = sorted(z.id for z in ge.detect_semantic_partition(nodes, edges))
    assert first == [0, 1]

    # x/y gone from the vault entirely; a/b keep theirs, a new pair arrives.
    nodes2 = _nodes("a.md", "b.md", "p.md", "q.md")
    edges2 = (_clique("a.md", "b.md", type="SIMILAR")
              + _clique("p.md", "q.md", type="SIMILAR"))
    second = sorted(z.id for z in ge.detect_semantic_partition(nodes2, edges2))
    assert second == [0, 2]


def test_snapshot_is_a_cache_deleting_it_is_legal(snapshot):
    """P2: with no snapshot the ids simply restart — no crash, no stale read."""
    nodes = _nodes("a.md", "b.md")
    edges = _clique("a.md", "b.md", type="SIMILAR")
    ge.detect_semantic_partition(nodes, edges)
    snapshot.unlink()
    assert [z.id for z in ge.detect_semantic_partition(nodes, edges)] == [0]


def test_labels_map_through_the_inherited_id(snapshot, monkeypatch):
    """community_labels is keyed by index; a zone label must not be read off it
    directly once the ids stop matching the ordering."""
    snapshot.write_bytes(orjson.dumps(
        {"next_id": 5, "clusters": {"4": ["a.md", "b.md"]}}))
    monkeypatch.setattr(
        "silica.kernel.recall.cooccurrence.CooccurStore.community_labels",
        lambda self, communities, **kw: {0: "topology · sheaf"},
    )
    nodes = _nodes("a.md", "b.md")
    zones = ge.detect_semantic_partition(nodes, _clique("a.md", "b.md", type="SIMILAR"))
    assert [(z.id, z.label) for z in zones] == [(4, "topology · sheaf")]


def test_unlabelled_zone_says_which_partition_it_came_from(snapshot, monkeypatch):
    monkeypatch.setattr(
        "silica.kernel.recall.cooccurrence.CooccurStore.community_labels",
        lambda self, communities, **kw: {},
    )
    zones = ge.detect_semantic_partition(
        _nodes("a.md", "b.md"), _clique("a.md", "b.md", type="SIMILAR"))
    assert zones[0].label == "Zone 0"   # never "Cluster 0"


def test_note_with_no_similar_edge_is_not_a_zone(monkeypatch):
    """An unembedded note gets sgroup -1, not a zone of one.

    The regression that made the viewer draw 737 labelled hulls on an 886-note
    vault whose embed index held 135 notes (2026-08-24).
    """
    from silica.kernel.recall import graph_export as ge

    # 20 notes, 18 of them embedded: 90% coverage, so the layer is drawn (the
    # coverage gate is a separate rule, tested below) and the two vectorless
    # notes are the only thing left for the singleton filter to catch.
    nodes = [{"id": f"n{i}.md"} for i in range(20)]
    covered = [f"n{i}.md" for i in range(18)]
    edges = [
        {"from": covered[i], "to": covered[(i + 1) % len(covered)], "type": "SIMILAR"}
        for i in range(len(covered))
    ]

    monkeypatch.setattr(ge, "load_semantic_snapshot", lambda: ({}, 0))
    monkeypatch.setattr(ge, "save_semantic_snapshot", lambda *a, **k: None)

    zones = ge.detect_semantic_partition(nodes, edges)

    assert zones and all(z.size > 1 for z in zones)
    assert {n["sgroup"] for n in nodes if n["id"] in {"n18.md", "n19.md"}} == {-1}


def _sim(a, b):
    return {"from": a, "to": b, "type": "SIMILAR"}


def _ring(ids):
    """A cycle over `ids` — every node covered, every zone bigger than one."""
    return [_sim(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]


def test_semantic_coverage_counts_notes_a_similar_edge_reaches():
    from silica.kernel.recall.graph_export import semantic_coverage

    nodes = [{"id": f"n{i}.md"} for i in range(5)] + [{"id": "g.md", "type": "ghost"}]
    edges = [_sim("n0.md", "n1.md"), {"from": "n2.md", "to": "n3.md", "type": "EXTRACTED"}]

    # n2/n3 are joined by a WIKILINK, which says nothing about having a vector.
    assert semantic_coverage(nodes, edges) == (2, 5)


def test_thin_embed_index_withholds_the_zones(monkeypatch):
    """A partial index draws a DIFFERENT map, not a partial one — so none.

    15% coverage agreed with the full partition at ARI 0.363 on an 887-note
    vault (2026-08-24), while looking exactly as confident.
    """
    from silica.kernel.recall import graph_export as ge

    monkeypatch.setattr(ge, "load_semantic_snapshot", lambda: ({}, 0))
    monkeypatch.setattr(ge, "save_semantic_snapshot", lambda *a, **k: None)

    # 20 notes, only the first 6 embedded: two clean zones, 30% coverage.
    nodes = [{"id": f"n{i}.md"} for i in range(20)]
    edges = _ring(["n0.md", "n1.md", "n2.md"]) + _ring(["n3.md", "n4.md", "n5.md"])

    assert ge.detect_semantic_partition(nodes, edges) == []
    assert {n["sgroup"] for n in nodes} == {-1}

    # Same nodes, same rule, once the index reaches the rest of the vault.
    edges += _ring([f"n{i}.md" for i in range(6, 20)])
    zones = ge.detect_semantic_partition(nodes, edges)
    assert zones and all(z.size > 1 for z in zones)
    assert all(n["sgroup"] >= 0 for n in nodes)


def test_gated_layer_says_why_in_the_hud():
    """The row replaces the layer; dropping it silently is the defect it fixes."""
    from silica.ui.web.graph_view import _zone_gate_row

    nodes = [{"id": f"n{i}.md"} for i in range(10)]
    row = _zone_gate_row(nodes, _ring(["n0.md", "n1.md", "n2.md"]))
    assert "3/10 embedded" in row
    assert "/embed" in row
    # No index at all needs no explaining — that layer was never promised.
    assert _zone_gate_row(nodes, []) == ""
