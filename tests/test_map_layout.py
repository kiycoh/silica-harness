"""Layout invariants for kernel/mindmap — deterministic radial wedge placement."""
from __future__ import annotations

import math

import networkx as nx

from silica.kernel.recall.mindmap import (
    BOX_H,
    BOX_W,
    MapMaterials,
    build_mapview,
)


def _materials(latent=()):
    """root — a,b (hop1); a — c, b — d (hop2). Two communities.  Plus latent."""
    g = nx.Graph()
    g.add_edges_from([
        ("root.md", "a.md"), ("root.md", "b.md"),
        ("a.md", "c.md"), ("b.md", "d.md"),
    ])
    return MapMaterials(
        graph=g,
        titles={p: p[:-3].upper() for p in g.nodes},
        community_of={"root.md": 0, "a.md": 0, "c.md": 0, "b.md": 1, "d.md": 1},
        latent=list(latent),
    )


def _boxes_overlap(a, b) -> bool:
    # Equal axis-aligned boxes centred at a, b overlap iff both axes are within a box.
    return abs(a.x - b.x) < BOX_W and abs(a.y - b.y) < BOX_H


def test_root_at_origin():
    mv = build_mapview("root.md", _materials())
    root = next(n for n in mv.nodes if n.id == "root.md")
    assert (root.x, root.y) == (0.0, 0.0)
    assert root.hop == 0


def test_no_box_overlap():
    mv = build_mapview("root.md", _materials([("e.md", "E", 0.9), ("f.md", "F", 0.8)]))
    ns = mv.nodes
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            assert not _boxes_overlap(ns[i], ns[j]), f"{ns[i].id} overlaps {ns[j].id}"


def test_communities_occupy_contiguous_wedges():
    # Each community must own a contiguous angular arc — no interleaving.
    mv = build_mapview("root.md", _materials())
    non_root = [n for n in mv.nodes if n.hop > 0]
    ordered = sorted(non_root, key=lambda n: math.atan2(n.y, n.x) % (2 * math.pi))
    runs = [n.community for n in ordered]
    # Number of contiguous community-runs equals the number of distinct communities.
    changes = sum(1 for k in range(1, len(runs)) if runs[k] != runs[k - 1])
    assert changes == len(set(runs)) - 1


def test_deterministic():
    m = _materials([("e.md", "E", 0.9)])
    a = build_mapview("root.md", m)
    b = build_mapview("root.md", _materials([("e.md", "E", 0.9)]))
    assert [(n.id, n.x, n.y) for n in a.nodes] == [(n.id, n.x, n.y) for n in b.nodes]


def test_root_argument_normalized_to_md():
    mv = build_mapview("root", _materials())  # no .md suffix
    assert mv.root == "root.md"
    assert any(n.id == "root.md" and n.hop == 0 for n in mv.nodes)


def test_radius_tracks_association_cost():
    # Radius IS data: a direct wikilink sits closer than a 2-hop one, a strong
    # latent tie closer than a weak one, and the strongest latent tie costs
    # exactly one wikilink hop (strengths are normalised to the map's best).
    mv = build_mapview("root.md", _materials([("e.md", "E", 0.9), ("f.md", "F", 0.45)]))
    r = {n.id: math.hypot(n.x, n.y) for n in mv.nodes}
    assert r["a.md"] < r["c.md"]
    assert r["e.md"] < r["f.md"]
    assert abs(r["a.md"] - r["e.md"]) < 1e-6
    costs = {n.id: n.cost for n in mv.nodes}
    assert costs["root.md"] == 0.0
    assert costs["a.md"] == 1.0 and costs["c.md"] == 2.0


def test_hub_root_radius_is_not_degenerate():
    # Regression: when every selected node is a direct wikilink, cost used to
    # tie at 1.0 for all of them and the radius spread was pure collision
    # slide — i.e. alphabetical. Similarity must break that tie.
    g = nx.Graph()
    for x in "abcd":
        g.add_edge("root.md", f"{x}.md")
    sims = {"a.md": 0.9, "b.md": 0.6, "c.md": 0.4, "d.md": 0.2}
    m = MapMaterials(
        graph=g,
        titles={p: p for p in g.nodes},
        community_of={p: 0 for p in g.nodes},
        sim=lambda p, q: sims.get(q if p == "root.md" else p, 0.0),
    )
    mv = build_mapview("root.md", m)
    cost = {n.id: n.cost for n in mv.nodes}
    assert cost["a.md"] < cost["b.md"] < cost["c.md"] < cost["d.md"]
    assert cost["a.md"] == 1.0            # the tightest tie anchors the map
    assert cost["d.md"] < 2.0             # one wikilink never costs two


def test_sibling_order_follows_similarity():
    # With a similarity signal, angular neighbours must be semantic neighbours:
    # a~z high beats the alphabetical a,b,z order.
    g = nx.Graph()
    for x in ("a.md", "b.md", "z.md"):
        g.add_edge("root.md", x)
    sims = {
        frozenset({"a.md", "z.md"}): 0.95,
        frozenset({"a.md", "b.md"}): 0.10,
        frozenset({"b.md", "z.md"}): 0.20,
    }
    m = MapMaterials(
        graph=g,
        titles={p: p for p in g.nodes},
        community_of={p: 0 for p in g.nodes},
        sim=lambda p, q: sims.get(frozenset({p, q}), 0.0),
    )
    mv = build_mapview("root.md", m)
    ordered = sorted(
        (n for n in mv.nodes if n.hop > 0),
        key=lambda n: math.atan2(n.y, n.x) % (2 * math.pi),
    )
    assert [n.id for n in ordered] == ["a.md", "z.md", "b.md"]


def test_subtitle_shows_link_context():
    m = _materials()
    m.link_context = lambda p, c: f"{p}→{c}"
    mv = build_mapview("root.md", m)
    subs = {n.id: n.subtitle for n in mv.nodes}
    assert subs["a.md"] == "root.md→a.md"
    assert subs["c.md"] == "a.md→c.md"


def test_latent_subtitle_shows_evidence():
    m = _materials([("e.md", "E", 0.9)])
    m.latent_evidence = {"e.md": "≈ embed:0.90"}
    m.link_context = lambda p, c: "should not be used for latent"
    mv = build_mapview("root.md", m)
    e = next(n for n in mv.nodes if n.id == "e.md")
    assert e.subtitle == "≈ embed:0.90"


def test_degree_recorded():
    mv = build_mapview("root.md", _materials([("e.md", "E", 0.9)]))
    deg = {n.id: n.degree for n in mv.nodes}
    assert deg["root.md"] == 2 and deg["a.md"] == 2 and deg["c.md"] == 1
    assert deg["e.md"] == 0  # latent-only, not in the wikilink graph


def test_clean_snippet_unwraps_wikilinks_and_truncates():
    from silica.kernel.recall.mindmap import _clean_snippet

    assert _clean_snippet("- See [[path/to/note|the alias]] for context") == (
        "See the alias for context"
    )
    assert _clean_snippet("## About [[grafo]]") == "About grafo"
    long = "x" * 200
    assert len(_clean_snippet(long)) == 96 and _clean_snippet(long).endswith("…")


def test_anchor_snippet_keeps_the_explanation_not_the_restated_title():
    from silica.kernel.recall.mindmap import _anchor_snippet

    line = "- **[[Devianza]] (scomposizione)**: la devianza totale si scompone in due parti"
    assert _anchor_snippet(line, {"devianza"}) == (
        "la devianza totale si scompone in due parti"
    )
    # No title restatement before the colon ⇒ the whole line survives.
    line2 = "- vedi anche [[Devianza]] per il calcolo"
    assert _anchor_snippet(line2, {"altro"}) == "vedi anche Devianza per il calcolo"
