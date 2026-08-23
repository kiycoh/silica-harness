"""The readings the metrics charts stand on, and the three ways they can lie.

The view moved from tables of numbers to charts, and a chart is a stronger
claim than a table: a table that shows twelve rows says twelve rows, a lens
coloured from those twelve rows says something about the whole vault. Three
seams here fail silently rather than loudly, so each one gets the test that
catches it.

  - `inter_cluster` is the coupling grid's whole content. Counted off a
    directed graph it would double every mutual pair, and the grid would
    disagree with the cohesion on its own diagonal.
  - `_signal_areas` colours the treemap. Computed from the twelve rows
    `/metrics` ships it would paint the top-12's areas hot and every other area
    clean, which is a confident statement about the wrong population -- the
    same trap `dangling_hist` exists to avoid for the unresolved tail.
  - `_area_matrix` and `/shape` are drawn by ONE client renderer. Let the two
    payload shapes drift and the second surface to be opened renders wrong.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from silica.kernel.report.graph_report import compute_report  # noqa: E402
from silica.ui.web.server import (  # noqa: E402
    _area_matrix,
    _area_of,
    _signal_areas,
)

APP_JS = Path(__file__).resolve().parents[1] / "silica" / "ui" / "web" / "static" / "app.js"


def _fn_body(src: str, name: str) -> str:
    """The source of one top-level `function name(...) { ... }`.

    Brace-counted rather than regex-matched to the closing brace: these bodies
    contain object literals and template strings, and a non-greedy match to the
    first `^}` stops at the first nested one.

    The parameter list is walked first. Both functions here destructure their
    options argument, so the first `{` after the name opens `{ dense = true }`
    and not the body -- counting from there closes one character later and
    returns the signature, which reads as a function that touches no fields.
    """
    start = src.index(f"function {name}(")
    i = src.index("(", start)
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                i = src.index("{", j)
                break
    else:
        raise AssertionError(f"{name} has no closing paren")
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[start:j + 1]
    raise AssertionError(f"{name} has no closing brace")


def _node(nid, group=0):
    return {"id": nid, "label": nid, "group": group, "type": "note"}


def _edge(src, dst):
    return {"id": f"{src}{dst}", "from": src, "to": dst, "type": "EXTRACTED"}


@pytest.fixture()
def two_triangles():
    """Two 3-cliques joined by one edge, plus a mutual pair inside the first.

    Louvain has one honest answer here, so the areas are stable across runs and
    the expected tally can be written down rather than derived from the code
    under test.
    """
    nodes = [_node(x, 0) for x in "ABC"] + [_node(x, 1) for x in "DEF"]
    edges = [
        _edge("A", "B"), _edge("B", "A"),   # a mutual pair: ONE linked pair
        _edge("B", "C"), _edge("A", "C"),
        _edge("D", "E"), _edge("E", "F"), _edge("D", "F"),
        _edge("C", "D"),                    # the bridge
    ]
    return compute_report(
        analytics=True, _nodes_edges_override=(nodes, edges), _mtimes_override={},
    )


def test_coupling_counts_linked_pairs_not_wikilinks(two_triangles):
    """A mutual link is one coupling, not two.

    The diagonal is cohesion's numerator: counted off the directed graph a
    2-note area that links both ways would score 2.0 on a ratio bounded in
    [0, 1], which is the defect `/shape` documents deduping to avoid.
    """
    r = two_triangles
    assert r.inter_cluster == {"0|0": 3, "1|1": 3, "0|1": 1}, r.inter_cluster

    # And the diagonal really is what cohesion divides: 3 pairs over C(3,2).
    for c in r.clusters:
        possible = c.size * (c.size - 1) / 2
        intra = r.inter_cluster[f"{c.cluster_id}|{c.cluster_id}"]
        assert c.cohesion == pytest.approx(round(intra / possible, 4))


def test_coupling_is_absent_not_empty_without_analytics():
    """{} at structural depth would read as "measured, nothing couples"."""
    nodes = [_node(x, 0) for x in "ABC"]
    edges = [_edge("A", "B"), _edge("B", "C")]
    r = compute_report(
        analytics=False, _nodes_edges_override=(nodes, edges), _mtimes_override={},
    )
    assert r.inter_cluster == {}


def test_area_matrix_matches_the_shape_payload(two_triangles):
    """One client renderer draws this grid and `/shape`'s. Same keys or it breaks."""
    m = _area_matrix(two_triangles)
    assert m is not None
    assert set(m) == {"areas", "matrix"}
    assert set(m["areas"][0]) == {"id", "label", "path", "size", "cohesion", "intra"}
    n = len(m["areas"])
    assert [len(row) for row in m["matrix"]] == [n] * n
    # Symmetric, and the diagonal is the area's own intra count.
    for i in range(n):
        assert m["matrix"][i][i] == m["areas"][i]["intra"]
        for j in range(n):
            assert m["matrix"][i][j] == m["matrix"][j][i]


def test_area_matrix_is_none_below_two_areas():
    """A 1x1 grid is a cell saying nothing; None is what lets the card say so."""
    nodes = [_node(x, 0) for x in "ABC"]
    edges = [_edge("A", "B"), _edge("B", "C"), _edge("A", "C")]
    r = compute_report(
        analytics=True, _nodes_edges_override=(nodes, edges), _mtimes_override={},
    )
    assert _area_matrix(r) is None


def test_area_index_answers_in_both_keyspaces(two_triangles):
    """The V7 rows arrive without `.md` and the graph ids carry it.

    A lookup that missed would not raise. It would report the area as clean.
    """
    idx = _area_of(two_triangles)
    member = two_triangles.clusters[0].members[0]
    bare = member[:-3] if member.endswith(".md") else member
    assert idx[bare] == idx[bare + ".md"]


def test_signal_areas_counts_the_whole_list_not_the_shipped_slice():
    """The lens is coloured from this, and `/metrics` ships twelve rows.

    Twenty lean notes in one area must tally twenty. Recomputed from the slice
    this reads twelve, and every area past the cut reads clean.
    """
    from silica.ui.web.server import _METRICS_ROWS

    members = [f"a/n{i}.md" for i in range(20)] + ["b/h1.md", "b/h2.md"]
    report = SimpleNamespace(
        clusters=[
            SimpleNamespace(cluster_id=0, size=20, members=members[:20]),
            SimpleNamespace(cluster_id=1, size=2, members=members[20:]),
            SimpleNamespace(cluster_id=9, size=1, members=["c/lonely.md"]),
        ],
        lean_notes=members[:20],
        orphans=["c/lonely.md"],
        attention_candidates=[],
        integration_deficits=[],
        contested=[],
        source_drift=[],
        sprawling=[SimpleNamespace(path="b/h1")],  # store keyspace, no .md
    )
    tallies = _signal_areas(report)
    assert tallies["lean"] == {"0": 20}
    assert tallies["lean"]["0"] > _METRICS_ROWS, "the slice would read 12"
    # The V7 row lands in its area despite arriving in the other keyspace.
    assert tallies["sprawling"] == {"1": 1}
    # A note in no multi-note area has no column, rather than a made-up one.
    assert "orphans" not in tallies


def test_signals_with_no_place_are_absent_by_construction():
    """A dangling target has no note; a gap is a fact about a PAIR of areas.

    Either one tallied into a single area would invent a location, so the lens
    must have no entry for them at all -- an empty dict would read as "that
    signal is clean everywhere".
    """
    report = SimpleNamespace(
        clusters=[SimpleNamespace(cluster_id=0, size=2, members=["a.md", "b.md"])],
        lean_notes=["a.md"], orphans=[], attention_candidates=[],
        integration_deficits=[], contested=[], source_drift=[], sprawling=[],
    )
    tallies = _signal_areas(report)
    assert "dangling" not in tallies
    assert "gaps" not in tallies


# --- the two renderers fed by two payloads ----------------------------------
# `couplingMatrix` and `areaTreemap` each draw data that reaches them from more
# than one endpoint, and the endpoints do not agree on their spelling: `/shape`
# names an area's hub `label`, `/metrics` names it `hub`. A renderer reading
# `label` off a `/metrics` cluster does not fail. It draws a tile with an empty
# name, which is how this shipped once already.

# What squarify() adds to a row on its way through the layout. Not payload
# fields, so a renderer reading them proves nothing about either endpoint.
_LAYOUT_FIELDS = {"x", "y", "w", "h", "value"}


def _fields_read(body: str, *vars_: str) -> set[str]:
    out: set[str] = set()
    for v in vars_:
        out |= set(re.findall(rf"\b{v}\.([a-z_][a-z0-9_]*)\b", body))
    return out - _LAYOUT_FIELDS


def test_the_coupling_matrix_gets_every_field_it_reads(two_triangles):
    """One renderer, two endpoints. `/metrics` mirrors `/shape`'s area shape so
    that stays true; this is the test that notices when it stops being."""
    body = _fn_body(APP_JS.read_text(), "couplingMatrix")
    reads = _fields_read(body, "a", "b")
    assert reads, "couplingMatrix stopped reading its rows by name"
    supplied = set(_area_matrix(two_triangles)["areas"][0])
    missing = sorted(reads - supplied)
    assert not missing, f"couplingMatrix reads fields /metrics does not ship: {missing}"


def test_the_area_treemap_gets_every_field_it_reads():
    """The treemap is fed from `d.clusters`, which spells the hub `hub`.

    renderMetrics maps that to `label` before handing it over. Without the
    mapping the tiles render nameless, so the mapping is the contract and this
    is what holds it: the reader is the JS, and the payload is `/metrics`.
    """
    src = APP_JS.read_text()
    body = _fn_body(src, "areaTreemap")
    reads = _fields_read(body, "a", "t")
    assert reads, "areaTreemap stopped reading its rows by name"

    # What `/metrics` puts on a cluster, read off the endpoint's own literal so
    # a renamed key fails here rather than rendering blank in a browser.
    server = (Path(__file__).resolve().parents[1] / "silica" / "ui" / "web" / "server.py").read_text()
    cluster = re.search(r'"clusters": \[\s*\{([^}]*)\}', server, re.S)
    assert cluster, "the /metrics clusters literal moved"
    shipped = set(re.findall(r'"(\w+)":', cluster.group(1)))

    # Plus whatever renderMetrics adds on the way in.
    field = re.search(r"const areaField = .*?;\n", src, re.S)
    assert field, "the areaField mapping moved"
    added = set(re.findall(r"(\w+):", field.group(0).split(".map(")[-1]))

    missing = sorted(reads - (shipped | added))
    assert not missing, f"areaTreemap reads fields nothing supplies: {missing}"


def test_the_lens_only_offers_signals_the_server_tallies():
    """A worklist row whose signal has no per-area tally falls back to cohesion.

    Silently, and correctly -- but only for the two the fallback NAMES. A third
    unplaceable signal added server-side would fall back with no explanation,
    and a coloured field that quietly stops meaning anything is worse than one
    that says why.
    """
    src = APP_JS.read_text()
    block = re.search(r"^  const signals = \[(.*?)^  \]\.sort", src, re.S | re.M)
    assert block, "the worklist's signal table moved"
    rows = set(re.findall(r'^\s*\["(\w+)",', block.group(1), re.M))

    unplaceable = re.search(r"const unplaceable = \{(.*?)\n    \};", src, re.S)
    assert unplaceable, "the lens fallback explanations moved"
    named = set(re.findall(r"^\s*(\w+):", unplaceable.group(1), re.M))

    # Everything the server can place, from the one place that decides it.
    server = (Path(__file__).resolve().parents[1] / "silica" / "ui" / "web" / "server.py").read_text()
    lists = re.search(r"    lists = \{(.*?)\n    \}", server, re.S)
    assert lists, "_signal_areas' list table moved"
    tallied = set(re.findall(r'"(\w+)":', lists.group(1)))

    unexplained = sorted(rows - tallied - named)
    assert not unexplained, (
        f"worklist signals with neither a per-area tally nor a stated reason: {unexplained}"
    )


def test_shape_and_metrics_ship_the_same_area(two_triangles):
    """The other half of the one-renderer contract.

    `couplingMatrix` is drawn from `/shape` on the explore surface and from
    `/metrics` in the gaps pane. Testing only one of them leaves the other free
    to drift, and the surface that breaks is whichever is opened second.
    """
    server = (Path(__file__).resolve().parents[1] / "silica" / "ui" / "web" / "server.py").read_text()
    lit = re.search(r"    areas = \[\n        \{(.*?)\}\n        for g in ids", server, re.S)
    assert lit, "the /shape areas literal moved"
    shape_keys = set(re.findall(r'"(\w+)":', lit.group(1)))
    metrics_keys = set(_area_matrix(two_triangles)["areas"][0])
    assert shape_keys == metrics_keys, (
        f"/shape ships {sorted(shape_keys)}, /metrics ships {sorted(metrics_keys)}"
    )
