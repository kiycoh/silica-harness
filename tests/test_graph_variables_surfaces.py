# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""The seven variables where a reader meets them.

The kernel side is pinned by tests/test_graph_variables.py. This file pins the
SURFACES: the per-note facade behind the Work panel, the reading-order pane,
the three graph layers, the metrics rows and the calendar strip. One rule runs
through all of them and is asserted rather than assumed: a variable that failed
its gate (docs/adr/0027) is allowed to state what it measured and is never
allowed to accuse a note.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import networkx as nx
import pytest

from silica.kernel.report import structure

WEB = Path(__file__).resolve().parent.parent / "silica" / "ui" / "web"
APP_JS = WEB / "static" / "app.js"
APP_CSS = WEB / "static" / "app.css"
WORK_JS = WEB / "static" / "work.js"
FRAME_JS = WEB / "static" / "graph-frame.js"
FRAME_HTML = WEB / "static" / "graph-frame.html"


# ---------------------------------------------------------------------------
# the facade: cut components, and the ladder's rungs
# ---------------------------------------------------------------------------

def test_cut_component_is_measured_inside_its_own_component():
    """A vault is already in pieces before anything is removed. Counting every
    piece as "stranded by this note" charged the first cut vertex on a 709-note
    vault with 159 notes it had nothing to do with; the honest number is what
    falls off ITS component, against the largest piece the removal leaves."""
    G = nx.Graph()
    G.add_edges_from([("a", "cut"), ("cut", "b"), ("b", "c")])   # cut splits 1 | 2
    G.add_edges_from([("x", "y"), ("y", "z")])                   # a separate island
    sizes = structure.cut_component_sizes(G, {"cut", "b"})
    assert sizes["cut"] == 1     # 'a' alone: the {b,c} side is the larger piece
    assert sizes["b"] == 1       # 'c' alone
    assert "x" not in sizes      # the island was already an island


def test_cut_component_size_is_zero_for_a_node_that_is_not_a_cut_vertex():
    G = nx.Graph([("a", "b"), ("b", "c"), ("c", "a")])
    assert structure.cut_component_sizes(G, set()) == {}


def _map(prereq, degree=None) -> structure.StructureMap:
    """A StructureMap with only the fields the ladder reads."""
    deg = degree or {n + ".md": 1 for n in {k for k in prereq} | {
        p for ps in prereq.values() for p in ps}}
    m = structure.StructureMap(degree=deg, prereq=prereq)
    object.__setattr__(m, "_sizes", {})
    return m


def test_ladder_depth_is_the_longest_path_not_the_hop_count(monkeypatch):
    """A -> B -> C with a shortcut A -> C. By hops, C sits one step from A and
    lands on B's rung; by longest path it is two, which is the reading: you can
    skip B, but B still comes first."""
    monkeypatch.setattr(structure, "structure_map",
                        lambda: _map({"B": ["A"], "C": ["B", "A"]}))
    lad = structure.ladder("A.md")
    depth = {n["path"]: n["depth"] for n in lad["nodes"]}
    assert depth == {"A.md": 0, "B.md": 1, "C.md": 2}
    assert lad["root"] == "A.md"
    assert lad["cycles"] == 0


def test_ladder_places_prerequisites_below_zero_and_dependents_above(monkeypatch):
    monkeypatch.setattr(structure, "structure_map",
                        lambda: _map({"ROOT": ["BASE"], "TIP": ["ROOT"]}))
    lad = structure.ladder("ROOT.md")
    depth = {n["path"]: n["depth"] for n in lad["nodes"]}
    assert depth["BASE.md"] == -1 and depth["ROOT.md"] == 0 and depth["TIP.md"] == 1


def test_ladder_reports_a_cycle_and_does_not_lay_its_edges_out(monkeypatch):
    """RefD can say A needs B, B needs C and C needs A. That is a real
    disagreement in the vault, not corrupt data, so it is counted and the
    edges that close it are left undrawn rather than making the rungs
    unassignable."""
    monkeypatch.setattr(structure, "structure_map",
                        lambda: _map({"A": ["C"], "B": ["A"], "C": ["B"]}))
    lad = structure.ladder("A.md")
    assert lad["cycles"] == 3
    assert lad["edges"] == []           # nothing settled, so nothing is drawn
    assert lad["root"] == "A.md"


def test_ladder_abstains_for_a_note_with_no_direction(monkeypatch):
    m = _map({"B": ["A"]})
    object.__setattr__(m, "degree", {**m.degree, "LONE.md": 3})
    monkeypatch.setattr(structure, "structure_map", lambda: m)
    lad = structure.ladder("LONE.md")
    assert lad["nodes"] == [] and lad["root"] == "LONE.md"


def test_ladder_is_empty_without_a_prerequisite_index(monkeypatch):
    monkeypatch.setattr(structure, "structure_map", lambda: structure.StructureMap())
    assert structure.ladder("anything.md") == {
        "root": "", "nodes": [], "edges": [], "cycles": 0, "truncated": False,
    }


def test_note_structure_abstains_on_dissonance_without_a_semantic_snapshot(monkeypatch):
    """None, not 0.0: no snapshot means nobody drew the zones, and a printed
    zero would read as "measured, the links all stay home"."""
    m = structure.StructureMap(degree={"N.md": 2}, core={"N.md": 1}, zoned=False)
    object.__setattr__(m, "_sizes", {})
    monkeypatch.setattr(structure, "structure_map", lambda: m)
    assert structure.note_structure("N.md")["dissonance"] is None


def test_note_structure_is_empty_for_a_note_the_graph_has_never_seen(monkeypatch):
    monkeypatch.setattr(structure, "structure_map", lambda: structure.StructureMap())
    assert structure.note_structure("ghost.md") == {}


# ---------------------------------------------------------------------------
# the graph frame: three layers and one node channel
# ---------------------------------------------------------------------------

def _n(nid, sgroup=-1, ntype="note"):
    return {"id": nid, "label": nid, "type": ntype, "sgroup": sgroup}


def test_discord_marks_existing_wikilinks_and_adds_none():
    """An overlay would have doubled the spring force on every link it marked,
    and ~40% of a vault's links cross a zone boundary (176 of 1340 measured on
    the dev vault, and that is the sparse case)."""
    from silica.ui.web.graph_view import _mark_discord

    nodes = [_n("A", 0), _n("B", 1), _n("C", 0), _n("D", -1)]
    edges = [
        {"id": "e0", "from": "A", "to": "B", "type": "EXTRACTED"},   # 0 vs 1
        {"id": "e1", "from": "A", "to": "C", "type": "EXTRACTED"},   # same zone
        {"id": "e2", "from": "A", "to": "D", "type": "EXTRACTED"},   # D has no zone
        {"id": "e3", "from": "A", "to": "B", "type": "SIMILAR"},     # not a wikilink
    ]
    before = len(edges)
    _mark_discord(edges, nodes)
    assert len(edges) == before
    assert [e.get("discord") for e in edges] == [True, None, None, None]


def test_discord_says_nothing_when_the_vault_has_no_zones():
    from silica.ui.web.graph_view import _mark_discord

    nodes = [_n("A"), _n("B")]
    edges = [{"id": "e0", "from": "A", "to": "B", "type": "EXTRACTED"}]
    _mark_discord(edges, nodes)
    assert "discord" not in edges[0]


def test_load_bearing_is_stamped_on_the_nodes_with_what_a_cut_costs():
    from silica.ui.web.graph_view import _stamp_load_bearing

    nodes = [_n("a"), _n("cut"), _n("b"), _n("c"), _n("G", ntype="ghost")]
    G = nx.Graph([("a", "cut"), ("cut", "b"), ("b", "c")])
    _stamp_load_bearing(nodes, G, {n: 0.0 for n in G})
    by_id = {n["id"]: n for n in nodes}
    assert by_id["cut"]["cut"] is True and by_id["cut"]["strands"] == 1
    assert "cut" not in by_id["a"]           # not a cut vertex: no flag at all
    assert by_id["a"]["coreness"] == 1
    assert "coreness" not in by_id["G"]      # a ghost is not a note


def test_proposed_layer_keeps_only_pairs_a_shared_concept_corroborates(monkeypatch):
    """Uncorroborated, V1 lost every golden pair it was tested on (ADR-0027).
    The layer is allowed to exist because the corroboration is the filter, so
    a pair the index cannot vouch for is not drawn."""
    import silica.ui.web.graph_view as gv

    monkeypatch.setattr(
        "silica.kernel.recall.graph_export.shared_concepts",
        lambda pairs, k=3: {("A", "D"): ["gradient"]},
    )
    out = gv._corroborate(
        [("A", "D", 2.0), ("B", "E", 9.0)], [_n("A"), _n("B"), _n("D"), _n("E")], 10,
        kind="PROPOSED", color="#000", paper="#fff",
    )
    assert [(e["from"], e["to"]) for e in out] == [("A", "D")]
    assert out[0]["type"] == "PROPOSED" and out[0]["shared"] == ["gradient"]


def test_proposed_layer_is_empty_without_an_index(monkeypatch):
    import silica.ui.web.graph_view as gv

    monkeypatch.setattr(
        "silica.kernel.recall.graph_export.shared_concepts", lambda pairs, k=3: {})
    assert gv._corroborate([("A", "D", 2.0)], [_n("A"), _n("D")], 10,
                           kind="COUPLED", color="#000", paper="#fff") == []


def test_the_frame_hides_a_layer_that_has_nothing_in_it():
    """A checkbox for an empty layer promises something behind it. Each of the
    four rows is rendered only where its own count is non-zero."""
    from silica.ui.web.graph_view import _frame_context

    ctx = _frame_context([_n("A"), _n("B")],
                         [{"id": "e0", "from": "A", "to": "B", "type": "EXTRACTED"}],
                         [], "t", "", "", [])
    assert ctx["proposed_row"] == "" and ctx["coupled_row"] == ""
    assert ctx["discord_row"] == "" and ctx["cut_row"] == ""


def test_the_frame_renders_each_layer_when_it_has_something(tmp_path):
    from silica.ui.web.graph_view import _frame_context

    nodes = [_n("A", 0), _n("B", 1)]
    nodes[0]["cut"] = True
    nodes[0]["strands"] = 4
    edges = [
        {"id": "e0", "from": "A", "to": "B", "type": "EXTRACTED", "discord": True},
        {"id": "p0", "from": "A", "to": "B", "type": "PROPOSED"},
        {"id": "c0", "from": "A", "to": "B", "type": "COUPLED"},
    ]
    ctx = _frame_context(nodes, edges, [], "t", "", "", [])
    assert 'id="cb-proposed"' in ctx["proposed_row"]
    assert 'id="cb-coupled"' in ctx["coupled_row"]
    assert 'id="cb-discord"' in ctx["discord_row"]
    # The load-bearing row is a legend entry, not a control: the ring it names
    # is always drawn, so it carries the swatch and the count and no input.
    assert "load-bearing" in ctx["cut_row"] and ">1<" in ctx["cut_row"]
    assert "<input" not in ctx["cut_row"]


def test_the_three_inferred_layers_are_off_until_asked_for():
    """The frame opens on what you WROTE. Every layer that is a claim about
    what you did not write starts unticked."""
    from silica.ui.web.graph_view import _frame_context

    js = FRAME_JS.read_text()
    for flag in ("showProposed", "showCoupled", "showDiscord"):
        assert re.search(rf"let {flag} = false;", js), flag
    nodes = [_n("A", 0), _n("B", 1)]
    nodes[0]["cut"] = True
    ctx = _frame_context(
        nodes,
        [{"id": "e0", "from": "A", "to": "B", "type": "EXTRACTED", "discord": True},
         {"id": "p0", "from": "A", "to": "B", "type": "PROPOSED"},
         {"id": "c0", "from": "A", "to": "B", "type": "COUPLED"}],
        [], "t", "", "", [],
    )
    for slot in ("proposed_row", "coupled_row", "discord_row"):
        # The <input> tag itself, not the row: one of these tooltips contains
        # the word "checked" in its prose ("were checked by hand").
        tag = ctx[slot].split("<input", 1)[1].split(">", 1)[0]
        assert "checked" not in tag, slot


def test_discord_is_part_of_the_link_paint_cache_key():
    """linkPaint memoises on the edge's own data. Toggling discord changes the
    colour of an edge whose data did not change, so a key that ignored it would
    keep painting the last answer."""
    js = FRAME_JS.read_text()
    block = js.split("function linkPaint(l) {")[1].split("\n}")[0]
    assert "showDiscord && l.discord" in block
    assert 'lit ? "x" : ""' in block


def test_the_load_bearing_ring_sits_outside_the_state_ring():
    """A cut vertex is usually a hub too. One ring would have had the two
    readings overwrite each other, which is why this is a second channel and
    not a fourth nodeState."""
    js = FRAME_JS.read_text()
    draw = js.split("function drawNode(n, ctx, scale) {")[1].split("\n}")[0]
    state_r = draw.index("r + 2 / scale")
    cut_r = draw.index("r + 4.5 / scale")
    assert state_r < cut_r


def test_the_load_bearing_ring_has_no_switch_at_all():
    """It shipped off by default, on the argument that 128 amber rings on a
    709-note vault is an alarm about a normal graph. A reading nobody switches
    on is a reading nobody has, and unlike the three inferred layers this one
    is a fact about the graph in front of you, not a claim about what you did
    not write. So: no flag, no checkbox, no handler, in either renderer."""
    js = FRAME_JS.read_text()
    assert "showCut" not in js and "updateNodeFilter" not in js
    assert 'id="cb-cut"' not in FRAME_HTML.read_text()
    draw = js.split("function drawNode(n, ctx, scale) {")[1].split("\n}")[0]
    assert "if (n.cut) {" in draw
    tip = js.split(".nodeLabel(n => {")[1].split("});")[0]
    assert "const cut = n.cut" in tip, "3D has no ring: the tooltip is the reading"


def test_the_zone_crossing_layer_has_a_colour_of_its_own():
    """It reused the similar layer's azure - the same hex - on the argument
    that this row IS the semantic layer talking. The similar layer is on by
    default, so the recolour landed in the hue thousands of k-NN edges were
    already wearing and ticking the box changed nothing anyone could see. Every
    edge colour the frame can draw at once has to differ from every other."""
    from silica.kernel.recall import graph_export as ge
    from silica.ui.web import graph_view as gv

    # By NAME first: graph_view re-imports three of graph_export's constants, and
    # a constant seen twice under its own name is one colour, not two.
    by_name: dict[str, str] = {
        name: getattr(mod, name).lower()
        for mod in (ge, gv)
        for name in dir(mod)
        if name.startswith("_EDGE_COLOR_") and not name.endswith("_PAPER")
    }
    seen: dict[str, str] = {}
    for name, hexv in sorted(by_name.items()):
        assert hexv not in seen, f"{name} wears {seen[hexv]}'s colour ({hexv})"
        seen[hexv] = name
    assert gv._EDGE_COLOR_DISCORD.lower() in seen


def test_the_zone_crossing_row_names_the_ends_and_not_the_route():
    """The first name, "crossing a zone", read as geometry: on a canvas most
    links cross some zone on their way, so the row seemed to name where a line
    PASSES rather than where its two ends sit."""
    from silica.ui.web.graph_view import _frame_context

    ctx = _frame_context(
        [_n("A", 0), _n("B", 1)],
        [{"id": "e0", "from": "A", "to": "B", "type": "EXTRACTED", "discord": True}],
        [], "t", "", "", [])
    assert "between zones" in ctx["discord_row"]
    assert "crossing a zone" not in ctx["discord_row"]


# ---------------------------------------------------------------------------
# the Work panel: what each number is allowed to claim
# ---------------------------------------------------------------------------

def _structure_rows(tmp_path, st) -> list[dict]:
    """Run structureRows() from work.js under node, on this structure block."""
    src = WORK_JS.read_text()
    m = re.search(r"// --- projection:begin.*?// --- projection:end", src, re.S)
    assert m, "projection markers not found in work.js"
    script = tmp_path / "rows.js"
    script.write_text(
        m.group(0)
        + "\nconsole.log(JSON.stringify(structureRows(JSON.parse(process.argv[2]))));\n"
    )
    out = subprocess.run(["node", str(script), json.dumps(st)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


NODE = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


@NODE
def test_a_cut_vertex_is_the_one_row_allowed_to_warn(tmp_path):
    rows = _structure_rows(tmp_path, {
        "articulation": True, "strands": 30, "coreness": 3, "surprise": 0,
        "dissonance": None,
    })
    assert rows[0]["warn"] is True
    assert "strands 30 notes" in rows[0]["text"]
    assert all(not r.get("warn") for r in rows[1:])


@NODE
def test_dissonance_states_what_it_measured_and_never_warns(tmp_path):
    """Its judge gate failed: the notes it called misfiled were filed right 13
    times out of 14 and 16 out of 19. So the row says the share of links that
    leave the zone, carries no warning swatch, and offers no action."""
    rows = _structure_rows(tmp_path, {
        "articulation": False, "strands": 0, "coreness": 0, "surprise": 0,
        "dissonance": 71,
    })
    assert len(rows) == 1
    assert rows[0]["text"] == "71% of its links leave its zone"
    assert not rows[0].get("warn")
    assert "misfiled" not in json.dumps(rows).lower()
    assert "wrong" not in json.dumps(rows).lower()


@NODE
def test_surprise_reads_as_a_comparison_in_both_directions(tmp_path):
    high = _structure_rows(tmp_path, {"articulation": False, "strands": 0,
                                      "coreness": 0, "surprise": 41,
                                      "dissonance": None})
    low = _structure_rows(tmp_path, {"articulation": False, "strands": 0,
                                     "coreness": 0, "surprise": -41,
                                     "dissonance": None})
    assert "more crossings than links" in high[0]["text"]
    assert "more links than crossings" in low[0]["text"]


@NODE
def test_an_ordinary_note_gets_no_structure_section_at_all(tmp_path):
    """Six numbers about a note in the middle of its own area is six lines
    saying "normal". The block is absent, not zeroed."""
    assert _structure_rows(tmp_path, {
        "articulation": False, "strands": 0, "coreness": 1, "surprise": 4,
        "dissonance": 12,
    }) == []
    assert _structure_rows(tmp_path, None) == []


def test_the_work_panel_reads_the_structure_block_through_the_projection():
    js = WORK_JS.read_text()
    proj = re.search(r"// --- projection:begin.*?// --- projection:end", js, re.S).group(0)
    assert "structure: c.structure || null," in proj
    assert "function structureRows(st)" in proj
    # Read first / Unlocks are rows that point at notes, so they use the same
    # section builder every other neighbour list does.
    assert 'nodeSection("Read first", st.prerequisites || [])' in js.replace(
        ", (r) => ({ why: folderOf(r.path) }))", ")")
    assert 'nodeSection("Unlocks", st.unlocks || [])' in js.replace(
        ", (r) => ({ why: folderOf(r.path) }))", ")")


def test_the_structure_row_has_a_warning_state_and_a_neutral_one():
    css = APP_CSS.read_text()
    assert ".wk-str {" in css and ".wk-str.warn .wk-str-sw {" in css


# ---------------------------------------------------------------------------
# the path surface, the metrics rows and the calendar strip
# ---------------------------------------------------------------------------

def test_path_is_a_pane_surface_that_keeps_the_root_search():
    """It renders where the shape views render and is not one of them: they are
    three readings of one /shape load, this one is rooted on a note and needs
    the search they hide."""
    app = APP_JS.read_text()
    body = app.split("function setGraphMode(m) {")[1].split("\n}")[0]
    assert 'const isPath = m === "path";' in body
    assert '$("#shape-pane").hidden = !(isShape || isPath);' in body
    assert '$("#node-search-wrap").hidden = isShape;' in body
    assert "drawPath();" in body


def test_the_shape_click_defers_to_the_re_root_button():
    """Both handlers live on #shape-pane, and registration order on one node is
    not a contract. The generic one declines rather than the specific one
    shouting."""
    app = APP_JS.read_text()
    assert 'if (e.target.closest("[data-root]")) return;' in app
    assert "rootPath(b.dataset.root)" in app


def test_the_ladder_wires_are_measured_from_the_chips_not_laid_out():
    app = APP_JS.read_text()
    assert "function drawPathWires()" in app
    assert "getBoundingClientRect()" in app.split("function drawPathWires()")[1][:900]
    # and they redraw on resize, because the chips are text of unknown width
    assert "ResizeObserver" in app.split("function drawPathWires()")[1][:2000]


def test_sprawling_is_a_worklist_row_and_bursting_is_not():
    """A worklist is what needs attention. A burst needs none: it is what the
    fortnight turned out to be about, so it is a card in Activity."""
    app = APP_JS.read_text()
    assert '["sprawling", full ? T.sprawling : null, "sprawling notes",' in app
    assert '"bursting"' not in app.split("].sort((a, b) => (b[1] || 0) - (a[1] || 0));")[0]
    assert 'case "sprawling":' in app
    assert 'mCard("Bursting concepts"' in app


def test_the_calendar_carries_the_burst_strip_above_its_days():
    app = APP_JS.read_text()
    assert "function calBurstStrip()" in app
    body = app.split("function renderCalAgenda() {")[1].split("\n}")[0]
    assert body.index("calBurstStrip()") < body.index('head.className = "cal-ag-head"')


# ---------------------------------------------------------------------------
# the endpoints
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_vault, monkeypatch):
    from silica.ui.web import server

    server._reset_session()
    return TestClient(server.app)


def test_context_carries_the_structure_block(client, tmp_vault, monkeypatch):
    tmp_vault.note("A.md", "alpha [[B]]")
    tmp_vault.note("B.md", "beta")
    monkeypatch.setattr(
        "silica.kernel.report.structure.note_structure",
        lambda nid: {"in_graph": True, "degree": 3, "betweenness": 0.4, "coreness": 2,
                     "articulation": True, "strands": 7, "surprise": 0.4123,
                     "dissonance": 0.5, "prerequisites": ["B.md"], "unlocks": []},
    )
    st = client.get("/context", params={"path": "A.md"}).json()["structure"]
    assert st["coreness"] == 2 and st["articulation"] is True and st["strands"] == 7
    # Rounded to whole percent at the seam: the raw pct-rank difference carries
    # four decimals of noise from a betweenness sampled at 400 pivots.
    assert st["surprise"] == 41 and st["dissonance"] == 50
    assert st["prerequisites"] == [{"name": "B", "path": "B.md"}]


def test_context_omits_the_block_for_a_note_the_graph_does_not_hold(client, tmp_vault,
                                                                   monkeypatch):
    tmp_vault.note("A.md", "alpha")
    monkeypatch.setattr("silica.kernel.report.structure.note_structure", lambda nid: {})
    assert client.get("/context", params={"path": "A.md"}).json()["structure"] == {}


def test_context_survives_a_structure_pass_that_raises(client, tmp_vault, monkeypatch):
    """A reading, never the drawer: the seven variables are the last section,
    and a broken index must cost that section and nothing else."""
    tmp_vault.note("A.md", "alpha")

    def _boom(nid):
        raise RuntimeError("index on fire")

    monkeypatch.setattr("silica.kernel.report.structure.note_structure", _boom)
    body = client.get("/context", params={"path": "A.md"}).json()
    assert body["structure"] == {} and body["title"] == "A"


def _fake_map(monkeypatch, **kw):
    m = structure.StructureMap(**kw)
    object.__setattr__(m, "_sizes", {})
    monkeypatch.setattr(structure, "structure_map", lambda: m)
    return m


def test_path_landing_ranks_two_sided_ladders_first(client, tmp_vault, monkeypatch):
    """A hub with 59 dependents and no prerequisites makes one rung of 59, which
    is a list with a title. The instructive root has notes on both sides."""
    tmp_vault.note("A.md", "a")
    _fake_map(
        monkeypatch,
        degree={"HUB.md": 1, "MID.md": 1, "BASE.md": 1, "TIP.md": 1, "F1.md": 1, "F2.md": 1},
        prereq={"F1": ["HUB"], "F2": ["HUB"], "MID": ["BASE"], "TIP": ["MID"]},
        unlocks={"HUB": ["F1", "F2"], "BASE": ["MID"], "MID": ["TIP"]},
    )
    picks = client.get("/path").json()["picks"]
    assert picks[0]["name"] == "MID"           # 1 before, 1 after
    assert picks[0]["before"] == 1 and picks[0]["after"] == 1


def test_path_says_why_it_is_empty_rather_than_showing_a_blank_pane(client, tmp_vault,
                                                                   monkeypatch):
    tmp_vault.note("A.md", "a")
    monkeypatch.setattr(structure, "structure_map", lambda: structure.StructureMap())
    body = client.get("/path").json()
    assert body["picks"] == []
    assert "co-occurrence index" in body["hint"]


def test_path_groups_the_ladder_into_rungs(client, tmp_vault, monkeypatch):
    tmp_vault.note("A.md", "a")
    _fake_map(monkeypatch, degree={"BASE.md": 1, "ROOT.md": 1, "TIP.md": 1},
              prereq={"ROOT": ["BASE"], "TIP": ["ROOT"]})
    body = client.get("/path", params={"note": "ROOT.md"}).json()
    assert body["root"] == {"name": "ROOT", "path": "ROOT.md"}
    assert [(lv["depth"], [n["name"] for n in lv["notes"]]) for lv in body["levels"]] == [
        (-1, ["BASE"]), (0, ["ROOT"]), (1, ["TIP"]),
    ]
    assert [n for lv in body["levels"] for n in lv["notes"] if n["root"]][0]["name"] == "ROOT"
    assert {"from": "BASE.md", "to": "ROOT.md"} in body["edges"]


def test_path_names_a_note_with_no_direction_instead_of_erroring(client, tmp_vault,
                                                                 monkeypatch):
    tmp_vault.note("A.md", "a")
    _fake_map(monkeypatch, degree={"A.md": 2, "B.md": 1, "C.md": 1}, prereq={"B": ["C"]})
    body = client.get("/path", params={"note": "A.md"}).json()
    assert body["levels"] == [] and "RefD found no direction" in body["hint"]


def test_calendar_carries_the_burst_and_survives_it_failing(client, tmp_vault, monkeypatch):
    rows = [{"concept": "gradient", "z": 3.1, "recent": 4, "total": 5}]
    monkeypatch.setattr("silica.kernel.report.structure.bursting", lambda: rows)
    assert client.get("/calendar", params={"days": 7}).json()["bursting"] == rows

    def _boom():
        raise RuntimeError("no index")

    monkeypatch.setattr("silica.kernel.report.structure.bursting", _boom)
    body = client.get("/calendar", params={"days": 7}).json()
    assert "bursting" not in body and "days" in body


def test_metrics_ships_the_two_store_derived_rows(client, tmp_vault):
    """Both ride the co-occurrence depth, so at structural depth they come back
    empty with their totals present, and the client is what decides how to say
    "this leg did not run"."""
    tmp_vault.note("A.md", "alpha [[B]]")
    tmp_vault.note("B.md", "beta")
    body = client.get("/metrics").json()
    assert body["sprawling"] == [] and body["bursting"] == []
    assert "sprawling" in body["totals"] and "bursting" in body["totals"]


def test_the_calendar_burst_and_the_report_burst_are_the_same_reading(tmp_vault,
                                                                     tmp_path,
                                                                     monkeypatch):
    """Two paths compute V6 because they cost two different amounts: the report
    pays 9.39 s for its co-occurrence depth, the calendar 0.33 s for this one
    variable. Two paths is a licence to disagree, so this is the test that
    revokes it. Both must call signals.burst over the same window.
    """
    from silica.kernel.recall.cooccurrence import CooccurStore, build_contribution
    from silica.kernel.report.graph_report import compute_report

    # Every note carries a wikilink: the structure map is built from the
    # wikilink graph, and a vault of unlinked files has no graph to be in.
    #
    # The proportions are what make the fixture work, not the words. burst()
    # fires at z >= 2, and z is a one-proportion score: four recent notes all
    # carrying a stem that eight older ones never do lands at 2.83. Three of
    # six lands at 1.73 and nothing burst at all, which is the correct answer
    # to a weaker contrast and a useless fixture.
    old, new = "2020-01-01", "2026-08-20"
    names = [f"R{i}" for i in range(4)] + [f"O{i}" for i in range(8)]
    text = {}
    for i, name in enumerate(names):
        nxt = names[(i + 1) % len(names)]
        recent = name.startswith("R")
        body = ("gradient descent tuning" if recent else "sailing boat harbour")
        text[name] = (f"{body} [[{nxt}]]", new if recent else old)
    for name, (body, date) in text.items():
        tmp_vault.note(f"{name}.md", f"---\ndate: {date}\n---\n{body}\n")
    store = CooccurStore(path=tmp_path / "c.json", lang="english")
    for name, (body, _d) in text.items():
        store.upsert_note(name, build_contribution(name, body))
    monkeypatch.setattr(
        "silica.kernel.recall.cooccurrence.get_cooccur_store", lambda *a, **k: store)

    structure._memo.clear()
    structure._burst_memo.clear()
    cheap = structure.bursting()

    report = compute_report(analytics=True, with_cooccurrence=True,
                            _cooccur_store_override=store)
    rich = [{"concept": b.concept, "z": b.z, "recent": b.recent, "total": b.total}
            for b in report.bursting_concepts]
    assert cheap, "the fixture has a real contrast: something must burst"
    assert cheap == rich[:len(cheap)]
