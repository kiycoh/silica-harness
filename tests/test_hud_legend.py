# tests/test_hud_legend.py
"""The graph HUD answers three questions, and hides its machinery under them.

It used to stack six panels: renderer, edge types, semantic zones, node state, a
91-row community list and three force sliders, with a rebuild button under all
of it. Two of those (the list, the sliders) held most of the height, and neither
is legend -- they are controls you reach for once. So the reading you actually
came for ("what is this line, what is this dot, what does the colour mean") was
buried in its own machinery.

Nothing was deleted, which is the part these tests exist to keep true: the list
opens from the row that counts it, the forces are one fold down, and every id
the JS drives is still there.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAME_JS = ROOT / "silica" / "ui" / "web" / "static" / "graph-frame.js"


def _hud(html: str) -> str:
    start = html.index('<div id="hud">')
    return html[start:html.index('<div id="drawer">', start)]


def _render() -> str:
    from silica.ui.web.graph_view import render_html
    nodes = [
        {"id": "a.md", "label": "A", "type": "note", "group": 0, "size": 3,
         "betweenness": 0.4, "path": "a.md"},
        {"id": "b.md", "label": "B", "type": "note", "group": 0, "size": 2,
         "betweenness": 0.0, "path": "b.md"},
    ]
    edges = [{"from": "a.md", "to": "b.md", "type": "EXTRACTED"},
             {"from": "b.md", "to": "a.md", "type": "SIMILAR"}]
    return render_html(nodes, edges, communities=[], lib_js="// x")


def test_the_panel_asks_three_questions_in_reading_order():
    """Lines, dots, colour: the order you ask them of a picture of a graph."""
    hud = _hud(_render())
    heads = [hud.index(">Edges<"), hud.index(">Nodes<"), hud.index(">Colour<")]
    assert heads == sorted(heads), "the three headings are out of reading order"
    # the old names are gone, not merely re-styled
    for dead in (">Edge types<", ">Node state<", ">Communities<", ">Semantic zones<"):
        assert dead not in hud, f"{dead} survived the regrouping"


def test_the_area_list_is_the_answer_to_the_row_that_counts_it():
    """Closed, the row states how many areas there are, which IS the legend.
    Open, it is the filter it always was: same ids, same handlers."""
    hud = _hud(_render())
    fold = hud[hud.index('<details id="areas-fold">'):hud.index("</details>")]
    assert "structural areas" in fold
    assert 'id="legend-box"' in fold and 'id="legend-all"' in fold
    assert 'id="sort-communities"' in fold and "toggleCommunitySort()" in fold
    assert "filterCommunity(-2)" in fold
    # and it is closed on arrival: an open 91-row list is the height problem
    assert "<details id=\"areas-fold\" open" not in hud


def test_the_controls_that_are_not_legend_are_one_fold_down():
    """Named Forces and not Layout: the app's own legend is painted directly
    above this panel and carries a Layout compartment of its own, which switches
    which of the six SURFACES you are on. Two folds a centimetre apart, one named
    after the other's subject, is one fold you open by mistake every time."""
    hud = _hud(_render())
    fold = hud[hud.index('<details id="forces-fold">'):]
    assert "<summary class=\"section-title\">Forces</summary>" in fold
    for control in ("sl-repel", "sl-dist", "sl-center", "resetForces()",
                    "onForceSlider()", "Rebuild"):
        assert control in fold, f"{control} is not under the Forces fold"
    assert '<details id="forces-fold" open' not in hud
    assert "layout-fold" not in hud, "the old name is still in the panel"


def test_nothing_the_frame_drives_lost_its_id():
    """The regrouping moved markup, not behaviour. Every id below is read or
    written by a function in this same file, and a rename here fails silently in
    the browser: the handler just stops finding its element."""
    html = _render()
    for el in ("cb-extracted", "cb-ambiguous", "cb-gaps", "cb-similar",
               "state-legend", "st-hub", "st-orphan", "st-ghost",
               "legend-box", "legend-all", "sort-communities",
               "mode-toggle", "renderer-section",
               "sl-repel", "sl-dist", "sl-center", "fv-repel", "fv-dist", "fv-center"):
        assert f'id="{el}"' in html, f"{el} is gone from the frame"


def test_the_nodes_heading_hides_with_the_rings_it_names():
    """The three rows under Nodes are a 2D-only channel and setMode hides the
    block. The heading rides INSIDE that block, or 3D shows a title with nothing
    under it."""
    hud = _hud(_render())
    block = hud[hud.index('<div id="state-legend">'):]
    block = block[:block.index("</div>\n\n")]
    assert ">Nodes<" in block
    js = FRAME_JS.read_text()
    assert 'document.getElementById("state-legend").style.display = is2D()' in js


def test_the_two_partitions_are_two_rows_of_one_section():
    """Structural areas and semantic zones are both answers to "what does this
    colour mean", so they read together. The zone rows only exist where the
    vault has vectors, which is why they are a separate f-string."""
    from silica.ui.web.graph_view import Zone, render_html
    nodes = [{"id": "a.md", "label": "A", "type": "note", "group": 0, "sgroup": 0,
              "size": 3, "betweenness": 0.4, "path": "a.md"}]
    zones = [Zone(id=0, label="z", color="#888", color_paper="#444", size=1)]
    hud = _hud(render_html(nodes, [], communities=[], lib_js="// x", zones=zones))
    colour = hud[hud.index(">Colour<"):]
    assert "structural areas" in colour and "semantic zones" in colour
    assert 'id="cb-zones"' in colour and 'id="cb-zone-nodes"' in colour
    # no vectors, no rows: an empty checkbox promises a layer with nothing behind it
    plain = _hud(_render())
    assert 'id="cb-zones"' not in plain
