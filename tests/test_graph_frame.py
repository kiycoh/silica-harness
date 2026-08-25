"""The graph frame is three static files, filled by graph_view (ADR-0026).

What these hold: the skeleton and the Python agree on every slot in both
directions, the JS is a file node can parse, the data island cannot be ended by
what it carries, and a missing file is a packaging error with a name.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from string import Template

import pytest

STATIC = Path(__file__).resolve().parents[1] / "silica" / "ui" / "web" / "static"


def _island(html: str) -> str:
    start = html.index('<script id="graph-data"')
    start = html.index(">", start) + 1
    return html[start:html.index("</script>", start)]


def test_every_slot_is_filled_and_every_value_is_read():
    """substitute() already raises on a slot with no value; this is the other
    direction, a value the skeleton stopped asking for, which nothing else
    would ever report."""
    import silica.ui.web.graph_view as gv

    slots = set(Template(gv._asset("graph-frame.html")).get_identifiers())
    ctx = gv._frame_context([], [], [], "t", "", "", [])
    assert slots == set(ctx), {"unfilled": slots - set(ctx), "unread": set(ctx) - slots}


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to parse graph-frame.js")
def test_the_frame_code_is_a_file_node_can_parse():
    """The point of the split: the JS is no longer inside an f-string where a
    brace slip is a runtime error in a browser and nothing else."""
    out = subprocess.run(["node", "--check", str(STATIC / "graph-frame.js")],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr


def test_the_frame_reads_its_data_from_the_island_and_nowhere_else():
    import silica.ui.web.graph_view as gv

    html = gv.render_html([{"id": "a.md", "label": "a", "type": "note", "group": 0, "path": "a.md"}],
                          [], lib_js="// x")
    js = gv._asset("graph-frame.js")
    assert 'JSON.parse(document.getElementById("graph-data").textContent)' in js
    for sym in ("RAW_NODES", "RAW_EDGES", "COMM_LABELS", "ZONES", "PARTICLES", "SHADING"):
        assert f"const {sym} = DATA." in js, f"{sym} is not read off the island"
    data = json.loads(_island(html))
    assert data["nodes"][0]["id"] == "a.md"
    assert set(data) == {"nodes", "edges", "comm_labels", "zones", "particles", "shading", "colors"}


def test_the_island_cannot_be_ended_by_what_it_carries():
    """`</script>` is the obvious one. `<!--<script>` is the other: it moves the
    HTML parser into the double-escaped state, where the real closing tag no
    longer closes, which is why every `<` goes and not only `</`."""
    import silica.ui.web.graph_view as gv

    label = "x</script><!--<script>alert(1)//"
    html = gv.render_html([{"id": "n", "label": label, "type": "note", "group": 0, "path": label + ".md"}],
                          [], title="t", lib_js="// x")
    island = _island(html)
    assert "<" not in island
    assert json.loads(island)["nodes"][0]["label"] == label
    assert html.count("</script>") == 4   # theme resolver, lib, island, frame: no fifth


def test_a_non_finite_score_fails_on_the_server_with_a_name():
    """NaN was a working JS literal and would be a JSON.parse syntax error and a
    blank frame; the ValueError here is what /graph shows instead."""
    import silica.ui.web.graph_view as gv

    edges = [{"id": "e", "from": "a", "to": "b", "type": "SIMILAR", "score": float("nan")}]
    with pytest.raises(ValueError, match="JSON compliant"):
        gv.render_html([], edges, lib_js="// x")


def test_a_missing_frame_file_is_a_packaging_error_with_a_name():
    import silica.ui.web.graph_view as gv

    with pytest.raises(RuntimeError, match="nope.css"):
        gv._asset("nope.css")
    for name in gv._FRAME_ASSETS:
        assert gv._asset(name)


def test_the_document_references_nothing_by_url():
    """Opened from file:// there is no server: everything the page needs is in
    the page. The bundles and the face were already inlined; the frame's own
    three files must not be the exception."""
    import silica.ui.web.graph_view as gv

    html = gv.render_html([], [], lib_js="// x")
    assert "<link" not in html and "<script src=" not in html
    assert "/static/" not in html


# --- the edge encoding: two channels, and each says what it is --------------
# Seven edge kinds on the hue channel alone is more than a 1px antialiased line
# can carry, which is how "between zones" (#E8559E, hue 330) and "unresolved"
# (#e2544f, hue 2) came to be the same colour on screen: 32 degrees apart at
# identical saturation and near-identical luminance. The fix splits the load in
# two - solid means you wrote it, broken or moving means the vault inferred it -
# and takes the recolour layer off the hue channel entirely.

def _edge_colors(paper: bool) -> dict[str, str]:
    """Every edge colour the canvas can draw, by constant name.

    Particle colours are excluded on purpose: they are blended against the
    floor they land on (graph_view's comment on _EDGE_COLOR_SIMILAR_PARTICLE),
    so their luminance is chosen to hide, not to separate.
    """
    import silica.kernel.recall.graph_export as ge
    import silica.ui.web.graph_view as gv

    out = {}
    for mod in (ge, gv):
        for name, val in vars(mod).items():
            if not name.startswith("_EDGE_COLOR_") or "PARTICLE" in name:
                continue
            if name.endswith("_PAPER") is paper:
                out[name] = val
    return out


def _luma(hex_color: str) -> float:
    """Rec.709 weights on the sRGB values as stored, no gamma linearisation.

    The question here is "does this line read as brighter than that one on the
    same floor", which is a question about the numbers the GPU is handed. A CIE
    round trip would move every value by the same curve and change no verdict.
    """
    n = int(hex_color.lstrip("#"), 16)
    r, g, b = (n >> 16) & 255, (n >> 8) & 255, n & 255
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255


@pytest.mark.parametrize("paper,margin", [(False, 0.15), (True, 0.15)])
def test_the_recolour_layer_lives_at_a_luminance_no_hue_reaches(paper, margin):
    """"between zones" recolours wikilinks that are already on screen, so it is
    a highlight and not a seventh category. It rides luminance for that reason:
    brightest of all on the void, darkest of all on paper, by a margin no other
    edge colour can close. Hue it cannot have - every hue on that wheel is one
    the reader has to tell apart from six others already.
    """
    colors = _edge_colors(paper)
    key = "_EDGE_COLOR_DISCORD_PAPER" if paper else "_EDGE_COLOR_DISCORD"
    mine = _luma(colors.pop(key))
    others = {n: _luma(v) for n, v in colors.items()}
    assert others, "no edge colours to separate from - the harvest broke"
    if paper:
        worst = min(others, key=lambda n: others[n])
        gap = others[worst] - mine
    else:
        worst = max(others, key=lambda n: others[n])
        gap = mine - others[worst]
    assert gap >= margin, (
        f"discord luma {mine:.3f} is {gap:.3f} from {worst} ({others[worst]:.3f}); "
        f"the highlight has to clear every hue by {margin}"
    )


def test_a_broken_line_is_a_link_that_is_not_there():
    """The second channel is only worth having if the panel and the canvas agree
    on it. ABSENT is the canvas's list; the three legend rows that carry .dashed
    are the panel's. A full swatch over a broken line is a legend that lies about
    the picture under it.

    SIMILAR is the row this pins hardest. It is inferred, so a "solid is what you
    wrote" rule would dash it, and it must not be dashed: it is a relation that
    holds rather than one that is missing, and at 2901 of 4246 visible edges it
    is the ground the other layers are read against.
    """
    import silica.ui.web.graph_view as gv

    js = gv._asset("graph-frame.js")
    line = next(ln for ln in js.splitlines() if ln.startswith("const ABSENT"))
    assert set(json.loads(line[line.index("["):line.rindex("]") + 1])) == {
        "GAP", "PROPOSED", "COUPLED"}

    # Rendered and not read off the skeleton: three of the seven rows are in
    # graph-frame.html and four are built in graph_view, and a check that saw
    # only one of the two halves would pass while the other half lied.
    kinds = ["EXTRACTED", "AMBIGUOUS", "GAP", "SIMILAR", "PROPOSED", "COUPLED"]
    edges = [{"from": "a.md", "to": "b.md", "type": k} for k in kinds]
    edges.append({"from": "a.md", "to": "b.md", "type": "EXTRACTED", "discord": True})
    nodes = [{"id": f"{c}.md", "label": c, "type": "note", "group": 0, "path": f"{c}.md"}
             for c in "ab"]
    html = gv.render_html(nodes, edges, lib_js="// x")

    for cb, dashed in (("cb-extracted", False), ("cb-ambiguous", False),
                       ("cb-discord", False), ("cb-similar", False),
                       ("cb-gaps", True), ("cb-proposed", True),
                       ("cb-coupled", True)):
        row = html[html.index(f'id="{cb}"'):]
        row = row[:row.index("</label>")]
        assert ('dot-edge dashed' in row) is dashed, f"{cb} swatch disagrees with the canvas"


def test_both_renderers_break_the_inferred_line():
    """2D has the dash natively; 3D has no dash at all, but every link there is
    already merged into one LineSegments, and LineSegments IS the dash
    primitive - it draws disjoint vertex pairs. So the buffer carries a
    per-edge segment count rather than a flat two vertices, and that count comes
    from a period in WORLD units: a long link gets more dashes, never longer
    ones, which is the only version of the pattern that reads the same on every
    edge in the picture.
    """
    import silica.ui.web.graph_view as gv

    js = gv._asset("graph-frame.js")
    assert ".linkLineDash(" in js, "2D draws the absent layers solid"
    assert "DASH_PERIOD" in js and "DASH_CAP" in js, (
        "the 3D dash is a world-unit period with a bounded per-edge count")
    assert "DASH_SEGS" not in js, (
        "a fixed segment count makes the dash a FRACTION of the link, so a long "
        "edge draws long strokes with furrows cut in them")


def test_the_3d_mark_carries_the_rings_2d_draws():
    """The legend's NODE rows promise four readings in either renderer, and two
    of them - the state ring and load-bearing - used to exist in 2D only. They
    ride the one per-node channel this view has: the bundle shares a material
    per COLOUR and a geometry per size, so the value has to arrive on the object
    itself, immediately before its draw.
    """
    import silica.ui.web.graph_view as gv

    js = gv._asset("graph-frame.js")
    assert "onBeforeRender = nodeRingStep" in js, (
        "the ring colour is per NODE, and onBeforeRender is the only hook that "
        "runs per object while the material and the geometry are shared")
    assert "uSilState" in js and "uSilCut" in js, (
        "both rings, not one: a note can be a hub and load-bearing at once, "
        "which is why 2D draws them as two circles and not one")
    assert "RING_MIN_PX" in js, (
        "under a floor there is no note left under 6.5px of ring - 2D spends "
        "its zoom gate on the same problem")
    assert "n._dim" in js[js.index("function nodeRingStep"):], (
        "a dimmed note is a dot in 2D; a ring on it would outrank the focus")
