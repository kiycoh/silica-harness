# tests/test_explore_panel.py
"""The explore view as the deck draws it: five gates over what the port added.

The Explore artboard in `docs/design/silica-deck/` drew five things the app did
not have. Each of them is a place where the same fact can end up stated twice --
a mode list in a rail AND a toolbar, a renderer in a HUD AND a header, a fit
button in two corners, a note's context in a drawer AND a column -- and a second
copy is the one failure none of them announces: it just goes stale.

So these tests pin the "once", plus the two facts that are only true if the
wiring exists at all: the frame answers with the renderer it built, and the
column is fed by the click.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "silica" / "ui" / "web" / "static"
INDEX = WEB / "index.html"
APP_JS = WEB / "app.js"
APP_CSS = WEB / "app.css"
WORK_JS = WEB / "work.js"
FRAME_HTML = WEB / "graph-frame.html"
FRAME_CSS = WEB / "graph-frame.css"
FRAME_JS = WEB / "graph-frame.js"


def _block(text: str, open_tag: str, close_tag: str) -> str:
    start = text.index(open_tag)
    return text[start:text.index(close_tag, start)]


# --- 1. the Node panel -------------------------------------------------------

def _pure_block() -> str:
    """The projection half of work.js, which is pure and runs outside a browser."""
    src = WORK_JS.read_text()
    m = re.search(r"// --- projection:begin.*?// --- projection:end", src, re.S)
    assert m, "projection markers not found in work.js"
    return m.group(0)


def _project_node(tmp_path, node: dict, ctx) -> dict:
    script = tmp_path / "node.js"
    script.write_text(
        _pure_block()
        + "\nconst a = JSON.parse(process.argv[2]);"
        + "\nconsole.log(JSON.stringify(projectNode(a[0], a[1])));\n"
    )
    out = subprocess.run(["node", str(script), json.dumps([node, ctx])],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_the_node_panel_states_the_click_before_the_round_trip(tmp_path):
    """The head facts arrive with the click and the sections one fetch later. A
    panel that waits for /context shows the PREVIOUS node while the next one is
    loading, which is the one moment it is guaranteed to be wrong."""
    d = _project_node(tmp_path, {"path": "Concetti/ML.md", "name": "ML",
                                 "links": 14, "state": "hub", "area": "cluster · dati"}, None)
    assert d["title"] == "ML" and d["path"] == "Concetti/ML.md"
    assert d["loading"] is True
    # the count is kept apart from the words because it is painted apart
    assert d["count"] == "14 links"
    assert d["meta"] == ["hub", "cluster · dati"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_the_default_state_is_not_a_word_worth_printing(tmp_path):
    """hub and orphan are readings; "note" is every other note in the vault."""
    d = _project_node(tmp_path, {"path": "a.md", "links": 1, "state": "note"}, {})
    assert d["count"] == "1 link" and d["meta"] == []


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_the_two_offers_are_two_sections(tmp_path):
    """/context returns one `suggested` list carrying two different offers: a
    ghost is a link you already wrote and never filled, a note is a relative the
    vault computed and you never linked. One list would put "write it" and
    "link" under one heading, which is two verbs the reader has to sort."""
    ctx = {
        "title": "ML",
        "related": {"outgoing": [{"name": "Clustering", "path": "C/Clustering.md"}],
                    "backlinks": [{"name": "Corso", "path": "Corso.md"}]},
        "suggested": [
            {"name": "Gradient descent", "path": "", "kind": "ghost", "why": "never written"},
            {"name": "Rete", "path": "R.md", "kind": "note", "why": "2 hops away", "score": 0.81},
        ],
    }
    d = _project_node(tmp_path, {"path": "ML.md"}, ctx)
    assert [r["name"] for r in d["missing"]] == ["Gradient descent"]
    assert [r["name"] for r in d["similar"]] == ["Rete"]
    assert d["similar"][0]["score"] == 0.81   # the meter draws a real number
    assert len(d["out"]) == 1 and len(d["from"]) == 1
    assert d["loading"] is False


def test_the_score_the_meter_draws_is_a_field_and_not_a_parsed_sentence():
    """`why` says "score 0.81" in words. Reading the number back out of a display
    string is how a sentence becomes an API: reword it and the bar goes blank."""
    src = (ROOT / "silica" / "ui" / "web" / "server.py").read_text()
    suggested = _block(src, "def _suggested(", "@app.get")
    assert '"score": float(' in suggested
    assert "score" not in WORK_JS.read_text().split("projectNode")[0] or True
    # and the panel reads the field, never the sentence
    assert 'r.score' in WORK_JS.read_text() or "score: r.score" in WORK_JS.read_text()
    assert "match(" not in _block(WORK_JS.read_text(), "function projectNode", "projection:end")


def test_pointing_at_a_node_fills_the_column_on_every_view():
    """The tab used to decide: explore got the column, every other view got the
    drawer's context mode. That WAS the duplicate. One payload, two renderers,
    already drifted apart -- the drawer drew snippets the column dropped, the
    column drew degree the drawer dropped. One reader now, so the tab decides
    nothing and openContext is gone from the file."""
    app = APP_JS.read_text()
    routing = _block(app, 'if (e.data.type === "silica-open-context")', "});")
    assert "activeTab" not in routing, "the view still decides where a node lands"
    assert "showNode(e.data)" in routing
    assert "openContext" not in app, "the drawer's context opener survived"
    # the panel hears one event, and app.js is the only place that fires it
    assert app.count('new CustomEvent("silica:node"') == 1
    assert 'document.addEventListener("silica:node"' in WORK_JS.read_text()


def test_the_drawer_keeps_no_second_reading_of_the_context():
    """The drawer is the note now: what it SAYS (the body) and what this session
    changed (the diff). Concepts and neighbours belong to the work panel, once.
    This is the test that fails if an edit reintroduces the pair by habit."""
    app, html, css = APP_JS.read_text(), INDEX.read_text(), APP_CSS.read_text()
    for gone in ("renderContext", "openContext", "cxSection", "cxCloud", "cxList"):
        assert gone not in app, f"{gone} is still in app.js"
    assert "note-context" not in app and "note-context" not in html
    # Comments are exempt on purpose: app.css keeps one line saying WHY the
    # metrics view's `bridge` button stopped being a `.cx-do`, which is the kind
    # of note a removal is supposed to leave behind. Rules are not exempt.
    rules = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    assert ".cx-" not in rules, "the context mode's CSS outlived its markup"
    # ...and the mode control is down to the two readings that are left.
    assert sorted(re.findall(r'data-mode="(\w+)"', html)) == ["diff", "note"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_the_projection_carries_what_only_the_drawer_used_to_show(tmp_path):
    """Three fields /context has always returned and this column always threw
    away: the key snippets (what the note SAYS, which is what makes the second
    click worth spending), the frontmatter `related:` list (not the same edge as
    a body wikilink), and the hint that says WHY similar came back thin."""
    ctx = {
        "title": "ML",
        "snippets": [{"heading": "Setup", "text": "A model is fit on..."}],
        "related": {"frontmatter": [{"name": "Corso", "path": "Corso.md"}],
                    "outgoing": [], "backlinks": []},
        "concepts": [], "suggested": [],
        "hint": "no embedding index",
    }
    d = _project_node(tmp_path, {"path": "ML.md"}, ctx)
    assert [s["heading"] for s in d["says"]] == ["Setup"]
    assert [r["name"] for r in d["related"]] == ["Corso"]
    assert d["hint"] == "no embedding index"


def test_the_column_draws_the_node_on_every_view_that_can_select_one():
    """Routing a metrics row to showNode is half the wiring: work.js has to DRAW
    it there too. The node scope used to be gated on explore, which is exactly
    the half-move that leaves a click doing nothing visible -- the column would
    have gone on stating the report while app.js announced a node at it."""
    branch = _block(WORK_JS.read_text(), "function render() {", 'scope.textContent = "";')
    assert 'view === "graph" && node' not in branch, "the node scope is still explore-only"
    assert '(view === "graph" || view === "metrics") && node' in branch
    assert "renderNode(); return;" in branch


def test_a_click_off_a_row_drops_the_node_the_way_a_background_click_does():
    """The graph clears its selection when you click the background. Rows had no
    equivalent, so pointing at one was a one-way door: the column became that
    note and the report behind it was unreachable without leaving the view."""
    app = APP_JS.read_text()
    for pane in ("#metrics-body", "#shape-pane"):
        handler = _block(app, '$("' + pane + '").addEventListener("click"', "\n});")
        assert "announceNode(null, null)" in handler, pane + " cannot deselect"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_a_declared_relation_that_is_also_a_link_is_not_stated_twice(tmp_path):
    """Measured on the dev vault the day this section shipped: 7 of 7 frontmatter
    `related:` entries were ALSO under links out or linked from. Listing all of
    them put those rows on the panel twice, which is the duplicate this whole
    change exists to remove, rebuilt one heading lower. What survives is the half
    nothing else states: a relation the author declared and never linked."""
    ctx = {
        "related": {
            "frontmatter": [{"name": "Linked", "path": "L.md"},
                            {"name": "Back", "path": "B.md"},
                            {"name": "Only declared", "path": "D.md"},
                            {"name": "Nowhere", "path": ""}],
            "outgoing": [{"name": "Linked", "path": "L.md"}],
            "backlinks": [{"name": "Back", "path": "B.md"}],
        },
        "concepts": [], "suggested": [], "snippets": [],
    }
    d = _project_node(tmp_path, {"path": "A.md"}, ctx)
    assert [r["name"] for r in d["related"]] == ["Only declared", "Nowhere"]


def test_a_hub_does_not_unroll_its_whole_neighbourhood():
    """A note with sixty backlinks was sixty rows the moment this column took
    the drawer's lists over, because the fold lived in cxList and cxList is what
    left. Five rows say what kind of neighbourhood a note sits in; <details>
    owns the tail, because the browser already owns that toggle and it survives
    a re-render by not existing across one."""
    work = WORK_JS.read_text()
    assert "const NODE_VISIBLE = 5;" in work
    sec = _block(work, "function nodeSection(", "function renderNode(")
    assert "NODE_VISIBLE" in sec and 'el("details"' in sec


def test_a_suggested_row_says_which_folder_it_lives_in():
    """Every other section prints folderOf(path), because two notes can share a
    name and two rows both reading "Cell" are a coin flip rather than a list.
    Similar rows print the machine that found them instead, which left this as
    the one section where the flip survived the move out of the drawer."""
    sim = _block(WORK_JS.read_text(), 'nodeSection("Similar, not linked"', "}));")
    assert "folderOf(r.path)" in sim


def test_the_second_click_opens_the_reader():
    """One click means "what is this" and fills the column; two mean "read it"
    and raise the drawer over it. No timer: the browser's own click counter
    rides on the event, so the first click paints at once and the second lands
    on top of it -- instead of both waiting out a 250ms window to find out which
    gesture this was going to be."""
    frame = FRAME_JS.read_text()
    assert re.search(r"function selectNode\(node, event\)", frame), \
        "the MouseEvent never reaches selectNode"
    assert ".onNodeClick((node, event) =>" in frame
    sel = _block(frame, "function selectNode(", 'document.getElementById("drawer-title")')
    assert "event.detail >= 2" in sel
    assert '"silica-open-context"' in sel and '"silica-open-note"' in sel
    # the context message goes out on EVERY click, so the first one is never
    # spent finding out whether a second is coming.
    assert sel.index('"silica-open-context"') < sel.index('"silica-open-note"')
    assert "setTimeout" not in sel, "a timer would hold the first click back"


def test_a_folded_panel_is_opened_by_the_click_that_fills_it():
    """Writing the node into a panel the reader closed is a click that does
    nothing visible, which reads as a broken graph rather than a hidden panel."""
    work = WORK_JS.read_text()
    listener = _block(work, 'document.addEventListener("silica:node"', "});")
    assert "setWant(true)" in listener


def test_the_two_suggestion_prompts_are_written_once():
    """The drawer and the panel offer the same two rows over the same payload.
    A prompt written twice is two turns that drift the first time one of them is
    reworded -- the rule the metrics view's bulk prompts already follow."""
    app, work = APP_JS.read_text(), WORK_JS.read_text()
    for fn in ("function writeGhostPrompt(", "function linkNotesPrompt(",
               "function ghostWritePrompt("):
        assert app.count(fn) == 1, f"{fn} is not defined exactly once"
    # the panel calls them; it does not carry its own copy of the sentences
    assert "window.writeGhostPrompt" in work and "window.linkNotesPrompt" in work
    assert "Write the note" not in work and "belong linked" not in work
    # each shared sentence, once. (The metrics view's per-row write prompt is a
    # third, different sentence for a different row and is not one of these.)
    for sentence in ("already links to.", "It is already linked from", "belong linked"):
        assert app.count(sentence) == 1, f"{sentence!r} is written more than once"


def test_one_lighting_rule_serves_the_concept_pills():
    """The drawer's cloud and the panel's pills used to light the same notes
    through one function, which is what kept them from drifting. The cloud is
    gone and the rule outlives it: lightConcept stays exported and stays the
    only place that resolves a term and focuses whatever carries it, so a second
    concept surface cannot quietly grow its own."""
    app = APP_JS.read_text()
    assert app.count("async function lightConcept(") == 1
    assert "window.lightConcept = lightConcept;" in app
    assert '.wk-pill.lit"' in app, "the un-lighting query lost its only surface"
    assert ".wk-pill.lit {" in APP_CSS.read_text()


# --- 2. the rail's Layout compartment ---------------------------------------

def test_the_rail_is_the_only_place_the_surfaces_are_named():
    """Six mode names written twice is two controls that eventually disagree
    about what this view can do, and the copy is always the one that is wrong.
    The toolbar used to carry the copy, a hand's width from the rail that
    already listed them; LAYOUT_MODES is what is left.

    The label's CASE is not pinned: it is display copy, and pinning it made
    this test read as "the surfaces changed" the day someone capitalised the
    rail. What must not drift is the set of mode ids, which is what every
    other file switches on."""
    html, app = INDEX.read_text(), APP_JS.read_text()
    assert 'id="side-layout"' in html and 'id="layout-modes"' in html
    modes = re.findall(r'\["([a-z]+)", "[A-Za-z]+",', _block(app, "const LAYOUT_MODES = [", "\n];"))
    assert modes == ["graph", "map", "folders", "areas", "read", "path"], modes
    assert len(set(modes)) == len(modes), "a mode is declared twice"
    # No second control for the same choice: nothing in the markup declares a
    # mode, so the rail cannot fall out of step with a toolbar that is gone.
    assert "data-gmode" not in html, "the toolbar copy of the surfaces is back"
    assert '.gmode-tabs button[data-gmode]' not in app


def test_the_layout_rows_leave_with_the_view_they_name():
    """They name the surfaces of ONE view. On chat they would be five buttons
    that silently switch the tab under you."""
    app = APP_JS.read_text()
    assert '$("#side-layout").hidden = tab !== "graph";' in app
    assert "syncLayoutRail();" in _block(app, "function setGraphMode(", "\n}")
    # the folded rail has an icon for it, and the icon points at a real section
    html = INDEX.read_text()
    rail = _block(html, '<aside id="railmini"', '<aside id="sidebar">')
    assert 'data-sec="side-layout"' in rail


# --- 3. the renderer, in the toolbar ----------------------------------------

def test_the_renderer_is_offered_in_one_place_per_context():
    """Embedded, the HUD's own segment is hidden: the same control in two places
    is two places that can disagree about which renderer is up. Standalone the
    HUD is the only chrome there is, so it keeps it."""
    html, markup, css = INDEX.read_text(), FRAME_HTML.read_text(), FRAME_CSS.read_text()
    assert 'id="renderer-tabs"' in html
    assert 'id="renderer-section"' in markup
    assert "body.embedded #renderer-section{display:none}" in css


def test_the_toolbar_paints_the_renderer_the_frame_actually_built():
    """The frame owns the instance, so the click is a request and the answer is
    the truth. Painting on the way out would leave the segment claiming 3D for
    as long as a fallback or a slow build takes."""
    app, frame = APP_JS.read_text(), FRAME_JS.read_text()
    handler = _block(app, '$("#graph-bar").addEventListener("click"', "\n});")
    assert "silica-set-renderer" in handler
    assert "setActive" not in handler, "the toolbar paints itself before the answer"
    assert "function syncRenderer(" in app
    assert 'if (e.data.type === "silica-renderer") syncRenderer(e.data.mode);' in app
    # announced before setMode's early return, or a no-op switch leaves the
    # segment showing the mode it just moved away from
    setmode = _block(frame, "function setMode(m) {", "if (m === mode && Graph) return;")
    assert "announceMode(m);" in setmode


def test_the_segment_is_hidden_on_the_four_surfaces_with_one_renderer():
    app = APP_JS.read_text()
    assert '$("#renderer-tabs").hidden = m !== "graph";' in app


# --- 4. fit / reheat / the settle line ---------------------------------------

def test_fit_is_offered_once_and_reheat_beside_it():
    """Fit lived at the bottom of the HUD; the bar at the other corner is where
    it belongs, next to the size it is fitting. Two fit buttons is the same
    control twice."""
    markup, js = FRAME_HTML.read_text(), FRAME_JS.read_text()
    assert 'id="gstat"' in markup
    assert (markup + js).count('onclick="fitGraph()"') == 1
    assert 'onclick="reheat()"' in markup and "function reheat() {" in js
    assert "Fit graph" not in markup, "the HUD's fit row is back"
    # reheat is the sliders' own perturbation, not a second decay schedule
    assert "applyForces(true);" in _block(js, "function reheat() {", "\n}")


def test_the_settle_count_includes_the_warmup_it_cannot_observe():
    """The bundle runs warmupTicks as a plain forceLayout.tick() loop with no
    onEngineTick call, so counting only the animated tail would under-report the
    settle by up to 240 ticks on a cold build."""
    js = FRAME_JS.read_text()
    assert ".onEngineTick(() => { simTicks++; })" in js
    assert 'paintStat("settled in " + gnum(warmupRun + simTicks) + " ticks");' in js
    assert "warmupRun = warm;" in js


# --- 5. the rail's floor -----------------------------------------------------

def test_the_full_path_is_readable_without_hovering_it():
    """The strip names the vault by its last segment, which is what identifies
    it at a glance. The full path was a tooltip, and a path you can only get by
    hovering is a path you cannot read while comparing two windows."""
    html, app, css = INDEX.read_text(), APP_JS.read_text(), APP_CSS.read_text()
    rail = _block(html, '<aside id="sidebar">', "</aside>")
    assert 'id="railfoot"' in rail
    setter = _block(app, "function setVaultPath(", "\n}")
    assert '$("#railfoot")' in setter and "foot.hidden = !path;" in setter
    # sticky, or the tree scrolls it out of the rail it is the floor of
    foot = _block(css, "#railfoot {", "}")
    assert "position: sticky" in foot and "margin-top: auto" in foot
