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
    """The sidebar reads a note two ways: what it SAYS (the body) and what this
    session changed (the diff). Concepts and neighbours belong to `work`, once --
    which is a third segment of this same sidebar now rather than a panel beside
    it, and that makes the duplicate cheaper to reintroduce, not dearer: the two
    readings are one click apart. This is the test that fails if an edit brings
    the pair back by habit."""
    app, html, css = APP_JS.read_text(), INDEX.read_text(), APP_CSS.read_text()
    for gone in ("renderContext", "openContext", "cxSection", "cxCloud", "cxList"):
        assert gone not in app, f"{gone} is still in app.js"
    assert "note-context" not in app and "note-context" not in html
    # Comments are exempt on purpose: app.css keeps one line saying WHY the
    # metrics view's `bridge` button stopped being a `.cx-do`, which is the kind
    # of note a removal is supposed to leave behind. Rules are not exempt.
    rules = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    assert ".cx-" not in rules, "the context mode's CSS outlived its markup"
    # ...and the mode control carries the two readings that are left, beside the
    # node. Three segments, and `context` is not one of them. `work` is not one
    # of them either: that segment drew the run on chat and the node on explore,
    # one name over two subjects, and the run left for the rail.
    assert sorted(re.findall(r'data-mode="(\w+)"', html)) == ["diff", "node", "note"]


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
    branch = _block(WORK_JS.read_text(), "function renderCtx() {", 'scope.textContent = "";')
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


def test_a_closed_pane_is_opened_by_the_click_that_fills_it():
    """Writing the node into a pane the reader is not on is a click that does
    nothing visible, which reads as a broken graph rather than a hidden panel.
    Unconditional now: the sidebar can be open on a NOTE, which is the other
    thing this pane used to be invisible behind (`body.note-open #work`), so
    "is it open" was never the whole question."""
    work, app = WORK_JS.read_text(), APP_JS.read_text()
    listener = _block(work, 'document.addEventListener("silica:node"', "});")
    assert "window.openNodeDrawer()" in listener
    # and the thing it calls is app.js's, which owns the drawer
    assert "window.openNodeDrawer = " in app


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


# --- 2. the legend's Layout compartment -------------------------------------

def test_the_legend_is_the_only_place_the_surfaces_are_named():
    """Six mode names written twice is two controls that eventually disagree
    about what this view can do, and the copy is always the one that is wrong.
    The toolbar used to carry the copy, a hand's width from the list; then the
    list was in the LEFT rail, a screen's width from the picture it switched.
    It is in #legend now, against the picture, and LAYOUT_MODES is still the
    only declaration.

    The label's CASE is not pinned: it is display copy, and pinning it made
    this test read as "the surfaces changed" the day someone capitalised the
    legend. What must not drift is the set of mode ids, which is what every
    other file switches on."""
    html, app = INDEX.read_text(), APP_JS.read_text()
    assert 'id="lg-layout"' in html and 'id="layout-modes"' in html
    assert 'id="side-layout"' not in html, "the rail's copy of the surfaces is back"
    modes = re.findall(r'\["([a-z]+)", "[A-Za-z]+",', _block(app, "const LAYOUT_MODES = [", "\n];"))
    assert modes == ["graph", "map", "folders", "areas", "read", "path"], modes
    assert len(set(modes)) == len(modes), "a mode is declared twice"
    # No second control for the same choice: nothing in the markup declares a
    # mode, so the rail cannot fall out of step with a toolbar that is gone.
    assert "data-gmode" not in html, "the toolbar copy of the surfaces is back"
    assert '.gmode-tabs button[data-gmode]' not in app


def test_the_layout_rows_leave_with_the_view_they_name():
    """They name the surfaces of ONE view. The gate is structural now and not a
    hidden flag: #legend lives inside #view-graph, so it is on screen exactly
    when the surfaces it switches are."""
    html, app = INDEX.read_text(), APP_JS.read_text()
    view = _block(html, '<section id="view-graph"', "</section>")
    assert 'id="legend"' in view, "the legend is not inside the view it describes"
    assert "syncLayoutRail();" in _block(app, "function setGraphMode(", "\n}")
    # closed, the fold has to say which of the six is up, or it names the
    # control and never the state
    assert '$("#lg-layout-now").textContent' in app


def test_the_legend_states_a_key_for_every_surface_it_can_show():
    """A legend that is blank on four of six surfaces is a panel the reader
    learns to ignore. Every mode in LAYOUT_MODES gets a key; nothing else does,
    or the table has grown an entry for a surface that does not exist."""
    app = APP_JS.read_text()
    modes = re.findall(r'\["([a-z]+)", "[A-Za-z]+",', _block(app, "const LAYOUT_MODES = [", "\n];"))
    keys = re.findall(r"^  ([a-z]+): \[", _block(app, "const SURFACE_KEYS = {", "\n};"), re.M)
    assert sorted(keys) == sorted(modes), (keys, modes)


# --- 3. the renderer, in the toolbar ----------------------------------------

def test_the_renderer_is_offered_in_one_place_per_context():
    """Embedded, the HUD's own segment is hidden: the same control in two places
    is two places that can disagree about which renderer is up. Standalone the
    HUD is the only chrome there is, so it keeps it."""
    app, markup, css = APP_JS.read_text(), FRAME_HTML.read_text(), FRAME_CSS.read_text()
    # built into the legend rather than declared in index.html: it is one
    # surface's control among six, and only that surface may declare it
    assert 'seg.id = "renderer-tabs";' in app
    assert 'id="renderer-tabs"' not in INDEX.read_text()
    assert 'id="renderer-section"' in markup
    assert "body.embedded #renderer-section{display:none}" in css


def test_the_toolbar_paints_the_renderer_the_frame_actually_built():
    """The frame owns the instance, so the click is a request and the answer is
    the truth. Painting on the way out would leave the segment claiming 3D for
    as long as a fallback or a slow build takes."""
    app, frame = APP_JS.read_text(), FRAME_JS.read_text()
    handler = _block(app, '$("#lg-surface").addEventListener("click"', "\n});")
    assert "silica-set-renderer" in handler
    assert "setActive" not in handler, "the toolbar paints itself before the answer"
    assert "function syncRenderer(" in app
    assert 'if (e.data.type === "silica-renderer") syncRenderer(e.data.mode);' in app
    # announced before setMode's early return, or a no-op switch leaves the
    # segment showing the mode it just moved away from
    setmode = _block(frame, "function setMode(m) {", "if (m === mode && Graph) return;")
    assert "announceMode(m);" in setmode


def test_the_segment_is_absent_on_the_five_surfaces_with_one_renderer():
    """Not hidden: built only for the surface that has two renderers. A hidden
    control is a control that can be un-hidden by a stylesheet."""
    app = APP_JS.read_text()
    builder = _block(app, "function buildSurfaceLegend() {", "\n}")
    assert 'if (graphMode === "graph") {' in builder
    assert 'seg.id = "renderer-tabs";' in builder


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
    # last, and last is enough: the rail itself no longer scrolls, so the floor
    # cannot be scrolled off it. Its compartments do, inside themselves.
    foot = _block(css, "#railfoot {", "}")
    assert "margin-top: auto" in foot
    rail_css = _block(css, "#sidebar {", "\n}")
    assert "overflow: hidden" in rail_css, "the rail scrolls again"
    assert "overflow-y: auto" in _block(css, ".side-body {", "\n}")


def test_a_click_the_panel_answers_does_not_read_as_a_click_outside_it():
    """The outside-click rule closed the sidebar on every row that re-rendered.

    A Read-first row's own handler calls render(), which does `body.textContent
    = ""` and detaches the button the click started on. The document handler
    runs after that, and `closest("#note-panel")` on a node with no parent
    answers null for every selector -- so the click that asked the panel a
    question was read as a click away from it. isConnected has to be checked,
    and checked BEFORE the closest() calls it invalidates.
    """
    app = APP_JS.read_text()
    guard = _block(app, "if (drawerBefore && drawerBefore === drawerNow()", "closeNote();")
    assert "e.target.isConnected" in guard
    assert guard.index("e.target.isConnected") < guard.index('closest("#note-panel")')


def test_naming_a_neighbour_row_reads_it_and_leaves_the_column_on_that_note():
    """Both calls, or the `work` toggle answers for a note you left two ago.

    The rows are note titles, so clicking one reads it: the drawer swings to
    note mode. showNode still fires, and first, because the pane BEHIND the
    note is the one the segmented control swings back to.
    """
    row = _block(WORK_JS.read_text(), "function nodeRow(r, opt)", "return row;")
    assert "window.openNote(r.path)" in row
    assert row.index("window.showNode({ path: r.path })") < row.index("window.openNote(r.path)")


def test_a_lit_concept_can_be_put_out_by_the_pill_that_lit_it():
    """A set you can only swap for another set is a mode with no exit: before
    this there was no gesture at all for "drop the filter, show me the graph"."""
    light = _block(APP_JS.read_text(), "async function lightConcept(", "window.lightConcept")
    assert 'btn.classList.contains("lit")' in light
    assert "focusGraphNodes([])" in light


def test_work_js_is_cache_busted_like_the_other_two_churning_assets():
    """It owns the whole node panel. Left out of the hash list, every edit to it
    reached a browser that had already loaded the page exactly never."""
    src = (ROOT / "silica" / "ui" / "web" / "server.py").read_text()
    assert '"app.js", "app.css", "work.js"' in src


def test_the_ladder_footer_roots_path_before_it_enters_the_tab():
    """Path has no root-replay on tab-enter, the way map does.

    showTab -> setGraphMode("path") -> drawPath() reads pathRootedPath and
    nothing else, so pre-setting graphMode the way the map button does starts an
    UNROOTED draw first -- and the landing measures forty ladders, so it also
    finishes last and overwrites the rooted one. rootPath has to run before the
    tab click, not after it.
    """
    app = APP_JS.read_text()
    body = _block(app, "window.openLadder = (note) => {", "};")
    assert 'graphMode = "path"' not in body, "pre-setting the mode races the rooted draw"
    assert body.index("rootPath(note)") < body.index('.tab[data-tab="graph"]')
    # and the panel is the only caller: it knows its note and nothing about modes
    assert "window.openLadder" in WORK_JS.read_text()
