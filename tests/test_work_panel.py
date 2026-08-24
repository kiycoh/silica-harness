# tests/test_work_panel.py
"""The work panel projects the narration stream, and the projection is the seam.

`/narration/sse` has been written and served since the narration shipped, and
until now nothing in the GUI read it: the panel is a projection of a file that
already exists. These tests pin the two things that can silently rot — the
fold that attaches an llm call to the thought that ran it, and the verb table
that has to stay in step with `silica/ui/renderer.py`.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "silica" / "ui" / "web"
WORK_JS = WEB / "static" / "work.js"
INDEX = WEB / "static" / "index.html"
APP_CSS = WEB / "static" / "app.css"

# The shape is the real one: silica/agent/narration.py writes these records and
# `read_beats` hands them back verbatim. A thought span WRAPS the llm call that
# produced it, both close on the same tick, and neither carries the other as a
# parent — which is exactly why the fold below cannot key on `parent`.
TRACE = [
    {"seq": 1, "ts": 100.0, "kind": "session", "status": "done", "id": None,
     "summary": "gui session, vault test", "payload": {"driver": "gui"}},
    {"seq": 2, "ts": 100.0, "kind": "turn", "status": "done", "id": None,
     "summary": "user: cerca utilitarismo",
     "payload": {"message": {"role": "user", "content": "cerca utilitarismo"}}},
    {"seq": 3, "ts": 100.0, "kind": "thought", "status": "running", "id": "th-3",
     "summary": "thinking", "payload": {}},
    {"seq": 4, "ts": 100.0, "kind": "call", "status": "running", "id": "c-1",
     "summary": "llm deepseek", "payload": {"model": "deepseek-v4-flash"}},
    {"seq": 5, "ts": 104.0, "kind": "call", "status": "done", "id": "c-1",
     "summary": "llm deepseek 10600 to 64 tok",
     "payload": {"model": "deepseek-v4-flash", "prompt_tokens": 10600,
                 "completion_tokens": 64, "cached_tokens": 0}},
    {"seq": 6, "ts": 104.0, "kind": "thought", "status": "done", "id": "th-3",
     "summary": "Cerchiamo note che menzionano utilitarismo.",
     "payload": {"text": "Cerchiamo note che menzionano utilitarismo.", "duration_s": 3.991}},
    {"seq": 7, "ts": 104.0, "kind": "tool", "status": "running", "id": "t-1",
     "summary": "silica_search_context",
     "payload": {"name": "silica_search_context", "args": {"query": "utilitarismo"}}},
    {"seq": 8, "ts": 104.02, "kind": "tool", "status": "done", "id": "t-1",
     "summary": "silica_search_context done",
     "payload": {"name": "silica_search_context", "result": "{}"}},
    {"seq": 9, "ts": 104.02, "kind": "tool", "status": "running", "id": "t-2",
     "summary": "silica_read_note",
     "payload": {"name": "silica_read_note", "args": {"name": "Morale"}}},
    {"seq": 10, "ts": 104.05, "kind": "tool", "status": "done", "id": "t-2",
     "summary": "silica_read_note done", "payload": {"name": "silica_read_note"}},
    {"seq": 11, "ts": 104.05, "kind": "tool", "status": "running", "id": "t-3",
     "summary": "silica_patch_note",
     "payload": {"name": "silica_patch_note", "args": {"name": "Filosofia/Morale.md"}}},
    {"seq": 12, "ts": 104.2, "kind": "tool", "status": "done", "id": "t-3",
     "summary": "silica_patch_note done", "payload": {"name": "silica_patch_note"}},
]


def _pure_block() -> str:
    """The projection half of work.js: pure, so it runs outside a browser."""
    src = WORK_JS.read_text()
    m = re.search(r"// --- projection:begin.*?// --- projection:end", src, re.S)
    assert m, "projection markers not found in work.js"
    block = m.group(0)
    for banned in ("document.", "window.", "fetch(", "EventSource"):
        assert banned not in block, f"the projection block reaches for {banned}"
    return block


def _project(tmp_path, beats: list[dict]) -> dict:
    """Run projectRun() from work.js under node, on these beats."""
    script = tmp_path / "project.js"
    script.write_text(
        _pure_block()
        + "\nconsole.log(JSON.stringify(projectRun(JSON.parse(process.argv[2]))));\n"
    )
    out = subprocess.run(["node", str(script), json.dumps(beats)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_a_call_folds_into_the_thought_that_ran_it(tmp_path):
    """One row, not two. The call and the thought open and close as one span in
    the log, so printing both prints the same 3.99 seconds twice."""
    run = _project(tmp_path, TRACE)
    thoughts = [r for r in run["rows"] if r["kind"] == "think"]
    assert len(thoughts) == 1
    assert thoughts[0]["text"] == "Cerchiamo note che menzionano utilitarismo."
    assert "10,600" in thoughts[0]["sub"] and "64" in thoughts[0]["sub"]
    assert thoughts[0]["dur"] == "4.0s"
    # and the call is not also a row of its own
    assert not [r for r in run["rows"] if r["kind"] == "call"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_tools_carry_the_verb_and_the_target_the_renderer_would_print(tmp_path):
    run = _project(tmp_path, TRACE)
    tools = [r for r in run["rows"] if r["kind"] == "tool"]
    assert [(t["verb"], t["target"]) for t in tools] == [
        ("search", "utilitarismo"),
        ("read", "Morale"),
        ("patch note", "Filosofia/Morale.md"),
    ]
    assert tools[0]["dur"] == "0.0s"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_a_write_tool_becomes_a_write_and_a_read_becomes_a_source(tmp_path):
    """The panel's two lists are derived, not separately reported: whether a
    tool changed the vault is the same table `_TOOL_EFFECT` keys on."""
    run = _project(tmp_path, TRACE)
    assert [w["target"] for w in run["writes"]] == ["Filosofia/Morale.md"]
    assert "Morale" in [s["name"] for s in run["sources"]]
    assert "Filosofia/Morale.md" not in [s["name"] for s in run["sources"]]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_an_unclosed_span_reads_as_running(tmp_path):
    """The live case: the panel renders mid-run, so the last beat is an open
    span and the projection has to say so rather than waiting for a close."""
    run = _project(tmp_path, TRACE[:7])
    assert run["running"] is True
    assert run["rows"][-1]["running"] is True
    done = _project(tmp_path, TRACE)
    assert done["running"] is False
    assert not [r for r in done["rows"] if r["running"]]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_the_user_turn_titles_the_run(tmp_path):
    run = _project(tmp_path, TRACE)
    assert run["title"] == "cerca utilitarismo"


def test_the_verb_table_cannot_drift_from_the_renderer():
    """work.js carries its own copy of the verb table, because the narration
    payload holds the RAW tool name and the browser has no renderer. A copy that
    nothing checks is a copy that goes stale the next time a tool is added."""
    from silica.ui.renderer import _TOOL_DESC

    src = WORK_JS.read_text()
    m = re.search(r"^const TOOL_VERBS = \{(.*?)^\};", src, re.S | re.M)
    assert m, "TOOL_VERBS not found in work.js"
    js_full = {name: (verb, key or None) for name, verb, key in
               re.findall(r'"(\w+)":\s*\["([^"]*)",\s*(?:"([^"]*)"|null)\]', m.group(1))}
    js = js_full
    assert js, "no entries parsed out of TOOL_VERBS"
    missing = sorted(set(_TOOL_DESC) - set(js))
    assert not missing, f"work.js TOOL_VERBS is missing {missing}"
    extra = sorted(set(js) - set(_TOOL_DESC))
    assert not extra, f"work.js TOOL_VERBS invents {extra}"
    for name, (verb, key) in _TOOL_DESC.items():
        assert js_full[name] == (verb, key), f"{name}: {js_full[name]} != {(verb, key)}"


def test_the_panel_is_wired_into_the_page():
    """The pane is part of the shell, not a widget bolted onto chat: the markup,
    the stylesheet and the script all have to be there."""
    html = INDEX.read_text()
    assert 'id="work-body"' in html
    assert '/static/work.js' in html
    # loaded after app.js, which owns openNote() and send()
    assert html.index("/static/app.js") < html.index("/static/work.js")
    assert "/narration/sse" in WORK_JS.read_text()


def test_the_run_is_a_mode_of_the_one_right_sidebar():
    """There were two asides at the right edge and the CSS said what that really
    was: `body.note-open #work { display: none }`, i.e. the one surface that says
    what the agent is doing was switched off for as long as any note was open.
    The run is a segment of the note sidebar now, so it cannot collide with the
    other two -- and the rules that kept the pair apart have to be gone, not
    merely unused, or the merged panel inherits a hiding rule for itself."""
    html, css, js = INDEX.read_text(), APP_CSS.read_text(), WORK_JS.read_text()
    drawer = html[html.index('<aside id="note-panel"'):]
    for pane in ("note-body", "note-diff", "work-body"):
        assert f'id="{pane}"' in drawer, f"{pane} is not a pane of the sidebar"
    modes = re.findall(r'data-mode="(\w+)"', drawer)
    assert sorted(modes) == ["diff", "note", "work"], modes
    # one surface at this edge, so nothing hides one half of it from the other
    assert '<aside id="work"' not in html
    # comments stripped: the rules are named in the prose that explains why they
    # are gone, and a rationale that quotes what it removed is the point of it
    rules = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for dead in ("body.note-open #work", "body.work-open", "body.work-fold"):
        assert dead not in rules, f"{dead} still arbitrates an edge with one panel"
    assert "work-open" not in js, "work.js still owns whether the sidebar is open"


def test_the_scope_chip_rides_the_shared_header_and_nothing_renames_it():
    """The pane used to rewrite its own header to "Node", which is what the note
    drawer's `context` mode said under a different word. The name is the active
    segment now and the scope is a chip beside it, in the sidebar's ONE header --
    above the panes, so it is stated once whichever pane is showing."""
    html, js = INDEX.read_text(), WORK_JS.read_text()
    head = html[html.index('<aside id="note-panel"'):html.index('id="note-actions"')]
    assert 'id="work-scope"' in head and 'id="work-state"' in head
    # nothing writes over the segment's name: the three views set the chip
    assert "work-title" not in js and "wk-title" not in html
    assert sorted(re.findall(r'scope\.textContent = "(\w*)"', js)) == ["", "node", "report"]


def test_the_sidebars_mark_is_a_sidebar_and_not_a_document():
    """A glyph each was how you told the two right-edge panels apart. With one
    panel there is nothing to tell apart, so the mark names the surface -- and it
    is the same glyph as the header button that opens it, or the control and the
    thing it opens are two signs for one act."""
    html = INDEX.read_text()
    drawer = html[html.index('<aside id="note-panel"'):html.index('id="note-actions"')]
    mark = re.search(r'<svg class="np-ico".*?</svg>', drawer, re.S)
    assert mark, "the sidebar has no mark"
    shape = re.search(r'<rect[^>]*>', mark.group(0))
    assert shape, "the mark is not the sidebar glyph"
    toggle = html[html.index('id="work-toggle"'):html.index('id="stop"')]
    assert shape.group(0) in toggle, "the button that opens it wears a different glyph"
    # and it is the ONE control: #note-last was an accent-blue document glyph
    # beside it opening the same sidebar on a different mode. Markup only —
    # index.html keeps a line saying what stood there and why it does not.
    markup = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    assert "note-last" not in markup
    assert "note-last" not in (WEB / "static" / "app.js").read_text()


def test_a_click_that_moves_the_sidebar_never_also_closes_it():
    """The outside-click rule closes the sidebar, and it used to key on "is the
    panel open", read live on the way UP. With three modes that is wrong twice:
    a metrics or shape row POINTS at a note, raising the sidebar on `work` from
    its own listener, and the delegated handler then closed what the click had
    just asked for -- whether the click opened the sidebar or swung it off a note.
    The sample is taken in the capture pass and compared, so the rule is "did this
    click leave it alone" rather than "was it open"."""
    app = (WEB / "static" / "app.js").read_text()
    assert '{ drawerBefore = drawerNow(); }, true)' in app, "the sample is not captured"
    assert "drawerBefore && drawerBefore === drawerNow()" in app, \
        "the close still keys on a live read"
    # the sample is registered before the handler that reads it
    assert app.index("drawerBefore = drawerNow()") < app.index("drawerBefore === drawerNow()")


def test_the_kept_pane_survives_its_own_boot():
    """Two orderings, both of which opened the sidebar and shut it again inside
    one frame -- which reads as a preference that was never saved rather than one
    that was undone. The restore has to run AFTER the boot tab, because switching
    tabs closes the sidebar; and the preference has to be READ before that, because
    syncDrawerMode() writes the key on every call and the first call happens with
    the sidebar still closed."""
    app = (WEB / "static" / "app.js").read_text()
    read = app.index("const bootWantsWork")
    sync = app.index("function syncDrawerMode(")
    assert read < sync, "the preference is read after the function that overwrites it"
    restore = app.index("if (bootWantsWork) openWork()")
    boot_tab = app.index('`.tab[data-tab="${[')
    assert boot_tab < restore, "the boot tab is clicked after the restore, and closes it"
    # ...and one place writes the key, or the two can disagree about it
    assert app.count('localStorage.setItem("work-open"') == 1


def test_every_rail_compartment_wears_its_own_icon():
    """Below the 1120 floor the compartments ARE their icons (#railmini), so a
    glyph that means "areas" collapsed and nothing open is a glyph you have to
    learn twice. Same mark in both, one per section."""
    html = INDEX.read_text()
    rail = html[html.index('<aside id="railmini"'):html.index('<aside id="sidebar">')]
    sections = re.findall(r'<details class="side-section" id="(side-\w+)"[^>]*>\s*'
                          r'<summary class="side-title">(.*?)</summary>', html, re.S)
    assert len(sections) == 6, [s[0] for s in sections]
    for sec, summary in sections:
        assert 'class="si"' in summary, f"{sec} has no icon in its header"
        # the shape itself, not just any icon: the first path/polygon/circle of
        # the railmini button for this section has to appear in the summary too
        btn = re.search(rf'data-sec="{sec}".*?</button>', rail, re.S)
        assert btn, f"{sec} has no button in the folded rail"
        shape = re.search(r'<(?:path|polygon|circle) [^>]*>', btn.group(0))
        assert shape and shape.group(0) in summary, f"{sec}: rail and header disagree"
