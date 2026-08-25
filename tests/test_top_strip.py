# tests/test_top_strip.py
"""The deck's substrate move: identity and counts are stated once.

The rail used to carry the vault's size as a 2x2 board while the metrics view
carried four of the same numbers again, and three boot-diagnostic cards held
roughly 420 of the rail's 1000 pixels from launch for the whole session. Both
are now in the top strip -- the counts because they are true on every view, the
notices behind a chip because configuration is a thing you open, not a thing
that sits on top of the vault you came to read.

These tests pin the "once", which is the only part that can silently come back:
a second copy of a count is added the day someone wants it closer to hand.
"""
from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "silica" / "ui" / "web" / "static"
INDEX = WEB / "index.html"
APP_JS = WEB / "app.js"
APP_CSS = WEB / "app.css"
WORK_JS = WEB / "work.js"


def _block(html: str, open_tag: str, close_tag: str) -> str:
    start = html.index(open_tag)
    return html[start:html.index(close_tag, start)]


def test_the_vault_states_its_size_in_the_strip_and_nowhere_else():
    html = INDEX.read_text()
    header = _block(html, "<header>", "</header>")
    rail = _block(html, '<aside id="sidebar">', "</aside>")

    assert 'id="top-counts"' in header and 'id="top-broken"' in header
    # the rail's old board, by every name it had
    for dead in ("stat-grid", "stat-notes", "stat-links", "stat-clusters",
                 "stat-unresolved"):
        assert dead not in html, f"the rail's {dead} is back"
    assert 'id="top-counts"' not in rail


def test_the_area_count_has_one_rule():
    """The strip and the chat landing both state 'N areas'. A cluster count made
    entirely of singletons names nothing, so the count is gated -- and the gate
    stated twice is two numbers that can disagree about the same vault."""
    src = APP_JS.read_text()
    assert src.count("function areaCount(") == 1
    # the raw expression lives only inside that function
    assert src.count("(data.topics || []).length ?") == 1
    assert src.count("= areaCount(data)") == 2  # strip + landing


def test_the_notices_are_behind_the_chip_not_camped_in_the_rail():
    html = INDEX.read_text()
    header = _block(html, "<header>", "</header>")
    rail = _block(html, '<aside id="sidebar">', "</aside>")
    assert 'id="notices-btn"' in header
    assert 'id="boot-notices"' not in rail
    panel = _block(html, 'id="notices-panel"', "</div>\n    <!-- the gear alone")
    assert 'id="boot-notices"' in panel
    # the chip IS the count: with nothing to report there is no chip
    assert "noticesBtn.hidden = !n;" in APP_JS.read_text()


def test_the_rail_holds_what_only_the_rail_can():
    """Pinned, this session's writes, the files, the chats, the run: compartments
    none of them stated anywhere else in the app. Layout and Areas stood here too
    and no longer do -- both are readings of ONE view, and a control for the
    picture belongs against the picture (#legend, inside #view-graph)."""
    rail = _block(INDEX.read_text(), '<aside id="sidebar">', "</aside>")
    order = [m for m in re.findall(r'id="(side-[a-z]+)"', rail)]
    # side-resize is the drag handle, not a compartment; side-search reads
    # INSIDE side-files, because the filter narrows that tree and floating on
    # its own between two compartments it was a box with no stated subject.
    assert order == ["side-resize", "side-pinned", "side-changes",
                     "side-files", "side-search", "side-history",
                     "side-work"], order
    assert 'id="pinned"' in rail
    # ...and the two that left are in the legend, once each
    view = _block(INDEX.read_text(), '<section id="view-graph"', "</section>")
    assert 'id="layout-modes"' in view and 'id="areas"' in view


def test_the_strip_is_one_bar_and_not_six_compartments():
    """Five vertical rules used to divide the strip (brand, every tab, the
    zones, and the two edge buttons). What separates a fact from the next one is
    now the space around it, which only works in a bar tall enough to have any:
    the height and the absence of the rules are one change, not two."""
    css = APP_CSS.read_text()
    header = css[css.index("header {"):css.index("#tree {")]
    assert "min-height: 46px" in header
    # the bar keeps its own bottom edge; what is gone is every rule INSIDE it
    assert header.count("border-bottom: 1px solid var(--line-2)") == 1
    for rule in ("border-right: 1px solid var(--line);",
                 "border-left: 1px solid var(--line);"):
        assert rule not in header, f"a divider is back in the strip: {rule}"
    # and the tabs run the full height, or the active accent floats above the edge
    assert ".tabs {\n  align-items: stretch;" in css


def test_a_pin_is_scoped_to_the_vault_it_was_made_in():
    """Switching vaults must not carry one vault's pins into another, so the
    storage key carries the path and a /vault switch re-reads it."""
    src = APP_JS.read_text()
    assert 'return "pinned:" + ($("#top-vname").title || "");' in src
    assert "if (path && path !== was) loadPins();" in src


def test_the_strip_and_the_panel_say_the_same_word():
    """Both are painted from one projectRun() call and one computed word. Two
    derivations would let the strip read idle while the panel beside it fills."""
    src = WORK_JS.read_text()
    assert src.count('run.running ? "live"') == 1
    assert "runState.textContent = state;" in src
    assert "paintTopRun(run, state);" in src


def test_no_two_functions_in_app_js_share_a_name():
    """A second `function renderAreas(...)` in the same file is not an error in
    JavaScript: the later declaration is hoisted over the earlier one and every
    call in the file silently reaches the wrong body. That is how the rail's
    area spectrum shipped invisible for one commit -- it collided with the shape
    view's area-coupling matrix, and nothing in the console said so."""
    src = APP_JS.read_text()
    names = re.findall(r"^(?:async )?function ([A-Za-z_$][\w$]*)\(", src, re.M)
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"declared twice at top level in app.js: {dupes}"
