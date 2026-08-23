# tests/test_narrow_fold.py
"""The 1120 floor: below it the deck FOLDS, it does not reflow.

The placeholder this replaces was a deletion: under the fit, work.js took the
work panel away entirely, and the rail went with it behind the header toggle.
A narrow window is the case where "what is the agent doing" matters most, so
the fold has to leave both surfaces reachable -- the rail as five icons that
summon it, the panel as an overlay with its own close control.

These tests pin the two things that rot: the threshold existing in one place,
and the icons standing for compartments that are actually there.
"""
from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "silica" / "ui" / "web" / "static"
INDEX = WEB / "index.html"
APP_JS = WEB / "app.js"
APP_CSS = WEB / "app.css"
WORK_JS = WEB / "work.js"


def test_the_fold_threshold_is_not_duplicated_across_the_two_scripts():
    """work.js used to carry WORK_W/MIN_PROSE/SIDE_W and app.css carried the
    same widths. Two constants that must match are two constants that eventually
    do not, so the threshold is the media query and both scripts ask it."""
    css, app, work = APP_CSS.read_text(), APP_JS.read_text(), WORK_JS.read_text()
    assert len(re.findall(r"max-width:\s*1195px", css)) == 1
    assert "1195" not in app and "1195" not in work
    # the old pixel arithmetic is gone, not merely unused
    for dead in ("MIN_PROSE", "SIDE_W", "WORK_W", "function affordable("):
        assert dead not in work, f"work.js still computes the fit with {dead}"
    assert work.count("window.isNarrow") >= 1
    assert app.count("function isNarrow(") == 1


def test_the_panel_folds_instead_of_disappearing():
    html, css, work = INDEX.read_text(), APP_CSS.read_text(), WORK_JS.read_text()
    assert 'id="work-close"' in html, "an overlay must be dismissable from itself"
    assert "body.work-fold #work" in css and "body.work-fold #work-close" in css
    assert 'classList.toggle("work-fold"' in work
    # the preference still decides; the width only decides HOW it is shown
    assert 'localStorage.setItem("work-open"' in work
    # and it reserves its inset rather than covering #send, which is the bug the
    # note drawer already had to fix once
    assert "body.work-fold #view-chat" in css and "padding-right: var(--work-w)" in css


def test_every_rail_icon_stands_for_a_compartment_that_exists():
    """data-sec is the id of the <details> the icon opens. A renamed section
    leaves a button that opens nothing, and nothing about that is an error --
    the click just does not land."""
    html = INDEX.read_text()
    rail = html[html.index('<aside id="railmini"'):html.index('<aside id="sidebar">')]
    secs = re.findall(r'data-sec="([^"]+)"', rail)
    assert secs, "the mini rail has no icons"
    for sec in secs:
        assert f'id="{sec}"' in html, f"#railmini points at {sec}, which is not in the rail"
    # every compartment of the full rail is reachable from the folded one
    sidebar = html[html.index('<aside id="sidebar">'):html.index("</aside>", html.index('<aside id="sidebar">'))]
    for det in re.findall(r'<details class="side-section" id="([^"]+)"', sidebar):
        assert det in secs, f"{det} has no icon and is unreachable when folded"


def test_an_icon_for_a_hidden_compartment_hides_itself():
    """Pinned before the first pin and This session before the first write hide
    themselves; an icon standing for one of them is a button that opens an
    empty drawer. The sections are unhidden from several places, so this watches
    the result rather than adding a call site to each."""
    app = APP_JS.read_text()
    assert "function syncRailIcons(" in app
    assert 'attributeFilter: ["hidden"]' in app


def test_the_spine_ends_at_the_running_beat():
    """The deck drew 62% and the port shipped that literal, so the rule painted
    the same two thirds whatever the run was doing. It is the run's own progress
    bar; measured, not counted, because a thought row is two lines and a tool
    row is one."""
    css, work = APP_CSS.read_text(), WORK_JS.read_text()
    assert "var(--spine, 62%)" in css
    assert re.search(r"linear-gradient\(var\(--accent\) 0 62%", css) is None
    spine = work[work.index("function paintSpine("):work.index("// --- the Report panel")]
    assert ".wk-beat.running" in spine
    assert 'setProperty("--spine"' in spine
    # and it is cleared, not left pointing at a beat that finished
    assert 'removeProperty("--spine")' in spine
