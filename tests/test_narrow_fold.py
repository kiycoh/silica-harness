# tests/test_narrow_fold.py
"""The 1120 floor: below it the deck FOLDS, it does not reflow.

The placeholder this replaces was a deletion: under the fit, work.js took the
work panel away entirely, and the rail went with it behind the header toggle.
A narrow window is the case where "what is the agent doing" matters most, so
the fold has to leave both surfaces reachable -- the rail as five icons that
summon it, the run as a mode of the right sidebar, which has been an overlay
with its own close control at every width since it existed.

These tests pin the two things that rot: the threshold existing in one place,
and the icons standing for compartments that are actually there.
"""
from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "silica" / "ui" / "web" / "static"
INDEX = WEB / "index.html"
from tests.webassets import app_css, app_js
WORK_JS = WEB / "work.js"


def test_the_fold_threshold_is_not_duplicated_across_the_two_scripts():
    """work.js used to carry WORK_W/MIN_PROSE/SIDE_W and the stylesheet carried the
    same widths. Two constants that must match are two constants that eventually
    do not, so the threshold is the media query and both scripts ask it."""
    css, app, work = app_css(), app_js(), WORK_JS.read_text()
    assert len(re.findall(r"max-width:\s*1195px", css)) == 1
    assert "1195" not in app and "1195" not in work
    # the old pixel arithmetic is gone, not merely unused
    for dead in ("MIN_PROSE", "SIDE_W", "WORK_W", "function affordable("):
        assert dead not in work, f"work.js still computes the fit with {dead}"
    # work.js used to ask app.js for it through window.isNarrow; the sidebar it
    # is a mode of now answers the width question for it, so the question and the
    # export that carried it are both gone rather than left dangling
    assert "isNarrow" not in work
    assert app.count("function isNarrow(") == 1 and "window.isNarrow" not in app


def test_the_run_needs_no_fold_of_its_own():
    """It used to be a second panel at the right edge with a fold of its own --
    an overlay, an inset, a close button, and a rule teaching it to yield to the
    note drawer at that same edge. It is a MODE of that drawer now, so it folds
    the way the drawer does and the three selectors that arbitrated between them
    are gone. The two things the fold has to keep are still here: it is
    dismissable from itself, and it reserves its inset rather than covering
    #send, which is the bug the drawer already had to fix once."""
    html, css, app, work = (INDEX.read_text(), app_css(),
                            app_js(), WORK_JS.read_text())
    assert 'id="note-close"' in html, "an overlay must be dismissable from itself"
    assert "max-width: 55vw" in css, "the sidebar does not fold against the viewport"
    assert "body.note-open #view-chat" in css
    rules = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for dead in ("work-fold", "--work-w", "#work-close"):
        assert dead not in rules, f"the stylesheet still folds a second panel via {dead}"
    assert "work-fold" not in work and "work-fold" not in app
    # There is no preference left to keep in step: the run moved to a rail
    # compartment, where <details open> in the markup IS the restore. app.js
    # still owns whether the rail is out, and reads the fold off the DOM.
    assert "work-open" not in app and "work-open" not in work


def test_the_composer_hint_is_a_clause_the_fitter_can_drop():
    """Chromium drops text-overflow on a <textarea> placeholder, so a squeezed
    field hard-cuts mid-word and fitPlaceholder() shortens the string instead.
    It shortens by stripping the trailing parenthetical, which silently does
    nothing the day someone rewrites the hint without one -- and the failure is
    the old clip, which looks like a rendering artefact rather than a rewrite."""
    html, app = INDEX.read_text(), app_js()
    assert "function fitPlaceholder(" in app
    hints = re.findall(r'<textarea[^>]*placeholder="([^"]*)"', html)
    assert hints, "no composer placeholder in index.html"
    for hint in hints:
        stem = re.sub(r"\s*\([^()]*\)\s*$", "", hint)
        assert stem and stem != hint, f"{hint!r} has no clause for the fitter to drop"


def test_every_view_but_the_graph_reserves_the_drawer_gap():
    """A view that does not reserve the gap renders UNDER the overlay, and the
    part that goes missing is silent: the calendar lost its seventh column, its
    agenda rail and its month/week toggle, and a month grid that stops at
    Saturday still looks like a month grid. #view-graph is the one exception on
    purpose -- it hides its own HUD instead, see syncDrawerToViews."""
    html, css = INDEX.read_text(), app_css()
    views = set(re.findall(r'id="(view-[a-z]+)"', html)) - {"view-graph"}
    assert views, "no views found in index.html"
    # one rule, because there is one overlay: the work panel had a second inset
    # of its own and a third for the case where both claimed the edge at once
    for view in sorted(views):
        assert f"body.note-open #{view}" in css, f"the sidebar does not inset #{view}"


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
    app = app_js()
    assert "function syncRailIcons(" in app
    assert 'attributeFilter: ["hidden"]' in app


def test_the_spine_ends_at_the_running_beat():
    """The deck drew 62% and the port shipped that literal, so the rule painted
    the same two thirds whatever the run was doing. It is the run's own progress
    bar; measured, not counted, because a thought row is two lines and a tool
    row is one."""
    css, work = app_css(), WORK_JS.read_text()
    assert "var(--spine, 62%)" in css
    assert re.search(r"linear-gradient\(var\(--accent\) 0 62%", css) is None
    spine = work[work.index("function paintSpine("):work.index("// --- the Report panel")]
    assert ".wk-beat.running" in spine
    assert 'setProperty("--spine"' in spine
    # and it is cleared, not left pointing at a beat that finished
    assert 'removeProperty("--spine")' in spine
