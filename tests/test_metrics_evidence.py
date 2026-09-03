# tests/test_metrics_evidence.py
"""The metrics view is a worklist you steer with and the evidence beside it.

It used to be three walls of tables: 63 orphans and 532 unresolved targets
printed as rows, in cards you had to go find, with nothing to do about any of
them from the view that found them. The worklist is the navigation now, the
pane beside it is that row's evidence, and the action sits on the row that
carries it.

Two things here can rot in silence. A signal added to the worklist with no
evidence builder falls through to a pane that says "nothing to act on" and
nothing raises. And the unresolved reading ("16% of the references point at 20
targets, 458 of the rest are asked for once") is a statement about ALL 532
targets while the table shows twelve: computed from the slice it would be a
confident sentence about the wrong population.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.webassets import app_css, app_js


@pytest.fixture
def client(tmp_vault, tmp_path, monkeypatch):
    """Fresh module-level session per test, backed by a tmp fs vault."""
    from silica.ui.web import server

    server._reset_session()
    return TestClient(server.app), server


def test_every_worklist_signal_has_an_evidence_pane():
    src = app_js()
    block = re.search(r"^  const signals = \[(.*?)^  \]\.sort", src, re.S | re.M)
    assert block, "the worklist's signal table moved"
    listed = re.findall(r'^\s*\["(\w+)",', block.group(1), re.M)
    assert len(listed) >= 7, listed

    sw = re.search(r"^function evidence\(key.*?^}", src, re.S | re.M)
    assert sw, "evidence() moved"
    handled = set(re.findall(r'case "(\w+)":', sw.group(0)))
    missing = sorted(set(listed) - handled)
    assert not missing, f"worklist signals with no evidence pane: {missing}"
    extra = sorted(handled - set(listed))
    assert not extra, f"evidence panes no worklist row can reach: {extra}"


def test_the_absorbed_cards_are_gone_from_the_sections():
    """A signal stated twice is two places that can disagree about the same
    vault, and the second one is the wall of tables this pass removed."""
    src = app_js()
    for gone in ("Maintenance", "Orphans", "Unresolved links", "Attention",
                 "Structural gaps", "Contested", "Integration deficits"):
        assert f'mCard("{gone}"' not in src, f"{gone} is a card again as well as a pane"


def test_the_unresolved_reading_is_computed_over_every_target(client, tmp_vault):
    """`dangling` is a twelve-row slice; the tail and the concentration are the
    whole distribution, so they ride the payload as their own fields."""
    tc, _server = client
    # One target three notes ask for, one only a single note asks for: two bins,
    # which is the smallest shape the reading can be checked on.
    tmp_vault.note("A.md", "[[Wanted]] and [[Once]]")
    tmp_vault.note("B.md", "[[Wanted]]")
    tmp_vault.note("C.md", "[[Wanted]]")

    d = tc.get("/metrics").json()
    hist = {h["refs"]: h["targets"] for h in d["dangling_hist"]}
    assert hist == {1: 1, 3: 1}, d["dangling_hist"]
    # the bins account for every target, not just the ones on screen
    assert sum(hist.values()) == d["totals"]["dangling_links"]
    # and the references they carry account for every reference
    assert sum(r * n for r, n in hist.items()) == d["totals"]["unresolved"]
    assert d["dangling_top_refs"] == 4


def test_a_missing_target_names_the_notes_that_ask_for_it(client, tmp_vault):
    """"Referenced from" is what decides whether a target is worth writing:
    three notes asking for it is a gap, one is probably a typo."""
    tc, _server = client
    for name in ("A", "B", "C", "D"):
        tmp_vault.note(f"{name}.md", "[[Wanted]]")

    d = tc.get("/metrics").json()
    row = next(x for x in d["dangling"] if x["target"] == "Wanted")
    assert row["refs"] == 4
    # three names and a count: the fourth is what starts wrapping the column
    assert len(row["from"]) == 3
    assert row["from_more"] == 1
    assert {n.removesuffix(".md") for n in row["from"]} <= {"A", "B", "C", "D"}


def test_the_column_chart_is_not_wrapped_in_the_tail_chart_s_box():
    """Two charts, one class name. `.hist` is the dangling tail's own box and
    carries `height: 44px`; the degree histogram is a 120px track plus a cap
    band plus an axis band, and it shipped inside a div.hist -- 150px of chart
    in a 44px box, overflowing the Link distribution card and painting over the
    line under it. The renderer returns its .hist-plot directly now."""
    src = app_js()
    body = re.search(r"^function histogram\(bins\) \{(.*?)^\}", src, re.S | re.M)
    assert body, "histogram() moved"
    assert 'mkEl("div", "hist")' not in body.group(1)
    assert "return plot;" in body.group(1)

    css = app_css()
    # and the box that made it a bug is still the fixed one, so this stays real
    assert re.search(r"^\.hist \{[^}]*height: 44px;", css, re.S | re.M)
