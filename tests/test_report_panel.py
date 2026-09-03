# tests/test_report_panel.py
"""The Report panel is a STORE before it is a surface.

Every other number the metrics view shows is derived from the report in front
of it. "Since the last report" is the one that cannot be: it needs a previous
reading, and until now nothing kept one, which is why this lane did not land
with the rest of Metrics.

These tests pin the three things that decide whether the delta is true:
what gets stored (only signals that mean the same at both report depths), when
a line is appended (movement, not "the tab was opened again"), and which
reading is handed back to diff against.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from silica.kernel.report.history import (
    SIGNALS,
    history_path,
    read_history,
    record_report,
    signals_of,
)

WEB = Path(__file__).resolve().parents[1] / "silica" / "ui" / "web" / "static"
WORK_JS = WEB / "work.js"
from tests.webassets import app_js
INDEX = WEB / "index.html"


def _sig(**over) -> dict[str, int]:
    base = dict.fromkeys(SIGNALS, 0)
    base.update(over)
    return base


# --- the store ---------------------------------------------------------------

def test_the_first_report_has_nothing_to_diff_against(tmp_path):
    """A delta invented against zero would read as if the whole vault had just
    been written in one go, which is the opposite of what the panel says."""
    assert record_report(tmp_path, _sig(notes=690, unresolved=566)) is None
    assert len(read_history(tmp_path)) == 1


def test_a_line_lands_only_when_something_moved(tmp_path):
    """Opening the metrics tab is not a data point. The view recomputes on every
    open, and a series that grew per open would make an untouched vault look
    busy and push the real previous reading out of reach."""
    a = _sig(notes=690, unresolved=566)
    record_report(tmp_path, a)
    record_report(tmp_path, a)
    record_report(tmp_path, a)
    assert len(read_history(tmp_path)) == 1

    record_report(tmp_path, _sig(notes=690, unresolved=532))
    assert len(read_history(tmp_path)) == 2


def test_the_previous_reading_is_the_last_DIFFERENT_one(tmp_path):
    """Not "the last line". Once a vault stops changing, the head equals the
    present, and diffing it against itself blanks the panel on every second
    open -- the reading a person wants ("532, was 566") has to survive until
    the vault actually moves again."""
    record_report(tmp_path, _sig(notes=690, unresolved=566))
    prev = record_report(tmp_path, _sig(notes=690, unresolved=532))
    assert prev["signals"]["unresolved"] == 566

    # same counts again: nothing appended, and 566 is still the answer
    again = record_report(tmp_path, _sig(notes=690, unresolved=532))
    assert again["signals"]["unresolved"] == 566
    assert len(read_history(tmp_path)) == 2


def test_a_torn_last_line_is_skipped_not_raised(tmp_path):
    """Same rule quiz.jsonl reads by: a half-written line costs the reading it
    held, never the whole series."""
    record_report(tmp_path, _sig(notes=690, unresolved=566))
    with history_path(tmp_path).open("ab") as fh:
        fh.write(b'{"at": "2026-08-2')
    assert len(read_history(tmp_path)) == 1
    assert record_report(tmp_path, _sig(notes=691, unresolved=560)) is not None


def test_only_depth_independent_signals_are_stored():
    """`integration_deficits` and the autolink family are zero unless the
    co-occurrence leg ran, so a structural report diffed against a full one
    would report 137 deficits "closed" for work nobody did. Same trap
    vault_energy documents for E(vault); the way out is to never store them."""
    totals = {"notes": 690, "unresolved": 532, "integration_deficits": 137,
              "autolink_candidates": 88, "missing_links": 40, "clusters": 210}
    sig = signals_of(totals, areas=90)
    assert "integration_deficits" not in sig
    assert "autolink_candidates" not in sig
    assert "missing_links" not in sig
    # and an area is a cluster of more than one note, which is why it is passed
    # in rather than read off totals["clusters"]
    assert sig["areas"] == 90 and "clusters" not in sig


# --- the projection ----------------------------------------------------------

def _pure_block() -> str:
    src = WORK_JS.read_text()
    m = re.search(r"// --- projection:begin.*?// --- projection:end", src, re.S)
    assert m, "projection markers not found in work.js"
    return m.group(0)


def _report(tmp_path, payload: dict) -> dict:
    script = tmp_path / "report.js"
    script.write_text(
        _pure_block()
        + "\nconsole.log(JSON.stringify(projectReport(JSON.parse(process.argv[2]))));\n"
    )
    out = subprocess.run(["node", str(script), json.dumps(payload)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


PAYLOAD = {
    "depth": "full",
    "report": {
        "at": "2026-08-22T13:18:04+00:00", "elapsed_s": 4.2, "notes": 690,
        "depth": "full", "since": "2026-08-06T09:00:00",
        "signals": {"notes": 690, "links": 1400, "unresolved": 532,
                    "dangling_links": 300, "orphans": 63, "lean_notes": 254,
                    "contested": 2, "structural_gaps": 5, "areas": 90},
        "previous": {"notes": 688, "links": 1380, "unresolved": 566,
                     "dangling_links": 320, "orphans": 58, "lean_notes": 254,
                     "contested": 2, "structural_gaps": 5, "areas": 88},
    },
}


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_a_fall_and_a_rise_are_not_the_same_word(tmp_path):
    """"34 closed" and "34 new" are the same integer and opposite news. The
    panel states the word, not a signed number the reader has to interpret."""
    r = _report(tmp_path, PAYLOAD)
    rows = {row["key"]: row for row in r["rows"]}
    assert rows["unresolved"]["sub"] == "was 566 · 34 closed"
    assert rows["unresolved"]["delta"] == -34
    assert rows["orphans"]["sub"] == "was 58 · 5 new notes nothing links to"
    assert rows["lean_notes"]["sub"] == "was 254 · unchanged"
    assert rows["areas"]["sub"] == "was 88 · 2 new communities"
    # the strip's "areas" counts every community and this one drops the
    # singletons, so the row says which of the two it is
    assert rows["areas"]["label"] == "areas > 1 note"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_a_delta_of_one_says_it_in_the_singular(tmp_path):
    """"1 communities merged" is the kind of sentence that makes a reader stop
    trusting the number beside it."""
    prev = {**PAYLOAD["report"]["signals"], "areas": 91, "orphans": 62, "lean_notes": 253}
    r = _report(tmp_path, {"report": {**PAYLOAD["report"], "previous": prev}})
    rows = {row["key"]: row for row in r["rows"]}
    assert rows["areas"]["sub"] == "was 91 · 1 community merged"
    assert rows["orphans"]["sub"] == "was 62 · 1 new note nothing links to"
    assert rows["lean_notes"]["sub"] == "was 253 · 1 new thin note"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_growth_is_neither_good_nor_bad(tmp_path):
    """Colouring `notes` and `areas` would congratulate or scold the user for
    writing. Only the three signals that name something out of place carry a
    direction."""
    r = _report(tmp_path, PAYLOAD)
    good = {row["key"]: row["good"] for row in r["rows"]}
    assert good["unresolved"] is True and good["orphans"] is False
    assert good["notes"] is None and good["areas"] is None
    assert good["lean_notes"] is None  # unchanged is not an improvement


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_with_no_previous_reading_every_row_says_so(tmp_path):
    r = _report(tmp_path, {"report": {**PAYLOAD["report"], "previous": None, "since": None}})
    assert {row["sub"] for row in r["rows"]} == {"first reading"}
    assert all(row["delta"] is None for row in r["rows"])
    assert r["since"] is None


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_the_head_names_the_depth_and_that_no_model_ran(tmp_path):
    """The two depths are not the same report -- the co-occurrence leg is what
    separates them and it is the expensive one -- and "local, no model" is said
    out loud because it is what people assume otherwise."""
    r = _report(tmp_path, PAYLOAD)
    assert r["title"] == "Structural audit · full report"
    assert r["meta"] == ["4.2s", "690 notes", "local, no model"]
    shallow = _report(tmp_path, {"report": {**PAYLOAD["report"], "depth": "structural"}})
    assert shallow["title"] == "Structural audit"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_a_report_served_from_the_memo_states_no_duration(tmp_path):
    """compute_report memoises per vault epoch, so re-opening the tab costs
    microseconds. "0.0s" there would read as "the audit is free" rather than
    "this is the one from two minutes ago", and the header's own clock already
    says when it was made."""
    r = _report(tmp_path, {"report": {**PAYLOAD["report"], "elapsed_s": None}})
    assert r["meta"] == ["690 notes", "local, no model"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run work.js")
def test_an_empty_payload_projects_to_an_empty_report(tmp_path):
    """The panel renders before the first /metrics lands and after one fails."""
    r = _report(tmp_path, {})
    assert r["rows"] == [] and r["since"] is None


# --- the wiring --------------------------------------------------------------

def test_the_report_rides_the_metrics_fetch(tmp_path):
    """One fetch, two surfaces. A second /metrics from work.js would recompute
    the whole vault to paint a panel beside the view that just computed it."""
    src = app_js()
    assert 'CustomEvent("silica:report"' in src
    assert 'CustomEvent("silica:view"' in src
    assert 'silica:report' in WORK_JS.read_text()
    assert '"/metrics' not in WORK_JS.read_text()


def test_the_two_bulk_turns_are_written_once():
    """The Report panel offers the same two turns the evidence panes do. A write
    prompt stated twice is two turns that can drift, and the one people would
    notice is the one that writes twenty notes."""
    src = app_js()
    assert src.count("function bulkWritePrompt(") == 1
    assert src.count("function bulkAutolinkPrompt(") == 1
    assert src.count("most referenced ones that do not exist yet") == 1
    assert src.count("have nothing linking to them") == 1
    # and work.js calls them rather than carrying its own copy
    work = WORK_JS.read_text()
    assert "window.bulkWritePrompt(" in work and "window.bulkAutolinkPrompt(" in work


def test_a_suggested_action_drafts_the_turn_it_never_sends_it():
    """Same contract as every action in the metrics view: the vault is changed
    by a turn you sent, never by a panel deciding on your behalf."""
    work = WORK_JS.read_text()
    card = work[work.index("function suggestedCard("):work.index("function renderReport(")]
    assert "window.prefillChat(" in card
    for sends in ("send(", "fetch(", "/chat"):
        assert sends not in card, f"the suggested card {sends} on its own"


# --- the endpoint ------------------------------------------------------------

def test_metrics_files_the_report_and_hands_back_yesterdays(tmp_vault):
    """End to end: the panel's whole reason to exist is a number the view in
    front of it cannot derive, so the seam that has to hold is /metrics filing
    this report and returning the last different one with it.

    The earlier reading is written straight into the store rather than produced
    by a second call, because compute_report memoises per vault epoch and the
    driver caches the file list: two calls in one test see the same vault, which
    is correct behaviour and useless for a delta.
    """
    from fastapi.testclient import TestClient

    from silica.config import CONFIG
    from silica.ui.web import server

    tmp_vault.note("A.md", "# A\n\nlinks to [[B]] and [[Ghost]]\n")
    tmp_vault.note("B.md", "# B\n\nback to [[A]]\n")
    yesterday = _sig(notes=1, unresolved=4, orphans=2)
    record_report(CONFIG.vault_path, yesterday, at="2026-08-06T09:00:00")

    payload = TestClient(server.app).get("/metrics").json()
    assert "error" not in payload, payload.get("error")
    head = payload["report"]
    assert head["signals"]["notes"] == 2
    assert head["previous"] == yesterday
    assert head["since"] == "2026-08-06T09:00:00"
    assert head["depth"] == "structural" and head["elapsed_s"] >= 0

    # and this report is now on file for the next one
    stored = read_history(CONFIG.vault_path)
    assert [r["signals"]["notes"] for r in stored] == [1, 2]


def test_a_memo_hit_does_not_claim_a_duration_it_did_not_spend(tmp_vault):
    """The second call inside one vault epoch is a dict lookup. It is still a
    true report and still worth showing; what it is not is a two-second audit."""
    from fastapi.testclient import TestClient

    from silica.ui.web import server

    tmp_vault.note("A.md", "# A\n\nto [[B]]\n")
    client = TestClient(server.app)
    first = client.get("/metrics").json()["report"]
    second = client.get("/metrics").json()["report"]
    assert first["elapsed_s"] is not None
    assert second["elapsed_s"] is None
    assert second["at"] == first["at"]      # same report, not a new one


def test_metrics_survives_a_vault_it_cannot_write_to(tmp_vault, monkeypatch):
    """Losing the delta costs a section of one panel. Failing the call costs the
    whole view, which is why the store is best-effort at the call site."""
    from fastapi.testclient import TestClient

    from silica.ui.web import server

    tmp_vault.note("A.md", "# A\n")
    monkeypatch.setattr("silica.kernel.report.history.record_report",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only vault")))
    payload = TestClient(server.app).get("/metrics").json()
    assert "error" not in payload
    assert payload["report"]["previous"] is None
    assert payload["report"]["signals"]["notes"] == 1
