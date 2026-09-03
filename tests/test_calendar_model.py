"""Event-note parsing: tolerant on read, strict on write (spec-calendar-events).

The contract under test: a note IS an Event note iff frontmatter has
`event_start`; PyYAML types that key three ways (date / str / datetime) and
all three must normalize; malformed start skips the note, but an invalid
rrule DEGRADES to a one-shot (the start is valid — vanishing a visible note
over one bad key is silent data loss); write-side validation is strict.
"""
from __future__ import annotations

import datetime as dt

from silica.kernel.calendar.model import (
    parse_event,
    validate_event,
    scan_events,
    event_rows,
)


def _ev(data, stem="dentist", path="calendar/dentist.md"):
    return parse_event(data, stem=stem, path=path)


# --- parse: the three YAML types of event_start -----------------------------

def test_date_only_yaml_date_object_is_all_day():
    e = _ev({"event_start": dt.date(2026, 8, 20)})
    assert e.all_day is True
    assert e.start == dt.datetime(2026, 8, 20, 0, 0)


def test_date_only_string_is_all_day():
    e = _ev({"event_start": "2026-08-20"})
    assert e.all_day is True
    assert e.start == dt.datetime(2026, 8, 20, 0, 0)


def test_no_seconds_string_is_timed():
    # PyYAML leaves "2026-08-20 15:00" as str (its resolver wants seconds)
    e = _ev({"event_start": "2026-08-20 15:00"})
    assert e.all_day is False
    assert e.start == dt.datetime(2026, 8, 20, 15, 0)


def test_yaml_datetime_object_is_timed():
    e = _ev({"event_start": dt.datetime(2026, 8, 20, 15, 0)})
    assert e.all_day is False
    assert e.start == dt.datetime(2026, 8, 20, 15, 0)


# --- parse: identity and fallbacks ------------------------------------------

def test_note_without_event_start_is_not_an_event():
    assert _ev({"title": "just a note"}) is None


def test_malformed_start_skips_the_note():
    assert _ev({"event_start": "not a date"}) is None


def test_title_falls_back_to_stem():
    e = _ev({"event_start": "2026-08-20 15:00"}, stem="dentist")
    assert e.title == "dentist"
    e2 = _ev({"event_start": "2026-08-20 15:00", "title": "Dentist"})
    assert e2.title == "Dentist"


# --- parse: end normalization (internal bound is EXCLUSIVE) -----------------

def test_all_day_end_is_inclusive():
    # start 20th, end 22nd = 3 days; internally end is the 23rd at midnight
    e = _ev({"event_start": "2026-08-20", "event_end": "2026-08-22"})
    assert e.end == dt.datetime(2026, 8, 23, 0, 0)


def test_all_day_without_end_covers_one_day():
    e = _ev({"event_start": "2026-08-20"})
    assert e.end == dt.datetime(2026, 8, 21, 0, 0)


def test_timed_end_kept_as_is():
    e = _ev({"event_start": "2026-08-20 15:00", "event_end": "2026-08-20 16:00"})
    assert e.end == dt.datetime(2026, 8, 20, 16, 0)


def test_timed_without_end_is_an_instant():
    e = _ev({"event_start": "2026-08-20 15:00"})
    assert e.end is None


def test_end_before_start_is_dropped_on_read():
    e = _ev({"event_start": "2026-08-20 15:00", "event_end": "2026-08-20 14:00"})
    assert e is not None
    assert e.end is None


# --- parse: rrule degrades, never vanishes ----------------------------------

def test_invalid_rrule_degrades_to_one_shot():
    e = _ev({"event_start": "2026-08-20 15:00", "event_rrule": "FREQ=BOGUS"})
    assert e is not None
    assert e.rrule is None


def test_valid_rrule_kept():
    e = _ev({"event_start": "2026-08-20 15:00", "event_rrule": "FREQ=WEEKLY;BYDAY=WE"})
    assert e.rrule == "FREQ=WEEKLY;BYDAY=WE"


# --- parse: reminder lead ----------------------------------------------------

def test_reminder_lead_grammar():
    assert _ev({"event_start": "2026-08-20 15:00", "event_reminder": "30m"}).reminder == dt.timedelta(minutes=30)
    assert _ev({"event_start": "2026-08-20 15:00", "event_reminder": "2h"}).reminder == dt.timedelta(hours=2)
    assert _ev({"event_start": "2026-08-20 15:00", "event_reminder": "1d"}).reminder == dt.timedelta(days=1)
    assert _ev({"event_start": "2026-08-20 15:00", "event_reminder": "0m"}).reminder == dt.timedelta(0)


def test_reminder_casefolded_on_read_invalid_dropped():
    assert _ev({"event_start": "2026-08-20 15:00", "event_reminder": "30M"}).reminder == dt.timedelta(minutes=30)
    assert _ev({"event_start": "2026-08-20 15:00", "event_reminder": "soon"}).reminder is None


# --- parse: status -----------------------------------------------------------

def test_status_values():
    assert _ev({"event_start": "2026-08-20"}).status == ""
    assert _ev({"event_start": "2026-08-20", "event_status": "done"}).status == "done"
    assert _ev({"event_start": "2026-08-20", "event_status": "Cancelled"}).status == "cancelled"
    # unknown value reads as open, never crashes
    assert _ev({"event_start": "2026-08-20", "event_status": "maybe"}).status == ""


# --- strict write-side validation --------------------------------------------

def test_validate_accepts_a_full_valid_event():
    errs = validate_event({
        "event_start": "2026-08-20 15:00",
        "event_end": "2026-08-20 16:00",
        "event_rrule": "FREQ=WEEKLY;BYDAY=WE",
        "event_reminder": "30m",
    })
    assert errs == []


def test_validate_rejects_each_defect():
    assert validate_event({"event_start": "nope"})
    assert validate_event({"event_start": "2026-08-20 15:00", "event_end": "2026-08-20 14:00"})
    assert validate_event({"event_start": "2026-08-20 15:00", "event_rrule": "FREQ=BOGUS"})
    assert validate_event({"event_start": "2026-08-20 15:00", "event_reminder": "soon"})
    # strict is strict: uppercase lead is a write-side error, read-side tolerance only
    assert validate_event({"event_start": "2026-08-20 15:00", "event_reminder": "30M"})


def test_validate_rejects_bad_status():
    assert validate_event({"event_start": "2026-08-20", "event_status": "maybe"})
    assert validate_event({"event_start": "2026-08-20", "event_status": "done"}) == []


# --- scan_events: walk parity with timeline ----------------------------------

def _write(vault, rel, text):
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


EVENT_NOTE = "---\nevent_start: 2026-08-20 15:00\n---\nbody\n"


def test_scan_finds_events_anywhere_and_skips_non_events(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write(vault, "calendar/dentist.md", EVENT_NOTE)
    _write(vault, "Projects/deadline.md", "---\nevent_start: 2026-09-01\n---\n")
    _write(vault, "Projects/plain.md", "---\ntags: [x]\n---\nno event\n")
    stems = {e.stem for e in scan_events(vault)}
    assert stems == {"dentist", "deadline"}


def test_scan_skips_dot_dirs_and_sources(tmp_path):
    from silica.kernel.recall.paths import SOURCES_DIR
    vault = tmp_path / "vault"
    vault.mkdir()
    _write(vault, ".trash/ghost.md", EVENT_NOTE)
    _write(vault, f"{SOURCES_DIR}/leaf.md", EVENT_NOTE)
    _write(vault, "ok.md", EVENT_NOTE)
    assert [e.stem for e in scan_events(vault)] == ["ok"]


def test_scan_skips_malformed_event_note_without_crashing(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write(vault, "bad.md", "---\nevent_start: [not, a, date]\n---\n")
    _write(vault, "ok.md", EVENT_NOTE)
    assert [e.stem for e in scan_events(vault)] == ["ok"]


def test_scan_memoized_on_vault_epoch(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write(vault, "ok.md", EVENT_NOTE)

    monkeypatch.setattr("silica.kernel.recall.paths.vault_epoch", lambda v=None: "epoch-1")
    first = scan_events(vault)
    _write(vault, "later.md", EVENT_NOTE)
    assert scan_events(vault) is first  # same epoch: memo hit, no rescan
    monkeypatch.setattr("silica.kernel.recall.paths.vault_epoch", lambda v=None: "epoch-2")
    assert {e.stem for e in scan_events(vault)} == {"ok", "later"}


# --- BI flattener contract ----------------------------------------------------

def test_event_rows_columns_and_display_semantics(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write(vault, "calendar/trip.md",
           "---\ntitle: Trip\nevent_start: 2026-08-20\nevent_end: 2026-08-22\n---\n")
    _write(vault, "calendar/dentist.md",
           "---\nevent_start: 2026-08-20 15:00\nevent_end: 2026-08-20 16:00\n"
           "event_reminder: 30m\nevent_status: done\n---\n")
    rows = {r["stem"]: r for r in event_rows(vault)}
    trip, dentist = rows["trip"], rows["dentist"]
    assert set(trip) == {"stem", "title", "start", "end", "all_day", "rrule",
                         "reminder", "status", "folder"}
    # all-day rows echo the user-facing INCLUSIVE dates, not the internal bound
    assert trip["start"] == "2026-08-20" and trip["end"] == "2026-08-22"
    assert trip["all_day"] is True and trip["folder"] == "calendar"
    assert dentist["start"] == "2026-08-20 15:00" and dentist["end"] == "2026-08-20 16:00"
    assert dentist["reminder"] == "30m" and dentist["status"] == "done"
