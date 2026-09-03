"""Tests for the expires_at temporal-forgetting linter."""
from __future__ import annotations

import datetime
from silica.kernel.link.linter import check_expires_at


def test_no_expires_at_returns_no_warning():
    data = {"title": "My Note", "tags": ["ai"]}
    warnings = check_expires_at(data)
    assert warnings == []


def test_expires_at_in_future_returns_no_warning():
    future = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()
    data = {"expires_at": future}
    assert check_expires_at(data) == []


def test_expires_at_today_returns_no_warning():
    today = datetime.date.today().isoformat()
    data = {"expires_at": today}
    assert check_expires_at(data) == []


def test_expires_at_in_past_returns_warning():
    past = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    data = {"expires_at": past}
    warnings = check_expires_at(data)
    assert len(warnings) == 1
    assert "expired" in warnings[0].lower()


def test_expires_at_far_past_mentions_date():
    past = "2020-01-01"
    data = {"expires_at": past}
    warnings = check_expires_at(data)
    assert "2020-01-01" in warnings[0]


def test_expires_at_invalid_format_returns_warning():
    data = {"expires_at": "not-a-date"}
    warnings = check_expires_at(data)
    assert len(warnings) == 1
    assert "invalid" in warnings[0].lower() or "expires_at" in warnings[0].lower()


def test_expires_at_none_returns_no_warning():
    data = {"expires_at": None}
    assert check_expires_at(data) == []


def test_check_expires_at_integrated_in_validate_note(tmp_path, monkeypatch):
    """validate_note emits an expires_at warning for an expired note."""
    import datetime

    past = (datetime.date.today() - datetime.timedelta(days=5)).isoformat()
    note_path = tmp_path / "expired_note.md"
    note_path.write_text(
        f"---\ntitle: Expired\nexpires_at: {past}\n---\n\n[[Hub]]\n\nContent here.\n",
        encoding="utf-8",
    )

    from silica.driver.fs_backend import ObsidianFSBackend
    import silica.kernel.link.linter as linter_mod
    monkeypatch.setattr(linter_mod, "DRIVER", ObsidianFSBackend(str(tmp_path)))

    from silica.kernel.link.linter import validate_note
    errors, warnings = validate_note(str(note_path), hub=None, op_type=None)
    assert any("expired" in w.lower() for w in warnings), f"warnings={warnings}"


def test_sentinel_leak_is_a_hard_error(tmp_path, monkeypatch):
    """A ===SILICA-…=== wire delimiter in a note body fails the post-write gate.

    2026-08-15: a garbled body-pass marker (`===SILICA-BILY 5===`) landed
    verbatim in a committed note together with the next note's whole body.
    The lint is the one gate every write path shares, so it hard-errors here.
    """
    note_path = tmp_path / "leaky.md"
    note_path.write_text(
        "---\ntitle: Leaky\n---\n\n[[Hub]]\n\nFine prose.\n\n"
        "===SILICA-BILY 5===\nAnother note's body.\n",
        encoding="utf-8",
    )

    from silica.driver.fs_backend import ObsidianFSBackend
    import silica.kernel.link.linter as linter_mod
    monkeypatch.setattr(linter_mod, "DRIVER", ObsidianFSBackend(str(tmp_path)))

    from silica.kernel.link.linter import validate_note
    errors, _ = validate_note(str(note_path), hub=None, op_type=None)
    assert any("delimiter" in e.lower() for e in errors), f"errors={errors}"


def test_clean_note_has_no_sentinel_error(tmp_path, monkeypatch):
    note_path = tmp_path / "clean.md"
    note_path.write_text(
        "---\ntitle: Clean\n---\n\n[[Hub]]\n\nProse mentioning === separators "
        "and even the word SILICA inline is fine.\n",
        encoding="utf-8",
    )
    from silica.driver.fs_backend import ObsidianFSBackend
    import silica.kernel.link.linter as linter_mod
    monkeypatch.setattr(linter_mod, "DRIVER", ObsidianFSBackend(str(tmp_path)))

    from silica.kernel.link.linter import validate_note
    errors, _ = validate_note(str(note_path), hub=None, op_type=None)
    assert not any("delimiter" in e.lower() for e in errors), f"errors={errors}"
