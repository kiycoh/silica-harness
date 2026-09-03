# tests/test_session_changes.py
import re
import pytest

from silica.kernel.write import session_changes


@pytest.fixture(autouse=True)
def _clean_ledger():
    session_changes.clear()
    yield
    session_changes.clear()


def _rows():
    from silica.ui.web.server import _change_rows
    return {r["path"]: r for r in _change_rows()}


def test_baseline_is_the_first_touch_not_the_last(tmp_vault):
    from silica.driver import DRIVER

    tmp_vault.note("Notes/Ada.md", "one\ntwo\n")
    DRIVER.overwrite("Notes/Ada.md", "one\ntwo\nthree\n")
    DRIVER.overwrite("Notes/Ada.md", "one\ntwo\nthree\nfour\n")

    row = _rows()["Notes/Ada.md"]
    assert row["kind"] == "modified"
    # Measured against how the note stood BEFORE the session, not before the
    # last write: two lines arrived, none left.
    assert (row["added"], row["removed"]) == (2, 0)


def test_create_then_delete_leaves_no_row(tmp_vault):
    from silica.driver import DRIVER

    DRIVER.create("Notes/Ghost.md", "body\n")
    assert _rows()["Notes/Ghost.md"]["kind"] == "created"
    DRIVER.delete("Notes/Ghost.md")
    assert "Notes/Ghost.md" not in _rows()


def test_delete_of_an_existing_note_is_all_red(tmp_vault):
    from silica.driver import DRIVER

    tmp_vault.note("Notes/Gone.md", "a\nb\n")
    DRIVER.read_note("Notes/Gone.md")  # warm the index the way a real turn does
    DRIVER.delete("Notes/Gone.md")

    row = _rows()["Notes/Gone.md"]
    assert row["kind"] == "deleted"
    assert (row["added"], row["removed"]) == (0, 2)


def test_a_move_keeps_one_row_carrying_its_origin(tmp_vault):
    from silica.driver import DRIVER

    tmp_vault.note("Inbox/Draft.md", "same bytes\n")
    DRIVER.move("Inbox/Draft.md", "Notes/Draft.md")

    rows = _rows()
    assert "Inbox/Draft.md" not in rows
    assert rows["Notes/Draft.md"]["kind"] == "moved"
    assert rows["Notes/Draft.md"]["from"] == "Inbox/Draft.md"


def test_a_write_that_changed_nothing_is_not_a_change(tmp_vault):
    from silica.driver import DRIVER

    tmp_vault.note("Notes/Same.md", "unchanged\n")
    DRIVER.overwrite("Notes/Same.md", "unchanged\n")
    assert "Notes/Same.md" not in _rows()


def test_diff_marks_the_lines_that_left_and_arrived(tmp_vault):
    from silica.ui.web.server import changes_diff
    from silica.driver import DRIVER

    tmp_vault.note("Notes/Edit.md", "keep\ndrop\ntail\n")
    DRIVER.overwrite("Notes/Edit.md", "keep\nadd\ntail\n")

    d = changes_diff(path="Notes/Edit.md")
    ops = [(l["op"], l["text"]) for l in d["lines"]]
    assert ("-", "drop") in ops
    assert ("+", "add") in ops
    assert (" ", "keep") in ops  # context lines survive with the space op
    assert (d["added"], d["removed"]) == (1, 1)


def test_a_gap_marker_only_appears_where_lines_were_skipped(tmp_vault):
    from silica.ui.web.server import changes_diff
    from silica.driver import DRIVER

    # Edit on line 1: nothing was skipped above it, so the diff must not open on
    # a "⋯" that stands for no lines at all.
    tmp_vault.note("Notes/Top.md", "first\nsecond\n")
    DRIVER.overwrite("Notes/Top.md", "FIRST\nsecond\n")
    assert changes_diff(path="Notes/Top.md")["lines"][0]["op"] != "@"

    # Edit far down a long note: now the gap is the truth.
    body = "\n".join(f"line {i}" for i in range(40)) + "\n"
    tmp_vault.note("Notes/Deep.md", body)
    DRIVER.overwrite("Notes/Deep.md", body.replace("line 30", "LINE 30"))
    assert changes_diff(path="Notes/Deep.md")["lines"][0]["op"] == "@"


def test_a_created_note_opens_on_its_first_line(tmp_vault):
    from silica.ui.web.server import changes_diff
    from silica.driver import DRIVER

    DRIVER.create("Notes/Fresh.md", "one\ntwo\n")
    lines = changes_diff(path="Notes/Fresh.md")["lines"]
    assert lines[0] == {"op": "+", "text": "one"}


def test_untracked_note_has_an_empty_diff(tmp_vault):
    from silica.ui.web.server import changes_diff

    tmp_vault.note("Notes/Quiet.md", "never touched\n")
    assert changes_diff(path="Notes/Quiet.md")["lines"] == []


def test_the_ws_backend_records_what_the_plugin_writes(tmp_vault):
    """With Obsidian connected, the driver is the ws backend and the plugin does
    the writing. The list is about the vault, not about who held the pen — so the
    row must appear all the same."""
    from pathlib import Path

    from silica.config import CONFIG
    from silica.driver.ws_backend import ObsidianWSBackend

    backend = ObsidianWSBackend("ws://127.0.0.1:0")

    def fake_rpc(method, **params):
        """The plugin's half: writes to the same folder CONFIG points at."""
        p = Path(CONFIG.vault_path) / params["path"]
        if method == "overwrite":
            p.write_text(params["content"], encoding="utf-8")
            return None
        if method == "create":
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(params["content"], encoding="utf-8")
            return {"name": p.stem, "path": params["path"]}
        if method == "move":
            (Path(CONFIG.vault_path) / params["to"]).parent.mkdir(parents=True, exist_ok=True)
            p.rename(Path(CONFIG.vault_path) / params["to"])
            return None
        raise AssertionError(f"unexpected rpc: {method}")

    backend._rpc = fake_rpc

    tmp_vault.note("Notes/Live.md", "one\ntwo\n")
    backend.overwrite("Notes/Live.md", "one\ntwo\nthree\n")
    backend.create("Notes/New.md", "fresh\n")
    tmp_vault.note("Inbox/Draft.md", "same bytes\n")
    backend.move("Inbox/Draft.md", "Notes/Draft.md")

    rows = _rows()
    assert rows["Notes/Live.md"]["kind"] == "modified"
    assert (rows["Notes/Live.md"]["added"], rows["Notes/Live.md"]["removed"]) == (1, 0)
    assert rows["Notes/New.md"]["kind"] == "created"
    assert rows["Notes/Draft.md"]["from"] == "Inbox/Draft.md"
    assert "Inbox/Draft.md" not in rows


def test_the_repl_lists_what_the_session_wrote(tmp_vault, capsys):
    """/changes is the TUI's half of the GUI drawer: same ledger, one line each."""
    from silica.cli import _dc_changes
    from silica.driver import DRIVER

    tmp_vault.note("Notes/Ada.md", "one\ntwo\n")
    DRIVER.overwrite("Notes/Ada.md", "one\ntwo\nthree\n")
    DRIVER.create("Notes/New.md", "fresh\n")

    assert _dc_changes([]) is True
    # Rich honours FORCE_COLOR (set by some harnesses): read the text, not the codes.
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    assert "M Notes/Ada.md" in out and "+1" in out
    assert "A Notes/New.md" in out
    assert "2 note(s)" in out


def test_the_repl_says_nothing_happened_rather_than_printing_an_empty_list(tmp_vault, capsys):
    from silica.cli import _dc_changes

    assert _dc_changes([]) is True
    assert "Nothing changed this session." in capsys.readouterr().out
