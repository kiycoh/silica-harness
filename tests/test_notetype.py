"""Tests for the OKF `type` field (spec-okf-export.md, Lane 1).

Three things must hold: the derivation reads the signals the vault already
carries, with a stable precedence; the stamp is additive and idempotent, and
never touches a note that already declares a type or whose frontmatter it
cannot parse; and the walker reports each §11 clause on the note that breaks it.
"""
from __future__ import annotations

import pytest

from silica.kernel.write.notetype import (
    derive_type,
    is_human_verified,
    okf_conformance,
    stamp_type,
    verified_entries,
)

PLAIN = "---\ntitle: X\n---\n\nBody.\n"


# --- derive_type -------------------------------------------------------------

def test_source_leaf_is_a_source():
    assert derive_type("sources/session_1.md", PLAIN) == "Source"


def test_code_binding_is_code():
    assert derive_type("notes/x.md", "---\ndocuments:\n  - silica/cli.py\n---\n\nB\n") == "Code"
    assert derive_type("notes/x.md", "---\ncode_ref: abc123\n---\n\nB\n") == "Code"


def test_plan_status_is_a_plan():
    assert derive_type("plans/x.md", "---\nstatus: in-progress\n---\n\nB\n") == "Plan"


def test_status_outside_the_plan_enum_is_a_note():
    assert derive_type("notes/x.md", "---\nstatus: shipped\n---\n\nB\n") == "Note"


def test_everything_else_is_a_note():
    assert derive_type("notes/x.md", PLAIN) == "Note"
    assert derive_type("notes/x.md", "# No frontmatter\n") == "Note"


def test_precedence_source_over_code_over_plan():
    both = "---\ncode_ref: abc\nstatus: todo\n---\n\nB\n"
    assert derive_type("sources/leaf.md", both) == "Source"
    assert derive_type("notes/x.md", both) == "Code"


# --- stamp_type --------------------------------------------------------------

def test_stamp_inserts_the_derived_type():
    out = stamp_type("plans/x.md", "---\nstatus: todo\n---\n\nB\n")
    assert out == "---\nstatus: todo\ntype: Plan\n---\n\nB\n"


def test_stamp_is_idempotent():
    once = stamp_type("notes/x.md", PLAIN)
    assert stamp_type("notes/x.md", once) == once
    assert once.count("type:") == 1


def test_stamp_never_overwrites_a_declared_type():
    """§4.1 tolerates unknown types: the user's own vocabulary wins."""
    hand = "---\ntype: Recipe\nstatus: todo\n---\n\nB\n"
    assert stamp_type("plans/x.md", hand) == hand


def test_stamp_leaves_alone_what_it_cannot_parse():
    no_fm = "# Just prose\n"
    assert stamp_type("notes/x.md", no_fm) == no_fm
    broken = "---\ntitle: [unclosed\n---\n\nB\n"
    assert stamp_type("notes/x.md", broken) == broken
    unterminated = "---\ntitle: X\n"
    assert stamp_type("notes/x.md", unterminated) == unterminated


def test_stamp_preserves_the_rest_of_the_frontmatter_byte_for_byte():
    fm = "---\ntitle: X\ntags: [a, b]\nAI: true\n---\n\nB\n"
    out = stamp_type("notes/x.md", fm)
    assert out.replace("\ntype: Note", "") == fm


# --- verified (§5.2) ---------------------------------------------------------

def test_a_bare_mapping_reads_as_one_entry():
    """§5.2 MUST: a single {by, at} is a one-element list to a reader."""
    entry = {"by": "human:alessandro", "at": "2026-08-03"}
    assert verified_entries({"verified": entry}) == [entry]
    assert verified_entries({"verified": [entry]}) == [entry]


def test_absent_or_malformed_verified_is_no_entries():
    assert verified_entries({}) == []
    assert verified_entries(None) == []
    assert verified_entries({"verified": "yesterday"}) == []
    assert verified_entries({"verified": ["yesterday"]}) == []


def test_only_a_human_actor_counts_as_verification():
    assert is_human_verified({"verified": {"by": "human:alessandro"}})
    assert is_human_verified({"verified": {"by": "HUMAN:Alessandro"}})
    assert not is_human_verified({"verified": {"by": "process:nucleate"}})
    assert not is_human_verified({"verified": {"at": "2026-08-03"}})
    assert not is_human_verified({})


# --- okf_conformance ---------------------------------------------------------

def _write(vault, rel, content):
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_clean_vault_has_no_violations(tmp_path):
    _write(tmp_path, "notes/a.md", "---\ntype: Note\n---\n\nB\n")
    _write(tmp_path, "sources/leaf.md", "---\ntype: Source\nsource_id: leaf.md\n---\n\nB\n")
    assert okf_conformance(tmp_path).violations == []


def test_walker_reports_each_clause(tmp_path):
    _write(tmp_path, "notes/no_fm.md", "# Just prose\n")
    _write(tmp_path, "notes/broken.md", "---\ntitle: [unclosed\n---\n\nB\n")
    _write(tmp_path, "notes/untyped.md", PLAIN)
    _write(tmp_path, "index.md", "---\ntype: Note\n---\n\nB\n")

    by_path = {v.path: v for v in okf_conformance(tmp_path).violations}
    assert by_path["notes/no_fm.md"].clause == "11.1"
    assert by_path["notes/broken.md"].clause == "11.1"
    assert by_path["notes/untyped.md"].clause == "11.2"
    assert by_path["index.md"].clause == "11.3"


def test_walker_skips_dot_dirs(tmp_path):
    _write(tmp_path, ".obsidian/plugins/x.md", "# no frontmatter\n")
    _write(tmp_path, ".trash/old.md", "# no frontmatter\n")
    assert okf_conformance(tmp_path).violations == []


def test_walker_refuses_a_missing_vault(tmp_path):
    """An empty result over a path that is not there read as a conformant
    bundle: a zero-file scan is an invocation error, never a pass."""
    with pytest.raises(NotADirectoryError):
        okf_conformance(tmp_path / "nope")


def test_backfill_makes_a_legacy_vault_conformant(tmp_path):
    """The gate the spec sets: backfill, then zero violations."""
    import importlib.util
    from pathlib import Path

    _write(tmp_path, "notes/a.md", PLAIN)
    _write(tmp_path, "plans/b.md", "---\nstatus: todo\n---\n\nB\n")
    _write(tmp_path, "sources/leaf.md", "---\nsource_id: leaf.md\n---\n\nB\n")
    assert len(okf_conformance(tmp_path).violations) == 3

    script = Path(__file__).resolve().parent.parent / "scripts" / "backfill_notetype.py"
    spec = importlib.util.spec_from_file_location("backfill_notetype", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    counts = mod.backfill(tmp_path)
    assert counts["stamped"] == 3
    assert okf_conformance(tmp_path).violations == []
    assert mod.backfill(tmp_path)["stamped"] == 0   # second pass is a no-op


def test_silicas_own_journal_is_not_a_violation(tmp_path):
    """`log.md` is written by Silica, and §11.3's advice for it is "rename any
    `index`/`log` note by hand" — an instruction the user cannot carry out,
    since the next run recreates it. Doctor warned about it on every run."""
    _write(tmp_path, "vault.yaml", "write_dir: silica\n")
    _write(tmp_path, "silica/log.md", "- 2026-08-16 · nucleate `a.md` · run abc\n")

    assert okf_conformance(tmp_path).violations == []


def test_a_users_own_log_note_is_still_a_violation(tmp_path):
    _write(tmp_path, "vault.yaml", "write_dir: silica\n")
    _write(tmp_path, "Journal/log.md", "---\ntype: Note\n---\n\nB\n")

    assert [v.clause for v in okf_conformance(tmp_path).violations] == ["11.3"]
