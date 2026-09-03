"""Tests for silica/kernel/write/provenance.py (spec-hermes-coherence §3).

Note<->source drift via sha256 provenance records. Pure filesystem module
(stdlib json/hashlib/datetime), mirroring silica/kernel/recall/run_log.py: append
is best-effort (never raises), absence of the store degrades to "no
records" everywhere.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from silica.kernel.write.provenance import (
    DEFAULT_PROVENANCE_FILENAME,
    append_record,
    check_renucleate,
    content_sha256,
    drifted_notes,
    read_records,
)


# ---------------------------------------------------------------------------
# append_record / read_records — record shape + append-only behaviour
# ---------------------------------------------------------------------------

def test_append_record_writes_exact_shape(tmp_path):
    ok = append_record(
        "lezione-03.md", "sha-v1", "run-abc123", ["Concepts/A", "Concepts/B"],
        vault_path=str(tmp_path), date="2026-07-02",
    )
    assert ok == "present"

    raw = json.loads((tmp_path / DEFAULT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert raw == [{
        "source": "lezione-03.md",
        "sha256": "sha-v1",
        "run_id": "run-abc123",
        "date": "2026-07-02",
        "notes": ["Concepts/A", "Concepts/B"],
    }]


def test_append_record_resume_same_run_source_sha_is_idempotent(tmp_path):
    """A resumed run re-firing CLEANUP for the same file hits this seam again
    with the same (source, sha256, run_id) — must not duplicate the record."""
    append_record("a.md", "sha1", "run1", ["N1"], vault_path=str(tmp_path))
    append_record("a.md", "sha1", "run1", ["N1"], vault_path=str(tmp_path))

    records = read_records(vault_path=str(tmp_path))
    assert len(records) == 1


def test_append_record_is_append_only(tmp_path):
    append_record("a.md", "sha1", "r1", ["N1"], vault_path=str(tmp_path))
    append_record("a.md", "sha2", "r2", ["N2"], vault_path=str(tmp_path))

    records = read_records(vault_path=str(tmp_path))
    assert len(records) == 2
    assert [r["sha256"] for r in records] == ["sha1", "sha2"]


def test_read_records_filters_by_source(tmp_path):
    append_record("a.md", "sha1", "r1", ["N1"], vault_path=str(tmp_path))
    append_record("b.md", "sha1", "r1", ["N2"], vault_path=str(tmp_path))

    records = read_records("a.md", vault_path=str(tmp_path))
    assert len(records) == 1
    assert records[0]["source"] == "a.md"


def test_read_records_missing_file_returns_empty(tmp_path):
    assert read_records(vault_path=str(tmp_path)) == []


def test_read_records_corrupt_file_returns_empty(tmp_path):
    (tmp_path / DEFAULT_PROVENANCE_FILENAME).write_text("{not json", encoding="utf-8")
    assert read_records(vault_path=str(tmp_path)) == []


def test_append_record_no_vault_path_is_absent(monkeypatch):
    import silica.config as config_mod
    monkeypatch.setattr(config_mod.CONFIG, "vault_path", "")
    assert append_record("a.md", "sha1", "r1", ["N1"]) == "absent"


def test_append_record_resume_refire_is_present(tmp_path):
    """The second CLEANUP of a resumed run changes nothing on disk, but the
    record IS there: the caller asked whether its record is durable, not
    whether this call wrote bytes."""
    append_record("a.md", "sha1", "run1", ["N1"], vault_path=str(tmp_path))
    assert append_record("a.md", "sha1", "run1", ["N1"], vault_path=str(tmp_path)) == "present"


def test_append_record_survives_unwritable_store(tmp_path, monkeypatch, caplog):
    """Best-effort: an I/O failure on write must not raise."""
    import silica.kernel.recall.paths as paths_mod

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(paths_mod, "atomic_write_bytes", _boom)
    with caplog.at_level(logging.WARNING, logger="silica.kernel.write.provenance"):
        out = append_record("a.md", "sha1", "r1", ["N1"], vault_path=str(tmp_path))
    # Nothing reached the disk, and the ledger says so: a proven absence, and
    # a warning that names what the absence costs (the next nucleate of this
    # source re-appends into the notes it already wrote).
    assert out == "absent"
    assert "re-ingest" in caplog.text


def test_append_record_is_present_when_the_error_came_after_the_replace(tmp_path, monkeypatch):
    """atomic_write_bytes can raise after os.replace landed: the record IS on
    disk, and calling it absent would double-append on the next nucleate."""
    import silica.kernel.recall.paths as paths_mod

    real = paths_mod.atomic_write_bytes

    def _late(path, data):
        real(path, data)
        raise OSError("late failure")

    monkeypatch.setattr(paths_mod, "atomic_write_bytes", _late)
    assert append_record("a.md", "sha1", "r1", ["N1"], vault_path=str(tmp_path)) == "present"


def test_append_record_is_uncertain_when_the_store_cannot_be_read_back(tmp_path, monkeypatch, caplog):
    """Garbage on disk after a failed write is neither presence nor absence,
    and the ledger must not turn unreadable evidence into a proven absence."""
    import silica.kernel.recall.paths as paths_mod

    def _torn(path, data):
        Path(path).write_bytes(data[: len(data) // 2])
        raise OSError("torn")

    monkeypatch.setattr(paths_mod, "atomic_write_bytes", _torn)
    with caplog.at_level(logging.WARNING, logger="silica.kernel.write.provenance"):
        out = append_record("a.md", "sha1", "r1", ["N1"], vault_path=str(tmp_path))
    assert out == "uncertain"
    assert "re-ingest" in caplog.text


# ---------------------------------------------------------------------------
# drifted_notes — the drift rule from the spec
# ---------------------------------------------------------------------------

def test_no_provenance_file_no_drift(tmp_path):
    assert drifted_notes(vault_path=str(tmp_path)) == []


def test_single_version_no_drift(tmp_path):
    append_record("a.md", "sha1", "r1", ["Nota A", "Nota B"], vault_path=str(tmp_path))
    assert drifted_notes(vault_path=str(tmp_path)) == []


def test_v2_touching_half_the_notes_drifts_the_other_half(tmp_path):
    """Acceptance criterion: nucleate v1 (A,B) -> modify -> re-nucleate v2 (A only)
    -> B is drifted, A is not."""
    append_record("lezione-03.md", "sha-v1", "r1", ["Nota A", "Nota B"], vault_path=str(tmp_path))
    append_record("lezione-03.md", "sha-v2", "r2", ["Nota A"], vault_path=str(tmp_path))

    drift = drifted_notes(vault_path=str(tmp_path))
    assert drift == [("Nota B", "lezione-03.md")]


def test_note_untouched_by_v2_but_present_in_v2_is_not_drifted(tmp_path):
    append_record("a.md", "sha1", "r1", ["A", "B"], vault_path=str(tmp_path))
    append_record("a.md", "sha2", "r2", ["A", "B"], vault_path=str(tmp_path))
    assert drifted_notes(vault_path=str(tmp_path)) == []


def test_drift_scoped_per_source(tmp_path):
    append_record("a.md", "sha1", "r1", ["A1"], vault_path=str(tmp_path))
    append_record("a.md", "sha2", "r2", [], vault_path=str(tmp_path))
    append_record("b.md", "shaB", "r3", ["B1"], vault_path=str(tmp_path))

    drift = drifted_notes(vault_path=str(tmp_path))
    assert drift == [("A1", "a.md")]


def test_v2_touching_nothing_drifts_all_v1_notes(tmp_path):
    """A re-nucleate whose sha changed but produced zero write/patch ops still
    means every v1 note is now stale relative to the new version."""
    append_record("a.md", "sha1", "r1", ["A", "B"], vault_path=str(tmp_path))
    append_record("a.md", "sha2", "r2", [], vault_path=str(tmp_path))

    drift = drifted_notes(vault_path=str(tmp_path))
    assert sorted(drift) == [("A", "a.md"), ("B", "a.md")]


# ---------------------------------------------------------------------------
# check_renucleate — the /nucleate warning seam
# ---------------------------------------------------------------------------

def test_check_renucleate_no_prior_record_no_warning(tmp_path):
    modified, count = check_renucleate("new-source.md", "sha1", vault_path=str(tmp_path))
    assert modified is False
    assert count == 0


def test_check_renucleate_same_sha_no_warning(tmp_path):
    append_record("a.md", "sha1", "r1", ["A", "B"], vault_path=str(tmp_path))
    modified, count = check_renucleate("a.md", "sha1", vault_path=str(tmp_path))
    assert modified is False
    assert count == 0


def test_check_renucleate_different_sha_warns_with_prior_note_count(tmp_path):
    append_record("a.md", "sha1", "r1", ["A", "B"], vault_path=str(tmp_path))
    modified, count = check_renucleate("a.md", "sha2", vault_path=str(tmp_path))
    assert modified is True
    assert count == 2


def test_check_renucleate_uses_most_recent_record(tmp_path):
    append_record("a.md", "sha1", "r1", ["A"], vault_path=str(tmp_path))
    append_record("a.md", "sha2", "r2", ["A", "B"], vault_path=str(tmp_path))
    modified, count = check_renucleate("a.md", "sha2", vault_path=str(tmp_path))
    assert modified is False
    assert count == 0

    modified2, count2 = check_renucleate("a.md", "sha3", vault_path=str(tmp_path))
    assert modified2 is True
    assert count2 == 2


# ---------------------------------------------------------------------------
# content_sha256 — matches orchestrator.run()'s hashing exactly (hash parity
# between the CLEANUP write side and the /nucleate pre-check read side)
# ---------------------------------------------------------------------------

def test_content_sha256_matches_manual_hash(tmp_path, monkeypatch):
    import hashlib
    import silica.config as config_mod
    from silica.driver import fs_backend
    import silica.driver as driver_mod

    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "Inbox" / "a.md"
    note.parent.mkdir(parents=True)
    note.write_text("hello world", encoding="utf-8")

    monkeypatch.setattr(config_mod.CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(driver_mod, "DRIVER", fs_backend.ObsidianFSBackend(str(vault)))

    expected = hashlib.sha256("hello world".encode("utf-8")).hexdigest()
    assert content_sha256("Inbox/a.md") == expected


def test_content_sha256_missing_file_returns_empty(tmp_path, monkeypatch):
    import silica.config as config_mod
    from silica.driver import fs_backend
    import silica.driver as driver_mod

    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(config_mod.CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(driver_mod, "DRIVER", fs_backend.ObsidianFSBackend(str(vault)))

    assert content_sha256("Inbox/does-not-exist.md") == ""


# ---------------------------------------------------------------------------
# read_records memo (bulk._execute_patch calls this once per patch op)
# ---------------------------------------------------------------------------

def test_read_records_parses_once_for_repeated_reads(tmp_path, monkeypatch):
    append_record("a.md", "sha1", "r1", ["N1"], vault_path=str(tmp_path))

    import silica.kernel.write.provenance as prov

    parses = 0
    real_loads = prov.json.loads

    def counting_loads(*a, **kw):
        nonlocal parses
        parses += 1
        return real_loads(*a, **kw)

    monkeypatch.setattr(prov.json, "loads", counting_loads)
    for _ in range(5):
        assert len(read_records(vault_path=str(tmp_path))) == 1
    assert parses == 1


def test_read_records_memo_invalidates_on_write(tmp_path):
    append_record("a.md", "sha1", "r1", ["N1"], vault_path=str(tmp_path))
    assert len(read_records(vault_path=str(tmp_path))) == 1

    append_record("b.md", "sha2", "r2", ["N2"], vault_path=str(tmp_path))
    assert len(read_records(vault_path=str(tmp_path))) == 2

    # an out-of-process writer must be seen too: rewrite the file behind our back
    store = tmp_path / DEFAULT_PROVENANCE_FILENAME
    store.write_text(json.dumps([]), encoding="utf-8")
    assert read_records(vault_path=str(tmp_path)) == []


def test_read_records_caller_cannot_poison_the_memo(tmp_path):
    """append_record mutates the list it gets back; the memo must not see it."""
    append_record("a.md", "sha1", "r1", ["N1"], vault_path=str(tmp_path))

    first = read_records(vault_path=str(tmp_path))
    first.append({"source": "ghost.md"})

    assert [r["source"] for r in read_records(vault_path=str(tmp_path))] == ["a.md"]


def test_append_is_atomic_a_failed_write_keeps_the_prior_ledger(tmp_path, monkeypatch):
    """A crash mid-rewrite must not truncate the store.

    This ledger is authoritative — run_id/sha history is not reconstructible
    from the vault, and read_records quarantines a truncated file — so a torn
    write silently loses every prior record.
    """
    import os

    append_record("a.md", "sha1", "r1", ["N1"], vault_path=str(tmp_path))
    store = tmp_path / DEFAULT_PROVENANCE_FILENAME
    before = store.read_bytes()

    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full")))
    assert append_record("b.md", "sha2", "r2", ["N2"], vault_path=str(tmp_path)) == "absent"

    assert store.read_bytes() == before
    assert list(tmp_path.iterdir()) == [store]  # no tmp leftovers


def test_concurrent_appends_lose_no_records(tmp_path, monkeypatch):
    """read->append->replace was atomic per write but not per window: two
    parallel nucleate workers each rewrote the ledger from their own read and
    the last one won. The lease serializes the window."""
    import threading

    from silica.kernel.write import provenance

    vault = tmp_path / "vault"
    vault.mkdir()
    barrier = threading.Barrier(8)

    def _append(i):
        barrier.wait()
        provenance.append_record(
            source=f"inbox/s{i}.md", sha256=f"sha{i}", run_id=f"run{i}",
            notes=[f"N{i}.md"], vault_path=str(vault))

    threads = [threading.Thread(target=_append, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    records = provenance.read_records(vault_path=str(vault))
    assert len(records) == 8


def test_append_record_merges_notes_into_a_matching_triple(tmp_vault):
    """Two worker commits for the same (source, sha, run) must both be recorded.
    The old idempotency dropped the second outright, so a dedup lane that
    committed twice for one chunk left its second note untraceable."""
    from silica.kernel.write.provenance import append_record, read_records

    assert append_record("s.md", "sha1", "run1", ["Notes/A"])
    append_record("s.md", "sha1", "run1", ["Notes/B"])

    rec = [r for r in read_records() if r["sha256"] == "sha1"]
    assert len(rec) == 1, "the triple must stay one record, not two"
    assert rec[0]["notes"] == ["Notes/A", "Notes/B"]


def test_append_record_of_an_identical_triple_changes_nothing(tmp_vault):
    """Resume safety: re-entering CLEANUP with the same triple and the same
    notes must not grow the ledger."""
    from silica.kernel.write.provenance import append_record, read_records

    append_record("s.md", "sha2", "run2", ["Notes/A"])
    append_record("s.md", "sha2", "run2", ["Notes/A"])
    rec = [r for r in read_records() if r["sha256"] == "sha2"]
    assert len(rec) == 1 and rec[0]["notes"] == ["Notes/A"]


def test_a_failed_merge_write_leaves_nothing_in_the_memo(tmp_path, monkeypatch):
    """The merge branch used to edit the record dict in place.

    read_records copies the LIST but hands back the very dicts the memo holds,
    so the edit landed in the cache before the rewrite was known to have
    succeeded — and when the rewrite failed, the memo went on reporting notes
    that were never persisted. note_authored_by then answered True for an
    unrecorded note and bulk._execute_patch dropped the patch as a no-op.
    """
    import os

    from silica.kernel.write.provenance import note_authored_by

    append_record("a.md", "sha1", "r1", ["N1"], vault_path=str(tmp_path))
    read_records(vault_path=str(tmp_path))            # warm the memo

    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full")))
    assert append_record("a.md", "sha1", "r1", ["N2"], vault_path=str(tmp_path)) == "absent"
    monkeypatch.undo()

    assert read_records(vault_path=str(tmp_path))[0]["notes"] == ["N1"]
    assert note_authored_by("N2", "a.md", vault_path=str(tmp_path)) is False
    assert note_authored_by("N1", "a.md", vault_path=str(tmp_path)) is True


def test_a_successful_merge_still_unions_the_notes(tmp_path):
    """The sub-agent lane commits several times under one triple; the later
    commits must not lose their notes."""
    append_record("a.md", "sha1", "r1", ["N1"], vault_path=str(tmp_path))
    assert append_record("a.md", "sha1", "r1", ["N2"], vault_path=str(tmp_path)) == "present"

    recs = read_records(vault_path=str(tmp_path))
    assert len(recs) == 1 and recs[0]["notes"] == ["N1", "N2"]


# ---------------------------------------------------------------------------
# is_deriving_op — one spelling for "this op derived a note from the source"
# ---------------------------------------------------------------------------

def test_overwrite_derives_a_note_like_write_and_patch():
    """Four sub-agent profiles in agent/bounds.py may emit NOTHING BUT
    overwrite, and all three dedup merges emit it. A predicate that lists only
    write/patch unrecords the majority of that lane."""
    from silica.kernel.write.ops import OpType
    from silica.kernel.write.provenance import is_deriving_op

    for op in ("write", "patch", "overwrite"):
        assert is_deriving_op(op), op
    for op in (OpType.write, OpType.patch, OpType.overwrite):
        assert is_deriving_op(op), op
    assert not is_deriving_op("skip")
    assert not is_deriving_op(OpType.skip)


# ---------------------------------------------------------------------------
# nonextractive_lines — the extractive invariant
# ---------------------------------------------------------------------------

_SRC = "The quick brown fox jumps over the lazy dog and then keeps on going."


def test_a_body_of_nothing_but_link_lines_is_still_checked():
    """The wikilink-caption exemption `continue`d without recording the line,
    so it was neither judged nor judgeable later: the `if not judged` fallback
    could not see it and a body that was never checked read as fully
    extractive."""
    from silica.kernel.write.provenance import nonextractive_lines

    assert nonextractive_lines("Related: [[Alpha]]\nSee also [[Beta]]", _SRC)
    assert nonextractive_lines("[[Kant]] rejected [[Hume]].", _SRC)


def test_a_link_footer_beside_real_content_is_still_exempt():
    """The exemption itself stands: judging link footers made every block
    carrying one unpassable."""
    from silica.kernel.write.provenance import nonextractive_lines

    body = "The quick brown fox jumps over the lazy dog\n\nCorrelati: [[Alpha]]"
    assert nonextractive_lines(body, _SRC) == []


def test_a_single_quoted_verbatim_span_is_extractive():
    """_norm_extract folds U+2018/U+2019 to a straight apostrophe, so the
    straight form is what reaches the wrapping-quote strip — and it was the one
    quote missing from the class."""
    from silica.kernel.write.provenance import nonextractive_lines

    for open_q, close_q in (("‘", "’"), ("“", "”"),
                            ("«", "»"), ('"', '"')):
        line = f"{open_q}The quick brown fox jumps over the lazy dog{close_q}"
        assert nonextractive_lines(line, _SRC) == [], (open_q, close_q)


def test_a_fabricated_line_is_still_rejected():
    from silica.kernel.write.provenance import nonextractive_lines

    assert nonextractive_lines("A slow green turtle ambles past the alert cat.", _SRC)


# ---------------------------------------------------------------------------
# rename_note — the ledger follows a moved note
# ---------------------------------------------------------------------------

def test_rename_note_follows_the_moved_note(tmp_path):
    """The ledger keys notes by bare path: after a move the old key answered
    nothing, so a re-ingest re-appended into the note it had already written."""
    from silica.kernel.write.provenance import note_authored_by, rename_note

    append_record("a.md", "sha1", "r1", ["Concepts/A", "Concepts/B"], vault_path=str(tmp_path))

    out = rename_note("Concepts/A.md", "Archive/A.md", vault_path=str(tmp_path))

    assert out == "present"
    assert note_authored_by("Archive/A.md", "a.md", vault_path=str(tmp_path))
    assert not note_authored_by("Concepts/A.md", "a.md", vault_path=str(tmp_path))
    # the sibling entry is untouched
    assert note_authored_by("Concepts/B.md", "a.md", vault_path=str(tmp_path))


def test_rename_note_rewrites_every_record_naming_the_note(tmp_path):
    from silica.kernel.write.provenance import rename_note

    append_record("a.md", "sha1", "r1", ["Concepts/A"], vault_path=str(tmp_path))
    append_record("b.md", "sha9", "r2", ["Concepts/A", "Concepts/C"], vault_path=str(tmp_path))

    rename_note("Concepts/A.md", "Concepts/A2.md", vault_path=str(tmp_path))

    notes = [r["notes"] for r in read_records(vault_path=str(tmp_path))]
    assert notes == [["Concepts/A2"], ["Concepts/A2", "Concepts/C"]]


def test_rename_note_with_nothing_to_rename_writes_nothing(tmp_path):
    from silica.kernel.write.provenance import rename_note

    append_record("a.md", "sha1", "r1", ["Concepts/A"], vault_path=str(tmp_path))
    store = tmp_path / DEFAULT_PROVENANCE_FILENAME
    before = store.stat().st_mtime_ns

    assert rename_note("Other/X.md", "Other/Y.md", vault_path=str(tmp_path)) == "present"
    assert store.stat().st_mtime_ns == before


def test_rename_note_failure_is_absent_and_named(tmp_path, monkeypatch, caplog):
    from silica.kernel.write.provenance import note_authored_by, rename_note
    import silica.kernel.recall.paths as paths_mod

    append_record("a.md", "sha1", "r1", ["Concepts/A"], vault_path=str(tmp_path))

    def _boom(*a, **k):
        raise OSError("read-only vault")

    monkeypatch.setattr(paths_mod, "atomic_write_bytes", _boom)
    with caplog.at_level(logging.WARNING, logger="silica.kernel.write.provenance"):
        out = rename_note("Concepts/A.md", "Archive/A.md", vault_path=str(tmp_path))

    assert out == "absent"
    assert "re-ingest" in caplog.text
    assert note_authored_by("Concepts/A.md", "a.md", vault_path=str(tmp_path))


# ---------------------------------------------------------------------------
# sources_of / drift_map — the ledger read from the note's side
# ---------------------------------------------------------------------------

def test_sources_of_lists_every_source_that_authored_the_note(tmp_path):
    from silica.kernel.write.provenance import sources_of

    append_record("a.md", "sha1", "r1", ["Concepts/X"], vault_path=str(tmp_path))
    append_record("b.md", "sha2", "r2", ["Concepts/X", "Concepts/Y"], vault_path=str(tmp_path))
    append_record("a.md", "sha3", "r3", ["Concepts/X"], vault_path=str(tmp_path))

    assert sources_of("Concepts/X.md", vault_path=str(tmp_path)) == ["a.md", "b.md"]
    assert sources_of("Concepts/Y", vault_path=str(tmp_path)) == ["b.md"]
    assert sources_of("Concepts/Nope", vault_path=str(tmp_path)) == []


def test_drift_map_keys_notes_the_way_the_stale_peek_does(tmp_path):
    """`.md`-suffixed keys, so codedocs.peek_level reads both maps alike."""
    from silica.kernel.write.provenance import drift_map

    append_record("lec.md", "sha-v1", "r1", ["Concepts/A", "Concepts/B"], vault_path=str(tmp_path))
    append_record("lec.md", "sha-v2", "r2", ["Concepts/A"], vault_path=str(tmp_path))

    assert drift_map(vault_path=str(tmp_path)) == {"Concepts/B.md": "lec.md"}


def test_partial_run_record_never_counts_as_already_distilled(tmp_vault):
    """A run that failed a chunk at DELEGATE but wrote two notes through
    COLLISION recorded them, and the next /nucleate of the unchanged file said
    "already distilled" and skipped the segment (probe run 2026-09-02). The
    record carries the run's partial verdict; only a complete run skips."""
    from silica.config import CONFIG
    from silica.kernel.write.provenance import already_distilled, append_record

    append_record("lez.md", "sha-1", "run-1", ["Corso/A"], vault_path=CONFIG.vault_path,
                  partial=True)
    assert already_distilled("lez.md", "sha-1", vault_path=CONFIG.vault_path) is False
    append_record("lez.md", "sha-1", "run-2", ["Corso/A", "Corso/B"], vault_path=CONFIG.vault_path)
    assert already_distilled("lez.md", "sha-1", vault_path=CONFIG.vault_path) is True
    assert already_distilled("lez.md", "sha-other", vault_path=CONFIG.vault_path) is False
