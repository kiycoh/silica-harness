"""TDD tests for WS2 — Ledger integrity (content-hash, UPSERT, path-keyed).

Written BEFORE the implementation — all RED until ledger.py is updated.
Covers contracts C2.1–C2.6.
"""
from __future__ import annotations

import hashlib
import os
import time
from unittest.mock import patch

import pytest

from silica.kernel.write.ledger import Ledger


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def ledger(tmp_path):
    """Fresh in-memory ledger backed by a temp file."""
    return Ledger(tmp_path / "test_ledger.db")


# ---------------------------------------------------------------------------
# C2.2 — Content-hash awareness
# ---------------------------------------------------------------------------

def test_ledger_skip_on_identical_content(ledger, tmp_path):
    """is_committed returns True when hash matches and outputs exist."""
    output = tmp_path / "out.md"
    output.write_text("content", encoding="utf-8")

    h = _sha256("source content")
    ledger.record(
        txn_id="t1",
        source_canonical="concepts/backpropagation",
        path=str(output),
        op="write",
        status="committed",
        content_hash=h,
    )

    assert ledger.is_committed("concepts/backpropagation", content_hash=h)


def test_ledger_reprocess_on_content_change(ledger, tmp_path):
    """is_committed returns False when content_hash differs (note was modified)."""
    output = tmp_path / "out.md"
    output.write_text("content", encoding="utf-8")

    ledger.record(
        txn_id="t1",
        source_canonical="concepts/backpropagation",
        path=str(output),
        op="write",
        status="committed",
        content_hash=_sha256("original content"),
    )

    # Different hash → must reprocess
    assert not ledger.is_committed(
        "concepts/backpropagation",
        content_hash=_sha256("modified content"),
    )


# ---------------------------------------------------------------------------
# C2.3 — Output-existence awareness
# ---------------------------------------------------------------------------

def test_ledger_reprocess_when_output_missing(ledger, tmp_path):
    """is_committed returns False when a registered output no longer exists."""
    output = tmp_path / "out.md"
    h = _sha256("source content")

    ledger.record(
        txn_id="t1",
        source_canonical="concepts/backpropagation",
        path=str(output),
        op="write",
        status="committed",
        content_hash=h,
    )

    # Output does NOT exist on disk → skip must be invalidated
    assert not output.exists()
    assert not ledger.is_committed("concepts/backpropagation", content_hash=h)


# ---------------------------------------------------------------------------
# C2.4 — UPSERT semantics
# ---------------------------------------------------------------------------

def test_ledger_upsert_overwrites_prior_state(ledger, tmp_path):
    """A second record call on the same (source_canonical, path) produces one row."""
    output = tmp_path / "out.md"
    output.write_text("x", encoding="utf-8")
    h = _sha256("src")

    ledger.record("t1", "notes/alpha", str(output), "write", "committed", content_hash=h)
    ledger.record("t2", "notes/alpha", str(output), "write", "committed", content_hash=h)

    rows = ledger._conn.execute(
        "SELECT COUNT(*) FROM ops WHERE source_canonical=?", ("notes/alpha",)
    ).fetchone()[0]
    assert rows == 1, f"Expected 1 row after UPSERT, got {rows}"


# ---------------------------------------------------------------------------
# C2.5 — mark_failed materialised
# ---------------------------------------------------------------------------

def test_ledger_marks_failed_on_abort(ledger, tmp_path):
    """mark_failed writes a 'failed' row; is_committed returns False."""
    output = tmp_path / "out.md"

    ledger.record(
        txn_id="t1",
        source_canonical="notes/gamma",
        path=str(output),
        op="write",
        status="failed",
        content_hash=_sha256("src"),
    )

    # Failed row → should NOT skip
    assert not ledger.is_committed("notes/gamma", content_hash=_sha256("src"))


# ---------------------------------------------------------------------------
# C2.1 — Path-canonical key disambiguates same-basename sources
# ---------------------------------------------------------------------------

def test_ledger_canonical_path_key_disambiguates(ledger, tmp_path):
    """Two sources with the same basename in different folders → two distinct rows."""
    out_a = tmp_path / "out_a.md"
    out_b = tmp_path / "out_b.md"
    out_a.write_text("a", encoding="utf-8")
    out_b.write_text("b", encoding="utf-8")

    h = _sha256("src")
    ledger.record("t1", "a/cell", str(out_a), "write", "committed", content_hash=h)
    ledger.record("t2", "b/cell", str(out_b), "write", "committed", content_hash=h)

    assert ledger.is_committed("a/cell", content_hash=h)
    assert ledger.is_committed("b/cell", content_hash=h)

    # Also verify they are stored as separate rows
    count = ledger._conn.execute("SELECT COUNT(*) FROM ops").fetchone()[0]
    assert count == 2


# ---------------------------------------------------------------------------
# C2.6 — Backward-safe: NULL hash is stale
# ---------------------------------------------------------------------------

def test_ledger_null_hash_is_stale(ledger, tmp_path):
    """Legacy rows with content_hash=NULL are treated as stale (is_committed→False)."""
    output = tmp_path / "out.md"
    output.write_text("x", encoding="utf-8")

    # Insert a legacy row manually without content_hash
    ledger._conn.execute(
        "INSERT INTO ops(txn_id, source_canonical, path, op, status, ts) "
        "VALUES (?,?,?,?,?,?)",
        ("t0", "legacy/note", str(output), "write", "committed", time.time()),
    )
    ledger._conn.commit()

    # Querying with any hash should return False (stale)
    assert not ledger.is_committed("legacy/note", content_hash=_sha256("anything"))


# ---------------------------------------------------------------------------
# C2.2 — Orchestrator propagates content_hash to ledger
# ---------------------------------------------------------------------------

def test_orchestrator_writes_content_hash(tmp_path, monkeypatch):
    """InjectorFSM ledger writes pass the source content_hash from context."""
    from silica.router.orchestrator import InjectorFSM
    from silica.config import CONFIG

    # Point the vault at tmp_path so to_vault_relative can relativize the inbox path
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))

    # Set up a minimal inbox file
    inbox = tmp_path / "Lecture.md"
    inbox.write_text("# Lecture\n\nContent.", encoding="utf-8")

    fsm = InjectorFSM(inbox_file=str(inbox), target_dir="Concepts")
    expected_hash = hashlib.sha256(inbox.read_bytes()).hexdigest()
    fsm.context["source_content_hash"] = expected_hash

    # Build a minimal ops_path with one op
    import json, tempfile
    ops_data = [{
        "op": "write",
        "path": str(tmp_path / "Concepts" / "New.md"),
        "heading": "New",
        "source_basename": "Lecture.md",
        "content": "# New\n\nContent.",
    }]
    fd, ops_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(ops_data, f)
        fsm.context.setdefault("chunk", {})["ops_path"] = ops_path
        fsm.context.setdefault("chunk", {})["txn_id"] = "t_test"

        from silica.kernel.write.ledger import Ledger
        test_ledger = Ledger(tmp_path / "test.db")
        with patch("silica.kernel.write.ledger.get_ledger", return_value=test_ledger):
            fsm._write_ledger_for_file(0, "committed")

        # Verify the ledger row has the content_hash
        rows = test_ledger._conn.execute(
            "SELECT content_hash FROM ops WHERE status='committed'"
        ).fetchall()
        assert len(rows) >= 1
        assert any(r[0] == expected_hash for r in rows), (
            f"Expected hash {expected_hash[:12]}… not found in ledger rows: {rows}"
        )
    finally:
        os.unlink(ops_path)
