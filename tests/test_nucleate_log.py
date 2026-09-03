"""CLEANUP-phase wiring of the log.md journal (silica.router.states.finalize).

`_log_nucleate_completion` is the seam: a pure projection of already-recorded
manifest entries + deferred-store state onto one log.md line, exercised
directly against a minimal fake FSM rather than the full injector pipeline
(that machinery is covered elsewhere; this file's job is the seam only).
"""
from __future__ import annotations

import types
from pathlib import Path

from silica.kernel.progress import RunManifestEntry
from silica.kernel.recall.run_log import DEFAULT_LOG_FILENAME
from silica.router.states import finalize


def _fake_fsm(entries, run_id, content_hashes):
    return types.SimpleNamespace(
        manifest=types.SimpleNamespace(entries=entries),
        progress=types.SimpleNamespace(run_id=run_id),
        _file_content_hashes=content_hashes,
    )


def _entry(source_basename: str, op: str, title: str = "X") -> RunManifestEntry:
    return RunManifestEntry(
        title=title, path=f"Dir/{title}", parent=None, cluster_id=-1,
        source_basename=source_basename, op=op,
    )


def test_writes_projected_new_and_patch_counts(tmp_vault):
    from silica.config import CONFIG

    entries = [
        _entry("lezione-03.md", "write", "A"),
        _entry("lezione-03.md", "write", "B"),
        _entry("lezione-03.md", "patch", "C"),
        _entry("other.md", "write", "Other"),  # different source — must not be counted
    ]
    fsm = _fake_fsm(entries, run_id="deadbeef1234", content_hashes=["hash0"])

    finalize._log_nucleate_completion(fsm, 0, "Inbox/lezione-03.md")

    content = (Path(CONFIG.vault_path) / DEFAULT_LOG_FILENAME).read_text(encoding="utf-8")
    assert "nucleate `lezione-03.md` → 2 new, 1 patch, 0 deferred" in content
    assert "run deadbeef" in content


def test_includes_deferred_count_from_deferred_store(tmp_vault, monkeypatch, tmp_path):
    from silica.config import CONFIG
    from silica.kernel.recall.deferred import get_deferred_store

    # conftest isolates the default store; populate it through the public seam.
    get_deferred_store().put(
        content_hash="hash0",
        source_path="Inbox/lezione-03.md",
        target_dir="Dir",
        hub=None,
        rejected_ops=[{"op": "write", "path": "Dir/X.md"}, {"op": "write", "path": "Dir/Y.md"}],
    )

    fsm = _fake_fsm([], run_id="cafef00dabcd", content_hashes=["hash0"])

    finalize._log_nucleate_completion(fsm, 0, "Inbox/lezione-03.md")

    content = (Path(CONFIG.vault_path) / DEFAULT_LOG_FILENAME).read_text(encoding="utf-8")
    assert "→ 0 new, 0 patch, 2 deferred" in content


def test_two_nucleates_two_lines_in_order(tmp_vault):
    from silica.config import CONFIG

    fsm1 = _fake_fsm([_entry("one.md", "write", "A")], run_id="runidone1234", content_hashes=["h1"])
    finalize._log_nucleate_completion(fsm1, 0, "Inbox/one.md")

    fsm2 = _fake_fsm([_entry("two.md", "write", "B")], run_id="runidtwo5678", content_hashes=["h2"])
    finalize._log_nucleate_completion(fsm2, 0, "Inbox/two.md")

    lines = (Path(CONFIG.vault_path) / DEFAULT_LOG_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "one.md" in lines[0]
    assert "two.md" in lines[1]


def test_multi_file_same_run_id_logs_one_line_per_file(tmp_vault):
    """One FSM run over N inbox_files shares one run_id; CLEANUP fires once per
    file. Every file must get its own line (idempotency must NOT be keyed on
    run_id alone), and a resume of the same run must still not duplicate any."""
    from silica.config import CONFIG

    entries = [
        _entry("one.md", "write", "A"),
        _entry("two.md", "write", "B"),
        _entry("two.md", "patch", "C"),
    ]
    fsm = _fake_fsm(entries, run_id="sharedrunid1", content_hashes=["h1", "h2"])

    finalize._log_nucleate_completion(fsm, 0, "Inbox/one.md")
    finalize._log_nucleate_completion(fsm, 1, "Inbox/two.md")

    log_file = Path(CONFIG.vault_path) / DEFAULT_LOG_FILENAME
    lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    assert "nucleate `one.md` → 1 new, 0 patch, 0 deferred" in lines[0]
    assert "nucleate `two.md` → 1 new, 1 patch, 0 deferred" in lines[1]

    # Resume of the same run: both files re-enter CLEANUP → still two lines.
    finalize._log_nucleate_completion(fsm, 0, "Inbox/one.md")
    finalize._log_nucleate_completion(fsm, 1, "Inbox/two.md")
    lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2


def test_never_raises_on_broken_fsm(tmp_vault):
    """Best-effort: a fsm missing the expected attributes must not blow up CLEANUP."""
    broken_fsm = types.SimpleNamespace()
    finalize._log_nucleate_completion(broken_fsm, 0, "Inbox/x.md")  # must not raise


# ---------------------------------------------------------------------------
# W1 (survey-provenance spec §10): extractive-gate rejections are counted in
# the run digest — a dropped claim is never invisible. The deferred bundle is
# the durable record of what the retries could not save; ops rejected by the
# extractive span gate carry a "extractive:" rejection_reason stamp.
# ---------------------------------------------------------------------------


def test_files_summary_counts_extractive_rejections(tmp_vault):
    from silica.kernel.recall.deferred import get_deferred_store

    get_deferred_store().put(
        content_hash="hash0",
        source_path="Inbox/lezione-03.md",
        target_dir="Dir",
        hub=None,
        rejected_ops=[
            {"op": "write", "path": "Dir/X.md", "heading": "X"},
            {"op": "write", "path": "Dir/Y.md", "heading": "Y"},
        ],
        rejection_reasons={
            "Dir/X.md": "extractive: 2 body line(s) not verbatim from source: foo",
            "Dir/Y.md": "snippet too short (10 < 80 chars)",
        },
    )
    fsm = _fake_fsm([], run_id="cafecafe0000", content_hashes=["hash0"])
    fsm.context = {}

    finalize._log_nucleate_completion(fsm, 0, "Inbox/lezione-03.md")

    [entry] = fsm.context["files_summary"]
    assert entry["deferred"] == 2
    assert entry["extractive_rejected"] == 1


def test_files_summary_omits_extractive_key_when_none(tmp_vault):
    from silica.kernel.recall.deferred import get_deferred_store

    get_deferred_store().put(
        content_hash="hash0",
        source_path="Inbox/lezione-03.md",
        target_dir="Dir",
        hub=None,
        rejected_ops=[{"op": "write", "path": "Dir/X.md", "heading": "X"}],
        rejection_reasons={"Dir/X.md": "snippet too short (10 < 80 chars)"},
    )
    fsm = _fake_fsm([], run_id="cafecafe0001", content_hashes=["hash0"])
    fsm.context = {}

    finalize._log_nucleate_completion(fsm, 0, "Inbox/lezione-03.md")

    [entry] = fsm.context["files_summary"]
    assert entry["deferred"] == 1
    assert "extractive_rejected" not in entry
