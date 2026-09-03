"""Test that the WRITE phase uses bulk_write_atomic per-note atomicity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import silica.config
import silica.driver
import silica.kernel.progress as prog_mod

from silica.kernel.write.ops import Op, OpType
from silica.router import states
from silica.router.orchestrator import InjectorFSM


def _patch(path: str, src: str = "m.md") -> Op:
    return Op(op=OpType.patch, heading="H", source_basename=src, path=path, snippet="x", hub="Hub")


@pytest.fixture
def vault_fsm(tmp_path, monkeypatch):
    """Minimal FSM setup against a temp vault for write-phase unit tests."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    monkeypatch.setattr(silica.config.CONFIG, "vault_path", str(vault_dir))
    silica.driver._driver = None

    # Redirect ProgressLedger to tmp so tests don't pollute ~/.silica/runs
    monkeypatch.setattr(prog_mod, "_RUNS_DIR", tmp_path / "runs")

    class _Ctx:
        def __init__(self):
            self.vault = vault_dir

        def note(self, rel: str, content: str = "") -> str:
            p = vault_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return str(p)

        def read(self, path: str) -> str:
            return Path(path).read_text(encoding="utf-8")

        def make_fsm(self) -> InjectorFSM:
            inbox = vault_dir / "inbox.md"
            inbox.write_text("# test\ncontent\n", encoding="utf-8")
            fsm = InjectorFSM(inbox_file=str(inbox), target_dir="Concepts")
            # Patch noisy methods that require full pipeline setup
            fsm._progress_note = lambda *a, **kw: None
            fsm.manifest.record = lambda *a, **kw: None
            fsm.manifest.save = lambda *a, **kw: None
            # Set up chunk state
            fsm._file_content_hashes = ["testhash"]
            fsm._file_chunks = [{"source_file": str(inbox), "chunks": [{}]}]
            fsm._chunk_flat_to_fi_ci = {0: (0, 0)}
            return fsm

        def ops_path(self, ops: list[Op]) -> str:
            p = str(tmp_path / "ops.json")
            Path(p).write_text(json.dumps([o.model_dump() for o in ops]), encoding="utf-8")
            return p

    yield _Ctx()
    silica.driver._driver = None


def test_write_hands_chunk_to_residue_dispatch(vault_fsm, monkeypatch):
    """WRITE ends by offering the chunk to the residue pre-dispatch seam
    (which itself decides last-chunk/draft/failure), so the check call rides
    the mechanical tail instead of the file boundary."""
    from unittest.mock import MagicMock
    a = vault_fsm.note("A.md", "---\n---\nseed\n")
    monkeypatch.setattr("silica.tools.composed.silica_lint",
                        lambda *a_, **kw: {"success": True, "errors": []})
    dispatch = MagicMock()
    monkeypatch.setattr("silica.router.states.finalize.maybe_dispatch_residue_check", dispatch)

    fsm = vault_fsm.make_fsm()
    fsm.context["chunk"] = {"ops_path": vault_fsm.ops_path([_patch(a)])}
    states.write.handle_write(fsm)
    dispatch.assert_called_once_with(fsm)


def test_injector_write_defers_failing_note_keeps_siblings(vault_fsm, monkeypatch):
    """A chunk with one lint-failing note: others commit, bad one deferred."""
    a = vault_fsm.note("A.md", "---\n---\nseed\n")
    bad = vault_fsm.note("Bad.md", "---\n---\nseed\n")
    c = vault_fsm.note("C.md", "---\n---\nseed\n")

    def fake_lint(note_name, op_type="", hub=""):
        # Diff-aware patch lint: the bad note only fails once the patch appends
        # its block, so the violation counts as newly introduced.
        from silica.driver import DRIVER
        introduced = "Bad" in str(note_name) and "Additional notes" in DRIVER.read_note(note_name).content
        return {"success": not introduced, "errors": ["e"] if introduced else []}
    monkeypatch.setattr("silica.tools.composed.silica_lint", fake_lint)

    fsm = vault_fsm.make_fsm()
    fsm.context["chunk"] = {"ops_path": vault_fsm.ops_path([_patch(a), _patch(bad), _patch(c)])}

    # WRITE should use bulk_write_atomic — per-note lint is applied
    states.write.handle_write(fsm)

    # Committed notes got the snippet; failed note was reverted
    assert "x" in vault_fsm.read(a)
    assert "x" in vault_fsm.read(c)
    assert "x" not in vault_fsm.read(bad)

    write_ctx = fsm.context.get("write", {})
    failed_paths = {f["path"] for f in write_ctx.get("failed", [])}
    assert bad in failed_paths

    # Surviving inverses should be accumulated on the FSM
    committed_inv_paths = {inv.path for _, inv, _ in fsm._run_inverses}
    assert a in committed_inv_paths
    assert c in committed_inv_paths
    assert bad not in committed_inv_paths


def test_clean_batch_parity_old_vs_new(tmp_path, monkeypatch):
    """All-clean batch: bulk_write_atomic produces same vault state as execute_operations."""
    import silica.config
    import silica.driver

    # Branch A — old path (execute_operations)
    vault_a = tmp_path / "vault_a"
    vault_a.mkdir()
    for name in ("X.md", "Y.md", "Z.md"):
        (vault_a / name).write_text(f"---\n---\nseed_{name}\n", encoding="utf-8")

    monkeypatch.setattr(silica.config.CONFIG, "vault_path", str(vault_a))
    silica.driver._driver = None

    from silica.kernel.write.bulk import execute_operations
    ops_a = [
        Op(op=OpType.patch, heading="H", source_basename="m.md",
           path=str(vault_a / name), snippet="fact", hub="Hub")
        for name in ("X.md", "Y.md", "Z.md")
    ]
    execute_operations(ops_a)
    content_a = {name: (vault_a / name).read_text(encoding="utf-8") for name in ("X.md", "Y.md", "Z.md")}

    # Branch B — new path (bulk_write_atomic, lint=False)
    vault_b = tmp_path / "vault_b"
    vault_b.mkdir()
    for name in ("X.md", "Y.md", "Z.md"):
        (vault_b / name).write_text(f"---\n---\nseed_{name}\n", encoding="utf-8")

    monkeypatch.setattr(silica.config.CONFIG, "vault_path", str(vault_b))
    silica.driver._driver = None

    from silica.kernel.write.atomic_write import bulk_write_atomic
    ops_b = [
        Op(op=OpType.patch, heading="H", source_basename="m.md",
           path=str(vault_b / name), snippet="fact", hub="Hub")
        for name in ("X.md", "Y.md", "Z.md")
    ]
    result = bulk_write_atomic(ops_b, lint=False)

    assert result.ok is True
    assert result.total == 3
    for name in ("X.md", "Y.md", "Z.md"):
        content_b = (vault_b / name).read_text(encoding="utf-8")
        assert content_b == content_a[name], f"Parity failure for {name}"


def test_revert_restores_clean_batch(tmp_path, monkeypatch):
    """/revert end-to-end: bulk_write_atomic + journal record → revert_run restores original."""
    import hashlib
    import silica.config
    import silica.driver
    from silica.kernel.write.atomic_write import bulk_write_atomic
    from silica.kernel.write.undo_journal import UndoJournalStore, revert_run

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "X.md").write_text("ORIGINAL x", encoding="utf-8")

    monkeypatch.setattr(silica.config.CONFIG, "vault_path", str(vault))
    silica.driver._driver = None

    op = Op(op=OpType.patch, heading="H", source_basename="m.md",
            path=str(vault / "X.md"), snippet="patch", hub="Hub")

    store = UndoJournalStore(tmp_path / "j.db")
    run_id = store.start_run("inbox/s.md")

    result = bulk_write_atomic([op], lint=False)
    assert result.ok is True

    # Record inverses with post-write hash
    for r in result.committed:
        post_content = (vault / "X.md").read_text(encoding="utf-8")
        post_hash = hashlib.sha256(post_content.encode()).hexdigest()
        for inv in r.inverses:
            store.record(run_id, inv, post_hash)

    # Verify patch was applied
    assert "patch" in (vault / "X.md").read_text(encoding="utf-8")

    # Revert
    out = revert_run(run_id, store=store)
    assert str(vault / "X.md") in out["reverted"]
    assert (vault / "X.md").read_text(encoding="utf-8") == "ORIGINAL x"
