# tests/test_atomic_write.py
from silica.kernel.write.ops import Op, OpType
from silica.kernel.write.atomic_write import commit_note_atomic, NoteCommitResult, bulk_write_atomic, AtomicBulkResult


def _patch_op(path: str, snippet: str = "body") -> Op:
    return Op(op=OpType.patch, heading="H", source_basename="src.md",
              path=path, snippet=snippet, hub="Hub")


def test_clean_patch_commits_and_returns_inverse(tmp_vault, monkeypatch):
    target = tmp_vault.note("People/Ada.md", "---\n---\nseed\n")
    res = commit_note_atomic(_patch_op(target), lint=False)
    assert isinstance(res, NoteCommitResult)
    assert res.ok is True
    assert res.path == target
    assert res.reverted is False
    assert res.post_hash is not None
    assert len(res.inverses) >= 1
    assert "body" in tmp_vault.read(target)


def test_lint_failure_reverts_only_this_note(tmp_vault, monkeypatch):
    target = tmp_vault.note("Areas/Roadmap.md", "---\n---\nseed\n")
    original = tmp_vault.read(target)

    # Diff-aware patch lint: fail only once the patch has appended its block, so
    # the violation is NEWLY introduced (a pre-existing one wouldn't revert).
    def fake_lint(note_name, op_type="", hub=""):
        from silica.driver import DRIVER
        introduced = "Additional notes" in DRIVER.read_note(note_name).content
        return {"success": not introduced, "errors": ["bad link"] if introduced else []}
    monkeypatch.setattr("silica.tools.composed.silica_lint", fake_lint)
    res = commit_note_atomic(_patch_op(target), lint=True)
    assert res.ok is False
    assert res.reverted is True
    assert "lint failed" in res.error
    assert tmp_vault.read(target) == original


def test_failing_sibling_does_not_roll_back_others(tmp_vault, monkeypatch):
    a = tmp_vault.note("People/Ada.md", "---\n---\nseed\n")
    b = tmp_vault.note("Areas/Roadmap.md", "---\n---\nseed\n")
    c = tmp_vault.note("People/Grace.md", "---\n---\nseed\n")

    def fake_lint(note_name, op_type="", hub=""):
        from silica.driver import DRIVER
        introduced = "Roadmap" in note_name and "Additional notes" in DRIVER.read_note(note_name).content
        return {"success": not introduced, "errors": ["bad"] if introduced else []}
    monkeypatch.setattr("silica.tools.composed.silica_lint", fake_lint)

    ops = [_patch_op(a), _patch_op(b), _patch_op(c)]
    result = bulk_write_atomic(ops, lint=True)

    assert isinstance(result, AtomicBulkResult)
    assert result.total == 3
    assert {r.path for r in result.committed} == {a, c}
    assert [r.path for r in result.failed] == [b]
    assert "body" in tmp_vault.read(a)
    assert "body" in tmp_vault.read(c)
    assert "body" not in tmp_vault.read(b)
    assert result.ok is False


def test_commit_note_atomic_waits_for_path_lease(tmp_vault):
    """The FSM ingest path must serialize with lease-holding writers (subagent
    commit_ops, MCP note tools). Without the lease, patch's read-modify-write
    interleaves with a concurrent writer and one append is silently lost."""
    import threading
    from silica.kernel.workqueue import path_lease

    target = tmp_vault.note("People/Ada.md", "---\n---\nseed\n")
    done = threading.Event()

    def worker():
        commit_note_atomic(_patch_op(target), lint=False)
        done.set()

    with path_lease(target):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        assert not done.wait(0.3), "write landed while another writer held the lease"
    assert done.wait(5), "write never completed after the lease was released"
    t.join(5)
    assert "body" in tmp_vault.read(target)
