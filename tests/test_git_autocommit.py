"""Tests for the git auto-commit hook in the orchestrator.

After a write batch is committed to the vault, the orchestrator optionally
snapshots the touched vault paths to git. This is the git safety net behind
SILICA_GIT_COMMIT=auto: an additive snapshot on top of the undo journal
(ADR-0002), never a replacement. The helper is best-effort and must never raise.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


from silica.kernel.write.ops import Op, OpType
from silica.router.orchestrator import _commit_docs_for_ops


# ---------------------------------------------------------------------------
# Helpers shared with test_gitstate.py
# ---------------------------------------------------------------------------

def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


def _commit(path: Path, rel: str, text: str, msg: str) -> None:
    f = path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "--", rel], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg, "--", rel], cwd=path, check=True)


def _head(path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path, capture_output=True, text=True,
    )
    out = proc.stdout.strip()
    return out if len(out) == 40 else None


# ---------------------------------------------------------------------------
# Op factories — same shape as test_freshness_hook.py
# ---------------------------------------------------------------------------

def _write_op(path: str) -> Op:
    return Op(op=OpType.write, heading=path, source_basename="inbox.md", path=path, snippet="x")


def _patch_op(path: str) -> Op:
    return Op(op=OpType.patch, heading=path, source_basename="inbox.md", path=path, snippet="x")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_auto_commits_the_batch(tmp_path):
    """git_commit='auto' commits the write/patch batch; returns new HEAD sha."""
    _init_repo(tmp_path)
    _commit(tmp_path, "seed.md", "seed\n", "seed commit")
    seed_sha = _head(tmp_path)

    vault = tmp_path / "docs"
    vault.mkdir()
    (vault / "Alpha.md").write_text("# Alpha\n", encoding="utf-8")
    (vault / "Beta.md").write_text("# Beta\n", encoding="utf-8")

    ops = [
        _write_op("Alpha.md"),
        _patch_op("Beta.md"),
    ]
    committed_paths = {"Alpha.md", "Beta.md"}

    sha = _commit_docs_for_ops(
        ops, committed_paths,
        vault=str(vault),
        git_commit="auto",
    )

    assert sha is not None
    assert isinstance(sha, str) and len(sha) == 40
    assert sha != seed_sha, "HEAD must have advanced"

    log = subprocess.run(
        ["git", "log", "-1", "--format=%s"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert log.stdout.strip() == "silica: write 2 note(s)"

    new_head = _head(tmp_path)
    assert sha == new_head


def test_off_is_a_noop(tmp_path):
    """git_commit='off' returns None and leaves HEAD unchanged."""
    _init_repo(tmp_path)
    _commit(tmp_path, "seed.md", "seed\n", "seed commit")
    seed_sha = _head(tmp_path)

    vault = tmp_path / "docs"
    vault.mkdir()
    (vault / "Alpha.md").write_text("# Alpha\n", encoding="utf-8")

    ops = [_write_op("Alpha.md")]
    committed_paths = {"Alpha.md"}

    sha = _commit_docs_for_ops(
        ops, committed_paths,
        vault=str(vault),
        git_commit="off",
    )

    assert sha is None
    assert _head(tmp_path) == seed_sha, "HEAD must be unchanged"


def test_vault_outside_any_repo(tmp_path):
    """vault not inside a git repo → returns None, no raise."""
    vault = tmp_path / "docs"
    vault.mkdir()
    (vault / "Note.md").write_text("x\n", encoding="utf-8")

    ops = [_write_op("Note.md")]
    committed_paths = {"Note.md"}

    sha = _commit_docs_for_ops(
        ops, committed_paths,
        vault=str(vault),
        git_commit="auto",
    )
    assert sha is None


def test_only_committed_paths_are_included(tmp_path):
    """Only ops whose touched_ref is in committed_paths end up in the commit."""
    _init_repo(tmp_path)
    _commit(tmp_path, "seed.md", "seed\n", "seed commit")

    vault = tmp_path / "docs"
    vault.mkdir()
    (vault / "Kept.md").write_text("# Kept\n", encoding="utf-8")
    (vault / "Dropped.md").write_text("# Dropped\n", encoding="utf-8")

    ops = [
        _write_op("Kept.md"),
        _write_op("Dropped.md"),
    ]
    # Only Kept.md was atomically committed; Dropped.md failed/was deferred
    committed_paths = {"Kept.md"}

    sha = _commit_docs_for_ops(
        ops, committed_paths,
        vault=str(vault),
        git_commit="auto",
    )

    assert sha is not None
    show = subprocess.run(
        ["git", "show", "--name-only", "--format="],
        cwd=tmp_path, capture_output=True, text=True,
    )
    changed_files = [ln for ln in show.stdout.splitlines() if ln.strip()]
    # normalise to basenames for comparison
    basenames = [Path(f).name for f in changed_files]
    assert "Kept.md" in basenames
    assert "Dropped.md" not in basenames


def test_non_write_patch_ops_ignored(tmp_path):
    """An overwrite op (not write/patch) must not trigger a commit."""
    _init_repo(tmp_path)
    _commit(tmp_path, "seed.md", "seed\n", "seed commit")
    seed_sha = _head(tmp_path)

    vault = tmp_path / "docs"
    vault.mkdir()
    (vault / "X.md").write_text("# X\n", encoding="utf-8")

    overwrite = Op(
        op=OpType.overwrite, heading="X", source_basename="i.md",
        path="X.md", content="# X\n",
    )
    committed_paths = {"X.md"}

    sha = _commit_docs_for_ops(
        [overwrite], committed_paths,
        vault=str(vault),
        git_commit="auto",
    )

    assert sha is None
    assert _head(tmp_path) == seed_sha, "HEAD must be unchanged for non-write ops"
