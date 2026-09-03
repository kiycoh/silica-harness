import subprocess
from pathlib import Path


from silica.kernel.code import gitstate


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


def test_find_repo_root_returns_toplevel(tmp_path):
    _init_repo(tmp_path)
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert gitstate.find_repo_root(sub) == tmp_path.resolve()


def test_find_repo_root_none_outside_repo(tmp_path):
    assert gitstate.find_repo_root(tmp_path) is None


def test_head_ref_none_on_empty_repo(tmp_path):
    _init_repo(tmp_path)
    assert gitstate.head_ref(tmp_path) is None


def test_head_ref_returns_sha_after_commit(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, "f.py", "x=1\n", "init")
    ref = gitstate.head_ref(tmp_path)
    assert isinstance(ref, str) and len(ref) == 40


def test_is_ignored_batch(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (tmp_path / "keep.md").write_text("a", encoding="utf-8")
    (tmp_path / "drop.log").write_text("b", encoding="utf-8")
    ignored = gitstate.is_ignored(tmp_path, [Path("keep.md"), Path("drop.log")])
    assert ignored == {Path("drop.log")}


def test_is_ignored_empty_without_git(tmp_path):
    # not a repo → nothing reported ignored, no raise
    assert gitstate.is_ignored(tmp_path, [Path("x.md")]) == set()


def test_latest_shas_one_walk_many_paths(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, "src/m.py", "v1\n", "add m")
    old = gitstate.head_ref(tmp_path)
    _commit(tmp_path, "src/n.py", "v1\n", "add n")
    head = gitstate.head_ref(tmp_path)
    shas = gitstate.latest_shas(tmp_path, ["src/m.py", "src/n.py", "ghost.py"])
    assert shas == {"src/m.py": old, "src/n.py": head}  # no-history path absent


def test_latest_shas_sees_rename_commit(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, "old.py", "v1\n", "add old")
    subprocess.run(["git", "mv", "old.py", "new.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "rename"], cwd=tmp_path, check=True)
    shas = gitstate.latest_shas(tmp_path, ["new.py"])
    assert shas == {"new.py": gitstate.head_ref(tmp_path)}


def test_latest_shas_soft_outside_repo(tmp_path):
    assert gitstate.latest_shas(tmp_path, ["x.py"]) == {}


def test_commits_since_lists_intervening(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, "src/m.py", "v1\n", "c1")
    base = gitstate.head_ref(tmp_path)
    _commit(tmp_path, "src/m.py", "v2\n", "c2")
    _commit(tmp_path, "src/m.py", "v3\n", "c3")
    commits = gitstate.commits_since(tmp_path, base, "src/m.py")
    subjects = [c.subject for c in commits]
    assert subjects == ["c3", "c2"]  # newest-first, base excluded


def test_commit_docs_commits_only_vault_paths(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, "README.md", "r\n", "seed")  # give repo a HEAD
    vault = tmp_path / "docs"
    vault.mkdir()
    (vault / "note.md").write_text("hello\n", encoding="utf-8")
    sha = gitstate.commit_docs(
        tmp_path, vault, [vault / "note.md"], "silica: write 1 note"
    )
    assert sha == gitstate.head_ref(tmp_path)
    log = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=tmp_path,
        capture_output=True, text=True,
    )
    assert log.stdout.strip() == "silica: write 1 note"


def test_commit_docs_refuses_path_outside_vault(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, "README.md", "r\n", "seed")
    vault = tmp_path / "docs"
    vault.mkdir()
    outside = tmp_path / "src.py"
    outside.write_text("danger\n", encoding="utf-8")
    sha = gitstate.commit_docs(tmp_path, vault, [outside], "should refuse")
    assert sha is None  # path outside vault → no commit


def test_commit_docs_none_without_git(tmp_path):
    vault = tmp_path / "docs"
    vault.mkdir()
    (vault / "n.md").write_text("x", encoding="utf-8")
    assert gitstate.commit_docs(tmp_path, vault, [vault / "n.md"], "m") is None
