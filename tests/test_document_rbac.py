import subprocess
from pathlib import Path


from silica.config import CONFIG


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    f = path / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_document_writes_only_under_inbox(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    vault = tmp_path / "docs"
    vault.mkdir()
    (vault / "existing.md").write_text("keep me\n", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(CONFIG, "inbox_dir", "Inbox")
    from silica.driver import fs_backend
    import silica.driver as driver_mod
    monkeypatch.setattr(driver_mod, "DRIVER", fs_backend.ObsidianFSBackend(str(vault)))

    from silica.tools.codedocs_tool import silica_document
    before = {p.relative_to(vault).as_posix() for p in vault.rglob("*.md")}
    silica_document(path="a.py")
    after = {p.relative_to(vault).as_posix() for p in vault.rglob("*.md")}

    new = after - before
    assert new, "tool should have created a note"
    assert all(p.startswith("Inbox/") for p in new), f"wrote outside Inbox: {new}"
    # existing vault note untouched
    assert (vault / "existing.md").read_text(encoding="utf-8") == "keep me\n"
