"""SourceAdapter contract + adapters (ADR-0014)."""
import pytest

from silica.config import CONFIG
from silica.sources.base import GroundedStub, RawItem, SourceAdapter
from silica.sources.prose import PROSE


class _Dummy:
    name = "dummy"

    def matches(self, target: str) -> bool:
        return target.endswith(".dummy")

    def read(self, target: str) -> RawItem:
        return RawItem(target=target, text="hello")

    def to_stub(self, item: RawItem) -> GroundedStub:
        return GroundedStub(lane="distill", body=item.text)


def test_conforming_adapter_satisfies_protocol():
    assert isinstance(_Dummy(), SourceAdapter)


def test_nonconforming_class_fails_protocol():
    class Nope:
        name = "nope"

    assert not isinstance(Nope(), SourceAdapter)


def test_rawitem_and_stub_defaults():
    item = RawItem(target="x", text="t")
    assert item.meta == {}
    stub = GroundedStub(lane="terminal")
    assert stub.note_path == "" and stub.body == ""


def test_prose_matches_md_and_txt_only():
    assert PROSE.matches("Inbox/a.md")
    assert PROSE.matches("notes.TXT")
    assert not PROSE.matches("m.py")
    assert not PROSE.matches("data.csv")


def test_prose_read_resolves_against_vault(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    (tmp_path / "a.md").write_text("# Hello\n\nbody", encoding="utf-8")
    item = PROSE.read("a.md")
    assert "Hello" in item.text and item.target == "a.md"


def test_prose_read_is_total_on_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    item = PROSE.read("nope.md")
    assert item.text == ""  # soft: dispatch must not block a batch on one bad path


def test_prose_stub_takes_distill_lane(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    stub = PROSE.to_stub(PROSE.read("a.md"))
    assert stub.lane == "distill"


import subprocess


@pytest.fixture
def code_repo(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "m.py").write_text('def hi():\n    """Say hi."""\n    return 1\n', encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    vault = tmp_path / ".silica"
    vault.mkdir()
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(CONFIG, "inbox_dir", "Inbox")
    return tmp_path, vault


def test_code_matches_source_not_prose():
    from silica.sources.code import CODE

    assert CODE.matches("m.py")
    assert not CODE.matches("a.md")
    assert not CODE.matches("data.csv")


def test_code_read_guards_path_escape(code_repo):
    from silica.sources.code import CODE

    with pytest.raises(ValueError):
        CODE.read("../outside.py")


def test_code_stub_is_terminal_with_grounding(code_repo):
    from silica.sources.code import CODE

    root, _vault = code_repo
    item = CODE.read("m.py")
    stub = CODE.to_stub(item)
    assert stub.lane == "terminal"
    # folder named after the source folder the file sits in (repo root → repo name)
    assert stub.note_path == f"{root.name}/m.md"
    assert "documents:" in stub.body and "code_ref:" in stub.body
    assert "def hi()" in stub.body and "return 1" not in stub.body  # skeleton, never full source


def test_code_note_dest_names_the_folder_after_the_nucleated_source():
    from silica.sources.code import code_note_dest

    root = "core/src/main/java/io/github/soulslight/controller"
    # folder = the nucleated folder's own name; stem = the path under it, dotted
    assert code_note_dest(f"{root}/GameController.java", root) == ("controller", "GameController")
    assert code_note_dest(f"{root}/commands/AttackCommand.java", root) == (
        "controller", "commands.AttackCommand")
    # a file outside the run falls back to its own parent — which is exactly what
    # a later run rooted there produces, so the wikilink written now resolves then
    outside = "core/src/main/java/io/github/soulslight/manager/GameManager.java"
    assert code_note_dest(outside, root) == ("manager", "GameManager")
    assert code_note_dest(outside, "core/src/main/java/io/github/soulslight/manager") == (
        "manager", "GameManager")
    # no run root (single-file nucleate) → the file's own folder
    assert code_note_dest("pkg/main.py") == ("pkg", "main")
    # repo-root file has no parent folder to borrow: the repo names it
    assert code_note_dest("m.py", "", "souls-light") == ("souls-light", "m")
    # relative subfolder root (e.g. "agent") rebased on top-level package "silica"
    assert code_note_dest("silica/agent/commit.py", "agent", "silica-harness") == ("silica", "agent.commit")


def test_code_stub_says_so_when_the_parser_fails(code_repo, monkeypatch):
    """A parser silica cannot drive must never render as "(no imports)": that
    ships a note asserting a real file is empty (tree-sitter 0.26 did exactly
    this, silently, for every language)."""
    from silica.kernel.code.codeast import ModuleSkeleton
    from silica.sources import code as code_mod

    monkeypatch.setattr(
        code_mod.codeast, "extract_skeleton",
        lambda *a, **k: ModuleSkeleton(path="m.py", language="python", parse_error=True),
    )
    body = code_mod.CODE.to_stub(code_mod.CODE.read("m.py")).body
    assert "Skeleton unavailable" in body
    assert "no imports" not in body and "no top-level symbols" not in body
    assert "code_ref:" in body  # staleness tracking still wired


def test_code_note_name_qualifies_and_folds_package_init():
    from silica.sources.code import code_note_name

    assert code_note_name("silica/kernel/code/codegraph.py") == "silica.kernel.code.codegraph"
    assert code_note_name("m.py") == "m"
    # package import links by the package name, not the __init__ file
    assert code_note_name("silica/kernel/code/codeast/__init__.py") == "silica.kernel.code.codeast"
    # degenerate root-level __init__: no parent package to fold to
    assert code_note_name("__init__.py") == "__init__"


def test_code_stub_wikilinks_first_party_imports(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "helper.py").write_text("def h():\n    return 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "main.py").write_text(
        "import os\nfrom pkg.helper import h\n\n\ndef run():\n    return h()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    vault = tmp_path / ".silica"
    vault.mkdir()
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(CONFIG, "inbox_dir", "Inbox")

    from silica.sources.code import CODE

    stub = CODE.to_stub(CODE.read("pkg/main.py"))
    assert stub.note_path == "pkg/main.md"   # folder = the source folder
    assert "[[helper]]" in stub.body  # first-party import → wikilink, same naming rule
    assert "`os`" in stub.body            # external dep → code span, not linked
    assert "[[os]]" not in stub.body


def test_adapter_for_dispatch():
    from silica.sources.registry import adapter_for

    assert adapter_for("a.md").name == "prose"
    assert adapter_for("m.py").name == "code"
    assert adapter_for("data.csv") is None


def test_adapter_for_respects_enabled_filter():
    from silica.sources.registry import adapter_for

    assert adapter_for("m.py", enabled=("prose",)) is None
    assert adapter_for("a.md", enabled=("prose",)).name == "prose"


def test_stage_routes_distill_lane_without_writing(tmp_path, monkeypatch):
    from silica.sources.prose import PROSE
    from silica.sources.registry import stage

    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    result = stage(PROSE, "a.md")
    assert result["status"] == "distill"


def test_stage_writes_terminal_stub(code_repo, monkeypatch):
    from silica.driver import fs_backend
    import silica.driver as driver_mod
    from silica.sources.code import CODE
    from silica.sources.registry import stage

    root, vault = code_repo
    monkeypatch.setattr(driver_mod, "DRIVER", fs_backend.ObsidianFSBackend(str(vault)))
    result = stage(CODE, "m.py")
    assert result["status"] == "ok"
    assert result["note_path"] == f"{root.name}/m.md"
    assert (vault / root.name / "m.md").is_file()
    assert result["meta"].get("code_ref")
