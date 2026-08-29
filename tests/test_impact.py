# tests/test_impact.py
"""/impact — changed files → documenting notes + 1-hop import neighbors (spec §5)."""
import subprocess
from pathlib import Path

from silica.kernel.code import codegraph


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


def _commit_all(path: Path, msg: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=path, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                          capture_output=True, text=True).stdout.strip()


def _note(vault: Path, name: str, documents: str) -> None:
    vault.mkdir(parents=True, exist_ok=True)
    (vault / name).write_text(
        f"---\ndocuments:\n  - {documents}\ncode_ref: x\n---\n\nbody\n", encoding="utf-8")


def test_impact_lists_documenting_and_neighbor_notes(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "core.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "pkg" / "user_a.py").write_text("from .core import f\n", encoding="utf-8")
    (tmp_path / "pkg" / "user_b.py").write_text("from .core import f\n", encoding="utf-8")
    vault = tmp_path / "docs"
    _note(vault, "core.md", "pkg/core.py")
    _note(vault, "user_a.md", "pkg/user_a.py")
    _note(vault, "user_b.md", "pkg/user_b.py")
    _commit_all(tmp_path, "seed")
    monkeypatch.setattr(codegraph, "store_path", lambda: tmp_path / "cg.json")

    # uncommitted signature change to a file imported by two others
    (tmp_path / "pkg" / "core.py").write_text("def f(x, y):\n    return x\n", encoding="utf-8")
    entries = codegraph.compute_impact(vault)
    assert entries is not None
    e = next(en for en in entries if en.path == "pkg/core.py")
    assert e.change_level == "structural"
    assert any("signature changed: f" in d for d in e.details)
    assert e.fan_in == 2
    assert e.notes == ["core.md"]
    assert e.neighbor_notes == ["user_a.md", "user_b.md"]


def test_impact_range_and_no_repo(tmp_path, monkeypatch):
    assert codegraph.compute_impact(tmp_path / "nowhere") is None  # no repo → None
    _init_repo(tmp_path)
    (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    ref0 = _commit_all(tmp_path, "c1")
    (tmp_path / "m.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    ref1 = _commit_all(tmp_path, "c2")
    vault = tmp_path / "docs"
    _note(vault, "m.md", "m.py")
    _commit_all(tmp_path, "notes")
    monkeypatch.setattr(codegraph, "store_path", lambda: tmp_path / "cg.json")
    entries = codegraph.compute_impact(vault, f"{ref0}..{ref1}")
    e = next(en for en in entries if en.path == "m.py")
    assert e.change_level == "cosmetic"   # body-only commit in the range


# --- silica_impact: the MCP-facing wrapper --------------------------------


def test_impact_tool_serializes_entries_and_flags_the_cap(monkeypatch):
    import silica.config
    from silica.tools.codedocs_tool import _IMPACT_CAP, silica_impact

    monkeypatch.setattr(silica.config.CONFIG, "vault_path", "/v")
    fake = [
        codegraph.ImpactEntry(
            path=f"m{i}.py", change_level="structural", details=["+ f"],
            fan_in=i, notes=["m.md"], neighbor_notes=[],
        )
        for i in range(_IMPACT_CAP + 1)
    ]
    monkeypatch.setattr(codegraph, "compute_impact", lambda v, r=None: fake)
    out = silica_impact()
    assert out["status"] == "ok" and out["total"] == _IMPACT_CAP + 1
    assert len(out["entries"]) == _IMPACT_CAP and out["truncated"] is True
    assert out["entries"][0] == {
        "path": "m0.py", "change_level": "structural", "details": ["+ f"],
        "fan_in": 0, "notes": ["m.md"], "neighbor_notes": [],
    }


def test_impact_tool_degrades_soft(monkeypatch):
    import silica.config
    from silica.tools.codedocs_tool import silica_impact

    monkeypatch.setattr(silica.config.CONFIG, "vault_path", "")
    assert silica_impact()["status"] == "error"  # no vault configured

    monkeypatch.setattr(silica.config.CONFIG, "vault_path", "/v")
    monkeypatch.setattr(codegraph, "compute_impact", lambda v, r=None: None)
    assert silica_impact()["status"] == "no_repo"  # outside git: soft, not a crash
