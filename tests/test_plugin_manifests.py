"""One artifact tree, three harnesses: the plugin manifests must agree on it."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from silica.onboarding.setup_client import MCP_COMMAND, skill_path

ROOT = Path(__file__).resolve().parent.parent


def _manifest(folder: str) -> dict:
    return json.loads((ROOT / folder / "plugin.json").read_text(encoding="utf-8"))


def _hooks() -> dict:
    return json.loads((ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8"))["hooks"]


def test_both_mcp_files_launch_the_command_setup_writes():
    # setup_client writes MCP_COMMAND into client configs; the plugin files
    # have to launch the very same thing or the install paths drift apart.
    # Two files because the dialects disagree on the wrapper key only:
    # Claude Code reads `mcpServers`, Codex reads `mcp_servers`.
    for name, key in (("mcp.json", "mcpServers"), ("mcp.codex.json", "mcp_servers")):
        doc = json.loads((ROOT / name).read_text(encoding="utf-8"))
        assert set(doc) == {key}, name
        srv = doc[key]["silica"]
        assert [srv["command"], *srv["args"]] == MCP_COMMAND, name
        assert "env" not in srv  # vault = the folder the client opened, never a pin


def test_claude_and_codex_manifests_point_at_the_same_parts():
    claude, codex = _manifest(".claude-plugin"), _manifest(".codex-plugin")
    assert (ROOT / claude["mcpServers"]).resolve() == ROOT / "mcp.json"
    assert (ROOT / codex["mcpServers"]).resolve() == ROOT / "mcp.codex.json"
    for m in (claude, codex):
        assert (ROOT / m["hooks"]).resolve() == ROOT / "hooks" / "hooks.json"
        assert (ROOT / m["skills"] / "silica" / "SKILL.md").resolve() == skill_path().resolve()
    assert (codex["name"], codex["version"]) == (claude["name"], claude["version"])
    # Codex: "Start them with ./" is a documented requirement for manifest paths.
    assert all(codex[k].startswith("./") for k in ("skills", "mcpServers", "hooks"))


def test_both_marketplaces_offer_the_same_plugin_from_the_repo_root():
    claude = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    codex = json.loads((ROOT / ".agents" / "plugins" / "marketplace.json").read_text(encoding="utf-8"))
    assert claude["name"] == codex["name"]
    (c_entry,), (x_entry,) = claude["plugins"], codex["plugins"]
    assert c_entry["name"] == x_entry["name"] == _manifest(".codex-plugin")["name"]
    # Both sources are the repo root, where both plugin manifests sit.
    assert c_entry["source"] == "./"
    assert x_entry["source"] in ("./", {"source": "local", "path": "./"})


def test_hooks_fire_only_shipped_subcommands():
    launcher = " ".join(MCP_COMMAND[:-1]).replace("silica-harness[mcp]", "'silica-harness[mcp]'")
    hooks = _hooks()
    assert set(hooks) == {"SessionStart", "SessionEnd", "PreCompact"}
    for entries in hooks.values():
        for h in (h for e in entries for h in e["hooks"]):
            assert h["type"] == "command"
            assert h["command"].startswith(launcher + " "), h["command"]
            assert h["command"].removeprefix(launcher + " ").split()[0] in {"hook", "capture"}


def test_session_start_hook_cannot_hold_the_session_hostage():
    # Claude Code waits on SessionStart before the first prompt; its default
    # hook timeout is ten minutes.
    for h in (h for e in _hooks()["SessionStart"] for h in e["hooks"]):
        assert h["timeout"] <= 60


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not on PATH")
def test_claude_plugin_validate_passes():
    r = subprocess.run(["claude", "plugin", "validate", str(ROOT)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


PLUGIN_FILES = (
    ".claude-plugin/plugin.json", ".claude-plugin/marketplace.json",
    ".codex-plugin/plugin.json", ".agents/plugins/marketplace.json",
    "mcp.json", "mcp.codex.json", "hooks/hooks.json",
)


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(),
                    reason="not a git checkout")
def test_plugin_files_are_not_gitignored():
    # A plugin file that passes the tests here and is missing from every
    # clone is the one failure the tests above cannot see: `.agents/` was
    # ignored when the Codex marketplace was added.
    ignored = subprocess.run(["git", "check-ignore", *PLUGIN_FILES], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    assert not ignored, f"gitignored plugin files: {ignored}"
