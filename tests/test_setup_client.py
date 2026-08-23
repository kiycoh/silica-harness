"""`silica setup <client>`: merge, never clobber, and stay idempotent."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml

from silica.onboarding import setup_client


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    # codex/dsh setups copy the skill under ~/.agents; the suite must never
    # write into the developer's real home.
    monkeypatch.setenv("HOME", str(tmp_path))


def test_codex_appends_block(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "gpt-5"\n', encoding="utf-8")
    assert setup_client.run_setup(["codex", "--config", str(cfg)]) == 0
    parsed = tomllib.loads(cfg.read_text(encoding="utf-8"))
    assert parsed["model"] == "gpt-5"  # existing content survives
    assert parsed["mcp_servers"]["silica"]["command"] == "uvx"
    # The [mcp] extra is the point of the block, and rich markup eats it if the
    # payload is ever printed or written through a markup-enabled path.
    assert parsed["mcp_servers"]["silica"]["args"] == ["--from", "silica-harness[mcp]", "silica", "mcp"]


def test_no_client_gets_a_pinned_vault(tmp_path):
    """The generated config must not carry SILICA_VAULT.

    The server resolves the vault from the working directory its client spawns
    it in, so a pin written once at setup time would serve that one vault to
    every project the user ever opens.
    """
    toml_cfg = tmp_path / "config.toml"
    setup_client.run_setup(["codex", "--config", str(toml_cfg)])
    assert "SILICA_VAULT" not in toml_cfg.read_text(encoding="utf-8")

    json_cfg = tmp_path / "opencode.json"
    setup_client.run_setup(["opencode", "--config", str(json_cfg)])
    assert "SILICA_VAULT" not in json_cfg.read_text(encoding="utf-8")

    yml_cfg = tmp_path / "cordis.patch.yml"
    setup_client.run_setup(["dsh", "--config", str(yml_cfg)])
    assert "SILICA_VAULT" not in yml_cfg.read_text(encoding="utf-8")

    assert "SILICA_VAULT" not in _printed(["claude", "--dry-run"])


def test_codex_is_idempotent(tmp_path):
    cfg = tmp_path / "config.toml"
    assert setup_client.run_setup(["codex", "--config", str(cfg)]) == 0
    once = cfg.read_text(encoding="utf-8")
    assert setup_client.run_setup(["codex", "--config", str(cfg)]) == 0
    assert cfg.read_text(encoding="utf-8") == once


def test_codex_refuses_broken_toml(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("this is [not toml\n", encoding="utf-8")
    assert setup_client.run_setup(["codex", "--config", str(cfg)]) == 1
    assert cfg.read_text(encoding="utf-8") == "this is [not toml\n"


def test_opencode_merges_into_existing_json(tmp_path):
    cfg = tmp_path / "opencode.json"
    cfg.write_text(json.dumps({"theme": "dark", "mcp": {"other": {}}}), encoding="utf-8")
    assert setup_client.run_setup(["opencode", "--config", str(cfg)]) == 0
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["theme"] == "dark"
    assert "other" in data["mcp"]
    assert data["mcp"]["silica"]["command"][0] == "uvx"


def test_dry_run_writes_nothing(tmp_path):
    cfg = tmp_path / "opencode.json"
    assert setup_client.run_setup(["opencode", "--config", str(cfg), "--dry-run"]) == 0
    assert not cfg.exists()


def test_backup_taken_before_write(tmp_path):
    cfg = tmp_path / "opencode.json"
    cfg.write_text('{"theme": "dark"}', encoding="utf-8")
    setup_client.run_setup(["opencode", "--config", str(cfg)])
    backups = list(tmp_path.glob("opencode.json.bak.*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == {"theme": "dark"}


def _printed(args: list[str]) -> str:
    """What the user actually sees, newlines folded so rich's wrapping at the
    console width cannot be mistaken for a dropped token."""
    from silica.ui.console import CONSOLE
    with CONSOLE.capture() as cap:
        setup_client.run_setup(args)
    return " ".join(cap.get().split())


def test_previews_survive_rich_markup(tmp_path):
    """Every bracketed token rich could read as a style tag has to reach the
    terminal intact: a preview that drops `[mcp]` prints a command which
    installs the wrong package, and one that drops the TOML headers shows a
    block the writer never produces."""
    out = _printed(["codex", "--config", str(tmp_path / "config.toml"), "--dry-run"])
    assert "silica-harness[mcp]" in out
    assert "[mcp_servers.silica]" in out

    out = _printed(["opencode", "--config", str(tmp_path / "opencode.json"), "--dry-run"])
    assert "silica-harness[mcp]" in out

    out = _printed(["dsh", "--config", str(tmp_path / "cordis.patch.yml"), "--dry-run"])
    assert "silica-harness[mcp]" in out

    assert "silica-harness[mcp]" in _printed(["claude", "--dry-run"])
    assert "[--dry-run]" in _printed(["nonsense"])


def test_claude_command_is_one_pastable_line():
    """A wrapped command pastes as fragments, so it must not be folded."""
    from silica.ui.console import CONSOLE
    with CONSOLE.capture() as cap:
        setup_client.run_setup(["claude", "--dry-run"])
    assert cap.get().strip().count("\n") == 0


def test_unknown_client_is_an_error(tmp_path):
    assert setup_client.run_setup(["cursor"]) == 1
    assert setup_client.run_setup([]) == 1


def test_codex_block_outlives_a_cold_uvx(tmp_path):
    # Codex gives a stdio server 10 s to answer `initialize`; a first-run uvx
    # resolve of silica-harness[mcp] takes longer than that on a cold cache.
    cfg = tmp_path / "config.toml"
    setup_client.run_setup(["codex", "--config", str(cfg)])
    parsed = tomllib.loads(cfg.read_text(encoding="utf-8"))
    assert parsed["mcp_servers"]["silica"]["startup_timeout_sec"] == 60


def test_dsh_inserts_the_mcp_client_row(tmp_path):
    cfg = tmp_path / "cordis.patch.yml"
    assert setup_client.run_setup(["dsh", "--config", str(cfg)]) == 0
    patches = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    rows = [r for p in patches for r in p.get("insert", [])]
    row = next(r for r in rows if r["id"] == "mcp-silica")
    assert row["name"] == "@deepseek-ai/dsh-mcp-client"
    c = row["config"]
    assert (c["serverName"], c["transport"]) == ("silica", "stdio")
    assert [c["command"], *c["args"]] == setup_client.MCP_COMMAND


def test_dsh_keeps_other_patches_and_is_idempotent(tmp_path):
    cfg = tmp_path / "cordis.patch.yml"
    cfg.write_text("- id: llm\n  config:\n    model: deepseek-v4\n", encoding="utf-8")
    assert setup_client.run_setup(["dsh", "--config", str(cfg)]) == 0
    once = cfg.read_text(encoding="utf-8")
    assert "model: deepseek-v4" in once
    assert setup_client.run_setup(["dsh", "--config", str(cfg)]) == 0
    assert cfg.read_text(encoding="utf-8") == once


def test_dsh_refuses_a_patch_file_that_is_not_a_list(tmp_path):
    cfg = tmp_path / "cordis.patch.yml"
    cfg.write_text("key: value\n", encoding="utf-8")
    assert setup_client.run_setup(["dsh", "--config", str(cfg)]) == 1
    assert cfg.read_text(encoding="utf-8") == "key: value\n"


def test_dsh_default_path_honours_dsh_home(monkeypatch):
    monkeypatch.setenv("DSH_HOME", "/srv/dsh")
    assert setup_client._default_path("dsh") == Path("/srv/dsh/cordis.patch.yml")


def test_skill_lands_in_the_shared_agents_root(tmp_path, monkeypatch):
    # Codex and DeepSeek Harness both discover ~/.agents/skills, so one copy
    # serves both; Claude Code gets the skill from the plugin instead.
    monkeypatch.setenv("HOME", str(tmp_path))
    for client, cfg in (("codex", "config.toml"), ("dsh", "cordis.patch.yml")):
        assert setup_client.run_setup([client, "--config", str(tmp_path / cfg)]) == 0
        installed = tmp_path / ".agents" / "skills" / "silica" / "SKILL.md"
        assert installed.read_text(encoding="utf-8") == setup_client.skill_path().read_text(encoding="utf-8")
        installed.unlink()


def test_dry_run_installs_no_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    setup_client.run_setup(["codex", "--config", str(tmp_path / "config.toml"), "--dry-run"])
    assert not (tmp_path / ".agents").exists()
