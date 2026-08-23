"""`silica hook <event>`: the SessionStart producer every harness can run."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from silica import hook as hook_mod
from silica.ui.mcp import CORE_TOOLS


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "vault.yaml").write_text("write_dir: silica\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("# a\n", encoding="utf-8")
    return tmp_path


def test_session_start_names_the_vault_and_the_loop(tmp_path):
    vault = _vault(tmp_path)
    sub = vault / "deep" / "er"
    sub.mkdir(parents=True)
    out = hook_mod.session_start(json.dumps({"hook_event_name": "SessionStart", "cwd": str(sub)}))
    assert str(vault) in out
    for tool in ("silica_recall", "silica_write_note", "silica_patch_note"):
        assert tool in out


def test_session_start_names_only_served_tools(tmp_path):
    out = hook_mod.session_start(json.dumps({"cwd": str(_vault(tmp_path))}))
    assert not set(re.findall(r"silica_\w+", out)) - set(CORE_TOOLS)


def test_no_vault_is_silence(tmp_path):
    assert hook_mod.session_start(json.dumps({"cwd": str(tmp_path)})) == ""


def test_garbage_stdin_is_silence():
    assert hook_mod.session_start("not json") == ""


def test_run_hook_prints_the_brief_and_ignores_unknown_events(tmp_path, capsys):
    vault = _vault(tmp_path)
    assert hook_mod.run_hook(["SessionStart"], json.dumps({"cwd": str(vault)})) == 0
    assert str(vault) in capsys.readouterr().out
    assert hook_mod.run_hook(["Nope"], "{}") == 0
    assert capsys.readouterr().out == ""


def test_cli_exits_zero_and_prints_nothing_outside_a_vault(tmp_path):
    # Fail-open is the contract: a hook that exits non-zero or prints a
    # traceback lands in someone else's session as an error.
    r = subprocess.run(
        [sys.executable, "-m", "silica.cli", "hook", "SessionStart"],
        input=json.dumps({"cwd": str(tmp_path)}),
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
    )
    assert r.returncode == 0
    assert r.stdout == ""
