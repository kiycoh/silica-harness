# SPDX-License-Identifier: AGPL-3.0-or-later

"""MCP exposure is a declaration, not an import side-effect (ADR-0033).

Found 2026-08-25: tool registration happens as a decorator side-effect of
whichever modules got imported, so silica_web_answer and silica_query_table
were unreachable even under `--all` (their modules were only imported by the
chat REPL and the GUI), while shipped prompts instructed calling a tool the
default surface did not serve. The server now imports the whole tool tree
deliberately and every registered tool must be served, internal, sensitive,
or excluded BY NAME with a reason.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import silica


def _tool_modules() -> list[str]:
    """Every silica module that registers tools (syntactic @tool( usage)."""
    root = Path(silica.__file__).resolve().parent
    pat = re.compile(r"^@tool\(", re.MULTILINE)
    mods = []
    for py in root.rglob("*.py"):
        try:
            if pat.search(py.read_text(encoding="utf-8", errors="replace")):
                rel = py.relative_to(root.parent).with_suffix("")
                mods.append(".".join(rel.parts))
        except OSError:
            continue
    return sorted(mods)


def test_every_registered_tool_has_a_declared_verdict():
    from silica.ui import mcp

    failed = []
    for mod in _tool_modules():
        try:
            importlib.import_module(mod)
        except ImportError:
            failed.append(mod)
    # Only a declared-optional extra may fail to import; anything else is the
    # silent-unreachable defect coming back.
    undeclared = [m for m in failed if m not in mcp.OPTIONAL_TOOL_MODULES]
    assert not undeclared, f"tool modules unimportable and undeclared: {undeclared}"

    from silica.tools import TOOLS

    served = set(mcp.exposed_tools(all_tools=True))
    orphans = [
        n for n, t in TOOLS.items()
        if not t.internal and not t.sensitive
        and n not in served and n not in mcp.MCP_EXCLUDED
    ]
    assert not orphans, f"registered but neither served nor excluded-with-reason: {orphans}"

    for name, reason in mcp.MCP_EXCLUDED.items():
        assert reason.strip(), f"{name}: an exclusion without a why is drift with paperwork"
        # When every tool module imported, an excluded name must actually be
        # registered — a stale exclusion for a deleted tool would hide drift.
        if not failed:
            assert name in TOOLS, f"{name}: excluded but no longer registered anywhere"


def test_default_surface_serves_the_tools_the_prompts_instruct():
    # cli's /quiz and /learn shortcuts literally say "Call silica_review_queue";
    # a default MCP client following shipped guidance must not fail silently.
    from silica.ui import mcp

    exposed = mcp.exposed_tools()
    assert "silica_review_queue" in exposed
    assert "silica_changes" in exposed


def test_silica_changes_reports_this_process_writes(tmp_path, monkeypatch):
    from silica.config import CONFIG
    from silica.kernel.write import session_changes

    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    session_changes.clear()
    (tmp_path / "n.md").write_text("after", encoding="utf-8")
    session_changes.touched("n.md", None)  # first write: note did not exist
    from silica.tools.atomic import silica_changes

    out = silica_changes()
    assert out["total"] == 1
    row = out["changes"][0]
    assert row["path"] == "n.md" and row["kind"] == "created"
    session_changes.clear()


def test_doctor_live_probe_is_a_parameter(monkeypatch):
    from silica.onboarding import checks

    calls = {"n": 0}

    def fake_probe(cfg):
        calls["n"] += 1
        return checks.CheckResult("live probe", "ok", "model replied", "")

    monkeypatch.setattr(checks, "live_probe", fake_probe)
    from silica.tools.atomic import silica_doctor

    cold = silica_doctor()
    assert calls["n"] == 0, "the paid probe must never run unasked"
    assert all(r["name"] != "live probe" for r in cold["results"])

    hot = silica_doctor(live=True)
    assert calls["n"] == 1
    assert any(r["name"] == "live probe" for r in hot["results"])
