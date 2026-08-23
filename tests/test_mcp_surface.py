"""The MCP surface must stay a valid slice of the tool registry."""
from __future__ import annotations

import json
import re
import signal
import subprocess
import sys
from pathlib import Path

from silica.onboarding.setup_client import skill_path
from silica.ui.mcp import CORE_TOOLS, exposed_tools

ROOT = Path(__file__).resolve().parent.parent


def test_core_tools_resolve_and_are_agent_visible():
    core = exposed_tools()
    assert set(core) == set(CORE_TOOLS)
    for t in core.values():
        assert not t.internal and not t.sensitive
        # every exposed tool must yield a servable JSON schema
        params = t.json_schema()["function"]["parameters"]
        assert params.get("type") == "object"


def test_write_tools_are_the_only_non_readonly_hints():
    from silica.ui.mcp import WRITE_TOOLS

    # A new mutating tool served without joining WRITE_TOOLS would be
    # advertised to MCP clients as read-only — catch the drift here.
    assert WRITE_TOOLS <= set(CORE_TOOLS)
    for name in WRITE_TOOLS:
        assert any(k in name for k in ("write", "patch", "flag", "create", "update"))


def test_every_served_tool_sets_all_four_hints():
    """An omitted hint is not "unspecified" to a host: the MCP spec defaults
    destructiveHint and openWorldHint to TRUE. Setting two of four advertised
    journaled, revertible writes as destructive and a closed-world vault as
    open-world."""
    from silica.ui.mcp import tool_annotations

    for name in CORE_TOOLS:
        hints = tool_annotations(name)
        assert set(hints) == {"readOnlyHint", "destructiveHint",
                              "idempotentHint", "openWorldHint"}, name
        assert hints["openWorldHint"] is False, name
        assert hints["destructiveHint"] is False, name


def test_write_tools_are_additive_and_reads_are_idempotent():
    from silica.ui.mcp import WRITE_TOOLS, tool_annotations

    for name in WRITE_TOOLS:
        hints = tool_annotations(name)
        assert hints["readOnlyHint"] is False, name
        # Re-running a write appends again; only reads are idempotent.
        assert hints["idempotentHint"] is False, name
    for name in set(CORE_TOOLS) - WRITE_TOOLS:
        hints = tool_annotations(name)
        assert hints["readOnlyHint"] is True and hints["idempotentHint"] is True, name


def test_the_hint_dicts_are_not_shared_between_tools():
    """Returned per call, so a caller mutating one tool's hints cannot rewrite
    the advertised contract of every other tool in the process."""
    from silica.ui.mcp import tool_annotations

    first = tool_annotations("silica_recall")
    first["openWorldHint"] = True

    assert tool_annotations("silica_recall")["openWorldHint"] is False


def test_all_surface_matches_agent_loop_filter():
    from silica.tools import TOOLS

    exposed = exposed_tools(all_tools=True)
    expected = {n for n, t in TOOLS.items() if not t.sensitive and not t.internal}
    assert set(exposed) == expected


def test_skill_references_only_core_tools():
    # The Claude skill teaches the default MCP surface — a tool name in the
    # skill that isn't in CORE_TOOLS is drift (renamed, or never exposed).
    skill = skill_path().read_text(encoding="utf-8")
    referenced = set(re.findall(r"silica_\w+", skill))
    unknown = referenced - set(CORE_TOOLS)
    assert not unknown, f"SKILL.md references tools outside the MCP core surface: {unknown}"


def test_plugin_manifest_launches_silica_mcp():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    servers = json.loads((ROOT / plugin["mcpServers"]).read_text(encoding="utf-8"))["mcpServers"]
    assert servers["silica"]["args"][-2:] == ["silica", "mcp"]
    marketplace = json.loads(
        (ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    assert marketplace["plugins"][0]["name"] == plugin["name"]


def test_server_instructions_teach_the_loop_to_clients_without_skills():
    # Codex and DSH read the skill from ~/.agents/skills; a client with no
    # skill surface at all (opencode) only ever sees what the server says of
    # itself, and that text must not name a tool the default surface hides.
    from silica.ui.mcp import make_server
    text = make_server().instructions or ""
    for tool in ("silica_recall", "silica_write_note", "silica_patch_note"):
        assert tool in text
    assert not set(re.findall(r"silica_\w+", text)) - set(CORE_TOOLS)


def test_one_sigint_stops_the_server():
    # The stdio transport parks a non-daemon worker thread in a blocking
    # readline(), so without run_mcp's own SIGINT handler a graceful shutdown
    # deadlocks and the server needs three Ctrl+C and dies on an abort.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; from silica.ui.mcp import run_mcp; sys.exit(run_mcp())"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Handshake first: a reply on stdout is the only proof the loop is up
        # and parked on the next stdin read. The stderr banner prints before
        # anyio.run, so signalling on it races the event loop's own setup.
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "test", "version": "0"}},
        }).encode() + b"\n")
        proc.stdin.flush()
        assert proc.stdout.readline(), f"no initialize reply, rc={proc.poll()}"
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            pipe.close()
