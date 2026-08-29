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
    # Every CORE name resolves: no served tier holds an optional tool.
    assert not set(CORE_TOOLS) - set(core)
    for t in core.values():
        assert not t.internal and not t.sensitive
        # every exposed tool must yield a servable JSON schema
        params = t.json_schema()["function"]["parameters"]
        assert params.get("type") == "object"


def test_declared_writers_are_real_served_tools():
    from silica.ui.mcp import DESTRUCTIVE_TOOLS, MCP_EXCLUDED, WRITE_TOOLS

    # A stale name in the hint sets (renamed tool) would silently stop hinting;
    # every declared writer must still exist on the served-or-excluded surface.
    surface = set(exposed_tools(all_tools=True)) | set(MCP_EXCLUDED)
    assert (WRITE_TOOLS | DESTRUCTIVE_TOOLS) <= surface, \
        (WRITE_TOOLS | DESTRUCTIVE_TOOLS) - surface
    assert not WRITE_TOOLS & DESTRUCTIVE_TOOLS


def test_every_served_tool_sets_all_four_hints():
    """An omitted hint is not "unspecified" to a host: the MCP spec defaults
    destructiveHint and openWorldHint to TRUE. Setting two of four advertised
    journaled, revertible writes as destructive and a closed-world vault as
    open-world."""
    from silica.ui.mcp import DESTRUCTIVE_TOOLS, tool_annotations

    for name in exposed_tools(all_tools=True):
        hints = tool_annotations(name)
        assert set(hints) == {"readOnlyHint", "destructiveHint",
                              "idempotentHint", "openWorldHint"}, name
        assert hints["openWorldHint"] is False, name
        # destructiveHint only where /undo cannot give the note back whole.
        assert hints["destructiveHint"] is (name in DESTRUCTIVE_TOOLS), name
    # The default surface stays free of destructive tools entirely.
    assert not set(exposed_tools()) & DESTRUCTIVE_TOOLS


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


def test_all_surface_matches_agent_loop_filter_minus_declared_exclusions():
    # ADR-0033: --all is the agent-loop filter minus MCP_EXCLUDED — tools whose
    # exposure verdict is written down (no consent surface, no accuracy gate),
    # never an accident of which modules got imported.
    from silica.tools import TOOLS
    from silica.ui.mcp import MCP_EXCLUDED

    exposed = exposed_tools(all_tools=True)
    expected = {n for n, t in TOOLS.items()
                if not t.sensitive and not t.internal and n not in MCP_EXCLUDED}
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
    args = servers["silica"]["args"]
    assert args[args.index("silica"):args.index("silica") + 2] == ["silica", "mcp"]


def test_plugin_serves_its_own_tree_not_the_published_wheel():
    """ADR-0025 amendment: `uvx --from silica-harness` resolved the PyPI wheel,
    so a tool added to the checkout was unreachable from every client and
    `/plugin marketplace update` did not fix it (three live versions, measured
    2026-08-29). The plugin must run the code it shipped with."""
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    args = json.loads(
        (ROOT / plugin["mcpServers"]).read_text(encoding="utf-8")
    )["mcpServers"]["silica"]["args"]
    assert "${CLAUDE_PLUGIN_ROOT}" in args
    # --project selects the environment and leaves the working directory alone;
    # --directory would serve the silica checkout as every project's vault.
    assert "--project" in args and "--directory" not in args
    assert not any("silica-harness" in a for a in args), "back on the PyPI wheel"

    # Codex is not a plugin host: it has no ${CLAUDE_PLUGIN_ROOT} to expand, so
    # its manifest deliberately keeps the published wheel.
    codex = json.loads((ROOT / "mcp.codex.json").read_text(encoding="utf-8"))
    codex_args = codex["mcp_servers"]["silica"]["args"]
    assert "${CLAUDE_PLUGIN_ROOT}" not in codex_args
    assert any("silica-harness" in a for a in codex_args)
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
