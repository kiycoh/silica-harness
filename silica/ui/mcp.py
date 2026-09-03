# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`silica mcp` — stdio MCP server exposing the tool registry to external agents.

Any MCP client (Claude Code first) gets Silica as vault memory: context
search through the relatedness facade, note reading, and gated writing.

Exposure is a three-step ladder, because every exposed schema is context the
client pays for on each session: the default surface is CORE_TOOLS (what a
navigating client needs — search, read, write, code/data navigation),
`--extended` adds EXTENDED_TOOLS (ingest, graph topology, vault management),
`--all` exposes the full toolset (same sensitive/internal filter as the chat
agent's loop). `--vault DIR` pins the served vault explicitly, which is how
one client config reaches several vaults: one server entry per vault.

stdout is the protocol channel: nothing here may print to it. Logging goes
to stderr (wired by the `silica mcp` dispatch in cli.py).

Only stdlib at import time — the `mcp` SDK is imported inside run_mcp so the
module loads without the [mcp] extra.
"""
from __future__ import annotations

import os
import signal
import sys
from typing import Any

# Search + read + single-note write + structural lookup: the surface a coding
# agent needs to use the vault as memory. Everything else (pipelines, batches,
# taxonomy, graph exports) stays behind --all.
CORE_TOOLS = (
    "silica_recall",
    "silica_timeline",
    "silica_semantic_search",
    "silica_related",
    "silica_concepts",
    "silica_search",
    "silica_search_context",
    "silica_read_note",
    "silica_outline",
    "silica_links",
    "silica_file_links",
    "silica_graph_explain",
    "silica_props",
    "silica_files",
    "silica_exists",
    "silica_write_note",
    "silica_patch_note",
    "silica_flag_note",
    "silica_event_create",
    "silica_event_update",
    "silica_agenda",
    # The shipped /quiz and /learn prompts literally instruct "Call
    # silica_review_queue"; serving the prompts without the tool made a default
    # MCP client fail silently (found 2026-08-25).
    "silica_review_queue",
    # The client's own write ledger: an MCP session otherwise reconstructs what
    # it changed from its tool-call memory, and /undo restores rows it cannot see.
    "silica_changes",
    # Which vault holds the answer. The peek (silica_recall vault=) is reachable
    # only if the client can NAME a vault, and the default session is the one
    # that meets "this lives in another vault" mid-task — so the scoreboard is
    # served by default, not behind --extended.
    "silica_vaults",
    # Not vault memory: the client's read of whether the memory it is talking to
    # is actually whole. A degraded leg (no embeddings, no rerank, unwritable
    # vault) answers plausibly instead of erroring, so without this the only way
    # to find out is to be told.
    "silica_doctor",
    # Code + data navigation (field probe 2026-08-29, datapolis vault): the gap
    # was reachability, not existence — without these a coding client re-derives
    # orientation by hand (code_pack replaces ~10 file reads, tables answers
    # "which csv holds column X" in one call instead of head-reading 143 files).
    "silica_code_pack",
    "silica_impact",
    "silica_tables",
    "silica_query_table",
)

# The useful second ring (--extended): not every session ingests material or
# manages the vault, so these stay out of the default schema payload — but a
# session that does needs them reachable without opening the whole --all
# surface of batch pipelines and index maintenance.
EXTENDED_TOOLS = (
    # Ingest: what turns non-markdown material (code, notebooks, data, PDFs)
    # into notes from a client session. Without these the lane exists but only
    # the local REPL can drive it — a notebook stays invisible to recall until
    # something can call the injector.
    "silica_document",
    "silica_run_injector",
    "silica_inbox_ls",
    # Graph topology beyond CORE's per-note reads.
    "silica_backlinks",
    "silica_graph_path",
    "silica_orphans",
    "silica_unresolved",
    "silica_vault_report",
    # Vault management that carries its own guards (journaled move, gated
    # delete): the guards live in the tools, so serving them adds reach, not risk.
    "silica_move",
    "silica_delete",
    # Pairs with silica_review_queue in CORE: the /quiz flow grades through it.
    "silica_record_quiz",
)

# Exposure is a declaration, not an import side-effect (ADR-0033). Every
# registered non-internal, non-sensitive tool is either served (CORE_TOOLS or
# --all) or named here with its why; test_mcp_exposure fails on drift. Before
# this map, silica_web_answer and silica_query_table were unreachable even
# under --all because only the chat REPL imported their modules.
MCP_EXCLUDED = {
    # silica_query_table left this map 2026-08-29 (navigation-exposure goal):
    # the schema-on-every-reply contract plus the silica_tables census give the
    # client the grounding the missing accuracy gate was guarding against, and
    # the remaining failure mode (aggregating a numeric-looking VARCHAR) is
    # named in the tool's own docstring.
    "silica_web_answer": "live-web egress asks user consent in the chat flow; "
                         "MCP has no consent surface, so the lane stays off it",
}

# Tool modules whose import may fail when their extra is absent; anything else
# failing to import is the silent-unreachable defect, and the test says so.
OPTIONAL_TOOL_MODULES = frozenset({
    "silica.sources.web_research",  # [web] extra
    "silica.sources.web_fetch",     # [web] extra
})

# MCP behavior hints. Writers are enumerated by hand — the registry carries no
# write-capability bit — and test_mcp_surface pins the sets, so a newly served
# writer missing here fails a test instead of shipping advertised "read-only".
WRITE_TOOLS = frozenset({
    # core
    "silica_write_note", "silica_patch_note", "silica_flag_note",
    "silica_event_create", "silica_event_update",
    # extended
    "silica_document", "silica_run_injector", "silica_move",
    "silica_record_quiz",
    # --all: batch curation, index refreshes, exports, orchestration. All of
    # these served with read-only hints before 2026-08-29, so hosts that gate
    # writes harder gated nothing.
    "silica_aliases", "silica_autolink", "silica_backlink", "silica_curate",
    "silica_enrich_batch", "silica_refine_batch", "silica_anneal",
    "silica_generate_taxonomy", "silica_run_organizer", "silica_embed_refresh",
    "silica_lexical_refresh", "silica_cooccurrence_refresh", "silica_mindmap",
    "silica_graph_export", "silica_deferred_retry", "silica_ledger_update",
    "silica_delegate",
})

# Removal is the one shape /undo cannot always give back whole: dedup merges
# delete one note of each pair, flush discards deferred bundles permanently,
# delete is delete. These advertise destructiveHint honestly, so hosts that
# gate destructive tools harder gate exactly these.
DESTRUCTIVE_TOOLS = frozenset({
    "silica_delete", "silica_dedup", "silica_dedup_pairs",
    "silica_deferred_flush",
})

# All four hints, always. Per the MCP spec an omitted hint defaults to the
# permissive reading — destructiveHint and openWorldHint both default TRUE — so
# setting only two advertised every journaled, revertible write as destructive
# and the whole closed-world vault as open-world. Hosts gate destructive tools
# harder, which bought extra confirmation prompts for nothing, and open-world
# was simply false: the domain is the vault, plus — for the doctor alone — the
# user's own configured endpoints. Named in advance either way, never the
# unpredictable set of entities the hint is about.
_READ_ONLY = dict(readOnlyHint=True, destructiveHint=False,
                  idempotentHint=True, openWorldHint=False)
# Additive, not destructive: every write goes through the undo journal
# (ADR-0002) and `/undo` reverses it, and none is idempotent — re-running
# appends again.
_ADDITIVE = dict(readOnlyHint=False, destructiveHint=False,
                 idempotentHint=False, openWorldHint=False)
_DESTRUCTIVE = dict(readOnlyHint=False, destructiveHint=True,
                    idempotentHint=False, openWorldHint=False)


def tool_annotations(name: str) -> dict:
    """The four behaviour hints for one served tool."""
    if name in DESTRUCTIVE_TOOLS:
        return dict(_DESTRUCTIVE)
    return dict(_ADDITIVE if name in WRITE_TOOLS else _READ_ONLY)


def exposed_tools(all_tools: bool = False, extended: bool = False) -> dict[str, Any]:
    """The registry slice served over MCP: Tool objects keyed by name.

    The ladder is monotone: default ⊂ --extended ⊂ --all."""
    # Registration side effect — the WHOLE tool tree, deliberately, not the
    # subset cli.py happens to share (ADR-0033): a module nobody imports is a
    # tool no flag can reach.
    import silica.tools.atomic  # noqa: F401
    import silica.tools.composed  # noqa: F401
    import silica.tools.wrapped  # noqa: F401
    import silica.tools.codedocs_tool  # noqa: F401
    import silica.tools.delegate_tool  # noqa: F401
    # The tabular lane rode in on OPTIONAL_TOOL_MODULES until duckdb became a
    # base dependency (2026-08-29); without a plain import here its two CORE
    # names never register and exposed_tools raises registry drift.
    import silica.tools.tabular  # noqa: F401
    for _mod in OPTIONAL_TOOL_MODULES:
        try:
            __import__(_mod)
        except ImportError:
            pass  # extra not installed: its tools cannot register; MCP_EXCLUDED
            #       still documents the verdict for when they can
    from silica.tools import TOOLS

    allowed = {n: t for n, t in TOOLS.items()
               if not t.sensitive and not t.internal and n not in MCP_EXCLUDED}
    if all_tools:
        return allowed
    out: dict[str, Any] = {}
    for n in CORE_TOOLS + (EXTENDED_TOOLS if extended else ()):
        if n in allowed:
            out[n] = allowed[n]
        else:
            raise KeyError(f"{n}: named in a served tier but not registered — registry drift")
    return out


# The server's own account of itself, sent in the initialize reply. Claude
# Code and Codex put it in front of the model; a client with no skill surface
# at all (opencode) has nothing else to go on. The skill is the long form,
# this is the loop, and `silica hook SessionStart` opens a session with the
# same text so the three never disagree.
INSTRUCTIONS = (
    "Silica serves the vault of the folder this server was started in as "
    "memory. Before answering from memory call silica_recall; read a hit with "
    "silica_read_note before citing it. For questions scoped to THIS vault "
    "(repo and code questions) pass memory=false so personal-memory notes "
    "stay out of the slots. Capture what should outlive the "
    "session with silica_write_note (new concept) or silica_patch_note "
    "(existing note): decisions and their why, non-obvious constraints, "
    "hard-won references. If retrieval behaves as if switched off, call "
    "silica_doctor. If the answer may live in another vault, silica_vaults(query) "
    "scores the vaults this machine knows (`home` names the ones that hold it) and "
    "silica_recall(query, vault=<path>) reads one without leaving this session's vault."
)


def parse_cli_args(args: list[str]) -> dict[str, Any]:
    """`silica mcp` flags → run options; `error` set means print-and-exit.

    Parsed here, beside the tier declarations the flags select, so the ladder
    and its parsing share one home and one test file.
    """
    opts: dict[str, Any] = {"all_tools": False, "extended": False, "vault": "", "error": ""}
    it = iter(args)
    for a in it:
        if a == "--all":
            opts["all_tools"] = True
        elif a == "--extended":
            opts["extended"] = True
        elif a == "--vault":
            opts["vault"] = next(it, "")
            if not opts["vault"]:
                opts["error"] = "--vault needs a directory"
        elif a.startswith("--vault="):
            opts["vault"] = a.split("=", 1)[1]
        else:
            opts["error"] = f"unknown flag for silica mcp: {a}"
    return opts


def make_server(all_tools: bool = False, extended: bool = False):
    """The MCP Server with the registry slice wired in, not yet serving."""
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server

    from silica.config import CONFIG

    tools = exposed_tools(all_tools, extended)
    # The served vault, named in the handshake: with one server entry per
    # vault in a client config (--vault), this line is how the model tells
    # two silica servers apart.
    vault = str(getattr(CONFIG, "vault_path", "") or "").strip()
    instructions = INSTRUCTIONS + (f" This server's vault: {vault}" if vault else "")
    server = Server("silica", instructions=instructions)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=t.name,
                description=t.description,
                inputSchema=t.json_schema()["function"]["parameters"],
                annotations=types.ToolAnnotations(**tool_annotations(t.name)),
            )
            for t in tools.values()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        t = tools.get(name)
        if t is None:
            raise ValueError(f"Unknown tool: {name}")
        # Tool.run validates args via pydantic and always returns a JSON string
        # (errors included) — exactly what a text content block wants.
        out = await anyio.to_thread.run_sync(lambda: t.run(**(arguments or {})))
        return [types.TextContent(type="text", text=out)]

    return server


def run_mcp(all_tools: bool = False, extended: bool = False) -> int:
    """Serve the tool registry over MCP stdio. Blocks until the client hangs up."""
    try:
        import anyio
        from mcp.server.stdio import stdio_server
    except ImportError:
        print(
            "silica mcp needs the [mcp] extra: uv pip install 'silica-harness[mcp]'",
            file=sys.stderr,
        )
        return 1

    server = make_server(all_tools, extended)
    tools = exposed_tools(all_tools, extended)

    async def _serve() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    # stderr, not stdout: stdout is the protocol channel. Without this the
    # server looks hung when a human runs it by hand instead of a client.
    print(
        f"silica mcp: serving {len(tools)} tools on stdio, waiting for a client "
        "(Ctrl+C to stop)",
        file=sys.stderr,
    )
    # Ctrl+C exits now, not on the third press. The SDK reads stdin through
    # anyio's to_thread.run_sync, which parks a *non-daemon* worker thread in a
    # blocking readline(): graceful cancellation cannot return while that thread
    # sits there, and interpreter shutdown then deadlocks joining it. Left alone
    # that costs three presses (cancel, KeyboardInterrupt traceback, break the
    # join) and an abort. Claiming SIGINT here also keeps asyncio's runner off
    # it, since the runner only installs its own handler over the default one.
    # Hard exit is safe: vault writes land through atomic_write_bytes, so the
    # worst case is a stray temp file, never a torn note.
    signal.signal(signal.SIGINT, lambda *_: (sys.stderr.flush(), os._exit(0)))
    anyio.run(_serve)
    return 0
