# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Optional constraints that turn run_agent into a bounded worker loop.

Carries only the three generic dials (tools, model, iteration cap). The leash is
deliberately NOT here — write safety lives inside the write tool / apply_op, so
run_agent stays domain-agnostic (Rune 1 / ADR set).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConstraints:
    tools: tuple[str, ...]          # subset of TOOLS the loop may expose + dispatch
    model: str | None = None        # override the model arg when set
    max_iterations: int | None = None  # override the default safety cap when set
    # An interactive turn with a constrained toolset. run_agent used to read
    # "constraints is None" as "interactive": streaming on, no worker slot.
    # That conflated two independent facts — the GUI chat wants the chat_tools
    # cut (a smaller tool block every iteration) while keeping its live stream
    # and staying off the worker-pool cap. This flag says which of the two a
    # constrained caller is; batch/worker callers leave it False.
    interactive: bool = False


# Batch/maintenance tools the conversational loop does not need to *choose*: the
# user reaches each one by name through its slash command. Excluding them from
# the chat toolset is worth ~2.4k tokens per turn (~24% of the tool block).
#
# The cut is bounded by one hard constraint, enforced by
# test_chat_tools_keeps_every_recovery_path_it_advertises: tool descriptions name
# each other as follow-ups ("run silica_cooccurrence_refresh first", "use
# silica_curate"). Hiding a tool that a visible tool tells the model to call
# turns that instruction into a dead end. That constraint is what keeps
# silica_curate, silica_dedup and silica_graph_export in the set despite each
# having a slash command, and why this list is 14 entries rather than 30.
#
# silica_anneal was here until 2026-08-23 and is deliberately NOT: it is the
# only tool that resolves a deferred bundle, and the nucleate summary tells the
# user how many are still deferred. Hidden, the obvious follow-up ("resolve the
# 24 deferred notes") reached a turn with no tool that could do it — the same
# dead end _summoned was written for, except no slash command names it.
#
# ponytail: one explicit list, not a per-tool flag. A tool added later defaults
# to being visible in chat, which costs tokens but never breaks — the safe
# direction. Revisit if this grows past ~20 entries.
_CHAT_EXCLUDED = frozenset({
    "silica_aliases",            # /aliases
    "silica_code_pack",          # MCP-facing (external rewrite agents); no chat flow
    "silica_deferred_list",      # deferred-ops bookkeeping, surfaced by the FSM
    "silica_deferred_flush",
    "silica_deferred_retry",
    "silica_delegate",           # fan-out, driven by the Coordinator not by chat
    "silica_document",           # /wiki
    "silica_generate_taxonomy",  # /organize
    "silica_health",             # /status
    "silica_inbox_ls",
    "silica_ledger_digest",      # /report runs it directly, no agent involved
    "silica_mindmap",            # /map
    "silica_record_quiz",        # /quiz and /learn name it in their directives
    "silica_run_organizer",      # /organize
})


def web_turn_constraints() -> AgentConstraints:
    """`/web` — the consented web turn: the four web tools and nothing else.
    (`remember` banks a verbatim quote from a fetched page and `find_in_page`
    greps one; both touch only per-turn module state in
    silica/sources/web_research.py, never the vault.)

    48 is the twin of `_DEFAULT_MAX_SEARCHES` in silica/sources/web_research.py,
    which carries the reasoning for the number. Duplicated rather than imported:
    the import direction is sources -> agent, so this module must not read that
    one; test_web_turn_iteration_cap_matches_web_research holds them equal.

    Held equal even though this lane is interactive and that one is batch: the
    cap is a ceiling, not a target, so a /web question that converges in four
    steps costs four either way. Split the two the day a /web turn is seen
    spending a ceiling it had no reason to spend.
    """
    return AgentConstraints(
        tools=("web_search", "web_fetch", "remember", "find_in_page"),
        max_iterations=48,
    )


def _summoned(messages: "list[dict] | None") -> set[str]:
    """Excluded tools that a slash command in THIS conversation named.

    `/organize` expands into a directive that tells the model to call
    silica_generate_taxonomy and then silica_run_organizer. Both are excluded
    above, so the turn dead-ended on tools the model could not see — the very
    failure the _CHAT_EXCLUDED comment warns about for tool descriptions, which
    the description-level test could not see because this instruction lives in
    prompt text, not in a description.

    Read over the whole history rather than the current message: /organize is a
    four-step protocol (generate -> confirm -> dry run -> apply) and the user's
    "yes, go ahead" carries no tool name of its own. That is the same reason
    chat_tools() is not scoped per turn.

    ponytail: substring match over cli-origin turns only, so ordinary prose that
    happens to quote a tool name cannot summon it. Compaction that drops the
    directive drops the tools with it and the user re-runs the command; revisit
    if that is ever seen for real.
    """
    if not messages:
        return set()
    text = "\n".join(
        m["content"] for m in messages
        if m.get("origin") == "cli" and isinstance(m.get("content"), str)
    )
    return {n for n in _CHAT_EXCLUDED if n in text} if text else set()


def chat_tools(messages: "list[dict] | None" = None) -> tuple[str, ...]:
    """Toolset for the interactive chat loop: everything except batch maintenance,
    plus any excluded tool a slash command in `messages` explicitly asked for.

    Deliberately NOT scoped per turn. The vault-review protocol spans turns —
    step 1 reports, step 2 applies via silica_ledger_next after the user agrees —
    so a plain "yes, go ahead" turn still needs the ledger tools. Anything that
    picked tools from the current message alone would strand that second turn.
    """
    from silica.tools import TOOLS

    extra = _summoned(messages)
    return tuple(
        n for n, t in TOOLS.items()
        if not t.sensitive and not t.internal and (n not in _CHAT_EXCLUDED or n in extra)
    )
