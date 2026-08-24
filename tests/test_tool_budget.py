# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Tool-block budget (chat toolset) and Anthropic prompt-cache breakpoint.

Both legs exist for the same reason: the tool schemas are the biggest static
chunk of every agentic turn. chat_tools() shrinks it, _with_prompt_cache stops
Anthropic re-billing it.
"""
from __future__ import annotations

import importlib
import json
import pkgutil

import silica.tools as T
from silica.agent.constraints import _CHAT_EXCLUDED, chat_tools
from silica.agent.llm import _with_prompt_cache


def _all_tools():
    for m in pkgutil.iter_modules(T.__path__):
        importlib.import_module("silica.tools." + m.name)
    return T.TOOLS


# --- Leg A: chat toolset ---------------------------------------------------

def test_chat_tools_excludes_only_the_named_batch_tools():
    tools = _all_tools()
    chat = set(chat_tools())
    default = {n for n, t in tools.items() if not t.sensitive and not t.internal}
    assert chat == default - _CHAT_EXCLUDED
    # Every excluded name must actually exist, or the exclusion is a silent no-op
    # that quietly stops saving anything.
    assert _CHAT_EXCLUDED <= set(tools)


def test_chat_tools_keeps_every_recovery_path_it_advertises():
    """A tool description that says "run X first" must leave X callable.

    This is the constraint that bounds the cut: hiding a tool some other tool
    points at turns its error hint into a dead end the model cannot act on.
    """
    tools = _all_tools()
    chat = set(chat_tools())
    for name in chat:
        desc = tools[name].description or ""
        for other in tools:
            # Bare-word tool names (plan, remember) collide with prose: the
            # ledger's "when the plan is exhausted" is not a recovery hint.
            # Only namespaced names are detectable by substring, and both
            # bare names are sensitive, so a chat description that pointed at
            # one would be a doc bug this check cannot see. Known ceiling.
            if "_" not in other:
                continue
            if other in desc and other != name:
                assert other in chat, f"{name} tells the model to call {other}, which chat_tools() hides"


def test_chat_tools_keeps_the_multi_turn_review_protocol():
    # The vault review applies on a LATER turn than the one that reported, so a
    # plain "yes, go ahead" message still has to reach the ledger tools.
    chat = set(chat_tools())
    for name in ("silica_vault_report", "silica_ledger_next", "silica_ledger_update"):
        assert name in chat


def test_a_slash_command_reaches_the_excluded_tools_it_names():
    """The /organize dead end: its directive tells the model to call
    silica_generate_taxonomy and silica_run_organizer, both hidden by
    _CHAT_EXCLUDED. The turn used to answer "those tools aren't available to me".

    Checked through the real expansion, not a hand-written string, so the day the
    directive is reworded the check follows it.
    """
    from silica.cli import _expand_workflow_shortcut

    directive = _expand_workflow_shortcut('/organize "group by domain"')
    assert directive and "silica_generate_taxonomy" in directive
    history = [{"role": "user", "content": directive, "origin": "cli"}]
    chat = set(chat_tools(history))
    assert "silica_generate_taxonomy" in chat
    assert "silica_run_organizer" in chat


def test_quiz_and_learn_reach_the_record_tool_they_name():
    """silica_record_quiz left the chat set: /quiz and /learn are its only
    entries, and both directives name it — the summon path must hand it back."""
    from silica.cli import _expand_workflow_shortcut

    for cmd in ('/quiz "Topic"', '/learn "Topic"'):
        directive = _expand_workflow_shortcut(cmd)
        assert directive and "silica_record_quiz" in directive, cmd
        history = [{"role": "user", "content": directive, "origin": "cli"}]
        assert "silica_record_quiz" in set(chat_tools(history)), cmd
    assert "silica_record_quiz" not in set(chat_tools())


def test_the_summoned_tools_survive_the_follow_up_turn():
    # /organize is generate -> confirm -> dry run -> apply. The confirming turn
    # names no tool at all; scoping to the current message would strand it.
    from silica.cli import _expand_workflow_shortcut

    history = [
        {"role": "user", "content": _expand_workflow_shortcut('/organize "by domain"'), "origin": "cli"},
        {"role": "assistant", "content": "Here is the taxonomy. Look right?"},
        {"role": "user", "content": "yes, go ahead"},
    ]
    assert "silica_run_organizer" in set(chat_tools(history))


def test_prose_can_reach_the_tool_that_resolves_deferred_bundles():
    """Measured 2026-08-23: "Resolve the 24 deferred notes" is prose, so
    _summoned returns nothing, and every tool touching the deferred queue was
    excluded — the turn had no move and answered with an empty string.
    silica_anneal is the one that stays visible so the ask lands somewhere."""
    history = [{"role": "user", "content": "Resolve the 24 deferred notes"}]
    assert "silica_anneal" in set(chat_tools(history))


def test_prose_quoting_a_tool_name_does_not_summon_it():
    # Only a cli-origin directive summons. A human asking *about* the organizer
    # must not silently hand the model a bulk-move tool.
    history = [{"role": "user", "content": "what does silica_run_organizer do?"}]
    assert "silica_run_organizer" not in set(chat_tools(history))
    assert set(chat_tools(history)) == set(chat_tools())


def test_chat_tools_actually_cuts_the_block():
    tools = _all_tools()
    def cost(names):
        return sum(len(json.dumps(tools[n].json_schema())) for n in names)
    default = [n for n, t in tools.items() if not t.sensitive and not t.internal]
    # Re-measured 2026-08-15 at 0.807 (was pinned 0.80 with a stale "~34%
    # saving" note): every legitimate chat tool joins BOTH sets, so the ratio
    # drifts toward 1 by construction — the calendar tools took the last
    # headroom. The invariant guarded is "the cut stays real", so the line
    # moves to 0.82; below ~18% saving the exclusion list needs new members.
    assert cost(chat_tools()) < cost(default) * 0.82


# --- Leg B: prompt cache breakpoint ---------------------------------------

_MSGS = [{"role": "system", "content": "you are silica"}, {"role": "user", "content": "hi"}]


def _marked(text):
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def test_cache_breakpoint_marks_system_for_anthropic():
    out = _with_prompt_cache("anthropic/claude-opus-4", _MSGS)
    assert out[0]["content"] == _marked("you are silica")
    # Second breakpoint on the last user message: without it the whole
    # conversation is re-billed at full price on every turn.
    assert out[1]["content"] == _marked("hi")


def test_cache_breakpoint_covers_the_whole_static_head():
    # The vault map is a second system message; marking only messages[0] left it
    # (and anything else appended to the head) outside the cached prefix.
    msgs = [
        {"role": "system", "content": "you are silica"},
        {"role": "system", "content": "vault map"},
        {"role": "user", "content": "hi"},
    ]
    out = _with_prompt_cache("anthropic/claude-opus-4", msgs)
    assert out[0] == msgs[0]  # not the breakpoint — the head's LAST block is
    assert out[1]["content"] == _marked("vault map")
    assert out[2]["content"] == _marked("hi")


def test_cache_breakpoint_skips_tool_results():
    # A tool message's content blocks get nested inside a tool_result by the
    # translation layer, where cache_control is not a valid position.
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]
    out = _with_prompt_cache("anthropic/claude-opus-4", msgs)
    assert out[3] == msgs[3]
    assert out[1]["content"] == _marked("hi")  # rolling marker walks back to it


def test_cache_breakpoint_marks_the_latest_user_turn_only():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    out = _with_prompt_cache("anthropic/claude-opus-4", msgs)
    assert out[1] == msgs[1]
    assert out[3]["content"] == _marked("second")


def test_cache_breakpoint_applies_through_a_proxy_prefix():
    out = _with_prompt_cache("openrouter/anthropic/claude-sonnet-4", _MSGS)
    assert isinstance(out[0]["content"], list)


def test_cache_breakpoint_skips_providers_that_cache_on_their_own():
    # OpenAI caches long prefixes itself; a local backend has no billing to
    # amortise. Gemini is NOT here: its explicit breakpoints raise the hit rate
    # through OpenRouter, and the native route ignores the marker.
    for model in ("ollama/gemma4:e4b", "openai/gpt-4o", "deepseek/deepseek-chat"):
        assert _with_prompt_cache(model, _MSGS) is _MSGS


def test_cache_breakpoint_applies_to_gemini():
    out = _with_prompt_cache("openrouter/google/gemini-2.5-pro", _MSGS)
    assert out[0]["content"] == _marked("you are silica")


def test_cache_breakpoint_never_mutates_caller_history():
    # run_agent keeps appending to this exact list; a marker written in place
    # would leak into the stored conversation and every later turn.
    msgs = [dict(m) for m in _MSGS]
    _with_prompt_cache("anthropic/claude-opus-4", msgs)
    assert msgs[0]["content"] == "you are silica"


def test_cache_breakpoint_tolerates_odd_histories():
    assert _with_prompt_cache("anthropic/claude-opus-4", []) == []
    # No system head: the rolling breakpoint still lands on the user turn.
    no_sys = _with_prompt_cache("anthropic/claude-opus-4", [{"role": "user", "content": "hi"}])
    assert no_sys[0]["content"] == _marked("hi")
    # Content already in block form, or empty: left exactly as it came in.
    blocks = [{"role": "system", "content": [{"type": "text", "text": "x"}]}]
    assert _with_prompt_cache("anthropic/claude-opus-4", blocks) == blocks
    empty = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
    assert _with_prompt_cache("anthropic/claude-opus-4", empty) == empty


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
