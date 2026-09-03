"""Wiring tests: compaction levers actually run in the live loop.

- eager projection: run_agent stores the one-line summary in history,
  while the TUI event still carries the full result.
- lazy compaction: cli._compact_context collapses old reads when the
  meter is over budget and refreshes the token count.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from silica.agent.loop import run_agent


class _FakeTool:
    sensitive = False
    internal = False
    summarize = None

    def __init__(self, name: str, collapse: str, result: str):
        self.name = name
        self.collapse = collapse
        self._result = result

    def json_schema(self):
        return {
            "type": "function",
            "function": {"name": self.name, "description": "", "parameters": {"type": "object", "properties": {}}},
        }

    def run(self, _cancel_token=None, **kw):
        return self._result


def _two_turn_llm(tool_name: str):
    """First call: one tool call. Second call: final text."""
    calls = [0]

    def fake_call_llm(*a, **k):
        calls[0] += 1
        if calls[0] == 1:
            return SimpleNamespace(
                assistant_message={
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": tool_name, "arguments": "{}"}}],
                },
                tool_calls=[SimpleNamespace(id="c1", name=tool_name, args={})],
                text="",
                reasoning=None, usage={},
            )
        return SimpleNamespace(
            assistant_message={"role": "assistant", "content": "done"},
            tool_calls=[],
            text="done",
            reasoning=None, usage={},
        )

    return fake_call_llm


def test_eager_tool_result_is_projected_in_history_but_full_in_event():
    fat = json.dumps({"written": 3, "ops": ["op"] * 50})
    tool = _FakeTool("fake_write", collapse="eager", result=fat)

    events = []
    messages = [{"role": "user", "content": "go"}]
    with patch.dict("silica.agent.loop.TOOLS", {"fake_write": tool}, clear=True), \
         patch("silica.agent.loop.call_llm", _two_turn_llm("fake_write")):
        run_agent(messages, model="test", tool_progress_callback=events.append)

    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    # generic_projection: scalars kept, bulk collapsed to a count the model can
    # re-call to expand. (Tools used to be able to declare their own
    # `summarize`; none ever did, so the hook is gone.)
    assert tool_msgs == [{
        "role": "tool", "tool_call_id": "c1",
        "content": "written=3; ops=<50 items> ⟨↻ re-call to expand⟩",
    }]
    # the TUI event still carried the fat payload
    from silica.agent.events import ToolCompleteEvent
    complete = [e for e in events if isinstance(e, ToolCompleteEvent)]
    assert complete and complete[0].result == fat


def test_lazy_tool_result_stays_verbatim_in_history():
    fat = "x" * 500
    tool = _FakeTool("fake_read", collapse="lazy", result=fat)

    messages = [{"role": "user", "content": "go"}]
    with patch.dict("silica.agent.loop.TOOLS", {"fake_read": tool}, clear=True), \
         patch("silica.agent.loop.call_llm", _two_turn_llm("fake_read")):
        run_agent(messages, model="test")

    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == fat


def test_compact_context_collapses_old_read_and_recounts(monkeypatch):
    from silica import cli
    from silica.config import CONFIG

    big = "x" * 300
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "fake_read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": big},   # 2 — old read, past the floor
        {"role": "assistant", "content": "t1"},
        {"role": "assistant", "content": "t2"},
        {"role": "assistant", "content": "t3"},
    ]
    monkeypatch.setattr(CONFIG, "max_context_tokens", 100)   # budget = 60
    monkeypatch.setattr(CONFIG, "context_tokens", 1_000)     # over budget → trigger

    # A whole registry entry, not just `collapse`: the meter now prices the tool
    # block the turn ships (_chat_tool_schemas), so it reads the same fields the
    # agent loop reads when it builds the schemas for the call.
    fake = SimpleNamespace(collapse="lazy", sensitive=False, internal=False,
                           json_schema=lambda: {"type": "function", "function": {
                               "name": "fake_read", "description": "d", "parameters": {}}})
    with patch.dict("silica.tools.TOOLS", {"fake_read": fake}, clear=True):
        collapsed = cli._compact_context(messages, set())

    assert collapsed == {2}
    assert "re-call fake_read" in messages[2]["content"]
    assert CONFIG.context_tokens < 1_000  # meter refreshed after the collapse


def test_loop_compacts_between_its_own_iterations(monkeypatch):
    """A fat read must not survive to the end of the turn just because the
    callers only sweep once run_agent has already returned. On a pinned local
    window that overrun is truncated in silence, so the loop compacts itself."""
    from silica.config import CONFIG

    big = "x" * 5_000
    tool = _FakeTool("fake_read", "lazy", big)
    calls = [0]

    def fake_call_llm(model, messages, **kw):
        calls[0] += 1
        # Four tool calls, then a text answer: enough turns to push the first
        # read past the 3-turn recency floor.
        if calls[0] <= 4:
            cid = f"c{calls[0]}"
            return SimpleNamespace(
                assistant_message={"role": "assistant", "tool_calls": [
                    {"id": cid, "type": "function",
                     "function": {"name": "fake_read", "arguments": "{}"}}]},
                tool_calls=[SimpleNamespace(id=cid, name="fake_read", args={})],
                text="", reasoning=None,
                usage={"prompt_tokens": 10_000},  # over budget from the first call
            )
        return SimpleNamespace(
            assistant_message={"role": "assistant", "content": "done"},
            tool_calls=[], text="done", reasoning=None, usage={"prompt_tokens": 10_000},
        )

    monkeypatch.setattr(CONFIG, "max_context_tokens", 1_000)  # budget = 600
    messages = [{"role": "user", "content": "q"}]
    with patch.dict("silica.tools.TOOLS", {"fake_read": tool}, clear=True):
        with patch("silica.agent.loop.call_llm", fake_call_llm):
            run_agent(messages=messages, model="m")

    bodies = [m["content"] for m in messages if m.get("role") == "tool"]
    assert any("re-call fake_read" in b for b in bodies), "no read was collapsed mid-turn"
    assert bodies[-1] == big, "the most recent read must stay verbatim"


def test_write_gate_tools_are_classified_eager():
    """Invariant: the write/gate toolset is projected at emission; everything
    else stays lazy (collapsible later, but readable in full by the model)."""
    import silica.cli  # noqa: F401 — registers the full toolset
    from silica.tools import TOOLS

    eager = {n for n, t in TOOLS.items() if t.collapse == "eager"}
    assert eager == {
        "silica_move", "silica_delete", "silica_snapshot", "silica_restore",
        "silica_cleanup", "silica_patch_note", "silica_write_note",
        "silica_flag_note", "silica_event_create", "silica_event_update",
        "silica_autolink", "silica_backlink", "silica_embed_refresh",
        "silica_cooccurrence_refresh", "silica_lexical_refresh", "silica_bulk_write",
        "silica_deferred_retry", "silica_deferred_flush", "silica_run_injector",
        "silica_anneal",
    }


def test_compact_context_noop_under_budget(monkeypatch):
    from silica import cli
    from silica.config import CONFIG

    messages = [{"role": "user", "content": "q"}]
    monkeypatch.setattr(CONFIG, "max_context_tokens", 100_000)
    monkeypatch.setattr(CONFIG, "context_tokens", 10)

    assert cli._compact_context(messages, set()) == set()
    assert CONFIG.context_tokens == 10  # no recount on the no-op path


def test_context_breakdown_parts_sum_to_the_single_count():
    """The context ring prints the three parts and the total side by side, so a
    reader can add them up. litellm bills a fixed chat envelope per CALL, and
    counting three groups separately therefore over-reports by two envelopes;
    _context_breakdown charges it once. Without that subtraction the panel shows
    parts that do not reach the total beside them.
    """
    from silica.cli import (
        _chat_tool_schemas,
        _context_breakdown,
        _count_context_tokens,
    )

    msgs = [
        {"role": "system", "content": "You are silica. " * 40},
        {"role": "user", "content": "what do I know about ethics?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "silica_recall", "arguments": '{"query": "ethics"}'}}]},
        {"role": "tool", "tool_call_id": "a", "content": "Etica.md\n" * 60},
        {"role": "assistant", "content": "You have several notes." * 10},
    ]
    def total(m):
        return _count_context_tokens(m, _chat_tool_schemas(m))

    parts = _context_breakdown(msgs)
    assert sum(parts.values()) == total(msgs)
    # An assistant turn that carries a call is billed to tool_io, not to the
    # conversation: its arguments are what filled the window, and its content is
    # empty by construction.
    assert parts["tool_io"] > parts["messages"] > 0
    assert parts["system"] > 0
    # The schemas the turn ships are the biggest resident of an idle window; a
    # total that leaves them out reads as a fraction of what the provider bills.
    assert parts["tool_specs"] > parts["system"]

    # A window missing a whole group still adds up: the envelope is charged to
    # whichever group happens to be first, not to `system` by name.
    only_user = [msgs[1]]
    assert sum(_context_breakdown(only_user).values()) == total(only_user)
    # An empty window is not an empty prompt: the tool block still ships, so the
    # meter opens on it rather than on zero. Priced as a difference, it moves a
    # few tokens with the messages beside it, which is why this is a band.
    empty = _context_breakdown([])
    assert sum(empty.values()) == total([])
    assert empty["system"] == empty["tool_io"] == empty["messages"] == 0
    assert abs(empty["tool_specs"] - parts["tool_specs"]) < 50
