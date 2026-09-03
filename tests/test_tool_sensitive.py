"""Step 0: tool sensitivity is declared on the Tool, and the main agent's
default toolset (run_agent with constraints=None) excludes sensitive tools.
A sensitive tool is reachable only when a caller names it in AgentConstraints.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from silica.agent.constraints import AgentConstraints
from silica.agent.loop import run_agent
from silica.tools import TOOLS, Tool, tool


class _EmptyArgs(BaseModel):
    pass


def _resp(tool_calls=None, text="done"):
    return SimpleNamespace(
        assistant_message={"role": "assistant", "content": text},
        tool_calls=tool_calls or [],
        text=text,
        reasoning=None, usage={},
    )


def test_tool_defaults_to_not_sensitive():
    t = Tool(lambda: "x", "plain", "doc", _EmptyArgs, "atomic")
    assert t.sensitive is False


def test_decorator_propagates_sensitive():
    @tool(_EmptyArgs, sensitive=True)
    def _probe_sensitive_tool():
        """probe"""
        return "x"

    try:
        assert TOOLS["_probe_sensitive_tool"].sensitive is True
    finally:
        TOOLS.pop("_probe_sensitive_tool", None)


def test_default_toolset_excludes_sensitive():
    """Invariant: default_toolset ∩ {sensitive} == ∅.

    Register a sensitive + a plain tool, capture the schemas run_agent sends
    when constraints is None, and assert only the plain one is exposed.
    """
    TOOLS["_plain_t"] = Tool(lambda: "ok", "_plain_t", "plain", _EmptyArgs, "atomic")
    TOOLS["_secret_t"] = Tool(
        lambda: "no", "_secret_t", "secret", _EmptyArgs, "atomic"
    )
    TOOLS["_secret_t"].sensitive = True
    captured = {}

    def fake_call_llm(model, messages, tools=None, cancel=None):
        captured["tools"] = tools
        return _resp(text="ok")

    try:
        with patch("silica.agent.loop.call_llm", fake_call_llm):
            run_agent(messages=[{"role": "user", "content": "hi"}], model="m")
        names = {t["function"]["name"] for t in (captured["tools"] or [])}
        assert "_plain_t" in names
        assert "_secret_t" not in names
    finally:
        TOOLS.pop("_plain_t", None)
        TOOLS.pop("_secret_t", None)


def test_default_toolset_excludes_internal():
    """Invariant: default_toolset ∩ {internal} == ∅ (same seam as sensitive)."""
    TOOLS["_plain_t"] = Tool(lambda: "ok", "_plain_t", "plain", _EmptyArgs, "atomic")
    TOOLS["_internal_t"] = Tool(
        lambda: "no", "_internal_t", "internal", _EmptyArgs, "composed", internal=True
    )
    captured = {}

    def fake_call_llm(model, messages, tools=None, cancel=None):
        captured["tools"] = tools
        return _resp(text="ok")

    try:
        with patch("silica.agent.loop.call_llm", fake_call_llm):
            run_agent(messages=[{"role": "user", "content": "hi"}], model="m")
        names = {t["function"]["name"] for t in (captured["tools"] or [])}
        assert "_plain_t" in names
        assert "_internal_t" not in names
    finally:
        TOOLS.pop("_plain_t", None)
        TOOLS.pop("_internal_t", None)


def test_pipeline_stage_tools_are_internal():
    """The FSM-driven injector stages must never surface in the chat toolset;
    the agent-facing entry points must."""
    import silica.tools.composed  # noqa: F401 — registers pipeline + wrapped tools
    import silica.tools.wrapped  # noqa: F401

    internal = {
        "silica_recon", "silica_payload", "silica_sanitize", "silica_validate_ops",
        "silica_bulk_write", "silica_lint", "silica_snapshot", "silica_restore",
        "silica_cleanup",
    }
    for name in internal:
        assert TOOLS[name].internal is True, name
    for name in ("silica_run_injector", "silica_deferred_retry",
                 "silica_write_note", "silica_patch_note"):
        assert TOOLS[name].internal is False, name


def test_sensitive_tool_reachable_when_named():
    """A sensitive tool IS exposed when a caller names it in constraints."""
    TOOLS["_secret_t"] = Tool(
        lambda: "ok", "_secret_t", "secret", _EmptyArgs, "atomic"
    )
    TOOLS["_secret_t"].sensitive = True
    captured = {}

    def fake_call_llm(model, messages, tools=None, cancel=None):
        captured["tools"] = tools
        return _resp(text="ok")

    try:
        with patch("silica.agent.loop.call_llm", fake_call_llm):
            run_agent(
                messages=[{"role": "user", "content": "hi"}],
                model="m",
                constraints=AgentConstraints(tools=("_secret_t",), max_iterations=2),
            )
        names = {t["function"]["name"] for t in (captured["tools"] or [])}
        assert names == {"_secret_t"}
    finally:
        TOOLS.pop("_secret_t", None)
