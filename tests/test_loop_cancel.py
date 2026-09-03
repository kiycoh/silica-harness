"""Tests for cancel_token integration in run_agent (silica/agent/loop.py).

LLM calls are patched so we never touch real providers.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from silica.agent.loop import run_agent


def _fake_resp(*, tool_calls=None, text="done"):
    """Minimal LLM response stub."""
    resp = SimpleNamespace(
        assistant_message={"role": "assistant", "content": text},
        tool_calls=tool_calls or [],
        text=text,
        reasoning=None, usage={},
    )
    return resp


def test_cancel_before_first_iteration():
    """Token already set → loop exits immediately before the first LLM call."""
    token = threading.Event()
    token.set()

    call_count = 0

    def fake_call_llm(*a, **k):
        nonlocal call_count
        call_count += 1
        return _fake_resp()

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        result = run_agent(
            messages=[{"role": "user", "content": "hi"}],
            model="test",
            cancel_token=token,
        )

    assert result == "(silica: cancelled)"
    assert call_count == 0


def test_cancel_after_first_iteration():
    """Token is set after the first LLM call completes; second call must not happen."""
    token = threading.Event()

    # Without tool calls the loop returns immediately after the first response,
    # so we need a tool call on iteration 1 to force a second LLM call.
    iter_count = [0]

    def fake_call_llm2(*a, **k):
        iter_count[0] += 1
        if iter_count[0] == 1:
            # First LLM call: return a (fake) tool call so loop continues
            tc = SimpleNamespace(
                name="unknown_tool_that_will_error",
                args={},
                id="c1",
            )
            resp = SimpleNamespace(
                assistant_message={"role": "assistant", "content": ""},
                tool_calls=[tc],
                text="",
                reasoning=None, usage={},
            )
            token.set()   # set token AFTER first LLM call returns
            return resp
        # Second LLM call should never happen
        return _fake_resp()

    with patch("silica.agent.loop.call_llm", fake_call_llm2):
        result = run_agent(
            messages=[{"role": "user", "content": "hi"}],
            model="test",
            cancel_token=token,
        )

    assert result == "(silica: cancelled)"
    assert iter_count[0] == 1   # only one LLM call was made


def test_no_cancel_token_runs_normally():
    """When cancel_token is None, the loop runs to completion as before."""
    def fake_call_llm(*a, **k):
        return _fake_resp(text="hello")

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        result = run_agent(
            messages=[{"role": "user", "content": "hi"}],
            model="test",
            cancel_token=None,
        )

    assert result == "hello"


def test_llm_call_runs_on_daemon_thread():
    """The LLM worker must be a daemon thread: a non-daemon orphan gets joined
    at interpreter shutdown, hanging exit while its retries die against
    executors already flagged shut ('cannot schedule new futures after shutdown')."""
    seen = {}

    def fake_call_llm(*a, **k):
        seen["daemon"] = threading.current_thread().daemon
        return _fake_resp()

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        run_agent(messages=[{"role": "user", "content": "hi"}], model="test")

    assert seen.get("daemon") is True


def test_interrupt_sets_cancel_event_on_abandoned_call():
    """Ctrl+C abandons the in-flight call: control returns immediately AND the
    cancel event handed to call_llm is set, so retry_transient stops retrying
    in the background instead of burning attempts for minutes."""
    captured = {}

    def fake_call_llm(*a, **k):
        captured["cancel"] = k.get("cancel")
        raise KeyboardInterrupt  # surfaces from _future.result() like a real Ctrl+C

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        with pytest.raises(KeyboardInterrupt):
            run_agent(messages=[{"role": "user", "content": "hi"}], model="test")

    assert captured.get("cancel") is not None, "call_llm never received a cancel event"
    assert captured["cancel"].is_set()


def test_bus_receives_events_during_run():
    """run_agent publishes agent/* events to BUS even with no callback."""
    import silica.agent.bus as bus_mod
    received = []
    bus_mod.BUS.subscribe("agent/*", received.append)

    def fake_call_llm(*a, **k):
        return _fake_resp(text="hi")

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        run_agent(
            messages=[{"role": "user", "content": "hi"}],
            model="test",
        )

    # At minimum ThinkingStartEvent and ThinkingEndEvent should arrive.
    from silica.agent.events import ThinkingStartEvent, ThinkingEndEvent
    types = [type(e) for e in received]
    assert ThinkingStartEvent in types
    assert ThinkingEndEvent in types
