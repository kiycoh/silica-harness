"""A turn that ends with neither text nor tool calls must say so.

Measured 2026-08-23 on openrouter/stealth/ox-alpha: finish=stop, tool_calls=0,
content=''. run_agent returned "" and the REPL's `if answer:` printed nothing,
so a 21-second turn was indistinguishable from no turn at all.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from silica.agent.loop import run_agent


def _empty_resp(*a, **k):
    return SimpleNamespace(
        assistant_message={"role": "assistant"},
        tool_calls=[], text=None, reasoning=None, usage={},
    )


def test_empty_completion_is_reported_not_swallowed():
    with patch("silica.agent.loop.call_llm", _empty_resp):
        result = run_agent(messages=[{"role": "user", "content": "hi"}], model="test")
    assert result.strip(), "an empty completion must not return an empty string"
    assert "silica:" in result
