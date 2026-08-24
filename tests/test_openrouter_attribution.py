"""Silica names itself to OpenRouter, and groups a session's calls under one id.

Without the headers litellm's own defaults win — main.py builds
`{"HTTP-Referer": "https://litellm.ai", "X-Title": "liteLLM"}` and then `update`s
the caller's over them — so every silica generation was filed under liteLLM.
`session_id` is what OpenRouter groups generations by in the dashboard, and also
its sticky-routing key.
"""
from __future__ import annotations

from unittest.mock import patch

from silica.agent.llm import call_llm
from tests.llm_mocks import litellm_mock_response


def _sent(model, **kw):
    """The kwargs silica hands litellm for one call."""
    with patch("litellm.completion", return_value=litellm_mock_response("ok")) as completion:
        call_llm(model=model, messages=[{"role": "user", "content": "hi"}], **kw)
    return completion.call_args.kwargs


def test_openrouter_call_names_silica():
    headers = _sent("openrouter/test/model").get("extra_headers") or {}
    assert headers.get("X-Title") == "Silica"
    assert "silica-harness" in headers.get("HTTP-Referer", "")


def test_session_id_is_the_narration_sid():
    """One id across both records: the OpenRouter session and the file under
    ~/.silica/narration are readable against each other only if they match."""
    from silica.agent import narration as narr

    with patch.object(type(narr.NARRATOR), "sid", property(lambda self: "abc123def456")):
        body = _sent("openrouter/test/model").get("extra_body") or {}
    assert body.get("session_id") == "abc123def456"


def test_session_id_survives_provider_routing_and_the_reasoning_knob():
    """extra_body is shared with the provider pin and the reasoning block —
    whichever is written last must not drop the others."""
    from silica.config import CONFIG

    with patch.object(CONFIG, "openrouter_provider", "Stealth"):
        body = _sent("openrouter/test/model", reasoning=False).get("extra_body") or {}

    assert body.get("session_id")
    assert body["provider"]["order"] == ["Stealth"]
    assert body["reasoning"] == {"enabled": False}


def test_batch_lane_without_a_narration_session_still_groups():
    """nucleate / subagents / the distill pool open no narration session; one id
    per process still beats N loose generations in the dashboard."""
    from silica.agent import narration as narr

    with patch.object(type(narr.NARRATOR), "sid", property(lambda self: None)):
        first = (_sent("openrouter/test/model").get("extra_body") or {}).get("session_id")
        second = (_sent("openrouter/test/model").get("extra_body") or {}).get("session_id")
    assert first and first == second


def test_other_providers_get_neither():
    """Both are OpenRouter dialect: an unknown body field is a 400 elsewhere."""
    sent = _sent("openai/gpt-4o")
    assert "extra_headers" not in sent
    assert "session_id" not in (sent.get("extra_body") or {})
