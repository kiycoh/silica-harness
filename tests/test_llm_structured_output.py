"""Tests for structured output support in call_llm (Tier 1 Item 4).

Goal: call_llm accepts a response_format Pydantic model and passes it to
litellm so the model returns valid JSON directly instead of prose + parse_json.
Tests use monkeypatching — no real LLM calls.
"""
from __future__ import annotations

from unittest.mock import patch

from pydantic import BaseModel

from silica.agent.llm import call_llm, LLMResponse

from tests.llm_mocks import litellm_mock_response as _mock_completion


class SimpleSchema(BaseModel):
    title: str
    score: float


def test_call_llm_accepts_response_format_parameter():
    """call_llm must not raise when response_format is a Pydantic model."""
    mock_resp = _mock_completion(text='{"title": "Test", "score": 0.9}')
    with patch("litellm.completion", return_value=mock_resp):
        result = call_llm(
            model="lmstudio/test-model",
            messages=[{"role": "user", "content": "test"}],
            response_format=SimpleSchema,
        )
    assert isinstance(result, LLMResponse)


def test_call_llm_passes_response_format_to_litellm():
    """response_format must be forwarded as response_format kwarg to litellm."""
    mock_resp = _mock_completion(text='{"title": "Test", "score": 0.9}')
    with patch("litellm.completion", return_value=mock_resp) as mock_lit:
        call_llm(
            model="lmstudio/test-model",
            messages=[{"role": "user", "content": "test"}],
            response_format=SimpleSchema,
        )
    call_kwargs = mock_lit.call_args[1]
    assert "response_format" in call_kwargs
    assert call_kwargs["response_format"] is SimpleSchema


def test_call_llm_without_response_format_does_not_pass_kwarg():
    """Without response_format, litellm must not receive the kwarg."""
    mock_resp = _mock_completion(text="plain text")
    with patch("litellm.completion", return_value=mock_resp) as mock_lit:
        call_llm(
            model="lmstudio/test-model",
            messages=[{"role": "user", "content": "test"}],
        )
    call_kwargs = mock_lit.call_args[1]
    assert "response_format" not in call_kwargs


def test_call_llm_response_format_none_not_forwarded():
    """Explicit None must not forward the kwarg."""
    mock_resp = _mock_completion(text="plain text")
    with patch("litellm.completion", return_value=mock_resp) as mock_lit:
        call_llm(
            model="lmstudio/test-model",
            messages=[{"role": "user", "content": "test"}],
            response_format=None,
        )
    call_kwargs = mock_lit.call_args[1]
    assert "response_format" not in call_kwargs


def test_call_llm_returns_the_usage_the_provider_reported():
    """Token accounting is the whole basis of the context meter and the
    max-token clamp, so a response whose usage is dropped bills invisibly.
    This used to be asserted on the OpenAI-SDK provider path, which is gone;
    litellm is now the only lane and has to carry it."""
    mock_resp = _mock_completion(text="hi")
    with patch("litellm.completion", return_value=mock_resp):
        result = call_llm(model="lmstudio/test-model",
                          messages=[{"role": "user", "content": "test"}])
    assert result.usage["prompt_tokens"] == 10
    assert result.usage["completion_tokens"] == 20
    assert result.usage["total_tokens"] == 30


def test_call_llm_reports_empty_usage_rather_than_raising():
    """A provider that reports no usage must yield {}, not an exception."""
    mock_resp = _mock_completion(text="hi")
    mock_resp.usage = None
    with patch("litellm.completion", return_value=mock_resp):
        result = call_llm(model="lmstudio/test-model",
                          messages=[{"role": "user", "content": "test"}])
    assert result.usage == {}


def test_explicit_api_key_overrides_the_prefix_default():
    """get_provider's worker role passes a credential litellm cannot resolve
    on its own; it must survive the per-prefix api_key blocks."""
    mock_resp = _mock_completion(text="hi")
    with patch("litellm.completion", return_value=mock_resp) as mock_lit:
        call_llm(model="custom/m", messages=[{"role": "user", "content": "t"}],
                 api_key="WORKER-KEY")
    assert mock_lit.call_args[1]["api_key"] == "WORKER-KEY"


def test_no_api_key_leaves_the_prefix_default_alone():
    mock_resp = _mock_completion(text="hi")
    with patch("litellm.completion", return_value=mock_resp) as mock_lit:
        call_llm(model="lmstudio/m", messages=[{"role": "user", "content": "t"}])
    assert mock_lit.call_args[1]["api_key"] == "lm-studio"
