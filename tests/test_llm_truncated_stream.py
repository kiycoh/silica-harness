"""A stream that carried nothing must be retried, not returned as an answer.

litellm's stream_chunk_builder cannot tell a stream that ENDED from one that was
CUT: a connection dropped after the opening chunk reassembles into a well-formed
message with content='' and a *synthesised* finish_reason='stop'. Measured
2026-08-24 on openrouter/stealth/ox-alpha — three turns (5.6s / 8.5s / 21.3s,
~/.silica/narration) with completion_tokens=0 and no cached-token detail, which
is what a locally counted usage looks like when the provider never sent one.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import litellm
import pytest

from silica.agent.llm import call_llm
from tests.llm_mocks import litellm_mock_response


def _opening_chunk():
    """What OpenRouter sends first and what a cut stream leaves behind."""
    delta = MagicMock()
    delta.content = ""
    delta.reasoning_content = None
    delta.reasoning = None
    return MagicMock(choices=[MagicMock(delta=delta)])


def _built(content, *, finish_reason="stop", completion_tokens=0):
    resp = litellm_mock_response(content, finish_reason=finish_reason)
    resp.usage.completion_tokens = completion_tokens
    return resp


def test_cut_stream_is_retried_not_answered():
    """Nothing generated and nothing billed -> transport failure -> retry."""
    built = [_built(None), _built("Hello", completion_tokens=20)]

    with patch("litellm.completion", side_effect=lambda **k: iter([_opening_chunk()])), \
         patch("litellm.stream_chunk_builder", side_effect=built), \
         patch("silica.agent.llm.time.sleep"):
        res = call_llm(model="test/model", messages=[{"role": "user", "content": "hi"}],
                       on_delta=lambda t, c: None)

    assert res.text == "Hello", "the retry's answer must reach the caller"


def test_spent_budget_is_not_retried():
    """finish=length with billed tokens is the model's own doing (the whole
    budget went to the trace). Retrying replays it identically, six times."""
    built = _built(None, finish_reason="length", completion_tokens=5)

    with patch("litellm.completion", side_effect=lambda **k: iter([_opening_chunk()])), \
         patch("litellm.stream_chunk_builder", return_value=built) as builder, \
         patch("silica.agent.llm.time.sleep"):
        res = call_llm(model="test/model", messages=[{"role": "user", "content": "hi"}],
                       on_delta=lambda t, c: None)

    assert builder.call_count == 1
    assert not res.text
    assert res.finish_reason == "length"


def test_cut_stream_that_never_recovers_raises():
    """Six cut streams in a row surface as an error, never as an empty answer."""
    with patch("litellm.completion", side_effect=lambda **k: iter([_opening_chunk()])), \
         patch("litellm.stream_chunk_builder", return_value=_built(None)), \
         patch("silica.agent.llm.time.sleep"), \
         pytest.raises(litellm.APIConnectionError):
        call_llm(model="test/model", messages=[{"role": "user", "content": "hi"}],
                 on_delta=lambda t, c: None)
