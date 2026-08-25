"""`doctor --live` must not report a healthy thinking model as broken.

Measured 2026-08-24 on openrouter/stealth/ox-alpha: the probe's 5-token budget
went entirely to the model's trace (finish=length, completion_tokens=5,
reasoning 20 chars, text=''), so a model answering perfectly well in the REPL
scored "live probe: empty reply". The probe is the one place a user is told
whether their model works, so a false red there sends them to fix a key that
was never wrong.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def _resp(text, *, finish_reason="stop", reasoning=None):
    return SimpleNamespace(text=text, finish_reason=finish_reason,
                           reasoning=reasoning, usage={}, tool_calls=[],
                           assistant_message={"role": "assistant"})


def _probe(resp):
    """Run the probe against `resp`, returning (verdict, max_tokens asked for)."""
    from silica import cli
    from silica.config import CONFIG

    seen: dict = {}

    def _fake_call_llm(model, messages, **kw):
        seen.update(kw)
        return resp

    # CONFIG.model and not cli._model_configured: the gate moved into
    # checks.live_probe, which reads the config itself, so patching the CLI
    # helper let the probe skip and every assertion below pass vacuously. It
    # passed anyway on a developer machine, because a provider key in the
    # environment resolves a model — the outcome hung on whose machine ran it,
    # which is how this reached CI green-locally and red there.
    with patch("silica.agent.llm.call_llm", _fake_call_llm), \
         patch.object(CONFIG, "model", "test/probe-model"):
        return cli._doctor_live_probe(), seen.get("max_tokens")


def test_probe_budget_fits_a_reasoning_model():
    """The trace is billed against max_tokens, so 5 cannot fit thinking + a word."""
    _verdict, budget = _probe(_resp("ok"))
    assert budget is not None and budget >= 128, \
        "a budget this tight is a guaranteed red on any thinking model"


def test_trace_that_ate_the_budget_is_not_reported_as_a_dead_model():
    """finish=length with a trace means the model answered — into the trace."""
    verdict, _ = _probe(_resp("", finish_reason="length", reasoning="Let me…"))
    assert verdict is True


def test_genuinely_empty_reply_still_fails():
    verdict, _ = _probe(_resp("", finish_reason="stop"))
    assert verdict is False
