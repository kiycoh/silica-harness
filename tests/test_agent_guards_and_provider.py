from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from silica.config import SilicaConfig
from silica.agent.providers import Provider
from silica.agent.llm import LLMResponse, ToolCall, call_llm
from silica.agent.loop import run_agent, _is_tool_failure
from silica.tools import TOOLS, Tool
from silica.kernel.prep_delegation import run_distiller


class TestAgentGuardsAndProvider(unittest.TestCase):
    def test_dynamic_provider_derivation(self):
        # Default scenario
        config = SilicaConfig()
        config.model = "openrouter/google/gemma-4-31b-it"
        config._provider = None
        self.assertEqual(config.provider, "openrouter")

        # Explicit override in env or field
        config._provider = "lmstudio"
        self.assertEqual(config.provider, "lmstudio")

        # Different model string prefix
        config._provider = None
        config.model = "lmstudio/anthropic/claude"
        self.assertEqual(config.provider, "lmstudio")

        # Non-matching prefix falls back to lmstudio
        config.model = "anthropic/claude"
        self.assertEqual(config.provider, "lmstudio")

    @patch("litellm.completion")
    def test_max_tokens_and_finish_reason_in_litellm(self, mock_completion):
        # Mock choice and message with finish_reason
        mock_choice = MagicMock()
        mock_choice.finish_reason = "length"
        mock_choice.message.content = "Truncated text"
        mock_choice.message.tool_calls = None
        
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = {}
        
        mock_completion.return_value = mock_resp

        # Call with max_tokens
        resp = call_llm(model="test_model", messages=[], max_tokens=100)

        # Assertions
        mock_completion.assert_called_once()
        kwargs = mock_completion.call_args[1]
        self.assertEqual(kwargs.get("max_tokens"), 100)
        self.assertEqual(resp.finish_reason, "length")

    @patch("litellm.completion")
    def test_openrouter_provider_routing(self, mock_completion):
        from silica.config import CONFIG
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = None
        mock_resp = MagicMock(choices=[mock_choice], usage={})
        mock_completion.return_value = mock_resp

        with patch.object(CONFIG, "openrouter_provider", "DeepInfra, Together"):
            call_llm(model="openrouter/xiaomi/mimo-v2.5", messages=[])
        self.assertEqual(
            mock_completion.call_args[1]["extra_body"]["provider"],
            {"order": ["DeepInfra", "Together"], "allow_fallbacks": False},
        )

        # Non-openrouter model: no routing injected even when set.
        with patch.object(CONFIG, "openrouter_provider", "DeepInfra"):
            call_llm(model="lmstudio/local", messages=[])
        self.assertNotIn("extra_body", mock_completion.call_args[1])

        # Unset → default behaviour, no pin. extra_body still exists: every
        # openrouter call carries the session_id OpenRouter groups by.
        with patch.object(CONFIG, "openrouter_provider", ""):
            call_llm(model="openrouter/xiaomi/mimo-v2.5", messages=[])
        self.assertNotIn("provider", mock_completion.call_args[1]["extra_body"])

    @patch("litellm.completion")
    def test_max_tokens_and_finish_reason_in_provider(self, mock_completion):
        from tests.llm_mocks import litellm_mock_response

        mock_completion.return_value = litellm_mock_response(
            text="Truncated structured", finish_reason="length")

        provider = Provider(base_url="http://dummy", api_key="dummy",
                            model="lmstudio/test-model")
        resp = provider.call_llm(messages=[], max_tokens=4000)

        self.assertEqual(mock_completion.call_args[1].get("max_tokens"), 4000)
        # A truncated answer must be reported as such: the distiller decides
        # whether to salvage a partial on exactly this field.
        self.assertEqual(resp.finish_reason, "length")

    @patch("litellm.completion")
    def test_distiller_path_honors_openrouter_provider(self, mock_completion):
        # The distiller lane routes through the same call_llm as the chat loop
        # now, but its own pin (openrouter_provider=) must still reach the wire
        # — and must not leak onto a local endpoint.
        from silica.config import CONFIG
        from tests.llm_mocks import litellm_mock_response

        mock_completion.return_value = litellm_mock_response(text="ok")

        prov = Provider(base_url="https://openrouter.ai/api/v1", api_key="k",
                        model="openrouter/xiaomi/mimo-v2.5")
        prov.call_llm(messages=[], openrouter_provider="DigitalOcean")
        self.assertEqual(
            mock_completion.call_args[1]["extra_body"]["provider"],
            {"order": ["DigitalOcean"], "allow_fallbacks": False},
        )

        # No explicit override: falls back to CONFIG.openrouter_provider.
        mock_completion.reset_mock()
        with patch.object(CONFIG, "openrouter_provider", "Together"):
            prov = Provider(base_url="https://openrouter.ai/api/v1", api_key="k",
                            model="openrouter/m")
            prov.call_llm(messages=[])
        self.assertEqual(
            mock_completion.call_args[1]["extra_body"]["provider"],
            {"order": ["Together"], "allow_fallbacks": False},
        )

        # Local (non-openrouter) model: never injected.
        mock_completion.reset_mock()
        prov = Provider(base_url="http://localhost:1234/v1", api_key="k",
                        model="lmstudio/m")
        prov.call_llm(messages=[], openrouter_provider="DigitalOcean")
        self.assertNotIn("extra_body", mock_completion.call_args[1])

    def test_distiller_provider_config_falls_back_to_general_pin(self):
        # OPENROUTER_PROVIDER_DISTILLER wins when set; otherwise inherits
        # OPENROUTER_PROVIDER so a single pin still covers the distiller.
        import os
        with patch.dict(os.environ, {"OPENROUTER_PROVIDER": "General",
                                     "OPENROUTER_PROVIDER_DISTILLER": "Special"}):
            self.assertEqual(SilicaConfig().openrouter_provider_distiller, "Special")
        with patch.dict(os.environ, {"OPENROUTER_PROVIDER": "General"}, clear=False):
            os.environ.pop("OPENROUTER_PROVIDER_DISTILLER", None)
            self.assertEqual(SilicaConfig().openrouter_provider_distiller, "General")

    @patch("silica.agent.providers.get_provider")
    def test_run_distiller_accepts_complete_output_at_length_limit(self, mock_get_provider):
        # New contract (#1): finish_reason == "length" is no longer fatal by
        # itself. When the emitted JSON is complete and parseable, use it.
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        mock_response = MagicMock()
        mock_response.finish_reason = "length"
        mock_response.text = '{"updates": []}'
        mock_provider.call_llm.return_value = mock_response

        payload = {"schema_version": 1, "batches": []}
        res = run_distiller(payload=payload, target="TargetFolder")

        self.assertNotIn("error", res)
        self.assertEqual(res["updates"], [])

    @patch("silica.agent.providers.get_provider")
    def test_run_distiller_errors_only_on_unrecoverable_truncation(self, mock_get_provider):
        # A length-truncated response with no complete `updates` entry to
        # salvage still surfaces an error.
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        mock_response = MagicMock()
        mock_response.finish_reason = "length"
        mock_response.text = '{"updates": [{"heading": "Cut", "op": "wri'
        mock_provider.call_llm.return_value = mock_response

        payload = {"schema_version": 1, "batches": []}
        res = run_distiller(payload=payload, target="TargetFolder")

        self.assertIn("error", res)

    def test_is_tool_failure(self):
        # Dict representation
        self.assertTrue(_is_tool_failure({"error": "something"}))
        self.assertFalse(_is_tool_failure({"success": True}))

        # String representation: only a structured {"error": ...} payload is a
        # failure. Plain prose is a successful tool output even when it happens
        # to contain words like "error"/"failed" (e.g. grep hits, "0 errors"
        # reports) — substring-sniffing those is a false positive.
        self.assertTrue(_is_tool_failure('{"error": "bad stuff"}'))
        self.assertFalse(_is_tool_failure('An error occurred'))
        self.assertFalse(_is_tool_failure('Failed to read file'))
        self.assertFalse(_is_tool_failure('Exception raised'))
        self.assertFalse(_is_tool_failure('All systems operational'))

    @patch("silica.agent.loop.call_llm")
    def test_convergence_guard_trigger(self, mock_call_llm):
        from pydantic import BaseModel

        class DummyParams(BaseModel):
            param: str

        def failing_fn(param: str):
            return '{"error": "Execution failed"}'

        failing_tool = Tool(
            fn=failing_fn,
            name="failing_tool",
            description="Fails consistently",
            params_model=DummyParams,
            cls="atomic"
        )

        # Mock LLM: the same failing tool call while tools are offered, a text
        # answer on the forced tool-less landing turn.
        resp1 = LLMResponse(
            text=None,
            tool_calls=[ToolCall(id="tc1", name="failing_tool", args={"param": "value"})],
            assistant_message={"role": "assistant", "tool_calls": []},
            usage={}
        )
        landing = LLMResponse(
            text="salvaged findings",
            assistant_message={"role": "assistant"},
            usage={}
        )
        mock_call_llm.side_effect = (
            lambda *a, **kw: landing if kw.get("tools") is None else resp1
        )

        with patch.dict(TOOLS, {"failing_tool": failing_tool}):
            messages = [{"role": "user", "content": "run the tool"}]

            # 3 identical failures stop the loop, but land it the same way the
            # iteration cap does — a 47-step research run is salvaged as an
            # answer, not discarded as a RuntimeError (measured live: one
            # stubbornly re-paraphrased `remember` killed a 30-minute run).
            result = run_agent(messages, model="test_model")

            self.assertEqual(result, "salvaged findings")

            # The warning was still injected at consecutive failure #2
            system_messages = [m for m in messages if m.get("role") == "system"]
            self.assertTrue(any("DO NOT call this tool again with the exact same arguments" in m["content"] for m in system_messages))

            # Landing left no orphaned tool_calls behind for the final call
            for i, m in enumerate(messages):
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    answered = {
                        t["tool_call_id"] for t in messages[i + 1:]
                        if t.get("role") == "tool"
                    }
                    for tc in m["tool_calls"]:
                        self.assertIn(tc["id"], answered)
