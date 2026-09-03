"""Tests for M2: Distiller schema wiring and provider preset selection.

ADR-0008 §M2 TDD requirements:
1. run_distiller passes DistillerOutput schema to provider.
2. A schema-valid response is parsed into list[Op] without error.
3. A malformed response (JSON in code-fence, truncated) is recovered by
   parse_json and does not raise an unhandled exception — simulates the
   OpenRouter best-effort branch.
4. get_provider with different preset names produces the same class with
   different base_url / api_key.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from silica.agent.providers import get_provider, Provider, PROVIDER_PRESETS
from silica.agent.llm import LLMResponse
from silica.kernel.write.ops import Op, OpType, DistillerOutput
from silica.kernel.text.sanitize import parse_json

from tests.llm_mocks import litellm_mock_response as _litellm_mock_response


# ---------------------------------------------------------------------------
# Helper: minimal valid Op dict
# ---------------------------------------------------------------------------

def _op_dict(path: str = "1 Cultura/Test.md") -> dict:
    return {
        "op": "write",
        "heading": "Test Concept",
        "source_basename": "test.md",
        "path": path,
        "snippet": "Test content.",
        "hub": None,
    }


# ---------------------------------------------------------------------------
# 1. DistillerOutput schema round-trip
# ---------------------------------------------------------------------------

class TestDistillerOutputSchema:
    def test_valid_updates_parses(self):
        data = {"updates": [_op_dict()]}
        output = DistillerOutput.model_validate(data)
        assert len(output.updates) == 1
        assert isinstance(output.updates[0], Op)
        assert output.updates[0].op == OpType.write

    def test_empty_updates_parses(self):
        output = DistillerOutput.model_validate({"updates": []})
        assert output.updates == []

    def test_op_concepts_default_none(self):
        # #9: concepts is optional and absent by default (backward compatible)
        op = Op.model_validate(_op_dict())
        assert op.concepts is None

    def test_op_accepts_llm_concepts(self):
        # #9: LLM-extracted concept phrases are parsed and preserved
        data = _op_dict()
        data["concepts"] = ["quantum entanglement", "Bell inequality"]
        op = Op.model_validate(data)
        assert op.concepts == ["quantum entanglement", "Bell inequality"]

    def test_invalid_op_type_raises(self):
        with pytest.raises(Exception):
            DistillerOutput.model_validate({"updates": [{"op": "invalid", "heading": "x", "source_basename": "x.md"}]})

    def test_missing_path_for_write_raises(self):
        """Op.path is required for write/patch/overwrite — the @model_validator closes C1."""
        with pytest.raises(Exception):
            DistillerOutput.model_validate({"updates": [
                {"op": "write", "heading": "x", "source_basename": "x.md", "path": None}
            ]})


# ---------------------------------------------------------------------------
# 2. parse_json fallback — simulates OpenRouter best-effort branch
# ---------------------------------------------------------------------------

class TestParseJsonFallback:
    def test_clean_json_parses(self):
        raw = json.dumps({"updates": [_op_dict()]})
        parsed, _ = parse_json(raw, strict=False)
        assert "updates" in parsed

    def test_json_in_code_fence_recovered(self):
        """Simulates a model wrapping its output in ```json ... ```."""
        inner = json.dumps({"updates": [_op_dict()]})
        raw = f"```json\n{inner}\n```"
        parsed, _ = parse_json(raw, strict=False)
        assert "updates" in parsed

    def test_json_with_bom_recovered(self):
        raw = "\ufeff" + json.dumps({"updates": []})
        parsed, _ = parse_json(raw, strict=False)
        assert parsed == {"updates": []}

    def test_completely_invalid_json_raises(self):
        with pytest.raises(Exception):
            parse_json("not json at all <<<", strict=False)


# ---------------------------------------------------------------------------
# 3. Provider preset selection — same class, different config
# ---------------------------------------------------------------------------

class TestProviderPresets:
    def test_lmstudio_preset_base_url(self):
        class Cfg:
            provider = "lmstudio"
            model = "test-model"

        p = get_provider(Cfg())
        assert isinstance(p, Provider)
        assert p.base_url == PROVIDER_PRESETS["lmstudio"]["base_url"]
        assert p.model == "lmstudio/test-model"

    def test_openrouter_preset_same_class(self):
        class Cfg:
            provider = "openrouter"
            model = "or-model"

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            p = get_provider(Cfg())
        assert isinstance(p, Provider)
        assert p.model == "openrouter/or-model"

    def test_unknown_preset_falls_back_to_lmstudio(self):
        class Cfg:
            provider = "nonexistent-provider"
            model = "m"

        p = get_provider(Cfg())
        assert isinstance(p, Provider)
        assert p.base_url == PROVIDER_PRESETS["lmstudio"]["base_url"]

    def test_all_defined_presets_have_base_url(self):
        for name, preset in PROVIDER_PRESETS.items():
            assert "base_url" in preset, f"Preset {name!r} missing base_url"


# ---------------------------------------------------------------------------
# 4. run_distiller wires response_schema=DistillerOutput
# ---------------------------------------------------------------------------

class TestRunDistillerSchemaPassing:
    @patch("silica.agent.providers.get_provider")
    def test_response_schema_passed_to_provider(self, mock_get_provider):
        """run_distiller must pass DistillerOutput as response_schema."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        # Return a valid schema response
        valid_output = json.dumps({"updates": [_op_dict()]})
        mock_provider.call_llm.return_value = LLMResponse(text=valid_output)

        from silica.kernel.prep_delegation import run_distiller
        result = run_distiller(
            payload={"schema_version": "1.0", "batches": []},
            target="1 Cultura/Test",
        )

        # Verify the schema was passed
        call_kwargs = mock_provider.call_llm.call_args
        assert call_kwargs is not None
        schema_arg = call_kwargs.kwargs.get("response_schema")
        assert schema_arg is DistillerOutput

        assert "updates" in result

    @patch("silica.agent.providers.get_provider")
    def test_malformed_response_recovered_by_parse_json(self, mock_get_provider):
        """Malformed output (code-fence wrapped) must not raise — simulates OpenRouter."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        inner = json.dumps({"updates": [_op_dict()]})
        malformed = f"```json\n{inner}\n```"
        mock_provider.call_llm.return_value = LLMResponse(text=malformed)

        from silica.kernel.prep_delegation import run_distiller
        result = run_distiller(
            payload={"schema_version": "1.0", "batches": []},
            target="1 Cultura/Test",
        )
        assert "updates" in result

    @patch("silica.agent.providers.get_provider")
    def test_empty_response_returns_error_dict(self, mock_get_provider):
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_provider.call_llm.return_value = LLMResponse(text="")

        from silica.kernel.prep_delegation import run_distiller
        result = run_distiller(
            payload={"schema_version": "1.0", "batches": []},
            target="1 Cultura/Test",
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# 5. litellm fallback branch also wires response_format=DistillerOutput
#
# When the primary provider call raises, run_distiller falls back to the
# generic litellm call_llm() path. That recovery path must not silently drop
# constrained decoding — it is the branch where reliability matters most.
# ---------------------------------------------------------------------------

class TestRunDistillerFallbackSchemaPassing:
    @patch("litellm.completion")
    @patch("silica.agent.providers.get_provider")
    def test_fallback_passes_response_format_to_litellm(self, mock_get_provider, mock_completion):
        """The litellm fallback must request response_format=DistillerOutput,
        exactly like the primary provider path does via response_schema."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_provider.call_llm.side_effect = RuntimeError("primary provider unavailable")

        valid_output = json.dumps({"updates": [_op_dict()]})
        mock_completion.return_value = _litellm_mock_response(valid_output)

        from silica.kernel.prep_delegation import run_distiller
        result = run_distiller(
            payload={"schema_version": "1.0", "batches": []},
            target="1 Cultura/Test",
        )

        assert mock_completion.called
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs.get("response_format") is DistillerOutput
        assert "updates" in result

    @patch("litellm.completion")
    @patch("silica.agent.providers.get_provider")
    def test_fallback_still_salvages_truncated_prefix(self, mock_get_provider, mock_completion):
        """The truncated-prefix salvage (#1) must keep working when the response
        came from the litellm fallback rather than the primary provider."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_provider.call_llm.side_effect = RuntimeError("primary provider unavailable")

        truncated = (
            '{"main_thematic_axes": ["x"], "updates": ['
            '{"heading": "Done", "op": "skip", "source_basename": "f.md"}, '
            '{"heading": "Cut", "op": "write", "snippet": "half'
        )
        mock_completion.return_value = _litellm_mock_response(truncated, finish_reason="length")

        from silica.kernel.prep_delegation import run_distiller
        result = run_distiller(
            payload={"schema_version": 1, "batches": []},
            target="1 Cultura/Test",
        )

        assert "error" not in result
        assert len(result["updates"]) == 1
        assert result["updates"][0]["heading"] == "Done"


# ---------------------------------------------------------------------------
# #9 prompt instruction: the distiller is told to emit normalized concepts
# ---------------------------------------------------------------------------

def test_distiller_prompt_documents_concepts_field():
    from pathlib import Path
    import silica.capabilities as _c
    prompt = (Path(_c.__file__).parent / "prompts" / "distiller_prompt.txt").read_text(encoding="utf-8")
    # A dedicated instruction block must exist (distinct from the INPUT payload's
    # per-batch "concepts" array, which predates #9).
    assert "Concept Keyphrases" in prompt
    # …explaining the co-occurrence purpose and normalization mandate…
    assert "co-occurrence" in prompt
    lowered = prompt.lower()
    assert "normaliz" in lowered  # normalize / normalized / normalization
