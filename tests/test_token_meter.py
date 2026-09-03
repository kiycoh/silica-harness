# SPDX-License-Identifier: AGPL-3.0-or-later
"""Token meter attributes usage to the calling frame, not the LLM plumbing."""
from silica.agent import llm


def _distill(usage):
    return llm.LLMResponse(usage=usage)


def _collision(usage):
    return llm.LLMResponse(usage=usage)


def test_meter_sums_per_callsite(monkeypatch):
    monkeypatch.setattr(llm, "_METER_ON", True)
    llm._meter.clear()

    _distill({"prompt_tokens": 100, "completion_tokens": 10})
    _distill({"prompt_tokens": 50, "completion_tokens": 5,
              "prompt_tokens_details": {"cached_tokens": 40}})
    _collision({"prompt_tokens": 7, "completion_tokens": 3})

    by_fn = {site.split(":")[-1]: counts for site, counts in llm._meter.items()}
    assert by_fn["_distill"] == [2, 150, 15, 40]
    assert by_fn["_collision"] == [1, 7, 3, 0]


def test_cached_tokens_provider_dialects():
    class PTD:  # pydantic-like object (OpenAI SDK shape)
        cached_tokens = 12

    assert llm._cached_tokens({"prompt_tokens_details": {"cached_tokens": 5}}) == 5
    assert llm._cached_tokens({"prompt_tokens_details": PTD()}) == 12
    assert llm._cached_tokens({"cache_read_input_tokens": 9}) == 9  # anthropic
    assert llm._cached_tokens({"prompt_cache_hit_tokens": 4}) == 4  # deepseek native
    assert llm._cached_tokens({"prompt_tokens": 100}) == 0


def test_meter_off_is_noop(monkeypatch):
    monkeypatch.setattr(llm, "_METER_ON", False)
    llm._meter.clear()
    llm.LLMResponse(usage={"prompt_tokens": 999, "completion_tokens": 999})
    assert not llm._meter


# --- Chat-turn prompt floor -------------------------------------------------
# The fixed prefix resent on every LLM call of an interactive turn. Measured
# 2026-07-28 against a real metered turn: ~9.6k tokens, of which the two
# deterministic parts pinned here are ~9.2k (tool schemas 8.2k, system prompt
# 0.95k). The rest (_vault_scope, build_vault_map) is vault-dependent, so it
# can't be pinned. Tokens are approximated as chars/4 — the same proxy
# cli._count_context_tokens falls back to, and it reproduced the metered split
# exactly. Without this a +30% prompt lands silently on every call.
#
# Re-measured 2026-08-14 at 10_298 (schemas 9.2k, system prompt 1.0k). Of the
# +1105 since the first pin, 909 are tool schemas the toolset grew before this
# measurement, and 196 are the in-chat web door: silica_web_answer's schema plus
# the prompt lines that tell the model to use it. The tolerance had absorbed the
# earlier schema growth silently down to 10 tokens of headroom, which is the
# reverse of what the pin is for: re-pin when it fires, and say which part moved.
# Re-measured 2026-08-15 at 11_633: +1335 is the calendar surface — the three
# event tools' schemas (create/update/agenda, ~650 tok after a deliberate
# description trim) joining the chat set, plus their share of the json wrapper.
# Re-measured 2026-08-16 at 11_848: the schemas did not move, the system prompt
# did (1214 -> 1428). +214 is the prompt audit — the recall-vs-search routing
# line, silica_flag_note, and the two report fields the review step 1 was not
# naming. Re-pinned rather than left to the tolerance, per the paragraph above.
# Re-measured 2026-08-17 at 10_663: the OpenViking-informed description trim —
# ~28 tool docstrings cut to routing + result contracts (litellm-measured
# 5_727 -> 4_046 tok), plus the anyOf-null/default-null schema compaction in
# _strip_titles. Deliberate: the whole point was lowering this number.
# Re-measured 2026-09-01 at 11_538: +875 is the cross-vault surface (ADR-0035)
# — silica_vaults joining the chat set (193 tok), `vault=` on silica_recall
# and silica_read_note, and scope/since/limit on silica_changes (218 tok, was
# a no-arg tool). Descriptions were cut to routing + contract before pinning;
# the first draft had fired the ceiling at 11_871.
CHAT_PREFIX_TOKENS = 11_538
CHAT_PREFIX_TOLERANCE = 0.10


def test_chat_prefix_token_floor():
    import json

    import silica.cli  # noqa: F401 — registers the full toolset
    from silica.agent.constraints import chat_tools
    from silica.prompts import system_prompt
    from silica.tools import TOOLS

    schemas = json.dumps(
        [TOOLS[n].json_schema() for n in chat_tools()], ensure_ascii=False
    )
    tokens = (len(schemas) + len(system_prompt())) // 4

    lo = int(CHAT_PREFIX_TOKENS * (1 - CHAT_PREFIX_TOLERANCE))
    hi = int(CHAT_PREFIX_TOKENS * (1 + CHAT_PREFIX_TOLERANCE))
    assert lo <= tokens <= hi, (
        f"chat prefix is {tokens} tok, pinned at {CHAT_PREFIX_TOKENS} "
        f"±{CHAT_PREFIX_TOLERANCE:.0%} — every LLM call of every turn pays it. "
        "If the change is intentional, re-measure and move the constant."
    )
