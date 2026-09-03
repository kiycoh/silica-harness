# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit
import httpx
import openai
from pydantic import BaseModel

from silica.agent.llm import LLMResponse
from silica.config import HOSTED_PROVIDERS, ensure_prefix

logger = logging.getLogger(__name__)

_OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"


def _ollama_base_url() -> str:
    """Ollama's endpoint, honouring its own OLLAMA_HOST convention.

    OLLAMA_HOST is what `ollama` itself reads, so a user who already points the
    CLI at a remote box or a non-default port expects silica to follow. Accepts
    every shape ollama does: `host`, `host:port`, `http://host:port`. Port
    defaults to 11434, matching ollama's client.
    """
    raw = (os.getenv("OLLAMA_HOST") or "").strip()
    if not raw:
        return _OLLAMA_DEFAULT_URL
    try:
        u = urlsplit(raw if "://" in raw else f"http://{raw}")
        return f"{u.scheme}://{u.hostname}:{u.port or 11434}/v1"
    except ValueError:
        # Falling back beats crashing every `silica` import on a typo'd env var;
        # the endpoint check then reports localhost unreachable.
        logger.warning("OLLAMA_HOST=%r is not a valid host — using %s", raw, _OLLAMA_DEFAULT_URL)
        return _OLLAMA_DEFAULT_URL


# Ollama loads a model with a 4096-token window unless told otherwise, and it
# drops whatever does not fit in silence: HTTP 200, no warning (measured — a
# 6645-token prompt came back with prompt_eval_count=2051, the tool definitions
# gone, and the model answered in prose instead of calling a tool). The chat
# toolset alone is ~8k tokens, so the default window cannot hold a single turn.
# Pinned per request rather than left to the server's OLLAMA_CONTEXT_LENGTH,
# which is invisible from inside silica: a user who never set it got garbage.
_OLLAMA_DEFAULT_NUM_CTX = 16384


def ollama_num_ctx() -> int:
    """Context window silica pins on every Ollama request.

    Costs VRAM — the KV cache scales with it — so OLLAMA_NUM_CTX raises or lowers
    it. model_limits caps the effective value at the model's trained maximum.
    """
    raw = (os.getenv("OLLAMA_NUM_CTX") or "").strip()
    if not raw:
        return _OLLAMA_DEFAULT_NUM_CTX
    try:
        return max(int(raw), 1024)
    except ValueError:
        logger.warning("OLLAMA_NUM_CTX=%r is not an integer — using %d",
                       raw, _OLLAMA_DEFAULT_NUM_CTX)
        return _OLLAMA_DEFAULT_NUM_CTX


# Hosted, OpenAI-compatible base URLs. litellm resolves the same prefixes
# natively for the interactive loop; the presets below serve the
# constrained-decoding/distiller path. Gemini goes through its OpenAI-compatible
# endpoint; both loops read the same GEMINI_API_KEY. "custom" (any other
# OpenAI-compatible URL) has no static row — its endpoint comes from
# config.provider_base_url/_api_key (see get_provider).
_HOSTED_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com",
    "mistral": "https://api.mistral.ai/v1",
    "xai": "https://api.x.ai/v1",
}

PROVIDER_PRESETS = {
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio"
    },
    "ollama": {
        "base_url": _ollama_base_url(),
        "api_key": "ollama"  # Ollama ignores it; the OpenAI SDK demands non-empty.
    },
    # Hosted rows derive their key env var from config.HOSTED_PROVIDERS — one
    # table, so the two can't drift (they used to be kept in sync by hand).
    **{
        name: {"base_url": _HOSTED_BASE_URLS[name], "api_key_env": key_env}
        for name, (key_env, _models) in HOSTED_PROVIDERS.items()
    },
}


SILICA_CLI_OPEN = "<silica-cli>"
SILICA_CLI_CLOSE = "</silica-cli>"


# TTL memo, not lru_cache: limits move while silica runs (LM Studio reload
# with a different context window), and a stale window mis-clamps every
# request until restart. The TTL also un-sticks the (0,0) an unreachable
# endpoint used to pin for the whole process. Price: one metadata call per
# provider+model per window.
_MODEL_LIMITS_TTL_S = 600.0
_model_limits_memo: dict[tuple[str, str], tuple[float, tuple[int, int]]] = {}


def model_limits(provider: str, model: str) -> tuple[int, int]:
    """(context_window, max_output_tokens) as reported by the live provider,
    memoized for _MODEL_LIMITS_TTL_S per (provider, model).

    lmstudio   → GET {base}/api/v0/models: `loaded_context_length` (the window
                 the model is loaded with RIGHT NOW, often below its max) with
                 `max_context_length` as fallback. No output cap.
    ollama     → the window silica pins per request (ollama_num_ctx), capped at
                 the trained max from POST {base}/api/show
                 `model_info["<arch>.context_length"]`. Neither the runtime's
                 4096 default nor a Modelfile `num_ctx` can widen or narrow this:
                 request options win over both. No output cap.
    openrouter → GET /api/v1/models: `context_length` plus the top provider's
                 `max_completion_tokens` (often far below the window — e.g.
                 qwen3-8b: 131k ctx, 8k out).

    (0, 0) means unknown/unreachable: callers keep their static defaults
    (and the TTL retries the probe instead of pinning the failure).
    """
    import time

    now = time.monotonic()
    hit = _model_limits_memo.get((provider, model))
    if hit is not None and now < hit[0]:
        return hit[1]
    value = _model_limits_fetch(provider, model)
    _model_limits_memo[(provider, model)] = (now + _MODEL_LIMITS_TTL_S, value)
    return value


def _model_limits_fetch(provider: str, model: str) -> tuple[int, int]:
    try:
        if provider == "ollama":
            base = PROVIDER_PRESETS["ollama"]["base_url"].removesuffix("/v1")
            wanted = model.removeprefix("ollama/")
            info = httpx.post(f"{base}/api/show", json={"model": wanted}, timeout=5.0).json()
            mi = info.get("model_info") or {}
            arch = mi.get("general.architecture", "")
            trained = int(mi.get(f"{arch}.context_length") or next(
                (v for k, v in mi.items() if k.endswith(".context_length")), 0) or 0)
            if not trained:
                return (0, 0)  # model not pulled / unknown arch → caller's default
            return min(ollama_num_ctx(), trained), 0
        if provider == "lmstudio":
            base = PROVIDER_PRESETS["lmstudio"]["base_url"].removesuffix("/v1")
            data = httpx.get(f"{base}/api/v0/models", timeout=5.0).json()["data"]
            wanted = model.removeprefix("lmstudio/")
            entry = next(m for m in data if m["id"] == wanted)
            window = entry.get("loaded_context_length") or entry.get("max_context_length") or 0
            return int(window), 0
        if provider == "openrouter":
            data = httpx.get("https://openrouter.ai/api/v1/models", timeout=5.0).json()["data"]
            wanted = model.removeprefix("openrouter/")
            entry = next(m for m in data if m["id"] == wanted)
            out_cap = (entry.get("top_provider") or {}).get("max_completion_tokens") or 0
            return int(entry.get("context_length") or 0), int(out_cap)
    except Exception as e:
        logger.debug("model_limits(%s, %s) unavailable: %s", provider, model, e)
    return (0, 0)


def clamp_max_tokens(provider: str, model: str, requested: int | None, input_chars: int = 0) -> int:
    """Output-token budget for a request: the caller's ask (or the MAX_TOKENS
    default), never above the provider's live max_completion_tokens, never
    above the window space left after the input.

    Providers validate input + max_tokens <= context window upfront; without
    this clamp a default above either limit makes them reject the request
    (e.g. claude-sonnet-5 on OpenRouter: 128k output cap, 262144 window).
    input_chars is the serialized request size; // 3 overestimates its token
    count (English runs ~4 chars/token, JSON/code closer to 3), which errs on
    the side of a smaller output budget.
    """
    # ponytail: 32768 default keeps the OpenRouter pool wide — cheap endpoints
    # advertise smaller output caps and get dropped above this. 256k measured
    # bad, 32768 measured good; in-between never A/B'd. Override via MAX_TOKENS;
    # sweep the in-between only if real runs start finishing at this ceiling
    # (finish_reason=length on distill).
    want = requested if requested is not None else int(os.getenv("MAX_TOKENS", "32768"))
    window, out_cap = model_limits(provider, model)
    if out_cap:
        want = min(want, out_cap)
    if window:
        # floor 1024 keeps the request well-formed when input nearly
        # fills the window; compaction is the real defense at that point.
        want = min(want, max(window - input_chars // 3, 1024))
    return want


# Keys the history carries for silica's own use and the provider must never see.
# `origin` is turn provenance; `silica_reasoning` is the model's thinking, kept
# so a reopened chat can replay the trace it showed while streaming — sending it
# back would re-bill a multi-thousand-token trace on every later iteration of the
# tool loop, which is exactly why it used to be thrown away.
INTERNAL_KEYS = ("origin", "silica_reasoning")


def _to_wire(msg: dict) -> dict:
    """Strip internal fields and render the CLI marker for the wire.

    The OpenAI message object rejects unknown fields, so `INTERNAL_KEYS` must
    never reach the SDK. When ``origin == "cli"`` the content is wrapped in
    <silica-cli> markers so the model can tell a harness directive apart from a
    human turn. Messages carrying none of them are returned unchanged.
    """
    if not any(k in msg for k in INTERNAL_KEYS):
        return msg
    wire = {k: v for k, v in msg.items() if k not in INTERNAL_KEYS}
    if msg.get("origin") == "cli" and wire.get("content"):
        wire["content"] = f"{SILICA_CLI_OPEN}{wire['content']}{SILICA_CLI_CLOSE}"
    return wire


class Provider:
    """The non-interactive LLM lane: distiller, capability workers, sub-agents.

    One `call_llm(messages, ...) -> LLMResponse`, routed through the same
    `silica.agent.llm.call_llm` the interactive loop uses, so both lanes share
    one retry policy, one wire boundary, one max-token clamp and one set of
    per-provider quirks (ollama's num_ctx, lmstudio's api_base, openrouter's
    routing pin). This used to be a second stack on the raw openai SDK, which
    meant those quirks had to be fixed twice — and one of them, Ollama's silent
    4096-token truncation, was only ever fixed on one side.

    `model` carries its provider prefix (litellm resolves the endpoint from it);
    `api_key` is passed only when the role overrides what litellm would resolve.
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        # Diagnostic only — NOT how the call is routed. llm.call_llm derives the
        # api_base from the model prefix, so changing this steers nothing; it is
        # kept because it is the endpoint a "which box answered?" question means.
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def call_llm(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        openrouter_provider: str | None = None,
        temperature: float | None = None,
        reasoning: bool | None = None,
        cancel: "threading.Event | None" = None,
    ) -> LLMResponse:
        """`cancel` marks the call abandoned: a caller that bounds this with its
        own wall-clock deadline sets it when that deadline fires, so retry_transient
        stops rescheduling instead of issuing requests nobody is waiting for."""
        from silica.agent.llm import call_llm  # lazy: llm.py imports this module

        # Structured decoding defaults to greedy: this path exists to extract a
        # fixed shape, and sampling at the provider default (0.8 on Ollama, 1.0
        # on OpenAI) buys variety nobody wants in an extraction. Ollama's
        # structured-outputs doc calls for temperature 0 explicitly. Pass an
        # explicit temperature to override.
        if temperature is None and response_schema is not None:
            temperature = 0.0
        return call_llm(
            model=self.model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            response_format=response_schema,
            temperature=temperature,
            openrouter_provider=openrouter_provider,
            api_key=self.api_key or None,
            # Forwarded, not defaulted: a hybrid model bills its thinking
            # against max_tokens, and a structured worker call that cannot
            # switch it off gets its JSON cut at the budget (the dedup judge,
            # 2026-08-23). Which workers turn it off is their call, not the
            # lane's: the distiller's output is prose the trace may improve.
            reasoning=reasoning,
            cancel=cancel,
        )


_warned_down: set[tuple[str, str]] = set()


# What a call to `OpenAIEmbedder.embed` can raise when the endpoint, not the
# caller, is at fault: the SDK's own family, the transport under it, and the
# socket errors a local server exposes. A fail-open guard that catches these
# and nothing else lets a programming error surface instead of reading as
# "embedder down" (silica_vaults and the recall probes catch this tuple).
EMBED_ERRORS: tuple[type[BaseException], ...] = (openai.APIError, httpx.HTTPError, OSError)


def _failure_kind(exc: Exception) -> str:
    """'down' | 'rejected' | 'failed' — what the user has to go fix.

    The distinction is whether an HTTP response came back at all: no response
    means nothing is listening (start the server), a 4xx means something IS
    listening and refused the request (wrong model name, or a base_url whose
    path is off). 5xx and non-HTTP errors stay 'failed' — the server is up and
    broken, which is neither.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return "rejected" if 400 <= status < 500 else "failed"
    # Not OSError at large: the local reranker's missing weights raise it too,
    # and "unreachable" would be a lie there.
    if isinstance(exc, (openai.APIConnectionError, httpx.TransportError, ConnectionError)):
        return "down"
    return "failed"


def warn_down_once(
    role: str, where: str, exc: Exception, model: str = "", fallback: str = ""
) -> None:
    """Report a degraded relatedness leg once per (role, kind), then at DEBUG.

    Both legs fail open by design (embed leg abstains, rerank keeps the fused
    order), and every caller swallows the failure at DEBUG — so a server the
    user forgot to start degrades recall invisibly. This is the one line that
    says so, and says which of the two problems it is. Once per kind and not
    per call, because a batch embeds hundreds of times; keying on the kind too
    means the follow-on problem still gets its own line after the first is fixed.
    """
    kind = _failure_kind(exc)
    args: tuple
    if kind == "down":
        msg, args = "%s unreachable at %s (%s)", (role, where, exc)
    elif kind == "rejected":
        msg = "%s at %s is up but rejected model `%s` (%s)"
        args = (role, where, model, exc)
    else:
        msg, args = "%s failed at %s (%s)", (role, where, exc)
    if fallback:
        # A caught failure is not a degradation: saying "recall degrades" here
        # contradicted doctor's own "using in-process fallback" on the same state.
        msg += f" — falling back to {fallback}"
    else:
        msg += " — recall degrades; run `silica doctor`"

    if (role, kind) in _warned_down:
        logger.debug(msg, *args)
        return
    _warned_down.add((role, kind))
    logger.warning(msg, *args)


class OpenAIEmbedder:
    """Thin wrapper for the OpenAI-compatible /v1/embeddings endpoint.

    Uses the same SDK already present in the project. Suitable for any
    provider that speaks the OpenAI API (LM Studio, OpenRouter, etc.).
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        # Mirror the LLM provider's hardening: a granular read-timeout turns a
        # frozen embedding server (e.g. a cold/contended local model) into a
        # fast failure instead of an indefinite hang, and max_retries=1 stops
        # the SDK's default 2 silent retries from stacking 60s waits. COLLISION
        # is best_effort, so a bounded failure degrades to "skip dedup" rather
        # than freezing the run.
        _timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)
        self.client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=_timeout, max_retries=1
        )
        self.base_url = base_url
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text.

        Vectors are normalised by most embedding models; cosine similarity is
        therefore equivalent to dot-product for those models.
        """
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
        except Exception as e:
            # Warn, then re-raise unchanged: the callers' fail-open guards stay
            # in charge of what to skip, this only makes the skip visible.
            warn_down_once("embedder", self.base_url, e, self.model)
            raise
        # The API guarantees ordering matches the input list
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def get_embedder(config: Any) -> OpenAIEmbedder:
    """Return an embedder configured from SilicaConfig."""
    return OpenAIEmbedder(
        base_url=getattr(config, "embedding_base_url", "http://localhost:1234/v1"),
        api_key=getattr(config, "embedding_api_key", "lm-studio"),
        model=getattr(config, "embedding_model", "qwen3-embedding-8b"),
    )


def get_embedder_or_none(config: Any, label: str, *, level: str = "warning") -> OpenAIEmbedder | None:
    """Acquire the embedder, or None if unavailable (logged at `level`).

    Centralizes the 'try get_embedder, on failure log and skip the phase' guard
    each embedding-gated FSM handler repeats; callers keep their own skip action.
    """
    try:
        return get_embedder(config)
    except Exception as e:
        getattr(logger, level)("%s: embedder unavailable (%s) — skipping", label, e)
        return None


class Reranker:
    """Cross-encoder reranker over a served /rerank endpoint.

    Speaks the de-facto protocol (llama.cpp --rerank, Infinity, Jina, Cohere):
    ``POST {model, query, documents} -> {results: [{index, relevance_score}]}``.
    A cross-encoder scores query x document *jointly* — the biggest precision
    lever retrieval has after first-stage recall — so it is used to reorder an
    already-fused candidate pool, never to retrieve.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 5.0):
        self.url = base_url.rstrip("/") + "/rerank"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout
        # Set by FallbackReranker: names the leg that catches this one's
        # abstention, so the down-warning says "falling back to X" instead of
        # claiming a degradation that never happens.
        self.fallback_note = ""

    def scores(self, query: str, documents: list[str]) -> list[float] | None:
        """Relevance score per document in input order, or None to abstain.

        Abstains (None) on any transport or response-shape failure so the caller
        keeps its prior ordering rather than dropping candidates. The short
        timeout keeps a slow reranker from stalling an interactive path.
        """
        if not query or not documents:
            return None
        try:
            resp = httpx.post(
                self.url,
                json={"model": self.model, "query": query, "documents": documents},
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results")
            if not isinstance(results, list):
                return None
            scored = [0.0] * len(documents)
            for r in results:
                i = r.get("index")
                if isinstance(i, int) and 0 <= i < len(documents):
                    scored[i] = float(r.get("relevance_score", r.get("score", 0.0)))
            return scored
        except Exception as e:
            warn_down_once("reranker", self.url, e, self.model, fallback=self.fallback_note)
            return None


# Multilingual by design: a vault is whatever language its owner writes in
# (conventions.language), and bge-reranker-base is English/Chinese only.
# mxbai-rerank-base-v2 (Apache-2.0, Qwen2.5-0.5B backbone) over bge-v2-m3 on the
# link/orphan A/B (bench/local_rerank_query_ab.json, 609 masked wikilink pairs on
# a 795-note Italian vault): mrr +0.055 over the fused baseline at p=2.5e-09,
# where bge scored 125 wins / 100 losses at p=0.109, i.e. indistinguishable from
# not reranking at all. Costs ~390ms per 10-doc call against bge's ~140ms served.
# A HuggingFace repo id, never a GGUF filename — see get_reranker.
LOCAL_RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"


@lru_cache(maxsize=1)
def has_local_rerank() -> bool:
    """Whether the optional [rerank] extra is installed. find_spec, not import:
    get_reranker runs per query and importing torch costs seconds."""
    return importlib.util.find_spec("sentence_transformers") is not None


@lru_cache(maxsize=1)
def _load_cross_encoder(model: str) -> Any:
    """Load the cross-encoder, cached for the process lifetime.

    Cached because get_reranker() is called per query: a fresh CrossEncoder per
    recall would reload ~2GB of weights every time. First call downloads to the
    HF cache.
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model)


class LocalReranker:
    """Cross-encoder reranker in-process, via the optional [rerank] extra.

    Exists because the reranker is the one leg of the stack with nowhere to run:
    LM Studio and Ollama serve generative and embedding models, and a cross-encoder
    is neither — it scores a [query, document] PAIR jointly rather than embedding
    texts independently, so those runtimes either refuse it or (LM Studio) coerce it
    into an embedding model whose output is meaningless for ranking. Without this
    class the only path is a llama-server the user starts and maintains by hand,
    which is why rerank silently never ran for anyone but the eval harness.

    Duck-types Reranker: same .scores() contract, same abstention, so every call
    site and the reorder in kernel/rerank stay untouched.
    """

    def __init__(self, model: str = LOCAL_RERANK_MODEL):
        self.model = model

    def scores(self, query: str, documents: list[str]) -> list[float] | None:
        """Relevance score per document in input order, or None to abstain.

        Abstains on any failure (missing weights, no disk, OOM) so a broken local
        reranker degrades to the fused pool's order, exactly as a down endpoint does.
        """
        if not query or not documents:
            return None
        try:
            encoder = _load_cross_encoder(self.model)
            return [float(s) for s in encoder.predict([[query, d] for d in documents])]
        except Exception as e:
            warn_down_once("local reranker", self.model, e, self.model)
            return None


class FallbackReranker:
    """Prefers a served /rerank endpoint, degrades to the in-process cross-encoder
    per call when it's down, rather than losing the pass entirely.

    Needed because config now defaults `rerank_base_url`/`rerank_model` to a local
    llama-server — a static "endpoint wins when configured" priority (the old
    get_reranker behaviour) would silently strand anyone who installed
    `silica-harness[rerank]` but isn't running that server: Reranker.scores() would
    abstain every call instead of LocalReranker ever getting a turn. Trying the
    served endpoint first and falling back on its own abstention (None) needs no
    separate liveness probe and can't race a check-then-call gap the way a
    probe-first design would.
    """

    def __init__(self, primary: Reranker, secondary: LocalReranker):
        self.primary = primary
        self.secondary = secondary
        primary.fallback_note = (
            f"in-process {getattr(secondary, 'model', 'cross-encoder')}"
        )

    def scores(self, query: str, documents: list[str]) -> list[float] | None:
        scores = self.primary.scores(query, documents)
        return scores if scores is not None else self.secondary.scores(query, documents)


def get_reranker(config: Any) -> Reranker | LocalReranker | FallbackReranker | None:
    """Return a reranker: a served /rerank endpoint (config defaults to a local
    llama-server) when it answers, else the in-process cross-encoder if the
    [rerank] extra is installed, else None (disabled).

    `config.rerank_model` names the model the SERVED endpoint loads and is not a
    valid identity for the in-process leg: the shipped default is a GGUF filename
    ("bge-reranker-v2-m3-Q8_0"), which HuggingFace resolves to a repo that does
    not exist, so LocalReranker abstained on every call and the documented
    fallback never once ran. The local leg therefore always uses
    LOCAL_RERANK_MODEL, which is what `silica doctor` has always reported it uses.
    """
    base_url = getattr(config, "rerank_base_url", "")
    model = getattr(config, "rerank_model", "")
    served = Reranker(
        base_url=base_url, model=model, api_key=getattr(config, "rerank_api_key", ""),
    ) if (base_url and model) else None
    # No per-config override for the local model (declined 2026-08-19): one
    # constant until someone needs two different cross-encoders on one machine.
    if served is not None and has_local_rerank():
        return FallbackReranker(served, LocalReranker(model=LOCAL_RERANK_MODEL))
    if served is not None:
        return served
    if has_local_rerank():
        return LocalReranker(model=LOCAL_RERANK_MODEL)
    return None


def get_provider(config: Any, role: str = "router") -> Provider:
    """Return an LLM provider for the given role.

    role="router" (default) → uses config.provider / config.model (the main model).
    role="worker"            → uses config.worker_provider / config.worker_model so
                               leashed sub-agents can run on a separate small model.
    role="escalation"        → uses config.distill_escalation_provider / _model for escalated steer retries.

    When the worker role specifies an explicit worker_api_key it wins over the
    preset; the endpoint always comes from the worker_provider preset.
    """
    # `or ""` on every read: these fields are Optional on the config, and the
    # "fall back to the router role" test below is a falsiness test either way.
    provider_name: str
    model_name: str
    if role == "escalation":
        provider_name = getattr(config, "distill_escalation_provider", "") or ""
        model_name = getattr(config, "distill_escalation_model", "") or ""
        if not provider_name or not model_name:
            provider_name = getattr(config, "provider", "lmstudio") or ""
            model_name = getattr(config, "model", "") or ""
            role = "router"
    elif role == "worker":
        provider_name = getattr(config, "worker_provider", "") or ""
        model_name = getattr(config, "worker_model", "") or ""
        if not provider_name or not model_name:
            provider_name = getattr(config, "provider", "lmstudio") or ""
            model_name = getattr(config, "model", "") or ""
            role = "router"
    else:
        provider_name = getattr(config, "provider", "lmstudio") or ""
        model_name = getattr(config, "model", "") or ""

    preset = PROVIDER_PRESETS.get(provider_name)
    if preset:
        base_url = preset["base_url"]
        api_key = preset.get("api_key", "lm-studio")
        if "api_key_env" in preset:
            api_key = os.getenv(preset["api_key_env"], "dummy-key")
    else:
        # custom / unknown provider: endpoint from config (SILICA_PROVIDER_BASE_URL
        # / _API_KEY). Falls back to the lmstudio localhost default so a bare
        # misconfig still points somewhere local rather than crashing.
        base_url = getattr(config, "provider_base_url", "") or PROVIDER_PRESETS["lmstudio"]["base_url"]
        api_key = getattr(config, "provider_api_key", "") or "dummy-key"

    # Worker role: explicit api-key override takes precedence over the preset.
    if role == "worker":
        api_key = getattr(config, "worker_api_key", None) or api_key

    # litellm resolves the endpoint from the prefix, so keep it (or restore it
    # when the user wrote a bare id). Ollama in particular MUST keep it: the
    # prefix is what routes it to /api/chat with an explicit num_ctx, and the
    # /v1 endpoint it would otherwise hit cannot size the context window and
    # truncates window-sized prompts in silence (measured: 2051 of 6645 prompt
    # tokens kept at Ollama's 4096 default, zero tool calls, HTTP 200).
    #
    # Same rule as config.ensure_prefix, guard included: the roles below read
    # the very fields it already prefixed, so re-deriving it without the guard
    # made the two disagree. Only a prefix litellm knows can resolve an
    # endpoint, so an unlisted provider (SILICA_WORKER_PROVIDER=vllm) keeps its
    # bare id — config leaves it bare for exactly that reason, and pinning
    # `vllm/` in front of it turns a routable model into a BadRequestError.
    model_name = ensure_prefix(model_name, provider_name)

    # The preset key is only worth passing when it is not what litellm would
    # resolve on its own — the worker role's explicit override. `custom` needs
    # no entry here: llm.call_llm reads provider_api_key off CONFIG directly.
    override = api_key if (role == "worker" and getattr(config, "worker_api_key", None)) else ""
    return Provider(base_url=base_url, api_key=override, model=model_name)
