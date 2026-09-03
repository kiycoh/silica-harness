# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Silica configuration — model, vault, provider settings.

Configuration is loaded from (in order of precedence):
  1. Environment variables (SILICA_MODEL, SILICA_VAULT, etc.)
  2. .env file in the project root
  3. Hardcoded defaults

The config module is imported early and provides a singleton CONFIG object
that the rest of the codebase reads from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values, load_dotenv

# Captured at package import (silica/__init__.py), before any third-party
# load_dotenv can blur an exported pin into a .env value. Re-exported here
# because this is where every caller expects to find it.
from silica import SHELL_ENV, VAULT_PINNED  # noqa: E402,F401

# The one ambient .env. Layered *under* the real environment (override=False),
# so a key the shell exported stays a deliberate per-invocation pin. It is the
# user's own file and follows them between folders; an installed silica has no
# .env beside its package, so without it every setting evaporated the moment you
# ran `silica` outside a checkout.
#
# There was a second layer above this one: find_dotenv(usecwd=True), the .env
# found by walking UP from the working directory. Removed, because it has no
# provenance. It is not "silica's project file" — it is the .env of whatever
# repository the shell happens to sit in, and it could repoint the model, the
# endpoints, the vault and the API keys of a user who only cd'd there. The
# *_SERVE_CMD keys used to carry an exception for exactly that reason (it was
# arbitrary command execution); the exception is gone with the layer that needed
# it. Settings that belong to one directory are that directory's job to export —
# `set -a; source .env; set +a` — which is explicit and outranks this file.
USER_ENV = Path.home() / ".silica" / ".env"
load_dotenv(USER_ENV, override=False)

# Silica's ledger of the SILICA_* keys it owns, derived from the two sources
# above rather than snapshotted out of os.environ — a snapshot would be right
# only if nothing had polluted os.environ yet, which makes its correctness a
# question about import order.
#
# Anything under the prefix that is in os.environ but not here did not come from
# a source silica reads. load_dotenv() is a function third parties call at their
# own import, and litellm calls it. With override=False it cannot change a key
# silica already set, but it can ADD one silica deliberately left unset, taken
# from the .env of whatever directory the process happens to sit in — which
# would restore the removed layer, one os.getenv call site at a time.
_SILICA_KEYS = {k for k in SHELL_ENV if k.startswith("SILICA_")}
if USER_ENV.exists():
    _SILICA_KEYS |= {
        k for k in dotenv_values(USER_ENV) if k and k.startswith("SILICA_")
    }


def drop_foreign_env() -> list[str]:
    """Take back out the SILICA_* keys that appeared behind silica's back.

    Called immediately after every `import litellm`; idempotent, and a plain
    dict scan, so a call site that runs often costs nothing measurable.
    """
    foreign = sorted(
        k for k in os.environ if k.startswith("SILICA_") and k not in _SILICA_KEYS
    )
    for key in foreign:
        del os.environ[key]
    return foreign


def claim_env(updates: dict[str, str]) -> None:
    """Make settings live in this process and record them as silica's own.

    The settings panel and the wizard both write a saved key into os.environ so
    the running session sees it. Without the record the next drop_foreign_env()
    would read those keys as injected and take them straight back out.
    """
    os.environ.update(updates)
    _SILICA_KEYS.update(k for k in updates if k.startswith("SILICA_"))


# One spelling of true for every setting silica reads. Kept as a name because
# hand-written call sites drifted apart once: one of them accepted "yes" while
# SILICA_VERBOSE=yes did not work.
_TRUE_WORDS = ("true", "1", "t")


def env_flag(key: str, default: bool) -> bool:
    """Read a boolean setting from the environment.

    True is spelled "true"/"1"/"t" case-insensitively; anything else the user
    writes (including "no", "off" and typos) reads as False, which is what
    every hand-written call site here already did.
    """
    return os.getenv(key, "true" if default else "false").lower() in _TRUE_WORDS


# Provider prefixes that map a `prefix/model` string to an endpoint and get
# auto-prefixed onto a bare model. Single source for the three checks below
# (provider, distill_escalation_provider, ensure_prefix). "custom" routes to
# SILICA_PROVIDER_BASE_URL/_API_KEY; the rest to PROVIDER_PRESETS in
# agent.providers (kept a subset of this set — see test_providers).
PROVIDER_PREFIXES = frozenset({
    "openrouter", "lmstudio", "ollama", "gemini",
    "openai", "groq", "deepseek", "mistral", "xai", "custom",
})


def ensure_prefix(model: str, provider: str | None) -> str:
    """`provider/model`, but only for a provider whose prefix routes somewhere.

    litellm resolves the endpoint from the prefix alone, so a prefix outside
    PROVIDER_PREFIXES does not name an endpoint — it just makes a bare id that
    would have worked unroutable. Those keep the id as written.

    Module-level because agent.providers.get_provider re-derives the same prefix
    from the same fields: two copies of this rule drifted apart once already
    (the copy without the guard emitted `vllm/qwen`).
    """
    if model and provider and provider in PROVIDER_PREFIXES:
        if not model.startswith(f"{provider}/"):
            return f"{provider}/{model}"
    return model

# Hosted providers in fallback order: (API key env var, model ids best first).
# One table, three readers — `model_from_env` takes the head of the first entry
# whose key is exported, the wizard offers the whole list as a pick-list, and
# agent.providers derives its PROVIDER_PRESETS hosted rows from it. Lives here
# rather than beside PROVIDER_PRESETS because config is imported on every path,
# including the ones that must not pay for the openai SDK.
# The wizard validates this list against each provider's live /models before
# offering it (wizard._live_hosted_models), so stale ids stop being offered.
# ponytail: the static heads still back model_from_env, which must stay
# HTTP-free on config load — a head can rot for env-derived defaults; refresh
# it when a provider retires a model.
HOSTED_PROVIDERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "openrouter": ("OPENROUTER_API_KEY", (
        "openrouter/deepseek/deepseek-v4-flash",
        "openrouter/anthropic/claude-sonnet-5",
        "openrouter/google/gemini-3.5-flash",
        "openrouter/mistralai/mistral-small-2603",
    )),
    "gemini": ("GEMINI_API_KEY", ("gemini/gemini-2.5-flash",)),
    "openai": ("OPENAI_API_KEY", ("openai/gpt-4o",)),
    "groq": ("GROQ_API_KEY", ("groq/llama-3.3-70b-versatile",)),
    "deepseek": ("DEEPSEEK_API_KEY", ("deepseek/deepseek-chat",)),
    "mistral": ("MISTRAL_API_KEY", ("mistral/mistral-large-latest",)),
    "xai": ("XAI_API_KEY", ("xai/grok-2-latest",)),
}


def model_from_env() -> tuple[str, str]:
    """Resolve the chat model: (model id, source env var).

    SILICA_MODEL wins. Failing that, the first hosted provider whose API key is
    already exported answers for it — a user who has OPENROUTER_API_KEY in their
    shell should not have to name a model before silica will run. Returns
    ("", "") when nothing answers, which keeps the fail-fast for a bare install.

    Env keys only: probing a local endpoint from here would put an HTTP call on
    every config load. SILICA_PROVIDER set means the user pinned an endpoint
    (custom, ollama, ...) and a hosted guess would contradict it, so the chain
    stands down.
    """
    explicit = os.getenv("SILICA_MODEL", "").strip()
    if explicit:
        return explicit, "SILICA_MODEL"
    if os.getenv("SILICA_PROVIDER"):
        return "", ""
    for key_env, models in HOSTED_PROVIDERS.values():
        if os.getenv(key_env):
            return models[0], key_env
    return "", ""


@dataclass
class SilicaConfig:
    """Runtime configuration singleton."""

    # LLM provider — litellm model string. SILICA_MODEL, else derived from
    # whichever provider key is already exported (see model_from_env), else
    # empty: the REPL then points the user to `silica init` rather than assume
    # a hosted model whose API key was never mentioned.
    # Examples: "openrouter/anthropic/claude-sonnet-4-20250514", "qwen3-30b"
    model: str = field(
        default_factory=lambda: model_from_env()[0]
    )

    # Provider preset name (derived from model prefix by default, or overridden)
    _provider: str | None = field(
        default_factory=lambda: os.getenv("SILICA_PROVIDER", None)
    )

    # Custom OpenAI-compatible endpoint (provider="custom"): base URL + key.
    # Covers any server speaking the OpenAI API without a dedicated preset —
    # vLLM, llama.cpp, LocalAI, Jan, or a hosted vendor we don't preset.
    provider_base_url: str = field(
        default_factory=lambda: os.getenv("SILICA_PROVIDER_BASE_URL", "")
    )
    provider_api_key: str = field(
        default_factory=lambda: os.getenv("SILICA_PROVIDER_API_KEY", "")
    )

    # OpenRouter upstream-provider routing (agent/llm.py). Comma-separated
    # provider names (e.g. "DeepInfra,Together") pinned as the routing `order`
    # for openrouter/* models; unset → OpenRouter's default auto-routing (as now).
    openrouter_provider: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_PROVIDER", "")
    )

    # Distiller-only upstream-provider pin. Lets the constrained-decoding path
    # (kernel.prep_delegation.run_distiller) route to a different OpenRouter
    # provider than the interactive loop and the other workers. Falls back to
    # OPENROUTER_PROVIDER when unset, so a single pin still covers everything.
    openrouter_provider_distiller: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_PROVIDER_DISTILLER", "")
        or os.getenv("OPENROUTER_PROVIDER", "")
    )

    @property
    def provider(self) -> str:
        if self._provider is not None:
            return self._provider
        if self.model and "/" in self.model:
            prefix = self.model.split("/", 1)[0]
            if prefix in PROVIDER_PREFIXES:
                return prefix
        return "lmstudio"

    @provider.setter
    def provider(self, val: str) -> None:
        self._provider = val

    @property
    def distill_escalation_provider(self) -> str | None:
        """Escalation provider: explicit env wins, else derived from the model
        prefix (same rule as the main model), else lmstudio for a bare name,
        else None (get_provider then degrades the role to router)."""
        if self._distill_escalation_provider is not None:
            return self._distill_escalation_provider
        m = self.distill_escalation_model
        if not m:
            return None
        if "/" in m:
            prefix = m.split("/", 1)[0]
            if prefix in PROVIDER_PREFIXES:
                return prefix
        return "lmstudio"

    # --- Sub-agent worker model (leashed sub-agents run on a separate, smaller model) ---
    # The router (agent loop) uses `model`/`provider` above; sub-agents (dedup, refiner)
    # use these worker_* fields so they can run concurrently on a small local model.
    worker_model: str | None = field(
        default_factory=lambda: os.getenv("SILICA_WORKER_MODEL", None)
    )
    # Worker provider preset name; falls back to "lmstudio" when unset.
    worker_provider: str | None = field(
        default_factory=lambda: os.getenv("SILICA_WORKER_PROVIDER", None)
    )
    # Explicit API-key override for the worker model (endpoint comes from the preset).
    worker_api_key: str | None = field(
        default_factory=lambda: os.getenv("SILICA_WORKER_API_KEY", None)
    )

    # --- Distiller escalation model (Tier 2 cascade) ---
    # A VALIDATE rejection escalates the steer retry to this model instead of
    # re-steering the worker (UCCI-style cascade). Unset: escalation falls back
    # to the router model. Opt-out: set it equal to the worker model.
    distill_escalation_model: str | None = field(
        default_factory=lambda: os.getenv("SILICA_DISTILL_ESCALATION_MODEL", None)
    )
    _distill_escalation_provider: str | None = field(
        default_factory=lambda: os.getenv("SILICA_DISTILL_ESCALATION_PROVIDER", None)
    )


    subagent_max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("SILICA_SUBAGENT_MAX_CONCURRENT", "3"))
    )
    # Global ceiling on concurrent worker-model LLM calls (the one true
    # concurrency budget; see ADR-0004). Sized to the worker backend
    # (API rate limit or local GPU slots).
    worker_max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("SILICA_WORKER_MAX_CONCURRENT", "4"))
    )

    # Distiller prefetch width for /ingest (Tier 1 speed): how many chunk
    # distillations may be in flight at once. 1 = fully sequential. Default is 3
    # since the 2026-07-18 k=1-vs-k=3 staleness A/B (bench/kway_diff.py): a
    # lookahead chunk's staler ledger_digest diverged from a k=1 baseline no more
    # than a second k=1 run did (title agreement k1/k3 0.355 >= k1/k1 0.303) —
    # the staleness effect sits inside the pipeline's own run-to-run noise.
    distill_concurrency: int = field(
        default_factory=lambda: int(os.getenv("SILICA_DISTILL_CONCURRENCY", "3"))
    )

    # Tier 2 novelty gate (SAGE-style): a concept whose top vault candidate
    # scores at or above this cosine leaves the payload BEFORE chunking and
    # goes to the dedup-judge lane (deferred store + concurrent ternary judge).
    # 0 = gate off. Flip the default to 0.93 only after the bench A/B passes
    # (see docs spec 2026-07-18-ingest-tier2-cost-design).
    novelty_tau: float = field(
        default_factory=lambda: float(os.getenv("SILICA_NOVELTY_TAU", "0"))
    )

    # Vault path — used by the fs backend and for context.
    vault_path: str = field(
        default_factory=lambda: os.getenv("SILICA_VAULT", "")
    )

    # Obsidian vault display name (prompt fallback when no vault path is set).
    vault_name: str = field(
        default_factory=lambda: os.getenv("SILICA_VAULT_NAME", "")
    )

    # Personal-memory vault — the second recall lane (ADR-0019). Read-only at
    # query time: its (embed, cooccur) stores join the RRF fusion; writes never
    # route here. Empty ⇒ the default user vault (~/.silica/vault). When it
    # resolves to the SAME path as the active vault the lane abstains and
    # behavior is bit-identical to single-vault.
    memory_vault: str = field(
        default_factory=lambda: os.getenv("SILICA_MEMORY_VAULT", "")
    )

    # Capture of Silica's OWN sessions (capture.py), default off: opting in
    # deposits each conversation in the WAL, from which /nucleate distills
    # facts into the episodic store. Machine memory never becomes a note by
    # itself — promotion is the only path into the vault. /incognito turns it
    # off for the running session without touching this.
    capture_sessions: bool = field(
        default_factory=lambda: env_flag("SILICA_CAPTURE_SESSIONS", False)
    )

    # Episodic memory lane (kernel/episodic.py): wall-clock TTL in days from a
    # fact chain's last_seen (0 = never expire), and the distinct-run count at
    # which a key becomes a nucleation candidate in the digest.
    episodic_ttl_days: int = field(
        default_factory=lambda: int(os.getenv("SILICA_EPISODIC_TTL_DAYS", "90"))
    )
    episodic_nucleation_runs: int = field(
        default_factory=lambda: int(os.getenv("SILICA_EPISODIC_NUCLEATION_RUNS", "3"))
    )
    # Supersede gate (key-collision diagnosis 2026-08-02): minimum TEXT cosine
    # between a same-key arrival and the live head it would bury for the
    # supersede to proceed; below it the arrival FORKS a sibling live chain
    # instead, so distinct facts sharing a slotty key ("event_date" holding
    # five different events) stop erasing each other. 0 = off (always
    # supersede, the pre-gate behavior). The probe
    # (bench/supersede_gate_probe.json, 1081 pairs) and the mid-band hand-label
    # (bench/supersede_gate_midband_labels.json) size the tau: genuine updates
    # sit >= ~0.83, collisions center at 0.53, and the 0.55-0.70 band is ~56%
    # collision with no internal separation — so if it is ever armed the tau
    # belongs at the band's TOP, i.e. 0.70. A false fork costs one near-duplicate live fact; a
    # missed fork fabricates a retraction, so the asymmetry picks the
    # aggressive end. Replay evidence (bench/gate_replay.py): on
    # conv-26 live facts go 107 → 134, rescuing 27 of 29 burials while keeping
    # 2 real supersedes; on the worst store 20 → 190, rescuing 170 while
    # keeping 20 genuine update chains. Known miss: event_date school→workshop
    # scores 0.771 and still buries — the residue needs the distiller's key
    # contract, not a lower tau. Band is qwen3-embedding-4b-relative;
    # re-measure before trusting under another embedder. Gate abstains (legacy
    # supersede) when either vector is unavailable.
    # SHIPS OFF, deliberately. The answer-path A/B (bench/gate_answer_ab.py,
    # conv-26, 199 questions, episodic block only) came back NULL: 52.8% ->
    # 53.8%, discordant 14/12, McNemar exact p=0.845. The gate demonstrably
    # fixes store integrity (it stops fabricated retractions) but buys nothing
    # measurable on the product metric, so it stays behind the flag until
    # something moves. Post-hoc and uncorrected, therefore only a hypothesis for
    # the re-run: single-hop rose (discordant 10/2, p=0.039, the predicted
    # direction since one buried fact is exactly what those questions need)
    # while abstention (0/3) and open-domain (0/4) fell, consistent with more
    # live facts giving the model more material to over-answer from.
    # Before arming: re-run the A/B across more conversations (n=1 here against
    # the 17 stores the collision scan covered) and watch abstention.
    # evals/locomo/runner.py pins this to 0 regardless, so frozen baselines stay
    # comparable even if the default moves.
    episodic_supersede_tau: float = field(
        default_factory=lambda: float(os.getenv("SILICA_EPISODIC_SUPERSEDE_TAU", "0"))
    )

    # Relevance floor on the episodic embed leg (cosine). Without one, top-k
    # over `score > 0` ships the whole store on every query: measured on a
    # 11-fact store, "pasta recipe with tomatoes" recalled the same 10
    # AI-history facts as an on-topic query, ~520 tokens of noise per recall.
    # Calibration knob: 0.5 separates this embedder's off-topic ceiling (0.464
    # over 5 unrelated queries) from its true matches (0.598, 0.833). A
    # different embedder shifts the whole band — re-measure before trusting it.
    # 0 = off (pre-floor behavior).
    episodic_recall_floor: float = field(
        default_factory=lambda: float(os.getenv("SILICA_EPISODIC_RECALL_FLOOR", "0.5"))
    )

    # Obsidian WebSocket bridge (backend="ws"): port `silica connect` binds (0 →
    # OS picks a free one) and the shared token (empty → minted on first connect,
    # written to <vault>/.obsidian/silica-bridge.json). The wire contract is
    # PROTOCOL.md in github.com/kiycoh/obsidian-silica — change both sides together.
    ws_port: int = field(
        default_factory=lambda: int(os.getenv("SILICA_WS_PORT", "0"))
    )
    ws_token: str = field(
        default_factory=lambda: os.getenv("SILICA_WS_TOKEN", "")
    )

    # Inbox folder inside the vault — used to archive and blacklist staging files.
    inbox_dir: str = field(
        default_factory=lambda: os.getenv("SILICA_INBOX_DIR", "Inbox")
    )

    # PDF→Markdown converter (ADR-0011 provider seam):
    # "pdfium" (default, BSD/Apache, in the base install — one ~3 MB wheel, no
    # torch and no JVM, reads the PDF outline for headings, but NO OCR), "mineru"
    # (best fidelity and the only OCR path; `pip install 'mineru[pipeline]'`,
    # 3.8 GB of torch+CUDA plus model downloads), "docling" (MIT but its PDF
    # pipeline hard-imports docling-ibm-models, so torch is unavoidable), or
    # "opendataloader" (Apache-2.0, strong on complex tables and multi-column
    # reading order, needs a JVM). Non-PDF formats (DOCX/EPUB/FB2) have their own
    # in-process readers — the providers only take PDFs. All three non-default
    # providers are binaries the user installs; none is a dependency of Silica,
    # and the doctor's converters row reports whether mineru is on PATH. The
    # pre-2026-08-31 value "pymupdf" is accepted as an alias of the default
    # (convert.resolve_pdf_provider). An unmet provider errors clearly.
    pdf_provider: str = field(
        default_factory=lambda: os.getenv("SILICA_PDF_PROVIDER", "pdfium")
    )

    # OCR languages for PDF conversion, comma-separated (split at point of use).
    # Only docling consumes it: mineru 3.x has no latin-script language option
    # (its default `ch` models cover latin), opendataloader only OCRs in its
    # generative `hybrid` mode, which we never enable, and pdfium has no OCR at
    # all. Default keeps docling's European coverage and adds Italian; all
    # latin-script languages share one EasyOCR model, so the list is cheap.
    # Language detection can't replace this: for a scanned PDF there is no text
    # to detect from until OCR runs.
    pdf_ocr_lang: str = field(
        default_factory=lambda: os.getenv("SILICA_PDF_OCR_LANG", "en,it,fr,de,es")
    )

    # Tavily API key: the /web-search backstop when DuckDuckGo challenges us.
    # Empty is fine — DuckDuckGo is the primary lane and needs no key.
    tavily_api_key: str = field(
        default_factory=lambda: os.getenv("SILICA_TAVILY_API_KEY", "")
        or os.getenv("TAVILY_API_KEY", "")
    )

    # Maximum context tokens before the agent warns.
    max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("SILICA_MAX_CONTEXT", "60000"))
    )

    # Tool progress display level (REPL-runtime, cycled with /verbose)
    # off     — total silence, only the final response
    # new     — shows the tool name only when it changes
    # all     — every tool call with an args preview (default)
    # verbose — full args, truncated result, duration
    tool_progress: Literal["off", "new", "all", "verbose"] = field(
        default_factory=lambda: os.getenv("SILICA_TOOL_PROGRESS", "all")  # type: ignore
    )

    # Debug logging to stderr (--verbose / -v CLI flag, not cycled)
    debug_logging: bool = field(
        default_factory=lambda: env_flag("SILICA_VERBOSE", False)
    )

    # Shows the model's reasoning blocks (runtime toggle with /thinking)
    show_thinking: bool = field(
        default_factory=lambda: env_flag("SILICA_SHOW_THINKING", True)
    )

    # Runtime session state — updated by cli.py after each agent turn
    context_tokens: int = 0

    # Startup banner art (True → wordmark, False → plain one-liner)
    show_banner: bool = field(
        default_factory=lambda: env_flag("SILICA_SHOW_BANNER", True)
    )

    # Graph viewer: the drifting particles on the GAP and SIMILAR edges (both
    # renderers), and the crystal rig in 3D — flat shading, the two lights, the
    # camera-following fog. Off in either case is the bundle's own default, and
    # off costs nothing: particles alone hold the canvas awake at IDLE_FPS
    # forever, which on a settled graph is the largest standing cost in that view.
    graph_particles: bool = field(
        default_factory=lambda: env_flag("SILICA_GRAPH_PARTICLES", True)
    )
    graph_shading: bool = field(
        default_factory=lambda: env_flag("SILICA_GRAPH_SHADING", True)
    )

    # The chat landing's second line: a sentence the worker model writes about
    # what the vault holds. The counted line above it — notes, areas, the topic
    # labels themselves — is computed and always shows. This is the part that
    # costs a call, so it is the part that can be switched off.
    vault_brief: bool = field(
        default_factory=lambda: env_flag("SILICA_VAULT_BRIEF", True)
    )

    # Web UI palette: "auto" follows the OS, "dark" and "light" pin it. Only the
    # browser can answer "auto", so the server ships the preference and a script
    # in <head> resolves it to a concrete data-theme before first paint. The
    # terminal UI has its own palette and does not read this.
    theme: str = field(default_factory=lambda: os.getenv("SILICA_THEME", "auto").lower())

    # Embedding model — used by silica/kernel/recall/embed.py (Phase 3)
    # Default targets a local llama-server (`llama-server -m ... --embedding`) or
    # LM Studio, whichever answers at the URL below — both speak the same
    # OpenAI-compatible /v1/embeddings shape, and a single-model server ignores
    # the `model` field anyway. "text-embedding-qwen3-embedding-4b" is LM
    # Studio's id for qwen3-embedding-4b; the id is cosmetic when llama-server is
    # what's actually listening. Example alternatives: "qwen3-embedding-8b",
    # "text-embedding-3-small" (OpenAI), "nomic-embed-text" (Ollama).
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "SILICA_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b"
        )
    )

    # Base URL for the embeddings endpoint — a local llama-server or LM Studio
    # instance by default.
    embedding_base_url: str = field(
        default_factory=lambda: os.getenv("SILICA_EMBEDDING_BASE_URL", "http://localhost:1234/v1")
    )

    # API key for embeddings endpoint (local runtimes ignore it; any non-empty
    # value satisfies the OpenAI SDK)
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv("SILICA_EMBEDDING_API_KEY", "lm-studio")
    )

    # Cross-encoder reranker: the precision pass over the fused candidate pool.
    # Neither LM Studio nor Ollama can serve one (it scores a [query, document]
    # pair jointly, not an embedding), so the default here is a local llama-server
    # started with --reranking. get_reranker (agent/providers.py) tries it first
    # and falls back automatically, per call, to the in-process cross-encoder
    # from `pip install silica-harness[rerank]` when it's down — set both empty to
    # skip straight to that path, or leave the extra uninstalled too to disable
    # reranking outright (a no-op that preserves the pool's order).
    rerank_base_url: str = field(
        default_factory=lambda: os.getenv("SILICA_RERANK_BASE_URL", "http://localhost:1235/v1")
    )
    rerank_model: str = field(
        default_factory=lambda: os.getenv("SILICA_RERANK_MODEL", "bge-reranker-v2-m3-Q8_0")
    )
    rerank_api_key: str = field(
        default_factory=lambda: os.getenv("SILICA_RERANK_API_KEY", "lm-studio")
    )

    # Speech-to-text: one endpoint for both lanes — the web GUI's dictation
    # button and /convert's media transcription (audio/video -> inbox note).
    # Any OpenAI-compatible /audio/transcriptions server (whisper.cpp's
    # whisper-server, faster-whisper-server, OpenAI itself). Next port after
    # embeddings (1234) and rerank (1235); SILICA_STT_SERVE_CMD starts it.
    # Empty turns the dictation button off outright.
    # The SILICA_ASR_* keys are the pre-merge spelling of this family and are
    # still honoured, so an existing .env keeps working; SILICA_STT_* wins.
    stt_base_url: str = field(
        default_factory=lambda: os.getenv("SILICA_STT_BASE_URL", "")
        or os.getenv("SILICA_ASR_BASE_URL", "")
        or "http://localhost:1236/v1"
    )
    # Cosmetic against whisper.cpp, which serves whatever model it was started
    # with; a hosted endpoint needs the real id.
    stt_model: str = field(
        default_factory=lambda: os.getenv("SILICA_STT_MODEL", "")
        or os.getenv("SILICA_ASR_MODEL", "")
        or "whisper-1"
    )
    # whisper-server assumes English when a request names no language, so a vault
    # dictated in any other one comes back translated instead of transcribed.
    # "auto" asks it to detect; a fixed code ("it", "en") is steadier on short
    # clips. Dictation sends "auto" through; /convert omits the field for it,
    # which is what its own default (empty) always did.
    stt_lang: str = field(
        default_factory=lambda: os.getenv("SILICA_STT_LANG", "")
        or os.getenv("SILICA_ASR_LANG", "")
        or "auto"
    )
    stt_api_key: str = field(
        default_factory=lambda: os.getenv("SILICA_STT_API_KEY", "lm-studio")
    )

    # "endpoint" (the server above) or "whispercpp" (spawn the CLI locally).
    # /convert honours this; dictation always goes to the endpoint, since the
    # browser is already talking to the server.
    stt_provider: str = field(
        default_factory=lambda: os.getenv("SILICA_STT_PROVIDER", "")
        or os.getenv("SILICA_ASR_PROVIDER", "")
        or "endpoint"
    )
    # Path to (or name on PATH of) the whisper.cpp CLI, for stt_provider=whispercpp.
    # Upstream renamed `main` to `whisper-cli` in 2024; both names are tried.
    stt_whispercpp_bin: str = field(
        default_factory=lambda: os.getenv("SILICA_STT_WHISPERCPP_BIN", "")
        or os.getenv("SILICA_ASR_WHISPERCPP_BIN", "")
    )
    # whisper.cpp needs an explicit model file (`-m`); there is no default it can
    # find on its own, so this is required for that provider.
    stt_whispercpp_model: str = field(
        default_factory=lambda: os.getenv("SILICA_STT_WHISPERCPP_MODEL", "")
        or os.getenv("SILICA_ASR_WHISPERCPP_MODEL", "")
    )

    # COLLISION routing thresholds on the cosine of the cosine-best vault note
    # (ADR-0030). Measured 2026-08-23 as a risk-coverage curve (677 human-kept
    # notes as distinct, 60 synthetic duplicates): the high bar at 0.85 auto-merged
    # one pair, which the vault itself holds twice (a true duplicate), and 0.88
    # none; the low bar is where the leaks were. At 0.75, 28% of the duplicates
    # scored under it against their own source and were written as new notes; at
    # 0.70 it is 10% for 1.65x the judge calls, at 0.66 3% for 2.1x. 0.70 is the
    # 2%-risk point of the curve; the judge on the 0.66-0.75 band was measured at
    # 0.85 accuracy with the prompt of ADR-0030, so what it is handed it decides.
    # A repo .env that pins SILICA_SIM_THRESHOLD_LOW keeps its own value.
    sim_threshold_high: float = field(
        default_factory=lambda: float(os.getenv("SILICA_SIM_THRESHOLD_HIGH", "0.85"))
    )
    sim_threshold_low: float = field(
        default_factory=lambda: float(os.getenv("SILICA_SIM_THRESHOLD_LOW", "0.70"))
    )

    # Number of candidates to retrieve per note during dedup scan.
    # Higher values increase recall at negligible BLAS cost (search is a single
    # matrix-vector product). k=1 misses borderline secondary matches when the
    # primary match lands above τ_high and is discarded.
    dedup_scan_k: int = field(
        default_factory=lambda: int(os.getenv("SILICA_DEDUP_SCAN_K", "5"))
    )

    # Minimum title-only cosine similarity to promote a pair into the dedup
    # borderline window, regardless of the full-note score.
    # Set higher than sim_threshold_low (0.75) to avoid spurious matches between
    # generically related titles (e.g. "Python" / "Python async").
    sim_title_threshold: float = field(
        default_factory=lambda: float(os.getenv("SILICA_SIM_TITLE_THRESHOLD", "0.80"))
    )

    # Language for the co-occurrence graph stemmer + stopwords (kernel/cooccurrence.py).
    # "auto" (default) detects the vault language from its own text at build time
    # and freezes it into the index; set an explicit Snowball language to override.
    cooccurrence_lang: str = field(
        default_factory=lambda: os.getenv("SILICA_COOCCURRENCE_LANG", "auto")
    )

    # BM25 tf term in the co-occurrence ranking leg (docs/specs/cooccur-scoring.md).
    # On by default since 2026-08-23 (ADR-0029). Gate 2026-07-25: +4.02pp recall@10,
    # p=0.0015. Re-measured 2026-08-23 on the 709-note vault: +5.82pp (0.8233 ->
    # 0.8815), cooccur alone 0.51 -> 0.86, which is what lets the two legs fuse
    # unweighted. No decision surface reads this seam any more: COLLISION routes
    # on the cosine-best note since ADR-0030 (the fused winner was the duplicate
    # 87% of the time against 98% for the cosine-best one) and autolink's gate is
    # embed-only. Answer-side (LoCoMo) not re-run: retrieval lift has failed to
    # reach answers before in this project, so SILICA_COOCCUR_BM25=0 is the kill
    # switch. k1/b stay module constants in relatedness.py: the sweep the spec
    # pre-declared ran 2026-08-23 (ADR-0030) and its held-out best is +0.9pp,
    # under the gate.
    cooccur_bm25: bool = field(
        default_factory=lambda: env_flag("SILICA_COOCCUR_BM25", True)
    )

    # Invocation-time index sweep (kernel/recall/sync.py): detect out-of-band
    # note edits/creates/deletes before the indexes are read. Off = you own
    # index freshness via explicit /embed, /cooccur, /lexical (eval harnesses
    # that need byte-identical retrieval across runs set this off).
    index_sweep: bool = field(
        default_factory=lambda: env_flag("SILICA_INDEX_SWEEP", True)
    )

    # Salience gate (Phase 2.05): concept kept only if cosine(concept, doc_centroid) >= threshold
    sim_threshold_theme: float = field(
        default_factory=lambda: float(os.getenv("SILICA_SIM_THRESHOLD_THEME", "0.35"))
    )

    # AUTOLINK relevance gate: a title is a linking candidate only if its note
    # vector is within this cosine of the note being linked. 0 disables it (the
    # whole title index is a candidate, which is what shipped before).
    #
    # 0.30 is a knee, not an optimum. Swept 0.00-0.85 against 678 hand-authored
    # notes' own wikilinks as ground truth: F1 peaks at 0.40 (0.800 vs 0.758 at
    # 0.00), but that ranks a link the author simply never made as an error.
    # Judged on the noise it exists to stop (a psychology note linked from an ML
    # lecture), 0.30 removes 7 of 11 such links for 2 lost real ones; every step
    # above costs ~9 real links per further noise link removed.
    autolink_min_sim: float = field(
        default_factory=lambda: float(os.getenv("SILICA_AUTOLINK_MIN_SIM", "0.30"))
    )

    # Mindmap (/map): radial map rooted on one note. Node cap is "breathing room"
    # (readable map, not a hairball); latent_k = neighbours asked of the
    # relatedness facade; hops = wikilink BFS depth from the root.
    mindmap_max_nodes: int = field(
        default_factory=lambda: int(os.getenv("SILICA_MINDMAP_MAX_NODES", "35"))
    )
    mindmap_latent_k: int = field(
        default_factory=lambda: int(os.getenv("SILICA_MINDMAP_LATENT_K", "10"))
    )
    mindmap_hops: int = field(
        default_factory=lambda: int(os.getenv("SILICA_MINDMAP_HOPS", "2"))
    )

    # Git commit safety net for docs/ writes. "off" (default) → never commit;
    # "auto" → after each write batch, commit the touched docs/ paths with a
    # structured message. Additive to the undo journal (ADR-0002), never a
    # replacement. Only takes effect when the vault sits inside a git repo.
    git_commit: Literal["off", "auto"] = field(
        default_factory=lambda: os.getenv("SILICA_GIT_COMMIT", "off")  # type: ignore
    )

    @property
    def verbose(self) -> bool:
        return self.debug_logging

    @verbose.setter
    def verbose(self, v: bool) -> None:
        self.debug_logging = v

    def __post_init__(self):
        self.model = ensure_prefix(self.model, self._provider)
        self.worker_model = ensure_prefix(self.worker_model, self.worker_provider)
        self.distill_escalation_model = ensure_prefix(
            self.distill_escalation_model, self._distill_escalation_provider)


CONFIG = SilicaConfig()
