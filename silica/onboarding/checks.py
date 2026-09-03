# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Onboarding checks — pure diagnostics shared by `silica doctor` and `silica init`.

Each check reads config / env / filesystem / HTTP and returns a CheckResult.
No check mutates state and none makes a paid LLM completion call — key
presence and HTTP reachability only.
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import httpx

from silica.agent.providers import (
    LOCAL_RERANK_MODEL,
    PROVIDER_PRESETS,
    has_local_rerank,
    model_limits,
)
from silica.config import USER_ENV, SilicaConfig
from silica.kernel.code import gitstate
from silica.kernel.scrub import scrub_credentials

_HTTP_TIMEOUT = 3.0

# One agentic turn is system prompt + tool schemas + history; below this the
# first turn already overflows. Ollama's own default window is 4096.
_MIN_OLLAMA_WINDOW = 8192


# A local server that is still loading answers 503 on every path, so the socket
# accepting proves nothing. Any other status means something served the request.
_LOADING = 503


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Literal["ok", "warn", "fail", "unknown"]
    detail: str
    hint: str = ""

    def __post_init__(self) -> None:
        # Scrubbed at composition, not per output surface: the doctor table,
        # the --json payload, the GUI's /health endpoint and the settings
        # panel's bug report all consume these fields, and a scrub call at
        # each renderer is the call the next surface forgets. httpx exception
        # text carries the full request URL, query included, so the endpoint
        # checks cannot promise these fields are clean on their own.
        object.__setattr__(self, "detail", scrub_credentials(self.detail))
        object.__setattr__(self, "hint", scrub_credentials(self.hint))


# `unknown` is deliberately distinct from `ok`: when a check cannot read the
# state it must say so softly, not imply the thing is live. Folding the two
# together is how a run reported "rerank ready" and marked rerank unreachable in
# the same session. It is not a failure either — nothing is known to be broken.


def ignored_env_path() -> Path | None:
    """A `.env` in the working directory, or above it — a file silica does not read.

    config.py layers ~/.silica/.env and nothing else, deliberately: a .env found
    by walking up from the cwd belongs to whatever repository the shell happens
    to sit in. But a config file that is inert *in silence* is that same defect
    seen from the other side, and dropping the layer left one on every machine
    that kept a project .env. So the doctor goes looking for exactly the file
    config.py refuses to load.

    None when there is nothing to report, the user's own file included:
    find_dotenv returns it when the doctor runs from inside ~/.silica.
    """
    from dotenv import find_dotenv

    found = find_dotenv(usecwd=True)
    if not found:
        return None
    path = Path(found)
    return None if path == USER_ENV else path


# How many key names the row spells out before it stops counting.
_STRANDED_SHOWN = 6


def check_ignored_env(config: SilicaConfig) -> CheckResult:
    """Name the settings that a .env silica does not read would have applied.

    Key names only. The values in an unread file are exactly as untrusted as the
    file is, and one of them is routinely an API key.

    A file whose keys are all live with the same value already is reported `ok`:
    it is redundant, not lost, and warning there would cry wolf in every checkout
    that keeps a copy of the same config.
    """
    from dotenv import dotenv_values

    path = ignored_env_path()
    if path is None:
        return CheckResult("stray .env", "ok", "none above the working directory")
    stranded = sorted(
        k for k, v in dotenv_values(path).items()
        if k and v is not None and os.getenv(k) != v
    )
    if not stranded:
        return CheckResult(
            "stray .env", "ok", f"{path} is not read, but sets nothing new")
    shown = ", ".join(stranded[:_STRANDED_SHOWN])
    if len(stranded) > _STRANDED_SHOWN:
        shown += f", +{len(stranded) - _STRANDED_SHOWN} more"
    return CheckResult(
        "stray .env", "warn",
        f"{path} is not read — {len(stranded)} setting(s) inactive: {shown}",
        "silica reads only ~/.silica/.env — move them there, or export them for "
        "this directory: set -a; source .env; set +a",
    )


def check_chat_model(config: SilicaConfig) -> CheckResult:
    if not config.model.strip():
        return CheckResult(
            "chat model", "fail",
            "SILICA_MODEL is not set, and no provider key is exported",
            "run `silica init` — or serve the vault read-only with `silica mcp`, "
            "whose recall tools need no model",
        )
    key_env = PROVIDER_PRESETS.get(config.provider, {}).get("api_key_env")
    if key_env and not os.getenv(key_env):
        return CheckResult(
            "chat model", "fail",
            f"{config.model} — provider {config.provider} but {key_env} is unset",
            f"export {key_env} or run `silica init`",
        )
    return CheckResult("chat model", "ok", f"{config.model} via {config.provider}")


def check_chat_endpoint(config: SilicaConfig) -> CheckResult:
    if not config.model.strip():
        # Not probed because there was nothing to probe: a skip, not a
        # fallback. Warning on it every run trains the operator to skim the
        # column where a real degradation appears.
        return CheckResult("chat endpoint", "unknown", "skipped — no model configured")
    if config.provider in ("lmstudio", "ollama"):
        base_url = PROVIDER_PRESETS[config.provider]["base_url"]
    elif config.provider == "custom":
        base_url = config.provider_base_url
        if not base_url:
            return CheckResult(
                "chat endpoint", "fail",
                "custom provider but SILICA_PROVIDER_BASE_URL is unset",
                "run `silica init`",
            )
    else:
        return CheckResult(
            "chat endpoint", "unknown", f"{config.provider} (hosted, not probed)"
        )
    label = {"lmstudio": "LM Studio", "ollama": "Ollama"}.get(config.provider, "the endpoint")
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=_HTTP_TIMEOUT)
    except Exception:
        return CheckResult(
            "chat endpoint", "fail",
            f"{base_url} unreachable",
            f"start {label}, or switch provider with `silica init`",
        )
    if resp.status_code == _LOADING:
        return CheckResult(
            "chat endpoint", "unknown",
            f"{base_url} answering but still loading the model",
            "re-run `silica doctor` once the server reports ready",
        )
    return CheckResult("chat endpoint", "ok", f"{base_url} reachable")


def check_ollama_context(config: SilicaConfig) -> CheckResult:
    """Report the window silica pins per request, and warn when it cannot hold a turn.

    Ollama does not reject an oversized prompt — it drops the overflow and
    answers anyway (measured: a 6645-token prompt came back with
    prompt_eval_count=2051 and the tool definitions gone, HTTP 200, no warning).
    Silica pins num_ctx on every request so the runtime's 4096 default cannot
    cause that, which leaves two ways to end up under water: OLLAMA_NUM_CTX set
    too low by hand, or a model whose trained maximum is smaller than one turn.
    """
    window, _ = model_limits(config.provider, config.model)
    if not window:
        return CheckResult(
            # The check could not read the window. Nothing is known to be
            # wrong, and the detail already says "unknown".
            "ollama context", "unknown",
            f"{config.model} — window unknown (model not pulled, or Ollama unreachable)",
            f"`ollama pull {config.model.removeprefix('ollama/')}`, then re-run `silica doctor`",
        )
    if window < _MIN_OLLAMA_WINDOW:
        return CheckResult(
            "ollama context", "warn",
            f"{window} tokens — too small for one turn, Ollama discards the rest with no error",
            f"raise OLLAMA_NUM_CTX to {_MIN_OLLAMA_WINDOW} or more (costs VRAM); if the model's "
            "own trained maximum is the limit, use a model with a wider window",
        )
    return CheckResult("ollama context", "ok", f"{window} tokens pinned per request")


def check_vault(config: SilicaConfig) -> CheckResult:
    vault = config.vault_path.strip()
    if vault:
        p = Path(vault)
        if not p.is_dir():
            return CheckResult(
                "vault", "fail", f"{vault} does not exist",
                "fix SILICA_VAULT or run `silica init`",
            )
        if not os.access(p, os.W_OK):
            return CheckResult("vault", "fail", f"{vault} is not writable", "fix permissions")
        # Doctor may be handed a config that is not the active vault (the wizard
        # does exactly that), so the boundary comes from this path's own
        # manifest rather than from active_inbox_dir().
        from silica.kernel.vault_manifest import load_manifest, resolve_inbox_dir

        write_dir = load_manifest(str(p)).write_dir or ""
        inbox = resolve_inbox_dir(p, write_dir, config.inbox_dir)
        if inbox and not (p / inbox).is_dir():
            return CheckResult(
                "vault", "warn",
                f"{vault} ok, but inbox folder `{inbox}/` is missing",
                f"create `{inbox}/` inside the vault for nucleation",
            )
        return CheckResult("vault", "ok", vault)
    root = gitstate.find_repo_root(Path.cwd())
    if root is not None:
        # Same rule startup applies: the repo root is the vault, as-is. The old
        # "is this repo a vault *yet*" test predates that and reported a failure
        # for the very repo `silica` would have opened without asking.
        return CheckResult("vault", "ok", f"repo mode → {root}")
    return CheckResult(
        "vault", "fail",
        "SILICA_VAULT not set and this repo is not a Silica vault yet",
        "set SILICA_VAULT=/path/to/vault in .env, or run `silica init`",
    )


def check_embeddings(config: SilicaConfig) -> CheckResult:
    """Report the embeddings leg's actual state.

    Probes with a real /v1/embeddings call rather than comparing against the
    /models id list: llama-server (the default local runtime) reports the
    loaded gguf's file PATH as that id, not a friendly name, and ignores the
    requested `model` field on a single-model server — a literal id match
    false-positives on every llama-server setup even though embeddings work
    fine. Never "fail": relatedness degrades to the co-occurrence leg by design.
    """
    url = f"{config.embedding_base_url.rstrip('/')}/embeddings"
    try:
        resp = httpx.post(
            url,
            json={"model": config.embedding_model, "input": ["ping"]},
            timeout=_HTTP_TIMEOUT,
        )
        if resp.status_code == _LOADING:
            # Still loading its weights (llama-server answers 503 on every
            # path meanwhile). The warn below would send the operator to edit
            # a config that is correct — this is a transient, not a rejection.
            return CheckResult(
                "embeddings", "unknown",
                f"{config.embedding_base_url} answering but still loading the model",
                "re-run `silica doctor` once the server reports ready",
            )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if not data or not data[0].get("embedding"):
            raise ValueError("no embedding vector in response")
    except Exception:
        return CheckResult(
            "embeddings", "warn",
            f"{config.embedding_base_url} unreachable or rejected `{config.embedding_model}`",
            "load the embedding model, or update SILICA_EMBEDDING_MODEL "
            "(dedup routing and /find fall back to co-occurrence)",
        )
    return CheckResult(
        "embeddings", "ok",
        f"{config.embedding_model} @ {config.embedding_base_url}",
    )


def check_rerank(config: SilicaConfig) -> CheckResult:
    """Report the rerank pass's actual state.

    Never "fail": recall degrades to the fused pool's order by design. It warns
    rather than staying silent because the failure mode is invisible — rerank
    just never runs, and the results look plausible.
    """
    if config.rerank_base_url and config.rerank_model:
        url = f"{config.rerank_base_url.rstrip('/')}/rerank"
        try:
            resp = httpx.post(
                url,
                json={"model": config.rerank_model, "query": "ping", "documents": ["ping"]},
                timeout=_HTTP_TIMEOUT,
            )
            if resp.status_code == _LOADING:
                # Still loading its weights. "unreachable" would send the
                # operator to start a server that is already starting.
                return CheckResult(
                    "rerank", "unknown",
                    f"{config.rerank_base_url} answering but still loading the model",
                    "re-run `silica doctor` once the reranker reports ready",
                )
            resp.raise_for_status()
        except Exception:
            # get_reranker (agent/providers.py) falls back to the in-process
            # extra per call when this endpoint is down — report that fallback
            # rather than a bare "unreachable" that reads as rerank being off.
            if has_local_rerank():
                return CheckResult(
                    "rerank", "warn",
                    f"{config.rerank_base_url} unreachable — using in-process fallback ({LOCAL_RERANK_MODEL})",
                    "start the reranker to use it instead of the in-process cross-encoder, "
                    "or set SILICA_RERANK_SERVE_CMD and silica starts it itself",
                )
            return CheckResult(
                "rerank", "warn",
                f"{config.rerank_base_url} unreachable",
                "set SILICA_RERANK_SERVE_CMD so silica starts the reranker itself, start it by hand, "
                "or unset SILICA_RERANK_* and `pip install silica-harness[rerank]`",
            )
        return CheckResult(
            "rerank", "ok", f"{config.rerank_model} @ {config.rerank_base_url}",
        )
    if has_local_rerank():
        return CheckResult("rerank", "ok", f"in-process ({LOCAL_RERANK_MODEL})")
    return CheckResult(
        "rerank", "warn",
        "disabled (no cross-encoder available): recall keeps the fused order and silica_vaults cannot rank vaults",
        "`pip install silica-harness[rerank]` sharpens recall; LM Studio and Ollama cannot serve one",
    )


_LANG_SAMPLE_MAX_FILES = 30
_LANG_SAMPLE_PER_FILE_CHARS = 150
_LANG_SAMPLE_TOTAL_CHARS = 4000

# `AI: true` anywhere in a frontmatter head — the system floor stamps it as a
# top-level key, and the sample only needs a cheap screen, not a YAML parse.
_AI_FLAG_RE = re.compile(r"^AI:\s*true\s*$", re.MULTILINE)


def sample_vault_text(vault: str) -> str:
    """Deterministic, cheap sample of a vault's prose for language detection.

    Up to `_LANG_SAMPLE_MAX_FILES` `.md` files (sorted rglob — deterministic
    across runs/platforms), the first `_LANG_SAMPLE_PER_FILE_CHARS` characters
    of each, concatenated and capped at `_LANG_SAMPLE_TOTAL_CHARS`. The
    per-file cap is kept small (well under total/max_files) so the budget is
    actually SPREAD across the file cap rather than exhausted by the first
    handful of alphabetically-sorted files — a minority-language head (e.g.
    a lone "AAA notes.md") must not drown out the vault's real majority
    language, which only shows up once later files get sampled too. Returns
    "" when the vault has no readable `.md` files. Degrades on any
    filesystem error instead of raising — matches this module's
    pure-diagnostic contract.

    Single seam for this sampling logic: both `check_language` (doctor) and
    the `/vault` info block in cli.py go through `detect_vault_language`
    below, which calls this — no duplicated sampling.
    """
    from silica.kernel.recall.graph_export import is_vault_artifact
    from silica.kernel.vault_manifest import load_manifest, resolve_done_dir

    root = Path(vault)
    try:
        files = sorted(root.rglob("*.md"))
    except Exception:
        return ""
    # The agent's own output never votes on the vault's language: 13 generated
    # English notes flipped a human-Italian vault to `english`, and doctor then
    # proposed rebuilding the co-occurrence store in English — freezing the
    # error. Excluded: root artifacts (GRAPH_REPORT.md sorts before every note
    # and is English scaffolding), the conversion archive under `done/` (the
    # input's language, not the vault's), and any note stamped `AI: true`.
    try:
        write_dir = load_manifest(vault).write_dir or ""
    except Exception:
        write_dir = ""
    done_prefix = (resolve_done_dir(vault, write_dir) or "done").casefold() + "/"
    parts: list[str] = []
    total = 0
    sampled = 0
    for f in files:
        if sampled >= _LANG_SAMPLE_MAX_FILES or total >= _LANG_SAMPLE_TOTAL_CHARS:
            break
        try:
            rel = f.relative_to(root).as_posix()
        except ValueError:
            rel = f.name
        if is_vault_artifact(rel) or rel.casefold().startswith(done_prefix):
            continue
        try:
            head = f.read_text(encoding="utf-8", errors="ignore")[:2000]
        except Exception:
            continue
        if head.startswith("---\n") and _AI_FLAG_RE.search(
            head[: head.find("\n---\n", 4) if head.find("\n---\n", 4) != -1 else len(head)]
        ):
            continue
        chunk = head[:_LANG_SAMPLE_PER_FILE_CHARS]
        parts.append(chunk)
        total += len(chunk)
        sampled += 1
    return "".join(parts)[:_LANG_SAMPLE_TOTAL_CHARS]


def detect_vault_language(vault: str) -> str | None:
    """Cheap, deterministic dominant-language detection for `vault`.

    None when there is nothing to sample (no `.md` files, or all unreadable)
    — callers treat that as "no notes yet". Never raises.
    """
    if not vault:
        return None
    sample = sample_vault_text(vault)
    if not sample.strip():
        return None
    from silica.kernel.text import language

    return language.detect(sample)


def frozen_store_language(vault: str) -> str | None:
    """Read `vault`'s persisted cooccurrence store's frozen `lang` field, if
    a store exists on disk for THIS vault.

    Thin pass-through to `kernel.cooccurrence.frozen_lang` — this module
    owns no on-disk store schema knowledge; the store's own module does.
    Resolved from the `vault` argument, never from the global CONFIG
    singleton, so a caller comparing a specific (possibly non-active) vault
    never cross-checks a different vault's store. None when no store file
    exists yet, or on any read/parse error (degrade, never raise — inherited
    from the accessor this delegates to).

    Direct leg import — allowlisted in tests/test_relatedness_boundary.py:
    metadata-only read via the public accessor, no store construction.
    """
    from silica.kernel.recall.cooccurrence import frozen_lang

    return frozen_lang(vault)


def declared_language(vault: str) -> str | None:
    """The language `vault` DECLARES in its `vault.yaml` (`cooccurrence_lang`),
    or None when it declares none — or declares the `auto` sentinel, meaning
    "detect me". A declaration is authority: it is the language the
    co-occurrence store is (or should be) frozen to, so it SUPERSEDES the
    stopword heuristic. A frontmatter-heavy sample fools `detect` into
    "english" (the bundled english stoplist matches `last:`/`related:`/`null`
    scaffolding), but a vault that declares italian is italian.

    Resolved from the `vault` argument, never the global CONFIG singleton —
    same contract as frozen_store_language above.
    """
    from silica.kernel.vault_manifest import load_manifest

    lang = load_manifest(vault).cooccurrence_lang
    return lang if lang and lang != "auto" else None


def language_status(vault: str) -> tuple[str | None, str | None, bool]:
    """`(authority, store, drift)` for `vault` — the single seam behind both
    the doctor's `check_language` and the `/vault` info block.

    authority = the declared language if any, else the heuristically detected
    dominant language (None when there is nothing to sample). store = the
    frozen co-occurrence store language (None if no store on disk yet). drift =
    both known and differing — the signal that `/cooccur --force` is needed to
    rebuild the store in the authoritative language.
    """
    authority = declared_language(vault) or detect_vault_language(vault)
    store = frozen_store_language(vault) if authority else None
    return authority, store, bool(authority and store and authority != store)


def reply_language_for(vault: str) -> str | None:
    """The language chat should default to for `vault`.

    Explicit conventions win (`reply_language`, else `language`); the fallback
    is the vault's own authority (declared `cooccurrence_lang`, else detected
    from the human notes). Without the fallback a /quiz on an Italian vault
    came back in English: slash-command turns carry no language of their own,
    and only the explicit conventions ever reached the prompt.
    """
    if not vault:
        return None
    from silica.kernel.vault_manifest import load_manifest

    conv = load_manifest(vault).conventions
    return conv.reply_language or conv.language or language_status(vault)[0]


def check_language(config: SilicaConfig) -> CheckResult:
    """The vault's authoritative language (declared in vault.yaml, else
    detected) vs. the cooccurrence store's frozen language. A divergence is
    the signature of the historic bug that froze stores to "english" on
    non-English vaults — this is how existing users discover a store needs a
    `/cooccur` rebuild.

    Resolved from `config.vault_path` — never from the global CONFIG
    singleton — so a caller that just reconfigured (e.g. the init wizard
    building a fresh `SilicaConfig()` right after a vault switch) never
    compares the newly-chosen vault against a *different*, still-active
    vault's frozen store.
    """
    vault = config.vault_path.strip()
    if not vault:
        return CheckResult("language", "ok", "no vault — skipped")

    authority, store_lang, drift = language_status(vault)
    if authority is None:
        # unknown, not ok: nothing was sampled, so nothing is known about the
        # language, and a walk that reached no notes is not evidence of one.
        return CheckResult("language", "unknown", "no notes sampled yet")
    if store_lang is None:
        return CheckResult("language", "ok", f"language={authority}, no store frozen yet")
    if not drift:
        return CheckResult("language", "ok", f"language={authority}, store={store_lang}")
    return CheckResult(
        "language", "warn",
        f"language={authority}, store frozen={store_lang} — mismatch",
        "run `/cooccur --force` to rebuild the co-occurrence store in the vault's language",
    )


def check_manifest(config: SilicaConfig) -> CheckResult:
    from silica.kernel.vault_manifest import MANIFEST_REL, load_manifest
    from silica.sources.registry import ALL_ADAPTERS

    vault = config.vault_path.strip()
    if not vault:
        return CheckResult("vault manifest", "ok", "no vault — defaults apply")
    path = Path(vault) / MANIFEST_REL
    if not path.is_file():
        return CheckResult("vault manifest", "ok", "absent — retro-compatible defaults")
    m = load_manifest(vault)
    known = {a.name for a in ALL_ADAPTERS}
    unknown = [s for s in m.sources if s not in known]
    if unknown:
        return CheckResult(
            "vault manifest", "warn",
            f"unknown source(s) {unknown} in {MANIFEST_REL}",
            f"known sources: {sorted(known)}",
        )
    detail = f"sources={list(m.sources)}"
    return CheckResult("vault manifest", "ok", detail)


def check_memory_lane(config: SilicaConfig) -> CheckResult:
    """The second recall lane (ADR-0019), and the seam it opens when it diverges.

    Recall fuses the memory vault's legs; every path-taking tool (search,
    exists, read_note) resolves inside the ACTIVE vault only. When the two are
    different trees a note can be served by recall and denied by exists in the
    same session — read as "the note is missing" by anyone who does not know a
    second vault is in play. Legitimate configuration, so it warns rather than
    fails, but it must have a surface: doctor named one vault and stopped.
    """
    from silica.kernel.recall.memory_lane import memory_vault

    mem = memory_vault()  # None ⇒ absent, or same tree as the active vault
    if mem is None:
        return CheckResult("memory lane", "ok", "off — single vault")
    return CheckResult(
        "memory lane", "warn",
        f"recall also answers from {mem}",
        "search/exists/read_note resolve in the active vault only, so recall "
        "may name notes they deny; unset SILICA_MEMORY_VAULT to use one vault, "
        "or scope one call with memory=false on recall/semantic_search/related",
    )


def check_recall_indexes(config: SilicaConfig) -> CheckResult:
    """The active vault's retrieval stores, counted.

    Every leg abstains gracefully by design, so a vault that was co-occurrence
    indexed but never embedded serves plausible cooccur-only results forever —
    measured 2026-08-25: the repo vault ran every silica_related call without
    its strongest leg (embed alone recalls 0.815@10, ADR-0029) and nothing said
    so. The doctor is the surface whose job is saying it.
    """
    try:
        from silica.kernel.recall.cooccurrence import get_cooccur_store
        from silica.kernel.recall.embed import get_store

        n_embed = len(get_store())
        try:
            n_cooc = len(get_cooccur_store(lang=config.cooccurrence_lang))
        except Exception:
            n_cooc = 0
    except Exception as exc:  # unreadable stores: report, never block the doctor
        return CheckResult("recall indexes", "unknown", f"unreadable ({exc})")
    if n_embed == 0 and n_cooc > 0:
        return CheckResult(
            "recall indexes", "warn",
            f"cooccurrence holds {n_cooc} notes but the embedding index is empty",
            "the strongest leg abstains on every call and results degrade to "
            "co-occurrence only; run silica_embed_refresh once to seed it",
        )
    if n_embed == 0 and n_cooc == 0:
        return CheckResult("recall indexes", "warn", "no retrieval index yet",
                           "run silica_embed_refresh and silica_cooccurrence_refresh")
    return CheckResult("recall indexes", "ok",
                       f"embed {n_embed} vectors, cooccurrence {n_cooc} notes")


def live_probe(config: SilicaConfig) -> CheckResult:
    """One tiny PAID completion proving the model actually answers.

    run_checks only probes key-presence and /models reachability, never a paid
    call — this is the row that costs money, so it runs only when asked
    (`silica doctor --live`, `silica_doctor(live=True)`); one implementation so
    the CLI and the MCP surface cannot drift.
    """
    if not (getattr(config, "model", "") or "").strip():
        return CheckResult("live probe", "warn", "skipped — no model configured",
                           "run silica init to set one")
    from silica.agent.llm import call_llm

    try:
        # 512 and not the 5 tokens the word needs: a hybrid model bills its
        # thinking against max_tokens, so 5 was spent on the trace and the reply
        # came back empty on a model answering fine in the REPL (measured
        # 2026-08-24 on openrouter/stealth/ox-alpha: finish=length,
        # completion_tokens=5, 20 chars of trace, text='').
        resp = call_llm(config.model,
                        [{"role": "user", "content": "Reply with: ok"}],
                        max_tokens=512)
    except Exception as e:
        # scrub: provider exceptions embed the full request URL, key included.
        from silica.kernel.scrub import scrub_credentials

        return CheckResult("live probe", "fail", f"failed: {scrub_credentials(e)}")
    if (resp.text or "").strip():
        return CheckResult("live probe", "ok", "model replied")
    # A trace that overran the budget still proves the model answered — it
    # answered into the trace; the same arithmetic is what makes a tight-budget
    # extraction lane come back empty.
    if resp.finish_reason == "length" and (resp.reasoning or "").strip():
        return CheckResult("live probe", "ok",
                           "model replied (in reasoning only — thinking filled the budget)")
    return CheckResult("live probe", "fail", "empty reply")


def check_quarantine(config: SilicaConfig) -> CheckResult:
    """Corrupt state files quarantined as *.corrupt.* — preserved, not lost."""
    from silica.kernel.recall.paths import index_dir_for

    roots = [Path(p) for p in (config.vault_path,) if p]
    roots.append(index_dir_for(config.vault_path or ""))
    # ~/.silica holds cross-vault state (undo_journal.db, checkpoints): its
    # quarantined copies were invisible to doctor, the only surface for them.
    roots.append(Path.home() / ".silica")
    found = [p.name for r in roots if r.exists() for p in sorted(r.glob("*.corrupt.*"))]
    if found:
        return CheckResult(
            "quarantine", "warn",
            f"{len(found)} corrupt state file(s) preserved: {', '.join(found)}",
            "inspect or delete; derived indexes rebuild via /cooccur",
        )
    return CheckResult("quarantine", "ok", "no quarantined state")


def _pdf_lane(config: SilicaConfig) -> str:
    """The PDF provider and whether it can read a scan.

    A library of photographed books is entirely unreadable under the default
    provider, and nothing said so until a conversion returned no text. pdfium is
    the only provider that ships with Silica; the other three are binaries the
    user installs, so naming an unknown one is a configuration error worth
    seeing here. A legacy "pymupdf" pin resolves to the default before the check.

    A configured mineru that is not on PATH is its own line rather than "OCR
    available": since the [pdf] extra went away (2026-09-02) installing Silica
    proves nothing about mineru, and a pin to an absent binary reads as working
    right up to the conversion that fails.
    """
    from silica.sources.convert import PDF_PROVIDERS, resolve_pdf_provider

    name = resolve_pdf_provider(getattr(config, "pdf_provider", "") or "pdfium")
    if name not in PDF_PROVIDERS:
        return f"{name} — unknown provider (known: {', '.join(PDF_PROVIDERS)})"
    if name == "pdfium":
        return f"{name} — no OCR, a scan with no text layer yields nothing"
    if name == "mineru" and not shutil.which("mineru"):
        return f"{name} — OCR provider set but not on PATH"
    return f"{name} — OCR available"


def _mineru_lane() -> str:
    """Whether the mineru CLI is reachable, and what cannot convert without it.

    Its own row and not a footnote to the PDF one: images and .pptx/.xlsx reach
    mineru whatever SILICA_PDF_PROVIDER says, because no other backend opens
    them at all (`_MINERU_ONLY_EXTS` in sources/convert.py). So a vault of
    screenshots is unconvertible under the default provider too, and the pdf
    row — which only ever reports the PDF lane — never said so.
    """
    if shutil.which("mineru"):
        return "ok (OCR, figures, images/.pptx/.xlsx)"
    return "missing (no OCR; images/.pptx/.xlsx cannot convert)"


def check_converters(config: SilicaConfig) -> CheckResult:
    """External binaries the ingestion lanes shell out to, and their real state.

    Never `fail`: these are optional lanes, and someone who never converts a
    .doc does not have a broken install. What this row exists for is the state
    between installed and working — `which` finding a binary is not evidence it
    can do the job. Apache OpenOffice is the case that proved it: a healthy
    office suite that starts fine and simply never implemented headless
    `-convert-to`, so the conversion it is asked for opens a GUI wizard instead.
    Reading that here beats meeting it mid-conversion.
    """
    from silica.sources.convert import probe_soffice

    ffmpeg = shutil.which("ffmpeg")
    status, detail = probe_soffice()
    parts = [
        f"pdf {_pdf_lane(config)}",
        f"mineru {_mineru_lane()}",
        f"ffmpeg {'ok' if ffmpeg else 'missing'} (audio/video)",
        f"office {status} (.doc/.ppt only)",
    ]
    # The blast radius shrank once ODF/RTF/.xls became in-process reads, so
    # every office sentence below has to say how small it now is — otherwise a
    # `warn` reads as "conversion is broken" when 5 of 7 formats are unaffected.
    scope = (
        ". Only .doc/.ppt need it; .odt/.odp/.ods/.rtf/.xls are read in process "
        "and .pptx/.docx/.xlsx were never affected"
    )
    # The hint names only what is actually absent. Listing all three every time
    # is the same defect the `scope` sentence exists to fix: an install
    # instruction for a tool the user already has reads as a fault report.
    absent = []
    if not shutil.which("mineru"):
        absent.append("mineru for OCR and images/.pptx/.xlsx (`pip install 'mineru[pipeline]'`)")
    if not ffmpeg:
        absent.append("ffmpeg for audio/video")
    if status == "missing":
        absent.append("libreoffice for .doc/.ppt (or just re-save those as .docx/.pptx)")

    def with_absent(hint: str) -> str:
        """The office verdict and the install list in one hint, never either or.

        An unsupported suite returns `warn` and used to return it alone, so a
        machine with Apache OpenOffice AND no mineru heard about .doc and never
        about OCR — the louder row silenced the one covering more formats.
        """
        tail = ("optional: install " + ", ".join(absent)) if absent else ""
        return f"{hint}. {tail}" if hint and tail else (hint or tail)

    if status == "unsupported":
        return CheckResult(
            "converters", "warn", "; ".join(parts),
            with_absent(
                f"{detail}{scope}. Those two are refused with that message rather "
                "than attempted; re-save them as .docx/.pptx to skip the suite"
            ),
        )
    if status in ("hung", "broken"):
        return CheckResult("converters", "warn", "; ".join(parts), with_absent(detail + scope))
    if absent:
        return CheckResult("converters", "unknown", "; ".join(parts), with_absent(""))
    return CheckResult("converters", "ok", "; ".join(parts))


def check_okf(config: SilicaConfig) -> CheckResult:
    """Open Knowledge Format §11: the vault IS a bundle, or it says why not.

    Only what the user can act on raises the status. A file with no frontmatter
    at all (§11.1) is counted and reported but stays `ok`: Silica's write path
    never produces one, and in repo mode the vault is a source tree whose
    README and prompt templates are markdown by right — warning about those
    every run would be noise nobody can clear.

    A census that saw no notes is `unknown`, never `ok`: "conformant bundle"
    over zero files says nothing about the vault and everything about the walk
    (a path that is not there, an ignore rule that swallowed every folder).
    """
    from silica.kernel.write.notetype import okf_conformance

    vault = config.vault_path.strip()
    if not vault:
        return CheckResult("OKF §11", "ok", "no vault — nothing to census")
    if not Path(vault).is_dir():
        return CheckResult("OKF §11", "unknown", f"{vault} is not a directory; nothing censused")
    scanned, violations = okf_conformance(vault)
    if scanned == 0:
        return CheckResult("OKF §11", "unknown", "no notes censused; an empty walk proves nothing")
    if not violations:
        return CheckResult("OKF §11", "ok", "conformant bundle")
    def tally(vs: list) -> str:
        by: dict[str, int] = {}
        for v in vs:
            by[v.clause] = by.get(v.clause, 0) + 1
        return ", ".join(f"§{c}: {n}" for c, n in sorted(by.items()))

    actionable = [v for v in violations if v.clause != "11.1"]
    if not actionable:
        return CheckResult("OKF §11", "ok", f"typed bundle, {tally(violations)} without frontmatter")
    hint = ""
    if any(v.clause == "11.2" for v in actionable):
        hint = "run `uv run python scripts/backfill_notetype.py` to stamp the missing types"
    if any(v.clause == "11.3" for v in actionable):
        hint = (hint + "; " if hint else "") + "rename any `index`/`log` note by hand"
    sample = ", ".join(v.path for v in actionable[:3])
    # The count and the breakdown have to be the same set of notes. The headline
    # counted `actionable` while the breakdown was built from every violation,
    # so a vault with 13 untyped notes and 2 bad names announced itself, in a
    # panel that is permanently on screen, as "2 non-conformant note(s), §11.1:
    # 13, §11.3: 2". The 11.1s are still worth naming; they are named as what
    # they are, which is a separate and non-actionable fact.
    skipped = len(violations) - len(actionable)
    aside = f"; {skipped} more without frontmatter" if skipped else ""
    return CheckResult(
        "OKF §11", "warn",
        f"{len(actionable)} non-conformant note(s): {tally(actionable)} "
        f"(e.g. {sample}){aside}",
        hint,
    )


HOOK_SNIPPET = """\
"hooks": {
  "SessionEnd": [{"hooks": [{"type": "command", "command": "silica capture"}]}],
  "PreCompact": [{"hooks": [{"type": "command", "command": "silica capture"}]}]
}"""


def check_capture_hook(config: SilicaConfig) -> CheckResult:
    """Session capture is opt-in and hand-registered: say so when it is absent.

    Silica never edits `.claude/settings.json` itself — a tool that rewrites
    another tool's config is a support burden, and the hook is three lines.
    """
    # Claude Code resolves project settings from the session's cwd, not from
    # the vault: for an adopted source tree the two differ, and looking only
    # under the vault warned about a hook that was registered and firing.
    roots = [Path.home(), Path.cwd()]
    vault = config.vault_path.strip()
    if vault:
        roots.append(Path(vault))
    candidates = [root / ".claude" / name for root in roots
                  for name in ("settings.json", "settings.local.json")]
    for path in candidates:
        try:
            if "silica capture" in path.read_text(encoding="utf-8"):
                return CheckResult("session capture", "ok", f"hook registered in {path}")
        except OSError:
            continue
    return CheckResult(
        "session capture", "warn",
        "no `silica capture` hook — sessions are not captured",
        f"add to .claude/settings.json:\n{HOOK_SNIPPET}",
    )


def check_session_capture(config: SilicaConfig) -> CheckResult:
    """Silica's own conversations: opt-in, and never notes.

    Off is a legitimate choice, so this never warns — it only says the knob
    exists, and where the memory goes when it is on.
    """
    if getattr(config, "capture_sessions", False):
        return CheckResult(
            "own sessions", "ok",
            "captured to the WAL; /nucleate distills them into episodic memory",
        )
    return CheckResult(
        "own sessions", "ok", "not captured",
        "set SILICA_CAPTURE_SESSIONS=true to remember your own sessions "
        "(facts only, never notes — promotion is what writes to the vault)",
    )


def _guarded(name: str, check: Callable[[SilicaConfig], CheckResult],
             config: SilicaConfig) -> CheckResult:
    """Run one check, degrading it in place instead of taking down the report.

    The HTTP checks guard themselves, but the filesystem and parsing ones do
    not: a single OSError on a vault the user just unmounted used to abort the
    whole run, including the twelve checks that would have answered. A check
    that raises is a failure, not a new state.

    `name` is the row name the check itself uses when it answers, so a consumer
    keying on `results[].name` finds the row in exactly the degraded run the
    guard exists to report — deriving it from `check.__name__` gave "manifest"
    for a row every healthy run calls "vault manifest".
    """
    try:
        return check(config)
    except Exception as exc:  # noqa: BLE001 — the doctor must survive any check
        return CheckResult(name, "fail", f"check raised: {type(exc).__name__}: {exc}")


def check_narration_store(config: SilicaConfig) -> CheckResult:
    """Warn at 1GB store / 100MB single session (spec ticket 06). Numbers,
    not conditions: eviction is never automatic, so the doctor is the one
    place growth becomes visible before it becomes a surprise."""
    from silica.agent.narration import store_stats
    st = store_stats()
    gb, mb = st["total_bytes"] / 1e9, st["biggest_bytes"] / 1e6
    if st["biggest_bytes"] > 100e6:
        return CheckResult(
            "narration store", "warn",
            f"session {st['biggest_sid']} is {mb:.0f}MB (threshold 100MB)",
            hint="a runaway session, not history — consider /sessions prune")
    if st["total_bytes"] > 1e9:
        return CheckResult(
            "narration store", "warn", f"{gb:.1f}GB on disk (threshold 1GB)",
            hint="prune old sessions with /sessions prune <days>d")
    return CheckResult("narration store", "ok",
                       f"{st['total_bytes'] / 1e6:.1f}MB across sessions")


def run_checks(config: SilicaConfig) -> list[CheckResult]:
    checks: list[tuple[str, Callable[[SilicaConfig], CheckResult]]] = [
        # Listed only when there is one to list: a row reading "no stray .env"
        # on every run is noise, and this is the row that explains why any of
        # the rows below it might be reporting the wrong thing.
        *([("stray .env", check_ignored_env)] if ignored_env_path() else []),
        ("chat model", check_chat_model),
        ("chat endpoint", check_chat_endpoint),
        # Ollama-only: the silent-truncation trap is specific to it.
        *([("ollama context", check_ollama_context)] if config.provider == "ollama" else []),
        ("vault", check_vault),
        ("memory lane", check_memory_lane),
        ("vault manifest", check_manifest),
        ("language", check_language),
        ("embeddings", check_embeddings),
        ("recall indexes", check_recall_indexes),
        ("rerank", check_rerank),
        ("quarantine", check_quarantine),
        ("converters", check_converters),
        ("OKF §11", check_okf),
        ("session capture", check_capture_hook),
        ("own sessions", check_session_capture),
        ("narration store", check_narration_store),
    ]
    return [_guarded(name, c, config) for name, c in checks]


def has_failures(results: list[CheckResult]) -> bool:
    return any(r.status == "fail" for r in results)


def verdict(results: list[CheckResult]) -> Literal["ok", "hold", "fail"]:
    """ok = every row ok; fail = a row failed; hold = nothing failed, but a
    row needs reading (a fallback was taken, or a check could not answer).

    One resolver for the three surfaces that report it (the CLI exit code,
    the --json and MCP payload, the GUI): a consumer that re-derived the rule
    from the rows is the consumer that folds warn back into ok, which is how
    `silica doctor && run` started on the half-answering endpoint the table
    had flagged in yellow. The wizard keeps has_failures: its exit says
    whether init finished, not whether the environment is clean.
    """
    statuses = {r.status for r in results}
    if "fail" in statuses:
        return "fail"
    if statuses & {"warn", "unknown"}:
        return "hold"
    return "ok"


def exit_code(results: list[CheckResult]) -> int:
    """0 = ok, 1 = fail, 2 = hold: 2 is neither success nor failure and a
    script must not fold it into either."""
    return {"ok": 0, "fail": 1, "hold": 2}[verdict(results)]


# `?` and not `⚠`: a warning says a fallback was taken, unknown says nothing
# could be read. Give them the same glyph and the real warnings drown, which is
# the whole reason routine lines must not shout.
_STATUS_GLYPH = {"ok": ("✓", "green"), "warn": ("⚠", "yellow"),
                 "fail": ("✗", "red"), "unknown": ("?", "dim")}


def report_payload(results: list[CheckResult]) -> dict:
    """Machine-readable mirror of `render_report` — how the agent reads its own health.

    Deliberately a flat mirror of the dataclass rather than a shaped schema:
    once something routes on the field names, growing a shape is a breaking
    change, and `CheckResult` is already the contract. Credentials are already
    scrubbed — CheckResult redacts its own fields at composition.
    """
    return {
        "results": [
            {"name": r.name, "status": r.status, "detail": r.detail, "hint": r.hint}
            for r in results
        ],
        "ok": not has_failures(results),
        # `ok` stays for the consumers that key on it; `verdict` is the same
        # three-way answer the exit code gives, so an agent reading the MCP
        # payload sees a hold instead of an `ok: true` over yellow rows.
        "verdict": verdict(results),
    }


def render_report(results: list[CheckResult]) -> None:
    from rich.markup import escape
    from rich.table import Table

    from silica.ui.console import CONSOLE

    table = Table(show_header=False, box=None, padding=(0, 1))
    for r in results:
        glyph, style = _STATUS_GLYPH[r.status]
        # escape: detail/hint carry data (paths, model ids, `silica-harness[rerank]`), and
        # rich reads a bare [word] as a style tag and swallows it.
        hint = f"[dim]→ {escape(r.hint)}[/]" if r.hint else ""
        table.add_row(f"[{style}]{glyph}[/]", f"[bold]{r.name}[/]", escape(r.detail), hint)
    CONSOLE.print()
    CONSOLE.print(table)
    CONSOLE.print()
