# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`silica init` — interactive setup wizard. Writes .env, then runs the doctor checks."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Callable

from silica.config import (
    HOSTED_PROVIDERS,
    USER_ENV,
    SilicaConfig,
    claim_env,
    model_from_env,
)
from silica.kernel.code import gitstate
from silica.kernel.vault_manifest import MANIFEST_REL
from silica.onboarding.checks import has_failures, render_report, run_checks
from silica.ui.banner import print_banner
from silica.ui.console import CONSOLE
from silica.ui.style import GLYPHS

# Optional leading `#` so merge_env can uncomment-and-fill a `# KEY=default`
# line seeded from .env.example, not just rewrite an already-active key.
_KEY_RE = re.compile(r"^\s*#?\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")

# Human label for each hosted key prompt. The env var and the model pick-list
# come from config.HOSTED_PROVIDERS, which the no-model fallback chain reads
# too — one table, so a model id added for the wizard is also a model the
# chain can pick up. First entry is what Enter accepts; `other` is appended by
# _pick so any id stays reachable.
_KEY_PROMPTS = {
    "openrouter": "OpenRouter API key",
    "gemini": "Google Gemini API key",
    "openai": "OpenAI API key",
    "groq": "Groq API key",
    "deepseek": "DeepSeek API key",
    "mistral": "Mistral API key",
    "xai": "xAI (Grok) API key",
}
_HOSTED = {
    name: (key_env, list(models), _KEY_PROMPTS[name])
    for name, (key_env, models) in HOSTED_PROVIDERS.items()
}

# Worker (sub-agent) suggestions: small and cheap, the job is dedup/refine, not
# reasoning. Hosted providers not listed fall back to their main list.
_WORKER_MODELS = {
    "openrouter": [
        "openrouter/mistralai/mistral-small-2603",
        "openrouter/deepseek/deepseek-v4-flash",
    ],
    "mistral": ["mistral/mistral-small-latest"],
}

EMBED_MODELS = [
    "text-embedding-qwen3-embedding-4b",  # LM Studio's id for the same weights
    "qwen3-embedding-4b",
    "nomic-embed-text",
    "text-embedding-3-small",
]

# Served /rerank endpoints only (llama.cpp --reranking, Infinity, Jina) — the
# in-process [rerank] extra picks its own weights.
RERANK_MODELS = ["bge-reranker-v2-m3-Q8_0", "bge-reranker-v2-m3"]

_EMBED_KEYS = ("SILICA_EMBEDDING_MODEL", "SILICA_EMBEDDING_BASE_URL", "SILICA_EMBEDDING_API_KEY",
               "SILICA_EMBEDDING_SERVE_CMD")

# Suggested autostart command per local runtime (see onboarding/serve.py). Asked
# once here so no later run has to remember to start the server by hand.
_SERVE_CMDS = {"lmstudio": "lms server start", "ollama": "ollama serve"}


class BackRequested(Exception):
    """Raised by _ask when the user types `back` — the wizard driver rewinds
    to the previous step that actually asked something."""

_LANG_PROMPT = (
    "Force a language for distilled notes? "
    "[Enter = no, follow the source language]"
)
# Bare language names only: letters and spaces. Rejects punctuation (a colon
# above all — see _ask_language) that would corrupt the raw YAML the answer
# is embedded into.
_LANG_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z ]*$")
# YAML 1.1 boolean literals: they'd pass the letters-only regex above but
# parse as `True`/`False`, which `_parse_conventions` folds to None — the
# user would believe they forced a language but silently didn't.
_LANG_ANSWER_REJECT = {"y", "n", "yes", "no", "true", "false", "on", "off"}


def _find_env_example(repo_root: Path | str | None) -> Path | None:
    """Locate `.env.example` to seed a fresh `.env`: the vault repo root first,
    then the copy shipped inside the installed package, then this package's own
    checkout root. `None` when none exists — the caller falls back to a minimal
    write. The packaged copy (silica/.env.example) is what makes seeding work for
    a real `pip`/`uv tool` install, not just an editable checkout."""
    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(Path(repo_root) / ".env.example")
    candidates.append(Path(__file__).resolve().parents[1] / ".env.example")  # silica/ package dir
    candidates.append(Path(__file__).resolve().parents[2] / ".env.example")  # dev checkout root
    return next((c for c in candidates if c.is_file()), None)


def endpoint_model_ids(base_url: str, api_key: str = "") -> list[str]:
    """Model ids advertised by an OpenAI-compatible `/models` endpoint, best-effort
    ([] on any error). Powers LM Studio autodetect, the local-embeddings
    suggestion, and the hosted pick-list validation (_live_hosted_models);
    `api_key` becomes a Bearer header for the hosted endpoints that demand one."""
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        data = httpx.get(
            f"{base_url.rstrip('/')}/models", headers=headers, timeout=3.0
        ).json()
        return [m["id"] for m in data.get("data", []) if m.get("id")]
    except Exception:
        return []


# OpenAI-compatible /models base per hosted provider. Hand-kept like the model
# lists were, but base URLs outlive model generations by years; a wrong or dead
# entry only costs the live validation, never the wizard (fetch failure falls
# back to the static list).
_MODELS_BASE = {
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com",
    "mistral": "https://api.mistral.ai/v1",
    "xai": "https://api.x.ai/v1",
}


def _live_hosted_models(provider: str, api_key: str, curated: list[str]) -> list[str]:
    """The hosted pick-list, validated against the provider's live `/models`.

    Curated entries keep their order and litellm prefixes; ids the provider no
    longer serves are dropped (the staleness a hand-kept list accrues). If the
    whole curated list has rotted, the first live ids stand in — prefixed, so
    whatever gets picked is a usable litellm model string. An unreachable
    /models must never block onboarding: the curated list returns verbatim.
    """
    base = _MODELS_BASE.get(provider)
    if not base:
        return list(curated)
    live = endpoint_model_ids(base, api_key=api_key)
    if not live:
        return list(curated)
    # Gemini's OpenAI-compat endpoint prefixes ids with "models/".
    norm = {m.removeprefix("models/") for m in live}
    alive = [m for m in curated if m.split("/", 1)[1] in norm]
    dropped = len(curated) - len(alive)
    if alive and dropped:
        CONSOLE.print(
            f"      [dim]{dropped} suggested id(s) no longer served — hidden[/]")
    if alive:
        return alive
    return [f"{provider}/{m.removeprefix('models/')}" for m in live[:8]]


def resolve_env_path() -> Path:
    """The `.env` a settings write must land in — the one that would win at boot.

    Always the user-level file, because config.py layers that one and nothing
    else: it is the only path where a write is still there at the next launch.
    This used to prefer a project .env when one existed, so a settings edit made
    inside any checkout landed in that checkout's file — and, once the project
    layer was removed, would have landed in a file nobody reads.

    Creates the parent directory, never the file.
    """
    USER_ENV.parent.mkdir(parents=True, exist_ok=True)
    return USER_ENV


def merge_env(existing: str, updates: dict[str, str]) -> str:
    """Update KEY=VALUE lines in place — uncommenting a `# KEY=default` line when
    KEY is collected — preserve every other line untouched, and append keys that
    were not present. Never deletes a line it did not write."""
    pending = dict(updates)
    out: list[str] = []
    for line in existing.splitlines():
        m = _KEY_RE.match(line)
        if m and m.group(1) in pending:
            key = m.group(1)
            out.append(f"{key}={pending.pop(key)}")
        else:
            out.append(line)
    for key, value in pending.items():
        out.append(f"{key}={value}")
    text = "\n".join(out)
    return text + "\n" if text else ""


def _ask(
    input_fn: Callable[[str], str],
    prompt: str,
    default: str = "",
    *,
    secret: bool = False,
) -> str:
    shown = f"…{default[-4:]}" if (secret and default) else default
    suffix = f" [{shown}]" if default else ""
    try:
        # `→` gutter marks every question with the TUI's arrow glyph (same one
        # render_report uses for hints). Plain text: input() ignores markup.
        raw = input_fn(f"  {GLYPHS['arrow']} {prompt}{suffix}: ").strip()
    except (EOFError, StopIteration):
        # EOF (Ctrl+D) or an exhausted scripted input — treat like Ctrl+C.
        raise KeyboardInterrupt
    if raw.lower() == "back":
        raise BackRequested
    return raw or default


def _pick(
    input_fn: Callable[[str], str],
    prompt: str,
    options: list[str],
    *,
    other_prompt: str = "Model id",
    default: str | None = "",
    required: bool = True,
) -> str:
    """Numbered pick-list so a model id gets chosen, not recalled from memory.

    A number takes that entry; the last slot is always `other`, which then asks
    for an id. Anything non-numeric IS the id — typing one straight past the
    list stays the fastest path for whoever already knows what they want.

    `default=""` (unset) makes Enter accept the first option; an explicit string
    overrides it; `None` means Enter answers nothing (an optional question).

    ponytail: a model id made only of digits would read as an index. No vendor
    ships one; if that ever changes, the `other` slot still reaches it.
    """
    fallback = options[0] if (default == "" and options) else (default or "")
    if not options:
        return _ask(input_fn, other_prompt, fallback)
    for n, opt in enumerate(options, 1):
        CONSOLE.print(f"      [dim]{n}.[/] {opt}")
    CONSOLE.print(f"      [dim]{len(options) + 1}.[/] other [dim]— type your own id[/]")
    answer = _ask(input_fn, prompt, fallback)
    if not answer.isdigit():
        return answer
    index = int(answer)
    if 1 <= index <= len(options):
        return options[index - 1]
    model = _ask(input_fn, other_prompt)
    while required and not model:
        model = _ask(input_fn, other_prompt)
    return model


def _ask_serve_cmd(
    input_fn: Callable[[str], str],
    updates: dict[str, str],
    key: str,
    base_url: str,
    suggested: str,
) -> None:
    """Ask how to start a local endpoint, so no later run has to be asked at all.

    Only for a loopback URL — a hosted endpoint is nobody's to start. Blank
    answer means "I start it myself" and writes nothing.
    """
    from silica.onboarding.serve import is_local

    if not is_local(base_url):
        return
    cmd = _ask(
        input_fn,
        "Command that starts it, run when the port is closed (blank = you start it)",
        suggested,
    )
    if cmd:
        updates[key] = cmd


def _ollama_installed_models() -> list[str]:
    """Tags installed in the local Ollama, best-effort ([] if it's down/absent).

    Lets the wizard offer a pick-list instead of asking the user to recall an
    exact tag. Never raises — a down Ollama just means no suggestions.
    """
    import httpx

    from silica.agent.providers import PROVIDER_PRESETS

    base = PROVIDER_PRESETS["ollama"]["base_url"].removesuffix("/v1")
    try:
        data = httpx.get(f"{base}/api/tags", timeout=3.0).json()
        return [m["name"] for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def _section(glyph_key: str, title: str, n: int, total: int) -> None:
    """Flat-gutter step header in the TUI's brand vocabulary: glyph + title in
    bold brand cyan, a dim `· n/N` counter riding after it."""
    CONSOLE.print()
    CONSOLE.print(f"  [bold brand.cyan]{GLYPHS[glyph_key]} {title}[/]  [dim]· {n}/{total}[/]")


def _rerank_extra_present() -> bool:
    """Whether the optional [rerank] extra is importable. Guarded find_spec
    (via has_local_rerank) — never raises."""
    try:
        from silica.agent.providers import has_local_rerank
        return has_local_rerank()
    except Exception:
        return False


def _rerank_install_cmd() -> str:
    """Exact install command for the [rerank] extra, matching how this process
    was installed (uv-managed interpreter → uv pip, else pip)."""
    pip = "uv pip" if "uv" in Path(sys.executable).resolve().parts else "pip"
    return f"{pip} install 'silica-harness[rerank]'"


def _ask_language(input_fn: Callable[[str], str]) -> str:
    """Ask the "force a language" question and return an answer safe to embed
    raw into vault.yaml: either a plausible bare language name or "" (no
    language forced — same as Enter).

    Both call sites below splice the answer directly into unquoted YAML.
    Left unvalidated: "yes"/"no"/"true" etc. parse as YAML booleans that
    `_parse_conventions` folds to None (the user believes they forced a
    language but silently didn't), and any other stray punctuation — a colon
    above all — can break the surrounding YAML, degrading the WHOLE manifest
    (in repo mode this silently drops sources too). Anything that
    isn't a bare name is treated as no answer rather than risking either
    failure mode.
    """
    raw = _ask(input_fn, _LANG_PROMPT).strip()
    if not raw:
        return ""
    if not _LANG_NAME_RE.match(raw) or raw.lower() in _LANG_ANSWER_REJECT:
        CONSOLE.print(
            f"  [yellow]'{raw}' doesn't look like a language name — skipping "
            "(no language forced; distiller follows the source language).[/]"
        )
        return ""
    return raw


def _ask_local_model(
    input_fn: Callable[[str], str],
    ids: list[str],
    label: str,
    base_url: str,
    example: str,
) -> str:
    """Ask for a local (LM Studio / Ollama) model id, offering the served ids as
    a pick-list with the first as default. When the endpoint returned nothing —
    server down or no model loaded — warn that the id is being entered blind
    (`silica doctor` verifies reachability) rather than silently asking for a
    guess that only fails at first chat."""
    if ids:
        return _pick(
            input_fn, f"{label} model — pick a number or type an id", ids,
            other_prompt=f"{label} model id",
        )
    CONSOLE.print(
        f"  [yellow]{GLYPHS['warn']} {label} not reachable at {base_url} — start it "
        "(and load a model) to get a pick-list. You can still type an id to set up "
        "offline; `silica doctor` will confirm once it's running.[/]"
    )
    model = ""
    while not model:
        model = _ask(input_fn, f"{label} model id (e.g. {example})")
    return model


# bounded walk — a repo-mode vault can be a source tree with
# node_modules, and init must not stall counting files it will only mention.
_DOC_SCAN_CAP = 20_000


def propose_form_fallback(vault: Path, cap: int = _DOC_SCAN_CAP) -> str | None:
    """Fallback distill profile to propose, from a mechanical `form:` census.

    docs/specs/nucleation-forms.md wizard step: no vault-wide auto-inference
    and zero extra LLM calls — only the ingress stamps already sitting in
    frontmatter count. Proposes only when the stamped sample is sizable
    (>= 10) and skewed (>= 70% one form), and never proposes draft: draft has
    no lens, so it cannot be a fallback profile.
    """
    from collections import Counter

    from silica.kernel.forms import profile_for, stamped_form

    counts: Counter[str] = Counter()
    scanned = 0
    for p in vault.rglob("*.md"):
        scanned += 1
        if scanned > cap:
            break
        try:
            form = stamped_form(p.read_text(encoding="utf-8", errors="replace")[:2000])
        except OSError:
            continue
        if form:
            counts[form] += 1
    total = sum(counts.values())
    if total < 10:
        return None
    form, n = counts.most_common(1)[0]
    if n / total < 0.7:
        return None
    return profile_for(form) or None


def _offer_form_fallback(input_fn: Callable[[str], str], vault: Path) -> None:
    """Propose `conventions.distill_profile` on a skewed vault; write on yes."""
    profile = propose_form_fallback(vault)
    if not profile:
        return
    answer = _ask(
        input_fn,
        f"This vault looks like mostly {profile}-form material — set "
        f"`conventions.distill_profile: {profile}` as the fallback lens? [y/n]",
        "y",
    )
    if answer.lower() not in ("y", "yes"):
        return
    from silica.kernel.vault_manifest import set_distill_profile

    set_distill_profile(vault, profile)
    CONSOLE.print(f"  {GLYPHS['ok']} conventions.distill_profile: {profile}")


def unindexable_docs(vault: Path, cap: int = _DOC_SCAN_CAP) -> list[Path]:
    """Documents present in the vault that the index cannot read.

    The index is markdown-only (`fs_backend._build_index` skips every non-`.md`
    file), so a vault of PDFs answers every question with nothing until
    `/convert` turns them into notes. Onboarding is the one place that sees the
    vault before the user asks their first question, so it is where the gap gets
    named.

    `CONVERTIBLE_DOC_EXTS`, not the full `DOC_EXTS`: the converter also takes
    images and media, and those sitting in a vault are attachments, not documents
    someone is waiting to ingest. Counting them would greet a user who pastes
    screenshots with "500 documents to convert", and would count the figures
    Silica itself extracted into `<inbox>/Images`.
    """
    from silica.sources.convert import CONVERTIBLE_DOC_EXTS

    out: list[Path] = []
    for i, p in enumerate(vault.rglob("*")):
        if i >= cap:
            break
        if p.suffix.lower() in CONVERTIBLE_DOC_EXTS and p.is_file():
            out.append(p)
    return out


def _warn_unindexable(vault: Path) -> None:
    """One line naming the documents `/convert` still has to bring in."""
    docs = unindexable_docs(vault)
    if not docs:
        return
    names = ", ".join(p.name for p in docs[:3])
    more = f" (+{len(docs) - 3} more)" if len(docs) > 3 else ""
    CONSOLE.print(
        f"  [yellow]{GLYPHS['warn']} {len(docs)} document(s) the index cannot read: "
        f"[/][dim]{names}{more}[/]"
    )
    CONSOLE.print(
        "  [dim]The vault is markdown-only. Run [/][bold]/convert <file>[/]"
        "[dim], then [/][bold]/nucleate[/][dim] on the result.[/]"
    )


def _run_wizard_inner(
    input_fn: Callable[[str], str],
    env_path: Path,
    advanced: bool = False,
) -> int:
    updates: dict[str, str] = {}
    # Cross-step state shared by the step closures below. Each step owns a
    # fixed set of `updates` keys and pops them on entry, so a re-run after
    # `back` never leaves stale leftovers.
    state: dict = {"advanced": advanced, "provider": "", "high_value": True, "write": False}
    # From the working directory, not env_path.parent: the vault question is
    # about where you launched `silica init`, and the settings file may well sit
    # in ~/.silica/ instead of in the repo.
    repo_root = gitstate.find_repo_root(Path.cwd())

    print_banner()
    CONSOLE.print()
    CONSOLE.print(
        "  [bold]Interactive setup[/]  [dim]· Enter accepts the shown default"
        " · type back to redo the previous step[/]"
    )

    def total() -> int:
        return 6 if state["advanced"] else 5

    def step_mode() -> bool:
        if advanced:  # `silica init --advanced` skips the question
            return False
        answer = ""
        while answer not in ("essential", "e", "advanced", "a"):
            answer = _ask(input_fn, "Setup mode — essential or advanced", "essential").lower()
        state["advanced"] = answer in ("advanced", "a")
        return True

    def step_vault() -> bool:
        # Repo mode: the repo root becomes the vault as-is, and `write_dir` says
        # where inside it Silica may write (docs/silica for a source tree).
        updates.pop("SILICA_VAULT", None)
        _section("vault", "Vault", 1, total())
        use_repo_mode = False
        if repo_root is not None:
            from silica.onboarding.adopt import write_dir_for

            # Same rule as every other entry point: the repo root is the vault.
            repo_vault = Path(repo_root)
            write_dir = write_dir_for(repo_vault)
            where = f"writes → {write_dir}/" if write_dir else "writes in place"
            answer = _ask(
                input_fn,
                f"Git repo detected — use repo mode? vault = {repo_vault} ({where}) [y/n]",
                "y",
            )
            if answer.lower() in ("y", "yes"):
                use_repo_mode = True
                repo_vault.mkdir(parents=True, exist_ok=True)
                manifest = repo_vault / MANIFEST_REL
                if not manifest.exists():
                    # Declared capabilities (ADR-0014): repo-mode vault wants the
                    # code source active.
                    lang_answer = _ask_language(input_fn)
                    content = "sources: [prose, code]\n"
                    if write_dir:
                        content += f"write_dir: {write_dir}\n"
                    if lang_answer:
                        # cooccurrence_lang (stemmer/stopwords) is separate from
                        # conventions.language (distiller translation intent). Pin
                        # both from the one answer so the co-occurrence store never
                        # falls back to fragile auto-detection.
                        content += f"cooccurrence_lang: {lang_answer.lower()}\n"
                        content += f"conventions:\n  language: {lang_answer}\n"
                    manifest.write_text(content, encoding="utf-8")
        if not use_repo_mode:
            while True:
                # The launch folder is the default vault — same rule as `silica`
                # itself. Without it this was the one step with no default, under
                # a banner promising "Enter accepts the shown default".
                path = _ask(input_fn, "Vault path (existing directory)", str(Path.cwd()))
                resolved = Path(path).expanduser() if path else None
                if resolved is not None and resolved.is_dir():
                    updates["SILICA_VAULT"] = str(resolved)
                    break
                CONSOLE.print(f"  [red]{GLYPHS['err']} Not a directory — try again.[/]")
            # The design's language question is unscoped to repo mode ("init asks
            # whether to force a language"): an explicit-path vault with no
            # vault.yaml yet must be asked too. Unlike repo mode there is no other
            # content due to be written for this vault, so Enter writes nothing —
            # a vault.yaml wouldn't otherwise exist, and conventions is the only
            # thing this question could ever put in it. An existing manifest is
            # never touched, and the question is skipped entirely in that case.
            manifest = resolved / MANIFEST_REL
            if not manifest.exists():
                lang_answer = _ask_language(input_fn)
                if lang_answer:
                    # Pin cooccurrence_lang (stemmer) alongside conventions.language
                    # (distiller) — two separate axes, one answer. See repo-mode note.
                    manifest.write_text(
                        f"cooccurrence_lang: {lang_answer.lower()}\n"
                        f"conventions:\n  language: {lang_answer}\n",
                        encoding="utf-8",
                    )
        vault_dir = repo_vault if use_repo_mode else resolved
        if vault_dir is None:  # neither path resolved: nothing to inspect
            return True
        _warn_unindexable(vault_dir)
        # nucleation-forms wizard step: proposal only, written only on yes,
        # and only when the stamped distribution is skewed — a fresh or mixed
        # vault asks nothing, so existing flows are untouched.
        _offer_form_fallback(input_fn, vault_dir)
        return True

    def step_provider() -> bool:
        # The hosted PROVIDER_PRESETS entries that need a key, plus `custom`
        # for any other OpenAI-compatible URL (vLLM, llama.cpp, ...).
        for key in (
            "SILICA_PROVIDER", "SILICA_MODEL", "SILICA_PROVIDER_BASE_URL",
            "SILICA_PROVIDER_API_KEY", *(v[0] for v in _HOSTED.values()),
        ):
            updates.pop(key, None)
        _section("model", "Chat provider", 2, total())
        # A model configured elsewhere (usually the global ~/.silica/.env)
        # survives a per-vault re-run: one Enter keeps it instead of re-asking
        # provider, model, and key every time init runs in a new vault. With
        # SILICA_MODEL unset the same Enter accepts whatever the fallback chain
        # derives from an exported provider key, so a user who already has one
        # answers zero questions here.
        current, source = model_from_env()
        if current:
            derived = source != "SILICA_MODEL"
            origin = f"from {source}" if derived else "already configured"
            answer = ""
            while answer not in ("y", "yes", "n", "no"):
                answer = _ask(
                    input_fn,
                    f"Chat model {current} ({origin}) — keep it? [y/n]",
                    "y",
                ).lower()
            if answer in ("y", "yes"):
                # A derived model lives only as long as that key stays exported:
                # pin it so the vault still works from a fresh shell. An explicit
                # SILICA_MODEL is left where it is, so it keeps following the
                # user between folders instead of being copied per vault.
                if derived:
                    updates["SILICA_MODEL"] = current
                state["provider"] = SilicaConfig().provider
                return True
        from silica.agent.providers import PROVIDER_PRESETS
        provider = ""
        while provider not in ("lmstudio", "ollama", "custom", *_HOSTED):
            provider = _ask(
                input_fn,
                "Chat provider — lmstudio or ollama (local), custom (any OpenAI-compatible URL), "
                "or hosted: " + ", ".join(_HOSTED),
                "lmstudio",
            ).lower()
        updates["SILICA_PROVIDER"] = provider
        state["provider"] = provider
        if provider in _HOSTED:
            key_env, models, key_prompt = _HOSTED[provider]
            # Key before model: with the key in hand the pick-list validates
            # against the provider's live /models instead of trusting the
            # hand-kept suggestions to still exist.
            key = ""
            while not key:
                key = _ask(input_fn, key_prompt, os.getenv(key_env, ""), secret=True)
            updates[key_env] = key
            model = _pick(
                input_fn, "Model — pick a number or type an id",
                _live_hosted_models(provider, key, models),
            )
        elif provider == "custom":
            base_url = ""
            while not base_url:
                base_url = _ask(input_fn, "Base URL (OpenAI-compatible, e.g. http://localhost:8000/v1)")
            updates["SILICA_PROVIDER_BASE_URL"] = base_url
            # Local servers usually ignore the key but the OpenAI SDK demands non-empty.
            updates["SILICA_PROVIDER_API_KEY"] = _ask(
                input_fn, "API key [Enter for none / local]", "dummy-key", secret=True
            )
            model = ""
            while not model:
                model = _ask(input_fn, "Model id served at that URL")
        elif provider == "ollama":
            ollama_base = PROVIDER_PRESETS["ollama"]["base_url"].removesuffix("/v1")
            model = _ask_local_model(
                input_fn, _ollama_installed_models(), "Ollama", ollama_base, "llama3.2"
            )
        else:  # lmstudio — probe /models like the Ollama branch does with tags.
            lmstudio_base = PROVIDER_PRESETS["lmstudio"]["base_url"]
            model = _ask_local_model(
                input_fn, endpoint_model_ids(lmstudio_base), "LM Studio", lmstudio_base, "qwen3-30b"
            )
        updates["SILICA_MODEL"] = model
        return True

    def step_gate() -> bool:
        # Essential-only: one question covering embeddings + reranker. `n`
        # jumps straight to write. Advanced asks both steps directly.
        if state["advanced"]:
            return False
        answer = ""
        while answer not in ("y", "yes", "n", "no"):
            answer = _ask(
                input_fn,
                "Configure high-value options now? Embeddings (semantic search, "
                "dedup) and reranker (better recall) [y/n]",
                "y",
            ).lower()
        state["high_value"] = answer in ("y", "yes")
        if not state["high_value"]:
            # A `back` may have left embedding keys from an earlier `y` pass;
            # the embeddings step won't run again to clear them.
            for key in _EMBED_KEYS:
                updates.pop(key, None)
        return True

    def step_embeddings() -> bool:
        # Optional; skipping degrades gracefully.
        if not (state["advanced"] or state["high_value"]):
            return False
        for key in _EMBED_KEYS:
            updates.pop(key, None)
        _section("think", "Embeddings", 3, total())
        from silica.agent.providers import PROVIDER_PRESETS
        defaults = SilicaConfig()
        provider = state["provider"]
        answer = _ask(
            input_fn,
            "Configure embeddings? `skip` degrades dedup//find to co-occurrence [y/skip]",
            "y",
        )
        if answer.lower() in ("skip", "s", "n", "no"):
            CONSOLE.print(
                "  [yellow]Embeddings skipped. Dedup routing and /find will not run; "
                "relatedness falls back to co-occurrence.[/]"
            )
            return True
        # Reuse the chat endpoint when it is local — it can usually serve
        # embeddings too, so a good setup needs no separate server.
        local = provider in ("lmstudio", "ollama")
        local_base = PROVIDER_PRESETS[provider]["base_url"] if local else defaults.embedding_base_url
        # ponytail: the "embed" substring is the ceiling — covers nomic-embed-text,
        # text-embedding-*; a served embedder without "embed" in its id needs the
        # explicit prompts below. Upgrade path: probe each model's capabilities.
        candidate = next(
            (m for m in endpoint_model_ids(local_base) if "embed" in m.lower()), ""
        ) if local else ""
        if candidate and _ask(
            input_fn, f"Use {candidate} at {local_base} for embeddings? [y/n]", "y"
        ).lower() in ("y", "yes"):
            updates["SILICA_EMBEDDING_MODEL"] = candidate
            updates["SILICA_EMBEDDING_BASE_URL"] = local_base
            updates["SILICA_EMBEDDING_API_KEY"] = defaults.embedding_api_key
        else:
            updates["SILICA_EMBEDDING_MODEL"] = _pick(
                input_fn, "Embedding model — pick a number or type an id",
                EMBED_MODELS, other_prompt="Embedding model",
                default=defaults.embedding_model,
            )
            updates["SILICA_EMBEDDING_BASE_URL"] = _ask(
                input_fn, "Embedding base URL", local_base
            )
            updates["SILICA_EMBEDDING_API_KEY"] = _ask(
                input_fn, "Embedding API key", defaults.embedding_api_key
            )
        _ask_serve_cmd(
            input_fn, updates, "SILICA_EMBEDDING_SERVE_CMD",
            updates["SILICA_EMBEDDING_BASE_URL"], _SERVE_CMDS.get(provider, ""),
        )
        return True

    def step_rerank() -> bool:
        # In-process cross-encoder via the [rerank] extra. The wizard never
        # installs anything — it prints the exact command and moves on.
        # SILICA_RERANK_* (an externally served reranker) is advanced-only.
        if not (state["advanced"] or state["high_value"]):
            return False
        _section("gear", "Reranker", 4, total())
        if _rerank_extra_present():
            # markup=False: the literal "[rerank]" would otherwise parse as a rich tag.
            CONSOLE.print(
                f"  {GLYPHS['ok']} Reranker active ([rerank] extra installed) — "
                "nothing to configure.",
                markup=False,
            )
            return False
        answer = ""
        while answer not in ("y", "yes", "n", "no"):
            answer = _ask(
                input_fn,
                "Enable the in-process reranker? ~2GB of weights download on first use [y/n]",
                "n",
            ).lower()
        if answer in ("y", "yes"):
            CONSOLE.print(f"  {GLYPHS['arrow']} Install it, then re-run [bold]silica doctor[/]:")
            CONSOLE.print(f"      {_rerank_install_cmd()}", markup=False)
        return True

    def step_worker() -> bool:
        if not state["advanced"]:
            return False
        updates.pop("SILICA_WORKER_MODEL", None)
        updates.pop("SILICA_WORKER_PROVIDER", None)
        _section("worker", "Advanced options", 5, total())
        provider = state["provider"]
        if provider == "lmstudio":
            from silica.agent.providers import PROVIDER_PRESETS
            options = endpoint_model_ids(PROVIDER_PRESETS["lmstudio"]["base_url"])
        elif provider == "ollama":
            options = _ollama_installed_models()
        else:
            options = _WORKER_MODELS.get(provider) or list(_HOSTED.get(provider, ("", [], ""))[1])
        model = _pick(
            input_fn,
            "Worker model for background tasks (dedup, refiner) "
            "[Enter = inherit main model]",
            options,
            other_prompt="Worker model id",
            default=None,  # Enter = inherit the main model, not options[0]
            required=False,
        )
        if model:
            updates["SILICA_WORKER_MODEL"] = model
            # get_provider(role="worker") falls back to the ROUTER model whenever
            # worker_provider is unset, so a worker model alone is silently
            # ignored — pin the provider with it, always.
            updates["SILICA_WORKER_PROVIDER"] = provider
        return True

    def step_git() -> bool:
        if not state["advanced"]:
            return False
        updates.pop("SILICA_GIT_COMMIT", None)
        answer = None
        while answer not in ("", "y", "yes", "n", "no"):
            answer = _ask(
                input_fn,
                "Git auto-commit every vault write — the undo safety net? "
                "y = auto, n = off [Enter = leave off]",
            ).lower()
        if answer:
            updates["SILICA_GIT_COMMIT"] = "auto" if answer in ("y", "yes") else "off"
        return True

    def step_websearch() -> bool:
        if not state["advanced"]:
            return False
        updates.pop("SILICA_TAVILY_API_KEY", None)
        key = _ask(
            input_fn,
            "Tavily API key — /web-search backstop when DuckDuckGo rate-limits "
            "[Enter = skip]",
            secret=True,
        )
        if key:
            updates["SILICA_TAVILY_API_KEY"] = key
        return True

    def step_pdf() -> bool:
        if not state["advanced"]:
            return False
        updates.pop("SILICA_PDF_PROVIDER", None)
        updates.pop("SILICA_PDF_OCR_LANG", None)
        answer = None
        while answer not in ("", "pdfium", "mineru", "docling", "opendataloader"):
            answer = _ask(
                input_fn,
                "PDF converter — pdfium, mineru, docling, or opendataloader "
                "[Enter = pdfium; the other three install separately, and "
                "mineru is the only one with OCR]",
            ).lower()
        if answer:
            updates["SILICA_PDF_PROVIDER"] = answer
        if answer == "docling":
            updates["SILICA_PDF_OCR_LANG"] = _ask(
                input_fn, "OCR languages for docling (comma-separated)", "en,it,fr,de,es"
            )
        return True

    def step_ext_rerank() -> bool:
        # Only for a reranker the user already serves; all three keys or none.
        if not state["advanced"]:
            return False
        for key in ("SILICA_RERANK_BASE_URL", "SILICA_RERANK_MODEL", "SILICA_RERANK_API_KEY",
                    "SILICA_RERANK_SERVE_CMD"):
            updates.pop(key, None)
        answer = ""
        while answer not in ("y", "yes", "n", "no"):
            answer = _ask(
                input_fn,
                "Point at an external reranker you already serve "
                "(llama.cpp --reranking, Infinity, Jina)? [y/n]",
                "n",
            ).lower()
        if answer in ("y", "yes"):
            updates["SILICA_RERANK_BASE_URL"] = _ask(
                input_fn, "Reranker base URL", "http://localhost:1235/v1"
            )
            updates["SILICA_RERANK_MODEL"] = _pick(
                input_fn, "Reranker model — pick a number or type an id",
                RERANK_MODELS, other_prompt="Reranker model id",
            )
            updates["SILICA_RERANK_API_KEY"] = _ask(
                input_fn, "Reranker API key", "lm-studio", secret=True
            )
            _ask_serve_cmd(
                input_fn, updates, "SILICA_RERANK_SERVE_CMD",
                updates["SILICA_RERANK_BASE_URL"], "",
            )
        return True

    def step_write() -> bool:
        _section("arrow", "Write configuration", total(), total())
        CONSOLE.print(
            f"  {len(updates)} key(s) → [bold]{env_path}[/]: "
            f"[dim]{', '.join(sorted(updates))}[/]"
        )
        state["write"] = _ask(input_fn, "Write? [y/n]", "y").lower() in ("y", "yes")
        return True

    # Driver: run steps in order; `back` rewinds to the previous step that
    # actually asked something (auto-skipped steps are transparent).
    steps = [
        step_mode, step_vault, step_provider, step_gate, step_embeddings,
        step_rerank, step_worker, step_git, step_websearch, step_pdf,
        step_ext_rerank, step_write,
    ]
    asked = [False] * len(steps)
    i = 0
    while i < len(steps):
        try:
            asked[i] = steps[i]()
        except BackRequested:
            j = i - 1
            while j >= 0 and not asked[j]:
                j -= 1
            if j < 0:
                CONSOLE.print("  [dim]Already at the first question.[/]")
            else:
                i = j
            continue
        i += 1

    if not state["write"]:
        CONSOLE.print(f"  [dim]{GLYPHS['err']} Aborted — nothing written.[/]")
        return 1
    # Fresh .env: seed from .env.example so every knob ships documented, with the
    # collected keys filled in. Existing .env: merge in place, untouched otherwise.
    if env_path.exists():
        base = env_path.read_text()
    else:
        example = _find_env_example(repo_root)
        base = example.read_text(encoding="utf-8") if example else ""
    # Atomic: a crash mid-write here would truncate the live .env — every API
    # key the user has — not just the values being merged.
    from silica.kernel.recall.paths import atomic_write_bytes
    atomic_write_bytes(env_path, merge_env(base, updates).encode("utf-8"))
    CONSOLE.print(f"  [green]{GLYPHS['ok']} Wrote {env_path}[/]")

    # Doctor checks against the values just chosen.
    CONSOLE.print()
    CONSOLE.print(f"  [bold brand.cyan]{GLYPHS['run']} Checking your setup[/]")
    claim_env(updates)
    results = run_checks(SilicaConfig())
    render_report(results)

    CONSOLE.print()
    CONSOLE.print(f"  [bold brand.cyan]{GLYPHS['arrow']} Next steps[/]")
    CONSOLE.print("  [dim]·[/] Run [bold]silica[/] — try ingesting a file or asking a question.")
    CONSOLE.print(
        "  [dim]·[/] Run [bold]silica doctor --live[/] to confirm the model actually replies "
        "(sends one tiny request)."
    )
    CONSOLE.print(
        f"  [dim]·[/] Every other option lives documented in [bold]{env_path}[/] — edit anytime."
    )
    CONSOLE.print(
        "  [dim]·[/] Re-run [bold]silica init[/] anytime — it updates values in place "
        "and never deletes your edits."
    )
    return 1 if has_failures(results) else 0


def run_wizard(
    input_fn: Callable[[str], str] = input,
    env_path: Path | None = None,
    advanced: bool = False,
) -> int:
    if env_path is None:
        env_path = resolve_env_path()  # shared with the web settings panel
    try:
        return _run_wizard_inner(input_fn, env_path, advanced=advanced)
    except KeyboardInterrupt:
        CONSOLE.print(
            f"\n  [dim]{GLYPHS['err']} Aborted — nothing written beyond what was already confirmed.[/]"
        )
        return 1
