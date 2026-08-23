# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Prepare and run Distiller delegation for the Injector pipeline.

This module ports `build_tasks()` from Hermes prep_delegation.py as a pure
function, and adds `run_distiller()` which calls the LLM directly via
`call_llm()` (stateless, single-turn, no tool use).

The protocol template uses {TARGET} as the only substitution. PAYLOAD_PATH
is passed as a file reference in the task context, not inlined into the prompt.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import typing
from pathlib import Path

logger = logging.getLogger(__name__)

# Distiller prompt template — vendored at install time
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "capabilities" / "prompts" / "distiller_prompt.txt"
# Shared anti-slop fragment, appended to every body-writing prompt (refine/enrich too).
_ANTI_SLOP_PATH = _PROMPT_PATH.parent / "_anti_slop.txt"
# Distill profiles: the template is the fixed validator-aligned contract; the
# {LENS_RUBRIC}/{LENS_QUALITY}/{LENS_EXAMPLES} placeholders are filled from
# profiles/<name>/{rubric,quality,examples}.md. `default` reproduces the
# pre-split prompt bit-identically.
_PROFILES_DIR = _PROMPT_PATH.parent / "profiles"
_LENS_FRAGMENTS = ("rubric", "quality", "examples", "ephemeral_routing")


def _load_prompt() -> str:
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Distiller prompt not found: {_PROMPT_PATH}")
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _vault_profiles_dir() -> Path | None:
    """<vault>/.silica/profiles/ for the active vault; None when unbound."""
    from silica.config import CONFIG

    vault = (getattr(CONFIG, "vault_path", "") or "").strip()
    return Path(vault) / ".silica" / "profiles" if vault else None


def _splice_lens(body: str, profile: str) -> str:
    """Fill the {LENS_*} placeholders from the named profile's fragments.

    Per-fragment search order: vault-local (<vault>/.silica/profiles/<name>/)
    > bundled <name>/ > bundled default/. A profile may override only some
    fragments, and a vault-local dir may shadow a bundled profile of the same
    name fragment-by-fragment. Unknown profile ⇒ warn + default (soft,
    matches vault.yaml parsing style).
    """
    # trust boundary: the name comes from vault.yaml/env and joins filesystem
    # paths — separators or ".." must not escape the profile roots
    if profile != "default" and (
        not profile.strip() or "/" in profile or "\\" in profile or ".." in profile
    ):
        logger.warning("Invalid distill profile name %r — using default", profile)
        profile = "default"
    roots = [d for d in (_vault_profiles_dir(), _PROFILES_DIR) if d is not None]
    if profile != "default" and not any((r / profile).is_dir() for r in roots):
        logger.warning("Unknown distill profile %r — using default", profile)
        profile = "default"
    for frag in _LENS_FRAGMENTS:
        candidates = [r / profile / f"{frag}.md" for r in roots]
        candidates.append(_PROFILES_DIR / "default" / f"{frag}.md")
        path = next(p for p in candidates if p.is_file())
        body = body.replace("{LENS_" + frag.upper() + "}",
                            path.read_text(encoding="utf-8"))
    return body


def active_distill_profile() -> str:
    """The distill profile in force: SILICA_DISTILL_PROFILE env >
    `conventions.distill_profile` > "default".

    Never raises: the validator asks on every ingest op, and a missing or
    unreadable manifest must read as the default profile, not an error."""
    try:
        from silica.kernel.vault_manifest import get_active_manifest

        declared = get_active_manifest().conventions.distill_profile
    except Exception:
        declared = ""
    return os.getenv("SILICA_DISTILL_PROFILE") or declared or "default"


def render_prompt(target: str, hub: str | None = None, source_text: str = "",
                  session_date: str = "", language: str | None = None,
                  profile: str | None = None) -> str:
    """Render the distiller prompt with TARGET/LANGUAGE/MAX_TAGS substitution.

    The prompt = fixed contract + profile lens (see `_splice_lens`). Profile
    precedence: explicit `profile` arg (per-run, e.g. /promote) >
    SILICA_DISTILL_PROFILE env > `conventions.distill_profile` > "default".

    `session_date` (F2a): the date the SOURCE session/document happened — not
    necessarily today (eval passes simulated time; dated documents pass their
    own date). Empty ⇒ "unknown", and the prompt rule keeps source wording.

    MAX_TAGS comes from the active vault's `conventions:` block
    (silica/kernel/vault_manifest.py) — single source shared with
    `ofm.ofm_lint`'s max-tags check. A vault without a manifest gets today's
    default (3), so this is bit-identical when unconfigured.

    LANGUAGE precedence: explicit `language` arg > declared
    `conventions.language` > per-call detection from `source_text`. An explicit
    arg is pinned once per file at PAYLOAD so the rendered template is
    byte-identical across a file's chunks/steer retries (cache-stable prefix);
    `conventions.language` unset (None) means "follow the source document's
    language" — detected from `source_text` (capped to 4000 chars, enough
    signal without scanning whole PDFs). A declared `conventions.language` is
    translation intent and always wins over detection. The {LANGUAGE}
    placeholder always receives a concrete language name ("Italian",
    "English", ...), never None.
    """
    from silica.kernel.text import language as lang_mod
    from silica.kernel.vault_manifest import get_active_manifest

    conventions = get_active_manifest().conventions
    profile = profile or active_distill_profile()
    body = _splice_lens(_load_prompt(), profile)
    body = body.replace("{TARGET}", target)
    if hub:
        body = body.replace("{HUB_NAME}", hub)
    # Cache-stable prefix: an explicit `language` (pinned once per file at
    # PAYLOAD) wins over per-call detection, so the rendered template is
    # byte-identical across all chunks and steer retries of a file.
    lang_name = language or conventions.language or lang_mod.display_name(
        lang_mod.detect(source_text[:4000])
    )
    body = body.replace("{LANGUAGE}", lang_name)
    body = body.replace("{MAX_TAGS}", str(conventions.max_tags))
    body = body.replace("{SESSION_DATE}", session_date.strip() or "unknown")
    # F1b: vault-declared capture rules. Empty ⇒ the placeholder line vanishes
    # entirely (consume its trailing newline), so an unconfigured vault renders
    # bit-identically to before this existed.
    rules = (conventions.capture_rules or "").strip()
    body = body.replace(
        "{CAPTURE_RULES}\n",
        f"## Vault capture rules\n{rules}\n\n" if rules else "",
    )
    if _ANTI_SLOP_PATH.exists():  # optional fragment, missing file must not break nucleation
        body += "\n\n" + _ANTI_SLOP_PATH.read_text(encoding="utf-8")
    return body


def _payload_sample_text(payload: dict, limit: int | None = 4000) -> str:
    """Concatenate inbox excerpts from the payload as a source-language sample.

    Used only when `conventions.language` is unset — detecting the dominant
    language of the batch's own inbox content is cheap and enough signal for
    the {LANGUAGE} placeholder; capped early so we never build a huge string
    for a single detect() call. `limit=None` returns everything.
    """
    parts: list[str] = []
    total = 0
    for batch in payload.get("batches", []):
        for concept in batch.get("concepts", []):
            excerpt = concept.get("inbox_excerpt") if isinstance(concept, dict) else None
            if excerpt:
                parts.append(excerpt)
                total += len(excerpt)
                if limit is not None and total >= limit:
                    return "\n".join(parts)[:limit]
    return "\n".join(parts)


def payload_inbox_text(payload: dict) -> str:
    """The chunk's full inbox text — the source the sanitize step anchors
    verbatim-body escape repair on (sanitize.normalize_ops verbatim_source)."""
    return _payload_sample_text(payload, limit=None)


# Per-string cap when echoing a rejected op back to the model: enough to
# recognize the op, not enough to blow up the prompt with full note bodies.
_STEER_ECHO_MAX_CHARS = 280


def _truncate_op_echo(obj, limit: int = _STEER_ECHO_MAX_CHARS):
    """Deep-copy `obj` with long strings truncated and empty fields dropped."""
    if isinstance(obj, str):
        return obj if len(obj) <= limit else obj[:limit] + f"… [truncated, {len(obj)} chars total]"
    if isinstance(obj, dict):
        return {k: _truncate_op_echo(v, limit) for k, v in obj.items()
                if v is not None and v != "" and v != []}
    if isinstance(obj, list):
        return [_truncate_op_echo(v, limit) for v in obj]
    return obj


def render_steer_feedback(
    rejected: list[dict],
    *,
    attempt: int,
    max_attempts: int,
    accepted: list[dict] | None = None,
    partial: bool = False,
    ungrounded: list[dict] | None = None,
) -> str:
    """Structured per-op steering feedback for a re-delegation attempt.

    Paper-aligned (PDDL-INSTRUCT): the corrective prompt echoes the previous
    output with a per-op verdict and the validator's detailed reason, instead
    of a flat concatenation of reasons — detailed feedback measurably beats
    binary feedback in verifier-guided refinement loops.

    Args:
        rejected: validator rejection entries, each `{"op": {...}, "reason": str}`
            (entries missing `op` degrade to a reason-only line).
        attempt/max_attempts: steer-arc position, shown in the header.
        accepted: validated op dicts from the same pass (partial steer) —
            listed so the model does not re-emit them.
        partial: True when the payload was filtered to the rejected concepts.
        ungrounded: span-grounding findings on ACCEPTED ops (`{"heading",
            "path", "spans"}`) — advisory only, the gate stays warn-only; the
            retry must not introduce more content untraceable to the payload.
    """
    accepted = accepted or []
    lines = [
        f"## STEERING CORRECTION (attempt {attempt}/{max_attempts})",
        f"Your previous output was validated: {len(accepted)} op(s) ACCEPTED, "
        f"{len(rejected)} op(s) REJECTED.",
    ]
    if partial:
        lines.append(
            "The payload in this message now contains ONLY the concepts whose ops "
            "were rejected; the accepted ops are already being written."
        )
    else:
        lines.append("Regenerate the full output, fixing every rejected op below.")

    if accepted:
        lines.append("\n### Accepted ops (do NOT re-emit these)")
        for op in accepted:
            if isinstance(op, dict):
                lines.append(f"- [{op.get('op', '?')}] {op.get('path') or op.get('heading', '?')}")

    for i, r in enumerate(rejected, 1):
        if not isinstance(r, dict):
            continue
        rej_op: dict | None = r.get("op") if isinstance(r.get("op"), dict) else None
        label = f"[{rej_op.get('op', '?')}] \"{rej_op.get('title') or rej_op.get('heading', '?')}\"" if rej_op else "(op not recorded)"
        lines.append(f"\n### Rejected op {i} — {label}")
        lines.append(f"Verdict: REJECTED — {r.get('reason') or 'no reason recorded'}")
        if rej_op:
            lines.append("Your op was:")
            lines.append("```json")
            lines.append(json.dumps(_truncate_op_echo(rej_op), ensure_ascii=False, indent=2))
            lines.append("```")

    if ungrounded:
        lines.append(
            "\n### Grounding warnings (accepted, but fix the habit)\n"
            "These accepted ops contain spans NOT traceable to any payload excerpt. "
            "They were not rejected, but your corrected ops must only carry facts "
            "grounded in inbox_excerpt:"
        )
        for u in ungrounded:
            if not isinstance(u, dict):
                continue
            spans = " | ".join(s[:80] for s in u.get("spans", [])[:3])
            lines.append(f"- \"{u.get('heading', '?')}\" ({u.get('path', '?')}): {spans}")

    lines.append(
        "\n### Instructions\n"
        "For EVERY rejected op above, re-emit a corrected op that fixes exactly "
        "the violated constraint stated in its verdict. Do not introduce new "
        "concepts and do not re-emit accepted ops."
    )
    return "\n".join(lines)


# Floor for the computed output budget. If the prompt is so large that no
# meaningful headroom remains, we still ask for at least this much rather than a
# negative/zero value — the API call will surface the real problem instead of us
# silently requesting nonsense.
_MIN_DISTILLER_OUTPUT_TOKENS = 1024


def estimate_prompt_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token), rounded up.

    Intentionally provider-agnostic: we only need a conservative figure to
    leave output headroom, not an exact tokenizer count.
    """
    return (len(text) + 3) // 4


def compute_distiller_max_tokens(
    prompt_text: str,
    *,
    context_window: int,
    safety_margin: int,
    ceiling: int = 0,
) -> int:
    """Size the output budget to the real prompt and the model's context window.

    `max_tokens = min(ceiling?, context_window - prompt_tokens - safety_margin)`,
    floored at `_MIN_DISTILLER_OUTPUT_TOKENS`.

    `ceiling <= 0` means "no manual cap" — use all available headroom.

    A 2026-06 note here blamed a 32k ceiling for "truncating dense batches";
    refuted 2026-08-21: the truncation users see in notes was hub_desc's raw
    120-char slice, and a 250k ask never once finished for length across a
    14-file run (0 salvage calls, ~3k mean completion). The caller bounds the
    default ask with MAX_TOKENS; DISTILLER_MAX_TOKENS pins it explicitly.
    """
    available = context_window - estimate_prompt_tokens(prompt_text) - safety_margin
    available = max(_MIN_DISTILLER_OUTPUT_TOKENS, available)
    if ceiling and ceiling > 0:
        return min(ceiling, available)
    return available


def salvage_distiller_json(raw: str) -> dict | None:
    """Recover the complete `updates` entries from a truncated distiller response.

    The distiller emits one large `{"main_thematic_axes": [...], "updates": [...]}`
    object. When generation is cut off mid-array the whole document is
    unparseable, but every element BEFORE the truncation point is valid JSON.
    This scans the `updates` array element-by-element and keeps every object that
    parses cleanly, discarding only the final half-written one.

    Returns `{"main_thematic_axes": [...], "updates": [...]}` with at least one
    recovered update, or None when nothing complete can be salvaged.
    """
    decoder = json.JSONDecoder()

    axes: list = []
    ax_key = raw.find('"main_thematic_axes"')
    if ax_key != -1:
        ax_bracket = raw.find("[", ax_key)
        if ax_bracket != -1:
            try:
                axes, _ = decoder.raw_decode(raw, ax_bracket)
            except ValueError:
                axes = []

    up_key = raw.find('"updates"')
    if up_key == -1:
        return None
    arr = raw.find("[", up_key)
    if arr == -1:
        return None

    updates: list = []
    i = arr + 1
    n = len(raw)
    while i < n:
        while i < n and raw[i] in " \t\r\n,":
            i += 1
        if i >= n or raw[i] == "]":
            break
        try:
            obj, end = decoder.raw_decode(raw, i)
        except ValueError:
            break  # trailing object is truncated — stop here
        if isinstance(obj, dict):
            updates.append(obj)
        i = end

    if not updates:
        return None
    return {"main_thematic_axes": axes if isinstance(axes, list) else [], "updates": updates}


# Backslash escapes that constrained decoding CANNOT protect: `\b \f \n \r \t`
# are valid JSON escapes that silently decode to control characters (`"\top"`
# becomes TAB + "op"). Every other single backslash is invalid JSON, so the
# grammar forces the model to double it. Bodies must be grounded in the inbox
# excerpts (anti-hallucination contract), so a clean excerpt cannot seed a
# hazardous body.
_ESCAPE_HAZARD_RE = re.compile(r"\\[bfnrt]")


def needs_body_pass(payload: dict) -> bool:
    """True when a chunk's excerpts can seed silent JSON-escape corruption."""
    for batch in payload.get("batches") or []:
        for concept in batch.get("concepts") or []:
            if _ESCAPE_HAZARD_RE.search(concept.get("inbox_excerpt") or ""):
                return True
    return False


_STRUCTURE_NOTE = (
    "\n\nSTRUCTURE PASS: do NOT write note bodies in this response — the "
    "output schema carries no snippet/content field. Emit the ops, skips and "
    "ephemerals only; bodies are requested separately."
)


def _body_ops(parsed: dict) -> list[dict]:
    """The pass-1 ops that carry a body, in output order (= body numbering)."""
    return [op for op in parsed.get("updates") or []
            if isinstance(op, dict)
            and op.get("op") in ("write", "patch", "overwrite")]


def _body_pass_instruction(ops: list[dict]) -> str:
    """The pass-2 user turn. Short by design: the full contract is already in
    the conversation prefix (continuation), so only the numbering and the
    verbatim rule need stating."""
    listing = "\n".join(
        f"{i}. {op.get('op')} → {op.get('path')} (heading: {op.get('heading')!r})"
        for i, op in enumerate(ops, 1)
    )
    return (
        "Now write the note bodies for the ops you planned. For each op in "
        "the list below, in order, emit exactly one block:\n\n"
        "===SILICA-BODY N===\n<body for op N>\n\n"
        "N is the op's number in the list. Output ONLY the blocks — no JSON, "
        "no prose before the first marker, nothing after the last body.\n"
        "Body text is copied to the note file as-is: literal single "
        "backslashes (\\frac stays \\frac, never \\\\frac), and real line "
        "breaks where you mean a new line — never \\n standing in for one. "
        "That rule is about layout, not content: when the source discusses an "
        "escape sequence itself (\\n, \\t, \\r, \\b, \\f as the subject), copy "
        "the two characters verbatim, backslash included. `splits on \\n` must "
        "not become `splits on ` followed by a line break. "
        "Every body rule from the contract still applies "
        "(prose language, wikilinks without .md, no self-links, no "
        "descriptive summaries).\n\n" + listing
    )


# overwrite carries its body in `content`; write/patch in `snippet`.
_BODY_FIELD = {"overwrite": "content"}


def _stitch_bodies(parsed: dict, raw: str) -> None:
    """Resolve pass-2 `===SILICA-BODY N===` blocks into the pass-1 ops.

    A missing block leaves the op bodyless: the validate floor rejects it
    downstream (fail closed, same path as a dangling snippet_ref)."""
    from silica.kernel.text.sanitize import VERBATIM_BODY, extract_body_appendix

    _, bodies = extract_body_appendix(raw)
    for i, op in enumerate(_body_ops(parsed), 1):
        body = bodies.get(i)
        if body:
            op[_BODY_FIELD.get(str(op.get("op") or ""), "snippet")] = body
            op[VERBATIM_BODY] = True
        else:
            logger.warning(
                "body pass: no ===SILICA-BODY %d=== block for %s — left empty",
                i, op.get("path") or op.get("heading") or "?")


def _call_with_deadline(fn, seconds: float, cancel: "threading.Event | None" = None):
    """Run fn() under a wall-clock deadline; raise TimeoutError past it.

    The transport read-timeout cannot bound a hung distiller call: OpenRouter
    trickles keep-alive bytes while "processing", and every byte resets httpx's
    per-chunk read timer, so a dead upstream holds the socket open forever.
    Only real elapsed time is trustworthy here.

    `cancel` is set when the deadline fires. Without it the abandoned worker kept
    its own retry schedule and went on issuing requests nobody was waiting for:
    measured 2026-08-23 on a throttled OpenRouter pool, this deadline gave up at
    300s and the orphan hit its 400s per-call cap at t+400 and started attempt
    2/3, so the run's own dead calls were producing the 429s it was retrying for.
    """
    from silica.agent.llm import run_with_deadline
    try:
        return run_with_deadline(
            fn, seconds,
            lambda: TimeoutError(f"distiller call exceeded {seconds:.0f}s wall-clock deadline"),
        )
    except TimeoutError:
        if cancel is not None:
            cancel.set()
        raise


def run_distiller(
    payload: dict,
    target: str,
    hub: str | None = None,
    ledger_digest: str | None = None,
    steer_context: str | None = None,
    substrate: str | None = None,
    session_date: str = "",
    language: str | None = None,
    escalate: bool = False,
    profile: str | None = None,
    structure_only: bool = False,
) -> dict:
    """Call the Distiller LLM for one payload chunk.

    Single call by default. When the chunk's excerpts can seed silent
    JSON-escape corruption (needs_body_pass), the call splits in two: a
    structure pass under constrained decode (schema without body fields), then
    a body pass as a same-conversation continuation where prose travels outside
    JSON entirely — verbatim backslashes, real line breaks. Every profile,
    extractive included. SILICA_DISTILL_TWO_PASS=0 forces single-call
    everywhere (re-admits the corruption class on hazard chunks).

    Args:
        payload: the payload dict (schema_version + batches)
        target: vault-relative target directory for new notes
        hub: optional [[Hub]] note name
        ledger_digest: compact run summary injected as context header (Phase 2)
        steer_context: corrective steering note injected when re-attempting after
            rejection (Phase 6). States why the previous output was rejected.
        escalate: route this call to the escalation model (steer retries; Tier 2 cascade).
        profile: per-run distill profile override (e.g. "promotion" for
            /promote); None follows env/manifest resolution.
        structure_only: never run the body pass and use the body-less schema —
            for callers that discard note bodies (the episodic drain keeps only
            ephemerals), so no body tokens are ever paid for.

    Returns:
        parsed dict with {"updates": [...]} or {"error": ...}
    """
    from silica.agent.providers import get_provider
    from silica.config import CONFIG
    from silica.kernel import distill_cache
    from silica.kernel.context_builder import build_context
    from silica.kernel.write.ops import DistillerOutput, DistillerStructure
    from silica.kernel.text.sanitize import parse_json

    # Extractive-class runs used to be excluded here, on the claim that a
    # corrupted body breaks the verbatim substring match so the validator
    # rejects it. Measured false for the newline class: the expansion splits
    # the line in two and both halves remain substrings of the source, so the
    # floor passes it silently. Extractive is in fact where verbatim transport
    # matters most — a selected span must reach the note as selected.
    two_pass = (
        not structure_only
        and os.getenv("SILICA_DISTILL_TWO_PASS", "1") != "0"
        and needs_body_pass(payload)
    )

    prompt_text = render_prompt(target=target, hub=hub,
                                source_text=_payload_sample_text(payload),
                                session_date=session_date, language=language,
                                profile=profile)
    # Assemble context through the context assembler (Phase 2 rails).
    # Only ledger_digest + the checkpoint payload reach the model — no other
    # vault content is forwarded here.
    ctx = build_context(
        checkpoint_id="distill",
        payload=payload,
        ledger_digest=ledger_digest,
        substrate=substrate,
    )

    # steer_context arrives fully rendered (render_steer_feedback), header included.
    steer_section = f"\n\n{steer_context}\n" if steer_context else ""
    ctx_text = f"---\n{ctx}"
    # Budget arithmetic runs on the same concatenation as the old single-message
    # prompt, so token sizing is unchanged by the split.
    budget_text = f"{prompt_text}\n\n{ctx_text}{steer_section}"

    # Cache-stable layout ("Don't Break the Cache", arXiv 2601.06007): the
    # per-file-stable template is a system block with a cache_control marker;
    # dynamic content (ctx) is user part 1 with a second marker so steer
    # retries reuse template+ctx prefill; steer is an appended trailing part.
    # Non-caching upstreams ignore the markers harmlessly.
    user_parts: list[dict[str, typing.Any]] = [
        {"type": "text", "text": ctx_text, "cache_control": {"type": "ephemeral"}}
    ]
    if steer_section:
        user_parts.append({"type": "text", "text": steer_section})
    if two_pass or structure_only:
        # Trailing part: the cached prefix (template + ctx) stays byte-identical.
        user_parts.append({"type": "text", "text": _STRUCTURE_NOTE})
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": prompt_text, "cache_control": {"type": "ephemeral"}}
        ]},
        {"role": "user", "content": user_parts},
    ]

    # Cache lookup before the budget arithmetic: sizing the output asks the
    # live provider for its window, so a hit that ran after it would pay a
    # network round trip to replay a stored reply. Fingerprints only under the
    # flag: entry_key serializes the full context payload, a per-note cost the
    # default-off live path must not pay.
    cache_ns = cache_key = None
    if distill_cache.enabled():
        cache_ns = distill_cache.prompt_fingerprint(prompt_text)
        cache_key = distill_cache.entry_key({
            "messages": messages,
            "schema": "structure" if (two_pass or structure_only) else "full",
            "two_pass": two_pass,
            "escalate": escalate,
            "model": (CONFIG.distill_escalation_model if escalate
                      else CONFIG.worker_model) or CONFIG.model,
            # max_tokens deliberately out of the key. It is derived
            # from live provider limits, so including it would turn an
            # unreachable provider into a cache miss instead of a replay. Put
            # it in the key if an arm ever needs to vary the output budget on
            # purpose.
        })
        if (hit := distill_cache.load(cache_ns, cache_key)) is not None:
            logger.info("Distiller reply replayed from cache (p%s)", cache_ns)
            return hit

    # Only a complete reply may enter the cache. A salvaged prefix, a length-
    # truncated generation, a bodyless stitch or a fallback-served call is a
    # recovery input: storing one replays the loss on every later run of the
    # arm, silently, which is worse than no cache at all.
    cache_ok = True

    logger.info("Calling Distiller LLM%s", " (escalated)" if escalate else "")

    # #2: size the output budget to the real prompt + model context window
    # instead of a fixed ceiling. Window and output cap come from the live
    # provider (LM Studio /api/v0/models, OpenRouter /api/v1/models);
    # MODEL_CONTEXT_WINDOW / DISTILLER_MAX_TOKENS stay as explicit operator
    # overrides, 262144 as the last-resort default when the provider is
    # unreachable or the model unmapped.
    context_window = int(os.getenv("MODEL_CONTEXT_WINDOW", "0"))
    ceiling = int(os.getenv("DISTILLER_MAX_TOKENS", "0"))
    explicit_ceiling = ceiling > 0
    if not context_window or not ceiling:
        from silica.agent.providers import model_limits
        # Same fallback chain as get_provider for the active role.
        if escalate:
            w_provider, w_model = (CONFIG.distill_escalation_provider,
                                   CONFIG.distill_escalation_model)
        else:
            w_provider, w_model = CONFIG.worker_provider, CONFIG.worker_model
        if not w_provider or not w_model:
            w_provider, w_model = CONFIG.provider, CONFIG.model
        window, out_cap = model_limits(w_provider, w_model)
        context_window = context_window or window or 262144
        ceiling = ceiling or out_cap
    if not explicit_ceiling:
        # Bound the default ask with the audited MAX_TOKENS (32768: the
        # token-cost audit measured 256k bad / 32k good on the router pool)
        # instead of the provider's own out_cap (384k on deepseek). The bare
        # out_cap ask was justified by a truncation story refuted on run
        # 262e6847 (see compute_distiller_max_tokens), and it has real costs:
        # OpenRouter drops pool endpoints advertising less than max_tokens,
        # and a reasoning model may bill its whole trace against the ask.
        # DISTILLER_MAX_TOKENS stays the explicit override, both directions;
        # a genuine length cut still recovers via salvage_distiller_json.
        ceiling = min(ceiling or (1 << 30), int(os.getenv("MAX_TOKENS", "32768")))
    safety_margin = int(os.getenv("DISTILLER_TOKEN_SAFETY_MARGIN", "2048"))
    max_tokens = compute_distiller_max_tokens(
        budget_text,
        context_window=context_window,
        safety_margin=safety_margin,
        ceiling=ceiling,
    )
    logger.info(
        "Distiller output budget: %d tokens (window=%d, prompt≈%d, margin=%d, ceiling=%s)",
        max_tokens, context_window, estimate_prompt_tokens(budget_text), safety_margin,
        ceiling if ceiling > 0 else "none",
    )

    deadline = float(os.getenv("DISTILLER_TIMEOUT", "300"))

    def _llm(msgs: list[dict], response_schema):
        nonlocal cache_ok
        provider = None
        try:
            provider = get_provider(CONFIG, role="escalation" if escalate else "worker")
            abandoned = threading.Event()
            return _call_with_deadline(lambda: provider.call_llm(
                messages=msgs,
                tools=None,
                response_schema=response_schema,
                max_tokens=max_tokens,
                # The distiller pin is tied to the worker model's provider routing;
                # an escalated call must not inherit it.
                openrouter_provider=None if escalate else CONFIG.openrouter_provider_distiller,
                cancel=abandoned,
            ), deadline, abandoned)
        except Exception as e:
            fallback_model = (CONFIG.distill_escalation_model or CONFIG.model) if escalate else CONFIG.model
            if provider is not None and fallback_model == provider.model:
                # Same model, same stack (Provider routes through llm.call_llm too):
                # a second full `deadline` round with the same retry schedule cannot
                # succeed where the first just failed, and on a rate-limited pool it
                # doubles the load that caused the failure. Only a DIFFERENT model
                # makes this branch worth its cost.
                raise
            # The reply now comes from the router model, not the worker the
            # cache key names: replaying it would blend two models' corpora
            # under one key.
            cache_ok = False
            logger.warning("Distiller call failed, retrying on %s: %s", fallback_model, e)
            from silica.agent.llm import call_llm
            return _call_with_deadline(lambda: call_llm(
                model=(CONFIG.distill_escalation_model or CONFIG.model) if escalate else CONFIG.model,
                messages=msgs,
                tools=None,
                max_tokens=max_tokens,
                response_format=response_schema,
                openrouter_provider=None if escalate else CONFIG.openrouter_provider_distiller,
            ), deadline)

    response = _llm(
        messages,
        DistillerStructure if (two_pass or structure_only) else DistillerOutput,
    )

    if getattr(response, "finish_reason", None) == "length":
        # Even a prefix that still parses is a truncation: the tail is lost.
        cache_ok = False

    raw_output = response.text or ""
    if not raw_output.strip():
        return {"error": "Distiller returned empty response"}

    # #1: a truncated response (finish_reason == "length", or any malformed
    # trailing object) must not kill the whole batch. Try a clean parse first;
    # on failure, salvage every complete `updates` entry from the valid prefix.
    try:
        parsed, _ = parse_json(raw_output, strict=False)
    except Exception as e:
        salvaged = salvage_distiller_json(raw_output)
        if salvaged and salvaged.get("updates"):
            logger.warning(
                "Distiller output truncated/malformed (%s); salvaged %d complete "
                "update(s) from the valid prefix — batch continues with partial set",
                "length-limit" if response.finish_reason == "length" else "parse-error",
                len(salvaged["updates"]),
            )
            parsed = salvaged
            cache_ok = False
        else:
            return {"error": f"Distiller output JSON parse failed: {e}", "raw": raw_output[:500]}

    if not isinstance(parsed, dict) or "updates" not in parsed:
        return {"error": "Distiller output missing 'updates' key", "raw": raw_output[:500]}

    if two_pass and (body_ops := _body_ops(parsed)):
        follow_up = messages + [
            {"role": "assistant", "content": raw_output},
            {"role": "user", "content": _body_pass_instruction(body_ops)},
        ]
        try:
            second = _llm(follow_up, None)
            _stitch_bodies(parsed, second.text or "")
        except Exception as e:
            logger.warning(
                "body pass failed (%s) — %d op(s) left bodyless; the validate "
                "floor rejects them downstream, ephemerals survive",
                e, len(body_ops))
        if any(not op.get(_BODY_FIELD.get(str(op.get("op") or ""), "snippet"))
               for op in body_ops):
            # A missing ===SILICA-BODY N=== block left an op bodyless; the
            # floor would reject that op again on every replay.
            cache_ok = False

    logger.info("Distiller produced %d updates", len(parsed["updates"]))
    # Successes only, and complete ones only: error paths returned already,
    # and cache_ok gates out salvaged/truncated/bodyless/fallback replies.
    # cache_key doubles as the enabled() witness — it exists only under the
    # flag, so a mid-run flag flip cannot store under a key never computed.
    if cache_ns is not None and cache_key is not None and cache_ok:
        distill_cache.store(cache_ns, cache_key, parsed)
    return parsed
