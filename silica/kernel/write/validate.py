# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

import os
import logging
import re
import unicodedata
from pydantic import BaseModel
from silica.driver import DRIVER
from silica.kernel.write.ops import Op, OpType
from silica.kernel.write.templates import slugify
from silica.kernel.link.ast import extract_links

logger = logging.getLogger(__name__)


def same_file_key(path: str) -> str:
    """Identity key for "these two refs are the same file on disk".

    `os.path.abspath` alone is case- and composition-sensitive, but macOS and
    Windows are not: `Percettrone.md` and `percettrone.md` are ONE file there,
    so two ops surviving the dedup commit in sequence and the second silently
    overwrites the first — both reporting ok. NFC folds the other half of the
    same failure (macOS stores decomposed filenames).

    ponytail: over-merges on a case-sensitive filesystem where two notes really
    do differ only by case. That is a vault the rest of the pipeline already
    treats as one note (title_key casefolds, the linter matches case-insensitively),
    so collapsing is the consistent answer, not a regression.
    """
    return unicodedata.normalize("NFC", os.path.abspath(path)).casefold()


# Precision gate: a write op whose snippet is shorter than this is deferred
# instead of written — execute_write would otherwise fill the note with a
# "(to be expanded)" placeholder (real incident: run 5d0a3350, 2026-07-04, the
# distiller returned whole chunks with snippet="" despite full inbox excerpts).
# Rejection routes through the existing defer + steer path, so the distiller
# gets re-prompted with the reason.
#
# Raised 100→400 (audit 2026-07-23): a 130-220 char body is a meta-summary of
# the source section, not the section — the run's note-body length was bimodal
# (thin ≤220 vs rich ≥1300), and 400 routes the thin band to expand.
# Lowered 400→275 (2026-07-27) to match the value actually run against real
# vaults: 400 also deferred short-but-complete notes, which cost a round trip.
# ponytail: length is a proxy for "content, not a description of it" — a true
# shape check needs semantic judgment, not a regex; left as the worker-model
# lever. Build it only with an eval that catches what the floor misses (a
# description of content that still clears the char floor).
MIN_WRITE_SNIPPET_CHARS = 275


# Extractive bodies are verbatim-selected spans, grounded by construction; a
# short durable fact is real content, not the prose-placeholder the default
# floor guards against.
EXTRACTIVE_MIN_SNIPPET_CHARS = 40

# Profiles whose bodies are SELECTED verbatim, not authored: extractivity is
# enforced and the lower floor applies. "promotion" copies lines from an
# episodic chain render; same contract, different lens.
EXTRACTIVE_PROFILES = frozenset({"extractive", "promotion"})


def min_write_snippet_chars(profile: str | None = None) -> int:
    """Effective write-body floor for the run's profile, read at call time.
    Single source so the expand recovery uses the SAME floor the gate enforces
    — they used to diverge (expand cleared bodies at 100 while the gate could
    want more, so expanded notes stayed thin, audit finding 1).

    Two independent levers, one per body kind: SILICA_MIN_WRITE_SNIPPET_CHARS
    pins the AUTHORING floor (default profile), SILICA_EXTRACTIVE_MIN_SNIPPET_CHARS
    the verbatim-selection one. Measured live: a single shared env pin, set for
    authoring, reached a promotion run and rejected its verbatim body at
    '0 < 275'."""
    if profile in EXTRACTIVE_PROFILES:
        return int(os.getenv("SILICA_EXTRACTIVE_MIN_SNIPPET_CHARS",
                             str(EXTRACTIVE_MIN_SNIPPET_CHARS)))
    return int(os.getenv("SILICA_MIN_WRITE_SNIPPET_CHARS",
                         str(MIN_WRITE_SNIPPET_CHARS)))


# --- meta-description shape gate (audit 2026-07-23, finding 1) ---------------
# A small worker model writes bodies that ANNOUNCE the source section instead of
# delivering it: "Task: classificazione supervisionata... Include definizione
# formale e esempio di applicazione pratica." Length alone can't catch a 400+
# char announcement, so this detects announcement *shape*: content-noun
# announcements ("include definizione", "provides an overview"), deixis to the
# source artifact ("questa sezione", "la lezione"), or patch-style openers
# ("Estende la sezione su..."). Rejection routes through defer/steer/expand,
# which re-prompts with this reason — a false positive costs one retry, not the
# note. ponytail: lexical heuristic, IT+EN only; a semantic judge is the upgrade
# path if new marker families appear.

# Announcement verb followed (same clause) by an abstract content-noun.
_META_ANNOUNCE_RE = re.compile(
    r"\b(?:include|contiene|presenta|descrive|copre|tratta|riassume|illustra|fornisce"
    r"|includes?|contains?|presents?|describes?|covers?|summarizes?|outlines?|provides?)\b"
    r"[^.\n]{0,60}?"
    r"\b(?:definizion\w*|esemp\w*|panoramic\w*|descrizion\w*|spiegazion\w*|introduzion\w*"
    r"|dettagl\w*|informazion\w*|overview|definitions?|examples?|details?"
    r"|explanations?|introductions?)\b",
    re.IGNORECASE,
)
# Deixis to the source artifact itself (demonstratives, or the unambiguous
# section words — "il documento" is left out: too common as subject matter).
_META_DEICTIC_RE = re.compile(
    r"\b(?:questa|questo)\s+(?:sezione|nota|documento|capitolo)\b"
    r"|\b(?:la\s+sezione|il\s+capitolo|la\s+lezione)\b"
    r"|\bthis\s+(?:section|note|document|chapter)\b|\bthe\s+section\b",
    re.IGNORECASE,
)
_META_VERB_RE = re.compile(
    r"\b(?:include|contiene|presenta|descrive|copre|tratta|riassume|illustra|spiega"
    r"|covers?|describes?|contains?|includes?|presents?|summarizes?|outlines?|explains?)\b",
    re.IGNORECASE,
)
# Patch-style opener on a WRITE body: extends/adds something instead of being it.
_META_OPENER_RE = re.compile(
    r"^(?:estende|aggiunge|aggiunto|integra|amplia|extends?|adds?|added|updates?)\b",
    re.IGNORECASE,
)


def _has_content_evidence(body: str) -> bool:
    """Signals the body delivers material rather than announcing it: code,
    math, a real list, or multi-paragraph length."""
    if "```" in body or "$" in body:
        return True
    if sum(1 for l in body.splitlines() if l.lstrip().startswith(("-", "*", "+"))) >= 3:
        return True
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    return len(paragraphs) >= 3 or len(body) >= 800


def meta_description_reason(body: str) -> str | None:
    """Reason string when a write body reads as a meta-description of its
    source section instead of the section's content; None when it looks fine."""
    b = (body or "").strip()
    if not b or _has_content_evidence(b):
        return None
    markers = []
    if _META_ANNOUNCE_RE.search(b):
        markers.append("content announcement")
    if _META_OPENER_RE.match(b):
        markers.append("patch-style opener")
    if not markers and _META_DEICTIC_RE.search(b) and _META_VERB_RE.search(b):
        markers.append("source-artifact deixis")
    if not markers:
        return None
    return (
        f"body reads as a meta-description of the source ({', '.join(markers)}) "
        f"— write the section's actual content, not a summary of what it contains"
    )


class Rejection(BaseModel):
    op: Op
    reason: str


def validate_operations(
    ops: list[Op] | list[dict],
    payloads: list,
    target_dir: str,
    hub: str | None = None,
    cleared_parents_out: list | None = None,
    future_ref_whitelist: list[str] | None = None,
    cleared_links_out: list | None = None,
    ungrounded_out: list | None = None,
    profile: str | None = None,
    normalized_out: dict | None = None,
) -> tuple[list[Op], list[Rejection]]:
    """Validates operations against payloads and target_dir using DRIVER.

    ungrounded_out (optional): collects warn-only span-grounding hits —
    write/patch ops whose math/code spans can't be located in their source
    excerpt (fabrication candidates). Never causes a rejection.

    profile (optional): the RUN's distill profile. When set it decides the
    extractive enforcement and the snippet floor instead of the process-global
    resolution — a /promote run judges its ops as extractive while a
    concurrent /nucleate keeps the default gate.

    normalized_out (optional): counts the malformed-but-unambiguous ops this
    call repaired, keyed by pattern. A reject total says nothing an author can
    act on; "nine of ten repairs were a path rebuilt from the title" names the
    prompt rule to change. Repairs already happened and only reached a debug
    log — this is the same set, counted. Absent keys mean no repair of that
    kind, never a zero row.
    """
    from silica.config import CONFIG
    from silica.kernel.write.ops_io import parse_ops
    from silica.kernel.vault_manifest import active_write_dir, within
    from silica.onboarding.adopt import SAFE_WRITE_DIR
    ops_parsed = parse_ops(ops)
    ops = [op.model_copy(deep=True) for op in ops_parsed]

    # Vault write boundary (`write_dir` in vault.yaml); "" ⇒ the whole vault, so
    # a vault that declares nothing behaves exactly as before.
    write_root = active_write_dir()
    # Safe mode only: `silica/` mirrors the vault's own tree, so a subfolder
    # inside it is a claim that the vault has no folder for this note. The rebase
    # below repairs a FORGOTTEN prefix for free, but an INVENTED subtree is
    # indistinguishable from a correct one — hence the check further down.
    mirror = write_root == SAFE_WRITE_DIR
    vault_root = (getattr(CONFIG, "vault_path", "") or "").strip()

    # The run's landing folder follows the ops into the boundary. Without this
    # the rebase below moves every write into `write_root/` while target_dir
    # still names the folder outside it, and the "not in target folder" check
    # downstream then rejects the whole batch — the two were comparing paths
    # from different spaces. Reachable before safe mode (a code vault plus any
    # target_dir outside docs/silica); reachable on every prose vault after it.
    if write_root and target_dir and not os.path.isabs(target_dir) and not within(
        target_dir, write_root
    ):
        target_dir = write_root + "/" + target_dir.replace("\\", "/").strip("/")

    def _invented_folder(rel_path: str) -> str:
        """The mirror-relative parent of `rel_path` when no folder answers for it
        yet; "" when one does, when it is the mirror root, or when no vault.

        Two places answer, because the mirror is a preview of the vault after the
        paste: a folder the vault already has, and a folder an earlier run
        already created INSIDE the mirror (justified once, at creation — the
        gate is per folder, not per note, and the user's veto is declining to
        paste it).

        Presence check only — whether the reason is a *good* one is the prompt's
        job, not an LLM adjudication this seam has to stay deterministic without.
        """
        under = (rel_path or "").replace("\\", "/").strip("/").split("/", 1)
        parent = os.path.dirname(under[1]) if len(under) == 2 else ""
        if not parent or not vault_root:
            return ""
        if os.path.isdir(os.path.join(vault_root, parent)):
            return ""
        if os.path.isdir(os.path.join(vault_root, SAFE_WRITE_DIR, parent)):
            return ""
        return parent

    def _repaired(pattern: str) -> None:
        if normalized_out is not None:
            normalized_out[pattern] = normalized_out.get(pattern, 0) + 1

    # Sanitize filesystem-illegal characters (e.g. ':') from path filenames.
    # When a write op carries a `title` field, rebuild the path from title so
    # the note is filed under the clean concept name rather than the heading.
    for op in ops:
        if op.op == OpType.write and op.title and op.path and target_dir:
            clean_title = slugify(op.title)
            if clean_title:
                new_path = f"{target_dir.rstrip('/')}/{clean_title}.md"
                if new_path != op.path:
                    logger.debug("validate: title-derived path '%s' → '%s'", op.path, new_path)
                    _repaired("title_path")
                    op.path = new_path

        # A NEW note's location is Silica's own filing decision, so a write
        # aimed outside the boundary is rebased into it (structure preserved)
        # rather than rejected — target_dir and the model both predate the
        # vault's declaration. Ops that touch an EXISTING note outside a MIRROR
        # are never rebased: that would silently hit a different file. They get
        # rejected in the main loop below.
        if write_root and op.op == OpType.write and op.path and not within(op.path, write_root):
            rebased = write_root + "/" + op.path.replace("\\", "/").strip("/")
            logger.debug("validate: write_dir rebase '%s' → '%s'", op.path, rebased)
            _repaired("write_dir_rebase")
            op.path = rebased

        if op.path:
            folder, filename = os.path.split(op.path)
            name, ext = os.path.splitext(filename)
            # lstrip("."): a leading dot hides the note from every vault walker
            # (the calendar scan, the timeline and ignore_matcher all skip a
            # dot-prefixed part), so a model-supplied ".secret.md" would be
            # written, reported as a success, and never read back again.
            sanitized = (slugify(name).lstrip(".") or "note") + ext
            if sanitized != filename:
                new_path = (os.path.join(folder, sanitized) if folder else sanitized).replace("\\", "/")
                logger.debug("validate: sanitized path '%s' → '%s'", op.path, new_path)
                _repaired("path_slugify")
                op.path = new_path

    valid_concepts: dict[str, set[str]] = {}
    expected_collision_paths: dict[tuple[str, str], str | None] = {}
    concept_excerpts: dict[tuple[str, str], str] = {}
    collision_excerpts: dict[tuple[str, str], str] = {}
    source_files: dict[str, str] = {}
    inbox_folders = set()
    has_payloads = bool(payloads)

    # Index payloads
    if has_payloads:
        for payload_data in payloads:
            batches = payload_data.get("batches", [])
            for batch in batches:
                inbox_file = batch.get("inbox_file")
                if not inbox_file:
                    continue
                    
                source_basename = os.path.basename(inbox_file)
                inbox_dir = os.path.dirname(os.path.abspath(inbox_file))
                inbox_folders.add(inbox_dir)
                source_files[source_basename] = inbox_file

                if source_basename not in valid_concepts:
                    valid_concepts[source_basename] = set()
                    
                for c in batch.get("concepts", []):
                    name = c.get("name")
                    if not name:
                        continue
                    valid_concepts[source_basename].add(name)
                    concept_excerpts[(source_basename, name)] = c.get("inbox_excerpt", "") or ""

                    collision = c.get("vault_collision")
                    if collision and isinstance(collision, dict) and collision.get("path"):
                        expected_collision_paths[(source_basename, name)] = collision["path"]
                        collision_excerpts[(source_basename, name)] = collision.get("excerpt", "") or ""
                    else:
                        expected_collision_paths[(source_basename, name)] = None

    _existence_cache: dict[str, bool] = {}
    def path_exists(p: str) -> bool:
        norm = os.path.abspath(p)
        if norm not in _existence_cache:
            try:
                DRIVER.read_note(p)
                _existence_cache[norm] = True
            except RuntimeError:
                _existence_cache[norm] = False
        return _existence_cache[norm]

    # 1. Global deduplication (executed before coercion to ensure correct richest op type determination)
    path_groups: dict[str, list[Op]] = {}
    for op in ops:
        path = op.touched_ref()
        if path:
            norm_path = same_file_key(path)
            if norm_path not in path_groups:
                path_groups[norm_path] = []
            path_groups[norm_path].append(op)

    for norm_path, group in path_groups.items():
        if len(group) > 1:
            richest_op = max(group, key=lambda o: len(o.snippet or o.content or ""))
            has_write = any(o.op in (OpType.write, OpType.overwrite) for o in group)
            for op in group:
                if op is not richest_op:
                    op.op = OpType.skip
                    op.reason = f"Duplicate operation to the same path '{op.path}'"
            if has_write:
                # If there's an overwrite in the group, richest_op becomes overwrite
                if any(o.op == OpType.overwrite for o in group):
                    richest_op.op = OpType.overwrite
                    # overwrite reads op.content; a coerced write/patch carries its body
                    # in snippet — copy it so _execute_overwrite doesn't crash on content=None (A18)
                    richest_op.content = richest_op.content or richest_op.snippet
                else:
                    richest_op.op = OpType.write
                    # write reads op.snippet; a coerced overwrite carries its body in content (A18)
                    richest_op.snippet = richest_op.snippet or richest_op.content or ""

    # C3 title-identity gate: existing note titles in target_dir, keyed by
    # title_key, built lazily once per call. Empty on any driver failure —
    # the gate abstains, never blocks the pipeline.
    _title_gate_cache: dict[str, tuple[str, str]] | None = None  # key -> (title, path)
    _title_gate_list_cache: list[str] | None = None

    def _target_dir_titles() -> dict[str, tuple[str, str]]:
        nonlocal _title_gate_cache
        if _title_gate_cache is not None:
            return _title_gate_cache
        from silica.kernel.text.title import title_key
        out: dict[str, tuple[str, str]] = {}
        try:
            norm_dir = (target_dir or "").replace("\\", "/").strip("/")
            for ref in DRIVER.list_files(norm_dir):
                ref_dir = os.path.dirname((ref.path or "").replace("\\", "/")).strip("/")
                if ref_dir != norm_dir:
                    continue
                key = title_key(ref.name)
                if key:
                    out[key] = (ref.name, ref.path)
        except Exception as e:
            logger.debug("validate: title gate enumeration failed (abstaining): %s", e)
        _title_gate_cache = out
        return out

    def _target_dir_title_list() -> list[str]:
        nonlocal _title_gate_list_cache
        if _title_gate_list_cache is None:
            _title_gate_list_cache = [t for (t, _p) in _target_dir_titles().values()]
        return _title_gate_list_cache

    # 2. Coerce write <-> patch and enforce default hub fallback
    if not hub and target_dir:
        hub = os.path.basename(target_dir.rstrip("/\\"))

    for op in ops:
        if op.op == OpType.skip:
            continue
        if op.op in (OpType.write, OpType.patch, OpType.overwrite) and hub:
            op.hub = hub

        if op.op == OpType.write and op.path and path_exists(op.path):
            op.op = OpType.patch
        elif op.op == OpType.write and op.path:
            # C3 gate, band 1: a title key-equal to an existing note in the
            # target folder is the SAME note under a cosmetic variant
            # («Machine Learning (9 CFU)») — mechanical coercion to patch,
            # extending the exact-path coercion above.
            from silica.kernel.text.title import title_key
            stem = os.path.splitext(os.path.basename(op.path))[0]
            match = _target_dir_titles().get(title_key(stem))
            if match is not None:
                logger.info(
                    "validate: title '%s' key-equal to existing '%s' — coercing write→patch",
                    stem, match[0],
                )
                op.op = OpType.patch
                op.path = match[1] if match[1].endswith(".md") else f"{match[1]}.md"
        elif op.op == OpType.patch and op.path and not path_exists(op.path):
            if has_payloads:
                expected_path = expected_collision_paths.get((op.source_basename, op.heading))
                if not expected_path or os.path.abspath(op.path) == os.path.abspath(expected_path):
                    op.op = OpType.write
            else:
                op.op = OpType.write

    # 2b. Band-1 title-key coercion above can retarget two different-path writes
    # onto the SAME existing note; the step-1 path dedup ran before coercion and
    # cannot see it. Re-dedup by post-coercion path so two ops never target one
    # note (also removes the duplicate touched_ref that would let write.py's
    # failed-op lookup defer one op twice and drop the other — audit A21/A4).
    coerced_groups: dict[str, list[Op]] = {}
    for op in ops:
        if op.op == OpType.skip:
            continue
        ref = op.touched_ref()
        if ref:
            coerced_groups.setdefault(same_file_key(ref), []).append(op)
    for _ref_path, group in coerced_groups.items():
        if len(group) > 1:
            keep = max(group, key=lambda o: len(o.snippet or o.content or ""))
            for op in group:
                if op is not keep:
                    op.op = OpType.skip
                    op.reason = f"Duplicate operation to the same path '{op.path}' after coercion"

    # Pre-compute note stems created in this run so parent validation can allow
    # forward references to notes being written in the same batch.
    _run_write_stems: set[str] = {
        os.path.splitext(os.path.basename(op.path))[0].lower()
        for op in ops
        if op.op in (OpType.write, OpType.overwrite) and op.path
    }

    def _resolve_parent(op: Op, cleared_out: list | None = None) -> None:
        """Neutralise an unresolvable parent — fall back to hub, no Rejection.

        If cleared_out is provided, records the cleared reference as a forward-link
        hint so the distiller can anticipate the note in future iterations.
        """
        if not op.parent:
            return
        p_key = op.parent.lower()
        if p_key in _run_write_stems:
            return
        matches = DRIVER.search_names(op.parent)
        if not any(r.name.lower() == p_key for r in matches):
            logger.warning(
                "validate: parent '%s' not found in vault or current run — clearing to hub fallback",
                op.parent,
            )
            if cleared_out is not None:
                cleared_out.append({
                    "cleared_parent": op.parent,
                    "note_heading": op.heading or "",
                    "note_path": op.path or "",
                })
            op.parent = None

    _source_text_cache: dict[str, str] = {}

    def _full_source_text(basename: str) -> str:
        """The whole source document, "" when unreadable. A finished source is
        archived at CLEANUP and RETRY/steer validates after the move — so look
        there too, at the path `archive_path_for` would have produced. The flat
        `<archive>/<basename>` stays last in line: sources archived before
        2026-08-23 kept no folder structure and still have to be readable."""
        if basename not in _source_text_cache:
            from silica.kernel.vault_manifest import active_done_dir, archive_path_for

            text = ""
            inbox_path = source_files.get(basename)
            candidates = [
                inbox_path,
                archive_path_for(inbox_path) if inbox_path else "",
                f"{active_done_dir()}/{basename}",
            ]
            for cand in candidates:
                if not cand:
                    continue
                try:
                    text = DRIVER.read_note(cand).content
                    break
                except Exception:
                    continue
            _source_text_cache[basename] = text
        return _source_text_cache[basename]

    def _check_grounding(op: Op) -> None:
        """Warn-only verbatim gate (never rejects): math/code spans in the body
        that can't be located in the source excerpt are fabrication candidates."""
        # Ground against everything the distiller legitimately saw for this
        # concept: inbox excerpt + colliding vault note excerpt — a patch
        # restating a vault formula for coherence is not fabrication.
        excerpt = concept_excerpts.get((op.source_basename, op.heading), "")
        collision = collision_excerpts.get((op.source_basename, op.heading), "")
        source_text = f"{excerpt}\n{collision}" if collision else excerpt
        body = op.snippet or op.content or ""
        if not source_text.strip() or not body:
            return
        from silica.kernel.write.provenance import ungrounded_spans
        spans = ungrounded_spans(body, source_text)
        if spans:
            # Second look against the whole document: the distiller (and the
            # steer model, which sees the entire bundle) legitimately stitches
            # from neighboring windows, and a span verbatim elsewhere in the
            # source is not fabrication (IDK.md, 2026-08-05: 3/3 flagged spans
            # were verbatim in the file, outside the concept's window). Only
            # ever clears spans — a class the excerpt didn't gate stays ungated.
            full = _full_source_text(op.source_basename)
            if full:
                still = set(ungrounded_spans(body, f"{source_text}\n{full}"))
                spans = [s for s in spans if s in still]
        if spans:
            logger.warning(
                "validate: '%s' — %d verbatim span(s) not grounded in source excerpt: %s",
                op.path, len(spans), " | ".join(s[:60] for s in spans),
            )
            if ungrounded_out is not None:
                ungrounded_out.append({
                    "path": op.path,
                    "heading": op.heading,
                    "source_basename": op.source_basename,
                    "spans": spans,
                })

    from silica.kernel.prep_delegation import active_distill_profile

    _profile = profile or active_distill_profile()
    _extract_enforce = (os.getenv("SILICA_EXTRACTIVE_ENFORCE") == "1"
                        or _profile in EXTRACTIVE_PROFILES)

    def _extractive_reject(op: Op) -> str | None:
        """Under the extractive distill profile a write/patch body must be
        SELECTED verbatim from the source, not rewritten. Returns a rejection
        reason when it isn't — routing the op through the normal defer/steer
        retry so a persistent violator becomes a declared hole, never silent
        loss. On when the extractive profile is active (the invariant is the
        profile's contract, one lever) or under SILICA_EXTRACTIVE_ENFORCE=1;
        otherwise off — the default distiller paraphrases legitimately, so
        this must never fire outside such a run."""
        if not _extract_enforce:
            return None
        excerpt = concept_excerpts.get((op.source_basename, op.heading), "")
        collision = collision_excerpts.get((op.source_basename, op.heading), "")
        source_text = f"{excerpt}\n{collision}" if collision else excerpt
        body = op.snippet or op.content or ""
        if not source_text.strip() or not body.strip():
            return None
        from silica.kernel.write.provenance import nonextractive_lines
        bad = nonextractive_lines(body, source_text)
        if bad:
            return ("extractive: %d body line(s) not verbatim from source: %s"
                    % (len(bad), " | ".join(s[:60] for s in bad[:3])))
        return None

    validated_ops = []
    rejected_ops = []

    target_dir_abs = os.path.abspath(target_dir) if target_dir else ""

    def _is_within_dir(path_abs: str, dir_abs: str) -> bool:
        if not dir_abs:
            return True
        try:
            return os.path.commonpath([path_abs, dir_abs]) == dir_abs
        except ValueError:
            return False

    # Safe mode: which invented folders this batch has already accounted for.
    # Resolved up front so the gate is order-independent — with a per-op check
    # five notes into one new folder rejected four of them, and WHICH four
    # depended on where the model happened to put the sentence.
    justified_folders: set[str] = set()
    if mirror:
        # The run's landing folder is the user's own filing decision (typed at
        # /nucleate, folded onto the tree by resolve_target_dir), so it never
        # owes the model's justification. Exact match, not `within`: a subtree
        # improvised UNDER it is still the model's invention.
        landing = (target_dir or "").replace("\\", "/").strip("/")
        if landing and not os.path.isabs(target_dir or ""):
            head, _, tail = landing.partition("/")
            if head.lower() == SAFE_WRITE_DIR:
                landing = tail  # the user typed the mirror path themselves
            if landing:
                justified_folders.add(landing.lower())
        for op in ops:
            if op.op == OpType.write and op.path and (op.reason or "").strip():
                stated = _invented_folder(op.path)
                if stated:
                    justified_folders.add(stated.lower())

    for op in ops:
        heading = op.heading
        op_type = op.op
        source_basename = op.source_basename
        path = op.path

        # Skip ops are no-ops (dedup/axis demotion keeps the original path on
        # them) — rejecting one inflates the rejection rate past its
        # actionable-only denominator, so short-circuit before any check.
        if op_type == OpType.skip:
            continue

        # Vault write boundary (`write_dir` in vault.yaml). Distinct from
        # target_dir, which is a per-run landing folder: this one is per-vault,
        # applies to every op type, and protects files Silica reads but must not
        # author (a source tree, someone's hand-written notes). "" ⇒ whole vault,
        # so a vault that declares nothing is unaffected.
        # Safe mode exempts patches: they are judged against the REAL note they
        # enrich (collision path, existence) and only then rebased onto its
        # mirror copy, at the bottom of the patch branch. Rejecting them here
        # left safe mode able to create and never to enrich, which is most of
        # what an ingest does — 11 of 20 deferred ops in one real run, and 6
        # patches on the same lecture without safe mode against 1 with.
        mirror_patch = mirror and op_type in (OpType.patch, OpType.overwrite)
        if write_root and not mirror_patch:
            touched = [op.touched_ref()]
            if op_type == OpType.move:
                touched.append(op.to_path)  # a move must not leave the boundary either
            outside = [t for t in touched if t and not within(t, write_root)]
            if outside:
                rejected_ops.append(Rejection(
                    op=op,
                    reason=f"outside the vault write boundary '{write_root}/': {outside[0]}",
                ))
                continue

        # Safe mode: a new note may land in any folder that already answers for
        # it without saying anything. A folder nobody has claimed yet is a
        # filing decision the model made alone — it costs one sentence to state
        # it, and the steering loop lets it retry with one.
        if mirror and op_type == OpType.write and path:
            invented = _invented_folder(path)
            if invented and invented.lower() not in justified_folders:
                rejected_ops.append(Rejection(
                    op=op,
                    reason=(
                        f"'{invented}/' does not exist in the vault — file this "
                        f"under a folder that does, or set \"reason\" to state "
                        f"why no existing folder fits"
                    ),
                ))
                continue

        if has_payloads:
            if not heading:
                rejected_ops.append(Rejection(op=op, reason="Missing 'heading' field"))
                continue
            if not source_basename:
                rejected_ops.append(Rejection(op=op, reason="Missing 'source_basename' field"))
                continue
            if source_basename not in valid_concepts:
                if len(valid_concepts) == 1:
                    # Unambiguous: exactly one source file in this chunk, so a
                    # mismatched source_basename can only mean that one —
                    # coerce instead of rejecting (mirrors the heading
                    # near-match fallback below, same unique-candidate logic).
                    (only_basename,) = valid_concepts.keys()
                    logger.info(
                        "validate: source_basename '%s' coerced to only candidate '%s'",
                        source_basename, only_basename,
                    )
                    op.source_basename = source_basename = only_basename
                else:
                    rejected_ops.append(Rejection(op=op, reason=f"Unknown source_basename '{source_basename}'"))
                    continue
            if heading not in valid_concepts[source_basename]:
                # The distiller re-cases names and swaps typographic for
                # straight apostrophes — remap a normalized unique match to
                # the canonical payload name instead of rejecting.
                def _heading_key(s: str) -> str:
                    return " ".join(s.replace("’", "'").casefold().split())

                near = [
                    n for n in valid_concepts[source_basename]
                    if _heading_key(n) == _heading_key(heading)
                ]
                if len(near) != 1:
                    # Fallback: the op title is often a cleaner rephrase of the
                    # registered concept ("DataFrame" vs "dataframe spark").
                    # Accept only a UNIQUE token-subset match (one side's word
                    # set contained in the other) — ambiguous / 1-to-many still
                    # rejects, so content can't attach to the wrong concept.
                    ht = frozenset(_heading_key(heading).split())
                    if ht:
                        subset = [
                            n for n in valid_concepts[source_basename]
                            if (nt := frozenset(_heading_key(n).split())) and (ht <= nt or nt <= ht)
                        ]
                        if len(subset) == 1:
                            near = subset
                if len(near) != 1:
                    # Fallback: same idea, one more step of normalization for
                    # morphological variants a raw-token subset can't see
                    # ("Stima parametrica" vs "Stima dei parametri" — no shared
                    # token, same Snowball stem). Same unique-match-only
                    # discipline as above.
                    from silica.config import CONFIG
                    from silica.kernel.text.text import stem_word

                    lang = getattr(CONFIG, "cooccurrence_lang", "auto")
                    hs = frozenset(
                        stem_word(t, lang=lang) for t in _heading_key(heading).split()
                    )
                    if hs:
                        stemmed = [
                            n for n in valid_concepts[source_basename]
                            if (ns := frozenset(
                                stem_word(t, lang=lang) for t in _heading_key(n).split()
                            )) and (hs <= ns or ns <= hs)
                        ]
                        if len(stemmed) == 1:
                            near = stemmed
                if len(near) != 1:
                    rejected_ops.append(Rejection(op=op, reason=f"Heading '{heading}' not present in payload concepts"))
                    continue
                logger.info("validate: heading '%s' normalized to payload concept '%s'", heading, near[0])
                op.heading = heading = near[0]

        if path:
            path_abs = os.path.abspath(path)
            # Guard the whole configured inbox tree, not just the folders of the
            # current run — a patch aimed at a *different* Inbox subfolder used
            # to slip through validate and die (or land) at WRITE.
            from silica.kernel.recall.paths import is_inbox_path
            forbidden = (
                is_inbox_path(path)
                or any(_is_within_dir(path_abs, folder) for folder in inbox_folders)
            )
            if "/0 Inbox/" in path or "/0 inbox/" in path.lower() or forbidden:
                rejected_ops.append(Rejection(op=op, reason=f"Target path '{path}' contains forbidden inbox segment"))
                continue

        if op_type == OpType.patch:
            if not path:
                rejected_ops.append(Rejection(op=op, reason="Missing 'path' field for patch operation"))
                continue

            # Safe mode: the model reads TARGET in mirror space and hands the
            # collision back mirror-prefixed. Undo the prefix so every check
            # below judges the note actually enriched, symmetric with the rebase
            # that re-applies it at the bottom of this branch. Only an exact
            # mirror image of the path the payload named is repaired — a patch
            # aimed at a different note stays a rejection.
            if mirror_patch and write_root and within(path, write_root):
                expected = expected_collision_paths.get((source_basename, heading))
                unmirrored = path.replace("\\", "/").strip("/")[len(write_root) + 1:]
                if expected and unmirrored == expected.replace("\\", "/").strip("/"):
                    logger.debug("validate: mirror patch un-rebase '%s' → '%s'",
                                 path, unmirrored)
                    _repaired("mirror_patch_unrebase")
                    op.path = path = unmirrored

            path_abs = os.path.abspath(path)
            if has_payloads:
                expected_path = expected_collision_paths.get((source_basename, heading))
                if expected_path:
                    if path_abs != os.path.abspath(expected_path):
                        rejected_ops.append(Rejection(op=op, reason=f"Path '{path}' does not match expected collision '{expected_path}'"))
                        continue
                else:
                    if not _is_within_dir(path_abs, target_dir_abs):
                        rejected_ops.append(Rejection(op=op, reason=f"Coerced patch path '{path}' not in target folder"))
                        continue
            elif not _is_within_dir(path_abs, target_dir_abs):
                rejected_ops.append(Rejection(op=op, reason=f"Path '{path}' not in target folder"))
                continue

            if not path_exists(path):
                rejected_ops.append(Rejection(op=op, reason=f"Collision path '{path}' does not exist in vault"))
                continue

            _resolve_parent(op, cleared_parents_out)
            _check_grounding(op)
            _reason = _extractive_reject(op)
            if _reason:
                rejected_ops.append(Rejection(op=op, reason=_reason))
                continue
            # Everything above judged the op against the note it actually
            # enriches. Only the landing moves: under safe mode the append goes
            # to the mirror copy (`_execute_patch` seeds it from the original),
            # so the vault note itself is untouched until the user pastes.
            if mirror_patch and not within(path, write_root):
                rebased = write_root + "/" + path.replace("\\", "/").strip("/")
                logger.debug("validate: mirror patch rebase '%s' → '%s'", path, rebased)
                _repaired("mirror_patch_rebase")
                op.path = rebased
            validated_ops.append(op)

        elif op_type == OpType.write:
            if not path:
                rejected_ops.append(Rejection(op=op, reason="Missing 'path' field for write operation"))
                continue

            path_abs = os.path.abspath(path)
            if not _is_within_dir(path_abs, target_dir_abs):
                rejected_ops.append(Rejection(op=op, reason=f"Path '{path}' not in target folder"))
                continue

            if path_exists(path):
                rejected_ops.append(Rejection(op=op, reason=f"Target path '{path}' already exists (should be patch/overwrite)"))
                continue

            # C3 gate, band 2: fuzzy-near an existing title (Descriptor vs
            # Description) → defer to the review queue so the dedup judge
            # decides — never a hard block, never a silent fourth duplicate.
            from silica.kernel.text.title import near_titles
            stem = os.path.splitext(os.path.basename(path))[0]
            near_hits = near_titles(stem, _target_dir_title_list())
            if near_hits:
                cand_title, ratio = near_hits[0]
                cand_path = next(
                    (p for (t, p) in _target_dir_titles().values() if t == cand_title), ""
                )
                rejected_ops.append(Rejection(
                    op=op,
                    reason=(
                        f"near_title candidate='{cand_title}' path='{cand_path}' "
                        f"ratio={ratio:.2f} — deferred for dedup review"
                    ),
                ))
                continue

            body_len = len((op.snippet or "").strip())
            if body_len == 0 and has_payloads:
                # Distinguish two ways a write lands with an empty body:
                #  (a) the source excerpt itself is empty — the concept was only
                #      *mentioned*, never defined. Nothing to distill or expand;
                #      deferring only churns (and a whole chunk of these drives the
                #      rejection rate to 100% and aborts the run). Skip it as a
                #      forward-reference — it stays linked from the notes that
                #      mention it, to be authored when a later source defines it.
                #  (b) the excerpt HAD content but the distiller dropped the body
                #      (run 5d0a3350 regression) — that must still be rejected so
                #      the expand arc retries it. Falls through below.
                excerpt = concept_excerpts.get((source_basename, heading))
                if excerpt is not None and not excerpt.strip():
                    logger.info(
                        "validate: write '%s' — empty source excerpt, skipped as a "
                        "forward-reference (nothing to distill)", op.path,
                    )
                    continue
            # Floor read at call time (not import) so an arm can lower it via env
            # without import-order fragility. Extractive selects verbatim spans, and
            # a durable fact can live in a legitimately short turn — a 60-char
            # verbatim fact is real content, not the prose-placeholder this gate
            # guards against — so the extractive arm sets a lower floor.
            _min_snippet = min_write_snippet_chars(_profile)
            if body_len < _min_snippet:
                rejected_ops.append(Rejection(
                    op=op,
                    reason=(
                        f"snippet too short ({body_len} < {_min_snippet} chars) "
                        f"— would write a placeholder note, deferred for retry"
                    ),
                ))
                continue

            # Shape gate: a long-enough body can still be an announcement of the
            # section instead of the section. Skipped under extractive enforce
            # (verbatim-selected content is grounded by construction, and the
            # source's own wording must not be judged); env kill-switch for
            # eval arms, same pattern as the length floor.
            if not _extract_enforce and os.getenv("SILICA_META_SHAPE_CHECK", "1") != "0":
                _shape_reason = meta_description_reason(op.snippet or "")
                if _shape_reason:
                    rejected_ops.append(Rejection(op=op, reason=_shape_reason))
                    continue

            _resolve_parent(op, cleared_parents_out)
            _check_grounding(op)
            _reason = _extractive_reject(op)
            if _reason:
                rejected_ops.append(Rejection(op=op, reason=_reason))
                continue
            validated_ops.append(op)

        elif op_type == OpType.overwrite:
            if not path:
                rejected_ops.append(Rejection(op=op, reason="Missing 'path' field for overwrite operation"))
                continue

            path_abs = os.path.abspath(path)
            if not _is_within_dir(path_abs, target_dir_abs):
                rejected_ops.append(Rejection(op=op, reason=f"Path '{path}' not in target folder"))
                continue

            if not path_exists(path):
                # If target note doesn't exist, overwrite degrades to write gracefully.
                # overwrite body lives in op.content; _execute_write reads op.snippet,
                # so carry it over or the degraded write persists an empty note (A17).
                op.op = OpType.write
                op.snippet = op.snippet or op.content or ""
            elif op.base_content is None:
                # LAST-RESORT fallback, deliberately weak: it only guards the
                # validate->execute window. An edit that landed while the
                # producer computed `content` is already on disk here and
                # becomes the base itself — no conflict detected, edit stomped.
                # Producers must snapshot base_content at READ time (refine
                # and enrich do); this exists only for future producers that
                # forget, so their overwrites are not entirely unguarded.
                op.base_content = DRIVER.read_note(path).content

            _resolve_parent(op, cleared_parents_out)
            validated_ops.append(op)

        else:
            rejected_ops.append(Rejection(op=op, reason=f"Unknown operation type '{op_type}'"))

    # 3. Auto-create missing Hub notes
    hubs_to_check = set()
    for op in validated_ops:
        op_type = op.op
        if op_type in (OpType.write, OpType.patch, OpType.overwrite):
            hub = op.hub
            if hub:
                clean_hub = hub.strip("[]")
                if clean_hub:
                    hubs_to_check.add(clean_hub)

    hub_ops = []
    for hub in sorted(hubs_to_check):
        if not path_exists(hub):
            hub_filename = f"{hub}.md"
            hub_path = os.path.join(target_dir, hub_filename).replace("\\", "/")
            
            already_creating = any(
                (o.op == OpType.write and o.heading == hub) or
                (o.path and same_file_key(o.path) == same_file_key(hub_path))
                for o in validated_ops
            )
            
            if not already_creating:
                source_basename = "auto_generated"
                if validated_ops:
                    source_basename = validated_ops[0].source_basename or "auto_generated"
                
                hub_op = Op(
                    op=OpType.write,
                    heading=hub,
                    path=hub_path,
                    snippet="Hub automatically generated by the Injector pipeline.",
                    hub=hub,
                    source_basename=source_basename
                )
                hub_ops.append(hub_op)
                logger.info("Validation: hub '%s' does not exist. Injected creation operation at %s", hub, hub_path)

    if hub_ops:
        validated_ops = hub_ops + validated_ops

    # 4. Prospective link check: surface wikilinks introduced by write/patch/overwrite
    # ops that cannot be resolved in the current vault, within this batch, or via the
    # future_ref_whitelist.  Unlike parents (see _resolve_parent), an unresolved inline
    # link does NOT reject the op — it is kept verbatim as a dangling forward-reference
    # and recorded in cleared_links_out, symmetric with cleared_parents.  This mirrors
    # Obsidian semantics (dangling links are first-class) and prevents a self-referential
    # source from losing whole notes to the rejection-rate gate.
    # Any op that leaves a note at op.path after this batch (write/patch/overwrite) is a
    # valid in-batch link target — not just freshly-written notes.
    batch_created_names: set[str] = {
        os.path.splitext(os.path.basename(op.path))[0].lower()
        for op in validated_ops
        if op.op in (OpType.write, OpType.patch, OpType.overwrite) and op.path
    }

    _link_resolve_cache: dict[str, bool] = {}
    whitelist_lower = {w.lower() for w in (future_ref_whitelist or [])}

    def _link_resolves(target: str) -> bool:
        stem = target.removesuffix(".md")
        key = stem.lower()
        if key in _link_resolve_cache:
            return _link_resolve_cache[key]
        if key in batch_created_names:
            _link_resolve_cache[key] = True
            return True
        if key in whitelist_lower:
            _link_resolve_cache[key] = True
            return True
        if "/" in stem:
            result = path_exists(stem + ".md") or path_exists(stem)
        else:
            matches = DRIVER.search_names(stem)
            result = any(r.name.lower() == key for r in matches)
        _link_resolve_cache[key] = result
        return result

    prospective_valid: list[Op] = []
    for op in validated_ops:
        if op.op not in (OpType.write, OpType.patch, OpType.overwrite):
            prospective_valid.append(op)
            continue
        text = op.snippet or op.content or ""
        if not text:
            prospective_valid.append(op)
            continue
        links = extract_links(text)
        broken = [lnk for lnk in links if not _link_resolves(lnk)]
        if broken:
            logger.debug(
                "validate: %d unresolved wikilink(s) kept as forward-ref in '%s': %r",
                len(broken), op.path or op.heading or "?", broken,
            )
            if cleared_links_out is not None:
                for lnk in broken:
                    cleared_links_out.append({
                        "cleared_link": lnk,
                        "note_heading": op.heading or "",
                        "note_path": op.path or "",
                    })
        prospective_valid.append(op)
    validated_ops = prospective_valid

    return validated_ops, rejected_ops
