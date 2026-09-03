# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Note<->source provenance ledger (spec-hermes-coherence §3).

Append-only record of which notes derive from which version (sha256) of an
nucleated source file, keyed by source basename. Written at CLEANUP alongside
archiving (silica.router.states.finalize, sibling to _log_nucleate_completion)
and read by graph_report (source drift section) and /nucleate (re-nucleate of a
modified source warning).

Storage: `<vault_path>/provenance.json` — a JSON array of records:
    {"source": "lecture-03.md", "sha256": "…", "run_id": "…",
     "date": "2026-07-02", "notes": ["Concepts/Note A", "Concepts/Note B"]}
`notes` entries are vault-relative note paths without the `.md` extension
(RunManifestEntry.path, which strips it) — NOT the same form as graph_report
node ids, which carry `.md` (driver index keys). Callers that intersect the
two (e.g. graph_report's source-drift section) must strip `.md` at the seam.

No hash lives in note frontmatter (user-facing noise) — provenance lives
only here, ledger-side.

Kernel-only: no router/capabilities imports (import-linter boundary).
Additive: a missing/unreadable/unwritable store degrades to "no records"
everywhere — nothing here ever raises out to its caller.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal

from silica.kernel.recall.paths import resolve_vault_path

logger = logging.getLogger(__name__)

DEFAULT_PROVENANCE_FILENAME = "provenance.json"

# What an append can prove about the ledger once it returns. `present`: the
# record with these notes is on disk. `absent`: nothing of it is (a clean
# failure before the replace, or no store to write to). `uncertain`: the store
# could not be read back, so neither claim holds. Kept apart because folding
# the last two into "not written" is how a failed append stayed invisible
# until the next nucleate of the source re-appended into the notes it had
# already written: note_authored_by reads this ledger to stop exactly that.
AppendOutcome = Literal["present", "absent", "uncertain"]


def _store_path(vault_path: str | None) -> Path | None:
    """`<vault>/<write_dir>/provenance.json` — Silica's own ledger, so it lives
    inside the write boundary. Reads fall back to a root file written before
    this: pointing Silica at an existing library used to leave both this and
    `log.md` in the user's root, beside their own README.
    """
    resolved = resolve_vault_path(vault_path)
    if not resolved:
        return None
    root = Path(resolved)
    try:
        from silica.kernel.vault_manifest import in_write_dir

        composed = root / in_write_dir(DEFAULT_PROVENANCE_FILENAME)
    except Exception as exc:
        logger.debug("provenance: write-dir compose failed (non-fatal): %s", exc)
        return root / DEFAULT_PROVENANCE_FILENAME
    legacy = root / DEFAULT_PROVENANCE_FILENAME
    if composed != legacy and not composed.exists() and legacy.exists():
        return legacy
    return composed


def source_event_date(source_text: str, seen_override: str | None = None) -> str | None:
    """Event clock for a source: when the thing it describes happened; None
    when it states none.

    Distinct from the ingest clock (this store's `date`, the day we read the
    file). Precedence: capture clock, then the source's own `date:` — never
    today. An undated source stays undated on purpose: on the write path None
    emits no claim stamp at all (an ingest-dated stamp is noise on the event
    axis — measured on a real vault: 107 stamps, one distinct day), and on the
    verdict path "undated" and "today" are opposite evidence when the question
    is whether an incoming claim is fresher than the note it contradicts.
    Callers that must agree share it: the source-leaf writer and the claim
    stamp the FSM puts on every note it writes, or a claim and its leaf would
    date the same event differently.
    """
    if seen_override:
        return str(seen_override)
    try:
        from silica.kernel.write import frontmatter

        data, _raw, _body = frontmatter.split(source_text or "")
        date = (data or {}).get("date")
        if date:
            return str(date)
    except Exception:
        pass
    return None


# Single-entry memo keyed on (path, mtime_ns, size) — a big re-ingest
# calls this once per patch op (bulk._execute_patch → note_authored_by) and would
# otherwise re-parse the whole ledger every time. Self-invalidating: any writer,
# in-process or not, moves mtime/size. An LRU would only matter if two vaults
# ever interleaved reads in one process, which the one-vault-per-process
# invariant rules out.
_records_memo: tuple[tuple[str, int, int], list[dict[str, Any]]] | None = None


def read_records(
    source: str | None = None,
    *,
    vault_path: str | None = None,
) -> list[dict[str, Any]]:
    """All provenance records, optionally filtered to one source basename.

    Missing store or unreadable file degrade to [] (additive: absence of
    the store must look like today's behaviour). Corrupt content is
    quarantined first (*.corrupt.<stamp>, surfaced by doctor): this ledger
    is authoritative — run_id/sha history is not reconstructible from the
    vault — and a later append_record would otherwise clobber the corrupt
    bytes with a fresh array.

    The returned list is always the caller's own: append_record mutates what it
    gets back, and the memo below must not see that.
    """
    global _records_memo

    path = _store_path(vault_path)
    if not path or not path.exists():
        return []
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None

    records: list[dict[str, Any]] | None = None
    if key is not None and _records_memo is not None and _records_memo[0] == key:
        records = _records_memo[1]

    if records is None:
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError(f"expected JSON array, got {type(records).__name__}")
        except Exception as exc:
            from silica.kernel.recall.paths import quarantine

            dest = quarantine(path)
            logger.warning("provenance: corrupt store quarantined to %s: %s", dest or path, exc)
            return []
        if key is not None:
            _records_memo = (key, records)

    if source is not None:
        return [r for r in records if isinstance(r, dict) and r.get("source") == source]
    return list(records)


# Op kinds that DERIVE a note from the source, i.e. what the ledger records.
# Named once because the three call sites (CLEANUP's manifest, the sub-agent
# worker lane, the anneal recovery) each spelled it inline and disagreed: the
# worker lane listed only write/patch, and `overwrite` is the ONLY op four
# sub-agent profiles in agent/bounds.py are allowed to emit — so the majority of
# that lane's notes reached the vault with no record at all.
DERIVING_OPS = frozenset({"write", "patch", "overwrite"})


def is_deriving_op(op: Any) -> bool:
    """True for an op kind that derives a note from its source. Accepts the
    OpType enum or its `.value`, so both lanes ask the identical question."""
    return getattr(op, "value", op) in DERIVING_OPS


def _observed(path: Path, source: str, sha256: str, run_id: str,
              notes: list[str]) -> AppendOutcome:
    """What the store holds for this triple, read raw after a failed append.

    Raw, not through read_records: its quarantine would move the very bytes
    whose state is the question, and its memo could answer from before the
    write. An unreadable or unparseable store is `uncertain`, never a proven
    absence.
    """
    try:
        if not path.exists():
            return "absent"
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            return "uncertain"
    except Exception:
        return "uncertain"
    wanted = set(notes)
    for r in records:
        if (isinstance(r, dict) and r.get("source") == source
                and r.get("sha256") == sha256 and r.get("run_id") == run_id
                and wanted <= set(r.get("notes") or [])):
            return "present"
    return "absent"


def append_record(
    source: str,
    sha256: str,
    run_id: str,
    notes: list[str],
    *,
    vault_path: str | None = None,
    date: str | None = None,
    partial: bool = False,
) -> AppendOutcome:
    """Append one record for `source` and report what is on disk afterwards.

    `partial`: the run lost a chunk. The record still lists what landed (those
    notes derive from this version), but it must not read as "this version is
    in the vault": a partial run that wrote two COLLISION notes made the next
    /nucleate skip the whole segment as already distilled (2026-09-02).

    Never raises (CLEANUP must never fail on this), but never hides a failure
    either: anything short of `present` is logged at WARNING with what it
    costs, because the note is already in the vault and a ledger that does
    not know it will let the next nucleate of this source patch its own
    concepts into the notes it already wrote.

    Idempotent on (source, sha256, run_id): a resumed run re-entering
    CLEANUP for the same file fires this again with an unchanged triple —
    that must not duplicate the record (mirrors run_log.append_log_line's
    resume-safety), and it answers `present` because the record is.
    """
    path = _store_path(vault_path)
    if not path:
        return "absent"

    record = {
        "source": source,
        "sha256": sha256,
        "run_id": run_id,
        "date": date or datetime.now().strftime("%Y-%m-%d"),
        "notes": list(notes),
        **({"partial": True} if partial else {}),
    }

    try:
        # The write is atomic but read→append→replace is not: two concurrent
        # appends (parallel nucleate workers) would each rewrite the whole
        # ledger from their own read and the last one wins, silently dropping
        # the other's record. The lease serializes the whole window.
        from silica.kernel.recall.paths import atomic_write_bytes
        from silica.kernel.workqueue import path_lease

        with path_lease(str(path)):
            existing = read_records(vault_path=vault_path)
            # Merge, not drop: a resumed CLEANUP re-fires with the same triple
            # AND the same notes, so union is a no-op there — but the sub-agent
            # lane commits several times under one triple, and dropping the
            # later ones left their notes with no record at all.
            idx = next(
                (i for i, r in enumerate(existing)
                 if r.get("source") == source and r.get("sha256") == sha256
                 and r.get("run_id") == run_id),
                None,
            )
            if idx is not None:
                prior = list(existing[idx].get("notes") or [])
                merged = list(dict.fromkeys(prior + list(notes)))
                if merged == prior:
                    return "present"
                # Replace the entry, never edit it in place: read_records copies
                # the LIST but hands back the very dicts `_records_memo` holds,
                # so an in-place edit lands in the cache BEFORE the write below
                # is known to have succeeded. That write can still fail (read-
                # only vault, ENOSPC, lease dir gone) — the except swallows it
                # and returns False while mtime/size are unchanged, leaving the
                # memo valid and reporting notes that were never persisted.
                existing[idx] = {**existing[idx], "notes": merged}
            else:
                existing.append(record)
            # Atomic: this ledger is authoritative (run_id/sha history is not
            # reconstructible from the vault) and read_records quarantines a
            # truncated file, so a torn rewrite would silently lose the history.
            atomic_write_bytes(
                path, json.dumps(existing, indent=2, ensure_ascii=False).encode("utf-8")
            )
    except Exception as exc:
        outcome = _observed(path, source, sha256, run_id, notes)
        if outcome != "present":  # the replace can land and a later step still raise
            logger.warning(
                "provenance: record for %s (run %s) did not land: %s (%s); a "
                "re-ingest of this source will re-append into the notes it "
                "already wrote", source, run_id, outcome, exc)
        return outcome
    return "present"


def drifted_notes(
    *,
    vault_path: str | None = None,
) -> list[tuple[str, str]]:
    """`[(note, source_basename), ...]` for notes derived from a superseded
    source version.

    Rule (spec-hermes-coherence §3): for each source with >=2 records at
    different sha256 values, take the most recent record whose sha differs
    from the source's CURRENT (latest) sha — its notes that do NOT appear
    in ANY record carrying the current sha are drifted.
    """
    records = read_records(vault_path=vault_path)
    by_source: dict[str, list[dict]] = {}
    for r in records:
        src = r.get("source")
        if src:
            by_source.setdefault(src, []).append(r)

    out: list[tuple[str, str]] = []
    for source, recs in by_source.items():
        shas = {r.get("sha256") for r in recs}
        if len(shas) < 2:
            continue
        current_sha = recs[-1].get("sha256")
        current_notes: set[str] = set()
        for r in recs:
            if r.get("sha256") == current_sha:
                current_notes.update(r.get("notes") or [])
        old_recs = [r for r in recs if r.get("sha256") != current_sha]
        if not old_recs:
            continue
        last_old = old_recs[-1]
        for note in last_old.get("notes") or []:
            if note not in current_notes:
                out.append((note, source))
    return out


def _bare_note_ref(p: str) -> str:
    """A note reference in the ledger's own spelling: vault-relative POSIX, no
    `.md`. Provenance stores RunManifestEntry.path in exactly this form; a
    caller may pass an absolute or `.md`-suffixed path, so relativize when
    possible and degrade to a plain strip otherwise."""
    ref = p or ""
    try:
        from silica.kernel.recall.paths import to_vault_relative

        ref = to_vault_relative(ref, ensure_md=False)
    except Exception:
        ref = ref.replace("\\", "/").strip("/")
    return ref.removesuffix(".md")


def note_key(p: str) -> str:
    """The comparable key of a note reference: `_bare_note_ref`, casefolded.

    Public because the undo journal scopes a revert by the ledger's note set
    and must fold its own (possibly absolute) inverse paths the same way;
    two spellings of this rule would disagree on exactly the rename cases
    the ledger exists to survive.
    """
    return _bare_note_ref(p).casefold()


_norm_note_ref = note_key


def already_distilled(source: str, sha256: str, *, vault_path: str | None = None) -> bool:
    """True when the last record for `source` is this exact content, from a
    run that completed and wrote at least one note. A zero-yield record (all
    ops deferred) or a partial run is a failure, not a completion: never skip
    a segment on either."""
    recs = read_records(source, vault_path=vault_path)
    last = recs[-1] if recs else None
    return bool(
        last and last.get("sha256") == sha256 and last.get("notes")
        and not last.get("partial")
    )


def note_authored_by(
    note_path: str,
    source: str,
    *,
    vault_path: str | None = None,
) -> bool:
    """True when `source` (a source basename) already authored `note_path`.

    Reads the provenance ledger: on any prior run, did this exact source file
    write or patch this note? The patch executor uses it to make a re-ingest
    idempotent — a source must not re-append its own concepts into the notes it
    already wrote (each re-append is a redundant "Additional notes (from <source>)"
    block). A genuinely new concept has no prior authored note, so it still
    flows to a fresh write; a DIFFERENT source enriching the same note still
    patches. Matches any recorded version of the source (an A->B->A edit still
    counts). Absent/unreadable ledger degrades to False.
    """
    target = _norm_note_ref(note_path)
    if not target:
        return False
    for r in read_records(source, vault_path=vault_path):
        if any(_norm_note_ref(n) == target for n in (r.get("notes") or [])):
            return True
    return False


def sources_by_note(*, vault_path: str | None = None) -> dict[str, list[str]]:
    """`{note_key: [source_basename, ...]}`, sources in first-seen ledger order.

    The ledger is keyed by source; the graph payload and the dedup verdict ask
    the other way round, for every node at once. One inversion per call is
    O(records) and the memo in read_records makes the parse free; per-node
    `sources_of` calls over the whole graph would be O(nodes * records).
    """
    out: dict[str, list[str]] = {}
    for r in read_records(vault_path=vault_path):
        src = r.get("source")
        if not src:
            continue
        for n in r.get("notes") or []:
            lst = out.setdefault(note_key(n), [])
            if src not in lst:
                lst.append(src)
    return out


def sources_of(note_path: str, *, vault_path: str | None = None) -> list[str]:
    """Every source basename that ever authored `note_path`, or []."""
    return list(sources_by_note(vault_path=vault_path).get(note_key(note_path), []))


def drift_map(*, vault_path: str | None = None) -> dict[str, str]:
    """`{note_path.md: source_basename}` for every drifted note.

    The read-time stale flag's shape: keys end in `.md` like codedocs.peek's,
    so `codedocs.peek_level` reads both maps with one lookup. The ledger
    stores bare paths, so the suffix is restored here and nowhere else.
    """
    return {f"{note}.md": source for note, source in drifted_notes(vault_path=vault_path)}


def rename_note(old_path: str, new_path: str, *, vault_path: str | None = None) -> AppendOutcome:
    """Make every record that names `old_path` name `new_path` instead.

    The ledger keys notes by bare path, so a move that rewrote every wikilink
    and not this left `note_authored_by` answering False for the moved note:
    the next nucleate of its source re-appended into a note it had already
    written, and drift reports named a path that no longer existed. Called by
    every backend's move(), right after the file itself has moved.

    Same outcome vocabulary as append_record: `present` means the ledger now
    names the new path wherever it named the old one (nothing to rename is
    present too, and writes nothing); `absent` means the old name is still
    there; `uncertain` means the store could not be read back.
    """
    path = _store_path(vault_path)
    if not path:
        return "absent"
    old_key = note_key(old_path)
    new_bare = _bare_note_ref(new_path)
    if not old_key or not new_bare:
        return "absent"

    def _rewritten(notes: list[str]) -> list[str] | None:
        if not any(note_key(n) == old_key for n in notes):
            return None
        moved = [new_bare if note_key(n) == old_key else n for n in notes]
        return list(dict.fromkeys(moved))

    try:
        from silica.kernel.recall.paths import atomic_write_bytes
        from silica.kernel.workqueue import path_lease

        with path_lease(str(path)):
            existing = read_records(vault_path=vault_path)
            changed = False
            for i, r in enumerate(existing):
                moved = _rewritten(list(r.get("notes") or []))
                if moved is not None:
                    # A new dict, never an in-place edit: read_records hands back
                    # the very dicts its memo holds (see append_record).
                    existing[i] = {**r, "notes": moved}
                    changed = True
            if not changed:
                return "present"
            atomic_write_bytes(
                path, json.dumps(existing, indent=2, ensure_ascii=False).encode("utf-8")
            )
    except Exception as exc:
        outcome = _observed_rename(path, old_key)
        if outcome != "present":
            logger.warning(
                "provenance: rename %s -> %s did not land: %s (%s); a re-ingest of "
                "its source will re-append into the moved note and drift reports "
                "will name the old path", old_path, new_path, outcome, exc)
        return outcome
    return "present"


def _observed_rename(path: Path, old_key: str) -> AppendOutcome:
    """Whether any record still names the old path, read raw after a failure."""
    try:
        if not path.exists():
            return "absent"
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            return "uncertain"
    except Exception:
        return "uncertain"
    for r in records:
        if isinstance(r, dict) and any(note_key(n) == old_key for n in (r.get("notes") or [])):
            return "absent"
    return "present"


def check_renucleate(
    source: str,
    incoming_sha256: str,
    *,
    vault_path: str | None = None,
) -> tuple[bool, int]:
    """`(is_modified, notes_derived_from_the_prior_version)`.

    Used by /nucleate right before staging a file: warns when the inbox file
    about to be re-nucleated carries a different sha256 than the last known
    record for that source basename. No prior record -> (False, 0) — a
    first nucleate is not a re-nucleate.
    """
    recs = read_records(source, vault_path=vault_path)
    if not recs:
        return False, 0
    last = recs[-1]
    if last.get("sha256") == incoming_sha256:
        return False, 0
    return True, len(last.get("notes") or [])


# --- Span grounding (verbatim-contract gate) --------------------------------
# The distiller must carry formulas and code verbatim from the source excerpt
# (distiller_prompt "Content Quality Requirements"). Prose is rewritten and
# translated by design, so it can't be checked mechanically — but math and
# code can: a $$...$$ / ```...``` span in the output that cannot be located
# in the source excerpt is a fabrication candidate. Warn-only signal:
# re-typesetting ASCII math into LaTeX is sanctioned by the prompt's own
# few-shot example, so a span class is gated only when the source itself
# uses that markup ($ for math, ``` for code).

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")

MIN_GROUNDABLE_CHARS = 12  # normalized; shorter spans ($x$, \top) match anywhere
GROUNDING_FLOOR = 0.85     # matched-char fraction under LOCAL difflib alignment
LOCALITY_WINDOW = 2        # matched blocks must fit in a window of N * len(span)

_NUMERAL_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


_SINGLE_BRACE_RE = re.compile(r"(?<=[_^])\{(\w)\}")
_BOLD_MACRO_RE = re.compile(r"\\(?:mathbf|boldsymbol|pmb|bm|bf)\b")
# Upright-text wrappers around a name: which one a formula carries is the
# converter's or the model's habit, not content. `\operatorname*{Pr}`,
# `\mathrm{err}`, `\mathsf{Fill}` and `\text{err}` all put the same upright
# word on the page; stripped to the bare word so they compare equal. Applied
# after whitespace collapsing, so MinerU's `\mathrm { e r r }` folds too.
# (Measured 2026-08-05: the dominant residue class of grounding flags was a
# single wrapper swap on a short span — \text vs \mathrm vs \mathsf — which
# alone drops the local match fraction below the floor.)
_UPRIGHT_MACRO_RE = re.compile(r"\\(?:operatorname\*?|text|mathrm|mathsf)\{([^{}]*)\}")
# `\Pr` is the one named operator the wrappers alias with (`\operatorname*{Pr}`).
_PR_MACRO_RE = re.compile(r"\\Pr(?![A-Za-z])")


def _norm_math(s: str) -> str:
    """Typesetting noise dropped: whitespace, braces around a lone subscript, and
    which macro was used to bold a symbol.

    Both are invisible to LaTeX and both differ systematically between a PDF
    converter and a model. MinerU spaces a formula out at every token and braces
    every index (`S _ {B} = \\sum_ {j = 1} ^ {K}`); a model re-emits it tight
    (`S_B = \\sum_{j=1}^K`). Under plain whitespace-collapsing the two differ
    every few characters, which shreds difflib's matching blocks below the 3-char
    floor and reported verbatim formulas as fabrications — the dominant rejection
    class on any converted-PDF source (measured: 31% of one run's flags).

    Math only: inside a fenced code span whitespace carries meaning.

    The bold macros are one closed family — `\\mathbf`, `\\boldsymbol`, `\\pmb`,
    `\\bm`, `\\bf` all put the same bold symbol on the page, and which one a
    formula carries says nothing about what it states. A converter emits the
    one its PDF encoded and a model re-emits the one it learned, so a vector
    equation copied faithfully read as fabricated (one run on a converted ML
    lecture flagged 61 spans over six notes, the two spellings interleaved
    between them).

    Same reasoning folds the upright-text wrappers (`\\operatorname*?`,
    `\\text`, `\\mathrm`, `\\mathsf`) to their bare content, plus the `\\Pr`
    alias they typeset.

    ponytail: these families only, not an alias table. Alphabets stay
    distinct — `\\mathbb{R}` and `\\mathcal{R}` are different sets, and this
    normalizer will not say otherwise.
    """
    s = "".join(s.split())
    prev = None
    while prev != s:  # nested wrappers: \operatorname*{\mathrm{Pr}}
        prev, s = s, _UPRIGHT_MACRO_RE.sub(r"\1", s)
    s = _PR_MACRO_RE.sub("Pr", s)
    return _BOLD_MACRO_RE.sub(r"\\bf", _SINGLE_BRACE_RE.sub(r"\1", s))


def _local_match_fraction(s: str, src: str) -> float:
    """Best matched-char fraction of *s* with all blocks inside one source
    window of LOCALITY_WINDOW * len(s). Global scatter would let a formula
    recombined from fragments across the excerpt self-ground; localization
    is the whole point of the gate."""
    from difflib import SequenceMatcher

    # blocks under 3 chars are coincidence ('v', ')'), not localization —
    # they inflate the fraction exactly on recombined formulas
    blocks = [b for b in SequenceMatcher(None, s, src, autojunk=False).get_matching_blocks() if b.size >= 3]
    if not blocks:
        return 0.0
    window = LOCALITY_WINDOW * len(s)
    best = 0
    for i in range(len(blocks)):  # blocks are few; O(n²) is fine
        lo = blocks[i].b
        best = max(best, sum(b.size for b in blocks[i:] if b.b + b.size <= lo + window))
    return best / len(s)


def ungrounded_spans(body: str, source: str) -> list[str]:
    """Verbatim-contract spans of *body* (math, fenced code) not locatable in *source*.

    Returns the offending spans (whitespace-normalized); empty list means
    fully grounded or nothing gateable. A span class is checked only when
    *source* itself contains that markup — LaTeX in the output for an
    ASCII-math source is legitimate re-typesetting, not drift.

    Two independent checks per span (either failing flags it):
    - numeric literals (≥2 chars) must appear verbatim in the source —
      numbers survive re-typesetting and translation, so an absent constant
      is the sharpest fabrication signal (altered 0.01→0.1, invented ε=10⁻⁸);
    - fuzzy match must be LOCAL (see _local_match_fraction).
    """
    # (span, normalizer) — each class is compared against a source normalized
    # the same way, so a code fence keeps its whitespace and math loses it.
    spans: list[tuple[str, Callable[[str], str]]] = []
    if "```" in source:
        spans += [(s, _norm_ws) for s in _FENCE_RE.findall(body)]
    if "$" in source:
        # "$" also matches currency; acceptable for a warn-only gate
        rest = _FENCE_RE.sub("", body)
        spans += [(s, _norm_math) for s in _DISPLAY_MATH_RE.findall(rest)]
        spans += [(s, _norm_math) for s in _INLINE_MATH_RE.findall(_DISPLAY_MATH_RE.sub("", rest))]

    # The key type must be the general callable, not the two specific function
    # objects mypy infers from the literal, since `norm` below is either one.
    sources: dict[Callable[[str], str], str] = {
        _norm_ws: _norm_ws(source), _norm_math: _norm_math(source)}
    out: list[str] = []
    for span, norm in spans:
        s, src = norm(span), sources[norm]
        if len(s) < MIN_GROUNDABLE_CHARS or s in src:
            continue
        numerals = [n for n in _NUMERAL_RE.findall(s) if len(n) >= 2]
        if any(n not in src for n in numerals):
            out.append(_norm_ws(span))
            continue
        if _local_match_fraction(s, src) < GROUNDING_FLOOR:
            out.append(_norm_ws(span))
    return out


# Extractive invariant: markers the model may prepend to a copied span (list
# bullets, ordered markers, headings, blockquotes) and inline wikilinks the
# autolink phase injects later — stripped before the substring check so added
# structure never reads as content drift.
_LEADING_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)]|#{1,6}|>)\s+")
# A span wrapped in quotation marks is still a selected span — quoting is the
# canonical extractive act, and the marker strip above already covers its
# block-level form (`>`). Stripping the wrapper cannot let non-verbatim content
# through: the substring test still runs on what is inside.
# It runs on text that already went through _norm_extract, which folds U+2018 /
# U+2019 to a straight `'` — so the straight form is the one that actually
# arrives here, and the curly pair in the class could never match. A genuinely
# verbatim single-quoted span was therefore rejected as non-extractive.
_QUOTE_CHARS = "\"'\u201c\u201d\u00ab\u00bb"
_WRAPPING_QUOTES_RE = re.compile(f"^[{_QUOTE_CHARS}]+|[{_QUOTE_CHARS}]+$")
_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s+")
# An index line: a wikilink LABEL at the head, a separator, then the payload.
# The label points at another note, so it is authored by construction and can
# never be a source span; only the payload is claim content. Kept tight (the
# link must open the line and a separator must follow) because a looser
# split-on-any-dash would let a fabricated label ride in on a verbatim quote.
_INDEX_LABEL_RE = re.compile(r"^\[\[[^\]]+\]\]\s*[\u2014\u2013:-]\s+")
_ANY_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_EXTRACTIVE_MIN_CHARS = 12  # normalized; shorter residues match trivially


def _norm_extract(s: str) -> str:
    """Whitespace + straight-apostrophe normalization for substring matching.
    Case is preserved: a case change is an edit, not a copied span."""
    return _norm_ws(s.replace("’", "'").replace("‘", "'"))


def nonextractive_lines(body: str, source: str) -> list[str]:
    """Body content-lines that are NOT verbatim spans of *source*.

    The `extractive` distill profile must SELECT spans from the transcript, not
    rewrite them; this is the mechanical check enforcing that. Split the body
    into lines, strip structural markers (bullets, headings, blockquotes) and
    inline wikilinks (autolink adds those post-write), normalize whitespace and
    apostrophes, and require each remaining line to be a substring of the
    normalized source. Returns the offending lines; empty means fully
    extractive.

    ponytail: lines under MIN chars after stripping are skipped — a short
    residue substring-matches almost any source, so enforcing them only yields
    false rejections; paraphrase and fabrication are prose-length anyway.
    """
    src = _norm_extract(source)
    out: list[str] = []
    exempt: list[str] = []   # structure set aside, judged only if nothing else was
    judged = 0

    def _claim(raw: str) -> str:
        """The judgeable residue of a line: structural marker stripped, index
        label (the wikilink that introduces the payload) dropped, remaining
        wikilinks unwrapped to the text they wrap."""
        marked = _INDEX_LABEL_RE.sub("", _LEADING_MARKER_RE.sub("", raw))
        return _norm_extract(_WIKILINK_RE.sub(r"\1", marked))

    for raw in body.splitlines():
        # Framework structure is not a claim. A heading is a label, and a line
        # whose prose is only a caption around wikilinks is a link footer
        # ("Correlati: [[X]]"); neither can be a selected span, and judging
        # them made every block carrying one unpassable.
        #
        # Set ASIDE, never dropped. A bare `continue` here left the line out of
        # `exempt` too, so it was not judged AND not judgeable later: a body of
        # nothing but link lines came back "fully extractive" with zero checks
        # run, and "[[Kant]] rejected [[Hume]]." passed against a source naming
        # neither. The `if not judged` fallback below exists for exactly that
        # case and could not see them.
        marked = _LEADING_MARKER_RE.sub("", raw)
        if (_HEADING_LINE_RE.match(raw)
                or len(_norm_extract(_ANY_WIKILINK_RE.sub("", marked)))
                < _EXTRACTIVE_MIN_CHARS):
            exempt.append(_claim(raw))
            continue
        norm = _claim(raw)
        if len(norm) < _EXTRACTIVE_MIN_CHARS:
            continue
        judged += 1
        if norm not in src and _WRAPPING_QUOTES_RE.sub("", norm) not in src:
            out.append(norm)
    if not judged:
        # Nothing in this body was verifiable, so an empty result would read as
        # "fully extractive" on a body that was never checked. Judge the
        # structure after all: exempting it is only safe while real content
        # carries the block.
        return [
            e for e in exempt
            if len(e) >= _EXTRACTIVE_MIN_CHARS
            and e not in src and _WRAPPING_QUOTES_RE.sub("", e) not in src
        ]
    return out


# Keyed attribution (OKF §5.1). One `## Sources` block covers a whole note, so
# a note patched from three transcripts says only "these three fed me" — never
# which line came from which. A footnote marker per grounded line closes that,
# and the label is the join key rather than the position: agents reorder lists,
# and a positional reference would silently re-point.
_FOOTNOTE_LABEL_RE = re.compile(r"[^A-Za-z0-9_-]+")
_SECTION_STOP_RE = re.compile(r"^#{1,6}\s+(Sources|Superseded)\b", re.IGNORECASE)


def footnote_label(source_basename: str) -> str:
    """The footnote label for a source leaf: its id, made markdown-safe.

    Derived from the leaf's `source_id` (the basename), so the label a reader
    sees and the leaf a consumer resolves are the same key.
    """
    return _FOOTNOTE_LABEL_RE.sub("-", source_basename.removesuffix(".md")).strip("-")


def attribute_lines(note: str, source: str, label: str) -> str:
    """Mark every line of *note* that is verbatim from *source* with `[^label]`.

    The same span test the extractive invariant uses (`nonextractive_lines`),
    read the other way round: there it names the lines that failed, here the
    ones that passed. Left alone: the frontmatter block, headings, blank and
    short lines, fenced code (a marker inside a fence would corrupt the code
    it attributes), everything from `## Sources` or `## Superseded` onward, and
    any line already carrying this label — re-running over a note is a no-op.
    """
    from silica.kernel.write import frontmatter

    if not label or not source.strip():
        return note
    _data, raw_fm, body = frontmatter.split(note)
    head = note[: len(note) - len(body)] if raw_fm is not None else ""
    src = _norm_extract(source)
    marker = f"[^{label}]"
    lines = body.splitlines()
    out: list[str] = []
    marked = False
    fenced = False
    for i, raw in enumerate(lines):
        if _SECTION_STOP_RE.match(raw):
            out.extend(lines[i:])
            break
        if raw.lstrip().startswith("```"):
            fenced = not fenced
            out.append(raw)
            continue
        line = _WIKILINK_RE.sub(r"\1", _LEADING_MARKER_RE.sub("", raw))
        norm = _norm_extract(line)
        if (fenced or raw.lstrip().startswith("#") or marker in raw
                or len(norm) < _EXTRACTIVE_MIN_CHARS or norm not in src):
            out.append(raw)
            continue
        out.append(raw.rstrip() + marker)
        marked = True
    if not marked:
        return note
    return head + "\n".join(out) + ("\n" if body.endswith("\n") else "")


def content_sha256(source_path: str) -> str:
    """SHA-256 hex digest of a source file's content.

    Mirrors silica.router.orchestrator.InjectorFSM.run()'s hashing exactly
    (DRIVER.read_note(...).content.encode("utf-8"), falling back to raw file
    bytes) so a value computed here (e.g. by the /nucleate pre-check) compares
    equal to the sha256 CLEANUP later records for an unmodified file. Never
    raises — returns "" when the file can't be read either way.
    """
    try:
        from silica.driver import DRIVER

        content_bytes = DRIVER.read_note(source_path).content.encode("utf-8")
        return hashlib.sha256(content_bytes).hexdigest()
    except Exception:
        try:
            content_bytes = Path(source_path).read_bytes()
            return hashlib.sha256(content_bytes).hexdigest()
        except OSError:
            return ""


# Citation fields a converted chunk carries in its frontmatter (stamped by
# sources/convert._provenance_fm) that written notes inherit, so a note can be
# cited without reopening the original document.
CITATION_KEYS = ("doi", "arxiv", "authors", "source_title")

_citation_memo: dict[str, dict[str, str]] = {}


def citation_of(source_basename: str, *, vault_path: str | None = None) -> dict[str, str]:
    """Citation frontmatter of the inbox chunk `source_basename` derives from.

    Searched under the active inbox tree (chunks live there until CLEANUP
    archives them into its `done/` subtree, still under the inbox). Missing
    file, unreadable frontmatter, or no citation keys all degrade to {} —
    provenance is additive, never a reason to fail a write. Memoized per
    basename for the duration of the process (a run stamps many notes from
    one chunk).
    """
    if not source_basename:
        return {}
    if source_basename in _citation_memo:
        return _citation_memo[source_basename]

    result: dict[str, str] = {}
    try:
        from silica.kernel.vault_manifest import active_inbox_dir
        from silica.kernel.write import frontmatter

        root = resolve_vault_path(vault_path)
        inbox = active_inbox_dir()
        if root and inbox:
            hits = sorted((Path(root) / inbox).rglob(source_basename))
            if hits:
                data, _, _ = frontmatter.split(hits[0].read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    result = {
                        k: str(data[k]) for k in CITATION_KEYS
                        if data.get(k) not in (None, "")
                    }
    except Exception as exc:
        logger.debug("citation_of(%s) failed (non-fatal): %s", source_basename, exc)
    _citation_memo[source_basename] = result
    return result
