# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Contested-claims layer (spec-hermes-coherence §1).

A contradiction is neither a duplicate nor a new concept: it is recorded on
the existing note (frontmatter flag + warning callout) and kept visible until
a human resolves it. Pure functions over note text — no I/O, no LLM.

Also home to the claim stamp (spec-contested-bitemporal §3): the per-claim
event clock. It rides in an HTML comment rather than frontmatter because
frontmatter is per-note while a note accumulates claims from many sources on
different dates; the comment is invisible in preview, greppable, and survives
every write path byte-for-byte (no YAML round-trip).
"""
from __future__ import annotations

import re

from silica.kernel.write import frontmatter
from silica.kernel.write.notetype import is_human_verified

CONTESTED_KEY = "contested"
CONTRADICTIONS_KEY = "contradictions"
_UNRESOLVED_TAIL = "Unresolved."

STAMP_RE = re.compile(r"<!--\s*silica:\s*(.*?)\s*-->")
# Values are dates and hex run ids. Anything outside this class is dropped so a
# value can never close the comment early or break the key=value split.
_STAMP_VALUE_RE = re.compile(r"[^\w.:+-]")


def stamp(**fields: str) -> str:
    """A claim stamp: `<!-- silica: valid_from=2023-05-08 run=b07f1268 -->`.

    Key order is the caller's, so the rendered line is deterministic. Fields
    with an empty value are dropped; all-empty yields "" so a caller can splice
    unconditionally.
    """
    parts = []
    for k, v in fields.items():
        if v is None:  # str(None) is "None", which is truthy — never emit it
            continue
        cleaned = _STAMP_VALUE_RE.sub("", str(v).strip())
        if cleaned:
            parts.append(f"{k}={cleaned}")
    return f"<!-- silica: {' '.join(parts)} -->" if parts else ""


def _stamp_fields(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in raw.split():
        k, _, v = tok.partition("=")
        if k and v:
            out[k] = v
    return out


def parse_stamp(text: str) -> dict[str, str]:
    """Fields of the first claim stamp in `text`; {} when there is none."""
    m = STAMP_RE.search(text or "")
    return _stamp_fields(m.group(1)) if m else {}


def parse_stamps(text: str) -> list[dict[str, str]]:
    """Fields of EVERY claim stamp in `text`, in document order.

    A note accumulates claims from several sources on different dates, so the
    first stamp is not the note's clock. The temporal report reads all of them;
    `parse_stamp` stays the single-claim accessor its callers expect.
    """
    return [_stamp_fields(m.group(1)) for m in STAMP_RE.finditer(text or "")]


def mark_contested(content: str, source_ref: str) -> str:
    """Set `contested: true` and append `source_ref` to `contradictions:`.

    Idempotent on source_ref. A note without frontmatter gains a minimal one;
    a note with unparseable YAML is returned unchanged (never destroy what we
    cannot round-trip).
    """
    data, raw, body = frontmatter.split(content)
    if data is None:
        if raw is not None:  # frontmatter present but broken YAML
            return content
        data, body = {}, content
    refs = list(data.get(CONTRADICTIONS_KEY) or [])
    if source_ref in refs:
        return content
    data[CONTESTED_KEY] = True
    data[CONTRADICTIONS_KEY] = refs + [source_ref]
    return frontmatter.dump(data, body)


def clear_contested(content: str) -> str:
    """Remove the `contested`/`contradictions` flag from a note's frontmatter.

    No-op when the note is not contested; a note with unparseable YAML is
    returned unchanged (mirror of `mark_contested`).
    """
    data, raw, body = frontmatter.split(content)
    if data is None or not data.get(CONTESTED_KEY):
        return content
    data.pop(CONTESTED_KEY, None)
    data.pop(CONTRADICTIONS_KEY, None)
    return frontmatter.dump(data, body)


def contested_refs(content: str) -> list[str]:
    """The note's `contradictions:` entries; [] when not contested."""
    data, _, _ = frontmatter.split(content)
    if not data or not data.get(CONTESTED_KEY):
        return []
    return list(data.get(CONTRADICTIONS_KEY) or [])


def contested_callout(claim: str, source_basename: str) -> str:
    """The warning callout recording a conflicting claim, with provenance."""
    quoted = "\n".join(f"> {line}".rstrip() for line in claim.strip().splitlines())
    return (
        f"> [!warning] Contradiction — from {source_basename}\n"
        f"{quoted}\n"
        f">\n"
        f"> Conflicts with this note. {_UNRESOLVED_TAIL}"
    )


# ---------------------------------------------------------------------------
# Superseded section (spec-contested-bitemporal §4)
#
# A claim that loses a contest is never deleted: it moves to the end of the
# note under `## Superseded`, keeping its own provenance and gaining a
# valid_to stamp. The section is ALWAYS the last one in the body, which is
# what append_before_superseded exists to guarantee — every EOF appender in
# the codebase routes through it.
# ---------------------------------------------------------------------------

SUPERSEDED_HEADING = "## Superseded"
_SUPERSEDED_RE = re.compile(r"^## Superseded\s*$", re.MULTILINE)
_CONTRADICTION_START_RE = re.compile(
    r"^> \[!warning\] Contradiction\b(?:[^\n]*?\bfrom (?P<src>[^\n]+?))?\s*$"
)


def append_before_superseded(content: str, block: str) -> str:
    """`content.rstrip() + "\\n" + block`, but above `## Superseded` if present.

    Without the section this is byte-identical to the plain EOF append it
    replaces. With it, the block lands above, so live content never ends up
    filed under the note's graveyard.
    """
    m = _SUPERSEDED_RE.search(content)
    if not m:
        return content.rstrip() + "\n" + block
    head, tail = content[: m.start()], content[m.start():]
    return head.rstrip() + "\n" + block.rstrip() + "\n\n" + tail


def _split_at_superseded(body: str) -> tuple[str, str]:
    """(live body, superseded section incl. heading). Tail is "" when absent."""
    m = _SUPERSEDED_RE.search(body)
    if not m:
        return body, ""
    return body[: m.start()], body[m.start():]


def ref_source(ref: str) -> str:
    """The source basename a `contradictions:` ref names; "" when it names none.

    Two spellings reach the list: `source: appunti.md` from the dedup judge,
    which has a callout in the body, and `flagged: … (by user, date)` from
    silica_flag_note, which has none. Only the first can be matched to a block.
    """
    kind, _, rest = ref.partition(":")
    return rest.strip() if kind.strip() == "source" else ""


def _extract_contradiction_callouts(
    body: str, source: str | None = None
) -> tuple[str, list[str]]:
    """Lift contradiction callouts out of `body`; all of them, or one source's.

    A callout is the run of contiguous `>`-prefixed lines opened by the
    warning marker (its internal blank lines are `>`-prefixed too, so the run
    is unbroken). A provenance header left with nothing but whitespace under
    it is dropped with its callout: the header exists only to attribute the
    block that just moved. Both header spellings count, so a note written
    before the header was translated still gets its orphan pruned.

    `source` restricts the lift to the callouts attributed to that basename,
    which is how one contradiction is resolved while the others stay open.
    """
    from silica.kernel.write.templates import PROVENANCE_HEADER_PREFIXES

    lines = body.splitlines()
    kept: list[str] = []
    callouts: list[str] = []
    i = 0
    while i < len(lines):
        m = _CONTRADICTION_START_RE.match(lines[i])
        if m and (source is None or (m.group("src") or "").strip() == source):
            j = i
            while j < len(lines) and lines[j].startswith(">"):
                j += 1
            callouts.append("\n".join(lines[i:j]))
            i = j
            continue
        kept.append(lines[i])
        i += 1

    if callouts:
        pruned: list[str] = []
        for idx, line in enumerate(kept):
            if line.startswith(PROVENANCE_HEADER_PREFIXES):
                rest = kept[idx + 1:]
                nxt = next((r for r in rest if r.strip()), "")
                if not nxt or nxt.startswith("## "):
                    continue  # header whose only content was the moved callout
            pruned.append(line)
        kept = pruned

    return "\n".join(kept).rstrip() + "\n", callouts


# ---------------------------------------------------------------------------
# Reliability tiers (spec-contested-bitemporal §5)
#
# Ordinal, not a posterior: the three signals available are coarse, and a
# calibrated number over three levels would be false precision. Every tier is
# derived from the note text alone, so both sides of a comparison are always
# ranked on the same information (an asymmetric lookup would turn "we know
# more about A" into "A wins", which is not the same claim).
# ---------------------------------------------------------------------------

SUPERSEDED_BY_KEY = "superseded_by"

TIER_HUMAN = 3
TIER_GROUNDED = 2
TIER_DISTILLED = 1


def reliability_tier(content: str, *, has_source_leaf: bool | None = None) -> int:
    """How much weight a claim's origin earns it: 3 human, 2 grounded, 1 distilled.

    Human means the agent never claimed authorship (`AI` absent or false, or no
    frontmatter at all: every agent write stamps the flag). Grounded means an
    agent note whose verbatim source is still reachable through its `## Sources`
    link. Distilled is everything else.

    Unparseable frontmatter ranks lowest on purpose: a parse accident must never
    win a contest. `has_source_leaf` overrides the note-side signal for a claim
    that is not a note yet (an incoming excerpt has no `## Sources` block).

    The human tier no longer decays on a touch: the bulk patch path stamps
    `AI: partial` on a legacy user note (the agent appended a section, the
    body stays the user's), and partial ranks human here. `AI: true` means
    the agent authored the body and reads as before. Notes stamped `true` by
    patches that predate the partial convention are recognized forever; OKF
    §5.2 `verified` is their way back — a person who vouches for the note
    (`verified: {by: human:…, at: …}`) restores the tier the old stamp cost
    it. A machine verifier does not: a pipeline re-reading its own output is
    not a second opinion.
    """
    # Three ordinal levels off signals that already exist, not a calibrated
    # score — deliberate: the signals are coarse and a number would be false
    # precision.
    data, raw, _body = frontmatter.split(content or "")
    if data is None:
        if raw is not None:  # frontmatter present but broken YAML
            return TIER_DISTILLED
        return TIER_HUMAN  # no frontmatter at all: the agent always stamps one
    ai = data.get("AI")
    if isinstance(ai, str) and ai.strip().lower() == "partial":
        return TIER_HUMAN  # the agent touched the note, the user wrote it
    if not ai or is_human_verified(data):
        return TIER_HUMAN
    if has_source_leaf is None:
        from silica.kernel.recall.paths import SOURCES_MARKER
        has_source_leaf = SOURCES_MARKER in (content or "")
    return TIER_GROUNDED if has_source_leaf else TIER_DISTILLED


def merge_rank(content: str) -> tuple[int, int]:
    """Sort key for picking the target of a duplicate merge. Higher wins.

    Replaces the bare `len(body)` heuristic, which systematically handed the
    merge to the verbose agent note over the terse hand-written one. Length
    still breaks a tie within a tier, where there is no reliability signal to
    prefer either side.
    """
    return (reliability_tier(content), len(content))


def superseded_claim(
    claim: str, *, source_basename: str, valid_from: str | None, valid_to: str
) -> list[str]:
    """The `## Superseded` entry for a claim that lost the contest on arrival.

    `from=` is the derivation axis (§4.1): the source the claim came from, never
    a note of the vault. The block is built here rather than by the judge, which
    only ever supplies the verdict and the claim text.
    """
    quoted = "\n".join(f"> {line}".rstrip() for line in claim.strip().splitlines())
    return [
        stamp(**{"valid_from": valid_from or "", "valid_to": valid_to,
                 "from": source_basename}),
        f"> [!quote] Superseded {valid_to} (from {source_basename})",
        quoted,
        "",
    ]


def note_clock(content: str) -> str | None:
    """The freshest date a note can show for its own claims; None when it shows none.

    Two sources, both already written by this codebase: the newest `valid_from`
    on a claim stamp (Fase A), and OKF §5.2 `verified.at`, a person recording
    the day they read the note. A note that carries neither is not thereby
    fresh — it is silent, which `suppress_contest` treats as the risk it is.
    """
    from silica.kernel.write.notetype import verified_entries

    dates = [s["valid_from"] for s in parse_stamps(content) if s.get("valid_from")]
    data, _raw, _body = frontmatter.split(content or "")
    dates += [
        str(e.get("at")) for e in verified_entries(data) if e.get("at")
    ]
    return max(dates) if dates else None


def suppress_contest(
    target_content: str, *, incoming_tier: int, incoming_clock: str | None
) -> bool:
    """Whether a contradiction may be auto-resolved in the target's favour (§6.1-bis).

    Two conditions, and the second is why this is not last-write-wins wearing a
    hat. First, the target must STRICTLY outrank the incoming claim: equal tier
    carries no signal and stays contested (§7.5). Second, nothing may indicate
    the losing claim is the fresher one — recency never resolves a contest here,
    it only refuses to let reliability resolve one it would get wrong.

    An unknown target clock vetoes a dated incoming claim on purpose. Silence
    about when a note was last true is not evidence that it still is, and the
    asymmetry is deliberate: declining to auto-resolve leaves a visible contest,
    while resolving wrongly buries a live claim under `## Superseded`.

    Measured on `evals/golden/fixtures/contests`: strict dominance alone acts on
    4 contests and gets 2 wrong (precision 0.50); with this veto it acts on 2
    and gets both right, at the cost of leaving one more settleable contest
    open. Both failures it removes were a stale note meeting a fresher source,
    which is the ordinary memory update rather than an exotic case.
    """
    if reliability_tier(target_content) <= incoming_tier:
        return False
    if incoming_clock is None:
        return True  # nothing suggests the losing claim is fresher
    target_clock = note_clock(target_content)
    return target_clock is not None and target_clock >= incoming_clock


def mark_superseded_by(content: str, winner: str) -> str:
    """Point a merged-away note at the note that absorbed it.

    The merge loser used to be left on disk with overlapping content and no
    link to the winner: two notes saying the same thing and no record that one
    replaced the other. Idempotent; unparseable YAML is returned unchanged.
    """
    data, raw, body = frontmatter.split(content)
    if data is None:
        if raw is not None:
            return content
        data, body = {}, content
    link = f"[[{winner.removesuffix('.md').rsplit('/', 1)[-1]}]]"
    if data.get(SUPERSEDED_BY_KEY) == link:
        return content
    data[SUPERSEDED_BY_KEY] = link
    return frontmatter.dump(data, body)


def follow_superseded(path: str, *, max_hops: int = 5) -> str:
    """The note that absorbed `path`, or `path` itself when nothing did.

    The inverse of mark_superseded_by, and until 2026-09-03 it had no reader:
    the loser stayed on disk with its pointer while the write gate coerced
    every later same-title op onto it by exact path, so "Percettrone" took
    Lezione 2, 8 and 10 and "Percettrone di Rosenblatt" kept only Lezione 1
    (run f30ace50). The pointer is a basename link, so the winner is looked
    up next to the loser; a dangling or cyclic pointer stops the walk and
    the last note that exists wins. Best-effort: unreadable → `path`.
    """
    import os
    from silica.driver import DRIVER

    seen = {path}
    for _ in range(max_hops):
        try:
            data, _raw, _body = frontmatter.split(DRIVER.read_note(path).content or "")
        except Exception:
            return path
        link = str((data or {}).get(SUPERSEDED_BY_KEY) or "").strip()
        if not link.startswith("[["):
            return path
        name = link.strip("[]").split("|", 1)[0].rsplit("/", 1)[-1].strip()
        nxt = os.path.join(os.path.dirname(path), name + ("" if name.endswith(".md") else ".md"))
        try:
            if nxt in seen or not DRIVER.read_note(nxt).content:
                return path
        except Exception:
            return path
        seen.add(nxt)
        path = nxt
    return path


def resolve_contested(
    content: str, *, resolved_by: str, valid_to: str, source_ref: str | None = None
) -> str:
    """Resolve a note's contradictions without erasing the record.

    Each resolved callout moves under `## Superseded`, stamped with `valid_to`
    and its "Unresolved." tail rewritten, before the frontmatter flags are
    dropped. Callouts already filed under `## Superseded` are left alone, so
    re-running is a no-op.

    `source_ref` resolves exactly one entry of `contradictions:` and files only
    the blocks attributed to it; `contested: true` survives until the list
    empties. Without it every open contradiction is resolved at once, which is
    all a caller holding no verdict can honestly ask for. A ref that is not in
    the list returns the note unchanged: resolving twice must not clear a
    contradiction nobody adjudicated.

    This is what `clear_contested` should have been: clearing the flag while
    leaving a body callout that still reads "Unresolved" makes the note lie
    about its own state, and drops every record of what was contested.
    """
    from silica.kernel.write.moc import merge_moc_section

    data, raw, body = frontmatter.split(content)
    if data is None:
        if raw is not None:  # frontmatter present but broken YAML
            return content
        data, body = {}, content
    if not data.get(CONTESTED_KEY):
        return content

    remaining: list[str] = []
    source: str | None = None
    if source_ref is not None:
        refs = list(data.get(CONTRADICTIONS_KEY) or [])
        if source_ref not in refs:
            return content
        remaining = [r for r in refs if r != source_ref]
        source = ref_source(source_ref)

    live, tail = _split_at_superseded(body)
    callouts: list[str]
    if source_ref is not None and not source:
        # A `flagged:` ref carries no body block — the ref drops, nothing moves.
        kept, callouts = live, []
    else:
        kept, callouts = _extract_contradiction_callouts(live, source)

    new_body = kept + tail
    if callouts:
        block: list[str] = []
        for callout in callouts:
            block.append(stamp(valid_to=valid_to, resolved_by=resolved_by))
            block.append(callout.replace(_UNRESOLVED_TAIL, f"Resolved {valid_to}."))
            block.append("")
        new_body = merge_moc_section(new_body, SUPERSEDED_HEADING, block)

    if remaining:
        data[CONTRADICTIONS_KEY] = remaining
    else:
        data.pop(CONTESTED_KEY, None)
        data.pop(CONTRADICTIONS_KEY, None)
    return frontmatter.dump(data, new_body)
