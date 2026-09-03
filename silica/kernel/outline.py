# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Outline-first nucleation lane: the model names the units, the machine guards.

Why this lane exists (ADR-0036; docs/specs/nucleation-paradigm-ab.md, measured
2026-09-02 on two SVM lectures): the keyphrase pipeline let a miner decide the
units (tokens // 20, cap 40), handed the distiller {name, excerpt} pairs with
no document structure, hid same-run notes from the related candidates and
linked by cosine + BM25. The result was a star around the hub: section-level
provenance 0%, typed edges 0, one real link between the two lectures. The
prototype that reads the lecture whole reached 100% section provenance, 37
typed and explained edges and 8 cross-lecture pairs at 7 calls and 5.5 min
against 53 min for the pipeline. This module is that prototype with the four
defects it showed fixed mechanically: a coverage pass over the source
headings, `same_as` handed to the dedup judge instead of merged blind, a cap
on edges per target, and lecture-level notes hidden from the edge stage.

Call plan per source file: A (outline) -> GAP (only when a heading has no
idea) -> BODIES (batches of 8) -> B (cross edges, only when the vault has
notes). Every stage is one JSON reply; `ask` is injectable so the plan is
testable without a model.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from collections.abc import Set as AbstractSet
from typing import Callable

logger = logging.getLogger(__name__)

RELATIONS = ("defines", "derives", "relaxes", "applies", "bounds", "contrasts",
             "instance_of", "motivates", "generalizes", "justifies")
CROSS_RELATIONS = RELATIONS + ("same_as",)

# Readable verb phrase per relation, subject = the note the line sits in.
# English on purpose: vault strings emitted by the harness are UI copy.
RELATION_LABEL = {
    "defines": "defined via", "derives": "derived from", "relaxes": "relaxes",
    "applies": "applies", "bounds": "bounds", "contrasts": "contrasts with",
    "instance_of": "instance of", "motivates": "motivates",
    "generalizes": "generalizes", "justifies": "justifies",
}

# Frontmatter `section:` value that marks a lesson spine note. The spine is
# the one note per source that is about the source's argument, not an idea,
# so the edge stage must never propose it as a target (measured: 6 of 9 edges
# went to the lecture-level note when it was visible).
SPINE_SECTION = "(outline)"

BODIES_BATCH = 8            # ideas per bodies call: 8 x ~600 tokens stays under every worker's output cap
MAX_EDGES_PER_TARGET = 3    # a target with more is a hub in disguise, not three real dependencies
MIN_WHY_CHARS = 20          # "e' collegato" is not a reason
MAX_OUTLINE_ROWS = 400      # ponytail: folder-scoped outline; reopen at 400 notes in one folder, then seed with cosine top-k

STAGE_A_TAG = "[silica outline: stage A]"
STAGE_GAP_TAG = "[silica outline: coverage]"
STAGE_BODIES_TAG = "[silica outline: bodies]"
STAGE_B_TAG = "[silica outline: stage B]"

STAGE_A = STAGE_A_TAG + """
You are reading one complete source document (markdown, LaTeX kept, image descriptions inside <details>). Reconstruct its ARGUMENT as a small graph of ideas, the way a careful student would after studying it.

Emit ONLY a JSON object:
{{
 "lesson_title": "<short title of what this document is about, in {lang}>",
 "spine": ["<idea title>", ...],
 "ideas": [
   {{
     "title": "<noun phrase in {lang} naming ONE idea; never a sentence fragment>",
     "section": "<heading of the source section(s) the idea comes from, verbatim>",
     "claim": "<one sentence in {lang}: what this idea asserts>",
     "depends_on": [{{"title": "<another idea title from THIS list>", "relation": "<one of {rels}>", "why": "<one sentence in {lang}>"}}]
   }}
 ]
}}
Rules:
- "spine" lists every idea title exactly once, in the order the argument develops.
- Between 6 and 16 ideas. Merge repeated slides of the same idea into one idea. Skip noise (author, email, course header, empty slides).
- Every depends_on.title MUST be another idea title of this same JSON. Relation meanings: defines (A is defined in terms of B), derives (A is derived from B), relaxes (A weakens B's assumptions), applies (A applies B), bounds (A bounds B), contrasts (A vs B), instance_of, motivates (A is why B is needed), generalizes, justifies.
- lesson_title, every claim and every why are written in {lang}.
- No markdown fences around the JSON.
"""

STAGE_GAP = STAGE_GAP_TAG + """
You already extracted the ideas listed below from the source document that follows. These source sections have NO idea yet:
{gaps}
For each of them either add an idea (same shape as before: title, section, claim, depends_on referring to existing or new titles) or record an explicit skip with the reason (empty slide, duplicate of an existing idea, noise).

Emit ONLY a JSON object: {{"ideas": [...], "skips": [{{"section": "<heading>", "reason": "<why>"}}]}}. No markdown fences.
"""

STAGE_BODIES = STAGE_BODIES_TAG + """
You are given one complete source document (markdown, LaTeX kept) and a list of ideas already identified in it (title | source section). For EACH listed idea write its note body in {lang}: the facts themselves (definitions, formulas in LaTeX verbatim, theorem statements, algorithm steps), copied from the source, never outside knowledge, never a description of the source. Markdown. Keep every formula that belongs to the idea, drop repeated slides.

Emit ONLY a JSON object: {{"bodies": {{"<idea title verbatim>": "<markdown body>", ...}}}}
JSON strings: escape every backslash (write \\\\alpha for \\alpha), newlines as \\n. No markdown fences.
"""

STAGE_B = STAGE_B_TAG + """
You maintain a knowledge vault. Below is the OUTLINE of the vault so far (one line per existing note: lesson | title | claim), then the ideas of a NEW source (title | claim).

Find the connections a careful student would draw between the NEW ideas and the EXISTING notes: where a new idea applies, relaxes, generalizes, bounds, justifies, contrasts with, is an instance of, or is the same idea as an existing one. Propose only connections you can justify in one sentence from the two claims; at most three connections per existing note.

Emit ONLY a JSON object: {{"edges": [{{"from": "<new idea title>", "to": "<existing note title>", "relation": "<one of {rels}>", "why": "<one sentence in {lang}>"}}]}}
"from" MUST be a new idea title and "to" an existing note title, both verbatim. Use "same_as" only for the same idea restated. Every why is written in {lang}. No markdown fences.
"""


# Reply schemas, one per stage: passed as `response_schema` so the provider
# constrains decoding (the distiller's guard since the two-pass split). A
# free-form reply on an upstream that ignores `reasoning: false` was the
# thinking trace (no JSON, finish=stop) or a JSON cut at the budget
# (finish=length): both seen on the 2026-09-02 live run.
from pydantic import BaseModel, Field


class DependencyReply(BaseModel):
    title: str
    relation: str
    why: str = ""


class IdeaReply(BaseModel):
    title: str
    section: str = ""
    claim: str = ""
    depends_on: list[DependencyReply] = Field(default_factory=list)


class OutlineReply(BaseModel):
    lesson_title: str = ""
    spine: list[str] = Field(default_factory=list)
    ideas: list[IdeaReply]


class SkipReply(BaseModel):
    section: str
    reason: str = ""


class GapReply(BaseModel):
    ideas: list[IdeaReply] = Field(default_factory=list)
    skips: list[SkipReply] = Field(default_factory=list)


class BodiesReply(BaseModel):
    bodies: dict[str, str]


class EdgeReply(BaseModel):
    from_: str = Field(alias="from")
    to: str
    relation: str
    why: str = ""


class EdgesReply(BaseModel):
    edges: list[EdgeReply] = Field(default_factory=list)


SCHEMA_BY_TAG: dict[str, type[BaseModel]] = {STAGE_A_TAG: OutlineReply, STAGE_GAP_TAG: GapReply,
                 STAGE_BODIES_TAG: BodiesReply, STAGE_B_TAG: EdgesReply}


@dataclass
class Idea:
    title: str
    section: str
    claim: str
    body: str = ""
    depends_on: list[dict] = field(default_factory=list)


@dataclass
class Outline:
    lesson_title: str
    spine: list[str]
    ideas: list[Idea]
    skips: list[dict] = field(default_factory=list)

    def titles(self) -> list[str]:
        return [i.title for i in self.ideas]


# ------------------------------------------------------------------ routing --
def lane_for(profile: str | None) -> str:
    """'outline' for document-shaped sources, 'keyphrase' for the memory forms.

    clip and promotion carry episodic facts the distiller's ephemeral schema
    routes; the outline lane has no such channel, so they keep the keyphrase
    pipeline. SILICA_NUCLEATE_LANE=keyphrase turns the lane off everywhere
    (the A/B rerun knob).
    """
    from silica.config import CONFIG
    if (getattr(CONFIG, "nucleate_lane", "outline") or "outline") != "outline":
        return "keyphrase"
    return "keyphrase" if (profile or "default") in ("clip", "promotion") else "outline"


# ------------------------------------------------------------------ parsing --
def _clean_title(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().strip("*").strip()


def parse_outline(obj: dict) -> Outline:
    """Model JSON -> Outline. Drops duplicate titles, unknown dependency
    targets, self-dependencies and relations outside the closed set."""
    ideas: list[Idea] = []
    seen: set[str] = set()
    for raw in obj.get("ideas") or []:
        if not isinstance(raw, dict):
            continue
        title = _clean_title(raw.get("title"))
        if not title or title.casefold() in seen:
            continue
        seen.add(title.casefold())
        ideas.append(Idea(title=title, section=_clean_title(raw.get("section")),
                          claim=str(raw.get("claim") or "").strip(),
                          body=str(raw.get("body") or ""),
                          depends_on=list(raw.get("depends_on") or [])))
    by_key = {i.title.casefold(): i.title for i in ideas}
    for idea in ideas:
        kept = []
        for d in idea.depends_on:
            if not isinstance(d, dict):
                continue
            target = by_key.get(_clean_title(d.get("title")).casefold())
            rel = str(d.get("relation") or "").strip()
            if not target or target == idea.title or rel not in RELATIONS:
                continue
            kept.append({"title": target, "relation": rel, "why": str(d.get("why") or "").strip()})
        idea.depends_on = kept
    spine = [by_key[_clean_title(t).casefold()] for t in (obj.get("spine") or [])
             if _clean_title(t).casefold() in by_key]
    spine += [i.title for i in ideas if i.title not in spine]
    return Outline(lesson_title=str(obj.get("lesson_title") or "").strip(), spine=spine, ideas=ideas)


_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.M)
_NOISE_HEADING_RE = re.compile(
    r"@|^\s*(dove|where|note|notes)\s*:?\s*$|\b\d+\s*cfu\b|^\s*lezione\s+\d+\s*$", re.I)


def source_headings(md: str) -> list[str]:
    """Substantive headings of the source, in order, once each.

    Noise is a closed list (author line, email, course header, "dove:"): the
    coverage pass asks the model about every heading left, so a false
    positive here costs one wasted question, a false negative a silent gap.
    """
    out: list[str] = []
    seen: set[str] = set()
    for h in _HEADING_RE.findall(md):
        h = _clean_title(h)
        if not h or _NOISE_HEADING_RE.search(h) or h.casefold() in seen:
            continue
        seen.add(h.casefold())
        out.append(h)
    return out


def _sec_key(s: str) -> str:
    return re.sub(r"[^\w]+", " ", _clean_title(s).casefold()).strip()


def coverage_gaps(outline: Outline, headings: list[str]) -> list[str]:
    """Headings no idea claims as its section (and no skip explains)."""
    claimed = {_sec_key(i.section) for i in outline.ideas} | {_sec_key(s.get("section", "")) for s in outline.skips}
    gaps = []
    for h in headings:
        k = _sec_key(h)
        if k in claimed or any(k and (k in c or c in k) for c in claimed if c):
            continue
        gaps.append(h)
    return gaps


# -------------------------------------------------------------------- edges --
def select_edges(raw: list, *, ideas: set[str], existing: dict[str, str],
                 spine_titles: AbstractSet[str] = frozenset()) -> list[dict]:
    """Stage B edges the vault will act on, in proposal order."""
    kept: list[dict] = []
    per_target: dict[str, int] = {}
    # Casefold lookups: the model re-cases titles ("Support Vector Machines"
    # for a note filed as "Support vector machines"), and an exact match threw
    # every edge of one live run away (2026-09-02, 0 of the proposals kept).
    def _endpoint_key(raw: object) -> str:
        # The model echoes the outline ROW it read ("Lezione 11 | Margine
        # geometrico") instead of the title alone: measured 2026-09-02, all 22
        # proposed edges of one run died on exact matching. Take the last
        # pipe-separated field, which is the title in every row shape we emit.
        return _clean_title(str(raw or "").split("|")[-1]).casefold()

    idea_by_key = {t.casefold(): t for t in ideas}
    existing_by_key = {t.casefold(): t for t in existing}
    spine_keys = {t.casefold() for t in spine_titles}
    dropped = 0
    for e in raw or []:
        if not isinstance(e, dict):
            continue
        src = idea_by_key.get(_endpoint_key(e.get("from")))
        tgt = existing_by_key.get(_endpoint_key(e.get("to")))
        rel = str(e.get("relation") or "").strip()
        why = str(e.get("why") or "").strip()
        if not src or not tgt or tgt.casefold() in spine_keys:
            dropped += 1
            logger.debug("outline stage B: dropped edge %r -> %r (unknown endpoint or spine)", e.get("from"), e.get("to"))
            continue
        if rel not in CROSS_RELATIONS or len(why) < MIN_WHY_CHARS:
            dropped += 1
            logger.debug("outline stage B: dropped edge %r -> %r (relation %r, why %d chars)", src, tgt, rel, len(why))
            continue
        if per_target.get(tgt, 0) >= MAX_EDGES_PER_TARGET:
            continue
        per_target[tgt] = per_target.get(tgt, 0) + 1
        kept.append({"from": src, "to": tgt, "relation": rel, "why": why})
    logger.info("outline stage B: %d edge(s) proposed, %d kept, %d dropped", len(raw or []), len(kept), dropped)
    return kept


# ---------------------------------------------------------------------- ops --
def _stem(basename: str) -> str:
    return os.path.splitext(os.path.basename(basename))[0]


def outline_ops(outline: Outline, *, target: str, hub: str | None, source_basename: str,
                edges: list[dict], existing: dict[str, str],
                only_titles: set[str] | None = None) -> list[dict]:
    """Outline + edges -> raw ops in the distiller's own shape.

    One write per idea (section, parent = the spine note, typed relations in
    the body), one write for the spine, one patch per cross edge on the
    existing note. A `same_as` edge is NOT a merge: the new note lands
    flagged with the near-title verdict format, so WRITE's settle path hands
    the pair to the dedup judge with a real loser to mark (the prototype
    merged blind and collapsed a variant into its parent).
    """
    tgt = target.rstrip("/")
    spine_title = _stem(source_basename)
    ops: list[dict] = []
    by_from: dict[str, list[dict]] = {}
    for e in edges:
        by_from.setdefault(e["from"], []).append(e)
    for idea in outline.ideas:
        if only_titles is not None and idea.title not in only_titles:
            continue
        lines = [f"- {RELATION_LABEL[d['relation']]} [[{d['title']}]]: {d['why']}".rstrip(": ")
                 for d in idea.depends_on]
        related = [d["title"] for d in idea.depends_on]
        review = None
        for e in by_from.get(idea.title, []):
            if e["relation"] == "same_as":
                review = f"near_title candidate='{e['to']}' path='{existing[e['to']]}' ratio=1.00"
                continue
            lines.append(f"- {RELATION_LABEL[e['relation']]} [[{e['to']}]] ({spine_title} -> earlier): {e['why']}")
            related.append(e["to"])
        body = f"> {idea.claim}\n\n{idea.body.strip()}\n" if idea.claim else idea.body.strip() + "\n"
        if lines:
            body += "\n## Relations\n" + "\n".join(lines) + "\n"
        op = {
            "op": "write", "path": f"{tgt}/{idea.title}.md", "heading": idea.title, "title": idea.title,
            "snippet": body, "source_basename": source_basename, "hub": hub, "parent": spine_title,
            "related": list(dict.fromkeys(related)), "section": idea.section,
        }
        if review:
            op["review"] = review
        ops.append(op)
    if only_titles is None:
        rows = [f"{n}. [[{t}]]: {next(i.claim for i in outline.ideas if i.title == t)}"
                for n, t in enumerate(outline.spine, 1)]
        head = f"> {outline.lesson_title}\n\n" if outline.lesson_title else ""
        ops.append({
            "op": "write", "path": f"{tgt}/{spine_title}.md", "heading": spine_title, "title": spine_title,
            "snippet": head + "\n".join(rows) + "\n", "source_basename": source_basename, "hub": hub,
            "section": SPINE_SECTION,
        })
        for e in edges:
            if e["relation"] == "same_as":
                continue
            ops.append({
                "op": "patch", "path": existing[e["to"]], "heading": e["to"],
                "snippet": f"- [[{e['from']}]] {RELATION_LABEL[e['relation']]} this ({spine_title}): {e['why']}\n",
                "source_basename": source_basename,
            })
    return ops


MAX_SECTION_EXCERPT = 12000  # a lecture section; the span-grounding gate needs the formulas whole, not a 2k window


def section_text(source: str, heading: str) -> str:
    """The source section under `heading` (to the next heading of the same or
    higher level, every repeat of the heading joined), or "" when absent."""
    key = _sec_key(heading)
    if not key:
        return ""
    parts: list[str] = []
    for m in _HEADING_RE.finditer(source):
        if _sec_key(m.group(1)) != key:
            continue
        level = len(m.group(0)) - len(m.group(0).lstrip("#"))
        nxt = re.compile(rf"^#{{1,{level}}}\s", re.M).search(source, m.end())
        parts.append(source[m.start():nxt.start() if nxt else len(source)].strip())
    return "\n\n".join(parts)


def concept_entries(ops: list[dict], source: str = "") -> list[dict]:
    """Payload concept rows that make VALIDATE hold for these ops, in the shape
    `payload.build_concept_entry` emits (name, excerpt, hint, collision).

    The excerpt is the idea's own source section, whole: the span-grounding
    gate and the extractive floor compare bodies against it, and the
    keyphrase pipeline's 450-char windows are what flagged verbatim formulas
    as "not grounded" (7 of 24 notes on the 2026-09-02 before arm). A section
    the source does not have falls back to the whole document."""
    entries: list[dict] = []
    seen: set[str] = set()
    for op in ops:
        name = op.get("heading")
        if not name or name in seen:
            continue
        seen.add(name)
        collision = {"path": op["path"], "match_type": "title", "total_hits": 1, "excerpt": ""} \
            if op.get("op") == "patch" else None
        excerpt = section_text(source, op.get("section") or "") or source
        entries.append({"name": name, "action_hint": "enrich" if collision else "create",
                        "inbox_excerpt": excerpt[:MAX_SECTION_EXCERPT], "vault_collision": collision})
    return entries


# ------------------------------------------------------------ vault outline --
_BLOCKQUOTE_RE = re.compile(r"^>\s*(.+?)\s*$", re.M)


def _claim_of(body: str) -> str:
    m = _BLOCKQUOTE_RE.search(body)
    if m:
        return m.group(1)
    prose = re.sub(r"\$\$.*?\$\$", " ", body, flags=re.S)          # display math first: it spans lines
    prose = re.sub(r"^#.*$|!\[\[.*?\]\]", " ", prose, flags=re.M)   # headings and embeds, line-bound
    prose = re.sub(r"\s+", " ", prose).strip()
    return (re.split(r"(?<=[.!?])\s", prose, 1)[0] or "")[:200]


def vault_outline(target_dir: str, *, exclude_titles: AbstractSet[str] = frozenset()) -> list[dict]:
    """Rows {title, claim, lesson, path} for every idea note under `target_dir`.

    Folder-scoped by design: stage B reads titles + one-line claims, and a
    folder is the unit the user nucleates into. Spine notes and the hub are
    excluded (see SPINE_SECTION). Best-effort per note: an unreadable note is
    one missing row, never a failed run.
    """
    from silica.driver import DRIVER
    from silica.kernel.write import frontmatter
    from silica.tools.atomic import notes_under

    rows: list[dict] = []
    for rel in notes_under(target_dir):
        title = _stem(rel)
        if title in exclude_titles:
            continue
        try:
            content = DRIVER.read_note(rel).content or ""
            data, _raw, body = frontmatter.split(content)
        except Exception as e:  # one bad note must not blind the whole stage
            logger.debug("vault_outline: skipping %s (%s)", rel, e)
            continue
        data = data if isinstance(data, dict) else {}
        if str(data.get("section") or "") == SPINE_SECTION:
            continue
        if data.get("superseded_by"):
            # A merge loser shown as a live note draws the model's same_as
            # edges onto the stub instead of the winner (Lezione 8, f30ace50).
            continue
        srcs = data.get("sources") or []
        if isinstance(srcs, str):
            srcs = [srcs]
        lesson = _stem(str(srcs[0])) if srcs else ""
        rows.append({"title": title, "claim": _claim_of(body or ""), "lesson": lesson, "path": rel})
        if len(rows) >= MAX_OUTLINE_ROWS:
            break
    return rows


# -------------------------------------------------------------- the runner --
def _default_ask(system: str, user: str, *, max_tokens: int) -> dict:
    """One JSON reply from the worker model, no tools, no thinking.

    reasoning=False: measured 2026-09-02 on deepseek-v4-flash, the thinking
    trace ate the whole completion budget and the reply carried no JSON.
    """
    import json as _json
    import threading
    from silica.agent.providers import get_provider
    from silica.config import CONFIG
    from silica.kernel.prep_delegation import _call_with_deadline
    provider = get_provider(CONFIG, role="worker")
    schema_for_retry = next((m for tag, m in SCHEMA_BY_TAG.items() if system.startswith(tag)), None)
    last = ""
    for attempt in range(2):
        # Free JSON first, constrained decode only on the retry: measured
        # 2026-09-02, the free reply closed lecture 12's outline in 1.3k
        # tokens (7/7 calls in the prototype, 9/9 in the first product runs)
        # while the schema-constrained one repeated itself past 12k tokens
        # twice. The two fail differently, which is what makes the retry a
        # second chance rather than the same ask again.
        schema = schema_for_retry if attempt else None
        # Wall-clock bound, same helper and knob as run_distiller: a dead
        # upstream keeps the socket alive with keep-alive bytes, and the first
        # live run of this lane (2026-09-02) sat ten minutes on one such call.
        abandoned = threading.Event()
        resp = _call_with_deadline(lambda: provider.call_llm(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=None, response_schema=schema, max_tokens=max_tokens, temperature=0.0,
            reasoning=False, openrouter_provider=CONFIG.openrouter_provider_distiller, cancel=abandoned,
        ), float(os.getenv("DISTILLER_TIMEOUT", "300")), abandoned)
        if resp is None:
            raise TimeoutError("outline stage call abandoned")
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (resp.text or "").strip())
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return _json.loads(text[start:end + 1])
            except ValueError:
                pass
        last = f"finish={resp.finish_reason} head={text[:200]!r}"
        logger.warning("outline stage reply unusable (attempt %d): %s", attempt + 1, last)
        if resp.finish_reason == "length":
            max_tokens *= 2  # a cut JSON: the same ask with room to finish
    raise ValueError(f"outline stage returned no JSON after retry ({last})")


def _strip_images(md: str) -> str:
    return re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)


def run_outliner(*, source_text: str, source_basename: str, target: str, hub: str | None,
                 language: str, vault_outline: list[dict],
                 ask: Callable[..., dict] | None = None,
                 outline: Outline | None = None, only_titles: set[str] | None = None,
                 steer_context: str | None = None) -> dict:
    """The lane's DELEGATE: source text -> {"updates", "ephemerals", "concepts", "outline", "gaps"}.

    `outline` + `only_titles` is the steer retry: VALIDATE rejected some
    bodies, so only those are regenerated (with the feedback appended) and the
    outline, spine and edges are not paid for again.
    """
    ask = ask or _default_ask
    src = _strip_images(source_text)
    rels = ", ".join(RELATIONS)
    if outline is None:
        outline = parse_outline(ask(STAGE_A.format(lang=language, rels=rels), src, max_tokens=8000))
        gaps = coverage_gaps(outline, source_headings(src))
        if gaps:
            extra = ask(STAGE_GAP.format(gaps="\n".join(f"- {g}" for g in gaps)),
                        "IDEAS SO FAR:\n" + "\n".join(f"{i.title} | {i.section}" for i in outline.ideas)
                        + "\n\nSOURCE:\n" + src, max_tokens=4000)
            merged = parse_outline({"lesson_title": outline.lesson_title, "spine": outline.spine,
                                    "ideas": [i.__dict__ for i in outline.ideas] + list(extra.get("ideas") or [])})
            merged.skips = [s for s in (extra.get("skips") or []) if isinstance(s, dict)]
            outline = merged
    titles = [i.title for i in outline.ideas if only_titles is None or i.title in only_titles]
    for k in range(0, len(titles), BODIES_BATCH):
        batch = titles[k:k + BODIES_BATCH]
        user = "SOURCE:\n" + src + "\n\nIDEAS:\n" + "\n".join(
            f"{t} | {next(i.section for i in outline.ideas if i.title == t)}" for t in batch)
        if steer_context:
            user += "\n\nPREVIOUS ATTEMPT WAS REJECTED:\n" + steer_context
        bodies = ask(STAGE_BODIES.format(lang=language), user, max_tokens=12000).get("bodies") or {}
        for idea in outline.ideas:
            if idea.title in batch:
                idea.body = str(bodies.get(idea.title) or "")
    edges: list[dict] = []
    existing = {r["title"]: r["path"] for r in vault_outline}
    if vault_outline and only_titles is None:
        reply = ask(STAGE_B.format(lang=language, rels=", ".join(CROSS_RELATIONS)),
                    "EXISTING NOTES:\n" + "\n".join(f"{r.get('lesson', '')} | {r['title']} | {r.get('claim', '')}" for r in vault_outline)
                    + "\n\nNEW SOURCE IDEAS:\n" + "\n".join(f"{i.title} | {i.claim}" for i in outline.ideas),
                    max_tokens=4000)
        edges = select_edges(reply.get("edges") or [], ideas=set(outline.titles()), existing=existing,
                             spine_titles={r["title"] for r in vault_outline if r.get("spine")})
    ops = outline_ops(outline, target=target, hub=hub, source_basename=source_basename,
                      edges=edges, existing=existing, only_titles=only_titles)
    return {
        "updates": ops, "ephemerals": [], "concepts": concept_entries(ops, src),
        "outline": {"lesson_title": outline.lesson_title, "spine": outline.spine,
                    "ideas": [i.__dict__ for i in outline.ideas], "skips": outline.skips},
        "gaps": coverage_gaps(outline, source_headings(src)),
    }
