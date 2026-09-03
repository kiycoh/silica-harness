# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Dedup capability — merge a borderline-duplicate concept into an existing note.

Given an incoming concept vs. an existing larger note, decide whether they are
the same concept and, if so, append only the genuinely-new information into the
existing note as a single ``patch`` under the dedup bounds.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel
from pydantic.json_schema import SkipJsonSchema

from silica.agent.commit import commit_ops
from silica.agent.bounds import dedup_bounds, dedup_spoke_bounds
from silica.kernel.write.ops import Op, OpType
from silica.kernel.workqueue import WorkItem
from silica.capabilities._base import emit_feedback, load_prompt, read_or_skip

logger = logging.getLogger(__name__)


# The three labels the judge may return; anything else is coerced to "distinct"
# at the parse boundary. (`DedupVerdict` below is the wire schema, not the label.)
VerdictLabel = Literal["duplicate", "distinct", "contradicts"]


class DedupVerdict(BaseModel):
    """Wire schema for the judge-only path — no authoring fields.

    Constrained decoding forces the model to emit every key the schema declares,
    so exposing title/body when no spoke was asked for made the judge re-author
    the note it was judging: hundreds of wasted tokens, and a truncated response
    (silent fallback verdict) whenever that regeneration outran max_tokens.
    """
    # duplicate    → append only the genuinely-new info
    # distinct     → pipeline concepts: author the spoke note in the same call
    #                (judge+author); ad-hoc pairs: no write
    # contradicts  → record the conflicting claim as a contested patch (never resolve)
    verdict: VerdictLabel = "distinct"
    # False when the verdict is the parse-failure default rather than the
    # model's answer. Off the wire schema: constrained decoding would
    # otherwise ask the model to fill it, and it is the framework's field.
    judged: SkipJsonSchema[bool] = True
    rationale: str = ""
    addition: str = ""


class DedupDecision(DedupVerdict):
    # Authored spoke (distinct + pipeline item only; empty otherwise).
    title: str = ""
    body: str = ""


class DedupVerdictBatch(BaseModel):
    """Batch wire schema for the judge-only path (no authoring fields)."""
    decisions: list[DedupVerdict] = []


class DedupBatchDecision(BaseModel):
    """One verdict per incoming concept, same order as presented."""
    decisions: list[DedupDecision] = []


def passes_dedup_gate(
    score: float,
    incoming_len: int,
    candidate_len: int,
    *,
    threshold: float = 0.85,
    max_ratio: float = 2.0,
) -> bool:
    """Cheap gate before the LLM judge (spec 2.1). True iff the effective cosine
    clears `threshold` AND the two bodies are within `max_ratio` in size. The
    size guard rejects the spoke-in-hub false positive (small spoke, big hub:
    high cosine, huge size gap). Cosine is symmetric, so the spec's mutual
    requirement is the single self-normalized score.
    """
    if score < threshold:
        return False
    small, large = sorted((max(incoming_len, 1), max(candidate_len, 1)))
    return large / small <= max_ratio


def run_dedup(item: WorkItem, config: Any) -> dict[str, Any]:
    ctx = item.context
    candidate_path = item.target_path
    budget = 8000

    emit_feedback(item, "reading")
    candidate_body, skip = read_or_skip(candidate_path)
    if skip is not None:
        return skip

    if item.cancel_token.is_set():
        return {"status": "cancelled"}

    if ctx.get("concepts"):
        return _run_batch(item, ctx["concepts"], candidate_body[:budget], config)

    if os.getenv("SILICA_DEDUP_GATE"):
        eff = max(ctx.get("full_score", ctx.get("score", 0.0)),
                  ctx.get("title_score", 0.0))
        if not passes_dedup_gate(eff, len(ctx.get("excerpt", "")),
                                 len(candidate_body[:budget])):
            gate_decision = DedupDecision(
                verdict="distinct",
                rationale="dedup gate: below threshold or size guard",
            )
            return _route_verdict(item, ctx, gate_decision, config)

    emit_feedback(item, "calling_llm")
    decision = _decide_dedup(
        config,
        concept=ctx.get("concept", ""),
        excerpt=ctx.get("excerpt", ""),
        candidate_name=ctx.get("candidate", candidate_path),
        candidate_body=candidate_body[:budget],
        score=ctx.get("score", 0.0),
        full_score=ctx.get("full_score", ctx.get("score", 0.0)),
        title_score=ctx.get("title_score", 0.0),
        author_spoke=bool(ctx.get("target_dir")),
        hub=ctx.get("hub"),
    )

    if item.cancel_token.is_set():
        return {"status": "cancelled"}

    return _route_verdict(item, ctx, decision, config)


def _run_batch(
    item: WorkItem, concepts: list[dict], candidate_body: str, config: Any
) -> dict[str, Any]:
    """Judge a family of concepts against one candidate in a single LLM call,
    then route every verdict through the exact same code as a single item."""
    ctx = item.context
    emit_feedback(item, "calling_llm")
    if os.getenv("SILICA_DEDUP_GATE"):
        decisions = _gated_batch_decisions(config, item, concepts, candidate_body)
    else:
        decisions = _decide_dedup_batch(
            config,
            concepts=concepts,
            candidate_name=ctx.get("candidate", item.target_path),
            candidate_body=candidate_body,
            author_spoke=bool(ctx.get("target_dir")),
            hub=ctx.get("hub"),
        )
    results: list[dict[str, Any]] = []
    followups: list[dict[str, Any]] = []
    for entry, decision in zip(concepts, decisions):
        if item.cancel_token.is_set():
            return {"status": "cancelled", "results": results}
        sub_ctx = {k: v for k, v in ctx.items() if k != "concepts"} | entry
        res = _route_verdict(item, sub_ctx, decision, config)
        fu = res.pop("followup", None)
        if isinstance(fu, dict):
            followups.append(fu)
        results.append({"concept": entry.get("concept", ""), **res})
    statuses = {r.get("status") for r in results}
    out: dict[str, Any] = {
        "status": statuses.pop() if len(statuses) == 1 else "partial",
        "batch": len(results),
        "results": results,
    }
    if followups:
        out["followups"] = followups
    return out


def _gated_batch_decisions(
    config: Any, item: WorkItem, concepts: list[dict], candidate_body: str
) -> list[DedupDecision]:
    """SILICA_DEDUP_GATE batch path: concepts failing the cheap cosine+size gate
    skip the LLM and are pre-judged 'distinct' (clearly not duplicates); the rest
    are judged in one batch call. Returned full-length and aligned to `concepts`,
    so the caller's zip stays correct and every gated-out concept still routes
    through the normal distinct path (authoring its spoke for pipeline items),
    identical to an LLM 'distinct'.
    """
    ctx = item.context
    to_judge: list[dict] = []
    prejudged: dict[int, DedupDecision] = {}
    for i, c in enumerate(concepts):
        eff = max(c.get("full_score", c.get("score", 0.0)) or 0.0,
                  c.get("title_score", 0.0) or 0.0)
        if passes_dedup_gate(eff, len(c.get("excerpt", "")), len(candidate_body)):
            to_judge.append(c)
        else:
            prejudged[i] = DedupDecision(
                verdict="distinct",
                rationale="dedup gate: below threshold or size guard",
            )
    judged = (
        _decide_dedup_batch(
            config,
            concepts=to_judge,
            candidate_name=ctx.get("candidate", item.target_path),
            candidate_body=candidate_body,
            author_spoke=bool(ctx.get("target_dir")),
            hub=ctx.get("hub"),
        )
        if to_judge else []
    )
    judged_iter = iter(judged)
    return [
        prejudged[i] if i in prejudged else next(judged_iter, DedupDecision())
        for i in range(len(concepts))
    ]


def _incoming_side(ctx: dict, source_basename: str) -> tuple[int, str | None]:
    """(tier, event clock) of the claim arriving from this source.

    NOT `reliability_tier` of the source text: a raw document has no `AI` key
    and would rank HUMAN, which says "the agent did not write this" and nothing
    at all about whether a person vouched for the claim (§6.1). §5's rule
    instead — grounded while the verbatim source is reachable, distilled once
    it is not. The clock is whatever the source actually states, never today.
    """
    from silica.kernel.recall.paths import SOURCES_DIR
    from silica.kernel.vault_manifest import in_write_dir
    from silica.kernel.write.contested import TIER_DISTILLED, TIER_GROUNDED
    from silica.kernel.write.provenance import source_event_date

    leaf, _err = read_or_skip(f"{in_write_dir(SOURCES_DIR)}/{source_basename}")
    if leaf:
        return TIER_GROUNDED, source_event_date(leaf)
    inbox_file = ctx.get("inbox_file") or ""
    source_text = (read_or_skip(inbox_file)[0] or "") if inbox_file else ""
    return TIER_DISTILLED, source_event_date(source_text)


def _pair_provenance(candidate_path: str, source_basename: str) -> dict[str, Any]:
    """Whether the merge joins two derivations of one source or two sources.

    Two notes from one source are distiller noise; two from different sources
    are a real convergence. The judge and its gates are untouched and the
    cross-file fusion path stays closed (see the CROSSDEDUP verdict): this is
    the ledger's answer riding on the result, for whoever reviews the merge.
    `unknown` when the ledger never saw the candidate, never a guess.
    """
    from silica.kernel.write.provenance import sources_of

    candidate_sources = sources_of(candidate_path)
    if not candidate_sources:
        relation = "unknown"
    elif source_basename in candidate_sources:
        relation = "same-source"
    else:
        relation = "cross-source"
    return {"candidate_sources": candidate_sources,
            "incoming_source": source_basename, "relation": relation}


def _file_dominated_claim(
    ctx: dict, path: str, decision: Any, source_basename: str, hub: str | None
) -> dict | None:
    """§6.1-bis: file a claim the note strictly outranks, instead of contesting it.

    Returns None when the contest must stay open, which is every case where the
    two sides rank equal, where the incoming claim wins, or where anything
    suggests the note is the stale side. The asymmetry is the design: this only
    ever suppresses in the one direction where the information suffices, and a
    contest left visible costs a human a decision, while one resolved wrongly
    buries a live claim.

    Priced on `evals/golden/fixtures/contests` before being written: bare tier
    dominance acts on 4 of 7 ranked contests and gets 2 wrong; with the recency
    veto in `suppress_contest` it acts on 2 and gets both right.
    """
    import datetime

    from silica.agent.bounds import dedup_supersede_bounds
    from silica.kernel.write.contested import (
        SUPERSEDED_HEADING,
        superseded_claim,
        suppress_contest,
    )
    from silica.kernel.write.moc import merge_moc_section

    prior, _err = read_or_skip(path)
    if not prior:
        return None
    incoming_tier, incoming_clock = _incoming_side(ctx, source_basename)
    if not suppress_contest(prior, incoming_tier=incoming_tier,
                            incoming_clock=incoming_clock):
        return None

    today = datetime.date.today().isoformat()
    filed = merge_moc_section(prior, SUPERSEDED_HEADING, superseded_claim(
        decision.addition, source_basename=source_basename,
        valid_from=incoming_clock, valid_to=today,
    ))
    result = commit_ops(
        [Op(
            op=OpType.overwrite,
            heading=ctx.get("concept", "") or "superseded claim",
            source_basename=source_basename,
            path=path,
            content=filed,
            base_content=prior,
            reason=f"contested: outranked on arrival ({decision.rationale[:80]})",
        )],
        target_dir=os.path.dirname(path),
        bounds=dedup_supersede_bounds(path, hub=hub),
    )
    if result.get("status") != "committed":
        return None  # bounds or a concurrent edit refused it: leave the contest open
    result.setdefault("rationale", decision.rationale)
    result.setdefault("verdict", "contradicts")
    result["superseded"] = True
    return result


def _mark_merge_loser(ctx: dict, winner_path: str) -> None:
    """Point the absorbed note at the one that took its content.

    Only fires when the loser is a real vault note: `loser_path` is set by the
    note-vs-note seams (/dedup pairs, /curate families) and by the soft gate
    once a flagged note has landed (write.py _settle_flagged); an incoming
    concept that was never written has none. Best-effort — a committed merge
    must not be reported failed over a bookkeeping key.
    """
    loser_path = ctx.get("loser_path")
    if not loser_path or loser_path == winner_path:
        return
    try:
        from silica.agent.bounds import dedup_supersede_bounds
        from silica.kernel.write.contested import mark_superseded_by

        prior = read_or_skip(loser_path)[0] or ""
        marked = mark_superseded_by(prior, winner_path)
        if marked == prior:
            return
        commit_ops(
            [Op(
                op=OpType.overwrite,
                heading=ctx.get("concept", "") or "merged note",
                source_basename=os.path.basename(loser_path),
                path=loser_path,
                content=marked,
                base_content=prior,
                reason="dedup merge: superseded_by pointer",
            )],
            target_dir=os.path.dirname(loser_path),
            bounds=dedup_supersede_bounds(loser_path),
        )
    except Exception as e:
        logger.debug("dedup: superseded_by mark failed (non-fatal): %s", e)


def _record_absorbed_alias(ctx: dict, winner_path: str, hub: str | None) -> None:
    """Keep the surface form of a concept that was merged away.

    Runs beside _mark_merge_loser, not instead of it: a landed loser keeps a
    superseded_by pointer that the write gate follows, but the pointer dies
    with the stub, and an incoming concept that was never written has no
    stub at all. Without the alias the heading is discarded at the merge —
    every later mention of that spelling stays unlinked and the concept
    quietly splits in two ("Percettrone", run f30ace50, 2026-09-03). Recorded
    as a frontmatter alias, so autolink resolves the spelling onto the note that
    absorbed it (build_alias_map + autolink(aliases=...)).

    Best-effort, like its sibling: a committed merge is not reported failed over
    a bookkeeping key. An alias that collides with a real note title is dropped
    downstream by build_alias_map, so no read is spent proving it here.
    """
    concept = (ctx.get("concept") or "").strip()
    winner_title = os.path.basename(winner_path).removesuffix(".md")
    if not concept or concept.lower() == winner_title.lower():
        return
    try:
        from silica.agent.bounds import dedup_alias_bounds
        from silica.kernel.write.frontmatter import add_alias

        prior = read_or_skip(winner_path)[0] or ""
        aliased = add_alias(prior, concept)
        if aliased == prior:
            return
        res = commit_ops(
            [Op(
                op=OpType.overwrite,
                heading=concept,
                source_basename=os.path.basename(winner_path),
                path=winner_path,
                content=aliased,
                base_content=prior,
                reason=f"dedup merge: '{concept}' kept as alias",
            )],
            target_dir=os.path.dirname(winner_path),
            bounds=dedup_alias_bounds(winner_path, hub=hub),
        )
        # commit_ops reports refusal in its status, it does not raise: without
        # this line a bounds rejection would be indistinguishable from success.
        if res.get("status") != "committed":
            logger.debug("dedup: alias not recorded (%s)", res.get("status"))
    except Exception as e:
        logger.debug("dedup: alias record failed (non-fatal): %s", e)


def _route_verdict(
    item: WorkItem, ctx: dict, decision: DedupDecision, config: Any
) -> dict[str, Any]:
    candidate_path = item.target_path

    if decision.verdict == "distinct":
        return _route_distinct(item, ctx, decision, config)

    if not decision.addition.strip():
        if decision.verdict == "duplicate":
            # A landed soft-gate note judged a pure duplicate: nothing to merge,
            # but the note is still on disk saying the same thing as the
            # candidate — point it at the winner (no-op without loser_path).
            _mark_merge_loser(ctx, candidate_path)
            _record_absorbed_alias(ctx, candidate_path, ctx.get("hub"))
            # The verdict is final, so the parked copy has nothing left to
            # wait for: release it, or the anneal re-validates it every run
            # until the TTL drops it (probe run 2026-09-02).
            _clean_twin_bundle(ctx)
        return {
            "status": "no_merge",
            "verdict": decision.verdict,
            "rationale": decision.rationale,
        }

    emit_feedback(item, "committing")
    hub = ctx.get("hub")
    inbox_file = ctx.get("inbox_file", "")
    source_basename = os.path.basename(inbox_file) if inbox_file else "dedup"
    if decision.verdict == "contradicts":
        filed = _file_dominated_claim(ctx, candidate_path, decision, source_basename, hub)
        if filed is not None:
            return filed
        from silica.kernel.write.contested import contested_callout
        op = Op(
            op=OpType.patch,
            heading=ctx.get("concept", "") or "contested claim",
            source_basename=source_basename,
            path=candidate_path,
            snippet=contested_callout(decision.addition, source_basename),
            hub=hub,
            reason=f"contested: {decision.rationale[:120]}",
            contested_by=f"source: {source_basename}",
        )
    else:
        op = Op(
            op=OpType.patch,
            heading=ctx.get("concept", "") or "merged concept",
            source_basename=source_basename,
            path=candidate_path,
            snippet=decision.addition,
            hub=hub,
            reason=f"dedup merge: {decision.rationale[:120]}",
        )
    bounds = dedup_bounds(candidate_path, hub=hub)
    result = commit_ops(
        [op],
        target_dir=os.path.dirname(candidate_path),
        hub=hub,
        bounds=bounds,
    )
    result.setdefault("rationale", decision.rationale)
    result.setdefault("verdict", decision.verdict)
    result.setdefault("provenance", _pair_provenance(candidate_path, source_basename))
    if result.get("status") == "committed":
        _clean_twin_bundle(ctx)
        if decision.verdict == "duplicate":
            if ctx.get("loser_path"):
                _mark_merge_loser(ctx, candidate_path)
            # The alias is recorded with or without a loser note: the loser's
            # pointer is read by the write gate only while the stub exists,
            # the alias on the winner is what outlives its deletion. Before
            # 2026-09-03 the branches were exclusive and "Percettrone" left
            # no trace on "Percettrone di Rosenblatt" (run f30ace50).
            _record_absorbed_alias(ctx, candidate_path, hub)
        if decision.verdict == "contradicts":
            # Without this the judge's contradictions never reach the run
            # digest worklist: only silica_flag_note used to feed the register,
            # so LLM-detected ones were visible solely to a full-vault scan.
            try:
                from silica.kernel import contested_register
                contested_register.add(candidate_path)
            except Exception as e:
                logger.debug("dedup: contested register add failed (non-fatal): %s", e)
    return result


def _route_distinct(
    item: WorkItem, ctx: dict, decision: DedupDecision, config: Any
) -> dict[str, Any]:
    """Distinct verdict routing (C2): the borderline concept becomes a spoke.

    Pipeline items (context carries ``target_dir``) commit the spoke the judge
    authored in the verdict call — or, when authoring failed, a mechanical
    write of the excerpt verbatim with provenance, refined right after
    (ADR-0001: mechanical inject + deferred refine). The parked twin bundle is
    cleaned only on verified commit, so the op degrades but is never lost.

    Ad-hoc pairs (two existing notes, no ``target_dir``) keep the historical
    contract: distinct → no write.
    """
    target_dir = ctx.get("target_dir", "")
    no_merge = {"status": "no_merge", "verdict": "distinct", "rationale": decision.rationale}
    if not target_dir:
        return no_merge
    if ctx.get("loser_path"):
        return _settle_landed_distinct(item, ctx, decision)

    from silica.kernel.write.templates import (
        has_related_trace, related_trace, related_unjudged, slugify,
    )

    concept = ctx.get("concept", "")
    candidate_name = ctx.get("candidate", "")
    inbox_file = ctx.get("inbox_file", "")
    source_basename = os.path.basename(inbox_file) if inbox_file else "dedup"
    hub = ctx.get("hub")

    title = decision.title.strip()
    body = decision.body.strip()
    mechanical = not (title and body)
    if mechanical:
        excerpt = (ctx.get("excerpt") or "").strip()
        if not excerpt:
            return no_merge  # nothing to materialize the spoke from
        title = concept or candidate_name
        body = f"{excerpt}\n\n*(from {source_basename})*"
    # The framework, not the model, guarantees the spoke is born linked — and
    # the judged relation survives as a typed, parseable trace (survey-
    # provenance spec, Lane A): the judge call is already paid, this line is
    # the only record of its verdict. In-body rather than a post-commit patch,
    # so the trace exists iff the spoke does and re-runs cannot stack copies.
    # A parse-failure default is not a verdict: the spoke is still born
    # linked, but no "judged distinct" record is written for a pair nobody
    # judged, and the fallback's diagnostic string stays out of vault copy
    # (2026-08-23: "(judged distinct: unparseable decision)" on a spoke).
    if candidate_name and not has_related_trace(body, candidate_name):
        body += "\n\n" + (
            related_trace(candidate_name, decision.rationale) if decision.judged
            else related_unjudged(candidate_name)
        )

    emit_feedback(item, "committing")
    spoke_path = f"{target_dir}/{slugify(title) or title}.md"
    op = Op(
        op=OpType.write,
        heading=concept or title,
        source_basename=source_basename,
        path=spoke_path,
        title=title,
        snippet=body,
        hub=hub,
        reason=f"dedup distinct spoke: {decision.rationale[:120]}",
    )
    result = commit_ops(
        [op],
        target_dir=target_dir,
        hub=hub,
        bounds=dedup_spoke_bounds(spoke_path, hub=hub),
    )
    result.setdefault("verdict", "distinct")
    result.setdefault("rationale", decision.rationale)
    result["spoke_path"] = spoke_path
    if result.get("status") == "committed":
        _clean_twin_bundle(ctx)
        if mechanical:
            # ADR-0001: mechanical inject + deferred refine. Capabilities are
            # peers (P9) — dedup proposes the follow-up; the BoundedSubAgent
            # engine dispatches it through the registry.
            result["followup"] = {
                "kind": "refine",
                "target_path": spoke_path,
                "context": {"hub": hub} if hub else {},
            }
    return result


def _settle_landed_distinct(item: WorkItem, ctx: dict, decision: DedupDecision) -> dict[str, Any]:
    """`distinct` on a note that landed under a soft gate: the flag has done
    its job. Drop `review:` and leave the judged relation as the same typed
    trace a spoke gets, so the verdict survives in the note and nothing is
    written a second time under a slugified sibling path.
    """
    from silica.agent.bounds import dedup_settle_bounds
    from silica.kernel.write import frontmatter
    from silica.kernel.write.templates import has_related_trace, related_trace, related_unjudged

    path = ctx["loser_path"]
    prior = read_or_skip(path)[0] or ""
    data, raw, body = frontmatter.split(prior)
    if data is None:
        if raw is not None:  # broken YAML: never round-trip what cannot be parsed
            return {"status": "no_merge", "verdict": "distinct", "rationale": decision.rationale}
        data, body = {}, prior
    data.pop("review", None)
    candidate_name = ctx.get("candidate", "")
    if candidate_name and not has_related_trace(body, candidate_name):
        body = body.rstrip("\n") + "\n\n" + (
            related_trace(candidate_name, decision.rationale) if decision.judged
            else related_unjudged(candidate_name)
        ) + "\n"
    emit_feedback(item, "committing")
    result = commit_ops(
        [Op(
            op=OpType.overwrite,
            heading=ctx.get("concept", "") or "judged distinct",
            source_basename=os.path.basename(path),
            path=path,
            content=frontmatter.dump(data, body),
            base_content=prior,
            reason=f"dedup distinct, flag cleared: {decision.rationale[:120]}",
        )],
        target_dir=os.path.dirname(path),
        hub=ctx.get("hub"),
        bounds=dedup_settle_bounds(path, hub=ctx.get("hub")),
    )
    result.setdefault("verdict", "distinct")
    result.setdefault("rationale", decision.rationale)
    if result.get("status") == "committed":
        _clean_twin_bundle(ctx)
    return result


def _clean_twin_bundle(ctx: dict) -> None:
    """Drop this concept's op from the deferred bundle COLLISION parked.

    Called only after a verified commit: the verdict has been routed into the
    vault, so the parked copy is no longer the durable one. Best-effort — a
    missing bundle (retry already flushed it, pre-C2 stub) is not an error.
    """
    content_hash = ctx.get("content_hash", "")
    if not content_hash:
        return
    try:
        from silica.kernel.recall.deferred import get_deferred_store
        get_deferred_store().remove_op(content_hash, ctx.get("concept", ""))
    except Exception as e:
        logger.debug("dedup: twin bundle cleanup failed (non-fatal): %s", e)


def _decide_dedup(
    config: Any,
    *,
    concept: str,
    excerpt: str,
    candidate_name: str,
    candidate_body: str,
    score: float = 0.0,
    full_score: float = 0.0,
    title_score: float = 0.0,
    author_spoke: bool = False,
    hub: str | None = None,
) -> DedupDecision:
    from silica.agent.providers import get_provider
    from silica.kernel.text.sanitize import parse_json

    prompt = load_prompt("dedup_prompt.txt")
    if author_spoke:
        # Judge+author (C2): the same call that judges "distinct" also
        # authors the spoke note — a second pass would just re-read the
        # context this call already has.
        hub_hint = f" and to the parent note [[{hub}]]" if hub else ""
        prompt += (
            "\n\nIf (and only if) your verdict is \"distinct\", ALSO author the new note"
            " for the INCOMING CONCEPT in the same response, adding two more JSON keys:"
            "\n  \"title\" — clean note name (no extension, no quotes)."
            f"\n  \"body\" — well-formed Obsidian Markdown grounded ONLY in the incoming"
            f" excerpt (never invent facts); no top-level heading; include a wikilink"
            f" to [[{candidate_name}]]{hub_hint}."
            "\nFor any other verdict leave \"title\" and \"body\" empty."
        )

    score_block = _score_block(score, full_score, title_score)

    user_message = (
        f"{prompt}\n\n"
        f"---\n{score_block}\n"
        f"---\nCANDIDATE NOTE ({candidate_name}):\n{candidate_body}\n\n"
        f"---\nINCOMING CONCEPT: {concept}\nEXCERPT:\n{excerpt}\n"
    )
    provider = get_provider(config, role="worker")
    response = provider.call_llm(
        messages=[{"role": "user", "content": user_message}],
        tools=None,
        response_schema=DedupDecision if author_spoke else DedupVerdict,
        max_tokens=2048,
        # The verdict is a few hundred tokens of JSON; with thinking on, a
        # hybrid model spends the budget on its trace and the JSON arrives
        # cut (finish=length, "Unterminated string", 2026-08-23 run on
        # deepseek-v4-flash). That trace had concluded "duplicate", and the
        # parse fallback then wrote "distinct". Same knob as residue.py.
        reasoning=False,
    )
    raw = response.text or ""
    try:
        parsed, _ = parse_json(raw, strict=False)
        if isinstance(parsed, dict):
            raw_verdict = parsed.get("verdict")
            verdict: VerdictLabel
            if raw_verdict not in ("duplicate", "distinct", "contradicts"):
                # Legacy binary schema, or anything unrecognised → conservative.
                legacy = parsed.get("is_duplicate")
                verdict = "duplicate" if legacy is True else "distinct"
            else:
                verdict = raw_verdict
            return DedupDecision(
                verdict=verdict,
                rationale=str(parsed.get("rationale", "")),
                addition=str(parsed.get("addition", "")),
                title=str(parsed.get("title", "") or ""),
                body=str(parsed.get("body", "") or ""),
            )
    except Exception as e:
        logger.debug("dedup decision parse failed: %s", e)
    # Conservative default: when in doubt, do not merge and do not contest.
    return DedupDecision(verdict="distinct", rationale="unparseable decision", judged=False)


def _score_block(score: float, full_score: float, title_score: float) -> str:
    # When both metrics are available we surface them separately so the model
    # can interpret the signal correctly: a high title score with a low body
    # score means "topically related but distinct" — very different from a
    # uniformly high score which strongly suggests a true duplicate.
    if title_score > 0.0 and full_score > 0.0:
        return (
            f"SEMANTIC CLOSENESS SCORE: {score:.3f} (effective = max of the two below)\n"
            f"  • Full-note similarity (body + title):  {full_score:.3f}\n"
            f"  • Title-only similarity:                {title_score:.3f}\n"
            f"Interpretation:\n"
            f"  - A high score means the two texts READ alike, not that they state the\n"
            f"    same things. Sibling notes filled from one template (same tables,\n"
            f"    parallel prose) score high while differing exactly where it matters.\n"
            f"  - High title score with low body score: topically related but distinct\n"
            f"    (e.g. 'ROS' vs 'JSON in ROS 2') → prefer linking over merging.\n"
            f"  - Judge the claims, never the layout: 'duplicate' only when the two texts\n"
            f"    assert the same things about the same subject."
        )
    return (
        f"SEMANTIC CLOSENESS SCORE: {score:.3f} (0.0 to 1.0, where 1.0 is identical)\n"
        f"Use this metric as an indicator only: a high score means the texts read alike,\n"
        f"not that they state the same things. Judge the claims, never the layout."
    )


def _decide_dedup_batch(
    config: Any,
    *,
    concepts: list[dict],
    candidate_name: str,
    candidate_body: str,
    author_spoke: bool = False,
    hub: str | None = None,
) -> list[DedupDecision]:
    """One LLM call judging every concept of a family against the candidate.

    The single-verdict prompt is reused verbatim; batch mode only appends the
    array contract and the numbered concept blocks, so the judging criteria
    cannot drift between the per-item and the family path.
    """
    from silica.agent.providers import get_provider

    prompt = load_prompt("dedup_prompt.txt")
    n = len(concepts)
    batch_note = (
        f"\n\nBATCH MODE: below are {n} INCOMING CONCEPTS, all matched against the SAME"
        " candidate note. Judge each one INDEPENDENTLY — verdicts within a batch may"
        " differ. Respond with JSON: {\"decisions\": [...]} containing EXACTLY"
        f" {n} entries, in the same order as the concepts, each with the single-verdict"
        " schema (verdict, rationale, addition)."
    )
    if author_spoke:
        hub_hint = f" and to the parent note [[{hub}]]" if hub else ""
        batch_note += (
            "\nFor every entry whose verdict is \"distinct\", ALSO author the new note"
            " in that entry, adding \"title\" (clean note name, no extension) and"
            " \"body\" (well-formed Obsidian Markdown grounded ONLY in that concept's"
            " excerpt — never invent facts; no top-level heading; include a wikilink"
            f" to [[{candidate_name}]]{hub_hint}). For any other verdict leave"
            " \"title\" and \"body\" empty."
        )
    blocks = []
    for i, c in enumerate(concepts, 1):
        score = c.get("score") or 0.0
        blocks.append(
            f"---\nINCOMING CONCEPT {i}/{n}: {c.get('concept', '')}\n"
            f"{_score_block(score, c.get('full_score') or score, c.get('title_score') or 0.0)}\n"
            f"EXCERPT:\n{c.get('excerpt', '')}\n"
        )
    user_message = (
        f"{prompt}{batch_note}\n\n"
        f"---\nCANDIDATE NOTE ({candidate_name}):\n{candidate_body}\n\n"
        + "\n".join(blocks)
    )
    provider = get_provider(config, role="worker")
    response = provider.call_llm(
        messages=[{"role": "user", "content": user_message}],
        tools=None,
        response_schema=DedupBatchDecision if author_spoke else DedupVerdictBatch,
        max_tokens=2048 * n,
        reasoning=False,  # same budget arithmetic as the single path
    )
    return _parse_batch(response.text or "", n)


def _parse_batch(raw: str, n: int) -> list[DedupDecision]:
    """Positional decisions, padded/truncated to exactly n.

    A missing or unparseable entry degrades to the same conservative default
    as the single path (distinct, no authorship → mechanical spoke or
    no_merge downstream) — never to a merge.
    """
    from silica.kernel.text.sanitize import parse_json

    def fallback() -> DedupDecision:
        return DedupDecision(
            verdict="distinct", rationale="missing from batch response", judged=False)

    decisions: list[DedupDecision] = []
    try:
        parsed, _ = parse_json(raw, strict=False)
        entries = parsed.get("decisions") if isinstance(parsed, dict) else parsed
        if isinstance(entries, list):
            for e in entries[:n]:
                if not isinstance(e, dict):
                    decisions.append(fallback())
                    continue
                raw_verdict = e.get("verdict")
                verdict: VerdictLabel
                if raw_verdict not in ("duplicate", "distinct", "contradicts"):
                    # Legacy binary schema, or anything unrecognised → conservative.
                    verdict = "duplicate" if e.get("is_duplicate") is True else "distinct"
                else:
                    verdict = raw_verdict
                decisions.append(DedupDecision(
                    verdict=verdict,
                    rationale=str(e.get("rationale", "")),
                    addition=str(e.get("addition", "")),
                    title=str(e.get("title", "") or ""),
                    body=str(e.get("body", "") or ""),
                ))
    except Exception as e:
        logger.debug("dedup batch parse failed: %s", e)
    while len(decisions) < n:
        decisions.append(fallback())
    return decisions
