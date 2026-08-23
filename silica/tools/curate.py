# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Curator dispatch — /curate: vault maintenance as a background policy.

The pure composer (silica.kernel.recall.curator) turns an L1 VaultReport into a typed
CurationPlan. This module executes that plan on the *existing* machinery:

  * strong autolink candidate → the mechanical, LLM-free silica_autolink path
    (graph-safe direct commit);
  * orphan / dedup / refine    → WorkItems drained through run_subagent_batch,
    the same leashed-sub-agent seam /dedup and /refine already use — so every
    write goes through commit_ops + bounds + the snapshot/rollback undo path.

The curator gains no new power: only initiative. `silica_curate` defaults to a
dry-run (compose + return the plan, enqueue and write nothing); `apply=True`
routes through `apply_curation_plan`, which also appends one idempotent journal
line via run_log.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from silica.agent.subagent import run_subagent_batch
from silica.kernel.recall.curator import CurationPlan, compose_curation_plan
from silica.kernel.report.graph_report import compute_report
from silica.kernel.recall.run_log import append_log_line, format_curate_event
from silica.kernel.workqueue import WorkItem
from silica.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers (patched wholesale in tests so no driver/index/LLM is touched)
# ---------------------------------------------------------------------------

def _read_body(path: str) -> str:
    """Note body, or "" on any error (a missing note simply yields no excerpt).

    Stale guard (spec-stale-triggers §4): curation reads bypass the read-gate
    banner, so a stale doc note gets one prefix line here, served from the
    read-only peek. Annotate only — the model decides, nothing blocks."""
    try:
        from silica.driver import DRIVER

        content = DRIVER.read_note(path).content or ""
    except Exception:
        return ""
    try:
        from silica.config import CONFIG
        from silica.kernel.code import codedocs

        lvl = codedocs.peek_level(codedocs.peek(CONFIG.vault_path), path)
        if lvl and content:
            return (f"[stale] {lvl}: verify against the source before "
                    f"reusing claims\n\n{content}")
    except Exception:
        pass  # the guard is an aid; the read succeeds without it
    return content


def _run_autolink(sources: list[str]) -> dict[str, Any]:
    """Mechanical, LLM-free autolink of the given source notes (direct commit)."""
    from silica.tools.graph import silica_autolink

    return silica_autolink(note_paths=list(dict.fromkeys(sources)))



def link_candidates(path: str, k: int = 5) -> list[dict]:
    """Link targets to offer for an orphan note: ``[{"name", "path"}, ...]``.

    Fuses embeddings + co-occurrence (RRF): pure candidate generation, no cosine
    thresholding, so it still produces targets when the embedding index is empty
    (the co-occurrence leg carries the routing on its own). A reranker, when one
    is configured, tightens a wider pool down to `k`.

    Lives here rather than beside `related_notes`: it needs `get_reranker`, and
    kernel.recall.relatedness must not reach into silica.agent — the structural
    modules that import it are contractually agent-free (see import-linter).

    Any failure degrades to `[]`, which every caller treats as a safe no-op:
    the orphan worker only links among the candidates it was offered.
    """
    try:
        from silica.config import CONFIG
        from silica.kernel.recall.cooccurrence import cooccur_key, get_cooccur_store
        from silica.kernel.recall.embed import get_store
        from silica.kernel.recall.relatedness import related_notes

        from silica.agent.providers import get_reranker
        from silica.kernel.recall.rerank import link_query, rerank_related

        # cooccur_key (case-PRESERVED, .md-stripped) is the store keyspace; _norm_path
        # would lowercase and miss the case-preserving stored keys -> empty results.
        key = cooccur_key(path)
        reranker = get_reranker(CONFIG)
        pool = max(k, 20) if reranker else k
        results = related_notes(
            key,
            embed_store=get_store(),
            cooccur_store=get_cooccur_store(lang=CONFIG.cooccurrence_lang),
            k=pool,
        )
        if reranker:
            results = rerank_related(reranker, link_query(key), results, k=k)
        return [{"name": r.name, "path": r.path} for r in results]
    except Exception as e:
        logger.debug("link candidate lookup failed for %r (non-fatal): %s", path, e)
        return []

# ---------------------------------------------------------------------------
# plan → WorkItems
# ---------------------------------------------------------------------------

def _orphan_workitems(plan: CurationPlan) -> list[WorkItem]:
    items: list[WorkItem] = []
    for it in plan.by_kind("orphan"):
        items.append(WorkItem(
            kind="orphan",
            target_path=it.target,
            context={"candidates": link_candidates(it.target)},
            reason=it.reason or "curate orphan",
        ))
    return items


def _dedup_workitems(plan: CurationPlan) -> list[WorkItem]:
    """Turn dedup pairs into merge WorkItems, collapsing duplicate *families*.

    Pairwise dedup leaves one survivor per local top-1 hub: a family {A,B,C,D,E}
    whose top-1 edges are A→B, C→B, D→E, F→E collapses to TWO notes, not one.
    So confirmed pairs (score ≥ τ_high) are union-found into connected components
    and each component funnels into its single largest note.

    Transitive closure is applied ONLY to confirmed pairs — chaining borderline
    (< τ_high) links would merge distant notes — so borderline pairs stay per-pair.
    Safety: every item still passes the ternary judge, and curate items carry no
    `target_dir`, so a "distinct" verdict is a no-op. Union-find proposes the
    merge target; the judge disposes. A false union costs a judge call, never a bad
    merge.
    """
    from silica.config import CONFIG

    tau_high = getattr(CONFIG, "sim_threshold_high", 0.85)
    pairs = list(plan.by_kind("dedup"))

    _body_cache: dict[str, str] = {}
    def body(p: str) -> str:
        if p not in _body_cache:
            _body_cache[p] = _read_body(p)
        return _body_cache[p]
    stem = lambda p: p.removesuffix(".md").rsplit("/", 1)[-1]

    # Connected components over confirmed pairs only (nx over a hand-rolled
    # union-find — same result, one well-known call).
    import networkx as nx

    graph = nx.Graph()
    node_score: dict[str, float] = {}
    for it in pairs:
        if it.score >= tau_high:
            graph.add_edge(it.target, it.partner)
            for n in (it.target, it.partner):
                node_score[n] = max(node_score.get(n, 0.0), it.score)

    components = [set(c) for c in nx.connected_components(graph)]
    # node -> component-id, so the borderline pass below can ask "same component?"
    comp_of: dict[str, int] = {n: i for i, c in enumerate(components) for n in c}

    items: list[WorkItem] = []
    # 1. Each confirmed component → collapse every member into its largest note.
    from silica.kernel.write.contested import merge_rank
    for members in components:
        canonical = max(members, key=lambda p: merge_rank(body(p)))
        for m in members:
            if m == canonical:
                continue
            sc = node_score.get(m, tau_high)
            items.append(WorkItem(
                kind="dedup",
                target_path=canonical,
                context={
                    "concept": stem(m),
                    "excerpt": body(m)[:4000],
                    "candidate": stem(canonical),
                    "score": sc,
                    "inbox_file": m,
                    "loser_path": m,
                },
                reason=f"curate dedup family → {stem(canonical)} (score={sc:.3f})",
            ))

    # 2. Borderline pairs — per-pair behaviour (more reliable note is target),
    #    skipping any pair already absorbed by a shared confirmed component.
    for it in pairs:
        if it.score >= tau_high:
            continue
        source, target = it.target, it.partner
        if comp_of.get(source, -1) == comp_of.get(target, -2):
            continue
        body_s, body_t = body(source), body(target)
        if merge_rank(body_t) >= merge_rank(body_s):
            larger, smaller, smaller_body = target, source, body_s
        else:
            larger, smaller, smaller_body = source, target, body_t
        items.append(WorkItem(
            kind="dedup",
            target_path=larger,
            context={
                "concept": stem(smaller),
                "excerpt": smaller_body[:4000],
                "candidate": stem(larger),
                "score": it.score,
                "inbox_file": smaller,
                "loser_path": smaller,
            },
            reason=it.reason or f"curate dedup score={it.score:.3f}",
        ))
    # Family members share target_path=canonical by construction → one judge
    # call per family instead of one per member.
    from silica.kernel.workqueue import batch_dedup_items
    return batch_dedup_items(items)


def _refine_workitems(plan: CurationPlan) -> list[WorkItem]:
    return [
        WorkItem(kind="refine", target_path=it.target, context={}, reason=it.reason or "curate refine")
        for it in plan.by_kind("refine")
    ]


def _execution_outcome_counts(
    autolink_result: dict[str, Any], batch: dict[str, Any]
) -> dict[str, int]:
    """Real per-item outcome counts for an apply run — NOT plan.counts().

    `batch["summary"]` (run_subagent_batch -> WorkQueue.summary()) is already
    a Counter of each dispatched WorkItem's REAL terminal status (e.g.
    "committed" via commit.py, "no_merge" for a dedup verdict of distinct,
    "no_link" when the orphan worker found nothing worth linking, "no_change",
    "failed", "cancelled") — a batch where every dedup came back distinct
    must show up as "no_merge", not as though the planned dedup succeeded.
    The mechanical autolink direct-commit isn't a WorkItem, so its real
    outcome (links actually added, from silica_autolink's own return value —
    not the candidate-pair count the plan carried) is folded in separately.
    silica_autolink returns {"notes_scanned", "notes_linked", "total_links_added"}
    (see silica/tools/graph.py); "added" is silica_backlink's key, not autolink's.
    """
    counts: dict[str, int] = dict(batch.get("summary", {}))
    added = (autolink_result or {}).get("total_links_added", 0)
    if added:
        counts["autolink"] = added
    return counts


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def apply_curation_plan(
    plan: CurationPlan,
    *,
    config: Any = None,
    run_id: str | None = None,
    vault_path: str | None = None,
    cancel_token: Any = None,
) -> dict[str, Any]:
    """Execute a CurationPlan on the existing seam, then journal it once.

    Fires the mechanical autolink direct-commit, enqueues orphan/dedup/refine
    WorkItems through run_subagent_batch (commit_ops + bounds + rollback), and
    appends one idempotent journal line. An empty plan is a no-op — nothing is
    enqueued, written, or journalled.
    """
    if plan.is_empty():
        return {"status": "nothing_to_do", "counts": {}}

    if config is None:
        from silica.config import CONFIG

        config = CONFIG
    run_id = run_id or uuid.uuid4().hex

    # 1. Mechanical autolink — LLM-free, graph-safe, reversible direct commit.
    autolink_sources = [it.target for it in plan.by_kind("autolink")]
    autolink_result = _run_autolink(autolink_sources) if autolink_sources else {}

    # 2. Orphan / dedup / refine → WorkItems on the leashed-sub-agent seam.
    work: list[WorkItem] = (
        _orphan_workitems(plan) + _dedup_workitems(plan) + _refine_workitems(plan)
    )
    batch = (
        run_subagent_batch(work, config, cancel_token=cancel_token)
        if work
        else {"items": 0, "summary": {}, "results": []}
    )

    # 3. Human journal — one line per run, deduped so a resume never doubles
    #    it. Reports the REAL outcome (what run_subagent_batch's per-item
    #    statuses and the autolink direct-commit actually did), not the
    #    planned item counts — those live in `counts` below for callers that
    #    want the plan shape, but "Applied" must mean applied.
    counts = plan.counts()
    outcome_counts = _execution_outcome_counts(autolink_result, batch)
    append_log_line(
        format_curate_event(outcome_counts),
        run_id,
        vault_path=vault_path,
        dedup_key="curate",
    )

    return {
        "status": "applied",
        "run_id": run_id,
        "counts": counts,
        "outcome_counts": outcome_counts,
        "autolink": autolink_result,
        "batch": batch,
    }


# ---------------------------------------------------------------------------
# tool
# ---------------------------------------------------------------------------

class CurateArgs(BaseModel):
    apply: bool = Field(
        default=False,
        description="If True, enqueue/execute the plan; default is a dry-run that only returns the plan.",
    )
    folder: str = Field(default="", description="Vault-relative folder to scope the audit (empty = whole vault)")
    kinds: list[str] = Field(
        default_factory=list,
        description='Keep only these item kinds (autolink|orphan|dedup|refine); empty = all kinds.',
    )
    targets: list[str] = Field(
        default_factory=list,
        description="Keep only items touching these note paths (matches target or dedup partner, by path suffix); empty = all.",
    )


@tool(CurateArgs, cls="composed")
def silica_curate(
    apply: bool = False,
    folder: str = "",
    kinds: list[str] | None = None,
    targets: list[str] | None = None,
    cancel_token: Any = None,
) -> dict[str, Any]:
    """Curate the vault: turn the report's findings (autolinks, orphans,
    near-duplicates, oversized/lean notes) into executed maintenance work.
    Default is a dry-run returning the plan; apply=True executes with undo
    journaling. Raw audit alone: silica_vault_report.

    Narrow with `kinds` (e.g. ["dedup"]) and `targets` (paths) — both shape
    dry-run and apply. For selection beyond kind/path ("all but X"), dry-run
    first, read the plan, then pass explicit `targets`.
    """
    # the plan is re-derived on every call, so a subset apply after a
    # dry-run replans against the current vault (already true today between
    # /curate and /curate --apply). No plan persistence.
    report = compute_report(
        folder=folder,
        analytics=True,          # lean_notes / reformat_notes triage
        with_embeddings=True,    # duplicate pairs
        with_cooccurrence=True,  # autolink candidates
    )
    plan = compose_curation_plan(report)
    available = plan.counts()                    # pre-filter shape (no_matches honesty)
    # Computed pre-filter: a bare `x.md` target that hit several folders would
    # otherwise apply to notes the caller never named, invisibly.
    ambiguous = plan.ambiguous_targets(targets)
    plan = plan.filtered(kinds, targets)         # may raise ValueError → tool error

    result: dict[str, Any] = {
        "apply": apply,
        "total": len(plan),
        "counts": plan.counts(),
        "items": [
            {"kind": i.kind, "target": i.target, "partner": i.partner, "reason": i.reason}
            for i in plan.items
        ],
        # Held back on purpose (load-bearing notes, V4): shown so a dry-run
        # never reads as "nothing to refine" when something was vetoed.
        "vetoed": [{"kind": i.kind, "target": i.target, "reason": i.reason} for i in plan.vetoed],
    }
    if ambiguous:
        result["ambiguous_targets"] = ambiguous

    if plan.is_empty():
        # A filter that emptied a non-empty plan is `no_matches`, not
        # `nothing_to_do` — the CLI must not claim "the vault is coherent"
        # when the filter, not the vault, produced the emptiness.
        if available:
            result["status"] = "no_matches"
            result["available"] = available
        else:
            result["status"] = "nothing_to_do"
        return result

    if not apply:
        result["status"] = "dry_run"
        return result

    from silica.config import CONFIG

    execution = apply_curation_plan(
        plan,
        config=CONFIG,
        vault_path=getattr(CONFIG, "vault_path", None),
        cancel_token=cancel_token,
    )
    result["status"] = execution.get("status", "applied")
    result["execution"] = execution
    return result
