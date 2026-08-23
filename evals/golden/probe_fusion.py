# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Reranker A/B over the fused ranking (opt-in, informational).

The gated fusion probe itself lives in ``silica.kernel.link.health.fusion_probe``
(also served by the silica_health tool); the runner calls it directly. This
module keeps only the reranker A/B — it depends on an HTTP provider, so it is
neither deterministic nor free: never gated, never in the baseline.
"""
from __future__ import annotations

from silica.kernel.link.health import K, eligible_pairs, wikilink_graph


def _pair_ranks(eligible: list[tuple[str, str]],
                topk: dict[str, list[str]]) -> dict[tuple[str, str], int | None]:
    """Best 1-based rank of each pair in one arm's per-endpoint top-k (None=miss)."""
    out: dict[tuple[str, str], int | None] = {}
    for a, b in eligible:
        ranks = []
        if b in topk[a]:
            ranks.append(topk[a].index(b) + 1)
        if a in topk[b]:
            ranks.append(topk[b].index(a) + 1)
        out[(a, b)] = min(ranks) if ranks else None
    return out


def _score_pairs(eligible: list[tuple[str, str]], topk: dict[str, list[str]]) -> tuple[float, float, set[tuple[str, str]]]:
    """(recall, mrr, recovered-pair set) for one arm's per-endpoint top-k."""
    ranks = _pair_ranks(eligible, topk)
    hit = [r for r in ranks.values() if r]
    n = len(eligible)
    return (round(len(hit) / n, 4),
            round(sum(1.0 / r for r in hit) / n, 4),
            {p for p, r in ranks.items() if r})


def run_rerank_ab(vault, store, *, embed_store=None, reranker=None,
                  k: int = K, verbose: bool = False) -> dict:
    """A/B on the same masked pairs: the gated fused top-k (arm A) vs the
    production rerank path (arm B: the SAME first-stage top-k, reordered by the
    cross-encoder — reorder-only per the retrieval-gates spec, so recall@k is
    invariant by construction and the A/B measures ORDERING: mrr and per-pair
    rank wins/losses).

    Arm B mirrors the production call sites (coordinator/_orphan_candidates,
    curate): query text = link_query(key) — bare title when informative, body
    head window otherwise (query-shape A/B, bench/local_rerank_excerpt_sweep
    + local_rerank_title_blind). Pre-gate baselines used note_document(key).

    ``empty_docs`` counts endpoints whose query text could not be read — for
    those, rerank_related no-ops on the empty query and arm B silently
    degenerates to the unreranked pool. A high count means the A/B measured
    nothing, not that the reranker is neutral.
    """
    from silica.kernel.recall.relatedness import related_notes
    from silica.kernel.recall.rerank import link_query, rerank_related

    empty = {
        "pairs_evaluated": 0, "endpoints": 0, "empty_docs": 0,
        "base_recall": 0.0, "base_mrr": 0.0,
        "rerank_recall": 0.0, "rerank_mrr": 0.0,
        "pairs_won": 0, "pairs_lost": 0,
    }
    if len(store) == 0:
        return empty
    eligible = eligible_pairs(wikilink_graph(vault, store))
    if not eligible:
        return empty

    es = embed_store if (embed_store is not None and len(embed_store)) else None
    endpoints = sorted({e for pr in eligible for e in pr})
    if verbose:
        print(f"\nrerank A/B: {len(endpoints)} endpoints, top-{k} …")

    base_topk: dict[str, list[str]] = {}
    rr_topk: dict[str, list[str]] = {}
    empty_docs = 0
    for i, key in enumerate(endpoints):
        # One first stage per endpoint: arm B reorders exactly arm A's top-k,
        # so the arms differ only by the rerank pass (membership is shared).
        results = related_notes(key, embed_store=es, cooccur_store=store, k=k)
        base_topk[key] = [r.path for r in results]
        doc = link_query(key)
        if not doc:
            empty_docs += 1
        rr_topk[key] = [r.path for r in rerank_related(reranker, doc, results, k=k)]
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(endpoints)}")

    base_recall, base_mrr, _ = _score_pairs(eligible, base_topk)
    rr_recall, rr_mrr, _ = _score_pairs(eligible, rr_topk)
    base_ranks = _pair_ranks(eligible, base_topk)
    rr_ranks = _pair_ranks(eligible, rr_topk)
    _worst = float("inf")

    res = {
        "pairs_evaluated": len(eligible),
        "endpoints": len(endpoints),
        "empty_docs": empty_docs,
        "base_recall": base_recall, "base_mrr": base_mrr,
        "rerank_recall": rr_recall, "rerank_mrr": rr_mrr,
        # Reorder-only makes recall invariant; wins/losses live in the ranks.
        "pairs_won": sum(1 for p in eligible
                         if (rr_ranks[p] or _worst) < (base_ranks[p] or _worst)),
        "pairs_lost": sum(1 for p in eligible
                          if (rr_ranks[p] or _worst) > (base_ranks[p] or _worst)),
    }
    if verbose:
        print(f"  fused    recall@{k} {base_recall:.1%}  mrr {base_mrr:.3f}")
        print(f"  reranked recall@{k} {rr_recall:.1%}  mrr {rr_mrr:.3f}  "
              f"(won +{res['pairs_won']} / lost -{res['pairs_lost']}, "
              f"empty docs {empty_docs}/{len(endpoints)})")
    return res
