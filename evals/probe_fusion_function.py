# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Fusion-function cells for the relatedness facade (ADR-0029 follow-up).

The facade fuses two legs with unweighted RRF over a 30-item pool per leg.
Three findings in docs/research/papers question that composition; each is a
cell here, all offline, all on the golden ground truth of kernel/link/health
(human body-wikilink pairs that stay >2 hops apart once masked):

  rrf_full  RRF over COMPLETE per-leg lists. Zhang 2026 (arXiv 2608.07152): a
            fixed top-L per retriever is not the complete-list fusion, and the
            L that works on one corpus snapshot does not transfer. Both legs
            already score every note (one matvec, one posting walk), so the
            pool is a truncation that costs nothing to remove.
  cc        Convex combination of TMM-normalised scores, alpha * embed +
            (1 - alpha) * cooccur (Bruch, Gai, Ingber 2023, arXiv 2210.11934):
            a rank-only fusion discards the score distances, a convex
            combination keeps them, and its single alpha converges on a few
            labels. Alpha is HELD OUT here: chosen on one half of the pairs,
            scored on the other, repeated.
  bm25      k1/b grid for the cooccur tf term (Hsu and Yang 2026, arXiv
            2605.10848: BM25 tuning moved answer accuracy by 18% on
            BrowseComp-Plus). Held out like alpha.
  maxsim    embed leg = max(body cosine, title cosine) per candidate, the two
            vectors the store already holds (Yu et al. 2026, arXiv 2606.23642:
            several matching points per document beat one pooled vector).

  uv run python -m evals.probe_fusion_function --vault ~/Documents/Obsidian/test

Every cell is scored against the shipped facade (arm A, asserted to be
reproduced path for path before any cell runs) with the phase-2 gate of
probe_graph_variables: +2pp recall@10, mrr non-regression, McNemar. A cell
whose parameter was fitted reports the held-out mean beside the in-sample
number, and only the held-out number can pass.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from evals.probe_graph_variables import K, _gate, _score, _stores

# Alpha grid of the convex combination; the paper's consistent range is
# 0.6 .. 0.8 and the grid brackets it on both sides so a held-out optimum at
# an edge is visible as such.
ALPHAS = tuple(round(0.05 * i, 2) for i in range(4, 20))
ALPHA_PAPER = 0.8
# Textbook neighbourhood of the shipped constants (k1 1.2, b 0.75).
K1_GRID = (0.9, 1.2, 1.5, 2.0)
B_GRID = (0.5, 0.75, 1.0)
SPLITS = 20
SEED = 42
# Theoretical infima for TMM normalisation (the paper's phi_tmm): cosine
# cannot go below -1, an IDF-weighted overlap cannot go below 0.
INF_COSINE = -1.0
INF_OVERLAP = 0.0


@dataclass(frozen=True)
class Legs:
    """One endpoint's complete per-leg rankings, best first, None = abstains."""

    embed: list[tuple[str, float]] | None
    title: list[tuple[str, float]] | None
    cooc: list[tuple[str, float]] | None


# ---------------------------------------------------------------------------
# legs
# ---------------------------------------------------------------------------

def _embed_complete(es, key: str, *, title: bool) -> list[tuple[str, float]] | None:
    """Every other note scored by cosine on one vector field, best first.

    Same abstention as the facade's `_rank_embeddings_from_vec`: no vector, or
    a flat-zero ranking, means the leg says nothing.
    """
    from silica.kernel.recall.relatedness import _NOISE_FLOOR

    vec = es.get_title_vec(key) if title else es.get_vec(key)
    if vec is None:
        return None
    search = es.title_cosine_top_k if title else es.cosine_top_k
    hits = search(vec, k=len(es), exclude={key})
    if not hits or max(h["score"] for h in hits) <= _NOISE_FLOOR:
        return None
    return [(h["path"], float(h["score"])) for h in hits]


def _cooc_complete(store, key: str) -> list[tuple[str, float]] | None:
    from silica.kernel.recall.relatedness import _cooccur_ranking

    return _cooccur_ranking(store, key, k=len(store), exclude={key}, scope=None, expand=False)


def _legs(es, store, endpoints: list[str]) -> dict[str, Legs]:
    return {
        key: Legs(
            embed=_embed_complete(es, key, title=False) if es is not None else None,
            title=_embed_complete(es, key, title=True) if es is not None else None,
            cooc=_cooc_complete(store, key),
        )
        for key in endpoints
    }


def _with_bm25(store, endpoints: list[str], k1: float, b: float) -> dict[str, list | None]:
    """The cooccur leg under other k1/b, read by `_rank_cooccur_from_profile`
    from the module constants at call time."""
    from silica.kernel.recall import relatedness as R

    old = (R.BM25_K1, R.BM25_B)
    R.BM25_K1, R.BM25_B = k1, b
    try:
        return {key: _cooc_complete(store, key) for key in endpoints}
    finally:
        R.BM25_K1, R.BM25_B = old


def _maxsim(legs: Legs) -> list[tuple[str, float]] | None:
    """Body and title as two matching points: a candidate scores the better one."""
    if legs.embed is None:
        return None
    if legs.title is None:
        return legs.embed
    best = dict(legs.embed)
    for p, s in legs.title:
        if s > best.get(p, INF_COSINE):
            best[p] = s
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# fusion functions (pure)
# ---------------------------------------------------------------------------

def _top(fused: dict[str, float], k: int) -> list[str]:
    from silica.kernel.recall.graph_export import is_vault_artifact

    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [p for p, _ in ranked if not is_vault_artifact(p)][:k]


def rrf(lists: list[list[tuple[str, float]] | None], *, pool: int | None, k: int) -> list[str]:
    """The facade's fusion: 1/(RRF_K + rank) per list, lists cut at `pool`
    (None = complete lists)."""
    from silica.kernel.recall.relatedness import RRF_K

    fused: dict[str, float] = {}
    for ranking in lists:
        if ranking is None:
            continue
        for rank, (p, _s) in enumerate(ranking if pool is None else ranking[:pool]):
            fused[p] = fused.get(p, 0.0) + 1.0 / (RRF_K + rank + 1)
    return _top(fused, k)


def cc_scores(embed: list[tuple[str, float]] | None, cooc: list[tuple[str, float]] | None,
              *, alpha: float) -> dict[str, float]:
    """TM2C2 of the paper: each leg min-max normalised between its theoretical
    infimum and the best score this query saw, then alpha-mixed. A note the
    cooccur leg did not score sits at that leg's infimum (no overlap is a real
    zero, not a missing value), which is what complete lists make exact. The
    result is bounded in [0, 1], which is what makes it a candidate routing
    scalar for COLLISION (probe_collision_risk_coverage).
    """
    fused: dict[str, float] = {}
    if embed is not None:
        span = embed[0][1] - INF_COSINE
        for p, s in embed:
            fused[p] = alpha * ((s - INF_COSINE) / span if span > 0 else 0.0)
    if cooc is not None:
        span = cooc[0][1] - INF_OVERLAP
        for p, s in cooc:
            fused[p] = fused.get(p, 0.0) + (1.0 - alpha) * ((s - INF_OVERLAP) / span if span > 0 else 0.0)
    return fused


def cc(embed: list[tuple[str, float]] | None, cooc: list[tuple[str, float]] | None,
       *, alpha: float, k: int) -> list[str]:
    return _top(cc_scores(embed, cooc, alpha=alpha), k)


# ---------------------------------------------------------------------------
# held-out selection
# ---------------------------------------------------------------------------

def held_out(pairs: list[tuple[str, str]], cells: dict[str, dict[str, list[str]]],
             *, reference: dict[str, list[str]], splits: int = SPLITS, seed: int = SEED) -> dict:
    """Pick the best cell on a random half of the pairs (recall, then mrr),
    score it on the other half, `splits` times; the reference is scored on the
    same halves so the delta is paired.

    Pairs share endpoints, so the halves are not independent samples of
    endpoints; with one or two scalars being chosen that leakage cannot fit
    anything, and it is stated here rather than hidden.
    """
    rng = random.Random(seed)
    order = list(pairs)
    picks: dict[str, int] = {}
    d_recall: list[float] = []
    d_mrr: list[float] = []
    test_recall: list[float] = []
    for _ in range(splits):
        rng.shuffle(order)
        half = len(order) // 2
        train, test = order[:half], order[half:]
        best = max(cells, key=lambda name: (_score(cells[name], train)["recall_at_10"],
                                            _score(cells[name], train)["mrr"], name))
        picks[best] = picks.get(best, 0) + 1
        chosen, ref = _score(cells[best], test), _score(reference, test)
        test_recall.append(chosen["recall_at_10"])
        d_recall.append(chosen["recall_at_10"] - ref["recall_at_10"])
        d_mrr.append(chosen["mrr"] - ref["mrr"])
    return {
        "picked": dict(sorted(picks.items(), key=lambda kv: -kv[1])),
        "heldout_recall": round(statistics.fmean(test_recall), 4),
        "heldout_delta_recall": round(statistics.fmean(d_recall), 4),
        "heldout_delta_recall_sd": round(statistics.pstdev(d_recall), 4),
        "heldout_delta_mrr": round(statistics.fmean(d_mrr), 4),
        "splits": splits,
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _timed(fn, keys: list[str]) -> tuple[dict[str, list[str]], float]:
    t0 = time.perf_counter()
    out = {key: fn(key) for key in keys}
    return out, (time.perf_counter() - t0) * 1000.0 / max(1, len(keys))


def run(vault: Path, *, verbose: bool = True) -> dict:
    from silica.config import CONFIG
    from silica.kernel.link.health import eligible_pairs, wikilink_graph
    from silica.kernel.recall import relatedness as R

    store, es = _stores(vault)
    CONFIG.cooccur_bm25 = True  # the shipped default, pinned so the run is the run
    pairs = eligible_pairs(wikilink_graph(vault, store))
    endpoints = sorted({e for pr in pairs for e in pr})
    pool = max(K * 3, R._POOL_MIN)

    arm_a, ms_a = _timed(
        lambda key: [r.path for r in R.related_notes(key, embed_store=es, cooccur_store=store, k=K)],
        endpoints,
    )
    t0 = time.perf_counter()
    legs = _legs(es, store, endpoints)
    ms_legs = (time.perf_counter() - t0) * 1000.0 / len(endpoints)

    mine = {key: rrf([legs[key].embed, legs[key].cooc], pool=pool, k=K) for key in endpoints}
    drift = sum(1 for key in endpoints if mine[key] != arm_a[key])
    assert drift == 0, f"probe RRF drifted from the facade on {drift}/{len(endpoints)} endpoints"

    base = _score(arm_a, pairs)
    out: dict = {
        "pairs": len(pairs), "endpoints": len(endpoints), "legs": "embed+cooccur" if es else "cooccur",
        "title_vec_coverage": round(sum(1 for key in endpoints if legs[key].title is not None) / len(endpoints), 4),
        "arm_a": {"recall_at_10": base["recall_at_10"], "mrr": base["mrr"], "ms_per_call": round(ms_a, 2),
                  "ms_legs_complete": round(ms_legs, 2)},
        "cells": {},
    }

    def cell(name: str, topk: dict[str, list[str]], ms: float | None = None, **extra) -> dict:
        g = _gate(base, _score(topk, pairs))
        g["recall_at_10"] = round(base["recall_at_10"] + g["delta_recall"], 4)
        if ms is not None:
            g["ms_per_call"] = round(ms, 3)
        g.update(extra)
        out["cells"][name] = g
        return g

    # rrf over complete lists: no parameter, gated in-sample is the whole story
    full, ms = _timed(lambda key: rrf([legs[key].embed, legs[key].cooc], pool=None, k=K), endpoints)
    cell("rrf_full", full, ms)

    # convex combination, alpha held out
    cc_cells = {}
    ms_cc = 0.0
    for alpha in ALPHAS:
        cc_cells[f"cc_{alpha}"], ms_cc = _timed(
            lambda key, a=alpha: cc(legs[key].embed, legs[key].cooc, alpha=a, k=K), endpoints)
    for alpha in ALPHAS:
        cell(f"cc_{alpha}", cc_cells[f"cc_{alpha}"], ms_cc, fitted=True)
    out["cc_heldout"] = held_out(pairs, cc_cells, reference=arm_a)
    out["cc_paper_alpha"] = out["cells"][f"cc_{ALPHA_PAPER}"]

    # bm25 grid under the shipped fusion, held out; the shipped constants are a cell
    bm25_cells = {}
    for k1 in K1_GRID:
        for b in B_GRID:
            cooc = _with_bm25(store, endpoints, k1, b)
            bm25_cells[f"bm25_k1{k1}_b{b}"] = {
                key: rrf([legs[key].embed, cooc[key]], pool=pool, k=K) for key in endpoints}
    for name, topk in bm25_cells.items():
        cell(name, topk, fitted=True)
    out["bm25_heldout"] = held_out(pairs, bm25_cells, reference=arm_a)

    # maxsim embed leg under the shipped fusion and under the paper alpha
    if out["title_vec_coverage"] > 0:
        ms_legs_max = {key: _maxsim(legs[key]) for key in endpoints}
        topk, ms = _timed(lambda key: rrf([ms_legs_max[key], legs[key].cooc], pool=pool, k=K), endpoints)
        cell("maxsim_rrf", topk, ms)
        topk, ms = _timed(lambda key: cc(ms_legs_max[key], legs[key].cooc, alpha=ALPHA_PAPER, k=K), endpoints)
        cell("maxsim_cc", topk, ms)
        alone, _ = _timed(lambda key: rrf([ms_legs_max[key]], pool=None, k=K), endpoints)
        body_alone, _ = _timed(lambda key: rrf([legs[key].embed], pool=None, k=K), endpoints)
        out["embed_leg_alone"] = {"body": _score(body_alone, pairs)["recall_at_10"],
                                  "maxsim": _score(alone, pairs)["recall_at_10"]}

    if verbose:
        _print(out)
    return out


def _print(out: dict) -> None:
    a = out["arm_a"]
    print(f"\narm A (facade): recall@10 {a['recall_at_10']:.4f} mrr {a['mrr']:.4f} "
          f"{a['ms_per_call']:.2f} ms/call; complete legs {a['ms_legs_complete']:.2f} ms/call; "
          f"{out['pairs']} pairs, {out['endpoints']} endpoints, title_vec coverage {out['title_vec_coverage']:.0%}")
    print(f"{'cell':20s} {'recall':>7s} {'d_rec':>7s} {'d_mrr':>7s} {'p':>7s}  verdict")
    for name, g in out["cells"].items():
        tag = " (fitted, read the held-out line)" if g.get("fitted") else ""
        print(f"{name:20s} {g['recall_at_10']:7.4f} {g['delta_recall']:+7.4f} {g['delta_mrr']:+7.4f} "
              f"{g['mcnemar_p']:7.4f}  {g['verdict']}{tag}")
    for key in ("cc_heldout", "bm25_heldout"):
        h = out[key]
        print(f"{key}: picked {h['picked']}; held-out recall {h['heldout_recall']:.4f}, "
              f"delta vs facade {h['heldout_delta_recall']:+.4f} (sd {h['heldout_delta_recall_sd']:.4f}), "
              f"delta mrr {h['heldout_delta_mrr']:+.4f} over {h['splits']} splits")
    if "embed_leg_alone" in out:
        print(f"embed leg alone recall@10: body {out['embed_leg_alone']['body']:.4f} "
              f"maxsim {out['embed_leg_alone']['maxsim']:.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vault", required=True)
    ap.add_argument("--record", help="write the results JSON here")
    args = ap.parse_args(argv)
    out = run(Path(args.vault).expanduser())
    if args.record:
        Path(args.record).write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
