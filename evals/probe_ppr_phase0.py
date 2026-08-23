# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Phase 0 of docs/spec-ppr-propagation.md — measure before building anything.

Throwaway instrument, NOT product code. Four measures, one of them a kill gate:

  1. |V|, |E| and degree distribution of the concept graph.
  2. Wall time of a degree-normalised walk per query (hops 1..3), median + p99.
  3. Which eligible pairs today's fused ranking misses (the real headroom).
  4. KILL GATE — of those missed pairs, how many are reachable at all on the
     concept graph within `hops` steps. If propagation cannot reach them, the
     mechanism cannot work and the spec stops here.

Measure 4 comes with its own vacuity check: on a dense graph "reachable in 3
hops" can mean "the whole vault", which would make the gate meaningless. The
report prints the reachable fraction of notes so the number can be read
honestly.

  uv run python -m evals.probe_ppr_phase0 --vault ~/Documents/Obsidian/test
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from silica.kernel.link.health import K, eligible_pairs, wikilink_graph

HOPS = 3
ALPHA = 0.5
# Spec 4/fase 0 stop criteria.
REACH_MIN_FRAC = 0.20
WALK_P99_BUDGET_MS = 50.0


def ppr(adj: dict[str, dict[str, float]], seeds: dict[str, float],
        *, hops: int = HOPS, alpha: float = ALPHA, eps: float = 0.0) -> dict[str, float]:
    """Personalised PageRank over dict-of-dicts, power iteration, stdlib only.

    Mass leaving a node is 1, split among its neighbours in proportion to edge
    weight — the degree normalisation `_EXPANSION_DISCOUNT` lacks. Mass on a
    dangling stem (absent from the graph) is dropped, not redistributed.

    `eps` is the spec's cheap mitigation: a node holding less mass than that
    does not propagate. Its own mass is below the threshold by construction, so
    dropping it only perturbs the tail of the profile — measured, not assumed.
    """
    total = sum(seeds.values())
    if total <= 0:
        return {}
    restart = {s: w / total for s, w in seeds.items()}
    r = dict(restart)
    for _ in range(hops):
        nxt = {s: alpha * w for s, w in restart.items()}
        for stem, mass in r.items():
            if mass < eps:
                continue
            nbrs = adj.get(stem)
            if not nbrs:
                continue
            degree = sum(nbrs.values())
            if degree <= 0:
                continue
            share = (1.0 - alpha) * mass / degree
            for nb, w in nbrs.items():
                nxt[nb] = nxt.get(nb, 0.0) + share * w
        r = nxt
    return r


def bfs_levels(adj: dict[str, dict[str, float]], seeds, max_hops: int) -> dict[str, int]:
    """{stem: hop distance} from the seed set, capped at `max_hops`."""
    dist = {s: 0 for s in seeds if s in adj}
    frontier = list(dist)
    for h in range(1, max_hops + 1):
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = h
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    return dist


def _pct(values: list[float], q: float) -> float:
    return sorted(values)[min(len(values) - 1, int(q * (len(values) - 1)))]


def run(vault: Path, *, hops: int = HOPS, alpha: float = ALPHA,
        samples: int = 100, verbose: bool = False) -> dict:
    from evals.golden.runner import _open_stores, vault_digest
    from silica.kernel.recall.relatedness import related_notes

    store, embed_store = _open_stores(vault)
    es = embed_store if (embed_store is not None and len(embed_store)) else None
    digest, notes = vault_digest(vault)
    print(f"vault {vault}  ({notes} notes, {digest[:19]}…)  embed leg: {'live' if es else 'OFF'}")

    # --- 1. graph size -----------------------------------------------------
    t0 = time.perf_counter()
    adj = store.adjacency()
    adj_ms = (time.perf_counter() - t0) * 1000
    degrees = sorted(len(n) for n in adj.values())
    edges = sum(degrees) // 2
    graph = {
        "concepts": len(adj), "edges": edges,
        "degree_mean": round(statistics.fmean(degrees), 1),
        "degree_median": degrees[len(degrees) // 2],
        "degree_p99": _pct([float(d) for d in degrees], 0.99),
        "degree_max": degrees[-1],
        "adjacency_build_ms": round(adj_ms, 1),
    }
    print(f"\n1. concept graph: |V|={graph['concepts']} |E|={graph['edges']} "
          f"degree mean {graph['degree_mean']} median {graph['degree_median']} "
          f"p99 {graph['degree_p99']:.0f} max {graph['degree_max']} "
          f"(adjacency built in {adj_ms:.0f}ms, cached)")

    # --- pairs the gate evaluates -----------------------------------------
    eligible = eligible_pairs(wikilink_graph(vault, store))
    endpoints = sorted({e for pr in eligible for e in pr})

    # --- 2. walk cost ------------------------------------------------------
    rng = random.Random(0)
    sample = rng.sample(endpoints, min(samples, len(endpoints)))
    walk = {}
    for h in range(1, hops + 1):
        times = []
        for key in sample:
            seeds = store.note_nodes(key)
            t0 = time.perf_counter()
            ppr(adj, seeds, hops=h, alpha=alpha)
            times.append((time.perf_counter() - t0) * 1000)
        walk[h] = {"median_ms": round(statistics.median(times), 1),
                   "p99_ms": round(_pct(times, 0.99), 1)}
    print(f"\n2. walk cost over {len(sample)} sampled endpoints (alpha={alpha}):")
    for h, w in walk.items():
        flag = "" if w["p99_ms"] <= WALK_P99_BUDGET_MS else "  OVER BUDGET (50ms)"
        print(f"   hops={h}  median {w['median_ms']:>7.1f}ms  p99 {w['p99_ms']:>7.1f}ms{flag}")

    # 2b. mass-pruning mitigation at full hops: cost, and how much of the
    # profile it perturbs (top-50 concept agreement against the exact walk).
    prune = {}
    exact = {key: ppr(adj, store.note_nodes(key), hops=hops, alpha=alpha) for key in sample}
    for eps in (1e-5, 1e-4, 1e-3):
        times, agree = [], []
        for key in sample:
            seeds = store.note_nodes(key)
            t0 = time.perf_counter()
            p = ppr(adj, seeds, hops=hops, alpha=alpha, eps=eps)
            times.append((time.perf_counter() - t0) * 1000)
            ref = {s for s, _w in sorted(exact[key].items(), key=lambda kv: -kv[1])[:50]}
            got = {s for s, _w in sorted(p.items(), key=lambda kv: -kv[1])[:50]}
            agree.append(len(ref & got) / len(ref) if ref else 1.0)
        prune[eps] = {"median_ms": round(statistics.median(times), 1),
                      "p99_ms": round(_pct(times, 0.99), 1),
                      "top50_agreement": round(statistics.fmean(agree), 4)}
        print(f"   hops={hops} eps={eps:<7g} median {prune[eps]['median_ms']:>7.1f}ms  "
              f"p99 {prune[eps]['p99_ms']:>7.1f}ms   top-50 agreement "
              f"{prune[eps]['top50_agreement']:.3f}")

    # --- 3. headroom: what the fused ranking misses today ------------------
    t0 = time.perf_counter()
    topk = {key: [r.path for r in related_notes(key, embed_store=es, cooccur_store=store, k=K)]
            for key in endpoints}
    fuse_s = time.perf_counter() - t0
    missed = [(a, b) for a, b in eligible if b not in topk[a] and a not in topk[b]]
    recall = 1 - len(missed) / len(eligible)
    fuse_ms = fuse_s * 1000 / len(endpoints)
    print(f"\n3. headroom: {len(eligible)} eligible pairs, recall@{K} {recall:.4f}, "
          f"{len(missed)} missed  ({len(endpoints)} endpoints, {fuse_ms:.1f}ms each "
          f"for ALL legs today)")

    # --- 4. KILL GATE: are the missed pairs even reachable? ----------------
    note_stems = {p: set(store.note_nodes(p)) for p in store.paths()}
    cache: dict[str, dict[str, int]] = {}

    def levels(key: str) -> dict[str, int]:
        if key not in cache:
            cache[key] = bfs_levels(adj, note_stems.get(key, ()), hops)
        return cache[key]

    buckets: dict[str, int] = {}
    reach_note_frac = []
    reach_concept_frac = []
    for a, b in missed:
        dist = levels(a)
        d = min((dist[s] for s in note_stems.get(b, ()) if s in dist), default=None)
        key = str(d) if d is not None else f">{hops}"
        buckets[key] = buckets.get(key, 0) + 1
    for key in cache:
        dist = cache[key]
        reach_concept_frac.append(len(dist) / len(adj))
        reach_note_frac.append(
            sum(1 for p, st in note_stems.items() if st & dist.keys()) / len(note_stems))

    shared = buckets.get("0", 0)                       # already overlap: rank miss, not reach
    newly = sum(v for k, v in buckets.items() if k.isdigit() and k != "0")
    unreachable = buckets.get(f">{hops}", 0)
    frac_within = (shared + newly) / len(missed) if missed else 0.0
    frac_newly = newly / len(missed) if missed else 0.0
    print(f"\n4. KILL GATE — min concept-hop distance of the {len(missed)} missed pairs:")
    for kb in sorted(buckets, key=lambda x: (not x.isdigit(), x)):
        label = {"0": "0 (already share a concept — ranking miss, not reach)"}.get(kb, kb)
        print(f"   d={label:<52} {buckets[kb]:>4}  ({buckets[kb]/len(missed):.1%})")
    print(f"   within {hops} hops: {frac_within:.1%}   newly reachable (d>=1): {frac_newly:.1%}"
          f"   unreachable: {unreachable/len(missed):.1%}")
    if reach_note_frac:
        print(f"   vacuity check: a seed set reaches {statistics.fmean(reach_concept_frac):.1%} "
              f"of concepts and {statistics.fmean(reach_note_frac):.1%} of notes within {hops} hops")

    # --- 5. which channel could even move them --------------------------------
    # PPR replaces the cooccur leg's profile, so it can only move a pair the
    # cooccur leg itself can surface. If the counterpart is already inside that
    # leg's pool and RRF still buries it, the lever is fusion, not the profile.
    from silica.kernel.recall.relatedness import _POOL_MIN, _cooccur_ranking
    pool_k = max(K * 3, _POOL_MIN)
    cooc: dict[str, dict[str, int]] = {}
    for key in {e for pr in missed for e in pr}:
        ranked = _cooccur_ranking(store, key, k=pool_k, exclude={key}, expand=False) or []
        cooc[key] = {p: i for i, (p, _s) in enumerate(ranked)}
    in_pool = sum(1 for a, b in missed if b in cooc[a] or a in cooc[b])
    print(f"\n5. channel: of the {len(missed)} missed pairs, {in_pool} "
          f"({in_pool/len(missed):.1%}) already sit inside the cooccur leg's "
          f"top-{pool_k} pool and are lost in RRF; {len(missed)-in_pool} "
          f"({1-in_pool/len(missed):.1%}) the leg never proposes at all")

    verdict = "PROCEED" if frac_within >= REACH_MIN_FRAC else "ABORT"
    print(f"\n   reach criterion (>= {REACH_MIN_FRAC:.0%} within {hops} hops): {verdict}")
    if walk[hops]["p99_ms"] > WALK_P99_BUDGET_MS:
        print(f"   cost criterion: p99 {walk[hops]['p99_ms']:.0f}ms > {WALK_P99_BUDGET_MS:.0f}ms "
              f"— mitigation required before phase 1")

    return {
        "vault": {"path": str(vault), "digest": digest, "notes": notes},
        "hops": hops, "alpha": alpha,
        "graph": graph,
        "walk_ms": walk,
        "walk_pruned": {str(e): v for e, v in prune.items()},
        "headroom": {"pairs_evaluated": len(eligible), "recall_at_10": round(recall, 4),
                     "missed": len(missed), "endpoints": len(endpoints),
                     "fused_ms_per_endpoint": round(fuse_ms, 1)},
        "reach": {"buckets": buckets, "frac_within_hops": round(frac_within, 4),
                  "frac_newly_reachable": round(frac_newly, 4),
                  "concept_frac_reached": round(statistics.fmean(reach_concept_frac), 4)
                  if reach_concept_frac else 0.0,
                  "note_frac_reached": round(statistics.fmean(reach_note_frac), 4)
                  if reach_note_frac else 0.0},
        "channel": {"cooccur_pool_k": pool_k, "missed_inside_cooccur_pool": in_pool,
                    "missed_outside_cooccur_pool": len(missed) - in_pool},
        "verdict": verdict,
    }


def main(argv=None) -> int:
    from evals.golden.runner import resolve_vault

    ap = argparse.ArgumentParser(prog="python -m evals.probe_ppr_phase0")
    ap.add_argument("--vault")
    ap.add_argument("--hops", type=int, default=HOPS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--json", default="bench/ppr_phase0.json")
    args = ap.parse_args(argv)

    vault = resolve_vault(args.vault)
    try:
        res = run(vault, hops=args.hops, alpha=args.alpha, samples=args.samples)
    finally:
        import silica.driver
        silica.driver._driver = None
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
