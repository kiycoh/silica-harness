# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Phase 2 of docs/spec-ppr-propagation.md — the pre-registered gate, measured.

Throwaway instrument, NOT product code: it swaps the profile builder by
rebinding the module global, so the whole run costs zero product diff. Only if
arm C wins does `_ppr_profile` get written into `relatedness.py`.

Two measures, in this order:

  1. CEILING — phase 0 established the reach channel is dead and the whole
     hypothesis now rests on re-weighting notes the cooccur leg can already
     see. That bounds the lever: for each missed pair, what is the best rank
     the counterpart would need in the cooccur leg to reach the fused top-10,
     and is any rank good enough? Pairs no cooccur ranking can save are out of
     the lever's reach whatever the profile does. If the ceiling sits below the
     +2pp the gate asks for, the arms are noise and the spec stops here.

  2. ARMS — three arms in one process on the same eligible pairs:
       A baseline          `expand=False` (today's frozen behaviour)
       B negative control  `expand=True`  (known worse, isolates normalisation)
       C PPR               degree-normalised walk, hops/alpha fixed in phase 0
     Reranker off in all three (it is off by default: `related_notes` does not
     rerank). Primary: C beats A on recall@10 by >= 2pp, measured in-run
     because the 2026-07-19 baseline is stale (phase 0, section 5-bis).
     Significance: exact McNemar on the discordant pairs, reusing
     `evals.paired_stats.paired` rather than writing a second binomial.

  uv run python -m evals.probe_ppr_phase2 --vault ~/Documents/Obsidian/test
  uv run python -m evals.probe_ppr_phase2 --vault ... --sweep
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
from pathlib import Path

from evals.probe_ppr_phase0 import ppr
from silica.kernel.link.health import K, eligible_pairs, wikilink_graph

HOPS = 2          # fixed in phase 0 (section 5-bis): hops=3 is over the 50ms budget
ALPHA = 0.5       # and buys no reach, the 3-hop closure is already saturated
# Pre-declared, before any arm ran: a hops=2 walk spreads mass over ~all 9773
# concepts, and `_rank_cooccur_from_profile` is O(candidates x |profile|), so an
# untruncated profile would multiply the projection cost by ~100x. 300 is about
# 10x the median arm-A profile size, so arm C is not handicapped against A.
# Sensitivity reported by --sweep; it must not be the load-bearing choice.
TOP_STEMS = 300
GATE_DELTA = 0.02  # primary: C - A >= +2pp on recall@10
MRR_TOLERANCE = 0.01

SWEEP_HOPS = (2, 3, 4)
SWEEP_ALPHA = (0.3, 0.5, 0.7)
SWEEP_TOP_STEMS = (100, 300, 1000)


# ---------------------------------------------------------------------------
# Arm C: PPR profile, swapped in by rebinding the module global
# ---------------------------------------------------------------------------

def _truncate(profile: dict[str, float], n: int) -> dict[str, float]:
    if len(profile) <= n:
        return profile
    return dict(sorted(profile.items(), key=lambda kv: -kv[1])[:n])


def _ppr_builder(*, hops: int, alpha: float, top_n: int, sizes: list[int] | None = None):
    """A `_profile_from_seeds`-compatible builder backed by the phase 0 walk."""

    def build(cooccur_store, seeds, *, scope, expand):
        profile = _truncate(ppr(cooccur_store.adjacency(scope=scope), seeds,
                                hops=hops, alpha=alpha), top_n)
        if sizes is not None:
            sizes.append(len(profile))
        return profile

    return build


@contextlib.contextmanager
def _swap(**attrs):
    """Rebind `relatedness` module globals for the duration of an arm.

    Both swappable seams (`_profile_from_seeds`, `_rank_cooccur_from_profile`)
    are looked up as module globals by all four call sites, so one rebind
    covers every lane and the arm costs zero product diff.
    """
    import silica.kernel.recall.relatedness as R

    saved = {name: getattr(R, name) for name in attrs}
    for name, fn in attrs.items():
        setattr(R, name, fn)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(R, name, fn)


def _profile_builder(fn):
    return _swap(_profile_from_seeds=fn)


# ---------------------------------------------------------------------------
# Metric, mirroring health.fusion_probe exactly
# ---------------------------------------------------------------------------

def _score(topk: dict[str, list[str]], eligible: list[tuple[str, str]]) -> dict:
    hits, rr_sum = 0, 0.0
    per_pair: dict[tuple[str, str], bool] = {}
    for a, b in eligible:
        ranks = []
        if b in topk[a]:
            ranks.append(topk[a].index(b) + 1)
        if a in topk[b]:
            ranks.append(topk[b].index(a) + 1)
        per_pair[(a, b)] = bool(ranks)
        if ranks:
            hits += 1
            rr_sum += 1.0 / min(ranks)
    n = len(eligible)
    return {"recall_at_10": round(hits / n, 4), "mrr": round(rr_sum / n, 4),
            "hits": hits, "pairs": n, "per_pair": per_pair}


def _run_arm(endpoints, eligible, store, es, *, expand=False,
             profile_fn=None, rank_fn=None) -> dict:
    from silica.kernel.recall.relatedness import related_notes

    swaps = {}
    if profile_fn:
        swaps["_profile_from_seeds"] = profile_fn
    if rank_fn:
        swaps["_rank_cooccur_from_profile"] = rank_fn
    ctx = _swap(**swaps) if swaps else contextlib.nullcontext()
    t0 = time.perf_counter()
    with ctx:
        topk = {key: [r.path for r in related_notes(
            key, embed_store=es, cooccur_store=store, k=K, expand=expand)]
            for key in endpoints}
    elapsed = (time.perf_counter() - t0) * 1000 / len(endpoints)
    res = _score(topk, eligible)
    res["ms_per_endpoint"] = round(elapsed, 1)
    res["topk"] = topk
    return res


# ---------------------------------------------------------------------------
# Measure 1: the ceiling of the profile lever
# ---------------------------------------------------------------------------

def _legs(key: str, store, es, *, pool: int):
    from silica.kernel.recall.relatedness import _cooccur_ranking, _embed_ranking

    blocked = {key}
    embed_rank = _embed_ranking(es, key, k=pool, exclude=blocked)
    cooc_rank = _cooccur_ranking(store, key, k=pool, exclude=blocked, expand=False)
    row = store.note_edges_for(key)
    ranked = sorted(((p, s) for p, s in row.items() if p not in blocked),
                    key=lambda kv: (-kv[1], kv[0]))
    return embed_rank, cooc_rank, (ranked or None)


def _fused(embed_rank, cooc_rank, edges_rank) -> dict[str, float]:
    """Reconstruct `_fuse`'s scores. Self-checked against `related_notes`."""
    from silica.kernel.recall.graph_export import is_vault_artifact
    from silica.kernel.recall.relatedness import _rrf_fuse

    rankings = []
    if embed_rank is not None:
        rankings.append([(p, s) for p, _n, s in embed_rank])
    if cooc_rank is not None:
        rankings.append(list(cooc_rank))
    if edges_rank is not None:
        rankings.append(list(edges_rank))
    return {p: s for p, s in _rrf_fuse(rankings).items() if not is_vault_artifact(p)}


def _beats(score: float, target: str, others: dict[str, float]) -> int:
    """How many competitors outrank `target` at `score`, with `_fuse`'s tie-break."""
    return sum(1 for p, s in others.items()
               if s > score or (s == score and p < target))


def ceiling(missed, store, es, *, pool: int, arm_a_topk: dict[str, list[str]]) -> dict:
    """For each missed pair: the best cooccur rank its counterpart would need.

    Two bounds per direction, both optimistic for the lever:
      `p_max`   competitors keep today's fused score, target promoted to rank p.
      `savable` absolute ceiling: target at cooccur rank 1 while every
                competitor loses its cooccur term. If this fails, no cooccur
                ranking whatsoever puts the pair in the top-10.
    """
    from silica.kernel.recall.relatedness import RRF_K

    cache: dict[str, tuple] = {}

    def legs(key):
        if key not in cache:
            e, c, g = _legs(key, store, es, pool=pool)
            base = _fused(e, None, g)
            full = _fused(e, c, g)
            top = [p for p, _s in sorted(full.items(), key=lambda kv: (-kv[1], kv[0]))[:K]]
            assert top == arm_a_topk[key], f"fusion reconstruction drifted for {key}"
            cache[key] = (base, full, {p: i + 1 for i, (p, _s) in enumerate(c or [])})
        return cache[key]

    rows = []
    for a, b in missed:
        cands = []
        for src, dst in ((a, b), (b, a)):
            base, full, cooc_pos = legs(src)
            others_full = {p: s for p, s in full.items() if p != dst}
            others_base = {p: s for p, s in base.items() if p != dst}
            floor = base.get(dst, 0.0)
            p_max = 0
            for p in range(1, pool + 1):
                if _beats(floor + 1.0 / (RRF_K + p), dst, others_full) < K:
                    p_max = p
            cands.append({"p_max": p_max,
                          "savable": _beats(floor + 1.0 / (RRF_K + 1), dst, others_base) < K,
                          "p_now": cooc_pos.get(dst)})
        # One row per pair, from the easier of the two directions: the pair is
        # recovered if EITHER endpoint surfaces the other.
        best = max(cands, key=lambda c: (c["savable"], c["p_max"]))
        rows.append({"pair": [a, b], **best,
                     "in_pool_either": any(c["p_now"] for c in cands)})

    reachable = [r for r in rows if r["p_max"] > 0]
    jumps = [(r["p_now"], r["p_max"]) for r in reachable if r["p_now"]]
    return {
        "missed": len(rows),
        "reachable_now": len(reachable),
        "savable_at_best": sum(1 for r in rows if r["savable"]),
        "p_max_median": statistics.median([r["p_max"] for r in reachable]) if reachable else 0,
        "in_pool_either_direction": sum(1 for r in rows if r["in_pool_either"]),
        "in_pool_needing_jump": len(jumps),
        "jump_median": statistics.median([n - m for n, m in jumps]) if jumps else 0,
        "rows": rows,
    }


# ---------------------------------------------------------------------------

def run(vault: Path, *, hops: int = HOPS, alpha: float = ALPHA,
        top_n: int = TOP_STEMS, sweep: bool = False) -> dict:
    from evals.golden.runner import _open_stores, vault_digest
    from silica.kernel.recall.relatedness import _POOL_MIN

    store, embed_store = _open_stores(vault)
    es = embed_store if (embed_store is not None and len(embed_store)) else None
    digest, notes = vault_digest(vault)
    pool = max(K * 3, _POOL_MIN)
    print(f"vault {vault}  ({notes} notes, {digest[:19]}…)  embed leg: {'live' if es else 'OFF'}")

    eligible = eligible_pairs(wikilink_graph(vault, store))
    endpoints = sorted({e for pr in eligible for e in pr})
    print(f"{len(eligible)} eligible pairs, {len(endpoints)} endpoints, cooccur pool top-{pool}")

    # --- arm A first: it is both the baseline and the input to the ceiling ---
    arm_a = _run_arm(endpoints, eligible, store, es)
    missed = [p for p, ok in arm_a["per_pair"].items() if not ok]
    print(f"\nA baseline: recall@{K} {arm_a['recall_at_10']:.4f} "
          f"({arm_a['hits']}/{arm_a['pairs']}), mrr {arm_a['mrr']:.4f}, "
          f"{arm_a['ms_per_endpoint']}ms/endpoint, {len(missed)} missed")

    # --- 1. CEILING -------------------------------------------------------
    ceil = ceiling(missed, store, es, pool=pool, arm_a_topk=arm_a["topk"])
    n_pairs = len(eligible)
    print(f"\n1. CEILING of the profile lever over the {ceil['missed']} missed pairs:")
    print(f"   savable by ANY cooccur ranking: {ceil['savable_at_best']} "
          f"({ceil['savable_at_best']/ceil['missed']:.1%} of missed, "
          f"+{ceil['savable_at_best']/n_pairs:.2%} recall if ALL were recovered)")
    print(f"   reachable with competitors unchanged: {ceil['reachable_now']}, "
          f"median rank needed {ceil['p_max_median']:.0f} of {pool}")
    print(f"   already in the cooccur pool (either direction): "
          f"{ceil['in_pool_either_direction']}; of the reachable ones "
          f"{ceil['in_pool_needing_jump']} are in-pool, median jump "
          f"{ceil['jump_median']:.0f} positions")
    headroom = ceil["savable_at_best"] / n_pairs
    if headroom < GATE_DELTA:
        print(f"   VERDICT: ceiling +{headroom:.2%} < gate +{GATE_DELTA:.0%} — "
              f"the lever cannot pass the primary even if perfect")
    else:
        print(f"   VERDICT: ceiling +{headroom:.2%} >= gate +{GATE_DELTA:.0%} — "
              f"arms are worth running")

    # --- 2. ARMS ----------------------------------------------------------
    sizes: list[int] = []
    arm_b = _run_arm(endpoints, eligible, store, es, expand=True)
    arm_c = _run_arm(endpoints, eligible, store, es,
                     profile_fn=_ppr_builder(hops=hops, alpha=alpha, top_n=top_n, sizes=sizes))
    print(f"\n2. ARMS (reranker off in all three, same {n_pairs} pairs)")
    for name, arm in (("A baseline   expand=False", arm_a),
                      ("B control    expand=True ", arm_b),
                      (f"C PPR        h={hops} a={alpha} n={top_n}", arm_c)):
        print(f"   {name}  recall@{K} {arm['recall_at_10']:.4f}  mrr {arm['mrr']:.4f}  "
              f"{arm['ms_per_endpoint']:>6.1f}ms/endpoint")
    if sizes:
        print(f"   arm C profile size: median {statistics.median(sizes):.0f} stems "
              f"(arm A seeds only)")

    gate = _gate(arm_a, arm_c)
    print(f"\n   primary   C - A = {gate['delta_recall']:+.4f} "
          f"(gate +{GATE_DELTA:.2f}): {'PASS' if gate['primary'] else 'FAIL'}")
    print(f"   mrr       C - A = {gate['delta_mrr']:+.4f} "
          f"(tolerance -{MRR_TOLERANCE:.2f}): {'ok' if gate['mrr_ok'] else 'REGRESSED'}")
    # paired(C, A): its "a_only" is C-right/A-wrong, "b_only" the reverse.
    print(f"   McNemar   discordant C-only {gate['discordant']['a_only']} / "
          f"A-only {gate['discordant']['b_only']}, p={gate['mcnemar_p']:.4f}"
          f" ({'significant' if gate['significant_05'] else 'not significant'})")
    print(f"   control   B - A = {arm_b['recall_at_10'] - arm_a['recall_at_10']:+.4f}"
          f"   C - B = {arm_c['recall_at_10'] - arm_b['recall_at_10']:+.4f}")
    print(f"\n   GATE: {gate['verdict']}")

    out = {
        "vault": {"path": str(vault), "digest": digest, "notes": notes},
        "config": {"hops": hops, "alpha": alpha, "top_stems": top_n,
                   "pool": pool, "k": K},
        "pairs_evaluated": n_pairs,
        "ceiling": {k: v for k, v in ceil.items() if k != "rows"},
        "ceiling_headroom": round(headroom, 4),
        "arms": {name: {k: v for k, v in arm.items() if k not in ("per_pair", "topk")}
                 for name, arm in (("A", arm_a), ("B", arm_b), ("C", arm_c))},
        "gate": gate,
    }

    if sweep:
        out["sweep"] = _sweep(endpoints, eligible, store, es, arm_a=arm_a, top_n=top_n)
    return out


def _gate(arm_a: dict, arm_c: dict) -> dict:
    from evals.paired_stats import paired

    def doc(arm):
        return {"questions": [{"question_id": f"{a}|{b}", "correct": ok}
                              for (a, b), ok in arm["per_pair"].items()]}

    st = paired(doc(arm_c), doc(arm_a))          # delta = C - A
    delta_recall = round(arm_c["recall_at_10"] - arm_a["recall_at_10"], 4)
    delta_mrr = round(arm_c["mrr"] - arm_a["mrr"], 4)
    primary = delta_recall >= GATE_DELTA
    mrr_ok = delta_mrr >= -MRR_TOLERANCE
    passed = primary and mrr_ok and st["significant_05"]
    if passed:
        verdict = "PASS — productionise behind a CONFIG flag, then confirm answer-side"
    elif not primary:
        verdict = "FAIL on the primary — lever stays unwritten, negative result stands"
    elif not st["significant_05"]:
        verdict = "FAIL on significance — the delta is inside noise"
    else:
        verdict = "FAIL on mrr non-regression"
    return {"delta_recall": delta_recall, "delta_mrr": delta_mrr,
            "primary": primary, "mrr_ok": mrr_ok,
            "mcnemar_p": st["mcnemar_p"], "significant_05": st["significant_05"],
            "discordant": st["discordant"], "delta_ci95": st["delta_ci95"],
            "verdict": verdict}


def _sweep(endpoints, eligible, store, es, *, arm_a: dict, top_n: int) -> dict:
    """Pre-declared sweep, reported in full — no cherry-picking the best cell."""
    base = arm_a["recall_at_10"]
    grid = {}
    print(f"\n3. SWEEP (pre-declared, reported in full; A = {base:.4f})")
    for h in SWEEP_HOPS:
        for a in SWEEP_ALPHA:
            arm = _run_arm(endpoints, eligible, store, es,
                           profile_fn=_ppr_builder(hops=h, alpha=a, top_n=top_n))
            grid[f"hops={h},alpha={a}"] = arm["recall_at_10"]
            print(f"   hops={h} alpha={a}  recall@{K} {arm['recall_at_10']:.4f} "
                  f"({arm['recall_at_10']-base:+.4f})  {arm['ms_per_endpoint']:>6.1f}ms")
    trunc = {}
    for n in SWEEP_TOP_STEMS:
        arm = _run_arm(endpoints, eligible, store, es,
                       profile_fn=_ppr_builder(hops=HOPS, alpha=ALPHA, top_n=n))
        trunc[f"top_stems={n}"] = arm["recall_at_10"]
        print(f"   top_stems={n:<5} recall@{K} {arm['recall_at_10']:.4f} "
              f"({arm['recall_at_10']-base:+.4f})  {arm['ms_per_endpoint']:>6.1f}ms")

    # Post-hoc, NOT part of the pre-declared grid, and run after the gate had
    # already failed: alpha is a continuum between "propagate" and "do not".
    # At alpha=1.0 no mass leaves the seeds, so arm C must reproduce arm A
    # exactly — that is also the validity check on the whole swap harness.
    cont = {}
    print("   post-hoc continuity in alpha (not pre-declared, gate already decided):")
    for a in (0.9, 1.0):
        arm = _run_arm(endpoints, eligible, store, es,
                       profile_fn=_ppr_builder(hops=HOPS, alpha=a, top_n=top_n))
        cont[f"alpha={a}"] = arm["recall_at_10"]
        note = ""
        if a == 1.0:
            note = ("   [harness check: identical to A]" if arm["recall_at_10"] == base
                    else "   [HARNESS BUG: alpha=1.0 must equal A]")
        print(f"   hops={HOPS} alpha={a}  recall@{K} {arm['recall_at_10']:.4f} "
              f"({arm['recall_at_10']-base:+.4f}){note}")
    return {"grid": grid, "truncation": trunc, "alpha_continuity_posthoc": cont,
            "arm_a": base}


DIAGNOSE_CONFIGS = (
    ("alpha", 0.3, 300), ("alpha", 0.5, 300), ("alpha", 0.7, 300), ("alpha", 0.9, 300),
    ("top", 0.5, 100), ("top", 0.5, 1000),
)


def diagnose(vault: Path, *, samples: int = 100, record: Path | None = None) -> dict:
    """Name the failure mode instead of only reporting it.

    The arms say propagation loses monotonically in propagated mass. This asks
    WHERE that mass lands. `_rank_cooccur_from_profile` scores a note by
    sum(profile_weight * tf * idf), so a profile whose mass sits on common
    (low-IDF) stems rewards notes that share generic concepts. Degree
    normalisation controls what a hub EMITS; PageRank centrality is about what
    a node RECEIVES, and nothing in the walk controls that.
    """
    import random

    from evals.golden.runner import _open_stores
    from silica.kernel.recall.relatedness import _concept_idf

    store, _embed = _open_stores(vault)
    adj = store.adjacency()
    rng = random.Random(0)
    keys = rng.sample(sorted(store.paths()), min(samples, len(store)))

    sweep = {}
    if record and record.exists():
        doc = json.loads(record.read_text(encoding="utf-8"))
        sweep = {**doc.get("sweep", {}).get("grid", {}),
                 **doc.get("sweep", {}).get("truncation", {}),
                 **doc.get("sweep", {}).get("alpha_continuity_posthoc", {})}
        base = doc.get("sweep", {}).get("arm_a")
    else:
        base = None

    def stats(profile: dict[str, float], seeds: dict[str, float]) -> tuple[float, float]:
        idf = _concept_idf(store, set(profile), scope=None)
        mass = sum(profile.values())
        off_seed = sum(w for s, w in profile.items() if s not in seeds)
        mean_idf = sum(w * idf.get(s, 0.0) for s, w in profile.items()) / mass if mass else 0.0
        return (off_seed / mass if mass else 0.0), mean_idf

    print(f"\n4. WHERE THE MASS LANDS ({len(keys)} sampled notes)")
    print("   config                 off-seed mass   mean IDF   profile stems   recall@10")
    rows = {}
    for axis, alpha, top_n in (("seeds", 1.0, 0), *DIAGNOSE_CONFIGS):
        off, idfs, sizes = [], [], []
        for key in keys:
            seeds = store.note_nodes(key)
            profile = (dict(seeds) if axis == "seeds"
                       else _truncate(ppr(adj, seeds, hops=HOPS, alpha=alpha), top_n))
            o, m = stats(profile, seeds)
            off.append(o)
            idfs.append(m)
            sizes.append(len(profile))
        label = "A seeds only" if axis == "seeds" else f"alpha={alpha} top={top_n}"
        key_sweep = (f"hops={HOPS},alpha={alpha}" if axis == "alpha"
                     else f"top_stems={top_n}")
        rec = base if axis == "seeds" else sweep.get(key_sweep, sweep.get(f"alpha={alpha}"))
        rows[label] = {"off_seed_mass": round(statistics.fmean(off), 4),
                       "mean_idf": round(statistics.fmean(idfs), 3),
                       "profile_stems": round(statistics.fmean(sizes)),
                       "recall_at_10": rec}
        rec_s = f"{rec:.4f}" if rec is not None else "n/a"
        print(f"   {label:<22} {statistics.fmean(off):>12.1%}   {statistics.fmean(idfs):>8.3f}   "
              f"{statistics.fmean(sizes):>13.0f}   {rec_s:>9}")
    return rows


def main(argv=None) -> int:
    from evals.golden.runner import resolve_vault

    ap = argparse.ArgumentParser(prog="python -m evals.probe_ppr_phase2")
    ap.add_argument("--vault")
    ap.add_argument("--hops", type=int, default=HOPS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--top-stems", type=int, default=TOP_STEMS)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--diagnose", action="store_true",
                    help="only measure where the walk puts its mass; joins the "
                         "recall of a previous --sweep run from --json")
    ap.add_argument("--json", default="bench/ppr_phase2.json")
    args = ap.parse_args(argv)

    vault = resolve_vault(args.vault)
    out = Path(args.json)
    try:
        if args.diagnose:
            res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
            res["mass_placement"] = diagnose(vault, record=out)
        else:
            res = run(vault, hops=args.hops, alpha=args.alpha,
                      top_n=args.top_stems, sweep=args.sweep)
    finally:
        import silica.driver
        silica.driver._driver = None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
