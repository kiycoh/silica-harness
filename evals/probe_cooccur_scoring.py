# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Is the cooccur leg's SCORING function the lever the PPR ceiling pointed at?

`docs/spec-ppr-propagation.md` section 5-ter closed the profile-builder
direction: adding concepts to the profile is measurably harmful. But its
ceiling measure said the room is still there (108 of 117 missed pairs reachable
at median rank 12 of 30) and phase 0 said 97.4% of missed pairs ALREADY share a
concept. So the untested lever is not which concepts enter the profile, it is
how the shared ones are weighted.

`_rank_cooccur_from_profile` scores a note as sum(w * tf * idf): tf enters
linearly and unbounded, and nothing normalises for note length. A note that
repeats one concept 30 times outscores one that mentions it once by 30x, and a
long note accumulates more terms in the sum purely by bulk. That is exactly the
pair of defects BM25 exists to fix, with two independent knobs:

  k1  tf saturation      does a note win by repeating a concept?
  b   length normalisation  does a note win by being long?

Same discipline as phase 2: zero product diff. `_rank_cooccur_from_profile` is
resolved as a module global by all four call sites, so an arm is a rebind.
Textbook defaults, declared here before the run, not tuned afterwards.

  uv run python -m evals.probe_cooccur_scoring --vault ~/Documents/Obsidian/test
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from evals.probe_ppr_phase2 import GATE_DELTA, _gate, _run_arm
from silica.kernel.link.health import K, eligible_pairs, wikilink_graph

K1 = 1.2      # textbook BM25 default, not tuned
B = 0.75      # textbook BM25 default, not tuned
_LENGTHS: dict[int, tuple[dict[str, int], float]] = {}


def _lengths(store) -> tuple[dict[str, int], float]:
    """{path: total stem count} and the mean, cached per store.

    Keyed by store identity because the memory lane passes a DIFFERENT store to
    the same seam; borrowing the active vault's lengths there would be a silent
    cross-vault bug even though the golden path never exercises it.
    """
    key = id(store)
    if key not in _LENGTHS:
        lens = {p: sum(store.note_nodes(p).values()) for p in store.paths()}
        _LENGTHS[key] = (lens, statistics.fmean(lens.values()) if lens else 1.0)
    return _LENGTHS[key]


def bm25_ranker(*, k1: float = K1, b: float = B, binary: bool = False):
    """A `_rank_cooccur_from_profile` drop-in with a BM25 tf term.

    Everything else is the production function verbatim: same candidate set
    (union of profile-stem postings), same blocked/scope filters, same abstain
    on no overlap, same tie-break. Only the tf term changes, so an arm measures
    the tf term and nothing else. The production confidence gate is skipped: it
    is inert (`_COOCCUR_MIN_CONFIDENCE == 0.0`, no probe hook) on every path
    this harness exercises.
    """

    def rank(cooccur_store, profile, *, k, blocked, scope):
        from silica.kernel.recall.paths import in_folder as _path_in_scope
        from silica.kernel.recall.relatedness import _concept_idf

        if not profile:
            return None
        idf = _concept_idf(cooccur_store, set(profile), scope=scope)
        postings = cooccur_store.stem_postings()
        lens, avgdl = _lengths(cooccur_store)
        candidates: set[str] = set()
        for stem in profile:
            plist = postings.get(stem)
            if plist:
                candidates.update(plist)
        note_scores: dict[str, float] = {}
        for path in candidates:
            if path in blocked or not _path_in_scope(path, scope):
                continue
            norm = k1 * (1.0 - b + b * (lens.get(path, 0) or 1) / avgdl)
            overlap = 0.0
            for stem, weight in profile.items():
                if not weight:
                    continue
                plist = postings.get(stem)
                if plist and path in plist:
                    tf = plist[path]
                    term = 1.0 if binary else tf * (k1 + 1.0) / (tf + norm)
                    overlap += weight * term * idf.get(stem, 0.0)
            if overlap > 0.0:
                note_scores[path] = overlap
        if not note_scores:
            return None
        return sorted(note_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    return rank


# ---------------------------------------------------------------------------
# Measure 1: does the premise hold at all?
# ---------------------------------------------------------------------------

def length_bias(store, groups: dict[str, list[tuple[str, str]]]) -> dict:
    """Is the gain real, or is the GROUND TRUTH biased toward short notes?

    A scorer that penalises length gets a free lift on any metric whose
    positives happen to be short, and that lift would not generalise past this
    probe. So: where do the endpoints of each group sit in the vault's own
    length distribution? Wikilinked pairs near the vault median means the
    ground truth is not length-biased and the gain has to come from ranking.
    """
    import bisect

    lens, _avg = _lengths(store)
    allv = sorted(lens.values())

    def pct(paths) -> float:
        if not paths:
            return 0.0
        return statistics.median(
            bisect.bisect_left(allv, lens.get(p, 0)) / len(allv) for p in paths)

    out = {"vault_median_length": statistics.median(allv) if allv else 0}
    for label, pairs in groups.items():
        endpoints = [e for pr in pairs for e in pr]
        out[label] = {"pairs": len(pairs),
                      "median_length": statistics.median(
                          [lens.get(e, 0) for e in endpoints]) if endpoints else 0,
                      "median_length_percentile": round(pct(endpoints), 3)}
        print(f"   {label:<26} {out[label]['pairs']:>4} pairs, endpoint median "
              f"{out[label]['median_length']:>5.0f} stems "
              f"(vault percentile {out[label]['median_length_percentile']:.2f})")
    return out


# ---------------------------------------------------------------------------

def run(vault: Path, *, k1: float = K1, b: float = B) -> dict:
    from evals.golden.runner import _open_stores, vault_digest
    from silica.config import CONFIG
    from silica.kernel.recall.relatedness import _POOL_MIN

    # Arm A is the frozen state whatever the operator's env says, and arm P below
    # turns the product flag on deliberately. Pinned here so a stray
    # SILICA_COOCCUR_BM25=1 cannot quietly make the baseline the treatment.
    _flag_was = CONFIG.cooccur_bm25
    CONFIG.cooccur_bm25 = False

    store, embed_store = _open_stores(vault)
    es = embed_store if (embed_store is not None and len(embed_store)) else None
    digest, notes = vault_digest(vault)
    pool = max(K * 3, _POOL_MIN)
    print(f"vault {vault}  ({notes} notes, {digest[:19]}…)  embed leg: {'live' if es else 'OFF'}")

    eligible = eligible_pairs(wikilink_graph(vault, store))
    endpoints = sorted({e for pr in eligible for e in pr})

    arm_a = _run_arm(endpoints, eligible, store, es)
    missed = [p for p, ok in arm_a["per_pair"].items() if not ok]
    print(f"{len(eligible)} eligible pairs, {len(endpoints)} endpoints, "
          f"A recall@{K} {arm_a['recall_at_10']:.4f}, {len(missed)} missed")

    print(f"\n1. ARMS (same {len(eligible)} pairs, only the tf term differs)")
    arms = {"A": arm_a}
    for name, label, fn in (
        ("D", f"saturation only  k1={k1} b=0  ", bm25_ranker(k1=k1, b=0.0)),
        ("E", f"full BM25        k1={k1} b={b}", bm25_ranker(k1=k1, b=b)),
        ("F", "binary tf        (control)   ", bm25_ranker(binary=True)),
        ("V", "validity         k1=1e9 b=0  ", bm25_ranker(k1=1e9, b=0.0)),
    ):
        arms[name] = _run_arm(endpoints, eligible, store, es, rank_fn=fn)
        arms[name]["label"] = label

    # Arm P: the SHIPPED function under CONFIG.cooccur_bm25, not a swapped-in
    # ranker. It must reproduce E pair for pair. This is the arm that catches
    # fase 1 implementing something subtly different from what passed the gate
    # and nobody noticing because the number moved the right way anyway.
    CONFIG.cooccur_bm25 = True
    try:
        arms["P"] = _run_arm(endpoints, eligible, store, es)
    finally:
        CONFIG.cooccur_bm25 = _flag_was
    arms["P"]["label"] = "product flag     CONFIG.cooccur_bm25"
    p_matches_e = (k1, b) == (K1, B) and arms["P"]["per_pair"] == arms["E"]["per_pair"]

    print(f"   A baseline       linear tf     recall@{K} {arm_a['recall_at_10']:.4f}  "
          f"mrr {arm_a['mrr']:.4f}  {arm_a['ms_per_endpoint']:>6.1f}ms")
    for name in ("D", "E", "F", "V", "P"):
        arm = arms[name]
        note = ""
        if name == "V":
            note = ("   [harness check: identical to A]"
                    if arm["recall_at_10"] == arm_a["recall_at_10"]
                    else "   [HARNESS BUG: k1=1e9 must equal A]")
        if name == "P":
            note = ("   [product check: identical to E]" if p_matches_e else
                    "   [knobs overridden, E is not comparable]" if (k1, b) != (K1, B) else
                    "   [PRODUCT BUG: the flag does not reproduce E]")
        print(f"   {name} {arm['label']} recall@{K} {arm['recall_at_10']:.4f}  "
              f"mrr {arm['mrr']:.4f}  {arm['ms_per_endpoint']:>6.1f}ms"
              f"  ({arm['recall_at_10'] - arm_a['recall_at_10']:+.4f}){note}")

    gates = {}
    for name in ("D", "E", "F"):
        g = _gate(arm_a, arms[name])
        gates[name] = g
        print(f"\n   {name}: delta {g['delta_recall']:+.4f} (gate +{GATE_DELTA:.2f}) "
              f"{'PASS' if g['primary'] else 'FAIL'}, mrr {g['delta_mrr']:+.4f} "
              f"{'ok' if g['mrr_ok'] else 'REGRESSED'}, McNemar {name}-only "
              f"{g['discordant']['a_only']} / A-only {g['discordant']['b_only']}, "
              f"p={g['mcnemar_p']:.4f}")
    best = max(("D", "E", "F"), key=lambda n: gates[n]["delta_recall"])
    print(f"\n   BEST: {best} {gates[best]['verdict']}")

    # --- 2. is the ground truth itself short-biased? -----------------------
    bpp = arms[best]["per_pair"]
    gained = [p for p, ok in bpp.items() if ok and not arm_a["per_pair"][p]]
    lost = [p for p, ok in bpp.items() if not ok and arm_a["per_pair"][p]]
    print(f"\n2. LENGTH BIAS of the ground truth (vault median "
          f"{statistics.median(_lengths(store)[0].values()):.0f} stems)")
    bias = length_bias(store, {"all eligible pairs": list(eligible),
                               "A missed": missed,
                               f"{best} gained over A": gained,
                               f"{best} lost vs A": lost})

    return {
        "vault": {"path": str(vault), "digest": digest, "notes": notes},
        "config": {"k1": k1, "b": b, "pool": pool, "k": K},
        "pairs_evaluated": len(eligible),
        "length_bias": bias,
        "arms": {n: {k: v for k, v in arm.items() if k not in ("per_pair", "topk")}
                 for n, arm in arms.items()},
        "gates": gates,
        "best_arm": best,
        "product_reproduces_E": p_matches_e,
    }


def main(argv=None) -> int:
    from evals.golden.runner import resolve_vault

    ap = argparse.ArgumentParser(prog="python -m evals.probe_cooccur_scoring")
    ap.add_argument("--vault")
    ap.add_argument("--k1", type=float, default=K1)
    ap.add_argument("--b", type=float, default=B)
    ap.add_argument("--json", default="bench/cooccur_scoring.json")
    args = ap.parse_args(argv)

    vault = resolve_vault(args.vault)
    try:
        res = run(vault, k1=args.k1, b=args.b)
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
