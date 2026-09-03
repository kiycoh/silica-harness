# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Vault router bench: which signal sends a question to the vault that holds it.

`silica_vaults(query)` orders the vaults this machine knows in two stages
(silica/kernel/recall/vault_registry.py): each vault's own indexes NOMINATE a
pool of notes, a cross-encoder scores the pool, and the best score orders the
rows. This bench measures both stages against a frozen query set in which
every query names its home vault (or none), so three claims become numbers
anyone can regenerate:

- which first-stage signal routes (top cosine, cosine spread, two-term hit
  count) against the reranker's best score, as top-1 routing accuracy;
- how wide the nomination pool must be, as nominee recall of the expected
  note per pool size and rerank routing accuracy per pool size;
- where the abstention floor sits, as a curve of (floor -> homed queries
  still routed, homeless queries correctly refused), reported for the
  reranker the scores came from, because a logit is one model's scale.

One reranker call per (query, vault) at the widest pool; narrower pools are
prefixes of the same nominee lists, and a cross-encoder scores pointwise, so
the best score over a prefix is what a separate call would have returned.

Usage:
    SILICA_RERANK_BASE_URL=http://127.0.0.1:1235 SILICA_RERANK_MODEL=bge-reranker-v2-m3-Q8_0 \\
    uv run python scripts/bench_vault_router.py docs/evaluation/vault-router/queries.json \\
        --vault /path/to/active/vault --pools 3,6,12,25 --out results.json

The query file: {"vaults": {alias: path}, "queries": [{"q", "home": alias|null,
"note": substring of the expected note's index path (optional)}]}.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from silica.config import CONFIG  # noqa: E402
from silica.kernel.recall import vault_registry as reg  # noqa: E402

SIGNALS = ("cosine", "spread", "hits")


def _argmax(scores: dict[str, float | None]) -> str | None:
    """Alias with the strictly highest score; None on no score or a tie."""
    ranked = sorted(((s, a) for a, s in scores.items() if s is not None), reverse=True)
    if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
        return None
    return ranked[0][1]


def _abstention_curve(homed: list[tuple[bool, float | None]], homeless: list[float | None]) -> list[dict]:
    """For every candidate floor: the share of homed queries still routed to
    their home (right vault AND above the floor) and of homeless queries
    refused (best score below the floor). Candidates are the observed scores."""
    cands = sorted({s for _ok, s in homed if s is not None} | {s for s in homeless if s is not None})
    curve = []
    for floor in cands:
        routed = sum(1 for ok, s in homed if ok and s is not None and s >= floor) / max(len(homed), 1)
        refused = sum(1 for s in homeless if s is None or s < floor) / max(len(homeless), 1)
        curve.append({"floor": round(floor, 3), "routed": round(routed, 3), "refused": round(refused, 3),
                      "balanced": round((routed + refused) / 2, 3)})
    return curve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("queries", type=Path)
    ap.add_argument("--vault", help="active vault for this run (its store is the process singleton)")
    ap.add_argument("--pools", default="3,6,12,25", help="pool sizes per stage to compare")
    ap.add_argument("--timeout", type=float, default=120.0, help="reranker call timeout for the wide pool")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    pools = sorted({int(p) for p in args.pools.split(",")})
    pool_max = pools[-1]
    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    alias_of = {str(Path(p).resolve()): a for a, p in spec["vaults"].items()}

    if args.vault:
        CONFIG.vault_path = str(Path(args.vault).resolve())

    from silica.agent.providers import Reranker, get_embedder
    if not (CONFIG.rerank_base_url and CONFIG.rerank_model):
        print("no reranker configured: set SILICA_RERANK_BASE_URL and SILICA_RERANK_MODEL", file=sys.stderr)
        return 2
    # Own client: the served default times out at 5 s, sized for k=15 windows,
    # not for a 50-document pool.
    reranker = Reranker(base_url=CONFIG.rerank_base_url, model=CONFIG.rerank_model,
                        api_key=getattr(CONFIG, "rerank_api_key", ""), timeout=args.timeout)
    if reranker.scores("ping", ["ping"]) is None:
        print(f"reranker at {CONFIG.rerank_base_url} does not answer", file=sys.stderr)
        return 2
    get_embedder(CONFIG).embed(["ping"])  # a dead embedder fails here, not per query

    vaults = [v for v in reg.known_vaults() if reg.coverage(v)["level"] != "cold"]
    aliases = {v: alias_of.get(str(v), v.name) for v in vaults}
    missing = {q["home"] for q in spec["queries"] if q["home"]} - set(aliases.values())
    if missing:
        print(f"home vaults not indexed on this machine: {sorted(missing)}", file=sys.stderr)
        return 2
    print("vaults: " + ", ".join(f"{aliases[v]}={reg.coverage(v)['level']}" for v in vaults))
    print(f"reranker: {reg.reranker_name(reranker)}  pools: {pools}  queries: {len(spec['queries'])}")

    rows = []
    t_all = time.perf_counter()
    for i, q in enumerate(spec["queries"], 1):
        query, home, note = q["q"], q.get("home"), (q.get("note") or "").lower()
        vec = reg._query_vec(query)
        per_vault: dict[str, dict] = {}
        for v in vaults:
            a = aliases[v]
            t0 = time.perf_counter()
            nom = reg.nominate(v, query, pool=pool_max, query_vec=lambda: vec)
            t_nom = time.perf_counter() - t0
            union = reg.pool_union(nom, pool_max)
            t0 = time.perf_counter()
            scores = reg.rerank_nominees(v, query, union, reranker) if union else None
            t_rr = time.perf_counter() - t0
            by_path = dict(zip((p for p, _n in union), scores or []))
            cos = [c for _p, _n, c in nom["embed"]]
            row = {
                "cosine": nom["best"],
                "spread": (round(cos[0] - statistics.median(cos), 4) if len(cos) >= 3 else None),
                "hits": nom["hits"],
                "rerank": {}, "nominated": {}, "top_is_note": {}, "top_path": {},
                "pool_docs": {}, "t_nominate": round(t_nom, 3), "t_rerank": round(t_rr, 3),
                "scored": scores is not None,
            }
            for p in pools:
                sub = reg.pool_union(nom, p)
                row["pool_docs"][p] = len(sub)
                got = [(by_path[pp], pp) for pp, _n in sub if pp in by_path]
                row["rerank"][p] = round(max(got)[0], 3) if got else None
                row["top_path"][p] = max(got)[1] if got else None
                if note:
                    row["nominated"][p] = any(note in pp.lower() for pp, _n in sub)
                    row["top_is_note"][p] = bool(got) and note in max(got)[1].lower()
            per_vault[a] = row
        picks = {s: _argmax({a: r[s] for a, r in per_vault.items()}) for s in SIGNALS}
        for p in pools:
            picks[f"rerank@{p}"] = _argmax({a: r["rerank"][p] for a, r in per_vault.items()})
        rows.append({"q": query, "home": home, "note": note or None, "picks": picks, "vaults": per_vault})
        best = per_vault.get(home) if home else None
        print(f"[{i:2d}/{len(spec['queries'])}] {query[:58]:58s} home={home or '-':9s} "
              f"rerank@{pool_max}: " + " ".join(f"{a}={r['rerank'][pool_max]}" for a, r in per_vault.items())
              + (f"  nominated@{pools[0]}={best['nominated'].get(pools[0])}" if best and note else ""))

    homed = [r for r in rows if r["home"]]
    homeless = [r for r in rows if not r["home"]]
    summary: dict = {
        "date": time.strftime("%Y-%m-%d"), "reranker": reg.reranker_name(reranker),
        "embedder": getattr(get_embedder(CONFIG), "model", ""),
        "vaults": {aliases[v]: str(v) for v in vaults}, "pools": pools,
        "n_homed": len(homed), "n_homeless": len(homeless),
        "elapsed_s": round(time.perf_counter() - t_all, 1),
        "routing_accuracy": {}, "nominee_recall": {}, "top_is_note": {}, "abstention": {},
        "mean_pool_docs": {}, "mean_t_rerank_s": round(statistics.mean(
            r["t_rerank"] for row in rows for r in row["vaults"].values()), 3),
    }
    for s in SIGNALS:
        summary["routing_accuracy"][s] = round(sum(1 for r in homed if r["picks"][s] == r["home"]) / len(homed), 3)
    for p in pools:
        key = f"rerank@{p}"
        summary["routing_accuracy"][key] = round(sum(1 for r in homed if r["picks"][key] == r["home"]) / len(homed), 3)
        labelled = [r for r in homed if r["note"]]
        summary["nominee_recall"][p] = round(sum(
            1 for r in labelled if r["vaults"][r["home"]]["nominated"][p]) / max(len(labelled), 1), 3)
        summary["top_is_note"][p] = round(sum(
            1 for r in labelled if r["vaults"][r["home"]]["top_is_note"][p]) / max(len(labelled), 1), 3)
        summary["mean_pool_docs"][p] = round(statistics.mean(
            r["pool_docs"][p] for row in rows for r in row["vaults"].values()), 1)
        homed_pts = [(r["picks"][key] == r["home"],
                      max((v["rerank"][p] for v in r["vaults"].values() if v["rerank"][p] is not None), default=None))
                     for r in homed]
        homeless_pts = [max((v["rerank"][p] for v in r["vaults"].values() if v["rerank"][p] is not None), default=None)
                        for r in homeless]
        curve = _abstention_curve(homed_pts, homeless_pts)
        best = max(curve, key=lambda c: (c["balanced"], c["floor"])) if curve else None
        # Runner-up margin, for the record: does "top1 - top2" add anything the floor lacks?
        margins = []
        for r in homed:
            sc = sorted((v["rerank"][p] for v in r["vaults"].values() if v["rerank"][p] is not None), reverse=True)
            if len(sc) >= 2 and r["picks"][key] == r["home"]:
                margins.append(round(sc[0] - sc[1], 3))
        summary["abstention"][p] = {
            "best_floor": best, "curve": curve,
            "homed_top_scores": sorted(round(s, 3) for _ok, s in homed_pts if s is not None),
            "homeless_top_scores": sorted(round(s, 3) for s in homeless_pts if s is not None),
            "routed_margins": sorted(margins),
        }

    print("\n== routing accuracy (top-1, homed queries) ==")
    for k_, v_ in summary["routing_accuracy"].items():
        print(f"  {k_:12s} {v_:.3f}")
    print("== nominee recall of the expected note, per pool ==")
    for p in pools:
        print(f"  pool {p:2d}: nominated {summary['nominee_recall'][p]:.3f}  reranked to top {summary['top_is_note'][p]:.3f}"
              f"  docs/vault {summary['mean_pool_docs'][p]}")
    print("== abstention (best floor by balanced accuracy), per pool ==")
    for p in pools:
        b = summary["abstention"][p]["best_floor"]
        print(f"  pool {p:2d}: floor {b['floor']:+.3f} routed {b['routed']:.3f} refused {b['refused']:.3f}"
              f"  homeless tops {summary['abstention'][p]['homeless_top_scores']}")
    print(f"elapsed {summary['elapsed_s']} s, mean rerank call {summary['mean_t_rerank_s']} s")

    misses = [(r["q"], r["home"], r["picks"][f"rerank@{pool_max}"]) for r in homed
              if r["picks"][f"rerank@{pool_max}"] != r["home"]]
    if misses:
        print("== rerank misses at the widest pool ==")
        for q_, h_, p_ in misses:
            print(f"  {q_[:60]:60s} home={h_} picked={p_}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
