# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Golden coherence harness runner — CLI + manifest digest + report + metrics.

One run output, three consumers: acceptance read (the printed table),
regression gate (``tests/golden/test_golden_regression.py`` imports ``collect``
+ ``compare``), and calibration instrument (``--verbose``).

Phase 1 = cheap/deterministic tier (classify, links, integrity). Embedder-tier
probes (dedup, neighbors) print a visible SKIP row.

  uv run python -m evals.golden --vault ~/Documents/Obsidian/test [--verbose]
  uv run python -m evals.golden --vault <v> --freeze-baseline
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

# Gated probes (fusion_probe, integrity_probe) live in kernel.health — shared
# with the silica_health tool; iter_notes re-exported so probes keep one source.
from silica.kernel.link.health import fusion_probe, integrity_probe, iter_notes  # noqa: F401
from evals.golden import (
    probe_classify,
    probe_correlate,
    probe_dedup,
    probe_fusion,
    probe_links,
    probe_supersede,
)

BASELINE_PATH = Path(__file__).parent / "baseline.json"
METRICS_PATH = Path(__file__).parent / "metrics.json"

# Gate rules (single source — the pytest imports compare()).
# fusion.recall_at_10 arms itself on the first freeze that records it
# (compare() skips keys absent from the baseline).
GATED_DROP_2PP = ("classify.agreement", "links.recall", "fusion.recall_at_10")
# Rates where HIGH is bad: a rise past tolerance fails (mirror image of DROP).
# dedup.tp_leak_rate self-arms — compare() only reads keys present in both docs,
# and the TP metric is recorded only when its labels resolve (pairs>0).
# supersede.* self-arm the same way: the wrong rate needs the embed leg for its
# pairs, the missed rate needs a contested note. §8.2 asks for the missed rate to
# be informational on the first run and gated after — which is what recording it
# in a baseline does, with no separate switch to remember to flip.
GATED_RISE_2PP = ("dedup.fp_auto_merge_rate", "dedup.tp_leak_rate",
                  "supersede.missed_resolution_rate")
GATED_EXACT_ONE = ("integrity.rate",)
# A merge inversion is a defect, not a rate to tolerate: `merge_rank` makes one
# structurally impossible, so any count above zero is a regression. Gated as a
# COUNT because the rate could not carry it — a full revert of §6.2 measured
# 0.0209 against a 2pp tolerance, i.e. it failed by 0.09pp and a partial revert
# would have slipped under. Same shape as integrity.rate: any new violation fails.
GATED_EXACT_ZERO = ("supersede.wrong_merges",)
_PRIMARIES = ("classify.agreement", "links.recall", "integrity.rate")

# Metrics that need the embed leg live. When the on-disk index is absent
# (deleted to reclaim RAM — it is a rebuildable derived artifact), the embed
# leg abstains and these become a different instrument, not a regression: the
# gate SKIPs them instead of failing, so an offline run never reads as drift.
_EMBED_DEPENDENT = ("fusion.recall_at_10", "dedup.fp_auto_merge_rate", "dedup.tp_leak_rate")


def _embed_live(legs: str | None) -> bool:
    return "embed" in (legs or "")


def _nonembed_legs(legs: str | None) -> str:
    """The leg set minus the embed leg — the part whose change is real drift."""
    return (legs or "").replace("embed+", "").replace("embed", "")


def _today() -> str:
    return datetime.date.today().isoformat()


def resolve_vault(cli: str | None) -> Path:
    """--vault > SILICA_VAULT; expanduser+resolve; exit 2 if missing or empty."""
    raw = cli or os.getenv("SILICA_VAULT")
    if not raw:
        print("no vault: pass --vault or set SILICA_VAULT")
        raise SystemExit(2)
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        print(f"vault not found: {p}")
        raise SystemExit(2)
    if not iter_notes(p):
        # Refused here, before any probe: a zero-note run would hand compare()
        # a 0.0 agreement and an unmeasured integrity rate, and --freeze-baseline
        # would write them down as the baseline. The per-probe None is the
        # defence for library callers; this is the one gate for the CLI.
        print(f"vault has 0 notes: {p} (a zero-note run is an invocation error)")
        raise SystemExit(2)
    return p


def vault_digest(vault: Path) -> tuple[str, int]:
    """sha256 over sorted ``relpath\\0sha256(content)`` lines → ("sha256:<hex>", count).

    Plain hashlib — provenance.content_sha256 goes through the DRIVER, not reused.
    """
    notes = iter_notes(vault)
    h = hashlib.sha256()
    for p in notes:
        rel = p.relative_to(vault).as_posix()
        content_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(f"{rel}\0{content_hash}\n".encode("utf-8"))
    return f"sha256:{h.hexdigest()}", len(notes)


def config_snapshot(store) -> dict:
    from silica.config import CONFIG

    return {
        "tau_high": 0.55,
        "tau_low": 0.15,
        "sim_threshold_high": getattr(CONFIG, "sim_threshold_high", None),
        "sim_threshold_low": getattr(CONFIG, "sim_threshold_low", None),
        "embedding_model": getattr(CONFIG, "embedding_model", None),
        "cooccur_store": "present" if len(store) else "absent",
        "cooccur_lang": getattr(store, "lang", None),
        "taxonomy_derivation": probe_classify.DERIVATION,
        "taxonomy_top_n": probe_classify.TOP_N,
        "relatedness_legs": None,  # filled by collect() from the fusion probe
    }


def _open_stores(vault: Path):
    """Point CONFIG/DRIVER at ``vault`` and open its on-disk indexes.

    Returns (cooccur_store, embed_store|None). The embed leg is offline for
    indexed notes (stored-vector lookup + cosine, no API); the explicit
    exists() check bypasses EmbedStore's legacy-path fallback so a vault
    without an index can never silently measure a foreign one.
    """
    import silica.driver
    from silica.config import CONFIG
    from silica.kernel.recall.cooccurrence import CooccurStore, _index_path_for
    from silica.kernel.recall.embed import EmbedStore
    from silica.kernel.recall.paths import index_dir_for

    CONFIG.vault_path = str(vault)
    silica.driver._driver = None
    store = CooccurStore(path=_index_path_for(str(vault)))
    embed_path = index_dir_for(str(vault)) / "embeddings.json"
    embed_store = EmbedStore(path=embed_path) if embed_path.exists() else None
    return store, embed_store


def collect(vault: Path, *, tier: str = "cheap", verbose: bool = False) -> dict:
    """Library entry point (the gate imports this). Runs the cheap-tier probes
    against ``vault`` and returns the full run document."""
    store, embed_store = _open_stores(vault)

    metrics: dict[str, float] = {}

    c = probe_classify.run(vault, store, verbose=verbose)
    metrics["classify.agreement"] = c["agreement"]
    metrics["classify.uncategorized_rate"] = c["uncategorized_rate"]
    metrics["classify.notes"] = c["notes"]

    lk = probe_links.run(vault, verbose=verbose)
    metrics["links.recall"] = lk["recall"]
    metrics["links.extra_per_note"] = lk["extra_per_note"]
    metrics["links.links_evaluated"] = lk["links_evaluated"]
    metrics["links.notes_evaluated"] = lk["notes_evaluated"]

    ig = integrity_probe(vault, verbose=verbose)
    metrics["integrity.rate"] = ig["rate"]
    metrics["integrity.notes"] = ig["notes"]
    metrics["integrity.vault_structural_violations"] = ig["vault_structural_violations"]
    metrics["integrity.vault_style_flags"] = ig["vault_style_flags"]
    metrics["integrity.vault_notes_with_structural"] = ig["vault_notes_with_structural"]

    # CORRELATE (ADR-0013): masked-pair recovery lift. Informational — the
    # fused-ranking regression is gated by fusion.recall_at_10 below. Both this
    # probe and fusion derive note_edges on the shared store in memory and are
    # self-contained (each recomputes), so their order does not matter; they run
    # after the leg probes only to keep the table grouping stable.
    cr = probe_correlate.run(vault, store, verbose=verbose)
    metrics["correlate.recall_expanded"] = cr["recall_expanded"]
    metrics["correlate.recall_union"] = cr["recall_union"]
    metrics["correlate.lift"] = cr["lift"]
    metrics["correlate.lift_pairs"] = cr["lift_pairs"]
    metrics["correlate.pairs_evaluated"] = cr["pairs_evaluated"]
    metrics["correlate.edges"] = cr["edges"]
    metrics["correlate.edges_wikilinked_frac"] = cr["edges_wikilinked_frac"]

    # FUSION: masked-pair recovery through the full relatedness facade — the
    # only end-to-end gate on RRF + leg wiring.
    fz = fusion_probe(vault, store, embed_store=embed_store, verbose=verbose)
    metrics["fusion.recall_at_10"] = fz["recall_at_10"]
    metrics["fusion.mrr"] = fz["mrr"]
    metrics["fusion.pairs_evaluated"] = fz["pairs_evaluated"]
    metrics["fusion.embed_coverage"] = fz["embed_coverage"]

    if tier in ("embedder", "all"):
        if embed_store is not None and len(embed_store):
            dd = probe_dedup.run(vault, store, embed_store=embed_store, verbose=verbose)
            metrics["dedup.fp_auto_merge_rate"] = dd["fp_auto_merge_rate"]
            metrics["dedup.fp_pairs_evaluated"] = dd["fp_pairs_evaluated"]
            metrics["dedup.fp_patches"] = dd["fp_patches"]
            metrics["dedup.route_defer"] = dd["route_defer"]
            metrics["dedup.route_keep"] = dd["route_keep"]
            metrics["dedup.tp_pairs_evaluated"] = dd["tp_pairs_evaluated"]
            if "tp_leak_rate" in dd:
                metrics["dedup.tp_leak_rate"] = dd["tp_leak_rate"]
                metrics["dedup.tp_leaks"] = dd["tp_leaks"]
        else:
            print("SKIP  dedup.*      — embed index absent (offline)")

    # SUPERSEDE (spec-contested-bitemporal §8.2): does a resolution ever hand the
    # merge to the weaker claim, and how many open contests would a tier rule
    # settle. Reuses the already-open stores; the wrong arm rides the embed leg
    # for its pairs and its keys are simply absent without it.
    sp = probe_supersede.run(vault, embed_store=embed_store, cooccur_store=store,
                             verbose=verbose)
    for key, val in sp.items():
        metrics[f"supersede.{key}"] = val

    # E(vault): lattice energy as one informational scalar + its six signed
    # terms (docs IV.1). Same report depth every run — analytics (cohesion,
    # gaps) + cooccurrence (integration deficits) — so E is comparable across
    # runs; the frozen baseline gives ΔE per term for free in the table. Never
    # gated: informational first (guardrail IV.5.1 — E is a thermometer, not an
    # objective to descend). Reuses the already-open cooccur store.
    # ponytail: on a byte-identical vault, energy.{orphans,dangling,deficits,
    # contested} are exactly stable but energy.{gaps,cohesion} carry ~0.5%
    # Louvain-order noise (seed is fixed; input edge order is not). Fine while
    # informational; to gate ΔE, gate those two with a tolerance band, not exact.
    # analytics=True runs PageRank/betweenness; fine forever on this vault,
    # which is a frozen fixture by design.
    from silica.kernel.report.graph_report import compute_report
    from silica.kernel.report.vault_energy import vault_energy

    e = vault_energy(compute_report(
        analytics=True, with_cooccurrence=True, _cooccur_store_override=store,
    ))
    for term, val in vars(e).items():
        metrics[f"energy.{term}"] = round(val, 4)

    # arbitrary single trend number — labeled as such, never gated
    metrics["coherence_index"] = round(sum(metrics[k] for k in _PRIMARIES) / len(_PRIMARIES), 4)

    cfg = config_snapshot(store)
    cfg["relatedness_legs"] = fz["legs"]

    digest, notes = vault_digest(vault)
    return {
        "generated_at": _today(),
        "tier": tier,
        "vault": {"digest": digest, "notes": notes, "path": str(vault)},
        "config": cfg,
        "metrics": metrics,
    }


def compare(baseline: dict, doc: dict) -> list[str]:
    """The single gate rule (pytest imports this). Digest/config refusals happen
    BEFORE this, not here."""
    b, d = baseline["metrics"], doc["metrics"]
    embed_live = _embed_live(doc["config"].get("relatedness_legs"))
    fails: list[str] = []
    for key in GATED_DROP_2PP:
        if key in _EMBED_DEPENDENT and not embed_live:
            continue  # embed index absent this run — SKIPped, not gated
        if key in b and key in d and d[key] < b[key] - 0.02:
            fails.append(f"{key}: {d[key]:.3f} < baseline {b[key]:.3f} − 2pp")
    for key in GATED_RISE_2PP:
        if key in _EMBED_DEPENDENT and not embed_live:
            continue  # embed index absent this run — SKIPped, not gated
        if key in b and key in d and d[key] > b[key] + 0.02:
            fails.append(f"{key}: {d[key]:.3f} > baseline {b[key]:.3f} + 2pp")
    for key in GATED_EXACT_ONE:
        if key in d and d[key] is None:
            # The probe saw zero notes: not a 1.0 and not a 0.0, an invocation
            # error (wrong vault, everything ignored) that must not pass the gate.
            fails.append(f"{key}: not measured, zero notes scanned (never a pass)")
        elif key in d and d[key] != 1.0:
            fails.append(f"{key}: {d[key]:.3f} != 1.0 (any new violation fails)")
    for key in GATED_EXACT_ZERO:
        if key in d and d[key] != 0:
            fails.append(f"{key}: {d[key]:.0f} != 0 (any inversion fails)")
    return fails


def print_table(doc: dict, baseline: dict | None) -> None:
    v = doc["vault"]
    cfg = doc["config"]
    print(f"\nvault: {v['path']}  ({v['notes']} notes)")
    print(f"digest: {v['digest']}  tier: {doc['tier']}  "
          f"cooccur: {cfg['cooccur_store']}/{cfg['cooccur_lang']}  "
          f"embedder: {cfg['embedding_model']}  "
          f"legs: {cfg.get('relatedness_legs') or '—'}")
    base_m = baseline["metrics"] if baseline else {}
    # Rise-gated dedup rates ARE gated by compare(); mark them so the table
    # matches the gate (they were silently unmarked before).
    gated = (set(GATED_DROP_2PP) | set(GATED_RISE_2PP)
             | set(GATED_EXACT_ONE) | set(GATED_EXACT_ZERO))
    print(f"\n{'metric':<38} {'value':>10} {'baseline':>10} {'delta':>9}  gate")
    for key in sorted(doc["metrics"]):
        val = doc["metrics"][key]
        base = base_m.get(key)
        measured = isinstance(val, (int, float))  # None: the probe saw zero notes
        delta = f"{val - base:+.3f}" if measured and isinstance(base, (int, float)) else "—"
        base_s = f"{base:.3f}" if isinstance(base, (int, float)) else "—"
        val_s = f"{val:.3f}" if measured else "—"
        mark = "GATE" if key in gated else ""
        print(f"{key:<38} {val_s:>10} {base_s:>10} {delta:>9}  {mark}")


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals.golden")
    ap.add_argument("--vault")
    ap.add_argument("--tier", choices=["cheap", "embedder", "all"], default="cheap")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--freeze-baseline", action="store_true")
    ap.add_argument("--rerank-ab", action="store_true",
                    help="A/B the configured cross-encoder over the fused ranking "
                         "(informational — HTTP provider, never gated, no baseline)")
    args = ap.parse_args(argv)

    vault = resolve_vault(args.vault)

    if args.rerank_ab:
        import silica.driver
        from silica.agent.providers import get_reranker
        from silica.config import CONFIG

        store, embed_store = _open_stores(vault)
        reranker = get_reranker(CONFIG)
        if reranker is None:
            print("no reranker configured (rerank_base_url/rerank_model) — nothing to A/B")
            return 2
        try:
            res = probe_fusion.run_rerank_ab(
                vault, store, embed_store=embed_store, reranker=reranker, verbose=args.verbose
            )
        finally:
            silica.driver._driver = None
        print(f"\nrerank A/B — informational, never gated "
              f"({res['pairs_evaluated']} pairs, {res['endpoints']} endpoints, "
              f"empty docs {res['empty_docs']})")
        print(f"{'arm':<22} {'recall@10':>10} {'mrr':>8}")
        print(f"{'fused (gated)':<22} {res['base_recall']:>10.3f} {res['base_mrr']:>8.3f}")
        print(f"{'reranked':<22} {res['rerank_recall']:>10.3f} {res['rerank_mrr']:>8.3f}")
        print(f"{'delta':<22} {res['rerank_recall'] - res['base_recall']:>+10.3f} "
              f"{res['rerank_mrr'] - res['base_mrr']:>+8.3f}   "
              f"pairs won +{res['pairs_won']} / lost -{res['pairs_lost']}")
        return 0
    try:
        doc = collect(vault, tier=args.tier, verbose=args.verbose)
    finally:
        import silica.driver
        silica.driver._driver = None

    _write_json(METRICS_PATH, doc)  # always

    if args.freeze_baseline:
        frozen = {**doc, "frozen_at": _today()}
        _write_json(BASELINE_PATH, frozen)
        print_table(doc, None)
        print(f"\nbaseline frozen → {BASELINE_PATH}")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else None
    if baseline is None:
        print_table(doc, None)
        print("\nno baseline yet — freeze one with --freeze-baseline")
        return 0

    if baseline["vault"]["digest"] != doc["vault"]["digest"]:
        print_table(doc, baseline)
        print("\nvault drifted — re-baseline deliberately with --freeze-baseline")
        return 1
    if baseline["config"]["cooccur_store"] != doc["config"]["cooccur_store"]:
        print_table(doc, baseline)
        print("\ncooccur mode changed — re-baseline deliberately with --freeze-baseline")
        return 1
    # A change in the NON-embed legs (cooccur/edges) is a different instrument —
    # refuse like digest/cooccur drift. But losing only the embed leg means the
    # index file is absent (deleted to reclaim RAM); that is an offline run, not
    # drift — SKIP the embed-dependent metrics and still gate the cheap tier.
    base_legs = baseline["config"].get("relatedness_legs")
    cur_legs = doc["config"].get("relatedness_legs")
    if _nonembed_legs(base_legs) != _nonembed_legs(cur_legs):
        print_table(doc, baseline)
        print("\nrelatedness legs changed — re-baseline deliberately with --freeze-baseline")
        return 1

    # Embedder-id drift: dedup/fusion cosines are not comparable across models.
    # Only refuse when the embed leg is actually live this run (else the id is moot).
    if _embed_live(cur_legs) and \
            baseline["config"].get("embedding_model") != doc["config"].get("embedding_model"):
        print_table(doc, baseline)
        print("\nembedder model changed — re-baseline deliberately with --freeze-baseline")
        return 1

    print_table(doc, baseline)
    if _embed_live(base_legs) and not _embed_live(cur_legs):
        print("\nSKIP  embed tier — index absent (offline); "
              f"{', '.join(_EMBED_DEPENDENT)} not gated this run")
    fails = compare(baseline, doc)
    if fails:
        print("\nFAIL:")
        for f in fails:
            print(f"  {f}")
        return 1
    print("\nPASS")
    return 0
