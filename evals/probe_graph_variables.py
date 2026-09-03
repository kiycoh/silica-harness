# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Gates for the seven inter-note variables (spec 2026-08-22-graph-variables-design).

One subcommand per variable, all on a real vault's on-disk indexes, no LLM
unless --judge is passed. Verdicts are PASS / FAIL / INFORMATIONAL and every
number that feeds a verdict is printed next to it.

  uv run python -m evals.probe_graph_variables --vault ~/Documents/Obsidian/test --which all
  uv run python -m evals.probe_graph_variables --vault ... --which prereq --judge

structural  V1  AUC of Adamic-Adar vs common-neighbours vs random on 10%
                masked wikilinks, then the fused leg: recall@10 of the masked
                pairs, arm A (production legs) vs arm C (+structural), McNemar.
coupling    V3  masked-pair fusion harness (the golden eligible pairs), arm A
                vs arm C (+coupling leg), McNemar.
loadbearing V4  deterministic: refine candidates that are cut vertices and the
                notes that would be cut off if their links went.
dissonance  V5  AUC of per-note dissonance against "endpoint of a stale link"
                (an independent text signal); --judge adds precision of the
                misfiled list against random linked notes.
prereq      V2  --judge only: blind direction agreement on the strongest edges.
burst       V6  informational: the window and the top concepts.
sprawling   V7  --judge only: precision of split candidates vs random notes.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

K = 10
MASK_FRACTION = 0.10
SEED = 42
GATE_DELTA = 0.02        # primary: C - A >= +2pp on recall@10 (phase-2 convention)
MRR_TOLERANCE = 0.01
JUDGE_N = 40
JUDGE_AGREEMENT = 0.65   # direction agreement a coin cannot explain at n=40


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------

def _stores(vault: Path):
    from evals.golden.runner import _open_stores

    store, es = _open_stores(vault)
    es = es if (es is not None and len(es)) else None
    return store, es


def _score(topk: dict[str, list[str]], pairs: list[tuple[str, str]]) -> dict:
    hits, rr = 0, 0.0
    per_pair: dict[tuple[str, str], bool] = {}
    for a, b in pairs:
        ranks = []
        if b in topk.get(a, []):
            ranks.append(topk[a].index(b) + 1)
        if a in topk.get(b, []):
            ranks.append(topk[b].index(a) + 1)
        per_pair[(a, b)] = bool(ranks)
        if ranks:
            hits += 1
            rr += 1.0 / min(ranks)
    n = max(1, len(pairs))
    return {"recall_at_10": round(hits / n, 4), "mrr": round(rr / n, 4),
            "hits": hits, "pairs": len(pairs), "per_pair": per_pair}


def _best_rank(topk: dict[str, list[str]], a: str, b: str) -> int | None:
    ranks = []
    if b in topk.get(a, []):
        ranks.append(topk[a].index(b) + 1)
    if a in topk.get(b, []):
        ranks.append(topk[b].index(a) + 1)
    return min(ranks) if ranks else None


def _ordering(topk_a: dict, topk_c: dict, pairs: list[tuple[str, str]]) -> dict:
    """Per-pair rank wins/losses between the arms (reranker A/B convention:
    recall@k can tie while the ORDER moves; a sign test on the discordant
    pairs says whether the order moved one way)."""
    from scipy.stats import binomtest

    wins = losses = 0
    for a, b in pairs:
        ra, rc = _best_rank(topk_a, a, b), _best_rank(topk_c, a, b)
        if ra == rc:
            continue
        if rc is not None and (ra is None or rc < ra):
            wins += 1
        else:
            losses += 1
    n = wins + losses
    p = binomtest(wins, n, 0.5).pvalue if n else 1.0
    return {"rank_wins": wins, "rank_losses": losses, "sign_p": round(p, 4),
            "ordering_significant_05": bool(n and p < 0.05 and wins > losses)}


def _gate(arm_a: dict, arm_c: dict) -> dict:
    from evals.paired_stats import paired

    def doc(arm):
        return {"questions": [{"question_id": f"{a}|{b}", "correct": ok}
                              for (a, b), ok in arm["per_pair"].items()]}

    st = paired(doc(arm_c), doc(arm_a))
    d_recall = round(arm_c["recall_at_10"] - arm_a["recall_at_10"], 4)
    d_mrr = round(arm_c["mrr"] - arm_a["mrr"], 4)
    primary = d_recall >= GATE_DELTA
    passed = primary and d_mrr >= -MRR_TOLERANCE and st["significant_05"]
    if passed:
        verdict = "PASS"
    elif not primary:
        verdict = "FAIL on the primary (recall@10 delta under +2pp)"
    elif not st["significant_05"]:
        verdict = "FAIL on significance (delta inside noise)"
    else:
        verdict = "FAIL on mrr non-regression"
    return {"delta_recall": d_recall, "delta_mrr": d_mrr, "mcnemar_p": st["mcnemar_p"],
            "discordant": st["discordant"], "delta_ci95": st["delta_ci95"], "verdict": verdict}


def _auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC: P(score(pos) > score(neg)) with ties at 1/2."""
    if not pos or not neg:
        return 0.0
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return round(wins / (len(pos) * len(neg)), 4)


def _judge(prompt: str, *, max_tokens: int = 64) -> str:
    from evals.oracle import cached_text
    from silica.config import CONFIG

    return cached_text(CONFIG.model, [{"role": "user", "content": prompt}],
                       max_tokens=max_tokens, temperature=0.0).strip().lower()


def _body(vault: Path, key: str, limit: int = 700) -> str:
    from silica.kernel.write import frontmatter

    p = vault / (key + ".md")
    try:
        _d, _r, body = frontmatter.split(p.read_text(encoding="utf-8"))
    except OSError:
        return ""
    return " ".join(body.split())[:limit]


def _title(key: str) -> str:
    return key.rsplit("/", 1)[-1]


def _structural_ranking(graph, query_path: str, *, k: int, exclude: set[str]):
    """V1 leg: Adamic-Adar neighbours of `query_path` over the wikilink graph.

    Lived in relatedness.py while the gate ran; moved here with the gate's
    verdict (ADR-0027, ADR-0029): no production caller ever passed a graph. The
    graph carries driver node ids (with `.md`), the facade speaks store keys;
    bridged both ways so the leg joins the product legs' keyspace.
    """
    if graph is None:
        return None
    from silica.kernel.recall.cooccurrence import cooccur_key
    from silica.kernel.recall.signals import adamic_adar_ranking

    key_of = {cooccur_key(n): n for n in graph.nodes}
    gid = key_of.get(cooccur_key(query_path))
    if gid is None:
        return None
    blocked_ids = {key_of[x] for x in exclude if x in key_of}
    ranking = adamic_adar_ranking(graph, gid, k=k, exclude=blocked_ids)
    if not ranking:
        return None
    return [(cooccur_key(n), s) for n, s in ranking]


def fused_with(key: str, es, store, *, k: int = K, graph=None, mode: str = "leg",
               coupling=None) -> list[str]:
    """The production fusion with one extra leg mounted, as the probe's arm C.

    Rebinds nothing: it calls the same `_embed_ranking` / `_cooccur_ranking`
    the facade calls and fuses them with the facade's own `_rrf_fuse`, so with
    no extra leg it reproduces `related_notes` path-for-path (asserted by
    `check_reproduces_facade`, the alpha=1.0 trick of the PPR probe). `mode`
    "leg" adds the structural ranking as a fourth list; "boost" multiplies the
    fused score of candidates the other legs surfaced by aa/(1+aa) and never
    introduces one (the 2026-08-22 corroboration arm).
    """
    from silica.kernel.recall import relatedness as R
    from silica.kernel.recall.cooccurrence import cooccur_key
    from silica.kernel.recall.graph_export import is_vault_artifact

    blocked = {cooccur_key(key)}
    pool = max(k * 3, R._POOL_MIN)
    rankings = []
    e = R._embed_ranking(es, key, k=pool, exclude=blocked)
    if e is not None:
        rankings.append([(p, sc) for p, _n, sc in e])
    c = R._cooccur_ranking(store, key, k=pool, exclude=blocked, scope=None, expand=False)
    if c is not None:
        rankings.append(list(c))
    boost = None
    if graph is not None:
        ranked = _structural_ranking(graph, key, k=pool, exclude=blocked)
        if ranked and mode == "leg":
            rankings.append(ranked)
        elif ranked:
            boost = dict(ranked)
    if coupling:
        rankings.append([(p, w) for p, w in coupling if p not in blocked][:pool])
    fused = R._rrf_fuse(rankings)
    if boost:
        for p in list(fused):
            aa = boost.get(p)
            if aa:
                fused[p] *= 1.0 + aa / (1.0 + aa)
    ranked_paths = [p for p, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
                    if not is_vault_artifact(p)]
    return ranked_paths[:k]


def check_reproduces_facade(keys, es, store, *, k: int = K) -> None:
    """Validity check: no extra leg == the product facade, path for path."""
    from silica.kernel.recall.relatedness import related_notes

    for key in keys:
        mine = fused_with(key, es, store, k=k)
        theirs = [r.path for r in related_notes(key, embed_store=es, cooccur_store=store, k=k)]
        assert mine == theirs, f"probe fusion drifted from the facade on {key!r}"



# ---------------------------------------------------------------------------
# V1 structural
# ---------------------------------------------------------------------------

def run_structural(vault: Path) -> dict:
    import networkx as nx

    from silica.kernel.link.health import wikilink_graph
    from silica.kernel.recall.relatedness import related_notes
    from silica.kernel.recall.signals import adamic_adar_ranking

    store, es = _stores(vault)
    adj = wikilink_graph(vault, store)
    G = nx.Graph()
    G.add_nodes_from(sorted(adj))
    G.add_edges_from(sorted((a, b) for a, nb in adj.items() for b in nb if a < b))
    edges = sorted(G.edges())
    rng = random.Random(SEED)
    masked = rng.sample(edges, max(1, int(len(edges) * MASK_FRACTION)))
    G_train = G.copy()
    G_train.remove_edges_from(masked)

    # (a) link prediction: masked edges vs as many random non-edges
    nodes = sorted(G.nodes)
    negatives: set[tuple[str, str]] = set()
    while len(negatives) < len(masked):
        a, b = rng.choice(nodes), rng.choice(nodes)
        if a != b and not G.has_edge(a, b):
            negatives.add((min(a, b), max(a, b)))

    def aa(u, v):
        return sum(1.0 / __import__("math").log(G_train.degree(z))
                   for z in nx.common_neighbors(G_train, u, v) if G_train.degree(z) > 1)

    def cn(u, v):
        return float(len(list(nx.common_neighbors(G_train, u, v))))

    pos_aa = [aa(u, v) for u, v in masked]
    neg_aa = [aa(u, v) for u, v in negatives]
    pos_cn = [cn(u, v) for u, v in masked]
    neg_cn = [cn(u, v) for u, v in negatives]
    reach = sum(1 for s in pos_aa if s > 0)
    auc = {"adamic_adar": _auc(pos_aa, neg_aa), "common_neighbours": _auc(pos_cn, neg_cn),
           "random": _auc([rng.random() for _ in masked], [rng.random() for _ in negatives]),
           "masked": len(masked), "masked_with_common_neighbour": reach}

    # (b) the fused leg on the masked pairs, both directions, k=10
    endpoints = sorted({e for pr in masked for e in pr})
    legs = ("embed+" if es is not None else "") + "cooccur"

    def arms(keys, Gx):
        check_reproduces_facade(keys[:20], es, store)
        a = {key: [r.path for r in related_notes(key, embed_store=es, cooccur_store=store, k=K)]
             for key in keys}
        out = {}
        for mode in ("leg", "boost"):
            out[mode] = {key: fused_with(key, es, store, graph=Gx, mode=mode) for key in keys}
        return a, out

    topk_a, variants = arms(endpoints, G_train)
    arm_a = _score(topk_a, masked)
    gates = {}
    for mode, topk_c in variants.items():
        arm_c = _score(topk_c, masked)
        g = _gate(arm_a, arm_c)
        g["ordering"] = _ordering(topk_a, topk_c, masked)
        g["arm_c"] = {k: v for k, v in arm_c.items() if k != "per_pair"}
        gates[mode] = g
    # how often the leg alone would have ranked the counterpart in its top-10
    alone = 0
    for a, b in masked:
        ra = adamic_adar_ranking(G_train, a, k=K) or []
        rb = adamic_adar_ranking(G_train, b, k=K) or []
        if b in dict(ra) or a in dict(rb):
            alone += 1
    # Regression read on the golden metric: the eligible pairs stay >2 hops
    # apart once masked, so the leg cannot recover them by construction and
    # can only dilute. A drop here would fail the golden gate after wiring.
    from silica.kernel.link.health import eligible_pairs

    eligible = eligible_pairs(adj)
    ends = sorted({e for pr in eligible for e in pr})
    reg_a, reg_variants = arms(ends, G)
    base = _score(reg_a, eligible)
    regression = {}
    for mode, reg_c in reg_variants.items():
        r = _gate(base, _score(reg_c, eligible))
        r["ordering"] = _ordering(reg_a, reg_c, eligible)
        r["pairs"] = len(eligible)
        r["verdict"] = ("no regression" if r["delta_recall"] > -GATE_DELTA
                        else "REGRESSION on the golden eligible pairs")
        regression[mode] = r
    verdict = {}
    for mode in gates:
        ok_gain = gates[mode]["verdict"] == "PASS" or gates[mode]["ordering"]["ordering_significant_05"]
        ok_reg = regression[mode]["verdict"] == "no regression"
        verdict[mode] = ("PASS" if ok_gain and ok_reg else
                         "FAIL (regresses the golden pairs)" if ok_gain else
                         "FAIL (no gain)" if ok_reg else "FAIL (no gain, regresses)")
    return {"variable": "structural", "legs": legs, "auc": auc,
            "leg_alone_recall_at_10": round(alone / len(masked), 4),
            "arm_a": {k: v for k, v in arm_a.items() if k != "per_pair"},
            "masked_gate": gates, "golden_regression": regression, "verdict": verdict}


# ---------------------------------------------------------------------------
# V3 coupling
# ---------------------------------------------------------------------------

def run_coupling(vault: Path) -> dict:
    from silica.kernel.link.health import eligible_pairs, wikilink_graph
    from silica.kernel.recall.relatedness import related_notes
    from silica.kernel.report.graph_report import compute_report

    store, es = _stores(vault)
    report = compute_report(analytics=True)
    cmap = report.coupling_map
    eligible = eligible_pairs(wikilink_graph(vault, store))
    endpoints = sorted({e for pr in eligible for e in pr})
    covered = sum(1 for e in endpoints if cmap.get(e))
    if not eligible:
        return {"variable": "coupling", "verdict": "no eligible pairs"}
    topk_a = {key: [r.path for r in related_notes(key, embed_store=es, cooccur_store=store, k=K)]
              for key in endpoints}
    check_reproduces_facade(endpoints[:20], es, store)
    topk_c = {}
    for key in endpoints:
        row = cmap.get(key) or {}
        rank = sorted(row.items(), key=lambda kv: (-kv[1], kv[0])) or None
        topk_c[key] = fused_with(key, es, store, coupling=rank)
    arm_a = _score(topk_a, eligible)
    arm_c = _score(topk_c, eligible)
    legs = ("embed+" if es is not None else "") + "cooccur"
    gate = _gate(arm_a, arm_c)
    gate["ordering"] = _ordering(topk_a, topk_c, eligible)
    alone = sum(1 for a, b in eligible if b in (cmap.get(a) or {}) or a in (cmap.get(b) or {}))
    return {"variable": "coupling", "legs": legs,
            "coupled_notes": len(cmap), "endpoints": len(endpoints),
            "endpoints_with_coupling": covered,
            "leg_alone_recall": round(alone / len(eligible), 4),
            "arm_a": {k: v for k, v in arm_a.items() if k != "per_pair"},
            "arm_c": {k: v for k, v in arm_c.items() if k != "per_pair"},
            "gate": gate}


# ---------------------------------------------------------------------------
# V4 load-bearing
# ---------------------------------------------------------------------------

def run_loadbearing(vault: Path) -> dict:
    import networkx as nx

    from silica.kernel.recall.graph_export import build_graph_data, edge_graph
    from silica.kernel.report.graph_report import compute_report

    _stores(vault)
    report = compute_report(analytics=True)
    nodes, edges = build_graph_data()
    G = edge_graph(nodes, edges)
    base = nx.number_connected_components(G)
    refine = set(report.reformat_notes) | set(report.lean_notes)
    cut = set(report.articulation)
    vetoed = sorted(refine & cut)
    severed = []
    for n in vetoed:
        H = G.copy()
        H.remove_edges_from(list(G.edges(n)))
        comps = nx.number_connected_components(H) - 1   # minus the now-isolated n itself
        # notes no longer reachable from the largest remaining component
        giant = max(nx.connected_components(H), key=len)
        lost = sum(1 for m in G.nodes if m != n and m not in giant and nx.has_path(G, n, m))
        severed.append({"note": n, "new_components": comps - base, "notes_cut_off": lost})
    severed.sort(key=lambda d: (-d["notes_cut_off"], d["note"]))
    protected = sum(d["notes_cut_off"] for d in severed)
    return {"variable": "loadbearing", "notes": report.totals["notes"],
            "articulation_points": len(cut), "refine_candidates": len(refine),
            "vetoed": len(vetoed), "notes_protected": protected,
            "top_vetoed": severed[:5],
            "top_surprise": [{"note": lb.path, "degree": lb.degree, "betweenness": lb.betweenness,
                              "surprise": lb.surprise, "cut": lb.articulation}
                             for lb in report.load_bearing[:5]],
            "verdict": ("PASS (veto protects %d notes behind %d refine candidates)" % (protected, len(vetoed))
                        if vetoed else "INFORMATIONAL (no refine candidate is a cut vertex)")}


# ---------------------------------------------------------------------------
# V5 dissonance
# ---------------------------------------------------------------------------

def run_dissonance(vault: Path, *, judge: bool) -> dict:
    from silica.kernel.report.graph_report import compute_report

    _stores(vault)
    report = compute_report(analytics=True, with_embeddings=True, with_cooccurrence=True, top_k=20)
    dmap = report.dissonance_map
    if not dmap:
        return {"variable": "dissonance", "verdict": "no embed index"}
    stale_endpoints = {s.source for s in report.stale_links} | {s.target for s in report.stale_links}
    pos = [d for n, d in dmap.items() if n in stale_endpoints]
    neg = [d for n, d in dmap.items() if n not in stale_endpoints]
    out = {"variable": "dissonance", "notes_with_dissonance": len(dmap),
           "stale_endpoints_with_dissonance": len(pos),
           "auc_stale": _auc(pos, neg), "misfiled": len(report.misfiled)}
    out["verdict"] = ("PASS (dissonance separates stale-link endpoints, AUC %.2f)" % out["auc_stale"]
                      if out["auc_stale"] >= 0.6 and len(pos) >= 10 else
                      "FAIL/INFORMATIONAL (AUC %.2f on %d stale endpoints)" % (out["auc_stale"], len(pos)))
    if judge and report.misfiled:
        rng = random.Random(SEED)
        linked = [n for n in dmap if n not in {m.path for m in report.misfiled}]
        sample = rng.sample(linked, min(20, len(linked)))
        from silica.kernel.recall.graph_export import build_graph_data, edge_graph
        G = edge_graph(*build_graph_data())

        def ask(n: str) -> bool | None:
            key = n.removesuffix(".md")
            nbrs = [m.removesuffix(".md") for m in G.neighbors(n)][:6]
            prompt = ("A note in a personal knowledge vault links to these notes: "
                      + "; ".join(_title(m) for m in nbrs)
                      + f".\n\nNote title: {_title(key)}\nNote text: {_body(vault, key)}\n\n"
                      "Judging by its content, is this note filed with the right neighbours, "
                      "or does it belong to a different topic area? Answer exactly one word: "
                      "RIGHT or WRONG.")
            t = _judge(prompt)
            return None if not t else ("wrong" in t)

        flagged = [ask(m.path) for m in report.misfiled[:20]]
        control = [ask(n) for n in sample]
        f = [x for x in flagged if x is not None]
        c = [x for x in control if x is not None]
        out["judge"] = {"misfiled_wrong_rate": round(sum(f) / len(f), 3) if f else None,
                        "control_wrong_rate": round(sum(c) / len(c), 3) if c else None,
                        "n": [len(f), len(c)]}
    return out


# ---------------------------------------------------------------------------
# V2 prerequisites (judge)
# ---------------------------------------------------------------------------

def run_prereq(vault: Path, *, judge: bool) -> dict:
    from scipy.stats import binomtest

    from silica.kernel.report.graph_report import compute_report

    _stores(vault)
    report = compute_report(analytics=True, with_cooccurrence=True)
    edges = report.prerequisites[:JUDGE_N]
    out = {"variable": "prereq", "edges": len(report.prerequisites),
           "dependents": len(report.prereq_map),
           "top": [[e.prereq, e.dependent, e.refd] for e in edges[:10]]}
    if not judge:
        out["verdict"] = "INFORMATIONAL (pass --judge for the direction gate)"
        return out
    rng = random.Random(SEED)
    agree = disagree = neither = 0
    for e in edges:
        first, second = (e.prereq, e.dependent) if rng.random() < 0.5 else (e.dependent, e.prereq)
        prompt = ("Two notes from a study vault.\n\n"
                  f"A: {_title(first)}\n{_body(vault, first)}\n\n"
                  f"B: {_title(second)}\n{_body(vault, second)}\n\n"
                  "If a learner had to read one BEFORE the other to understand it, which "
                  "comes first? Answer exactly one word: A, B or NEITHER.")
        t = _judge(prompt)
        pick = "a" if t.startswith("a") else "b" if t.startswith("b") else None
        if pick is None:
            neither += 1
            continue
        chosen = first if pick == "a" else second
        if chosen == e.prereq:
            agree += 1
        else:
            disagree += 1
    n = agree + disagree
    p = binomtest(agree, n, 0.5, alternative="greater").pvalue if n else 1.0
    rate = round(agree / n, 3) if n else None
    out.update({"judged": n, "agree": agree, "disagree": disagree, "neither": neither,
                "agreement": rate, "binomial_p": round(p, 4)})
    out["verdict"] = ("PASS (direction agreement %.2f, p=%.3f)" % (rate, p)
                      if n >= 20 and rate is not None and rate >= JUDGE_AGREEMENT and p < 0.05
                      else "FAIL (agreement %s on %d judged, p=%.3f)" % (rate, n, p))
    return out


# ---------------------------------------------------------------------------
# V6 burst, V7 sprawling
# ---------------------------------------------------------------------------

def run_burst(vault: Path) -> dict:
    from silica.kernel.report.graph_report import compute_report

    _stores(vault)
    report = compute_report(analytics=True, with_cooccurrence=True)
    return {"variable": "burst", "verdict": "INFORMATIONAL",
            "bursting": [[b.concept, b.z, b.recent, b.total] for b in report.bursting_concepts]}


def run_sprawling(vault: Path, *, judge: bool) -> dict:
    from silica.kernel.report.graph_report import compute_report

    store, _es = _stores(vault)
    report = compute_report(analytics=True, with_cooccurrence=True)
    out = {"variable": "sprawling",
           "rows": [[s.path, s.concepts, s.entropy, s.flatness] for s in report.sprawling]}
    if not judge or not report.sprawling:
        out["verdict"] = "INFORMATIONAL"
        return out
    rng = random.Random(SEED)
    flagged = {s.path for s in report.sprawling}
    pool = [p for p in store.paths() if p not in flagged and (vault / (p + ".md")).exists()]
    control = rng.sample(pool, min(10, len(pool)))

    def ask(key: str) -> bool | None:
        prompt = (f"Note title: {_title(key)}\nNote text (beginning): {_body(vault, key, 1500)}\n\n"
                  "Does this note cover several distinct topics that would be clearer as "
                  "separate notes? Answer exactly one word: SPLIT or KEEP.")
        t = _judge(prompt)
        return None if not t else ("split" in t)

    f = [x for x in (ask(p) for p in list(flagged)[:10]) if x is not None]
    c = [x for x in (ask(p) for p in control) if x is not None]
    out["judge"] = {"flagged_split_rate": round(sum(f) / len(f), 3) if f else None,
                    "control_split_rate": round(sum(c) / len(c), 3) if c else None,
                    "n": [len(f), len(c)]}
    fr, cr = out["judge"]["flagged_split_rate"], out["judge"]["control_split_rate"]
    out["verdict"] = ("PASS (split rate %.2f vs %.2f control)" % (fr, cr)
                      if fr is not None and cr is not None and fr >= cr + 0.3 else
                      "FAIL/INFORMATIONAL (%s vs %s)" % (fr, cr))
    return out


# ---------------------------------------------------------------------------

RUNNERS = {
    "structural": lambda v, j: run_structural(v),
    "coupling": lambda v, j: run_coupling(v),
    "loadbearing": lambda v, j: run_loadbearing(v),
    "dissonance": lambda v, j: run_dissonance(v, judge=j),
    "prereq": lambda v, j: run_prereq(v, judge=j),
    "burst": lambda v, j: run_burst(v),
    "sprawling": lambda v, j: run_sprawling(v, judge=j),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vault", required=True)
    ap.add_argument("--which", default="all", help="comma list of " + ",".join(RUNNERS) + " or all")
    ap.add_argument("--judge", action="store_true", help="run the LLM-judged gates (network)")
    ap.add_argument("--record", help="write the results JSON here")
    args = ap.parse_args(argv)
    vault = Path(args.vault).expanduser().resolve()
    which = list(RUNNERS) if args.which == "all" else [w.strip() for w in args.which.split(",")]
    results = []
    for w in which:
        res = RUNNERS[w](vault, args.judge)
        results.append(res)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.record:
        Path(args.record).write_text(json.dumps({"vault": str(vault), "results": results},
                                                indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
