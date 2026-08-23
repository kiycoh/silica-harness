# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""probe_dedup — distinctness routing on real embeddings (embedder tier).

Absorbs the retired ``tests/eval/coherence_harness.py``: it replays COLLISION's
real routing decision (``route_concept`` + ``_names_agree``) on the vault's own
notes and reports where a merge would go. No replica of pipeline logic — the
harness calls the same functions, so it can never drift from what it measures.

FP arm (gated on RISE):
    Each non-inbox note's cosine-best candidate (the store's own
    ``cosine_top_k``, inbox-filtered exactly as ``collision_pass`` does since
    ADR-0030; until then the facade's winner). Any pair the router auto-routes
    ``patch`` — a mechanical merge with no judge — is a FALSE POSITIVE: the
    human kept the two notes distinct. Zero-API: the query is the note's stored
    vector and hub status comes from the cached cluster context, the same
    offline signals COLLISION reads.

TP arm (report; gate self-arms once labels resolve):
    Analyst-labeled real duplicates (same concept, different surface) ported
    from coherence_harness. A pair routed ``keep`` (new note) is a LEAK. On a
    vault where the labels no longer resolve, ``tp_pairs_evaluated`` is 0 and
    the metric is omitted (nothing to gate) until the notes exist again.

Ceiling (accepted): the stored candidate vector carries its folder, so the
intra-domain cosine is an optimistic upper bound — conservative for a rise
gate. The faithful asymmetric embed (query side without folder) is the upgrade
path if the gate ever needs tightening.
"""
from __future__ import annotations

# Analyst-labeled real duplicates ported from coherence_harness (CORRECT ME via
# the retired harness's --dump matrix if the vault changes). Resolved against
# live store keys by basename; unresolved pairs are reported, never fatal.
TRUE_DUPS: list[tuple[str, str]] = [
    ("Neurone artificiale", "Neurone Artificiale (ANN)"),
    ("Neurone artificiale", "Neurone Artificiale 1"),
    ("Neurone Artificiale (ANN)", "Neurone Artificiale 1"),
    ("One-Class SVDD", "One-Class Support Vector Data Description"),
    ("One-Class Support Vector Data Description", "One-Class Support Vector Data Descriptor"),
    ("One-Class SVDD", "One-Class Support Vector Data Descriptor"),
    ("One-class classification", "Classificatori One-class"),
    ("Storia dell'AI", "Storia dell'intelligenza artificiale"),
    ("Maximum Likelihood estimator", "Stimatore massima verosimiglianza"),
    ("Interpretazione frequentista e bayesiana della probabilità", "Probabilità frequentista e bayesiana"),
    ("Variabile casuale", "Random variables"),
]

_EMPTY = {
    "fp_auto_merge_rate": 0.0,
    "fp_pairs_evaluated": 0,
    "fp_patches": 0,
    "route_patch": 0,
    "route_defer": 0,
    "route_keep": 0,
    "tp_pairs_evaluated": 0,
}


def _hub_keys(vault) -> set[str]:
    """Cached cluster hub set (``clusters_ctx.json``), keyed like store paths.

    Absent/cold cache ⇒ every note reads non-hub — bounded staleness, exactly
    what COLLISION sees on a note added since the last graph refresh.
    """
    import orjson

    from silica.kernel.recall.paths import index_dir_for

    p = index_dir_for(str(vault)) / "clusters_ctx.json"
    if not p.exists():
        return set()
    try:
        ctx = (orjson.loads(p.read_bytes()) or {}).get("ctx") or {}
    except Exception:
        return set()
    return {k for k, d in ctx.items() if isinstance(d, dict) and d.get("is_hub")}


def _cos(a, b) -> float:
    import numpy as np

    x, y = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
    return float(x @ y / ((x @ x) ** 0.5 * (y @ y) ** 0.5 + 1e-12))


def run(vault, store, *, embed_store=None, verbose: bool = False) -> dict:
    from silica.config import CONFIG
    from silica.kernel.link.health import iter_notes
    from silica.kernel.recall.paths import is_inbox_path
    from silica.router.states.collision import _names_agree, route_concept

    if embed_store is None or not len(embed_store):
        return dict(_EMPTY)  # embed index absent — the runner SKIPs, never gates

    tau_high = getattr(CONFIG, "sim_threshold_high", 0.85)
    tau_low = getattr(CONFIG, "sim_threshold_low", 0.65)
    hubs = _hub_keys(vault)

    keys = [
        p.relative_to(vault).with_suffix("").as_posix()
        for p in iter_notes(vault)
    ]
    keys = [k for k in keys if embed_store.get_vec(k) is not None and not is_inbox_path(k)]
    domain = {k: k.split("/")[0] for k in keys}

    # --- FP arm: cosine-best candidate per note, routed by the real functions ---
    counts = {"patch": 0, "defer": 0, "keep": 0}
    evaluated = 0
    fp_examples: list[tuple[float, str, str]] = []
    for k in keys:
        # k=5 as in collision_pass, plus the self row the exclude removes.
        hits = embed_store.cosine_top_k(embed_store.get_vec(k), k=6, exclude={k, k + ".md"})
        best = next(
            (h for h in hits
             if not is_inbox_path(h["path"])
             and domain.get(h["path"].removesuffix(".md")) == domain[k]),
            None,
        )
        if best is None:
            continue  # cold path: no same-domain candidate → COLLISION keeps
        evaluated += 1
        is_hub = best["path"].removesuffix(".md") in hubs
        d = route_concept(
            best["score"],
            names_agree=_names_agree(k.split("/")[-1], best["name"]),
            is_hub=is_hub, tau_high=tau_high, tau_low=tau_low,
        )
        counts[d] += 1
        if d == "patch":
            fp_examples.append((round(best["score"], 3), k.split("/")[-1], best["name"]))

    out = {
        "fp_auto_merge_rate": round(counts["patch"] / evaluated, 4) if evaluated else 0.0,
        "fp_pairs_evaluated": evaluated,
        "fp_patches": counts["patch"],
        "route_patch": counts["patch"],
        "route_defer": counts["defer"],
        "route_keep": counts["keep"],
    }

    # --- TP arm: labeled duplicates; a `keep` is a leak. Zero-API (stored vecs) ---
    by_stem: dict[str, str] = {}
    for k in store.paths():
        by_stem.setdefault(k.split("/")[-1], k)
    tp_eval = tp_leak = 0
    tp_examples: list[tuple[float, str, str, str]] = []
    for a, b in TRUE_DUPS:
        ka, kb = by_stem.get(a), by_stem.get(b)
        if not ka or not kb:
            continue
        va, vb = embed_store.get_vec(ka), embed_store.get_vec(kb)
        if va is None or vb is None:
            continue
        tp_eval += 1
        s = _cos(va, vb)
        d = route_concept(
            s, names_agree=_names_agree(a, b),
            is_hub=kb.removesuffix(".md") in hubs, tau_high=tau_high, tau_low=tau_low,
        )
        tp_leak += d == "keep"
        tp_examples.append((round(s, 3), d, a, b))
    out["tp_pairs_evaluated"] = tp_eval
    if tp_eval:  # gate self-arms only when labels resolve
        out["tp_leak_rate"] = round(tp_leak / tp_eval, 4)
        out["tp_leaks"] = tp_leak

    if verbose:
        print(f"\ndedup FP: {counts['patch']} auto-merge / {evaluated} eval "
              f"(defer {counts['defer']}, keep {counts['keep']})")
        for s, a, b in sorted(fp_examples, reverse=True)[:10]:
            print(f"   FP cos={s:.3f}  {a!r} <- {b!r}")
        print(f"dedup TP: {tp_leak} leak / {tp_eval} labeled dups resolved")
        for s, d, a, b in tp_examples:
            print(f"   {d:5} cos={s:.3f}  {a!r} | {b!r}")
    return out
