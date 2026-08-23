# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Embedding-based PROPOSED signals: missing links and duplicate pairs.

Both functions degrade to [] when the embedding index is empty or unreadable.
They read stored vectors only and never construct an embedder, so a provider
that is down does not suppress the proposals: this module reaches nothing
under silica.agent, which is what lets the P2/DKB contract cover it without
an exemption.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from silica.kernel.report.graph_report.models import DuplicatePair, MisfiledNote, MissingLink, VaultReport
from silica.kernel.recall.paths import in_folder

logger = logging.getLogger(__name__)

# Temporal decay half-life: a note updated 30 days ago gets ~50% boost; at
# 90 days the boost is negligible.  The constant is chosen so the recency
# factor degrades smoothly without overwhelming the cosine signal.
_RECENCY_HALFLIFE_DAYS = 30.0


def _pair_key(d: DuplicatePair) -> tuple[float, str, str]:
    """Deterministic pair ordering — best score first, then path. Byte-stable output."""
    return (-d.score, d.source, d.target)


def _compute_missing_links(
    report: VaultReport,
    G_und: Any,
    *,
    tau: float = 0.82,
    k: int = 10,
) -> list[MissingLink]:
    """Propose missing links via embedding similarity (PROPOSED — not authoritative).

    Paper-inspired refinements (Marwitz et al. 2026):
      - common_neighbors: a structural boost from the 2-length path count, the
        paper's Baseline core feature — likelier links rank above structurally
        isolated but equally-similar pairs.
      - d_prev annotation: each result carries its shortest-path distance before
        prediction. Only direct neighbours (d<=1) are hard-gated; d=2 candidates
        (likely links) and d>=3 (novel, high creative value) both surface.
      - Temporal decay: recent note pairs receive a modest cosine boost based on
        EmbedStore timestamps, capturing the paper's velocity-of-growth signal.
    """
    try:
        from silica.kernel.recall.cooccurrence import cooccur_key
        from silica.kernel.recall.embed import get_store
        import networkx as nx

        store = get_store()
        if len(store) == 0:
            return []
    except Exception as exc:
        logger.debug("graph_report: embeddings unavailable (%s)", exc)
        return []

    now = time.time()
    god_paths = [n.id for n in report.god_nodes]
    results: list[MissingLink] = []
    seen: set[tuple[str, str]] = set()

    # Keyspace bridge, as in cooccur_delta: G_und node ids are graph paths WITH
    # '.md', embed-store keys are stripped. Every candidate coming back from the
    # store must be mapped BACK to a node id before it touches the graph — without
    # it `tgt not in G_und` was true for every candidate and this whole section
    # returned [] on any real vault (measured: 0 links, 12 once the keyspaces line
    # up). The tests missed it because their fixtures use bare ids like "A"/"B",
    # where the two keyspaces coincide.
    gid_by_key = {cooccur_key(n): n for n in G_und.nodes}

    for source in god_paths:
        skey = cooccur_key(source)
        vec = store.get_vec(skey)
        if vec is None:
            continue
        try:
            candidates = store.cosine_top_k(vec, k=k, exclude={skey})
        except Exception:
            continue

        for cand in candidates:
            tgt = gid_by_key.get(cand["path"])
            if tgt is None:
                continue                       # scored note is not a graph node
            score = cand.get("score", 0.0)
            if score < tau:
                break  # results are sorted desc
            if tgt not in G_und or source not in G_und:
                continue

            # --- #7 d_prev: annotate instead of hard-gating at d<=2 ----------
            try:
                d_prev = nx.shortest_path_length(G_und, source, tgt)
            except nx.NetworkXNoPath:
                d_prev = 0  # unreachable → highest novelty
            if d_prev <= 1:
                continue  # only skip direct neighbours

            # --- #2 common_neighbors: structural-likelihood boost ------------
            # Paper Baseline uses sum_i A^2_u,i (2-length path count) as a core
            # feature; more shared neighbours → likelier real link. Mapped to
            # [0, 1) for diminishing returns so it nudges the ranking without
            # overwhelming the cosine signal.
            cn = len(list(nx.common_neighbors(G_und, source, tgt)))
            structural = cn / (1.0 + cn)

            # --- #5 temporal decay: boost recent note pairs ------------------
            # Store keyspace, not node ids — the old code passed `source` with its
            # '.md' still on, so ts_src was always 0 and the pair's recency silently
            # collapsed to the target's alone.
            ts_src = store.get_ts(skey)
            ts_tgt = store.get_ts(cooccur_key(tgt))
            age_days = max(0.0, (now - max(ts_src, ts_tgt)) / 86400.0)
            recency = 2.0 ** (-age_days / _RECENCY_HALFLIFE_DAYS)  # [0, 1]
            adjusted = score * (1.0 + 0.3 * structural) * (1.0 + 0.1 * recency)

            key = (min(source, tgt), max(source, tgt))
            if key not in seen:
                seen.add(key)
                results.append(MissingLink(
                    source=source, target=tgt,
                    cosine=round(adjusted, 4),
                    d_prev=d_prev,
                ))

    results.sort(key=lambda m: (-m.cosine, m.source, m.target))
    return results[:k]


def _minhash_duplicate_pairs(report: VaultReport) -> list[DuplicatePair]:
    """Embedder-free near-duplicate pairs — the STABLE leg of the maintenance dedup.

    COLLISION already falls back to MinHash when the embedder is down
    (router/states/collision.py), so INGEST keeps protecting itself against
    duplicates. The maintenance path did not: `/dedup` and `/curate` read
    `duplicate_pairs`, which is pure cosine, so an embedder-less vault got an
    empty plan and reported itself coherent. Same signature scheme, same
    threshold as the collision leg.

    Pairs land in the BORDERLINE band only, never `confirmed`: an estimated
    Jaccard over character shingles is a surface-form signal on a different
    scale from cosine, so it must not be compared against the cosine taus. The
    curator feeds both bands to the ternary dedup judge, which reads the two
    bodies — so these still get judged, never mechanically merged.

    Band-LSH after one O(n) signing pass: only pairs sharing a signature band
    are verified (the all-pairs loop this replaced was interpreted O(n^2) and
    took /curate to minutes at a few thousand notes). The sweep is now
    probabilistic — pairs sitting exactly at the threshold have ~11% miss odds
    (see banded_duplicate_pairs) — which the judge-not-merge contract absorbs.
    """
    from silica.config import CONFIG
    from silica.driver import DRIVER
    from silica.kernel.write import frontmatter
    from silica.kernel.report.minhash_dedup import banded_duplicate_pairs, minhash_signature

    threshold = getattr(CONFIG, "minhash_dup_threshold", 0.6)
    sigs: dict[str, tuple[int, ...]] = {}
    for nid in report.pagerank_map:  # every real node, always populated
        if not in_folder(nid, report.scope):
            continue
        try:
            content = DRIVER.read_note(nid).content
        except Exception:
            continue
        if not content:
            continue
        _data, _raw, body = frontmatter.split(content)
        # Title + body, mirroring the collision leg's "name + excerpt" query: two
        # notes on the same concept usually differ in surface title, and body
        # alone loses that signal.
        name = nid.rsplit("/", 1)[-1].removesuffix(".md")
        sig = minhash_signature(f"{name}\n{body}")
        if sig:
            sigs[nid] = sig

    out = [DuplicatePair(source=a, target=b, score=round(score, 4))
           for a, b, score in banded_duplicate_pairs(sigs, threshold=threshold)]
    out.sort(key=_pair_key)
    return out


def _compute_duplicate_pairs(
    report: VaultReport,
) -> tuple[list[DuplicatePair], list[DuplicatePair]]:
    """Find cosine-close note pairs (PROPOSED — not authoritative).

    Returns ``(borderline, confirmed)``:
      - borderline (τ_low < score < τ_high): topically related but distinct —
        the "link, don't merge" band. This is what the old report mislabelled
        as "possible duplicates"; it scales with vault size and domain coherence.
      - confirmed (score ≥ τ_high): bodies cover the same topic → likely true
        duplicates and genuine merge candidates.

    One cosine pass over the vault feeds both bands; a pair lands in exactly one.
    """
    store = None
    try:
        from silica.kernel.recall.embed import get_store

        store = get_store()
        if len(store) == 0:
            store = None
    except Exception as exc:
        logger.debug("graph_report: embeddings unavailable for dedup (%s)", exc)

    if store is None:
        # Degrade to the embedder-free leg instead of reporting zero duplicates:
        # an empty list here reads as "the vault is clean" and silently disarms
        # /dedup and /curate. Mirrors the abstention contract in relatedness.
        try:
            return _minhash_duplicate_pairs(report), []
        except Exception as exc:
            logger.debug("graph_report: minhash dedup leg also unavailable (%s)", exc)
            return [], []

    from silica.config import CONFIG

    tau_high = getattr(CONFIG, "sim_threshold_high", 0.85)
    tau_low = getattr(CONFIG, "sim_threshold_low", 0.75)

    borderline: list[DuplicatePair] = []
    confirmed: list[DuplicatePair] = []
    seen: set[tuple[str, str]] = set()

    scope = [p for p in store.paths() if in_folder(p, report.scope)]
    # One blocked matmul for the whole scope instead of one matvec per note — the
    # loop below only ever wants each note's single nearest neighbour, but paid a
    # full pass over the index for each. Same top-1 (see cosine_top_k_batch).
    try:
        neighbours = store.cosine_top_k_batch(scope, k=1)
    except Exception as exc:
        # Same contract as the missing-store branch above: [] would read as "the
        # vault is clean" and silently disarm /dedup and /curate.
        logger.debug("graph_report: cosine dedup leg failed (%s)", exc)
        return _minhash_duplicate_pairs(report), []

    for p in scope:
        candidates = neighbours.get(p)
        if not candidates:
            continue

        cand = candidates[0]
        tgt = cand["path"]
        score = cand.get("score", 0.0)

        if score <= tau_low:
            continue
        key = (min(p, tgt), max(p, tgt))
        if key in seen:
            continue
        seen.add(key)
        pair = DuplicatePair(source=p, target=tgt, score=round(score, 4))
        (confirmed if score >= tau_high else borderline).append(pair)

    borderline.sort(key=_pair_key)
    confirmed.sort(key=_pair_key)
    return borderline, confirmed


def _compute_dissonance(
    report: VaultReport,
    nodes: list[dict],
    G_und: Any,
    *,
    knn_k: int = 6,
    k: int = 10,
    min_degree: int = 2,
    threshold: float = 0.75,
) -> tuple[dict[str, float], list[MisfiledNote]]:
    """Partition dissonance (V5, ADR-0023): linked like one area, reads like another.

    The semantic partition is Louvain over the embedding k-NN, recomputed here
    without touching the persisted snapshot: the report needs membership only,
    and the snapshot's id inheritance exists for the viewer's colours.
    `dissonance_map` covers every zoned note with a zoned neighbour; `misfiled`
    keeps the notes with at least `min_degree` zoned neighbours of which
    `threshold` or more disagree, so a single odd link cannot flag a note.
    Degrades to ({}, []) without an embed index.
    """
    try:
        from silica.kernel.recall.graph_export import _louvain_partition, knn_edges
        from silica.kernel.recall.signals import dissonance
    except Exception as exc:  # pragma: no cover - import guard
        logger.debug("graph_report: dissonance unavailable (%s)", exc)
        return {}, []
    try:
        sim = knn_edges(nodes, k=knn_k)
        if not sim:
            return {}, []
        partition = _louvain_partition(nodes, sim, "SIMILAR")
    except Exception as exc:
        logger.debug("graph_report: semantic partition skipped (%s)", exc)
        return {}, []
    zone_of = {nid: i for i, comm in enumerate(partition) for nid in comm}
    dmap = {n: round(v, 4) for n, v in dissonance(G_und, zone_of).items()}
    rows: list[MisfiledNote] = []
    for nid, d in dmap.items():
        zoned = sum(1 for m in G_und.neighbors(nid) if m != nid and m in zone_of)
        if zoned >= min_degree and d >= threshold:
            rows.append(MisfiledNote(path=nid, degree=int(G_und.degree(nid)), dissonance=d))
    rows.sort(key=lambda m: (-m.dissonance, -m.degree, m.path))
    return dmap, rows[:k]
