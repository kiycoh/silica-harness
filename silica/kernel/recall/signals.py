# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Seven inter-note variables, computed the way co-occurrence is: L1, no LLM,
no I/O, every input injected (an nx graph, a store-shaped dict, timestamps).

Design: docs/superpowers/specs/2026-08-22-graph-variables-design.md. Each
function is a candidate generator or a per-note scalar; none is authoritative
about vault structure. Ranking functions abstain with None (the fusion
contract in relatedness._fuse) when they cannot rank, never with [].

Keyspace: these functions do not know about '.md'. Callers hand in one
keyspace and get the same one back; the report and the facade bridge at
their own seams (cooccur_key), as every other signal does.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable


# ---------------------------------------------------------------------------
# V1  Structural link prediction: Adamic-Adar over the wikilink graph
# ---------------------------------------------------------------------------

def adamic_adar_ranking(
    G, node: str, *, k: int, exclude: Iterable[str] = (),
) -> list[tuple[str, float]] | None:
    """Non-adjacent notes two hops from `node`, by Adamic-Adar score.

    score(a, b) = sum over common neighbours z of 1 / log(deg z). Adamic-Adar
    rather than a raw common-neighbour count because a hub is a common
    neighbour of everything it links: the 1/log(deg) term is what keeps a MOC
    from making all its members predicted neighbours of each other. None when
    `node` is absent, isolated, or has no distance-2 candidate at all.
    """
    if node not in G or G.degree(node) == 0:
        return None
    blocked = set(exclude) | {node}
    scores: dict[str, float] = {}
    for z in G.neighbors(node):
        dz = G.degree(z)
        if dz < 2:
            continue  # cannot be a common neighbour of two distinct notes
        w = 1.0 / math.log(dz)
        for cand in G.neighbors(z):
            if cand in blocked or G.has_edge(node, cand):
                continue
            scores[cand] = scores.get(cand, 0.0) + w
    if not scores:
        return None
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def structural_links(G, *, top_k: int) -> list[tuple[str, str, float, list[str]]]:
    """Every unlinked pair at distance 2, scored by Adamic-Adar, best first.

    Returns (u, v, score, common) with u < v. O(sum of deg^2): each common
    neighbour enumerates its own neighbour pairs once, which is the same
    bound nx.adamic_adar_index pays, minus the nx ebunch construction.
    """
    scores: dict[tuple[str, str], float] = {}
    common: dict[tuple[str, str], list[str]] = {}
    for z in G.nodes:
        nbrs = sorted(G.neighbors(z))
        if len(nbrs) < 2:
            continue
        w = 1.0 / math.log(len(nbrs)) if len(nbrs) > 1 else 0.0
        for i, u in enumerate(nbrs):
            for v in nbrs[i + 1:]:
                if u == v or G.has_edge(u, v):
                    continue
                key = (u, v)
                scores[key] = scores.get(key, 0.0) + w
                common.setdefault(key, []).append(z)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [(u, v, s, sorted(common[(u, v)])) for (u, v), s in ranked]


# ---------------------------------------------------------------------------
# V2  Prerequisite direction: RefD (Liang, Wu, Huang & Giles, CIKM 2015)
# ---------------------------------------------------------------------------

def refd_edges(
    links: dict[str, set[str]],
    related: dict[str, list[tuple[str, float]]],
    *,
    theta: float = 0.1,
    min_related: int = 1,
) -> list[tuple[str, str, float]]:
    """Directed prerequisite edges (prereq, dependent, refd), strongest first.

    RefD(A -> B) = sum_{v in R(B)} w_B(v) l(v, A)
                 - sum_{v in R(A)} w_A(v) l(v, B)
    with w_X normalised to sum 1 over R(X) and l(v, A) = 1/log2(1 + outdeg v)
    when v links A, else 0. Notes related to B that cite A without the
    reverse holding is the asymmetry a reading order needs; the wikilink
    graph alone cannot give it (a link is written in one direction for
    editorial reasons as often as for conceptual ones).

    Two guards the paper's Wikipedia setting did not need. A related set
    smaller than `min_related` abstains (its side reads 0): measured
    2026-08-22 on a 709-note vault, a note whose only related note was an
    index page made that page's four out-links "prerequisites" at RefD 1.0.
    And a citing note is damped by its out-degree: a map-of-content that
    links everything testifies for nothing in particular.
    """
    weights: dict[str, dict[str, float]] = {}
    for x, rel in related.items():
        row = {v: w for v, w in rel if v != x and w > 0}
        total = sum(row.values())
        if len(row) < min_related or total <= 0:
            weights[x] = {}
            continue
        weights[x] = {v: w / total for v, w in row.items()}
    damp = {v: 1.0 / math.log2(1 + len(t)) for v, t in links.items() if t}

    # Candidate pairs are the citation pairs (X, t): t is linked by some note
    # related to X. RefD is zero for any other pair by construction, so
    # enumerating linked or related pairs on top would only add zeros.
    pairs: set[tuple[str, str]] = set()
    for x, row in weights.items():
        for v in row:
            for t in links.get(v, ()):
                if t != x and t != v:
                    pairs.add((x, t) if x < t else (t, x))

    def cites(a: str, b: str) -> float:
        # weighted share of B's related notes that link to A
        return sum(w * damp[v] for v, w in weights.get(b, {}).items() if a in links.get(v, ()))

    out: list[tuple[str, str, float]] = []
    for a, b in pairs:
        r = cites(a, b) - cites(b, a)
        if r >= theta:
            out.append((a, b, r))
        elif r <= -theta:
            out.append((b, a, -r))
    out.sort(key=lambda e: (-e[2], e[0], e[1]))
    return out


# ---------------------------------------------------------------------------
# V3  Coupling: notes that share transactions (sources cited, runs written)
# ---------------------------------------------------------------------------

def coupling_adjacency(
    transactions: Iterable[Iterable[str]], *, cap: int = 30, report_dropped: bool = False,
):
    """{note: {other: weight}} from the note sets of each transaction.

    weight(a, b) = sum over shared transactions t of 1 / log(1 + |t|): a
    source cited by two notes says more than one cited by thirty (the
    Adamic-Adar reading of bibliographic coupling). Transactions above `cap`
    are dropped and counted, never silently: a 456-note nucleate batch is an
    ingest, not evidence that its notes belong together (Zimmermann et al.
    2005 drop such transactions the same way).
    """
    adj: dict[str, dict[str, float]] = {}
    dropped = 0
    for t in transactions:
        members = sorted(set(t))
        if len(members) < 2:
            continue
        if len(members) > cap:
            dropped += 1
            continue
        w = 1.0 / math.log(1 + len(members))
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                adj.setdefault(a, {})[b] = adj.get(a, {}).get(b, 0.0) + w
                adj.setdefault(b, {})[a] = adj.get(b, {}).get(a, 0.0) + w
    return (adj, dropped) if report_dropped else adj


def coupling_ranking(
    adj: dict[str, dict[str, float]], note: str, *, k: int, exclude: Iterable[str] = (),
) -> list[tuple[str, float]] | None:
    row = adj.get(note)
    if not row:
        return None
    blocked = set(exclude) | {note}
    ranked = sorted(((p, w) for p, w in row.items() if p not in blocked),
                    key=lambda kv: (-kv[1], kv[0]))
    return ranked[:k] or None


# ---------------------------------------------------------------------------
# V4  Load-bearing notes: coreness, articulation points, surprise
# ---------------------------------------------------------------------------

def _pct_rank(values: dict[str, float]) -> dict[str, float]:
    """Fraction of the other nodes strictly below each node (ties share)."""
    n = len(values)
    if n < 2:
        return {k: 0.0 for k in values}
    ordered = sorted(values.values())
    below: dict[float, int] = {}
    for i, v in enumerate(ordered):
        below.setdefault(v, i)   # first index == count strictly below
    return {k: below[v] / (n - 1) for k, v in values.items()}


def load_bearing(G, *, betweenness: dict[str, float], degree: dict[str, int]):
    """(core_map, articulation points, surprise_map) over the wikilink graph.

    surprise = pct-rank(betweenness) - pct-rank(degree), in [-1, 1]: the note
    that many shortest paths cross although few notes link it. That is the
    InfraNodus "influential but not frequent" reading made per note, and
    the one case a degree-based hub list cannot show. Articulation points are
    the notes whose removal disconnects the graph; coreness is the k-core
    number, the depth at which a note is embedded.
    """
    import networkx as nx

    H = G
    loops = list(nx.selfloop_edges(G))
    if loops:  # core_number refuses self-loops; a note linking itself is legal
        H = G.copy()
        H.remove_edges_from(loops)
    core = nx.core_number(H) if H.number_of_nodes() else {}
    articulation = set(nx.articulation_points(H)) if H.number_of_edges() else set()
    nodes = list(H.nodes)
    b_rank = _pct_rank({n: float(betweenness.get(n, 0.0)) for n in nodes})
    d_rank = _pct_rank({n: float(degree.get(n, 0)) for n in nodes})
    surprise = {n: round(b_rank[n] - d_rank[n], 4) for n in nodes}
    return core, articulation, surprise


# ---------------------------------------------------------------------------
# V5  Partition dissonance: linked like one area, reads like another
# ---------------------------------------------------------------------------

def dissonance(G, zone_of: dict[str, int]) -> dict[str, float]:
    """Share of a note's wikilink neighbours whose semantic zone differs.

    Only notes with a zone and at least one zoned neighbour get a value: a
    neighbour without a vector is unknown, not disagreeing.
    """
    out: dict[str, float] = {}
    for n in G.nodes:
        z = zone_of.get(n)
        if z is None:
            continue
        zoned = [zone_of[m] for m in G.neighbors(n) if m != n and m in zone_of]
        if not zoned:
            continue
        out[n] = sum(1 for zz in zoned if zz != z) / len(zoned)
    return out


# ---------------------------------------------------------------------------
# V6  Concept burst: what the recent notes are about, against the baseline
# ---------------------------------------------------------------------------

def burst(
    created: dict[str, float],
    stems: dict[str, dict[str, int]],
    *,
    window_days: float = 14.0,
    min_recent: int = 3,
    z_min: float = 2.0,
) -> list[tuple[str, float, int, int]]:
    """(stem, z, n_recent, n_all) for stems over-represented in the window.

    The window is the last `window_days` of WRITING activity (relative to the
    newest note), not of wall-clock time: a vault written two months ago still
    has a most-recent fortnight. z is the one-proportion z-score of the stem's
    recent document frequency against its overall share. Abstains when every
    note is recent (no baseline) or none is.
    """
    notes = [p for p in stems if p in created]
    n_all = len(notes)
    if n_all < 2:
        return []
    latest = max(created[p] for p in notes)
    recent = {p for p in notes if created[p] >= latest - window_days * 86400.0}
    n_recent = len(recent)
    if n_recent == 0 or n_recent == n_all:
        return []
    df_all: Counter = Counter()
    df_recent: Counter = Counter()
    for p in notes:
        for s in stems[p]:
            df_all[s] += 1
            if p in recent:
                df_recent[s] += 1
    out: list[tuple[str, float, int, int]] = []
    for s, nr in df_recent.items():
        if nr < min_recent:
            continue
        p_all = df_all[s] / n_all
        var = n_recent * p_all * (1.0 - p_all)
        if var <= 0:
            continue  # present everywhere (or nowhere): no contrast to measure
        z = (nr - n_recent * p_all) / math.sqrt(var)
        if z >= z_min:
            out.append((s, round(z, 3), nr, df_all[s]))
    out.sort(key=lambda r: (-r[1], r[0]))
    return out


# ---------------------------------------------------------------------------
# V7  Note entropy: breadth and flatness of a note's concept distribution
# ---------------------------------------------------------------------------

def entropy_bits(counts: dict[str, int]) -> float:
    total = sum(c for c in counts.values() if c > 0)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def _nearest_rank(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = max(0, math.ceil(pct * len(ordered)) - 1)
    return ordered[idx]


def sprawling(
    stems: dict[str, dict[str, int]], *, pct: float = 0.9, flat: float = 0.8, min_notes: int = 10,
) -> list[tuple[str, int, float, float]]:
    """(path, n_stems, H, flatness) for notes both broad and flat.

    breadth: distinct stems at or above the vault's `pct` percentile;
    flatness: H / log2(n_stems) >= `flat`, i.e. no concept dominates. A long
    note about one thing is peaked (excluded); a note touching forty concepts
    evenly is the split candidate. Percentiles need a population: under
    `min_notes` the list is empty rather than a ranking of noise.
    """
    rows = []
    for path, counts in stems.items():
        n = sum(1 for c in counts.values() if c > 0)
        if n == 0:
            continue
        h = entropy_bits(counts)
        flatness = h / math.log2(n) if n > 1 else 0.0
        rows.append((path, n, h, flatness))
    if len(rows) < min_notes:
        return []
    breadth = _nearest_rank([r[1] for r in rows], pct)
    out = [(p, n, round(h, 3), round(f, 3)) for p, n, h, f in rows if n >= breadth and f >= flat]
    out.sort(key=lambda r: (-r[2], r[0]))
    return out
