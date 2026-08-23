# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""CORRELATE (ADR-0013, revised by ADR-0029): note-to-note edges from co-occurrence.

L1 kernel: no LLM, no API, no embedder, no store. Pure math over plain
{stem: count} maps: each note's top-k stems by RAW count, an edge where the
Jaccard of two top-k sets reaches tau. IDF was rejected by the data (it made a
note's top-k a function of the whole corpus; raw count keeps it a function of
that note alone).

The edges are a MEMO on CooccurStore (`note_adjacency`), never a persisted
section: they are a pure function of the contributions, and the 2026-07-10
persistence (prune on delete, orphan prune on load, a dirty-edge overlay in the
disk sync) bought nothing a recomputation does not. Measured 2026-08-23 on the
709-note vault: as a fusion leg the edges recovered 0 pairs the embed leg lacked
and cost 0.03 mrr when they fired, so the leg is gone too (ADR-0029). What stays
is what the mindmap radius and the report's "direct" provenance read.
"""
from __future__ import annotations

from collections.abc import Mapping

# Module constants (config promotion declined 2026-08-19; revisit only if a
# second vault ever needs different values).
_TOP_K = 30
_TAU = 0.25


def topk_set(nodes: Mapping[str, int], k: int = _TOP_K) -> frozenset[str]:
    """The k highest-count stems of one note, as a set. Tie-break lexicographic.

    `k` defaults to the module constant; production never passes it. It exists
    so a fixture can pin a small k instead of needing 30+ synthetic stems.
    """
    ranked = sorted(nodes.items(), key=lambda kv: (-kv[1], kv[0]))
    return frozenset(stem for stem, _count in ranked[:k])


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    if not union:
        return 0.0  # two empty top-k sets share nothing; 0.0, not a ZeroDivisionError
    return len(a & b) / len(union)


def compute_edges(
    nodes_by_path: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, float]]:
    """Symmetric adjacency {path: {other: jaccard}} over every pair at or above tau.

    Top-k sets are taken ONCE per note and candidates come from an inverted
    index of those sets, so the cost is O(N*k) to bucket plus one k-element set
    intersection per candidate pair. The refresh this replaced re-sorted the
    neighbour's stems for every candidate it met: 5.3 s for 709 notes, against
    well under a second here for the same 218 edges.
    """
    tops = {path: topk_set(nodes) for path, nodes in nodes_by_path.items()}
    by_stem: dict[str, list[str]] = {}
    for path, stems in tops.items():
        for stem in stems:
            by_stem.setdefault(stem, []).append(path)
    adj: dict[str, dict[str, float]] = {}
    for a, stems_a in tops.items():
        candidates: set[str] = set()
        for stem in stems_a:
            candidates.update(by_stem[stem])
        for b in candidates:
            if b <= a:
                continue  # each unordered pair once; the row is written both ways
            score = jaccard(stems_a, tops[b])
            if score >= _TAU:
                adj.setdefault(a, {})[b] = score
                adj.setdefault(b, {})[a] = score
    return adj
