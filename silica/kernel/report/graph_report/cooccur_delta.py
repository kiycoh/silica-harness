# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Co-occurrence vs wikilink delta — PROPOSED, embedder-free signals.

AUTOLINK (co-occurrence − wikilink), STALE (wikilink − co-occurrence) and
MISSING HUB (central concept with no hub note). This is the designated
landing zone for the ADR-0013 CORRELATE wiring: changes to the delta logic
must not leak into compute.py or render.py.
"""
from __future__ import annotations

import logging
from typing import Any

from silica.kernel.report.graph_report.models import (
    AutolinkCandidate,
    BurstingConcept,
    PrerequisiteEdge,
    SprawlingNote,
    IntegrationDeficit,
    MissingHub,
    StaleLink,
    VaultReport,
)

logger = logging.getLogger(__name__)


def _compute_cooccur_delta(
    report: VaultReport,
    G_und: Any,
    node_label: dict[str, str],
    *,
    cooccur_store: Any | None = None,
    k: int = 10,
) -> tuple[list[AutolinkCandidate], list[StaleLink], list[MissingHub], list[IntegrationDeficit]]:
    """Delta between what the text relates and what the wikilink graph holds.

    Four PROPOSED signals, no network, pure local compute:

      - AUTOLINK  (relatedness − wikilink): note pairs the relatedness facade
        ranks together (embed + cooccur, ADR-0029) but never wikilinked, more
        than 2 hops apart. Embedder-free when the embed index is empty: the
        facade's embed leg abstains and the cooccur leg carries the pass.
      - STALE     (wikilink − co-occurrence): wikilinked pairs whose notes share
        no concepts in text — a structural link without textual co-presence.
      - MISSING HUB (centrality − hub): a concept central in the discourse for
        which no note is titled — the next hub note to create.
      - INTEGRATION DEFICIT (concepts − degree): per-note divergence between
        textual richness and wikilink integration — a dense note never linked in.

    `cooccur_store` is injectable for testing; loaded from disk when None.
    Returns empty lists when the index is empty (best-effort, never raises).
    """
    from silica.kernel.recall.cooccurrence import cooccur_key, get_cooccur_store, tokenize
    from silica.kernel.recall.relatedness import _concept_idf, _cooccur_ranking, related_notes_many

    try:
        store = cooccur_store if cooccur_store is not None else get_cooccur_store()
    except Exception as exc:
        logger.debug("graph_report: co-occurrence index unavailable (%s)", exc)
        return [], [], [], []
    if len(store) == 0:
        return [], [], [], []

    scope = report.scope or None

    def _shared_labels(a: str, b: str) -> list[str]:
        na, nb = store.note_nodes(a), store.note_nodes(b)
        return sorted(store.node_label(s) for s in (set(na) & set(nb)))

    # Shared-concept evidence orders stems by IDF descending (top 5), so the
    # boilerplate-template stems a raw-count metric admits sink below the
    # discriminative ones. IDF lives ONLY here (display), never in the metric.
    # Computed lazily on the first candidate — one O(N) pass at render.
    idf_map: dict[str, float] | None = None

    def _shared_by_idf(a: str, b: str) -> list[str]:
        nonlocal idf_map
        if idf_map is None:
            all_stems: set[str] = set()
            for p in store.paths():
                all_stems |= set(store.note_nodes(p))
            idf_map = _concept_idf(store, all_stems, scope=scope)
        shared = set(store.note_nodes(a)) & set(store.note_nodes(b))
        ranked = sorted(shared, key=lambda s: (-idf_map.get(s, 0.0), s))
        return [store.node_label(s) for s in ranked[:5]]

    # Keyspace bridge: G_und node ids are graph paths WITH '.md'; store keys are
    # stripped (cooccur_key). Membership and hop checks must cross that boundary
    # here — on a real vault a raw `nid in G_und` matches nothing and the whole
    # AUTOLINK section silently comes out empty.
    gid_by_key = {cooccur_key(n): n for n in G_und.nodes}

    # Precompute once for the AUTOLINK loop below: the symmetric note-edge
    # adjacency (else note_edges_for does an O(E) reverse scan per note -> O(N*E))
    # and G_und neighbour sets (the gate only needs distance <= 2, not a full
    # shortest-path search per candidate).
    note_adj = store.note_adjacency()
    adj_sets: dict[str, set[str]] = {n: set(G_und.neighbors(n)) for n in G_und.nodes}

    def _within_2_hops(s: str, t: str) -> bool:
        if s == t or t in adj_sets.get(s, ()):
            return True  # 0 or 1 hop
        return bool(adj_sets.get(s, set()) & adj_sets.get(t, set()))  # common neighbour

    # --- AUTOLINK: the relatedness facade over every in-graph note, pairs >2
    #     hops away and not already wikilinked. A direct Jaccard edge
    #     (CORRELATE memo, a free lookup) wins provenance when the facade also
    #     surfaces the pair; the rest is "fused" and carries the RRF score.
    #     Candidates keep STORE keys (stripped): the cosine-band filter and the
    #     shared-concept evidence below consume them, and render strips anyway.
    #     One generator: the proposer that ranked the expanded cooccur profile
    #     per note scored recall 0.10 at 6.8 ms/note against the facade's 0.82
    #     at 5 ms on the 709-note vault (2026-08-23), and the embed-only
    #     god-node proposer it sat beside was a subset of this one.
    try:
        from silica.kernel.recall.embed import get_store as _get_embed_store
        _es = _get_embed_store()
        embed_store = _es if len(_es) > 0 else None
    except Exception:
        embed_store = None  # no index: the embed leg abstains, cooccur carries the pass
    in_graph = [nid for nid in store.paths() if nid in gid_by_key]
    fused = related_notes_many(in_graph, embed_store=embed_store, cooccur_store=store, k=k, scope=scope)
    autolinks: list[AutolinkCandidate] = []
    seen: set[tuple[str, str]] = set()
    for nid in in_graph:
        src_gid = gid_by_key[nid]
        direct = note_adj.get(nid, {})  # {tgt: jaccard}, both directions
        legs = (
            [(tgt, w, "direct") for tgt, w in direct.items()]
            + [(r.path, r.score, "fused") for r in fused.get(nid, [])]
        )
        for tgt, weight, provenance in legs:
            tgt_gid = gid_by_key.get(tgt)
            if tgt_gid is None:
                continue
            if _within_2_hops(src_gid, tgt_gid):
                continue  # already linked or trivially close (disconnected -> valid)
            key = (min(nid, tgt), max(nid, tgt))
            if key in seen:
                continue
            seen.add(key)
            autolinks.append(AutolinkCandidate(
                source=key[0], target=key[1],
                # Jaccard for a direct edge, the RRF score for a fused pair:
                # two decimals would flatten every RRF score to 0.02.
                weight=round(float(weight), 2 if provenance == "direct" else 4),
                shared=_shared_by_idf(nid, tgt),
                provenance=provenance,
            ))

    # --- #6 cosine-band: filter trivially-similar or nonsensically-distant ---
    # Paper (Marwitz 2026) S_own×other^filtered: removes pairs whose semantic
    # similarity is too high (trivial, A2 in expert ratings) or too low
    # (nonsensical, B in expert ratings). Best-effort: skipped silently when
    # embeddings are unavailable.
    try:
        from silica.kernel.recall.embed import get_store, _cosine
        _embed_store = get_store()
        if len(_embed_store) > 0:
            _cos_hi = 0.92
            _cos_lo = 0.35
            filtered: list[AutolinkCandidate] = []
            for a in autolinks:
                v_src = _embed_store.get_vec(a.source)
                v_tgt = _embed_store.get_vec(a.target)
                if v_src and v_tgt:
                    cos = _cosine(v_src, v_tgt)
                    if cos > _cos_hi or cos < _cos_lo:
                        continue  # too trivial or too alien
                filtered.append(a)
            autolinks = filtered
    except Exception:
        pass  # embeddings unavailable → no filtering, degrade gracefully

    # --- #8 convergence: S_(many_own)×other --------------------------------
    # Paper (Marwitz 2026, Table 2): the highest-interest section connects a
    # candidate to MANY of the researcher's own concepts. Silica's "own
    # concepts" are the god-node hubs; a candidate touching more hubs (either
    # endpoint co-occurring with the hub, or being a hub itself) earns a higher
    # convergence and is ranked by convergence × weight. Degrades to the prior
    # weight-only ordering when there are no god nodes.
    # Same keyspace bridge: god-node ids are graph paths, candidates carry store
    # keys — normalise once so the hub-exclusion and reach comparisons line up.
    god_ids = [cooccur_key(n.id) for n in report.god_nodes]
    if god_ids:
        god_set = set(god_ids)
        # expand=False: count only DIRECT concept overlap with the hub, so
        # convergence measures genuine reach into distinct hubs rather than
        # transitive bleed through a single shared concept.
        god_related: dict[str, set[str]] = {}
        for g in god_ids:
            ranking = _cooccur_ranking(store, g, k=50, exclude=set(), scope=scope, expand=False)
            god_related[g] = {p for p, _w in (ranking or [])}
        for a in autolinks:
            # The "other" endpoint(s) are those not themselves hubs; convergence
            # counts how many distinct hubs that other concept reaches into.
            others = [e for e in (a.source, a.target) if e not in god_set]
            a.convergence = sum(
                1 for g in god_ids
                if any(o in god_related[g] for o in others)
            )

    # Per-leg quota: direct weights are Jaccard (<=1) while fused weights are
    # RRF scores (~0.02) — one mixed sort would order the two legs by their
    # scale, not their evidence. Direct gets up to half the slots ranked by its
    # native Jaccard; fused keeps the convergence ranking for the rest; either
    # leg backfills when the other runs short.
    direct_leg = sorted(
        (a for a in autolinks if a.provenance == "direct"),
        key=lambda a: (-a.weight, a.source, a.target),
    )
    fused_leg = sorted(
        (a for a in autolinks if a.provenance != "direct"),
        key=lambda a: (-(a.convergence * a.weight), -a.weight, a.source, a.target),
    )
    take = min(len(direct_leg), k - k // 2)
    autolinks = direct_leg[:take] + fused_leg[: k - take]
    if len(autolinks) < k:
        autolinks += direct_leg[take: take + k - len(autolinks)]

    # --- INTEGRATION DEFICIT: concept-rich note, weakly wikilinked ----------
    # Per-note divergence between textual richness (concepts contributed to the
    # co-occurrence graph) and structural integration (wikilink degree). The
    # common decay pattern: a dense note written and never linked in. Pure
    # ranking, no weights — same shape as AttentionCandidate's score.
    # ponytail: raw concept count favours long notes; IDF-weight the count if
    # boilerplate stems ever dominate the ranking.
    deficits: list[IntegrationDeficit] = []
    for nid in store.paths():
        gid = gid_by_key.get(nid)
        if gid is None:
            continue  # outside the graph scope
        concepts = len(store.note_nodes(nid))
        if concepts == 0:
            continue  # abstain: a note with no concepts can't be assessed
        d = int(G_und.degree(gid))
        deficits.append(IntegrationDeficit(
            path=nid, concepts=concepts, degree=d,
            score=round(concepts / (1 + d), 3),
        ))
    deficits.sort(key=lambda x: (-x.score, x.path))
    deficits = deficits[:k]

    # --- STALE: wikilinked but the two notes share no concepts --------------
    stale: list[StaleLink] = []
    for u, v in G_und.edges():
        if not store.note_nodes(u) or not store.note_nodes(v):
            continue  # a note with no concepts can't be assessed -> don't flag
        if not _shared_labels(u, v):
            stale.append(StaleLink(source=min(u, v), target=max(u, v)))
    stale.sort(key=lambda s: (s.source, s.target))
    stale = stale[:k]

    # --- MISSING HUB: central concept with no note titled after it ----------
    adj = store.adjacency(scope)  # the aggregate to_networkx() would wrap, unwrapped
    titled_stems: set[str] = set()
    for label in node_label.values():
        for sentence in tokenize(label, stem_lang=store.lang, stopword_lang=store.lang):
            titled_stems.update(stem for stem, _surface in sentence)

    hubs: list[MissingHub] = []
    for stem, nbrs in adj.items():
        if stem in titled_stems:
            continue  # a hub note already formalises this concept
        wdeg = sum(nbrs.values())
        hubs.append(MissingHub(concept=store.node_label(stem), centrality=round(wdeg, 2)))
    hubs.sort(key=lambda h: (-h.centrality, h.concept))
    hubs = hubs[:k]

    return autolinks, stale, hubs, deficits


# RefD related-set depth: the paper's related concepts are the top neighbours
# by similarity; 30 is the facade pool size (_POOL_MIN) so a prerequisite
# reads the same neighbourhood /related shows.
_REFD_RELATED_K = 30
_REFD_THETA = 0.1
# Fewer related notes than this and the note's side abstains; see
# signals.refd_edges for the measured failure this guards.
_REFD_MIN_RELATED = 5


def _compute_cooccur_variables(
    report: VaultReport,
    G_und: Any,
    G_dir: Any,
    *,
    cooccur_store: Any | None = None,
    created: dict[str, float] | None = None,
    k: int = 10,
) -> tuple[list[PrerequisiteEdge], dict[str, list[str]], list[SprawlingNote], list[BurstingConcept]]:
    """Store-derived variables (V2, V6, V7) and the shared-concept evidence for
    the structural (V1) and coupled (V3) pairs already on the report.

    Same contract as `_compute_cooccur_delta`: injectable store, empty results
    when the index is empty, never raises. `created` is {graph id: creation
    timestamp} from the analytics body scan; None skips the burst.
    """
    from silica.kernel.recall.cooccurrence import cooccur_key, get_cooccur_store
    from silica.kernel.recall.relatedness import _cooccur_ranking
    from silica.kernel.recall.signals import burst, refd_edges, sprawling

    try:
        store = cooccur_store if cooccur_store is not None else get_cooccur_store()
    except Exception as exc:
        logger.debug("graph_report: co-occurrence index unavailable (%s)", exc)
        return [], {}, [], []
    if len(store) == 0:
        return [], {}, [], []

    scope = report.scope or None
    gid_by_key = {cooccur_key(n): n for n in G_und.nodes}
    in_graph = [p for p in store.paths() if p in gid_by_key]

    def _shared(a_gid: str, b_gid: str) -> list[str]:
        na, nb = store.note_nodes(cooccur_key(a_gid)), store.note_nodes(cooccur_key(b_gid))
        return sorted(store.node_label(s) for s in (set(na) & set(nb)))[:5]

    for row in list(report.structural_links) + list(report.coupled_pairs):
        row.shared = _shared(row.source, row.target)

    # --- V2 prerequisites: RefD over (directed wikilinks, cooccur related sets)
    links: dict[str, set[str]] = {}
    for u, v in G_dir.edges():
        links.setdefault(cooccur_key(u), set()).add(cooccur_key(v))
    related: dict[str, list[tuple[str, float]]] = {}
    for p in in_graph:
        ranking = _cooccur_ranking(
            store, p, k=_REFD_RELATED_K, exclude={p}, scope=scope, expand=False,
        )
        if ranking:
            related[p] = [(q, float(w)) for q, w in ranking if q in gid_by_key]
    prereqs = [
        PrerequisiteEdge(prereq=a, dependent=b, refd=round(r, 4))
        for a, b, r in refd_edges(links, related, theta=_REFD_THETA, min_related=_REFD_MIN_RELATED)
    ]
    prereq_map: dict[str, list[str]] = {}
    for e in prereqs:
        prereq_map.setdefault(e.dependent, []).append(e.prereq)

    # --- V7 sprawling and V6 bursting, both over the store's note -> stems map
    stems = {p: dict(store.note_nodes(p)) for p in in_graph}
    sprawl = [
        SprawlingNote(path=p, concepts=n, entropy=h, flatness=f)
        for p, n, h, f in sprawling(stems)
    ][:k]
    bursting: list[BurstingConcept] = []
    if created:
        created_keys = {cooccur_key(g): ts for g, ts in created.items() if ts is not None}
        bursting = [
            BurstingConcept(concept=store.node_label(s), z=z, recent=nr, total=na)
            for s, z, nr, na in burst(created_keys, stems)
        ][:k]
    return prereqs[:max(k, 50)], prereq_map, sprawl, bursting
