# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Relatedness facade — fuses the two PROPOSE-layers into one note ranking.

Silica has two independent signals about which notes belong together:

  - **embeddings** (`kernel/embed.py`)        — holistic semantic similarity
  - **co-occurrence** (`kernel/cooccurrence.py`) — how the author actually
    co-mentions concepts (deterministic, embedder-free)

They live at different granularities (note-level vs concept-level) and on
incomparable scales (cosine in [0,1] vs unbounded integer weight). This module
is the single place where they meet. It:

  1. Reconciles granularity via a concept->notes **inverted index**, turning the
     concept-level co-occurrence graph into a note-level ranking.
  2. Fuses the two note rankings with **Reciprocal Rank Fusion** (RRF), which
     only consults rank position, so incomparable scores combine cleanly.
  3. Lets a degenerate proponent **abstain** (return None) instead of emitting a
     flat zero ranking that would poison RRF. A leg that abstains contributes no
     reciprocal-rank terms, so fusion degrades automatically to the survivor's
     ranking — "embedder down => routing on co-occurrence", with no special-case
     branch.

Provenance is preserved: every returned note carries an `evidence` list
(`embed:0.83`, `cooccur:w9`, or both).

Generalises the existing rule "embeddings PROPOSE, graph DISPOSES" into
"the proponents propose, the graph disposes": this facade is a proposer, never
authoritative about vault structure.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Callable

from silica.kernel.recall.cooccurrence import CooccurStore, cooccur_key
from silica.kernel.recall.embed import EmbedStore
from silica.kernel.recall.graph_export import is_vault_artifact
from silica.kernel.recall.paths import in_folder

logger = logging.getLogger(__name__)

# Standard RRF damping constant (Cormack et al. 2009). Larger -> flatter weight
# decay across ranks; 60 is the widely-used default.
RRF_K = 60

# A leg whose best candidate scores at or below this is treated as signal-free
# (e.g. a zero query vector makes every cosine 0.0) and abstains.
_NOISE_FLOOR = 1e-6

# BM25 knobs for the cooccur tf term (CONFIG.cooccur_bm25, spec section 9.2).
# Textbook defaults, DELIBERATELY UNTUNED: the +4.02pp gate is credible precisely
# because they were not fitted to this vault, so they stay constants rather than
# config knobs. A sweep, if ever, is pre-declared after phase 3.
BM25_K1 = 1.2
BM25_B = 0.75

# Cooccur confidence gate (retrieval-gates spec 2026-07-14). ponytail: dormant —
# 0.0 never fires. Phase-0 (2026-07-17, bench/phase0_gates.json) recorded the
# no-fire reference only: vault coverage p10 0.259 / lme_s p10 0.432, so any
# future threshold <=0.1 is home-turf-safe. The fire side (MuSiQue, vocabulary
# mismatch) is no longer on disk; freeze only after a MuSiQue re-run, and shelve
# the gate if that run shows no wide separation (spec abort criterion).
_COOCCUR_MIN_CONFIDENCE = 0.0
# Calibration hook: harnesses set it to capture per-query
# {"coverage", "flatness", "fired"}; production leaves it None.
COOCCUR_GATE_PROBE: Callable[[dict], None] | None = None

# Neighbour edges are associative, not direct membership: discount their pull on
# the query concept profile so notes literally sharing concepts still dominate.
_EXPANSION_DISCOUNT = 0.25

# Minimum per-leg candidate pool fed to RRF, independent of the caller's k, so
# fusion has enough material to find agreement before the final top-k cut.
_POOL_MIN = 25


def reset_vault_caches() -> None:
    """Drop all vault-scoped leg caches so a /vault switch releases the previous
    vault's index/vectors instead of retaining them for the process lifetime.

    Lives on the facade so callers (the CLI /vault handler) don't import the
    legs directly — the leg caches are path-keyed, so this is memory release,
    not correctness (a stale-keyed lookup would miss and rebuild anyway).
    """
    from silica.kernel.recall import cooccurrence, embed, lexical

    embed.clear()
    cooccurrence.clear()
    lexical.clear()


@dataclass
class RelatedNote:
    """A fused related-note candidate with its provenance.

    `score` is the RRF score (only meaningful for ordering, not as a similarity).
    `evidence` records which legs proposed it and their native scores, as
    display strings; `embed_score` / `cooccur_weight` expose the same raw signals
    structurally (None when that leg did not propose the note) so callers can
    threshold or render without parsing the evidence strings.
    """
    path: str
    name: str
    score: float
    evidence: list[str]
    embed_score: float | None = None
    cooccur_weight: float | None = None
    edge_score: float | None = None  # CORRELATE (ADR-0013): direct note_edges Jaccard
    # ADR-0019: "vault" = active vault, "memory" = personal-memory lane. A
    # memory result's `path` is relative to the MEMORY vault — consumers must
    # respect this marker (open the right note in the right vault) and never
    # write through it.
    origin: str = "vault"


# ---------------------------------------------------------------------------
# RRF fusion (pure)
# ---------------------------------------------------------------------------

def _rrf_fuse(rankings: list[list[tuple[str, float]]]) -> dict[str, float]:
    """Reciprocal Rank Fusion over several ranked lists of (key, _score).

    Each list must be sorted best-first. The native score is ignored — only the
    rank position counts — so lists on incomparable scales combine cleanly. A key
    appearing in multiple lists accumulates a reciprocal-rank term from each.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (key, _score) in enumerate(ranking):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    return fused


# ---------------------------------------------------------------------------
# Embedding leg
# ---------------------------------------------------------------------------

def _rank_embeddings_from_vec(
    embed_store: EmbedStore | None,
    vec: list[float] | None,
    *,
    k: int,
    exclude: set[str],
) -> list[tuple[str, str, float]] | None:
    """Note ranking from a query vector, or None if the embed leg abstains.

    Abstains when: no store, no vector, the search errors, or the output is
    degenerate (every score at the noise floor — a flat zero ranking that would
    poison RRF rather than inform it).
    """
    if embed_store is None or vec is None:
        return None
    try:
        cands = embed_store.cosine_top_k(vec, k=k, exclude=exclude)
    except Exception:
        return None
    if not cands:
        return None
    if max((c.get("score", 0.0) for c in cands), default=0.0) <= _NOISE_FLOOR:
        return None
    return [(c["path"], c["name"], float(c.get("score", 0.0))) for c in cands]


def _embed_ranking(
    embed_store: EmbedStore | None,
    query_path: str,
    *,
    k: int,
    exclude: set[str],
) -> list[tuple[str, str, float]] | None:
    """Embed ranking for an INDEXED note: resolve its vector by path, then rank."""
    if embed_store is None:
        return None
    return _rank_embeddings_from_vec(
        embed_store, embed_store.get_vec(cooccur_key(query_path)), k=k, exclude=exclude
    )


# ---------------------------------------------------------------------------
# Co-occurrence leg (granularity reconciliation)
# ---------------------------------------------------------------------------

def _profile_from_seeds(
    cooccur_store: CooccurStore,
    # Mapping, not dict: note_nodes() hands back integer counts and a
    # dict[str, int] is not a dict[str, float] (invariant), while the body only
    # ever reads.
    seeds: Mapping[str, float],
    *,
    scope: str | None,
    expand: bool,
) -> dict[str, float]:
    """Weighted concept profile {stem: weight} from seed concepts.

    When `expand`, adds each seed's co-occurrence neighbours at a discounted
    weight (associative reach: a note about a strongly-linked neighbour concept
    is related even without a literal shared concept).
    """
    if not seeds:
        return {}
    profile: dict[str, float] = dict(seeds)
    if expand:
        adj = cooccur_store.adjacency(scope=scope)
        for stem, weight in list(profile.items()):
            for neighbour, edge_weight in adj.get(stem, {}).items():
                profile[neighbour] = (
                    profile.get(neighbour, 0.0)
                    + weight * edge_weight * _EXPANSION_DISCOUNT
                )
    return profile


def _seed_from_text(text: str, lang: str) -> dict[str, float]:
    """Seed concepts {stem: count} from raw query text (for fresh queries)."""
    from silica.kernel.recall.cooccurrence import tokenize

    seeds: dict[str, float] = {}
    for sentence in tokenize(text, stem_lang=lang):
        for stem, _surface in sentence:
            seeds[stem] = seeds.get(stem, 0.0) + 1.0
    return seeds


def _concept_idf(
    cooccur_store: CooccurStore,
    stems: set[str],
    *,
    scope: str | None,
) -> dict[str, float]:
    """Inverse document frequency per stem: log((N+1) / df), N = in-scope notes.

    Without this, a hub concept present in hundreds of notes (e.g. "data
    science", "statistica") dominates every ranking purely by breadth, burying
    the discriminative concepts that actually make two notes the same. IDF is the
    standard fix — a stem in every note scores ~0, a rare stem scores high — and
    on the real vault it lifts true twins from rank 6/miss into the visible top-k
    where plain overlap left them buried. Rarity is a corpus property, so `blocked`
    (query + excludes) still counts toward df.

    The `N+1` numerator (smoothed IDF) keeps the weight strictly positive even
    when a stem sits in every note — the raw `log(N/df)` collapses to exactly 0
    there, which on a tiny or brand-new vault (N=1, every stem ubiquitous) would
    zero the whole co-occurrence signal and silently drop the leg. On a real
    corpus the smoothing is negligible, so hub suppression is unchanged.

    ponytail: backed by `CooccurStore.stem_postings()` (Task 3.4) instead of an
    O(notes) scan — df is just posting length (scope-filtered), n is the
    in-scope note count. Only queried `stems` are looked up, so this is
    O(|stems| + in-scope postings) rather than O(all notes).
    """
    import math

    postings = cooccur_store.stem_postings()
    if scope is None:
        n = len(cooccur_store)
        df = {stem: len(postings[stem]) for stem in stems if stem in postings}
    else:
        in_scope = cooccur_store.paths_in_scope(scope)
        n = len(in_scope)
        scope_set = set(in_scope)
        df = {}
        for stem in stems:
            plist = postings.get(stem)
            if not plist:
                continue
            c = sum(1 for p in plist if p in scope_set)
            if c:
                df[stem] = c

    return {stem: math.log((n + 1) / c) for stem, c in df.items() if c > 0}


def _rank_cooccur_from_profile(
    cooccur_store: CooccurStore,
    profile: dict[str, float],
    *,
    k: int,
    blocked: set[str],
    scope: str | None,
) -> list[tuple[str, float]] | None:
    """Rank in-scope notes by IDF-weighted concept overlap with `profile`
    (implicit concept->notes inverted index). Returns None when nothing overlaps.

    The tf term is raw count by default and BM25-saturated under
    CONFIG.cooccur_bm25 (docs/specs/cooccur-scoring.md phase 1). Nothing else moves
    with the flag: same IDF, same candidate set, same filters, same abstain, same
    tie-break, so the flag isolates the tf term the probe measured.
    """
    if not profile:
        return None
    from silica.config import CONFIG

    bm25 = bool(getattr(CONFIG, "cooccur_bm25", False))
    idf = _concept_idf(cooccur_store, set(profile), scope=scope)
    postings = cooccur_store.stem_postings()
    lens, avgdl = cooccur_store.doc_lengths() if bm25 else ({}, 1.0)

    # Accumulate over the inverted index, stem-major: walk each profile stem's
    # posting list once and add its term to the notes that actually appear in it.
    #
    # This used to run note-major — build the candidate union, then for every
    # candidate re-scan every profile stem and test `path in plist`. That is
    # O(|candidates| x |profile|) with a `postings.get(stem)` inside the inner
    # loop, so the same posting list was fetched once per candidate. Profiled on
    # a 718-note vault with expand=True: 113M dict.get calls over 40 queries,
    # 0.52s each, and the report's per-note loop over it took 129s. Stem-major
    # visits only the (stem, note) pairs that contribute — the non-matches, which
    # were the overwhelming majority of the work, are never enumerated.
    #
    # Bit-identical, not merely equivalent: `profile` is the outer loop in both
    # forms, so each note still receives its terms in profile order and the
    # float summation never reassociates. The per-term expression is kept
    # verbatim for the same reason.
    note_scores: dict[str, float] = {}
    norms: dict[str, float] = {}      # BM25 length norm per note, computed once
    allowed: dict[str, bool] = {}     # blocked/in_folder verdict per note, once
    for stem, weight in profile.items():
        if not weight:
            continue
        plist = postings.get(stem)
        if not plist:
            continue
        for path, tf in plist.items():
            ok = allowed.get(path)
            if ok is None:
                ok = path not in blocked and in_folder(path, scope)
                allowed[path] = ok
            if not ok:
                continue
            if bm25:
                # Length normalisation depends on the document alone, so it is
                # memoized per note. `or 1`: a candidate always has a posting,
                # hence a positive length — the fallback only guards an
                # incoherent store, where a 0 would collapse norm to K1*(1-B)
                # and hand that note the top score.
                norm = norms.get(path)
                if norm is None:
                    norm = BM25_K1 * (1.0 - BM25_B + BM25_B * (lens.get(path, 0) or 1) / avgdl)
                    norms[path] = norm
                term = tf * (BM25_K1 + 1.0) / (tf + norm)
            else:
                term = tf
            note_scores[path] = note_scores.get(path, 0.0) + weight * term * idf.get(stem, 0.0)

    # A note whose terms all cancelled to zero never scored an overlap — the
    # note-major form dropped it with `if overlap > 0.0`, so drop it here too.
    note_scores = {p: s for p, s in note_scores.items() if s > 0.0}
    if not note_scores:
        return None
    ranked = sorted(note_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    # Confidence gate is off in production (_COOCCUR_MIN_CONFIDENCE == 0.0, no probe),
    # so skip the whole signal compute — an extra note_nodes() call plus two IDF sums
    # per query — unless the experiment hook or a real threshold is active.
    if COOCCUR_GATE_PROBE or _COOCCUR_MIN_CONFIDENCE > 0.0:
        # Confidence signals (retrieval-gates spec): coverage measures the diagnosed
        # cause (query/corpus vocabulary mismatch — IDF mass of profile stems the top
        # hit actually matches), flatness the symptom (indiscriminate near-uniform
        # scores). Values already in hand; no extra corpus pass.
        total_mass = sum(w * idf.get(s, 0.0) for s, w in profile.items())
        top_stems = set(cooccur_store.note_nodes(ranked[0][0]))
        matched = sum(w * idf.get(s, 0.0) for s, w in profile.items() if s in top_stems)
        coverage = (matched / total_mass) if total_mass > 0 else 0.0
        scores = [s for _p, s in ranked]
        flatness = scores[0] / statistics.median(scores)
        fired = coverage < _COOCCUR_MIN_CONFIDENCE
        if COOCCUR_GATE_PROBE:
            COOCCUR_GATE_PROBE({"coverage": coverage, "flatness": flatness, "fired": fired})
        if fired:
            return None
    return ranked


def _cooccur_ranking(
    cooccur_store: CooccurStore | None,
    query_path: str,
    *,
    k: int,
    exclude: set[str],
    scope: str | None = None,
    expand: bool = True,
) -> list[tuple[str, float]] | None:
    """Co-occurrence ranking for an INDEXED note: seed from its own concepts."""
    if cooccur_store is None:
        return None
    profile = _profile_from_seeds(
        cooccur_store, cooccur_store.note_nodes(query_path), scope=scope, expand=expand
    )
    return _rank_cooccur_from_profile(
        cooccur_store, profile, k=k, blocked=set(exclude) | {query_path}, scope=scope
    )


# ---------------------------------------------------------------------------
# Facade entry point
# ---------------------------------------------------------------------------

def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


# Key namespace for memory-lane candidates inside the shared RRF dict: the two
# lanes are different vaults, so identical relative paths are DIFFERENT notes.
# NUL cannot appear in a filename, so the prefix can never collide.
_MEM = "\x00memory:"


def _fuse(
    embed_rank: list[tuple[str, str, float]] | None,
    cooc_rank: list[tuple[str, float]] | None,
    *,
    k: int,
    edges_rank: list[tuple[str, float]] | None = None,
    mem_embed_rank: list[tuple[str, str, float]] | None = None,
    mem_cooc_rank: list[tuple[str, float]] | None = None,
    recall_rank: list[tuple[str, float]] | None = None,
    lexical_rank: list[tuple[str, float]] | None = None,
    structural_rank: list[tuple[str, float]] | None = None,
    coupling_rank: list[tuple[str, float]] | None = None,
    structural_boost: dict[str, float] | None = None,
) -> list[RelatedNote]:
    """RRF-fuse the per-leg rankings into RelatedNotes with provenance.

    Shared by both facade entry points. A None ranking is an abstaining leg and
    contributes no terms; [] is returned only when all legs abstain.

    `edges_rank` is the CORRELATE third leg — direct note_edges of the query,
    ranked by Jaccard. Only related_notes(query_path) supplies it; the
    fresh-query facade always abstains here (fresh text has no note_edges row).

    `mem_embed_rank` / `mem_cooc_rank` are the personal-memory lane (ADR-0019):
    same fusion, key-namespaced under `_MEM` so a memory note never collides
    with (or masquerades as) an active-vault note. Its results come out with
    origin="memory" and `memory:`-prefixed evidence.

    `recall_rank` (phase 1 of `improve`, LoCoMo eval-only): notes previously
    confirmed helpful for this vault/conversation, best-first by weight. None
    abstains — every caller except the LoCoMo runner leaves this unset.

    `lexical_rank`: the hand-written BM25/fuzzy leg (opt-in via `use_lexical`
    on `facade_retrieve`), vault-only. None abstains — every caller except
    that opt-in path leaves this unset.
    """
    rankings: list[list[tuple[str, float]]] = []
    embed_scores: dict[str, float] = {}
    names: dict[str, str] = {}
    if embed_rank is not None:
        rankings.append([(path, score) for path, _name, score in embed_rank])
        for path, name, score in embed_rank:
            embed_scores[path] = score
            names[path] = name
    if mem_embed_rank is not None:
        rankings.append([(_MEM + path, score) for path, _name, score in mem_embed_rank])
        for path, name, score in mem_embed_rank:
            embed_scores[_MEM + path] = score
            names[_MEM + path] = name

    cooc_scores: dict[str, float] = {}
    if cooc_rank is not None:
        rankings.append(list(cooc_rank))
        cooc_scores = dict(cooc_rank)
    if mem_cooc_rank is not None:
        rankings.append([(_MEM + path, w) for path, w in mem_cooc_rank])
        cooc_scores.update({_MEM + path: w for path, w in mem_cooc_rank})

    edge_scores: dict[str, float] = {}
    if edges_rank is not None:
        rankings.append(list(edges_rank))
        edge_scores = dict(edges_rank)

    recall_scores: dict[str, float] = {}
    if recall_rank is not None:
        rankings.append(list(recall_rank))
        recall_scores = dict(recall_rank)

    lexical_scores: dict[str, float] = {}
    if lexical_rank is not None:
        rankings.append(list(lexical_rank))
        lexical_scores = dict(lexical_rank)

    # V1/V3 (spec 2026-08-22-graph-variables-design): two more abstaining
    # legs. `structural_rank` is Adamic-Adar over the wikilink graph (notes
    # two hops away through shared neighbours), `coupling_rank` the shared
    # transactions (sources cited, runs written). Store keyspace, like every
    # other leg; None abstains and fusion is bit-identical to before.
    structural_scores: dict[str, float] = {}
    if structural_rank is not None:
        rankings.append(list(structural_rank))
        structural_scores = dict(structural_rank)
    coupling_scores: dict[str, float] = {}
    if coupling_rank is not None:
        rankings.append(list(coupling_rank))
        coupling_scores = dict(coupling_rank)

    fused = _rrf_fuse(rankings)
    # V1 as corroboration rather than generation: a candidate the other legs
    # already surfaced is lifted by its Adamic-Adar score with the query
    # (aa/(1+aa), in [0,1)); a candidate no leg surfaced is never introduced.
    # A leg introduces its whole pool, and measured 2026-08-22 that pushed the
    # true counterpart down on every one of 481 and 845 golden pairs.
    if structural_boost:
        for p in list(fused):
            aa = structural_boost.get(p)
            if aa:
                fused[p] *= 1.0 + aa / (1.0 + aa)
                structural_scores.setdefault(p, aa)
    # Vault-root artifacts (log.md, GRAPH_REPORT.md) are excluded at index-build,
    # but a stale vector embedded before that exclusion outlives it: the store is
    # upsert-only and never prunes departed notes. Drop them here, before the
    # top-k cut, so no store consumer (map/autolink/dedup) ever surfaces one.
    fused = {
        p: s for p, s in fused.items()
        if not is_vault_artifact(p.removeprefix(_MEM))
    }
    if not fused:
        return []

    out: list[RelatedNote] = []
    for path, score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0])):
        evidence: list[str] = []
        embed_score = embed_scores.get(path)
        cooc_weight = cooc_scores.get(path)
        edge_score = edge_scores.get(path)
        if embed_score is not None:
            evidence.append(f"embed:{embed_score:.2f}")
        if cooc_weight is not None:
            evidence.append(f"cooccur:w{int(round(cooc_weight))}")
        if edge_score is not None:
            evidence.append(f"edge:{edge_score:.2f}")
        recall_weight = recall_scores.get(path)
        if recall_weight is not None:
            evidence.append(f"recall:{int(recall_weight)}")
        lexical_weight = lexical_scores.get(path)
        if lexical_weight is not None:
            evidence.append(f"lex:{lexical_weight:.2f}")
        structural_score = structural_scores.get(path)
        if structural_score is not None:
            evidence.append(f"struct:{structural_score:.2f}")
        coupling_score = coupling_scores.get(path)
        if coupling_score is not None:
            evidence.append(f"coupled:{coupling_score:.2f}")
        origin = "vault"
        if path.startswith(_MEM):
            origin = "memory"
            evidence = [f"memory:{e}" for e in evidence]
        out.append(
            RelatedNote(
                path=path.removeprefix(_MEM),
                name=names.get(path, _basename(path.removeprefix(_MEM))),
                score=score,
                evidence=evidence,
                embed_score=embed_score,
                cooccur_weight=cooc_weight,
                edge_score=edge_score,
                origin=origin,
            )
        )
    return dedupe_mirror_shadows(out)[:k]


def _mirror_prefix() -> str:
    """`"silica/"` when the ACTIVE vault stages writes in the mirror, else "".

    Keyed on the one write_dir name whose contents mirror the vault tree
    (onboarding.adopt.SAFE_WRITE_DIR — same function-level import as
    vault_manifest.seed_mirror_copy); `docs/silica` is Silica's own folder in
    a repo, not a mirror of it, so no dedupe applies there.
    """
    try:
        from silica.kernel.vault_manifest import active_write_dir
        from silica.onboarding.adopt import SAFE_WRITE_DIR

        wd = active_write_dir()
        return wd + "/" if wd == SAFE_WRITE_DIR else ""
    except Exception:
        return ""


def dedupe_mirror_shadows(results: list[RelatedNote]) -> list[RelatedNote]:
    """Drop originals shadowed by their own mirror copy.

    Safe mode stages `silica/X` as the pending update of `X`: one note, two
    paths — and /find listed both (observed 2026-08-15). The mirror copy is
    the curated, newer state, so it wins; an original stays only when no
    mirror twin made the pool. Memory-lane results are never compared: their
    paths belong to the memory vault (ADR-0019).
    """
    prefix = _mirror_prefix()
    if not prefix:
        return results
    mirrored = {
        r.path[len(prefix):].removesuffix(".md").casefold()
        for r in results
        if r.origin == "vault" and r.path.startswith(prefix)
    }
    if not mirrored:
        return results
    return [
        r for r in results
        if not (
            r.origin == "vault"
            and not r.path.startswith(prefix)
            and r.path.removesuffix(".md").casefold() in mirrored
        )
    ]


def neighbours_above(query_path: str, floor: float) -> list[str] | None:
    """Note names within cosine `floor` of an INDEXED note; None ⇒ can't answer.

    Score-gated, not rank-gated, so it does not go through fusion: the caller
    (AUTOLINK's relevance gate) needs "everything relevant enough", and a top-k
    would silently drop relevant titles on a large vault. Lives here anyway
    because it IS a relatedness query, and the legs are the facade's to touch.

    None is the abstention — no embed index, or this note has no vector yet — and
    is distinct from `[]`, which means the vault holds nothing close enough.
    Callers must keep the two apart: one is "gate unavailable, don't narrow",
    the other is "gate says nothing".
    """
    if floor <= 0:
        return None
    try:
        from silica.kernel.recall.cooccurrence import cooccur_key
        from silica.kernel.recall.embed import get_store

        store = get_store()
        if len(store) == 0:
            return None
        key = cooccur_key(query_path)
        vec = store.get_vec(key)
        if not vec:
            return None
        # Exclude by the STORE's key, not the raw path: the store drops `.md`
        # (cooccur_key), so a raw-path exclude never matches and the query note
        # comes back as its own closest neighbour.
        hits = store.cosine_top_k(vec, k=len(store), exclude={key})
    except Exception as exc:
        logger.debug("neighbours_above: abstaining for '%s' (%s)", query_path, exc)
        return None
    return [
        _basename(h.get("path") or "")
        for h in hits
        if h.get("score", 0.0) >= floor and h.get("path")
    ]


def _structural_ranking(
    graph, query_path: str, *, k: int, exclude: set[str],
) -> list[tuple[str, float]] | None:
    """V1 leg: Adamic-Adar neighbours of `query_path` over the wikilink graph.

    The graph carries driver node ids (with `.md`); the facade speaks store
    keys. Bridged here, both ways, so the leg's output joins the other legs'
    keyspace in `_fuse` — the same seam autolink/missing-links cross in the
    report. None abstains: no graph, query not a node, or no two-hop candidate.
    """
    if graph is None:
        return None
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


def related_notes(
    query_path: str,
    *,
    embed_store: EmbedStore | None = None,
    cooccur_store: CooccurStore | None = None,
    memory_embed_store: EmbedStore | None = None,
    memory_cooccur_store: CooccurStore | None = None,
    k: int = 10,
    scope: str | None = None,
    exclude: set[str] | None = None,
    expand: bool = False,
    graph=None,
    coupling_rank: list[tuple[str, float]] | None = None,
    structural_mode: str = "boost",
) -> list[RelatedNote]:
    """Return the top-k notes related to an INDEXED note `query_path`.

    Stores are injected (pass None for a leg that is unavailable — that leg
    abstains and fusion degrades to the survivor). Returns [] only when both
    legs abstain. Each result carries `evidence` recording its provenance.

    `memory_*_store` are the personal-memory lane (ADR-0019): the same query
    signals (the note's vector / concept stems) ranked against the memory
    vault's stores. None (the default) ⇒ the lane abstains and fusion is
    bit-identical to single-vault. `scope`/`exclude` are active-vault concepts
    and do not apply to the memory lane.

    `expand` (default off) adds associative co-occurrence neighbours to the
    concept profile. On a real vault this re-inflates hub concepts and buries
    true matches even under IDF weighting, so it stays opt-in for the one caller
    that wants pure associative reach (autolink candidate discovery).

    `graph` (an nx.Graph over driver node ids, e.g. graph_export's cached
    wikilink graph) switches on the structural signal (V1, Adamic-Adar) in
    `structural_mode`: "boost" (default) lifts candidates the other legs
    surfaced, "leg" adds the Adamic-Adar ranking as a fusion leg of its own.
    `coupling_rank` is the coupling leg (V3) already ranked in store keyspace
    (`VaultReport.coupling_map[key]`). Both default to abstaining.
    Gate 2026-08-22 (evals/probe_graph_variables, ADR-0027): "leg" recovered
    more masked links on a 1199-note vault (+5.7pp recall@10) but pushed every
    golden eligible pair down (-18pp); "boost" kept +2.3/+6.3pp on masked
    links and still lost 3.3/9.1pp on golden pairs; coupling as a leg lost
    25pp. No production caller passes `graph` or `coupling_rank`; the probe
    does, so the negative results stay reproducible.
    """
    # Normalize to the STORE keyspace before anything is excluded. CooccurStore
    # normalizes at its own boundary (note_nodes -> cooccur_key); EmbedStore does
    # not, so a caller handing in a graph-style '<path>.md' built an exclude that
    # never matched any store key and the query note came back as its own closest
    # neighbour, burning a slot in the fusion pool. /map did exactly that.
    blocked = {cooccur_key(x) for x in (exclude or ())} | {cooccur_key(query_path)}
    pool = max(k * 3, _POOL_MIN)

    embed_rank = _embed_ranking(embed_store, query_path, k=pool, exclude=blocked)
    cooc_rank = _cooccur_ranking(
        cooccur_store, query_path, k=pool, exclude=blocked, scope=scope, expand=expand
    )

    mem_embed_rank = None
    if memory_embed_store is not None and embed_store is not None:
        vec = embed_store.get_vec(cooccur_key(query_path))
        mem_embed_rank = _rank_embeddings_from_vec(
            memory_embed_store, vec, k=pool, exclude=set()
        )
    mem_cooc_rank = None
    if memory_cooccur_store is not None and cooccur_store is not None:
        profile = _profile_from_seeds(
            memory_cooccur_store,
            cooccur_store.note_nodes(query_path),
            scope=None,
            expand=expand,
        )
        mem_cooc_rank = _rank_cooccur_from_profile(
            memory_cooccur_store, profile, k=pool, blocked=set(), scope=None
        )
    # CORRELATE third leg: the query's direct note_edges row, ranked by Jaccard.
    # Abstains (None) when the row is empty — 0.57 edges/note means most queries
    # abstain and fusion is identical to before; when it fires, high precision.
    edges_rank = None
    if cooccur_store is not None:
        row = cooccur_store.note_edges_for(query_path)
        ranked = sorted(
            ((p, s) for p, s in row.items() if p not in blocked),
            key=lambda kv: (-kv[1], kv[0]),
        )
        edges_rank = ranked or None
    structural_rank = None
    structural_boost = None
    if graph is not None:
        ranked = _structural_ranking(graph, query_path, k=pool, exclude=blocked)
        if structural_mode == "leg":
            structural_rank = ranked
        elif ranked:
            structural_boost = dict(ranked)
    coupling_leg = None
    if coupling_rank:
        coupling_leg = [(p, w) for p, w in coupling_rank if p not in blocked][:pool] or None
    return _fuse(
        embed_rank,
        cooc_rank,
        k=k,
        edges_rank=edges_rank,
        mem_embed_rank=mem_embed_rank,
        mem_cooc_rank=mem_cooc_rank,
        structural_rank=structural_rank,
        coupling_rank=coupling_leg,
        structural_boost=structural_boost,
    )


def related_notes_for_query(
    *,
    query_vec: list[float] | None = None,
    query_text: str | None = None,
    embed_store: EmbedStore | None = None,
    cooccur_store: CooccurStore | None = None,
    memory_embed_store: EmbedStore | None = None,
    memory_cooccur_store: CooccurStore | None = None,
    k: int = 10,
    scope: str | None = None,
    exclude: set[str] | None = None,
    expand: bool = False,
    recall_rank: list[tuple[str, float]] | None = None,
    lexical_rank: list[tuple[str, float]] | None = None,
) -> list[RelatedNote]:
    """Return the top-k notes related to a FRESH query (not an indexed note).

    The embed leg ranks against `query_vec`; the co-occurrence leg seeds its
    concept profile from `query_text`. This is the fusion path for routing an
    incoming concept (COLLISION) or autolinking a freshly-written note: either
    input may be omitted, and that leg abstains. Returns [] when both abstain.

    `memory_*_store` are the personal-memory lane (ADR-0019); see
    `related_notes`. The memory co-occurrence leg seeds from `query_text`
    using the MEMORY store's frozen language.

    `recall_rank` (phase 1 of `improve`, LoCoMo eval-only): a best-first
    `(key, weight)` list of notes previously confirmed helpful, folded into
    fusion as an extra abstaining leg. None (the default) leaves every caller
    but the LoCoMo runner unaffected.

    `lexical_rank`: the hand-written BM25/fuzzy leg (opt-in via `use_lexical`
    on `facade_retrieve`), folded into fusion as an extra abstaining leg. None
    (the default) leaves every caller but that opt-in path unaffected.
    """
    blocked = set(exclude or ())
    pool = max(k * 3, _POOL_MIN)

    embed_rank = _rank_embeddings_from_vec(embed_store, query_vec, k=pool, exclude=blocked)

    cooc_rank = None
    if cooccur_store is not None and query_text:
        profile = _profile_from_seeds(
            cooccur_store,
            _seed_from_text(query_text, cooccur_store.lang),
            scope=scope,
            expand=expand,
        )
        cooc_rank = _rank_cooccur_from_profile(
            cooccur_store, profile, k=pool, blocked=blocked, scope=scope
        )

    mem_embed_rank = _rank_embeddings_from_vec(
        memory_embed_store, query_vec, k=pool, exclude=set()
    )
    mem_cooc_rank = None
    if memory_cooccur_store is not None and query_text:
        profile = _profile_from_seeds(
            memory_cooccur_store,
            _seed_from_text(query_text, memory_cooccur_store.lang),
            scope=None,
            expand=expand,
        )
        mem_cooc_rank = _rank_cooccur_from_profile(
            memory_cooccur_store, profile, k=pool, blocked=set(), scope=None
        )
    return _fuse(
        embed_rank,
        cooc_rank,
        k=k,
        mem_embed_rank=mem_embed_rank,
        mem_cooc_rank=mem_cooc_rank,
        recall_rank=recall_rank,
        lexical_rank=lexical_rank,
    )

