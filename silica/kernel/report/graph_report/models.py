# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Dataclasses for the L1 graph report.

Pure data containers — no I/O, no graph computation. Authoritative
structures (NodeStat … VaultReport) and PROPOSED-signal records
(DuplicatePair, AutolinkCandidate, StaleLink, MissingHub,
AttentionCandidate) live together because they all describe one VaultReport payload.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeStat:
    id: str           # vault-relative path (no .md)
    label: str        # display name
    cluster: int      # node["group"], -1 if none
    out_degree: int
    in_degree: int
    degree: int       # out+in
    # No per-node pagerank: nothing read it (the markdown dropped the column, it
    # reads 0.0 at vault scale) and VaultReport.pagerank_map already carries the
    # value for every node, god-nodes included.
    betweenness: float = 0.0  # fraction of shortest paths through the node — bottleneck signal, distinct from degree


@dataclass
class BridgeStat:
    source: str
    target: str
    source_cluster: int
    target_cluster: int
    weight: float     # surprise score: (deg(u)+deg(v)) / (1 + shared_neighbors)


@dataclass
class StructuralGap:        # mirror of BridgeStat — two well-formed areas with few/no links between them
    cluster_a: int
    cluster_b: int
    hub_a: str             # highest-degree node of cluster_a (overlay endpoint)
    hub_b: str             # highest-degree node of cluster_b
    inter_edges: int       # EXTRACTED edges joining the two clusters (0 = a full structural hole)
    gap_score: float       # size_a * size_b / (1 + inter_edges) — big disconnected areas rank highest (report ranking)
    gap_density: float = 0.0  # 1 - inter/(size_a*size_b): absent-link fraction ∈ [0,1); bounded term E(vault) sums


@dataclass
class ClusterStat:
    cluster_id: int
    size: int
    hub: str | None        # highest-degree node in cluster
    members: list[str]     # capped at 25 in markdown, full in JSON
    cohesion: float        # intra-cluster edges / C(size,2)


@dataclass
class DuplicatePair:        # PROPOSED — cosine-close pair (band depends on which list it lands in)
    source: str
    target: str
    score: float


# --- Co-occurrence vs wikilink delta (PROPOSED, embedder-free) -------------

@dataclass
class AutolinkCandidate:    # relatedness − wikilink: related in text, unlinked
    source: str
    target: str
    weight: float          # Jaccard (direct) or RRF score (fused); higher = stronger, scales differ
    shared: list[str]      # directly shared concept labels (evidence)
    convergence: int = 0   # #8: number of god-node hubs this pair connects to
    provenance: str = "fused"  # "direct" (CORRELATE Jaccard edge) | "fused" (relatedness facade, ADR-0029)


@dataclass
class StaleLink:           # wikilink − co-occurrence: linked, no textual co-presence
    source: str
    target: str


@dataclass
class MissingHub:          # central concept in the discourse with no hub note
    concept: str           # surface label of the concept
    centrality: float      # weighted degree in the co-occurrence graph


@dataclass
class IntegrationDeficit:  # PROPOSED — concept-rich note, weakly wikilinked
    path: str              # node id (store key, no .md)
    concepts: int          # distinct concepts contributed to the co-occurrence graph
    degree: int            # wikilink degree (structural integration)
    score: float           # concepts / (1 + degree) — higher = richer text, fewer links


@dataclass
class AttentionCandidate:   # PROPOSED — spaced-repetition: recall misses × idle-time × weak-linkage
    path: str              # node id
    days_idle: int         # days since the last graded quiz answer, else since file mtime (see compute.py)
    degree: int            # wikilink degree: integration proxy standing in for a per-note "confidence"
    score: float           # (days_idle+1)(1+misses) / ((1+degree)(1+correct)) — higher = more neglected
    misses: int = 0        # graded quiz answers this note got WRONG (0 = never quizzed, or never missed)
    attempts: int = 0      # graded quiz answers total; 0 means the recall signal is absent, not clean


@dataclass
class ContestedNote:       # AUTHORITATIVE — frontmatter `contested: true`
    path: str              # node id
    refs: list[str]        # `contradictions:` entries (sources / notes in conflict)


@dataclass
class SourceDrift:         # AUTHORITATIVE — derived from <vault>/provenance.json
    note: str               # node id, derived from a superseded source version
    source: str              # source basename whose version moved on without this note


@dataclass
class TemporalStat:        # AUTHORITATIVE — read from note text (frontmatter + claim stamps)
    """What the bi-temporal layer says about the vault, as counts.

    Every field already existed on disk (reliability tiers, `## Superseded`,
    `superseded_by`, `<!-- silica: valid_from=... -->`) and nothing read it.
    Deliberately NOT a term of E(vault): E is pinned by a frozen perturbation
    bench, and "the vault records its own history" is not an entropic cost.
    """
    notes_scanned: int
    by_tier: dict[int, int]        # reliability_tier -> count (3 human, 2 grounded, 1 distilled)
    superseded_sections: int       # notes carrying a `## Superseded` graveyard
    superseded_notes: int          # notes pointed at a winner via `superseded_by`
    stamped: int                   # notes carrying >= 1 claim stamp
    oldest_valid_from: str = ""    # earliest valid_from seen ("" when none)


@dataclass
class CodeCoverage:        # AUTHORITATIVE — derived codegraph vs documents: frontmatter
    documented: int        # supported files documented by at least one note
    total: int             # supported files in the codegraph index
    undocumented: list[list] = field(default_factory=list)  # [path, fan_in], fan-in desc


@dataclass
class StructuralLink:       # PROPOSED — Adamic-Adar over the wikilink graph (V1)
    source: str             # graph node id
    target: str
    score: float            # sum over common neighbours of 1/log(deg)
    common: list[str]       # the common neighbours (evidence)
    shared: list[str] = field(default_factory=list)  # shared concept labels, when the cooccur store is present


@dataclass
class PrerequisiteEdge:     # PROPOSED — RefD direction (V2); store keyspace (no .md)
    prereq: str
    dependent: str
    refd: float             # in (0, 1]: share of the dependent's related notes citing the prereq, net


@dataclass
class CoupledPair:          # PROPOSED — shared transactions (sources cited, runs written) (V3)
    source: str             # graph node id
    target: str
    score: float            # sum over shared transactions of 1/log(1+size)
    shared: list[str] = field(default_factory=list)  # shared concept labels, when the cooccur store is present


@dataclass
class LoadBearingNote:      # analytics — the notes the graph stands on (V4)
    path: str
    degree: int
    betweenness: float
    coreness: int           # k-core number
    articulation: bool      # removal disconnects the graph
    surprise: float         # pct-rank(betweenness) - pct-rank(degree), in [-1, 1]


@dataclass
class MisfiledNote:         # PROPOSED — linked like one area, reads like another (V5, ADR-0023)
    path: str
    degree: int
    dissonance: float       # share of wikilink neighbours in a different semantic zone


@dataclass
class BurstingConcept:      # analytics — over-represented in the recent writing window (V6)
    concept: str
    z: float
    recent: int             # notes in the window carrying it
    total: int              # notes overall carrying it


@dataclass
class SprawlingNote:        # PROPOSED — broad AND flat concept distribution: split candidate (V7)
    path: str               # store keyspace (no .md)
    concepts: int
    entropy: float          # bits
    flatness: float         # entropy / log2(concepts), in [0, 1]


@dataclass
class VaultReport:
    generated_at: str
    scope: str
    totals: dict[str, int]
    god_nodes: list[NodeStat]
    bridges: list[BridgeStat]
    orphans: list[str]
    dangling: list[dict]   # [{"target": str, "refs": int}]
    clusters: list[ClusterStat]
    duplicate_pairs: list[DuplicatePair] = field(default_factory=list)          # borderline band (τ_low..τ_high): link, don't merge
    confirmed_duplicate_pairs: list[DuplicatePair] = field(default_factory=list)  # ≥ τ_high: likely true duplicates (merge candidates)
    autolink_candidates: list[AutolinkCandidate] = field(default_factory=list)
    stale_links: list[StaleLink] = field(default_factory=list)
    missing_hubs: list[MissingHub] = field(default_factory=list)
    integration_deficits: list[IntegrationDeficit] = field(default_factory=list)
    attention_candidates: list[AttentionCandidate] = field(default_factory=list)
    lean_notes: list[str] = field(default_factory=list)
    # Body length of each lean note, from the scan that already held the text to
    # decide it was lean. A separate map rather than a field on the row because
    # `lean_notes` is a list of ids two other callers iterate as ids
    # (analyst_plan's chunker and the curator's refinement bucket), and widening
    # it to carry a second value would rewrite both to read what neither wants.
    lean_chars: dict[str, int] = field(default_factory=dict)
    reformat_notes: list[str] = field(default_factory=list)
    contested: list[ContestedNote] = field(default_factory=list)
    source_drift: list[SourceDrift] = field(default_factory=list)
    structural_gaps: list[StructuralGap] = field(default_factory=list)
    # The area x area coupling tally, keyed "a|b" with a <= b: linked note PAIRS
    # between two multi-note areas, and inside one where a == b. One field and
    # not two because the diagonal is the same measurement as the off-diagonal
    # on the same currency -- cohesion is this number over C(size,2) -- and two
    # fields would let a caller mix a deduped count with an edge count.
    # structural_gaps is a top_k ranking of the emptiest pairs, which cannot
    # show what a hole is a hole against: an absence only reads against the
    # pairs that are present. Analytics-only, and empty rather than zero-filled
    # at structural depth, so a caller can tell "not measured" from "no links".
    inter_cluster: dict[str, int] = field(default_factory=dict)
    discourse_state: str = ""     # "Focused" | "Diversified" | "Fragmented" | "" — topology one-word diagnosis
    pagerank_map: dict[str, float] = field(default_factory=dict)  # all nodes: vault-relative path (no .md) → pagerank
    betweenness_map: dict[str, float] = field(default_factory=dict)  # all nodes → betweenness (analytics-only; zero-filled otherwise)
    # all nodes → resolved-link degree (in+out). Unlike the two maps above this
    # one is populated at EVERY depth: degree falls out of the structural core
    # the cheap nucleate path already computes, so it costs a dict comprehension.
    # god_nodes only carries the top_k; the whole map is what a distribution needs.
    degree_map: dict[str, int] = field(default_factory=dict)
    # Seven inter-note variables (docs/superpowers/specs/2026-08-22-graph-variables-design.md).
    # core_map and articulation are populated at every depth (O(V+E), the
    # structural core can afford them); the rows are analytics-only and the
    # store-derived ones (prerequisites, sprawling, bursting, shared labels)
    # need with_cooccurrence; misfiled/dissonance need with_embeddings.
    structural_links: list[StructuralLink] = field(default_factory=list)
    prerequisites: list[PrerequisiteEdge] = field(default_factory=list)
    prereq_map: dict[str, list[str]] = field(default_factory=dict)   # dependent -> prereqs, store keyspace
    coupled_pairs: list[CoupledPair] = field(default_factory=list)
    coupling_map: dict[str, dict[str, float]] = field(default_factory=dict)  # store keyspace; the relatedness leg reads it
    load_bearing: list[LoadBearingNote] = field(default_factory=list)
    core_map: dict[str, int] = field(default_factory=dict)           # all nodes -> k-core number
    articulation: list[str] = field(default_factory=list)            # sorted cut vertices
    misfiled: list[MisfiledNote] = field(default_factory=list)
    dissonance_map: dict[str, float] = field(default_factory=dict)
    bursting_concepts: list[BurstingConcept] = field(default_factory=list)
    sprawling: list[SprawlingNote] = field(default_factory=list)
    code_coverage: CodeCoverage | None = None
    temporal: TemporalStat | None = None   # analytics-only; None when the body scan did not run
