# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""L3 Analyst Plan — maps VaultReport anomalies to a three-tier task plan.

Translates structural anomalies into actionable TaskCandidates without executing
anything. The three confidence tiers enforce the rule:

  auto      — reversible, graph-safe by construction, unambiguous signal
  propose   — reversible but borderline/opinion-dependent → needs confirmation
  escalate  — irreversible or requires human judgment → IssueCard only

§3.2-bis invariant: capability_name in plan.auto must NEVER be an irreversible
tool (silica_merge, silica_move, silica_delete, etc.). Only silica_autolink
qualifies today — it is graph-safe by construction and fully reversible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx

from silica.kernel.report.graph_report import AutolinkCandidate, VaultReport
from silica.kernel.progress import PlanStep

Tier = Literal["auto", "propose", "escalate"]

# Edge provenance vocabulary (ported from Graphify, MIT, © 2026 Safi Shamsi).
# The tier follows the confidence, so it is never a loose threshold buried in a
# call site — one mapping, one source of truth.
#   EXTRACTED  = corroborated by structure  → auto (reversible, unambiguous)
#   INFERRED   = single-signal / embedding  → propose (confirm before writing)
#   AMBIGUOUS  = conflicting / needs a human → escalate
Confidence = Literal["EXTRACTED", "INFERRED", "AMBIGUOUS"]

def classify_autolink(cand: "AutolinkCandidate") -> Confidence:
    """Confidence of an embedder-free co-occurrence autolink, from its evidence.

    Never EXTRACTED: an autolink is by construction the co-occurrence − wikilink
    delta (>2 hops, unlinked), so the graph does not corroborate it as a link.
    A directly shared concept is textual evidence the pair belongs together
    (INFERRED → propose); a pair related only through transitive/associative
    expansion has no shared concept and needs a human (AMBIGUOUS → escalate).
    """
    return "INFERRED" if cand.shared else "AMBIGUOUS"

# Capability names that are IRREVERSIBLE — never allowed in plan.auto regardless
# of confidence. Expand this set when new write tools are added.
_IRREVERSIBLE = frozenset({
    "silica_move",
    "silica_delete",
    "silica_bulk_write",
    "silica_run_injector",
    "silica_bulk_move",
    "silica_run_organizer",
})

# Threshold for orphan "linkable" heuristic: an orphan goes to `auto` only
# when its title appears as a substring of an existing note name (graph-safe
# title-match heuristic). If no title candidates are known at plan-time we use
# `propose` instead. The actual mention lookup happens in silica_backlink.
_CLUSTER_SIZE_THRESHOLD = 40   # clusters > this → propose audit
_DANGLING_REFS_THRESHOLD = 2   # dangling targets seen >= this many times → escalate


@dataclass
class TaskCandidate:
    capability_name: str   # name of an existing tool in TOOLS
    payload: dict          # args dict for that tool
    reason: str            # human-readable explanation
    tier: Tier             # auto | propose | escalate
    priority: int = 0      # 0 = highest priority
    confidence: Confidence | None = None  # provenance that drove the tier (None = N/A)
    # Escalate-only: the choices the human is actually being offered. None keeps
    # the caller's generic default. Without this every escalate card inherited
    # the dangling-link options (create note / rename existing / ignore), which
    # are meaningless for a stale link or a missing hub.
    options: list[dict] | None = None


@dataclass
class AnalystPlan:
    checkpoints: list[PlanStep]
    auto:     list[TaskCandidate] = field(default_factory=list)
    propose:  list[TaskCandidate] = field(default_factory=list)
    escalate: list[TaskCandidate] = field(default_factory=list)


def build_task_plan(report: VaultReport) -> AnalystPlan:
    """Translate a VaultReport into a three-tier AnalystPlan.

    Rules (applied in priority order, deterministic):

    1. Orphan with linkable title candidates → auto  silica_autolink
       Orphan without candidates            → propose silica_autolink
    2. Missing link (embedding, cosine ≥ τ) → propose silica_autolink on source
    3. Cluster size > threshold             → propose silica_graph_explain (audit)
    4. Dangling wikilink refs ≥ threshold   → escalate (create vs rename decision)
    """
    auto: list[TaskCandidate] = []
    propose: list[TaskCandidate] = []
    escalate: list[TaskCandidate] = []

    # Build a set of existing note paths for title-index heuristic
    # (we use god_node IDs + cluster members as a proxy for "existing titles")
    known_ids: set[str] = set()
    for n in report.god_nodes:
        known_ids.add(n.id.lower())
    for c in report.clusters:
        for m in c.members:
            known_ids.add(m.lower())

    def _has_title_candidate(orphan_path: str) -> bool:
        """Heuristic: orphan stem matches another known note's stem (substring)."""
        stem = orphan_path.rsplit("/", 1)[-1].removesuffix(".md").lower()
        if not stem:
            return False
        orphan_lower = orphan_path.lower()
        for kid in known_ids:
            if kid == orphan_lower:
                continue  # skip the orphan itself
            kid_stem = kid.rsplit("/", 1)[-1].removesuffix(".md").lower()
            if stem in kid_stem or kid_stem in stem:
                return True
        return False

    # Pre-compute cluster assignments for topology-aware chunking
    node_to_cluster: dict[str, int] = {}
    for c in report.clusters:
        for m in c.members:
            node_to_cluster[m] = c.cluster_id

    def _chunk_groups(group_map: dict[int, list[str]], max_bytes: int = 4096) -> list[list[str]]:
        chunks: list[list[str]] = []
        current_chunk: list[str] = []
        current_size = 0
        for _, nodes in group_map.items():
            nodes_size = len(json.dumps(nodes))
            if current_size + nodes_size > max_bytes and current_chunk:
                chunks.append(current_chunk)
                current_chunk = list(nodes)
                current_size = nodes_size
            else:
                current_chunk.extend(nodes)
                current_size += nodes_size
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    # 1. Orphans
    #
    # BACKLINK, not autolink. `report.orphans` is in-degree == 0, so the link has
    # to be injected into the *neighbours*, pointing at the orphan. Autolinking
    # the orphan scans ITS body for mentions of other titles and gives it
    # out-degree, which leaves the in-degree at 0: on a 795-note vault 39 of the
    # 100 orphans already carried outgoing links and were still counted, and the
    # `orphans` term of E(vault) never moved. Same guarantees either way — both
    # only wrap literal mentions of notes that exist, so both stay graph-safe.
    auto_orphans_by_cluster: dict[int, list[str]] = {}
    propose_orphans_by_cluster: dict[int, list[str]] = {}

    for orphan in report.orphans:
        cid = node_to_cluster.get(orphan, -1)
        if _has_title_candidate(orphan):
            auto_orphans_by_cluster.setdefault(cid, []).append(orphan)
        else:
            propose_orphans_by_cluster.setdefault(cid, []).append(orphan)

    cluster_members: dict[int, list[str]] = {c.cluster_id: list(c.members) for c in report.clusters}

    def _backlink_payload(chunk: list[str]) -> dict:
        """Orphan titles to inject, and the cluster peers to scan for a mention.

        Scoped to the orphans' own clusters: that is where prose about them
        already lives, and a vault-wide scan would re-read every note per chunk.
        An orphan with no cluster (-1) contributes no peers, so the pass is a
        no-op for it rather than a full-vault sweep.
        """
        cids = {node_to_cluster.get(o, -1) for o in chunk}
        peers = {m for cid in cids if cid != -1 for m in cluster_members.get(cid, ())}
        return {
            "new_titles": [o.rsplit("/", 1)[-1].removesuffix(".md") for o in chunk],
            # backlink_pass passes self_title, so an orphan left in the
            # neighbourhood can gain a link to a *different* orphan and never
            # to itself.
            "neighbourhood": sorted(peers),
        }

    for chunk in _chunk_groups(auto_orphans_by_cluster):
        auto.append(TaskCandidate(
            capability_name="silica_backlink",
            payload=_backlink_payload(chunk),
            reason=f"{len(chunk)} orphans are named in existing notes → auto-backlink",
            tier="auto",
            priority=0,
        ))

    for chunk in _chunk_groups(propose_orphans_by_cluster):
        propose.append(TaskCandidate(
            capability_name="silica_backlink",
            payload=_backlink_payload(chunk),
            reason=f"{len(chunk)} orphans — no clear title match, confirm before backlinking",
            tier="propose",
            priority=1,
        ))

    # 2. Autolink candidates — the relatedness facade's proposals (ADR-0029),
    #    embedder-free when the embed index is empty. Tier follows evidence via
    #    classify_autolink: a shared concept is INFERRED → propose; an
    #    associative-only pair is AMBIGUOUS → escalate (human review).
    autolink_inferred_by_cluster: dict[int, list[str]] = {}
    seen_inferred: set[str] = set()
    for cand in getattr(report, "autolink_candidates", []):
        if classify_autolink(cand) == "INFERRED":
            if cand.source in seen_inferred:
                continue
            seen_inferred.add(cand.source)
            cid = node_to_cluster.get(cand.source, -1)
            autolink_inferred_by_cluster.setdefault(cid, []).append(cand.source)
        else:
            escalate.append(TaskCandidate(
                capability_name="",  # review-only — a human decides if the association is real
                payload={"source": cand.source, "target": cand.target},
                reason=(
                    f"'{cand.source}' and '{cand.target}' co-occur only associatively "
                    f"(no shared concept) — confirm relevance before linking"
                ),
                tier="escalate",
                confidence="AMBIGUOUS",
                priority=4,
            ))

    for chunk in _chunk_groups(autolink_inferred_by_cluster):
        propose.append(TaskCandidate(
            capability_name="silica_autolink",
            payload={"note_paths": chunk, "use_candidates": True},
            reason=f"Relatedness proposes links for {len(chunk)} source note(s) — confirm before writing",
            tier="propose",
            confidence="INFERRED",
            priority=2,
        ))

    # 2.5 Duplicate pairs (dedup suggestions)
    if hasattr(report, "duplicate_pairs") and report.duplicate_pairs:
        dup_graph = nx.Graph()
        for dp in report.duplicate_pairs:
            dup_graph.add_edge(dp.source, dp.target, score=dp.score)

        dedup_components = list(nx.connected_components(dup_graph))
        component_pairs = []
        for comp in dedup_components:
            comp_pairs = []
            for dp in report.duplicate_pairs:
                if dp.source in comp and dp.target in comp:
                    comp_pairs.append({"source": dp.source, "target": dp.target, "score": dp.score})
            if comp_pairs:
                component_pairs.append(comp_pairs)

        def _chunk_components(components: list[list[dict]], max_bytes: int = 4096) -> list[list[dict]]:
            chunks: list[list[dict]] = []
            current_chunk: list[dict] = []
            current_size = 0
            for comp in components:
                comp_size = len(json.dumps(comp))
                if current_size + comp_size > max_bytes and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = list(comp)
                    current_size = comp_size
                else:
                    current_chunk.extend(comp)
                    current_size += comp_size
            if current_chunk:
                chunks.append(current_chunk)
            return chunks

        for pair_chunk in _chunk_components(component_pairs):
            propose.append(TaskCandidate(
                capability_name="silica_dedup_pairs",
                payload={"pairs": pair_chunk},
                reason=f"Embedding proposes {len(pair_chunk)} duplicate pairs for merge — confirm before executing",
                tier="propose",
                priority=2,
            ))

    # 3. Refiner & Enricher (from OFM triage) → propose
    if getattr(report, "lean_notes", []):
        lean_chunks = _chunk_groups({1: report.lean_notes})
        for chunk in lean_chunks:
            propose.append(TaskCandidate(
                capability_name="silica_enrich_batch",
                payload={"note_paths": chunk},
                reason=f"Identified {len(chunk)} lean or empty note(s) → propose semantic enrichment",
                tier="propose",
                priority=3,
            ))
            
    if getattr(report, "reformat_notes", []):
        ref_chunks = _chunk_groups({1: report.reformat_notes})
        for chunk in ref_chunks:
            propose.append(TaskCandidate(
                capability_name="silica_refine_batch",
                payload={"note_paths": chunk},
                reason=f"Identified {len(chunk)} note(s) with missing or invalid tags → propose stylistic refinement",
                tier="propose",
                priority=3,
            ))

    # 4. Oversized clusters → propose a read-only audit
    for c in report.clusters:
        if c.size > _CLUSTER_SIZE_THRESHOLD and c.hub:
            propose.append(TaskCandidate(
                capability_name="silica_graph_explain",
                payload={"note": c.hub, "depth": 1},
                reason=(
                    f"Cluster {c.cluster_id} is large (size={c.size} > {_CLUSTER_SIZE_THRESHOLD}) "
                    f"— audit hub '{c.hub}' to decide if refactoring is needed"
                ),
                tier="propose",
                priority=4,
            ))

    # 5. Recurring dangling wikilinks → escalate (create vs rename is irreversible)
    for d in report.dangling:
        if d["refs"] >= _DANGLING_REFS_THRESHOLD:
            escalate.append(TaskCandidate(
                capability_name="",  # no automatic capability — human decides
                payload={"target": d["target"], "refs": d["refs"]},
                reason=(
                    f"Unresolved wikilink '{d['target']}' appears {d['refs']} time(s) "
                    f"— decide: create note, rename existing, or ignore"
                ),
                tier="escalate",
                priority=5,
            ))

    # 6. Stale links → escalate. The contrary reading of AUTOLINK: the human
    #    linked two notes whose text shares no concept at all. Either the link is
    #    wrong or a note drifted out from under it, and both readings are the
    #    human's call — deleting someone's wikilink is never an `auto`.
    #    Unlike an unlinked pair (humans under-link, so AUTOLINK's misses are
    #    cheap), a stale link is a labelled negative: the human asserted a
    #    relation the text no longer supports.
    for sl in getattr(report, "stale_links", []):
        escalate.append(TaskCandidate(
            capability_name="",  # no automatic capability — removing a link is destructive
            payload={"source": sl.source, "target": sl.target},
            reason=(
                f"'{sl.source}' links '{sl.target}' but the two share no concept in text "
                f"— decide: the link is stale, or a note drifted"
            ),
            tier="escalate",
            confidence="AMBIGUOUS",
            priority=5,
            options=[
                {"label": "keep_link", "description": "The relation is real; the text just does not name it"},
                {"label": "remove_link", "description": "The link no longer holds — unlink the two notes"},
                {"label": "enrich_source", "description": "The relation is real but the note drifted — re-enrich it"},
            ],
        ))

    # 7. Missing hubs → escalate. The only GENERATIVE signal in the report: a
    #    concept central in the discourse that no note is titled after. Creating
    #    a note is irreversible-ish and needs content, so it never goes in auto.
    for mh in getattr(report, "missing_hubs", []):
        escalate.append(TaskCandidate(
            capability_name="",  # human decides; creation runs through the normal write path
            payload={"concept": mh.concept, "centrality": mh.centrality},
            reason=(
                f"'{mh.concept}' is central in the discourse (centrality={mh.centrality}) "
                f"but no note is titled after it — decide: create the hub note, or ignore"
            ),
            tier="escalate",
            priority=6,
            options=[
                {"label": "create_hub", "description": f"Write a hub note for '{mh.concept}'"},
                {"label": "alias_existing", "description": "An existing note already covers it — add the alias"},
                {"label": "ignore", "description": "Not a concept worth its own note"},
            ],
        ))

    # §3.2-bis safety check: strip any irreversible capability that leaked into auto
    auto = [c for c in auto if c.capability_name not in _IRREVERSIBLE]

    # Sort by priority within each tier
    auto.sort(key=lambda c: (c.priority, c.reason))
    propose.sort(key=lambda c: (c.priority, c.reason))
    escalate.sort(key=lambda c: (c.priority, c.reason))

    checkpoints = [
        PlanStep(id="audit",    kind="mechanical", objective="silica_vault_report"),
        PlanStep(id="remediate", kind="gate",      objective="silica_autolink"),
        PlanStep(id="report",   kind="mechanical", objective="silica_ledger_digest"),
    ]

    return AnalystPlan(
        checkpoints=checkpoints,
        auto=auto,
        propose=propose,
        escalate=escalate,
    )
