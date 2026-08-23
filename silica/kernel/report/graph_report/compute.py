# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Core deterministic computation of the VaultReport.

Builds degree/PageRank/Louvain/bridge/orphan/dangling stats from the
driver's wikilink graph, then attaches the optional PROPOSED signal
sections computed by embed_signals and cooccur_delta.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from silica.kernel.report.graph_report.cooccur_delta import (
    _compute_cooccur_delta,
    _compute_cooccur_variables,
)
from silica.kernel.report.graph_report.embed_signals import (
    _compute_dissonance,
    _compute_duplicate_pairs,
)
from silica.kernel.report.graph_report.models import (
    AttentionCandidate,
    BridgeStat,
    ClusterStat,
    ContestedNote,
    CoupledPair,
    LoadBearingNote,
    NodeStat,
    SourceDrift,
    StructuralGap,
    StructuralLink,
    TemporalStat,
    VaultReport,
)

logger = logging.getLogger(__name__)

# Epoch-keyed memo for compute_report — see the comment inside it. The epoch
# lives INSIDE the key: freshness checked against a separate global raced under
# the MCP server's one-thread-per-request dispatch (a slow analytics pass could
# store its stale report after a newer epoch had cleared the memo). A key that
# names its own epoch cannot be served stale; entries from older epochs are
# swept on the next miss so they cannot pile up.
_report_memo: dict[tuple, VaultReport] = {}


def _is_staging(path: str) -> bool:
    """True for Silica's own staging paths: the inbox and the `done/` archive.

    Neither is knowledge — the inbox is material awaiting distillation, `done/`
    is what has already been distilled — so nothing links to them by design and
    calling them orphaned reports Silica's bookkeeping as vault damage.
    """
    from silica.kernel.recall.paths import is_inbox_path
    from silica.kernel.vault_manifest import active_done_dir, in_write_dir

    from silica.kernel.recall.run_log import DEFAULT_LOG_FILENAME

    p = (path or "").replace("\\", "/").lstrip("/")
    if is_inbox_path(p):
        return True
    if p.casefold() == in_write_dir(DEFAULT_LOG_FILENAME).casefold():
        return True  # the run journal, same reason
    done = active_done_dir().casefold()
    return bool(done) and p.casefold().startswith(done + "/")


def _index_stores_sig(with_embeddings: bool, with_cooccurrence: bool,
                      analytics: bool = False) -> tuple:
    """(mtime_ns, size) of the index-side stores these flags pull in.

    They live in ~/.silica/index/, outside the vault walk `vault_epoch`
    hashes, so a re-embed, a co-occurrence refresh, or a graded quiz answer
    changes the report without touching any note. The quiz log rides the
    analytics flag because only the attention section reads it. Absent file
    or failed stat -> None slot: still a stable key, and it moves the moment
    the file appears.
    """
    files = []
    if with_embeddings:
        from silica.kernel.recall.embed import _index_path
        files.append(_index_path())
    if with_cooccurrence:
        from silica.kernel.recall.cooccurrence import _index_path as _cooccur_path
        files.append(_cooccur_path())
    if analytics:
        from silica.kernel.report.quiz import log_path
        files.append(log_path())
    sig: list[tuple[int, int] | None] = []
    for p in files:
        try:
            st = p.stat()
            sig.append((st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append(None)  # a vanished file is its own signature
    return tuple(sig)


def _note_mtimes(real_ids: set[str], override: dict[str, float] | None) -> dict[str, float]:
    """{node id: mtime} via the driver, or the injected override (tests)."""
    if override is not None:
        return dict(override)
    mtimes: dict[str, float] = {}
    from silica.driver import DRIVER

    mtime_of = getattr(DRIVER, "mtime_of", None)
    if mtime_of is None:
        return mtimes
    for nid in real_ids:
        try:
            ts = mtime_of(nid)
        except Exception:
            ts = None
        if ts is not None:
            mtimes[nid] = ts
    return mtimes


def compute_report(
    folder: str = "",
    *,
    top_k: int = 10,
    analytics: bool = False,
    with_embeddings: bool = False,
    with_cooccurrence: bool = False,
    _nodes_edges_override: tuple[list[dict], list[dict]] | None = None,
    _cooccur_store_override: Any | None = None,
    _mtimes_override: dict[str, float] | None = None,
    _quiz_override: dict[str, dict] | None = None,
    _transactions_override: list[set[str]] | None = None,
    _dissonance_knn_k: int = 6,
) -> VaultReport:
    """Build a VaultReport from the driver's wikilink graph.

    Uses build_graph_data + detect_communities from graph_export, then
    computes degree, Louvain clusters, orphans, and dangling links from the
    resolved (EXTRACTED) edge set only — the cheap *structural core* nucleate
    reads (cluster routing + orphan repair).

    `analytics=True` additionally computes the expensive read-only signals that
    only the on-demand /graph and /report commands consume: PageRank, god-nodes,
    cross-cluster bridges, and per-cluster cohesion. Nucleate leaves it False to
    skip the 200-iteration PageRank and the bridge/cohesion edge traversals.

    Pass _nodes_edges_override for testing without a live driver.
    """
    import networkx as nx
    from silica.kernel.recall.graph_export import (
        build_graph_data,
        detect_communities,
        edge_graph,
        structural_gaps,
    )

    # Memoized on the vault's file-state epoch: silica_graph_explain calls this
    # with analytics=True on EVERY invocation, and a full analytics pass reads
    # every body + runs PageRank and betweenness — seconds on a real vault, to
    # answer a question about one note. Any file change bumps the epoch; the
    # override seams (tests, custom feeds) bypass the memo entirely.
    # The quiz log lives outside the epoch walk, so its stat rides the key
    # (analytics only — nothing else reads it). Every other analytics input,
    # the contested/temporal scan included, is note text the epoch hashes.
    overrides = (_nodes_edges_override, _cooccur_store_override,
                 _mtimes_override, _quiz_override, _transactions_override)
    memo_key = None
    if all(o is None for o in overrides):
        from silica.kernel.recall.paths import vault_epoch

        if epoch := vault_epoch():
            memo_key = (epoch,
                        _index_stores_sig(with_embeddings, with_cooccurrence,
                                          analytics),
                        folder, top_k, analytics, with_embeddings,
                        with_cooccurrence, _dissonance_knn_k)
            if (hit := _report_memo.get(memo_key)) is not None:
                return hit
            # Miss: sweep entries from other epochs. A concurrent slow pass
            # may still store under its own (older) epoch key afterwards —
            # unreadable by construction, swept on the next miss.
            for k in [k for k in _report_memo if k[0] != epoch]:
                del _report_memo[k]

    if _nodes_edges_override is not None:
        nodes, edges = _nodes_edges_override
        detect_communities(nodes, edges)
    else:
        try:
            nodes, edges = build_graph_data(folder=folder)
            detect_communities(nodes, edges)
        except Exception as exc:
            logger.warning("graph_report: build_graph_data failed (%s) — returning empty report", exc)
            return _empty_report(folder)

    # Split real nodes from ghost nodes
    real_nodes = [n for n in nodes if n.get("type") != "ghost"]
    real_ids: set[str] = {n["id"] for n in real_nodes}

    # One build over EXTRACTED edges only (authoritative); the undirected view
    # everything but in/out-degree wants is a projection of the same edges.
    G_dir = edge_graph(nodes, edges, directed=True)
    G_und = G_dir.to_undirected()

    # Degree maps
    out_deg: dict[str, int] = dict(G_dir.out_degree())
    in_deg: dict[str, int] = dict(G_dir.in_degree())
    deg: dict[str, int] = {n: out_deg.get(n, 0) + in_deg.get(n, 0) for n in real_ids}

    # Triage for stylistic refinement and enrichment — analytics-only. It reads
    # EVERY note body (the dominant report cost on a large vault) and its output
    # (lean_notes/reformat_notes) is consumed only by build_task_plan + render on
    # the /graph,/report path. Nucleate never reads it, so the structural core skips
    # the per-note read entirely.
    lean_notes: list[str] = []
    lean_chars: dict[str, int] = {}
    reformat_notes: list[str] = []
    contested: list[ContestedNote] = []
    # Bi-temporal counters ride this same loop: reliability tiers, `## Superseded`
    # graveyards, `superseded_by` pointers and claim stamps are all note text the
    # scan is already holding, so reading the temporal layer costs no extra I/O.
    temporal: TemporalStat | None = None
    # Read once, before the body scan: the attention ranking needs mtimes, and
    # the burst (V6) needs creation dates that fall back to the same mtime.
    mtimes: dict[str, float] = _note_mtimes(real_ids, _mtimes_override) if analytics else {}
    # Frontmatter `sources:` per note (V3 transactions) and creation timestamp
    # per note (V6 window), both harvested from the scan already holding the text.
    sources_of: dict[str, list[str]] = {}
    created_ts: dict[str, float] = {}
    if analytics:
        try:
            from silica.kernel.write import contested as contested_kernel
            from silica.kernel.link import ofm
            from silica.kernel.write import frontmatter
            from silica.kernel.report.learner import _created_and_ai
            from silica.driver import DRIVER

            tiers: Counter = Counter()
            scanned = superseded_sections = superseded_notes = stamped = 0
            valid_froms: list[str] = []

            for nid in real_ids:
                try:
                    nc = DRIVER.read_note(nid)
                    if not nc.content:
                        continue
                    data, _, body = frontmatter.split(nc.content)
                    if data and data.get("contested"):
                        contested.append(
                            ContestedNote(path=nid, refs=list(data.get("contradictions") or []))
                        )

                    scanned += 1
                    if data:
                        srcs = data.get("sources")
                        if isinstance(srcs, list):
                            sources_of[nid] = [str(x) for x in srcs if isinstance(x, (str, int))]
                    mt = mtimes.get(nid)
                    ts_created, _ai = _created_and_ai(nc.content, mt if mt is not None else 0.0)
                    if mt is not None or ts_created != 0.0:
                        created_ts[nid] = ts_created
                    tiers[contested_kernel.reliability_tier(nc.content)] += 1
                    if contested_kernel.SUPERSEDED_HEADING in body:
                        superseded_sections += 1
                    if data and data.get(contested_kernel.SUPERSEDED_BY_KEY):
                        superseded_notes += 1
                    stamps = contested_kernel.parse_stamps(body)
                    if stamps:
                        stamped += 1
                    valid_froms.extend(s["valid_from"] for s in stamps if s.get("valid_from"))

                    body_chars = len(body.strip())
                    is_empty = body_chars == 0
                    is_lean = ofm.is_lean(body)
                    if is_empty or is_lean:
                        lean_notes.append(nid)
                        # The same figure is_lean compared against its limit,
                        # kept rather than recomputed: a second read of the note
                        # would be I/O for a number this loop is already holding,
                        # and a second measurement is a second thing to disagree.
                        lean_chars[nid] = body_chars
                    elif data is None or frontmatter.lint_tags(data):
                        reformat_notes.append(nid)
                except Exception as exc:
                    # Silent here made a systematic regression render as a clean
                    # vault: `scanned` is already incremented, so the note lands
                    # in notes_scanned while contributing to no counter.
                    logger.debug("graph_report: triage skipped '%s' — %s", nid, exc)

            temporal = TemporalStat(
                notes_scanned=scanned,
                by_tier=dict(sorted(tiers.items(), reverse=True)),
                superseded_sections=superseded_sections,
                superseded_notes=superseded_notes,
                stamped=stamped,
                # Stamps carry ISO dates, which sort lexicographically — min() is
                # the earliest without parsing anything.
                oldest_valid_from=min(valid_froms) if valid_froms else "",
            )
        except Exception as exc:
            logger.warning("graph_report: triage failed — %s", exc)

    # Source drift (spec-hermes-coherence §3) — analytics-only for parity with
    # the other on-demand /report signals above, though the read itself is
    # cheap (one <vault>/provenance.json parse, no per-note driver reads).
    source_drift: list[SourceDrift] = []
    if analytics:
        try:
            from silica.kernel.write.provenance import drifted_notes

            # Provenance notes are recorded WITHOUT the `.md` extension
            # (RunManifestEntry.path strips it), but graph node ids (real_ids)
            # carry `.md` (driver index keys) — strip at the seam before
            # intersecting, per codebase convention.
            real_stems = {i.removesuffix(".md") for i in real_ids}
            source_drift = [
                SourceDrift(note=note, source=source)
                for note, source in drifted_notes()
                if note in real_stems
            ]
        except Exception as exc:
            logger.warning("graph_report: source drift check failed — %s", exc)

    # PageRank — analytics-only (200-iteration power method); the structural
    # core leaves it empty so god-node tiebreaks and pagerank_map are all-zero.
    pr: dict[str, float] = {}
    if analytics:
        try:
            pr = _pagerank(G_und, max_iter=200) if G_und.number_of_edges() > 0 else {}
        except Exception:
            pr = {}

    # Betweenness — analytics-only (O(V·E), the most expensive metric). Sampled
    # at k<=400 pivots so it stays bounded on big vaults; k==n (small vaults) is
    # exact. seed fixed for deterministic output. Distinct from degree: it flags
    # bottleneck nodes whose removal fragments the discourse.
    # k-sampled approximation, decided: exact betweenness buys only the last
    # digit at a cost no report reader has asked for.
    bet: dict[str, float] = {}
    if analytics and G_und.number_of_edges() > 0:
        try:
            bet = nx.betweenness_centrality(
                G_und, k=min(G_und.number_of_nodes(), 400), seed=42, normalized=True
            )
        except Exception:
            bet = {}

    # ------------------------------------------------------------------
    # Attention candidates — analytics-only. Spaced-repetition
    # surfacing, embedder-free: a note the reader gets WRONG, or leaves
    # untouched while weakly linked, floats up.
    #   score = (days_idle + 1)(1 + misses) / ((1 + degree)(1 + correct))
    # Pure ranking, no weights, no config. With no quiz history the two quiz
    # terms are 1 and the score is bit-identical to the mtime-only ranking it
    # replaces; degree still stands in for a per-note "confidence" until a
    # graded answer measures the reader's instead of guessing from structure.
    #
    # The recall log also supplies the review date mtime could not: mtime is
    # "last touch by ANYONE" (a bulk autolink resets it), while a graded answer
    # is the reader, on a date, on this note. Quizzed notes date from that;
    # the rest keep the mtime proxy and its ceiling.
    attention: list[AttentionCandidate] = []
    if analytics:
        quiz_stats = _quiz_override
        if quiz_stats is None:
            from silica.kernel.report import quiz as _quiz

            try:
                quiz_stats = _quiz.stats()
            except Exception as exc:  # a broken log must not sink the report
                logger.warning("graph_report: quiz log unreadable (%s)", exc)
                quiz_stats = {}
        if mtimes or quiz_stats:
            from silica.kernel.report.quiz import key as _quiz_key

            now_ts = datetime.now(timezone.utc).timestamp()
            for nid in real_ids:
                q = quiz_stats.get(_quiz_key(nid)) or {}
                ts = mtimes.get(nid)
                if q.get("last"):
                    try:
                        ts = datetime.fromisoformat(q["last"]).timestamp()
                    except ValueError:
                        pass  # unparseable stamp: fall back to mtime
                if ts is None:
                    continue  # abstain: no recency signal for this note
                days_idle = max(0, int((now_ts - ts) // 86400))
                d = deg.get(nid, 0)
                misses, correct = int(q.get("misses", 0)), int(q.get("correct", 0))
                attention.append(AttentionCandidate(
                    path=nid, days_idle=days_idle, degree=d,
                    misses=misses, attempts=misses + correct,
                    score=round((days_idle + 1) * (1 + misses) / ((1 + d) * (1 + correct)), 3),
                ))
            # Two tiers, because idle-days are unbounded and would otherwise
            # drown the measurement: a note failed more often than recalled
            # outranks ANY note whose weakness is only guessed at from file age.
            # Recall enough times and it retires to the proxy tier on its own.
            attention.sort(key=lambda a: (0 if 2 * a.misses > a.attempts else 1, -a.score, a.path))
            attention = attention[:top_k]

    # Cluster map from detect_communities output
    cluster_map: dict[str, int] = {n["id"]: n.get("group", -1) for n in real_nodes}
    node_label: dict[str, str] = {n["id"]: n.get("label", n["id"]) for n in real_nodes}

    # ------------------------------------------------------------------
    # God nodes + cross-cluster bridges — analytics-only (read by /graph,
    # /report; nucleate never touches them). Skipped for the structural core.
    # ------------------------------------------------------------------
    god_nodes: list[NodeStat] = []
    bridges: list[BridgeStat] = []
    if analytics:
        sorted_nodes = sorted(
            real_ids,
            key=lambda n: (-deg.get(n, 0), -pr.get(n, 0.0), n),
        )
        for nid in sorted_nodes[:top_k]:
            god_nodes.append(NodeStat(
                id=nid,
                label=node_label.get(nid, nid),
                cluster=cluster_map.get(nid, -1),
                out_degree=out_deg.get(nid, 0),
                in_degree=in_deg.get(nid, 0),
                degree=deg.get(nid, 0),
                betweenness=round(bet.get(nid, 0.0), 4),
            ))

        seen_bridge: set[tuple[str, str]] = set()
        for u, v in G_und.edges():
            cu, cv = cluster_map.get(u, -1), cluster_map.get(v, -1)
            if cu < 0 or cv < 0 or cu == cv:
                continue
            shared = len(list(nx.common_neighbors(G_und, u, v)))
            weight = (deg.get(u, 0) + deg.get(v, 0)) / (1 + shared)
            key = (min(u, v), max(u, v))
            if key not in seen_bridge:
                seen_bridge.add(key)
                bridges.append(BridgeStat(
                    source=u, target=v,
                    source_cluster=cu, target_cluster=cv,
                    weight=round(weight, 4),
                ))
        bridges.sort(key=lambda b: (-b.weight, b.source, b.target))
        bridges = bridges[:top_k]

    # ------------------------------------------------------------------
    # Load-bearing notes (V4). Coreness and cut vertices cost O(V+E), so the
    # structural core carries them (nucleate's hub choice can read coreness);
    # surprise needs betweenness and the ranked rows are analytics-only.
    # ------------------------------------------------------------------
    from silica.kernel.recall import signals as _signals

    core_map: dict[str, int] = {}
    articulation_set: set[str] = set()
    surprise: dict[str, float] = {}
    try:
        core_map, articulation_set, surprise = _signals.load_bearing(
            G_und, betweenness=bet, degree=deg,
        )
    except Exception as exc:
        logger.warning("graph_report: load-bearing signals failed — %s", exc)
    load_bearing: list[LoadBearingNote] = []
    structural_links: list[StructuralLink] = []
    coupled_pairs: list[CoupledPair] = []
    coupling_map: dict[str, dict[str, float]] = {}
    if analytics:
        ranked_lb = sorted(
            (n for n in real_ids if deg.get(n, 0) > 0),
            key=lambda n: (n not in articulation_set, -surprise.get(n, 0.0), n),
        )
        load_bearing = [
            LoadBearingNote(
                path=n, degree=deg.get(n, 0), betweenness=round(bet.get(n, 0.0), 4),
                coreness=core_map.get(n, 0), articulation=n in articulation_set,
                surprise=surprise.get(n, 0.0),
            )
            for n in ranked_lb[:top_k]
        ]
        # V1: unlinked pairs that share neighbours, Adamic-Adar. Disjoint from
        # AUTOLINK by construction (that list keeps only pairs >2 hops apart).
        try:
            structural_links = [
                StructuralLink(source=u, target=v, score=round(sc, 4), common=cm)
                for u, v, sc, cm in _signals.structural_links(G_und, top_k=top_k)
            ]
        except Exception as exc:
            logger.warning("graph_report: structural links failed — %s", exc)
        # V3: transactions = frontmatter `sources:` (one per cited source) plus
        # the notes each run wrote together (manifest, scoped to this vault).
        try:
            from silica.kernel.recall.cooccurrence import cooccur_key as _ck

            if _transactions_override is not None:
                transactions = [set(t) for t in _transactions_override]
                dropped_runs = 0
            else:
                from silica.config import CONFIG as _CFG
                from silica.kernel.report.cowrite import coupling_transactions

                # One assembly, shared with the graph viewer's COUPLED layer:
                # two copies of "what counts as written together" would drift
                # the moment either learned about a third kind of transaction.
                transactions, dropped_runs = coupling_transactions(
                    str(getattr(_CFG, "vault_path", "") or ""), sources_of, set(real_ids),
                )
            adj, dropped_big = _signals.coupling_adjacency(transactions, report_dropped=True)
            if dropped_runs or dropped_big:
                logger.info("graph_report: coupling dropped %d over-cap transaction(s)",
                            dropped_runs + dropped_big)
            coupling_map = {
                _ck(a): {_ck(b): round(w, 4) for b, w in row.items()} for a, row in adj.items()
            }
            seen_cp: set[tuple[str, str]] = set()
            for a, row in adj.items():
                for b, w in row.items():
                    key = (min(a, b), max(a, b))
                    if key in seen_cp or G_und.has_edge(a, b) or a not in real_ids or b not in real_ids:
                        continue
                    seen_cp.add(key)
                    coupled_pairs.append(CoupledPair(source=key[0], target=key[1], score=round(w, 4)))
            coupled_pairs.sort(key=lambda c: (-c.score, c.source, c.target))
            coupled_pairs = coupled_pairs[:top_k]
        except Exception as exc:
            logger.warning("graph_report: coupling failed — %s", exc)

    # ------------------------------------------------------------------
    # Clusters
    # ------------------------------------------------------------------
    cluster_members: dict[int, list[str]] = {}
    for nid in real_ids:
        cid = cluster_map.get(nid, -1)
        if cid >= 0:
            cluster_members.setdefault(cid, []).append(nid)

    # Cohesion (intra-cluster edges / possible pairs) — analytics-only. One O(E)
    # pass tallies intra-edges per cluster; the per-cluster scan was O(C x E).
    intra_edges: dict[int, int] = {}
    # The off-diagonal of the same tally, in the same pass. Walking G_und twice
    # to get the other half of one contingency table is O(E) spent to keep two
    # loops in step, and they would drift the first time one of them learns to
    # skip an edge kind. Keyed on the ordered pair because a coupling is
    # symmetric and (a,b) and (b,a) counted apart would halve every cell.
    inter_pairs: dict[tuple[int, int], int] = {}
    if analytics:
        for u, v in G_und.edges():
            cu, cv = cluster_map.get(u, -1), cluster_map.get(v, -1)
            if cu < 0 or cv < 0:
                continue
            if cu == cv:
                intra_edges[cu] = intra_edges.get(cu, 0) + 1
            else:
                key = (cu, cv) if cu < cv else (cv, cu)
                inter_pairs[key] = inter_pairs.get(key, 0) + 1

    clusters: list[ClusterStat] = []
    for cid, members in sorted(cluster_members.items()):
        size = len(members)
        hub_node = max(members, key=lambda n: (deg.get(n, 0), n)) if members else None
        cohesion = 0.0
        possible = size * (size - 1) / 2 if size >= 2 else 0
        if analytics and possible > 0:
            cohesion = round(intra_edges.get(cid, 0) / possible, 4)
        clusters.append(ClusterStat(
            cluster_id=cid,
            size=size,
            hub=hub_node,
            members=sorted(members),
            cohesion=cohesion,
        ))

    # Structural gaps + discourse shape — analytics-only, mirror of the bridge
    # signal (areas that SHOULD connect but don't) plus a one-word topology read.
    sizes_by_cluster = {c.cluster_id: c.size for c in clusters}

    structural_gaps_list: list[StructuralGap] = []
    discourse_state = ""
    if analytics:
        structural_gaps_list = [
            StructuralGap(
                cluster_a=ca, cluster_b=cb, hub_a=ha, hub_b=hb,
                inter_edges=ie, gap_score=score, gap_density=dens,
            )
            for ca, cb, ha, hb, ie, score, dens in structural_gaps(nodes, edges, top_k=top_k)
        ]
        discourse_state = _discourse_state(G_und, clusters)

    # ------------------------------------------------------------------
    # Orphans (in-degree == 0, scoped to folder)
    #
    # Staging excluded: the inbox holds sources awaiting distillation and
    # `done/` holds the ones already consumed. Nothing is supposed to link to
    # either, so counting them made E(vault) a reading of the inbox — on a
    # freshly-ingested library, 12 of 12 orphans were staging files and the
    # orphan term was 92% of the total energy.
    # ------------------------------------------------------------------
    orphans: list[str] = sorted(
        nid for nid in real_ids
        if in_deg.get(nid, 0) == 0 and not _is_staging(nid)
    )

    # ------------------------------------------------------------------
    # Fragmentation: connected components of the wikilink layer. Catches what
    # the orphan signal can't — a linked clique detached from the rest counts
    # as one island. Metric only; bridging stays a human decision.
    # ------------------------------------------------------------------
    import networkx as nx
    n_components = (
        nx.number_connected_components(G_und) if G_und.number_of_nodes() else 0
    )

    # ------------------------------------------------------------------
    # Dangling (unresolved wikilinks aggregated by target name)
    # ------------------------------------------------------------------
    # edge target is "__unresolved__<name>" from graph_export
    ghost_refs = Counter(
        e.get("to", "").removeprefix("__unresolved__")
        for e in edges
        if e.get("type") == "AMBIGUOUS"
    )

    # Who asks for each missing target. "Referenced from" is what decides
    # whether a target is worth writing: three notes asking for it is a gap in
    # the vault, one is probably a typo. The list is bounded by the number of
    # AMBIGUOUS edges, which is the same thing already counted above.
    ghost_sources: dict[str, list[str]] = {}
    for e in edges:
        if e.get("type") != "AMBIGUOUS":
            continue
        target = e.get("to", "").removeprefix("__unresolved__")
        src = e.get("from", "")
        seen = ghost_sources.setdefault(target, [])
        if src and src not in seen:
            seen.append(src)

    dangling: list[dict] = sorted(
        [{"target": t, "refs": c, "sources": ghost_sources.get(t, [])}
         for t, c in ghost_refs.items()],
        key=lambda d: (-d["refs"], d["target"]),
    )

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------
    n_links = sum(1 for e in edges if e.get("type") == "EXTRACTED")
    # The share of the wikilink graph written by structure, not prose
    # (frontmatter properties, heading lines). A reading, not a filter: every
    # structural signal keeps both classes (ADR-0029).
    n_scaffold = sum(1 for e in edges if e.get("type") == "EXTRACTED" and e.get("scaffold"))
    n_unresolved = sum(1 for e in edges if e.get("type") == "AMBIGUOUS")

    # Initialize report shell to allow recursive calculation of totals if needed
    report = VaultReport(
        generated_at=_now(),
        scope=folder,
        totals={}, # Placeholder
        god_nodes=god_nodes,
        bridges=bridges,
        orphans=orphans,
        dangling=dangling,
        clusters=clusters,
        pagerank_map={nid: round(pr.get(nid, 0.0), 5) for nid in real_ids},
        betweenness_map={nid: round(bet.get(nid, 0.0), 4) for nid in real_ids},
        degree_map={nid: deg.get(nid, 0) for nid in real_ids},
        attention_candidates=attention,
        lean_notes=lean_notes,
        lean_chars=lean_chars,
        reformat_notes=reformat_notes,
        contested=contested,
        source_drift=source_drift,
        structural_gaps=structural_gaps_list,
        # Multi-note areas only, the same cut the coupling matrix and the app's
        # own "areas" count already make: a singleton is a row and a column of
        # zeroes with a perfect diagonal, and on a real vault there are more of
        # them than there are areas that carry it.
        inter_cluster={
            f"{a}|{b}": n
            for (a, b), n in ({(c, c): intra_edges.get(c, 0) for c in intra_edges}
                              | inter_pairs).items()
            if sizes_by_cluster.get(a, 0) > 1 and sizes_by_cluster.get(b, 0) > 1
        },
        discourse_state=discourse_state,
        temporal=temporal,
        structural_links=structural_links,
        coupled_pairs=coupled_pairs,
        coupling_map=coupling_map,
        load_bearing=load_bearing,
        core_map={nid: int(core_map.get(nid, 0)) for nid in real_ids},
        articulation=sorted(n for n in articulation_set if n in real_ids),
    )

    if with_embeddings:
        report.duplicate_pairs, report.confirmed_duplicate_pairs = _compute_duplicate_pairs(report)
        report.dissonance_map, report.misfiled = _compute_dissonance(
            report, nodes, G_und, knn_k=_dissonance_knn_k, k=top_k,
        )

    if with_cooccurrence:
        autolinks, stale, hubs, deficits = _compute_cooccur_delta(
            report, G_und, node_label,
            cooccur_store=_cooccur_store_override, k=top_k,
        )
        report.autolink_candidates = autolinks
        report.stale_links = stale
        report.missing_hubs = hubs
        report.integration_deficits = deficits
        (report.prerequisites, report.prereq_map,
         report.sprawling, report.bursting_concepts) = _compute_cooccur_variables(
            report, G_und, G_dir,
            cooccur_store=_cooccur_store_override, created=created_ts, k=top_k,
        )

    if analytics:
        try:
            from silica.config import CONFIG as _CFG
            from silica.kernel.report.graph_report.code_signals import _compute_code_signals
            vault_path = getattr(_CFG, "vault_path", "") or ""
            if vault_path:
                wl = {(min(u, v), max(u, v)) for u, v in G_und.edges()}
                cov, import_autolinks = _compute_code_signals(vault_path, wl)
                report.code_coverage = cov
                if import_autolinks:
                    report.autolink_candidates = list(report.autolink_candidates) + import_autolinks
        except Exception as exc:
            logger.warning("graph_report: code signals skipped — %s", exc)

    totals = {
        "notes": len(real_ids),
        "links": n_links,
        "scaffold_links": n_scaffold,
        # Every unresolved wikilink REFERENCE (the digest header's `unresolved=`),
        # not the count of distinct missing targets — that is `dangling_links`.
        "unresolved": n_unresolved,
        "dangling_links": len(dangling),
        "duplicate_pairs": len(report.duplicate_pairs),
        "confirmed_duplicates": len(report.confirmed_duplicate_pairs),
        "autolink_candidates": len(report.autolink_candidates),
        "stale_links": len(report.stale_links),
        "missing_hubs": len(report.missing_hubs),
        "integration_deficits": len(report.integration_deficits),
        "attention_candidates": len(attention),
        "lean_notes": len(lean_notes),
        "reformat_notes": len(reformat_notes),
        "contested": len(contested),
        "superseded_notes": (temporal.superseded_notes if temporal else 0),
        "superseded_sections": (temporal.superseded_sections if temporal else 0),
        "source_drift": len(source_drift),
        "orphans": len(orphans),
        "components": n_components,
        "clusters": len(clusters),
        "structural_gaps": len(structural_gaps_list),
        "structural_links": len(report.structural_links),
        "prerequisites": len(report.prerequisites),
        "coupled_pairs": len(report.coupled_pairs),
        "load_bearing": len(report.load_bearing),
        "articulation": len(report.articulation),
        "misfiled": len(report.misfiled),
        "bursting": len(report.bursting_concepts),
        "sprawling": len(report.sprawling),
        "code_files_documented": (report.code_coverage.documented if report.code_coverage else 0),
        "code_files_total": (report.code_coverage.total if report.code_coverage else 0),
    }
    report.totals = totals

    if memo_key is not None:
        # Callers treat VaultReport as read-only (its fields are "read-only
        # signals" by contract); a caller that mutates one would poison the
        # memo for the rest of the epoch.
        _report_memo[memo_key] = report
    return report


def _pagerank(G_und, alpha: float = 0.85, max_iter: int = 200, tol: float = 1.0e-6) -> dict[str, float]:
    """PageRank by power iteration on numpy, replacing nx.pagerank.

    `nx.pagerank` is scipy-only, and scipy was 104 MB of a 351 MB base install for
    this one call. Same recurrence nx runs (row-normalized adjacency, dangling mass
    redistributed uniformly, `sum|x - xlast| < n * tol` stop, raise on non-convergence),
    but the sparse matvec is a bincount over the edge arrays, so it needs numpy only
    (already a base dep). Measured against nx.pagerank before the swap: max abs diff
    1e-18 (float64 noise) and identical top-20 order on vault-shaped, two-component,
    self-loop, star, path and 10k/50k graphs, at 2-3x the speed.
    """
    import numpy as np

    nodes = list(G_und)
    n = len(nodes)
    if n == 0:
        return {}
    idx = {v: i for i, v in enumerate(nodes)}
    s: list[int] = []
    d: list[int] = []
    for u, v in G_und.edges():
        iu, iv = idx[u], idx[v]
        s.append(iu)
        d.append(iv)
        if iu != iv:  # a self-link lands once, as nx's adjacency writes it
            s.append(iv)
            d.append(iu)
    src = np.asarray(s, dtype=np.intp)
    dst = np.asarray(d, dtype=np.intp)
    out = np.bincount(src, minlength=n).astype(float)
    dangling = out == 0.0  # orphan notes: no mass of their own to hand out
    out[dangling] = 1.0    # keeps the division finite; the dangling term does the work
    p = np.full(n, 1.0 / n)
    x = p.copy()
    for _ in range(max_iter):
        last = x
        x = np.bincount(dst, weights=(last / out)[src], minlength=n)
        x = alpha * (x + last[dangling].sum() * p) + (1.0 - alpha) * p
        if np.abs(x - last).sum() < n * tol:
            return dict(zip(nodes, x.tolist()))
    # Same contract as nx: a non-converged run is not a result. The caller catches
    # and degrades to an all-zero pagerank_map.
    raise RuntimeError(f"pagerank failed to converge in {max_iter} iterations")


def _discourse_state(G_und, clusters: list[ClusterStat]) -> str:
    """Report-side wrapper: measure giant-component share on G_und, then apply
    the shared discourse_shape rule (single source, also used by the graph HUD)."""
    import networkx as nx

    from silica.kernel.recall.graph_export import discourse_shape

    giant = max((len(c) for c in nx.connected_components(G_und)), default=0)
    return discourse_shape(G_und.number_of_nodes(), giant, [c.size for c in clusters])


def _empty_report(scope: str = "") -> VaultReport:
    return VaultReport(
        generated_at=_now(),
        scope=scope,
        totals={"notes": 0, "links": 0, "unresolved": 0, "orphans": 0, "clusters": 0},
        god_nodes=[],
        bridges=[],
        orphans=[],
        dangling=[],
        clusters=[],
        duplicate_pairs=[],
        lean_notes=[],
        reformat_notes=[],
        pagerank_map={},
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
