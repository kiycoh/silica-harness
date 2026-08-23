# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""L1 Graph Data — deterministic, no LLM calls, no network.

Builds the vault's wikilink graph as node/edge lists (build_graph_data) and
detects topic communities via Louvain (detect_communities). Works with both CLI
and FS backends: triggers the driver index via graph_snapshot(), then reads
_graph / _unresolved_links / _notes directly to avoid O(N) subprocess calls on
the CLI backend.

Two partitions live here and never mix (ADR-0023): detect_communities is Louvain
over the wikilinks (structural, node["group"]) and detect_semantic_partition is
Louvain over the embedding k-NN (semantic, node["sgroup"]). Same algorithm, same
labeller, separate fields, colour keys and persisted ids.

Community detection via networkx.algorithms.community.louvain_communities
(built-in since networkx >= 3.0, already declared in pyproject.toml).
Degrades gracefully to no-community mode if unavailable.

The HTML viewer that renders this data (render_html / export_graph) lives in
silica.ui.web.graph_view — kept out of the kernel so this layer stays offline.
"""
from __future__ import annotations

from typing import Any

import colorsys
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Community:
    id: int
    label: str
    color: str
    size: int
    color_paper: str = ""   # the same community on a light floor


@dataclass
class Zone:
    """A cluster of the SEMANTIC partition — Louvain over the embedding k-NN.

    A separate type from Community on purpose (ADR-0023): the two partitions
    coexist, never substitute, and share neither colour key nor id space. A
    function taking `list[Zone]` cannot be handed communities by accident.

    `color_paper` is the same zone on a light floor. It rides on the zone rather
    than being derived in the viewer because the phase shift that keeps zone i
    away from community i lives here, and a viewer recomputing it would be a
    second place for ADR-0023's "no shared colour key" to break.
    """
    id: int
    label: str
    color: str
    size: int
    color_paper: str = ""


logger = logging.getLogger(__name__)


# Cluster colors: one distinct, vivid hue per community — the color encodes
# Louvain membership (real structure), so it must be unique per community and
# stable for a given id. Hues are confined to the arc of the mascot's body,
# its blue pole (212 deg) through to its warm magenta one (306 deg); the
# previous golden-angle walk covered the whole wheel and so emitted green,
# chartreuse and orange, hues that appear nowhere in the mark. The arc stops
# short of cyan deliberately: the mascot's body contains no cyan, only its eye
# does, and here cyan is already spoken for by _EDGE_COLOR_SIMILAR — a k-NN
# mesh that outnumbers the wikilinks two to one and swallowed any node sharing
# its hue. Ending at 212 rather than 186 puts the nearest community 0.087 away
# from that mesh in OKLab instead of 0.042.
#
# Within the arc a golden-ratio sequence keeps consecutive ids far apart and
# lightness alternates between two bands, which buys back most of what the
# shorter arc costs: minimum pairwise dE across 12 communities is 0.023 against
# 0.029 for the full wheel. Neither is a real number to lean on. The arc is not
# what limits color-as-identity, count is — past roughly ten communities both
# sets fall below the perceptual floor, so read the labels, not the hues.
_COMMUNITY_ARC = (212.0, 306.0)
_COMMUNITY_LIGHTNESS = (0.70, 0.54)
# The same two-band alternation, reflected for a paper floor. Only the bands
# move: the arc, the golden-ratio walk and the saturation are what make two
# communities tell apart, and none of them cares which way the floor goes.
# Chosen the same way the dark pair was, by measuring against the floor they
# land on — 4.93:1 to 11.11:1 across the arc on the light --void, against
# 3.07:1 to 8.59:1 for the dark pair on the dark one.
_COMMUNITY_LIGHTNESS_ON_PAPER = (0.42, 0.30)
_GOLDEN_RATIO_INV = 0.6180339887


def _community_color(i: int, on_paper: bool = False) -> str:
    lo, hi = _COMMUNITY_ARC
    hue = lo + ((i * _GOLDEN_RATIO_INV) % 1.0) * (hi - lo)
    bands = _COMMUNITY_LIGHTNESS_ON_PAPER if on_paper else _COMMUNITY_LIGHTNESS
    light = bands[i % 2]
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, light, 0.66)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


# Zones walk the same arc as communities but never the same KEY: the phase shift puts zone i half an
# arc away from community i, so the two partitions cannot be read off one
# swatch (ADR-0023: no shared colour key). Lightness bands are shared, because a
# zone hue also lands on a node dot when the zone layer is on and the dot has to
# stay legible on --void.
_ZONE_PHASE = 0.5


def _zone_color(i: int, on_paper: bool = False) -> str:
    lo, hi = _COMMUNITY_ARC
    hue = lo + ((i * _GOLDEN_RATIO_INV + _ZONE_PHASE) % 1.0) * (hi - lo)
    bands = _COMMUNITY_LIGHTNESS_ON_PAPER if on_paper else _COMMUNITY_LIGHTNESS
    light = bands[i % 2]
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, light, 0.66)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))

# Edge colours. Only ONE layer is vivid, the inferred one; the written links are
# the unlit neutral they have always been. They carry arrows, width and the most
# opacity, so they never needed a hue to rank first, and two saturated meshes
# stacked on one frame is how both stop being legible.
#
# The neutrals were unlit blue-violet, keyed to the mascot. The substrate is
# graphite now (docs/specs/visual-identity.md), so they are 214deg at 10%, each
# holding its predecessor's relative luminance to four decimals. That invariant
# is the one this block already committed to, and it is why the mesh ranking
# below survives the recolour untouched.
#
# Ambiguous stays red on purpose — an alarm that matches the brand stops reading
# as an alarm — but off the neon and into the same lightness band as its
# neighbours.
_EDGE_COLOR_EXTRACTED = "#868f9c"   # unlit graphite — resolved links
_EDGE_COLOR_AMBIGUOUS = "#e2544f"   # alarm red, deliberately outside the arc
_EDGE_COLOR_SIMILAR   = "#22b4cc"   # the signal — embedding k-NN (semantic map)

# The paper set. Every one of these is drawn ON the floor and read AGAINST it,
# so a light floor needs its own values or the whole mesh disappears — an edge
# is one or two pixels wide and has no second channel to fall back on. The
# ranking between the three layers is what is preserved, not the values: the
# written links stay the quiet neutral, ambiguous stays the alarm outside the
# arc, and similar stays the one vivid layer. Each is the same hue walked down
# into the range paper can hold, which for a hairline means roughly half the
# lightness of its crystal twin.
_EDGE_COLOR_EXTRACTED_PAPER = "#7c7261"
_EDGE_COLOR_AMBIGUOUS_PAPER = "#b3211b"
_EDGE_COLOR_SIMILAR_PAPER   = "#0a6070"

_NODE_DEFAULT_COLOR = {"background": "#545d67", "border": "#868f9c",
                       "background_paper": "#79705f",
                       "highlight": {"background": "#868f9c", "border": "#F4F5F7"}}
# Ghost is the one node that does not simply invert. On crystal it is the
# darkest thing on screen — "unlit, never black" — and the paper equivalent of
# unlit is not a dark dot but a pale one: 468 near-black dots on a white field
# would read as the loudest layer in the view, which is the opposite of what a
# ghost is. So it goes the other way, to the faintest thing the floor can hold.
_NODE_GHOST_COLOR   = {"background": "#141619", "border": "#545d67",
                       "background_paper": "#cbc6bd",
                       "highlight": {"background": "#22262a", "border": "#868f9c"}}


def _infer_type(path: str) -> str:
    p = path.lower().replace("\\", "/")
    if "_inbox" in p or p.startswith("inbox/"):
        return "inbox"
    stem = Path(path).stem.lower()
    if "hub" in stem:
        return "hub"
    return "note"


# Silica's own generated files at the vault ROOT (log.md, GRAPH_REPORT.md) are
# tooling output, not knowledge notes. The driver indexes them like any note, so
# without this filter GRAPH_REPORT.md's hundreds of `[[...]]` would make it the
# top hub AND give every note it lists an incoming link — silently zeroing the
# orphan count on the next run. Matched by root-relative stem only, so a genuine
# note in a subfolder (e.g. "Concepts/log.md") stays in the graph.
_VAULT_ROOT_ARTIFACT_STEMS = frozenset({"log", "GRAPH_REPORT"})


def is_vault_artifact(note_id: str) -> bool:
    """True if `note_id` is a Silica-generated file at the vault root.

    Id form varies by caller (graph node ids carry `.md`; other callers may
    not), so this matches on the `.md`-stripped stem and requires no path
    separator — i.e. vault-root only.
    """
    stem = note_id.replace("\\", "/").removesuffix(".md")
    return "/" not in stem and stem in _VAULT_ROOT_ARTIFACT_STEMS


def build_graph_data(folder: str = "") -> tuple[list[dict], list[dict]]:
    """Build node and edge lists from the driver's internal nx.DiGraph.

    Calls driver.graph_snapshot() once to populate _graph, _notes, and
    _unresolved_links, then reads them directly. This avoids O(N) subprocess
    calls on the CLI backend.
    """
    from silica.driver import get_driver

    driver = get_driver()
    internal_notes, unresolved_links, internal_graph = driver.graph_data()

    def _in_scope(path: str) -> bool:
        if not folder:
            return True
        prefix = folder.rstrip("/") + "/"
        return path.startswith(prefix) or path == folder.rstrip("/")

    in_scope: set[str] = {
        p.replace("\\", "/") for p in internal_notes if _in_scope(p.replace("\\", "/"))
    }

    # Provenance is a third grouping of the graph, disjoint from both Louvain
    # partitions (ADR-0023 names two; this is neither): read off the ledger,
    # never computed from topology. One inversion for the whole payload.
    from silica.kernel.write.provenance import note_key, sources_by_note

    by_note = sources_by_note()

    nodes: list[dict] = []
    for raw_path, ref in internal_notes.items():
        path = raw_path.replace("\\", "/")
        if path not in in_scope:
            continue
        if is_vault_artifact(path):   # keep Silica's own log.md/GRAPH_REPORT.md out of the graph
            continue
        node = {
            "id":    path,
            "label": ref.name,
            "title": path,
            "type":  _infer_type(path),
            "group": -1,
            "color": dict(_NODE_DEFAULT_COLOR),
            "path":  path,
            "font":  {"color": "#F4F5F7", "size": 13},
            "size":  16,
        }
        sources = by_note.get(note_key(path))
        if sources:
            node["sources"] = list(sources)
        nodes.append(node)

    node_ids: set[str] = {n["id"] for n in nodes}

    edges: list[dict] = []
    edge_set: set[tuple[str, str]] = set()
    edge_idx = 0

    for src_raw, tgt_raw, attrs in internal_graph.edges(data=True):
        src = src_raw.replace("\\", "/")
        tgt = tgt_raw.replace("\\", "/")
        if src not in node_ids or tgt not in node_ids:
            continue
        key = (src, tgt)
        if key in edge_set:
            continue
        edge_set.add(key)
        # Only the scaffold class is stamped: a prose link is the payload every
        # consumer already reads, and the viewer ignores keys it does not know.
        edges.append({
            **({"scaffold": True} if attrs.get("scaffold") else {}),
            "id":     f"e{edge_idx}",
            "from":   src,
            "to":     tgt,
            "type":   "EXTRACTED",
            # opacity and width are the RANK, and the viewer honours both. A
            # wikilink is the strongest thing in the graph because it is the only
            # edge a person actually wrote; everything else is inferred.
            "color":  {"color": _EDGE_COLOR_EXTRACTED,
                       "paper": _EDGE_COLOR_EXTRACTED_PAPER, "opacity": 0.72},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
            "width":  1.6,
        })
        edge_idx += 1

    ghost_nodes: dict[str, dict] = {}
    for src_raw, tgt_raw in unresolved_links:
        src = src_raw.replace("\\", "/")
        if src not in node_ids:
            continue
        tgt_name = tgt_raw.removesuffix(".md").rsplit("/", 1)[-1]
        ghost_id  = f"__unresolved__{tgt_name}"

        if ghost_id not in ghost_nodes:
            ghost_nodes[ghost_id] = {
                "id":           ghost_id,
                "label":        tgt_name,
                "title":        f"⚠ Unresolved: {tgt_name}",
                "type":         "ghost",
                "group":        -1,
                "color":        dict(_NODE_GHOST_COLOR),
                "path":         "",
                "font":         {"color": "#AEB4BE", "size": 11},
                "size":         10,
                "borderWidth":  2,
                "borderDashes": True,
            }

        key = (src, ghost_id)
        if key not in edge_set:
            edge_set.add(key)
            edges.append({
                "id":     f"e{edge_idx}",
                "from":   src,
                "to":     ghost_id,
                "type":   "AMBIGUOUS",
                "color":  {"color": _EDGE_COLOR_AMBIGUOUS,
                           "paper": _EDGE_COLOR_AMBIGUOUS_PAPER, "opacity": 0.55},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
                "width":  1.0,
                "dashes": [4, 4],
            })
            edge_idx += 1

    nodes.extend(ghost_nodes.values())
    return nodes, edges


def knn_edges(nodes: list[dict], k: int = 6) -> list[dict]:
    """Cosine k-NN edges over the embed store — the "semantic map" edge set.

    One undirected edge per note-pair to each note's k nearest neighbours by
    embedding cosine. Rendered like links (schema-compatible) but typed SIMILAR;
    the client force-layout positions notes by semantic proximity instead of by
    explicit wikilinks. Notes without a stored vector simply get no edges.

    Deterministic, offline (stored vectors + one blocked BLAS matmul for the whole
    vault, via EmbedStore.cosine_top_k_batch). Empty list when the embed index is
    absent.
    """
    from silica.kernel.recall.cooccurrence import cooccur_key
    from silica.kernel.recall.embed import get_store

    store = get_store()
    if len(store) == 0:
        return []

    # Store keyspace is stripped-.md/posix/case-preserved (cooccur_key); node ids
    # carry '.md'. Map both through cooccur_key so a store hit resolves to its node.
    id_by_key = {cooccur_key(n["id"]): n["id"] for n in nodes if n.get("type") != "ghost"}

    # One matmul for the whole vault, not one matvec per note: this is the only
    # all-pairs consumer of the embed store, and the per-note loop made it the
    # sole O(N²) pass in the system (2.9 s at 1238 notes, ~6 min at 12k). Same
    # top-k, exactly — see cosine_top_k_batch.
    neighbours = store.cosine_top_k_batch(list(id_by_key), k=k)

    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    idx = 0
    for key, nid in id_by_key.items():
        for cand in neighbours.get(key, ()):
            tid = id_by_key.get(cooccur_key(cand["path"]))
            if tid is None or tid == nid:
                continue
            p = (nid, tid) if nid < tid else (tid, nid)
            if p in seen:
                continue
            seen.add(p)
            score = float(cand["score"])
            edges.append({
                "id":    f"s{idx}",
                "from":  p[0],
                "to":    p[1],
                "type":  "SIMILAR",
                # 0.35, where this layer has always drawn. It dropped to 0.26
                # while the wikilinks were vivid too and the two meshes were
                # fighting; with the written links back to gray there is nothing
                # to stay out of the way of, and at 0.26 the k-NN reads as haze
                # rather than as lines. The rank still holds — a wikilink is
                # 0.72 and nearly three times the width.
                "color": {"color": _EDGE_COLOR_SIMILAR,
                          "paper": _EDGE_COLOR_SIMILAR_PAPER, "opacity": 0.35},
                # Width still carries the similarity score, but inside a band
                # that ends BELOW a wikilink's 1.6. It used to run to 3.0, so
                # the strongest inferred edges drew wider than every written one
                # — the rank exactly inverted — which went unnoticed while the
                # viewer ignored width entirely.
                "width": round(0.5 + 0.6 * score, 2),
                "score": round(score, 4),
            })
            idx += 1
    return edges


def edge_graph(
    nodes: list[dict], edges: list[dict], *, directed: bool = False,
    edge_type: str = "EXTRACTED",
):
    """nx graph over one edge type, between non-ghost nodes.

    The one builder every consumer of the driver graph shares (communities,
    distances, canvas metrics, the vault report) — they all filtered the same
    way by hand. `G.nodes()` is the real-id set, so callers need not recompute it.

    `edge_type` selects the layer: EXTRACTED (wikilinks) is the structural
    default every existing caller keeps; SIMILAR (embedding k-NN) is the
    semantic partition's substrate. One parameter, not a second builder, so the
    node set and the sorted insertion order below stay shared between them.

    Nodes go in SORTED. `G.nodes()` order is insertion order, and Louvain is a
    greedy local-move heuristic: it shuffles that list with its seeded RNG, so a
    different starting order lands on a different local optimum. Feeding it a set
    made the order hash-dependent, hence per-process, and E(vault) drifted on an
    unchanged vault (cohesion moved in the third decimal, cluster sizes by ~5%).

    EDGES go in sorted for the same reason, one layer down. Sorting the nodes
    only fixed where Louvain STARTS; each local move then scans `G[u]`, whose
    order is edge insertion order, and `build_graph_data` hands back a list whose
    order varies per process. Measured 2026-08-22 on a 709-note vault: three runs
    of an unchanged vault gave 24, 24 and 23 areas and a largest community of 74,
    74 and 96. With both sorted the three runs agree exactly. Anything that
    persists a partition depends on this -- cluster_id in clusters_ctx.json, the
    colour a community gets, E(vault), and the report history's `areas` delta,
    which otherwise reports movement that never happened.
    """
    import networkx as nx

    real = {n["id"] for n in nodes if n.get("type") != "ghost"}
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(sorted(real))
    G.add_edges_from(sorted(
        (e["from"], e["to"]) for e in edges
        if e.get("type") == edge_type and e.get("from") in real and e.get("to") in real
    ))
    return G


def _louvain_partition(
    nodes: list[dict], edges: list[dict], edge_type: str
) -> list[set[str]]:
    """Louvain over one edge layer, size-descending. [] when it cannot run.

    The ordering rule both partitions share — see detect_communities for why the
    id must be a function of the partition rather than of Louvain's return order.
    """
    from networkx.algorithms.community import louvain_communities

    G = edge_graph(nodes, edges, edge_type=edge_type)

    if G.number_of_edges() == 0:
        logger.info("graph_export: no %s edges — partition skipped.", edge_type)
        return []

    try:
        communities = louvain_communities(G, seed=42)
    except Exception as exc:
        logger.warning("graph_export: louvain_communities raised %s: %s", type(exc).__name__, exc)
        return []

    return sorted(communities, key=lambda c: (-len(c), min(c)))


def detect_communities(nodes: list[dict], edges: list[dict]) -> list[Community]:
    """Louvain community detection on the EXTRACTED (wikilink) edges, in-place.

    The STRUCTURAL partition — the vault as you linked it. Its semantic sibling
    is detect_semantic_partition, and the two never mix (ADR-0023): this one
    owns node["group"], node["color"] and the `clusters_ctx.json` id space, and
    reads nothing the other writes.

    Assigns node["group"] (int) and node["color"]. Ghost nodes keep group == -1.

    Returns a list of Community objects with topic labels where available.
    """
    # The id must be a function of the PARTITION, not of the order louvain
    # happened to return it in. Sorting `edge_graph`'s node list already froze
    # WHICH partition we land on, but `louvain_communities` returns its list of
    # sets in an order that still varies per process: three runs on an unchanged
    # vault gave the same sizes and labels with the 78-member community numbered
    # 60, 61, 60. That id is not cosmetic — `_community_color` is keyed on it, so
    # a community changed colour between two runs of the same vault, and
    # `clusters_ctx.json` persists `cluster_id` for readers in other processes.
    # Largest first, ties broken by the smallest member id: total, content-only.
    communities = _louvain_partition(nodes, edges, "EXTRACTED")
    if not communities:
        return []

    node_to_comm: dict[str, int] = {
        node_id: i
        for i, comm in enumerate(communities)
        for node_id in comm
    }

    for node in nodes:
        if node.get("type") == "ghost":
            continue
        comm_id = node_to_comm.get(node["id"], -1)
        node["group"] = comm_id
        if comm_id >= 0:
            color = _community_color(comm_id)
            paper = _community_color(comm_id, on_paper=True)
            node["color"] = {
                "background": color,
                "border":     color,
                # the same community, at the lightness the light floor holds
                "background_paper": paper,
                "highlight":  {"background": color, "border": "#F4F5F7"},
            }

    # Fetch community labels from the co-occurrence index; degrade to {} on any failure.
    # Member ids carry '.md'; CooccurStore.note_nodes normalises via cooccur_key, so
    # no manual strip is needed here (single source of truth for the key).
    from silica.kernel.recall.cooccurrence import get_cooccur_store
    try:
        labels = get_cooccur_store().community_labels([set(c) for c in communities])
    except Exception:
        labels = {}

    logger.info(
        "graph_export: %d communities across %d nodes.",
        len(communities), sum(len(c) for c in communities),
    )

    return [
        Community(
            id=i,
            label=labels.get(i, f"Cluster {i}"),
            color=_community_color(i),
            color_paper=_community_color(i, on_paper=True),
            size=len(comm),
        )
        for i, comm in enumerate(communities)
    ]


# ---------------------------------------------------------------------------
# The semantic partition (ADR-0023)
#
# Louvain over the SIMILAR (embedding k-NN) edges: notes grouped by what they
# are about, not by what you linked. It never substitutes for the structural
# community — the two are not nested on a real vault (14 of 78 wikilink
# clusters split across k-NN boundaries), so folding one into the other under a
# single name would make a cluster's meaning unreadable. Separate node field,
# separate colour key, separate persisted id space.
# ---------------------------------------------------------------------------


def semantic_snapshot_path() -> Path:
    # Function, not constant: resolves per current vault; tests monkeypatch it.
    from silica.kernel.recall import paths

    return paths.index_dir() / "semantic_partition.json"


def load_semantic_snapshot() -> tuple[dict[int, set[str]], int]:
    """Last generation's clusters and the id high-water mark, or ({}, 0).

    A cache under P2, never source of truth: deleting the file is always legal,
    the partition recomputes and the ids simply restart. It exists only as the
    MEMORY the inheritance reads — the partition itself is not stable enough to
    persist as data (removing one note moved 53 of 681 notes on the real vault).
    """
    import orjson

    try:
        raw = orjson.loads(semantic_snapshot_path().read_bytes())
        clusters = {int(k): set(v) for k, v in (raw.get("clusters") or {}).items()}
        return clusters, int(raw.get("next_id", 0))
    except Exception:
        return {}, 0


def save_semantic_snapshot(clusters: dict[int, list[str]], next_id: int) -> None:
    import orjson

    from silica.kernel.recall.paths import vault_epoch

    try:
        p = semantic_snapshot_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(orjson.dumps({
            # Stamped with the vault state it was taken from — the inheritance
            # does not check it, it is there so a stale snapshot can be told
            # apart from a fresh one when reading the file by hand.
            "epoch": vault_epoch(),
            "next_id": next_id,
            # Keys stringified here, parsed back with int() on load: orjson
            # refuses int keys outright, and this write is inside a
            # best-effort try — so the raise would have been swallowed and the
            # snapshot would silently never exist (ids restarting every run).
            "clusters": {str(k): v for k, v in clusters.items()},
        }))
    except Exception as e:
        logger.debug("graph_export: semantic snapshot save skipped (%s)", e)


def inherit_ids(
    partition: list[set[str]], prev: dict[int, set[str]], next_id: int
) -> tuple[list[int], int]:
    """Ids for a fresh partition: each cluster keeps its predecessor's.

    A cluster inherits the id of the previous-generation cluster it shares the
    most members with; each predecessor is claimed at most once, and clusters
    with no predecessor take ids that have never been handed out (the high-water
    mark, not max-of-current — a colour must not be recycled onto an unrelated
    cluster the generation after its own disappeared).

    Without this the ids track Louvain's size ordering, which is not stable: the
    partition is a global function of every vector, so one note added or removed
    can lawfully reshuffle it and every colour in the view jumps.

    ponytail: greedy by descending overlap, not optimal bipartite matching.
    Ties go to the earlier cluster then the lower id, so it stays deterministic;
    upgrade to scipy's linear_sum_assignment if two clusters ever visibly swap.
    """
    pairs = sorted(
        (
            (len(members & pmembers), i, pid)
            for i, members in enumerate(partition)
            for pid, pmembers in prev.items()
            if members & pmembers
        ),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    inherited: dict[int, int] = {}
    claimed: set[int] = set()
    for _overlap, i, pid in pairs:
        if i in inherited or pid in claimed:
            continue
        inherited[i] = pid
        claimed.add(pid)

    ids: list[int] = []
    for i in range(len(partition)):
        if i in inherited:
            ids.append(inherited[i])
        else:
            ids.append(next_id)
            next_id += 1
    return ids, next_id


def detect_semantic_partition(nodes: list[dict], edges: list[dict]) -> list[Zone]:
    """Louvain on the SIMILAR (k-NN) edges, in-place — the SEMANTIC partition.

    Assigns node["sgroup"] (int, -1 for ghosts and for notes with no vector) and
    touches node["group"]/node["color"] never. Returns the zones, size-descending,
    with ids inherited from the previous generation so the colours hold still.

    Degrades to [] with sgroup == -1 everywhere when the vault has no embed
    index: absent, never a community in disguise.
    """
    for node in nodes:
        node["sgroup"] = -1

    partition = _louvain_partition(nodes, edges, "SIMILAR")
    if not partition:
        return []

    prev, next_id = load_semantic_snapshot()
    ids, next_id = inherit_ids(partition, prev, next_id)

    zone_of: dict[str, int] = {
        node_id: ids[i] for i, comm in enumerate(partition) for node_id in comm
    }
    for node in nodes:
        node["sgroup"] = zone_of.get(node["id"], -1)

    # Same labeller as the communities — the c-TF-IDF is a function of the
    # member set, not of which partition produced it. Keyed by INDEX, so it maps
    # through `ids` before it can be read as a zone label.
    from silica.kernel.recall.cooccurrence import get_cooccur_store
    try:
        labels = get_cooccur_store().community_labels([set(c) for c in partition])
    except Exception:
        labels = {}

    save_semantic_snapshot(
        {ids[i]: sorted(comm) for i, comm in enumerate(partition)}, next_id
    )
    logger.info(
        "graph_export: %d semantic zones across %d nodes.",
        len(partition), sum(len(c) for c in partition),
    )

    return [
        Zone(
            id=ids[i],
            # "Zone N", never "Cluster N": the fallback label is a surface too,
            # and it says which partition it came from.
            label=labels.get(i, f"Zone {ids[i]}"),
            color=_zone_color(ids[i]),
            color_paper=_zone_color(ids[i], on_paper=True),
            size=len(comm),
        )
        for i, comm in enumerate(partition)
    ]


def shared_concepts(pairs, *, k: int = 3) -> dict[tuple[str, str], list[str]]:
    """{(a, b): shared concept labels} for graph-id pairs, best-effort.

    Lives beside community_labels for the same reason that does: reading the
    co-occurrence store for a LABEL is not a relatedness query, and the graph
    surfaces need one place that knows how to ask. A pair with no shared
    concept is absent from the result, which is what makes the caller's
    "corroborated only" filter a filter and not a preference.

    Empty when there is no index: an uncorroborated proposal is exactly what
    the gates refuted (docs/adr/0027), so no index means no proposals rather
    than proposals nobody checked.
    """
    from silica.kernel.recall.cooccurrence import cooccur_key, get_cooccur_store

    try:
        store = get_cooccur_store()
        if len(store) == 0:
            return {}
    except Exception as exc:
        logger.debug("graph_export: shared concepts need the co-occurrence index (%s)", exc)
        return {}
    out: dict[tuple[str, str], list[str]] = {}
    for a, b in pairs:
        try:
            common = set(store.note_nodes(cooccur_key(a))) & set(store.note_nodes(cooccur_key(b)))
        except Exception:
            continue
        if common:
            out[(a, b)] = sorted(store.node_label(x) for x in common)[:k]
    return out


def structural_gaps(
    nodes: list[dict], edges: list[dict], top_k: int = 10, min_size: int = 2
) -> list[tuple[int, int, str, str, int, float, float]]:
    """Community-pairs with the largest structural hole, score-descending.

    The mirror of the cross-cluster bridge signal: a bridge is a weak link that
    EXISTS between two areas; a structural gap is a pair of well-formed areas
    where the linking edges are ABSENT. Returns tuples
    (cluster_a, cluster_b, hub_a, hub_b, inter_edges, gap_score, gap_density).

    Two scores, two consumers:
    * gap_score = size_a * size_b / (1 + inter_edges) — RANKS the report: big
      disconnected areas surface first. Unbounded and size-scaling by design.
    * gap_density = 1 - inter_edges / (size_a * size_b) — the fraction of the
      possible inter-cluster edges that are ABSENT, in [0, 1). Bounded and
      near-invariant to cluster growth, so E(vault) sums this instead of the
      size-scaling gap_score (which made E reward fragmentation over growth).

    Reads node["group"] (set by detect_communities) and EXTRACTED edges, so it
    agrees node-for-node with the exported graph. Pure dict counting, no nx.
    ponytail: in a sparse vault gap_score mostly ranks by size-product — that IS
    the signal (biggest disconnected areas). Upgrade to a modularity-gain score
    if it ever reads noisy.
    """
    group_of = {
        n["id"]: n["group"]
        for n in nodes
        if n.get("type") != "ghost" and n.get("group", -1) >= 0
    }
    if not group_of:
        return []

    sizes = Counter(group_of.values())
    deg: Counter = Counter()
    inter: Counter = Counter()  # (lo_cluster, hi_cluster) -> #EXTRACTED edges joining them
    for e in edges:
        if e.get("type") != "EXTRACTED":
            continue
        ga, gb = group_of.get(e.get("from")), group_of.get(e.get("to"))
        if ga is None or gb is None:
            continue
        deg[e["from"]] += 1
        deg[e["to"]] += 1
        if ga != gb:
            inter[(min(ga, gb), max(ga, gb))] += 1

    # Hub per cluster = highest-degree member (ties → larger id), matching
    # ClusterStat.hub so the report row and the overlay edge point at the same note.
    hub: dict[int, str] = {}
    for nid, g in group_of.items():
        if g not in hub or (deg[nid], nid) > (deg[hub[g]], hub[g]):
            hub[g] = nid

    big = sorted(g for g, s in sizes.items() if s >= min_size)
    gaps: list[tuple[int, int, str, str, int, float, float]] = []
    for i, ca in enumerate(big):
        for cb in big[i + 1:]:
            ie = inter.get((ca, cb), 0)
            potential = sizes[ca] * sizes[cb]
            score = potential / (1 + ie)
            density = 1.0 - ie / potential   # absent fraction of possible links
            gaps.append((ca, cb, hub[ca], hub[cb], ie, round(score, 2), round(density, 4)))
    gaps.sort(key=lambda t: (-t[5], t[0], t[1]))
    return gaps[:top_k]


def discourse_shape(n_nodes: int, giant: int, cluster_sizes: list[int]) -> str:
    """One-word topology diagnosis from primitives.

    The single source of the rule, shared by the /graph report Summary and the
    graph HUD badge so the two surfaces never disagree.

    Fragmented: the graph splits into pieces — the giant component holds under
                half the notes.
    Focused:    one area dominates — the largest linked cluster is over half the
                linked notes (a star around a few hubs).
    Diversified: several comparably-sized, well-connected areas.
    ponytail: two 0.5 thresholds on component-share and cluster-share. Good
    enough for a headline; tune the knobs if a real vault reads wrong.
    """
    if n_nodes <= 0:
        return ""
    if giant / n_nodes < 0.5:
        return "Fragmented"
    linked = sorted((s for s in cluster_sizes if s >= 2), reverse=True)
    if linked and linked[0] / sum(linked) > 0.5:
        return "Focused"
    return "Diversified"


# ---------------------------------------------------------------------------
# Vault cluster-context cache
#
# Per-note cluster membership ({path: {cluster_id, hub, is_hub}}) persisted
# under index_dir() so any layer can annotate a note with its graph community
# without re-running Louvain. Written by the router's payload phase
# (build_vault_graph_ctx) and by silica_vault_report; read best-effort by
# build_substrate and silica_related. Bounded staleness by design: a note
# added after the last write reads as "no cluster" until the next refresh.
# ---------------------------------------------------------------------------

def cluster_ctx_path() -> Path:
    # Function, not constant: resolves per current vault; tests monkeypatch it.
    from silica.kernel.recall import paths

    return paths.index_dir() / "clusters_ctx.json"


# Bumped when the ctx KEY SHAPE changes: a cache written under the old shape is
# indistinguishable from a fresh one and would keep every consumer missing.
_CLUSTER_CTX_VERSION = 2


def load_cluster_ctx() -> dict | None:
    """Full cache envelope {"v", "sig": [nodes, edges], "ctx": {...}} or None.

    A cache from an older key shape is discarded, not migrated: one Louvain
    recompute is cheaper than a migration that has to be right forever.
    """
    import orjson

    p = cluster_ctx_path()
    if not p.exists():
        return None
    try:
        env = orjson.loads(p.read_bytes())
    except Exception:
        return None
    return env if env.get("v") == _CLUSTER_CTX_VERSION else None


def save_cluster_ctx(sig: list[int], ctx: dict) -> None:
    import orjson

    try:
        p = cluster_ctx_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(orjson.dumps({"v": _CLUSTER_CTX_VERSION, "sig": sig, "ctx": ctx}))
    except Exception as e:
        logger.debug("graph_export: cluster ctx cache save skipped (%s)", e)


def ctx_from_report(report) -> dict[str, dict]:
    """Per-note cluster ctx from a VaultReport (duck-typed: importing
    graph_report here would be a cycle — its compute imports this module).

    Keyed WITHOUT `.md`, and `hub` carries the same shape. Graph node ids keep
    the extension, and every consumer of this dict strips it before looking up
    (`collision`, `write` twice, `distill`, `linking` via basename) — so with
    the extension left on, all six missed on every note. Measured on a 717-note
    vault: 0 hits out of 717. The whole graph-context feature was inert —
    AUTOLINK's same-cluster narrowing, COLLISION's hub detection, HUB_UPDATE's
    cross-cluster warning, and the distiller's graph_context alike. Normalized
    here, once, rather than at six call sites that already agree with each other.
    """
    def key(node_id: str) -> str:
        return node_id.removesuffix(".md") if isinstance(node_id, str) else node_id

    ctx: dict[str, dict] = {}
    for cs in report.clusters:
        for member in cs.members:
            ctx[key(member)] = {
                "cluster_id": cs.cluster_id,
                "hub": key(cs.hub) if cs.hub else cs.hub,
                "is_hub": member == cs.hub,
            }
    # Isolated nodes too (cluster -1 = "no cluster" for consumers);
    # pagerank_map carries every real node id, zero-valued without analytics.
    for node_id in report.pagerank_map:
        if key(node_id) not in ctx:
            ctx[key(node_id)] = {"cluster_id": -1, "hub": None, "is_hub": False}
    return ctx


# (epoch, folder) -> nx.Graph. `Any`, not `object`: the value IS a graph and is
# used as one three lines below the read, and `object` only moved the knowledge
# into a comment.
_distance_graph_memo: dict[tuple[str, str], Any] = {}
# Entry bound: the epoch sweep already drops stale epochs, so this only caps
# how many FOLDER scopes can pin a graph at once inside one epoch. Oldest out.
_DISTANCE_MEMO_CAP = 8


def wikilink_graph_cached(folder: str = ""):
    """The undirected resolved-wikilink nx.Graph, memoized on the vault epoch.

    Shared by graph_distances (BFS) and the structural relatedness leg (V1):
    both read the graph and never mutate it. See graph_distances for the
    memo's bound and eviction.
    """
    from silica.kernel.recall.paths import vault_epoch

    epoch = vault_epoch()
    G = _distance_graph_memo.get((epoch, folder)) if epoch else None
    if G is None:
        nodes, edges = build_graph_data(folder=folder)
        G = edge_graph(nodes, edges)
        if epoch:
            for k in [k for k in _distance_graph_memo if k[0] != epoch]:
                del _distance_graph_memo[k]
            _distance_graph_memo[(epoch, folder)] = G
            while len(_distance_graph_memo) > _DISTANCE_MEMO_CAP:
                del _distance_graph_memo[next(iter(_distance_graph_memo))]
    return G


def graph_distances(source: str, *, folder: str = "") -> dict[str, int] | None:
    """BFS hop distances from `source` over the resolved wikilink graph.

    Returns {node_id (no .md): hops} for every note reachable from `source`,
    or None when the graph is unavailable or `source` is not in it. The
    structural complement of a semantic score: high similarity + absent/large
    distance = a missing link worth creating; distance 1 = already linked.

    The BFS-ready graph is memoized on the vault's file-state epoch: every
    silica_related call pays one of these, and building the decorated
    node/edge payload dominated it. A hit costs one vault stat walk (~10 ms
    per 1k notes) instead of the full build; the BFS reads the cached graph,
    never mutates it. The memo is bounded at _DISTANCE_MEMO_CAP folder
    scopes (oldest evicted), so many scoped calls cannot pin one graph per
    folder for the epoch's lifetime.
    """
    try:
        import networkx as nx

        G = wikilink_graph_cached(folder=folder)
    except Exception:
        return None
    src = source if source in G else source + ".md"
    if src not in G:
        src = source.removesuffix(".md")
        if src not in G:
            return None
    dist = nx.single_source_shortest_path_length(G, src)
    return {p.removesuffix(".md"): d for p, d in dist.items()}


def cluster_hub_of(ctx: dict, path: str) -> str | None:
    """Short hub label of `path`'s cluster in a cached ctx map, or None.

    Ctx keys are driver graph node ids, whose .md suffix varies by backend;
    facade paths carry none — try both forms at the seam.
    """
    p = path.removesuffix(".md")
    gctx = ctx.get(p) or ctx.get(p + ".md")
    hub = (gctx or {}).get("hub")
    if not hub:
        return None
    return hub.rsplit("/", 1)[-1].removesuffix(".md")


def canvas_metrics(nodes: list[dict], edges: list[dict], k: int = 400) -> tuple[dict[str, float], int]:
    """One nx build over EXTRACTED edges → (betweenness per node, giant-component
    size), for node sizing and the discourse-shape badge on the exported graph.

    Kept separate from graph_report.compute_report (the report path) so the
    offline export doesn't drag in the full report machinery; the betweenness
    formula matches compute_report's so the two agree.
    ponytail: betweenness is O(V·E), sampled at k pivots to stay bounded on big
    vaults (k==n on small ones is exact).
    """
    import networkx as nx

    G = edge_graph(nodes, edges)

    giant = max((len(c) for c in nx.connected_components(G)), default=0)
    bet: dict[str, float] = {}
    if G.number_of_edges() > 0:
        try:
            bet = nx.betweenness_centrality(
                G, k=min(G.number_of_nodes(), k), seed=42, normalized=True
            )
        except Exception:
            bet = {}
    return bet, giant
