# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""L1 Mindmap — deterministic radial map rooted on one note (zero LLM).

Builds a `MapView` — a radial mind-map centred on a single note — from the vault's
wikilink graph plus the latent (embeddings + co-occurrence) relatedness leg. The
builder computes the 2D coordinates **server-side**, so the two surfaces that
materialise a MapView (an Obsidian `.canvas` file and the web GUI's static SVG)
show the *identical* map and cannot diverge.

Complementary to `graph_export` (which draws the flat whole-vault network): `/graph`
is the network, `/map <note>` is a rooted association field.

Both polar coordinates carry data (viewers read display distance as semantic
distance — the distance-similarity metaphor — so they had better agree):
radius is the association cost to the root (weighted shortest path: wikilink
hop = 1.0, latent tie = 1/normalised strength), and angular neighbours are
ordered by embedding similarity inside each community wedge.

Layout is deterministic (same input → same positions, no `random`, no physics)
and non-overlap is guaranteed by placement order: nodes are placed
closest-first and each slides outward along its own angle until it clears
every already-placed card (exact AABB check against the fixed box size).
"""
from __future__ import annotations

import heapq
import math
import re
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from silica.kernel.recall.cooccurrence import CooccurStore


class GraphLike(Protocol):
    """The slice of an nx.Graph this module actually uses.

    A Protocol, not an import: the mindmap is pure geometry and must not pull
    networkx in. Structural typing states the same contract the caller already
    satisfies, and now states it where a checker can see it.
    """

    def __contains__(self, node: object) -> bool: ...

    def neighbors(self, node: str) -> Iterable[str]: ...


# (node id, node id) -> cosine.
SimFn = Callable[[str, str], float]
# (parent id, child id) -> the anchor line that justifies the link, or None.
LinkContextFn = Callable[[str, str], "str | None"]

# Fixed node-box size in canvas units. The non-overlap guarantee is stated in
# terms of these: distance ≥ hypot(W, H) ⇒ the two boxes' AABBs are disjoint.
BOX_W = 220.0
BOX_H = 64.0
# shrinking these is the ONLY compactness knob. _DIAG is the spacing
# unit every ring radius is a multiple of, so smaller boxes scale the whole
# layout tighter with wedge angles and ring count untouched; non-overlap still
# holds by construction (distance >= hypot(W,H), recomputed from the new size).
_DIAG = math.hypot(BOX_W, BOX_H)

# Muted slate for community-less nodes (group == -1) — never black/white.
_MUTED = "#5a6372"
_MUTED_PAPER = "#7d8496"   # the same no-community slate, read against paper


@dataclass
class MapNode:
    id: str          # stable: vault-relative path WITH .md (e.g. "concetti/x.md")
    path: str        # same as id; the file the Canvas node points at
    title: str
    x: float
    y: float
    community: int   # global Louvain membership (reused from graph_export); -1 = none
    hop: int         # 0 = root, 1, 2
    subtitle: str | None = None
    cost: float = 0.0    # association cost to the root (radius input); 0 = root
    degree: int = 0      # global wikilink degree — landmark salience, not layout


@dataclass
class MapEdge:
    src: str          # node id
    dst: str          # node id
    kind: str         # "wikilink" | "latent"
    weight: float


@dataclass
class MapView:
    root: str
    nodes: list[MapNode]
    edges: list[MapEdge]


@dataclass
class MapMaterials:
    """Everything build_mapview needs, injectable so tests need no live vault.

    `graph` is an undirected view of the wikilink graph (ids carry `.md`).
    `latent` is the relatedness leg in fused-ranking order: (id_with_md, title,
    strength) where strength is a native 0-1 signal (embed cosine / edge
    Jaccard), NOT the RRF score — the radius consumes it as 1/strength.
    """
    graph: GraphLike
    titles: dict[str, str]                # id -> display title
    community_of: dict[str, int]          # id -> global community (missing ⇒ -1)
    latent: list[tuple[str, str, float]] = field(default_factory=list)
    latent_evidence: dict[str, str] = field(default_factory=dict)  # id -> display "why"
    sim: SimFn | None = None              # None to abstain
    link_context: LinkContextFn | None = None


def node_color(community: int, on_paper: bool = False) -> str:
    """Community colour, shared with /graph; muted slate for -1 (no community).

    `on_paper` picks the light-floor band. Both are emitted onto every card so
    the map can switch theme without a rebuild — see render_map_svg.
    """
    if community < 0:
        return _MUTED_PAPER if on_paper else _MUTED
    from silica.kernel.recall.graph_export import _community_color
    return _community_color(community, on_paper=on_paper)


# ---------------------------------------------------------------------------
# Neighbourhood selection + cap
# ---------------------------------------------------------------------------

def _with_md(path: str) -> str:
    return path if path.endswith(".md") else path + ".md"


def _stem_title(node_id: str) -> str:
    return node_id.rsplit("/", 1)[-1].removesuffix(".md")


def _bfs(root: str, graph: GraphLike, hops: int) -> dict[str, tuple[int, str | None]]:
    """BFS up to `hops` on the undirected wikilink graph.

    Returns id -> (hop, parent_id); root maps to (0, None). Neighbours are
    visited in sorted order so the result is deterministic.
    """
    seen: dict[str, tuple[int, str | None]] = {root: (0, None)}
    q: deque[str] = deque([root])
    while q:
        u = q.popleft()
        hop, _ = seen[u]
        if hop >= hops or u not in graph:
            continue
        for v in sorted(graph.neighbors(u)):
            if v not in seen:
                seen[v] = (hop + 1, u)
                q.append(v)
    return seen


def _select(
    root: str,
    materials: MapMaterials,
    *,
    max_nodes: int,
    hops: int,
) -> tuple[dict[str, tuple[int, str | None]], set[str]]:
    """Pick the capped node set: root + wikilink BFS + latent, priority-ordered.

    Returns (selected, latent_only): selected is id -> (hop, parent);
    latent_only are the ids that entered through the latent leg alone.
    Priority tiers (kept top `max_nodes`): root, wikilink hop-1, deeper
    wikilink hops, then latent in list order — `materials.latent` arrives in
    fused-ranking order, so the index IS the selection priority (the per-item
    strength is a radius signal, not a ranking one).
    """
    reached = _bfs(root, materials.graph, hops=hops)
    candidates: list[tuple[int, float, str]] = []  # (tier, rank, id) sort key

    for nid, (hop, _parent) in reached.items():
        if nid == root:
            continue
        tier = 1 if hop == 1 else 2  # hop-1 outranks all deeper wikilink hops
        candidates.append((tier, 0.0, nid))

    for i, (lid, _title, _strength) in enumerate(materials.latent):
        if lid == root or lid in reached:
            continue
        candidates.append((3, float(i), lid))

    candidates.sort()
    kept = {root}
    for _tier, _rank, nid in candidates[: max(0, max_nodes - 1)]:
        kept.add(nid)

    selected: dict[str, tuple[int, str | None]] = {root: (0, None)}
    latent_only: set[str] = set()
    for nid in kept:
        if nid == root:
            continue
        if nid in reached:
            selected[nid] = reached[nid]
        else:
            selected[nid] = (1, root)  # latent neighbours hang off the root
            latent_only.add(nid)
    return selected, latent_only


# ---------------------------------------------------------------------------
# Radial association layout (deterministic; the only new algorithmic piece)
# ---------------------------------------------------------------------------

def _similarity_chain(members: list[MapNode], sim: SimFn | None) -> list[MapNode]:
    """Greedy nearest-neighbour ordering: each next node is the one most
    similar to the last placed, so angular neighbours are semantic ones.
    Falls back to id order when there is no similarity signal."""
    rest = sorted(members, key=lambda n: n.id)
    if sim is None or len(rest) < 3:
        return rest
    out = [rest.pop(0)]
    while rest:
        rest.sort(key=lambda n: (-sim(out[-1].id, n.id), n.id))
        out.append(rest.pop(0))
    return out


def _wedge_order(by_comm: dict[int, list[MapNode]], sim: SimFn | None) -> list[int]:
    """Communities as an angular sequence: largest first, then greedy by mean
    cross-similarity, so related communities share a wedge border."""
    comms = sorted(by_comm)
    if sim is None or len(comms) < 3:
        return comms

    def cross(a: int, b: int) -> float:
        vals = [sim(x.id, y.id) for x in by_comm[a] for y in by_comm[b]]
        return sum(vals) / len(vals) if vals else 0.0

    rest = sorted(comms, key=lambda c: (-len(by_comm[c]), c))
    out = [rest.pop(0)]
    while rest:
        rest.sort(key=lambda c: (-cross(out[-1], c), c))
        out.append(rest.pop(0))
    return out


# A wikilink's cost band. The floor is the map's anchor (the tightest tie of
# either kind costs exactly 1.0); the ceiling stays under 2.0 so ONE wikilink,
# however loose, never costs more than two tight ones — the written link keeps
# its structural rank over any chain that replaces it.
_WIKI_COST_SPAN = 0.9


def _association_costs(
    root: str, edges: list[MapEdge], ids: set[str], sim: SimFn | None = None
) -> dict[str, float]:
    """Weighted shortest-path cost root → every selected node (Dijkstra).

    Edge costs. Latent: s_max/s, where s_max is the best latent strength on the
    map — the strongest inferred tie costs exactly one tight wikilink. Wikilink:
    1.0 + span·(1 − t), where t is the pair's embedding similarity min-max
    normalised over the map's own wikilinks. Both legs are normalised WITHIN the
    map because embed cosine and edge Jaccard share a 0-1 range but not a
    calibration; only their ratios are trusted.

    Flat 1.0 without a similarity signal — which is also what kept hub roots
    degenerate: a root whose neighbours fill the node cap put every one of them
    at cost 1.0, so radius carried nothing and the collision slide (alphabetical)
    decided the spread. Similarity is what makes the band continuous there.
    """
    latent_s = [e.weight for e in edges if e.kind == "latent" and e.weight > 0]
    s_max = max(latent_s) if latent_s else 1.0

    wiki_sim: dict[int, float] = {}
    if sim is not None:
        for i, e in enumerate(edges):
            if e.kind == "wikilink":
                wiki_sim[i] = sim(e.src, e.dst)
    lo, hi = (min(wiki_sim.values()), max(wiki_sim.values())) if wiki_sim else (0.0, 0.0)

    def wiki_cost(i: int) -> float:
        if i not in wiki_sim or hi <= lo:
            return 1.0
        t = (wiki_sim[i] - lo) / (hi - lo)
        return 1.0 + _WIKI_COST_SPAN * (1.0 - t)

    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for i, e in enumerate(edges):
        w = wiki_cost(i) if e.kind == "wikilink" else s_max / max(float(e.weight), 1e-6)
        adj[e.src].append((e.dst, w))
        adj[e.dst].append((e.src, w))

    dist = {root: 0.0}
    heap: list[tuple[float, str]] = [(0.0, root)]   # node id breaks ties ⇒ deterministic
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        for v, w in sorted(adj[u]):
            nd = d + w
            if nd < dist.get(v, math.inf) - 1e-12:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    far = max(dist.values(), default=0.0) + 1.0     # disconnected safety: park at the rim
    for nid in ids:
        dist.setdefault(nid, far)
    return dist


def _layout(nodes: list[MapNode], *, costs: dict[str, float], sim: SimFn | None = None) -> None:
    """Place nodes radially, mutating each node's x/y. Root stays at (0, 0).

    Angle: 360° is partitioned into one contiguous wedge per community, width
    ∝ the community's node count; wedges and the siblings inside them are
    ordered by similarity chains (see _wedge_order/_similarity_chain), so
    angular proximity reads as semantic proximity.

    Radius: association cost mapped into a radial band — closer to the root ⇒
    more strongly tied. Non-overlap is guaranteed by placement order: nodes
    are placed cheapest-first and each slides outward along its own angle
    until it clears every already-placed card (exact AABB test), so a node
    only ever moves AWAY from its data-given radius, never inward.
    """
    non_root = [n for n in nodes if n.hop > 0]
    if not non_root:
        return

    by_comm: dict[int, list[MapNode]] = defaultdict(list)
    for n in non_root:
        by_comm[n.community].append(n)

    total = len(non_root)
    order = _wedge_order(by_comm, sim)
    angle: dict[str, float] = {}
    cursor = 0.0
    for c in order:
        width = 2 * math.pi * len(by_comm[c]) / total
        members = _similarity_chain(by_comm[c], sim)
        slot = width / len(members)
        for i, n in enumerate(members):
            angle[n.id] = cursor + (i + 0.5) * slot
        cursor += width

    cmin = min(costs[n.id] for n in non_root)
    cmax = max(costs[n.id] for n in non_root)
    # More nodes ⇒ a wider band, capped: the band buys radial fidelity and pays
    # in card legibility (the viewBox fits the whole map, so a wider band shrinks
    # every card). Measured on a 34-node hub map, cost↔radius correlation vs the
    # map's outer radius: ×2.5→.88/892, ×3→.91/958, ×4→.97/1146, ×6.8→1.00/1787.
    # ×3 is the knee — the text is still readable and the gradient still reads.
    spread = _DIAG * min(3.0, max(2.0, total / 5.0))

    def desired(c: float) -> float:
        t = (c - cmin) / (cmax - cmin) if cmax > cmin else 0.0
        return _DIAG + t * spread                   # _DIAG floor clears the root box

    placed = [n for n in nodes if n.hop == 0]       # the root blocks the centre
    step = _DIAG / 3
    for n in sorted(non_root, key=lambda m: (costs[m.id], m.id)):
        a = angle[n.id]
        ca, sa = math.cos(a), math.sin(a)
        r = desired(costs[n.id])
        guard = 0
        while guard < 2000 and any(
            abs(r * ca - p.x) < BOX_W and abs(r * sa - p.y) < BOX_H for p in placed
        ):
            r += step
            guard += 1
        n.x, n.y = r * ca, r * sa
        placed.append(n)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_mapview(
    root: str,
    materials: MapMaterials,
    *,
    max_nodes: int = 35,
    hops: int = 2,
) -> MapView:
    """Build a MapView rooted on `root` from injected materials (pure)."""
    root = _with_md(root)
    selected, latent_only = _select(root, materials, max_nodes=max_nodes, hops=hops)
    graph = materials.graph

    def _degree(nid: str) -> int:
        try:
            return len(list(graph.neighbors(nid))) if nid in graph else 0
        except Exception:
            return 0

    parent = {nid: p for nid, (_hop, p) in selected.items()}
    nodes: list[MapNode] = []
    for nid, (hop, _p) in selected.items():
        nodes.append(
            MapNode(
                id=nid,
                path=nid,
                title=materials.titles.get(nid, _stem_title(nid)),
                x=0.0,
                y=0.0,
                community=materials.community_of.get(nid, -1),
                hop=hop,
                subtitle=nid.rsplit("/", 1)[0] if "/" in nid else None,
                degree=_degree(nid),
            )
        )

    ids = {n.id for n in nodes}
    edges: list[MapEdge] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Wikilink edges: any graph edge among the selected nodes (tree + cross-branch).
    for u in ids:
        if u not in graph:
            continue
        for v in graph.neighbors(u):
            if v not in ids:
                continue
            key = (u, v) if u <= v else (v, u)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            edges.append(MapEdge(src=key[0], dst=key[1], kind="wikilink", weight=1.0))

    # Latent edges: root → each surviving latent neighbour not already wiki-linked.
    for lid, _title, strength in materials.latent:
        lid = _with_md(lid)
        if lid not in ids or lid == root:
            continue
        key = (root, lid) if root <= lid else (lid, root)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        edges.append(MapEdge(src=root, dst=lid, kind="latent", weight=float(strength)))

    costs = _association_costs(root, edges, ids, materials.sim)
    for n in nodes:
        n.cost = costs[n.id]

    # Subtitle = the WHY of the relation, when the materials can supply it:
    # the anchor line for wikilinks, the leg evidence for latent ties. The
    # folder path (set above) stays as the fallback and as the root's own.
    for n in nodes:
        if n.hop == 0:
            continue
        if n.id in latent_only:
            n.subtitle = materials.latent_evidence.get(n.id) or n.subtitle
        elif materials.link_context is not None:
            p = parent.get(n.id)
            ctx = materials.link_context(p, n.id) if p else None
            if ctx:
                n.subtitle = ctx

    _layout(nodes, costs=costs, sim=materials.sim)
    return MapView(root=root, nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

def _xy(n: MapNode) -> tuple[int, int]:
    """Rounded integer coordinates — shared by both serializers so they agree."""
    return round(n.x), round(n.y)


def _side(dx: float, dy: float) -> str:
    """Nearest box side for an edge leaving toward (dx, dy). Canvas y grows down."""
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "bottom" if dy >= 0 else "top"


# Wikilink edges keep a full colour; latent edges degrade to a muted colour +
# the "≈" label, because JSON Canvas has no dashed-edge style (only colour+label).
_CANVAS_WIKI_COLOR = "#22d3ee"
_CANVAS_LATENT_COLOR = _MUTED


def mapview_to_canvas(mv: MapView) -> dict:
    """Serialize a MapView to a JSON Canvas dict (jsoncanvas.org)."""
    nodes = []
    for n in mv.nodes:
        x, y = _xy(n)
        node: dict = {
            "id": n.id,
            "type": "file",
            "file": n.path,
            "x": x - round(BOX_W / 2),
            "y": y - round(BOX_H / 2),
            "width": round(BOX_W),
            "height": round(BOX_H),
        }
        if n.community >= 0:
            node["color"] = node_color(n.community)
        nodes.append(node)

    by_id = {n.id: n for n in mv.nodes}
    edges = []
    for i, e in enumerate(mv.edges):
        s, d = by_id[e.src], by_id[e.dst]
        latent = e.kind == "latent"
        edges.append({
            "id": f"e{i}",
            "fromNode": e.src,
            "toNode": e.dst,
            "fromSide": _side(d.x - s.x, d.y - s.y),
            "toSide": _side(s.x - d.x, s.y - d.y),
            "color": _CANVAS_LATENT_COLOR if latent else _CANVAS_WIKI_COLOR,
            "label": "≈" if latent else "",
        })
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Static SVG render (GUI surface; native, zero deps, positions precomputed)
# ---------------------------------------------------------------------------

def _clip_to_box(ox: float, oy: float, hw: float, hh: float, dx: float, dy: float) -> tuple[float, float]:
    """Point where the ray from (ox,oy) toward (ox+dx,oy+dy) exits the axis-aligned
    box of half-extents (hw,hh) centred at (ox,oy). Used to trim edge endpoints to
    a card's border instead of its centre."""
    tx = hw / abs(dx) if dx else math.inf
    ty = hh / abs(dy) if dy else math.inf
    t = min(tx, ty)
    if not math.isfinite(t):
        return ox, oy
    return ox + t * dx, oy + t * dy


def render_map_svg(mv: MapView, title: str = "Mindmap") -> str:
    """Render a MapView as a self-contained, interactive SVG page.

    Consumes the precomputed positions (no force layout ⇒ cannot diverge from the
    canvas). Radius encodes association strength (closer = stronger tie), so the
    guide circles mark the nearest/mid/farthest association distance actually on
    the map. Cards carry a community wash + community border whose weight scales
    with the note's wikilink degree (hubs read as landmarks); wikilink edges are
    solid curves with an arrowhead on true parent→child hops (same-tier wikilinks
    and all latent edges stay arrowless — they aren't "downstream" relationships).
    Pan/zoom/click-to-focus are plain SVG + vanilla JS (no new dependency),
    mirroring the dim-on-focus idiom already shipped for /graph.
    """
    import html

    by_id = {n.id: n for n in mv.nodes}
    pad = BOX_W
    xs = [n.x for n in mv.nodes] or [0.0]
    ys = [n.y for n in mv.nodes] or [0.0]
    min_x = min(xs) - pad
    min_y = min(ys) - pad
    vb_w = (max(xs) - min(xs)) + 2 * pad
    vb_h = (max(ys) - min(ys)) + 2 * pad

    # Radius is data (association cost), so the guides mark the real extremes:
    # nearest node, farthest node, and the midpoint between them.
    rs = sorted(math.hypot(n.x, n.y) for n in mv.nodes if n.hop > 0)
    guide_rs = sorted({rs[0], (rs[0] + rs[-1]) / 2, rs[-1]}) if rs else []
    guide_svg = "".join(
        f'<circle class="ring-guide" cx="0" cy="0" r="{r:.1f}"/>' for r in guide_rs
    )
    halo_r = min(70.0, (rs[0] if rs else 200.0) * 0.55)

    edge_svg = []
    for e in mv.edges:
        s, d = by_id[e.src], by_id[e.dst]
        latent = e.kind == "latent"
        # Bow the line into a deterministic quadratic curve (perpendicular
        # offset from the midpoint) — an organic arc instead of a straight
        # ruler line, with no randomness so the render stays reproducible.
        parent, child = (s, d) if s.hop <= d.hop else (d, s)
        x1, y1, x2, y2 = parent.x, parent.y, child.x, child.y
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1.0
        bow = min(36.0, length * 0.16)
        cx, cy = mx - dy / length * bow, my + dx / length * bow
        # Trim both ends to the card's border along the curve's own tangent
        # (the direction toward the control point) so the line touches the
        # badge edge and never its interior.
        sx, sy = _clip_to_box(x1, y1, BOX_W / 2, BOX_H / 2, cx - x1, cy - y1)
        ex, ey = _clip_to_box(x2, y2, BOX_W / 2, BOX_H / 2, cx - x2, cy - y2)
        is_tree = parent.hop != child.hop
        cls = "edge latent" if latent else ("edge wiki" if is_tree else "edge wiki lateral")
        marker = ' marker-end="url(#arrow)"' if is_tree and not latent else ""
        edge_svg.append(
            f'<path class="{cls}" data-src="{html.escape(e.src, quote=True)}" '
            f'data-dst="{html.escape(e.dst, quote=True)}" '
            f'd="M {sx:.1f} {sy:.1f} Q {cx:.1f} {cy:.1f} {ex:.1f} {ey:.1f}"{marker}/>'
        )

    node_svg = []
    deg_max = max((n.degree for n in mv.nodes), default=0)
    for n in mv.nodes:
        color = node_color(n.community)
        color_paper = node_color(n.community, on_paper=True)
        rx, ry = n.x - BOX_W / 2, n.y - BOX_H / 2
        root_cls = " root" if n.hop == 0 else ""
        title_esc = html.escape(n.title)
        sub_html = f'<div class="card-sub">{html.escape(n.subtitle)}</div>' if n.subtitle else ""
        # Landmark salience: border weight + wash track the note's wikilink
        # degree (relative to the map's own hub). Visual weight only — box
        # geometry is fixed, so the layout's non-overlap guarantee holds.
        t = (n.degree / deg_max) if deg_max else 0.0
        style = (f"--fo:{0.12 + 0.14 * t:.3f};--sw:{1.2 + 1.6 * t:.2f};"
                 f"--c:{color};--cp:{color_paper}")
        tip = html.escape("\n".join(
            s for s in (n.title, n.subtitle or "", f"{n.degree} links") if s
        ))
        node_svg.append(
            f'<g class="card{root_cls}" data-id="{html.escape(n.id, quote=True)}" '
            f'style="{style}" transform="translate({rx:.1f},{ry:.1f})">'
            f'<title>{tip}</title>'
            f'<rect class="frame" width="{BOX_W}" height="{BOX_H}" rx="10"/>'
            f'<foreignObject x="14" y="0" width="{BOX_W - 28}" height="{BOX_H}">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" class="card-body">'
            f'<div class="card-title">{title_esc}</div>{sub_html}</div>'
            f'</foreignObject></g>'
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<!-- ?theme= when the app embeds this; the OS when it is opened on its own.
     Same contract as /graph — see graph_view.render_html. -->
<script>
  (function () {{
    var q = new URLSearchParams(location.search).get("theme");
    var mq = window.matchMedia("(prefers-color-scheme: light)");
    var pinned = q === "light" || q === "dark";
    document.documentElement.dataset.theme = pinned ? q : (mq.matches ? "light" : "dark");
    if (!pinned) mq.addEventListener("change", function () {{ location.reload(); }});
  }})();
</script>
<style>
  :root{{
    color-scheme:dark;
    --void:#0A0D14;--slate-2:#161B27;--line:#232A3A;--line-2:#38425A;
    --frost:#E8ECF5;--ash:#8B95AC;--ash-dim:#566076;--cyan:#00A5E1;
    --card-shadow:rgba(0,0,0,.55);
    --mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,"DejaVu Sans Mono",monospace;
  }}
  /* Warm paper, the app shell's ramp — see static/app-base.css for how the values
     were derived. Restated rather than imported because this document is
     emitted standalone and has no stylesheet to link to. */
  :root[data-theme="light"]{{
    color-scheme:light;
    --void:#EFEAE0;--slate-2:#E9E2D4;--line:#D9D1C0;--line-2:#C6BCA8;
    --frost:#1A1815;--ash:#5B554B;--ash-dim:#615B4F;--cyan:#096275;
    --card-shadow:rgba(58,44,20,.22);
  }}
  *{{box-sizing:border-box}}
  html,body{{margin:0;height:100%;background:var(--void);overflow:hidden;
             -webkit-user-select:none;user-select:none}}
  svg{{width:100%;height:100vh;display:block;cursor:grab;touch-action:none}}
  svg.panning{{cursor:grabbing}}
  .ring-guide{{fill:none;stroke:var(--line-2);stroke-width:1;stroke-dasharray:2 6;opacity:.6}}
  .root-halo{{animation:halo 2.8s ease-in-out infinite}}
  @media (prefers-reduced-motion:reduce){{.root-halo{{animation:none}}}}
  @keyframes halo{{0%,100%{{opacity:.5;transform:scale(1)}}50%{{opacity:.18;transform:scale(1.12)}}}}
  .edge{{fill:none;stroke:var(--cyan);stroke-width:1.6;stroke-opacity:.55;
         transition:opacity .15s ease,stroke-opacity .15s ease}}
  .edge.lateral{{stroke-width:1.1;stroke-opacity:.28}}
  .edge.latent{{stroke:var(--ash-dim);stroke-dasharray:6 6;stroke-opacity:.55}}
  .edge.dim{{opacity:.1}}
  .card{{cursor:pointer;transition:opacity .15s ease}}
  .card.dim{{opacity:.3}}
  /* --fo/--sw are set inline per card: degree-scaled wash + border weight. */
  .card .frame{{fill:var(--c);stroke:var(--c);
                fill-opacity:var(--fo,.15);stroke-opacity:1;stroke-width:var(--sw,1.5);
                filter:drop-shadow(0 2px 6px var(--card-shadow))}}
  :root[data-theme="light"] .card .frame{{fill:var(--cp,var(--c));stroke:var(--cp,var(--c))}}
  .card:hover .frame{{fill-opacity:calc(var(--fo,.15) + .09)}}
  .card.root .frame{{stroke:var(--cyan);stroke-opacity:1;stroke-width:2.5}}
  .card-body{{font-family:var(--mono);height:100%;display:flex;flex-direction:column;
              justify-content:center;gap:3px;pointer-events:none;overflow:hidden}}
  .card-title{{font-size:12.5px;font-weight:600;line-height:1.25;color:var(--frost);
               display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}}
  .card-sub{{font-size:10px;color:var(--ash-dim);letter-spacing:.03em;
             white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
  #hud{{position:fixed;top:14px;right:14px;display:flex;flex-direction:column;
        align-items:flex-end;gap:8px;font-family:var(--mono);z-index:2}}
  #fit-btn{{padding:7px 10px;background:var(--slate-2);border:1px solid var(--line-2);
            color:var(--ash);font-family:var(--mono);font-size:12px;cursor:pointer;border-radius:0}}
  #fit-btn:hover{{border-color:var(--cyan);color:var(--cyan)}}
  #legend{{display:flex;flex-direction:column;gap:5px;background:var(--slate-2);
           border:1px solid var(--line);border-radius:0;padding:8px 10px;
           font-size:11px;color:var(--ash)}}
  .legend-row{{display:flex;align-items:center;gap:7px}}
  .swatch{{width:20px;height:0;border-top:2px solid var(--cyan)}}
  .swatch.dashed{{border-top-style:dashed;border-color:var(--ash-dim)}}
  .legend-hint{{font-size:10px;color:var(--ash-dim);letter-spacing:.02em}}
</style>
</head>
<body>
<svg id="stage" viewBox="{min_x:.0f} {min_y:.0f} {vb_w:.0f} {vb_h:.0f}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--cyan)" fill-opacity=".8"/>
    </marker>
    <radialGradient id="halo-grad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="var(--cyan)" stop-opacity=".4"/>
      <stop offset="100%" stop-color="var(--cyan)" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <g id="scene">
    {guide_svg}
    <circle class="root-halo" cx="0" cy="0" r="{halo_r:.0f}" fill="url(#halo-grad)"/>
    <g id="edges">{"".join(edge_svg)}</g>
    <g id="nodes">{"".join(node_svg)}</g>
  </g>
</svg>
<div id="hud">
  <button id="fit-btn" type="button">⊹ Fit map</button>
  <div id="legend">
    <div class="legend-row"><span class="swatch"></span>wikilink</div>
    <div class="legend-row"><span class="swatch dashed"></span>related (≈)</div>
    <div class="legend-hint">closer = stronger tie</div>
    <div class="legend-hint">bolder = more linked</div>
  </div>
</div>
<script>
const stage = document.getElementById("stage");
const scene = document.getElementById("scene");
let tx = 0, ty = 0, scale = 1;
function applyTransform() {{
  scene.setAttribute("transform", "translate(" + tx + "," + ty + ") scale(" + scale + ")");
}}
function toBase(evt) {{
  const pt = stage.createSVGPoint();
  pt.x = evt.clientX; pt.y = evt.clientY;
  return pt.matrixTransform(stage.getScreenCTM().inverse());
}}

// Pan: drag the background. Zoom: wheel, anchored on the cursor so the point
// under it stays put (tx/ty live in the SVG's own root coordinate space, which
// getScreenCTM() reports independently of the inner group's own transform).
let dragging = false, lastX = 0, lastY = 0;
stage.addEventListener("pointerdown", (e) => {{
  if (e.target.closest(".card")) return;
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  stage.classList.add("panning");
  stage.setPointerCapture(e.pointerId);
}});
stage.addEventListener("pointermove", (e) => {{
  if (!dragging) return;
  const k = {vb_w:.1f} / stage.clientWidth;
  tx += (e.clientX - lastX) * k; ty += (e.clientY - lastY) * k;
  lastX = e.clientX; lastY = e.clientY;
  applyTransform();
}});
function endDrag() {{ dragging = false; stage.classList.remove("panning"); }}
stage.addEventListener("pointerup", endDrag);
stage.addEventListener("pointerleave", endDrag);

stage.addEventListener("wheel", (e) => {{
  e.preventDefault();
  const p = toBase(e);
  const wx = (p.x - tx) / scale, wy = (p.y - ty) / scale;
  scale = Math.min(4, Math.max(0.3, scale * (e.deltaY > 0 ? 0.9 : 1.1)));
  tx = p.x - wx * scale; ty = p.y - wy * scale;
  applyTransform();
}}, {{ passive: false }});

document.getElementById("fit-btn").addEventListener("click", () => {{
  tx = 0; ty = 0; scale = 1; applyTransform();
}});

// --- click-to-focus: dim everything except the clicked card + its 1-hop
// edges (same idiom as /graph); background click clears. Embedded in the app
// iframe, a click also hands the note off to the parent's note panel.
const neighbors = {{}};
document.querySelectorAll(".edge").forEach((el) => {{
  const a = el.dataset.src, b = el.dataset.dst;
  (neighbors[a] = neighbors[a] || new Set()).add(b);
  (neighbors[b] = neighbors[b] || new Set()).add(a);
}});

function focusNode(id) {{
  document.querySelectorAll(".card").forEach((el) => {{
    const nb = neighbors[id] || new Set();
    el.classList.toggle("dim", id != null && el.dataset.id !== id && !nb.has(el.dataset.id));
  }});
  document.querySelectorAll(".edge").forEach((el) => {{
    el.classList.toggle("dim", id != null && el.dataset.src !== id && el.dataset.dst !== id);
  }});
}}

document.querySelectorAll(".card").forEach((el) => {{
  el.addEventListener("click", () => {{
    focusNode(el.dataset.id);
    if (window.parent !== window) {{
      // A card click means "what is this, and what is around it" — the host's
      // context drawer, same contract as a graph node click. Naming a note (a
      // wikilink, the file tree) is what opens the reader.
      window.parent.postMessage({{ type: "silica-open-context", path: el.dataset.id }}, "*");
    }}
  }});
}});
stage.addEventListener("click", (e) => {{ if (!e.target.closest(".card")) focusNode(null); }});

const knownIds = new Set(Array.from(document.querySelectorAll(".card")).map((el) => el.dataset.id));
window.addEventListener("message", (e) => {{
  if (e.data && e.data.type === "silica-focus-path") {{
    focusNode(knownIds.has(e.data.path) ? e.data.path : null);
  }}
}});
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Live-vault material gathering (IO; the tool/endpoint call this)
# ---------------------------------------------------------------------------

def _resolve_in(note: str, note_paths: list[str], titles: dict[str, str]) -> str | None:
    """Resolve `note` (a path OR a title) to a graph key. Pure; sorted ⇒ stable.

    Graph keys are full vault-relative paths WITH .md; a user (or the GUI input)
    may give a bare title or a path with/without .md. Try exact path first, then
    match by basename/title case-insensitively.
    """
    keys = sorted(p.replace("\\", "/") for p in note_paths)
    key_set = set(keys)
    cand = note.replace("\\", "/").strip()
    for c in (cand, cand + ".md", cand.removesuffix(".md")):
        if c in key_set:
            return c
    target = cand.removesuffix(".md").rsplit("/", 1)[-1].lower()
    for path in keys:
        stem = path.removesuffix(".md").rsplit("/", 1)[-1].lower()
        if stem == target or (titles.get(path, "") or "").lower() == target:
            return path
    return None


def _driver_graph():
    """One driver read → (raw notes map, {path: title}, undirected wikilink graph)."""
    from silica.driver import get_driver

    notes, _unresolved, g = get_driver().graph_data()
    return (
        notes,
        {p.replace("\\", "/"): ref.name for p, ref in notes.items()},
        g.to_undirected(as_view=True) if hasattr(g, "to_undirected") else g,
    )


def note_resolver():
    """One driver read → a pure closure: ref (path or title) -> graph key | None.

    Reuse when resolving many refs per render (e.g. linkifying a message): the
    driver graph is read once, the returned callable does no further IO.
    """
    notes, titles, _g = _driver_graph()
    paths = list(notes)
    return lambda ref: _resolve_in(ref, paths, titles)


def reading_path(
    src: str,
    dst: str,
    *,
    graph: GraphLike | None = None,
    cooccur_store: "CooccurStore | None" = None,
    weighted: bool = False,
) -> list[tuple[str, str]] | None:
    """Shortest reading path src → dst: BFS over wikilinks + latent cooccur edges.

    Endpoints are resolved graph keys (vault-relative, with .md). Returns
    [(path, leg), ...] where leg says how the node was reached from the
    previous one ("start" | "wikilink" | "cooccur"); None when the two notes
    are not connected. Read-only. Pass graph/cooccur_store to skip loading
    the live vault (tests).

    With `weighted=True` the hop count gives way to association strength:
    edge cost = 1/strength (wikilink strength 1.0, cooccur strength = its
    Jaccard score, so a weak 0.25 bridge costs 4 wikilink hops) and Dijkstra
    picks the strongest chain instead of the fewest hops.
    """
    if graph is None:
        _notes, _titles, graph = _driver_graph()
    if cooccur_store is None:  # embed leg unused here — load only the cooccur half
        cooccur_store = _cooccur_store()

    def neighbors(u: str) -> list[tuple[str, tuple[str, float]]]:
        out: dict[str, tuple[str, float]] = {}
        if cooccur_store is not None:
            # O(deg) per node: the store serves this from a two-way adjacency
            # built once per mutation, so the walk is O(V+E), not O(V·E).
            for nb, score in cooccur_store.note_edges_for(u).items():
                out[nb + ".md"] = ("cooccur", max(float(score), 1e-6))
        if u in graph:
            for nb in graph.neighbors(u):
                out[nb] = ("wikilink", 1.0)  # wikilink wins when both legs share an edge
        return sorted(out.items())

    prev: dict[str, tuple[str | None, str]] = {src: (None, "start")}
    if weighted:
        dist = {src: 0.0}
        heap: list[tuple[float, str]] = [(0.0, src)]  # node id breaks ties ⇒ deterministic
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, math.inf):
                continue
            if u == dst:
                break
            for v, (leg, strength) in neighbors(u):
                nd = d + 1.0 / strength
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = (u, leg)
                    heapq.heappush(heap, (nd, v))
    else:
        q: deque[str] = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v, (leg, _strength) in neighbors(u):
                if v not in prev:
                    prev[v] = (u, leg)
                    q.append(v)
    if dst not in prev:
        return None
    steps: list[tuple[str, str]] = []
    node: str | None = dst
    while node is not None:
        parent, leg = prev[node]
        steps.append((node, leg))
        node = parent
    steps.reverse()
    return steps


_WIKILINK_ALIAS = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")
_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


def _unwrap_line(line: str) -> str:
    """A body line stripped to prose: wikilinks unwrapped (alias wins), bold
    and backtick markers removed, list/heading markers dropped, whitespace
    collapsed."""
    s = _WIKILINK_ALIAS.sub(r"\2", line)
    s = _WIKILINK.sub(r"\1", s)
    s = re.sub(r"\*\*|`", "", s)
    return " ".join(s.strip().lstrip("#>*- \t").split())


def _clean_snippet(line: str, limit: int = 96) -> str:
    s = _unwrap_line(line)
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _anchor_snippet(line: str, names: set[str]) -> str:
    """The learnable part of an anchor line. When the line opens by restating
    the linked title ("X (…): why…"), keep the part after the colon — the card
    already shows the title; the explanation is the payload."""
    full = _unwrap_line(line)
    head, sep, tail = full.partition(":")
    if sep and tail.strip() and any(nm in head.lower() for nm in names):
        full = tail.strip()
    return _clean_snippet(full)


def _embed_sim():
    """(id, id) -> stored-vector cosine, or None when the embed index is empty.

    Ids carry .md, store keys don't — bridged through cooccur_key like
    knn_edges. Vectors are cached per id; a missing vector abstains with 0.0.
    """
    try:
        from silica.kernel.recall.cooccurrence import cooccur_key
        from silica.kernel.recall.embed import _cosine, get_store

        store = get_store()
        if len(store) == 0:
            return None
    except Exception:
        return None

    cache: dict[str, list[float] | None] = {}

    def _vec(nid: str):
        if nid not in cache:
            cache[nid] = store.get_vec(cooccur_key(nid))
        return cache[nid]

    def sim(a: str, b: str) -> float:
        va, vb = _vec(a), _vec(b)
        return _cosine(va, vb) if va is not None and vb is not None else 0.0

    return sim


def _link_context_fn(titles: dict[str, str]):
    """(parent_id, child_id) -> the cleaned body line where one wikilinks the
    other, or None. Checks the parent's body first, then the child's (the BFS
    parent is traversal order, not authorship — either side may hold the link).
    Bodies are read lazily through the driver, one read per note per map.
    """
    from silica.driver import get_driver

    drv = get_driver()
    bodies: dict[str, str] = {}

    def _body(nid: str) -> str:
        if nid not in bodies:
            try:
                bodies[nid] = drv.read_note(nid).content
            except Exception:
                bodies[nid] = ""
        return bodies[nid]

    def ctx(parent_id: str, child_id: str) -> str | None:
        for host, other in ((parent_id, child_id), (child_id, parent_id)):
            text = _body(host)
            if not text:
                continue
            stem = other.rsplit("/", 1)[-1].removesuffix(".md").lower()
            names = {stem, (titles.get(other) or "").lower()} - {""}
            for line in text.splitlines():
                low = line.lower()
                if "[[" in low and any(nm in low for nm in names):
                    snippet = _anchor_snippet(line, names)
                    if snippet:
                        return snippet
        return None

    return ctx


def gather_materials(root: str, *, latent_k: int = 10) -> MapMaterials:
    """Collect wikilink graph, titles, global communities, the latent leg, and
    the two evidence closures (embedding similarity, wikilink anchor lines)."""
    from silica.kernel.recall.graph_export import build_graph_data, detect_communities
    from silica.kernel.recall.relatedness import related_notes

    _notes, titles, undirected = _driver_graph()

    nodes, edges = build_graph_data()
    detect_communities(nodes, edges)  # assigns node["group"] in place (global, seed=42)
    community_of = {n["id"]: n.get("group", -1) for n in nodes}

    latent: list[tuple[str, str, float]] = []
    latent_evidence: dict[str, str] = {}
    try:
        embed_store, cooccur_store = _load_stores()
        for r in related_notes(
            _with_md(root), embed_store=embed_store, cooccur_store=cooccur_store, k=latent_k
        ):
            lid = _with_md(r.path)
            # Radius wants a native 0-1 strength, not the RRF score (ordering
            # only). Cooccur-only rows carry no such signal: park them just
            # under the weakest scored one — they are the ranking's tail anyway.
            latent.append((lid, r.name, r.embed_score or 0.0))
            latent_evidence[lid] = "≈ " + " ".join(r.evidence)
    except Exception:
        latent = []
    if latent:
        floor = min((s for _, _, s in latent if s > 0), default=0.3) * 0.9
        latent = [(lid, t, s if s > 0 else floor) for lid, t, s in latent]

    return MapMaterials(
        graph=undirected,
        titles=titles,
        community_of=community_of,
        latent=latent,
        latent_evidence=latent_evidence,
        sim=_embed_sim(),
        link_context=_link_context_fn(titles),
    )


def _cooccur_store() -> "CooccurStore | None":
    """The cooccur store, or None when empty/unavailable ⇒ that leg abstains."""
    try:
        from silica.config import CONFIG
        from silica.kernel.recall.cooccurrence import get_cooccur_store

        cs = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
        return cs if len(cs) > 0 else None
    except Exception:
        return None


def _load_stores():
    """(embed_store, cooccur_store), each None when empty/unavailable ⇒ leg abstains."""
    embed_store = None
    try:
        from silica.kernel.recall.embed import get_store
        es = get_store()
        embed_store = es if len(es) > 0 else None
    except Exception:
        embed_store = None

    return embed_store, _cooccur_store()
