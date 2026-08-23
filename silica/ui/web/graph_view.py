# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Graph viewer — the force-graph HTML emitter for the vault wikilink graph.

Split out of `silica.kernel.recall.graph_export` (which keeps the deterministic *data*
role: build_graph_data / detect_communities). This module owns only the viewer:
it turns nodes/edges/communities into a fully self-contained HTML file.

The document is three files under silica/ui/web/static/, each in the language
a browser and a linter read it in: graph-frame.html is the string.Template skeleton,
graph-frame.css the sheet, graph-frame.js the frame's code. This module fills the skeleton
with the counts, the legend and tree fragments, and one JSON island carrying
everything graph-frame.js needs for this document. All three are inlined at render
and never referenced by URL, because the emitted file opens from file:// with
no server behind it (ADR-0026); the renderer bundles and the Lexend face are
inlined for the same reason.

TWO renderers ship in every document: `3d-force-graph` (WebGL) and `force-graph`
(2D canvas), switched at runtime from the HUD. Not `numDimensions(2)` on the 3D
bundle — the point of 2D is readable node text, which canvas gives and a
flattened WebGL scene does not. They are kapsule siblings and share nearly the
whole chainable API, so `buildGraph()` is one builder with four branches.

Both bundles are *vendored* (silica/ui/web/static/, pinned to
3d-force-graph@1.80.0 and force-graph@1.51.2) and inlined into every emitted
file — the artifact opens offline, with no network at render time. There is no
CDN fallback anywhere: `lib_js=""` from a direct caller emits an empty script
(the tests pass a dummy), and `export_graph` (the production path) always
inlines the vendored bundles and raises loudly if either asset is missing.
"""
from __future__ import annotations

import html
import importlib.resources
import json
import logging
import string
from pathlib import Path

# The edge colours ride along so each legend swatch and the edges it stands for
# cannot drift apart: they were two literals of the same hex until one moved.
from silica.kernel.recall.graph_export import (
    _EDGE_COLOR_AMBIGUOUS,
    _EDGE_COLOR_AMBIGUOUS_PAPER,
    _EDGE_COLOR_EXTRACTED,
    _EDGE_COLOR_EXTRACTED_PAPER,
    _EDGE_COLOR_SIMILAR,
    _EDGE_COLOR_SIMILAR_PAPER,
    Community,
    Zone,
)

logger = logging.getLogger(__name__)

# Both renderers, in load order: WebGL first (the default mode), canvas second.
_VENDORED_BUNDLES = ("3d-force-graph.min.js", "force-graph.min.js")


def _vendored_lib_js() -> str:
    """Both renderer bundles, concatenated: they are independent UMD modules
    exporting `ForceGraph3D` and `ForceGraph`, so callers keep one `lib_js`
    string. A missing bundle raises (see _asset) and nothing falls back to a
    CDN `<script src>`: that would silently reintroduce the network dependency
    this split removed and hide the packaging bug."""
    return ";\n".join(_asset(name) for name in _VENDORED_BUNDLES)


def _vendored_font_face() -> str:
    """@font-face rule with the Lexend woff2 inlined as a data: URI, so the
    exported HTML stays fully self-contained (it is opened from file:// too).
    Cosmetic asset: if missing, degrade to the system-ui fallback, not a raise."""
    import base64

    res = importlib.resources.files("silica.ui.web") / "static" / "lexend-latin.woff2"
    if not res.is_file():
        return ""
    b64 = base64.b64encode(res.read_bytes()).decode("ascii")
    return (
        '@font-face{font-family:"Lexend";'
        f'src:url("data:font/woff2;base64,{b64}") format("woff2");'
        "font-weight:100 900;font-style:normal;font-display:swap}"
    )


def render_tree(nodes: list[dict], *, actions: bool = False) -> str:
    """Build a collapsible <details> file tree from real note paths.

    Pure: nodes -> HTML. Folders become nested <details>/<summary> (native
    collapse, no JS); notes become
    <button type=button class="tree-note" data-id=ID>NAME</button>.

    `actions` wraps each note in a `.tree-row` and adds a pin toggle beside it.
    Off by default because this same tree renders inside the graph frame, where
    a pin would point at a rail that document does not have: the app's rail
    asks for it, the frame's sidebar does not. A wrapper and not a button in a
    button, which is what a nested control would have been.

    A button and not a div: the tree is the primary route into a note, and as a
    click-only div every one of a vault's notes reported to the accessibility
    tree as `generic` and sat outside the tab order. Nothing else changes —
    the class, the data-id and the delegated `closest('.tree-note')` handlers
    are the same, and the CSS resets the button back to a row.
    Ghost nodes (type == "ghost" or empty path) are unresolved links, not files,
    so they are skipped. Folders sort before notes at each level; both groups
    sort case-insensitively.
    """
    root: dict = {}
    for n in nodes:
        if n.get("type") == "ghost":
            continue
        path = n.get("path") or ""
        if not path:
            continue
        *folders, leaf = path.split("/")
        cur = root
        for f in folders:
            cur = cur.setdefault(f, {})
        cur.setdefault("__notes__", []).append((leaf, n.get("id", path)))

    def emit(tree: dict, depth: int) -> str:
        out = []
        for name in sorted((k for k in tree if k != "__notes__"), key=str.lower):
            attr = " open" if depth == 0 else ""
            out.append(f"<details{attr}><summary>{html.escape(name)}</summary>")
            out.append(emit(tree[name], depth + 1))
            out.append("</details>")
        for leaf, nid in sorted(tree.get("__notes__", []), key=lambda x: x[0].lower()):
            note = (
                f'<button type="button" class="tree-note" '
                f'data-id="{html.escape(nid, quote=True)}">'
                f"{html.escape(leaf)}</button>"
            )
            if not actions:
                out.append(note)
                continue
            out.append(
                f'<div class="tree-row">{note}'
                f'<button type="button" class="tree-pin" aria-pressed="false" '
                f'data-pin="{html.escape(nid, quote=True)}" '
                f'title="keep this note in the rail" '
                f'aria-label="pin {html.escape(leaf, quote=True)}">'
                f'<svg aria-hidden="true" width="11" height="11" viewBox="0 0 24 24" fill="none" '
                f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                f'<line x1="12" y1="17" x2="12" y2="22"></line>'
                f'<path d="M5 17h14l-2-4V4H7v9z"></path></svg></button></div>'
            )
        return "".join(out)

    return f'<div id="file-tree">{emit(root, 0)}</div>'


# The frame's own three files, beside the bundles and inlined the same way. Not
# `<script src>` as index.html does it: the emitted document is opened from
# file:// with no server behind it, so a URL is a reference to nothing there
# (ADR-0026). The cost is one read per render, which /graph already pays
# fifteen-fold for the bundles.
_FRAME_ASSETS = ("graph-frame.html", "graph-frame.css", "graph-frame.js")


def _asset(name: str) -> str:
    """One file under ui/web/static/, or a RuntimeError that names it.

    Loud on purpose: a missing asset is a packaging bug, and the alternative
    (an empty string in its slot) renders a frame that looks like a vault bug.
    """
    res = importlib.resources.files("silica.ui.web") / "static" / name
    if not res.is_file():
        raise RuntimeError(
            f"graph_export: vendored {name} is missing from silica/ui/web/static/ "
            "This is a packaging bug. Reinstall silica or re-vendor the assets (pinned "
            "3d-force-graph@1.80.0, force-graph@1.51.2)."
        )
    return res.read_text(encoding="utf-8")


def _json_island(payload: dict) -> str:
    """The document's data as the body of a <script type="application/json">.

    `<` becomes \\u003c, which is still JSON and the same string again after
    JSON.parse. Every `<`, not only `</`: the parser ends script data at the
    first `</script`, but a `<!--` followed by a `<script` inside the data
    moves it to the double-escaped state, where the real `</script>` no
    longer ends the element. One character closes both holes.

    allow_nan=False because NaN is not JSON: a non-finite score would reach
    JSON.parse as a syntax error and a blank frame, and this names the defect
    on the server side instead, where the log and the /graph error can show it.
    """
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")


def _frame_context(
    nodes: list[dict],
    edges: list[dict],
    communities: "list[Community]",
    title: str,
    lib_js: str,
    discourse: str,
    zones: "list[Zone]",
) -> dict:
    """Every value graph-frame.html asks for, under the name it asks for it by.

    A dict rather than locals(): a test holds its keys and the template's
    identifiers equal in both directions, so a slot nobody fills and a value
    nobody reads both fail in the suite, not as a blank in the sidebar.
    """
    # The title is a tool argument the model chooses, so it is untrusted markup
    # in both slots it lands in (<title> and the sidebar <h1>).
    title = html.escape(title)

    n_notes      = sum(1 for n in nodes if n.get("type") != "ghost")
    n_ghost      = sum(1 for n in nodes if n.get("type") == "ghost")
    n_extracted  = sum(1 for e in edges if e.get("type") == "EXTRACTED")
    n_ambiguous  = sum(1 for e in edges if e.get("type") == "AMBIGUOUS")
    n_gaps       = sum(1 for e in edges if e.get("type") == "GAP")
    n_similar    = sum(1 for e in edges if e.get("type") == "SIMILAR")
    n_proposed   = sum(1 for e in edges if e.get("type") == "PROPOSED")
    n_coupled    = sum(1 for e in edges if e.get("type") == "COUPLED")
    n_discord    = sum(1 for e in edges if e.get("discord"))
    n_cut        = sum(1 for n in nodes if n.get("cut"))
    n_communities = len(communities)
    # Semantic-map edges: only surface the row when present (the links view
    # have none, so the row would just read 0 and confuse).
    similar_row = (
        f'<label class="filter-row" style="margin-top:4px" title="Embedding k-NN: notes pulled together by semantic similarity">'
        f'<input type="checkbox" id="cb-similar" checked onchange="updateEdgeFilter()">'
        f'<div class="dot-edge" style="--c:{_EDGE_COLOR_SIMILAR};--cp:{_EDGE_COLOR_SIMILAR_PAPER}"></div>similar'
        f'<span class="ct">{n_similar}</span>'
        f'</label>'
    ) if n_similar else ""

    # Three rows that only exist when their layer does. Each states the rule it
    # obeys in the tooltip, because none of the three is a thing the reader
    # wrote: they are what the vault INFERS, and a layer you did not author has
    # to say where it came from before it is worth ticking.
    proposed_row = (
        f'<label class="filter-row" style="margin-top:4px" title="Pairs two hops apart that share '
        f'neighbours (Adamic-Adar) AND share a concept in the co-occurrence index. Corroborated '
        f'only: on its own this signal lost every golden pair it was tested on, so an uncorroborated '
        f'proposal is not drawn.">'
        f'<input type="checkbox" id="cb-proposed" onchange="updateEdgeFilter()">'
        f'<div class="dot-edge dashed" style="--c:{_EDGE_COLOR_PROPOSED};--cp:{_EDGE_COLOR_PROPOSED_PAPER}"></div>'
        f'proposed links'
        f'<span class="ct">{n_proposed}</span>'
        f'</label>'
    ) if n_proposed else ""
    coupled_row = (
        f'<label class="filter-row" style="margin-top:4px" title="Notes written from the same source '
        f'or in the same nucleate run, that never came to link each other. Provenance, not topology.">'
        f'<input type="checkbox" id="cb-coupled" onchange="updateEdgeFilter()">'
        f'<div class="dot-edge dashed" style="--c:{_EDGE_COLOR_COUPLED};--cp:{_EDGE_COLOR_COUPLED_PAPER}"></div>'
        f'coupled'
        f'<span class="ct">{n_coupled}</span>'
        f'</label>'
    ) if n_coupled else ""
    # "between zones" and not "crossing a zone", which was the first name and read
    # as geometry: on a canvas most links cross some zone or other on their way,
    # so the row appeared to name where a line PASSES rather than where its two
    # ends sit. This name says the ends.
    discord_row = (
        f'<label class="filter-row" style="margin-top:4px" title="Wikilinks you wrote whose two ends '
        f'fall in different SEMANTIC zones. It recolours links that are already there rather than '
        f'adding any. A reading, not a fault: the notes it flags were checked by hand and were '
        f'filed correctly 13 times out of 14.">'
        f'<input type="checkbox" id="cb-discord" onchange="updateEdgeFilter()">'
        f'<div class="dot-edge" style="--c:{_EDGE_COLOR_DISCORD};--cp:{_EDGE_COLOR_DISCORD_PAPER}"></div>'
        f'between zones'
        f'<span class="ct">{n_discord}</span>'
        f'</label>'
    ) if n_discord else ""

    # The fragility reading rides the Nodes section, not Edges: coreness and cut
    # vertices are facts about a note. 2D only, like the other two rings - in 3D
    # a ring means rebuilding every node's geometry.
    #
    # No checkbox, unlike the three rows above: this ring is always drawn. Those
    # three are claims about what you did NOT write, so each has to be asked for;
    # a cut vertex is a fact about the graph in front of you, on the same footing
    # as hub and orphan, which have no checkbox either. The row is here to say
    # what the amber ring means, and it aligns with its two neighbours because it
    # carries the same swatch-then-word shape they do.
    cut_row = (
        f'<div class="filter-row" title="Removing this note disconnects the graph: every path '
        f'between the notes it strands and the rest goes through it. Always ringed in 2D and '
        f'named on hover in 3D; the Work panel says what each one carries.">'
        f'<span class="ring" style="border-color:var(--warn)"></span>load-bearing'
        f'<span class="ct">{n_cut}</span>'
        f'</div>'
    ) if n_cut else ""

    discourse_badge = (
        f'<div style="font-size:11px;color:var(--ash);letter-spacing:.04em" '
        f'title="Shape of the wikilink graph: how much of the vault sits in the largest connected '
        f'component and how evenly the clusters split it.">'
        f'discourse: <span style="color:var(--warn);font-weight:600">{html.escape(discourse)}</span></div>'
        if discourse else ""
    )

    # The gap list used to live here, under Edge types. It is a vault-level
    # worklist, not a key to what the canvas is painting, and in a legend it read
    # as neither. It now sits on the vault-level surface that already measured it
    # -- the Structural gaps card in metrics -- where each row carries the
    # bridging action. The amber GAP overlay and its checkbox stay: that IS a key.
    legend_items = "".join(
        f'<div class="legend-item" data-community="{c.id}" data-size="{c.size}" onclick="filterCommunity({c.id})">'
        f'<span class="dot" style="--c:{c.color};--cp:{c.color_paper or c.color}"></span>{html.escape(c.label)} '
        f'<span class="ct">{c.size}</span>'
        f'</div>\n'
        # Biggest first: the legend is read top-down, and the clusters that
        # carry the vault are the ones worth seeing without scrolling.
        for c in sorted(communities, key=lambda c: (-c.size, c.id))
    )

    # The zone layer only exists where the vault has vectors, so the rows do too:
    # an empty "semantic zones" checkbox would promise a layer with nothing
    # behind it. They are two rows of the Colour section and not a section of
    # their own, because what the HUD answers there is "what does a colour mean
    # here", and the answer is a choice between two partitions. The tooltip
    # states the rule the layer obeys, since a second grouping over the same
    # notes is unreadable without one.
    zone_section = (
        f'<label class="filter-row" style="margin-top:4px" title="Louvain over the embedding k-NN, '
        f'not over your wikilinks: notes grouped by what they are about, drawn as a hull and a '
        f'name. Note colour does not change - it always means the structural area - so the two '
        f'partitions can be read against each other in one frame. In 3D the zones are names only; '
        f'the hulls are drawn on the 2D canvas.">'
        f'<input type="checkbox" id="cb-zones" onchange="updateZoneFilter()">'
        f'<span class="dot" style="--c:{zones[0].color if zones else "#565a77"};'
        f'--cp:{(zones[0].color_paper or zones[0].color) if zones else "#6b6f8c"};opacity:.5"></span>semantic zones'
        f'<span class="ct">{len(zones)}</span>'
        f'</label>'
        f'<label class="filter-row" style="margin-top:4px" title="Untick to leave the zones alone '
        f'in the frame: the macro read of the vault, with the individual notes and their edges out '
        f'of the way.">'
        f'<input type="checkbox" id="cb-zone-nodes" checked onchange="updateZoneFilter()">'
        f'notes'
        f'</label>'
    ) if zones else ""

    tree_html = render_tree(nodes)

    # Two display settings, baked in at render time rather than read from the
    # frame: /graph regenerates the whole document per request anyway, so there
    # is nothing here for a live toggle to be live *against*.
    from silica.config import CONFIG

    data = {
        "nodes": nodes,
        "edges": edges,
        "comm_labels": {c.id: c.label for c in communities},
        # The semantic partition rides as its own list (ADR-0023): the JS keys
        # it by node.sgroup, never by node.group.
        "zones": [
            {"id": z.id, "label": z.label, "color": z.color,
             "color_paper": z.color_paper or z.color, "size": z.size}
            for z in zones
        ],
        "particles": bool(CONFIG.graph_particles),
        "shading": bool(CONFIG.graph_shading),
        # The same constants paint the legend swatches below, so the JS reads
        # the edge colours from here and not from literals of its own.
        "colors": {
            "extracted": _EDGE_COLOR_EXTRACTED,
            "extracted_paper": _EDGE_COLOR_EXTRACTED_PAPER,
            "similar_particle": _EDGE_COLOR_SIMILAR_PARTICLE,
            "similar_particle_paper": _EDGE_COLOR_SIMILAR_PARTICLE_PAPER,
            "gap": _EDGE_COLOR_GAP,
            "gap_paper": _EDGE_COLOR_GAP_PAPER,
            "discord": _EDGE_COLOR_DISCORD,
            "discord_paper": _EDGE_COLOR_DISCORD_PAPER,
        },
    }

    return {
        "title": title,
        "lib_js": lib_js,
        "font_face": _vendored_font_face(),
        "graph_css": _asset("graph-frame.css"),
        "n_notes": n_notes,
        "n_ghost": n_ghost,
        "n_extracted": n_extracted,
        "n_ambiguous": n_ambiguous,
        "n_gaps": n_gaps,
        "n_communities": n_communities,
        "color_extracted": _EDGE_COLOR_EXTRACTED,
        "color_extracted_paper": _EDGE_COLOR_EXTRACTED_PAPER,
        "color_ambiguous": _EDGE_COLOR_AMBIGUOUS,
        "color_ambiguous_paper": _EDGE_COLOR_AMBIGUOUS_PAPER,
        "color_gap": _EDGE_COLOR_GAP,
        "color_gap_paper": _EDGE_COLOR_GAP_PAPER,
        "tree_html": tree_html,
        "similar_row": similar_row,
        "proposed_row": proposed_row,
        "coupled_row": coupled_row,
        "discord_row": discord_row,
        "cut_row": cut_row,
        "discourse_badge": discourse_badge,
        "legend_items": legend_items,
        "zone_section": zone_section,
        "graph_data": _json_island(data),
        "graph_js": _asset("graph-frame.js"),
    }


def render_html(
    nodes: list[dict],
    edges: list[dict],
    communities: "list[Community]" = (),  # type: ignore[assignment]
    title: str = "Vault Graph",
    lib_js: str = "",
    discourse: str = "",
    zones: "list[Zone]" = (),  # type: ignore[assignment]
) -> str:
    """Produce a fully self-contained force-graph HTML string.

    lib_js is the vendored renderer bundle, embedded inline (offline-capable;
    export_graph always supplies it via _vendored_lib_js).
    communities is a list of Community objects; legend is built from it.
    zones is the semantic partition (node["sgroup"]); it draws the zone layer,
    which is off until asked for and is a different grouping from communities.

    string.Template and not an f-string: the skeleton is a file the browser
    and a linter read as HTML, and its `$name` slots need no brace doubling,
    which is what had every `{` in 2,200 lines of CSS and JS written twice.
    substitute() is the strict form, so a slot with no value is a KeyError at
    render and never an empty sidebar.
    """
    ctx = _frame_context(nodes, edges, communities, title, lib_js, discourse, zones)
    return string.Template(_asset("graph-frame.html")).substitute(ctx)


_EDGE_COLOR_GAP = "#E0A93B"  # --warn — "a bridge could go here, and doesn't"

# The similarity particles, pre-dimmed rather than made transparent. There are
# five gap links and roughly two thousand similar ones, so the same particle
# treatment would drown the frame; the effect has to survive at a fortieth of
# the weight. Alpha is not available to do that dimming: 3d-force-graph builds
# each photon as a mesh whose material takes a THREE.Color, which discards the
# alpha channel outright. So the colour is blended against the void here, once,
# and both renderers get an opaque colour that already looks faint.
#
# It is derived UPWARD from the line, not downward. The line is _EDGE_COLOR_SIMILAR
# at opacity 0.35, which over --void lands at about #08405E; a particle at or
# below that is a moving dot you cannot see. This sits a step above it — the same
# azure blended at 0.65 — which is the whole budget the effect gets: enough that
# the drift registers, not enough that two thousand of them become the subject.
_EDGE_COLOR_SIMILAR_PARTICLE = "#056E9A"  # one step up from the line's apparent #08405E

# The paper pair. Same derivation, opposite direction: a photon that cannot use
# alpha has to be blended against the floor it will sit on, and on paper that
# blend goes toward white. The similar particle is _EDGE_COLOR_SIMILAR_PAPER
# blended one step DOWN from its line's apparent value, because on a light floor
# a faint dot is a dark one — the reverse of the crystal case above and the
# reason this is a second constant rather than the same value reused.
_EDGE_COLOR_GAP_PAPER = "#7A5305"              # --warn, on paper
_EDGE_COLOR_SIMILAR_PARTICLE_PAPER = "#5CA8C4"

# Two more proposal layers, and one reading. All three are off by default: the
# frame opens on what you WROTE, and every layer that is a claim has to be asked
# for. Amber was not reused for the note-level proposals even though a gap and a
# structural link make the same kind of claim, because the legend is read by
# swatch before it is read by word, and two amber rows a line apart say "these
# are the same thing at two widths" - which is exactly the mistake a reader
# would then make about the 5 area bridges and the 12 note pairs.
#
# V1 (structural): green, the one hue the frame was not using. "The shape of
# what you already wrote predicts this link."
_EDGE_COLOR_PROPOSED = "#3FBF8F"
_EDGE_COLOR_PROPOSED_PAPER = "#0E6B4C"
# V3 (coupled): violet. A different KIND of claim - provenance, not topology -
# so it gets a hue rather than a weight.
_EDGE_COLOR_COUPLED = "#B06BD9"
_EDGE_COLOR_COUPLED_PAPER = "#5B2E86"
# V5 (discord): no hue at all. It recolours EXISTING wikilinks rather than
# adding edges - an overlay would have doubled the spring force on every link it
# marked, and 176 of a 709-note vault's 1340 links cross a zone boundary.
#
# Two hues were tried and both failed the same way. First _EDGE_COLOR_SIMILAR,
# on the argument that this row IS the semantic layer talking: same hex as the
# 2718 edges the similar layer was already painting, so ticking the box changed
# nothing a reader could see. Then magenta, chosen as "the widest gap left on
# the wheel, 40 degrees off the ambiguous red". The 40 was wrong - #E8559E sits
# at hue 330 and #e2544f at hue 2, which is 32 - and the degrees were the wrong
# measure anyway: the two also shared a saturation (delta 147 in both) and a
# luminance (0.477 against 0.446). Three channels agreeing is one colour, and on
# a 1px antialiased line that is what they were.
#
# The wheel had no seventh place to give. Six hues were already spoken for and
# every one of them has to survive being told apart from the other five, so a
# seventh is not a gap to find, it is a cost to stop paying. This layer can stop
# paying it: it is a HIGHLIGHT of links that are already on screen, not a
# category beside them, and "these, among those" is what luminance says. So it
# takes the extreme of that channel - brightest of all on the void, darkest of
# all on paper - which no hue on either floor can reach, today or after the next
# layer is added. Warm bone and not the labels' cool #EBEFF8, so a lit link and
# a note title stay two different things where they cross.
#
# Not red, whatever its loudness: the judge gate FAILED (docs/adr/0027), so this
# is a reading and never an alarm.
_EDGE_COLOR_DISCORD = "#F0E9DC"
_EDGE_COLOR_DISCORD_PAPER = "#17141F"


def _gap_edges(nodes: list[dict], edges: list[dict], top_k: int = 5) -> list[dict]:
    """Top structural gaps as overlay edges between two area hubs.

    Reads: 'these two well-formed areas should probably connect, and don't.'
    Reuses graph_export.structural_gaps so the overlay agrees with the /graph
    report's Structural Gaps section node-for-node. Only the keys 3d-force-graph
    actually honours: from/to (linkSource/linkTarget), color.color (linkColor),
    and type (visibility toggle + particle accessor). The lib draws these as
    amber directional-particle links — WebGL has no dashed line, so motion, not
    a dash pattern, is what sets a gap apart. score rides along for the title map.
    """
    from silica.kernel.recall.graph_export import structural_gaps

    return [
        {
            "id":    f"gap{i}",
            "from":  hub_a,
            "to":    hub_b,
            "type":  "GAP",
            # Five of them against four thousand others: a gap can afford to be
            # the most opaque and the widest thing on screen, because there is
            # never enough of it to crowd anything.
            "color": {"color": _EDGE_COLOR_GAP,
                      "paper": _EDGE_COLOR_GAP_PAPER, "opacity": 0.95},
            "width": 2.0,
            "score": score,
        }
        for i, (ca, cb, hub_a, hub_b, ie, score, _dens) in enumerate(
            structural_gaps(nodes, edges, top_k=top_k)
        )
    ]


def _proposed_edges(G_und, nodes: list[dict], top_k: int = 12) -> list[dict]:
    """Top Adamic-Adar pairs (V1) as overlay edges: "these two share
    neighbours and do not link".

    Corroborated only, which is the whole reason the layer is honest: a pair
    also has to share at least one CONCEPT in the co-occurrence index. As a
    fusion leg V1 was killed (0 wins against 417 losses on the golden pairs,
    ADR-0027); corroborated, it is the same rule the curator accepts an
    autolink under, so the layer and the curator cannot disagree about what
    is worth proposing.
    """
    from silica.kernel.recall.signals import structural_links

    ranked = structural_links(G_und, top_k=top_k * 4)
    return _corroborate(
        [(u, v, sc) for u, v, sc, _cm in ranked], nodes, top_k,
        kind="PROPOSED", color=_EDGE_COLOR_PROPOSED, paper=_EDGE_COLOR_PROPOSED_PAPER,
    )


def _coupled_edges(G_und, nodes: list[dict], top_k: int = 12) -> list[dict]:
    """Top coupled pairs (V3) as overlay edges: "written from the same source,
    or in the same run, and never linked".

    Same corroboration rule and the same assembly of transactions the report
    uses, so the layer cannot claim a pair the report does not carry.
    """
    from silica.config import CONFIG
    from silica.kernel.recall.signals import coupling_adjacency
    from silica.kernel.report.cowrite import coupling_transactions

    ids = {n["id"] for n in nodes if n.get("type") != "ghost"}
    transactions, _dropped = coupling_transactions(
        str(getattr(CONFIG, "vault_path", "") or ""), _sources_of(nodes), ids,
    )
    adj = coupling_adjacency(transactions)
    pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for a, row in adj.items():
        for b, w in row.items():
            key = (min(a, b), max(a, b))
            if key in seen or a not in ids or b not in ids or G_und.has_edge(a, b):
                continue
            seen.add(key)
            pairs.append((key[0], key[1], w))
    pairs.sort(key=lambda r: (-r[2], r[0], r[1]))
    return _corroborate(
        pairs[:top_k * 4], nodes, top_k,
        kind="COUPLED", color=_EDGE_COLOR_COUPLED, paper=_EDGE_COLOR_COUPLED_PAPER,
    )


def _sources_of(nodes: list[dict]) -> dict[str, list[str]]:
    """{node id: frontmatter `sources:`}, read straight off the files.

    compute_report harvests this during its body scan; the export has no such
    scan, and a 709-note frontmatter read measured 0.01 s, so it does its own
    rather than dragging the report in for one field.
    """
    import re

    from silica.config import CONFIG

    root = Path(str(getattr(CONFIG, "vault_path", "") or ""))
    if not root.is_dir():
        return {}
    out: dict[str, list[str]] = {}
    for n in nodes:
        if n.get("type") == "ghost":
            continue
        try:
            head = (root / n["id"]).read_text(encoding="utf-8", errors="replace")[:1200]
        except OSError:
            continue  # a note the graph knows and the disk no longer has
        m = re.search(r"^sources:\s*(.*?)(?=^\S|\Z)", head, re.M | re.S)
        if not m:
            continue
        vals = [v.strip().strip("\"'[]") for v in re.findall(r"^\s*-\s*(.+)$", m.group(1), re.M)]
        inline = m.group(1).strip()
        if not vals and inline:
            vals = [v.strip().strip("\"'[]") for v in inline.split(",")]
        vals = [v for v in vals if v]
        if vals:
            out[n["id"]] = vals
    return out


def _corroborate(
    pairs: list[tuple[str, str, float]], nodes: list[dict], top_k: int,
    *, kind: str, color: str, paper: str,
) -> list[dict]:
    """Keep only the pairs that also share a concept, as overlay edges.

    The co-occurrence index is what corroborates; without one the layer is
    EMPTY rather than uncorroborated, because an uncorroborated proposal is
    the thing the gates refuted. The store read itself belongs to
    graph_export, which already owns the frame's other label lookups.
    """
    from silica.kernel.recall.graph_export import shared_concepts

    shared_of = shared_concepts([(u, v) for u, v, _s in pairs])
    out: list[dict] = []
    for i, (u, v, score) in enumerate(pairs):
        labels = shared_of.get((u, v))
        if not labels:
            continue
        out.append({
            "id": f"{kind.lower()}{i}",
            "from": u,
            "to": v,
            "type": kind,
            "color": {"color": color, "paper": paper, "opacity": 0.75},
            "width": 1.2,
            "score": round(score, 4),
            "shared": labels,
        })
        if len(out) >= top_k:
            break
    return out


def _stamp_load_bearing(nodes: list[dict], G_und, bet: dict[str, float]) -> None:
    """Coreness, cut-vertex status and surprise onto the nodes themselves (V4).

    On the nodes and not as a layer, because these are facts ABOUT a note the
    way betweenness already is: the frame decides what to do with them (a ring,
    a hover line), and a note carries its own reading wherever it is drawn.
    `strands` is what a cut vertex costs, which is the number the curator's veto
    is really about and the only one that says whether a hold is worth honouring.

    Best-effort: a graph nx cannot read leaves every node exactly as it was.
    """
    from silica.kernel.recall.signals import load_bearing
    from silica.kernel.report.structure import cut_component_sizes

    try:
        core, articulation, surprise = load_bearing(
            G_und, betweenness=bet, degree=dict(G_und.degree()),
        )
        strands = cut_component_sizes(G_und, articulation)
    except Exception as exc:
        logger.warning("graph_view: load-bearing signals skipped (%s)", exc)
        return
    for n in nodes:
        if n.get("type") == "ghost":
            continue
        nid = n["id"]
        n["coreness"] = core.get(nid, 0)
        n["surprise"] = surprise.get(nid, 0.0)
        if nid in articulation:
            n["cut"] = True
            n["strands"] = strands.get(nid, 0)


def _mark_discord(edges: list[dict], nodes: list[dict]) -> None:
    """Flag the wikilinks whose two ends sit in DIFFERENT semantic zones (V5).

    A flag on the existing edge, never a new one: an overlay would have doubled
    the spring force on every link it marked, and ~40% of a vault's links cross
    a zone boundary. Nothing is flagged when there are no zones, which is the
    same abstention the per-note scalar makes.
    """
    zone = {n["id"]: n.get("sgroup", -1) for n in nodes}
    for e in edges:
        if e.get("type") != "EXTRACTED":
            continue
        a, b = zone.get(e.get("from"), -1), zone.get(e.get("to"), -1)
        if a >= 0 and b >= 0 and a != b:
            e["discord"] = True


def export_graph(
    output_path: str,
    folder: str = "",
    title: str = "Vault Graph",
    knn_k: int = 6,
) -> dict:
    """Build and write the unified vault-graph HTML to output_path.

    One build, two edge layers on a shared force layout:
      - the wikilink graph (EXTRACTED/AMBIGUOUS) — the explicit structure;
      - the embedding k-NN overlay (SIMILAR) — meaning-space proximity.
    Communities are Louvain on the WIKILINKS; the SIMILAR layer is a toggleable
    HUD overlay whose forces pull link-orphans (e.g. book extracts with no
    wikilinks) next to their semantic neighbours instead of leaving them
    floating. Structural-gap particles ride the wikilink layer.

    Reads the vendored JS first (fail fast on a packaging bug) and always inlines
    it, so the emitted file is self-contained/offline. Returns dict with keys:
    success, path, nodes, edges (wikilinks), similar (k-NN), communities,
    unresolved, gaps.
    """
    from silica.kernel.recall.graph_export import (
        build_graph_data,
        canvas_metrics,
        detect_communities,
        detect_semantic_partition,
        discourse_shape,
        knn_edges,
    )

    from silica.kernel.recall.graph_export import edge_graph

    lib_js = _vendored_lib_js()  # fail fast before the graph build
    nodes, edges = build_graph_data(folder=folder)   # wikilink edges (the structure)
    sim = knn_edges(nodes, k=knn_k)                   # embedding k-NN overlay
    communities = detect_communities(nodes, edges)   # Louvain on the wikilinks
    # ...and the second partition, on the same nodes: Louvain on the k-NN. It
    # writes node["sgroup"] only, so the structural colours above stand.
    zones = detect_semantic_partition(nodes, sim)

    # Betweenness → node size (bottleneck nodes swell) + discourse-shape badge,
    # from one shared nx build over the wikilinks. Base size 16 for ordinary nodes.
    bet, giant = canvas_metrics(nodes, edges)
    if bet:
        for n in nodes:
            if n.get("type") != "ghost":
                b = round(bet.get(n["id"], 0.0), 4)
                n["betweenness"] = b
                n["size"] = round(16 + 40 * b, 2)
    discourse = discourse_shape(
        sum(1 for n in nodes if n.get("type") != "ghost"),
        giant, [c.size for c in communities],
    )

    G_und = edge_graph(nodes, edges)
    _stamp_load_bearing(nodes, G_und, bet)
    _mark_discord(edges, nodes)

    # Gap particles ride the wikilink layer (they answer a linking question).
    gaps = _gap_edges(nodes, edges)
    # The two note-level proposal layers. Both are best-effort: a vault with no
    # co-occurrence index gets no layer and the same graph it always got, which
    # is why neither is allowed to raise into the export.
    proposed: list[dict] = []
    coupled: list[dict] = []
    try:
        proposed = _proposed_edges(G_und, nodes)
    except Exception as exc:
        logger.warning("graph_view: proposed layer skipped (%s)", exc)
    try:
        coupled = _coupled_edges(G_und, nodes)
    except Exception as exc:
        logger.warning("graph_view: coupled layer skipped (%s)", exc)

    html_out = render_html(
        nodes, edges + sim + gaps + proposed + coupled, communities, title=title,
        lib_js=lib_js, discourse=discourse, zones=zones,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out, encoding="utf-8")

    n_notes       = sum(1 for n in nodes if n.get("type") != "ghost")
    n_ghost       = sum(1 for n in nodes if n.get("type") == "ghost")
    n_links       = sum(1 for e in edges if e.get("type") == "EXTRACTED")
    n_similar     = len(sim)
    n_communities = len(communities)

    logger.info(
        "graph_export: wrote %s — %d notes, %d links, %d similar, %d clusters, "
        "%d zones, %d unresolved",
        out, n_notes, n_links, n_similar, n_communities, len(zones), n_ghost,
    )
    return {
        "success":     True,
        "path":        str(out.resolve()),
        "nodes":       n_notes,
        "edges":       n_links,
        "similar":     n_similar,
        # Two counts, two names: `communities` is the structural partition,
        # `zones` the semantic one. Never summed, never swapped (ADR-0023).
        "communities": n_communities,
        "zones":       len(zones),
        "unresolved":  n_ghost,
        "gaps":        len(gaps),
    }
