"""Tests for render_html() in silica/kernel/recall/graph_export.py.

Exercises the 3d-force-graph renderer output without any network access.
render_html() accepts lib_js as a string parameter, so we can pass a dummy
string or "" to avoid CDN fetches entirely.
"""
from __future__ import annotations

import json

import pytest

from silica.ui.web.graph_view import render_html


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

def _node(nid: str, label: str = "", group: int = -1, node_type: str = "note") -> dict:
    return {
        "id": nid,
        "label": label or nid,
        "type": node_type,
        "group": group,
        "color": {"background": "#2d4a6e", "border": "#4a9eff"},
        "path": nid,
        "size": 16,
    }


def _edge(eid: str, src: str, dst: str, etype: str = "EXTRACTED") -> dict:
    color = "#4a9eff" if etype == "EXTRACTED" else "#ffaa33"
    return {
        "id": eid,
        "from": src,
        "to": dst,
        "type": etype,
        "color": {"color": color, "opacity": 0.6},
        "width": 1.5,
    }


@pytest.fixture()
def small_graph():
    nodes = [_node("A"), _node("B"), _node("C", node_type="ghost")]
    edges = [
        _edge("e0", "A", "B", "EXTRACTED"),
        _edge("e1", "A", "C", "AMBIGUOUS"),
    ]
    return nodes, edges


# ---------------------------------------------------------------------------
# 1. Output contains ForceGraph3D(
# ---------------------------------------------------------------------------

class TestForceGraph3DPresent:
    def test_contains_forcegraph3d_constructor(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "ForceGraph3D(" in html

    def test_forcegraph3d_present_with_empty_lib_js(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="")
        assert "ForceGraph3D(" in html

    def test_uses_constructor_form_not_legacy_curried(self, small_graph):
        # 3d-force-graph >= 1.x uses `new ForceGraph3D(element)`. The legacy
        # curried `ForceGraph3D()(element)` form throws at runtime in 1.80.0,
        # leaving the graph area blank. Lock the constructor form in.
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "new ForceGraph3D(" in html
        assert "ForceGraph3D()(" not in html


# ---------------------------------------------------------------------------
# 2. linkSource("from") and linkTarget("to") are present
# ---------------------------------------------------------------------------

class TestLinkSourceTarget:
    def test_link_source_from(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert '.linkSource("from")' in html

    def test_link_target_to(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert '.linkTarget("to")' in html


# ---------------------------------------------------------------------------
# 3. Inline bundle vs CDN fallback
# ---------------------------------------------------------------------------

class TestLibJsInlining:
    def test_inline_when_lib_js_provided(self, small_graph):
        """When lib_js is non-empty, bundle is inlined as <script>…</script>."""
        nodes, edges = small_graph
        bundle = "/* 3d-force-graph bundle */"
        html = render_html(nodes, edges, lib_js=bundle)
        assert f"<script>{bundle}</script>" in html

    def test_never_a_cdn_script_src(self, small_graph):
        """The bundle is always inlined — no <script src= network dependency,
        even with an empty lib_js (the CDN fallback is gone)."""
        nodes, edges = small_graph
        for lib_js in ("/* bundle */", ""):
            html = render_html(nodes, edges, lib_js=lib_js)
            assert "<script src=" not in html


# ---------------------------------------------------------------------------
# 4. No vis.Network or new vis.DataSet in output
# ---------------------------------------------------------------------------

class TestNoVisReferences:
    def test_no_vis_network(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "vis.Network" not in html

    def test_no_vis_dataset(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "new vis.DataSet" not in html

    def test_no_vis_network_cdn_path(self):
        """Even with empty lib_js (CDN mode), no vis-network CDN URL appears."""
        html = render_html([], [], lib_js="")
        assert "vis-network" not in html


# ---------------------------------------------------------------------------
# Additional sanity checks
# ---------------------------------------------------------------------------

class TestRenderSanity:
    def test_title_appears_in_output(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, title="My Test Graph", lib_js="// x")
        assert "My Test Graph" in html

    def test_graph_data_json_embedded(self, small_graph):
        """RAW_NODES and RAW_EDGES constants should appear in the output."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "RAW_NODES" in html
        assert "RAW_EDGES" in html

    def test_outdeg_indeg_precompute_present(self, small_graph):
        """Degree precompute block should be present."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "outDeg[e.from]" in html
        assert "inDeg[e.to]" in html

    def test_fit_button_uses_graph_zoom(self, small_graph):
        """Fit graph button should call Graph.zoomToFit, not network.fit."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "Graph.zoomToFit(400, 40)" in html
        assert "network.fit(" not in html

    def test_node_visibility_accessor(self, small_graph):
        """nodeVisibility accessor should be wired up."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ".nodeVisibility(" in html

    def test_link_visibility_accessor(self, small_graph):
        """linkVisibility accessor should be wired up."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ".linkVisibility(" in html

    def test_visibility_refresh_trick_present(self, small_graph):
        """applyFilters() should use the re-pass trick to force a visibility refresh."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "Graph.nodeVisibility(Graph.nodeVisibility())" in html

    def test_on_node_click_used(self, small_graph):
        """Drawer open should use onNodeClick, not network.on click."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "onNodeClick" in html
        assert 'network.on("click"' not in html

    def test_on_background_click_closes_drawer_and_clears_focus(self, small_graph):
        """Background tap closes the drawer AND reverts the focus/dim state.

        (A camera-orbit drag never reaches here: the vendored bundle sets
        clickAfterDrag=false, so onBackgroundClick fires only on a clean tap.)
        """
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "onBackgroundClick(" in html
        assert "closeDrawer()" in html
        assert "clearFocus()" in html

    def test_empty_graph_renders(self):
        """render_html with empty node/edge lists should not raise."""
        html = render_html([], [], lib_js="// x")
        assert "<!DOCTYPE html>" in html


# ---------------------------------------------------------------------------
# Search → results list → fly-to-focus (findability for the searching user)
# ---------------------------------------------------------------------------

class TestSearchResultsFlyTo:
    """Typing a query should produce a clickable ranked list, and choosing a
    result should fly the camera to that node and select it — not just dim the
    rest of the cloud."""

    def test_results_container_present(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert 'id="search-results"' in html

    def test_onsearch_renders_results(self, small_graph):
        """onSearch should populate the results list, not only set a filter."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "renderResults(" in html

    def test_scorer_searches_beyond_label(self, small_graph):
        """Ranking should consider path and tags, not just the label."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "function scoreNode(" in html
        assert ".path" in html

    def test_focus_node_uses_camera_position(self, small_graph):
        """Choosing a result flies the camera via the 3d-force-graph API."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "function focusNode(" in html
        assert ".cameraPosition(" in html

    def test_select_node_shared_between_click_and_result(self, small_graph):
        """Node-click and result-click should both route through selectNode."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "function selectNode(" in html

    def test_enter_focuses_top_result(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "onSearchKey(" in html

    def test_embedded_node_click_posts_open_context_to_parent(self, small_graph):
        """When embedded in the web-UI iframe, a node click hands off to the
        parent's drawer instead of opening the internal metadata drawer — and to
        its CONTEXT mode: pointing at a node asks "what is this", where naming a
        note (a wikilink, the file tree) asks to read it. A ghost node rides the
        same message, since it has no path and no reader."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "window.parent !== window" in html
        assert "postMessage" in html
        assert "silica-open-context" in html
        assert 'ghost: node.type === "ghost"' in html


class TestFocusDim:
    def test_neighbors_map_precomputed(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "neighbors" in html

    def test_apply_and_clear_focus_present(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "function applyFocus(" in html
        assert "function clearFocus(" in html

    def test_node_color_has_dim_branch(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "_dim" in html
        assert "#1d192f" in html  # unlit blue-violet; the palette has no gray

    def test_choose_node_applies_focus(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "applyFocus(node.id)" in html

    def test_clear_focus_zooms_to_fit(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "Graph.zoomToFit(600, 40)" in html

    def test_direct_node_click_dims_without_camera_fly(self, small_graph):
        """Clicking a node in the view itself dims non-neighbours like tree/search
        picks, but must NOT call focusNode — the user is already looking at the
        spot. Bound inside buildGraph, so a 2D/3D switch rebinds it. The
        MouseEvent rides along because selectNode reads its click counter: one
        click points at a node, two mean read it."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ".onNodeClick((node, event) => {" in html
        assert "selectNode(node, event); applyFocus(node.id);" in html

    def test_parent_can_sync_focus_by_path(self, small_graph):
        """The embedding page (note-panel navigation) can tell the graph which
        note is open elsewhere so it mirrors the dim state, without moving the camera."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "silica-focus-path" in html
        assert "NODE_BY_ID[e.data.path]" in html


class TestFileTree:
    def test_file_tree_container_present(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert 'id="file-tree"' in html

    def test_files_section_title_present(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ">Files<" in html

    def test_tree_leaf_for_real_note(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        # node "A" is a real note (id == path == "A")
        assert 'class="tree-note" data-id="A"' in html

    def test_node_by_id_map_built(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "NODE_BY_ID" in html

    def test_choose_node_function_present(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "function chooseNode(" in html

    def test_tree_click_delegated(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert 'getElementById("file-tree").addEventListener("click"' in html

    def test_choose_result_routes_through_choose_node(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "chooseNode(" in html


# ---------------------------------------------------------------------------
# 5. Perf knobs for big vaults — keep WebGL geometry count low
# ---------------------------------------------------------------------------

class TestBigVaultPerfKnobs:
    """1200-node vaults lag because 3d-force-graph defaults turn every edge into
    a cylinder + arrow-cone mesh and never stop the layout. Lock the cheap path.
    """

    def test_links_are_zero_width_gl_lines(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ".linkWidth(0)" in html

    def test_no_directional_arrow_cones(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "linkDirectionalArrowLength" not in html

    def test_layout_stops_on_alpha_not_on_a_tick_count(self, small_graph):
        """The freeze gate is the physics, never a counted budget.

        A tick count was the original gate and it cut the layout off at alpha
        0.028, twenty-eight times above the 0.001 d3 calls converged: the view
        froze on a half-unfolded graph. Both ceilings are deliberately infinite
        — alpha decays deterministically, so it always converges, and either
        ceiling can only re-introduce the early cut through another door.
        """
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ".d3AlphaMin(ALPHA_MIN)" in html
        assert "const ALPHA_MIN = 0.001" in html
        assert ".cooldownTicks(Infinity)" in html
        assert ".cooldownTime(Infinity)" in html

    def test_forces_are_set_before_the_data_lands(self, small_graph):
        """graphData runs the warmup loop, so late forces shape only the tail.

        With the tuned forces applied after graphData, the bulk of the layout
        was being built by d3's untouched defaults.
        """
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert html.index("applyForces(false, G)") < html.index("G.graphData(")

    def test_low_node_resolution(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ".nodeResolution(" in html

    def test_dynamic_resolution_governor_wired(self, small_graph):
        """Streaming 2D repaints drop the backing store to half ratio.

        The 2D canvas is fill-bound (~15ms per megapixel on software raster):
        a hot simulation on a full-size canvas delivered 15-22fps with the
        main thread idle, rAF starved by raster back-pressure. The governor
        overrides window.devicePixelRatio and watches paint bursts from
        onRenderFramePre; dropping any of the three pieces silently restores
        the jank, so pin them all.
        """
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert 'Object.defineProperty(window, "devicePixelRatio"' in html
        assert "drsOnPaint(); drawZones(ctx, scale);" in html
        # the crisp restore must repaint AND re-derive the budget state
        assert "Graph.resumeAnimation(); Graph.pauseAnimation(); renderBudget();" in html
        # the lib's resize path re-centers using old-pixels / NEW-ratio, wrong by
        # exactly the ratio step: without the camera save/restore every res
        # transition lurches the view half a screen (shipped once, 2026-08-17)
        assert html.index("const z = Graph.zoom(), c = Graph.centerAt();") \
            < html.index("Graph.width(Graph.width()).height(Graph.height());") \
            < html.index("Graph.centerAt(c.x, c.y);")
        # pointer-driven camera work engages BEFORE the first heavy paint;
        # leaving it to the 3-paint burst warm-up made every pan resume chop
        # for 150-270ms at full res (user-reported, 2026-08-17)
        assert '(e.type === "pointermove" && e.buttons)' in html


# ---------------------------------------------------------------------------
# The host page's note drawer overlays this frame's right edge, where the HUD
# sits, and the drawer is translucent. Both halves of the fix live inside an
# f-string template, so a brace-escaping slip would drop them silently and the
# legend would read through the open note again.
# ---------------------------------------------------------------------------

class TestHostDrawerHidesTheHud:
    def test_css_rule_survives_the_template(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="")
        assert "body.host-drawer-open #hud{display:none}" in html

    def test_listener_toggles_the_class(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="")
        assert 'e.data.type === "silica-host-drawer"' in html
        assert 'classList.toggle("host-drawer-open", !!e.data.open)' in html


class TestDragReheat:
    """d3AlphaMin gates tick(), and alpha only rises inside tick().

    So a settled graph could never answer a drag: the bundles reheat with
    d3AlphaTarget(0.3).resetCountdown(), the gate saw the old alpha and stopped
    the engine on the same frame, and only the grabbed node moved. The pair of
    handlers below is the whole fix — lifting the gate without restoring it
    would leave the layout running forever instead.
    """

    def test_drag_lifts_the_alpha_gate_and_dragend_puts_it_back(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="")
        assert ".onNodeDrag(" in html
        assert "G.d3AlphaMin(0);" in html
        assert ".onNodeDragEnd(() => G.d3AlphaMin(ALPHA_MIN))" in html


# ---------------------------------------------------------------------------
# Display settings baked in at render time (settings.py Display section). The
# flags are read from CONFIG when the document is built, so a change only lands
# on the next /graph request — which is what makes these two island fields the
# whole contract between the panel and the viewer.
# ---------------------------------------------------------------------------

def _island(html: str) -> dict:
    """The document's data, as graph.js parses it."""
    start = html.index('<script id="graph-data"')
    start = html.index(">", start) + 1
    return json.loads(html[start:html.index("</script>", start)])


class TestGraphEffectToggles:
    def test_on_by_default(self, small_graph, monkeypatch):
        from silica.config import CONFIG

        # Set, not assumed: CONFIG is read live, so a machine whose panel (or
        # SILICA_GRAPH_PARTICLES) has the effects off failed this on the default
        # rather than on the render.
        nodes, edges = small_graph
        monkeypatch.setattr(CONFIG, "graph_particles", True)
        monkeypatch.setattr(CONFIG, "graph_shading", True)
        html = render_html(nodes, edges, lib_js="")
        assert _island(html)["particles"] is True
        assert _island(html)["shading"] is True
        assert "const PARTICLES = DATA.particles;" in html
        assert "const SHADING = DATA.shading;" in html

    def test_off_reaches_the_gates(self, small_graph, monkeypatch):
        from silica.config import CONFIG

        nodes, edges = small_graph
        monkeypatch.setattr(CONFIG, "graph_particles", False)
        monkeypatch.setattr(CONFIG, "graph_shading", False)
        html = render_html(nodes, edges, lib_js="")
        assert _island(html)["particles"] is False
        assert _island(html)["shading"] is False
        # The constants are only worth anything if the three call sites read them.
        assert "(!PARTICLES || l._dim || l._hidden) ? 0" in html
        assert "PARTICLES && RAW_EDGES.some" in html
        assert html.count("if (!SHADING || is2D()") == 2


class TestCameraFitAndWarmup:
    """The 2D -> 3D switch used to land the camera inside the node cloud.

    zoomToFit measures the SCENE, not the data: 3d-force-graph unions the node
    meshes' world boxes, and a mesh only takes its node's position on a rendered
    frame. The fit was deferred to onEngineStop, and with a cached layout the
    engine stopped inside the SYNCHRONOUS warmup (FAST_DECAY reaches ALPHA_MIN
    at tick 66, warmupTicks was 150 in 3D and 240 in 2D), so it fired before the
    first frame existed. getGraphBbox then returned the node radii alone
    (measured +/-14 units against real positions spanning +/-2300), and the
    camera was placed ~41 units from the centre of a ~3900-unit graph.
    """

    def test_a_seeded_layout_runs_no_synchronous_warmup(self, small_graph):
        nodes, edges = small_graph
        js = render_html(nodes, edges, lib_js="// dummy")
        assert "function WARMUP_TICKS(seeded)" in js, \
            "warmup must be able to see whether the positions were seeded"
        assert "if (seeded) return 0;" in js, \
            "a seeded layout must not also run a warmup: the cache IS the warmup, " \
            "and running one consumes the whole alpha schedule before any frame"

    def test_the_deferred_fit_waits_for_a_painted_frame(self, small_graph):
        nodes, edges = small_graph
        js = render_html(nodes, edges, lib_js="// dummy")
        assert "function fitWhenPainted(" in js
        assert "requestAnimationFrame" in js[js.index("function fitWhenPainted("):][:600], \
            "the fit must run on the far side of a real frame, or it measures meshes at the origin"
        stop = js[js.index(".onEngineStop("):][:400]
        assert "fitWhenPainted(" in stop, "onEngineStop still fits directly"
        assert "G.zoomToFit(" not in stop, "onEngineStop still calls zoomToFit before a frame exists"

    def test_the_warmup_freeze_is_bounded_by_graph_size(self, small_graph):
        """The warmup is synchronous: every tick in it is a frame the browser
        cannot paint. 240 ticks cost ~0.8s at the ~4.5k links these counts were
        tuned on, and ~2.1s at 12.2k (measured 8.94ms/tick). Hold the freeze,
        not the tick count."""
        nodes, edges = small_graph
        js = render_html(nodes, edges, lib_js="// dummy")
        assert "WARMUP_REF_EDGES" in js, "warmup does not scale with the graph it runs on"
