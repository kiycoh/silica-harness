"""Tests for Task 2: named community labels in graph_export.

Tests:
1. Fallback to "Cluster N" when cooccurrence index is empty.
2. Named labels from a populated cooccurrence store (via monkeypatch).
3. HTML-escape of labels in render_html legend and COMM_LABELS JS map.
4. Existing render_html signature still works (communities defaults to ()).
"""
from __future__ import annotations

import json
from unittest.mock import patch


from silica.kernel.recall.graph_export import (
    Community,
    _community_color,
    detect_communities,
)
from silica.ui.web.graph_view import render_html


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(nid: str, node_type: str = "note") -> dict:
    return {
        "id": nid,
        "label": nid,
        "type": node_type,
        "group": -1,
        "color": {"background": "#2d4a6e", "border": "#4a9eff"},
        "path": nid,
        "size": 16,
    }


def _edge(eid: str, src: str, dst: str, etype: str = "EXTRACTED") -> dict:
    return {
        "id": eid,
        "from": src,
        "to": dst,
        "type": etype,
        "color": {"color": "#4a9eff", "opacity": 0.6},
        "width": 1.5,
    }


# ---------------------------------------------------------------------------
# Test 1 — fallback to "Cluster N" when index is empty
# ---------------------------------------------------------------------------

class TestCommunityFallback:
    """detect_communities returns list[Community] with Cluster-N labels when
    the cooccurrence store has no data for the node paths."""

    def test_returns_list_not_none(self):
        nodes = [_node("A"), _node("B"), _node("C"), _node("D")]
        edges = [_edge("e0", "A", "B"), _edge("e1", "C", "D")]
        result = detect_communities(nodes, edges)
        assert isinstance(result, list)

    def test_communities_have_correct_size(self):
        nodes = [_node("A"), _node("B"), _node("C"), _node("D")]
        edges = [_edge("e0", "A", "B"), _edge("e1", "C", "D")]
        result = detect_communities(nodes, edges)
        # Should detect 2 communities (two disconnected pairs)
        assert len(result) == 2
        sizes = {c.size for c in result}
        assert sizes == {2}

    def test_fallback_labels_are_cluster_n(self):
        """Empty index means community_labels returns {} → label = 'Cluster N'.

        CooccurStore is patched to return {} so the test is hermetic regardless
        of any on-disk index that may have accumulated real data.
        """
        nodes = [_node("A"), _node("B"), _node("C"), _node("D")]
        edges = [_edge("e0", "A", "B"), _edge("e1", "C", "D")]
        with patch(
            "silica.kernel.recall.cooccurrence.CooccurStore.community_labels",
            return_value={},
        ):
            result = detect_communities(nodes, edges)
        for c in result:
            assert c.label == f"Cluster {c.id}"

    def test_community_colors_come_from_palette(self):
        nodes = [_node("A"), _node("B"), _node("C"), _node("D")]
        edges = [_edge("e0", "A", "B"), _edge("e1", "C", "D")]
        result = detect_communities(nodes, edges)
        for c in result:
            expected_color = _community_color(c.id)
            assert c.color == expected_color

    def test_community_dataclass_fields(self):
        nodes = [_node("A"), _node("B"), _node("C"), _node("D")]
        edges = [_edge("e0", "A", "B"), _edge("e1", "C", "D")]
        result = detect_communities(nodes, edges)
        for c in result:
            assert isinstance(c, Community)
            assert isinstance(c.id, int)
            assert isinstance(c.label, str)
            assert isinstance(c.color, str)
            assert isinstance(c.size, int)

    def test_early_return_no_edges_returns_empty_list(self):
        """No EXTRACTED edges → early return should be [] not None."""
        nodes = [_node("A"), _node("B")]
        edges = []
        result = detect_communities(nodes, edges)
        assert result == []

    def test_per_node_group_still_assigned(self):
        """detect_communities must still mutate node['group'] in place."""
        nodes = [_node("A"), _node("B"), _node("C"), _node("D")]
        edges = [_edge("e0", "A", "B"), _edge("e1", "C", "D")]
        detect_communities(nodes, edges)
        real_nodes = [n for n in nodes if n["type"] != "ghost"]
        assert all(n["group"] >= 0 for n in real_nodes)


# ---------------------------------------------------------------------------
# Test 2 — named labels from a mocked community_labels
# ---------------------------------------------------------------------------

class TestNamedLabels:
    """Use monkeypatch so CooccurStore.community_labels returns a known label."""

    def test_label_comes_from_community_labels(self):
        nodes = [_node("A"), _node("B"), _node("C"), _node("D")]
        edges = [_edge("e0", "A", "B"), _edge("e1", "C", "D")]

        with patch(
            "silica.kernel.recall.cooccurrence.CooccurStore.community_labels",
            return_value={0: "Kernel · Ledger", 1: "Notes · Graph"},
        ):
            result = detect_communities(nodes, edges)

        labels = {c.id: c.label for c in result}
        assert labels[0] == "Kernel · Ledger"
        assert labels[1] == "Notes · Graph"

    def test_partial_labels_fallback_to_cluster_n(self):
        """community_labels may omit some indices; those fall back to Cluster N."""
        nodes = [_node("A"), _node("B"), _node("C"), _node("D")]
        edges = [_edge("e0", "A", "B"), _edge("e1", "C", "D")]

        with patch(
            "silica.kernel.recall.cooccurrence.CooccurStore.community_labels",
            return_value={0: "Kernel · Ledger"},  # only community 0 named
        ):
            result = detect_communities(nodes, edges)

        labels = {c.id: c.label for c in result}
        assert labels[0] == "Kernel · Ledger"
        assert labels[1] == "Cluster 1"

    def test_exception_in_community_labels_degrades_to_cluster_n(self):
        """Any exception from community_labels → degrade to Cluster N."""
        nodes = [_node("A"), _node("B"), _node("C"), _node("D")]
        edges = [_edge("e0", "A", "B"), _edge("e1", "C", "D")]

        with patch(
            "silica.kernel.recall.cooccurrence.CooccurStore.community_labels",
            side_effect=RuntimeError("index corrupt"),
        ):
            result = detect_communities(nodes, edges)

        for c in result:
            assert c.label == f"Cluster {c.id}"


# ---------------------------------------------------------------------------
# Test 3 — HTML escaping and COMM_LABELS in render_html
# ---------------------------------------------------------------------------

def _legend(rendered: str) -> str:
    return rendered[rendered.index('<div id="legend-box">'):rendered.index('id="legend-all"')]


def _island_text(rendered: str) -> str:
    """The raw body of the graph-data island, before JSON.parse."""
    start = rendered.index('<script id="graph-data"')
    start = rendered.index(">", start) + 1
    return rendered[start:rendered.index("</script>", start)]


class TestRenderHtmlCommunities:
    def _make_community(self, cid: int, label: str, size: int = 2) -> Community:
        return Community(
            id=cid,
            label=label,
            color=_community_color(cid),
            size=size,
        )

    def test_html_escaped_label_in_legend(self):
        """Label 'A & B' must appear as 'A &amp; B' in the HTML legend (full output)."""
        communities = [self._make_community(0, "A & B")]
        rendered = render_html([], [], communities=communities, lib_js="// x")
        # The legend uses html.escape — verify the escaped form is present
        assert "A &amp; B" in rendered
        # The raw unescaped '&' must not appear in the legend itself
        assert "A & B" not in _legend(rendered)

    def test_html_escaped_lt_gt(self):
        """Labels with < and > must be escaped in the legend (full output)."""
        communities = [self._make_community(0, "<topic>")]
        rendered = render_html([], [], communities=communities, lib_js="// x")
        assert "&lt;topic&gt;" in rendered
        # Raw < must not appear in the legend itself
        assert "<topic>" not in _legend(rendered)

    def test_xss_script_injection_blocked(self):
        """A label containing '</script>' must not end the data island early.

        Without mitigation, a label like 'x</script><script>alert(1)//' would
        terminate the enclosing <script> element and inject markup. The island
        emits every '<' as \\u003c, which is still JSON and the same string
        again after JSON.parse, but never a tag to the HTML parser.
        """
        evil_label = "x</script><script>alert(1)//"
        communities = [self._make_community(0, evil_label)]
        rendered = render_html([], [], communities=communities, lib_js="// x")

        island = _island_text(rendered)
        assert "</script>" not in island and "<script>" not in island, (
            "raw tag found inside the graph-data island — XSS vector open"
        )
        assert "\\u003c/script>" in island, "the < escape was not applied"
        # and the label survives the round trip the browser makes
        assert json.loads(island)["comm_labels"]["0"] == evil_label

    def test_comm_labels_js_map_present(self):
        """COMM_LABELS JS constant must appear in the script block."""
        communities = [
            self._make_community(0, "Kernel · Ledger"),
            self._make_community(1, "Notes · Graph"),
        ]
        html = render_html([], [], communities=communities, lib_js="// x")
        assert "COMM_LABELS" in html

    def test_comm_labels_contains_label_text(self):
        """The COMM_LABELS map must carry the label strings (JSON-encoded)."""
        communities = [self._make_community(0, "Kernel · Ledger")]
        html = render_html([], [], communities=communities, lib_js="// x")
        assert "Kernel · Ledger" in html

    def test_legend_shows_community_label(self):
        """Legend should display the named label, not 'Cluster 0'."""
        communities = [self._make_community(0, "Kernel · Ledger")]
        html = render_html([], [], communities=communities, lib_js="// x")
        assert "Kernel · Ledger" in html
        # 'Cluster 0' should NOT appear in legend when a named label is provided
        assert "Cluster 0" not in html

    def test_legend_shows_community_size(self):
        """Legend must show the community size."""
        communities = [self._make_community(0, "Kernel · Ledger", size=5)]
        html = render_html([], [], communities=communities, lib_js="// x")
        assert ">5<" in html

    def test_legend_item_carries_data_size(self):
        """Each legend item needs its size in the DOM so the sort button can read it."""
        communities = [self._make_community(0, "Kernel · Ledger", size=5)]
        html = render_html([], [], communities=communities, lib_js="// x")
        assert 'data-size="5"' in html

    def test_sort_communities_button_present(self):
        """A button toggles the communities legend between size-ascending/descending."""
        communities = [self._make_community(0, "A", size=1), self._make_community(1, "B", size=2)]
        html = render_html([], [], communities=communities, lib_js="// x")
        assert 'id="sort-communities"' in html
        assert "function toggleCommunitySort(" in html
        assert "onclick=\"toggleCommunitySort()\"" in html

    def test_comm_text_uses_comm_labels_not_cluster_id(self):
        """The onNodeClick drawer must use COMM_LABELS[node.group], not cluster N."""
        communities = [self._make_community(0, "Topic")]
        html = render_html([], [], communities=communities, lib_js="// x")
        assert "COMM_LABELS[node.group]" in html

    def test_empty_communities_no_error(self):
        """render_html with communities=() must not raise."""
        html = render_html([], [], communities=(), lib_js="// x")
        assert "<!DOCTYPE html>" in html

    def test_render_html_old_signature_still_works(self):
        """render_html called without communities kwarg must not raise."""
        html = render_html([], [], lib_js="// x")
        assert "<!DOCTYPE html>" in html

    def test_node_label_xss_escape(self):
        """A node whose label contains '</script>' must not produce a raw
        </script> inside the data island the nodes ride in."""
        nodes = [
            {
                "id": "a</script>b.md",
                "label": "a</script>b",
                "type": "note",
                "group": -1,
                "color": {"background": "#2d4a6e", "border": "#4a9eff"},
                "path": "a</script>b.md",
                "size": 16,
            }
        ]
        rendered = render_html(nodes, [], lib_js="// x")
        island = _island_text(rendered)
        assert "</script>" not in island, "raw </script> in the island — XSS vector open"
        assert "\\u003c/script>" in island, "the < escape was not applied"
        assert json.loads(island)["nodes"][0]["label"] == "a</script>b"


# ---------------------------------------------------------------------------
# Test 4 — non-mocked integration: .md suffix stripping at the cooccur seam
# ---------------------------------------------------------------------------

class TestCommunityLabelsIntegration:
    """Guard test for Fix 1: node ids in detect_communities carry .md, but the
    CooccurStore is keyed WITHOUT .md.  Without the fix, community_labels gets
    the wrong keys, note_nodes() misses on all of them, and every cluster falls
    back to 'Cluster N'."""

    def test_md_suffix_stripped_before_cooccur_lookup(self, tmp_path, monkeypatch):
        """Community labels are resolved from a real CooccurStore seeded with
        stem keys (no .md), while the graph nodes have .md-suffixed ids.

        This test MUST fail without Fix 1 (the .removesuffix call) and pass
        with it.
        """
        import silica.kernel.recall.cooccurrence as cooc_mod

        # Redirect the default index to a private tmp file for this test
        index_path = tmp_path / "test_cooccur.json"
        monkeypatch.setattr(cooc_mod, "_index_path", lambda: index_path)

        from silica.kernel.recall.cooccurrence import CooccurStore

        # Seed keyed WITHOUT .md — this is the production convention
        store = CooccurStore(path=index_path)
        store.upsert_note("note_a", {
            "nodes": {
                "kernel": {"label": "kernel", "count": 5},
                "ledger": {"label": "ledger", "count": 3},
            },
            "edges": [["kernel", "ledger", 4.0]],
        })
        store.upsert_note("note_b", {
            "nodes": {
                "kernel": {"label": "kernel", "count": 4},
                "router": {"label": "router", "count": 2},
            },
            "edges": [["kernel", "router", 3.0]],
        })
        store.save()

        # Graph nodes carry .md suffixes — the production shape
        nodes = [_node("note_a.md"), _node("note_b.md")]
        edges = [_edge("e0", "note_a.md", "note_b.md")]

        result = detect_communities(nodes, edges)

        # At least one community must get a concept-derived label, not "Cluster N"
        assert result, "detect_communities returned no communities"
        labels = {c.label for c in result}
        cluster_n_labels = {f"Cluster {c.id}" for c in result}
        assert labels != cluster_n_labels, (
            f"All communities fell back to Cluster N: {labels}. "
            "Fix 1 (.removesuffix) is not applied."
        )
