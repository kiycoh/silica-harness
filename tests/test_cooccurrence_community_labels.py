"""Tests for CooccurStore.community_labels (pure c-TF-IDF method)."""
from __future__ import annotations

from pathlib import Path


from silica.kernel.recall.cooccurrence import CooccurStore


def _make_store(tmp_path: Path, notes: dict[str, dict]) -> CooccurStore:
    """Build a CooccurStore with explicit path; upsert notes from hand-built contributions."""
    store = CooccurStore(path=tmp_path / "test_cooccur.json")
    for path, contrib in notes.items():
        store.upsert_note(path, contrib)
    return store


# ---------------------------------------------------------------------------
# Case 1 — Distinctive term beats shared generic term (TF-IDF flip)
# ---------------------------------------------------------------------------
def test_distinctive_term_beats_shared_generic(tmp_path):
    """TF-IDF should rank distinctive stems above shared high-count stems.

    Worked example (N=2):
      'gener' count 10 in both comms → df=2 → score 10·ln(1+2/2)=10·ln(2)≈6.93
      'ledger' count 8 in comm 0 only → df=1 → score 8·ln(1+2/1)=8·ln(3)≈8.79
      'graph'  count 8 in comm 1 only → df=1 → score 8·ln(3)≈8.79
    → comm 0 label leads with 'ledger', comm 1 leads with 'graph'
    → neither label should lead with the generic surface.
    """
    # Hand-built contributions: nodes only (edges not needed for community_labels)
    store = _make_store(
        tmp_path,
        {
            "note_a.md": {
                "nodes": {
                    "gener": {"label": "generic", "count": 10},
                    "ledger": {"label": "ledger", "count": 8},
                },
                "edges": [],
            },
            "note_b.md": {
                "nodes": {
                    "gener": {"label": "generic", "count": 10},
                    "graph": {"label": "graph", "count": 8},
                },
                "edges": [],
            },
        },
    )

    communities = [{"note_a.md"}, {"note_b.md"}]
    labels = store.community_labels(communities, terms=2)

    assert 0 in labels, "Community 0 should have a label"
    assert 1 in labels, "Community 1 should have a label"

    # The distinctive term must appear first (leading the label)
    comm0_parts = labels[0].split(" · ")
    comm1_parts = labels[1].split(" · ")

    assert comm0_parts[0] == "ledger", (
        f"Comm 0 should lead with 'ledger', got: {labels[0]!r}"
    )
    assert comm1_parts[0] == "graph", (
        f"Comm 1 should lead with 'graph', got: {labels[1]!r}"
    )

    # Generic surface must NOT lead either label
    assert comm0_parts[0] != "generic", "Generic should not lead comm 0"
    assert comm1_parts[0] != "generic", "Generic should not lead comm 1"


# ---------------------------------------------------------------------------
# Case 2 — Empty store → {} (communities reference unknown paths)
# ---------------------------------------------------------------------------
def test_empty_store_returns_empty_dict(tmp_path):
    """If no notes are indexed, all communities are empty → return {}."""
    store = CooccurStore(path=tmp_path / "empty.json")
    communities = [{"ghost_a.md"}, {"ghost_b.md"}]
    result = store.community_labels(communities)
    assert result == {}


# ---------------------------------------------------------------------------
# Case 3 — Graceful partial: one indexed + one absent note
# ---------------------------------------------------------------------------
def test_partial_community_uses_indexed_note_only(tmp_path):
    """Community with one indexed + one absent note → label from indexed note, no crash."""
    store = _make_store(
        tmp_path,
        {
            "real.md": {
                "nodes": {
                    "concept": {"label": "concept", "count": 5},
                },
                "edges": [],
            },
        },
    )
    communities = [{"real.md", "ghost.md"}]
    labels = store.community_labels(communities, terms=1)

    assert 0 in labels
    assert labels[0] == "concept"


# ---------------------------------------------------------------------------
# Case 4 — Empty communities list → {}
# ---------------------------------------------------------------------------
def test_empty_communities_list_returns_empty_dict(tmp_path):
    """An empty communities list must return {}."""
    store = _make_store(
        tmp_path,
        {
            "note.md": {
                "nodes": {"concept": {"label": "concept", "count": 3}},
                "edges": [],
            }
        },
    )
    result = store.community_labels([])
    assert result == {}


# ---------------------------------------------------------------------------
# Case 5 (optional) — terms=1 returns single-surface labels
# ---------------------------------------------------------------------------
def test_terms_1_returns_single_surface_no_separator(tmp_path):
    """With terms=1, label has no ' · ' separator — just one surface form."""
    store = _make_store(
        tmp_path,
        {
            "note_x.md": {
                "nodes": {
                    "alpha": {"label": "alpha", "count": 5},
                    "beta": {"label": "beta", "count": 3},
                },
                "edges": [],
            },
        },
    )
    communities = [{"note_x.md"}]
    labels = store.community_labels(communities, terms=1)

    assert 0 in labels
    assert " · " not in labels[0], "terms=1 must not produce a separator"


# ---------------------------------------------------------------------------
# Extra — terms larger than available distinct stems → use all available
# ---------------------------------------------------------------------------
def test_terms_larger_than_available_uses_all(tmp_path):
    """Requesting more terms than stems available → all stems used, no error."""
    store = _make_store(
        tmp_path,
        {
            "note.md": {
                "nodes": {
                    "onlystem": {"label": "onlystem", "count": 2},
                },
                "edges": [],
            }
        },
    )
    communities = [{"note.md"}]
    labels = store.community_labels(communities, terms=10)
    assert 0 in labels
    assert labels[0] == "onlystem"


# ---------------------------------------------------------------------------
# Extra — community omitted from result when all member notes absent
# ---------------------------------------------------------------------------
def test_all_absent_community_omitted(tmp_path):
    """A community whose members are all absent from the store is not in the result."""
    store = _make_store(
        tmp_path,
        {
            "real.md": {
                "nodes": {"concept": {"label": "concept", "count": 2}},
                "edges": [],
            }
        },
    )
    # comm 0 = real, comm 1 = entirely absent
    communities = [{"real.md"}, {"ghost_x.md", "ghost_y.md"}]
    labels = store.community_labels(communities, terms=1)

    assert 0 in labels
    assert 1 not in labels, "All-absent community must be omitted from result"
