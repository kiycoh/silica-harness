# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Note edges are DERIVED at read (CORRELATE, ADR-0029).

`note_adjacency()` is a memo over the contributions: symmetric, dropped on any
mutation, never written to disk. A file from before ADR-0029 still carries a
`note_edges` section; it is ignored, not migrated.
"""
from __future__ import annotations

import orjson
import pytest

from silica.kernel.recall.cooccurrence import CooccurStore


def _nodes(**counts: int) -> dict:
    return {"nodes": {s: {"label": s, "count": c} for s, c in counts.items()}, "edges": []}


def test_edges_derive_from_contributions_in_both_directions() -> None:
    store = CooccurStore()
    store.upsert_note("a", _nodes(quick=1, sort=1, array=1, tree=1))
    store.upsert_note("b", _nodes(quick=1, sort=1, array=1, heap=1))  # jaccard 3/5
    assert store.note_edges_for("a") == {"b": pytest.approx(0.6)}
    assert store.note_edges_for("b") == {"a": pytest.approx(0.6)}
    assert store.note_edges_for("a.md") == store.note_edges_for("a")  # store keyspace


def test_no_edge_below_tau() -> None:
    store = CooccurStore()
    store.upsert_note("a", _nodes(quick=1, sort=1, array=1))
    store.upsert_note("b", _nodes(bread=1, flour=1, yeast=1))
    assert store.note_edges_for("a") == {}
    assert store.note_adjacency() == {}


def test_memo_is_dropped_by_every_mutation() -> None:
    store = CooccurStore()
    store.upsert_note("a", _nodes(quick=1, sort=1, array=1, tree=1))
    store.upsert_note("b", _nodes(quick=1, sort=1, array=1, heap=1))
    assert store.note_edges_for("a") == {"b": pytest.approx(0.6)}
    store.upsert_note("a", _nodes(xxx=1, yyy=1))  # diverged: the edge must go
    assert store.note_edges_for("a") == {} and store.note_edges_for("b") == {}
    store.upsert_note("c", _nodes(xxx=1, yyy=1, zzz=1))
    assert store.note_edges_for("a") == {"c": pytest.approx(2 / 3)}
    store.delete_note("c")
    assert store.note_edges_for("a") == {}


def test_edges_are_not_persisted_and_a_legacy_section_is_ignored(tmp_path) -> None:
    p = tmp_path / "cooccurrence.json"
    store = CooccurStore(path=p)
    store.upsert_note("a", _nodes(quick=1, sort=1, array=1, tree=1))
    store.upsert_note("b", _nodes(quick=1, sort=1, array=1, heap=1))
    store.save()
    on_disk = orjson.loads(p.read_bytes())
    assert "note_edges" not in on_disk
    # A pre-ADR-0029 file: the stale section names an edge the contributions
    # do not support and misses one they do. Contributions win, silently.
    on_disk["note_edges"] = {"a": {"zzz": 0.9}}
    p.write_bytes(orjson.dumps(on_disk))
    reloaded = CooccurStore(path=p)
    assert reloaded.note_edges_for("a") == {"b": pytest.approx(0.6)}
    assert reloaded.note_edges_for("zzz") == {}
