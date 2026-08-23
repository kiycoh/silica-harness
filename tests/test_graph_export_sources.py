# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Graph nodes carry the sources that authored them.

Provenance is a third grouping of the graph, disjoint from both Louvain
partitions (ADR-0023 names two, this is neither): it is read off the ledger,
never computed from topology. The payload carries it per node so any frame
or report can group by it; a node no source authored carries nothing.
"""
from __future__ import annotations

from silica.kernel.write.provenance import append_record


def test_nodes_carry_their_sources(tmp_vault):
    from silica.kernel.recall.graph_export import build_graph_data

    tmp_vault.note("Concepts/A.md", "---\ntitle: A\n---\n\n[[B]]\n")
    tmp_vault.note("Concepts/B.md", "---\ntitle: B\n---\n\nb\n")
    tmp_vault.note("Concepts/C.md", "---\ntitle: C\n---\n\nc\n")
    append_record("lec.md", "sha1", "r1", ["Concepts/A", "Concepts/B"])
    append_record("paper.md", "sha2", "r2", ["Concepts/B"])

    nodes, _edges = build_graph_data()
    by_id = {n["id"]: n for n in nodes}

    assert by_id["Concepts/A.md"]["sources"] == ["lec.md"]
    assert by_id["Concepts/B.md"]["sources"] == ["lec.md", "paper.md"]
    assert "sources" not in by_id["Concepts/C.md"]
