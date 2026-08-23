# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""A dedup verdict says whether the two sides share a source.

Two notes from one source in one run are distiller noise; two from different
sources are a real convergence. The judge and its gates are untouched (the
cross-file fusion path stays closed, see the CROSSDEDUP verdict): the ledger's
answer rides on the result as information for whoever reviews the merge.
"""
from __future__ import annotations

from unittest.mock import patch

from silica.capabilities.dedup import DedupDecision, _route_verdict
from silica.config import SilicaConfig
from silica.kernel.workqueue import WorkItem
from silica.kernel.write.provenance import append_record


def _item(path: str) -> WorkItem:
    return WorkItem(kind="dedup", target_path=path, context={})


def _route(ctx, path="Concepts/A.md"):
    with patch("silica.capabilities.dedup.commit_ops", return_value={"status": "committed"}), \
         patch("silica.capabilities.dedup._clean_twin_bundle"), \
         patch("silica.capabilities.dedup._record_absorbed_alias"), \
         patch("silica.capabilities.dedup.emit_feedback"):
        return _route_verdict(_item(path), ctx,
                              DedupDecision(verdict="duplicate", addition="more", rationale="same"),
                              SilicaConfig())


def test_same_source_is_named(tmp_vault):
    append_record("lec.md", "sha1", "r1", ["Concepts/A"])

    res = _route({"inbox_file": "Inbox/lec.md"})

    assert res["provenance"] == {"candidate_sources": ["lec.md"],
                                 "incoming_source": "lec.md", "relation": "same-source"}


def test_cross_source_is_named(tmp_vault):
    append_record("lec.md", "sha1", "r1", ["Concepts/A"])

    res = _route({"inbox_file": "Inbox/paper.md"})

    assert res["provenance"]["relation"] == "cross-source"


def test_no_ledger_knowledge_is_unknown_not_a_guess(tmp_vault):
    res = _route({"inbox_file": "Inbox/paper.md"})

    assert res["provenance"] == {"candidate_sources": [], "incoming_source": "paper.md",
                                 "relation": "unknown"}
