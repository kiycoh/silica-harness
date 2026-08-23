# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Re-nucleating a changed source is the operator's decision, so it is said.

`check_renucleate` existed, was tested, and had no caller: the ledger knew a
source had been nucleated at another version and nothing told the user that
the notes derived from the old version would stay in place beside the new
ones. The fact is recorded where the run is described (the run report every
frontend reads) and in the completion line, with the command that undoes it.
"""
from __future__ import annotations

import types

from silica.kernel.write.provenance import append_record


class TestTheFsmNotesIt:
    def test_a_changed_source_is_recorded_in_the_context(self, tmp_vault):
        from silica.router.orchestrator import InjectorFSM

        append_record("lec.md", "sha-v1", "r1", ["Concepts/A", "Concepts/B"])
        fsm = types.SimpleNamespace(context={})

        InjectorFSM._note_renucleation(fsm, "Inbox/lec.md", "sha-v2")

        assert fsm.context["renucleated"] == {"lec.md": 2}

    def test_an_unchanged_or_first_source_records_nothing(self, tmp_vault):
        from silica.router.orchestrator import InjectorFSM

        append_record("lec.md", "sha-v1", "r1", ["Concepts/A"])
        fsm = types.SimpleNamespace(context={})

        InjectorFSM._note_renucleation(fsm, "Inbox/lec.md", "sha-v1")
        InjectorFSM._note_renucleation(fsm, "Inbox/new.md", "sha-x")
        InjectorFSM._note_renucleation(fsm, "Inbox/lec.md", "")  # unreadable file: no claim

        assert "renucleated" not in fsm.context


class TestTheRunReportCarriesIt:
    def test_files_summary_entry_names_the_prior_notes(self, tmp_vault):
        from silica.kernel.progress import RunManifestEntry
        from silica.router.states import finalize

        entry = RunManifestEntry(title="Concepts/A", path="Concepts/A", parent=None,
                                 cluster_id=-1, source_basename="lec.md", op="write")
        fsm = types.SimpleNamespace(
            manifest=types.SimpleNamespace(entries=[entry]),
            progress=types.SimpleNamespace(run_id="run-1"),
            _file_content_hashes=["sha-v2"],
            context={"renucleated": {"lec.md": 2}},
        )

        finalize._log_nucleate_completion(fsm, 0, "Inbox/lec.md")

        (summary,) = fsm.context["files_summary"]
        assert summary["renucleated_prior_notes"] == 2

    def test_a_first_nucleation_has_no_such_key(self, tmp_vault):
        from silica.kernel.progress import RunManifestEntry
        from silica.router.states import finalize

        entry = RunManifestEntry(title="Concepts/A", path="Concepts/A", parent=None,
                                 cluster_id=-1, source_basename="lec.md", op="write")
        fsm = types.SimpleNamespace(
            manifest=types.SimpleNamespace(entries=[entry]),
            progress=types.SimpleNamespace(run_id="run-1"),
            _file_content_hashes=["sha-v1"],
            context={},
        )

        finalize._log_nucleate_completion(fsm, 0, "Inbox/lec.md")

        assert "renucleated_prior_notes" not in fsm.context["files_summary"][0]


class TestTheCompletionLineSaysIt:
    def test_the_line_names_the_count_and_the_way_back(self):
        from silica.cli import _nucleate_result_line

        line = _nucleate_result_line({"final_status": "Success",
                                      "renucleated": {"lec.md": 2}})

        assert "2 note(s)" in line
        assert "/revert --source lec.md" in line

    def test_a_plain_run_has_no_such_clause(self):
        from silica.cli import _nucleate_result_line

        assert "revert" not in _nucleate_result_line({"final_status": "Success"})
