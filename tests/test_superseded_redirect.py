# SPDX-License-Identifier: AGPL-3.0-or-later

"""A merge loser must not attract the next write.

Run f30ace50 (2026-09-03, Obsidian/test, Lezione 1-10): the judge ruled
"Percettrone" a duplicate of "Percettrone di Rosenblatt", merged Lezione 1
into the winner and stamped the loser `superseded_by`. Nothing read that
stamp: the exact-path coercion sent Lezione 2 and 10 into the loser, stage B
listed it as a live note so the model tied Lezione 8 to it, and the winner
never saw another line. Three seams now honour the pointer: the write gate
follows it, the vault outline hides the loser, and the duplicate verdict
records the absorbed spelling as an alias on the winner.
"""
from __future__ import annotations

from unittest.mock import patch

from silica.kernel.write.ops import OpType
from silica.kernel.write.validate import validate_operations

LONG = "corpo della nota " * 30

WINNER = "Corso/Percettrone di Rosenblatt.md"
LOSER = "Corso/Percettrone.md"
LOSER_TEXT = '---\nAI: true\nsuperseded_by: "[[Percettrone di Rosenblatt]]"\n---\n\n# Percettrone\n\ncorpo\n'


def _write_op(heading: str, path: str) -> dict:
    return {"op": "write", "path": path, "heading": heading,
            "source_basename": "lez.md", "snippet": LONG}


def _only(validated, heading):
    [op] = [o for o in validated if o.heading == heading]
    return op


# ---------------------------------------------------------------- validate --


def test_write_on_superseded_path_patches_the_winner(tmp_vault):
    tmp_vault.note(WINNER, "# Rosenblatt\n\ncorpo")
    tmp_vault.note(LOSER, LOSER_TEXT)
    validated, rejected = validate_operations([_write_op("Percettrone", LOSER)], [], "Corso")
    assert rejected == []
    op = _only(validated, "Percettrone")
    assert op.op == OpType.patch and op.path == WINNER


def test_key_equal_match_on_superseded_note_patches_the_winner(tmp_vault):
    tmp_vault.note(WINNER, "# Rosenblatt\n\ncorpo")
    tmp_vault.note(LOSER, LOSER_TEXT)
    validated, _ = validate_operations(
        [_write_op("Percettrone (1958)", "Corso/Percettrone (1958).md")], [], "Corso")
    op = _only(validated, "Percettrone (1958)")
    assert op.op == OpType.patch and op.path == WINNER


def test_near_title_candidate_on_superseded_note_points_at_the_winner(tmp_vault):
    tmp_vault.note("Corso/Descriptor concept.md", "# Descriptor concept\n\ncorpo")
    tmp_vault.note("Corso/Descriptor.md",
                   '---\nsuperseded_by: "[[Descriptor concept]]"\n---\n\n# Descriptor\n\ncorpo\n')
    validated, _ = validate_operations(
        [_write_op("Description", "Corso/Description.md")], [], "Corso")
    op = _only(validated, "Description")
    assert op.op == OpType.write
    assert "candidate='Descriptor concept'" in op.review
    assert "path='Corso/Descriptor concept.md'" in op.review


def test_dangling_or_cyclic_pointer_keeps_the_original_path(tmp_vault):
    tmp_vault.note("Corso/A.md", '---\nsuperseded_by: "[[B]]"\n---\n\n# A\n')
    tmp_vault.note("Corso/B.md", '---\nsuperseded_by: "[[A]]"\n---\n\n# B\n')
    tmp_vault.note("Corso/C.md", '---\nsuperseded_by: "[[Missing]]"\n---\n\n# C\n')
    validated, _ = validate_operations(
        [_write_op("A", "Corso/A.md"), _write_op("C", "Corso/C.md")], [], "Corso")
    assert _only(validated, "A").path in ("Corso/A.md", "Corso/B.md")
    assert _only(validated, "C").path == "Corso/C.md"


# ------------------------------------------------------------ vault_outline --


def test_vault_outline_hides_superseded_notes(tmp_vault):
    from silica.kernel import outline as ol

    tmp_vault.note("ML/Percettrone di Rosenblatt.md", "# R\n\n> Il percettrone di Rosenblatt.\n")
    tmp_vault.note("ML/Percettrone.md", LOSER_TEXT)
    rows = ol.vault_outline("ML")
    assert {r["title"] for r in rows} == {"Percettrone di Rosenblatt"}


# -------------------------------------------------------------------- judge --


def _decision(verdict, addition="", rationale="why"):
    from silica.capabilities.dedup import DedupDecision
    return DedupDecision(verdict=verdict, addition=addition, rationale=rationale)


def test_duplicate_with_landed_loser_records_alias_on_the_winner(tmp_vault):
    from silica.capabilities.dedup import _route_verdict
    from silica.kernel.workqueue import WorkItem
    from silica.kernel.write import frontmatter

    tmp_vault.note(WINNER, "---\nAI: partial\n---\n\n# Rosenblatt\n\ncorpo\n")
    tmp_vault.note(LOSER, "---\nAI: true\n---\n\n# Percettrone\n\ncorpo\n")
    ctx = {"concept": "Percettrone", "candidate": "Percettrone di Rosenblatt",
           "loser_path": LOSER, "target_dir": "Corso", "hub": "Hub",
           "inbox_file": "Inbox/lez.md", "content_hash": "h"}
    item = WorkItem(kind="dedup", target_path=WINNER, context=ctx)
    with patch("silica.capabilities.dedup.commit_ops",
               return_value={"status": "committed", "committed": 1}) as commit, \
         patch("silica.kernel.recall.deferred.get_deferred_store"):
        res = _route_verdict(item, ctx, _decision("duplicate", addition="nuovo fatto"), None)

    assert res["status"] == "committed"
    by_reason = {c.args[0][0].reason: c.args[0][0] for c in commit.call_args_list}
    alias_op = by_reason["dedup merge: 'Percettrone' kept as alias"]
    assert alias_op.path == WINNER
    data, _, _ = frontmatter.split(alias_op.content)
    assert "Percettrone" in data["aliases"]
    assert "dedup merge: superseded_by pointer" in by_reason  # the loser is still marked


def test_duplicate_without_addition_records_alias_too():
    from silica.capabilities.dedup import _route_verdict
    from silica.kernel.workqueue import WorkItem

    ctx = {"concept": "Percettrone", "loser_path": LOSER, "hub": "Hub"}
    item = WorkItem(kind="dedup", target_path=WINNER, context=ctx)
    with patch("silica.capabilities.dedup._mark_merge_loser") as mark, \
         patch("silica.capabilities.dedup._record_absorbed_alias") as alias:
        res = _route_verdict(item, ctx, _decision("duplicate"), None)
    assert res["status"] == "no_merge"
    mark.assert_called_once_with(ctx, WINNER)
    alias.assert_called_once_with(ctx, WINNER, "Hub")


# ------------------------------------------------------- alias in the gate --


def test_write_key_equal_to_an_alias_patches_the_aliased_note(tmp_vault):
    """The pointer dies with the stub; the alias the judge left on the winner
    is what routes "Percettrone" once the loser is gone."""
    tmp_vault.note(WINNER, "---\naliases:\n  - Percettrone\n---\n\n# Rosenblatt\n\ncorpo\n")
    validated, rejected = validate_operations([_write_op("Percettrone", LOSER)], [], "Corso")
    assert rejected == []
    op = _only(validated, "Percettrone")
    assert op.op == OpType.patch and op.path == WINNER


def test_filename_outranks_alias_on_the_same_key(tmp_vault):
    tmp_vault.note(WINNER, "---\naliases:\n  - Percettrone\n---\n\n# Rosenblatt\n\ncorpo\n")
    tmp_vault.note(LOSER, "# Percettrone\n\ncorpo\n")  # a live note, no pointer
    validated, _ = validate_operations(
        [_write_op("Percettrone (1958)", "Corso/Percettrone (1958).md")], [], "Corso")
    op = _only(validated, "Percettrone (1958)")
    assert op.op == OpType.patch and op.path == LOSER
