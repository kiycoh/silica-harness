# SPDX-License-Identifier: AGPL-3.0-or-later

"""Soft gates: a note that is merely suspicious lands flagged, never parked.

Measured 2026-09-02 on the ML lessons (Obsidian/test): the dedup judge was
down for one whole run (73/73 items "Reasoning is mandatory"), and every
near_title op it should have ruled on stayed in the deferred store because the
retry path re-validates without a judge, so the same gate fired forever. Two
gates decide nothing on their own (fuzzy title band, short-but-real body) and
only ever meant "someone should look": those now let the op land with a
`review:` frontmatter key, a warning-ledger row, and the judge asked AFTER the
note exists. Empty bodies and invented paths stay hard.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

from silica.config import CONFIG
from silica.kernel.write.ops import Op, OpType
from silica.kernel.write.validate import validate_operations

LONG = "corpo della nota " * 30  # 510 chars, over the 275 floor


def _write_op(heading: str, path: str, snippet: str = LONG) -> dict:
    return {"op": "write", "path": path, "heading": heading,
            "source_basename": "lez.md", "snippet": snippet}


# ---------------------------------------------------------------------------
# validate: the two soft gates flag instead of rejecting
# ---------------------------------------------------------------------------


def test_near_title_lands_flagged(tmp_vault):
    tmp_vault.note("Corso/Descriptor.md", "# Descriptor\n\ncorpo")
    validated, rejected = validate_operations(
        [_write_op("Description", "Corso/Description.md")], [], "Corso",
    )
    assert rejected == []
    [op] = [o for o in validated if o.heading == "Description"]  # validate also adds the hub op
    assert op.op == OpType.write and op.path == "Corso/Description.md"
    assert "near_title" in op.review and "Descriptor" in op.review


def test_short_body_lands_flagged(tmp_vault):
    validated, rejected = validate_operations(
        [_write_op("Percettrone", "Corso/Percettrone.md", snippet="x" * 50)], [], "Corso",
    )
    assert rejected == []
    [op] = [o for o in validated if o.heading == "Percettrone"]
    assert "too short" in op.review


def test_empty_body_stays_hard(tmp_vault):
    from tests.test_snippet_gate import _payload

    payload = _payload("Inbox/lez.md", [{"name": "Percettrone", "inbox_excerpt": "x" * 500}])
    validated, rejected = validate_operations(
        [_write_op("Percettrone", "Corso/Percettrone.md", snippet="")], payload, "Corso",
    )
    assert not any(o.heading == "Percettrone" for o in validated)
    assert len(rejected) == 1 and "too short" in rejected[0].reason


def test_both_soft_gates_stack_in_one_review(tmp_vault):
    tmp_vault.note("Corso/Descriptor.md", "# Descriptor\n\ncorpo")
    validated, _ = validate_operations(
        [_write_op("Description", "Corso/Description.md", snippet="x" * 50)], [], "Corso",
    )
    [op] = [o for o in validated if o.heading == "Description"]
    assert "near_title" in op.review and "too short" in op.review


def test_clean_write_has_no_review(tmp_vault):
    validated, _ = validate_operations(
        [_write_op("Percettrone", "Corso/Percettrone.md")], [], "Corso",
    )
    assert all(o.review is None for o in validated)


# ---------------------------------------------------------------------------
# write: the flag reaches the note
# ---------------------------------------------------------------------------


def test_write_stamps_review_in_frontmatter(tmp_vault):
    from silica.kernel.write.bulk import _execute_write
    from silica.kernel.write import frontmatter

    op = Op(op=OpType.write, heading="Description", source_basename="lez.md",
            path="Corso/Description.md", snippet=LONG, hub="Corso",
            review="near_title candidate='Descriptor' path='Corso/Descriptor.md' ratio=0.90")
    _execute_write(op, "Corso/Description.md")
    data, _, _ = frontmatter.split(open(os.path.join(CONFIG.vault_path, "Corso/Description.md"), encoding="utf-8").read())
    assert data["review"].startswith("near_title candidate='Descriptor'")


# ---------------------------------------------------------------------------
# WRITE state: landed flags go to the ledger and to the judge, after landing
# ---------------------------------------------------------------------------


class _FakeQueue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)


class _FakeLedger:
    def __init__(self):
        self.rows = []

    def add(self, path, kind, detail=""):
        self.rows.append((path, kind, detail))


def _fsm(queue, ledger):
    return SimpleNamespace(
        work_queue=queue, warning_ledger=ledger, inbox_file="in.md",
        _current_source_file="in.md", hub="Hub", _current_content_hash="h",
        target_dir="Corso", progress=SimpleNamespace(run_id="run-1"),
        _chunks=[{"batches": [{"concepts": [
            {"name": "Percettrone", "inbox_excerpt": "excerpt"}]}]}],
        _current_chunk_idx=0,
    )


def test_settle_flagged_records_ledger_and_asks_judge_with_loser_path():
    from silica.router.states.write import _settle_flagged

    q, ledger = _FakeQueue(), _FakeLedger()
    ops = [
        Op(op=OpType.write, heading="Description", source_basename="lez.md",
           path="Corso/Description.md", snippet=LONG, hub="Hub",
           review="near_title candidate='Descriptor' path='Corso/Descriptor.md' ratio=0.90"),
        Op(op=OpType.write, heading="Percettrone", source_basename="lez.md",
           path="Corso/Percettrone.md", snippet="x" * 50, hub="Hub",
           review="snippet too short (50 < 275 chars)"),
        Op(op=OpType.write, heading="Clean", source_basename="lez.md",
           path="Corso/Clean.md", snippet=LONG, hub="Hub"),
    ]
    _settle_flagged(_fsm(q, ledger), ops, committed={"Corso/Description.md", "Corso/Percettrone.md", "Corso/Clean.md"})

    assert sorted(r[:2] for r in ledger.rows) == [
        ("Corso/Description.md", "soft_gate"), ("Corso/Percettrone.md", "soft_gate")]
    kinds = {it.kind: it for it in q.items}
    assert set(kinds) == {"dedup", "expand"}
    # The judge sees a note that exists: a duplicate verdict has a loser to mark.
    assert kinds["dedup"].context["loser_path"] == "Corso/Description.md"
    assert kinds["dedup"].context["title_score"] == 0.90
    assert kinds["expand"].target_path == "Corso/Percettrone.md"


def test_settle_flagged_ignores_ops_that_did_not_land():
    from silica.router.states.write import _settle_flagged

    q, ledger = _FakeQueue(), _FakeLedger()
    op = Op(op=OpType.write, heading="Description", source_basename="lez.md",
            path="Corso/Description.md", snippet=LONG, hub="Hub",
            review="near_title candidate='Descriptor' path='Corso/Descriptor.md' ratio=0.90")
    _settle_flagged(_fsm(q, ledger), [op], committed=set())
    assert ledger.rows == [] and q.items == []


# ---------------------------------------------------------------------------
# judge verdicts on a note that already landed
# ---------------------------------------------------------------------------


def _decision(verdict, addition="", rationale="why"):
    from silica.capabilities.dedup import DedupDecision
    return DedupDecision(verdict=verdict, addition=addition, rationale=rationale)


def test_duplicate_without_addition_still_marks_the_landed_loser():
    from silica.capabilities.dedup import _route_verdict
    from silica.kernel.workqueue import WorkItem

    ctx = {"concept": "Description", "loser_path": "Corso/Description.md", "hub": "Hub"}
    item = WorkItem(kind="dedup", target_path="Corso/Descriptor.md", context=ctx)
    with patch("silica.capabilities.dedup._mark_merge_loser") as mark:
        res = _route_verdict(item, ctx, _decision("duplicate"), None)
    assert res["status"] == "no_merge"
    mark.assert_called_once_with(ctx, "Corso/Descriptor.md")


def test_distinct_on_landed_note_clears_flag_and_leaves_trace(tmp_vault):
    from silica.capabilities.dedup import _route_distinct
    from silica.kernel.workqueue import WorkItem
    from silica.kernel.write import frontmatter
    from silica.kernel.write.templates import has_related_trace

    tmp_vault.note("Corso/Description.md",
                   "---\nAI: true\nreview: near_title candidate='Descriptor'\n---\n\n# Description\n\ncorpo\n")
    ctx = {"concept": "Description", "candidate": "Descriptor",
           "loser_path": "Corso/Description.md", "target_dir": "Corso", "hub": "Hub",
           "content_hash": "h"}
    item = WorkItem(kind="dedup", target_path="Corso/Descriptor.md", context=ctx)
    with patch("silica.capabilities.dedup.commit_ops",
               return_value={"status": "committed", "committed": 1}) as commit, \
         patch("silica.kernel.recall.deferred.get_deferred_store"):
        res = _route_distinct(item, ctx, _decision("distinct", rationale="different things"), None)

    assert res["status"] == "committed"
    [op] = commit.call_args.args[0]
    assert op.op == OpType.overwrite and op.path == "Corso/Description.md"
    data, _, body = frontmatter.split(op.content)
    assert "review" not in data
    assert has_related_trace(body, "Descriptor")
    assert commit.call_args.kwargs["bounds"].allowed_ops == frozenset({OpType.overwrite})


# ---------------------------------------------------------------------------
# expand re-authors a landed short note in place
# ---------------------------------------------------------------------------


def test_expand_overwrites_a_landed_short_note(tmp_vault):
    from silica.capabilities.expand import run_expand
    from silica.config import CONFIG
    from silica.kernel.workqueue import WorkItem
    from silica.kernel.write import frontmatter

    tmp_vault.note("Corso/Percettrone.md",
                   "---\nAI: true\nreview: snippet too short (50 < 275 chars)\n---\n\n# Percettrone\n\ncorto\n")
    item = WorkItem(kind="expand", target_path="Corso/Percettrone.md", context={
        "op": {"op": "write", "heading": "Percettrone", "path": "Corso/Percettrone.md",
               "snippet": "corto", "source_basename": "lez.md"},
        "excerpt": "il percettrone separa due classi", "reason": "snippet too short",
        "hub": "Corso", "target_dir": "Corso", "inbox_file": "Inbox/lez.md", "content_hash": "h",
    })
    with patch("silica.capabilities.expand._author_body", return_value=LONG), \
         patch("silica.capabilities.expand.commit_ops",
               return_value={"status": "committed", "committed": 1}) as commit, \
         patch("silica.kernel.recall.deferred.get_deferred_store"):
        res = run_expand(item, CONFIG)

    assert res["status"] == "committed"
    [op] = commit.call_args.args[0]
    assert op.op == OpType.overwrite and op.path == "Corso/Percettrone.md"
    data, _, body = frontmatter.split(op.content)
    assert "review" not in data and LONG.strip() in body
    assert commit.call_args.kwargs["bounds"].allowed_ops == frozenset({OpType.overwrite})


# ---------------------------------------------------------------------------
# the run's closing line counts what landed flagged
# ---------------------------------------------------------------------------


def test_coverage_line_counts_flagged_notes(tmp_path, monkeypatch):
    from silica.router.coordinator import Coordinator
    from silica.kernel.recall import deferred

    monkeypatch.setattr(deferred, "_store_dir", lambda: tmp_path / "deferred")
    deferred._stores.clear()
    ledger = _FakeLedger()
    ledger.paths = lambda kind=None: ["Corso/Description.md", "Corso/Percettrone.md"]
    fsm = SimpleNamespace(
        progress=SimpleNamespace(inputs={"residue": {}}),
        _file_content_hashes=[], _annealed_ops=0, warning_ledger=ledger,
    )
    c = object.__new__(Coordinator)
    c.fsm = fsm
    result: dict = {}
    c._coverage_summary(result)
    assert result["coverage"]["flagged_notes"] == 2


def test_duplicate_without_addition_releases_the_parked_twin():
    """The verdict is final: nothing to write, the candidate already says it.
    Leaving the op in the deferred store (as before) meant the boundary anneal
    re-validated it every run until the 30-day TTL ate it; measured 2026-09-02
    on the probe run ('iperpiano soddisfa', judged duplicate, still parked)."""
    from silica.capabilities.dedup import _route_verdict
    from silica.kernel.workqueue import WorkItem

    ctx = {"concept": "iperpiano soddisfa", "content_hash": "h10", "hub": "Hub"}
    item = WorkItem(kind="dedup", target_path="Corso/Iperpiano.md", context=ctx)
    with patch("silica.capabilities.dedup._clean_twin_bundle") as clean:
        res = _route_verdict(item, ctx, _decision("duplicate"), None)
    assert res["status"] == "no_merge"
    clean.assert_called_once_with(ctx)


def test_boundary_anneal_settles_the_ops_the_sweep_landed_flagged(monkeypatch):
    """The sweep's retry path has no judge and no ledger; the FSM running it
    in its own `finally` has both, so flagged landings get the same settle as
    a chunk write (probe run 2026-09-02: a near_title note recovered by the
    anneal carried `review:` but no ledger row and no judge item)."""
    from silica.router import orchestrator
    from silica.router.orchestrator import InjectorFSM

    monkeypatch.delenv("SILICA_BOUNDARY_ANNEAL", raising=False)
    landed = {"op": "write", "heading": "Description", "source_basename": "f.md",
              "path": "Reti/Description.md", "snippet": LONG, "hub": "Reti",
              "review": "near_title candidate='Descriptor' path='Reti/Descriptor.md' ratio=0.90"}
    fake = SimpleNamespace(
        _undo_run_id="undo-1", progress=SimpleNamespace(run_id="run-1"),
        _annealed_ops=0, _lift_recovered_partial=lambda: None,
        work_queue=_FakeQueue(), warning_ledger=_FakeLedger(),
        _current_source_file="in.md", hub="Reti", _current_content_hash="h",
        target_dir="Reti", _chunks=[], _current_chunk_idx=0,
    )
    store = SimpleNamespace(list_all=lambda: [{"content_hash": "fff6"}])
    with patch("silica.kernel.recall.deferred.get_deferred_store", return_value=store), \
         patch("silica.tools.pipeline.silica_anneal",
               return_value={"written": 1, "flagged_ops": [landed]}):
        InjectorFSM._boundary_anneal(fake)

    assert fake._annealed_ops == 1
    assert [r[:2] for r in fake.warning_ledger.rows] == [("Reti/Description.md", "soft_gate")]
    [item] = fake.work_queue.items
    assert item.kind == "dedup" and item.context["loser_path"] == "Reti/Description.md"
