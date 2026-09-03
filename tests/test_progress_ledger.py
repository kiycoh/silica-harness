"""Phase 1 tests — ProgressLedger: schema, serialisation, dependency ordering."""
from __future__ import annotations

import pytest

from silica.kernel.progress import (
    PlanStep,
    IssueCard,
    ProgressLedger,
    Task,
    TaskLedger,
)


@pytest.fixture(autouse=True)
def _runs_dir_is_tmp(tmp_path, monkeypatch):
    """Every test here writes a ledger, so every one of them has to redirect
    `_RUNS_DIR` away from ~/.silica. Autouse + monkeypatch instead of 26
    hand-rolled assignments because the hand-rolled ones did not restore it: 23
    had no teardown at all and the three that did put `tmp_path` back instead of
    the original, so the first test to run left every later test in the session
    pointed at a tmp dir pytest had already removed.
    """
    import silica.kernel.progress as _mod

    monkeypatch.setattr(_mod, "_RUNS_DIR", tmp_path)


# ---------------------------------------------------------------------------
# Round-trip serialisation
# ---------------------------------------------------------------------------

def test_roundtrip_empty(tmp_path):
    p = ProgressLedger.new(mode="inject", inputs={"inbox": "notes.md"})
    p.save()
    p2 = ProgressLedger.load(p.run_id)

    assert p2.run_id == p.run_id
    assert p2.mode == "inject"
    assert p2.inputs == {"inbox": "notes.md"}
    assert p2.tasks == []
    assert p2.issues == []
    assert p2.cursor is None


def test_roundtrip_with_tasks(tmp_path):
    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("recon", task_id="recon")
    p.add_task("payload", task_id="payload", depends_on=["recon"])
    p.set_status("recon", "done")
    p.set_status("payload", "running")
    p.save()
    p2 = ProgressLedger.load(p.run_id)

    assert len(p2.tasks) == 2
    recon = next(t for t in p2.tasks if t.id == "recon")
    payload = next(t for t in p2.tasks if t.id == "payload")
    assert recon.status == "done"
    assert payload.status == "running"
    assert payload.depends_on == ["recon"]
    assert payload.attempts == 1


def test_roundtrip_preserves_issue_cards(tmp_path):
    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("recon", task_id="recon")
    p.issues.append(IssueCard(
        task_id="recon",
        question="Merge or create?",
        options=[{"label": "merge"}, {"label": "create"}],
        default_option="create",
    ))
    p.save()
    p2 = ProgressLedger.load(p.run_id)

    assert len(p2.issues) == 1
    assert p2.issues[0].question == "Merge or create?"
    assert p2.issues[0].default_option == "create"


# ---------------------------------------------------------------------------
# next_pending — dependency ordering
# ---------------------------------------------------------------------------

def test_next_pending_respects_depends_on(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("recon",   task_id="recon")
    p.add_task("payload", task_id="payload", depends_on=["recon"])

    # payload has unmet dep → only recon is available
    nxt = p.next_pending()
    assert nxt is not None and nxt.id == "recon"

    p.mark_done("recon")
    nxt = p.next_pending()
    assert nxt is not None and nxt.id == "payload"


def test_next_pending_returns_none_when_all_done(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("recon", task_id="recon")
    p.mark_done("recon")

    assert p.next_pending() is None


def test_next_pending_skips_running_tasks(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("recon",   task_id="recon")
    p.add_task("payload", task_id="payload", depends_on=["recon"])

    p.set_status("recon", "running")
    # recon is running (not done) → payload dep unmet, nothing available
    assert p.next_pending() is None


# ---------------------------------------------------------------------------
# blocked tasks are never returned
# ---------------------------------------------------------------------------

def test_next_pending_never_returns_blocked(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("recon", task_id="recon")
    p.set_status("recon", "blocked")  # human must resolve

    assert p.next_pending() is None


def test_next_pending_skips_blocked_even_with_deps_met(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("recon",   task_id="recon")
    p.add_task("payload", task_id="payload", depends_on=["recon"])
    p.mark_done("recon")
    p.set_status("payload", "blocked")  # all deps met but blocked

    assert p.next_pending() is None


# ---------------------------------------------------------------------------
# set_status / mark_done / mark_failed
# ---------------------------------------------------------------------------

def test_set_status_running_increments_attempts(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("recon", task_id="recon")

    p.set_status("recon", "running")
    assert p.tasks[0].attempts == 1
    assert p.cursor == "recon"

    p.set_status("recon", "failed", error="oops")
    assert p.tasks[0].error == "oops"
    assert p.cursor is None


def test_mark_done_clears_cursor(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("recon", task_id="recon")
    p.set_status("recon", "running")
    p.mark_done("recon", output_ref="/tmp/recon_out.json")

    assert p.tasks[0].status == "done"
    assert p.tasks[0].output_ref == "/tmp/recon_out.json"
    assert p.cursor is None


def test_set_status_unknown_task_raises(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    with pytest.raises(KeyError, match="no-such-id"):
        p.set_status("no-such-id", "running")


def test_mark_failed_records_error(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("distill", task_id="distill")
    p.mark_failed("distill", "LLM timeout")

    t = p.tasks[0]
    assert t.status == "failed"
    assert t.error == "LLM timeout"


# ---------------------------------------------------------------------------
# add_task returns the Task and it is in ledger.tasks
# ---------------------------------------------------------------------------

def test_add_task_returns_task_appended(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    t = p.add_task("recon", task_id="recon", input_ref="/tmp/inbox.md")

    assert isinstance(t, Task)
    assert t in p.tasks
    assert t.input_ref == "/tmp/inbox.md"
    assert t.status == "pending"


# ---------------------------------------------------------------------------
# last_updated advances on mutation
# ---------------------------------------------------------------------------

def test_last_updated_advances_on_mutation(tmp_path):
    import time

    p = ProgressLedger.new(mode="inject", inputs={})
    before = p.last_updated
    time.sleep(0.01)

    p.add_task("recon", task_id="recon")
    p.set_status("recon", "running")

    assert p.last_updated > before


# ---------------------------------------------------------------------------
# content_hash field on Task (Phase 2 idempotency hook)
# ---------------------------------------------------------------------------

def test_task_content_hash_survives_roundtrip(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    t = p.add_task("recon", task_id="recon")
    t.content_hash = "abc123"
    p.save()
    p2 = ProgressLedger.load(p.run_id)

    assert p2.tasks[0].content_hash == "abc123"


# ---------------------------------------------------------------------------
# deferred status
# ---------------------------------------------------------------------------

def test_deferred_status_is_accepted(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("distill", task_id="distill")
    p.set_status("distill", "running")
    p.set_status("distill", "deferred")

    assert p.tasks[0].status == "deferred"
    assert p.cursor is None


def test_next_pending_skips_deferred(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("distill", task_id="distill")
    p.set_status("distill", "deferred")

    assert p.next_pending() is None


# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TaskLedger — save / load / write-once idempotency
# ---------------------------------------------------------------------------

def test_task_ledger_roundtrip(tmp_path):

    specs = [
        PlanStep("recon",   "mechanical", "silica_recon"),
        PlanStep("distill", "semantic",   "distiller"),
        PlanStep("validate","gate",        "silica_validate_ops"),
    ]
    tl = TaskLedger.new(
        run_id="testrun001",
        user_request="inject Inbox/foo.md → Concepts/",
        checkpoints=specs,
        facts={"source": "foo.md"},
    )
    tl.save()
    tl2 = TaskLedger.load("testrun001")

    assert tl2.run_id == "testrun001"
    assert tl2.user_request == "inject Inbox/foo.md → Concepts/"
    assert len(tl2.checkpoints) == 3
    assert tl2.checkpoints[1].kind == "semantic"
    assert tl2.facts == {"source": "foo.md"}


def test_task_ledger_save_is_write_once(tmp_path):
    """Second save() must not overwrite the file."""

    tl = TaskLedger.new(run_id="testrun002", user_request="original", checkpoints=[])
    tl.save()

    # Mutate and save again — disk must still have the original content
    tl.user_request = "mutated"
    tl.save()

    tl2 = TaskLedger.load("testrun002")
    assert tl2.user_request == "original"


# ---------------------------------------------------------------------------
# ProgressLedger.digest()
# ---------------------------------------------------------------------------

def test_digest_contains_run_id(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={"inbox_file": "Inbox/test.md"})
    p.add_task("recon", task_id="recon")
    p.set_status("recon", "running")

    d = p.digest()
    assert p.run_id[:8] in d
    assert "inject" in d
    assert "recon" in d


def test_digest_under_500_tokens(tmp_path):
    """Digest must stay compact enough for LLM context injection."""

    specs = [PlanStep(f"phase_{i}", "mechanical", f"tool_{i}") for i in range(10)]
    tl = TaskLedger.new(run_id="bigrun", user_request="inject big vault", checkpoints=specs)
    tl.save()

    p = ProgressLedger.new(mode="inject", inputs={"inbox_file": "Inbox/big.md", "target_dir": "Concepts/"})
    p.run_id = "bigrun"  # align with TaskLedger
    for i in range(10):
        p.add_task(f"tool_{i}", task_id=f"phase_{i}")
    p.set_status("phase_0", "done")
    p.set_status("phase_1", "running")

    d = p.digest()
    # Rough estimate: 1 token ≈ 4 chars; 500 tokens ≈ 2000 chars
    assert len(d) < 2000, f"digest too long ({len(d)} chars)"


def test_digest_shows_task_ledger_plan(tmp_path):

    specs = [
        PlanStep("recon",   "mechanical", "silica_recon"),
        PlanStep("distill", "semantic",   "distiller"),
    ]
    tl = TaskLedger.new(run_id="plantest", user_request="inject test", checkpoints=specs)
    tl.save()

    p = ProgressLedger.new(mode="inject", inputs={})
    p.run_id = "plantest"
    d = p.digest()

    assert "PLAN" in d
    assert "recon(mechanical)" in d
    assert "distill(semantic)" in d


def test_digest_graceful_without_task_ledger(tmp_path):
    """digest() must not raise when TaskLedger doesn't exist on disk."""

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("recon", task_id="recon")
    # No TaskLedger saved — digest() should still return something useful
    d = p.digest()
    assert "RUN" in d
    assert "recon" in d


# ---------------------------------------------------------------------------
# silica_ledger_digest tool
# ---------------------------------------------------------------------------

def test_silica_ledger_digest_tool(tmp_path):

    p = ProgressLedger.new(mode="inject", inputs={"inbox_file": "Inbox/x.md"})
    p.add_task("recon", task_id="recon")
    p.set_status("recon", "done")
    p.save()

    from silica.tools.composed import silica_ledger_digest
    result = silica_ledger_digest(run_id=p.run_id)

    assert "error" not in result
    assert result["run_id"] == p.run_id
    assert "digest" in result
    assert p.run_id[:8] in result["digest"]


def test_silica_ledger_digest_tool_latest_run(tmp_path):
    """Passing run_id='' should pick the most recently modified run."""

    p = ProgressLedger.new(mode="inject", inputs={})
    p.add_task("recon", task_id="recon")
    p.save()

    from silica.tools.composed import silica_ledger_digest
    result = silica_ledger_digest(run_id="")

    assert "error" not in result
    assert result["run_id"] == p.run_id


def test_silica_ledger_digest_tool_unknown_run(tmp_path):

    from silica.tools.composed import silica_ledger_digest
    result = silica_ledger_digest(run_id="doesnotexist")
    assert "error" in result
