# SPDX-License-Identifier: AGPL-3.0-or-later
"""The emit-site grafts: each seam gets the smallest test that fails
without it (call_llm wrapper, consume spans, commit_ops write beats,
the GUI turn round-trip)."""
from __future__ import annotations

import pytest

import silica.agent.narration as narr


@pytest.fixture()
def narrator(_fresh_narrator, monkeypatch):
    monkeypatch.setattr("silica.config.CONFIG.vault_path", "/v/demo", raising=False)
    _fresh_narrator.ensure_session(driver="tui")
    return _fresh_narrator


def _beats(narrator):
    return list(narr.read_beats(narr.narration_dir() / f"{narrator.sid}.jsonl"))


def test_call_llm_wrapper_narrates_the_envelope(narrator, monkeypatch):
    import silica.agent.llm as llm

    monkeypatch.setattr(llm, "_call_llm", lambda *a, **k: llm.LLMResponse(
        text="hi", usage={"prompt_tokens": 900, "completion_tokens": 40}))
    llm.call_llm("test/model", [{"role": "user", "content": "q"}])
    calls = [b for b in _beats(narrator) if b["kind"] == "call"]
    assert [b["status"] for b in calls] == ["running", "done"]
    assert calls[1]["payload"]["prompt_tokens"] == 900
    assert calls[0]["id"] == calls[1]["id"]


def test_call_llm_failure_closes_the_span_failed(narrator, monkeypatch):
    import silica.agent.llm as llm

    def boom(*a, **k):
        raise RuntimeError("provider down")
    monkeypatch.setattr(llm, "_call_llm", boom)
    with pytest.raises(RuntimeError):
        llm.call_llm("test/model", [])
    calls = [b for b in _beats(narrator) if b["kind"] == "call"]
    assert calls[-1]["status"] == "failed"
    assert "provider down" in calls[-1]["payload"]["error"]


def test_consume_wraps_items_in_attributed_subagent_spans(narrator):
    from silica.agent.subagent import consume
    from silica.kernel.workqueue import WorkItem, WorkQueue

    class FakeAgent:
        def handle(self, item):
            narr.NARRATOR.narrate("write", "committed", "inner write")
            return {"status": "done"}

    wq = WorkQueue()
    item = WorkItem(kind="dedup", target_path="Concepts/X.md", id="it1")
    wq.enqueue(item)
    wq.close()
    consume(wq, FakeAgent(), parent_span="run-r9", run_id="r9")
    beats = _beats(narrator)
    sub = [b for b in beats if b["kind"] == "subagent"]
    assert [b["status"] for b in sub] == ["running", "done"]
    assert sub[0]["parent"] == "run-r9" and sub[0]["run"] == "r9"
    inner = next(b for b in beats if b["summary"] == "inner write")
    assert inner["parent"] == "it1"          # attributed to the subagent span
    work = [b for b in beats if b["kind"] == "work"]
    assert work[-1]["status"] == "done" and work[-1]["id"] == "wk-it1"


def test_consume_cancelled_item_narrates_cancelled(narrator):
    from silica.agent.subagent import consume
    from silica.kernel.workqueue import WorkItem, WorkQueue

    wq = WorkQueue()
    item = WorkItem(kind="refine", target_path="A.md", id="it2")
    item.cancel_token.set()
    wq.enqueue(item)
    wq.close()
    consume(wq, object())
    sub = [b for b in _beats(narrator) if b["kind"] == "subagent"]
    assert sub[-1]["status"] == "cancelled"
    assert sub[-1]["payload"]["phase"] == "pre_handle"


def test_commit_ops_narrates_proposed_then_committed(narrator, monkeypatch):
    import silica.agent.commit as commit
    import silica.tools.composed as composed
    import silica.tools.wrapped as wrapped
    from silica.kernel.write.ops import Op, OpType

    op = Op(op=OpType.patch, heading="X", source_basename="s.md", path="Concepts/X.md",
            snippet="body")
    monkeypatch.setattr(composed, "silica_validate_ops", lambda *a, **k: {"validated_count": 1})
    monkeypatch.setattr(commit, "load_ops", lambda _p: [op])
    monkeypatch.setattr(wrapped, "silica_snapshot", lambda *a, **k: {"txn_id": "t" * 12, "inverses": []})
    monkeypatch.setattr(composed, "silica_bulk_write", lambda *a, **k: {"successful": 1, "total": 1})
    monkeypatch.setattr(composed, "silica_lint", lambda *a, **k: {"success": True})
    res = commit.commit_ops([op])
    assert res["status"] == "committed"
    writes = [b for b in _beats(narrator) if b["kind"] == "write"]
    assert [b["status"] for b in writes] == ["proposed", "committed"]
    assert writes[0]["payload"]["touched"] == [["Concepts/X.md", "patch", ""]]
    assert writes[0]["id"] == writes[1]["id"]


def test_commit_ops_rollback_narrates_rolled_back(narrator, monkeypatch):
    import silica.agent.commit as commit
    import silica.tools.composed as composed
    import silica.tools.wrapped as wrapped
    from silica.kernel.write.ops import Op, OpType

    op = Op(op=OpType.patch, heading="X", source_basename="s.md", path="Concepts/X.md",
            snippet="body")
    monkeypatch.setattr(composed, "silica_validate_ops", lambda *a, **k: {"validated_count": 1})
    monkeypatch.setattr(commit, "load_ops", lambda _p: [op])
    monkeypatch.setattr(wrapped, "silica_snapshot", lambda *a, **k: {"txn_id": "t" * 12, "inverses": []})
    monkeypatch.setattr(composed, "silica_bulk_write", lambda *a, **k: {"error": "disk full"})
    monkeypatch.setattr(wrapped, "silica_restore", lambda *a, **k: None)
    res = commit.commit_ops([op])
    assert res["status"] == "rolled_back"
    writes = [b for b in _beats(narrator) if b["kind"] == "write"]
    assert [b["status"] for b in writes] == ["proposed", "rolled_back"]
    assert "disk full" in writes[1]["summary"]


def test_gui_turn_roundtrip_replay_and_endpoints(narrator, monkeypatch, tmp_path):
    """The GUI seam end to end: a turn lands as beats, /narration replays it,
    and a reloaded session equals what the user saw."""
    from fastapi.testclient import TestClient
    from silica.ui.web import server

    narrator.close()   # the server owns its own session
    server._reset_session()

    def fake_run_agent(messages, model, tool_progress_callback=None,
                       cancel_token=None, **kw):
        messages.append({"role": "assistant", "content": "Reply"})
        return "Reply"

    monkeypatch.setattr(server, "run_agent", fake_run_agent)
    tc = TestClient(server.app)
    tc.post("/chat", json={"text": "hello narration"})

    sid = server.current_session_id
    assert sid
    body = tc.get("/narration").json()
    kinds = [b["kind"] for b in body["beats"]]
    assert body["sid"] == sid
    assert kinds[0] == "session"
    turns = [b["payload"]["message"] for b in body["beats"] if b["kind"] == "turn"]
    assert turns[0]["content"] == "hello narration"
    assert turns[-1]["content"] == "Reply"

    # from_seq is a real cursor, not a rendering
    tail = tc.get("/narration", params={"from_seq": body["beats"][-1]["seq"]}).json()
    assert tail["beats"] == []

    # reset then load: the replay is the store (no snapshot anywhere)
    tc.post("/reset")
    assert tc.post("/session/load", json={"id": sid}).status_code == 200
    assert server.messages[-1]["content"] == "Reply"


def test_gui_stop_narrates_cancel(narrator, monkeypatch):
    import threading
    from fastapi.testclient import TestClient
    from silica.ui.web import server

    server.current_cancel = threading.Event()
    tc = TestClient(server.app)
    tc.post("/stop")
    assert server.current_cancel.is_set()
    beats = _beats(narrator)
    assert beats[-1]["kind"] == "cancel"
    assert beats[-1]["payload"]["driver"] == "gui"


def test_repl_sessions_resume_handlers(narrator, monkeypatch, capsys):
    """The REPL surface: /sessions lists, /resume rebuilds messages from the
    narration, and the meta dispatcher owns both."""
    from silica.cli import _handle_slash_command

    narrator.turn({"role": "user", "content": "prima domanda"})
    narrator.turn({"role": "assistant", "content": "prima risposta"})
    sid = narrator.sid
    narrator.close()

    monkeypatch.setattr("silica.cli._fresh_messages", lambda: [{"role": "system", "content": "seed"}])
    monkeypatch.setattr("silica.cli._update_context_tokens", lambda m: None)

    assert _handle_slash_command("/sessions", []) is True
    messages: list[dict] = []
    assert _handle_slash_command("/resume 1", messages) is True
    assert [m["content"] for m in messages] == ["seed", "prima domanda", "prima risposta"]
    from silica.agent.narration import NARRATOR
    assert NARRATOR.sid == sid              # appending continues the same account

    assert _handle_slash_command("/resume 99", []) is True   # graceful, no raise
    assert _handle_slash_command("/sessions prune", []) is True  # usage line, no delete
