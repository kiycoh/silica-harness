# SPDX-License-Identifier: AGPL-3.0-or-later
"""Seam tests for the narration store (docs/specs/narration/spec.md).

Each test is the smallest thing that fails if its seam breaks: gap-free seq,
torn-tail repair, single-writer flock, legacy list merge, fold join equality.
"""
from __future__ import annotations

import json
import os
import threading

import pytest

import silica.agent.narration as narr
from silica.agent.narration import (
    ViewState, fold, fold_all, list_sessions, load_session_messages,
    messages_from_beats, prune, read_beats, store_stats,
)


@pytest.fixture()
def narrator(_fresh_narrator, monkeypatch):
    monkeypatch.setattr("silica.config.CONFIG.vault_path", "/v/demo", raising=False)
    return _fresh_narrator


def _open(narrator, driver="tui"):
    return narrator.ensure_session(driver=driver)


# --- writer -----------------------------------------------------------------

def test_seq_is_gap_free_and_session_beat_first(narrator):
    sid = _open(narrator)
    narrator.turn({"role": "user", "content": "ciao"})
    narrator.narrate("tool", "done", "read_note x")
    beats = list(read_beats(narr.narration_dir() / f"{sid}.jsonl"))
    assert [b["seq"] for b in beats] == [1, 2, 3]
    assert beats[0]["kind"] == "session"
    assert beats[0]["payload"]["v"] == 1
    assert beats[0]["payload"]["driver"] == "tui"


def test_no_session_means_silent_noop(narrator):
    assert narrator.narrate("tool", "done", "x") is None


def test_ensure_session_is_idempotent(narrator):
    assert _open(narrator) == _open(narrator)


def test_appends_are_thread_safe_gap_free(narrator):
    sid = _open(narrator)
    def worker():
        for _ in range(50):
            narrator.narrate("tool", "done", "t")
    ts = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    seqs = [b["seq"] for b in read_beats(narr.narration_dir() / f"{sid}.jsonl")]
    assert seqs == list(range(1, 202))


def test_bus_carries_the_appended_record(narrator):
    from silica.agent.bus import BUS
    got = []
    BUS.subscribe(narr.BEAT_TOPIC, got.append)
    _open(narrator)
    rec = narrator.narrate("tool", "done", "x")
    assert got[-1] is rec  # same object: file line and bus payload cannot drift


# --- torn writes and resume -------------------------------------------------

def test_torn_tail_is_truncated_on_reopen(narrator):
    sid = _open(narrator)
    narrator.turn({"role": "user", "content": "hello"})
    path = narr.narration_dir() / f"{sid}.jsonl"
    narrator.close()
    with open(path, "ab") as fh:
        fh.write(b'{"seq":3,"kind":"tool"')   # crash mid-write, no newline
    beats = narrator.resume(sid)
    assert [b["seq"] for b in beats] == [1, 2]
    narrator.narrate("tool", "done", "after resume")
    beats = list(read_beats(path))
    assert [b["seq"] for b in beats] == [1, 2, 3]   # seq continues, no glue


def test_reader_surfaces_midfile_corruption_as_degraded_beat(narrator):
    sid = _open(narrator)
    narrator.turn({"role": "user", "content": "a"})
    path = narr.narration_dir() / f"{sid}.jsonl"
    narrator.close()
    lines = path.read_bytes().split(b"\n")
    lines[0] = b"garbage{{{"
    path.write_bytes(b"\n".join(lines))
    kinds = [b["kind"] for b in read_beats(path)]
    assert "narration/corrupt" in kinds and "turn" in kinds


def test_resume_replays_turn_beats_into_messages(narrator):
    sid = _open(narrator)
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "tool_calls": []}]
    for m in msgs:
        narrator.turn(m)
    narrator.close()
    beats = narrator.resume(sid)
    assert messages_from_beats(beats) == msgs


# --- single writer ----------------------------------------------------------

def test_second_writer_is_refused_with_owner(narrator):
    sid = _open(narrator)
    second = narr.Narrator()
    with pytest.raises(narr.SessionBusy) as e:
        second.resume(sid)
    assert e.value.owner.get("driver") == "tui"
    assert e.value.owner.get("pid") == os.getpid()


# --- spans, adapter, cancel -------------------------------------------------

def test_span_attach_parents_inner_beats(narrator):
    sid = _open(narrator)
    narrator.span_open("subagent", "a1", "dedup X", attach=True)
    inner = narrator.narrate("write", "committed", "patch X")
    narrator.span_close("subagent", "a1", "done", "dedup done")
    beats = list(read_beats(narr.narration_dir() / f"{sid}.jsonl"))
    assert inner["parent"] == "a1"
    assert beats[-1]["id"] == "a1" and beats[-1]["parent"] is None


def test_render_event_adapter_tool_lifecycle(narrator):
    from silica.agent.events import ToolStartEvent, ToolCompleteEvent
    sid = _open(narrator)
    narrator.on_render_event(ToolStartEvent("silica_recall", {"q": "x"}, "c1", 1))
    assert narr.current_parent() == "c1"   # writes inside the tool attribute to it
    narrator.on_render_event(ToolCompleteEvent("silica_recall", {"q": "x"}, "c1",
                                               "ok", 0.5, 1))
    assert narr.current_parent() is None
    beats = list(read_beats(narr.narration_dir() / f"{sid}.jsonl"))
    tool = [b for b in beats if b["kind"] == "tool"]
    assert [b["status"] for b in tool] == ["running", "done"]
    assert tool[0]["id"] == tool[1]["id"] == "c1"


def test_thought_pairs_and_dangling_repair(narrator):
    sid = _open(narrator)
    narrator.thought_open()
    narrator.thought_open()          # a crash between pairs must not dangle
    narrator.thought_close("full reasoning text")
    beats = [b for b in read_beats(narr.narration_dir() / f"{sid}.jsonl")
             if b["kind"] == "thought"]
    assert [b["status"] for b in beats] == ["running", "done", "running", "done"]
    assert beats[1]["payload"]["text"] == ""          # dangling closed empty
    assert beats[3]["payload"]["text"] == "full reasoning text"


def test_cancel_beat_records_driver_and_target(narrator):
    sid = _open(narrator)
    narrator.cancel(driver="tui", target="r1", scope="run")
    b = list(read_beats(narr.narration_dir() / f"{sid}.jsonl"))[-1]
    assert b["kind"] == "cancel" and b["payload"] == {
        "driver": "tui", "target": "r1", "scope": "run"}


# --- list / load / prune / stats -------------------------------------------

def _legacy_session(sid, vault, title="old chat"):
    d = narr._legacy_sessions_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}.json").write_text(json.dumps({
        "id": sid, "title": title, "vault": vault, "updated": 5.0,
        "messages": [{"role": "user", "content": title}]}))


def test_list_merges_both_stores_and_filters_vault(narrator):
    _open(narrator)
    narrator.turn({"role": "user", "content": "narration chat"})
    _legacy_session("aaa111", "/v/demo")
    _legacy_session("bbb222", "/v/other")
    rows = list_sessions("/v/demo")
    assert [r["store"] for r in rows] == ["narration", "legacy"]
    assert rows[0]["title"] == "narration chat"
    assert all(r["id"] != "bbb222" for r in rows)


def test_session_without_user_turn_is_unlisted(narrator):
    _open(narrator)
    assert list_sessions("/v/demo") == []


def test_load_reads_either_store(narrator):
    sid = _open(narrator)
    narrator.turn({"role": "user", "content": "hi"})
    _legacy_session("ccc333", "/v/demo", "legacy hi")
    assert load_session_messages(sid, "/v/demo")[0]["content"] == "hi"
    assert load_session_messages("ccc333", "/v/demo")[0]["content"] == "legacy hi"
    assert load_session_messages("nope", "/v/demo") is None


def test_prune_spares_live_session_and_legacy(narrator):
    sid = _open(narrator)
    narrator.turn({"role": "user", "content": "live"})
    old = narr.narration_dir() / "deadbeef0000.jsonl"
    old.write_text('{"seq":1,"kind":"session","payload":{"vault":"/v/demo"}}\n')
    os.utime(old, (0, 0))
    _legacy_session("ddd444", "/v/demo")
    assert prune(30) == 1
    assert not old.exists()
    assert (narr.narration_dir() / f"{sid}.jsonl").exists()
    assert (narr._legacy_sessions_dir() / "ddd444.json").exists()


def test_store_stats_names_the_biggest(narrator):
    sid = _open(narrator)
    narrator.turn({"role": "user", "content": "x" * 500})
    s = store_stats()
    assert s["total_bytes"] > 0 and s["biggest_sid"] == sid


# --- fold -------------------------------------------------------------------

def _beats():
    return [
        dict(seq=1, ts=0.0, kind="session", status="done", id=None, parent=None,
             summary="s", payload={"v": 1}),
        dict(seq=2, ts=1.0, kind="run", status="running", id="r1", parent=None,
             summary="nucleate", payload={}),
        dict(seq=3, ts=1.1, kind="subagent", status="running", id="a1",
             parent="r1", summary="dedup", payload={}),
        dict(seq=4, ts=1.2, kind="call", status="done", id="c1", parent="a1",
             summary="llm", payload={"prompt_tokens": 4100, "completion_tokens": 210}),
        dict(seq=5, ts=1.3, kind="codewiki/index", status="done", id=None,
             parent="r1", summary="indexed", payload={}),
        dict(seq=6, ts=2.0, kind="cancel", status="done", id=None, parent=None,
             summary="stop", payload={"target": "r1"}),
        dict(seq=7, ts=2.1, kind="subagent", status="cancelled", id="a1",
             parent="r1", summary="cancelled", payload={}),
        dict(seq=8, ts=2.2, kind="run", status="cancelled", id="r1", parent=None,
             summary="cancelled", payload={}),
    ]


def test_fold_join_then_follow_equals_having_been_there():
    live = fold_all(_beats())
    joined = fold_all(_beats()[:4])
    for b in _beats()[4:]:
        fold(joined, b)
    assert joined == live
    assert live.spans["r1"].children == ["a1", 5]
    assert live.context_tokens == 4100 and live.cost_tokens == 210
    assert not live.cancelling                     # terminal beats cleared it


def test_fold_records_gaps_and_drops_duplicates_instead_of_raising():
    st = ViewState()
    fold(st, _beats()[0])
    fold(st, _beats()[0])          # SSE replay overlap: duplicate dropped
    fold(st, _beats()[3])          # seq jumps 1 -> 4
    assert st.cursor == 4 and st.gaps == [(2, 3)]
