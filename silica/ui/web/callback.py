# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""RenderEvent -> JSON. The single source of truth for the wire event map.

Mirrors the table in docs/specs/gui-web.md. Reasoning/thinking events are
dropped in v1 (return None -> the callback skips them).
"""
from __future__ import annotations

from typing import Any

import json

from silica.agent.events import (
    BatchRunStartEvent,
    LLMStreamEvent,
    PhaseEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ToolStartEvent,
)
from silica.ui.renderer import (  # same verb + target the TUI shows
    _CHUNK_PHASES,
    _FILE_PHASES,
    _PHASE_LABELS,
    _tool_target,
    _tool_verb,
)

# How a tool changes the vault, for the chat footer's grouping. Only the tools
# that mutate notes are listed; everything else is a read. Deliberately keyed on
# the note-touching surface (the tools whose args `_note_refs` can resolve), so a
# batch pipeline that names an ops file rather than a note produces no chip and
# needs no entry here.
_TOOL_EFFECT: dict[str, str] = {
    "silica_write_note": "written",
    "silica_patch_note": "written",
    "silica_flag_note": "written",
    "silica_bulk_write": "written",
    "silica_restore": "written",
    "silica_delete": "deleted",
    "silica_move": "moved",
}

# Arg keys that name a note across the tool surface (read=name, write=path,
# related=note, mindmap=note_path, move/delete=ref). A small allowlist, not
# per-tool logic: missing one only omits a chip from the chat 'sources' footer,
# it never reports a wrong note.
_NOTE_KEYS = ("name", "path", "note", "note_path", "ref")


# What the expandable tool card carries. A read of a long note returns the whole
# note, and the transcript would grow a second copy of the vault; the drawer is
# one click away and holds the real thing, so the card only has to show enough to
# recognise WHICH result this was. The number is a character count and not a line
# count on purpose: one minified JSON line and forty lines of frontmatter cost
# the reader the same attention and must cost the frame the same bytes.
_CARD_CHARS = 1200


def _card_text(value: Any) -> dict | None:
    """One side of a tool card: the text, and whether it was cut.

    Cut is reported rather than elided into the string, because "…" at the end of
    a result is indistinguishable from a result that genuinely ends in an
    ellipsis, and the card has to be able to say which happened.
    """
    if value is None:
        return None
    # An argument-less call would otherwise open onto the two characters "{}",
    # which is a disclosure that costs a click to say nothing.
    if isinstance(value, (dict, list)) and not value:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.strip()
    if not text:
        return None
    return {"text": text[:_CARD_CHARS], "cut": max(0, len(text) - _CARD_CHARS)}


def _note_refs(args: dict) -> list[str]:
    refs = [args[k].strip() for k in _NOTE_KEYS
            if isinstance(args.get(k), str) and args[k].strip()]
    paths = args.get("note_paths")
    if isinstance(paths, list):
        refs += [p.strip() for p in paths if isinstance(p, str) and p.strip()]
    return refs


# The injector's two phase tracks, in run order, sent once at tool_start so the
# client can draw the whole pipeline greyed out instead of growing it a row at a
# time. rollback is absent by design: it is an exception branch (on_gate_fail in
# recipes/injector.yaml), and listing it made every healthy run display a pending
# "rollback" step that was never going to run.
_PHASE_TRACKS = {
    "file": list(_FILE_PHASES.values()),
    "chunk": list(_CHUNK_PHASES.values()),
}

# final_status as the FSM writes it -> what the user is told. The FSM emits
# "Success" capitalised and the rest lowercase (states/finalize.py); normalising
# here rather than renaming at the source keeps cli.py's _DRAIN_SETTLED and the
# FSM tests on their existing contract.
_STATUS_TEXT: dict[str, tuple[str, str]] = {
    "success":           ("ok", ""),
    "partial":           ("partial", ""),
    "no_ops":            ("empty", "no operations produced"),
    "already_nucleated": ("empty", "already in the vault"),
}


def _injector_summary(result: str | None) -> dict:
    """The completion line of a run_injector call, from its stored result.

    Shared by the live tool_done event and by transcript replay, so a reloaded
    chat states the same outcome it stated while streaming.
    """
    try:
        data = json.loads(result) if isinstance(result, str) else {}
    except (TypeError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    raw = str(data.get("final_status") or "unknown")
    kind, reason = _STATUS_TEXT.get(raw.lower(), ("failed", ""))
    failed = [f for f in (data.get("failed_chunks") or []) if isinstance(f, dict)]
    return {
        "kind": kind,               # ok | partial | empty | failed
        "reason": reason,
        "status": raw.lower(),
        "notes": data.get("yield_notes") or 0,
        "links": data.get("yield_links") or 0,
        "files": data.get("files_total") or 0,
        "committed": data.get("chunks_committed") or 0,
        "failed_chunks": [{"chunk": f.get("chunk", ""), "phase": f.get("phase", "")}
                          for f in failed],
    }


def event_to_json(ev) -> dict | None:
    if isinstance(ev, LLMStreamEvent):
        return {"type": "delta", "kind": ev.chunk_type, "text": ev.content}
    if isinstance(ev, ToolStartEvent):
        # A move leaves the note at `to`, so that is the ref worth offering as a
        # chip; `ref` is a path that no longer resolves once the move lands.
        notes = _note_refs(ev.args)
        if ev.name == "silica_move" and isinstance(ev.args.get("to"), str):
            notes = [ev.args["to"].strip()]
        out: dict[str, Any] = {
               "type": "tool_start", "name": _tool_verb(ev.name), "id": ev.call_id,
               "target": _tool_target(ev.name, ev.args),
               "effect": _TOOL_EFFECT.get(ev.name, "read"),
               "notes": notes}
        # The call's own arguments are the IN half of the expandable card: the
        # one-word target the row prints answers "which note", never "asking
        # what of it", and a recall whose query the reader cannot see is a step
        # they have to take on trust.
        if (card := _card_text(ev.args)):
            out["input"] = card
        if ev.name == "silica_run_injector":
            out["pipeline"] = _PHASE_TRACKS
        return out
    if isinstance(ev, ToolCompleteEvent):
        out = {"type": "tool_done", "name": _tool_verb(ev.name), "id": ev.call_id}  # noqa: F841 (same shape)
        # Measured by the loop around the actual call, not by a client timer
        # started when the event was painted: an SSE frame can land a frame late
        # and a browser tab in the background stops scheduling altogether, so a
        # client clock reports the transport's latency as the tool's cost.
        out["ms"] = round(ev.duration_s * 1000)
        if (card := _card_text(ev.result)):
            out["output"] = card
        if ev.name == "silica_run_injector":
            out["summary"] = _injector_summary(ev.result)
        return out
    if isinstance(ev, PhaseEvent):
        # The label, not the recipe id: the client matches rows by exact string,
        # and no id->label rule it could apply covers hub_update/hub-update.
        # Guessing left the renamed phase grey for the whole run.
        return {"type": "phase", "phase": _PHASE_LABELS.get(ev.phase, ev.phase),
                "status": ev.status,
                "scope": ev.scope, "source_file": ev.source_file,
                "file_idx": ev.file_idx, "file_total": ev.file_total,
                "chunk_idx": ev.chunk_idx, "chunk_total": ev.chunk_total}
    if isinstance(ev, ToolErrorEvent):
        return {"type": "tool_error", "name": _tool_verb(ev.name), "id": ev.call_id, "error": ev.error}
    if isinstance(ev, BatchRunStartEvent):
        return {"type": "batch", "kind": ev.kind, "label": ev.label}
    return None  # ReasoningEvent / Thinking* — ignored in v1


def tool_calls_to_json(
    msg: dict,
    failed: set[str] | None = None,
    results: dict[str, str] | None = None,
) -> list[dict]:
    """The tool lines of a *stored* assistant message, for transcript replay.

    Same verb + target the live `tool_start` event carries, so reopening a chat
    shows the steps it showed while streaming. Without this the reload dropped
    every tool call and the answer read as if the agent had touched nothing.

    `results` maps tool_call_id -> stored result content. A nucleate run's
    outcome lives only in that result, so without it a reloaded chat showed a
    bare "injector" line for a run that had reported notes, links and failures.
    """
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        line = {"name": _tool_verb(name), "target": _tool_target(name, args),
                "error": bool(failed and tc.get("id") in failed)}
        if name == "silica_run_injector" and results:
            line["summary"] = _injector_summary(results.get(tc.get("id", "")))
        out.append(line)
    return out
