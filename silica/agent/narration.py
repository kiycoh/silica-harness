# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""The narration: one append-only account per session, the source of truth
for what Silica is doing. Spec: docs/specs/narration/spec.md.

Layering: this module is agent-level on purpose. Every emit site is agent or
above (loop, subagent, commit, router, the UIs); the kernel never narrates,
so the kernel/agent import boundary stays intact (spec ticket 05).

Concurrency model, chosen over a writer thread on measurement (ticket 02:
2.4-8µs per flushed append, 0.00046% of wall-clock at the max observed
rate): every thread appends synchronously under one process-wide RLock.
The BUS publish happens inside the same lock so a live consumer can never
see beat N+1 before beat N; the file and the bus carry the same serialized
record, which is what makes projection drift impossible by construction.
fsync is deliberately absent: measured at 6.5ms per call it would cost 39%
of wall-clock, and flush-per-beat already means a process crash loses
nothing (only kernel panic / power loss can eat the page-cache tail).

NARRATOR is a process-global singleton importable anywhere, like BUS.
Tests get a fresh one per test via the conftest autouse fixture.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator

logger = logging.getLogger(__name__)

BEAT_TOPIC = "narration/beat"   # BUS topic carrying each appended record

# Core kinds (spec §4). The set is open: extension kinds carry an owner
# prefix ("codewiki/index") and a free payload; projections render unknowns
# through the mandatory `summary`, never drop them.
CORE_KINDS = frozenset({
    "session", "turn", "thought", "call", "tool", "subagent",
    "run", "phase", "work", "write", "compaction", "cancel",
})

TERMINAL_STATUSES = frozenset({
    "done", "failed", "cancelled", "committed", "rolled_back",
})

# Innermost open span of the current execution context. ContextVar and not a
# Narrator attribute for the same reason commit.py uses one for provenance:
# subagent threads set their own value without seeing each other's.
_parent_var: ContextVar[str | None] = ContextVar("narration_parent", default=None)
_run_var: ContextVar[str | None] = ContextVar("narration_run", default=None)


def narration_dir() -> Path:
    """Resolved per call, not at import: tests monkeypatch _SILICA_HOME."""
    from silica.kernel.recall.paths import _SILICA_HOME
    return _SILICA_HOME / "narration"


def _legacy_sessions_dir() -> Path:
    from silica.kernel.recall.paths import _SILICA_HOME
    return _SILICA_HOME / "web_sessions"


def _title_from(text: str) -> str:
    line = str(text).strip().splitlines()[0] if str(text).strip() else "untitled"
    return line[:57] + "…" if len(line) > 58 else line


class SessionBusy(RuntimeError):
    """Another process holds this session's narration (spec §5: single writer)."""

    def __init__(self, sid: str, owner: dict):
        self.sid, self.owner = sid, owner
        who = owner.get("driver", "unknown")
        pid = owner.get("pid", "?")
        super().__init__(f"session {sid} in use by {who}, pid {pid}")


class Narrator:
    """Appends beats to the current session file and publishes each on the BUS.

    No session open → every narrate() is a silent no-op: narration is
    session-scoped by definition, and batch entry points that never open a
    session simply do not narrate (spec §1).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._fh: BinaryIO | None = None   # open file, flock held for its lifetime
        self._sid: str | None = None
        self._seq = 0
        self._span_tokens: dict[str, Any] = {}   # call_id -> parent-var reset token
        self._thought: str | None = None         # open thought span id, if any
        self._thought_t0 = 0.0

    # --- session lifecycle -------------------------------------------------

    @property
    def sid(self) -> str | None:
        return self._sid

    def ensure_session(self, *, driver: str, sid: str | None = None,
                       seed: dict | None = None) -> str:
        """Open (or adopt) the session, writing the `session` beat on creation.

        Idempotent: an already-open session returns its sid untouched, so the
        REPL can call this at every user turn and only the first one creates
        (born at the first user turn, spec §5).
        """
        with self._lock:
            if self._fh is not None:
                return self._sid  # type: ignore[return-value]
            sid = sid or uuid.uuid4().hex[:12]
            path = narration_dir() / f"{sid}.jsonl"
            fresh = not path.exists()
            self._open_locked(sid, path)
            if fresh:
                from silica.config import CONFIG
                self._append_locked({
                    "kind": "session", "status": "done", "id": None,
                    "summary": f"{driver} session, vault "
                               f"{os.path.basename(CONFIG.vault_path or '') or '(none)'}",
                    "payload": {"v": 1, "driver": driver, "pid": os.getpid(),
                                "vault": CONFIG.vault_path or "",
                                "seed": seed or {}},
                })
            return sid

    def resume(self, sid: str) -> list[dict]:
        """Reopen an existing session for append; returns its replayed beats.

        The caller folds `turn` beats back into `messages` (the one resume
        path, spec §5). Raises FileNotFoundError for an unknown sid and
        SessionBusy when another process holds the flock.
        """
        path = narration_dir() / f"{sid}.jsonl"
        if not path.exists():
            raise FileNotFoundError(sid)
        with self._lock:
            self.close()
            self._open_locked(sid, path)
            return list(read_beats(path))

    def close(self) -> None:
        """Release the flock and forget the session. No close beat, ever
        (spec §3: an end marker that exists only on clean paths lies)."""
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()   # closing releases the flock
                except OSError:
                    pass
            self._fh = None
            self._sid = None
            self._seq = 0
            self._span_tokens.clear()
            self._thought = None

    def _open_locked(self, sid: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "a+b")
        try:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise SessionBusy(sid, _session_owner(path)) from None
        # Torn-tail repair: a crash can leave a final line without its
        # newline. Appending after it would glue two records into one corrupt
        # line, so the never-durable fragment is truncated away — safe exactly
        # because the flock above makes us the only writer.
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        if size:
            fh.seek(size - 1)
            if fh.read(1) != b"\n":
                data = path.read_bytes()
                keep = data.rfind(b"\n") + 1
                fh.truncate(keep)
                fh.seek(0, os.SEEK_END)
        self._fh = fh
        self._sid = sid
        self._seq = _last_seq(path)

    # --- the one write path -------------------------------------------------

    def narrate(self, kind: str, status: str = "done", summary: str = "",
                payload: dict | None = None, *, id: str | None = None,
                parent: str | None | object = ...) -> dict | None:
        """Append one beat. Returns the record, or None when no session is open.

        `parent` defaults to the ambient span (ContextVar); pass an explicit
        value (including None) to override. Best-effort by contract: a
        narration failure must never break the work it narrates, so I/O
        errors are logged and swallowed — the failure tolerated here is a
        full disk or a vanished home, and the work continuing without its
        account is strictly better than the work dying for it.
        """
        with self._lock:
            if self._fh is None:
                return None
            rec: dict[str, Any] = {
                "seq": self._seq + 1,
                "ts": time.time(),
                "sid": self._sid,
                "run": _run_var.get(),
                "id": id,
                "parent": _parent_var.get() if parent is ... else parent,
                "kind": kind,
                "status": status,
                "summary": summary,
                "payload": payload or {},
            }
            try:
                self._append_raw(rec)
            except OSError as e:
                logger.warning("narration append failed (non-fatal): %s", e)
                return None
            self._seq = rec["seq"]
            # Inside the lock on purpose: a live consumer must never observe
            # N+1 before N. RLock, so a subscriber that narrates cannot deadlock.
            from silica.agent.bus import BUS
            BUS.publish(BEAT_TOPIC, rec)
            return rec

    def _append_locked(self, partial: dict) -> dict:
        rec = {"seq": self._seq + 1, "ts": time.time(), "sid": self._sid,
               "run": _run_var.get(), **partial}
        self._append_raw(rec)
        self._seq = rec["seq"]
        from silica.agent.bus import BUS
        BUS.publish(BEAT_TOPIC, rec)
        return rec

    def _append_raw(self, rec: dict) -> None:
        # default=str: a tool payload that isn't JSON degrades to text rather
        # than crash — same posture _save_session took for the same data.
        line = json.dumps(rec, default=str, ensure_ascii=False) + "\n"
        # Both call sites hold _lock and have already established the session
        # is open (narrate() returns early on None, _append_locked runs right
        # after _open_locked): the assert states that invariant rather than
        # inventing a no-session branch nothing can reach.
        fh = self._fh
        assert fh is not None
        fh.write(line.encode("utf-8"))
        fh.flush()   # to the OS, never fsync (ticket 02: 6.5ms = 39%)

    # --- convenience emitters (the vocabulary, spec §4) ---------------------

    def turn(self, message: dict) -> None:
        """One beat per conversation message, full content (ticket 01 Q10)."""
        role = message.get("role", "?")
        self.narrate("turn", "done",
                     f"{role}: {_title_from(str(message.get('content') or ''))}",
                     {"message": message}, parent=None)

    def span_open(self, kind: str, span_id: str, summary: str,
                  payload: dict | None = None, *, attach: bool = False,
                  parent: str | None | object = ...) -> None:
        """Open a span; with attach=True the current context's ambient parent
        becomes this span until span_close (tool/subagent nesting)."""
        self.narrate(kind, "running", summary, payload, id=span_id, parent=parent)
        if attach and self._sid is not None:
            self._span_tokens[span_id] = _parent_var.set(span_id)

    def span_close(self, kind: str, span_id: str, status: str, summary: str,
                   payload: dict | None = None) -> None:
        tok = self._span_tokens.pop(span_id, None)
        if tok is not None:
            _parent_var.reset(tok)
        # The terminal beat carries the span's own parent, not itself: with
        # the ambient var already reset the default would self-parent when
        # tok existed, and point at the span itself when it didn't.
        self.narrate(kind, status, summary, payload, id=span_id)

    def thought_open(self) -> None:
        """ThinkingStart. A prior thought still open is closed empty first:
        text arrives only with the response, and an exception between the two
        must not leave the pair dangling into the next iteration."""
        if self._sid is None:
            return
        if self._thought is not None:
            self.thought_close("")
        self._thought = f"th-{self._seq + 1}"
        self._thought_t0 = time.time()
        self.narrate("thought", "running", "thinking", id=self._thought)

    def thought_close(self, text: str) -> None:
        """Close the open thought with the full reasoning text (durable per
        ticket 01 Q5). No open thought → no-op, not an error: worker turns
        never opened one."""
        if self._sid is None or self._thought is None:
            return
        tid, self._thought = self._thought, None
        dur = time.time() - self._thought_t0
        self.narrate("thought", "done", _title_from(text) if text else "thought (empty)",
                     {"text": text, "duration_s": round(dur, 3)}, id=tid)

    def cancel(self, *, driver: str, target: str | None, scope: str) -> None:
        self.narrate("cancel", "done",
                     f"{driver} cancelled {scope}" + (f" {target}" if target else ""),
                     {"driver": driver, "target": target, "scope": scope},
                     parent=None)

    # --- render-event adapter (loop.py's _emit calls this once) -------------

    def on_render_event(self, event: Any) -> None:
        """Translate the loop's RenderEvents into beats (spec §4 mapping).

        Stream deltas are deliberately absent: they stay ephemeral on the BUS
        (ticket 01 Q9). Thinking start maps here; the close is explicit in
        loop.py where the reasoning text exists.
        """
        from silica.agent import events as ev
        if self._sid is None:
            return
        if isinstance(event, ev.ToolStartEvent):
            args = json.dumps(event.args, default=str)[:120]
            self.span_open("tool", event.call_id, f"{event.name} {args}",
                           {"name": event.name, "args": event.args}, attach=True)
        elif isinstance(event, ev.ToolCompleteEvent):
            self.span_close("tool", event.call_id, "done",
                            f"{event.name} done in {event.duration_s:.1f}s",
                            {"name": event.name, "result": event.result,
                             "duration_s": event.duration_s})
        elif isinstance(event, ev.ToolErrorEvent):
            self.span_close("tool", event.call_id, "failed",
                            f"{event.name} failed: {_title_from(event.error)}",
                            {"name": event.name, "error": event.error})
        elif isinstance(event, ev.ThinkingStartEvent):
            self.thought_open()
        elif isinstance(event, ev.PhaseEvent):
            status = event.status if event.status in ("running", "done", "failed") else "done"
            pid = f"ph-{event.phase}-{event.file_idx}-{event.chunk_idx}"
            self.narrate("phase", status,
                         f"{event.phase} {event.status} "
                         f"[file {event.file_idx + 1}/{event.file_total}]",
                         {"phase": event.phase, "scope": event.scope,
                          "file_idx": event.file_idx, "file_total": event.file_total,
                          "chunk_idx": event.chunk_idx, "chunk_total": event.chunk_total,
                          "source_file": event.source_file, "elapsed": event.elapsed},
                         id=pid)
        # ThinkingEnd, Reasoning, LLMStream: no durable record here.


NARRATOR = Narrator()


def current_parent() -> str | None:
    return _parent_var.get()


def set_run(run_id: str | None):
    """Bind the ambient run id for this context; returns the reset token."""
    return _run_var.set(run_id)


def reset_run(token) -> None:
    _run_var.reset(token)


def set_parent(span_id: str | None):
    return _parent_var.set(span_id)


def reset_parent(token) -> None:
    _parent_var.reset(token)


# --- reader ----------------------------------------------------------------

def read_beats(path: Path, *, from_seq: int = 0) -> Iterator[dict]:
    """Yield parsed beats with seq > from_seq, in file order.

    A trailing line without newline (torn write) is skipped silently: it was
    never durable. Any other unparseable line is surfaced as a degraded
    corrupt-beat (spec §2) — mid-file corruption is a defect a projection
    must show, not swallow.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return
    lines = data.split(b"\n")
    torn = lines and lines[-1] != b""   # no final newline: last piece is torn
    body = lines[:-1]
    for i, raw in enumerate(body):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            yield {"seq": None, "ts": None, "sid": None, "run": None, "id": None,
                   "parent": None, "kind": "narration/corrupt", "status": "done",
                   "summary": f"unparseable line {i + 1}",
                   "payload": {"raw": raw[:200].decode("utf-8", "replace")}}
            continue
        if isinstance(rec, dict) and (rec.get("seq") or 0) > from_seq:
            yield rec
    if torn:
        logger.debug("narration: torn trailing line in %s skipped", path.name)


def _last_seq(path: Path) -> int:
    last = 0
    for rec in read_beats(path):
        if rec.get("seq"):
            last = rec["seq"]
    return last


def _session_owner(path: Path) -> dict:
    for rec in read_beats(path):
        if rec.get("kind") == "session":
            return rec.get("payload", {})
        break
    return {}


def messages_from_beats(beats: list[dict]) -> list[dict]:
    """The conversation as the narration recorded it: `turn` beats in order.
    This is what resume/load rebuild `messages` from (spec §5)."""
    return [b["payload"]["message"] for b in beats
            if b.get("kind") == "turn" and isinstance(b.get("payload"), dict)
            and isinstance(b["payload"].get("message"), dict)]


# --- session list (merged with legacy web_sessions, spec §5) ----------------

def list_sessions(vault: str) -> list[dict]:
    """Sessions for `vault`, newest first, across both stores.

    Narration rows are derived per file from line one (session beat: vault),
    the first turn beat (title) and mtime — no sidecar index. Reopen at 5000
    sessions, where a scan plausibly crosses ~100ms (ticket 05). Legacy
    web_sessions/*.json are recognised forever, read-only.
    """
    out: list[dict] = []
    nd = narration_dir()
    if nd.exists():
        for f in nd.glob("*.jsonl"):
            row = _narration_row(f, vault)
            if row:
                out.append(row)
    ld = _legacy_sessions_dir()
    if ld.exists():
        seen = {r["id"] for r in out}
        for f in ld.glob("*.json"):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue   # corrupt/half-written legacy file: skip, as before
            if rec.get("vault", "") != vault or rec.get("id") in seen:
                continue
            out.append({"id": rec.get("id"), "title": rec.get("title", "untitled"),
                        "updated": rec.get("updated", 0), "store": "legacy"})
    out.sort(key=lambda r: r["updated"], reverse=True)
    return out


def _narration_row(path: Path, vault: str) -> dict | None:
    title, sid = "untitled", path.stem
    for rec in read_beats(path):
        if rec.get("kind") == "session":
            if rec.get("payload", {}).get("vault", "") != vault:
                return None
        elif rec.get("kind") == "turn":
            msg = rec.get("payload", {}).get("message", {})
            if msg.get("role") == "user" and msg.get("content"):
                title = _title_from(str(msg["content"]))
                break
    else:
        # A session with no user turn yet is real but unlisted, matching the
        # old _save_session's no-op-until-named behaviour.
        return None
    try:
        updated = path.stat().st_mtime
    except OSError:
        updated = 0
    return {"id": sid, "title": title, "updated": updated, "store": "narration"}


def load_session_messages(sid: str, vault: str) -> list[dict] | None:
    """Messages for a session id from whichever store holds it, else None."""
    np = narration_dir() / f"{sid}.jsonl"
    if np.exists():
        return messages_from_beats(list(read_beats(np)))
    lp = _legacy_sessions_dir() / f"{sid}.json"
    if lp.exists():
        try:
            rec = json.loads(lp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if rec.get("vault", "") == vault:
            return rec.get("messages", [])
    return None


def prune(older_than_days: float) -> int:
    """Delete narration sessions older than the explicit age. Explicit only:
    deletion is the user's sentence to write (ticket 06). Legacy files are
    left alone — they predate the store and are not ours to garbage-collect."""
    cutoff = time.time() - older_than_days * 86400
    n = 0
    nd = narration_dir()
    if not nd.exists():
        return 0
    for f in nd.glob("*.jsonl"):
        try:
            if f.stat().st_mtime < cutoff and f.stem != (NARRATOR.sid or ""):
                f.unlink()
                n += 1
        except OSError:
            continue
    return n


def store_stats() -> dict:
    """Bytes on disk, for the doctor's thresholds (1GB store / 100MB session)."""
    nd = narration_dir()
    total = biggest = 0
    biggest_sid = ""
    if nd.exists():
        for f in nd.glob("*.jsonl"):
            try:
                s = f.stat().st_size
            except OSError:
                continue
            total += s
            if s > biggest:
                biggest, biggest_sid = s, f.stem
    return {"total_bytes": total, "biggest_bytes": biggest, "biggest_sid": biggest_sid}


# --- the fold (spec §6): shared projection state ----------------------------

@dataclass
class SpanView:
    id: str
    kind: str
    parent: str | None
    summary: str
    status: str
    started_ts: float
    ended_ts: float | None = None
    children: list = field(default_factory=list)   # span ids and point seqs, in order


@dataclass
class ViewState:
    """`fold(state, beat)`: one pure synchronous unit for every surface.

    Unlike the design prototype (assets/proto_fold.py), which asserts on a
    seq gap, this fold records gaps and duplicates instead of raising: a
    projection's job is to render the truth, including the truth that data
    is missing.
    """
    cursor: int = 0
    spans: dict = field(default_factory=dict)
    roots: list = field(default_factory=list)
    points: dict = field(default_factory=dict)
    context_tokens: int = 0
    cost_tokens: int = 0
    cancelling: set = field(default_factory=set)
    gaps: list = field(default_factory=list)


def fold(st: ViewState, b: dict) -> ViewState:
    seq = b.get("seq")
    if seq is not None:
        if seq <= st.cursor:
            return st                      # duplicate (SSE replay overlap): drop
        if seq > st.cursor + 1:
            st.gaps.append((st.cursor + 1, seq - 1))
        st.cursor = seq
    kind, status = b.get("kind", "?"), b.get("status", "done")
    span_id = b.get("id")
    parent = b.get("parent")
    if span_id is None:
        key = seq if seq is not None else f"x{len(st.points)}"
        st.points[key] = b
        (st.spans[parent].children if parent in st.spans else st.roots).append(key)
        if kind == "cancel":
            tgt = (b.get("payload") or {}).get("target")
            if tgt:
                st.cancelling.add(tgt)
        return st
    if span_id not in st.spans:
        sv = SpanView(span_id, kind, parent, b.get("summary", ""), status,
                      b.get("ts") or 0.0)
        st.spans[span_id] = sv
        (st.spans[parent].children if parent in st.spans else st.roots).append(span_id)
    else:
        sv = st.spans[span_id]
        sv.status, sv.summary = status, b.get("summary", sv.summary)
    if status in TERMINAL_STATUSES and sv.ended_ts is None:
        sv.ended_ts = b.get("ts")
        st.cancelling.discard(span_id)
    if kind == "call" and status == "done":
        p = b.get("payload") or {}
        st.context_tokens = p.get("prompt_tokens") or st.context_tokens
        st.cost_tokens += p.get("completion_tokens") or 0
    return st


def fold_all(beats) -> ViewState:
    st = ViewState()
    for b in beats:
        fold(st, b)
    return st
