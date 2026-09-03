# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Capture lane producer — agent session transcripts into a per-vault WAL.

No structure at capture time, ever: this module never calls an LLM and never
writes a note. It deposits the raw transcript as a JSON envelope in
~/.silica/inbox/<digest12>/ and stops. Parsing lives in
`silica.sources.transcript`; the vault write happens later, at drain, through
the same FSM and write gate as everything else.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import time
from datetime import UTC, datetime
from pathlib import Path

from silica.kernel.recall.paths import inbox_dir_for, is_system_dir

logger = logging.getLogger(__name__)

# Under a KiB a transcript is a session that did nothing; the drain would pay
# an LLM call to learn that.
MIN_TRANSCRIPT_BYTES = 1024

_EVENTS = {"SessionEnd": "session_end", "PreCompact": "pre_compact"}


def find_vault(cwd: str) -> Path | None:
    """The vault `cwd` belongs to: nearest ancestor holding `vault.yaml`.

    The walk-up is what makes a hook fired from `<repo>/silica/kernel` capture
    to `<repo>`. Any adopted vault has the manifest — `declare_write_dir` writes
    it on first launch, since every vault now declares a write boundary — so a
    folder without one is a folder Silica has never been pointed at.
    """
    d = Path(cwd)
    for candidate in (d, *d.parents):
        # A manifest a stray launch left in /tmp is not a vault, and stopping
        # here would hand every session under it someone else's.
        if is_system_dir(candidate):
            continue
        if (candidate / "vault.yaml").is_file():
            return candidate
    return None


def write_envelope(vault: str, name: str, envelope: dict) -> Path:
    """Atomically deposit one envelope in the vault's WAL. Returns its path."""
    d = inbox_dir_for(vault)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, stat.S_IRWXU)  # mkdir's mode is umask-masked; this is not
    path = d / name
    tmp = d / f".tmp-{name}"
    tmp.write_text(json.dumps(envelope), encoding="utf-8")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(tmp, path)
    return path


def _move(path: Path, subdir: str) -> Path:
    """Retire one envelope into `processed/` or `failed/`, name preserved."""
    dest_dir = path.parent / subdir
    dest_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    dest = dest_dir / path.name
    os.replace(path, dest)
    return dest


def mark_processed(path: Path) -> Path:
    return _move(path, "processed")


def mark_failed(path: Path) -> Path:
    return _move(path, "failed")


def _session_of(path: Path) -> str:
    """The session an envelope belongs to, from its deterministic name.

    `<source>-<session>-end.json` / `<source>-<session>-precompact-<ts>.json`;
    anything else (an import envelope) is its own session, so the supersede
    rule below never groups two imports together.

    `silica-<session>-clear-<ts>.json` deliberately falls in that "anything
    else" bucket: `/clear` destroys the conversation, so the session's own end
    transcript does not contain it and must not supersede it. Adding a
    `-clear-` marker here would silently drop every cleared conversation.
    """
    stem = path.stem
    for marker in ("-end", "-precompact-"):
        head, sep, _ = stem.partition(marker)
        if sep:
            return head
    return stem


def housekeep(vault: str, *, days: int = 30) -> None:
    """Truncate drained envelopes older than `days` to zero bytes.

    Truncated, not deleted: the filename is what carries import idempotency,
    so deleting would silently expire the dedup guarantee and let a re-import
    re-nucleate a year of conversations.
    """
    cutoff = time.time() - days * 86400
    processed = inbox_dir_for(vault) / "processed"
    for p in processed.glob("*.json") if processed.is_dir() else ():
        if p.stat().st_mtime < cutoff and p.stat().st_size:
            os.truncate(p, 0)


def collect(vault: str, *, cap: int = 10) -> tuple[list[Path], int]:
    """Envelopes to drain now, plus how many stay pending after this batch.

    Superseded envelopes are retired to `processed/` here rather than
    processed: an end-of-session transcript is cumulative, so its session's
    compactions carry nothing new, and when a session has only compactions the
    latest one subsumes the earlier ones the same way.
    """
    d = inbox_dir_for(vault)
    if not d.is_dir():
        return [], 0
    pending = sorted(d.glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name))

    by_session: dict[str, list[Path]] = {}
    for p in pending:
        by_session.setdefault(_session_of(p), []).append(p)
    keep: list[Path] = []
    for group in by_session.values():
        ends = [p for p in group if p.stem.endswith("-end")]
        # No end envelope ⇒ latest compaction wins, and the UTC stamp in the
        # name is what orders them (mtime would order by capture-to-disk,
        # which a retroactive import writes out of order).
        winner = ends[-1] if ends else max(group, key=lambda p: p.name)
        keep.append(winner)
        for p in group:
            if p is not winner:
                mark_processed(p)

    keep.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return keep[:cap], max(0, len(keep) - cap)


def _import_envelope(source: str, fmt: str, conversation: dict, *,
                     session_id: str, title: str) -> dict:
    return {
        "version": 1,
        "source": source,
        "event": "import",
        "format": fmt,
        "captured_at": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "cwd": "",
        "title": title,
        "payload": json.dumps(conversation),
    }


def _explode_conversations(conversations: list[dict]) -> list[tuple[str, dict]]:
    """One (envelope name, envelope) pair per conversation in an export.

    ChatGPT and claude.ai ship the same file name (`conversations.json`) with
    different schemas; the discriminator is which message container the object
    carries, not anything either vendor declares.
    """
    out: list[tuple[str, dict]] = []
    for conversation in conversations:
        if not isinstance(conversation, dict):
            continue
        if "mapping" in conversation:
            source, fmt = "chatgpt", "chatgpt-mapping"
            cid = str(conversation.get("conversation_id") or conversation.get("id") or "")
            title = str(conversation.get("title") or "")
        elif "chat_messages" in conversation:
            source, fmt = "claude-ai", "claude-ai-messages"
            cid = str(conversation.get("uuid") or "")
            title = str(conversation.get("name") or "")
        else:
            continue
        if not cid:
            continue
        digest = hashlib.sha256(cid.encode("utf-8")).hexdigest()[:12]
        out.append((
            f"{source}-{digest}.json",
            _import_envelope(source, fmt, conversation, session_id=cid, title=title),
        ))
    return out


# A conversation this small is a greeting or a one-shot lookup: the drain
# would pay an LLM call to discover it holds nothing. Measured on extracted
# text, so it holds for any source.
MIN_CONVERSATION_MESSAGES = 2
MIN_CONVERSATION_CHARS = 200


def _is_trivial(envelope: dict) -> bool:
    from silica.sources.transcript import parse

    turns = parse(envelope)
    return (len(turns) < MIN_CONVERSATION_MESSAGES
            or sum(len(text) for _, text, _ in turns) < MIN_CONVERSATION_CHARS)


def _jsonl_envelopes(target: Path) -> list[tuple[str, dict]]:
    """Local Claude Code transcripts: one envelope per `.jsonl` session file.

    The live hook's own name is reused (`-end`), which is what makes a session
    already captured live skip itself on a retroactive import.
    """
    files = sorted(target.rglob("*.jsonl")) if target.is_dir() else [target]
    out: list[tuple[str, dict]] = []
    for f in files:
        session_id = f.stem
        envelope = _import_envelope(
            "claude-code", "claude-code-jsonl", {},
            session_id=session_id, title="",
        )
        envelope["payload"] = f.read_text(encoding="utf-8", errors="replace")
        out.append((f"claude-code-{session_id}-end.json", envelope))
    return out


def _export_envelopes(target: str) -> list[tuple[str, dict]]:
    """Envelope candidates for whatever the user pointed `silica import` at.

    Three shapes cover every migration path: an export archive, the
    `conversations.json` inside it, and a tree of local Claude Code
    transcripts.
    """
    path = Path(target)
    if path.suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as z:
            name = next((n for n in z.namelist()
                         if n.rsplit("/", 1)[-1] == "conversations.json"), "")
            if not name:
                raise ValueError(f"no conversations.json inside {path.name}")
            return _explode_conversations(json.loads(z.read(name)))
    if path.is_dir() or path.suffix == ".jsonl":
        return _jsonl_envelopes(path)
    return _explode_conversations(json.loads(path.read_text(encoding="utf-8")))


def run_import(target: str, vault: str) -> tuple[int, int]:
    """Explode a conversation export into the vault's WAL. Zero LLM calls.

    Returns (created, skipped). The only parsing here is the triviality
    measure, and it goes through the transcript adapter rather than reading
    the vendor schemas a second time.
    """
    d = inbox_dir_for(vault)
    created = skipped = 0
    for name, envelope in _export_envelopes(target):
        # Existence, not content: a `processed/` entry truncated by
        # housekeeping is still the record that this conversation went
        # through, and re-importing it would re-nucleate it.
        if any((d / sub / name).exists() for sub in ("", "processed", "failed")):
            skipped += 1
            continue
        if _is_trivial(envelope):
            skipped += 1
            continue
        write_envelope(vault, name, envelope)
        created += 1
    return created, skipped


def capture_session(messages: list[dict], *, session_id: str, driver: str,
                    event: str = "session_end",
                    notes_touched: list[str] | tuple = ()) -> Path | None:
    """Deposit one of Silica's OWN conversations in the WAL. Opt-in, no LLM.

    Not a hook: the TUI and GUI call this in-process at the points where a
    conversation ends (`/exit`, `/clear`, new chat, shutdown). What lands is
    the conversation and nothing else — tool traffic is dropped here, at the
    source, because the session knows its own history; what mattered about the
    tools travels as `notes_touched` instead.
    Returns the envelope path, or None when capture is off or the session said
    nothing worth keeping. Fail-open like the hook producer, and for the same
    reason: a capture bug must never break — or noise up — the session it was
    capturing, so a failure is a None here rather than a raise at every call
    site.
    """
    from silica.config import CONFIG

    try:
        if not getattr(CONFIG, "capture_sessions", False):
            return None
        vault = (getattr(CONFIG, "vault_path", "") or "").strip()
        if not vault:
            return None
        turns = [
            # No per-turn clock (declined 2026-08-19): the TUI keeps none, and
            # stamping every message for a default-off feature costs every
            # session. `captured_at` dates the conversation; real per-turn ts
            # only when something reads it.
            {"role": str(m["role"]), "content": str(m["content"]), "ts": ""}
            for m in messages
            if m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str) and m["content"].strip()
        ]
        # Same floor as the importer, and for the same reason. Deliberately NOT
        # the 1 KiB of section 3: that one measures a raw JSONL file, whose
        # per-line overhead is most of it, while here the text is already
        # extracted.
        if (len(turns) < MIN_CONVERSATION_MESSAGES
                or sum(len(t["content"]) for t in turns) < MIN_CONVERSATION_CHARS):
            return None
        # A session ends once (idempotent name) but is cleared many times, and
        # each clear is a whole conversation the end transcript will not contain.
        name = (
            f"silica-{session_id}-end.json" if event == "session_end"
            else f"silica-{session_id}-clear-"
                 f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        return write_envelope(vault, name, {
            "version": 1,
            "source": "silica",
            "event": event,
            "format": "silica-session",
            "captured_at": datetime.now(UTC).isoformat(),
            "session_id": session_id,
            "cwd": os.getcwd(),
            "title": "",
            "payload": json.dumps(turns),
            "driver": driver,
            "notes_touched": list(notes_touched),
        })
    except Exception as exc:
        logger.debug("session capture skipped (non-fatal): %s", exc)
        return None


def run_capture(stdin_text: str) -> int:
    """Claude Code hook entry: read the hook JSON, deposit one envelope.

    Fail-open is the contract, hence the blanket except: every path exits 0
    with nothing on stdout, because a capture bug must never break or noise up
    a Claude Code session.
    """
    try:
        hook = json.loads(stdin_text)
        cwd = hook.get("cwd") or os.getcwd()
        vault = find_vault(cwd)
        if vault is None:
            return 0
        transcript = Path(hook.get("transcript_path") or "")
        if transcript.stat().st_size < MIN_TRANSCRIPT_BYTES:
            return 0
        session_id = hook.get("session_id") or ""
        event = _EVENTS[hook.get("hook_event_name", "")]
        captured_at = datetime.now(UTC).isoformat()
        # A session ends once (idempotent name, re-capture overwrites) but
        # compacts many times, so each compaction keeps its own envelope.
        name = (
            f"claude-code-{session_id}-end.json" if event == "session_end"
            else f"claude-code-{session_id}-precompact-"
                 f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        write_envelope(str(vault), name, {
            "version": 1,
            "source": "claude-code",
            "event": event,
            "format": "claude-code-jsonl",
            "captured_at": captured_at,
            "session_id": session_id,
            "cwd": cwd,
            "title": "",
            "payload": transcript.read_text(encoding="utf-8", errors="replace"),
        })
    except Exception:
        pass
    return 0
