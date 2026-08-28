# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

from __future__ import annotations
import re
import json
import logging
import time
from typing import Callable
from urllib.parse import urlsplit
from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text
from rich.markup import escape
from rich.console import Group, RenderableType
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TaskID, TimeElapsedColumn
from silica.agent.events import (
    ToolStartEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ReasoningEvent,
    RenderEvent,
    ThinkingStartEvent,
    ThinkingEndEvent,
    LLMStreamEvent,
    BatchRunStartEvent,
)
from silica.config import CONFIG
from silica.ui.console import CONSOLE
from silica.ui.style import GLYPHS

logger = logging.getLogger(__name__)

def emit_phase(event) -> None:
    """Called by InjectorFSM to surface a phase transition to whoever is watching.

    Publishes on BUS rather than calling a registered hook: the TUI is not the
    only frontend any more (the web turn subscribes too, silica/ui/web/server.py),
    and a module-global hook admits exactly one consumer and cannot be observed
    from a test without writing into that global.
    """
    try:
        from silica.agent.bus import BUS
        BUS.publish("work/phase", event)
    except Exception:
        pass


_batch_run_hook: Callable[[RenderEvent], None] | None = None


def emit_batch_event(event: RenderEvent) -> None:
    """Called from cli.py to signal a batch run start. No-op if renderer not registered."""
    if _batch_run_hook is not None:
        try:
            _batch_run_hook(event)
        except Exception:
            pass


_REASONING_MAX_LINES = 20
_RESULT_HEAD_LINES = 3
_RESULT_LINE_CHARS = 120

_REDACT_PATTERNS = [
    re.compile(r'(api_?key|token|secret|password|auth|bearer)\s*[=:]\s*\S+', re.I),
    re.compile(r'"(api_?key|token|secret|password)"\s*:\s*"[^"]*"', re.I),
]


def _redact(text: str) -> str | None:
    """Returns None if redaction fails (fail-closed)."""
    try:
        for pattern in _REDACT_PATTERNS:
            text = pattern.sub(r'\1=[REDACTED]', text)
        return text
    except Exception as exc:
        logger.debug("Redaction failed (omitting detail): %s", exc)
        return None


def _head_cap(text: str, max_lines: int = _REASONING_MAX_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    extra = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n[… +{extra} lines]"


def _fmt_dur(seconds: float) -> str:
    """Human duration: omitted under 0.1s, one decimal under a minute, 1m04s beyond."""
    if seconds < 0.1:
        return ""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _head_result(text: str) -> str:
    """Return first N lines of a result, each line length-capped, with overflow count."""
    lines = text.splitlines()
    extra = max(0, len(lines) - _RESULT_HEAD_LINES)
    shown = [
        ln if len(ln) <= _RESULT_LINE_CHARS else ln[:_RESULT_LINE_CHARS] + "…"
        for ln in lines[:_RESULT_HEAD_LINES]
    ]
    result = "\n".join(shown)
    if extra:
        result += f"\n(+{extra} more lines)"
    return result


# tool → (verb, arg key shown as the bold target). Command-form grammar: `verb "target"`.
_TOOL_DESC: dict[str, tuple[str, str | None]] = {
    # Two tools, two verbs. Both read "search" until 2026-08-27, and the panel
    # then showed the same `search "boosting"` line answering `{"paths": []}`
    # once and a page of body snippets the next — same badge, same argument,
    # nothing on screen saying one matches titles and the other matches text.
    "silica_search": ("search titles", "query"),
    "silica_search_context": ("search inside", "query"),
    "silica_read_note": ("read", "name"),
    "silica_write_note": ("write note", "path"),
    "silica_patch_note": ("patch note", "name"),
    "silica_flag_note": ("flag note", "name"),
    "silica_record_quiz": ("record quiz", None),
    "silica_review_queue": ("review queue", None),
    "silica_props": ("props", "name"),
    "silica_outline": ("outline", "name"),
    "silica_links": ("links", "name"),
    "silica_backlinks": ("backlinks", "name"),
    "silica_orphans": ("orphans", None),
    "silica_unresolved": ("unresolved links", None),
    "silica_files": ("list files", "folder"),
    "silica_exists": ("exists", "path"),
    "silica_deferred_list": ("deferred list", None),
    "silica_deferred_flush": ("deferred flush", None),
    "silica_deferred_retry": ("deferred retry", None),
    "silica_anneal": ("anneal", None),
    "silica_inbox_ls": ("inbox", None),
    "silica_recon": ("recon", "inbox_file"),
    "silica_payload": ("payload", "recon_report_path"),
    "silica_sanitize": ("sanitize", "distiller_output_path"),
    "silica_validate_ops": ("validate", "ops_json_path"),
    "silica_bulk_write": ("bulk write", "ops_json_path"),
    "silica_lint": ("lint", "note_name"),
    "silica_run_injector": ("injector", "inbox_file"),
    "silica_delete": ("delete", "ref"),
    "silica_snapshot": ("snapshot", "ops_json_path"),
    "silica_restore": ("restore", "txn_id"),
    "silica_cleanup": ("cleanup", "inbox_file"),
    "web_search": ("web search", "query"),
    "web_fetch": ("web fetch", "url"),
    # The question the chat model handed to the web lane — it rewords the user's
    # own, and this row is the only place that rewording is ever visible. Without
    # it a wrong answer reads as "web search is bad" rather than "it asked
    # something else".
    "silica_web_answer": ("web answer", "question"),
}

# Value transforms for targets that are unreadable raw. Beside _TOOL_DESC and
# applied by both readers below, so the next one lands in one place instead of
# growing a third copied branch in each of two near-duplicate functions.
_TARGET_FMT: dict[str, Callable[[str], str]] = {
    # The host, not the URL. Live, "which sources" is a site you judge at a
    # glance; the full URL with its title arrives at the end of the turn anyway,
    # in the answer's own `## Sources (web)` block.
    "web_fetch": lambda url: urlsplit(url).netloc or url,
}


def _tool_verb(name: str) -> str:
    verb, _ = _TOOL_DESC.get(name, (name.removeprefix("silica_").replace("_", " "), None))
    return verb


def _tool_target(name: str, args: dict) -> str:
    """The one argument that names what the tool acted on, as plain text.

    Same table as `_synthetic_tool_desc`, without the rich markup, so the web
    transcript can name the note a write touched instead of showing a bare verb.
    """
    if name == "silica_move":
        return f'{args.get("ref", "")} {GLYPHS["arrow"]} {args.get("to", "")}'.strip()
    if name == "silica_run_injector":
        return _injector_label(args)
    _, key = _TOOL_DESC.get(name, (None, None))
    val = str(args.get(key) or "").strip() if key else ""
    return _TARGET_FMT.get(name, str)(val) if val else ""


def _injector_files(args: dict) -> list[str]:
    """The inbox files of a run_injector call, plural key first.

    `inbox_file` is the legacy singular; every real caller passes `inbox_files`
    (the CLI tells the agent to, silica/cli.py). Reading only the singular is why
    the web transcript showed a bare "injector" with no filename at all.
    """
    files = args.get("inbox_files")
    files = [f for f in files if isinstance(f, str) and f.strip()] if isinstance(files, list) else []
    single = args.get("inbox_file")
    if isinstance(single, str) and single.strip() and single not in files:
        files.insert(0, single)
    return files


def _injector_label(args: dict) -> str:
    files = _injector_files(args)
    if not files:
        return ""
    extra = f" +{len(files) - 1}" if len(files) > 1 else ""
    return f"{files[0]}{extra}"


def _synthetic_tool_desc(name: str, args: dict) -> str:
    """Compact command-form description; the target argument is the only bold element."""
    if name == "silica_move":
        ref, to = args.get("ref", ""), args.get("to", "")
        return f'move [bold]"{escape(str(ref))}"[/bold] {GLYPHS["arrow"]} [bold]"{escape(str(to))}"[/bold]'
    if name == "silica_run_injector":
        label = _injector_label(args)
        return f'injector [bold]"{escape(label)}"[/bold]' if label else "injector"
    verb, key = _TOOL_DESC.get(name, (name.removeprefix("silica_").replace("_", " "), None))
    val = args.get(key, "") if key else ""
    if val:
        val = _TARGET_FMT.get(name, str)(str(val))
        return f'{verb} [bold]"{escape(str(val))}"[/bold]'
    return verb


# The injector recipe's two nested scopes, as the FSM actually runs them.
# File-scope phases run once per inbox file; the chunk cycle re-enters at
# COLLISION/DELEGATE for every chunk of that file (orchestrator.py) and never at
# RECON. Splitting them is what lets a frontend reset the chunk track per chunk
# without wrongly clearing the file track.
#
# Hardcoded on purpose: this can only drift from an edit to
# recipes/injector.yaml — which tests/test_phase_track.py pins.
_FILE_PHASES: dict[str, str] = {
    "recon":      "recon",
    "payload":    "payload",
    "salience":   "salience",
}

_CHUNK_PHASES: dict[str, str] = {
    "collision":  "collision",
    "distill":    "distill",
    "sanitize":   "sanitize",
    "validate":   "validate",
    "snapshot":   "snapshot",
    "write":      "write",
    "hub_update": "hub-update",
    "autolink":   "autolink",
    "backlink":   "backlink",
    "lint":       "lint",
    "cleanup":    "cleanup",
}

# rollback carries on_gate_fail in the recipe: an exception branch, not a step.
# It is deliberately absent from _PHASE_ORDER — as the 16th entry it rendered as
# "· rollback" pending on every healthy run, showing "everything went wrong" as
# something about to happen. It is drawn only once it actually fires.
_PHASE_LABELS: dict[str, str] = {**_FILE_PHASES, **_CHUNK_PHASES, "rollback": "rollback"}

# Tools whose ✓ lines never collapse into `✓ verb ×N`. For these the target IS
# the payload — `✓ web fetch ×5` throws away the five sites the answer was read
# from, which is the whole reason the line is printed.
_NO_AGGREGATE = frozenset({"web_search", "web_fetch"})

_PHASE_ORDER: list[str] = [*_FILE_PHASES.values(), *_CHUNK_PHASES.values()]
_MICRO_PHASE_ORDER: tuple[str, ...] = ("reading", "calling_llm", "committing")


def _stage_track(pipeline_phases: list[dict], console_width: int) -> "Text":
    """Compact horizontal stage track for the injector panel.

    Shows a ±3 window around the running phase. Each phase is rendered as:
      done    → dim  "✓ label"
      running → bold #22d3ee "◉ label"
      failed  → bold red "✗ label"
      pending → dim  "· label"
    Leading/trailing "…" indicate clipped phases.
    """
    status_map: dict[str, str] = {e["phase"]: e["status"] for e in pipeline_phases}
    start_map: dict[str, float] = {e["phase"]: e["start"] for e in pipeline_phases if e.get("start")}

    running_idx = next(
        (i for i, p in enumerate(_PHASE_ORDER) if status_map.get(p) == "running"),
        None,
    )
    if running_idx is None:
        done_indices = [
            i for i, p in enumerate(_PHASE_ORDER)
            if status_map.get(p) in ("done", "failed")
        ]
        center = done_indices[-1] if done_indices else 0
    else:
        center = running_idx

    start = max(0, center - 3)
    end = min(len(_PHASE_ORDER), center + 4)

    t = Text()
    if start > 0:
        t.append("… ", style="dim")
    for i in range(start, end):
        phase = _PHASE_ORDER[i]
        st = status_map.get(phase, "pending")
        if st == "done":
            t.append(f"✓ {phase}", style="dim")
        elif st == "running":
            # Not `start`: that name is the window's lower bound, and rebinding
            # it mid-loop is only harmless because range() was already evaluated.
            phase_start = start_map.get(phase)
            suffix = f" {_fmt_dur(time.monotonic() - phase_start)}" if phase_start else ""
            t.append(f"◉ {phase}{suffix}", style="bold #22d3ee")
        elif st == "failed":
            t.append(f"✗ {phase}", style="bold red")
        else:
            t.append(f"· {phase}", style="dim")
        if i < end - 1:
            t.append("  ")
    if end < len(_PHASE_ORDER):
        t.append("  …", style="dim")
    # Keep the track on a single line: on a narrow console the ±3 window would wrap,
    # changing the panel height between frames and tearing the Live region. Crop with
    # an ellipsis instead of wrapping so the panel height stays constant at any width.
    t.no_wrap = True
    t.overflow = "ellipsis"
    return t


class _StageTrack:
    """Re-renders ``_stage_track`` on every Live auto-refresh tick.

    Same trick ``rich.spinner.Spinner`` uses to animate without explicit
    ``.update()`` calls: Live's own refresh thread re-renders whatever is
    currently displayed, so a running phase's elapsed time ticks live instead
    of freezing at the value captured when the phase started.
    """

    def __init__(self, phases: list[dict], width: int) -> None:
        self._phases = phases
        self._width = width

    def __rich_console__(self, console, options):
        yield _stage_track(self._phases, self._width)


class _ProgressRenderer:
    def __init__(self) -> None:
        self._live: Live | None = None
        self._last_tool_name: str = ""
        self._active_tools: dict[str, dict] = {}
        # Pipeline phase tracking (populated when silica_run_injector is active)
        self._injector_call_id: str | None = None
        self._pipeline_phases: list[dict] = []   # ordered: {phase, status, elapsed}
        self._phase_start_times: dict[str, float] = {}
        # Run position, restated by every PhaseEvent (file/chunk counters + label).
        self._inject_pos: dict = {}
        # Inject progress bar (tracks files processed / total)
        self._inject_inbox_label: str = ""
        self._inject_file_count: int = 0
        self._inject_progress: Progress | None = None
        self._inject_task_id: TaskID | None = None
        # Batch run progress (refine/enrich)
        self._batch: dict | None = None
        # Pending ✓ line — consecutive completions of the same tool collapse into one
        # aggregated line (`✓ read ×4 · 1.2s`), flushed before any other output.
        self._ok_run: dict | None = None
        # Streamed answer preview shown in the transient live region while the
        # final LLM response is being generated.
        self._stream_buf: str = ""
        # Register as batch hook so emit_batch_event reaches this renderer
        global _batch_run_hook
        _batch_run_hook = self.__call__
        from silica.agent.bus import BUS
        BUS.subscribe("work/feedback", self._on_work_feedback)
        BUS.subscribe("work/phase", self._on_pipeline_phase)

    def _flush_ok_run(self) -> None:
        """Print (and clear) the buffered ✓ line, aggregated when count > 1."""
        run, self._ok_run = self._ok_run, None
        if run is None:
            return
        dur = _fmt_dur(run["dur"])
        dur_str = f" [dim]{dur}[/]" if dur else ""
        if run["count"] > 1:
            CONSOLE.print(
                f"  [tool.ok]{GLYPHS['ok']}[/] {_tool_verb(run['name'])}"
                f" [dim]×{run['count']}[/]{dur_str}"
            )
        else:
            CONSOLE.print(f"  [tool.ok]{GLYPHS['ok']}[/] {run['desc']}{dur_str}")

    def _start_spinner(self) -> None:
        if self._live is not None:
            return
        if not CONSOLE.is_terminal:
            return
        spinner = Spinner("dots", text="  thinking…", style="dim cyan")
        self._live = Live(
            spinner,
            console=CONSOLE,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()

    _STREAM_TAIL_LINES = 6

    def _update_stream_live(self) -> None:
        """Render the tail of the streaming answer in the transient live region."""
        if self._batch is not None or not self._stream_buf:
            return
        tail = self._stream_buf.splitlines()[-self._STREAM_TAIL_LINES:]
        # Each line clipped, not wrapped: the region height must stay bounded so
        # the Live never tears (same invariant as the stage track).
        body = Text("\n".join(tail), style="dim", no_wrap=True, overflow="ellipsis")
        header = Spinner("dots", text=" [role.assistant]silica[/] [dim]·[/]", style="brand.cyan")
        group = Group(header, Padding(body, (0, 0, 0, 2)))
        if self._live is not None:
            self._live.update(group)
        else:
            self._live = Live(group, console=CONSOLE, refresh_per_second=12, transient=True)
            self._live.start()

    def _stop_spinner(self) -> None:
        if self._batch is not None:
            return  # Batch Live is managed by _update_batch_live / _finalize_batch
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _on_pipeline_phase(self, event) -> None:
        """Called from BUS when the InjectorFSM publishes a PhaseEvent.

        Subscribed for the renderer's whole life, so it guards on the injector
        being the live tool exactly as _on_work_feedback guards on _batch.
        """
        if self._injector_call_id is None:
            return
        phase, status, elapsed = event.phase, event.status, event.elapsed
        # Position rides on every event, so a missed one can't leave the header
        # naming the wrong file or chunk.
        self._inject_pos = {
            "file_idx": event.file_idx, "file_total": event.file_total,
            "chunk_idx": event.chunk_idx, "chunk_total": event.chunk_total,
        }
        if event.source_file:
            self._inject_inbox_label = event.source_file
        if self._inject_progress is not None and self._inject_task_id is not None:
            self._inject_progress.update(
                self._inject_task_id,
                completed=min(event.file_idx, event.file_total),
                total=event.file_total,
            )
        label = _PHASE_LABELS.get(phase, phase)
        if status == "running":
            now = time.monotonic()
            self._phase_start_times[phase] = now
            for entry in self._pipeline_phases:
                if entry["phase"] == label:
                    entry["status"] = "running"
                    entry["elapsed"] = None
                    entry["start"] = now
                    break
            else:
                self._pipeline_phases.append({"phase": label, "status": "running", "elapsed": None, "start": now})
        elif status in ("done", "failed"):
            start = self._phase_start_times.pop(phase, time.monotonic())
            dur = elapsed if elapsed is not None else (time.monotonic() - start)
            for entry in self._pipeline_phases:
                if entry["phase"] == label:
                    entry["status"] = status
                    entry["elapsed"] = dur
                    break
            else:
                self._pipeline_phases.append({"phase": label, "status": status, "elapsed": dur})
        self._update_live()

    def _position_suffix(self) -> str:
        """` · file 1/2 · chunk 3/5` for the injector header, empty before the
        first PhaseEvent. Both denominators are per-file and known when shown:
        the run-wide chunk total does not exist until the last file is
        partitioned (see PhaseEvent)."""
        pos = self._inject_pos
        if not pos:
            return ""
        bits = []
        if pos.get("file_total", 0) > 1:
            bits.append(f"file {pos['file_idx'] + 1}/{pos['file_total']}")
        if pos.get("chunk_total", 0) > 0:
            bits.append(f"chunk {pos['chunk_idx'] + 1}/{pos['chunk_total']}")
        return f" [dim]·[/] [dim]{' · '.join(bits)}[/]" if bits else ""

    def _on_work_feedback(self, event) -> None:
        """Called from BUS when a sub-agent publishes a WorkFeedbackEvent."""
        if self._batch is None:
            return
        if event.kind != self._batch["kind"]:
            return
        self._batch["micro_phase"] = event.phase
        self._update_batch_live()

    def _update_live(self) -> None:
        if self._batch is not None:
            return  # Batch block managed by _update_batch_live
        mode = CONFIG.tool_progress
        if mode == "off" or not CONSOLE.is_terminal:
            return

        if not self._active_tools:
            if self._live:
                self._live.update(Spinner("dots", text="  thinking…", style="dim cyan"))
            return

        renderables: list[RenderableType] = []
        for cid, tinfo in self._active_tools.items():
            name = tinfo["name"]
            args = tinfo["args"]
            desc = _synthetic_tool_desc(name, args)

            if name == "silica_run_injector" and (self._pipeline_phases or self._inject_progress is not None):
                # Flat block, no borders: spinner header + indented stage track (+ file bar).
                # The running phase is already marked ◉ in the track — no separate phase line.
                header = Spinner(
                    "dots",
                    text=f" [bold]injector[/] [dim]·[/] {escape(self._inject_inbox_label)}"
                         f"{self._position_suffix()}",
                    style="brand.cyan",
                )
                parts: list = [header, Padding(_StageTrack(self._pipeline_phases, CONSOLE.width), (0, 0, 0, 2))]
                if self._inject_progress is not None:
                    parts.append(Padding(self._inject_progress, (0, 0, 0, 2)))
                renderables.append(Group(*parts))
            else:
                renderables.append(Spinner("dots", text=f"  {desc}…", style="cyan"))

        if self._live:
            self._live.update(Group(*renderables))
        else:
            self._live = Live(
                Group(*renderables),
                console=CONSOLE,
                refresh_per_second=12,
                transient=True,
            )
            self._live.start()

    def _update_batch_live(self) -> None:
        if self._batch is None or not CONSOLE.is_terminal:
            return
        batch = self._batch
        batch["progress_obj"].update(batch["task_id"], completed=batch["done"])
        cur = batch["current_label"]
        cur_str = f" [dim]· {escape(cur)}[/]" if cur else ""
        # Flat block, no borders: spinner header + indented bar (+ micro-phase track).
        # done/total lives in the bar's MofN column — not repeated in the header.
        header = Spinner(
            "dots",
            text=f" [bold]{escape(batch['kind'])}[/] [dim]·[/] {escape(batch['label'])}{cur_str}",
            style="brand.cyan",
        )
        parts: list = [header, Padding(batch["progress_obj"], (0, 0, 0, 2))]
        micro = batch.get("micro_phase", "")
        if micro:
            micro_parts = []
            for mp in _MICRO_PHASE_ORDER:
                display = mp.replace("_", " ")
                if mp == micro:
                    micro_parts.append(f"[bold brand.cyan]{GLYPHS['active']} {display}[/]")
                else:
                    micro_parts.append(f"[dim]{GLYPHS['pending']} {display}[/]")
            parts.append(Padding(Text.from_markup("   ".join(micro_parts)), (0, 0, 0, 2)))
        group = Group(*parts)
        if self._live is None:
            self._live = Live(group, console=CONSOLE, refresh_per_second=12, transient=True)
            self._live.start()
        else:
            self._live.update(group)

    def _finalize_batch(self) -> None:
        batch = self._batch
        if batch is None:
            return
        self._batch = None
        batch["progress_obj"].stop()
        if self._live is not None:
            self._live.stop()
            self._live = None
        kind = escape(batch["kind"])
        label = escape(batch["label"])
        done = batch["done"]
        total = batch["total"]
        failed = batch["failed"]
        elapsed = f"{time.monotonic() - batch['start_time']:.1f}s"
        if failed > 0:
            CONSOLE.print(
                f"  [tool.err]✗[/] [bold]{kind}[/] [dim]·[/] {label}"
                f"   {done}/{total} batches [dim]·[/] {failed} failed [dim]·[/] {elapsed}"
            )
        else:
            CONSOLE.print(
                f"  [tool.ok]✓[/] [bold]{kind}[/] [dim]·[/] {label}"
                f"   {done}/{total} batches [dim]·[/] {elapsed}"
            )

    def __call__(self, event: RenderEvent) -> None:
        try:
            self._dispatch(event)
        except Exception as exc:
            logger.debug("tool_progress_callback error (swallowed): %s", exc)

    def _dispatch(self, event: RenderEvent) -> None:
        mode = CONFIG.tool_progress

        # Any output other than a further completion breaks a ✓-aggregation run.
        # ToolStartEvent is exempt: interactively it only redraws the live region.
        if not isinstance(event, (ToolStartEvent, ToolCompleteEvent)):
            self._flush_ok_run()

        if isinstance(event, ThinkingStartEvent):
            self._stream_buf = ""
            if mode != "off":
                self._start_spinner()
            return

        if isinstance(event, ThinkingEndEvent):
            # The transient preview vanishes here; cli.py prints the final
            # formatted answer once — streaming never double-renders.
            self._stream_buf = ""
            self._stop_spinner()
            return

        if isinstance(event, LLMStreamEvent):
            if not CONSOLE.is_terminal:
                return
            if event.chunk_type == "reset":
                self._stream_buf = ""
            elif event.chunk_type == "reasoning":
                if CONFIG.show_thinking or mode == "verbose" or CONFIG.verbose:
                    self._stream_buf += event.content
            elif event.chunk_type == "text":
                self._stream_buf += event.content
            self._update_stream_live()
            return

        if isinstance(event, ReasoningEvent):
            self._stop_spinner()
            if CONFIG.show_thinking or mode == "verbose" or CONFIG.verbose:
                body = _head_cap(event.text).strip()
                indented = "\n".join(f"  [reasoning.gutter]│[/] [dim]{line}[/]" for line in body.splitlines())
                CONSOLE.print(f"  [reasoning]{GLYPHS['think']} thinking[/]\n{indented}\n")
            return

        if isinstance(event, ToolErrorEvent):
            if self._batch is not None:
                if event.name in ("silica_refine_batch", "silica_enrich_batch"):
                    self._batch["failed"] += 1
                CONSOLE.print(f"  [tool.err]✗[/] {escape(event.name)}  {escape(event.error[:80])}")
                return
            self._stop_spinner()
            CONSOLE.print(f"  [tool.err]✗[/] [bold]{escape(event.name)}[/]: [tool.err]{escape(event.error)}[/]")
            self._active_tools.pop(event.call_id, None)
            self._update_live()
            return

        if mode == "off":
            return

        if isinstance(event, BatchRunStartEvent):
            if not CONSOLE.is_terminal:
                return
            self._stop_spinner()
            progress_obj = Progress(
                BarColumn(bar_width=24, style="dim", complete_style="brand.cyan", finished_style="brand.cyan"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                auto_refresh=False,
            )
            task_id = progress_obj.add_task("", total=event.total)
            # Not started on purpose — embedded in self._live (see injector note); a
            # Progress.start() here would open a second Live and orphan the bar.
            self._batch = {
                "run_id": event.run_id,
                "kind": event.kind,
                "label": event.label,
                "total": event.total,
                "done": 0,
                "failed": 0,
                "start_time": time.monotonic(),
                "progress_obj": progress_obj,
                "task_id": task_id,
                "current_label": "",
                "micro_phase": "",
            }
            self._update_batch_live()
            return

        if isinstance(event, ToolStartEvent):
            if self._batch is not None:
                return  # All tool spinners suppressed during batch run
            if CONSOLE.is_terminal:
                self._active_tools[event.call_id] = {"name": event.name, "args": event.args}
                if event.name == "silica_run_injector":
                    self._injector_call_id = event.call_id
                    files = _injector_files(event.args)
                    self._inject_file_count = max(1, len(files))
                    self._inject_inbox_label = str(files[0] if files else "?")
                    self._pipeline_phases = []
                    self._phase_start_times = {}
                    self._inject_pos = {}
                    # File bar only when there is more than one file — a 0/1→1/1 bar is noise.
                    if self._inject_file_count > 1:
                        self._inject_progress = Progress(
                            BarColumn(bar_width=24, style="dim", complete_style="brand.cyan", finished_style="brand.cyan"),
                            MofNCompleteColumn(),
                            TimeElapsedColumn(),
                            auto_refresh=False,
                        )
                        self._inject_task_id = self._inject_progress.add_task("", total=self._inject_file_count)
                        # Do NOT start() it: Progress.start() spins up its own Live on the
                        # global console, which double-renders the bar (an orphan above the
                        # live block on small consoles). We embed it as a renderable in
                        # self._live, which drives the rendering.
                self._update_live()
            else:
                # Non-interactive fallback: print immediately
                desc = _synthetic_tool_desc(event.name, event.args)
                if mode == "new":
                    if event.name == self._last_tool_name:
                        return
                    self._last_tool_name = event.name
                    CONSOLE.print(f"  [dim]{GLYPHS['gear']}[/] {desc}")
                elif mode == "all":
                    CONSOLE.print(f"  [cyan]→[/] {desc}")
                elif mode == "verbose":
                    CONSOLE.print(f"  [cyan]→[/] {desc}")

        elif isinstance(event, ToolCompleteEvent):
            # Batch: ledger_next sets the label, ledger_update advances the bar.
            # One next now hands out a whole frontier of tasks, so counting its
            # completions would undercount the run; one update lands per task.
            if self._batch is not None and event.name in (
                "silica_ledger_next", "silica_ledger_update"
            ):
                try:
                    result_data = json.loads(event.result)
                except Exception:
                    result_data = {}
                if event.name == "silica_ledger_update":
                    self._batch["done"] = min(self._batch["done"] + 1, self._batch["total"])
                    self._batch["micro_phase"] = ""
                    self._update_batch_live()
                    return
                if result_data.get("done"):
                    self._finalize_batch()
                else:
                    tasks = result_data.get("tasks") or []
                    payload = (tasks[0].get("payload") if tasks else None) or {}
                    note_paths = payload.get("note_paths", [])
                    if isinstance(note_paths, list) and note_paths:
                        name0 = note_paths[0].rsplit("/", 1)[-1]
                        extra = f" +{len(note_paths) - 1}" if len(note_paths) > 1 else ""
                        self._batch["current_label"] = f"{name0}{extra}"
                    self._batch["micro_phase"] = ""
                    self._update_batch_live()
                return
            # Suppress all other tool completions during batch
            if self._batch is not None:
                return

            if CONSOLE.is_terminal:
                self._stop_spinner()
                desc = _synthetic_tool_desc(event.name, event.args)

                if event.name == "silica_run_injector" and self._injector_call_id == event.call_id:
                    # Close the live block; print compact single-line summary. The
                    # BUS subscription stays for the renderer's life — clearing
                    # _injector_call_id is what silences it (see _on_pipeline_phase).
                    self._flush_ok_run()
                    self._injector_call_id = None
                    self._inject_pos = {}
                    if self._inject_progress is not None:
                        self._inject_progress.stop()
                        self._inject_progress = None
                        self._inject_task_id = None
                    failed_phases = [e for e in self._pipeline_phases if e["status"] == "failed"]
                    lbl = escape(self._inject_inbox_label)
                    inject_dur = _fmt_dur(event.duration_s) or "0s"
                    # Brief numeric yield from the run result (files always; notes/links if any).
                    try:
                        _data = json.loads(event.result) if isinstance(event.result, str) else {}
                    except Exception:
                        _data = {}
                    n_files = self._inject_file_count or 1
                    bits = [f"{n_files} file" if n_files == 1 else f"{n_files} files"]
                    if _data.get("yield_notes"):
                        bits.append(f"{_data['yield_notes']} notes")
                    if _data.get("yield_links"):
                        bits.append(f"{_data['yield_links']} links")
                    yield_str = " [dim]·[/] ".join(bits)
                    if failed_phases:
                        # The phase that FAILED, not the last one seen:
                        # _pipeline_phases is ordered by first appearance and
                        # its tail is whatever ran afterwards (cleanup, or
                        # rollback), which sends the user to a phase that
                        # succeeded.
                        last_phase = failed_phases[-1].get("phase") or "?"
                        CONSOLE.print(
                            f"  [tool.err]✗[/] [bold]injector[/] [dim]·[/] {lbl}"
                            f"   {yield_str} [dim]·[/] failed at {last_phase} [dim]·[/] {inject_dur}"
                        )
                    else:
                        CONSOLE.print(
                            f"  [tool.ok]✓[/] [bold]injector[/] [dim]·[/] {lbl}"
                            f"   {yield_str} [dim]·[/] {inject_dur}"
                        )
                    self._pipeline_phases = []
                    self._inject_inbox_label = ""
                    self._inject_file_count = 0
                elif mode == "verbose":
                    # Verbose prints per-call result heads — aggregation would hide them.
                    self._flush_ok_run()
                    dur = _fmt_dur(event.duration_s)
                    dur_str = f" [dim]{dur}[/]" if dur else ""
                    CONSOLE.print(f"  [tool.ok]{GLYPHS['ok']}[/] {desc}{dur_str}")
                    redacted = _redact(event.result)
                    if redacted is not None:
                        head = _head_result(redacted.strip())
                        if head:
                            CONSOLE.print(f"    [dim]{escape(head)}[/]")
                    else:
                        CONSOLE.print("    [dim][result redacted][/]")
                else:
                    # Buffer the ✓ line: consecutive completions of the same tool
                    # collapse into one aggregated line at flush time.
                    run = self._ok_run
                    if (
                        run is not None
                        and run["name"] == event.name
                        and event.name not in _NO_AGGREGATE
                    ):
                        run["count"] += 1
                        run["dur"] += event.duration_s
                    else:
                        self._flush_ok_run()
                        self._ok_run = {
                            "name": event.name,
                            "desc": desc,
                            "count": 1,
                            "dur": event.duration_s,
                        }

                self._active_tools.pop(event.call_id, None)
                self._update_live()
            else:
                # Non-interactive fallback
                desc = _synthetic_tool_desc(event.name, event.args)
                dur = _fmt_dur(event.duration_s)
                dur_str = f" [dim]{dur}[/]" if dur else ""
                if mode in ("new", "all"):
                    CONSOLE.print(f"  [tool.ok]{GLYPHS['ok']}[/] {desc}{dur_str}")
                elif mode == "verbose":
                    redacted = _redact(event.result)
                    if redacted is not None:
                        head = _head_result(redacted.strip())
                        CONSOLE.print(f"  [tool.ok]{GLYPHS['ok']}[/] {desc}{dur_str}")
                        if head:
                            CONSOLE.print(f"  [dim]{escape(head)}[/]")
                    else:
                        CONSOLE.print(f"  [tool.ok]{GLYPHS['ok']}[/] {desc}{dur_str} [dim][result redacted][/]")


    def close(self) -> None:
        """Unconditionally stop the live display and deregister all hooks.

        Called on KeyboardInterrupt / uncaught exceptions so the terminal is
        always restored before the next prompt is printed.
        """
        global _batch_run_hook
        _batch_run_hook = None
        self._flush_ok_run()
        self._stream_buf = ""
        self._injector_call_id = None
        self._inject_pos = {}
        from silica.agent.bus import BUS
        BUS.unsubscribe("work/feedback", self._on_work_feedback)
        BUS.unsubscribe("work/phase", self._on_pipeline_phase)
        if self._batch is not None:
            batch = self._batch
            self._batch = None
            batch["progress_obj"].stop()
            if self._live is not None:
                self._live.stop()
                self._live = None
            CONSOLE.print(
                f"  [bold yellow]⚠[/]  {escape(batch['kind'])} [dim]·[/] {escape(batch['label'])}"
                f"   interrupted at {batch['done']}/{batch['total']}"
            )
        if self._inject_progress is not None:
            self._inject_progress.stop()
            self._inject_progress = None
            self._inject_task_id = None
        self._stop_spinner()


def make_progress_callback() -> _ProgressRenderer:
    return _ProgressRenderer()
