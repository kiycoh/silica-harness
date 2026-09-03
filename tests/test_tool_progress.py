from unittest.mock import patch, MagicMock, PropertyMock
from silica.config import CONFIG
from silica.cli import _handle_slash_command
from silica.ui.renderer import make_progress_callback, _redact
from silica.ui.console import CONSOLE
from silica.agent.events import ToolStartEvent, ToolCompleteEvent, ToolErrorEvent
from silica.agent.loop import run_agent
from silica.agent.llm import LLMResponse, ToolCall
from silica.tools import TOOLS

def test_verbose_slash_command_cycle():
    orig_mode = CONFIG.tool_progress
    try:
        CONFIG.tool_progress = "off"
        messages = []
        
        # off -> new
        _handle_slash_command("/verbose", messages)
        assert CONFIG.tool_progress == "new"
        
        # new -> all
        _handle_slash_command("/verbose", messages)
        assert CONFIG.tool_progress == "all"
        
        # all -> verbose
        _handle_slash_command("/verbose", messages)
        assert CONFIG.tool_progress == "verbose"
        
        # verbose -> off
        _handle_slash_command("/verbose", messages)
        assert CONFIG.tool_progress == "off"
    finally:
        CONFIG.tool_progress = orig_mode


def test_callback_noop_when_off(capsys):
    orig_mode = CONFIG.tool_progress
    CONFIG.tool_progress = "off"
    
    try:
        cb = make_progress_callback()
        event = ToolStartEvent(name="test_tool", args={}, call_id="1", iteration=1)
        cb(event)
        
        captured = capsys.readouterr()
        assert captured.out == ""
    finally:
        CONFIG.tool_progress = orig_mode


def test_redact_patterns():
    # Test credentials redaction
    assert "api_key=[REDACTED]" in _redact('api_key = "abc-123"')
    assert "token=[REDACTED]" in _redact('token:123')
    assert "secret=[REDACTED]" in _redact('"secret" : "password"')
    
    # Fail-closed test
    assert _redact(None) is None


@patch("silica.agent.loop.call_llm")
def test_agent_loop_swallows_callback_exceptions(mock_call_llm):
    response1 = LLMResponse(
        text=None,
        tool_calls=[ToolCall(id="tc1", name="silica_read_note", args={"name": "test_note"})],
        assistant_message={"role": "assistant", "tool_calls": []},
        usage={}
    )
    response2 = LLMResponse(
        text="Final answer",
        tool_calls=[],
        assistant_message={"role": "assistant", "content": "Final answer"},
        usage={}
    )
    mock_call_llm.side_effect = [response1, response2]
    
    with patch.dict(TOOLS, {"silica_read_note": MagicMock()}):
        TOOLS["silica_read_note"].run.return_value = "note content"
        
        def bad_callback(event):
            raise ValueError("bad callback")
            
        messages = [{"role": "user", "content": "hello"}]
        
        ans = run_agent(messages, model="test_model", tool_progress_callback=bad_callback)
        assert ans == "Final answer"


def test_tool_error_event_always_emitted(capsys):
    orig_mode = CONFIG.tool_progress
    CONFIG.tool_progress = "off"
    
    try:
        cb = make_progress_callback()
        event = ToolErrorEvent(name="error_tool", call_id="1", error="Some error", iteration=1)
        cb(event)
        
        captured = capsys.readouterr()
        assert "error_tool" in captured.out
        assert "Some error" in captured.out
    finally:
        CONFIG.tool_progress = orig_mode


def test_callback_modes_output(capsys, monkeypatch):
    orig_mode = CONFIG.tool_progress
    # This test exercises the non-interactive (plain print) branch; pin the console
    # off-terminal so it doesn't take the Live branch when FORCE_COLOR/a TTY is present.
    monkeypatch.setattr(CONSOLE, "_force_terminal", False)
    cb = make_progress_callback()

    try:
        # Test "new" mode
        CONFIG.tool_progress = "new"
        cb(ToolStartEvent(name="silica_read_note", args={"name": "noteA"}, call_id="1", iteration=1))
        captured = capsys.readouterr()
        assert "read" in captured.out
        assert "noteA" in captured.out

        # Same tool consecutive call should be skipped in "new" mode
        cb(ToolStartEvent(name="silica_read_note", args={"name": "noteB"}, call_id="2", iteration=2))
        captured = capsys.readouterr()
        assert captured.out == ""

        # Different tool should print
        cb(ToolStartEvent(name="silica_search", args={"query": "searchQ"}, call_id="3", iteration=3))
        captured = capsys.readouterr()
        assert "search" in captured.out
        assert "searchQ" in captured.out

        # Test "all" mode
        CONFIG.tool_progress = "all"
        cb(ToolStartEvent(name="silica_read_note", args={"name": "noteA"}, call_id="4", iteration=4))
        captured = capsys.readouterr()
        assert "read" in captured.out
        assert "noteA" in captured.out

        # Test "verbose" mode
        CONFIG.tool_progress = "verbose"
        cb(ToolStartEvent(name="silica_read_note", args={"name": "noteA"}, call_id="5", iteration=5))
        captured = capsys.readouterr()
        assert "read" in captured.out
        assert "noteA" in captured.out

        cb(ToolCompleteEvent(name="silica_read_note", args={"name": "noteA"}, call_id="5", result="some result", duration_s=1.23, iteration=5))
        captured = capsys.readouterr()
        assert "read" in captured.out
        assert "some result" in captured.out
        
    finally:
        CONFIG.tool_progress = orig_mode


def test_web_fetches_are_never_collapsed_into_one_line(capsys, monkeypatch):
    """`✓ web fetch ×5` throws away the five sites the answer was read from,
    which is the only reason that line is printed at all.

    The aggregation is right for everything else and stays; the exemption is
    what a future tidy-up of `_flush_ok_run` would silently undo, taking the
    feature with it while the suite stayed green.
    """
    from silica.agent.events import ThinkingEndEvent

    monkeypatch.setattr(CONSOLE, "_force_terminal", True)
    monkeypatch.setattr(CONFIG, "tool_progress", "new")
    cb = make_progress_callback()

    for url in ("https://docs.kernel.org/scheduler/sched-domains.html", "https://lwn.net/Articles/1"):
        cb(ToolCompleteEvent(name="web_fetch", args={"url": url}, call_id=url,
                             result="", duration_s=0.0, iteration=1))
    cb(ThinkingEndEvent(iteration=1))  # anything but a completion flushes the pending run

    out = capsys.readouterr().out
    assert "docs.kernel.org" in out
    assert "lwn.net" in out
    assert "×" not in out           # not aggregated
    assert "/scheduler/" not in out  # the host, not the URL — the path is the Sources block's job


def test_reasoning_event_renders_when_enabled(capsys):
    from silica.agent.events import ReasoningEvent
    from silica.ui.renderer import make_progress_callback
    orig_thinking = CONFIG.show_thinking
    orig_tool_progress = CONFIG.tool_progress
    orig_verbose = CONFIG.verbose
    try:
        cb = make_progress_callback()
        event = ReasoningEvent(text="This is my deep reasoning process.", iteration=1)

        # Case 1: show_thinking=True, verbose=False, progress=all
        CONFIG.show_thinking = True
        CONFIG.verbose = False
        CONFIG.tool_progress = "all"
        cb(event)
        captured = capsys.readouterr()
        assert "thinking" in captured.out.lower()
        assert "reasoning" in captured.out.lower()

        # Case 2: show_thinking=False, verbose=False, progress=all
        CONFIG.show_thinking = False
        CONFIG.verbose = False
        CONFIG.tool_progress = "all"
        cb(event)
        captured = capsys.readouterr()
        assert captured.out == ""

        # Case 3: show_thinking=False, verbose=True, progress=all
        CONFIG.show_thinking = False
        CONFIG.verbose = True
        CONFIG.tool_progress = "all"
        cb(event)
        captured = capsys.readouterr()
        assert "thinking" in captured.out.lower()
        assert "reasoning" in captured.out.lower()

        # Case 4: show_thinking=False, verbose=False, progress=verbose
        CONFIG.show_thinking = False
        CONFIG.verbose = False
        CONFIG.tool_progress = "verbose"
        cb(event)
        captured = capsys.readouterr()
        assert "thinking" in captured.out.lower()
        assert "reasoning" in captured.out.lower()

    finally:
        CONFIG.show_thinking = orig_thinking
        CONFIG.tool_progress = orig_tool_progress
        CONFIG.verbose = orig_verbose


@patch("litellm.completion")
def test_llm_captures_reasoning(mock_completion):
    from silica.agent.llm import call_llm
    
    mock_message = MagicMock()
    mock_message.content = "My answer"
    mock_message.tool_calls = []
    mock_message.reasoning_content = "Thinking hard..."
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = {}
    
    mock_completion.return_value = mock_resp
    
    messages = [{"role": "user", "content": "hello"}]
    res = call_llm(model="test_model", messages=messages)
    
    assert res.reasoning == "Thinking hard..."
    assert res.text == "My answer"
    assert res.assistant_message["role"] == "assistant"
    assert res.assistant_message["content"] == "My answer"
    # Reasoning is surfaced on the response but kept OUT of the history: it would
    # be re-sent (and re-billed) on every later iteration of the tool loop.
    assert "reasoning_content" not in res.assistant_message

    mock_message2 = MagicMock()
    mock_message2.content = "Answer with blocks"
    mock_message2.tool_calls = []
    mock_message2.reasoning_content = None
    mock_message2.thinking_blocks = [{"thinking": "Block reasoning"}]
    
    mock_choice2 = MagicMock()
    mock_choice2.message = mock_message2
    
    mock_resp2 = MagicMock()
    mock_resp2.choices = [mock_choice2]
    mock_resp2.usage = {}
    
    mock_completion.return_value = mock_resp2
    
    res2 = call_llm(model="test_model", messages=messages)
    assert res2.reasoning == "Block reasoning"
    assert res2.assistant_message["role"] == "assistant"
    assert res2.assistant_message["content"] == "Answer with blocks"
    assert res2.assistant_message["thinking_blocks"] == [{"thinking": "Block reasoning"}]


@patch("litellm.completion")
def test_llm_openrouter_include_reasoning(mock_completion, monkeypatch):
    from silica.agent.llm import call_llm

    # Pin the output budget: the expected max_tokens must not track the
    # developer's .env MAX_TOKENS.
    monkeypatch.setenv("MAX_TOKENS", "256000")
    
    mock_message = MagicMock()
    mock_message.content = "My answer"
    mock_message.tool_calls = []
    mock_message.reasoning_content = "Thinking hard..."
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = {}
    mock_completion.return_value = mock_resp
    
    messages = [{"role": "user", "content": "hello"}]
    
    orig_thinking = CONFIG.show_thinking
    orig_verbose = CONFIG.verbose
    try:
        # Test openrouter model with show_thinking=True
        CONFIG.show_thinking = True
        CONFIG.verbose = False
        call_llm(model="openrouter/some-model", messages=messages)
        # Named kwargs, not assert_called_with: every openrouter call also
        # carries the attribution headers and the session_id, and this test is
        # about include_reasoning and the pinned budget.
        sent = mock_completion.call_args.kwargs
        assert sent["model"] == "openrouter/some-model"
        assert sent["messages"] == messages
        assert sent["max_tokens"] == 256000
        assert sent["include_reasoning"] is True
        assert sent["timeout"] == 120.0

        # Test openrouter model with show_thinking=False and verbose=True
        CONFIG.show_thinking = False
        CONFIG.verbose = True
        call_llm(model="openrouter/some-model", messages=messages)
        # Named kwargs, not assert_called_with: every openrouter call also
        # carries the attribution headers and the session_id, and this test is
        # about include_reasoning and the pinned budget.
        sent = mock_completion.call_args.kwargs
        assert sent["model"] == "openrouter/some-model"
        assert sent["messages"] == messages
        assert sent["max_tokens"] == 256000
        assert sent["include_reasoning"] is True
        assert sent["timeout"] == 120.0
        
        # Test non-openrouter model
        call_llm(model="openai/gpt-4o", messages=messages)
        args, kwargs = mock_completion.call_args
        assert "include_reasoning" not in kwargs
        assert kwargs.get("timeout") == 120.0
        
    finally:
        CONFIG.show_thinking = orig_thinking
        CONFIG.verbose = orig_verbose


def test_thinking_slash_toggle():
    orig_thinking = CONFIG.show_thinking
    try:
        messages = []
        CONFIG.show_thinking = True
        _handle_slash_command("/thinking", messages)
        assert CONFIG.show_thinking is False
        
        _handle_slash_command("/thinking", messages)
        assert CONFIG.show_thinking is True
    finally:
        CONFIG.show_thinking = orig_thinking


def test_stage_track_centers_on_running_phase():
    from silica.ui.renderer import _stage_track
    phases = [
        {"phase": "recon",     "status": "done",    "elapsed": 1.0},
        {"phase": "payload",   "status": "done",    "elapsed": 0.8},
        {"phase": "salience",  "status": "done",    "elapsed": 0.3},
        {"phase": "collision", "status": "done",    "elapsed": 0.4},
        {"phase": "distill",   "status": "running", "elapsed": None},
    ]
    track = _stage_track(phases, console_width=120)
    plain = track.plain
    # Running phase is visible
    assert "◉ distill" in plain
    # Phases in window are visible (distill is index 4; window is indices 1-7)
    assert "✓ payload" in plain
    assert "✓ salience" in plain
    # Phases outside window are NOT visible (recon is index 0, outside window start=1)
    assert "✓ recon" not in plain
    # Leading ellipsis present because window doesn't start at 0
    assert plain.startswith("…")


def test_stage_track_empty_shows_pending_from_start():
    from silica.ui.renderer import _stage_track
    track = _stage_track([], console_width=120)
    plain = track.plain
    # No running phase → center=0, window starts at 0, no leading ellipsis
    assert not plain.startswith("…")
    assert "· recon" in plain
    assert "· payload" in plain


def test_injector_block_height_constant_across_widths():
    """Regression: the injector live block (spinner header + indented stage track) must
    keep the same height on a narrow vs wide console. A wrapping track grew the block
    between frames and tore the Live region on a small / non-fullscreen terminal."""
    import io
    from rich.console import Console, Group
    from rich.padding import Padding
    from rich.text import Text
    from silica.ui.renderer import _stage_track
    phases = [
        {"phase": "payload",   "status": "done",    "elapsed": 0.8},
        {"phase": "salience",  "status": "done",    "elapsed": 0.3},
        {"phase": "collision", "status": "running", "elapsed": None},
    ]

    def block_height(width: int) -> int:
        buf = io.StringIO()
        c = Console(file=buf, width=width)
        header = Text(" injector · some/long/inbox/path/with a long file name.md",
                      no_wrap=True, overflow="ellipsis")
        c.print(Group(header, Padding(_stage_track(phases, width), (0, 0, 0, 2))))
        return len(buf.getvalue().rstrip("\n").split("\n"))

    heights = {w: block_height(w) for w in (30, 50, 80, 120)}
    assert len(set(heights.values())) == 1, f"block height varies with width: {heights}"


def test_stage_track_failed_phase_shown():
    from silica.ui.renderer import _stage_track
    phases = [
        {"phase": "recon",    "status": "done",   "elapsed": 1.0},
        {"phase": "payload",  "status": "failed", "elapsed": 0.2},
    ]
    track = _stage_track(phases, console_width=120)
    plain = track.plain
    assert "✗ payload" in plain


def _make_mock_batch(kind: str = "refine") -> dict:
    """Minimal _batch dict for micro-phase tests (bypasses terminal-gated BatchRunStartEvent)."""
    from unittest.mock import MagicMock
    return {
        "run_id": "r1", "kind": kind, "label": "Concepts",
        "total": 5, "done": 0, "failed": 0,
        "start_time": 0.0, "progress_obj": MagicMock(),
        "task_id": 0, "current_label": "", "micro_phase": "",
    }


def test_batch_micro_phase_tracked_from_work_feedback(monkeypatch):
    import silica.agent.bus as bus_mod
    from silica.agent.events import WorkFeedbackEvent
    from silica.ui.renderer import make_progress_callback

    # Logic-only test: pin off-terminal so the batch panel (full of MagicMocks)
    # is never rendered via the Live branch when FORCE_COLOR/a TTY is present.
    monkeypatch.setattr(CONSOLE, "_force_terminal", False)
    orig_bus = bus_mod.BUS
    bus_mod.BUS = bus_mod.EventBus()
    try:
        cb = make_progress_callback()
        cb._batch = _make_mock_batch("refine")

        bus_mod.BUS.publish("work/feedback", WorkFeedbackEvent(
            item_id="i1", kind="refine", phase="calling_llm"
        ))
        assert cb._batch["micro_phase"] == "calling_llm"
    finally:
        cb.close()
        bus_mod.BUS = orig_bus


def test_batch_micro_phase_ignored_for_wrong_kind():
    import silica.agent.bus as bus_mod
    from silica.agent.events import WorkFeedbackEvent
    from silica.ui.renderer import make_progress_callback

    orig_bus = bus_mod.BUS
    bus_mod.BUS = bus_mod.EventBus()
    try:
        cb = make_progress_callback()
        cb._batch = _make_mock_batch("refine")

        bus_mod.BUS.publish("work/feedback", WorkFeedbackEvent(
            item_id="i1", kind="dedup", phase="reading"
        ))
        assert cb._batch["micro_phase"] == ""
    finally:
        cb.close()
        bus_mod.BUS = orig_bus


def test_batch_micro_phase_resets_on_ledger_next_complete(monkeypatch):
    import json
    import silica.agent.bus as bus_mod
    from silica.agent.events import WorkFeedbackEvent, ToolCompleteEvent
    from silica.ui.renderer import make_progress_callback

    # Logic-only test: pin off-terminal (see sibling) to avoid rendering MagicMocks.
    monkeypatch.setattr(CONSOLE, "_force_terminal", False)
    orig_bus = bus_mod.BUS
    orig_mode = CONFIG.tool_progress
    bus_mod.BUS = bus_mod.EventBus()
    try:
        CONFIG.tool_progress = "all"  # must not be "off" — ToolCompleteEvent is behind mode guard
        cb = make_progress_callback()
        cb._batch = _make_mock_batch("refine")

        # Set micro phase via BUS
        bus_mod.BUS.publish("work/feedback", WorkFeedbackEvent(
            item_id="i1", kind="refine", phase="committing"
        ))
        assert cb._batch["micro_phase"] == "committing"

        # ledger_next complete (not done — advances done counter, resets micro_phase)
        cb(ToolCompleteEvent(
            name="silica_ledger_next",
            args={},
            call_id="c1",
            result=json.dumps({"done": False, "payload": {"note_paths": ["a/b.md"]}}),
            duration_s=0.1,
            iteration=1,
        ))
        assert cb._batch["micro_phase"] == ""
    finally:
        cb.close()
        bus_mod.BUS = orig_bus
        CONFIG.tool_progress = orig_mode


def test_batch_bar_advances_per_ledger_update_not_per_next(monkeypatch):
    """One ledger_next now hands out a whole frontier, so counting its
    completions would undercount a 5-task run as 1. The bar counts the
    per-task ledger_update instead."""
    import json
    import silica.agent.bus as bus_mod
    from silica.agent.events import ToolCompleteEvent
    from silica.ui.renderer import make_progress_callback

    monkeypatch.setattr(CONSOLE, "_force_terminal", False)
    orig_bus = bus_mod.BUS
    orig_mode = CONFIG.tool_progress
    bus_mod.BUS = bus_mod.EventBus()
    try:
        CONFIG.tool_progress = "all"
        cb = make_progress_callback()
        cb._batch = _make_mock_batch("autolink")

        def _complete(name, result):
            cb(ToolCompleteEvent(
                name=name, args={}, call_id="c", result=json.dumps(result),
                duration_s=0.1, iteration=1,
            ))

        _complete("silica_ledger_next", {
            "tasks": [{"task_id": f"t{i}", "payload": {"note_paths": [f"a/n{i}.md"]}}
                      for i in range(3)],
            "remaining": 0,
        })
        assert cb._batch["done"] == 0            # handed out, not finished
        assert cb._batch["current_label"] == "n0.md"

        for _ in range(3):
            _complete("silica_ledger_update", {"ok": True})
        assert cb._batch["done"] == 3
    finally:
        cb.close()
        bus_mod.BUS = orig_bus
        CONFIG.tool_progress = orig_mode


def test_live_aware_handler_resolves_stderr_dynamically():
    """Regression for torn panels: a log handler that caches ``sys.stderr`` at
    construction writes raw bytes while a ``rich.Live`` has redirected stderr to its
    coordinating proxy → the live region tears. The handler must resolve ``sys.stderr``
    at emit time so Live can print the log above the region cleanly."""
    import sys, io
    from silica.ui.logging import LiveAwareStreamHandler

    h = LiveAwareStreamHandler()
    orig = sys.stderr
    proxy = io.StringIO()  # stand-in for rich.Live's FileProxy
    try:
        sys.stderr = proxy  # simulate Live.start() redirect
        assert h.stream is proxy
    finally:
        sys.stderr = orig
    # Live.stop() restores stderr → handler follows it back, no stale reference.
    assert h.stream is orig


def test_injector_progress_not_given_its_own_live():
    """Regression: the embedded Progress must NOT own an active Live. Progress.start()
    opens a second Live on the global console that double-renders the bar — an orphan
    progress bar above the panel on small consoles. The outer self._live drives it."""
    from unittest.mock import patch
    from rich.console import Console
    from silica.agent.events import ToolStartEvent
    from silica.ui.renderer import make_progress_callback

    orig_mode = CONFIG.tool_progress
    cb = make_progress_callback()
    try:
        CONFIG.tool_progress = "all"
        with patch.object(Console, "is_terminal", new_callable=PropertyMock, return_value=True), \
             patch.object(cb, "_update_live", lambda: None):  # isolate: no real outer Live
            cb(ToolStartEvent(name="silica_run_injector",
                              args={"inbox_files": ["a.md", "b.md"]}, call_id="1", iteration=1))
        assert cb._inject_progress is not None
        assert cb._inject_progress.live.is_started is False
    finally:
        cb.close()
        CONFIG.tool_progress = orig_mode


def test_injector_single_file_has_no_bar():
    """A 0/1→1/1 bar is noise: the file bar only appears for multi-file runs."""
    from unittest.mock import patch
    from rich.console import Console
    from silica.agent.events import ToolStartEvent
    from silica.ui.renderer import make_progress_callback

    orig_mode = CONFIG.tool_progress
    cb = make_progress_callback()
    try:
        CONFIG.tool_progress = "all"
        with patch.object(Console, "is_terminal", new_callable=PropertyMock, return_value=True), \
             patch.object(cb, "_update_live", lambda: None):
            cb(ToolStartEvent(name="silica_run_injector",
                              args={"inbox_files": ["a.md"]}, call_id="1", iteration=1))
        assert cb._inject_progress is None
    finally:
        cb.close()
        CONFIG.tool_progress = orig_mode


def _phase(phase, status, *, scope="chunk", fi=0, ft=1, ci=0, ct=0, src="", elapsed=None):
    from silica.agent.events import PhaseEvent
    return PhaseEvent(phase=phase, status=status, scope=scope, file_idx=fi,
                      file_total=ft, chunk_idx=ci, chunk_total=ct,
                      source_file=src, elapsed=elapsed)


def test_phase_refires_do_not_touch_file_bar():
    """The bar tracks FILES, not phases — so phase re-fires (retries / deferred
    reprocessing) within one file never move it. The track still dedups by label."""
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn,
    )
    from silica.ui.renderer import make_progress_callback

    cb = make_progress_callback()
    try:
        cb._injector_call_id = "1"   # the guard: events are ignored otherwise
        cb._inject_progress = Progress(
            SpinnerColumn(), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            auto_refresh=False,
        )
        cb._inject_task_id = cb._inject_progress.add_task("", total=3)  # 3 files
        cb._pipeline_phases = []
        cb._phase_start_times = {}

        phases = ["payload", "salience", "collision"]
        for _ in range(3):  # three passes over the same phases, all inside file 0
            for p in phases:
                cb._on_pipeline_phase(_phase(p, "running", ft=3))
                cb._on_pipeline_phase(_phase(p, "done", ft=3, elapsed=0.1))

        # Bar untouched by phase events; track deduped to 3 distinct phases.
        assert cb._inject_progress.tasks[cb._inject_task_id].completed == 0
        assert len(cb._pipeline_phases) == len(phases)
    finally:
        cb.close()


def test_phase_events_ignored_when_injector_not_running():
    """The renderer subscribes to work/phase for its whole life, so it must drop
    events arriving outside an injector call — as _on_work_feedback does for
    batches. Without the guard a stray event would open a phase track under
    whatever tool happened to be live."""
    from silica.ui.renderer import make_progress_callback

    cb = make_progress_callback()
    try:
        assert cb._injector_call_id is None
        cb._on_pipeline_phase(_phase("distill", "running"))
        assert cb._pipeline_phases == []
    finally:
        cb.close()


def test_committed_file_counts_toward_bar_done():
    """Regression: an already-committed (dedup'd) file is in the denominator
    (len(inbox_files)) but is skipped before PAYLOAD, so it never enters the
    chunk map. It must still count as done, or the bar stalls below 100%.

    _current_file_idx starts past the committed prefix, so the position the FSM
    reports already accounts for it — this pins that, not the old chunk-map count.
    """
    from silica.router.orchestrator import InjectorFSM

    with patch("silica.kernel.write.ledger.get_ledger"):
        fsm = InjectorFSM(inbox_files=["Inbox/a.md", "Inbox/b.md"], target_dir="Concepts")
    fsm._committed_file_indices = {0}          # file 0 already nucleated → skipped
    fsm._current_file_idx = fsm._next_uncommitted_file_idx(0)

    pos = fsm._phase_position("recon")
    assert pos["file_idx"] == 1, "committed file not counted as behind us"
    assert pos["file_total"] == 2
    assert pos["source_file"] == "Inbox/b.md"


def test_phase_position_names_the_file_being_processed():
    """Regression: during file 2's RECON/PAYLOAD the chunk map still
    points into file 1 (it only gains file 2's entries at file 2's PAYLOAD), so
    deriving the position from it named the previous document."""
    from silica.router.orchestrator import InjectorFSM

    with patch("silica.kernel.write.ledger.get_ledger"):
        fsm = InjectorFSM(inbox_files=["Inbox/a.md", "Inbox/b.md"], target_dir="Concepts")
    fsm._chunk_flat_to_fi_ci = {0: (0, 0), 1: (0, 1)}   # only file 0 payloaded
    fsm._current_chunk_idx = 1
    fsm._current_file_idx = 1                            # advanced, RECON of file 1

    pos = fsm._phase_position("recon")
    assert pos["source_file"] == "Inbox/b.md"
    assert pos["file_idx"] == 1
    assert pos["scope"] == "file"
    # A file-scope phase runs before this file has chunks: reporting the previous
    # file's chunk index here would rewind the counter under the reader.
    assert pos["chunk_idx"] == 0


def test_phase_position_chunk_total_is_per_file():
    """The run-wide chunk total does not exist until the last file is
    partitioned (_chunks grows one file-group at a time), so the denominator
    shown must be the current file's."""
    from silica.router.orchestrator import InjectorFSM

    with patch("silica.kernel.write.ledger.get_ledger"):
        fsm = InjectorFSM(inbox_files=["Inbox/a.md", "Inbox/b.md"], target_dir="Concepts")
    fsm._file_chunks = {0: {"chunks": [{}, {}, {}]}, 1: {"chunks": [{}, {}]}}
    fsm._chunk_flat_to_fi_ci = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1)}
    fsm._chunks = [{}] * 5

    fsm._current_file_idx, fsm._current_chunk_idx = 0, 1
    assert fsm._phase_position("distill")["chunk_total"] == 3
    assert fsm._phase_position("distill")["chunk_idx"] == 1

    fsm._current_file_idx, fsm._current_chunk_idx = 1, 4
    pos = fsm._phase_position("distill")
    assert (pos["chunk_idx"], pos["chunk_total"]) == (1, 2), "counter must restart per file"


def test_failed_chunk_records_its_phase():
    """The failure ledger must say WHERE a chunk died structurally: on reload
    there is no phase stream, and the phase used to exist only inside the error
    prose. rollback must not overwrite the gate that actually failed."""
    from silica.router.orchestrator import InjectorFSM

    with patch("silica.kernel.write.ledger.get_ledger"):
        fsm = InjectorFSM(inbox_files=["Inbox/a.md"], target_dir="Concepts")
    fsm._progress_note("f0_c0_lint", "lint", "running")
    fsm._progress_note("f0_c0_rollback", "rollback", "running")
    assert fsm._failed_phase_id() == "lint"


def test_injector_bar_total_is_file_count():
    from unittest.mock import patch
    from rich.console import Console
    from silica.agent.events import ToolStartEvent
    from silica.ui.renderer import make_progress_callback

    orig_mode = CONFIG.tool_progress
    cb = make_progress_callback()
    try:
        CONFIG.tool_progress = "all"
        with patch.object(Console, "is_terminal", new_callable=PropertyMock, return_value=True), \
             patch.object(cb, "_update_live", lambda: None):
            cb(ToolStartEvent(name="silica_run_injector",
                              args={"inbox_files": ["a.md", "b.md", "c.md"]},
                              call_id="1", iteration=1))
        task = cb._inject_progress.tasks[cb._inject_task_id]
        assert task.total == 3  # files, not 16 phases
    finally:
        cb.close()
        CONFIG.tool_progress = orig_mode


def test_phase_event_advances_file_bar():
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn,
    )
    from silica.ui.renderer import make_progress_callback

    cb = make_progress_callback()
    try:
        cb._injector_call_id = "1"
        cb._inject_progress = Progress(
            SpinnerColumn(), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            auto_refresh=False,
        )
        cb._inject_task_id = cb._inject_progress.add_task("", total=3)
        cb._on_pipeline_phase(_phase("recon", "running", scope="file", fi=2, ft=3))
        assert cb._inject_progress.tasks[cb._inject_task_id].completed == 2
        cb._on_pipeline_phase(_phase("recon", "running", scope="file", fi=9, ft=3))
        assert cb._inject_progress.tasks[cb._inject_task_id].completed == 3  # clamped
    finally:
        cb.close()


def test_phase_event_updates_inbox_label_and_position():
    """Regression: the panel title must follow the document currently processed,
    not stay frozen on the first file. Position rides on every event, so a
    missed one cannot leave the header naming the wrong file."""
    from silica.ui.renderer import make_progress_callback

    cb = make_progress_callback()
    try:
        cb._injector_call_id = "1"
        cb._inject_inbox_label = "a.md"
        cb._on_pipeline_phase(_phase("distill", "running", fi=1, ft=2, ci=2, ct=5, src="b.md"))
        assert cb._inject_inbox_label == "b.md"
        assert "file 2/2" in cb._position_suffix()
        assert "chunk 3/5" in cb._position_suffix()
        cb._on_pipeline_phase(_phase("lint", "running", fi=1, ft=2, ci=2, ct=5, src=""))
        assert cb._inject_inbox_label == "b.md"  # empty label must not clobber
    finally:
        cb.close()


def test_single_file_run_shows_no_file_counter():
    """A `file 1/1` counter is noise, the same reason the bar is suppressed."""
    from silica.ui.renderer import make_progress_callback

    cb = make_progress_callback()
    try:
        cb._injector_call_id = "1"
        cb._on_pipeline_phase(_phase("distill", "running", fi=0, ft=1, ci=1, ct=4))
        suffix = cb._position_suffix()
        assert "file" not in suffix
        assert "chunk 2/4" in suffix
    finally:
        cb.close()


def test_injector_summary_shows_yield(capsys):
    import json
    from unittest.mock import patch
    from rich.console import Console
    from silica.agent.events import ToolCompleteEvent
    from silica.ui.renderer import make_progress_callback

    orig_mode = CONFIG.tool_progress
    cb = make_progress_callback()
    try:
        CONFIG.tool_progress = "all"
        cb._injector_call_id = "1"
        cb._inject_file_count = 3
        cb._inject_inbox_label = "a.md"
        cb._pipeline_phases = [{"phase": "write", "status": "done", "elapsed": 0.1}]
        with patch.object(Console, "is_terminal", new_callable=PropertyMock, return_value=True):
            cb(ToolCompleteEvent(
                name="silica_run_injector", args={}, call_id="1",
                result=json.dumps({"yield_notes": 7, "yield_links": 12}),
                duration_s=4.2, iteration=1,
            ))
        out = capsys.readouterr().out
        assert "3 files" in out
        assert "7 notes" in out
        assert "12 links" in out
    finally:
        cb.close()
        CONFIG.tool_progress = orig_mode


def test_injector_projection_carries_yield_counts():
    """The summary line above reads yield_notes/yield_links off the tool result,
    but the projection never copied them out of fsm.context — so every real run
    printed "1 file · 3m12s" and never what it had created. Pins the projection,
    not just the renderer's willingness to display it."""
    from unittest.mock import MagicMock, patch as _patch

    with _patch("silica.router.coordinator.Coordinator") as Coord, \
         _patch("silica.kernel.vault_manifest.get_active_manifest") as manifest, \
         _patch("silica.sources.registry.adapter_for", return_value=object()):
        manifest.return_value.sources = []
        inst = MagicMock()
        inst.run.return_value = {
            "final_status": "Success", "committed_chunks": 2,
            "yield_notes": 7, "yield_links": 12,
        }
        Coord.return_value = inst
        from silica.tools.runners import silica_run_injector
        out = silica_run_injector(inbox_files=["Inbox/a.md"], target_dir="Concepts")

    assert out["yield_notes"] == 7
    assert out["yield_links"] == 12
    assert out["files_total"] == 1


