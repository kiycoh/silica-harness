"""Streaming path: call_llm(on_delta=…) emits deltas and reassembles the response;
run_agent gates streaming to the interactive main loop; the renderer accumulates
the transient preview buffer and resets it at the turn boundaries."""
from unittest.mock import MagicMock, patch, PropertyMock

from silica.config import CONFIG
from tests.llm_mocks import litellm_mock_response


def _chunk(content=None, reasoning=None):
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = reasoning
    delta.reasoning = None
    choice = MagicMock()
    choice.delta = delta
    ch = MagicMock()
    ch.choices = [choice]
    return ch


def test_call_llm_streams_deltas_and_reassembles():
    from silica.agent.llm import call_llm

    chunks = [
        _chunk(reasoning="hmm"),
        _chunk(content="Hel"),
        _chunk(content="lo"),
        MagicMock(choices=[]),  # usage-only trailing chunk → skipped
    ]
    built = litellm_mock_response("Hello")
    deltas: list[tuple[str, str]] = []

    with patch("litellm.completion", return_value=iter(chunks)) as mock_completion, \
         patch("litellm.stream_chunk_builder", return_value=built) as mock_builder:
        res = call_llm(model="test_model", messages=[{"role": "user", "content": "hi"}],
                       on_delta=lambda t, c: deltas.append((t, c)))

    assert mock_completion.call_args.kwargs.get("stream") is True
    assert mock_builder.called
    assert deltas == [("reset", ""), ("reasoning", "hmm"), ("text", "Hel"), ("text", "lo")]
    assert res.text == "Hello"
    # The rebuilt message carries no reasoning, so the deltas are the only copy
    # left: without keeping them, a reopened chat has no thinking to replay.
    assert res.reasoning == "hmm"
    assert res.assistant_message["silica_reasoning"] == "hmm"


def test_bounded_stream_times_out_on_stalled_chunk():
    """A stream that hangs (provider accepts then sends nothing) raises
    litellm.Timeout via the per-chunk deadline, instead of blocking forever (A3)."""
    import threading

    import litellm
    import pytest

    from silica.agent.llm import _bounded_stream

    block = threading.Event()

    def _hung_stream():
        block.wait()  # never released before the deadline → simulates a hang
        yield  # pragma: no cover

    gen = _bounded_stream(_hung_stream, per_chunk_timeout=0.05, model="test/model")
    with pytest.raises(litellm.Timeout):
        next(gen)
    block.set()  # let the daemon pump exit cleanly


def test_bounded_stream_forwards_chunks_and_propagates_error():
    import litellm
    import pytest

    from silica.agent.llm import _bounded_stream

    assert list(_bounded_stream(lambda: iter([1, 2, 3]), 5.0, "test/model")) == [1, 2, 3]

    def _boom():
        yield 1
        raise litellm.APIConnectionError("dropped", model="m", llm_provider="p")

    gen = _bounded_stream(_boom, 5.0, "test/model")
    assert next(gen) == 1
    with pytest.raises(litellm.APIConnectionError):
        next(gen)


def test_call_llm_without_on_delta_does_not_stream():
    from silica.agent.llm import call_llm

    with patch("litellm.completion", return_value=litellm_mock_response("Hi")) as mock_completion:
        res = call_llm(model="test_model", messages=[{"role": "user", "content": "hi"}])

    assert "stream" not in mock_completion.call_args.kwargs
    assert res.text == "Hi"


@patch("silica.agent.loop.call_llm")
def test_run_agent_streams_only_with_callback(mock_call_llm):
    from silica.agent.llm import LLMResponse
    from silica.agent.loop import run_agent

    mock_call_llm.return_value = LLMResponse(
        text="done", tool_calls=[],
        assistant_message={"role": "assistant", "content": "done"}, usage={},
    )

    run_agent([{"role": "user", "content": "x"}], model="m",
              tool_progress_callback=lambda e: None)
    assert callable(mock_call_llm.call_args.kwargs["on_delta"])

    # Without a callback the kwarg is omitted entirely (bare-signature doubles keep working)
    run_agent([{"role": "user", "content": "x"}], model="m")
    assert "on_delta" not in mock_call_llm.call_args.kwargs


def _tool_then_answer(preamble="let me look"):
    """A provider that streams `preamble` as text and only then reveals that the
    turn is a tool call; the second turn answers. Returns (fake_call_llm, TOOLS
    patch target name)."""
    from silica.agent.llm import LLMResponse, ToolCall

    responses = [
        LLMResponse(
            text=preamble,
            tool_calls=[ToolCall(id="tc1", name="silica_read_note", args={"name": "n"})],
            assistant_message={"role": "assistant", "tool_calls": []}, usage={},
        ),
        LLMResponse(
            text="Final answer", tool_calls=[],
            assistant_message={"role": "assistant", "content": "Final answer"}, usage={},
        ),
    ]

    def fake_call_llm(model, messages, **kw):
        resp = responses.pop(0)
        if kw.get("on_delta") is not None and resp.text:
            kw["on_delta"]("text", resp.text)
        return resp

    return fake_call_llm


def _run_collecting(events, **run_kw):
    from silica.agent.loop import run_agent
    from silica.tools import TOOLS

    tool = MagicMock()
    tool.run.return_value = "note content"
    # Explicit, not MagicMock defaults: a bare mock's .sensitive is truthy, so the
    # loop filters it out of `allowed` and dispatch takes the forbidden-tool
    # branch — no ToolStartEvent, and the ordering this file asserts is vacuous.
    tool.sensitive = False
    tool.internal = False
    with patch.dict(TOOLS, {"silica_read_note": tool}):
        return run_agent([{"role": "user", "content": "x"}], model="m",
                         tool_progress_callback=events.append, **run_kw)


@patch("silica.agent.loop.call_llm")
def test_tool_call_retracts_the_text_it_already_streamed(mock_call_llm):
    """The loop learns the streamed text was a preamble only after painting it,
    so it retracts it — before the first tool line, since the TUI's live region
    is shared and a later reset would clear a region that has moved on."""
    from silica.agent.events import LLMStreamEvent, ToolStartEvent

    mock_call_llm.side_effect = _tool_then_answer()
    events: list = []
    assert _run_collecting(events) == "Final answer"

    kinds = [(type(e), getattr(e, "chunk_type", None)) for e in events]
    resets = [i for i, k in enumerate(kinds) if k == (LLMStreamEvent, "reset")]
    texts = [i for i, k in enumerate(kinds) if k == (LLMStreamEvent, "text")]
    starts = [i for i, (t, _) in enumerate(kinds) if t is ToolStartEvent]
    assert len(resets) == 1, kinds
    assert texts and starts
    assert texts[0] < resets[0] < starts[0], kinds
    # Scoped: the reasoning that produced the tool call stays on screen.
    assert events[resets[0]].content == "text"


@patch("silica.agent.loop.call_llm")
def test_a_plain_answer_is_never_retracted(mock_call_llm):
    from silica.agent.events import LLMStreamEvent
    from silica.agent.llm import LLMResponse

    def fake_call_llm(model, messages, **kw):
        if kw.get("on_delta") is not None:
            kw["on_delta"]("text", "Final answer")
        return LLMResponse(text="Final answer", tool_calls=[],
                           assistant_message={"role": "assistant", "content": "Final answer"},
                           usage={})

    mock_call_llm.side_effect = fake_call_llm
    events: list = []
    assert _run_collecting(events) == "Final answer"
    assert not [e for e in events
                if isinstance(e, LLMStreamEvent) and e.chunk_type == "reset"]


@patch("silica.agent.loop.call_llm")
def test_an_interactive_constrained_run_still_streams(mock_call_llm):
    """The chat_tools cut rides in via AgentConstraints, but an interactive
    turn (CLI REPL, GUI chat) must keep its live stream and stay off the
    worker-slot cap — "constrained toolset" no longer implies "worker"."""
    from silica.agent.constraints import AgentConstraints

    mock_call_llm.side_effect = _tool_then_answer()
    events: list = []
    with patch("silica.agent.loop.worker_slot",
               side_effect=AssertionError("interactive turn grabbed a worker slot")):
        assert _run_collecting(events, constraints=AgentConstraints(
            tools=("silica_read_note",), interactive=True)) == "Final answer"
    assert callable(mock_call_llm.call_args.kwargs["on_delta"])


@patch("silica.agent.loop.call_llm")
def test_no_reset_when_the_run_never_streamed(mock_call_llm):
    """Constrained runs stay on the non-streaming call (no on_delta), so nobody
    subscribed to a reset — emitting one anyway would clear an unrelated region."""
    from silica.agent.constraints import AgentConstraints
    from silica.agent.events import LLMStreamEvent

    mock_call_llm.side_effect = _tool_then_answer()
    events: list = []
    assert _run_collecting(events, constraints=AgentConstraints(
        tools=("silica_read_note",))) == "Final answer"
    assert "on_delta" not in mock_call_llm.call_args.kwargs
    assert not [e for e in events
                if isinstance(e, LLMStreamEvent) and e.chunk_type == "reset"]


def test_renderer_stream_buffer_accumulates_and_resets():
    from rich.console import Console
    from silica.agent.events import LLMStreamEvent, ThinkingEndEvent
    from silica.ui.renderer import make_progress_callback

    orig = (CONFIG.tool_progress, CONFIG.show_thinking, CONFIG.verbose)
    cb = make_progress_callback()
    try:
        CONFIG.tool_progress = "all"
        CONFIG.show_thinking = False
        CONFIG.verbose = False
        with patch.object(Console, "is_terminal", new_callable=PropertyMock, return_value=True), \
             patch.object(cb, "_update_stream_live", lambda: None):
            cb(LLMStreamEvent(chunk_type="text", content="Hel", iteration=1))
            cb(LLMStreamEvent(chunk_type="text", content="lo", iteration=1))
            assert cb._stream_buf == "Hello"
            # Reasoning deltas hidden unless thinking/verbose is on
            cb(LLMStreamEvent(chunk_type="reasoning", content="secret", iteration=1))
            assert cb._stream_buf == "Hello"
            # Retry reset clears the preview
            cb(LLMStreamEvent(chunk_type="reset", content="", iteration=1))
            assert cb._stream_buf == ""
            # Turn boundary clears it too — the final answer is printed by cli.py
            cb(LLMStreamEvent(chunk_type="text", content="again", iteration=1))
            cb(ThinkingEndEvent(iteration=1))
            assert cb._stream_buf == ""
    finally:
        CONFIG.tool_progress, CONFIG.show_thinking, CONFIG.verbose = orig
        cb.close()
