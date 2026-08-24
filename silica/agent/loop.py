# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""The agentic loop — the core of Silica.

This is the 'while True' from SILICA.md §8.1:
  loop:
    response = LLM(system_prompt, message_history, tool_schemas)
    if response has tool_calls:
        for each tool_call:
            result = execute_tool(name, args)
            append tool_result to history
        continue  (re-call LLM with results)
    else:
        return response text to user

Everything else (streaming, TUI, context compression) is ergonomics
around this nucleus. Build this first, then ergonomics.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Any
import concurrent.futures as _cf
import threading
import time
import logging
import json

if TYPE_CHECKING:
    from silica.kernel.progress import ProgressLedger

import silica.agent.bus as _bus_mod
from silica.agent.events import (
    ToolStartEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ReasoningEvent,
    RenderEvent,
    ThinkingStartEvent,
    ThinkingEndEvent,
    LLMStreamEvent,
)
from silica.agent.llm import call_llm
from silica.agent import narration as _narr_mod
from silica.agent.compaction import (
    COMPACT_FLOOR_TURNS,
    COMPACT_FRACTION,
    compact_read_history,
    eager_stub,
)
from silica.config import CONFIG
from silica.agent.concurrency import worker_slot
from silica.agent.constraints import AgentConstraints
from silica.tools import TOOLS, Tool
from contextlib import nullcontext

logger = logging.getLogger(__name__)


def _topic_for(event: RenderEvent) -> str | None:
    if isinstance(event, ToolStartEvent):
        return "agent/tool_start"
    if isinstance(event, ToolCompleteEvent):
        return "agent/tool_complete"
    if isinstance(event, ToolErrorEvent):
        return "agent/tool_error"
    if isinstance(event, (ThinkingStartEvent, ThinkingEndEvent)):
        return "agent/thinking"
    if isinstance(event, ReasoningEvent):
        return "agent/reasoning"
    if isinstance(event, LLMStreamEvent):
        return "agent/stream"
    return None


def _is_tool_failure(result: Any) -> bool:
    """Helper to detect if a tool result indicates a failure."""
    if not result:
        return False
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "error" in parsed:
                return True
        except Exception:
            pass
        # A non-JSON / non-error-keyed string result is a successful tool
        # output, not a failure. Substring-sniffing for "error"/"failed" here
        # misclassifies legitimate content (grep hits, "0 errors" reports).
    elif isinstance(result, dict) and "error" in result:
        return True
    return False


ToolProgressCallback = Callable[[RenderEvent], None] | None


# How many steps from the wall the model is told about it. Silent above this:
# a budget line on every iteration is noise the model learns to skip, and the
# number only changes the right move once finishing IS the right move.
BUDGET_NOTICE_AT = 2

# What a turn that produced neither text nor a tool call returns. A sentinel and
# not "": every caller gates its output on the returned string being truthy, so
# an empty return renders as silence and the user cannot tell a broken provider
# call from a turn that never ran.
_EMPTY_COMPLETION = "(silica: the model returned an empty response)"

_FINAL_TURN_INSTRUCTION = (
    "The tool phase of this turn is over and no tools are available now. "
    "Answer from what you already have: report what you found, and name any "
    "note you created or edited. Do not request, announce, or simulate another "
    "tool call. If the work is incomplete, say plainly what is missing."
)


def _budget_notice(iteration: int, max_iterations: int) -> dict | None:
    """Tell the model how much of the loop is left, at the point where the
    answer changes.

    The loop used to stop dead at the cap having never told the model a cap
    existed, so it could spend its last steps opening work it had no room to
    close. Silent while there is room; a system notice near the wall, on the
    same channel as the convergence guard so tool results stay contiguous.
    """
    remaining = max_iterations - iteration
    if remaining > BUDGET_NOTICE_AT:
        return None
    return {
        "role": "system",
        "content": (
            f"Budget: {remaining} tool step(s) remain in this turn. Prefer to "
            f"finish now and report what you have, including anything already "
            f"written. Use another tool only when it is strictly required to "
            f"complete work you have already started."
        ),
    }


def repair_tool_call_history(messages: list[dict]) -> int:
    """Ensure every assistant `tool_calls` block is answered by a tool result.

    An interrupt (KeyboardInterrupt is a BaseException, uncaught by the dispatch
    loop) or the convergence abort can exit mid-dispatch, leaving an assistant
    message whose tool_calls have no matching `tool` responses. The next
    `call_llm` then rejects the orphaned block with a 400 and the session is dead
    until /clear. Insert a synthetic error result for each unanswered id, right
    after that block's existing tool responses. Idempotent. Returns count inserted.
    """
    inserted = 0
    i = 0
    while i < len(messages):
        m = messages[i]
        calls = m.get("tool_calls") if m.get("role") == "assistant" else None
        if calls:
            j = i + 1
            answered: set = set()
            while j < len(messages) and messages[j].get("role") == "tool":
                answered.add(messages[j].get("tool_call_id"))
                j += 1
            missing = [c.get("id") for c in calls if c.get("id") not in answered]
            for k, cid in enumerate(missing):
                messages.insert(j + k, {
                    "role": "tool", "tool_call_id": cid,
                    "content": '{"error": "tool call interrupted before it produced a result"}',
                })
            inserted += len(missing)
            i = j + len(missing)
        else:
            i += 1
    return inserted


def identical_prior_calls(messages: list[dict], name: str, args_str: str) -> int:
    """How many times (name, args) already appears as an assistant tool_call.

    Reuse probe: a result the chat computed, spent, and now asks for again. Reads
    the history instead of a counter so it spans turns (the caller accumulates
    `messages`), and `args_str` must be the canonical json.dumps(sort_keys=True)
    form since providers echo their own spacing back.
    """
    n = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") != name:
                continue
            try:
                prior = json.loads(fn.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if json.dumps(prior, sort_keys=True) == args_str:
                n += 1
    return n


def run_agent(
    messages: list[dict],
    model: str,
    tool_progress_callback: ToolProgressCallback = None,
    progress: "ProgressLedger | None" = None,
    cancel_token: "threading.Event | None" = None,
    constraints: "AgentConstraints | None" = None,
    temperature: float | None = None,
) -> str:
    """Execute the agentic loop until the model produces a text response.

    The loop calls the LLM, dispatches any tool calls, appends results,
    and re-calls until the model responds with text (no tool calls).

    Args:
        messages: mutable conversation history (modified in-place)
        model: litellm model string
        tool_progress_callback: callback for tool progress events

    Returns:
        The model's final text response
    """
    # Effective tool registry: full global, or the constrained subset.
    if constraints is not None:
        allowed: dict[str, "Tool"] = {
            name: TOOLS[name] for name in constraints.tools if name in TOOLS
        }
    else:
        # Non-ambient authority: the main agent's default toolset excludes
        # sensitive tools (ADR-0009 / ADR-0015) and pipeline internals the FSM
        # drives programmatically. Both are reachable only when a caller names
        # them in AgentConstraints.tools.
        allowed = {n: t for n, t in TOOLS.items() if not t.sensitive and not t.internal}

    # Collect tool schemas for the LLM
    schemas = [t.json_schema() for t in allowed.values()] if allowed else None

    effective_model = (
        constraints.model if (constraints is not None and constraints.model) else model
    )

    # A33: a prior turn interrupted (Ctrl+C, BaseException) or convergence-aborted
    # mid-dispatch can leave an assistant tool_calls block with unanswered ids;
    # the first call_llm below would then 400 the whole session. Self-heal on entry.
    repair_tool_call_history(messages)
    # ponytail: a single meter for every tool. Split retrieval from writes if a
    # turn is ever seen spending its whole allowance on reads, or vice versa.

    iteration = 0
    max_iterations = (
        constraints.max_iterations
        if (constraints is not None and constraints.max_iterations is not None)
        else 20
    )  # Hard safety cap lowered from 50

    # Track consecutive failures for the same (tool_name, args) pair
    # Key: (tool_name, args_json_string)
    # Value: consecutive failure count
    consecutive_failures: dict[tuple[str, str], int] = {}
    # Set when the convergence guard stops the loop: the run lands with the
    # same tool-less final turn as the iteration cap, instead of discarding
    # everything it found behind a RuntimeError.
    convergence_landing = False

    # Message indices already elided by this run's compaction sweeps. The callers
    # keep their own set across turns; the two never conflict because a stub is
    # shorter than MIN_COLLAPSE_CHARS, so a second sweep skips it.
    collapsed: set[int] = set()

    def _emit(event: RenderEvent) -> None:
        """Best-effort event emission to callback and bus."""
        if tool_progress_callback is not None:
            try:
                tool_progress_callback(event)
            except Exception as exc:
                logger.debug("tool_progress_callback error (swallowed): %s", exc)
        topic = _topic_for(event)
        if topic is not None:
            _bus_mod.BUS.publish(topic, event)
        # Durable record: one adapter call, all render-event kinds (spec §4).
        # Best-effort like the callback above — narration must never kill a turn.
        try:
            _narr_mod.NARRATOR.on_render_event(event)
        except Exception as exc:
            logger.debug("narration adapter error (swallowed): %s", exc)

    # Set once the main thread stops waiting on the LLM (Ctrl+C or normal return).
    # The LLM runs on a detached daemon thread that keeps streaming deltas after an
    # interrupt; this gate stops those late deltas from re-opening the live region
    # and printing thinking below "(interrupted)". Also passed as `cancel` to abort
    # retries. Cleared at the top of each iteration.
    _abandon = threading.Event()

    def _stream_delta(chunk_type: str, content: str) -> None:
        # Called from the LLM worker thread; `iteration` reads the current loop pass.
        if _abandon.is_set():
            return  # interrupted/abandoned — stop feeding the renderer
        _emit(LLMStreamEvent(chunk_type=chunk_type, content=content, iteration=iteration))

    # Streaming is a TUI ergonomic: only interactive turns get it — worker and
    # batch runs stay on the plain non-streaming call. A constrained toolset no
    # longer implies "worker": an interactive caller that wants the chat_tools
    # cut says so with constraints.interactive.
    # The kwarg is only passed when active, so call_llm test doubles only need
    # the bare signature plus cancel=None.
    _interactive = constraints is None or constraints.interactive
    _llm_kwargs: dict = {"tools": None}
    if temperature is not None:
        # None keeps the provider default (product behavior). Eval agent arms
        # pin 0.0 so a single-run A/B measures the lever, not sampling noise.
        _llm_kwargs["temperature"] = temperature
    if tool_progress_callback is not None and _interactive:
        _llm_kwargs["on_delta"] = _stream_delta

    def _interruptible_llm(kwargs: dict):
        """One call_llm on a *daemon* thread — see the comment in the loop
        body for why (Ctrl+C delivery, no shutdown join, retry abandonment).
        Caller clears `_abandon` first; this always sets it on the way out so
        an orphaned worker stops retrying and streaming."""
        _future: _cf.Future = _cf.Future()

        def _llm_worker(k=dict(kwargs)):
            try:
                _future.set_result(call_llm(effective_model, messages, **k))
            except BaseException as e:
                _future.set_exception(e)

        threading.Thread(target=_llm_worker, daemon=True, name="llm-call").start()
        try:
            return _future.result()
        finally:
            _abandon.set()

    while iteration < max_iterations:
        if cancel_token is not None and cancel_token.is_set():
            logger.info("Agent loop cancelled at iteration %d", iteration)
            return "(silica: cancelled)"
        iteration += 1
        logger.debug("Agent loop iteration %d", iteration)

        _emit(ThinkingStartEvent(iteration=iteration))
        try:
            # Run the (synchronous, potentially slow) LLM call on a *daemon*
            # thread so a Ctrl+C on the main thread raises KeyboardInterrupt out
            # of _future.result() instead of being trapped in a C-level network
            # recv(). Daemon matters: a non-daemon orphan (the old throwaway
            # ThreadPoolExecutor worker) gets joined at interpreter shutdown,
            # hanging exit for minutes while its retries die against executors
            # already flagged shut ("cannot schedule new futures after shutdown").
            # The finally sets `_abandon` so retry_transient stops rescheduling
            # once nobody is waiting (harmless on success: the call already
            # returned). ponytail: sync litellm can't abort the in-flight HTTP
            # request — best we can do is stop waiting and stop retrying.
            slot = nullcontext() if _interactive else worker_slot()
            with slot:
                _abandon.clear()
                _llm_kwargs["tools"] = schemas
                _llm_kwargs["cancel"] = _abandon
                resp = _interruptible_llm(_llm_kwargs)
        finally:
            _emit(ThinkingEndEvent(iteration=iteration))
        # Closed here and not in the adapter: the reasoning text only exists
        # once resp does, and the durable thought carries it whole (spec §4).
        _narr_mod.NARRATOR.thought_close(resp.reasoning or "")
        messages.append(resp.assistant_message)
        # Only intermediate (tool-calling) assistant messages narrate here.
        # The FINAL one is narrated by the caller after attribution mutates
        # messages[-1] in place (WebTurn.attribute appends the Sources block):
        # narrating it now would freeze the pre-citation text into the replay.
        if resp.tool_calls:
            _narr_mod.NARRATOR.turn(resp.assistant_message)

        if resp.reasoning:
            _emit(ReasoningEvent(text=resp.reasoning, iteration=iteration))

        # No tool calls → model produced a final text response
        if not resp.tool_calls:
            # ...or produced NOTHING: measured 2026-08-23 on
            # openrouter/stealth/ox-alpha, finish=stop with tool_calls=0 and
            # content='' after 21s. `or ""` sent that straight into the REPL's
            # `if answer:`, which prints nothing — a failed turn rendered
            # exactly like no turn at all. Same shape as the cancelled and
            # max-iterations sentinels: the caller always gets text back.
            return resp.text or _EMPTY_COMPLETION

        # Tool calls, so whatever text this iteration streamed was a preamble and
        # not the answer — but its deltas are already painted. Retract them with
        # the signal the stream contract already carries (llm.py sends the same
        # event when a transient retry replays an attempt); content names the
        # scope, so a renderer keeps the reasoning that produced the call.
        # Must precede the first ToolStartEvent below: the TUI's live region is
        # shared, and a reset arriving after a tool line would clear a region
        # that has already moved on. Guarded on on_delta because streaming is
        # only wired for the interactive loop — a worker run must not emit an
        # event nobody subscribed to. _emit and not _stream_delta: _abandon is
        # already set by the time the call returns.
        if _llm_kwargs.get("on_delta") is not None:
            _emit(LLMStreamEvent(chunk_type="reset", content="text", iteration=iteration))

        # The loop is where the history explodes — a single fat read can add
        # thousands of tokens, and every later iteration re-sends it. The
        # callers only sweep once run_agent has already returned, so on a local
        # backend a long turn could overrun the pinned window mid-flight and get
        # truncated in silence. `prompt_tokens` is the provider's own count of
        # what we just sent, so this costs no tokenizer pass; when a provider
        # reports no usage, fall back to the same chars/4 estimate the meter uses.
        prompt_tokens = resp.usage.get("prompt_tokens") or sum(
            len(str(m.get("content") or "")) for m in messages
        ) // 4
        _pre_collapsed = set(collapsed)
        collapsed = compact_read_history(
            messages,
            collapsed,
            prompt_tokens=prompt_tokens,
            budget=int(COMPACT_FRACTION * CONFIG.max_context_tokens),
            floor_turns=COMPACT_FLOOR_TURNS,
            tools=TOOLS,
        )
        _newly = sorted(set(collapsed) - _pre_collapsed)
        if _newly:
            # Without this beat the turn beats alone cannot reconstruct the
            # live context (ticket 05): the elision would be invisible.
            _narr_mod.NARRATOR.narrate("compaction", "done",
                              f"compacted {len(_newly)} message(s)",
                              {"indices": _newly, "prompt_tokens": prompt_tokens},
                              parent=None)

        # Dispatch each tool call
        pending_notices: list[dict] = []  # A34: convergence warnings, flushed AFTER
        for tc in resp.tool_calls:        # the tool block, never between sibling results
            logger.info("Tool call: %s(%s)", tc.name, tc.args)

            # Key representing the specific tool call + args
            args_str = json.dumps(tc.args, sort_keys=True)
            tool_key = (tc.name, args_str)

            # Reuse probe. The current call is already in `messages` (:270), so
            # >1 means this exact result was produced earlier in the chat and is
            # being recomputed — either a plain re-ask or compaction's read_stub
            # telling the model to re-call. Grep "reuse-probe" over a session log
            # to size the waste before caching anything.
            _seen = identical_prior_calls(messages, tc.name, args_str)
            if _seen > 1:
                logger.info("reuse-probe: %s recomputed (call #%d) with identical args: %s",
                            tc.name, _seen, args_str[:200])

            failed = False
            if tc.name not in allowed:
                failed = True
                result = f'{{"error": "Unknown or forbidden tool: {tc.name}"}}'
                _emit(
                    ToolErrorEvent(
                        name=tc.name,
                        call_id=tc.id,
                        error=f"Unknown or forbidden tool: {tc.name}",
                        iteration=iteration,
                    )
                )
            else:
                _emit(
                    ToolStartEvent(
                        name=tc.name,
                        args=tc.args,
                        call_id=tc.id,
                        iteration=iteration,
                    )
                )
                start_time = time.perf_counter()
                try:
                    # The frontend callback, deliberately not `_emit`: `_emit`
                    # also publishes on the bus, and a tool that runs a loop of
                    # its own has its own `_emit` publishing there already — so
                    # forwarding that one would put every inner event on
                    # `agent/tool_*` twice.
                    result = allowed[tc.name].run(
                        _cancel_token=cancel_token,
                        _progress=tool_progress_callback,
                        **tc.args,
                    )
                    duration = time.perf_counter() - start_time
                    _emit(
                        ToolCompleteEvent(
                            name=tc.name,
                            args=tc.args,
                            call_id=tc.id,
                            result=result,
                            duration_s=duration,
                            iteration=iteration,
                        )
                    )
                    if _is_tool_failure(result):
                        failed = True
                except Exception as e:
                    duration = time.perf_counter() - start_time
                    _emit(
                        ToolErrorEvent(
                            name=tc.name,
                            call_id=tc.id,
                            error=str(e),
                            iteration=iteration,
                        )
                    )
                    failed = True
                    result = f'{{"error": "{type(e).__name__}: {str(e)}"}}'

            # Eager projection: a write/gate tool's fat JSON never enters the
            # history — the TUI already got the full result via the event above.
            # Errors stay verbatim so the model can react to them.
            if not failed and tc.name in allowed and allowed[tc.name].collapse == "eager":
                result = eager_stub(allowed[tc.name], result)

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
            messages.append(tool_msg)
            _narr_mod.NARRATOR.turn(tool_msg)

            # Update convergence guard
            if failed:
                consecutive_failures[tool_key] = consecutive_failures.get(tool_key, 0) + 1
                failures_count = consecutive_failures[tool_key]
                if failures_count >= 3:
                    logger.error("Convergence guard: tool '%s' with args %s failed %d times consecutively. Aborting agent run.", tc.name, tc.args, failures_count)
                    if progress is not None and progress.cursor:
                        try:
                            progress.set_status(
                                progress.cursor,
                                "blocked",
                                error=f"Convergence guard: '{tc.name}' failed 3× consecutively",
                            )
                            progress.save()
                        except Exception:
                            pass
                    # Stop the loop, keep the run: a deep research turn that
                    # stubs its toes on one tool (measured live: a re-pasted
                    # non-verbatim `remember` quote, three times) still holds
                    # 40+ steps of findings worth answering from.
                    convergence_landing = True
                    break
                elif failures_count == 2:
                    logger.warning("Convergence guard: tool '%s' with args %s failed consecutively. Injecting warning message.", tc.name, tc.args)
                    pending_notices.append(
                        {
                            "role": "system",
                            "content": f"IMPORTANT: Tool '{tc.name}' failed consecutively with these parameters. DO NOT call this tool again with the exact same arguments."
                        }
                    )
            else:
                consecutive_failures[tool_key] = 0

        # A34: convergence notices go after ALL tool results for this assistant
        # block, never interleaved between sibling results (strict tool protocols
        # require the tool messages contiguous immediately after the assistant).
        messages.extend(pending_notices)

        if convergence_landing:
            # The guard broke out mid-dispatch: siblings of the aborted call
            # may be unanswered, and the landing call below rejects orphaned
            # tool_calls blocks.
            repair_tool_call_history(messages)
            break

        notice = _budget_notice(iteration, max_iterations)
        if notice is not None:
            messages.append(notice)
            _narr_mod.NARRATOR.turn(notice)

        # Loop continues: re-call LLM with tool results

    if convergence_landing:
        logger.warning("Agent loop stopped by the convergence guard; landing")
    else:
        logger.warning("Agent loop hit max iterations (%d)", max_iterations)
    # One last turn with the tools removed, rather than discarding the whole
    # turn. Everything the model found or wrote is still in `messages`; asking
    # for it back costs one call and turns a completed write reported as
    # "maximum iterations reached" into an actual answer. Same discipline as
    # every other call this function makes: cancellation honoured, the
    # worker-slot cap held, the daemon-thread pattern kept (a bare in-thread
    # call here would trap the turn's LAST Ctrl+C in a C-level recv()), and
    # `_abandon` cleared so the TUI still streams this closing answer.
    if cancel_token is not None and cancel_token.is_set():
        logger.info("Agent loop cancelled before the final turn")
        return "(silica: cancelled)"
    messages.append({"role": "system", "content": _FINAL_TURN_INSTRUCTION})
    _narr_mod.NARRATOR.turn(messages[-1])
    _emit(ThinkingStartEvent(iteration=iteration))
    try:
        slot = nullcontext() if _interactive else worker_slot()
        with slot:
            _abandon.clear()
            final_kwargs = dict(_llm_kwargs)
            final_kwargs["tools"] = None   # removed, not merely discouraged
            final_kwargs["cancel"] = _abandon
            final = _interruptible_llm(final_kwargs)
    except Exception as e:
        logger.warning("Forced final turn failed (%s)", e)
        return "(silica: maximum iterations reached)"
    finally:
        _emit(ThinkingEndEvent(iteration=iteration))
    _narr_mod.NARRATOR.thought_close(final.reasoning or "")
    text = (final.text or "").strip() or "(silica: maximum iterations reached)"
    # The landing answer never reaches messages here (callers print it), so the
    # narration records it explicitly or the transcript ends mid-question.
    _narr_mod.NARRATOR.turn({"role": "assistant", "content": text})
    return text
