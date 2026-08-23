"""`silica connect` — the WS bridge server the Obsidian plugin dials into.

Covers PROTOCOL.md's server half: bridge-file discovery, the handshake gate
(token / protocolVersion / Origin), the chat channel framed over `run_turn`
(chat_event*/chat_done, busy refusal, cancel), and the driver channel — the
accepted connection is wrapped in an attached ObsidianWSBackend, installed as
the global driver, and uninstalled (fallback to CONFIG.backend) on drop.

The fake plugin here is a plain websockets client: production topology
(Python hosts, plugin dials) — the inverse of test_ws_backend's unit-3 seam.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

import pytest

pytest.importorskip("websockets")
pytest.importorskip("fastapi")  # run_turn lives in the web server module
from websockets.asyncio.client import connect  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402

from silica.agent.events import LLMStreamEvent  # noqa: E402
from silica.config import CONFIG  # noqa: E402
from silica.driver.base import NoteRef  # noqa: E402


@pytest.fixture
def bridge_env(tmp_vault, tmp_path, monkeypatch):
    """Fresh web-session state + driver singleton around each bridge test."""
    from silica.ui.web import server as web

    web._reset_session()
    yield web


def _run(scenario, **srv_kwargs) -> None:
    """Boot a BridgeServer, run the async scenario against it, tear down."""
    from silica.ui.connect import BridgeServer

    async def main():
        srv = BridgeServer(**srv_kwargs)
        await srv.start()
        try:
            await asyncio.wait_for(scenario(srv), timeout=15)
        finally:
            await srv.stop()

    asyncio.run(main())


async def _dial(srv, token=None, **kwargs):
    """Dial + hello like the plugin; return (ws, first non-rpc reply frame).

    The handshake installs the connection as the global driver BEFORE sending
    welcome, so a process-wide DRIVER call can inject an rpc frame ahead of
    it — demux via _recv_chat, like the real plugin. Refused handshakes never
    install the driver, so their `bye` is always the first frame anyway."""
    ws = await connect(f"ws://127.0.0.1:{srv.port}", **kwargs)
    await ws.send(json.dumps({
        "type": "hello",
        "token": srv.token if token is None else token,
        "protocolVersion": 1, "role": "plugin",
    }))
    return ws, await _recv_chat(ws)


# ---------------------------------------------------------------------------
# Discovery + handshake gate
# ---------------------------------------------------------------------------

def test_bridge_file_written_and_removed(bridge_env):
    path = Path(CONFIG.vault_path) / ".obsidian" / "silica-bridge.json"

    async def scenario(srv):
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec == {"port": srv.port, "token": srv.token,
                       "pid": os.getpid(), "protocolVersion": 1}
        assert path.stat().st_mode & 0o777 == 0o600

    _run(scenario)
    assert not path.exists()  # removed on stop


def test_handshake_welcome(bridge_env):
    async def scenario(srv):
        ws, welcome = await _dial(srv)
        assert welcome["type"] == "welcome"
        assert welcome["protocolVersion"] == 1
        assert welcome["vault"] == Path(CONFIG.vault_path).name
        await ws.close()

    _run(scenario)


def test_bad_token_gets_bye(bridge_env):
    async def scenario(srv):
        ws, reply = await _dial(srv, token="wrong")
        assert reply["type"] == "bye"
        assert "token" in reply["reason"]
        await ws.close()

    _run(scenario)


def test_obsidian_electron_origin_accepted(bridge_env):
    async def scenario(srv):
        # Obsidian's Electron renderer always sends Origin: app://obsidian.md.
        ws, welcome = await _dial(srv, origin="app://obsidian.md")
        assert welcome["type"] == "welcome"
        await ws.close()

    _run(scenario)


def test_web_page_origins_refused(bridge_env):
    async def scenario(srv):
        # Every page origin is refused — loopback included: any open browser
        # tab can reach a loopback port, so only Obsidian's renderer passes.
        for origin in ("https://evil.example", "http://127.0.0.1:8000"):
            ws, reply = await _dial(srv, origin=origin)
            assert reply["type"] == "bye"
            assert "origin" in reply["reason"].lower()
            await ws.close()

    _run(scenario)


def test_garbage_hello_is_refused_quietly(bridge_env):
    async def scenario(srv):
        ws = await connect(f"ws://127.0.0.1:{srv.port}")
        await ws.send("not json")
        with pytest.raises(ConnectionClosed):  # server closes; no welcome, no crash
            await asyncio.wait_for(ws.recv(), timeout=5)

    _run(scenario)


# ---------------------------------------------------------------------------
# Chat channel — run_turn framed as chat_event*/chat_done
# ---------------------------------------------------------------------------

_EMPTY_GRAPH_RPC = {
    "list_files": [],
    "resolved_links": {"resolved": {}, "unresolved": {}},
    "mention_index": {},
}


async def _recv_chat(ws):
    """Next non-rpc frame (chat_*, welcome, bye), answering interleaved driver
    rpc frames (graph reads from html rendering, or any process-wide DRIVER
    call once the handshake installs this socket as the driver) with empty
    results — this is what proves rpc and chat share the socket without
    deadlocking the loop."""
    while True:
        frame = json.loads(await ws.recv())
        if frame.get("type") == "rpc":
            await ws.send(json.dumps({"type": "rpc_result", "id": frame["id"],
                                      "result": _EMPTY_GRAPH_RPC.get(frame["method"])}))
            continue
        return frame

def test_chat_streams_events_then_done(bridge_env, monkeypatch):
    web = bridge_env

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        tool_progress_callback(LLMStreamEvent("content", "Hello", 0))
        messages.append({"role": "assistant", "content": "Hello"})
        return "Hello"

    monkeypatch.setattr(web, "run_agent", fake_run_agent)

    async def scenario(srv):
        ws, _ = await _dial(srv)
        await ws.send(json.dumps({"type": "chat", "turnId": "t1", "text": "hi"}))
        frames = []
        while True:
            frame = await _recv_chat(ws)
            frames.append(frame)
            if frame["type"] in ("chat_done", "chat_error"):
                break
        assert all(f["turnId"] == "t1" for f in frames)
        assert any(f["type"] == "chat_event" and f["event"]["type"] == "delta"
                   and f["event"]["text"] == "Hello" for f in frames)
        assert frames[-1]["type"] == "chat_done"
        assert frames[-1]["answer"] == "Hello"
        assert frames[-1]["html"]  # rendered markdown travels with the answer
        await ws.close()

    _run(scenario)


def test_second_chat_while_busy_is_refused(bridge_env, monkeypatch):
    web = bridge_env
    release = threading.Event()

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        tool_progress_callback(LLMStreamEvent("content", "…", 0))
        release.wait(10)
        messages.append({"role": "assistant", "content": "ok"})
        return "ok"

    monkeypatch.setattr(web, "run_agent", fake_run_agent)

    async def scenario(srv):
        ws, _ = await _dial(srv)
        await ws.send(json.dumps({"type": "chat", "turnId": "t1", "text": "first"}))
        first = await _recv_chat(ws)  # t1 is in flight once its delta lands
        assert first["type"] == "chat_event"
        await ws.send(json.dumps({"type": "chat", "turnId": "t2", "text": "second"}))
        refusal = await _recv_chat(ws)
        assert refusal == {"type": "chat_error", "turnId": "t2",
                           "error": "a turn is already in progress"}
        release.set()
        done = await _recv_chat(ws)
        assert done["type"] == "chat_done" and done["turnId"] == "t1"
        await ws.close()

    _run(scenario)


def test_chat_cancel_sets_the_cancel_token(bridge_env, monkeypatch):
    web = bridge_env

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        tool_progress_callback(LLMStreamEvent("content", "…", 0))
        cancelled = cancel_token.wait(10)
        answer = "cancelled" if cancelled else "timeout"
        messages.append({"role": "assistant", "content": answer})
        return answer

    monkeypatch.setattr(web, "run_agent", fake_run_agent)

    async def scenario(srv):
        ws, _ = await _dial(srv)
        await ws.send(json.dumps({"type": "chat", "turnId": "t1", "text": "go"}))
        first = await _recv_chat(ws)
        assert first["type"] == "chat_event"
        await ws.send(json.dumps({"type": "chat_cancel", "turnId": "t1"}))
        done = await _recv_chat(ws)
        assert done["type"] == "chat_done"
        assert done["answer"] == "cancelled"
        await ws.close()

    _run(scenario)


# ---------------------------------------------------------------------------
# Driver channel — attached ws backend installed on handshake, dropped on close
# ---------------------------------------------------------------------------

def test_rpc_frames_serve_the_installed_driver(bridge_env):
    from silica.driver import get_driver
    from silica.driver.ws_backend import ObsidianWSBackend

    async def scenario(srv):
        ws, _ = await _dial(srv)
        be = get_driver()
        assert isinstance(be, ObsidianWSBackend)

        async def plugin():
            frame = json.loads(await ws.recv())
            assert frame["type"] == "rpc" and frame["method"] == "read"
            await ws.send(json.dumps({
                "type": "rpc_result", "id": frame["id"],
                "result": {"path": frame["params"]["path"],
                           "content": "over the bridge", "size": 15},
            }))

        replier = asyncio.create_task(plugin())
        # Driver calls arrive from the sync agent worker thread in production.
        nc = await asyncio.to_thread(be.read_note, NoteRef(name="X", path="X.md"))
        await replier
        assert nc.content == "over the bridge"
        await ws.close()

    _run(scenario)


def test_plugin_drop_falls_back_to_local_backend(bridge_env):
    from silica.driver import get_driver
    from silica.driver.ws_backend import ObsidianWSBackend

    async def scenario(srv):
        ws, _ = await _dial(srv)
        assert isinstance(get_driver(), ObsidianWSBackend)
        await ws.close()
        for _ in range(200):  # the handler notices the drop asynchronously
            if not isinstance(get_driver(), ObsidianWSBackend):
                break
            await asyncio.sleep(0.02)
        assert not isinstance(get_driver(), ObsidianWSBackend)  # fs fallback (tmp_vault)

    _run(scenario)


# ---------------------------------------------------------------------------
# Startup fallback + CLI wiring
# ---------------------------------------------------------------------------

def test_cli_dispatches_silica_connect(monkeypatch):
    import silica.cli as cli
    import silica.ui.connect as connect_mod

    calls = []
    monkeypatch.setattr(connect_mod, "run_connect", lambda: calls.append(1) or 0)
    monkeypatch.setattr(cli, "_activate_repo_mode", lambda: None)
    monkeypatch.setattr(cli, "_resolve_context_budget", lambda: None)
    assert cli._dispatch_subcommand(["connect"]) == 0
    assert calls == [1]


# ---------------------------------------------------------------------------
# Auto-hosting — the TUI thread host and the GUI lifespan host
# ---------------------------------------------------------------------------

def test_bridge_supported_requires_obsidian_dir(bridge_env):
    from silica.ui.connect import bridge_supported

    assert not bridge_supported()  # tmp_vault has no .obsidian/
    (Path(CONFIG.vault_path) / ".obsidian").mkdir()
    assert bridge_supported()


def test_chat_refused_when_hosted_by_tui(bridge_env):
    async def scenario(srv):
        ws, _ = await _dial(srv)
        await ws.send(json.dumps({"type": "chat", "turnId": "t1", "text": "ciao"}))
        reply = await _recv_chat(ws)
        assert reply["type"] == "chat_error"
        assert reply["turnId"] == "t1"
        assert "silica --gui" in reply["error"]
        await ws.close()

    _run(scenario, chat_enabled=False)


def test_large_frames_survive_the_bridge(bridge_env):
    """A frame over websockets' 1 MiB default must not sever the connection
    (rpc replies carry whole note bodies). The follow-up chat probe proves
    the socket is still alive."""
    async def scenario(srv):
        ws, _ = await _dial(srv, max_size=None)
        big = json.dumps({"type": "rpc_result", "id": 999, "result": "x" * (2 * 2**20)})
        await ws.send(big)  # unknown id — routed to the driver demux, ignored
        await ws.send(json.dumps({"type": "chat", "turnId": "t2", "text": "ping"}))
        reply = await _recv_chat(ws)
        assert reply["type"] == "chat_error"  # chat disabled, but the reply arrived
        await ws.close()

    _run(scenario, chat_enabled=False)


def test_chat_probe_survives_interleaved_driver_rpc(bridge_env):
    """Regression for the 2026-08-05 full-suite flake: the socket is one
    channel for two protocols, and the moment the handshake installs this
    connection as the global driver, ANY process-wide DRIVER call (a leaked
    worker thread from an earlier test, a post-turn capture flush) lands as an
    rpc frame between a chat probe and its reply. The probe must demux like
    the real plugin does — asserting on the first frame is a protocol
    violation, and it failed exactly here with reply["type"] == "rpc"."""
    import contextlib

    from silica.driver import get_driver

    async def scenario(srv):
        ws, _ = await _dial(srv)
        be = get_driver()
        # the leaked-thread stand-in: a sync driver call from a worker thread
        rpc_call = asyncio.create_task(asyncio.to_thread(be.list_files))
        await asyncio.sleep(0.1)  # let its rpc frame reach the wire first
        await ws.send(json.dumps({"type": "chat", "turnId": "t9", "text": "ping"}))
        reply = await _recv_chat(ws)  # answers the rpc, then gets the chat reply
        assert reply["type"] == "chat_error"
        assert await rpc_call == []  # the interleaved caller got unblocked too
        with contextlib.suppress(Exception):
            await ws.close()

    _run(scenario, chat_enabled=False)


def test_maybe_start_bridge_noop_without_obsidian_dir(bridge_env):
    from silica.ui.connect import maybe_start_bridge

    assert asyncio.run(maybe_start_bridge()) is None


def test_gui_lifespan_hosts_bridge(bridge_env):
    """`silica --gui` hosts the bridge for the whole app lifetime: the
    discovery file appears on startup and is gone after shutdown."""
    from fastapi.testclient import TestClient

    from silica.ui.web.server import app

    (Path(CONFIG.vault_path) / ".obsidian").mkdir()
    bridge_file = Path(CONFIG.vault_path) / ".obsidian" / "silica-bridge.json"
    with TestClient(app):
        assert bridge_file.exists()
    assert not bridge_file.exists()


def test_start_bridge_thread_hosts_rpc_only(bridge_env):
    import atexit

    import silica.ui.connect as connect_mod
    from silica.driver import get_driver
    from silica.driver.ws_backend import ObsidianWSBackend

    (Path(CONFIG.vault_path) / ".obsidian").mkdir()
    srv = connect_mod.start_bridge_thread()
    assert srv is not None
    try:
        async def scenario():
            ws, welcome = await _dial(srv)
            assert welcome["type"] == "welcome"
            assert isinstance(get_driver(), ObsidianWSBackend)  # driver hot-swapped
            await ws.send(json.dumps({"type": "chat", "turnId": "t3", "text": "hi"}))
            reply = json.loads(await ws.recv())
            assert reply["type"] == "chat_error"  # rpc-only host
            await ws.close()

        asyncio.run(asyncio.wait_for(scenario(), timeout=15))
    finally:
        atexit.unregister(connect_mod._stop_bridge_thread)
        connect_mod._stop_bridge_thread(srv)
    assert not (Path(CONFIG.vault_path) / ".obsidian" / "silica-bridge.json").exists()
