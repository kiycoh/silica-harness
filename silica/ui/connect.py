# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`silica connect` — host the WS bridge server the Obsidian plugin dials into.

Also the auto-host used by the TUI (start_bridge_thread, rpc-only) and the GUI
lifespan (maybe_start_bridge, full chat) so a running Obsidian upgrades the
driver without a separate `silica connect` session.

PROTOCOL.md's server half. One loopback socket carries both channels: the
plugin's chat turns stream through the transport-neutral `run_turn` (framed as
`chat_event*` + one `chat_done`/`chat_error`), while DRIVER `rpc` frames
interleave on the same connection. On handshake the accepted connection is
wrapped in an attached ObsidianWSBackend and installed as the global driver;
on drop the driver falls back to the local fs backend.

Only stdlib at import time — websockets is imported inside start()/run_connect
so the module loads without the [connect] extra.
"""
from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import secrets
import threading
from pathlib import Path
from typing import Any

from silica.config import CONFIG

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1

# websockets' 1 MiB default max_size would sever the connection on any note
# body over ~1 MB (rpc read replies, bulk-nucleate creates). 32 MiB matches the
# no-ceiling behavior of the fs backend. Mirrored in ws_backend.py.
MAX_FRAME = 2**25


def _origin_ok(origin: str) -> bool:
    """Only two callers are legitimate: Obsidian's Electron renderer (its
    WebSocket always sends `Origin: app://obsidian.md`) and native clients,
    which send no Origin at all. Every web-page origin — loopback included —
    is refused: any open browser tab can reach a loopback port, and the token
    should not be the only gate."""
    return origin in ("", "app://obsidian.md")


async def _send(ws: Any, frame: dict) -> None:
    await ws.send(json.dumps(frame))


class BridgeServer:
    """The ws://127.0.0.1 bridge server (PROTOCOL.md, server half)."""

    def __init__(self, chat_enabled: bool = True) -> None:
        self.port: int = CONFIG.ws_port  # 0 → OS picks; real port set by start()
        self.token: str = CONFIG.ws_token or secrets.token_hex(16)
        # False when the TUI hosts the bridge: the REPL owns the conversation
        # and doesn't share the GUI's turn gate, so plugin chat is refused.
        self.chat_enabled = chat_enabled
        self._server: Any = None
        self._backend: Any = None  # the attached ws driver, while a plugin is connected
        self._bridge_file: Path | None = None
        self._chat_task: Any = None  # the in-flight turn (one at a time — _begin_turn gates)
        # Set by the auto-start helper below when the TUI hosts the bridge on
        # its own thread; _stop_bridge_thread needs it to reach that loop.
        self._host_loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        from websockets.asyncio.server import serve

        self._server = await serve(self._handler, "127.0.0.1", self.port, max_size=MAX_FRAME)
        self.port = self._server.sockets[0].getsockname()[1]
        self._bridge_file = self._write_bridge_file()
        logger.info("bridge: listening on ws://127.0.0.1:%d", self.port)

    async def stop(self) -> None:
        self._uninstall(self._backend)
        if self._bridge_file is not None:
            self._bridge_file.unlink(missing_ok=True)
            self._bridge_file = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def _write_bridge_file(self) -> Path:
        """Discovery file the plugin reads to find port + token (mode 0600)."""
        path = Path(CONFIG.vault_path) / ".obsidian" / "silica-bridge.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "port": self.port, "token": self.token,
            "pid": os.getpid(), "protocolVersion": PROTOCOL_VERSION,
        })
        # Created 0600 from the first byte — the token is never briefly readable.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        path.chmod(0o600)  # a stale file from a dead session may carry another mode
        return path

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _handler(self, ws: Any) -> None:
        if not await self._handshake(ws):
            return
        backend = self._install(ws)
        await _send(ws, {
            "type": "welcome", "vault": Path(CONFIG.vault_path).name,
            "obsidianVersion": "", "protocolVersion": PROTOCOL_VERSION,
        })
        try:
            async for raw in ws:
                try:
                    frame = json.loads(raw)
                except ValueError:
                    logger.warning("bridge: non-JSON frame ignored")
                    continue
                await self._route(ws, frame, backend)
        finally:
            self._uninstall(backend)

    async def _handshake(self, ws: Any) -> bool:
        try:
            hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        except Exception:  # closed early, silent for 10s, or non-JSON — no one to greet
            logger.warning("bridge: handshake refused (no valid hello)")
            await ws.close()
            return False
        if not isinstance(hello, dict):
            hello = {}
        origin = ws.request.headers.get("Origin", "") if ws.request is not None else ""
        if not _origin_ok(origin):
            reason = f"origin refused: {origin}"
        elif not secrets.compare_digest(str(hello.get("token", "")), self.token):
            reason = "bad token"
        elif hello.get("protocolVersion") != PROTOCOL_VERSION:
            reason = f"protocol mismatch: {hello.get('protocolVersion')!r}"
        else:
            return True
        logger.warning("bridge: handshake refused (%s)", reason)
        await _send(ws, {"type": "bye", "reason": reason})
        await ws.close()
        return False

    def _install(self, ws: Any) -> Any:
        from silica.driver import set_driver
        from silica.driver.ws_backend import ObsidianWSBackend

        backend = ObsidianWSBackend.attached(ws, asyncio.get_running_loop())
        self._backend = backend
        set_driver(backend)
        logger.info("bridge: plugin connected — ws driver installed")
        return backend

    def _uninstall(self, backend: Any) -> None:
        from silica.driver import set_driver

        if backend is None or backend is not self._backend:
            return  # a newer connection already took over
        self._backend = None
        backend.detach("plugin disconnected")
        set_driver(None)
        logger.info("bridge: plugin disconnected — driver falls back to the fs backend")

    # ------------------------------------------------------------------
    # Frame routing
    # ------------------------------------------------------------------

    async def _route(self, ws: Any, frame: dict, backend: Any) -> None:
        # ponytail: reaches across module seams into privates — web._begin_turn /
        # web.current_cancel (the GUI's turn gate) and backend._on_frame (the ws
        # driver's rpc demux). Deliberate for now: the bridge is the only second
        # consumer, so promoting them to public API would be speculative. Promote
        # to a shared public interface if a third caller needs the same gate/demux.
        # The web import stays inside the chat branches: rpc frames must keep
        # working when the TUI hosts the bridge without the [gui] extra.
        kind = frame.get("type")
        if kind in ("rpc_result", "rpc_error"):
            backend._on_frame(frame)
        elif kind == "chat":
            tid, text = str(frame.get("turnId", "")), str(frame.get("text", ""))
            if not self.chat_enabled:
                await _send(ws, {"type": "chat_error", "turnId": tid,
                                 "error": "chat is unavailable: this bridge is hosted by a "
                                          "TUI session; use `silica --gui` or `silica connect`"})
                return
            from silica.ui.web import server as web

            if not web._begin_turn():
                await _send(ws, {"type": "chat_error", "turnId": tid,
                                 "error": "a turn is already in progress"})
                return
            # Held on self — the event loop only weakly references tasks, and a
            # bare create_task can be garbage-collected mid-turn.
            self._chat_task = asyncio.create_task(self._chat_turn(ws, tid, text))
        elif kind == "chat_cancel":
            if not self.chat_enabled:
                return
            from silica.ui.web import server as web

            if web.current_cancel is not None:
                web.current_cancel.set()
        elif kind == "event":
            logger.debug("bridge: metadata event: %s", frame)  # non-fatal, LINT audits later

    async def _chat_turn(self, ws: Any, tid: str, text: str) -> None:
        from silica.ui.web.server import run_turn

        try:
            async for item in run_turn(text):
                kind = item.get("type")
                if kind == "done":
                    await _send(ws, {"type": "chat_done", "turnId": tid,
                                     "answer": item.get("answer", ""),
                                     "html": item.get("html", "")})
                elif kind == "error":
                    await _send(ws, {"type": "chat_error", "turnId": tid,
                                     "error": item.get("error", "")})
                else:
                    await _send(ws, {"type": "chat_event", "turnId": tid, "event": item})
        except Exception as exc:  # socket died mid-turn; run_turn's finally cleans up
            logger.warning("bridge: chat turn aborted: %s", exc)


def bridge_supported() -> bool:
    """True when auto-hosting the bridge makes sense: the [connect] extra is
    installed and the vault is a real Obsidian vault (has .obsidian/) — repo
    .silica/ vaults have no plugin to dial in."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False
    return bool(CONFIG.vault_path) and (Path(CONFIG.vault_path) / ".obsidian").is_dir()


async def maybe_start_bridge(chat_enabled: bool = True) -> BridgeServer | None:
    """Async host for the GUI lifespan: start the bridge on the running loop.
    Returns None when unsupported; caller owns stop()."""
    if not bridge_supported():
        return None
    server = BridgeServer(chat_enabled=chat_enabled)
    await server.start()
    return server


def start_bridge_thread() -> BridgeServer | None:
    """Sync host for the TUI: bridge on a daemon loop thread, rpc channel only
    (chat_enabled=False — the REPL owns the conversation). Stops via atexit so
    the discovery file is unlinked on clean exit."""
    if not bridge_supported():
        return None
    server = BridgeServer(chat_enabled=False)
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="silica-bridge", daemon=True).start()
    try:
        asyncio.run_coroutine_threadsafe(server.start(), loop).result(10)
    except Exception as exc:  # port clash, etc. — the TUI must not die for the bridge
        logger.warning("bridge auto-start failed: %s", exc)
        loop.call_soon_threadsafe(loop.stop)
        return None
    server._host_loop = loop  # for _stop_bridge_thread
    atexit.register(_stop_bridge_thread, server)
    return server


def _stop_bridge_thread(server: BridgeServer) -> None:
    loop = server._host_loop
    if loop is None:  # never auto-started: nothing hosts a loop to stop
        return
    try:
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result(5)
    except Exception:
        pass  # exiting anyway — stop() only tidies the socket and bridge file
    loop.call_soon_threadsafe(loop.stop)


def run_connect() -> int:
    """`silica connect` entry: host the bridge until Ctrl-C. Returns exit code."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        print("silica connect requires the [connect] extra: pip install 'silica-harness[connect]'")
        return 1
    if not CONFIG.vault_path:
        print("silica connect needs a vault: set SILICA_VAULT or run inside a repo with .silica/")
        return 1
    logger.info("bridge: serving from the fs backend until a plugin attaches")

    async def _main() -> None:
        server = BridgeServer()
        await server.start()
        print(f"silica connect — ws://127.0.0.1:{server.port} (Ctrl-C to stop)")
        try:
            await asyncio.get_running_loop().create_future()  # serve until interrupted
        finally:
            await server.stop()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
    return 0
