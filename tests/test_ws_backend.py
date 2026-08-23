"""ObsidianWSBackend read + write paths — driven against a Python stub WS server.

The stub is the machine-checkable contract the TS plugin must satisfy: it
speaks PROTOCOL.md's RPCs, but its answers are sourced from the FS backend
over the synthetic vault, so it is a faithful oracle. A WS backend consuming it
must therefore match the FS backend op-for-op — that's the parity test.

Topology here is the inverse of production (unit 5: Python hosts, plugin dials);
the RPC channel is symmetric, so backend-as-client is the simplest unit-3 seam.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import threading

import pytest

pytest.importorskip("websockets")
from websockets.asyncio.server import serve  # noqa: E402

from silica.driver.base import NoteRef  # noqa: E402
from silica.driver.fs_backend import ObsidianFSBackend  # noqa: E402


# ---------------------------------------------------------------------------
# Stub WS server — PROTOCOL.md read RPCs, answered from an FS oracle.
# ---------------------------------------------------------------------------

class StubPluginServer:
    """A loopback WS server that answers read RPCs from a real FS vault.

    Stands in for the Obsidian plugin: validates the token on `hello`, replies
    `welcome`, then serves `rpc` frames from the FS backend so results carry the
    exact PROTOCOL.md shapes a correct plugin would return.
    """

    def __init__(self, vault_path, token: str = "test-token"):
        self.token = token
        self._fs = ObsidianFSBackend(vault_path=str(vault_path))
        self._fs._ensure_index()
        self.port: int | None = None
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("stub server failed to start")

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)

    # -- server internals --------------------------------------------------

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._boot())
        self._loop.run_forever()

    async def _boot(self) -> None:
        self._server = await serve(self._handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        self._ready.set()

    async def _handler(self, ws) -> None:
        hello = json.loads(await ws.recv())
        if hello.get("token") != self.token or hello.get("protocolVersion") != 1:
            await ws.send(json.dumps({"type": "bye", "reason": "bad token or version"}))
            return
        await ws.send(json.dumps({
            "type": "welcome", "vault": "synthetic",
            "obsidianVersion": "stub", "protocolVersion": 1,
        }))
        async for raw in ws:
            frame = json.loads(raw)
            if frame.get("type") != "rpc":
                continue
            rid, method, params = frame.get("id"), frame.get("method"), frame.get("params") or {}
            if method == "__noreply__":
                continue  # test hook: never answer, to exercise the driver's RPC-timeout cleanup
            try:
                result = self._dispatch(method, params)
            except Exception as exc:  # fatal op → rpc_error, driver raises
                await ws.send(json.dumps({"type": "rpc_error", "id": rid, "error": str(exc)}))
            else:
                await ws.send(json.dumps({"type": "rpc_result", "id": rid, "result": result}))

    def _dispatch(self, method: str, params: dict):
        fs = self._fs
        if method == "read":
            path = params["path"]
            full = fs.vault_path / path
            if not full.exists():
                raise RuntimeError(f"read: file not found: {path}")
            # newline="" = byte-faithful, like the plugin's vault.cachedRead;
            # fs.read_note would normalize CRLF via universal-newlines read_text.
            with full.open(encoding="utf-8", newline="") as fh:
                content = fh.read()
            return {"path": path, "content": content, "size": len(content)}
        if method == "list_files":
            return [{"name": r.name, "path": r.path} for r in fs.list_files(params.get("folder", ""))]
        if method == "props_of":
            return fs.props_of(NoteRef(name="", path=params["path"]))
        if method == "outline":
            return [{"level": h.level, "text": h.text, "position": h.position}
                    for h in fs.outline(NoteRef(name="", path=params["path"]))]
        if method == "search_context":
            return self._grouped_search(params["query"])
        if method == "search_context_batch":
            return {q: self._grouped_search(q) for q in params["queries"]}
        if method == "resolved_links":
            resolved: dict = {}
            for s, t in fs._graph.edges():
                resolved.setdefault(s, {})[t] = 1
            unresolved: dict = {}
            for s, t in fs._unresolved_links:
                unresolved.setdefault(s, {})[t] = 1
            return {"resolved": resolved, "unresolved": unresolved}
        if method == "create":
            path = params["path"]
            if (fs.vault_path / path).exists():  # vault.create errors on existing files
                raise RuntimeError(f"create: path already exists: {path}")
            ref = fs.create(path, params["content"])
            return {"name": ref.name, "path": ref.path}
        if method == "overwrite":
            # Verbatim, like the real plugin: the Obsidian side is a dumb pipe
            # and does no stamping of its own (the WS backend already did it).
            fs._overwrite_raw(params["path"], params["content"])
            return {"ok": True}
        if method == "append":
            fs.append(self._ref(params["path"]), params["content"])
            return {"ok": True}
        if method == "set_prop":
            fs.set_prop(self._ref(params["path"]), params["name"], params["value"],
                        params.get("type", "text"))
            return {"ok": True}
        if method == "move":
            fs.move(self._ref(params["path"]), params["to"])
            return {"ok": True}
        if method == "delete":
            fs.delete(self._ref(params["path"]))
            return {"ok": True}
        if method == "autolink_note":
            return fs.autolink_note(params["path"], params.get("candidates"))
        raise RuntimeError(f"unknown method: {method}")

    @staticmethod
    def _ref(path: str) -> NoteRef:
        return NoteRef(name=path.rsplit("/", 1)[-1].removesuffix(".md"), path=path)

    def _grouped_search(self, query: str) -> list[dict]:
        by_path: dict[str, dict] = {}
        for hit in self._fs.search_context(query):
            entry = by_path.setdefault(hit.ref.path, {"path": hit.ref.path, "name": hit.ref.name, "matches": []})
            entry["matches"].append({"line": hit.line, "content": hit.snippet})
        return list(by_path.values())


@pytest.fixture(scope="module")
def stub(synthetic_vault):
    srv = StubPluginServer(synthetic_vault)
    yield srv
    srv.stop()


@pytest.fixture
def ws_backend(stub):
    from silica.driver.ws_backend import ObsidianWSBackend

    be = ObsidianWSBackend(url=stub.url, token=stub.token)
    yield be
    be.close()


@pytest.fixture
def fs(synthetic_vault):
    """FS backend over the same vault — the parity oracle."""
    return ObsidianFSBackend(vault_path=str(synthetic_vault))


# ---------------------------------------------------------------------------
# Read-path behaviour — measured by parity with the FS backend on one vault.
# ---------------------------------------------------------------------------

def test_read_note_round_trips_content_over_the_socket(ws_backend, synthetic_vault):
    """A `read` RPC returns the note's verbatim content, correlated by id."""
    nc = ws_backend.read_note(NoteRef(name="Concepts", path="Hub/Concepts.md"))
    assert "central hub" in nc.content


def test_read_note_by_bare_path_appends_md(ws_backend):
    """A path-shaped name without extension (as it arrives from wikilinks/search)
    resolves to the .md file, matching getFileByPath's exact-path requirement."""
    nc = ws_backend.read_note("Hub/Concepts")
    assert "central hub" in nc.content


def test_path_arg_extension_handling(ws_backend):
    pa = ws_backend._path_arg
    assert pa("Hub/Concepts") == "Hub/Concepts.md"      # bare path → .md
    assert pa("Hub/Concepts.md") == "Hub/Concepts.md"   # already .md → verbatim
    assert pa("Board/Sketch.canvas") == "Board/Sketch.canvas"  # other ext preserved


def test_list_files_matches_fs(ws_backend, fs):
    assert {r.path for r in ws_backend.list_files()} == {r.path for r in fs.list_files()}


def test_list_files_drops_vault_root_artifacts(ws_backend, monkeypatch):
    # Obsidian's list_files RPC returns EVERY .md, including Silica's own
    # GRAPH_REPORT.md — unlike fs/cli, which filter at their own list_files.
    # ws must strip it too, else it seeds the graph as a real wikilink node.
    monkeypatch.setattr(ws_backend, "_rpc", lambda *a, **k: [
        {"name": "GRAPH_REPORT", "path": "GRAPH_REPORT.md"},
        {"name": "log", "path": "log.md"},
        {"name": "Real", "path": "Concepts/Real.md"},
    ])
    assert {r.path for r in ws_backend.list_files()} == {"Concepts/Real.md"}


def test_search_names_matches_fs(ws_backend, fs):
    assert {r.path for r in ws_backend.search_names("cell")} == {r.path for r in fs.search_names("cell")}


def test_props_of_matches_fs(ws_backend, fs):
    assert ws_backend.props_of(NoteRef(name="Concepts", path="Hub/Concepts.md")) == \
        fs.props_of(NoteRef(name="Concepts", path="Hub/Concepts.md"))


def test_outline_matches_fs(ws_backend, fs):
    ref = NoteRef(name="Concepts", path="Hub/Concepts.md")
    assert [(h.level, h.text) for h in ws_backend.outline(ref)] == \
        [(h.level, h.text) for h in fs.outline(ref)]


def test_search_context_matches_fs(ws_backend, fs):
    ws_hits = {(h.ref.path, h.line, h.snippet) for h in ws_backend.search_context("hub")}
    fs_hits = {(h.ref.path, h.line, h.snippet) for h in fs.search_context("hub")}
    assert ws_hits == fs_hits


def test_search_context_batch_matches_fs(ws_backend, fs):
    ws_res = ws_backend.search_context_batch(["hub", "gradient"])
    fs_res = fs.search_context_batch(["hub", "gradient"])
    assert set(ws_res) == set(fs_res)
    for q in ws_res:
        assert {(h.ref.path, h.line) for h in ws_res[q]} == {(h.ref.path, h.line) for h in fs_res[q]}


# ---------------------------------------------------------------------------
# Graph path — _ensure_graph via resolved_links + mention_index, parity w/ FS.
# ---------------------------------------------------------------------------

_HUB = NoteRef(name="Concepts", path="Hub/Concepts.md")


def test_links_match_fs(ws_backend, fs):
    assert {r.name.lower() for r in ws_backend.links(_HUB)} == \
        {r.name.lower() for r in fs.links(_HUB)}


def test_backlinks_match_fs(ws_backend, fs):
    ref = NoteRef(name="Backpropagation", path="Concepts/Backpropagation.md")
    assert {r.path for r in ws_backend.backlinks(ref)} == {r.path for r in fs.backlinks(ref)}


def test_orphans_match_fs(ws_backend, fs):
    assert {r.path for r in ws_backend.orphans()} == {r.path for r in fs.orphans()}


def test_unresolved_match_fs(ws_backend, fs):
    assert {lnk.target.lower() for lnk in ws_backend.unresolved()} == \
        {lnk.target.lower() for lnk in fs.unresolved()}


def test_graph_snapshot_matches_fs(ws_backend, fs):
    ws_snap = ws_backend.graph_snapshot()
    fs_snap = fs.graph_snapshot()
    assert ws_snap.link_counts == fs_snap.link_counts
    assert ws_snap.backlink_counts == fs_snap.backlink_counts
    assert {r.path for r in ws_snap.orphans} == {r.path for r in fs_snap.orphans}


# ---------------------------------------------------------------------------
# Write path — each write is one RPC; the reply IS the settle (PROTOCOL §2.4).
# Function-scoped stub over a private vault copy: writes never touch the
# module-scoped read fixtures above.
# ---------------------------------------------------------------------------

@pytest.fixture
def write_stub(synthetic_vault, tmp_path):
    vault = tmp_path / "vault"
    shutil.copytree(synthetic_vault, vault)
    srv = StubPluginServer(vault)
    yield srv
    srv.stop()


@pytest.fixture
def wbe(write_stub):
    from silica.driver.ws_backend import ObsidianWSBackend

    be = ObsidianWSBackend(url=write_stub.url, token=write_stub.token)
    yield be
    be.close()


def test_create_lands_verbatim_and_returns_ref(wbe, write_stub):
    body = "# New\n\nFresh note body.\n"
    ref = wbe.create("Concepts/NewNote.md", body)
    assert (ref.name, ref.path) == ("NewNote", "Concepts/NewNote.md")
    on_disk = (write_stub._fs.vault_path / "Concepts/NewNote.md").read_text(encoding="utf-8")
    assert on_disk == body
    assert wbe.read_note(ref).content == body


def test_create_on_existing_path_raises(wbe):
    # match= keeps NotImplementedError (a RuntimeError subclass) from passing this
    with pytest.raises(RuntimeError, match="already exists"):
        wbe.create("Hub/Concepts.md", "clobber")


@pytest.mark.parametrize("body", [
    pytest.param("line one\r\nline two\r\n", id="crlf"),
    pytest.param("\\begin{align}\n\\nabla f = \\sum_i x_i\n\\end{align}\n", id="latex"),
    # No 30 KB special case exists: JSON framing carries any size in one write.
    pytest.param("x" * 1_000_000, id="1mb"),
])
def test_create_round_trips_hostile_bodies_verbatim(wbe, body):
    ref = wbe.create("Stress/Body.md", body)
    assert wbe.read_note(ref).content == body


def test_overwrite_replaces_content_in_place(wbe):
    new = "# Gradient\n\nRewritten body.\n"
    ref = wbe.overwrite("Concepts/Gradient.md", new)
    assert ref.path == "Concepts/Gradient.md"
    assert wbe.read_note(ref).content == new


def test_overwrite_of_missing_note_raises(wbe):
    with pytest.raises(RuntimeError, match="non-existent"):
        wbe.overwrite("Nope/Missing.md", "x")


def test_append_adds_fragment_and_patches_graph(wbe):
    empty = NoteRef(name="Empty", path="Lean/Empty.md")
    perceptron = NoteRef(name="Perceptron", path="Concepts/Perceptron.md")
    before_backlinks = {r.path for r in wbe.backlinks(perceptron)}  # primes the graph
    before = wbe.read_note(empty).content
    wbe.append(empty, "\nSee [[Perceptron]].\n")
    assert wbe.read_note(empty).content == before + "\nSee [[Perceptron]].\n"
    assert {r.path for r in wbe.backlinks(perceptron)} == before_backlinks | {"Lean/Empty.md"}


def test_set_prop_visible_via_props_of(wbe):
    ref = NoteRef(name="Gradient", path="Concepts/Gradient.md")
    wbe.set_prop(ref, "status", "reviewed")
    assert wbe.props_of(ref)["status"] == "reviewed"


def test_create_patches_graph_for_name_and_path_links(wbe):
    hub = NoteRef(name="Concepts", path="Hub/Concepts.md")
    gradient = NoteRef(name="Gradient", path="Concepts/Gradient.md")
    hub_before = {r.path for r in wbe.backlinks(hub)}  # primes the graph
    gradient_before = {r.path for r in wbe.backlinks(gradient)}
    ref = wbe.create("Concepts/Fresh.md", "Path link [[Hub/Concepts]] and name link [[Gradient]].\n")
    assert {r.path for r in wbe.backlinks(hub)} == hub_before | {"Concepts/Fresh.md"}
    assert {r.path for r in wbe.backlinks(gradient)} == gradient_before | {"Concepts/Fresh.md"}
    assert {r.path for r in wbe.links(ref)} == {"Hub/Concepts.md", "Concepts/Gradient.md"}


def test_move_rewrites_referrers_and_reindexes(wbe):
    src = NoteRef(name="Gradient", path="Concepts/Gradient.md")
    referrers = {"Hub/Concepts.md", "Concepts/Backpropagation.md"}
    assert {r.path for r in wbe.backlinks(src)} == referrers  # primes the graph
    original = wbe.read_note(src).content
    wbe.move(src, "Archive/Slope.md")
    dest = NoteRef(name="Slope", path="Archive/Slope.md")
    assert wbe.read_note(dest).content == original
    with pytest.raises(RuntimeError):
        wbe.read_note(NoteRef(name="Gradient", path="Concepts/Gradient.md"))
    assert wbe.backlinks(src) == []
    assert {r.path for r in wbe.backlinks(dest)} == referrers


def test_delete_removes_note_from_vault_and_graph(wbe):
    stub_note = NoteRef(name="Stub", path="Lean/Stub.md")
    hub = NoteRef(name="Concepts", path="Hub/Concepts.md")
    assert "Lean/Stub.md" in {r.path for r in wbe.backlinks(hub)}  # primes the graph
    wbe.delete(stub_note)
    assert "Lean/Stub.md" not in {r.path for r in wbe.backlinks(hub)}
    with pytest.raises(RuntimeError):
        wbe.read_note(stub_note)


def test_delete_of_missing_note_raises(wbe):
    with pytest.raises(RuntimeError, match="not found"):
        wbe.delete(NoteRef(name="Ghost", path="Nope/Ghost.md"))


def test_autolink_note_wraps_unlinked_mentions(wbe):
    ref = wbe.create("Notes/Mention.md", "This note discusses Perceptron at length.\n")
    added = wbe.autolink_note("Notes/Mention.md")
    assert added == ["Perceptron"]
    assert "[[Perceptron]]" in wbe.read_note(ref).content


def test_rpc_timeout_drops_pending_entry(wbe):
    """A timed-out RPC must not strand its future in _pending (leak on timeout)."""
    wbe._ensure_connected()
    wbe._timeout = 0.2
    with pytest.raises(Exception):
        wbe._rpc("__noreply__")
    assert wbe._pending == {}


def test_snapshot_restore_round_trips_prior_content(wbe):
    ref = NoteRef(name="Gradient", path="Concepts/Gradient.md")
    original = wbe.read_note(ref).content
    txn = wbe.snapshot_versions([ref])
    assert txn.id
    wbe.overwrite("Concepts/Gradient.md", "clobbered\n")
    created = wbe.create("Tmp/Scratch.md", "scratch\n")
    txn.created_paths.append(created.path)  # the caller (build_txn) records creations
    wbe.restore(txn)
    assert wbe.read_note(ref).content == original
    with pytest.raises(RuntimeError):
        wbe.read_note(created)
