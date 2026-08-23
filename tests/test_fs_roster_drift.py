"""Out-of-band roster drift: notes created/deleted while the process is alive.

The body cache is mtime-keyed (`_read_cached`) and the derived indexes
re-embed by content signature (kernel/recall/sync.py), so an EDIT made in
Obsidian was already picked up. A CREATE or a DELETE was not: `self._notes`
is built once and only ever patched by Silica's own writes, so a long-lived
process (REPL, `silica gui`, MCP server) stayed blind to it — and since
`sync.sweep` enumerates the vault through `DRIVER.list_files("")`, the whole
freshness chain inherited that blindness.

The signal is the directory mtime: POSIX bumps it on create/delete/rename
inside the folder, never on an in-place edit. That asymmetry is what these
tests pin, in both directions — the drift must be seen, and an edit must NOT
pay for a rebuild.
"""
from __future__ import annotations

import pytest

from silica.driver import fs_backend
from silica.driver.fs_backend import ObsidianFSBackend


@pytest.fixture
def vault_dir(tmp_path):
    (tmp_path / "A.md").write_text("alpha body\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "C.md").write_text("charlie body\n")
    return tmp_path


@pytest.fixture
def backend(vault_dir, monkeypatch):
    """Backend with the roster re-check debounce off (its own test covers it)."""
    monkeypatch.setattr(fs_backend, "_ROSTER_RECHECK_INTERVAL", 0.0)
    b = ObsidianFSBackend(vault_path=str(vault_dir))
    b._ensure_index()  # first build, so later drift is drift and not cold start
    return b


def _rebuild_counter(backend, monkeypatch):
    """Count full rebuilds without suppressing them."""
    calls: list[int] = []
    original = backend._rebuild_index

    def counting():
        calls.append(1)
        return original()

    monkeypatch.setattr(backend, "_rebuild_index", counting)
    return calls


def _paths(backend):
    return sorted(r.path for r in backend.list_files(""))


def test_out_of_band_create_enters_the_roster(backend, vault_dir):
    """A note created in Obsidian mid-session must become visible."""
    assert _paths(backend) == ["A.md", "sub/C.md"]

    (vault_dir / "B.md").write_text("beta body\n")

    assert _paths(backend) == ["A.md", "B.md", "sub/C.md"]
    # And to the grep lane, which iterates the same roster.
    assert [h.ref.path for h in backend.search_context("beta")] == ["B.md"]


def test_out_of_band_create_in_subfolder_enters_the_roster(backend, vault_dir):
    """Notes live in folders: the drift signal has to reach past the root."""
    (vault_dir / "sub" / "D.md").write_text("delta body\n")

    assert "sub/D.md" in _paths(backend)


def test_out_of_band_delete_leaves_the_roster(backend, vault_dir):
    """A phantom entry is worse than a missing one: `read_note` raises on it,
    and `sync._prune_orphans` keeps its vectors alive because the roster still
    reports the note as live."""
    (vault_dir / "A.md").unlink()

    assert _paths(backend) == ["sub/C.md"]


def test_in_place_edit_does_not_rebuild(backend, vault_dir, monkeypatch):
    """The hot-path guard. `_ensure_index` runs on every read op, so an edit
    (already covered by the mtime-keyed body cache) must not drag the whole
    vault through a rebuild."""
    calls = _rebuild_counter(backend, monkeypatch)

    (vault_dir / "A.md").write_text("alpha body EDITED gamma\n")

    assert backend.read_note("A").content.endswith("gamma\n")  # fresh anyway
    assert [h.ref.path for h in backend.search_context("gamma")] == ["A.md"]
    assert calls == [], "an in-place edit must not trigger a full rebuild"


def test_churn_in_an_ignored_tree_does_not_rebuild(vault_dir, monkeypatch):
    """A vault adopted as-is can be a repo root, where `.git/` alone rewrites
    directory mtimes on every command. `_rebuild_index` never walks those
    trees, so the drift check must not either — otherwise the roster sits in a
    permanent rebuild loop over files it does not even index."""
    monkeypatch.setattr(fs_backend, "_ROSTER_RECHECK_INTERVAL", 0.0)
    (vault_dir / ".git").mkdir()
    (vault_dir / "node_modules").mkdir()
    b = ObsidianFSBackend(vault_path=str(vault_dir))
    b._ensure_index()
    calls = _rebuild_counter(b, monkeypatch)

    (vault_dir / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (vault_dir / "node_modules" / "README.md").write_text("vendored\n")

    assert _paths(b) == ["A.md", "sub/C.md"]
    assert calls == [], "hidden and vendored trees must not drive the roster"


def test_silica_own_write_does_not_rebuild(backend, monkeypatch):
    """`_patch_index` exists to keep Silica's own writes off the rebuild path.
    Its write bumps the folder mtime like any other, so the stamps have to be
    carried forward or every write would undo the optimisation."""
    calls = _rebuild_counter(backend, monkeypatch)

    backend.create("E.md", "echo body\n")

    assert "E.md" in _paths(backend)
    assert calls == [], "a backend write must not trigger a full rebuild"


def test_recheck_is_debounced(vault_dir, monkeypatch):
    """Several reads land in one agent turn; a roster that just rescanned has
    nothing new to see, and the scan is a whole-tree walk."""
    monkeypatch.setattr(fs_backend, "_ROSTER_RECHECK_INTERVAL", 3600.0)
    b = ObsidianFSBackend(vault_path=str(vault_dir))
    b._ensure_index()

    (vault_dir / "B.md").write_text("beta body\n")

    assert _paths(b) == ["A.md", "sub/C.md"], "within the interval: no rescan"


def test_out_of_band_create_reaches_the_embed_index(vault_dir, monkeypatch):
    """The chain the drift check exists for, end to end.

    `sync.sweep` enumerates the vault through `DRIVER.list_files("")`, so a
    note the roster misses is one no derived index can ever hold — and
    nucleate's dedup (states/collision.py reads `get_store()` directly) would
    then write it a second time as new. test_index_sync.py cannot catch this:
    it stubs the driver out, which is exactly the layer that was lying.
    """
    import silica.agent.providers as providers
    import silica.driver as driver_mod
    from silica.kernel.recall import sync
    from silica.kernel.recall.embed import build_index, get_store

    class _Emb:
        model = "fake-model"

        def embed(self, texts):
            return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    monkeypatch.setattr(fs_backend, "_ROSTER_RECHECK_INTERVAL", 0.0)
    monkeypatch.setattr(sync, "_MIN_INTERVAL", 0.0)
    monkeypatch.setattr("silica.config.CONFIG.index_sweep", True)
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(vault_dir))
    monkeypatch.setattr(providers, "get_embedder", lambda cfg: _Emb())
    driver_mod.set_driver(ObsidianFSBackend(vault_path=str(vault_dir)))

    store = get_store()
    build_index(
        _Emb(),
        [("A", "A", (vault_dir / "A.md").read_text()),
         ("sub/C", "C", (vault_dir / "sub" / "C.md").read_text())],
        store=store,
    )
    # Seed BOTH notes so the baseline sweep is a genuine no-op: the `embedded`
    # count below then means the new note and nothing else.
    assert sync.sweep(force=True).embedded == 0
    assert sorted(store.paths()) == ["A", "sub/C"]

    (vault_dir / "B.md").write_text("beta body about welding\n")

    assert sync.sweep(force=True).embedded == 1
    assert sorted(store.paths()) == ["A", "B", "sub/C"]
