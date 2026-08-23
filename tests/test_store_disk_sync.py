"""Two Silica processes on one index: the singleton stores stay honest.

`path_keyed_singleton` hands every caller in a process ONE store, and that
store used to read its file exactly once. So `silica nucleate` in a terminal
while the GUI or the MCP server was up left the long-lived process blind to
the new vectors until restart, and the long-lived process's next save() then
wrote its stale memory over the terminal's work. Last writer won, silently.

A second store instance on the same path stands in for the other process
here: it shares nothing in memory with the singleton, only the file.

Three stores, one contract (kernel/recall/paths.py `DiskSynced`), so every
test that can be shape-agnostic runs against all three.
"""
from __future__ import annotations

import pytest

from silica.kernel.recall import cooccurrence as cooc
from silica.kernel.recall import embed, lexical, paths


# --- per-store adapters: (module, accessor, constructor, put, value, drop) ---

def _embed_put(s, key, v):
    s.upsert(key, key, [v, 0.0, 0.0], content_hash=f"h{v}")


def _embed_value(s, key):
    vec = s.get_vec(key)
    return None if vec is None else vec[0]


def _cooc_put(s, key, v):
    s.upsert_note(key, {"nodes": {f"stem{int(v)}": {"count": 1}}, "edges": {}})


def _cooc_value(s, key):
    nodes = s.note_nodes(key)
    return None if not nodes else float(next(iter(nodes)).removeprefix("stem"))


def _lex_put(s, key, v):
    s.upsert(key, f"name{int(v)}", "body")


def _lex_value(s, key):
    name = s._name.get(key)
    return None if name is None else float(name.removeprefix("name"))


STORES = [
    pytest.param(
        embed, "get_store", lambda p: embed.EmbedStore(p),
        _embed_put, _embed_value, lambda s, k: s.delete(k), id="embed",
    ),
    pytest.param(
        cooc, "get_cooccur_store", lambda p: cooc.CooccurStore(p),
        _cooc_put, _cooc_value, lambda s, k: s.delete_note(k), id="cooccur",
    ),
    pytest.param(
        lexical, "get_lexical_store", lambda p: lexical.LexicalStore.load(p),
        _lex_put, _lex_value, lambda s, k: s.remove(k), id="lexical",
    ),
]
PARAMS = "module,accessor,make,put,value,drop"


@pytest.fixture
def index_path(tmp_path, monkeypatch):
    """One index file per test; every store module keyed to it."""
    p = tmp_path / "index" / "store.bin"
    for m in (embed, cooc, lexical):
        monkeypatch.setattr(m, "_index_path", lambda p=p: p)
        m.clear()
    yield p
    for m in (embed, cooc, lexical):
        m.clear()


@pytest.mark.parametrize(PARAMS, STORES)
def test_other_process_write_is_visible_through_the_accessor(
    module, accessor, make, put, value, drop, index_path
):
    """The staleness half: the GUI must see what the terminal just indexed."""
    mine = getattr(module, accessor)()
    assert value(mine, "X") is None

    other = make(index_path)
    put(other, "X", 1.0)
    other.save()

    assert value(getattr(module, accessor)(), "X") == 1.0


@pytest.mark.parametrize(PARAMS, STORES)
def test_a_store_holding_nothing_new_says_so(
    module, accessor, make, put, value, drop, index_path
):
    """is_dirty() is what lets a caller skip a save that would write the same
    bytes. It is not an optimisation: a rewrite moves the file's mtime, and
    `vault_version()` digests exactly those mtimes to tell the GUI that its
    cached views are stale. The co-occurrence refresh every graph export runs
    made the strip say "vault changed" after every build, and rewrote megabytes
    to say it."""
    s = getattr(module, accessor)()
    assert s.is_dirty() is False          # freshly loaded: nothing of ours
    put(s, "X", 1.0)
    assert s.is_dirty() is True
    s.save()
    assert s.is_dirty() is False


@pytest.mark.parametrize(PARAMS, STORES)
def test_save_merges_instead_of_clobbering(
    module, accessor, make, put, value, drop, index_path
):
    """The loss half: my save must carry the other process's entries forward."""
    mine = getattr(module, accessor)()
    put(mine, "Y", 2.0)            # unsaved, as during a nucleate run

    other = make(index_path)
    put(other, "X", 1.0)
    other.save()

    mine.save()

    fresh = make(index_path)        # a third process reading what landed
    assert value(fresh, "X") == 1.0, "the other process's entry was clobbered"
    assert value(fresh, "Y") == 2.0


@pytest.mark.parametrize(PARAMS, STORES)
def test_unsaved_local_entry_wins_over_disk(
    module, accessor, make, put, value, drop, index_path
):
    """Same path in both: mine is the newer embedding of a note I just wrote."""
    mine = getattr(module, accessor)()
    put(mine, "X", 2.0)

    other = make(index_path)
    put(other, "X", 1.0)
    other.save()

    assert value(getattr(module, accessor)(), "X") == 2.0
    mine.save()
    assert value(make(index_path), "X") == 2.0


@pytest.mark.parametrize(PARAMS, STORES)
def test_unsaved_local_delete_survives_a_reload(
    module, accessor, make, put, value, drop, index_path
):
    """`_drop_embed_vector` promises this process never sees the phantom again;
    a reload must not resurrect it from the other process's file."""
    mine = getattr(module, accessor)()
    put(mine, "X", 1.0)
    mine.save()
    drop(mine, "X")                 # unsaved (the fs backend buffers these)

    other = make(index_path)        # still has X
    put(other, "Z", 3.0)
    other.save()

    synced = getattr(module, accessor)()
    assert value(synced, "X") is None
    assert value(synced, "Z") == 3.0
    mine.save()
    assert value(make(index_path), "X") is None


@pytest.mark.parametrize(PARAMS, STORES)
def test_own_save_does_not_reload(
    module, accessor, make, put, value, drop, index_path, monkeypatch
):
    """A save records the stamp of the file it wrote: the next access must not
    pay a deserialize for our own write."""
    mine = getattr(module, accessor)()
    put(mine, "X", 1.0)
    mine.save()
    monkeypatch.setattr(mine, "_read_disk", lambda: pytest.fail("reloaded own write"))

    assert mine.sync_from_disk() is False
    getattr(module, accessor)()


@pytest.mark.parametrize(PARAMS, STORES)
def test_failed_write_keeps_entries_dirty(
    module, accessor, make, put, value, drop, index_path, monkeypatch
):
    """A write that dies must leave the entries marked unsaved, or the next
    external change would reload over them and they would be gone for good."""
    mine = getattr(module, accessor)()
    put(mine, "Y", 2.0)

    def _boom(path, data):
        raise OSError("disk full")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paths, "atomic_write_bytes", _boom)
        with pytest.raises(OSError):
            mine.save()

    other = make(index_path)
    put(other, "X", 1.0)
    other.save()

    assert value(getattr(module, accessor)(), "Y") == 2.0


@pytest.mark.parametrize(PARAMS, STORES)
def test_mutation_during_the_write_stays_dirty(
    module, accessor, make, put, value, drop, index_path, monkeypatch
):
    """The serialize runs outside the store lock (it is the slow part), so a
    sibling thread can upsert while it runs. That entry is not in the bytes
    being written and must still be dirty afterwards."""
    mine = getattr(module, accessor)()
    put(mine, "X", 1.0)
    original = mine._serialize

    def _serialize_and_race(snapshot):
        put(mine, "Y", 2.0)          # lands after the snapshot was taken
        return original(snapshot)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mine, "_serialize", _serialize_and_race)
        mine.save()

    assert value(make(index_path), "Y") is None, "premise: Y missed the write"
    other = make(index_path)
    put(other, "Z", 3.0)
    other.save()

    assert value(getattr(module, accessor)(), "Y") == 2.0, "Y was dropped on reload"


@pytest.mark.parametrize(PARAMS, STORES)
def test_stamp_is_taken_before_the_read(
    module, accessor, make, put, value, drop, index_path, monkeypatch
):
    """A write landing between our stat and our read must be seen by the NEXT
    sync: the stamp we keep has to be the pre-read one. Stamping after the
    read would record a file we never loaded and hide that write until the
    one after it."""
    mine = getattr(module, accessor)()
    other = make(index_path)
    put(other, "X", 1.0)
    other.save()
    original = mine._read_disk

    def _read_then_race():
        state = original()
        put(other, "Z", 3.0)         # another process writes mid-read
        other.save()
        return state

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mine, "_read_disk", _read_then_race)
        assert mine.sync_from_disk() is True
    assert value(mine, "Z") is None, "premise: the mid-read write was not loaded"

    assert mine.sync_from_disk() is True, "the mid-read write was hidden"
    assert value(mine, "Z") == 3.0



def test_sweep_save_does_not_clobber_the_other_process(index_path, monkeypatch):
    """The caller that bit: the GUI's index sweep prunes and saves while a
    terminal nucleate has just written. Through the real sweep, not the store."""
    from types import SimpleNamespace

    from silica.driver.base import NoteRef
    from silica.kernel.recall import sync

    monkeypatch.setattr("silica.config.CONFIG.index_sweep", True)
    monkeypatch.setattr(sync, "_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(sync, "_stamps_path", lambda: index_path.parent / "stamps.json")
    mine = embed.get_store()
    for k in ("A", "B", "C"):
        _embed_put(mine, k, 1.0)
    mine.save()

    other = embed.EmbedStore(index_path)
    _embed_put(other, "N", 5.0)         # the terminal's new note
    other.save()

    # The GUI's vault view: B was deleted out-of-band, N exists on disk too.
    live = ["A", "C", "N"]
    monkeypatch.setattr("silica.driver.DRIVER", SimpleNamespace(
        list_files=lambda folder="": [NoteRef(p, p + ".md") for p in live],
        read_note=lambda ref: SimpleNamespace(content="body"),
        mtime_of=lambda ref: 1.0,
    ))
    monkeypatch.setattr("silica.agent.providers.get_embedder",
                        lambda cfg: SimpleNamespace(model="fake", embed=lambda t: [[1.0, 0, 0]] * len(t)))
    stats = sync.sweep(force=True)

    assert stats.pruned == 1
    on_disk = embed.EmbedStore(index_path)
    assert on_disk.has("N"), "the sweep's save clobbered the terminal's vector"
    assert not on_disk.has("B")


def test_contended_saves_across_real_processes(tmp_path):
    """Three OS processes, 40 read-merge-write saves each on one index file.

    The in-process tests above prove the merge; only real processes prove the
    lock, and without it two saves serialising at the same moment lose one
    process's entries for the whole window. Every entry must land."""
    import subprocess
    import sys

    idx = tmp_path / "index" / "embed.npz"
    worker = f"""
import sys
from pathlib import Path
from silica.kernel.recall.embed import EmbedStore
s = EmbedStore(Path({str(idx)!r}))
tag = sys.argv[1]
for i in range(40):
    s.upsert(f"{{tag}}{{i}}", f"{{tag}}{{i}}", [float(i), 0.0, 0.0], content_hash="h")
    s.save()
"""
    procs = [subprocess.Popen([sys.executable, "-c", worker, tag]) for tag in "abc"]
    assert [p.wait(timeout=60) for p in procs] == [0, 0, 0]

    assert len(embed.EmbedStore(idx)) == 120
