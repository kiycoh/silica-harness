"""ADR-0019: personal memory as a second recall lane.

The memory vault's (embed, cooccur) stores join the same RRF fusion as the
active vault's legs, read-only, with visible provenance (origin="memory",
`memory:`-prefixed evidence). Degenerate case: active vault == memory vault
⇒ the lane abstains and fusion is bit-identical to single-vault.
"""
from __future__ import annotations

from silica.config import CONFIG
from silica.kernel.recall import memory_lane
from silica.kernel.recall.embed import EmbedStore
from silica.kernel.recall.relatedness import related_notes, related_notes_for_query


def _store(path, notes) -> EmbedStore:
    es = EmbedStore(path=path)
    for p, name, vec in notes:
        es.upsert(p, name, vec)
    return es


# ---------------------------------------------------------------------------
# Lane resolution (memory_lane.py)
# ---------------------------------------------------------------------------

def test_memory_vault_abstains_when_same_as_active(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(CONFIG, "memory_vault", str(vault))
    assert memory_lane.memory_vault() is None
    assert memory_lane.memory_stores() == (None, None)


def test_memory_vault_resolves_when_distinct(tmp_path, monkeypatch):
    active = tmp_path / "active"
    mem = tmp_path / "mem"
    active.mkdir()
    mem.mkdir()
    monkeypatch.setattr(CONFIG, "vault_path", str(active))
    monkeypatch.setattr(CONFIG, "memory_vault", str(mem))
    assert memory_lane.memory_vault() == mem.resolve()


def test_memory_vault_abstains_when_missing_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(CONFIG, "memory_vault", str(tmp_path / "nope"))
    assert memory_lane.memory_vault() is None


def test_memory_stores_abstain_without_indexes(tmp_path, monkeypatch):
    active = tmp_path / "active"
    mem = tmp_path / "mem"
    active.mkdir()
    mem.mkdir()
    monkeypatch.setattr(CONFIG, "vault_path", str(active))
    monkeypatch.setattr(CONFIG, "memory_vault", str(mem))
    memory_lane.clear()
    assert memory_lane.memory_stores() == (None, None)


# ---------------------------------------------------------------------------
# Fusion with the memory legs (relatedness facade)
# ---------------------------------------------------------------------------

def test_fresh_query_fuses_memory_leg_with_provenance(tmp_path):
    active = _store(tmp_path / "a.json", [("A", "A note", [1.0, 0.0])])
    memory = _store(tmp_path / "m.json", [("Spec X", "Spec X", [0.95, 0.05])])
    out = related_notes_for_query(
        query_vec=[1.0, 0.0],
        embed_store=active,
        memory_embed_store=memory,
        k=5,
    )
    origins = {r.path: r.origin for r in out}
    assert origins == {"A": "vault", "Spec X": "memory"}
    mem_hit = next(r for r in out if r.origin == "memory")
    assert mem_hit.evidence and all(e.startswith("memory:") for e in mem_hit.evidence)


def test_same_relative_path_in_both_lanes_stays_two_results(tmp_path):
    # Identical rel path in the two vaults = two DIFFERENT notes; the key
    # namespace must keep them apart instead of collapsing/duplicating RRF terms.
    active = _store(tmp_path / "a.json", [("N", "N", [1.0, 0.0])])
    memory = _store(tmp_path / "m.json", [("N", "N", [1.0, 0.0])])
    out = related_notes_for_query(
        query_vec=[1.0, 0.0], embed_store=active, memory_embed_store=memory, k=5
    )
    assert len(out) == 2
    assert {r.origin for r in out} == {"vault", "memory"}


def test_indexed_note_query_reaches_memory_lane(tmp_path):
    active = _store(
        tmp_path / "a.json", [("A", "A", [1.0, 0.0]), ("B", "B", [0.9, 0.1])]
    )
    memory = _store(tmp_path / "m.json", [("M", "M", [0.95, 0.05])])
    out = related_notes("A", embed_store=active, memory_embed_store=memory, k=5)
    assert {r.path for r in out} == {"B", "M"}
    assert next(r for r in out if r.path == "M").origin == "memory"
    assert next(r for r in out if r.path == "B").origin == "vault"


def test_no_memory_stores_is_bit_identical_to_single_vault(tmp_path):
    active = _store(
        tmp_path / "a.json", [("A", "A", [1.0, 0.0]), ("B", "B", [0.9, 0.1])]
    )
    baseline = related_notes_for_query(query_vec=[1.0, 0.0], embed_store=active, k=5)
    with_lane = related_notes_for_query(
        query_vec=[1.0, 0.0],
        embed_store=active,
        memory_embed_store=None,
        memory_cooccur_store=None,
        k=5,
    )
    assert baseline == with_lane


def test_memory_artifacts_filtered_like_vault_ones(tmp_path):
    memory = _store(
        tmp_path / "m.json",
        [("GRAPH_REPORT", "Graph Report", [1.0, 0.0]), ("real", "Real", [0.9, 0.1])],
    )
    out = related_notes_for_query(query_vec=[1.0, 0.0], memory_embed_store=memory, k=5)
    assert [r.path for r in out] == ["real"]


def test_foreign_store_cache_is_bounded_and_evicts_the_least_recently_used(tmp_path, monkeypatch):
    # silica_vaults loads one embed store per known vault and a peek adds its
    # co-occurrence store: unbounded, a process serving one vault ends up
    # holding every vault on the machine (108 MB for the repo vault alone).
    from silica.kernel.recall.paths import index_dir_for

    memory_lane.clear()
    vaults = []
    for i in range(memory_lane._CACHE_MAX + 1):
        v = tmp_path / f"v{i}"
        v.mkdir()
        d = index_dir_for(str(v))
        d.mkdir(parents=True, exist_ok=True)
        _store(d / "embeddings.json", [("N", "N", [1.0, 0.0])]).save()
        vaults.append(v)
    for v in vaults:
        assert memory_lane.stores_for(v)[0] is not None
    assert len(memory_lane._CACHE) == memory_lane._CACHE_MAX
    assert str(index_dir_for(str(vaults[0]))) not in memory_lane._CACHE  # the first one went
    memory_lane.stores_for(vaults[1])  # touch: v1 becomes the most recent
    memory_lane.stores_for(vaults[0])  # re-read from disk, evicts v2, not v1
    assert str(index_dir_for(str(vaults[1]))) in memory_lane._CACHE
    assert str(index_dir_for(str(vaults[2]))) not in memory_lane._CACHE
