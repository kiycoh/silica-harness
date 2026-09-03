"""Phase 3 tests — embedding store, cosine similarity, and semantic search tools."""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from silica.kernel.recall.embed import (
    EmbedStore,
    _cosine,
    _note_text,
    _note_title_text,
    build_index,
    refresh_note,
)


# ---------------------------------------------------------------------------
# _cosine — pure maths
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine(a, b) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine(a, b) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine([1.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_length_mismatch_returns_zero():
    assert _cosine([1.0], [1.0, 2.0]) == 0.0


def test_cosine_normalised_vectors():
    # normalised: dot == cosine
    a = [3.0 / 5, 4.0 / 5]
    b = [4.0 / 5, 3.0 / 5]
    expected = (3 * 4 + 4 * 3) / 25
    assert _cosine(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _note_text — truncation
# ---------------------------------------------------------------------------

def test_note_text_truncates_at_1200():
    long_body = "x" * 2000
    result = _note_text("Title", long_body)
    assert len(result) <= 1200


def test_note_text_includes_title():
    result = _note_text("Neural Networks", "Deep learning intro")
    assert "Neural Networks" in result


def test_note_text_with_folder_prefix():
    result = _note_text("CAN", "Controller Area Network bus", folder="Robotica")
    assert result.startswith("[Robotica] CAN")


def test_note_text_no_folder_is_backward_compatible():
    result = _note_text("CAN", "Controller Area Network bus")
    assert result.startswith("CAN"), "No folder → no prefix, backward-compatible"


def test_note_text_folder_respects_max_chars():
    result = _note_text("T", "x" * 2000, folder="Robotica")
    assert len(result) <= 1200


# ---------------------------------------------------------------------------
# EmbedStore — in-memory and persistence
# ---------------------------------------------------------------------------

def test_embed_store_empty_on_missing_file(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    assert len(store) == 0
    assert store.paths() == []


def test_embed_store_upsert_and_get(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("Concepts/NeuralNet", "Neural Net", [1.0, 0.0, 0.0])
    assert store.has("Concepts/NeuralNet")
    assert store.get_vec("Concepts/NeuralNet") == [1.0, 0.0, 0.0]


def test_embed_store_roundtrip(tmp_path):
    idx = tmp_path / "embeddings.json"
    store = EmbedStore(path=idx)
    store.upsert("Concepts/A", "A", [1.0, 0.0])
    store.upsert("Concepts/B", "B", [0.0, 1.0])
    store.save()

    store2 = EmbedStore(path=idx)
    assert store2.has("Concepts/A")
    assert store2.has("Concepts/B")
    assert store2.get_vec("Concepts/A") == [1.0, 0.0]


def test_embed_store_delete(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("Concepts/A", "A", [1.0, 0.0])
    store.delete("Concepts/A")
    assert not store.has("Concepts/A")


def test_embed_store_len(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("a", "A", [1.0])
    store.upsert("b", "B", [0.5])
    assert len(store) == 2


# ---------------------------------------------------------------------------
# EmbedStore.cosine_top_k
# ---------------------------------------------------------------------------

def test_cosine_top_k_ordering(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("best",   "Best",   [1.0, 0.0])
    store.upsert("middle", "Middle", [0.8, 0.6])
    store.upsert("worst",  "Worst",  [0.0, 1.0])

    results = store.cosine_top_k([1.0, 0.0], k=3)
    assert [r["path"] for r in results] == ["best", "middle", "worst"]


def test_cosine_top_k_respects_k(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    for i in range(10):
        store.upsert(f"note_{i}", f"Note {i}", [float(i), 0.0])

    results = store.cosine_top_k([1.0, 0.0], k=3)
    assert len(results) == 3


def test_cosine_top_k_exclude(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("query_note", "Query", [1.0, 0.0])
    store.upsert("other",      "Other", [0.9, 0.0])

    results = store.cosine_top_k([1.0, 0.0], k=5, exclude={"query_note"})
    paths = [r["path"] for r in results]
    assert "query_note" not in paths
    assert "other" in paths


def test_build_matrix_uses_modal_dimension_not_first(tmp_path):
    # A10: the first-inserted note is a MINORITY dimension. The majority must not
    # be dropped, and a query in the majority dim must not zero the whole leg.
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("legacy", "Legacy", [0.1, 0.2, 0.3])  # minority dim=3, inserted first
    store.upsert("hit",    "Hit",    [1.0, 0.0])        # majority dim=2
    store.upsert("other",  "Other",  [0.6, 0.8])        # majority dim=2

    scores = {r["path"]: r["score"] for r in store.cosine_top_k([1.0, 0.0], k=3)}
    assert scores["hit"] > 0.9          # majority-dim note scored — leg not zeroed
    assert scores.get("other", 0.0) > 0.0


def test_cosine_top_k_pads_off_matrix_zero_rows_above_negatives(tmp_path):
    # B13: the top-k fast path skips the full-vault scan only when the matrix has
    # >= k strictly-positive hits. Here it does not, so an off-dimension note (0.0,
    # off-matrix) must still be placed above a negative in-matrix hit.
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("pos", "Pos", [1.0, 0.0])          # sim +1.0, modal dim 2
    store.upsert("neg", "Neg", [-1.0, 0.0])         # sim -1.0, modal dim 2
    store.upsert("off", "Off", [0.1, 0.2, 0.3])     # off-dim → 0.0, off-matrix

    results = store.cosine_top_k([1.0, 0.0], k=3)
    assert [r["path"] for r in results] == ["pos", "off", "neg"]
    assert results[1]["score"] == 0.0


def test_model_swap_reembeds_unchanged_notes(tmp_path):
    # A11: swapping the embedder model must re-embed content-unchanged notes,
    # or the store ends up mixed-dimension (the A10 hazard).
    idx = tmp_path / "idx.json"
    notes = [("Concepts/A", "A", "body a"), ("Concepts/B", "B", "body b")]
    build_index(_make_embedder(), notes, store=EmbedStore(idx))

    e2 = _make_embedder()
    e2.model = "other-model"
    e2.embed.reset_mock()
    build_index(e2, notes, store=EmbedStore(idx))
    assert e2.embed.call_count > 0  # model changed → stale despite unchanged bodies


def test_cosine_top_k_returns_score(tmp_path):
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("a", "A", [1.0, 0.0])
    results = store.cosine_top_k([1.0, 0.0], k=1)
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# build_index — incremental and force
# ---------------------------------------------------------------------------

def _make_embedder(dim: int = 2):
    """Return a mock embedder that produces deterministic unit vectors."""
    embedder = MagicMock()
    embedder.model = "test-model"  # real embedders expose a str .model (A11 signature)
    call_count = [0]

    def fake_embed(texts):
        vecs = []
        for _ in texts:
            angle = call_count[0] * 0.1
            vecs.append([math.cos(angle), math.sin(angle)])
            call_count[0] += 1
        return vecs

    embedder.embed.side_effect = fake_embed
    return embedder


def test_build_index_embeds_new_notes(tmp_path):
    embedder = _make_embedder()
    notes = [
        ("Concepts/A", "A", "body a"),
        ("Concepts/B", "B", "body b"),
    ]
    store = build_index(embedder, notes, store=EmbedStore(tmp_path / "idx.json"))
    assert store.has("Concepts/A")
    assert store.has("Concepts/B")
    assert embedder.embed.call_count == 1  # both in one batch


def test_build_index_prune_reconciles_out_of_band_deletion(tmp_path):
    # A note deleted from the vault (out-of-band, e.g. Obsidian) is absent from
    # the authoritative `notes`; prune=True drops its phantom vector, scoped to
    # `folder` so notes in other folders survive.
    idx = tmp_path / "idx.json"
    store = EmbedStore(idx)
    store.upsert("Concepts/Gone", "Gone", [1.0, 0.0])   # deleted since last build
    store.upsert("Other/Keep", "Keep", [0.0, 1.0])      # different folder

    build_index(_make_embedder(), [("Concepts/A", "A", "a")],
                store=store, prune=True, folder="Concepts")

    assert store.has("Concepts/A")           # new note embedded
    assert not store.has("Concepts/Gone")    # phantom pruned
    assert store.has("Other/Keep")           # out-of-folder untouched


def test_build_index_default_does_not_prune(tmp_path):
    # Guard: incremental callers pass a PARTIAL `notes` (only missing paths).
    # Without prune, unlisted entries must survive — else the reconcile path
    # (_reconcile_embed_index) would wipe every valid vector.
    idx = tmp_path / "idx.json"
    store = EmbedStore(idx)
    store.upsert("Concepts/Present", "Present", [1.0, 0.0])

    build_index(_make_embedder(), [("Concepts/New", "New", "n")], store=store)

    assert store.has("Concepts/Present")     # not listed, but kept
    assert store.has("Concepts/New")


def test_build_index_skips_existing(tmp_path):
    embedder = _make_embedder()
    idx = tmp_path / "idx.json"
    # Seed A through build_index so it carries a real content signature (an
    # unchanged note is one whose stored hash still matches its text).
    build_index(embedder, [("Concepts/A", "A", "body a")], store=EmbedStore(idx))
    embedder.embed.reset_mock()

    notes = [
        ("Concepts/A", "A", "body a"),  # already indexed, unchanged
        ("Concepts/B", "B", "body b"),  # new
    ]
    store2 = build_index(embedder, notes, store=EmbedStore(idx))
    assert store2.has("Concepts/A")
    assert store2.has("Concepts/B")
    # Only B should have been sent to the embedder — A is unchanged.
    embedded_texts = [
        t for call in embedder.embed.call_args_list for t in call.args[0]
    ]
    assert all("A" not in t for t in embedded_texts)


def test_build_index_force_reembeds_all(tmp_path):
    embedder = _make_embedder()
    idx = tmp_path / "idx.json"
    store = EmbedStore(idx)
    store.upsert("Concepts/A", "A", [1.0, 0.0])
    store.save()

    notes = [("Concepts/A", "A", "body a")]
    build_index(embedder, notes, store=EmbedStore(idx), force=True)
    assert embedder.embed.call_count == 1


# ---------------------------------------------------------------------------
# build_index — content-change detection (incremental refresh re-embeds edits)
# ---------------------------------------------------------------------------

def test_build_index_reembeds_changed_body(tmp_path):
    """A note whose embedded text changed must be re-embedded incrementally."""
    embedder = _make_embedder()
    idx = tmp_path / "idx.json"
    build_index(embedder, [("Concepts/A", "A", "body v1")], store=EmbedStore(idx))
    n = embedder.embed.call_count

    build_index(embedder, [("Concepts/A", "A", "body v2 — edited")], store=EmbedStore(idx))
    assert embedder.embed.call_count == n + 1  # changed → re-embedded, not skipped


def test_build_index_skips_unchanged_body_across_reload(tmp_path):
    """An unchanged note must be skipped even by a fresh store loaded from disk
    — the content signature has to survive serialize/deserialize."""
    embedder = _make_embedder()
    idx = tmp_path / "idx.json"
    build_index(embedder, [("Concepts/A", "A", "same body")], store=EmbedStore(idx))
    n = embedder.embed.call_count

    # Fresh EmbedStore(idx) reloads from disk; identical text → no re-embed.
    build_index(embedder, [("Concepts/A", "A", "same body")], store=EmbedStore(idx))
    assert embedder.embed.call_count == n


# ---------------------------------------------------------------------------
# title_vec — EmbedStore dual-vector support
# ---------------------------------------------------------------------------

def test_upsert_stores_title_vec(tmp_path):
    store = EmbedStore(path=tmp_path / "idx.json")
    store.upsert("Robotica/ROS", "ROS", [1.0, 0.0], title_vec=[0.9, 0.1])
    # float32 at upsert time, not only after a save/load roundtrip: a fresh
    # entry and a reloaded one now carry identical precision.
    assert store.get_title_vec("Robotica/ROS") == pytest.approx([0.9, 0.1], rel=1e-6)


def test_get_title_vec_missing_returns_none(tmp_path):
    """Old index entries without title_vec must not raise — they return None."""
    store = EmbedStore(path=tmp_path / "idx.json")
    store.upsert("Concepts/A", "A", [1.0, 0.0])  # no title_vec
    assert store.get_title_vec("Concepts/A") is None


def test_upsert_preserves_existing_title_vec_when_omitted(tmp_path):
    """Re-upserting with only vec (no title_vec) must not wipe an existing title_vec."""
    store = EmbedStore(path=tmp_path / "idx.json")
    store.upsert("Concepts/A", "A", [1.0, 0.0], title_vec=[0.5, 0.5])
    store.upsert("Concepts/A", "A", [0.8, 0.2])  # title_vec omitted
    assert store.get_title_vec("Concepts/A") == [0.5, 0.5]


def test_build_index_stores_title_vecs(tmp_path):
    """build_index must populate title_vec for every note via interleaved batch."""
    call_count = [0]

    def interleaved_embedder(texts):
        # Returns a distinct vector for each text based on call order
        vecs = []
        for _ in texts:
            i = call_count[0]
            # Even positions → full vec, odd → title vec (simplified: just use index)
            vecs.append([float(i), 0.0])
            call_count[0] += 1
        return vecs

    embedder_mock = MagicMock()
    embedder_mock.model = "test-model"  # real embedders expose a str .model (A11 signature)
    embedder_mock.embed.side_effect = interleaved_embedder

    notes = [("Robotica/ROS", "ROS", "Robot Operating System")]
    store = build_index(embedder_mock, notes, store=EmbedStore(tmp_path / "idx.json"))

    # build_index calls embed once with interleaved [full, title]
    assert embedder_mock.embed.call_count == 1
    texts_sent = embedder_mock.embed.call_args.args[0]
    assert len(texts_sent) == 2  # 1 note × 2 texts

    # Both vec and title_vec must be present
    assert store.get_vec("Robotica/ROS") is not None
    assert store.get_title_vec("Robotica/ROS") is not None
    # They must be distinct (full vec at index 0, title vec at index 1)
    assert store.get_vec("Robotica/ROS") != store.get_title_vec("Robotica/ROS")


def test_note_title_text_with_folder():
    result = _note_title_text("ROS", folder="Robotica")
    assert result == "[Robotica] ROS"


def test_note_title_text_no_folder():
    result = _note_title_text("ROS")
    assert result == "ROS"


# ---------------------------------------------------------------------------
# refresh_note
# ---------------------------------------------------------------------------

def test_refresh_note_updates_vec(tmp_path):
    embedder = _make_embedder()
    idx = tmp_path / "idx.json"

    store = EmbedStore(idx)
    store.upsert("Concepts/A", "A", [0.0, 1.0])
    store.save()

    refresh_note(embedder, "Concepts/A", "A", "new body", store=EmbedStore(idx))

    store2 = EmbedStore(idx)
    # The vector should have been updated (not the old one)
    assert store2.get_vec("Concepts/A") != [0.0, 1.0]


def test_refresh_note_then_build_index_skips_it(tmp_path):
    """A note freshened via refresh_note carries a signature, so a later
    incremental build_index with the same body does not re-embed it."""
    embedder = _make_embedder()
    idx = tmp_path / "idx.json"
    refresh_note(embedder, "Concepts/A", "A", "the body", store=EmbedStore(idx))
    n = embedder.embed.call_count

    build_index(embedder, [("Concepts/A", "A", "the body")], store=EmbedStore(idx))
    assert embedder.embed.call_count == n  # unchanged since refresh → skipped


# ---------------------------------------------------------------------------
# silica_semantic_search tool — mocked embedder
# ---------------------------------------------------------------------------

def test_silica_semantic_search_empty_index(tmp_path, monkeypatch):

    # Ensure the store loaded by the tool sees an empty index
    monkeypatch.setattr("silica.kernel.recall.embed._index_path", lambda: tmp_path / "empty.json")

    from silica.tools.composed import silica_semantic_search
    result = silica_semantic_search(query="neural networks")
    assert "error" in result


def test_silica_semantic_search_uses_cooccurrence_when_embed_empty(tmp_path, monkeypatch):
    """Routing through the relatedness facade: with an empty embedding index but
    a populated co-occurrence store, search still returns results (the co-occur
    leg carries it) instead of the old cosine-only 'empty index' error."""
    monkeypatch.setattr("silica.kernel.recall.embed._index_path", lambda: tmp_path / "empty_embed.json")

    from silica.kernel.recall.cooccurrence import build_index as cooc_build
    cooc_build(
        [
            ("Concepts/Neural", "Neural", "neural network architecture deep learning"),
            ("Concepts/Sailing", "Sailing", "sailing boat harbour wind"),
        ],
        lang="english",
        force=True,
    )

    from silica.tools.composed import silica_semantic_search
    result = silica_semantic_search(query="neural network", k=5)
    assert "error" not in result
    paths = [r["path"] for r in result["results"]]
    assert any("Neural" in p for p in paths)


def test_silica_semantic_search_returns_results(tmp_path, monkeypatch):
    from silica.kernel.recall.embed import EmbedStore as _ES

    idx = tmp_path / "embeddings.json"
    monkeypatch.setattr("silica.kernel.recall.embed._index_path", lambda: idx)
    # Pre-populate the store
    store = _ES(idx)
    store.upsert("Concepts/A", "A", [1.0, 0.0])
    store.upsert("Concepts/B", "B", [0.0, 1.0])
    store.save()

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[1.0, 0.0]]

    with patch("silica.agent.providers.get_embedder", return_value=mock_embedder):
        from silica.tools.composed import silica_semantic_search
        result = silica_semantic_search(query="test", k=2)

    assert "error" not in result
    assert len(result["results"]) == 2
    assert result["results"][0]["path"] == "Concepts/A"


# ---------------------------------------------------------------------------
# OpenAIEmbedder — unit test the wrapper class
# ---------------------------------------------------------------------------

def test_openai_embedder_returns_sorted_embeddings():
    """The embedder should return vectors in input order regardless of API ordering."""
    from silica.agent.providers import OpenAIEmbedder

    # Build a mock openai client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(index=1, embedding=[0.0, 1.0]),
        MagicMock(index=0, embedding=[1.0, 0.0]),
    ]
    mock_client.embeddings.create.return_value = mock_response

    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder.client = mock_client
    embedder.model = "test-model"

    result = embedder.embed(["text_a", "text_b"])
    assert result == [[1.0, 0.0], [0.0, 1.0]]


def test_openai_embedder_empty_input():
    from silica.agent.providers import OpenAIEmbedder

    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    assert embedder.embed([]) == []


# ---------------------------------------------------------------------------
# Per-vault index keying (Task 8)
# ---------------------------------------------------------------------------

def test_index_path_is_keyed_by_vault(tmp_path, monkeypatch):
    from silica.config import CONFIG
    from silica.kernel.recall import paths
    from silica.kernel.recall.embed import _index_path

    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path / "v1"))
    p1 = _index_path()
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path / "v2"))
    p2 = _index_path()
    assert p1 != p2
    assert p1.name == "embeddings.json" and p2.name == "embeddings.json"
    assert paths.index_dir() == p2.parent


def test_index_path_falls_back_to_global_without_vault(monkeypatch):
    from silica.config import CONFIG
    from silica.kernel.recall import paths

    monkeypatch.setattr(CONFIG, "vault_path", "")
    # The global (non-vault-namespaced) dir — _SILICA_HOME, not Path.home(),
    # so the invariant holds under the suite's home sandbox too.
    assert paths.index_dir() == paths._SILICA_HOME / "index"


def test_legacy_index_migrates_on_load(tmp_path, monkeypatch):
    import orjson
    from silica.kernel.recall import embed as embed_mod
    from silica.kernel.recall.embed import EmbedStore

    legacy = tmp_path / "legacy" / "embeddings.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(orjson.dumps({"version": 1, "notes": {"old/note": {"vec": [1.0], "name": "old", "ts": 0}}}))
    keyed = tmp_path / "keyed" / "embeddings.json"
    monkeypatch.setattr(embed_mod, "_LEGACY_INDEX_PATH", legacy)
    monkeypatch.setattr(embed_mod, "_index_path", lambda: keyed)

    store = EmbedStore()
    assert "old/note" in store._notes  # loaded from legacy
    store.save()
    assert keyed.exists()              # copied forward into the keyed namespace


# ---------------------------------------------------------------------------
# document_theme_vector — bounded cost on book-sized bodies
# ---------------------------------------------------------------------------

def _marker_body(n_blocks: int) -> str:
    """n_blocks of exactly 1200 chars, each starting with its own index marker."""
    return "".join(f"<{i:04d}>" + "a" * 1194 for i in range(n_blocks))


def _fake_embedder(calls: list):
    fake = MagicMock()
    fake.model = "fake-theme-model"
    def _embed(texts):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]
    fake.embed.side_effect = _embed
    return fake


def test_theme_vector_samples_book_sized_body():
    """A 40k-char body must cost a bounded, equispaced sample of blocks — not
    one embed input per 1200 chars (RECON paid 34 blocks per book segment)."""
    from silica.kernel.recall.embed import document_theme_vector

    calls: list = []
    vec = document_theme_vector(_fake_embedder(calls), _marker_body(40))

    assert vec
    assert len(calls) == 1
    sent = calls[0]
    assert len(sent) <= 8
    picked = sorted(int(s[1:5]) for s in sent)
    assert picked[0] == 0        # the opening block is always sampled
    assert picked[-1] >= 30      # and the sample reaches the back quarter


def test_theme_vector_short_body_embeds_every_block():
    """At or under the sample budget the behavior is bit-identical to before:
    every block is embedded, in order."""
    from silica.kernel.recall.embed import document_theme_vector

    calls: list = []
    vec = document_theme_vector(_fake_embedder(calls), _marker_body(5))

    assert vec
    assert [int(s[1:5]) for s in calls[0]] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# EmbedStore.cosine_top_k_batch
# ---------------------------------------------------------------------------

def _random_store(tmp_path, n=40, dim=16, seed=7):
    import random
    rng = random.Random(seed)
    store = EmbedStore(path=tmp_path / "embeddings.json")
    for i in range(n):
        store.upsert(f"note_{i}", f"Note {i}", [rng.uniform(-1.0, 1.0) for _ in range(dim)])
    return store


def test_cosine_top_k_batch_matches_per_note_path(tmp_path):
    """The whole point of the batch path: same top-k as N separate calls.

    It replaces the per-note loop in graph_export.knn_edges, so any divergence in
    membership or ORDER silently rewires the semantic map. Scores may differ in the
    last decimal (the query is normalized once instead of twice), ranking may not.
    """
    store = _random_store(tmp_path)
    keys = store.paths()

    batch = store.cosine_top_k_batch(keys, k=6)
    assert set(batch) == set(keys)
    for key in keys:
        per_note = store.cosine_top_k(store.get_vec(key), k=6, exclude={key})
        assert [r["path"] for r in batch[key]] == [r["path"] for r in per_note]
        for got, want in zip(batch[key], per_note):
            assert got["score"] == pytest.approx(want["score"], abs=1e-3)
            assert got["name"] == want["name"]


def test_cosine_top_k_batch_excludes_self(tmp_path):
    store = _random_store(tmp_path, n=10)
    for key, hits in store.cosine_top_k_batch(store.paths(), k=9).items():
        assert key not in [h["path"] for h in hits]


def test_cosine_top_k_batch_skips_unknown_and_off_dimension(tmp_path):
    store = _random_store(tmp_path, n=5, dim=4)
    store.upsert("odd_dim", "Odd", [1.0, 0.0, 0.0])   # minority dim, off the matrix

    out = store.cosine_top_k_batch([*store.paths(), "never_indexed"], k=3)
    assert "never_indexed" not in out
    assert "odd_dim" not in out
    assert len(out) == 5


def test_cosine_top_k_batch_blocks_do_not_change_results(tmp_path):
    """block only bounds peak memory; it must not be observable in the output."""
    store = _random_store(tmp_path, n=25)
    keys = store.paths()
    assert store.cosine_top_k_batch(keys, k=4, block=3) == store.cosine_top_k_batch(keys, k=4)


def test_cosine_top_k_batch_degenerate_tail_matches_per_note_path(tmp_path):
    """Fewer than k strictly-positive neighbours: the fast path cannot decide the
    tail (0.0-scored off-matrix notes and their ordering belong to _search), so it
    delegates. This is the branch the random-vector test never reaches."""
    store = EmbedStore(path=tmp_path / "embeddings.json")
    store.upsert("east",  "East",  [1.0, 0.0])
    store.upsert("north", "North", [0.0, 1.0])    # cosine 0.0 vs east
    store.upsert("west",  "West",  [-1.0, 0.0])   # cosine -1.0 vs east

    batch = store.cosine_top_k_batch(["east"], k=2)["east"]
    per_note = store.cosine_top_k(store.get_vec("east"), k=2, exclude={"east"})
    assert [r["path"] for r in batch] == [r["path"] for r in per_note] == ["north", "west"]
    assert [r["score"] for r in batch] == [r["score"] for r in per_note]
