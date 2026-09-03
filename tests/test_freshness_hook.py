"""Tests for the co-occurrence freshness hook in the orchestrator.

After a write commit the orchestrator refreshes BOTH PROPOSE-layers side by
side. The co-occurrence leg is the STABLE leg: it must refresh independently of
the embedder (works with LM Studio down), so it lives in its own helper that
imports only the embedder-free cooccurrence module.
"""
from __future__ import annotations

import ast

import snowballstemmer

from silica.kernel.recall.cooccurrence import CooccurStore
from silica.kernel.write.ops import Op, OpType
from silica.router.orchestrator import _refresh_cooccurrence_for_ops


def _write_op(path: str) -> Op:
    return Op(op=OpType.write, heading=path, source_basename="inbox.md", path=path, snippet="x")


def _patch_op(path: str) -> Op:
    return Op(op=OpType.patch, heading=path, source_basename="inbox.md", path=path, snippet="x")


def _en(word: str) -> str:
    return snowballstemmer.stemmer("english").stemWord(word)


def test_refreshes_committed_write_and_patch_ops(tmp_path):
    store = CooccurStore(path=tmp_path / "c.json", lang="english")
    ops = [_write_op("Concepts/Neural.md"), _patch_op("Concepts/Boats.md")]
    bodies = {
        "Concepts/Neural.md": "neural network architecture",
        "Concepts/Boats.md": "sailing boat harbour",
    }
    n = _refresh_cooccurrence_for_ops(
        ops, {"Concepts/Neural.md", "Concepts/Boats.md"},
        read_body=bodies.get, lang="english", store=store,
    )
    assert n == 2
    # keys are vault-relative, .md stripped (mirrors the embed index keying)
    assert "Concepts/Neural" in store.paths()
    assert "Concepts/Boats" in store.paths()
    # the actual contribution is indexed (queryable)
    assert store.neighbors("network", k=5)  # neural<->network edge present


def test_skips_uncommitted_paths(tmp_path):
    store = CooccurStore(path=tmp_path / "c.json", lang="english")
    ops = [_write_op("Concepts/Kept.md"), _write_op("Concepts/Dropped.md")]
    bodies = {"Concepts/Kept.md": "alpha beta", "Concepts/Dropped.md": "gamma delta"}
    n = _refresh_cooccurrence_for_ops(
        ops, {"Concepts/Kept.md"},  # only one was committed
        read_body=bodies.get, lang="english", store=store,
    )
    assert n == 1
    assert "Concepts/Kept" in store.paths()
    assert "Concepts/Dropped" not in store.paths()


def test_ignores_non_write_ops(tmp_path):
    store = CooccurStore(path=tmp_path / "c.json", lang="english")
    overwrite = Op(op=OpType.overwrite, heading="X", source_basename="i.md",
                   path="Concepts/X.md", content="alpha beta")
    n = _refresh_cooccurrence_for_ops(
        ops=[overwrite], committed_paths={"Concepts/X.md"},
        read_body=lambda p: "alpha beta", lang="english", store=store,
    )
    # only write/patch participate in the freshness hook (mirrors embed refresh)
    assert n == 0
    assert len(store) == 0


def test_replacement_not_inflation_on_repeated_refresh(tmp_path):
    store = CooccurStore(path=tmp_path / "c.json", lang="english")
    op = _write_op("Concepts/N.md")
    rb = lambda p: "alpha beta"
    _refresh_cooccurrence_for_ops([op], {"Concepts/N.md"}, read_body=rb, lang="english", store=store)
    w1 = next(c["weight"] for c in store.neighbors("alpha", k=5) if c["concept"] == "beta")
    _refresh_cooccurrence_for_ops([op], {"Concepts/N.md"}, read_body=rb, lang="english", store=store)
    w2 = next(c["weight"] for c in store.neighbors("alpha", k=5) if c["concept"] == "beta")
    assert w1 == w2  # force=True replaces the note's contribution, never accumulates


def test_hook_creates_note_edges_between_related_notes(tmp_path):
    # CORRELATE (ADR-0013): the freshness hook also refreshes note_edges, so two
    # notes sharing enough top stems come out linked after a write commit.
    store = CooccurStore(path=tmp_path / "c.json", lang="english")
    ops = [_write_op("Concepts/A.md"), _write_op("Concepts/B.md")]
    bodies = {
        "Concepts/A.md": "quick sort compares array elements",
        "Concepts/B.md": "quick sort swaps array elements",
    }
    _refresh_cooccurrence_for_ops(
        ops, {"Concepts/A.md", "Concepts/B.md"},
        read_body=bodies.get, lang="english", store=store,
    )
    assert "Concepts/B" in store.note_edges_for("Concepts/A")


def test_best_effort_never_raises_on_read_failure(tmp_path):
    store = CooccurStore(path=tmp_path / "c.json", lang="english")

    def boom(_path):
        raise OSError("driver down")

    # a per-note read failure must be swallowed, not propagated
    n = _refresh_cooccurrence_for_ops(
        [_write_op("Concepts/N.md")], {"Concepts/N.md"},
        read_body=boom, lang="english", store=store,
    )
    assert n == 0


def test_op_concepts_flow_into_contribution(tmp_path):
    """#9: an op's LLM concepts reinforce the note's co-occurrence contribution."""
    store = CooccurStore(path=tmp_path / "c.json", lang="english")
    op = Op(
        op=OpType.write, heading="N", source_basename="inbox.md",
        path="Concepts/N.md", snippet="x",
        concepts=["quantum entanglement"],
    )
    n = _refresh_cooccurrence_for_ops(
        [op], {"Concepts/N.md"},
        read_body=lambda p: "alpha beta", lang="english", store=store,
    )
    assert n == 1
    nodes = store.note_nodes("Concepts/N")
    assert _en("quantum") in nodes
    assert _en("entanglement") in nodes


def test_auto_lang_hook_never_reflips_frozen_store_language(tmp_path):
    """Round 2 mainline pin: the hook passes force=True (replacement semantics
    for a note's contribution) and lang="auto" (CONFIG default). That exact
    shape must NOT re-freeze store.lang from the batch — a single English note
    landing on an Italian-frozen store keeps store.lang="italian" and gets
    stemmed with the frozen Italian stemmer (uniform node keys)."""
    from silica.kernel.recall.cooccurrence import build_contribution

    store = CooccurStore(path=tmp_path / "c.json", lang="italian")
    store.upsert_note(
        "it/nota.md",
        build_contribution("Nota", "La rete della azienda migliora la produttivita", lang="italian"),
    )
    store.save()

    n = _refresh_cooccurrence_for_ops(
        [_write_op("en/note.md")], {"en/note.md"},
        read_body=lambda p: "The company improves productivity for the whole team.",
        lang="auto", store=store,
    )
    assert n == 1
    assert store.lang == "italian"
    # Italian stemmer leaves "company" unchanged; English would give "compani".
    assert "company" in store.note_nodes("en/note")
    assert "compani" not in store.note_nodes("en/note")


def test_helper_is_embedder_free():
    """The freshness helper is the stable leg: it must not pull in the embedder
    or provider stack, so it refreshes even when LM Studio is down."""
    import inspect
    import silica.router.orchestrator as orch
    src = inspect.getsource(orch._refresh_cooccurrence_for_ops)
    tree = ast.parse(src)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    assert not any("providers" in m for m in imported)
    assert not any(m.endswith("embed") or ".embed" in m for m in imported)
