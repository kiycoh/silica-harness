# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Stale flags on the fused-retrieval payloads (spec-stale-triggers §3).

Store-keyspace paths carry no .md; the peek map's keys do. peek_level bridges
the two, and memory-lane results are never flagged (another vault)."""
from types import SimpleNamespace

from silica.kernel.code import codedocs
from silica.tools import graph


def _r(path, origin="vault", score=0.9):
    return SimpleNamespace(path=path, name=path.rsplit("/", 1)[-1],
                           score=score, origin=origin)


def _patch_retrieve(monkeypatch, results):
    import silica.kernel.recall.perception as perception_mod
    monkeypatch.setattr(perception_mod, "facade_retrieve",
                        lambda text, k, **kw: (results, None), raising=False)
    # The facade reranks the surviving notes itself now; these tests assert the
    # stale-flag projection, so keep the cross-encoder out of the payload.
    import silica.agent.providers as providers_mod
    monkeypatch.setattr(providers_mod, "get_reranker", lambda cfg: None)


def test_stale_entry_flags_vault_results():
    m = {"wiki/m.md": "structural"}
    assert graph._stale_entry(m, _r("wiki/m")) == {"stale": "structural"}
    assert graph._stale_entry(m, _r("other/n")) == {}


def test_stale_entry_never_flags_memory_lane():
    m = {"wiki/m.md": "structural"}
    assert graph._stale_entry(m, _r("wiki/m", origin="memory")) == {}


def test_semantic_search_flags_stale_results(monkeypatch):
    _patch_retrieve(monkeypatch, [_r("wiki/m"), _r("other/n")])
    monkeypatch.setattr(codedocs, "peek", lambda v: {"wiki/m.md": "structural"})
    out = graph.silica_semantic_search("q")
    assert out["results"][0]["stale"] == "structural"
    assert "stale" not in out["results"][1]


def test_semantic_search_fresh_vault_payload_unchanged(monkeypatch):
    _patch_retrieve(monkeypatch, [_r("wiki/m")])
    monkeypatch.setattr(codedocs, "peek", lambda v: {})
    assert set(graph.silica_semantic_search("q")["results"][0]) == {
        "path", "name", "score"}


def test_peek_failure_never_fails_the_search(monkeypatch):
    _patch_retrieve(monkeypatch, [_r("wiki/m")])

    def boom(v):
        raise RuntimeError("boom")

    monkeypatch.setattr(codedocs, "peek", boom)
    assert graph.silica_semantic_search("q")["results"]


def _related_result(path, origin="vault", score=0.9):
    # silica_related's result dict also reads r.evidence, unlike the facade
    # search results above.
    # embed_score / cooccur_weight are on the RelatedNote contract; the
    # payload reads them structurally since ADR-0032.
    return SimpleNamespace(path=path, name=path.rsplit("/", 1)[-1],
                           score=score, origin=origin, evidence=["embed:0.9"],
                           embed_score=None, cooccur_weight=None)


def test_related_flags_stale_vault_result_not_memory_lane(tmp_path, monkeypatch):
    """silica_related must get the same stale-flag treatment as
    silica_semantic_search. Fails if the **_stale_entry(...) spread were
    ever dropped from silica_related's result-dict comprehension."""
    from silica.kernel.recall.embed import EmbedStore
    from silica.kernel.recall.cooccurrence import CooccurStore
    import silica.kernel.recall.relatedness as relatedness_mod

    # Non-empty stores so silica_related doesn't take the empty-index early
    # return; related_notes itself is stubbed below, so their content is
    # otherwise irrelevant.
    es = EmbedStore(path=tmp_path / "e.json")
    es.upsert("wiki/m", "m", [1.0, 0.0])
    monkeypatch.setattr("silica.kernel.recall.embed.get_store", lambda: es)
    monkeypatch.setattr(
        "silica.kernel.recall.cooccurrence.get_cooccur_store",
        lambda **_: CooccurStore(path=tmp_path / "c.json", lang="english"),
    )
    monkeypatch.setattr(
        "silica.driver.DRIVER",
        SimpleNamespace(read_note=lambda note: (_ for _ in ()).throw(KeyError(note))),
    )

    results = [_related_result("wiki/m"), _related_result("mem/x", origin="memory")]
    monkeypatch.setattr(relatedness_mod, "related_notes", lambda *a, **k: results)
    monkeypatch.setattr(codedocs, "peek", lambda v: {"wiki/m.md": "structural"})

    out = graph.silica_related("wiki/m", k=5)
    by_path = {r["path"]: r for r in out["results"]}
    assert by_path["wiki/m"]["stale"] == "structural"
    assert "stale" not in by_path["mem/x"]
