# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""silica_vaults: the vaults this machine knows, scored for one question.

Routing, not fusion (ADR-0019 keeps recall at two lanes): the model reads a
per-vault scoreboard and chooses where to peek. A cold vault says so instead
of scoring zero, because "0 hits" and "never indexed" call for different next
steps.
"""
from __future__ import annotations

import json

import httpx

from silica.config import CONFIG
from silica.kernel.recall import vault_registry as reg
from silica.kernel.recall.lexical import LexicalStore
from silica.kernel.recall.paths import index_dir_for


def _adopted(tmp_path, name: str):
    v = tmp_path / name
    v.mkdir()
    (v / "vault.yaml").write_text("write_dir: ''\n", encoding="utf-8")
    return v


def _lexical(vault, docs) -> None:
    d = index_dir_for(str(vault))
    d.mkdir(parents=True, exist_ok=True)
    st = LexicalStore(path=d / "lexical.json")
    for path, name, body in docs:
        st.upsert(path, name, body)
    st.save()


def _obsidian(tmp_path, monkeypatch, vaults) -> None:
    j = tmp_path / "obsidian.json"
    j.write_text(json.dumps({"vaults": {
        f"id{i}": {"path": str(v), "ts": 0} for i, v in enumerate(vaults)}}), encoding="utf-8")
    monkeypatch.setattr(reg, "_obsidian_json", lambda: j)


def _no_memory(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(CONFIG, "memory_vault", str(tmp_path / "no-such-memory"))


def test_known_vaults_active_first_then_adopted_obsidian_vaults(tmp_path, monkeypatch):
    active, other = _adopted(tmp_path, "active"), _adopted(tmp_path, "other")
    bare = tmp_path / "bare"
    bare.mkdir()  # in Obsidian, never adopted: no vault.yaml, so not a Silica vault
    monkeypatch.setattr(CONFIG, "vault_path", str(active))
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [other, bare, active])
    assert reg.known_vaults() == [active.resolve(), other.resolve()]


def test_memory_vault_is_known_when_it_exists(tmp_path, monkeypatch):
    active, mem = _adopted(tmp_path, "active"), _adopted(tmp_path, "mem")
    monkeypatch.setattr(CONFIG, "vault_path", str(active))
    monkeypatch.setattr(CONFIG, "memory_vault", str(mem))
    _obsidian(tmp_path, monkeypatch, [])
    assert reg.known_vaults() == [active.resolve(), mem.resolve()]


def test_unreadable_obsidian_registry_is_just_absent(tmp_path, monkeypatch):
    active = _adopted(tmp_path, "active")
    monkeypatch.setattr(CONFIG, "vault_path", str(active))
    _no_memory(tmp_path, monkeypatch)
    monkeypatch.setattr(reg, "_obsidian_json", lambda: tmp_path / "missing.json")
    assert reg.known_vaults() == [active.resolve()]


def test_coverage_levels(tmp_path):
    cold, lex = _adopted(tmp_path, "cold"), _adopted(tmp_path, "lex")
    _lexical(lex, [("N", "N", "body")])
    assert reg.coverage(cold)["level"] == "cold"
    assert reg.coverage(lex)["level"] == "lexical-only"
    assert reg.coverage(lex)["notes"] == 1


def test_describe_reads_the_cached_brief_and_the_manifest(tmp_path):
    v = _adopted(tmp_path, "v")
    (v / "vault.yaml").write_text("write_dir: docs/silica\n", encoding="utf-8")
    d = index_dir_for(str(v))
    d.mkdir(parents=True, exist_ok=True)
    (d / "vault_brief.json").write_text(json.dumps({"stamp": "x", "text": "A vault about X."}),
                                        encoding="utf-8")
    got = reg.describe(v)
    assert got["brief"] == "A vault about X."
    assert got["write_dir"] == "docs/silica"
    assert got["name"] == "v" and got["path"] == str(v.resolve())


def test_scoreboard_without_a_reranker_keeps_discovery_order_and_says_so(tmp_path, monkeypatch):
    # Measured 2026-09-01 on 4 real vaults x 4 queries: raw cosine routed 2/4,
    # cosine spread 0/3, a lexical hit count was inflated 250/270 by one
    # corpus-wide token. None of them may ORDER the rows; they stay indicative.
    stats, cook, cold = (_adopted(tmp_path, n) for n in ("stats", "cook", "cold"))
    _lexical(stats, [("Notes/Heteroscedasticity", "Heteroscedasticity",
                      "the variance of the residuals grows along the regression line")])
    _lexical(cook, [("Notes/Risotto", "Risotto", "rice broth stir butter")])
    monkeypatch.setattr(CONFIG, "vault_path", str(cook))
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [stats, cold])
    _no_reranker(monkeypatch)

    rows = reg.scoreboard("regression residuals variance", diagnostics=True)
    by = {r["name"]: r for r in rows}
    assert by["stats"]["hits"] == 1 and by["stats"]["top"] == ["Heteroscedasticity"]
    assert by["cook"]["hits"] == 0 and by["cook"]["active"] is True
    assert by["cold"]["hits"] is None and by["cold"]["coverage"] == "cold"
    assert [r["name"] for r in rows] == ["cook", "stats", "cold"]  # discovery order, active first
    assert all(r["scored"] is False for r in rows)
    assert reg.route("regression residuals variance")["ranked"] is False


def test_scoreboard_without_a_query_leads_with_the_active_vault(tmp_path, monkeypatch):
    a, b = _adopted(tmp_path, "a"), _adopted(tmp_path, "b")
    monkeypatch.setattr(CONFIG, "vault_path", str(b))
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [a])
    rows = reg.scoreboard("")
    assert [r["name"] for r in rows] == ["b", "a"]
    assert "hits" not in rows[0]


def test_tool_is_served_by_default_and_returns_the_scoreboard(tmp_path, monkeypatch):
    from silica.tools.graph import silica_vaults
    from silica.ui import mcp

    assert "silica_vaults" in mcp.exposed_tools()
    a = _adopted(tmp_path, "a")
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [])
    # "no reranker configured" is the case under test, not the machine's .env.
    monkeypatch.setattr(reg, "_reranker", lambda: None)
    out = silica_vaults(query="anything")
    assert out["active"] == str(a.resolve())
    assert [v["path"] for v in out["vaults"]] == [str(a.resolve())]
    assert out["ranked"] is False and out["reranker"] is None
    # The two numbers measured to mislead (cosine 2/4, hit count 250/270) stay
    # out of what the model reads; the bench asks for them with diagnostics=True.
    assert "hits" not in out["vaults"][0] and "best" not in out["vaults"][0]


# ---------------------------------------------------------------------------
# Field finding 2026-09-01: no real vault on the machine had a lexical.json
# (the lexical leg is opt-in), every indexed one had embeddings. A scoreboard
# that only knows the lexical index scores nothing where it matters.
# ---------------------------------------------------------------------------

def _embedded(vault, notes) -> None:
    from silica.kernel.recall.embed import EmbedStore

    d = index_dir_for(str(vault))
    d.mkdir(parents=True, exist_ok=True)
    es = EmbedStore(path=d / "embeddings.json")
    for path, name, vec in notes:
        es.upsert(path, name, vec)
    es.save()


def _fake_embedder(monkeypatch, vec) -> None:
    import silica.agent.providers as providers

    class _E:
        def embed(self, texts):
            return [list(vec) for _ in texts]

    monkeypatch.setattr(providers, "get_embedder", lambda cfg: _E())


def test_probe_falls_back_to_embeddings_when_there_is_no_lexical_index(tmp_path, monkeypatch):
    a, stats = _adopted(tmp_path, "a"), _adopted(tmp_path, "stats")
    _embedded(stats, [("Notes/Heteroscedasticity", "Heteroscedasticity", [1.0, 0.0])])
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [stats])
    _fake_embedder(monkeypatch, [1.0, 0.0])

    _no_reranker(monkeypatch)
    row = {r["name"]: r for r in reg.scoreboard("residual variance", diagnostics=True)}["stats"]
    assert row["probe"] == "embed" and row["hits"] is None
    assert row["best"] > 0.99 and row["top"] == ["Heteroscedasticity"]
    assert row["coverage"] == "indexed" and row["notes"] == 1
    assert row["scored"] is False and row["rerank"] is None


def _fake_reranker(monkeypatch, liked: str) -> None:
    import silica.agent.providers as providers

    class _R:
        def scores(self, query, docs):
            return [1.0 if liked in d else -5.0 for d in docs]

    monkeypatch.setattr(providers, "get_reranker", lambda cfg: _R())


def _no_reranker(monkeypatch) -> None:
    import silica.agent.providers as providers

    monkeypatch.setattr(providers, "get_reranker", lambda cfg: None)


def test_scoreboard_orders_by_the_reranker_when_one_is_configured(tmp_path, monkeypatch):
    # The one signal measured to route across vaults (3/3 on the field): the
    # first stage nominates a few notes per vault, the cross-encoder scores
    # them, and the best score is comparable between vaults.
    a, near, far = (_adopted(tmp_path, n) for n in ("a", "near", "far"))
    _embedded(near, [("N", "Near note", [1.0, 0.0])])
    (near / "N.md").write_text("the answer lives here\n", encoding="utf-8")
    _embedded(far, [("F", "Far note", [0.9, 0.1])])
    (far / "F.md").write_text("nothing of the kind\n", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [far, near])
    _fake_embedder(monkeypatch, [1.0, 0.0])
    _fake_reranker(monkeypatch, liked="answer lives here")

    rows = reg.scoreboard("q")
    assert [r["name"] for r in rows] == ["near", "far", "a"]  # a nominated nothing: last
    assert rows[0]["scored"] is True and rows[0]["rerank"] == 1.0 and rows[1]["rerank"] == -5.0
    assert rows[0]["top"] == ["Near note"]
    assert rows[2]["rerank"] is None and rows[2]["scored"] is False
    assert reg.route("q")["ranked"] is True


def test_lexical_hits_count_notes_matching_two_query_terms(tmp_path):
    # Field 2026-09-01: every nucleated note carries the token "silica", so a
    # union count answered 250/270 for a query that had nothing to do with the
    # vault. Two distinct terms is the cheapest cut that keeps a real signal.
    from silica.kernel.recall.lexical import LexicalStore

    st = LexicalStore(path=tmp_path / "lexical.json")
    st.upsert("a", "a", "silica wrote this note about tea")
    st.upsert("b", "b", "silica wrote this note about drift in indexes")
    st.upsert("c", "c", "silica wrote this note about coffee")
    assert st.match_count("silica drift") == 1
    assert st.match_count("silica") == 3  # a one-term query keeps its one term
    assert st.match_count("nothing here") == 0


def test_scoreboard_without_an_embedder_still_answers(tmp_path, monkeypatch):
    import silica.agent.providers as providers

    a, v = _adopted(tmp_path, "a"), _adopted(tmp_path, "v")
    _embedded(v, [("N", "N", [1.0, 0.0])])
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [v])

    class _Down:
        def embed(self, texts):
            raise httpx.ConnectError("embedder offline")  # what a dead endpoint really raises

    monkeypatch.setattr(providers, "get_embedder", lambda cfg: _Down())
    row = {r["name"]: r for r in reg.scoreboard("q", diagnostics=True)}["v"]
    assert row["probe"] is None and row["best"] is None and row["top"] == []
    assert row["coverage"] == "indexed"  # the index exists; only the probe could not run


# ---------------------------------------------------------------------------
# Stage one is a POOL, stage two orders it. Both sizes were one `k` until the
# bench (scripts/bench_vault_router.py) separated them: the reranker can only
# promote a note the first stage put forward, and the recall rank probe had
# already shown the tail of a ranking carrying gold.
# ---------------------------------------------------------------------------

def _basis(i: int, n: int = 8) -> list[float]:
    v = [0.0] * n
    v[i] = 1.0
    return v


def test_nominate_lists_each_stage_up_to_the_pool(tmp_path, monkeypatch):
    v = _adopted(tmp_path, "v")
    _lexical(v, [(f"N{i}", f"N{i}", "regression residuals " + "variance " * i) for i in range(4)])
    _embedded(v, [(f"E{i}", f"E{i}", _basis(i)) for i in range(4)])
    _fake_embedder(monkeypatch, [0.9, 0.5, 0.3, 0.1, 0, 0, 0, 0])

    got = reg.nominate(v, "regression residuals variance", pool=2)
    assert got["stages"] == ["lexical", "embed"]
    assert len(got["lexical"]) == 2 and all(len(t) == 2 for t in got["lexical"])
    assert [p for p, _n, _c in got["embed"]] == ["E0", "E1"]  # cosine order, with the cosine
    assert got["embed"][0][2] > got["embed"][1][2] > 0
    assert got["hits"] == 4 and got["best"] == round(got["embed"][0][2], 3) and got["notes"] == 4


def test_probe_pool_is_wider_than_the_titles_it_shows(tmp_path, monkeypatch):
    # Six notes; the query vector likes E0 most and E5 least; the cross-encoder
    # likes only E5. With pool == k (the first cut) E5 never reached it.
    v = _adopted(tmp_path, "v")
    _embedded(v, [(f"E{i}", f"E{i}", _basis(i)) for i in range(6)])
    for i in range(6):
        (v / f"E{i}.md").write_text("gold" if i == 5 else "dust", encoding="utf-8")
    _fake_embedder(monkeypatch, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0, 0])
    _fake_reranker(monkeypatch, liked="gold")
    import silica.agent.providers as providers

    row = reg.probe(v, "q", k=2, reranker=providers.get_reranker(None))
    assert row["top"] == ["E5", "E0"] and len(row["top"]) == 2
    assert row["scored"] is True and row["rerank"] == 1.0


def test_probe_with_an_abstaining_reranker_keeps_nomination_order_and_says_unscored(tmp_path, monkeypatch):
    v = _adopted(tmp_path, "v")
    _embedded(v, [("E0", "E0", _basis(0)), ("E1", "E1", _basis(1))])
    _fake_embedder(monkeypatch, [0.9, 0.5, 0, 0, 0, 0, 0, 0])

    class _Mute:
        def scores(self, query, docs):
            return None  # the provider's own abstention on a down endpoint

    row = reg.probe(v, "q", k=2, reranker=_Mute())
    assert row["top"] == ["E0", "E1"] and row["scored"] is False and row["rerank"] is None


def test_a_row_the_reranker_could_not_score_is_marked_and_sorted_last(tmp_path, monkeypatch):
    import silica.agent.providers as providers

    a, near, far = (_adopted(tmp_path, n) for n in ("a", "near", "far"))
    _embedded(near, [("N", "Near note", [1.0, 0.0])])
    (near / "N.md").write_text("the answer lives here\n", encoding="utf-8")
    _embedded(far, [("F", "Far note", [0.9, 0.1])])
    (far / "F.md").write_text("nothing of the kind\n", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [far, near])
    _fake_embedder(monkeypatch, [1.0, 0.0])

    class _Flaky:
        model = "bge-reranker-v2-m3-Q8_0"

        def scores(self, query, docs):
            if any("nothing of the kind" in d for d in docs):
                return None  # timed out on this vault only
            return [1.0 for _ in docs]

    monkeypatch.setattr(providers, "get_reranker", lambda cfg: _Flaky())
    out = reg.route("q")
    assert out["ranked"] is True and out["reranker"] == "bge-reranker-v2-m3-Q8_0"
    by_name = [r["name"] for r in out["vaults"]]
    assert by_name[0] == "near"
    far_row = next(r for r in out["vaults"] if r["name"] == "far")
    assert far_row["scored"] is False and far_row["rerank"] is None
    assert far_row["top"] == ["Far note"]  # nominated, just not scored


def test_default_rows_carry_no_diagnostic_numbers(tmp_path, monkeypatch):
    v = _adopted(tmp_path, "v")
    _lexical(v, [("N", "N", "regression residuals variance")])
    monkeypatch.setattr(CONFIG, "vault_path", str(v))
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [])
    _no_reranker(monkeypatch)
    row = reg.scoreboard("regression residuals")[0]
    assert "hits" not in row and "best" not in row
    assert row["top"] == ["N"] and row["probe"] == "lexical" and row["scored"] is False


# ---------------------------------------------------------------------------
# `home`: the vaults whose best rerank score clears a floor calibrated on the
# bench (42 homed + 10 homeless queries, bge-reranker-v2-m3 logits, floor
# -2.5: 41/42 kept, 9/10 refused, chosen because a refused home costs the
# answer and an admitted stranger costs a peek). A logit is one model's scale,
# so the rule applies only under the model it was measured on; anything else
# gets `home: null` while `ranked` stays true.
# ---------------------------------------------------------------------------

def _two_vaults(tmp_path, monkeypatch):
    a, near, far = (_adopted(tmp_path, n) for n in ("a", "near", "far"))
    _embedded(near, [("N", "Near note", [1.0, 0.0])])
    (near / "N.md").write_text("the answer lives here\n", encoding="utf-8")
    _embedded(far, [("F", "Far note", [0.9, 0.1])])
    (far / "F.md").write_text("nothing of the kind\n", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [far, near])
    _fake_embedder(monkeypatch, [1.0, 0.0])
    return a, near, far


def _scoring(monkeypatch, model: str, near: float, far: float) -> None:
    import silica.agent.providers as providers

    class _R:
        def scores(self, query, docs):
            return [near if "answer lives here" in d else far for d in docs]

    _R.model = model
    monkeypatch.setattr(providers, "get_reranker", lambda cfg: _R())


def test_home_lists_the_vaults_that_clear_the_calibrated_floor(tmp_path, monkeypatch):
    _a, near, _far = _two_vaults(tmp_path, monkeypatch)
    _scoring(monkeypatch, "bge-reranker-v2-m3-Q8_0", near=1.2, far=-5.0)
    out = reg.route("q")
    assert out["ranked"] is True and out["home"] == [str(near.resolve())]


def test_home_is_empty_when_no_vault_clears_the_floor(tmp_path, monkeypatch):
    _two_vaults(tmp_path, monkeypatch)
    _scoring(monkeypatch, "bge-reranker-v2-m3-Q8_0", near=-2.8, far=-7.0)  # homeless: best is below -2.5
    out = reg.route("q")
    assert out["ranked"] is True and out["home"] == []
    assert out["vaults"][0]["name"] == "near"  # still ordered: the floor refuses, it does not reorder


def test_home_is_unknown_under_a_reranker_the_floor_was_not_measured_on(tmp_path, monkeypatch):
    _two_vaults(tmp_path, monkeypatch)
    _scoring(monkeypatch, "mxbai-rerank-base-v2", near=1.2, far=-5.0)
    out = reg.route("q")
    assert out["ranked"] is True and out["home"] is None and out["reranker"] == "mxbai-rerank-base-v2"


def test_home_is_unknown_without_a_reranker(tmp_path, monkeypatch):
    _two_vaults(tmp_path, monkeypatch)
    _no_reranker(monkeypatch)
    out = reg.route("q")
    assert out["ranked"] is False and out["home"] is None


def test_home_is_unknown_while_an_indexed_vault_could_not_be_probed(tmp_path, monkeypatch):
    import silica.agent.providers as providers

    _a, near, far = _two_vaults(tmp_path, monkeypatch)
    _scoring(monkeypatch, "bge-reranker-v2-m3-Q8_0", near=1.2, far=-5.0)

    class _Down:
        def embed(self, texts):
            raise httpx.ConnectError("embedder offline")

    monkeypatch.setattr(providers, "get_embedder", lambda cfg: _Down())
    out = reg.route("q")  # both vaults are embed-only: nothing nominated, nothing scored
    assert out["ranked"] is False and out["home"] is None


def test_home_is_empty_when_every_stage_ran_and_nominated_nothing(tmp_path, monkeypatch):
    a, v = _adopted(tmp_path, "a"), _adopted(tmp_path, "v")
    _lexical(v, [("N", "N", "sourdough starter feeding schedule")])
    monkeypatch.setattr(CONFIG, "vault_path", str(a))
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [v])
    _scoring(monkeypatch, "bge-reranker-v2-m3-Q8_0", near=1.2, far=-5.0)
    out = reg.route("kubernetes ingress")  # no term in common: the lexical stage ran and put nothing forward
    assert out["ranked"] is False and out["home"] == []
    assert next(r for r in out["vaults"] if r["name"] == "v")["probe"] == "lexical"


# ---------------------------------------------------------------------------
# Discovery beyond Obsidian: a vault served from the REPL or `silica mcp
# --vault` and never opened in Obsidian used to be known only while active.
# ---------------------------------------------------------------------------

def test_a_vault_the_sweep_marked_is_known_without_obsidian(tmp_path, monkeypatch):
    from silica.kernel.recall import sync

    active, served = _adopted(tmp_path, "active"), _adopted(tmp_path, "served")
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [])
    monkeypatch.setattr(CONFIG, "vault_path", str(served))
    sync.sweep()  # gated off suite-wide: the marker is written before the gate
    assert (index_dir_for(str(served)) / "vault.json").is_file()
    monkeypatch.setattr(CONFIG, "vault_path", str(active))
    assert reg.known_vaults() == [active.resolve(), served.resolve()]


def test_a_marker_for_a_folder_no_longer_adopted_or_present_is_ignored(tmp_path, monkeypatch):
    active = _adopted(tmp_path, "active")
    gone = tmp_path / "gone"
    gone.mkdir()  # exists, but no vault.yaml any more
    d = index_dir_for(str(gone))
    d.mkdir(parents=True, exist_ok=True)
    (d / "vault.json").write_text(json.dumps({"path": str(gone)}), encoding="utf-8")
    torn = index_dir_for(str(tmp_path / "torn"))
    torn.mkdir(parents=True, exist_ok=True)
    (torn / "vault.json").write_text("{", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(active))
    _no_memory(tmp_path, monkeypatch)
    _obsidian(tmp_path, monkeypatch, [])
    assert reg.known_vaults() == [active.resolve()]
