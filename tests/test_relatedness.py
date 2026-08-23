"""Tests for the relatedness facade (kernel/relatedness.py).

The facade fuses two PROPOSE-layers into a single note-level ranking:
  - embeddings (EmbedStore.cosine_top_k)  — semantic similarity
  - co-occurrence (CooccurStore + inverted index) — associative reach
via Reciprocal Rank Fusion, with degenerate proponents abstaining so the
survivor's ranking passes through unchanged ("embedder down -> cooccur routing").
"""
from __future__ import annotations

import pytest


from silica.kernel.recall.embed import EmbedStore
from silica.kernel.recall.cooccurrence import CooccurStore, build_contribution


# ---------------------------------------------------------------------------
# RRF fusion (pure)
# ---------------------------------------------------------------------------

from silica.kernel.recall.relatedness import _rrf_fuse, RRF_K


@pytest.fixture(autouse=True)
def _linear_tf_unless_the_test_says_otherwise(monkeypatch):
    """The reference-equivalence tests below pin the LINEAR tf form of the
    cooccur leg against its old note-major implementation; they predate the
    BM25 default (ADR-0029, 2026-08-23) and measure the other branch. Tests
    that want BM25 set the flag themselves, after this fixture ran."""
    from silica.config import CONFIG
    monkeypatch.setattr(CONFIG, "cooccur_bm25", False)


def test_rrf_fuse_single_ranking_orders_by_rank():
    fused = _rrf_fuse([[("A", 9.0), ("B", 4.0), ("C", 1.0)]])
    # earlier rank -> higher RRF contribution
    assert fused["A"] > fused["B"] > fused["C"]


def test_rrf_fuse_rewards_agreement_across_rankings():
    # X is rank-2 in both lists; Y is rank-1 in one and absent in the other.
    embed = [("Y", 0.9), ("X", 0.8), ("Z", 0.1)]
    cooc = [("W", 50.0), ("X", 30.0), ("Q", 1.0)]
    fused = _rrf_fuse([embed, cooc])
    # X appears in both -> accumulates two reciprocal-rank terms -> beats
    # single-list leaders Y and W.
    assert fused["X"] > fused["Y"]
    assert fused["X"] > fused["W"]


def test_rrf_fuse_uses_standard_damping_constant():
    fused = _rrf_fuse([[("A", 1.0)]])
    assert fused["A"] == 1.0 / (RRF_K + 1)


def test_rrf_fuse_empty_is_empty():
    assert _rrf_fuse([]) == {}
    assert _rrf_fuse([[]]) == {}


def test_fuse_drops_vault_root_artifacts():
    # A stale GRAPH_REPORT vector can outlive its index-build exclusion (the
    # store is upsert-only); _fuse must never surface it. "real" survives.
    from silica.kernel.recall.relatedness import _fuse
    out = _fuse([("GRAPH_REPORT", "Graph Report", 0.99), ("real", "Real", 0.5)], None, k=5)
    assert [r.path for r in out] == ["real"]


# ---------------------------------------------------------------------------
# Embed leg + abstention
# ---------------------------------------------------------------------------

from silica.kernel.recall.relatedness import _embed_ranking


def _embed_store(tmp_path) -> EmbedStore:
    es = EmbedStore(path=tmp_path / "e.json")
    es.upsert("A", "A note", [1.0, 0.0])
    es.upsert("B", "B note", [0.9, 0.1])   # close to A
    es.upsert("C", "C note", [0.0, 1.0])   # orthogonal to A
    return es


def test_embed_ranking_returns_path_name_score(tmp_path):
    es = _embed_store(tmp_path)
    ranking = _embed_ranking(es, "A", k=5, exclude={"A"})
    assert ranking is not None
    paths = [p for p, _n, _s in ranking]
    assert paths[0] == "B"            # nearest neighbour first
    assert ("B", "B note") == (ranking[0][0], ranking[0][1])


def test_related_notes_excludes_the_query_when_it_carries_md(tmp_path):
    """A caller may hand in a graph-style '<path>.md'; the store keys are stripped.

    Regression: `blocked` was built from the raw path, so the exclude never matched
    a store key and the query note came back as its own closest neighbour at cosine
    1.0, burning a slot in the fusion pool. /map (mindmap._with_md) hits this on
    every call. Both keyspaces must now give the same answer.
    """
    from silica.kernel.recall.relatedness import related_notes

    es = _embed_store(tmp_path)

    with_md = [r.path for r in related_notes("A.md", embed_store=es, k=3)]
    stripped = [r.path for r in related_notes("A", embed_store=es, k=3)]

    assert "A" not in with_md and "A.md" not in with_md
    assert with_md == stripped == ["B", "C"]


def test_embed_ranking_normalises_the_query_keyspace(tmp_path):
    """_embed_ranking resolves the vector through cooccur_key, so '<path>.md'
    finds the same vector as the stripped key instead of abstaining."""
    es = _embed_store(tmp_path)
    assert _embed_ranking(es, "A.md", k=5, exclude={"A"}) == \
           _embed_ranking(es, "A",    k=5, exclude={"A"})


def test_embed_ranking_abstains_when_note_not_indexed(tmp_path):
    es = _embed_store(tmp_path)
    assert _embed_ranking(es, "DOES_NOT_EXIST", k=5, exclude=set()) is None


def test_embed_ranking_abstains_on_degenerate_all_zero_scores(tmp_path):
    es = EmbedStore(path=tmp_path / "e.json")
    es.upsert("Z", "Z", [0.0, 0.0])   # zero query vector -> every score 0.0
    es.upsert("B", "B", [1.0, 0.0])
    # degenerate output must abstain, NOT return a flat zero ranking (poison for RRF)
    assert _embed_ranking(es, "Z", k=5, exclude={"Z"}) is None


def test_embed_ranking_handles_md_suffixed_query(tmp_path):
    es = _embed_store(tmp_path)
    # graph_report-style callers may pass paths with a trailing .md
    ranking = _embed_ranking(es, "A.md", k=5, exclude={"A"})
    assert ranking is not None
    assert ranking[0][0] == "B"


# ---------------------------------------------------------------------------
# Co-occurrence leg + abstention
# ---------------------------------------------------------------------------

from silica.kernel.recall.relatedness import _cooccur_ranking


def _cooc_store(tmp_path) -> CooccurStore:
    st = CooccurStore(path=tmp_path / "c.json", lang="english")
    st.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    st.upsert_note("B", build_contribution("B", "beta gamma delta"))  # shares beta, gamma
    st.upsert_note("C", build_contribution("C", "zeta eta theta"))    # disjoint
    return st


def test_cooccur_ranking_ranks_notes_sharing_concepts(tmp_path):
    st = _cooc_store(tmp_path)
    ranking = _cooccur_ranking(st, "A", k=5, exclude=set(), scope=None, expand=False)
    assert ranking is not None
    paths = [p for p, _w in ranking]
    assert paths[0] == "B"      # shares two concepts with A
    assert "C" not in paths     # shares nothing -> not a candidate


def test_cooccur_ranking_excludes_query_and_exclude_set(tmp_path):
    st = _cooc_store(tmp_path)
    ranking = _cooccur_ranking(st, "A", k=5, exclude={"B"}, scope=None, expand=False)
    paths = [p for p, _w in ranking or []]
    assert "A" not in paths     # never returns the query itself
    assert "B" not in paths     # honours the exclude set


def test_cooccur_ranking_abstains_when_query_absent(tmp_path):
    st = _cooc_store(tmp_path)
    assert _cooccur_ranking(st, "UNKNOWN", k=5, exclude=set(), scope=None) is None


def test_cooccur_gate_probe_receives_coverage_and_flatness(tmp_path, monkeypatch):
    # Phase-0 calibration hook (retrieval-gates spec): per-query signals are
    # emitted when the probe is set; the dormant gate must not fire.
    from silica.kernel.recall import relatedness

    st = _cooc_store(tmp_path)
    seen = []
    monkeypatch.setattr(relatedness, "COOCCUR_GATE_PROBE", seen.append)
    ranking = _cooccur_ranking(st, "A", k=5, exclude=set(), scope=None, expand=False)
    assert ranking is not None
    sig = seen[0]
    # top hit B matches beta+gamma but not alpha -> partial IDF-mass coverage
    assert 0.0 < sig["coverage"] < 1.0
    assert sig["flatness"] >= 1.0
    assert sig["fired"] is False


def test_cooccur_gate_abstains_below_threshold(tmp_path, monkeypatch):
    # Gate plumbing: with a frozen threshold above the fixture's coverage the
    # leg must abstain via the existing None protocol.
    from silica.kernel.recall import relatedness

    st = _cooc_store(tmp_path)
    monkeypatch.setattr(relatedness, "_COOCCUR_MIN_CONFIDENCE", 0.99)
    assert _cooccur_ranking(st, "A", k=5, exclude=set(), scope=None,
                            expand=False) is None


def test_cooccur_ranking_idf_beats_hub_over_rare_match(tmp_path):
    # 'hub' appears in every note (zero discriminating power); 'rare' is shared
    # by only the query and TWIN. IDF must rank the rare-sharing TWIN above a
    # note that merely piles on the ubiquitous hub concept.
    st = CooccurStore(path=tmp_path / "c.json", lang="english")
    st.upsert_note("Q",    build_contribution("Q", "hub rare"))
    st.upsert_note("TWIN", build_contribution("TWIN", "hub rare"))       # shares the rare concept
    st.upsert_note("HUB1", build_contribution("HUB1", "hub hub hub"))    # only the ubiquitous hub
    st.upsert_note("HUB2", build_contribution("HUB2", "hub filler"))     # keeps 'hub' near-ubiquitous

    ranking = _cooccur_ranking(st, "Q", k=5, exclude=set(), scope=None, expand=False)
    paths = [p for p, _w in ranking or []]
    assert paths and paths[0] == "TWIN"   # rare shared concept wins over hub breadth


def test_cooccur_ranking_expansion_reaches_associative_notes(tmp_path):
    # A is about alpha. Elsewhere alpha co-occurs strongly with omega.
    # A note about omega (but not alpha) is associatively related ONLY via expansion.
    st = CooccurStore(path=tmp_path / "c.json", lang="english")
    st.upsert_note("A", build_contribution("A", "alpha alpha"))
    st.upsert_note("BRIDGE", build_contribution("BRIDGE", "alpha omega"))  # links alpha<->omega
    st.upsert_note("OMEGA", build_contribution("OMEGA", "omega omega"))    # no alpha at all

    direct = _cooccur_ranking(st, "A", k=5, exclude={"BRIDGE"}, scope=None, expand=False)
    expanded = _cooccur_ranking(st, "A", k=5, exclude={"BRIDGE"}, scope=None, expand=True)

    direct_paths = [p for p, _w in direct or []]
    expanded_paths = [p for p, _w in expanded or []]
    assert "OMEGA" not in direct_paths      # no shared concept without expansion
    assert "OMEGA" in expanded_paths        # reached via alpha->omega neighbour edge


# ---------------------------------------------------------------------------
# Facade integration: related_notes
# ---------------------------------------------------------------------------

from silica.kernel.recall.relatedness import related_notes, RelatedNote


def test_related_notes_fuses_both_legs_with_evidence(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes("A", embed_store=es, cooccur_store=st, k=5)
    assert out and isinstance(out[0], RelatedNote)
    by_path = {r.path: r for r in out}
    # B is both A's nearest embed neighbour AND its strongest cooccur overlap
    assert "B" in by_path
    ev = by_path["B"].evidence
    assert any(e.startswith("embed:") for e in ev)
    assert any(e.startswith("cooccur:") for e in ev)


def test_related_notes_embedder_down_routes_on_cooccurrence(tmp_path):
    # No embed store at all -> embed leg abstains -> pure cooccurrence ranking.
    st = _cooc_store(tmp_path)
    out = related_notes("A", embed_store=None, cooccur_store=st, k=5)
    paths = [r.path for r in out]
    assert paths and paths[0] == "B"
    # provenance is cooccur-only when the embedder is down
    assert all(e.startswith("cooccur:") for r in out for e in r.evidence)


def test_related_notes_cooccur_empty_routes_on_embeddings(tmp_path):
    es = _embed_store(tmp_path)
    out = related_notes("A", embed_store=es, cooccur_store=None, k=5)
    paths = [r.path for r in out]
    assert paths and paths[0] == "B"
    assert all(e.startswith("embed:") for r in out for e in r.evidence)


def test_related_notes_both_abstain_returns_empty(tmp_path):
    out = related_notes("A", embed_store=None, cooccur_store=None, k=5)
    assert out == []


def test_related_notes_respects_k(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes("A", embed_store=es, cooccur_store=st, k=1)
    assert len(out) <= 1


def test_related_notes_never_returns_the_query(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes("A", embed_store=es, cooccur_store=st, k=10)
    assert "A" not in [r.path for r in out]


def test_related_notes_evidence_score_formats(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes("A", embed_store=es, cooccur_store=st, k=5)
    ev_all = [e for r in out for e in r.evidence]
    # embed evidence carries a 2-decimal cosine; cooccur carries an integer weight
    assert any(e.startswith("embed:0.") or e.startswith("embed:1.") for e in ev_all)
    assert any(e.startswith("cooccur:w") for e in ev_all)




def test_related_note_exposes_structured_per_leg_scores(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes("A", embed_store=es, cooccur_store=st, k=5)
    b = next(r for r in out if r.path == "B")
    # raw signals are accessible without parsing the evidence strings
    assert b.embed_score is not None and b.embed_score > 0.9
    assert b.cooccur_weight is not None and b.cooccur_weight > 0


# ---------------------------------------------------------------------------
# Fresh-query facade: related_notes_for_query (vec + text, no indexed path)
# ---------------------------------------------------------------------------

from silica.kernel.recall.relatedness import related_notes_for_query


def test_for_query_embed_only_ranks_by_vector(tmp_path):
    es = _embed_store(tmp_path)
    out = related_notes_for_query(query_vec=[0.9, 0.1], embed_store=es, k=5)
    assert out and out[0].path == "B"          # nearest to the query vector
    assert out[0].embed_score is not None and out[0].cooccur_weight is None
    assert all(e.startswith("embed:") for r in out for e in r.evidence)


def test_for_query_cooccur_only_from_text(tmp_path):
    st = _cooc_store(tmp_path)                  # A:alpha beta gamma, B:beta gamma delta
    out = related_notes_for_query(query_text="alpha beta gamma", cooccur_store=st, k=5)
    paths = [r.path for r in out]
    assert "A" in paths and "B" in paths        # both share concepts with the text
    assert all(r.embed_score is None for r in out)
    assert any(r.cooccur_weight for r in out)


def test_for_query_fuses_vec_and_text(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes_for_query(
        query_vec=es.get_vec("A"), query_text="alpha beta gamma",
        embed_store=es, cooccur_store=st, k=5, exclude={"A"},
    )
    b = next(r for r in out if r.path == "B")
    assert b.embed_score is not None and b.cooccur_weight is not None


def test_for_query_degenerate_vector_abstains_cooccur_carries(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes_for_query(
        query_vec=[0.0, 0.0], query_text="alpha beta gamma",
        embed_store=es, cooccur_store=st, k=5, exclude={"A"},
    )
    # zero query vector -> embed leg abstains rather than poisoning the fusion
    assert out
    assert all(r.embed_score is None for r in out)
    assert any(r.cooccur_weight for r in out)


def test_for_query_respects_exclude(tmp_path):
    st = _cooc_store(tmp_path)
    out = related_notes_for_query(query_text="alpha beta gamma", cooccur_store=st, k=5, exclude={"B"})
    assert "B" not in [r.path for r in out]



def test_for_query_both_absent_returns_empty(tmp_path):
    assert related_notes_for_query(k=5) == []
    assert related_notes_for_query(query_text="alpha", k=5) == []        # no cooccur store
    assert related_notes_for_query(query_vec=[1.0, 0.0], k=5) == []      # no embed store


# ---------------------------------------------------------------------------
# Boundary / robustness contract
# ---------------------------------------------------------------------------

# --- recall-outcome leg (phase 1 of `improve`) ------------------------------

def test_fuse_recall_rank_none_is_identical_to_before():
    from silica.kernel.recall.relatedness import _fuse
    embed = [("A", "A", 0.9), ("B", "B", 0.5)]
    cooc = [("B", 4.0), ("C", 1.0)]
    with_none = _fuse(embed, cooc, k=5, recall_rank=None)
    without_param = _fuse(embed, cooc, k=5)
    assert with_none == without_param


def test_fuse_includes_recall_leg_in_evidence():
    from silica.kernel.recall.relatedness import _fuse
    out = _fuse(None, None, recall_rank=[("B", 3.0)], k=5)
    assert out and out[0].path == "B"
    assert "recall:3" in out[0].evidence


def test_fuse_recall_leg_proposes_notes_absent_from_semantic_legs():
    from silica.kernel.recall.relatedness import _fuse
    embed = [("A", "A", 0.9), ("B", "B", 0.8)]
    out_without = _fuse(embed, None, k=10)
    assert "Z" not in [r.path for r in out_without]
    out_with = _fuse(embed, None, k=10, recall_rank=[("Z", 3.0)])
    assert "Z" in [r.path for r in out_with]


# ---------------------------------------------------------------------------
# Task 3.4 (perf/hot-paths): stem-postings inverted index backs _concept_idf
# and _rank_cooccur_from_profile's candidate scoring. This is a
# REFERENCE-EQUIVALENCE suite: the postings-backed implementations must
# return IDENTICAL values (idf map / ranked list / None) to the old full-scan
# implementations, reimplemented below as reference functions frozen at the
# pre-refactor behavior. The coverage/flatness/gate tail is exercised through
# the REAL `_rank_cooccur_from_profile`, not reimplemented, since it must stay
# byte-identical rather than merely equivalent.
# ---------------------------------------------------------------------------

import math as _math
import statistics as _statistics

from silica.kernel.recall import relatedness as _relatedness_mod
from silica.kernel.recall.paths import in_folder as _path_in_scope
from silica.kernel.recall.relatedness import _concept_idf, _rank_cooccur_from_profile

_snowball = __import__("snowballstemmer").stemmer("english").stemWord


def _old_concept_idf(cooccur_store, stems, *, scope):
    """Frozen copy of the pre-3.4 `_concept_idf`: full paths()/note_nodes() scan."""
    df: dict[str, int] = {}
    n = 0
    for path in cooccur_store.paths():
        if not _path_in_scope(path, scope):
            continue
        n += 1
        for stem in cooccur_store.note_nodes(path):
            if stem in stems:
                df[stem] = df.get(stem, 0) + 1
    return {stem: _math.log((n + 1) / c) for stem, c in df.items() if c > 0}


def _old_rank_cooccur_from_profile(cooccur_store, profile, *, k, blocked, scope):
    """Frozen copy of the pre-3.4 `_rank_cooccur_from_profile`: two full scans.

    Reads `_COOCCUR_MIN_CONFIDENCE` off the live module (not a bound import) so
    a test that monkeypatches the threshold affects this reference the same
    way it affects the real function under test.
    """
    if not profile:
        return None
    idf = _old_concept_idf(cooccur_store, set(profile), scope=scope)
    note_scores: dict[str, float] = {}
    for path in cooccur_store.paths():
        if path in blocked or not _path_in_scope(path, scope):
            continue
        overlap = 0.0
        for stem, count in cooccur_store.note_nodes(path).items():
            weight = profile.get(stem)
            if weight:
                overlap += weight * count * idf.get(stem, 0.0)
        if overlap > 0.0:
            note_scores[path] = overlap
    if not note_scores:
        return None
    ranked = sorted(note_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    total_mass = sum(w * idf.get(s, 0.0) for s, w in profile.items())
    top_stems = set(cooccur_store.note_nodes(ranked[0][0]))
    matched = sum(w * idf.get(s, 0.0) for s, w in profile.items() if s in top_stems)
    coverage = (matched / total_mass) if total_mass > 0 else 0.0
    scores = [s for _p, s in ranked]
    flatness = scores[0] / _statistics.median(scores)
    fired = coverage < _relatedness_mod._COOCCUR_MIN_CONFIDENCE
    if fired:
        return None
    return ranked


def _dozen_notes_store(tmp_path) -> CooccurStore:
    """A dozen notes with varied stems/counts, some in a subfolder (scope tests).

    Profile stems used across the cases below: alpha, beta, gamma, zeta.
      root1: alpha beta gamma delta epsilon  (full spread)
      root2: alpha beta                      (partial, low count)
      root3: gamma delta epsilon zeta        (partial, no alpha/beta)
      root4: alpha x3                        (single stem, high count)
      root5: beta x3                         (single stem, high count)
      root6: eta theta iota                  (disjoint -> never a candidate)
      root7: mu nu                           (disjoint -> never a candidate)
      root8: alpha beta gamma delta          (near-full spread)
      folder/n1: alpha beta gamma            (subfolder, full profile overlap)
      folder/n2: alpha zeta                  (subfolder, partial)
      folder/n3: kappa lambda                (subfolder, disjoint)
      folder/n4: beta gamma                  (subfolder, partial)
    """
    st = CooccurStore(path=tmp_path / "dozen.json", lang="english")
    st.upsert_note("root1", build_contribution("root1", "alpha beta gamma delta epsilon"))
    st.upsert_note("root2", build_contribution("root2", "alpha beta"))
    st.upsert_note("root3", build_contribution("root3", "gamma delta epsilon zeta"))
    st.upsert_note("root4", build_contribution("root4", "alpha alpha alpha"))
    st.upsert_note("root5", build_contribution("root5", "beta beta beta"))
    st.upsert_note("root6", build_contribution("root6", "eta theta iota"))
    st.upsert_note("root7", build_contribution("root7", "mu nu"))
    st.upsert_note("root8", build_contribution("root8", "alpha beta gamma delta"))
    st.upsert_note("folder/n1", build_contribution("n1", "alpha beta gamma"))
    st.upsert_note("folder/n2", build_contribution("n2", "alpha zeta"))
    st.upsert_note("folder/n3", build_contribution("n3", "kappa lambda"))
    st.upsert_note("folder/n4", build_contribution("n4", "beta gamma"))
    return st


def _main_profile() -> dict[str, float]:
    return {
        _snowball("alpha"): 2.0,
        _snowball("beta"): 1.0,
        _snowball("gamma"): 1.5,
        _snowball("zeta"): 3.0,
    }


def test_concept_idf_reference_equivalence_scope_none_and_scoped(tmp_path):
    st = _dozen_notes_store(tmp_path)
    stems = {_snowball(w) for w in ("alpha", "beta", "gamma", "zeta", "missingstem")}
    for scope in (None, "folder"):
        new = _concept_idf(st, stems, scope=scope)
        old = _old_concept_idf(st, stems, scope=scope)
        assert new == old


def test_rank_cooccur_reference_equivalence_scope_none(tmp_path):
    st = _dozen_notes_store(tmp_path)
    profile = _main_profile()
    new = _rank_cooccur_from_profile(st, profile, k=100, blocked=set(), scope=None)
    old = _old_rank_cooccur_from_profile(st, profile, k=100, blocked=set(), scope=None)
    assert new is not None
    assert new == old


def test_rank_cooccur_reference_equivalence_real_scope(tmp_path):
    st = _dozen_notes_store(tmp_path)
    profile = _main_profile()
    new = _rank_cooccur_from_profile(st, profile, k=100, blocked=set(), scope="folder")
    old = _old_rank_cooccur_from_profile(st, profile, k=100, blocked=set(), scope="folder")
    assert new is not None
    assert new == old
    # scope filtering actually took effect (root-level notes excluded)
    assert all(p.startswith("folder/") for p, _s in new)


def test_rank_cooccur_reference_equivalence_nonempty_blocked_incl_query(tmp_path):
    st = _dozen_notes_store(tmp_path)
    profile = _main_profile()
    blocked = {"root1", "root4"}  # "root1" stands in for the query path
    new = _rank_cooccur_from_profile(st, profile, k=100, blocked=blocked, scope=None)
    old = _old_rank_cooccur_from_profile(st, profile, k=100, blocked=blocked, scope=None)
    assert new is not None
    assert new == old
    assert "root1" not in [p for p, _s in new]
    assert "root4" not in [p for p, _s in new]


def test_rank_cooccur_reference_equivalence_empty_profile_is_none(tmp_path):
    st = _dozen_notes_store(tmp_path)
    new = _rank_cooccur_from_profile(st, {}, k=10, blocked=set(), scope=None)
    old = _old_rank_cooccur_from_profile(st, {}, k=10, blocked=set(), scope=None)
    assert new is None
    assert new == old


def test_rank_cooccur_reference_equivalence_absent_stems_is_none(tmp_path):
    st = _dozen_notes_store(tmp_path)
    profile = {_snowball("zzznotinstore"): 1.0, _snowball("wontmatch"): 2.0}
    new = _rank_cooccur_from_profile(st, profile, k=10, blocked=set(), scope=None)
    old = _old_rank_cooccur_from_profile(st, profile, k=10, blocked=set(), scope=None)
    assert new is None
    assert new == old


def test_rank_cooccur_reference_equivalence_ties_broken_by_path(tmp_path):
    st = CooccurStore(path=tmp_path / "ties.json", lang="english")
    st.upsert_note("zeta_note", build_contribution("zeta_note", "alpha"))
    st.upsert_note("alpha_note", build_contribution("alpha_note", "alpha"))
    profile = {_snowball("alpha"): 1.0}
    new = _rank_cooccur_from_profile(st, profile, k=10, blocked=set(), scope=None)
    old = _old_rank_cooccur_from_profile(st, profile, k=10, blocked=set(), scope=None)
    assert new == old
    # equal overlap score -> tie broken by path ascending
    assert [p for p, _s in new] == ["alpha_note", "zeta_note"]


def test_rank_cooccur_reference_equivalence_k_smaller_than_candidates(tmp_path):
    st = _dozen_notes_store(tmp_path)
    profile = _main_profile()
    new = _rank_cooccur_from_profile(st, profile, k=2, blocked=set(), scope=None)
    old = _old_rank_cooccur_from_profile(st, profile, k=2, blocked=set(), scope=None)
    assert new is not None
    assert len(new) == 2
    assert new == old


def test_rank_cooccur_reference_equivalence_k_larger_than_candidates(tmp_path):
    st = _dozen_notes_store(tmp_path)
    profile = _main_profile()
    new = _rank_cooccur_from_profile(st, profile, k=1000, blocked=set(), scope=None)
    old = _old_rank_cooccur_from_profile(st, profile, k=1000, blocked=set(), scope=None)
    assert new is not None
    assert new == old


def test_rank_cooccur_reference_equivalence_gate_not_fired(tmp_path, monkeypatch):
    # Dormant gate (default 0.0): must not fire, new == old, non-None.
    monkeypatch.setattr(_relatedness_mod, "_COOCCUR_MIN_CONFIDENCE", 0.0)
    st = _dozen_notes_store(tmp_path)
    profile = _main_profile()
    new = _rank_cooccur_from_profile(st, profile, k=100, blocked=set(), scope=None)
    old = _old_rank_cooccur_from_profile(st, profile, k=100, blocked=set(), scope=None)
    assert new is not None
    assert new == old


def test_rank_cooccur_reference_equivalence_gate_fires(tmp_path, monkeypatch):
    # Coverage is a ratio <= 1.0 (matched/total_mass, matched <= total_mass);
    # a threshold well above 1.0 is guaranteed to fire regardless of the
    # fixture's actual coverage value.
    monkeypatch.setattr(_relatedness_mod, "_COOCCUR_MIN_CONFIDENCE", 999.0)
    st = _dozen_notes_store(tmp_path)
    profile = _main_profile()
    new = _rank_cooccur_from_profile(st, profile, k=100, blocked=set(), scope=None)
    old = _old_rank_cooccur_from_profile(st, profile, k=100, blocked=set(), scope=None)
    assert new is None
    assert new == old


def test_rank_cooccur_postings_invalidated_on_upsert_and_delete(tmp_path):
    st = CooccurStore(path=tmp_path / "inv.json", lang="english")
    st.upsert_note("A", build_contribution("A", "alpha beta"))
    st.upsert_note("B", build_contribution("B", "gamma"))
    profile = {_snowball("alpha"): 1.0}

    ranked = _rank_cooccur_from_profile(st, profile, k=10, blocked=set(), scope=None)
    assert [p for p, _s in ranked] == ["A"]

    # upsert moves A off 'alpha' -> A drops out of the ranking entirely
    st.upsert_note("A", build_contribution("A", "gamma delta"))
    ranked2 = _rank_cooccur_from_profile(st, profile, k=10, blocked=set(), scope=None)
    assert ranked2 is None

    # a fresh note carrying 'alpha' becomes the new candidate
    st.upsert_note("C", build_contribution("C", "alpha alpha"))
    ranked3 = _rank_cooccur_from_profile(st, profile, k=10, blocked=set(), scope=None)
    assert [p for p, _s in ranked3] == ["C"]

    st.delete_note("C")
    ranked4 = _rank_cooccur_from_profile(st, profile, k=10, blocked=set(), scope=None)
    assert ranked4 is None


# ---------------------------------------------------------------------------
# BM25 tf term (CONFIG.cooccur_bm25, docs/specs/cooccur-scoring.md fase 1)
# ---------------------------------------------------------------------------

def _nodes(**counts: int) -> dict:
    """A hand-built contribution: exact stem counts, so tf and note length are
    controlled instead of inferred from prose."""
    return {"nodes": {s: {"label": s, "count": c} for s, c in counts.items()}, "edges": []}


def _bm25_store(tmp_path, name: str = "bm25.json") -> CooccurStore:
    """Two notes sharing one stem: `long` says it 3 times in 200 stems, `short`
    twice in 5. Raw tf ranks long first; saturation plus length normalisation
    ranks short first. avgdl = 102.5.
    """
    st = CooccurStore(path=tmp_path / name, lang="english")
    st.upsert_note("long", _nodes(graph=3, filler=197))
    st.upsert_note("short", _nodes(graph=2, filler=3))
    return st


def test_cooccur_bm25_flag_off_keeps_the_linear_tf_ranking(tmp_path, monkeypatch):
    """Default path is frozen: score stays weight * tf * idf, so the two notes
    keep the raw-count order AND the raw-count ratio (3/2). Flipping the flag on
    saturates the tf and normalises by length, which reverses them.
    """
    from silica.config import CONFIG

    st = _bm25_store(tmp_path)
    profile = {"graph": 1.0}

    monkeypatch.setattr(CONFIG, "cooccur_bm25", False)
    ranked = _rank_cooccur_from_profile(st, profile, k=10, blocked=set(), scope=None)
    assert [p for p, _s in ranked] == ["long", "short"]
    # idf is common to both candidates, so the score ratio IS the tf ratio.
    assert ranked[0][1] / ranked[1][1] == 3 / 2

    monkeypatch.setattr(CONFIG, "cooccur_bm25", True)
    ranked_bm25 = _rank_cooccur_from_profile(st, profile, k=10, blocked=set(), scope=None)
    assert [p for p, _s in ranked_bm25] == ["short", "long"]


def test_cooccur_bm25_lengths_are_per_store_and_current(tmp_path, monkeypatch):
    """avgdl belongs to the store it was measured on, and to its current content.

    Two failure modes, one invariant: a cache shared across stores would lend
    this store the OTHER vault's avgdl (the memory lane passes a different store
    to this same seam), and a cache that outlives a write would keep an avgdl the
    vault no longer has.
    """
    from silica.config import CONFIG

    monkeypatch.setattr(CONFIG, "cooccur_bm25", True)
    profile = {"graph": 1.0}

    # A different store, whose notes are ~100x longer (avgdl 10000).
    other = CooccurStore(path=tmp_path / "other.json", lang="english")
    other.upsert_note("huge", _nodes(graph=1, filler=9999))
    assert _rank_cooccur_from_profile(other, profile, k=10, blocked=set(), scope=None)

    # Queried after it, this store still ranks by ITS OWN avgdl (102.5). Under
    # `other`'s avgdl the length term would vanish and long would win on tf.
    st = _bm25_store(tmp_path)
    ranked = _rank_cooccur_from_profile(st, profile, k=10, blocked=set(), scope=None)
    assert [p for p, _s in ranked] == ["short", "long"]

    # A write moves avgdl to ~10068: the length penalty on long all but
    # disappears and the order flips back. A stale cache would not notice.
    st.upsert_note("bulk", _nodes(filler=30000))
    ranked_after = _rank_cooccur_from_profile(st, profile, k=10, blocked=set(), scope=None)
    assert [p for p, _s in ranked_after] == ["long", "short"]


def _old_rank_bm25(cooccur_store, profile, *, k, blocked, scope):
    """Note-major BM25 reference: the shape `_rank_cooccur_from_profile` had
    before it went stem-major, with the BM25 tf term inlined.

    The stem-major rewrite memoizes the length norm per note instead of
    recomputing it once per candidate, so this reference is what pins that the
    memo returns the same number the inline expression did.
    """
    from silica.kernel.recall.relatedness import BM25_B, BM25_K1

    if not profile:
        return None
    idf = _concept_idf(cooccur_store, set(profile), scope=scope)
    lens, avgdl = cooccur_store.doc_lengths()
    note_scores: dict[str, float] = {}
    for path in cooccur_store.paths():
        if path in blocked or not _path_in_scope(path, scope):
            continue
        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * (lens.get(path, 0) or 1) / avgdl)
        overlap = 0.0
        nodes = cooccur_store.note_nodes(path)
        for stem, weight in profile.items():
            if not weight:
                continue
            tf = nodes.get(stem)
            if tf:
                overlap += weight * (tf * (BM25_K1 + 1.0) / (tf + norm)) * idf.get(stem, 0.0)
        if overlap > 0.0:
            note_scores[path] = overlap
    if not note_scores:
        return None
    return sorted(note_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def test_rank_cooccur_bm25_reference_equivalence_across_a_varied_store(tmp_path, monkeypatch):
    """The BM25 arm survives the stem-major rewrite bit-for-bit.

    Twelve notes of unequal length so the per-note length norm actually varies —
    a memo that leaked one note's norm onto another would move the scores here.
    """
    from silica.config import CONFIG

    monkeypatch.setattr(CONFIG, "cooccur_bm25", True)
    st = _dozen_notes_store(tmp_path)
    profile = _main_profile()

    for scope, blocked in ((None, set()), ("folder", set()), (None, {"root1", "root4"})):
        new = _rank_cooccur_from_profile(st, profile, k=100, blocked=blocked, scope=scope)
        old = _old_rank_bm25(st, profile, k=100, blocked=blocked, scope=scope)
        assert new is not None
        assert new == old, f"bm25 mismatch at scope={scope!r} blocked={blocked!r}"


def test_rank_cooccur_skips_zero_weight_stems_like_the_note_major_form(tmp_path):
    """A zero-weight profile stem contributes nothing and cannot resurrect a
    note that matches on nothing else — the `if not weight` guard moved from the
    inner loop to the outer one, so it needs its own pin."""
    st = _dozen_notes_store(tmp_path)
    profile = dict(_main_profile())
    profile[_snowball("eta")] = 0.0  # only root6 carries it, and nothing else

    ranked = _rank_cooccur_from_profile(st, profile, k=100, blocked=set(), scope=None)
    assert "root6" not in [p for p, _s in ranked]
    assert ranked == _old_rank_cooccur_from_profile(st, profile, k=100, blocked=set(), scope=None)


def test_related_notes_many_equals_the_per_note_facade(tmp_path):
    """One blocked matmul for the embed leg must reproduce the per-note path
    exactly — same pool, same abstention, same fusion (ADR-0029)."""
    from silica.kernel.recall.relatedness import related_notes_many
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    paths = sorted(set(st.paths()) | set(es.paths()))
    many = related_notes_many(paths, embed_store=es, cooccur_store=st, k=5)
    for p in paths:
        single = related_notes(p, embed_store=es, cooccur_store=st, k=5)
        assert [(r.path, r.evidence) for r in many[p]] == [(r.path, r.evidence) for r in single], p
    # an empty embed store abstains on that leg for every note, like the per-note path
    cooc_only = related_notes_many(paths, embed_store=None, cooccur_store=st, k=5)
    for p in paths:
        assert [r.path for r in cooc_only[p]] == [r.path for r in related_notes(p, cooccur_store=st, k=5)]
