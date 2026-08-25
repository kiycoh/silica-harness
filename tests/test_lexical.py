import math
from difflib import SequenceMatcher

from silica.kernel.recall.lexical import LexicalStore, _BM25_K1, _BM25_B, _FUZZY_MIN, _tokens


def _reference_rank(store: LexicalStore, query: str, k: int = 25) -> list[tuple[str, float]]:
    """Old (pre-inverted-index) rank() logic, reimplemented verbatim for the
    equivalence test: full scan of every doc for df, BM25 over ALL docs, and
    a plain SequenceMatcher.ratio() fuzzy leg over every name."""
    if not store._docs:
        return []
    q_terms = _tokens(query)
    n = len(store._docs)
    avgdl = (sum(store._len.values()) / n) if n else 0.0
    df: dict[str, int] = {}
    for term in set(q_terms):
        df[term] = sum(1 for tf in store._docs.values() if term in tf)

    bm25: dict[str, float] = {}
    for path, tf in store._docs.items():
        dl = store._len[path] or 1
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f or df.get(term, 0) == 0:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = f + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / (avgdl or 1))
            score += idf * (f * (_BM25_K1 + 1)) / denom
        if score > 0.0:
            bm25[path] = score
    bm25_ranked = sorted(bm25.items(), key=lambda kv: (-kv[1], kv[0]))

    ql = query.strip().lower()
    fuzzy: dict[str, float] = {}
    for path, name in store._name.items():
        r = SequenceMatcher(None, ql, name.lower()).ratio()
        if r >= _FUZZY_MIN:
            fuzzy[path] = r
    fuzzy_ranked = sorted(fuzzy.items(), key=lambda kv: (-kv[1], kv[0]))

    fused: dict[str, float] = {}
    for ranking in (bm25_ranked, fuzzy_ranked):
        for rank, (path, _s) in enumerate(ranking):
            fused[path] = fused.get(path, 0.0) + 1.0 / (60 + rank + 1)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def _build_reference_corpus() -> LexicalStore:
    """A dozen docs: varied shared/rare terms, a fuzzy-close title with no
    shared tokens, and zero-hit docs — enough surface to exercise every
    equivalence case in the brief."""
    s = LexicalStore()
    s.upsert("notes/apollo", "Apollo 11", "The Apollo 11 moon landing in 1969.")
    s.upsert("notes/apollo13", "Apollo 13", "Apollo 13 was a near-disaster in space.")
    s.upsert("notes/cooking", "Risotto", "Stir the rice slowly with stock.")
    s.upsert("notes/meeting1", "Meeting notes", "The meeting was about the meeting agenda meeting.")
    s.upsert("notes/zbigniew", "Zbigniew", "Zbigniew attended the meeting.")
    s.upsert("notes/kubernetes", "Kubernetes", "container orchestration platform")
    # Fuzzy-close title, zero shared BM25 tokens with the "kubernets" query.
    s.upsert("notes/kubernetsy", "Kubernetsy", "an unrelated made-up word, no other overlap")
    s.upsert("notes/moonwalk", "Moonwalk", "Astronauts walked on the moon during Apollo missions.")
    s.upsert("notes/space", "Space history", "A history of space exploration and moon landings.")
    s.upsert("notes/empty1", "Untouched note", "nothing relevant here at all")
    s.upsert("notes/empty2", "Another blank", "completely unrelated filler text")
    s.upsert("notes/tie_a", "Tie Alpha", "shared unique wordxyz appears here once")
    s.upsert("notes/tie_b", "Tie Beta", "shared unique wordxyz appears here once")
    return s


_EQUIVALENCE_QUERIES = [
    ("apollo moon landing", 5),   # multi-term query
    ("apollo", 25),               # multi-term query, larger k
    ("nonexistentxyzterm", 5),    # query term absent from all docs
    ("kubernets", 5),             # fuzzy-only hit path (also a real BM25 miss)
    ("wordxyz", 5),               # ties (tie_a vs tie_b, same score)
    ("meeting", 1),               # k smaller than match count
    ("zbigniew meeting apollo moon", 100),  # k larger than match count
    ("", 5),                      # empty query
]


def test_upsert_rank_and_remove():
    s = LexicalStore()
    s.upsert("notes/apollo", "Apollo 11", "The Apollo 11 moon landing in 1969.")
    s.upsert("notes/cooking", "Risotto", "Stir the rice slowly with stock.")
    ranked = s.rank("apollo moon landing", k=5)
    assert ranked[0][0] == "notes/apollo"           # rare-token query hits
    s.remove("notes/apollo")
    assert all(p != "notes/apollo" for p, _ in s.rank("apollo", k=5))


def test_empty_index_abstains():
    assert LexicalStore().rank("anything") == []    # abstain -> RRF fuses fewer legs


def test_rare_proper_noun_beats_common_words():
    s = LexicalStore()
    s.upsert("n/a", "Zbigniew", "Zbigniew attended the meeting.")
    s.upsert("n/b", "Meeting notes", "The meeting was about the meeting agenda meeting.")
    ranked = s.rank("Zbigniew", k=5)
    assert ranked[0][0] == "n/a"                    # proper noun, BM25 idf lift


def test_fuzzy_title_match_survives_typo():
    s = LexicalStore()
    s.upsert("n/a", "Kubernetes", "container orchestration platform")
    ranked = s.rank("kubernets", k=5)               # one-char typo on the title
    assert ranked and ranked[0][0] == "n/a"


def test_save_load_roundtrip(tmp_path):
    """save() persists and load() reconstitutes an equivalent, queryable index."""
    idx = tmp_path / "lexical.json"
    s = LexicalStore(idx)
    s.upsert("notes/apollo", "Apollo 11", "The Apollo 11 moon landing in 1969.")
    s.upsert("notes/cooking", "Risotto", "Stir the rice slowly with stock.")
    s.save()
    reloaded = LexicalStore.load(idx)
    assert len(reloaded) == 2
    assert reloaded.rank("apollo moon landing", k=5)[0][0] == "notes/apollo"


def test_corrupt_index_quarantines_and_abstains(tmp_path):
    """A corrupt index file loads as an empty, abstaining store (not a crash)."""
    idx = tmp_path / "lexical.json"
    idx.write_bytes(b"{ not valid json ]")
    store = LexicalStore.load(idx)
    assert len(store) == 0
    assert store.rank("anything") == []


def test_rank_matches_reference_implementation():
    """CENTRAL equivalence test: the inverted-index rank() must byte-for-byte
    match the old full-scan reference across multi-term, absent-term,
    fuzzy-only, tie, and k</>match-count cases."""
    s = _build_reference_corpus()
    for query, k in _EQUIVALENCE_QUERIES:
        assert s.rank(query, k=k) == _reference_rank(s, query, k=k), (query, k)


def test_rank_matches_reference_after_save_load(tmp_path):
    """Derived indexes (_postings/_name_lower) must be rebuilt on load() so
    rank() after a save->load round trip still matches the reference."""
    idx = tmp_path / "lexical.json"
    s = _build_reference_corpus()
    s._path = idx
    s.save()
    reloaded = LexicalStore.load(idx)
    for query, k in _EQUIVALENCE_QUERIES:
        assert reloaded.rank(query, k=k) == _reference_rank(reloaded, query, k=k), (query, k)


def test_fuzzy_only_hit_with_no_shared_bm25_term():
    """A title fuzzy-close to the query but sharing no BM25 token must still
    surface — the fuzzy leg must not be restricted to BM25 candidates."""
    s = LexicalStore()
    s.upsert("n/a", "Kubernetsy", "an unrelated made-up word, no other overlap")
    s.upsert("n/b", "Something else entirely", "totally different content")
    ranked = s.rank("kubernets", k=5)
    assert ranked and ranked[0][0] == "n/a"


def test_upsert_overwrite_purges_stale_postings():
    """Re-upserting a path with new terms must drop postings for terms the
    old version had but the new one doesn't."""
    s = LexicalStore()
    s.upsert("n/a", "Doc A", "alpha bravo charlie")
    assert "alpha" in s._postings and "n/a" in s._postings["alpha"]
    s.upsert("n/a", "Doc A", "delta echo foxtrot")
    assert "alpha" not in s._postings or "n/a" not in s._postings.get("alpha", {})
    assert "delta" in s._postings and "n/a" in s._postings["delta"]
    assert s.rank("alpha", k=5) == []
    assert s.rank("delta", k=5)[0][0] == "n/a"


def test_remove_cleans_postings_and_name_lower():
    """remove() must purge the path from every posting list and from
    _name_lower, not just from _docs/_len/_name."""
    s = LexicalStore()
    s.upsert("n/a", "Apollo 11", "moon landing apollo")
    s.upsert("n/b", "Risotto", "rice stock")
    s.remove("n/a")
    assert "n/a" not in s._name_lower
    for term, postings in s._postings.items():
        assert "n/a" not in postings
    assert all(p != "n/a" for p, _ in s.rank("apollo", k=5))


# --- G3: idf for the window scan (rerank.window_weights) ---

def test_query_idf_rare_above_common_unknown_omitted():
    s = LexicalStore()
    s.upsert("a", "A", "einstein relativity")
    s.upsert("b", "B", "relativity waves")
    s.upsert("c", "C", "relativity fields")
    w = s.query_idf({"relativity", "einstein", "neutrino"})
    assert w["einstein"] > w["relativity"]
    assert "neutrino" not in w   # df=0 = unknown-or-stopword -> neutral 1.0 in the scan


def test_query_idf_empty_store_abstains():
    assert LexicalStore().query_idf({"anything"}) == {}
