"""Tests for the co-occurrence graph kernel (kernel/cooccurrence.py)."""
from __future__ import annotations

from pathlib import Path



from silica.kernel.recall.cooccurrence import tokenize


def test_tokenize_breaks_sentences_on_terminators():
    text = "Prima frase. Seconda frase! Terza?\nQuarta frase"
    sents = tokenize(text, stem_lang="italian", stopword_lang="italian")
    assert len(sents) == 4
    assert [s for (_t, s) in sents[0]] == ["prima", "frase"]


def test_tokenize_lowercases_and_drops_short_and_stopwords_english():
    # "the" is a stopword, "a" and "is" too; "of" stopword; "ai" is < 3 chars
    # stopword_lang pinned explicitly: the assertion depends on the English
    # stopword set specifically, not on whatever language.detect() picks for
    # this tiny sample.
    sents = tokenize("The cat is on a mat", stem_lang="english", stopword_lang="english")
    # one sentence, stopwords/short removed, remaining stemmed (cat, mat)
    stems = [stem for sent in sents for (stem, _surface) in sent]
    assert "cat" in stems
    assert "mat" in stems
    assert all(s not in stems for s in ("the", "is", "on", "a"))


def snow_stem_it(word: str) -> str:
    import snowballstemmer
    return snowballstemmer.stemmer("italian").stemWord(word)


def test_tokenize_collapses_italian_inflections():
    sents = tokenize("La rete e le reti neurali", stem_lang="italian", stopword_lang="italian")
    stems = [stem for sent in sents for (stem, _surface) in sent]
    # rete and reti must collapse to the same stem
    assert stems.count(snow_stem_it("rete")) >= 1
    # both inflections map to one stem
    assert snow_stem_it("rete") == snow_stem_it("reti")


def test_tokenize_keeps_surface_form():
    sents = tokenize("Neural networks", stem_lang="english", stopword_lang="english")
    surfaces = [surface for sent in sents for (_stem, surface) in sent]
    assert "neural" in surfaces  # surface is lowercased original token


def test_tokenize_stopword_lang_explicit_overrides_detection():
    # "della" is an Italian stopword; pinning stopword_lang="italian" filters
    # it even though stem_lang is "english" (store's frozen stemmer).
    sents = tokenize("della rete", stem_lang="english", stopword_lang="italian")
    stems = [stem for sent in sents for (stem, _surface) in sent]
    assert "della" not in stems


def test_tokenize_stopword_lang_none_detects_from_text():
    # stopword_lang=None (default) -> language.detect(text); Italian function
    # words get dropped even when stem_lang is frozen to "english".
    sents = tokenize(
        "La rete della azienda migliora molto il lavoro del team.",
        stem_lang="english",
    )
    stems = [stem for sent in sents for (stem, _surface) in sent]
    assert "della" not in stems
    assert "il" not in stems


from silica.kernel.recall.cooccurrence import build_contribution


def _edge_weight(contribution, a, b):
    """Sum directed edge weight a->b in a contribution's edge list."""
    return sum(w for (f, t, w) in contribution["edges"] if f == a and t == b)


def test_build_contribution_narrative_adjacent_weight_3():
    # four distinct content words, no stopwords, single sentence
    c = build_contribution("N", "alpha beta gamma delta", lang="english")
    st = __import__("snowballstemmer").stemmer("english").stemWord
    # adjacent pair alpha->beta has narrative weight 3
    assert _edge_weight(c, st("alpha"), st("beta")) == 3


def test_build_contribution_gap_scan_decays_3_2_1():
    c = build_contribution("N", "alpha beta gamma delta", lang="english")
    st = __import__("snowballstemmer").stemmer("english").stemWord
    a, b, g, d = st("alpha"), st("beta"), st("gamma"), st("delta")
    # delta links back: to gamma (dist1=3), beta (dist2=2), alpha (dist3=1)
    assert _edge_weight(c, g, d) == 3
    assert _edge_weight(c, b, d) == 2
    assert _edge_weight(c, a, d) == 1


def test_build_contribution_no_edge_across_sentence_boundary():
    c = build_contribution("N", "alpha beta. gamma delta", lang="english")
    st = __import__("snowballstemmer").stemmer("english").stemWord
    # beta (end of sentence 1) must NOT link to gamma (start of sentence 2)
    assert _edge_weight(c, st("beta"), st("gamma")) == 0


def test_build_contribution_math_never_becomes_nodes():
    """C1: LaTeX must be stripped before tokenization — `frac`/`nabla` were
    real nodes in the vault graph before the kernel/text seam."""
    c = build_contribution(
        "Gradiente",
        "La discesa $\\nabla f = \\frac{a}{b}$ converge sempre verso il minimo.\n\n"
        "$$E = \\sum_i \\epsilon_i$$\n\nChiude con \\alpha residuo.",
        lang="italian",
    )
    for stem in c["nodes"]:
        assert not stem.startswith(("frac", "nabla", "sum", "epsilon", "alpha")), (
            f"math token leaked into the graph: {stem!r}"
        )
    assert any(s.startswith("convergere"[:7]) or s.startswith("converg") for s in c["nodes"])


def test_build_contribution_nodes_have_label_and_count():
    c = build_contribution("N", "alpha alpha beta", lang="english")
    st = __import__("snowballstemmer").stemmer("english").stemWord
    assert c["nodes"][st("alpha")]["count"] == 2
    assert c["nodes"][st("alpha")]["label"] == "alpha"


def test_build_contribution_strip_fences_flag():
    """Prose vaults drop ```code``` identifiers as noise; code vaults keep them."""
    body = "alpha beta\n```python\nzzident zzident\n```\ngamma delta"
    st = __import__("snowballstemmer").stemmer("english").stemWord
    kept = build_contribution("N", body, strip_fences=False)  # legacy default
    dropped = build_contribution("N", body, strip_fences=True)
    assert st("zzident") in kept["nodes"]
    assert st("zzident") not in dropped["nodes"]
    assert st("gamma") in dropped["nodes"]  # prose around the fence survives


def test_build_contribution_strips_inline_code_and_urls():
    st = __import__("snowballstemmer").stemmer("english").stemWord
    body = "vedi `zzident` e https://example.com/x?q=zznoise poi gamma delta"
    c = build_contribution("N", body, strip_fences=True)
    assert st("zzident") not in c["nodes"]  # inline code (prose gate)
    assert st("zznoise") not in c["nodes"]  # url query token
    assert st("gamma") in c["nodes"]
    # URLs are stripped unconditionally, even on a code vault (fences kept)
    c2 = build_contribution("N", "alpha https://x.io/zzurl beta", strip_fences=False)
    assert st("zzurl") not in c2["nodes"]
    assert st("alpha") in c2["nodes"]


def test_build_contribution_strips_schemeless_urls_and_block_ids():
    st = __import__("snowballstemmer").stemmer("english").stemWord
    body = "alpha www.zzsite.org/path beta ^zzblockid delta"
    c = build_contribution("N", body, strip_fences=False)  # unconditional
    assert st("zzsite") not in c["nodes"]     # schemeless www. url
    assert st("zzblockid") not in c["nodes"]  # excalidraw/obsidian block ref
    assert st("alpha") in c["nodes"] and st("delta") in c["nodes"]


def test_build_contribution_excludes_excalidraw_drawings():
    drawing = (
        "---\ntags: excalidraw\nexcalidraw-plugin: parsed\n---\n"
        "# Excalidraw Data\n## Text Elements\nSoulsLightGame ^g37aKW7i\n+create() ^D3qzgBfU\n"
    )
    c = build_contribution("Drawing 2026", drawing)
    assert c["nodes"] == {} and c["edges"] == []
    # a prose note that merely mentions excalidraw (no plugin marker) is kept
    ok = build_contribution("N", "uso excalidraw come strumento di disegno utile")
    assert ok["nodes"]


def test_strip_fences_keyed_on_manifest_sources(monkeypatch):
    from silica.kernel.recall import cooccurrence
    from silica.kernel import vault_manifest

    def _manifest(sources):
        return vault_manifest.VaultManifest(sources=sources)

    monkeypatch.setattr(vault_manifest, "get_active_manifest", lambda: _manifest(("prose",)))
    assert cooccurrence._strip_fences_for_active_vault() is True
    monkeypatch.setattr(vault_manifest, "get_active_manifest", lambda: _manifest(("prose", "code")))
    assert cooccurrence._strip_fences_for_active_vault() is False


from silica.kernel.recall.cooccurrence import CooccurStore


def test_store_empty_on_missing_file(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    assert len(store) == 0


def test_store_upsert_and_len(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    assert len(store) == 1
    assert "A" in store.paths()


def test_store_roundtrip(tmp_path):
    idx = tmp_path / "cooc.json"
    store = CooccurStore(path=idx, lang="english")
    store.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    store.save()

    store2 = CooccurStore(path=idx)
    assert len(store2) == 1
    assert store2.lang == "english"


def test_store_delete_note(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    store.delete_note("A")
    assert len(store) == 0


def test_neighbors_returns_sorted_candidates(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    # gamma co-occurs strongly with beta (dist1) and weaker with alpha (dist2)
    store.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    cands = store.neighbors("gamma", k=5)
    assert cands[0]["evidence"] == "cooccur"
    labels = [c["concept"] for c in cands]
    # beta (weight 3) ranks above alpha (weight 2)
    assert labels.index("beta") < labels.index("alpha")


def test_neighbors_undirected_sums_both_directions(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))   # alpha->beta w3
    store.upsert_note("B", build_contribution("B", "beta alpha"))   # beta->alpha w3
    cands = store.neighbors("alpha", k=5)
    beta = next(c for c in cands if c["concept"] == "beta")
    assert beta["weight"] == 6  # 3 + 3, undirected


def test_neighbors_respects_k(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta gamma delta epsilon"))
    assert len(store.neighbors("alpha", k=2)) <= 2


def test_neighbors_missing_concept_returns_empty(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    assert store.neighbors("nonexistentword", k=5) == []


def test_neighbors_empty_store_returns_empty(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    assert store.neighbors("alpha", k=5) == []


def test_note_nodes_returns_stem_counts_for_one_note(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha alpha beta"))
    st = __import__("snowballstemmer").stemmer("english").stemWord
    nodes = store.note_nodes("A")
    assert nodes[st("alpha")] == 2
    assert nodes[st("beta")] == 1


def test_note_nodes_missing_note_returns_empty(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    assert store.note_nodes("NOPE") == {}


# ---------------------------------------------------------------------------
# Task 3.2 (perf/hot-paths): note_nodes caches its derived {stem: count} dict
# per path, invalidated through the store's existing _invalidate() seam.
# ---------------------------------------------------------------------------

def test_note_nodes_matches_fresh_recompute_across_notes(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha alpha beta"))
    store.upsert_note("B", build_contribution("B", "gamma gamma gamma delta"))
    st = __import__("snowballstemmer").stemmer("english").stemWord
    assert store.note_nodes("A") == {st("alpha"): 2, st("beta"): 1}
    assert store.note_nodes("B") == {st("gamma"): 3, st("delta"): 1}


def test_note_nodes_returns_fresh_dict_each_call(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha alpha beta"))
    d = store.note_nodes("A")
    d["zzznotreal"] = 999
    d.clear()
    d2 = store.note_nodes("A")
    assert "zzznotreal" not in d2
    assert d2  # second call unaffected by mutating/clearing the first result


def test_note_nodes_invalidated_on_upsert(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha alpha beta"))
    store.note_nodes("A")  # warms the cache
    store.upsert_note("A", build_contribution("A", "gamma delta"))
    st = __import__("snowballstemmer").stemmer("english").stemWord
    nodes = store.note_nodes("A")
    assert st("alpha") not in nodes
    assert st("gamma") in nodes


def test_note_nodes_invalidated_on_delete(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    store.note_nodes("A")  # warms the cache
    store.delete_note("A")
    assert store.note_nodes("A") == {}


def test_note_nodes_second_call_served_from_cache(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    store.note_nodes("A")
    store.note_nodes("A")
    assert "A" in store._note_nodes_cache


# ---------------------------------------------------------------------------
# Task 3.4 (perf/hot-paths): stem_postings() — cached stem -> {path: count}
# inverted index, invalidated through the same _invalidate() seam as
# note_nodes()/_adj/_labels.
# ---------------------------------------------------------------------------

def test_stem_postings_builds_stem_to_path_count_map(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha alpha beta"))
    store.upsert_note("B", build_contribution("B", "alpha gamma"))
    st = __import__("snowballstemmer").stemmer("english").stemWord
    postings = store.stem_postings()
    assert postings[st("alpha")] == {"A": 2, "B": 1}
    assert postings[st("beta")] == {"A": 1}
    assert postings[st("gamma")] == {"B": 1}


def test_stem_postings_matches_manual_aggregation_across_notes(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    store.upsert_note("B", build_contribution("B", "beta gamma delta"))
    store.upsert_note("C", build_contribution("C", "delta epsilon"))
    postings = store.stem_postings()
    manual: dict[str, dict[str, int]] = {}
    for path in store.paths():
        for stem, count in store.note_nodes(path).items():
            manual.setdefault(stem, {})[path] = count
    assert postings == manual


def test_stem_postings_empty_store_returns_empty(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    assert store.stem_postings() == {}


def test_stem_postings_invalidated_on_upsert(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    store.stem_postings()  # warms the cache
    store.upsert_note("A", build_contribution("A", "gamma delta"))
    st = __import__("snowballstemmer").stemmer("english").stemWord
    postings = store.stem_postings()
    assert st("alpha") not in postings
    assert postings[st("gamma")] == {"A": 1}


def test_stem_postings_invalidated_on_delete(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    store.upsert_note("B", build_contribution("B", "alpha gamma"))
    store.stem_postings()  # warms the cache
    store.delete_note("A")
    st = __import__("snowballstemmer").stemmer("english").stemWord
    postings = store.stem_postings()
    assert "A" not in postings.get(st("alpha"), {})
    assert postings[st("alpha")] == {"B": 1}


def test_stem_postings_served_from_cache_on_second_call(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    store.stem_postings()
    assert store._stem_postings is not None
    assert store.stem_postings() is store._stem_postings


def test_to_networkx_builds_weighted_undirected_graph(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    G = store.to_networkx()
    st = __import__("snowballstemmer").stemmer("english").stemWord
    assert G.has_edge(st("alpha"), st("beta"))
    assert G[st("alpha")][st("beta")]["weight"] == 3


def test_scope_restricts_aggregation(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("Robotica/A", build_contribution("A", "alpha beta"))
    store.upsert_note("Cucina/B", build_contribution("B", "gamma delta"))
    # within Robotica/, gamma has no neighbors
    assert store.neighbors("gamma", scope="Robotica") == []
    # but alpha does
    assert store.neighbors("alpha", scope="Robotica") != []


from silica.kernel.recall.cooccurrence import build_index, refresh_note


def test_build_index_bulk(tmp_path):
    idx = tmp_path / "cooc.json"
    notes = [
        ("A", "A", "alpha beta gamma"),
        ("B", "B", "beta gamma delta"),
    ]
    store = build_index(notes, store=CooccurStore(path=idx, lang="english"))
    assert len(store) == 2
    assert idx.exists()


def test_build_index_mixed_vault_per_note_stopwords_uniform_stemming(tmp_path):
    # One Italian note + one English note land in the SAME store. Stemming
    # must be uniform at store.lang (one stemmer per store — node keys are
    # stemmed tokens, a per-note stemmer would split cross-language shared
    # terms), but stopword filtering is per-note: neither "della" (Italian)
    # nor "the" (English) may survive as a node, regardless of which
    # language the store's dominant-language freeze picks.
    idx = tmp_path / "cooc.json"
    store = CooccurStore(path=idx, lang="auto")
    notes = [
        ("it/nota.md", "Nota",
         "La rete neurale della azienda migliora la produttivita del team. "
         "Gli algoritmi della rete sono ottimizzati per la performance."),
        ("en/note.md", "Note",
         "The network architecture of the company improves the productivity "
         "of the team. The algorithms of the network are optimized for the "
         "performance."),
    ]
    build_index(notes, store=store, lang="auto")

    # freeze behavior unchanged: store.lang is resolved once, never "auto"
    assert store.lang in ("english", "italian")

    it_labels = {meta["label"] for meta in store._notes["it/nota"]["nodes"].values()}
    en_labels = {meta["label"] for meta in store._notes["en/note"]["nodes"].values()}
    assert "della" not in it_labels
    assert "the" not in en_labels

    # stemming uniform at store.lang: the Italian note's words are stemmed
    # with the SAME stemmer as the frozen store language, not its own.
    import snowballstemmer
    frozen_stem = snowballstemmer.stemmer(store.lang).stemWord("rete")
    assert frozen_stem in store.note_nodes("it/nota.md")


def test_refresh_note_replaces_contribution_no_inflation(tmp_path):
    idx = tmp_path / "cooc.json"
    store = CooccurStore(path=idx, lang="english")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    store.save()

    before = store.neighbors("alpha", k=5)
    w_before = next(c["weight"] for c in before if c["concept"] == "beta")

    # refresh the SAME note with identical content — weight must NOT double
    refresh_note("A", "A", "alpha beta", store=store)
    after = store.neighbors("alpha", k=5)
    w_after = next(c["weight"] for c in after if c["concept"] == "beta")
    assert w_after == w_before  # replacement, not accumulation


def test_refresh_note_reflects_new_content(tmp_path):
    idx = tmp_path / "cooc.json"
    store = CooccurStore(path=idx, lang="english")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    refresh_note("A", "A", "alpha gamma", store=store)
    # beta no longer co-occurs with alpha; gamma now does
    labels = [c["concept"] for c in store.neighbors("alpha", k=5)]
    assert "gamma" in labels
    assert "beta" not in labels


import ast


def test_module_never_imports_embedder():
    """cooccurrence.py is the stable leg: it must not depend on the embedder
    or provider stack (works with LM Studio down)."""
    src = (Path(__file__).parent.parent / "silica" / "kernel" / "recall" / "cooccurrence.py").read_text()
    tree = ast.parse(src)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    assert not any("providers" in m for m in imported)
    assert not any("embed" in m for m in imported)


def test_neighbors_never_raises_on_garbage(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    # empty/garbage queries must return [] rather than raising
    assert store.neighbors("", k=5) == []
    assert store.neighbors("   ", k=5) == []


def test_corrupt_index_loads_empty(tmp_path):
    idx = tmp_path / "cooc.json"
    idx.write_text("{ this is not valid json ")
    store = CooccurStore(path=idx)
    assert len(store) == 0


# ---------------------------------------------------------------------------
# #9 LLM concept augmentation — concepts reinforce the co-occurrence graph
#
# Paper (Marwitz et al. 2026, Table 1): LLM-extracted concept phrases beat
# rule-based extraction (nominalization, formula cleanup, synonym resolution).
# build_contribution accepts optional `concepts`; they enter the SAME tokenize
# pipeline so their stems become nodes and their words co-occur, lifting
# LLM-validated concepts above body noise. `concepts=None` is byte-identical to
# today (graceful degradation).
# ---------------------------------------------------------------------------

def test_build_contribution_concepts_add_nodes_absent_from_body():
    st = __import__("snowballstemmer").stemmer("english").stemWord
    c = build_contribution("N", "alpha beta", concepts=["quantum entanglement"], lang="english")
    assert st("quantum") in c["nodes"]
    assert st("entanglement") in c["nodes"]


def test_build_contribution_concepts_create_intra_concept_edge():
    st = __import__("snowballstemmer").stemmer("english").stemWord
    c = build_contribution("N", "alpha beta", concepts=["quantum entanglement"], lang="english")
    # the two words of one concept are adjacent -> narrative weight 3
    assert _edge_weight(c, st("quantum"), st("entanglement")) == 3


def test_build_contribution_concepts_none_is_identical_to_today():
    base = build_contribution("N", "alpha beta gamma", lang="english")
    none = build_contribution("N", "alpha beta gamma", concepts=None, lang="english")
    empty = build_contribution("N", "alpha beta gamma", concepts=[], lang="english")
    assert none == base
    assert empty == base


def test_build_index_threads_concepts_by_path(tmp_path):
    """#9: build_index forwards per-path LLM concepts into build_contribution."""
    st = __import__("snowballstemmer").stemmer("english").stemWord
    store = CooccurStore(path=tmp_path / "c.json", lang="english")
    build_index(
        [("Notes/A", "A", "alpha beta")],
        store=store,
        concepts_by_path={"Notes/A": ["quantum entanglement"]},
        force=True,
    )
    nodes = store.note_nodes("Notes/A")
    assert st("quantum") in nodes
    assert st("entanglement") in nodes


def test_top_stems_orders_by_total_weight(tmp_path):
    store = CooccurStore(path=tmp_path / "cooccur.json")
    store.upsert_note(
        "a.md",
        build_contribution("a", "neural networks learn. neural networks generalize. neural networks overfit."),
    )
    store.upsert_note(
        "b.md",
        build_contribution("b", "backpropagation tunes neural networks slowly."),
    )

    stems = store.top_stems(5)

    assert 0 < len(stems) <= 5
    # 'neural'/'network' dominate by accumulated weight across both notes.
    joined = " ".join(s.lower() for s in stems[:2])
    assert "neural" in joined or "network" in joined


def test_top_stems_respects_n(tmp_path):
    store = CooccurStore(path=tmp_path / "cooccur.json")
    store.upsert_note("a.md", build_contribution("a", "alpha beta gamma delta epsilon zeta"))
    assert len(store.top_stems(2)) == 2


def test_top_stems_empty_store(tmp_path):
    store = CooccurStore(path=tmp_path / "cooccur.json")
    assert store.top_stems(10) == []


# ---------------------------------------------------------------------------
# Finding 1 (final multilingua review): incremental refresh must not
# re-freeze store.lang when the caller passes the "auto" sentinel (the
# write-hook default post-Task-5). store.lang is frozen at FIRST build; an
# "auto" request on an already-populated store must stick to the frozen
# language, never re-detect from a single (possibly foreign-language) batch.
# "company"/"productivity"/"improves" are unambiguous probes: the English
# Snowball stemmer changes them ("compani"/"product"/"improv"), the Italian
# one leaves them unchanged.
# ---------------------------------------------------------------------------

def test_refresh_note_auto_lang_sticky_to_frozen_store(tmp_path):
    idx = tmp_path / "cooc.json"
    store = CooccurStore(path=idx, lang="italian")
    store.upsert_note(
        "it/nota.md",
        build_contribution("Nota", "La rete della azienda migliora la produttivita", lang="italian"),
    )
    store.save()

    refresh_note(
        "en/note.md", "Note",
        "The company improves productivity for the whole team.",
        store=store, lang="auto",
    )

    assert store.lang == "italian"
    # node key uses the FROZEN (italian) stemmer, not english's "compani"
    assert "company" in store.note_nodes("en/note.md")
    assert "compani" not in store.note_nodes("en/note.md")


def test_build_index_write_hook_shape_force_true_sticky_to_frozen_store(tmp_path):
    # THE mainline shape (Round 2): the post-write freshness hook
    # (orchestrator._refresh_cooccurrence_for_ops) calls build_index with
    # force=True — there force means "replace this note's prior contribution,
    # never inflate" (replacement semantics), NOT "rebuild the store". It must
    # NOT re-detect the frozen language; only an explicit refreeze=True may.
    idx = tmp_path / "cooc.json"
    store = CooccurStore(path=idx, lang="italian")
    store.upsert_note(
        "it/nota.md",
        build_contribution("Nota", "La rete della azienda migliora la produttivita", lang="italian"),
    )
    store.save()

    build_index(
        [("en/note.md", "Note", "The company improves productivity for the whole team.")],
        store=store, lang="auto", force=True,
    )

    assert store.lang == "italian"
    assert "company" in store.note_nodes("en/note.md")
    assert "compani" not in store.note_nodes("en/note.md")


def test_build_index_refreeze_true_redetects_auto_lang(tmp_path):
    # The deliberate-rebuild shape (/cooccur --force → silica_cooccurrence_refresh
    # force=True → refreeze=True): the doctor remedy for a wrong-frozen store.
    idx = tmp_path / "cooc.json"
    store = CooccurStore(path=idx, lang="english")
    store.upsert_note(
        "en/old.md",
        build_contribution("Old", "The company improves productivity.", lang="english"),
    )
    store.save()

    italian_notes = [
        ("it/a.md", "A", "La rete neurale della azienda migliora la produttivita del team."),
        ("it/b.md", "B", "Gli algoritmi della rete sono ottimizzati per la performance."),
    ]
    build_index(italian_notes, store=store, lang="auto", force=True, refreeze=True)

    assert store.lang == "italian"
