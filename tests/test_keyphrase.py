"""Tests for silica.kernel.text.keyphrase — content-based concept extraction (Fase 1).

The thesis: markup-only extraction (recon.extract_concepts) returns ~0 real
concepts on prose with no headings/bold/acronyms; the candidate miner
(kernel/text/candidates.py, in-house since 2026-08-31, YAKE before) recovers them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BUNDLED_OVERLAYS = Path(__file__).resolve().parent.parent / "silica" / "overlays"

# Italian prose, NO markup: the case that broke the old markup-only recon.
_PROSE = (
    "La discesa del gradiente stocastico ottimizza la funzione di perdita "
    "aggiornando i pesi della rete neurale a ogni iterazione del training. "
    "Il tasso di apprendimento controlla l'ampiezza del passo di aggiornamento. "
    "La retropropagazione calcola i gradienti rispetto a ciascun parametro del modello."
)


@pytest.fixture
def it_overlay():
    path = _BUNDLED_OVERLAYS / "italian.yaml"
    if not path.exists():
        pytest.skip(f"bundled overlay not found: {path}")
    from silica.kernel.text.overlay import load_overlay
    return load_overlay(path)


def test_prose_extracts_content_concepts(it_overlay):
    """Prose with no markup yields real domain concepts (markup-only gave ~0)."""
    from silica.kernel.text.keyphrase import extract_keyphrases

    cands = extract_keyphrases(_PROSE, overlay=it_overlay, lang="italian")
    phrases = " ".join(c.phrase.lower() for c in cands)

    assert cands, "no concepts extracted from prose"
    assert "gradiente" in phrases or "rete neurale" in phrases


def _fake_ranked(n):
    from silica.kernel.text.keyphrase import ConceptCandidate
    return [ConceptCandidate(phrase=f"c{i}", score=float(i)) for i in range(n)]


def test_cutoff_scales_with_tokens_and_caps():
    """k = clamp(n_tok / TOKENS_PER_CONCEPT, MIN, MAX), capped at candidates."""
    from silica.kernel.text.keyphrase import (
        MAX_CONCEPTS, MIN_CONCEPTS, TOKENS_PER_CONCEPT, _cutoff,
    )
    pool = _fake_ranked(100)

    huge = "w " * (TOKENS_PER_CONCEPT * (MAX_CONCEPTS + 10))   # well past MAX
    assert len(_cutoff(huge, pool)) == MAX_CONCEPTS

    mid = "w " * (TOKENS_PER_CONCEPT * 12)                     # 12 in [MIN, MAX]
    assert len(_cutoff(mid, pool)) == 12

    tiny = "w " * 5                                            # below MIN => floor
    assert len(_cutoff(tiny, pool)) == MIN_CONCEPTS

    assert len(_cutoff(huge, _fake_ranked(7))) == 7           # never exceed candidates


def test_frontmatter_ignored(it_overlay):
    """YAML front matter is metadata, not content: it must not change concepts."""
    from silica.kernel.text.keyphrase import extract_keyphrases

    body = _PROSE
    with_fm = "---\ntitle: ZzzParolaSegreta\ntags: [nascosto]\n---\n" + body
    a = [c.phrase for c in extract_keyphrases(with_fm, overlay=it_overlay, lang="italian")]
    b = [c.phrase for c in extract_keyphrases(body, overlay=it_overlay, lang="italian")]

    assert a == b


def test_empty_content_abstains(it_overlay):
    """No content => empty list (silica_recon handles it as an empty report)."""
    from silica.kernel.text.keyphrase import extract_keyphrases

    assert extract_keyphrases("", overlay=it_overlay, lang="italian") == []


# ---------------------------------------------------------------------------
# Fase 2: miner = pool generator, embedder + MMR = ranker, structural = boost
# ---------------------------------------------------------------------------

_AXES = ("graph", "memory", "planning", "noise")


class FakeEmbedder:
    """Deterministic embedder: vector over topic axes by word presence."""
    def embed(self, texts):
        return [[float(ax in t.lower()) for ax in _AXES] for t in texts]


def test_structural_concepts_from_markup():
    """Heading / bold / acronym concepts are extracted and overlay-filtered (lowercased)."""
    from silica.kernel.text.keyphrase import _structural_concepts
    from silica.kernel.text.overlay import DEFAULT_OVERLAY

    body = "# Reti Neurali\n\nUso di **Gradient Descent** e il PID controller."
    concs = _structural_concepts(body, DEFAULT_OVERLAY)

    assert "reti neurali" in concs   # heading
    assert "gradient descent" in concs  # bold
    assert "pid" in concs            # acronym


def test_markup_acronym_is_seeded_in_the_author_case():
    """`_structural_concepts` lowercased every markup term and `_seed_structural`
    reused those strings AS the candidate phrase, so an acronym the author wrote
    in a heading arrived in the pool as `pid` / `rdf` and became a note titled
    that way (audit of 2026-08-23). The map keys stay lowercase (that is what
    the boost and the confidence stamp test against); the VALUE keeps the case."""
    from silica.kernel.text.keyphrase import _structural_concepts, extract_keyphrases
    from silica.kernel.text.overlay import DEFAULT_OVERLAY

    body = ("# Reti Neurali\n\nUso di **Gradient Descent** e il PID controller "
            "nel loop di controllo del PID controller. " * 3)
    concs = _structural_concepts(body, DEFAULT_OVERLAY)
    assert concs["pid"] == "PID"

    pool = [c.phrase for c in extract_keyphrases(body, overlay=DEFAULT_OVERLAY,
                                                 lang="italian", embedder=None)]
    assert "PID" in pool and "pid" not in pool


def test_roman_numeral_markup_is_not_a_concept():
    """`from_acronyms` matches any run of 2-6 capitals, so a numbered section
    seeded `III` (lowercased to `iii`) as a concept. Screened where every other
    name-hygiene rule lives, so headings and bold get the same treatment."""
    from silica.kernel.text.keyphrase import _structural_concepts
    from silica.kernel.text.overlay import DEFAULT_OVERLAY

    body = "La sezione III introduce la **PCA** e la sua varianza."
    concs = _structural_concepts(body, DEFAULT_OVERLAY)
    assert "iii" not in concs
    assert concs.get("pca") == "PCA"


def test_mmr_demotes_near_duplicate():
    """MMR picks a diverse candidate over a near-duplicate of an already-selected one."""
    from silica.kernel.text.keyphrase import _mmr

    vecs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]  # 0 and 1 identical, 2 orthogonal
    order = _mmr(vecs, theme=[1.0, 1.0], k=2, lam=0.6)

    assert order[0] in (0, 1)
    assert 2 in order                       # diversity reaches the orthogonal item
    assert not (0 in order and 1 in order)  # not both duplicates


def test_rerank_orders_thematic_above_junk_and_abstains_without_embedder():
    from silica.kernel.text.keyphrase import _rerank, ConceptCandidate
    from silica.kernel.text.overlay import DEFAULT_OVERLAY

    pool = [ConceptCandidate("promise to enhance", 0.0),
            ConceptCandidate("knowledge graph", 0.0),
            ConceptCandidate("graph memory", 0.0)]
    body = "graph memory planning graph memory knowledge graph"

    ranked = _rerank(pool, body, DEFAULT_OVERLAY, FakeEmbedder())
    phrases = [c.phrase for c in ranked]
    assert phrases.index("knowledge graph") < phrases.index("promise to enhance")

    assert _rerank(pool, body, DEFAULT_OVERLAY, None) is None  # no embedder => abstain


def test_structural_boost_promotes_markup_concept():
    """A thematically-flat concept that appears in a heading is lifted by the structural boost."""
    from silica.kernel.text.keyphrase import _rerank, ConceptCandidate
    from silica.kernel.text.overlay import DEFAULT_OVERLAY

    pool = [ConceptCandidate("alpha widget", 0.0), ConceptCandidate("beta gadget", 0.0)]
    body = "# Beta Gadget\n\nsome unrelated prose"  # both flat on theme; beta is in a heading

    ranked = _rerank(pool, body, DEFAULT_OVERLAY, FakeEmbedder())
    phrases = [c.phrase for c in ranked]
    assert phrases.index("beta gadget") < phrases.index("alpha widget")


# ---------------------------------------------------------------------------
# Fase A: structural markup is also a *candidate source*, not only a boost
# ---------------------------------------------------------------------------

def test_structural_phrase_beyond_the_miner_span_enters_pool():
    """A markup-marked phrase longer than the miner's span (MAX_CONTENT_WORDS=4)
    can never be a mined candidate, yet the author bolded it. The structural leg
    must seed it into the pool so it survives even in the embedder-down fallback."""
    from silica.kernel.text.keyphrase import _pool_leg, extract_keyphrases
    from silica.kernel.text.overlay import DomainOverlay

    overlay = DomainOverlay(stopwords=frozenset(), noise_patterns=())
    body = ("This work studies sequential decision making in agents. "
            "The setting is a **partially observable markov decision process** "
            "and we evaluate planning under it across many tasks and domains.")

    # precondition: five content words exceed the miner's span
    pool = _pool_leg(body, overlay, "english") or []
    assert all("partially observable markov decision process" not in c.phrase.lower() for c in pool)

    # behaviour: the embedder-down fallback still surfaces the bolded concept
    out = [c.phrase.lower()
           for c in extract_keyphrases(body, overlay=overlay, lang="english", embedder=None)]
    assert any("partially observable markov decision process" in p for p in out)


# ---------------------------------------------------------------------------
# Pool-leg abstention: no content => None (never [] and never a raise), so
# extract_keyphrases keeps its "[] only when both legs abstain" contract.
# ---------------------------------------------------------------------------

def test_pool_leg_abstains_on_text_without_content():
    from silica.kernel.text.keyphrase import _pool_leg
    from silica.kernel.text.overlay import DEFAULT_OVERLAY

    assert _pool_leg("   \n  ", DEFAULT_OVERLAY, "english") is None
    assert _pool_leg("il la di e che", DEFAULT_OVERLAY, "italian") is None


def test_pool_leg_norwegian_does_not_raise():
    """A language with a stopword list but no dedicated overlay: a real,
    unmocked _pool_leg("norwegian") call must not raise."""
    from silica.kernel.text.keyphrase import _pool_leg
    from silica.kernel.text.overlay import DEFAULT_OVERLAY

    result = _pool_leg(
        "dette er en test av norsk tekst med flere ord i teksten for gradientnedstigning",
        DEFAULT_OVERLAY, "norwegian",
    )
    assert result is None or isinstance(result, list)


def test_pool_leg_never_needs_yake():
    """The AGPL yake package is gone from the dependency set; the pool must
    come from the in-house miner even if a stale environment still has yake."""
    import importlib.util
    import silica.kernel.text.keyphrase as kp

    src = importlib.util.find_spec(kp.__name__).origin
    assert "import yake" not in open(src, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# Corroboration tier: structural markup is a *second axis*, not only a boost.
# EXTRACTED <=> structurally corroborated (embedder-free); else INFERRED.
# ---------------------------------------------------------------------------

def test_embedder_down_structural_is_extracted_mined_only_is_inferred(it_overlay):
    """Embedder-down — the only corroboration available is author markup.

    A heading concept has a second independent signal → EXTRACTED. A prose-only
    mined concept has a single signal → INFERRED. This is the gate the salience
    path cannot supply when the embedder is down.
    """
    from silica.kernel.text.keyphrase import extract_keyphrases

    body = (
        "# Discesa Del Gradiente Stocastico\n\n"
        "La discesa del gradiente stocastico ottimizza la funzione di perdita "
        "aggiornando i pesi della rete neurale a ogni iterazione del training. "
        "Il tasso di apprendimento controlla l'ampiezza del passo di aggiornamento. "
        "La retropropagazione calcola i gradienti rispetto a ciascun parametro del modello. "
        "La regolarizzazione riduce il sovradattamento penalizzando i pesi troppo grandi. "
        "La convalida incrociata stima la capacita di generalizzazione del modello."
    )
    cands = extract_keyphrases(body, overlay=it_overlay, lang="italian", embedder=None)
    by = {c.phrase.lower(): c.confidence for c in cands}

    assert by.get("discesa del gradiente stocastico") == "EXTRACTED"  # heading → second axis
    assert any(conf == "INFERRED" for conf in by.values())           # prose-only → single signal


def test_embedder_up_tier_independent_of_ranking_axis():
    """With an embedder, the tier still follows markup, not the theme cosine the
    ranker already uses: a heading concept is EXTRACTED, a theme-only one INFERRED."""
    from silica.kernel.text.keyphrase import extract_keyphrases
    from silica.kernel.text.overlay import DEFAULT_OVERLAY

    body = (
        "# Graph Memory\n\n"
        "The planning module reads the graph memory and writes planning results back. "
        "Planning over the graph memory improves planning quality and memory recall. "
        "A planning agent stores planning state in graph memory for later planning. "
        "The memory layer indexes planning episodes so planning can resume from memory."
    )
    cands = extract_keyphrases(body, overlay=DEFAULT_OVERLAY, lang="english", embedder=FakeEmbedder())
    by = {c.phrase.lower(): c.confidence for c in cands}

    assert by.get("graph memory") == "EXTRACTED"          # in a heading → corroborated
    assert any(conf == "INFERRED" for conf in by.values())  # theme-only candidates stay single-signal


def test_extract_keyphrases_rerank_end_to_end():
    """With an embedder, extract_keyphrases reranks; without, it falls back to the mined order."""
    from silica.kernel.text.keyphrase import extract_keyphrases
    from silica.kernel.text.overlay import DEFAULT_OVERLAY

    body = ("The knowledge graph stores memory. Planning over the graph memory improves "
            "planning. A knowledge graph is a memory structure for planning.")
    with_emb = [c.phrase for c in extract_keyphrases(body, overlay=DEFAULT_OVERLAY, lang="english", embedder=FakeEmbedder())]
    no_emb = [c.phrase for c in extract_keyphrases(body, overlay=DEFAULT_OVERLAY, lang="english", embedder=None)]

    assert with_emb and no_emb
    assert with_emb != no_emb  # reranking actually changed the order


def test_code_fences_never_surface_as_concepts():
    """C1 fork ⚑: keyphrase strips code fences — the miner must not rank identifiers."""
    from silica.kernel.text.keyphrase import extract_keyphrases
    from silica.kernel.text.overlay import DEFAULT_OVERLAY

    body = (
        "The knowledge graph stores memory. Planning over the graph memory "
        "improves planning quality. A knowledge graph is a memory structure.\n\n"
        + "```python\ntrainstepalpha = trainstepalpha + 1\nprint(trainstepalpha)\n```\n" * 3
    )
    cands = extract_keyphrases(body, overlay=DEFAULT_OVERLAY, lang="english")
    assert cands, "prose concepts must survive"
    assert not any("trainstepalpha" in c.phrase.lower() for c in cands)


def test_latex_body_yields_no_math_token_concepts():
    """LaTeX commands in the body never surface as concepts (stripped before mining)."""
    from silica.kernel.text.keyphrase import extract_keyphrases
    body = (
        "# Gradient descent\n\n"
        "The loss function $\\mathcal{L}$ is minimized by gradient descent. "
        "We compute $$\\sum_{i} \\nabla_w \\mathcal{L}_i \\leq \\epsilon$$ each step, "
        "updating the weights of the neural network until convergence. " * 3
    )
    cands = extract_keyphrases(body)  # default overlay/lang, no embedder
    phrases = " ".join(c.phrase.lower() for c in cands)
    for junk in ("mathcal", "sum", "nabla", "leq", "epsilon"):
        assert junk not in phrases, f"{junk!r} leaked from LaTeX"
    assert cands, "prose should still yield concepts"


def test_auto_lang_resolves_so_the_miner_drops_italian_function_words():
    """lang='auto' is resolved to a real Snowball language before mining, so the
    miner treats Italian function words as boundaries (no bogus 'auto' list)."""
    from silica.kernel.text.keyphrase import extract_keyphrases
    from silica.kernel.text.overlay import DomainOverlay
    # Empty overlay isolates the language effect: is_concept filters nothing,
    # so a leaked function word would survive to the output if lang were wrong.
    empty = DomainOverlay(stopwords=frozenset(), noise_patterns=())
    # Longer body ensures "della" survives _cutoff and makes it to the final output
    # if the miner does not treat it as a function word.
    body = (
        "La discesa del gradiente della rete neurale aggiorna i pesi della rete. "
        "La funzione di perdita della rete dipende dai pesi della rete neurale. "
        "Il tasso di apprendimento della rete regola il passo. "
        "La retropropagazione della rete calcola i gradienti. " * 8
    )
    cands = extract_keyphrases(body, overlay=empty, lang="auto")
    phrases = {c.phrase.lower() for c in cands}
    assert "della" not in phrases  # IT function word treated as a boundary


def test_pool_leg_unions_language_and_overlay_stopwords():
    """The language's own stopword 'ancora' is filtered even though it is absent
    from the Italian overlay: the miner sees the UNION of the two lists, the
    overlay never replaces the language list (that replacement was a real bug
    in the YAKE era, which dropped ~300 function words).

    Word verified:
      'ancora' in language.stopwords_for('italian')      → True
      'ancora' in overlay_for_lang('italian').stopwords  → False
    """
    from silica.kernel.text import language
    from silica.kernel.text.keyphrase import extract_keyphrases
    from silica.kernel.text.overlay import overlay_for_lang

    overlay = overlay_for_lang("italian")
    assert "ancora" in language.stopwords_for("italian")
    assert "ancora" not in overlay.stopwords

    # Body: 'ancora' repeated many times alongside a real content word so that
    # if the miner does not filter it, it would rank highly and survive _cutoff.
    body = (
        "Il percettrone e ancora un modello ancora usato ancora nella rete neurale. "
        "Ancora oggi il percettrone e ancora studiato e ancora applicato ancora. "
        "La regola di apprendimento del percettrone e ancora fondamentale ancora. " * 6
    )
    cands = extract_keyphrases(body, overlay=overlay, lang="italian")
    phrases = {c.phrase.lower() for c in cands}

    # 'ancora' must not surface as a standalone concept (built-in stopword)
    assert not any(p == "ancora" or p.startswith("ancora ") or p.endswith(" ancora")
                   for p in phrases), f"'ancora' leaked as concept: {phrases}"
    # Real content word must still appear
    assert any("percettrone" in p for p in phrases), f"content word lost: {phrases}"


def test_recon_italian_drops_latex_and_structural_keeps_content():
    """End-to-end (overlay=None -> overlay_for_lang('italian')): LaTeX, the
    'Lezione' heading, and the CFU acronym are gone; an Italian content word
    survives."""
    from silica.kernel.text.keyphrase import extract_keyphrases
    from silica.kernel.text.overlay import reset_overlay_cache
    reset_overlay_cache()
    body = (
        "## Lezione 10\n\n"
        "Il percettrone e un modello della rete neurale. "
        "Per ogni CFU si studia il percettrone e la sua regola di apprendimento. "
        "La funzione $\\mathbb{R} \\to \\mathbb{R}$ con $\\sum_i w_i x_i \\leq \\theta$ "
        "definisce l'attivazione del percettrone. " * 3
    )
    cands = extract_keyphrases(body, lang="italian")  # overlay=None on purpose
    phrases = " ".join(c.phrase.lower() for c in cands)
    for junk in ("mathbb", "lezione", "cfu", "sum", "leq", "theta"):
        assert junk not in phrases, f"{junk!r} should be filtered"
    assert "percettrone" in phrases, "content word lost"


# ---------------------------------------------------------------------------
# _mmr — cost on a real embedding pool
# ---------------------------------------------------------------------------

def _naive_mmr(vecs, theme, k, lam, rel):
    """Reference: the per-pair formulation, kept only as the correctness oracle."""
    from silica.kernel.recall.embed import _cosine

    cand, sel = list(range(len(vecs))), []
    while cand and len(sel) < k:
        if not sel:
            i = max(cand, key=lambda i: rel[i])
        else:
            i = max(cand, key=lambda i: lam * rel[i]
                    - (1 - lam) * max(_cosine(vecs[i], vecs[j]) for j in sel))
        sel.append(i)
        cand.remove(i)
    return sel


def test_mmr_ranks_full_pool_in_seconds_not_minutes():
    """RECON ranks POOL_SIZE=100 vectors of the embedder's real width on every
    note. Recomputing each pair through _cosine (4 np.asarray per call) made
    this 26-46s per note; it must cost a fraction of a second."""
    import random
    import time
    from silica.kernel.text.keyphrase import _mmr

    random.seed(7)
    dim, n = 2560, 100  # Qwen3-Embedding-4B width, POOL_SIZE
    vecs = [[random.random() for _ in range(dim)] for _ in range(n)]
    theme = [random.random() for _ in range(dim)]
    rel = [random.random() for _ in range(n)]

    t0 = time.monotonic()
    order = _mmr(vecs, theme, k=n, lam=0.6, rel=rel)
    elapsed = time.monotonic() - t0

    assert sorted(order) == list(range(n))  # a full ranking, nothing dropped
    assert elapsed < 2.0, f"_mmr took {elapsed:.1f}s for {n} vectors of dim {dim}"


def test_mmr_selection_matches_the_per_pair_reference():
    """The faster formulation must pick the same order as the naive one."""
    import random
    from silica.kernel.text.keyphrase import _mmr

    random.seed(11)
    dim, n = 32, 12
    vecs = [[random.random() for _ in range(dim)] for _ in range(n)]
    theme = [random.random() for _ in range(dim)]
    rel = [random.random() for _ in range(n)]

    assert _mmr(vecs, theme, k=n, lam=0.6, rel=rel) == \
        _naive_mmr(vecs, theme, k=n, lam=0.6, rel=rel)


def test_rerank_abstains_on_ragged_embeddings():
    """A short embed reply (the A6 ragged case the other legs already guard)
    must abstain so the caller keeps the mined rank — never rank on a pool whose
    vectors are not the same width."""
    from unittest.mock import MagicMock
    from silica.kernel.text.keyphrase import _rerank, ConceptCandidate
    from silica.kernel.text.overlay import DomainOverlay

    pool = [ConceptCandidate(phrase=p, score=0.1) for p in ("alpha", "beta", "gamma")]
    fake = MagicMock()
    fake.model = "fake"
    # theme call: one well-formed vector; phrase call: one row of the wrong width
    fake.embed.side_effect = [[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0]]]

    out = _rerank(pool, "alpha beta gamma body text", DomainOverlay(stopwords=frozenset(), noise_patterns=()), fake)

    assert out is None


# ---------------------------------------------------------------------------
# _complete_phrases — boundary snap + edge trim (2026-08-21: 29 of 181 notes
# in a real run carried a truncated/fragment title verbatim from extraction).
# ---------------------------------------------------------------------------

_STOP = frozenset({"ha", "il", "la", "lo", "a", "di", "per", "è", "un",
                   "della", "quando", "sono", "e"})


def _cands(*phrases):
    from silica.kernel.text.keyphrase import ConceptCandidate
    return [ConceptCandidate(phrase=p, score=0.0) for p in phrases]


def _ov():
    from silica.kernel.text.overlay import DomainOverlay
    return DomainOverlay(stopwords=frozenset(), noise_patterns=())


def test_complete_snaps_truncated_ngram_to_phrase_boundary():
    # YAKE n=3 cut "stimatore a massima [verosimiglianza]"; every occurrence
    # continues with the same word, so the walk completes it.
    from silica.kernel.text.keyphrase import _complete_phrases

    body = ("Lo stimatore a massima verosimiglianza massimizza la likelihood. "
            "Uno stimatore a massima verosimiglianza esiste sempre.")
    out = _complete_phrases(body, _cands("stimatore a massima"), _STOP, _ov())
    assert [c.phrase for c in out] == ["stimatore a massima verosimiglianza"]
    assert "snap:+1" in out[0].evidence


def test_complete_requires_two_occurrences():
    # One occurrence is zero evidence of a collocation: a singleton walk
    # absorbed the sentence's verb ("Errore quadratico atteso dipende").
    from silica.kernel.text.keyphrase import _complete_phrases

    body = "L'errore quadratico atteso dipende dal modello."
    out = _complete_phrases(body, _cands("errore quadratico atteso"), _STOP, _ov())
    assert [c.phrase for c in out] == ["errore quadratico atteso"]


def test_complete_stops_on_divergent_continuations_and_stopwords():
    from silica.kernel.text.keyphrase import _complete_phrases

    body = ("il kernel polinomiale e il kernel gaussiano; "
            "la regressione lineare per i dati e la regressione lineare per tutti")
    out = _complete_phrases(
        body, _cands("kernel", "regressione lineare"), _STOP, _ov())
    assert [c.phrase for c in out] == ["kernel", "regressione lineare"]


def test_complete_never_crosses_identifiers():
    # "Giosuè Lo Bosco\n giosue.lobosco@..." — appending the email stem would
    # mint a new fragment.
    from silica.kernel.text.keyphrase import _complete_phrases

    body = ("Giosuè Lo Bosco giosue.lobosco@unipa.it dice. "
            "Giosuè Lo Bosco giosue.lobosco@unipa.it insegna.")
    out = _complete_phrases(body, _cands("Giosuè Lo Bosco"), _STOP, _ov())
    assert [c.phrase for c in out] == ["Giosuè Lo Bosco"]


def test_edge_trim_drops_stopword_edges_and_markup_tails():
    from silica.kernel.text.keyphrase import _complete_phrases

    body = "Il modello ha bias quando la stima è distorta."
    out = _complete_phrases(
        body, _cands("ha bias", "proprietà delle svm***"), _STOP, _ov())
    assert [c.phrase for c in out] == ["bias", "proprietà delle svm"]
    balanced = _complete_phrases(
        body, _cands("deep learning (6 crediti)"), _STOP, _ov())
    assert [c.phrase for c in balanced] == ["deep learning (6 crediti)"]


def test_completed_duplicates_collapse_on_best_rank():
    from silica.kernel.text.keyphrase import _complete_phrases

    body = ("Lo stimatore a massima verosimiglianza domina. "
            "Lo stimatore a massima verosimiglianza vince.")
    out = _complete_phrases(
        body,
        _cands("stimatore a massima", "stimatore a massima verosimiglianza"),
        _STOP, _ov())
    assert [c.phrase for c in out] == ["stimatore a massima verosimiglianza"]
