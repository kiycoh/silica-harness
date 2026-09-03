"""Tests for silica.kernel.text.candidates : the in-house keyphrase candidate miner.

Replaces YAKE (AGPL) as the pool generator behind keyphrase.extract_keyphrases.
Every property below is one the pool consumer (embedder + MMR ranker, or the
deterministic fallback rank) relies on; the two that YAKE could not give are
spans past three tokens and stem-level collapsing of inflected variants.
"""
from __future__ import annotations


def _phrases(text, **kw):
    from silica.kernel.text.candidates import mine_candidates
    return [c.phrase for c in mine_candidates(text, **kw)]


def test_inner_stopwords_are_allowed_but_edges_never_are():
    """Romance-language terms carry a preposition inside ("funzione di perdita");
    a candidate may contain one, but can neither start nor end on a stopword."""
    from silica.kernel.text import language

    text = ("La discesa del gradiente ottimizza la funzione di perdita. "
            "La discesa del gradiente converge quando la funzione di perdita cala.")
    phrases = _phrases(text, lang="italian")
    stop = language.stopwords_for("italian")

    assert "discesa del gradiente" in phrases
    assert "funzione di perdita" in phrases
    for p in phrases:
        words = p.split()
        assert words[0].lower() not in stop and words[-1].lower() not in stop, p


def test_spans_reach_four_content_words():
    """YAKE's n=3 window cut "stimatore a massima [verosimiglianza]" and the
    fragment became a note title (29 of 181 notes, 2026-08-21). The miner emits
    the whole term."""
    text = ("Lo stimatore a massima verosimiglianza massimizza la likelihood. "
            "Uno stimatore a massima verosimiglianza esiste sempre.")
    assert "stimatore a massima verosimiglianza" in _phrases(text, lang="italian")


def test_repetition_outranks_a_single_mention():
    """A phrase seen four times outranks every phrase seen once that is not one
    of its own extensions (the paper's product lets an early singleton
    extension such as "knowledge graph stores" sit above the term itself; the
    pool consumer's completion and cutoff steps deal with those)."""
    text = ("The knowledge graph stores memory. A knowledge graph links facts. "
            "Every knowledge graph has edges. The knowledge graph is queried. "
            "One promise remains.")
    phrases = _phrases(text, lang="english")
    at = phrases.index("knowledge graph")
    assert all("knowledge graph" in p for p in phrases[:at])
    assert at < phrases.index("promise")
    assert at < phrases.index("promise remains")


def test_inflected_variants_collapse_onto_the_most_frequent_surface():
    """"rete neurale" and "reti neurali" are one concept: one candidate, whose
    surface is the form the author used most, whose count sums both."""
    from silica.kernel.text.candidates import mine_candidates

    text = ("La rete neurale apprende. Le reti neurali apprendono. "
            "La rete neurale generalizza.")
    cands = mine_candidates(text, lang="italian")
    by_phrase = {c.phrase: c for c in cands}
    assert "rete neurale" in by_phrase and by_phrase["rete neurale"].count == 3
    assert not any("reti neurali" in c.phrase for c in cands)


def test_subphrase_seen_only_inside_a_longer_term_is_dropped():
    """Every "massima verosimiglianza" sits inside "stimatore a massima
    verosimiglianza": the fragment adds nothing and would compete with the term
    for the same pool slot. A sub-phrase that also stands alone survives."""
    only_inside = ("Lo stimatore a massima verosimiglianza domina. "
                   "Lo stimatore a massima verosimiglianza vince.")
    phrases = _phrases(only_inside, lang="italian")
    assert "stimatore a massima verosimiglianza" in phrases
    assert "massima verosimiglianza" not in phrases

    also_alone = only_inside + " La massima verosimiglianza è un principio."
    assert "massima verosimiglianza" in _phrases(also_alone, lang="italian")


def test_punctuation_and_line_breaks_bound_a_span():
    text = "il gradiente. funzione convessa\nmatrice hessiana, vettore gradiente"
    phrases = _phrases(text, lang="italian")
    assert "gradiente funzione" not in phrases
    assert "convessa matrice" not in phrases
    assert "hessiana vettore" not in phrases
    assert "funzione convessa" in phrases and "matrice hessiana" in phrases


def test_numbers_and_single_letters_are_boundaries_not_content():
    text = "In 2024 the model x reached 95 accuracy. In 2024 the model x reached 95 accuracy."
    phrases = _phrases(text, lang="english")
    assert "model" in phrases and "accuracy" in phrases
    assert not any(p in ("2024", "x", "95") for p in phrases)
    assert not any("2024" in p or " x " in f" {p} " for p in phrases)


def test_sentence_initial_capital_is_not_kept_but_an_acronym_is():
    text = ("Gradient descent converges. We use gradient descent daily. "
            "The POMDP setting is hard. A pomdp is partially observable. POMDP planning is slow.")
    phrases = _phrases(text, lang="english")
    assert "gradient descent" in phrases and "Gradient descent" not in phrases
    assert "POMDP" in phrases and "pomdp" not in phrases


def test_acronym_outranks_a_plain_word_at_equal_count():
    """YAKE's casing feature, kept: an acronym or a capitalised mid-sentence
    token is a proper term more often than a common word is."""
    text = "The POMDP is hard. The widget is hard."
    phrases = _phrases(text, lang="english")
    assert phrases.index("POMDP") < phrases.index("widget")


def test_unknown_language_never_raises_and_still_mines():
    """No stopword list and no stemmer for the language: degrade to raw tokens
    rather than crash extraction (language.py's contract, kept here)."""
    from silica.kernel.text.candidates import mine_candidates

    out = mine_candidates("plasma torus plasma torus drift", lang="klingon")
    assert isinstance(out, list) and out
    assert isinstance(mine_candidates("plasma torus plasma torus", lang="auto"), list)


def test_top_cap_and_determinism():
    from silica.kernel.text.candidates import mine_candidates

    words = [f"term{i} alpha{i}" for i in range(60)]
    text = ". ".join(words) + ". " + ". ".join(reversed(words)) + "."
    a = mine_candidates(text, lang="english", top=25)
    b = mine_candidates(text, lang="english", top=25)
    assert len(a) == 25
    assert [c.phrase for c in a] == [c.phrase for c in b]
    assert all(a[i].strength >= a[i + 1].strength for i in range(len(a) - 1))


def test_extra_stopwords_extend_the_language_list():
    """The vault overlay contributes its own stopwords (course scaffolding such as
    "cfu", "lezione"); they must act as boundaries like the language's own."""
    text = "Ogni lezione tratta il percettrone. Ogni lezione tratta il percettrone."
    assert "lezione tratta il percettrone" in _phrases(text, lang="italian")
    with_extra = _phrases(text, lang="italian", stopwords=frozenset({"lezione", "tratta"}))
    assert "percettrone" in with_extra
    assert not any("lezione" in p for p in with_extra)


def test_empty_or_stopword_only_text_yields_nothing():
    assert _phrases("", lang="italian") == []
    assert _phrases("il la di e che", lang="italian") == []


# ---------------------------------------------------------------------------
# Name hygiene (2026-09-02 vault run): a span bounded by punctuation and
# stopwords is not yet a NAME. Six of the notes written from one Italian ML
# lecture were titled with a clause or with a fragment whose head noun a
# stopword had eaten. One junk example and one keeper per rule.
# ---------------------------------------------------------------------------

def _is_fragment(p):
    from silica.kernel.text.candidates import is_fragment
    return is_fragment(p)


def test_a_trailing_finite_verb_is_a_clause_not_a_name():
    """"L'algoritmo del percettrone consente di trovare..." yielded the note
    `percettrone consente`. A name never ends on the sentence's verb."""
    assert _is_fragment("percettrone consente")
    assert _is_fragment("iperpiano soddisfa")
    assert _is_fragment("data stewardship includes")   # the 2026-08-23 English twin

    assert not _is_fragment("Algoritmo online del percettrone")
    assert not _is_fragment("Errore del percettrone batch")
    # "Means" here is half of a hyphenated name, not the English verb: the
    # phrase is tokenised with the miner's own hyphen-aware word regex.
    assert not _is_fragment("Clustering K-Means")


def test_an_adjective_first_fragment_lost_its_head_noun():
    """The H1 "Appunti integrativi sul percettrone" starts on an overlay
    stopword, so the span opens on the adjective and the note is named after a
    fragment. An adjective followed by an Italian preposition or article is
    never the head of a term."""
    assert _is_fragment("integrativi sul percettrone")
    assert _is_fragment("logiche tramite percettrone")
    assert _is_fragment("separabili l'errore del percettrone")

    assert not _is_fragment("Classificatori lineari")
    assert not _is_fragment("gradiente discendente (significato geometrico)")
    # -ente and -ici head real nouns on this vault ("Coefficiente di Gini",
    # "Indici di posizione"), so they are not adjective-first suffixes.
    assert not _is_fragment("Coefficiente di variazione")
    assert not _is_fragment("Indici di posizione")
    # English adjectives end in -ive too: only Italian function words may fire
    # the rule, or every "objective of the model" would die with them.
    assert not _is_fragment("objective of the model")


def test_an_adverb_never_heads_a_term():
    assert _is_fragment("linearmente separabili")

    assert not _is_fragment("Stima parametrica")
    assert not _is_fragment("Epoca di addestramento")


def test_roman_numerals_and_short_lowercase_tokens_are_not_names():
    """The 2026-08-23 audit: `iii`, `rdf`, `pid` became notes. An acronym is
    kept by its CASE, which is why the markup seed must not be lowercased."""
    assert _is_fragment("iii")
    assert _is_fragment("VIII")
    assert _is_fragment("rdf")
    assert _is_fragment("pid")

    assert not _is_fragment("RDF")
    assert not _is_fragment("PID")
    assert not _is_fragment("Delta del livello nascosto")
    # Two-letter roman shapes name real concepts here (CV = cross validation,
    # ML = machine learning): the rule starts at three characters.
    assert not _is_fragment("CV")
    assert not _is_fragment("ML")


def test_the_pool_never_carries_a_fragment():
    """Wiring: the screens run inside the miner, so no consumer has to repeat
    them. "appunti" is an overlay stopword, which is what cuts the head noun."""
    text = ("Appunti integrativi sul percettrone. Il percettrone consente la "
            "separazione. Appunti integrativi sul percettrone. Il percettrone "
            "consente la separazione lineare.")
    phrases = _phrases(text, lang="italian", stopwords=frozenset({"appunti"}))
    assert "integrativi sul percettrone" not in phrases
    assert "percettrone consente" not in phrases
    assert "percettrone" in phrases
