"""kernel/text — the single seam for «note text → tokens/stems» (C1).

One deep module owns stripping (frontmatter, math, images, fences) and
tokenization (language stopwords, optional Snowball stemming); cooccurrence,
keyphrase, cohesion, classify and the MOC writer all cross the same seam, so
a stripping bug is fixed once, everywhere.
"""
from __future__ import annotations


SAMPLE = """---
tags: [math]
---
# Gradiente

La formula $\\nabla f$ guida la discesa e il blocco

$$E = \\frac{a}{b}$$

resta trasparente. ![[grafico.png]] ![alt](media/plot.jpeg)

```python
def fenced_token():
    pass
```

Chiude con \\alpha residuo.
"""


def test_clean_body_strips_frontmatter_math_images():
    from silica.kernel.text.text import clean_body

    out = clean_body(SAMPLE, fences=False)
    assert "tags:" not in out, "frontmatter must be stripped"
    assert "nabla" not in out and "frac" not in out, "math spans must be stripped"
    assert "alpha" not in out, "residual latex commands must be stripped"
    assert "grafico.png" not in out and "plot.jpeg" not in out, "images must be stripped"
    assert "discesa" in out, "prose must survive"


def test_clean_body_fences_are_callers_choice():
    from silica.kernel.text.text import clean_body

    # cooccurrence keeps fences (identifiers are the graph signal of code notes)
    assert "fenced_token" in clean_body(SAMPLE, fences=False)
    # keyphrase strips them (the miner must not rank code identifiers)
    assert "fenced_token" not in clean_body(SAMPLE, fences=True)


def test_tokens_folds_plurals_and_drops_stopwords():
    from silica.kernel.text.text import tokens

    sents = tokens("La rete neurale. Le reti neurali!", lang="italian")
    assert len(sents) == 2, "sentence boundary must be preserved"
    surfaces = [s for sent in sents for (_t, s) in sent]
    assert "la" not in surfaces and "le" not in surfaces, "stopwords dropped"
    stems = [[t for (t, _s) in sent] for sent in sents]
    assert stems[0] == stems[1], "singular and plural must share a stem"


def test_tokens_without_stemming_keeps_surfaces():
    from silica.kernel.text.text import tokens

    sents = tokens("Descrittori compatti", lang="italian", stem=False)
    assert [(t, s) for sent in sents for (t, s) in sent] == [
        ("descrittori", "descrittori"), ("compatti", "compatti"),
    ]


def test_classify_stems_match_cooccur_nodes_on_dirty_body():
    """C1: classify and the co-occurrence index share ONE notion of «body» —
    math and images never diverge the two profiles again."""
    from silica.kernel.organize.classify import _stems_from_body
    from silica.kernel.recall.cooccurrence import build_contribution

    body = (
        "La discesa $\\nabla f = \\frac{a}{b}$ converge. ![[plot.png]]\n\n"
        "Rete neurale profonda con retropropagazione."
    )
    stems = set(_stems_from_body(body, "italian"))
    nodes = set(build_contribution("", body, lang="italian")["nodes"])
    assert stems == nodes
    assert not any(s.startswith(("frac", "nabla", "png", "plot")) for s in stems)


def test_wikilinks_unwrap_to_words_without_moving_tokens():
    """C1: `[[a|b]]` is markup, so it must not reach a phrase-level consumer as
    syntax — keyphrase kept pool n-grams verbatim and emitted `decidibilità]]`.
    Unwrapping (not dropping) is what keeps the token-level consumers frozen:
    `_TOKEN_RE` already yielded `a` and `b` from the raw form, so the
    co-occurrence node set must come out byte-identical to the pre-fix one.
    """
    from silica.kernel.recall.cooccurrence import build_contribution
    from silica.kernel.text.text import clean_body

    body = (
        "La [[Decidibilità]] separa i [[linguaggi context-free|linguaggi CF]] "
        "dal resto, vedi [[Analisi asintotica#Ricorrenze]] e ![[Teorema di Rice]]. "
        # adjacent links with nothing between them: `]][[` used to do the
        # separating, so an unpadded unwrap glued the two aliases into one token
        "Algoritmi [[Analisi asintotica|asintoticamente]][[Efficienza| efficienti]]. "
        # markdown link: _URL_RE deletes the target and used to leave `]( )`
        "Vedi [Maltego](https://example.com/x) e [Ricorrenze](Analisi asintotica.md)."
    )
    out = clean_body(body, fences=True)
    assert "[[" not in out and "]]" not in out and "|" not in out, out
    assert "](" not in out, "markdown link scaffolding must not survive either"
    assert "Maltego" in out and "example.com" not in out, "text kept, URL target dropped"
    assert "Analisi asintotica.md" in out, "a relative target is prose, keep its words"
    assert "Decidibilità" in out and "linguaggi context-free" in out
    assert "linguaggi CF" in out, "the alias is prose too, not just the target"
    assert "Ricorrenze" in out, "a heading anchor names something; keep its word"

    # The frozen half: same nodes as the raw body would have produced, because
    # unwrapping yields exactly the words the tokenizer already found.
    raw_nodes = set(build_contribution("", body, lang="italian")["nodes"])
    unwrapped_nodes = set(build_contribution("", out, lang="italian")["nodes"])
    assert raw_nodes == unwrapped_nodes, raw_nodes ^ unwrapped_nodes


def test_moc_heading_is_english_whatever_the_note_language():
    """Vault strings are UI copy: emitted in English, the Italian spelling is
    only recognised. Two chunks of one hub used to get `## Da:` and `## From:`
    from the same source when the language sample flipped (2026-09-02)."""
    from silica.router.states.write import _moc_heading

    it_strong = "Questa è una lezione sulle reti neurali con il gradiente."
    assert _moc_heading("lezione.md", it_strong) == "## From: lezione.md"
    assert _moc_heading("lecture.md", "This lecture covers gradients.") == "## From: lecture.md"


def test_merge_moc_section_appends_into_a_legacy_italian_section():
    from silica.kernel.write.moc import merge_moc_section

    hub = "# Hub\n\n## Da: lezione\n\n- [[A]]\n"
    out = merge_moc_section(hub, "## From: lezione", ["- [[B]]"])
    assert out.count("## Da: lezione") == 1 and "## From: lezione" not in out
    assert "- [[A]]\n- [[B]]" in out


def test_tokens_min_len_is_callers_choice():
    from silica.kernel.text.text import tokens

    # zk/xj: verified absent from the (aggressive) english stopword list —
    # this asserts the length gate alone.
    text = "zk xj gradient"
    flat = lambda sents: [s for sent in sents for (_t, s) in sent]
    assert flat(tokens(text, lang="english", stem=False)) == ["gradient"]
    assert flat(tokens(text, lang="english", stem=False, min_len=2)) == [
        "zk", "xj", "gradient",
    ]
