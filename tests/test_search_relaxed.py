"""silica_search scores every title once, instead of testing the phrase whole.

The old shape was one substring test against one string, so a query that says
more than the title does ("regressione lineare multipla" vs `Regressione
lineare.md`) answered 0 and the caller concluded the note was missing. Words
now score in the same scan that used to do the phrase, and the literal tiers
still answer alone when they answer at all, so `matched: 1` keeps meaning
"this is the note".
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from silica.tools.atomic import silica_search


def _ref(name: str, folder: str = "F"):
    return SimpleNamespace(name=name, path=f"{folder}/{name}.md")


VAULT = [_ref("Regressione lineare"), _ref("Regressione logistica"),
         _ref("Reti neurali"), _ref("Reti neurali convolutive"),
         _ref("Autoencoders"), _ref("Probabilità bayesiana"), _ref("Poincare")]


def _driver(drv):
    """search_names as both backends implement it: substring, case-folded."""
    drv.search_names.side_effect = lambda q: [
        r for r in VAULT if q.lower() in r.name.lower()]
    return drv


def test_whole_query_match_answers_alone():
    with patch("silica.tools.atomic.DRIVER") as drv:
        _driver(drv)
        res = silica_search(query="regressione logistica")
    # `Regressione lineare` carries one word; it must not dilute a literal hit.
    assert res == {"paths": ["F/Regressione logistica.md"], "matched": 1}


def test_query_longer_than_the_title_still_finds_it():
    with patch("silica.tools.atomic.DRIVER") as drv:
        _driver(drv)
        res = silica_search(query="regressione lineare multipla")
    # No title carries all three words, so the head is the partial one, flagged.
    assert res["paths"][0] == "F/Regressione lineare.md"
    assert res["matched"] == 2
    assert "silica_semantic_search" in res["relaxed"]


def test_all_words_present_but_scattered_is_a_hit_not_a_guess():
    """`Reti neurali convolutive` carries both words of "reti convolutive" out of
    order — a fact about the title, so it answers alone and unflagged."""
    with patch("silica.tools.atomic.DRIVER") as drv:
        _driver(drv)
        res = silica_search(query="reti convolutive")
    assert res == {"paths": ["F/Reti neurali convolutive.md"], "matched": 1}


def test_one_enumeration_per_search():
    """The phrase and its words are scored in the same scan: a second lookup is
    a second round-trip on the ws backend and finds nothing this scan missed."""
    with patch("silica.tools.atomic.DRIVER") as drv:
        _driver(drv)
        silica_search(query="regressione lineare multipla")
    assert drv.search_names.call_count == 1


def test_single_word_miss_stays_a_miss():
    """One word has nothing to relax to — a 0 there is the honest answer."""
    with patch("silica.tools.atomic.DRIVER") as drv:
        _driver(drv)
        res = silica_search(query="dropout")
    assert res["paths"] == [] and res["matched"] == 0
    # …but not in silence: a bare 0 is what the caller read as "not in the vault".
    assert "silica_search_context" in res["hint"]


def test_query_of_only_short_words_falls_back_to_the_phrase():
    """`di`/`la` match half the vault; under 3 chars they only stand in whole."""
    with patch("silica.tools.atomic.DRIVER") as drv:
        _driver(drv)
        res = silica_search(query="la di")
    assert res["paths"] == [] and res["matched"] == 0


def test_case_does_not_decide_a_match():
    with patch("silica.tools.atomic.DRIVER") as drv:
        _driver(drv)
        res = silica_search(query="REGRESSIONE LOGISTICA")
    assert res["paths"] == ["F/Regressione logistica.md"]


def test_accents_fold_in_both_directions():
    """44 of 872 titles on a real Italian vault carry one; typing it must not be
    the price of reaching them, and a title spelled without it must still answer
    a query that has it."""
    with patch("silica.tools.atomic.DRIVER") as drv:
        _driver(drv)
        bare = silica_search(query="probabilita")
        accented = silica_search(query="Poincaré")
    assert bare["paths"] == ["F/Probabilità bayesiana.md"]
    assert accented["paths"] == ["F/Poincare.md"]
