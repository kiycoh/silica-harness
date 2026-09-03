# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Keyphrase candidate mining: one document, no model, no corpus, no download.

This is the pool generator behind `keyphrase.extract_keyphrases`. It replaced
the `yake` package on 2026-08-31 (ADR-0034) for a licensing reason first: the yake wheel
ships an AGPL-3.0 LICENSE (PyPI says LGPLv3, the classifier says GPLv3; every
reading is copyleft), which blocks any proprietary redistribution of the
engine. Nothing on PyPI fit the slot: the RAKE family splits on every stopword
and so can never emit "funzione di perdita", the spaCy family downloads a model
per language, KeyBERT ties the pool to the embedder this leg must survive
without. The two dependencies used here were already installed: `stop-words`
(33 languages) and `snowballstemmer`.

The salience statistics are the ones of the YAKE paper (Campos et al.,
Information Sciences 2020: term frequency, position, casing, relatedness to
context, sentence spread, aggregated over the phrase as a product), recomputed
here from the paper. A first attempt with a plain count-times-length score lost
recall on 246 real vault notes (.220 vs .259) because frequent unigrams
outranked the multi-word terms the notes actually link to; the paper's
aggregation favours multi-word phrases of good words, which is what a note
title is.

Sweep of 2026-08-31 on those 246 notes (gold = the note's own wikilinks,
fallback rank, yake .259 recall / .159 precision; the harness is
scripts/bench_keyphrase.py): this configuration .256 / .157, and .309 / .171
against yake's .314 / .173 with the embedder ranking the pool. Measured and NOT kept: the paper's stopword-bridge penalty (.245), a
damping of single-mention phrases (.242), four content words per span (.256 /
.167 but longer, junkier titles), absorbing singleton extensions of repeated
phrases (.257 / .152). Kept because measured: the redundancy rule (+.006 over
none) and stem collapsing (+.003 over surface keys).

What the consumer relies on: a pool of ~100 phrases with a rough, deterministic
salience order. With the embedder up the order is discarded (cosine + MMR rank
the pool); in the fallback it IS the rank, so it has to be sane, not perfect.

Two defects of the yake pool that this mines away, both measured on the vault:
  - a span may hold four content words, so "stimatore a massima verosimiglianza"
    is a candidate and not a fragment `_complete_phrases` has to repair;
  - inflected variants collapse at stem level (rete neurale / reti neurali),
    so one concept takes one pool slot and the count of both.

Candidate = a run of tokens inside one punctuation-bounded chunk that starts
and ends on a content word and may carry function words INSIDE (Romance terms
need them). A sub-phrase that never occurs outside one repeated longer
candidate is dropped: it would spend a pool slot on the fragment of a term
already there.
"""
from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

from silica.kernel.text import language

# A word starts with a letter of any script and may continue with letters,
# digits, and inner apostrophes/hyphens ("dell'agente", "multi-agent", "gpt-4").
# Anything else that is word-like (a number, "3d") is a boundary, not content:
# a year or a figure never names a concept and must not glue two terms together.
_WORD_RE = re.compile(r"[^\W\d_][^\W_]*(?:['’\-][^\W_]+)*|[^\W_]+")
# Chunk boundaries: sentence punctuation, brackets, quotes, list/markup
# remnants (`*`, `#`, `>`, `|`), math leftovers. A span never crosses one.
_BOUNDARY_RE = re.compile(r"[.!?;:,()\[\]{}\"“”„«»‹›…/\\|*#>+=<%$€£§~^`\n\r\t]+")
# A line break bounds a span but does not end a sentence: the position and
# spread features count sentences, and a note full of list items and headings
# would otherwise read as 29 "sentences" where the prose has 12, pushing every
# median mention later and flattening the position signal (measured on the
# vault note "Variabile statistica").
_SENTENCE_END_RE = re.compile(r"[.!?]")

# Content words per span. Function words inside the span are not counted, so
# three already covers "stimatore a massima verosimiglianza" and "capacità di
# generalizzazione del modello", the truncations yake's three-TOKEN window
# produced. Four let singleton four-word spans dominate the fallback rank
# (the paper's product favours length), measured worse on 246 vault notes.
MAX_CONTENT_WORDS = 3
# One function word inside a span, as in yake's three-token window (a
# candidate never starts or ends on one, so its window held at most one). Two
# let "neurale a ogni iterazione del training" outrank "discesa del gradiente"
# on a three-sentence note: the paper's product rewards every extra word of an
# early sentence. Terms with two bridges ("funzione di costo del contesto")
# reach the pool through `_complete_phrases`' boundary snap instead.
MAX_INNER_RUN = 1
MAX_INNER_TOTAL = 1

_CONTENT, _FUNCTION, _BOUNDARY = 0, 1, 2


@dataclass(frozen=True)
class Candidate:
    phrase: str      # surface form: the author's most frequent wording and casing
    strength: float  # salience, higher = better; ordering only, not calibrated
    count: int       # whole-phrase occurrences (all inflections summed)


@dataclass
class _Word:
    """Per-term statistics behind the paper's five features (one per stem)."""
    tf: int = 0
    upper: int = 0                 # capitalised mentions, sentence-initial excluded
    acronym: int = 0
    sentences: set[int] = field(default_factory=set)
    left: Counter = field(default_factory=Counter)
    right: Counter = field(default_factory=Counter)


def _stemmer_for(lang: str) -> Callable[[str], str]:
    """Snowball stemmer for `lang`, identity when the language has none.

    Built per call, not cached: a snowballstemmer instance is not reentrant
    (see text.py's per-thread cache) and construction costs ~0.5 µs.
    """
    try:
        import snowballstemmer
        stemmer = snowballstemmer.stemmer("english" if lang == "auto" else lang)
    except Exception:
        # An unknown language degrades to surface-form keys: variants stop
        # collapsing, extraction keeps working (language.py's contract).
        return lambda w: w
    return stemmer.stemWord


def _acronym(tok: str) -> bool:
    return 2 <= len(tok) <= 6 and tok.isupper() and tok.isalpha()


def _strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    )


def _kind(tok: str, stop: frozenset[str]) -> int:
    """Content word, function word (may sit INSIDE a span), or boundary.

    Digits-first tokens (years, figures) and single letters end a span. A
    single letter is a function word only when it is a vowel: the articles and
    prepositions that are one letter long in these languages all are ("a",
    "e", "o", "è", "y"), while the English stop list carries every letter of
    the alphabet, so without the rule the variable in "model x reached" would
    read as a preposition and glue two terms together.
    """
    if not tok[0].isalpha():
        return _BOUNDARY
    low = tok.lower()
    if len(low) == 1:
        return _FUNCTION if low in stop and _strip_accents(low) in "aeiouy" else _BOUNDARY
    return _FUNCTION if low in stop else _CONTENT


# ---------------------------------------------------------------------------
# Name hygiene: a span bounded by punctuation and stopwords is not yet a NAME
# ---------------------------------------------------------------------------
# Measured on a real Italian ML vault, 2026-09-02: six notes written from one
# lecture were titled with a clause ("percettrone consente", "iperpiano
# soddisfa") or with a fragment whose head noun a stopword had eaten (the H1
# "Appunti integrativi sul percettrone" gives "integrativi sul percettrone",
# because the overlay lists "appunti"). The 2026-08-23 audit had recorded the
# English twins: `rdf`, `pid`, `iii`, `data stewardship includes`.
#
# The screens below are morphological, not syntactic: two closed word lists,
# four suffixes and a numeral parser, no tagger and no new dependency. They run
# inside the miner and on the markup seeds (keyphrase.py), the only two ways a
# phrase becomes a title.
#
# ponytail: a POS tagger (spaCy it_core_news_sm, ~15 MB downloaded per
# language) is the upgrade that makes agreement and part of speech exact,
# and ADR-0034 rejected exactly that cost for the pool. Reopen when a vault run
# shows junk of a shape no closed list can name, or when a keeper is lost to
# one of these rules that is not fixable by editing its list.

# The words are the gate, not the detected language: the same lecture resolves
# to english before `clean_body` and to italian after it (measured on Lezione
# 13, whose LaTeX fills the text with single letters that are all English
# stopwords), so a lang-gated rule would have missed most of the junk it was
# written for. No Italian form below is an English word and vice versa.
#
# An EXPLICIT list, not the 3rd-person endings (-a/-e/-isce/-ono/-ano): those
# are the endings of most Italian nouns too, and a suffix test rejects "Stima
# parametrica", "Epoca di addestramento" and "varianza campionaria". Forms that
# are as often a noun as a verb are left OUT on purpose (misura, stima,
# verifica, classifica, forma, causa, somma, deriva, dice, mostra in Italian;
# returns, yields, holds, works, needs, shows in English): each of them names a
# concept somewhere in a real vault.
_VERB_TAILS: frozenset[str] = frozenset("""
è sono ha hanno può possono deve devono viene vengono esiste esistono
dipende dipendono vale valgono risulta risultano consente consentono
permette permettono indica indicano rappresenta rappresentano definisce
definiscono esprime esprimono soddisfa soddisfano significa significano
corrisponde corrispondono contiene contengono include includono restituisce
restituiscono produce producono genera generano calcola calcolano ottiene
ottengono utilizza utilizzano usa usano minimizza massimizza converge
convergono cresce decresce aumenta diminuisce applica assume considera
fornisce serve tende riduce aggiorna apprende richiede riceve appartiene
denota trova separa predice implica coincide equivale comporta presenta
occorre accade avviene diventa distribuisce
is are was were be been has have had does do did includes include uses means
allows allow provides requires contains gives makes becomes refers depends
consists enables supports describes defines represents produces applies
occurs exists can may must should will would
""".split())

# Plural adjective / participle endings that open a headless fragment. -ente
# and -ici are deliberately absent: on the 1520 wikilink titles of the vault
# they head real nouns ("Coefficiente di variazione", "Indici di posizione",
# "gradiente discendente"), so including them would have deleted ten concepts
# to win one.
_HEADLESS_SUFFIXES = ("ivi", "ive", "ili", "iche")

# Italian prepositions and articles only, and no bare "a"/"in"/"e"/"o": those
# are English words too, and the rule must not fire on "alternative in the
# loop" or "objective of the model". This list IS the language gate for the
# adjective-first rule, since English adjectives end in -ive as well.
_IT_FUNCTION_AFTER_HEAD: frozenset[str] = frozenset("""
di da del dello della dei degli delle al allo alla ai agli alle dal dallo
dalla dai dagli dalle nel nello nella nei negli nelle sul sullo sulla sui
sugli sulle per con su tra fra il lo la i gli le un uno una tramite mediante
attraverso
""".split())
# The tokeniser keeps an elision inside one token ("l'errore", "dell'iperpiano"),
# so the article that follows the cut head has to be matched as a prefix.
_ELIDED_IT_ARTICLE = re.compile(r"^(?:l|dell|nell|all|dall|sull|un|d)['’]", re.IGNORECASE)

# Well-formed roman numerals. Any-subset-of-IVXLCDM would eat "ML" and "CV",
# which name concepts in this vault (18 titles carry CV); the three-character
# floor keeps the two-letter shapes as well. Residual ceiling: a three-letter
# acronym that IS a well-formed numeral (CLI, DIV, MIX) is lost, measured zero
# across the vault's note titles and link targets.
_ROMAN_RE = re.compile(
    r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})", re.IGNORECASE)


def is_fragment(phrase: str) -> bool:
    """True when `phrase` is a clause or a headless fragment, not a name.

    Case is evidence and not decoration here: an acronym earns its exemption
    from the short-token rule by being upper case, which is why a markup seed
    must reach this function in the author's own casing (keyphrase.py).
    """
    words = _WORD_RE.findall(phrase)
    if not words:
        return True

    # (a) ends on a finite verb: the sentence's predicate, not the term's tail.
    # Same all-caps guard as recon._dangles: "usa" is a verb, "USA" is not.
    last = words[-1]
    if not last.isupper() and last.lower() in _VERB_TAILS:
        return True

    head = words[0].lower()
    # (b) an Italian -mente adverb never heads a name ("linearmente separabili"
    # is a property of something, not a thing). The seven-character floor keeps
    # the noun "mente" itself out of the rule.
    if len(head) >= 7 and head.endswith("mente"):
        return True
    # (b') an adjective followed by a preposition or article: the head noun was
    # cut off to its left ("[Appunti] integrativi sul percettrone").
    if len(words) > 1 and head.endswith(_HEADLESS_SUFFIXES):
        after = words[1]
        if after.lower() in _IT_FUNCTION_AFTER_HEAD or _ELIDED_IT_ARTICLE.match(after):
            return True

    # (c) a lone short token is a name only as an acronym, and never as a
    # section number.
    if len(words) == 1:
        if len(last) >= 3 and _ROMAN_RE.fullmatch(last):
            return True
        if len(last) < 4 and not _acronym(last):
            return True
    return False


def mine_candidates(
    text: str,
    *,
    lang: str,
    stopwords: frozenset[str] = frozenset(),
    top: int = 100,
    max_words: int = MAX_CONTENT_WORDS,
) -> list[Candidate]:
    """Ranked candidate phrases from `text`, best-first, at most `top`.

    `lang` is a Snowball name ("italian"); `stopwords` extends the language's
    own list (the vault overlay's scaffolding words). Never raises on an
    unknown language or empty input: returns what it can, or [].
    """
    if not text or not text.strip():
        return []
    stop = language.stopwords_for(lang) | stopwords
    stem = _stemmer_for(lang)

    parsed, words, n_sentences = _analyse(text, stop, stem)
    if not words:
        return []
    score = _word_scores(words, n_sentences)

    # --- pass 3: spans -> phrase occurrences --------------------------------
    count: Counter[tuple[str, ...]] = Counter()
    surfaces: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    first_seen: dict[tuple[str, ...], int] = {}
    cased: dict[tuple[str, ...], list[Counter[str]]] = {}
    order: list[tuple[str, ...]] = []
    for ci, (toks, kinds, stems, si, starts) in enumerate(parsed):
        for i, k in enumerate(kinds):
            if k != _CONTENT:
                continue
            n_content = inner_run = inner_total = 0
            key_parts: list[str] = []
            for j in range(i, len(toks)):
                kj = kinds[j]
                if kj == _BOUNDARY:
                    break
                if kj == _FUNCTION:
                    inner_run += 1
                    inner_total += 1
                    if inner_run > MAX_INNER_RUN or inner_total > MAX_INNER_TOTAL:
                        break
                    continue
                n_content += 1
                if n_content > max_words:
                    break
                inner_run = 0
                key_parts.append(stems[j])
                key = tuple(key_parts)
                span = toks[i:j + 1]
                if key not in count:
                    order.append(key)
                    first_seen[key] = ci
                    cased[key] = [Counter() for _ in key]
                count[key] += 1
                surfaces[key][" ".join(span).lower()] += 1
                initial = starts and i == 0
                kk = 0
                for pos in range(i, j + 1):
                    if kinds[pos] != _CONTENT:
                        continue
                    if not (initial and pos == i):
                        cased[key][kk][toks[pos]] += 1
                    kk += 1

    # Drop a sub-phrase that only ever occurs inside one longer candidate: every
    # sub-span of an occurrence was emitted too, so equal counts mean "never
    # seen on its own". A shorter phrase with more mentions stands. Only a
    # REPEATED longer span counts as a term: one mention of "promise remains"
    # is no evidence of a collocation and must not swallow "promise" (the same
    # >= 2 rule `_complete_phrases` learned on "Errore quadratico atteso dipende").
    redundant: set[tuple[str, ...]] = set()
    for key in order:
        n = len(key)
        if n < 2 or count[key] < 2:
            continue
        for width in range(1, n):
            for start in range(0, n - width + 1):
                sub = key[start:start + width]
                if count.get(sub) == count[key]:
                    redundant.add(sub)

    # --- pass 4: the paper's phrase score, inverted so higher = better ------
    #   S(kw) = prod S(w) / (TF(kw) * (1 + sum S(w)))      (lower = better)
    out: list[tuple[float, int, str, int]] = []
    for key in order:
        if key in redundant:
            continue
        prod = 1.0
        total = 0.0
        for st in key:
            s = score[st]
            prod *= s
            total += s
        strength = count[key] * (1.0 + total) / max(prod, 1e-12)
        surface = surfaces[key].most_common(1)[0][0]
        phrase = _recase(surface, cased[key], stop)
        # After `_recase`, because the acronym exemption reads the casing the
        # author attested and not the lowercased surface key.
        if is_fragment(phrase):
            continue
        out.append((strength, first_seen[key], phrase, count[key]))

    out.sort(key=lambda r: (-r[0], r[1], r[2]))
    return [Candidate(phrase=p, strength=s, count=c) for s, _, p, c in out[:top]]


_Chunk = tuple[list[str], list[int], list[str], int, bool]


def _analyse(
    text: str, stop: frozenset[str], stem: Callable[[str], str],
) -> tuple[list[_Chunk], dict[str, _Word], int]:
    """Tokenise into punctuation-bounded chunks and gather per-term statistics.

    Each chunk carries its tokens, their kinds, their stems, the index of the
    sentence it belongs to and whether it opens that sentence (a capital there
    is not casing evidence: "Gradient descent converges" says nothing about
    proper nouns). Sentence indices feed the position and spread features.
    """
    chunks = _BOUNDARY_RE.split(text)
    seps = _BOUNDARY_RE.findall(text)
    opens = [True] + [bool(_SENTENCE_END_RE.search(s)) for s in seps]
    parsed: list[_Chunk] = []
    sent = -1
    for ci, chunk in enumerate(chunks):
        starts = opens[ci] if ci < len(opens) else False
        if starts:
            sent += 1
        toks = _WORD_RE.findall(chunk)
        if not toks:
            continue
        kinds = [_kind(t, stop) for t in toks]
        stems = [stem(t.lower()) if k == _CONTENT else t.lower() for t, k in zip(toks, kinds)]
        parsed.append((toks, kinds, stems, max(sent, 0), starts))
    n_sentences = max(sent + 1, 1)

    words: dict[str, _Word] = defaultdict(_Word)
    for toks, kinds, stems, si, starts in parsed:
        content_idx = [i for i, k in enumerate(kinds) if k == _CONTENT]
        for n, i in enumerate(content_idx):
            w = words[stems[i]]
            w.tf += 1
            w.sentences.add(si)
            tok = toks[i]
            if _acronym(tok):
                w.acronym += 1
            elif tok[0].isupper() and not (starts and i == 0):
                w.upper += 1
            if n > 0:
                w.left[stems[content_idx[n - 1]]] += 1
            if n + 1 < len(content_idx):
                w.right[stems[content_idx[n + 1]]] += 1
    return parsed, dict(words), n_sentences


def _word_scores(words: dict[str, _Word], n_sentences: int) -> dict[str, float]:
    """The paper's term score S(w), lower = more likely part of a keyphrase.

        S(w) = (T_rel * T_pos) / (T_case + T_f / T_rel + T_sent / T_rel)

    T_f is frequency normalised by the corpus-free mean + std of the document
    itself; T_pos favours early first mentions (log2 log2 of the median
    sentence); T_case counts capitalised or acronym mentions; T_rel penalises a
    term that co-occurs with many different neighbours (a generic word, not a
    term); T_sent is the share of sentences the word appears in.
    """
    tfs = [w.tf for w in words.values()]
    mean_tf = statistics.fmean(tfs)
    std_tf = statistics.pstdev(tfs) if len(tfs) > 1 else 0.0
    max_tf = max(tfs)
    out: dict[str, float] = {}
    for st, w in words.items():
        t_case = max(w.upper, w.acronym) / (1.0 + math.log(w.tf))
        # Natural log, as in the reference implementation (the paper prints
        # log2): with log2 an early first mention scores 0.66 and the position
        # signal barely moves the product; with ln it scores 0.09 and early
        # terms dominate, which is the behaviour the fallback rank inherited.
        t_pos = math.log(math.log(3.0 + statistics.median(sorted(w.sentences))))
        t_f = w.tf / (mean_tf + std_tf) if (mean_tf + std_tf) > 0 else 0.0
        left_total = sum(w.left.values())
        right_total = sum(w.right.values())
        dl = len(w.left) / left_total if left_total else 0.0
        dr = len(w.right) / right_total if right_total else 0.0
        t_rel = 1.0 + (dl + dr) * (w.tf / max_tf)
        t_sent = len(w.sentences) / n_sentences
        out[st] = (t_rel * t_pos) / (t_case + t_f / t_rel + t_sent / t_rel)
    return out


def _recase(surface: str, evidence: list[Counter[str]], stop: frozenset[str]) -> str:
    """Rebuild the phrase with each content word in its best-attested casing.

    Best = the most frequent form among mentions that are NOT sentence-initial
    (ties go to the form seen first). No such mention: lowercase, unless the
    only form is an acronym, which is a proper term wherever it stands.
    """
    out: list[str] = []
    k = 0
    for w in surface.split():
        if _kind(w, stop) != _CONTENT:
            out.append(w)
            continue
        forms = evidence[k] if k < len(evidence) else Counter()
        k += 1
        if forms:
            best = max(forms.items(), key=lambda kv: kv[1])[0]
            # Only the casing is taken from the evidence; the surface (the
            # most frequent inflection) decides which word stands here.
            out.append(_apply_casing(w, best))
        else:
            out.append(w)
    return " ".join(out)


def _apply_casing(word: str, attested: str) -> str:
    if attested.lower() != word:
        # Evidence comes from another inflection ("reti" for "rete"): copy its
        # shape (acronym / capitalised / lower) rather than its letters.
        if _acronym(attested):
            return word.upper()
        if attested[:1].isupper():
            return word[:1].upper() + word[1:]
        return word
    return attested
