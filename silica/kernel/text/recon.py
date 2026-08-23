# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Mechanical concept-recon over note content.

Vocabulary (stopwords + noise patterns) is supplied by the domain overlay seam;
see silica.kernel.text.overlay for the default English-generic overlay and the
load_overlay / get_active_overlay API.
"""
from __future__ import annotations

import re

from silica.kernel.text.overlay import DomainOverlay, get_active_overlay

MIN_LEN, MAX_LEN = 3, 50
TITLE_BONUS = 50
TOP_K_HITS = 3

LEADING_GARBAGE = re.compile(r'^[\W_]+')

# Leading IT/EN articles (and articulated prepositions) that YAKE/markup
# candidates drag along («della matrice Hessiana») — name hygiene only, the
# rest of the string is untouched (C3.4).
_LEADING_ARTICLE = re.compile(
    r"^(?:il|lo|la|i|gli|le|un|uno|una|del|dello|della|dei|degli|delle|"
    r"al|alla|nel|nella|the|an?)\s+|^(?:l|dell|un)'",
    re.IGNORECASE,
)


# Tail twin of _LEADING_ARTICLE: function words that can never END a concept
# name. Closed class on purpose — see _dangles for why the overlay's stopword
# set is the wrong instrument here.
_TRAILING_FUNCTION_WORDS = frozenset("""
il lo la i gli le un uno una del dello della dei degli delle al allo alla ai
agli alle nel nello nella nei negli nelle dal dallo dalla dai dagli dalle sul
sullo sulla sui sugli sulle di a da in con su per tra fra e o ed od ma che chi
cui come se non piu ogni quindi cioe ovvero oppure mentre perche quando dove
the an of on at to for with and or but as by from that which into over under
""".split())


def normalize(s: str) -> str:
    s = LEADING_GARBAGE.sub('', s)
    s = _LEADING_ARTICLE.sub('', s)
    return re.sub(r'\s+', ' ', s).rstrip()


def is_concept(s: str, overlay: DomainOverlay | None = None) -> bool:
    """Return True if *s* qualifies as a candidate concept under *overlay*.

    If *overlay* is None, the active vault overlay is resolved via
    ``get_active_overlay()`` (CONFIG-dependent, cached at module level).
    Pass an explicit overlay to make the call CONFIG-free.
    """
    if overlay is None:
        overlay = get_active_overlay()
    if s.lower().strip() in overlay.stopwords:
        return False
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return False
    if not re.search(r'[A-Za-zÀ-ÿ]{3,}', s):
        return False
    if _stutters(s):
        return False
    if _dangles(s):
        return False
    return not any(p.search(s) for p in overlay.noise_patterns)


def _stutters(s: str) -> bool:
    """True when a word repeats inside the phrase — a sliding n-gram artefact.

    YAKE's window walks the text, so prose that repeats a word within three
    tokens ("the holy angels, the holy ones") yields "holy angels holy" beside
    the real "holy angels", and both compete for the same candidate budget. No
    concept names the same thing twice.

    ponytail: exact lowercase tokens, no stemming — "angels/angel" is a
    different candidate, not a stutter, and collapsing those is dedup's job.
    """
    words = [w for w in re.findall(r'\w+', s.lower()) if len(w) > 2]
    return len(words) != len(set(words))


# Math stripping migrated to the kernel/text seam (C1): see text.strip_math.


def _dangles(s: str) -> bool:
    """True when the phrase ENDS on a function word — a clause, not a name.

    `normalize` already strips a LEADING article, because YAKE and markup both
    drag them in. The tail needed the same rule and never had it, so a lecture
    slide headed `## Da notare che` became a note called `Da notare che`, and
    YAKE n-grams cut mid-sentence (`intera presentazione del`, `Hidden Layer
    Nel`) survived beside real concepts. A concept name never ends on a
    conjunction, preposition or article — `Chain rules per le derivate`,
    `Algoritmi di apprendimento` and `Kernel trick` all end on content words.

    Closed class, hand-written, like `_LEADING_ARTICLE` — NOT `overlay.stopwords`.
    The overlay carries domain noise ("cfu") and generic nouns ("analysis"), and
    testing the tail against it deleted `Fisher discriminant analysis` and
    `Machine Learning (9 CFU)`. The all-caps guard is the other half: Italian
    "ai" is a preposition and "AI" is the field, and only case tells them apart
    — without it `Paradigmi di AI` and `Storia dell'AI` died too.

    ponytail: reject, don't strip. `Hidden Layer Nel` -> `Hidden Layer` would
    recover one good name, but `Da notare che` -> `Da notare` recovers nothing
    and the tail-stripped form is a new string to dedup against.
    """
    tokens = re.findall(r"[\w'’]+", s)
    if len(tokens) < 2:
        return False
    last = tokens[-1]
    if last.isupper():  # acronym, not a preposition: "Paradigmi di AI"
        return False
    return last.lower() in _TRAILING_FUNCTION_WORDS


def mentions_whole_word(phrase: str, line: str) -> bool:
    """True when *line* contains *phrase* as whole words, case-insensitively.

    The driver's body search is a substring scan and stays one: the agent's
    `silica_search_context` is a grep, and a prefix query is a feature there.
    Collision evidence is not a grep. Measured 2026-08-23 on the OpenAlex
    payload: 6 of 15 reported collisions had zero whole-word matches in the
    note they named (posi in position, gui in guide, MAG in image, ror in
    error), and `doi` counted 28 lines where 8 mention it. Each one was shown
    to the distiller as a vault collision to reason away.

    `\\w` is Unicode-aware on str patterns, so an accented letter is a word
    character and `rete` does not match inside `retè`.
    """
    words = phrase.casefold().split()
    if not words:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(w) for w in words) + r"(?!\w)"
    return re.search(pattern, line.casefold()) is not None


def is_title_match(c: str, stem: str) -> bool:
    c_lower, stem_lower = c.lower(), stem.lower()
    if c_lower == stem_lower: return True
    if c_lower in stem_lower or stem_lower in c_lower: return True
    c_words = set(re.findall(r'\w+', c_lower))
    s_words = set(re.findall(r'\w+', stem_lower))
    if c_words and s_words and (c_words.issubset(s_words) or s_words.issubset(c_words)):
        return True
    return False


def hit_score(body_count: int, in_title: bool) -> int:
    return body_count + (TITLE_BONUS if in_title else 0)


def rank_hits(raw: list, top_k: int = TOP_K_HITS) -> list:
    return sorted(raw, key=lambda h: hit_score(h["count"], h["in_title"]), reverse=True)[:top_k]


def collision_priority(c: dict) -> tuple:
    if c["best_match"] == "title": return (0, -c["total_hits"])
    if c["total_hits"] >= 3: return (1, -c["total_hits"])
    return (2, -c["total_hits"])
