# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Content-based concept extraction (UKE).

The old `recon.extract_concepts` keyed concepts on markdown markup, so prose
papers — concepts living in unmarked sentences — extracted to nearly nothing.
This module instead generates candidates from the *content* and ranks them.
Design split (validated on a real corpus, see the eval and the spec):

  - **miner = candidate generator** (`candidates.mine_candidates`, in-house
    since 2026-08-31; YAKE before, retired for its AGPL licence, see that
    module). Its rank is a rough salience order, discarded once an embedder is
    available — the miner only supplies the candidate *pool*.
  - **embedder + MMR = the ranker.** Candidates are ordered by cosine to the
    document theme, with MMR for diversity (plain cosine collapses onto
    near-synonym clusters). This is the primary signal.
  - **structural (markup) = boost.** Concepts that appear in a heading/bold/
    acronym get a relevance bonus — lifts lecture-genre concepts; on prose with
    no markup the boost set is empty (no effect).
  - **embedder down => fall back to the mined rank** (degraded, deterministic).

Return shape is `list[ConceptCandidate]`, ranked best-first.
See docs/superpowers/specs/2026-06-19-concept-recon-design.md.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from silica.kernel.text import language
from silica.kernel.recall.embed import _cosine, document_theme_vector
from silica.kernel.text.candidates import is_fragment, mine_candidates
from silica.kernel.text.overlay import DomainOverlay, overlay_for_lang
from silica.kernel.text.recon import is_concept, normalize
from silica.kernel.text.text import clean_body

# Cutoff knobs (calibration — tune on a real paper + lecture via the eval).
# Tuned on the eval (3 real docs). Concept density varies wildly across genres
# (~40 tok/concept for a paper, ~8 for a dense distilled lecture note), so one
# linear ratio is a compromise; a density-aware cutoff (cosine elbow) would serve
# short dense notes better — deferred until the linear clamp proves insufficient.
# ponytail: linear clamp, tuned once on a 3-doc eval (paper + lectures). No
# current harness exercises this path (golden/LME/MuSiQue never call recon),
# so "tune via the eval" is not actionable today — recalibrate only if ingest
# capture quality regresses, with an ingest-extraction eval built for it.
TOKENS_PER_CONCEPT = 20
MIN_CONCEPTS = 1          # a note may map to a single concept — no forced padding
MAX_CONCEPTS = 40
# Candidates the miner proposes (also the rerank pool); operator knob. The
# legacy name SILICA_YAKE_POOL stays recognized: a pin in someone's ~/.silica/.env
# must not silently revert to the default because the extractor changed.
POOL_SIZE = int(os.getenv("SILICA_KEYPHRASE_POOL") or os.getenv("SILICA_YAKE_POOL") or "100")

# Rerank knobs (Phase 2 — tune via the eval).
MMR_LAMBDA = 0.6          # relevance vs diversity in MMR; lower = more diverse
STRUCT_BOOST = 0.3        # relevance bonus for a concept present in markup


@dataclass
class ConceptCandidate:
    phrase: str
    score: float                       # ordering only (a cost; lower = better). NOT calibrated.
    evidence: list[str] = field(default_factory=list)  # provenance/debug, e.g. ["mine:3.40"]
    # Corroboration tier (vocabulary mirrors links, see analyst_plan.py):
    #   EXTRACTED — structurally corroborated (author markup; second, embedder-free axis)
    #   INFERRED  — single signal only (embedding cosine or mined rank), uncorroborated
    confidence: str = "INFERRED"


def _pool_leg(text: str, overlay: DomainOverlay, lang: str) -> list[ConceptCandidate] | None:
    """Mined candidates (best-first), filtered through the overlay.

    Abstains (None) when the text holds no content word, so the caller's
    "[] only when both legs abstain" contract holds. The miner unions the
    language stopwords with the overlay's: passing the overlay list ALONE would
    drop the ~300 function words of the language (the replace-vs-union bug of
    the YAKE era, kept as a test). Its strength is inverted into the cost the
    `score` field carries (lower = better), so callers ordering by score keep
    working unchanged.
    """
    mined = mine_candidates(
        text, lang=lang, stopwords=frozenset(overlay.stopwords), top=POOL_SIZE,
    )
    out: list[ConceptCandidate] = []
    for c in mined:
        norm = normalize(c.phrase)
        if is_concept(norm, overlay=overlay):
            out.append(ConceptCandidate(phrase=norm, score=1.0 / c.strength,
                                        evidence=[f"mine:{c.strength:.2f}"]))
    return out or None


# ---------------------------------------------------------------------------
# Structural (markup) signal — re-added for the boost; pure regex extractors.
# ---------------------------------------------------------------------------

def from_headings(content: str) -> set:
    return {m.group(1) for m in re.finditer(r'^#{1,4}\s+(.+?)\s*$', content, re.MULTILINE)}


def from_bold(content: str) -> set:
    return {m.group(1) for m in re.finditer(r'\*\*(.+?)\*\*', content)}


def from_acronyms(content: str) -> set:
    return set(re.findall(r'\b[A-Z]{2,6}\b', content))


def _structural_concepts(body: str, overlay: DomainOverlay) -> dict[str, str]:
    """Markup concepts (heading/bold/acronym), keyed lowercase, valued as written.

    Empty on prose with no markup — that is the leg "abstaining" for the boost.

    The key is what the boost and the confidence stamp test membership against,
    and stays lowercase. The VALUE is new: this function used to return the
    lowercased string, and `_seed_structural` reused it AS the candidate
    phrase, so an acronym the author put in a heading reached the vault as a
    note named `pid` or `rdf` (audit of 2026-08-23). Sorting before the
    setdefault makes the collision deterministic and picks the upper-case form,
    since "PID" sorts before "pid".

    `is_fragment` screens here too: `from_acronyms` matches any run of 2-6
    capitals, so a numbered section seeded `III` as a concept. One home for the
    name-hygiene rules, so headings and bold get the same treatment as the
    mined pool.
    """
    raw = from_headings(body) | from_bold(body) | from_acronyms(body)
    out: dict[str, str] = {}
    for r in sorted(raw):
        n = normalize(r)
        if is_concept(n, overlay=overlay) and not is_fragment(n):
            out.setdefault(n.lower(), n)
    return out


# ---------------------------------------------------------------------------
# Embedder + MMR ranker
# ---------------------------------------------------------------------------

def _mmr(vecs, theme, k, lam: float = MMR_LAMBDA, rel=None) -> list[int]:
    """Maximal Marginal Relevance selection. Returns selected indices, best-first.

    `rel[i]` is the relevance of candidate i (default: cosine to `theme`). Each
    pick maximises `lam*rel - (1-lam)*max similarity to already-picked`, so
    near-duplicates of a selected candidate are demoted.

    Candidate-candidate similarity is taken once as a matrix and each candidate
    carries a running max against the already-picked set. The per-pair form
    re-derived every cosine on every iteration: on a real pool (POOL_SIZE=100
    vectors of the embedder's width) that was ~166k `_cosine` calls, ~1.1M
    `np.asarray` conversions and 26-46s per note, which is 94% of RECON.
    Vectors must be uniform width — `_rerank` abstains before calling here.
    """
    if not vecs:
        return []
    M = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    M = M / np.where(norms == 0.0, 1.0, norms)  # a zero row stays zero → sim 0
    sim = M @ M.T
    if rel is None:
        rel = [_cosine(v, theme) for v in vecs]
    cand = list(range(len(vecs)))
    # Running max similarity to the picked set. -inf, not 0.0: cosine may be
    # negative and the per-pair form never clamped it.
    best = [float("-inf")] * len(vecs)
    sel: list[int] = []
    while cand and len(sel) < k:
        if not sel:
            i = max(cand, key=lambda i: rel[i])
        else:
            i = max(cand, key=lambda i: lam * rel[i] - (1 - lam) * best[i])
        sel.append(i)
        cand.remove(i)
        for j in cand:
            s = float(sim[j, i])
            if s > best[j]:
                best[j] = s
    return sel


def _rerank(
    pool: list[ConceptCandidate],
    body: str,
    overlay: DomainOverlay,
    embedder,
) -> list[ConceptCandidate] | None:
    """Rerank the mined pool by embedder cosine-to-theme + MMR + structural boost.

    Returns None (abstain -> caller falls back to the mined rank) when no embedder, an
    empty document theme, or an embedding failure.
    """
    if embedder is None:
        return None
    theme = document_theme_vector(embedder, body)
    if not theme:
        return None
    phrases = [c.phrase for c in pool]
    try:
        vecs = embedder.embed(phrases)
    except Exception:
        return None
    if not vecs:
        return None
    # Ragged reply (A6, guarded the same way in NOVELTY/COLLISION): a short
    # list would zip short and silently drop the tail, and mixed widths would
    # rank a pool on similarities that are not comparable. Abstain instead.
    if len(vecs) != len(phrases) or len({len(v) for v in vecs}) != 1:
        return None

    structural = _structural_concepts(body, overlay)
    rel = [
        _cosine(vecs[i], theme) + (STRUCT_BOOST if phrases[i].lower() in structural else 0.0)
        for i in range(len(pool))
    ]
    order = _mmr(vecs, theme, k=len(pool), lam=MMR_LAMBDA, rel=rel)

    out: list[ConceptCandidate] = []
    for i in order:
        ev = [f"embed:{_cosine(vecs[i], theme):.2f}"]
        if phrases[i].lower() in structural:
            ev.append("struct")
        out.append(ConceptCandidate(phrase=pool[i].phrase, score=rel[i], evidence=ev))
    return out


def _seed_structural(
    body: str, overlay: DomainOverlay, pool: list[ConceptCandidate],
) -> list[ConceptCandidate]:
    """Prepend markup concepts (heading/bold/acronym) absent from the mined pool.

    Author markup is frequency-independent: it recovers concepts the miner can't
    reach — terms past its span (MAX_CONTENT_WORDS=3) and low-count mentions
    that fell under the pool cap. With an embedder these are reranked by cosine
    like any candidate; in the fallback they lead, since author markup is the
    strongest deterministic signal we have.
    """
    structural = _structural_concepts(body, overlay)  # lower key -> author casing
    have = {c.phrase.lower() for c in pool}
    seeded = [ConceptCandidate(phrase=cased, score=0.0, evidence=["struct"])
              for key, cased in sorted(structural.items()) if key not in have]
    return seeded + pool


_PHRASE_WORD = r"[\w\u00C0-\u00FF'-]+"


def _complete_phrases(
    body: str,
    ranked: list[ConceptCandidate],
    stop: frozenset[str],
    overlay: DomainOverlay,
    max_words: int = 6,
) -> list[ConceptCandidate]:
    """Snap truncated candidates to their phrase boundary, mechanically.

    YAKE's n=3 window cut 4+-word terms mid-phrase and the fragment became a
    note title verbatim ("stimatore a massima" [verosimiglianza] — 29 of 181
    notes in the 2026-08-21 run carried one). The miner now spans four content
    words, so this mostly serves structural seeds and terms longer than that.
    Two deterministic repairs, no LLM:

    - completion: while EVERY occurrence of the phrase in the body is
      followed by the same next word (same line, nothing but spaces between)
      and that word is not a stopword, append it. Divergent continuations or
      a punctuation boundary stop the walk, so a singleton only absorbs the
      rest of its own noun phrase.
    - edge trim: drop leading/trailing stopwords ("ha bias" -> "bias") —
      the miner never emits a stopword edge, but the structural seed leg has
      no such screen.

    Candidates that complete onto an already-present phrase collapse onto
    the best-ranked survivor; a candidate the repairs empty out is dropped.
    """
    body_l = body.lower()
    out: list[ConceptCandidate] = []
    seen: set[str] = set()
    for c in ranked:
        # Markup remnants at the tail ("proprietà delle svm***", "variabile
        # discreta."): strip trailing non-word chars, keeping a ")" that
        # closes a "(" inside the phrase.
        ph = c.phrase
        while ph and re.match(r"[\W_]", ph[-1]) and not (ph[-1] == ")" and "(" in ph):
            ph = ph[:-1].rstrip()
        words = ph.split()
        # edge trim first: a stopword edge is never part of the term
        while words and words[0].lower() in stop:
            words.pop(0)
        while words and words[-1].lower() in stop:
            words.pop()
        if not words:
            continue
        grew = 0
        while len(words) < max_words:
            phrase_l = " ".join(words).lower()
            occ = re.compile(
                r"(?<![\w\u00C0-\u00FF])" + re.escape(phrase_l)
                + r"(?![\w\u00C0-\u00FF])")
            nexts: set[str | None] = set()
            n_occ = 0
            for m in occ.finditer(body_l):
                n_occ += 1
                nm = re.compile(r"[ \t]+(" + _PHRASE_WORD + ")").match(
                    body_l, m.end())
                if nm and nm.end() < len(body_l) and body_l[nm.end()] in ".@":
                    # Mid-identifier cut (emails, dotted names): appending
                    # would mint a new fragment ("Giosuè Lo Bosco giosue").
                    nm = None
                nexts.add(nm.group(1) if nm else None)
            # >= 2 occurrences: one occurrence is zero evidence of a stable
            # collocation, and a singleton walk absorbed the sentence's verb
            # ("Errore quadratico atteso dipende", measured on Lezione 7).
            if n_occ < 2 or len(nexts) != 1:
                break
            nxt = nexts.pop()
            if nxt is None or nxt in stop:
                break
            # Commit the word only if the longer phrase still qualifies:
            # recon's closed-class tail check (_dangles) knows function words
            # the language stop list may miss, and a candidate must never be
            # dropped for having grown badly.
            if not is_concept(normalize(" ".join(words + [nxt])), overlay=overlay):
                break
            words.append(nxt)
            grew += 1
        phrase = normalize(" ".join(words))
        if not phrase or phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        if phrase != c.phrase:
            if not is_concept(phrase, overlay=overlay):
                continue
            c = ConceptCandidate(
                phrase=phrase, score=c.score,
                evidence=c.evidence + ([f"snap:+{grew}"] if grew else ["trim"]),
                confidence=c.confidence)
        out.append(c)
    return out


def _cutoff(content: str, ranked: list[ConceptCandidate]) -> list[ConceptCandidate]:
    # Operator knobs, same idiom as POOL_SIZE. Read at call time (not import) so a
    # harness A/B arm can coarsen extraction in-process without an import-timing
    # race: lowering the cap keeps the top-ranked (most salient) concepts and
    # drops the trivial tail (peripheral single-mention atoms), so fewer notes are
    # nucleated. Defaults bit-identical to the module constants.
    max_c = int(os.getenv("SILICA_MAX_CONCEPTS", str(MAX_CONCEPTS)))
    per_tok = int(os.getenv("SILICA_TOKENS_PER_CONCEPT", str(TOKENS_PER_CONCEPT)))
    n_tok = len(content.split())
    k = max(MIN_CONCEPTS, min(max_c, n_tok // max(1, per_tok)))
    return ranked[:min(k, len(ranked))]


def extract_keyphrases(
    content: str,
    *,
    overlay: DomainOverlay | None = None,
    lang: str = "english",
    embedder=None,
) -> list[ConceptCandidate]:
    """Ranked concept candidates from *content*.

    The miner generates the candidate pool, seeded with markup concepts it can't
    reach (see `_seed_structural`); if an `embedder` is given it ranks the pool
    (cosine-to-theme + MMR + structural boost), otherwise the structural-first /
    mined rank is used (degraded fallback). Returns [] only when both legs
    abstain, which `silica_recon` already handles as an empty report.
    """
    # Transient: the note keeps its LaTeX/images/fences on disk. fences=True is
    # the C1 fork ⚑ — the miner must never rank code identifiers as concepts.
    body = clean_body(content, fences=True)
    lang = language.resolve(lang, body)  # "auto" -> concrete Snowball lang via language.detect
    if overlay is None:
        overlay = overlay_for_lang(lang)  # lang already resolved by language.resolve above
    pool = _seed_structural(body, overlay, _pool_leg(body, overlay, lang) or [])
    if not pool:
        return []
    ranked = _rerank(pool, body, overlay, embedder)
    if ranked is None:
        ranked = pool  # fallback: structural-first, then mined rank
    # Boundary snap + edge trim before the confidence stamp, so a completed
    # phrase that now equals a heading earns its EXTRACTED tier.
    ranked = _complete_phrases(
        body, ranked, language.stopwords_for(lang) | set(overlay.stopwords),
        overlay)

    # Stamp the corroboration tier from the second (embedder-free) axis: a concept
    # present in author markup is corroborated → EXTRACTED; otherwise INFERRED.
    # One rule, both paths — survives the embedder-down fallback, where it is the
    # only gate left (salience needs the embedder). ponytail: recompute structural
    # here (cheap regex) rather than thread it through _rerank/_seed_structural.
    structural = _structural_concepts(body, overlay)
    for c in ranked:
        if c.phrase.lower() in structural:
            c.confidence = "EXTRACTED"
    return _cutoff(body, ranked)


if __name__ == "__main__":  # self-check, no framework
    txt = ("La discesa del gradiente ottimizza la funzione di perdita aggiornando "
           "i pesi della rete neurale a ogni iterazione del training. " * 3)
    from silica.kernel.text.overlay import DomainOverlay as _DO
    cands = extract_keyphrases(txt, overlay=_DO(stopwords=frozenset(), noise_patterns=()),
                               lang="italian")
    assert cands, "expected concepts from prose"
    assert len(cands) <= MAX_CONCEPTS  # lower bound not guaranteed: cutoff caps at available
    print(f"OK: {len(cands)} concepts; top={cands[0].phrase!r}")

    # _cutoff env knobs: deterministic, miner-independent (stub the ranked list).
    stub = [ConceptCandidate(phrase=f"c{i}", score=0.0, evidence=[]) for i in range(30)]
    long_txt = "w " * 600  # 600 tokens -> default k = min(40, 600//20) = 30
    assert len(_cutoff(long_txt, stub)) == 30, "default cutoff regressed"
    os.environ["SILICA_MAX_CONCEPTS"] = "8"
    assert len(_cutoff(long_txt, stub)) == 8, "SILICA_MAX_CONCEPTS not honored"
    del os.environ["SILICA_MAX_CONCEPTS"]
    os.environ["SILICA_TOKENS_PER_CONCEPT"] = "100"  # 600//100 = 6
    assert len(_cutoff(long_txt, stub)) == 6, "SILICA_TOKENS_PER_CONCEPT not honored"
    del os.environ["SILICA_TOKENS_PER_CONCEPT"]
    print("OK: cutoff knobs 30 -> 8 (max) -> 6 (per-tok)")
