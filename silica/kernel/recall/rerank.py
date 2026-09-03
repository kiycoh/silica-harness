# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Cross-encoder rerank pass over a fused candidate pool.

The relatedness facade fuses embeddings + co-occurrence by RANK (RRF); neither leg
ever reads the query and a candidate *together*. A cross-encoder does exactly that,
scoring query x document jointly — the strongest precision lever after first-stage
recall. This module applies that pass to an already-retrieved pool: it reorders,
never retrieves, and abstains (leaves the pool's order untouched) whenever the
reranker is absent or errors, mirroring a down leg in the facade.

The reranker CLIENT lives in agent/providers.py (Reranker/get_reranker); this module
holds only the note-aware reorder, so the client stays a plain HTTP provider.
"""
import re
import statistics
from typing import Any, Callable

_WINDOW_CHARS = 800  # cross-encoder document budget (chars): excerpt window + gate unit
# Frozen by the phase-0 run (2026-07-17, bench/phase0_gates.json): the spec's
# clean-gap expectation was REFUTED — real-vault notes are long (ratio p50 3.6,
# p90 10.8, max 20.8) and overlap LME chat sessions (9.2–21.7), so no corpus-level
# all-or-nothing threshold exists. 8 sits just under the measured-damage floor
# (LME min 9.2): every query where rerank damage was measured still fires, with
# margin, while ~81% of vault queries keep the reranker; the ~19% vault tail that
# fires holds >6.4k-char bodies, unreadable for the same reason chat sessions are.
_RERANK_WINDOW_FACTOR = 8
# Calibration hook: harnesses set it to capture {"median_len", "min_len",
# "window", "fired"} per query; production leaves it None. The factor above was
# frozen against the MEDIAN; the gate now votes on `min_len` (see rerank_related),
# which keeps every all-long pool firing — the LME chat sessions the factor was
# calibrated on are uniformly 9.2k+ chars, so their min fires too — while a pool
# holding even one window-sized note keeps its cross-encoder scores.
RERANK_GATE_PROBE: Callable[[dict], None] | None = None


def _query_terms(query: str) -> set[str]:
    return {t for t in re.findall(r"\w+", query.lower()) if len(t) > 3}


def window_weights(query: str) -> dict[str, float]:
    """Per-term BM25 idf from the vault's lexical index, for weighting the
    window-density scan. The unweighted scan counts 'what' and 'gradient'
    the same, so on long notes the window can centre on function-word-dense
    prose instead of the discriminative passage (offline-signals-map §3,
    graft G3). Reading the index is the only coupling to the lexical lane:
    no leg is fused, the use_lexical ARM stays an eval flag (ADR-0019).
    {} — the unweighted scan — when the vault has no lexical index, it is
    empty, or the store is unreadable: the lever must never make windowing
    worse than it was without an index."""
    terms = _query_terms(query)
    if not terms:
        return {}
    try:
        from silica.kernel.recall.lexical import get_lexical_store

        return get_lexical_store().query_idf(terms)
    except Exception:
        return {}  # tolerated: no index dir / driver down -> unweighted scan


def best_window_spans(text: str, query: str, width: int, n: int = 1,
                      weights: dict[str, float] | None = None,
                      ) -> list[tuple[int, str]]:
    """Up to `n` non-overlapping (offset, slice) windows of `text` densest in
    query terms, in document order (multi-window spec 2026-07-15; offsets
    added for the section chain, graft G1).

    A cross-encoder sees ~512 tokens (~2k chars); on a long note the naive
    head slice `text[:width]` can miss the passage the query is actually about
    entirely, so the reranker scores irrelevant opening text and demotes a true
    match (measured: on LongMemEval's multi-turn chat sessions the head slice
    evicts gold sessions whose relevant turn sits past char 800). Anchoring
    windows on query-term density fixes that with no extra model call; on
    9-21k-char chat bodies a single window still cuts gold spans (gic 0.533 on
    the raw arm), so perception can ask for several.

    `weights` (term -> idf, from `window_weights`) recentres density on
    discriminative terms; None/{} keeps the historical unweighted count
    bit-identical (w.get default 1.0, and float==int ties break the same).

    Greedy top-N with masking: hits per position never change (masking removes
    candidate positions, not text), so one density scan feeds every pick. The
    first window is always taken even at zero hits (n=1 stays bit-identical to
    the historical single-window behavior); each later window needs hits > 0 —
    never pad with irrelevant text, returning fewer than n windows is normal.
    Document order preserves chat chronology for temporal questions.
    """
    if len(text) <= n * width:
        return [(0, text)]
    terms = _query_terms(query)
    if not terms:
        return [(0, text[:width])]
    w = weights or {}
    low = text.lower()
    step = max(1, width // 4)
    candidates = [(pos, sum(low.count(t, pos, pos + width) * w.get(t, 1.0)
                            for t in terms))
                  for pos in range(0, max(1, len(text) - width) + step, step)]
    chosen: list[int] = []
    while candidates and len(chosen) < n:
        pos, hits = max(candidates, key=lambda c: c[1])  # earliest max, as before
        if chosen and hits == 0:
            break
        chosen.append(pos)
        candidates = [c for c in candidates
                      if c[0] + width <= pos or c[0] >= pos + width]
    return [(p, text[p:p + width]) for p in sorted(chosen)]


def best_windows(text: str, query: str, width: int, n: int = 1,
                 weights: dict[str, float] | None = None) -> list[str]:
    """The slices of `best_window_spans`, for callers that never need the
    offsets (the rationale lives there)."""
    return [s for _p, s in best_window_spans(text, query, width, n, weights)]


def best_window(text: str, query: str, width: int,
                weights: dict[str, float] | None = None) -> str:
    """The single `width`-char slice of `text` densest in query terms
    (see `best_window_spans`; this is its n=1 case, bit-identical)."""
    return best_windows(text, query, width, 1, weights)[0]


_best_window = best_window  # transitional alias; drop once callers migrate


def _read_body(path: str, *, origin: str = "vault") -> tuple[str, str]:
    """(note name, full body text) for one note; ('', '') when unreadable —
    a length of 0 fails open toward reranking, and '' scores as irrelevant.
    origin='memory' (ADR-0019) resolves the path in the personal-memory vault
    and an absolute-path origin in that peeked vault — folders the active-vault
    driver cannot open — so rerank never buries a foreign lane."""
    if origin != "vault":
        from silica.kernel.recall.memory_lane import foreign_root

        root = foreign_root(origin)
        if root is None:
            return "", ""
        p = root / (path if path.endswith(".md") else path + ".md")
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "", ""
    else:
        try:
            from silica.driver import DRIVER

            content = DRIVER.read_note(path).content or ""
        except Exception:
            return "", ""
    from silica.kernel.write import frontmatter

    _data, _raw, body = frontmatter.split(content)
    name = path.rsplit("/", 1)[-1].removesuffix(".md")
    return name, (body or content)


# Obsidian's default capture names: a real word the informativeness check below
# would wave through. ponytail: en + it only — an unlisted locale's default name
# falls through to the measured body-window branch, never to a wrong answer.
_JUNK_TITLE = re.compile(r"^(?:untitled|new note|senza titolo)\b[\s\d]*$", re.IGNORECASE)


def link_query(path: str) -> str:
    """Cross-encoder query for the note-as-query call sites (orphan/link repair).

    The title alone when it carries at least one real word and is not a default
    capture name; else the head window of the body. Two branches, both measured
    on the masked-wikilink A/B (609 pairs, 795-note vault, mxbai-rerank-base-v2):

      * bare title: mrr +0.081 over fused, vs +0.055 for the old title+excerpt
        document (paired p=0.028) — and ANY body text in the query erased the
        difference, title+150 chars already scored as title+800, so there is no
        excerpt length to tune (bench/local_rerank_excerpt_sweep.json);
      * junk title -> bare body head window, +0.037: in the title-blind
        ablation no syntactic surrogate (first document heading, YAKE top-2)
        beat it, so the fallback is one branch, not a ladder
        (bench/local_rerank_title_blind.json).

    A date-like title ("2026-08-09") fails the one-real-word check with no date
    regex needed; "2026-08-09 riunione" passes as a title. Returns '' when the
    note is unreadable — rerank_related abstains on an empty query, preserving
    first-stage order.
    """
    name, text = _read_body(path)
    if re.search(r"[^\W\d_]{3,}", name) and not _JUNK_TITLE.match(name.strip()):
        return name
    # This branch is a FLOOR, not a fix, and read time cannot raise it: the two
    # measured numbers above say a good title roughly doubles what the reranker
    # recovers (+0.081 vs +0.037), and the title-blind ablation refuted every
    # cheap way of rebuilding one from the body — first heading tied the bare
    # body (86/92, p=0.71) and YAKE top-2 lost to it (72/107, p=0.011). Short
    # was never the reason the title won: heading is 24 chars like the title and
    # bought nothing. The title wins because it is the note's IDENTITY, which
    # the body simply does not carry in extractable form.
    # So the missing half is only recoverable at WRITE time, by giving the note
    # a real title instead of giving the query a surrogate: synthesis at
    # /promote (the capture is already in front of an LLM there, zero extra
    # calls) and a rename work item in curate beside the orphan repairs. That
    # pays on co-occurrence, the graph and wikilinks too, since build_contribution
    # indexes `name` into the note's own text — not on this one call.
    return text[:_WINDOW_CHARS].strip()


def _path_of(item: Any) -> str:
    if isinstance(item, dict):
        return item.get("path", "")
    return getattr(item, "path", "")


def _origin_of(item: Any) -> str:
    if isinstance(item, dict):
        return item.get("origin", "vault")
    return getattr(item, "origin", "vault")


def rerank_related(
    reranker: Any,
    query_text: str,
    results: list,
    *,
    k: int,
    document_of: Callable[[Any], str] | None = None,
    stats: dict | None = None,
) -> list:
    """Reorder the first-stage top-k of `results` by cross-encoder relevance.

    Reorder-only (retrieval-gates spec, 2a): the pool is truncated to k BEFORE
    scoring, so membership belongs to the first stage and recall@k is
    rerank-invariant by construction — every measured rerank damage was
    selection damage, the only measured gain is ordering. Re-tested 2026-08-25
    (graft G4, depth-50 pool, bench/g_depth.json vs bench/g_header.json): the
    cross-encoder promoted tail candidates over notes whose windows carried
    the gold text — gold_in_context 71 -> 67 (4 lost, 0 gained) on locomo,
    flat on LME18 — so the deeper pool was selection damage again and the
    rule stands on two measurements, not one.

    Granularity abstain (2b): when the median candidate body dwarfs the
    cross-encoder window, the model cannot read the evidence and its ordering
    is noise — skip the call, keep first-stage order.

    `stats` is an optional out-dict: `rerank_related` sets `stats["reranked"]`
    to whether the cross-encoder actually scored this pool. Callers that DISPLAY
    the score need it — on abstention the surviving numbers are first-stage
    fusion cosines (~0.03), on a real rerank they are relevance probabilities
    (~0.99), and nothing in the list itself distinguishes the two scales.

    Each result is any object/dict exposing a note path (`.path` or `["path"]`).
    Abstention — no reranker, empty query, gate fired, or the reranker erroring
    — falls back to the pool's existing order, so a disabled or down reranker
    is a pure no-op. `document_of(item) -> str` supplies each candidate's text
    (its lengths then feed gate 2b as-is); it defaults to reading the note.
    """
    pool = results[:k]
    if stats is not None:
        stats["reranked"] = False
    if reranker is None or not pool or not query_text:
        return pool
    if document_of is not None:
        docs = [document_of(it) for it in pool]
        lengths = [len(d) for d in docs]
    else:
        bodies = [_read_body(_path_of(it), origin=_origin_of(it)) for it in pool]
        lengths = [len(text) for _name, text in bodies]
        docs = None
    # Gate 2b is about whether the cross-encoder can READ the evidence, which is
    # a property of a candidate, not of a pool. On the median it was a majority
    # vote: a vault that indexes its verbatim lectures beside the notes distilled
    # from them puts two 20k-char sources in a top-5, and every short note beside
    # them lost its cross-encoder score to their length. Fire only when NO
    # candidate fits the window — then the ordering really is noise. One short
    # note in the pool is enough for the reranker to produce a real ordering
    # (measured on the ML vault: same query, gold at rank 5 scoring 0.025
    # un-reranked, rank 1 scoring 0.9999 once the gate stopped firing).
    min_len = min(lengths)
    median_len = statistics.median(lengths)
    fired = min_len > _RERANK_WINDOW_FACTOR * _WINDOW_CHARS
    if RERANK_GATE_PROBE:
        RERANK_GATE_PROBE({"median_len": median_len, "min_len": min_len,
                           "window": _WINDOW_CHARS, "fired": fired})
    if stats is not None:
        stats["gate_fired"] = fired
        stats["reranked"] = not fired
    if fired:
        return pool
    if docs is None:
        w = window_weights(query_text)
        docs = [f"{name}\n{best_window(text, query_text, _WINDOW_CHARS, w)}".strip()
                for name, text in bodies]
    scores = reranker.scores(query_text, docs)
    if scores is None or len(scores) != len(pool):
        if stats is not None:
            stats["reranked"] = False  # reranker down/short reply: cosines survive
        return pool
    order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
    out = [pool[i] for i in order]
    # The first-stage score is ordering-only (RelatedNote contract) — and after
    # a reorder it no longer even orders. Overwrite it with the cross-encoder
    # relevance so the number a caller displays always matches the ranking
    # (observed: /find printing 0.033, 0.032, 0.031, 0.031, 0.032).
    for item, i in zip(out, order):
        if isinstance(item, dict):
            if "score" in item:
                item["score"] = scores[i]
        elif hasattr(item, "score"):
            try:
                item.score = scores[i]
            except Exception:
                pass  # frozen carriers keep their first-stage score
    return out
