# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Answer-time perception — the one assembly of recalled memory into context.

Validated on the LongMemEval perception grid (frozen corpus A, 2026-07-14):
facts-first episodic block + per-note query-densest window + rank/evidence/date
headers. The LME harness consumes perceive() directly, so the eval and the
product cannot diverge on this seam — the measured number belongs to Silica.

Kernel rule: no ``datetime.now()`` here — ``now`` is supplied by the caller
(the tool layer passes today, the eval adapter passes the simulated question
date).

Failure behavior: the episodic lane is additive and best-effort (a broken
store never blocks answering); retrieval errors propagate — a silently empty
context would score as a memory miss with no signal.
"""
from __future__ import annotations

from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silica.kernel.recall.cooccurrence import CooccurStore

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Perception-grid winners as plain defaults (config promotion declined
# 2026-08-19; revisit only when a real vault needs different values).
DEFAULT_K = 15
# Window grid decided 2026-07-30 (bench/window_sweep_150.json + the paired A/B in
# bench/ab_win_*.metrics.json). 3x1000 beats the old 1x3000 on answer accuracy:
# 0.520 vs 0.427 over the same 150 LME questions and the SAME retrieved blocks
# (rerank carries its own _WINDOW_CHARS, so the render window cannot move
# ranking), McNemar exact p=0.0336, 26 questions won against 12 lost. Uniform
# NARROWING was the losing move — every 1xN cell below 3000 lost gold and cost
# 12-28pp on some question type; splitting the same budget across more windows
# is what wins. k stays 15: probe_recall_rank showed the rank tail carries gold.
WINDOW_CHARS = 1000
DEFAULT_WINDOWS = 3
FACTS_K = 10


@dataclass
class NoteBlock:
    """One recalled note, ready for the prompt."""
    path: str       # store-keyspace rel path (no .md)
    date: str       # frontmatter `date`, '' when absent
    evidence: str   # joined per-leg provenance ("embed:0.83 cooccur:w9"), '' in --stuff
    body: str       # full body, frontmatter stripped
    excerpt: str    # query-densest window of the body
    contested: str | None = None  # correction reason when flagged, else None
    section: str = ""  # heading chain above the first window ("A > B"), '' at top
    builds_on: str = ""  # rendered prereqs ("A, B"), set only by study order (G6)
    origin: str = "vault"  # "memory" = personal-memory lane (ADR-0019): the path
    #                        resolves in ANOTHER vault, so read_note denies it


@dataclass
class Perception:
    """perceive()'s result: render() is the prompt string, the rest is telemetry."""
    query: str
    orientation: str = ""  # G2: vault map block, filled only by orient=True
    facts_block: str = ""
    fact_hits: list = field(default_factory=list)    # episodic.FactHit
    fact_chains: list = field(default_factory=list)  # per-hit supersede chain (episodic.Fact)
    blocks: list[NoteBlock] = field(default_factory=list)

    def render(self, *, facts_first: bool = True, windowed: bool = True,
               stale: dict[str, str] | None = None,
               plain_headers: bool = False) -> str:
        """The context string. Defaults are the validated perception; the flags
        exist as A/B arms for the eval harness (legacy layouts).

        `stale` maps note_path (.md-suffixed, codedocs.peek's shape) to change
        level; a matching block's header gains a stale:<level> token, because
        the model answers from this string and a side map alone never reaches
        it."""
        parts: list[str] = []
        for rank, b in enumerate(self.blocks, 1):
            # block paths are store-keyspace (no .md); peek keys carry .md
            lvl = (stale.get(b.path) or stale.get(b.path + ".md")) if stale else None
            if windowed:
                # Path + section chain give the excerpt its place in the vault
                # (graft G1): a window that starts mid-section otherwise reaches
                # the model with no anchor at all — the header carried only rank
                # and score provenance. Chunk text is untouched (the query-side
                # variant of 2608.00824), so the embed index never rebuilds.
                if plain_headers:
                    # Pre-G1 header (rank + provenance only): the A/B legacy
                    # arm, same standing as flat_context/facts_last.
                    head = f"[#{rank}"
                else:
                    head = f"[#{rank} | {b.path}"
                    head += (f" | sec: {b.section}" if b.section else "")
                    head += (f" | builds-on: {b.builds_on}" if b.builds_on else "")
                head += (f" | {b.evidence}" if b.evidence else "")
                head += (f" | dated {b.date}" if b.date else "")
                head += (f" | contested: {b.contested}" if b.contested else "")
                head += (f" | stale:{lvl}" if lvl else "") + "]"
                parts.append(f"{head}\n{b.excerpt}")
            else:
                marks = ([f"dated {b.date}"] if b.date else []) \
                    + ([f"contested: {b.contested}"] if b.contested else []) \
                    + ([f"stale:{lvl}"] if lvl else [])
                head = f"[{' | '.join(marks)}]\n" if marks else ""
                parts.append(f"{head}{b.body}")
        ctx = "\n\n---\n\n".join(parts)
        if not self.facts_block or not ctx:
            out = self.facts_block or ctx
        else:
            out = (f"{self.facts_block}\n\n---\n\n{ctx}" if facts_first
                   else f"{ctx}\n\n---\n\n{self.facts_block}")
        # Orientation leads (G2): a map after the evidence reads as more
        # evidence; before it, it frames what the evidence is a sample OF.
        if self.orientation and out:
            out = f"{self.orientation}\n\n---\n\n{out}"
        return out


def _peek_dir(vault: str | None) -> str | None:
    """The resolved folder a `vault=` peek reads, or None for a plain call: no
    vault named, or the active vault named (then the active path, with its
    sweep and its singletons, is the right one)."""
    if not (vault or "").strip():
        return None
    from silica.config import CONFIG

    p = Path(vault).expanduser().resolve()
    active = (getattr(CONFIG, "vault_path", "") or "").strip()
    if active and Path(active).resolve() == p:
        return None
    return str(p)


def facade_retrieve(query: str, *, k: int, use_embedder: bool = True,
                    use_rerank: bool = True, use_recall_weights: bool = False,
                    use_lexical: bool = False, rerank_stats: dict | None = None,
                    use_memory: bool = True, vault: str | None = None):
    """Fused first-stage retrieval + cross-encoder rerank for a fresh text query.

    The single retrieval path shared by the chat tools
    (silica_semantic_search) and perceive() — and therefore by
    the eval adapter. Both lanes (active vault + personal memory, ADR-0019) are
    queried; a down leg abstains to the survivor.

    Returns ``(results, query_vec)``: results is the RelatedNote list ([] for
    no hits), or None when no leg is available at all (no query embedding AND
    no co-occurrence index in either lane). query_vec is surfaced for reuse —
    episodic fact recall scores against the same vector.

    ``use_recall_weights`` (phase 1 of `improve`, LoCoMo eval-only): when True,
    folds the vault's recall-outcome weights in as an extra fusion leg. False
    (the default) leaves the retrieval path byte-identical for every other
    caller.

    ``use_lexical`` (default off, opt-in like ``use_recall_weights``): when
    True, folds the hand-written BM25/fuzzy leg into fusion as an extra leg.
    Abstains when the lexical index is absent or empty.

    ``rerank_stats`` is an optional out-dict filled with ``{"reranked": bool}``
    (see ``rerank_related``): the returned ``.score`` is a cross-encoder
    relevance when True and a first-stage fusion cosine when False, and the two
    are an order of magnitude apart. A caller that shows the number, or
    thresholds on it, has to know which one it got.

    ``vault`` (peek): the path of ANOTHER adopted vault whose stores stand in
    for the active legs, read-only and read as they lie on disk. The memory
    lane still applies under ``use_memory``; results carry that folder as
    their ``origin`` so body readers open the right files. The active vault's
    singletons, sweep and CONFIG are never touched — this is the memory lane
    (ADR-0019) pointed where the caller says, not a vault switch.
    """
    from silica.agent.providers import get_embedder, get_reranker
    from silica.config import CONFIG
    from silica.kernel.recall.cooccurrence import get_cooccur_store
    from silica.kernel.recall.embed import get_store
    from silica.kernel.recall.memory_lane import memory_stores, memory_vault
    from silica.kernel.recall.relatedness import related_notes_for_query
    from silica.kernel.recall.rerank import rerank_related
    from silica.kernel.recall.sync import sweep

    peek = _peek_dir(vault)
    cooccur_store: CooccurStore | None
    if peek is None:
        # Out-of-band freshness: hand-edits (Obsidian, rm, git) land in the
        # indexes before this query reads them. Debounced, never raises.
        sweep()

        embed_store = get_store()
        try:
            loaded = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
            cooccur_store = loaded if len(loaded) else None  # empty store ⇒ abstain
        except Exception:
            cooccur_store = None
    else:
        # No sweep for a peek: the sweep reconciles the ACTIVE vault's indexes
        # with its disk and must not seed a foreign one from here. A cold
        # peeked index answers nothing; `vault_registry.coverage` says so.
        from silica.kernel.recall.memory_lane import stores_for

        embed_store, cooccur_store = stores_for(peek)
    # ADR-0032: lane scope is caller intent. False = "this vault only" — the
    # memory legs are never loaded, fusion is bit-identical to single-vault.
    mem_embed, mem_cooccur = memory_stores() if use_memory else (None, None)
    if peek is not None and memory_vault() == Path(peek):
        # Peeking AT the memory vault: it already fills the primary legs, and
        # the same store on both lanes would double every RRF term.
        mem_embed = mem_cooccur = None

    query_vec = None
    if use_embedder and ((embed_store is not None and len(embed_store) > 0)
                         or mem_embed is not None):
        try:
            query_vec = get_embedder(CONFIG).embed([query])[0]
        except Exception:
            query_vec = None  # embed leg abstains; co-occurrence may still carry

    if query_vec is None and cooccur_store is None and mem_cooccur is None:
        return None, None

    recall_rank = None
    if use_recall_weights:
        from silica.kernel.recall.recall_weights import ranking

        recall_rank = ranking()

    lexical_rank = None
    if use_lexical:
        if peek is None:
            from silica.kernel.recall.lexical import get_lexical_store

            lex = get_lexical_store()
        else:
            from silica.kernel.recall.vault_registry import lexical_for

            lex = lexical_for(Path(peek))
        lexical_rank = (lex.rank(query, k=k) if lex is not None else None) or None

    results = related_notes_for_query(
        query_vec=query_vec,
        query_text=query,
        embed_store=embed_store,
        cooccur_store=cooccur_store,
        memory_embed_store=mem_embed,
        memory_cooccur_store=mem_cooccur,
        k=k,
        recall_rank=recall_rank,
        lexical_rank=lexical_rank,
    ) or []
    if peek is not None:
        # Fusion marks the primary legs "vault", which every body reader takes
        # to mean the ACTIVE vault. Stamp the peeked folder before rerank reads
        # bodies: a wrong-vault read scores as irrelevant and buries the peek.
        for r in results:
            if r.origin == "vault":
                r.origin = peek
    reranker = get_reranker(CONFIG) if use_rerank else None
    if rerank_stats is not None:
        rerank_stats["reranked"] = False  # no reranker configured ⇒ cosines stand
    if reranker:
        # Default document path: gate 2b sees full body lengths, the scored
        # docs are query-densest windows, memory-lane bodies resolve by origin.
        results = rerank_related(reranker, query, results, k=k, stats=rerank_stats)
    return results, query_vec


def _read_dated_body(path: str, origin: str = "vault") -> tuple[str, str | None, str | None]:
    """(frontmatter date, contested reason, body) for one note; ('', None, None)
    when unreadable. `contested` is the note's flag reason (first `contradictions`
    entry) or None. origin='memory' resolves in the personal-memory vault
    (ADR-0019); an absolute-path origin resolves in that peeked vault."""
    if origin != "vault":
        from silica.kernel.recall.memory_lane import foreign_root

        root = foreign_root(origin)
        if root is None:
            return "", None, None
        p = root / (path if path.endswith(".md") else path + ".md")
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "", None, None
    else:
        from silica.driver import DRIVER

        try:
            content = DRIVER.read_note(
                path if path.endswith(".md") else path + ".md").content or ""
        except Exception:
            return "", None, None
    from silica.kernel.write import frontmatter

    data, _raw, body = frontmatter.split(content)
    # data is None for a body-only note (no frontmatter) or a YAML error —
    # product notes from the FSM write path can lack frontmatter entirely.
    data = data or {}
    date = str(data.get("date") or "").strip()
    contested = None
    if data.get("contested"):
        refs = data.get("contradictions") or []
        contested = str(refs[0]) if refs else "contested"
    # `body` is the frontmatter-stripped text; for a body-only note split()
    # already returns the whole content as body. The old `or content` fallback
    # leaked YAML frontmatter into context whenever the body was empty (A7).
    return date, contested, body


def _recall_facts(perception: Perception, query: str, query_vec, *, now: str,
                  facts_k: int, episodic_ttl_days: int | None,
                  use_embedder: bool) -> None:
    """Fill the Personal-memory side of `perception`. Best-effort: additive
    evidence must never block answering (mirror of capture_from_distill)."""
    try:
        from silica.kernel.recall.episodic import EpisodicStore, render as render_facts

        store = EpisodicStore()
        if not store.live_facts():
            return
        if query_vec is None and use_embedder:
            try:
                from silica.agent.providers import get_embedder
                from silica.config import CONFIG

                query_vec = get_embedder(CONFIG).embed([query])[0]
            except Exception:
                query_vec = None  # lexical fact recall
        hits = store.recall(query, query_vec, k=facts_k, now=now,
                            ttl_days=episodic_ttl_days)
        if not hits:
            return
        perception.fact_hits = hits
        perception.fact_chains = [store.chain(h.fact) for h in hits]
        perception.facts_block = "Personal memory:\n" + render_facts(hits, store=store)
    except Exception as e:
        logger.warning("perceive: episodic recall failed (context continues): %s", e)


def _maybe_assemble(blocks: list[NoteBlock], *, assemble: bool, query: str) -> list[NoteBlock]:
    """Gate: assemble=False returns blocks untouched (bit-identical default)."""
    if not assemble or not blocks:
        return blocks
    return _assemble_blocks(blocks, query)


def _driver_neighbors(path: str):
    """`assembly.Neighbors` for one note, read live from DRIVER + cooccurrence.

    Keyspace note: seeds and `body_of`/`by_path` live in the store keyspace
    (no ".md"); `NoteRef.path` (children via backlinks, related via links)
    carries ".md", so it is stripped here to match. `parent` is transcribed
    as the raw `parent note` prop value (a NAME, not necessarily a store
    path) and `edges` as raw cooccurrence-store keys — both may not resolve
    through `body_of`; see the caller's keyspace concerns.
    """
    from silica.driver import DRIVER
    from silica.kernel.recall import assembly
    from silica.kernel.recall.cooccurrence import cooccur_key, get_cooccur_store
    from silica.config import CONFIG

    parent = None
    try:
        raw = (DRIVER.props_of(path) or {}).get("parent note") or ""
        parent = str(raw).strip().strip("[]").strip() or None
    except Exception:
        parent = None
    try:
        related = [r.path.removesuffix(".md") for r in DRIVER.links(path)]
    except Exception:
        related = []
    children: list[str] = []
    try:
        for b in DRIVER.backlinks(path):
            bp = (DRIVER.props_of(b.path) or {}).get("parent note") or ""
            if str(bp).strip().strip("[]").strip().lower() == _name_of(path).lower():
                children.append(b.path.removesuffix(".md"))
    except Exception:
        children = []
    edges: list[str] = []
    try:
        store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
        row = store.note_edges_for(cooccur_key(path))
        edges = [p for p, _w in sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))]
    except Exception:
        edges = []
    return assembly.Neighbors(parent=parent, children=children,
                              related=related, edges=edges)


def _name_of(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".md")


def _assembly_body(path: str) -> str:
    _date, _contested, body = _read_dated_body(path)
    return body or ""


def _assemble_blocks(blocks: list[NoteBlock], query: str) -> list[NoteBlock]:
    from silica.kernel.recall import assembly

    by_path = {b.path: b for b in blocks}

    def _body(p: str) -> str:
        # Seeds already carry the correctly-fetched body (right origin, memory
        # or vault, per _read_dated_body) on NoteBlock.body — assemble() calls
        # body_of() for every unit including seeds, and a re-read here would
        # default to origin="vault" and silently drop memory-lane seed bodies.
        # Only genuine periphery paths (not in by_path) fall back to a fresh read.
        seed = by_path.get(p)
        return seed.body if seed is not None else _assembly_body(p)

    res = assembly.assemble(
        [b.path for b in blocks],
        neighbors_of=_driver_neighbors,
        body_of=_body,
    )
    out: list[NoteBlock] = []
    for ab in res.blocks:
        head = by_path.get(ab.members[0])
        # `contested` is a trust signal, not decoration: perceive() promises a
        # flagged note is never dropped, only marked, so the answer step can
        # distrust it. Folding lost it twice — off the head, and off every
        # periphery member whose text is squashed into this block with no
        # NoteBlock of its own. A periphery member is not a seed, so its flag
        # costs one frontmatter read; seeds carry it already.
        reasons: list[str] = []
        for member in ab.members:
            seed = by_path.get(member)
            reason = seed.contested if seed is not None else _read_dated_body(member)[1]
            if reason and reason not in reasons:
                reasons.append(reason)
        out.append(NoteBlock(
            path=ab.members[0],
            date=head.date if head else "",
            evidence=head.evidence if head else "",
            body=ab.text,
            excerpt=ab.text,   # assembled text is already budgeted
            contested="; ".join(reasons) or None,
        ))
    return out


def _study_order(blocks: list) -> list:
    """Prerequisite-first order + builds-on annotations over the RENDERED set
    (graft G6, study surfaces only).

    Membership is untouched: topology orders what the semantic legs already
    chose, so the crowding-out that killed every structural RRF leg (V1
    0/417, V3 -25pp) cannot occur here by construction. V2 RefD is the one
    PASSED directed signal (judge 26/33, p=0.0007) and by the surface rule a
    PASS may carry an imperative reading. Contested blocks keep their demoted
    tail position: distrust outranks didactic order.
    """
    from silica.kernel.report.learner import prerequisites_map

    try:
        prereqs = prerequisites_map() or {}
    except Exception:
        return blocks  # tolerated: no cooccur depth -> retrieval order stands
    present = {b.path for b in blocks}
    if not any(p in present for deps in prereqs.values() for p in deps):
        return blocks

    def topo(group: list) -> list:
        # Kahn over the induced sub-DAG, stable: ties keep retrieval order,
        # and any cycle survivor appends in retrieval order rather than
        # dropping (ordering may never change membership).
        paths = [b.path for b in group]
        need = {b.path: [p for p in prereqs.get(b.path, []) if p in paths]
                for b in group}
        out: list = []
        placed: set[str] = set()
        while len(out) < len(group):
            ready = [b for b in group if b.path not in placed
                     and all(p in placed for p in need[b.path])]
            if not ready:  # cycle: emit the rest as retrieved
                out.extend(b for b in group if b.path not in placed)
                break
            out.extend(ready)
            placed.update(b.path for b in ready)
        for b in out:
            if need[b.path]:
                b.builds_on = ", ".join(_name_of(p) for p in need[b.path])
        return out

    clean = [b for b in blocks if not b.contested]
    contested = [b for b in blocks if b.contested]
    return topo(clean) + topo(contested)


def _section_chain(body: str, offset: int, depth: int = 3) -> str:
    """Markdown heading chain still open at `offset` ("Training > Gradients").
    Deepest `depth` levels only — the nearest heading carries the most anchor
    per token, and a full chain on a deeply nested note is header bloat.
    '' above the first heading, so headingless notes cost zero tokens."""
    chain: list[tuple[int, str]] = []
    for m in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", body[:offset], re.MULTILINE):
        lvl = len(m.group(1))
        while chain and chain[-1][0] >= lvl:
            chain.pop()
        chain.append((lvl, m.group(2)))
    return " > ".join(t for _lvl, t in chain[-depth:])


def perceive(query: str, *, now: str, k: int = DEFAULT_K,
             window_chars: int = WINDOW_CHARS, windows: int = DEFAULT_WINDOWS,
             facts_k: int = FACTS_K,
             episodic_ttl_days: int | None = None, with_facts: bool = True,
             use_embedder: bool = True, use_rerank: bool = True,
             paths: list[str] | None = None,
             use_recall_weights: bool = False,
             assemble: bool = False,
             use_lexical: bool = False,
             study_order: bool = False,
             orient: bool = False,
             use_memory: bool = True,
             vault: str | None = None) -> Perception:
    """Retrieve + assemble the answer-time context for `query`.

    ``paths`` skips retrieval and assembles the given notes in order (the eval
    adapter's --stuff arm, or a caller that already holds a shortlist);
    unreadable paths are skipped and ranks stay dense. ``episodic_ttl_days``:
    None = CONFIG default, 0 = never expire. ``use_recall_weights`` (phase 1 of
    `improve`, eval-only, default off) is forwarded to `facade_retrieve`; it
    has no effect when ``paths`` is set, since that bypasses retrieval.
    ``assemble`` (default off) folds each seed's 1-hop neighbours into a
    squashed, breadcrumbed block; no effect when ``paths`` is set (that
    bypasses retrieval). ``study_order`` (default off, study surfaces /
    answer-side A/B): prerequisite-first block order + builds-on header
    tokens via V2 RefD — see `_study_order` for why ordering is the one
    safe topological surface. ``orient`` (default off, agent-mode A/B —
    offline-signals-map §4.1) prepends the session vault map so a one-shot
    caller can carry the orientation a REPL session gets at start; the
    honest prior from PEEK's content ablation is single digits, so it stays
    an arm until an agent-mode gate passes.
    ``use_lexical`` (default off) forwards to `facade_retrieve`'s lexical leg;
    no effect when ``paths`` is set. ``vault`` (default None) peeks at another
    adopted vault — see `facade_retrieve`; blocks then carry that folder as
    ``origin``, and assembly stays off because it walks the ACTIVE vault's
    link graph.
    """
    from silica.kernel.recall.rerank import best_window_spans, window_weights

    query_vec = None
    if paths is not None:
        hits = [(p, "", "vault") for p in paths]
    else:
        results, query_vec = facade_retrieve(
            query, k=k, use_embedder=use_embedder, use_rerank=use_rerank,
            use_recall_weights=use_recall_weights, use_lexical=use_lexical,
            use_memory=use_memory, vault=vault)
        hits = [(r.path, " ".join(r.evidence), getattr(r, "origin", "vault"))
                for r in (results or [])]

    # One idf map per query, shared by every note's window scan (graft G3);
    # {} — no lexical index — keeps the scan bit-identical to the unweighted one.
    wts = window_weights(query) if query else {}
    blocks: list[NoteBlock] = []
    for path, evidence, origin in hits:
        date, contested, body = _read_dated_body(path, origin)
        if body is None:
            continue
        spans = (best_window_spans(body, query, window_chars, windows, wts)
                 if query else [(0, body[:window_chars])])
        excerpt = "\n[…]\n".join(s for _p, s in spans)
        if not excerpt.strip():
            continue  # empty body renders as a bare "[#n | evidence]" header, zero content
        blocks.append(NoteBlock(path=path, date=date, evidence=evidence,
                                body=body, excerpt=excerpt, contested=contested,
                                section=_section_chain(body, spans[0][0]),
                                origin=origin))
    # Correction loop: contested notes are demoted behind clean ones (stable),
    # never dropped — the render marks them so the answer step can distrust them.
    blocks = [b for b in blocks if not b.contested] + [b for b in blocks if b.contested]
    if study_order:
        blocks = _study_order(blocks)

    if paths is None:
        blocks = _maybe_assemble(blocks, assemble=assemble and vault is None, query=query)

    perception = Perception(query=query, blocks=blocks)
    if orient:
        try:
            from silica.kernel.recall.vault_map import build_vault_map

            perception.orientation = build_vault_map() or ""
        except Exception as e:
            logger.warning("perceive: orientation skipped (%s)", e)
    # use_memory=False means "this vault only": the episodic store homes in the
    # memory vault with no abstain rule of its own (episodic.py), so the facts
    # block is the same foreign lane through a second door and goes dark with it.
    if with_facts and use_memory:
        _recall_facts(perception, query, query_vec, now=now, facts_k=facts_k,
                      episodic_ttl_days=episodic_ttl_days, use_embedder=use_embedder)
    return perception
