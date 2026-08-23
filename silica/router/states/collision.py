# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Injector COLLISION state: embedding-based dedup routing.

Handler bodies for InjectorFSM, extracted from orchestrator.py: each function
takes the FSM instance and mutates its context/state exactly as the former
method did. Patchable collaborators (DRIVER, CONFIG, tools, load_ops, time)
are resolved through the orchestrator module namespace (orch.X) so tests that
patch silica.router.orchestrator.* keep working.
"""
from __future__ import annotations

import logging
import os
import hashlib
from typing import Any, TYPE_CHECKING

from silica.router import orchestrator as orch

if TYPE_CHECKING:
    from silica.router.orchestrator import InjectorFSM

logger = logging.getLogger(__name__)


# Logical-negation particles (it+en). title_key drops these as stopwords, so a
# folded-key match silently collapses a concept onto its negation
# ("supervisionato" ⇄ "non supervisionato", "primitive" ⇄ "non-primitive").
# A high-cosine pair differing only by one of these must reach the ternary
# judge, never mechanically auto-patch. Lexical antonyms (forward/backward,
# top/bottom) are an open set — left to the judge, not enumerated here.
_NEGATION = frozenset({"non", "not", "no", "senza", "without", "né", "neither", "nor"})


def _differ_by_negation(a: str, b: str) -> bool:
    """True when the two names differ by a logical-negation token — the folded
    keys match but one side negates the other."""
    import re

    ta = set(re.findall(r"\w+", a.casefold()))
    tb = set(re.findall(r"\w+", b.casefold()))
    return bool((ta ^ tb) & _NEGATION)


def _names_agree(concept: str, note_name: str) -> bool:
    """Conservative lexical gate for the mechanical high-score auto-patch.

    A high cosine can be driven by a single shared word — e.g. the concept
    "MEMORY" against the note "RAM (Random Access Memory)" — which is a domain
    collision, not the same concept. COLLISION may bypass the distiller and patch
    directly ONLY when the names genuinely agree; otherwise the concept is demoted
    to normal distillation so the distiller can judge from the excerpts. A wrong
    demotion only costs an extra distillation pass; a wrong auto-patch pollutes the
    vault, so the gate is deliberately strict.

    Agreement holds when the names share the same title identity (title_key —
    casefold, suffix/punctuation fold, plural-fold; C3) or the concept equals
    the note's acronym — its parenthetical token (e.g. "(GPT)") or the head
    token before the first parenthesis. A folded-key match is REJECTED when the
    surfaces differ by a logical negation (title_key strips "non" as a stopword,
    so it would otherwise fuse a concept with its opposite).
    """
    import re
    from silica.kernel.text.title import title_key

    if not concept.strip() or not note_name.strip():
        return False
    key = title_key(concept)
    if key and key == title_key(note_name):
        return not _differ_by_negation(concept, note_name)
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    acronyms = set(re.findall(r"\(([^)]*)\)", note_name))   # parenthetical contents
    acronyms.add(note_name.split("(", 1)[0])                # head before any paren
    nc = norm(concept)
    return bool(nc) and nc in {norm(a) for a in acronyms if a.strip()}


def route_concept(
    score: float,
    *,
    names_agree: bool,
    is_hub: bool,
    tau_high: float,
    tau_low: float,
) -> str:
    """Pure COLLISION routing decision for one scored candidate — the single
    source of truth shared by the live FSM and the coherence eval harness.

    Returns:
      "patch" — mechanical merge into the existing note (fast path, no judge).
                Only when the names genuinely agree; the caller still confirms
                the node exists in the graph and falls back to "keep" otherwise.
      "defer" — hand to the ternary dedup judge (which reads both bodies). Covers
                the borderline band AND high-cosine pairs whose names DISAGREE: a
                surface-name duplicate ("SVDD" vs "Support Vector Data Description")
                and a domain collision ("MEMORY" vs "RAM (Random Access Memory)")
                are indistinguishable from names+cosine alone, so neither is
                decided mechanically — the judge decides. (This is fix #1: the old
                gate demoted high-cosine name-mismatches to a silent new note,
                starving the judge of exactly the pairs it exists to resolve.)
      "keep"  — below τ_low: not close enough to route; normal distillation.
    """
    tau_eff = tau_high - (0.08 if is_hub else 0.0)
    if score >= tau_eff:
        return "patch" if names_agree else "defer"
    if score > tau_low:
        return "defer"
    return "keep"


def _deferred_op_dict(fsm: "InjectorFSM", d: dict, reason_prefix: str) -> dict:
    """Full, re-materializable op for a borderline concept (never a stub).

    The bundle in the deferred store is the only durable copy of the concept:
    it must carry the excerpt (snippet) and a real write path, so a retry — or
    the dedup verdict routing — can act on it without re-nucleating the source.
    """
    from silica.kernel.write.templates import slugify

    concept = d["concept"]
    name = concept.get("name", "") if isinstance(concept, dict) else str(concept)
    excerpt = concept.get("inbox_excerpt", "") if isinstance(concept, dict) else ""
    return {
        "op": "write",
        "heading": name,
        "source_basename": os.path.basename(d["inbox_file"]),
        "path": f"{fsm.target_dir}/{slugify(name) or name}.md",
        "snippet": excerpt,
        "hub": fsm.hub,
        "reason": f"{reason_prefix} score={d['score']:.3f} candidate={d['top_match'].get('name', '?')}",
    }


def _embedder_free_near_dups(
    chunk: dict, corpus: dict[str, str], *, threshold: float = 0.6, sig_cache: dict | None = None
) -> list[dict]:
    """Concepts in `chunk` that have a MinHash near-duplicate in `corpus`.

    Pure: no FSM, no DRIVER. `corpus` maps note path → body text. Returns dicts
    shaped like the borderline `deferred_concepts` entries so the existing defer
    plumbing handles them unchanged. The returned `concept` is the same object
    held in the chunk, so callers can drop it by identity.

    `sig_cache` (optional) memoizes corpus signatures across chunks in a run, so
    the vault's notes are hashed once instead of once per chunk.
    """
    from silica.kernel.report.minhash_dedup import near_duplicates

    out: list[dict] = []
    for batch in chunk.get("batches", []):
        inbox_file = batch.get("inbox_file", "")
        for concept in batch.get("concepts", []):
            if isinstance(concept, dict):
                name, excerpt = concept.get("name", ""), concept.get("inbox_excerpt", "")
            else:
                name, excerpt = str(concept), ""
            query = f"{name}\n{excerpt}".strip()
            if not query:
                continue
            hits = near_duplicates(query, corpus, threshold=threshold, sig_cache=sig_cache)
            if not hits:
                continue
            path, score = hits[0]
            note_name = os.path.splitext(os.path.basename(path))[0]
            out.append({
                "concept": concept,
                "inbox_file": inbox_file,
                "top_match": {"path": path, "name": note_name, "score": score},
                "score": score,
            })
    return out


def _collapse_near_dup_concepts(chunk: dict, *, is_near_dup) -> dict:
    """Drop near-duplicate SIBLING concepts within a chunk, keeping one per group.

    COLLISION routes each concept only against the committed vault, so two twins
    distilled from the same chunk (same source passage) are both absent from the
    index and both get written — the intra-chunk blind spot (#4). A union-find over
    the chunk's own concepts keeps the richest member of each near-dup group (longest
    excerpt) and drops the rest.

    `is_near_dup((name_a, excerpt_a), (name_b, excerpt_b)) -> bool` is injected so the
    caller supplies the signal it has cheapest: the main path passes embedding cosine
    (vectors already computed for vault routing); the cold path passes title-identity
    + MinHash (no embedder). Pure: no FSM, no I/O. Drop-only (no excerpt merge) so the
    surviving concept's embed text is unchanged and its vault-routing vector stays
    valid. O(n²) over a chunk's concepts.
    ponytail: drops the twin's delta; it is assumed subsumed by the longer sibling.
    """
    batches = chunk.get("batches", [])
    entries: list[list] = []  # [batch_idx, concept, name, excerpt]
    for bi, batch in enumerate(batches):
        for c in batch.get("concepts", []):
            name = c.get("name", "") if isinstance(c, dict) else str(c)
            excerpt = c.get("inbox_excerpt", "") if isinstance(c, dict) else ""
            entries.append([bi, c, name, excerpt])

    n = len(entries)
    if n < 2:
        return chunk

    parent = list(range(n))
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if is_near_dup((entries[i][2], entries[i][3]), (entries[j][2], entries[j][3])):
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    if all(len(g) == 1 for g in groups.values()):
        return chunk

    keep: set[int] = set()
    for members in groups.values():
        keep.add(max(members, key=lambda k: len(entries[k][3])))  # richest excerpt survives

    new_batches: list[dict] = []
    for bi, batch in enumerate(batches):
        kept = [entries[i][1] for i in range(n) if entries[i][0] == bi and i in keep]
        if kept:
            new_batches.append({"inbox_file": batch.get("inbox_file", ""), "concepts": kept})
    return {"schema_version": chunk.get("schema_version", 1), "batches": new_batches}


def _cold_intra_chunk_near_dup(a: tuple, b: tuple) -> bool:
    """Embedder-free intra-chunk near-dup predicate for the STABLE leg.

    Conservative on purpose — a wrong drop here is unrecoverable (the concept is
    never written), so this only fires on high-precision signals:
      • same title identity (title_key) — a title variant of the same note
        ("Neurone Artificiale" ≡ "Neurone Artificiale (ANN)" ≡ "… 1"); or
      • near-verbatim bodies (MinHash char-shingle Jaccard ≥ threshold) — catches
        body twins whose titles differ ("… Description" vs "… Descriptor").
    Paraphrased twins slip through (MinHash at this threshold is high-precision,
    low-recall); the main embedding path catches those when the embedder is up.
    """
    from silica.kernel.text.title import title_key
    from silica.kernel.report.minhash_dedup import minhash_signature, estimate_jaccard

    (na, ea), (nb, eb) = a, b
    ka, kb = title_key(na), title_key(nb)
    # Require a NON-DEGENERATE shared key: title_key can collapse to a single common
    # word when the other token is a stopword ("Test Concept" & "Another Concept"
    # both → "concept"), which is not the same note. Two+ shared stems is a real
    # title-variant match ("Neurone Artificiale" ≡ "Neurone Artificiale (ANN)").
    if ka and ka == kb and len(ka.split()) >= 2:
        return True
    # MinHash is a BODY signal — comparing near-empty/short excerpts (or the titles)
    # lets a single shared word inflate the char-shingle Jaccard into a false drop.
    # Only compare excerpts, and only when both carry enough text to be meaningful.
    if len(ea) < 40 or len(eb) < 40:
        return False
    thr = getattr(orch.CONFIG, "minhash_dup_threshold", 0.6)
    sa, sb = minhash_signature(ea), minhash_signature(eb)
    return bool(sa and sb) and estimate_jaccard(sa, sb) >= thr


def _run_embedder_free_dedup_leg(fsm: "InjectorFSM", idx: int, chunk: dict) -> None:
    """STABLE dedup leg: MinHash near-dup pass for when the embedder is unavailable.

    Near-dups are DEFERRED for review and dropped from the chunk so DELEGATE does
    not write a duplicate note. They are never mechanically patched — the
    embedder-free signal is weaker than a cosine, so it routes to escalate-tier,
    not an auto-write. Best-effort: any failure leaves the chunk untouched.
    """
    # Fix #4 (cold path): collapse intra-chunk sibling near-dups before the vault
    # check, so twins from the SAME chunk don't both get written. Embedder-free,
    # conservative predicate (title identity + near-verbatim MinHash).
    _before = sum(len(b.get("concepts", [])) for b in chunk.get("batches", []))
    chunk = fsm._chunks[idx] = _collapse_near_dup_concepts(
        chunk, is_near_dup=_cold_intra_chunk_near_dup
    )
    if sum(len(b.get("concepts", [])) for b in chunk.get("batches", [])) < _before:
        logger.info(
            "COLLISION(stable leg): collapsed %d intra-chunk sibling near-dup concept(s)",
            _before - sum(len(b.get("concepts", [])) for b in chunk.get("batches", [])),
        )

    threshold = getattr(orch.CONFIG, "minhash_dup_threshold", 0.6)
    try:
        corpus: dict[str, str] = {}
        # "" matches every note name → full-vault enumeration. Acceptable on this
        # cold path (only reached when the embedder/index is down). Re-read per
        # chunk on purpose: chunk N must see notes chunk N-1 just wrote, or
        # cross-chunk twins land as duplicates. Only the signatures are memoized
        # (sig_cache below) — the corpus text is cheap I/O, the MinHash is not.
        for ref in orch.DRIVER.search_names(""):
            try:
                corpus[ref.path] = orch.DRIVER.read_note(ref).content or ""
            except Exception:
                continue
    except Exception as _e:
        logger.debug("COLLISION minhash leg: corpus build failed (%s) — skipping", _e)
        return
    if not corpus:
        return

    sig_cache = fsm.__dict__.setdefault("_minhash_sig_cache", {})
    deferred = _embedder_free_near_dups(chunk, corpus, threshold=threshold, sig_cache=sig_cache)
    if not deferred:
        return

    dup_ids = {id(d["concept"]) for d in deferred}
    new_batches: list[dict] = []
    for batch in chunk.get("batches", []):
        kept = [c for c in batch.get("concepts", []) if id(c) not in dup_ids]
        if kept:
            new_batches.append(
                {"inbox_file": batch.get("inbox_file", fsm._current_source_file), "concepts": kept}
            )
    fsm._chunks[idx] = {"schema_version": chunk.get("schema_version", 1), "batches": new_batches}

    deferred_op_dicts = [_deferred_op_dict(fsm, d, "minhash_near_dup") for d in deferred]
    fsm._defer_ops(
        deferred_op_dicts,
        {
            (d["concept"].get("name", str(i)) if isinstance(d["concept"], dict) else str(i)):
            f"minhash_near_dup score={d['score']:.3f}"
            for i, d in enumerate(deferred)
        },
        phase="COLLISION",
    )
    logger.info(
        "COLLISION minhash leg: deferred %d near-duplicate concept(s) (embedder-free)",
        len(deferred),
    )


def handle_collision(fsm: "InjectorFSM") -> None:
    """COLLISION state entry: run the pass unless a prefetch already did."""
    idx = fsm._current_chunk_idx
    if fsm.context.pop(f"chunk_{idx}_collision_done", False):
        fsm._progress_note(fsm._chunk_task_id("collision", idx), "collision", "done",
                           output_ref="prefetched")
        fsm._transition_success()
        return
    collision_pass(fsm, idx)
    fsm._transition_success()


def collision_pass(fsm: "InjectorFSM", idx: int) -> None:
    """Dedup/collision routing — Phase 5.

    The candidate is the cosine-best vault note, from the embedding index
    alone. Until ADR-0030 it was the winner of the relatedness facade (RRF of
    embeddings + co-occurrence), on the idea that a note the author co-mentions
    heavily should outrank a merely cosine-close one; the thresholds below are
    cosine thresholds, and measured on 60 duplicates (2026-08-23) the
    cosine-best note WAS the duplicate 98% of the time against 87% for the
    fused winner, while a duplicate handed to the judge under any other
    candidate is lost whatever the judge says. The co-occurrence pull is an
    associative signal; sameness is a similarity question.

    For each concept in the current chunk (see route_concept for the decision):
    - score ≥ τ_high & names agree → pre-route as a 'patch' op (graph check)
    - score ≥ τ_high & names disagree → defer to the dedup judge (fix #1: a
                        surface-name duplicate vs a domain collision can't be
                        told apart mechanically, so the judge decides)
    - τ_low < score < τ_high → defer (borderline, ambiguous)
    - score ≤ τ_low   → keep for normal distillation (new write)

    Best-effort: any failure (missing index, embedder down) silently skips
    the check and lets the chunk flow to DELEGATE unchanged.

    Re-entrant: takes an explicit chunk index so the distill prefetcher can run
    it early for lookahead chunks; performs no state transition.
    """
    fsm._progress_note(fsm._chunk_task_id("collision", idx), "collision", "running")

    τ_high = getattr(orch.CONFIG, "sim_threshold_high", 0.85)
    τ_low = getattr(orch.CONFIG, "sim_threshold_low", 0.75)

    from silica.agent.providers import get_embedder_or_none
    from silica.kernel.recall.embed import get_store

    def _minhash_fallback() -> None:
        fsm._get_chunks_from_context_if_empty()
        _run_embedder_free_dedup_leg(fsm, idx, fsm._chunks[idx])
        fsm._progress_note(fsm._chunk_task_id("collision", idx), "collision", "done")

    try:
        store = get_store()
    except Exception as _e:
        logger.warning("COLLISION: embed store unavailable (%s) — falling back to MinHash dedup leg", _e)
        _minhash_fallback()
        return
    if len(store) == 0:
        logger.info("COLLISION: embedding index empty — falling back to MinHash dedup leg")
        _minhash_fallback()
        return
    embedder = get_embedder_or_none(orch.CONFIG, "COLLISION")
    if embedder is None:
        _minhash_fallback()
        return

    fsm._get_chunks_from_context_if_empty()
    chunk = fsm._chunks[idx]

    pre_routed_ops: list[dict] = []
    deferred_concepts: list[dict] = []
    modified_batches: list[dict] = []

    # Embed every concept in the chunk in a SINGLE call (one network
    # round-trip per chunk instead of one per concept).  Falls back to
    # per-concept embedding only if the embedder returns a ragged response,
    # so a short/odd reply can never silently drop concepts.
    #
    # The query text is name+excerpt, built with the SAME _note_text used to
    # index notes, so the query vector is comparable to the stored title+body
    # vectors. Embedding the bare name lets short acronyms ("MEM", "ACE") score
    # spuriously high against unrelated short-acronym notes ("RAM (Random Access
    # Memory)", "MACE"); the excerpt anchors the concept in its real neighbourhood.
    from silica.kernel.recall.embed import _note_text

    def _concept_embed_text(concept: Any) -> str:
        if isinstance(concept, dict):
            return _note_text(concept.get("name", ""), concept.get("inbox_excerpt", ""))
        return _note_text(str(concept), "")

    all_texts: list[str] = []
    for batch in chunk.get("batches", []):
        for concept in batch.get("concepts", []):
            et = _concept_embed_text(concept)
            if et.strip():
                all_texts.append(et)

    vec_by_text: dict[str, Any] = {}
    uniq_texts = list(dict.fromkeys(all_texts))
    if uniq_texts:
        try:
            embedded = embedder.embed(uniq_texts)
            batched_ok = len(embedded) == len(uniq_texts)
        except Exception as _embed_err:
            logger.debug("COLLISION: batch embed failed (%s) — keeping concepts unrouted", _embed_err)
            embedded, batched_ok = [], False
        if batched_ok:
            vec_by_text = dict(zip(uniq_texts, embedded))
        else:
            for _t in uniq_texts:
                try:
                    _ev = embedder.embed([_t])
                    vec_by_text[_t] = _ev[0]
                except Exception as _embed_err:
                    logger.debug("COLLISION: embed failed for '%s': %s", _t, _embed_err)

    # Fix #4: collapse intra-chunk sibling near-dups before routing. COLLISION only
    # compares against committed notes, so twins distilled from the SAME chunk would
    # both be written. Reuse the embeddings just computed (cosine ≥ τ_high) — no extra
    # calls. Drop-only, so survivors keep the vectors already in vec_by_text.
    # (Cold/empty-store path has no vectors here; a MinHash predicate could feed
    # _collapse_near_dup_concepts there — left as a known small gap.)
    import numpy as _np

    def _emb_near_dup(a, b) -> bool:
        va = vec_by_text.get(_note_text(a[0], a[1]))
        vb = vec_by_text.get(_note_text(b[0], b[1]))
        if va is None or vb is None:
            return False
        va, vb = _np.asarray(va, dtype=float), _np.asarray(vb, dtype=float)
        denom = (va @ va) ** 0.5 * (vb @ vb) ** 0.5
        return bool(denom) and float(va @ vb) / denom >= τ_high

    _before = sum(len(b.get("concepts", [])) for b in chunk.get("batches", []))
    chunk = fsm._chunks[idx] = _collapse_near_dup_concepts(chunk, is_near_dup=_emb_near_dup)
    _after = sum(len(b.get("concepts", [])) for b in chunk.get("batches", []))
    if _after < _before:
        logger.info("COLLISION: collapsed %d intra-chunk sibling near-dup concept(s)", _before - _after)

    for batch in chunk.get("batches", []):
        inbox_file = batch.get("inbox_file", fsm._current_source_file)
        kept: list = []

        for concept in batch.get("concepts", []):
            concept_text = concept.get("name", "") if isinstance(concept, dict) else str(concept)
            if not concept_text:
                kept.append(concept)
                continue

            vec = vec_by_text.get(_concept_embed_text(concept))
            if vec is None:
                # Embedding unavailable for this concept (batch failed or
                # missing) — keep it for normal distillation.
                kept.append(concept)
                continue
            try:
                hits = store.cosine_top_k(vec, k=5)  # a few extra so the inbox filter below still leaves a candidate
            except Exception as _search_err:
                logger.debug("COLLISION: cosine lookup failed for '%s': %s", concept_text, _search_err)
                kept.append(concept)
                continue

            # Inbox notes index like any vault note but are staging, not merge
            # targets — validate rejects every Inbox path, so surfacing one as
            # the collision candidate guarantees a rejected op + steer churn.
            from silica.kernel.recall.paths import is_inbox_path
            hits = [h for h in hits if not is_inbox_path(h["path"])]

            if not hits:
                kept.append(concept)
                continue

            best = hits[0]
            top = {"path": best["path"], "name": best["name"], "score": float(best["score"])}
            score: float = top["score"]
            existing_path = top["path"]

            # Lower effective threshold for cluster hubs: merging into an
            # anchor note is safer than creating a competing shadow note.
            _vault_ctx = fsm.context.get("vault_graph_ctx", {})
            _match_key = existing_path.removesuffix(".md")
            _is_hub = _vault_ctx.get(_match_key, {}).get("is_hub", False)
            τ_eff = τ_high - (0.08 if _is_hub else 0.0)

            names_agree = _names_agree(concept_text, str(top["name"]))
            decision = route_concept(
                score, names_agree=names_agree,
                is_hub=_is_hub, tau_high=τ_high, tau_low=τ_low,
            )

            if decision == "patch":
                try:
                    orch.DRIVER.read_note(existing_path)
                    # Graph confirms node exists — safe to patch
                    logger.info(
                        "COLLISION: '%s' → patch '%s' (score=%.3f ≥ τ_eff=%.2f%s)",
                        concept_text, existing_path, score, τ_eff,
                        " [hub]" if _is_hub else "",
                    )
                    pre_routed_ops.append({
                        "op": "patch",
                        "path": existing_path,
                        "heading": concept_text,
                        "source_basename": os.path.basename(inbox_file),
                        "snippet": concept.get("inbox_excerpt", "") if isinstance(concept, dict) else "",
                        "hub": fsm.hub,
                        "reason": f"collision_routed score={score:.3f}{' [hub]' if _is_hub else ''}",
                    })
                except Exception:
                    # Node not in graph — treat as new write
                    logger.debug(
                        "COLLISION: '%s' high score but '%s' not in graph → keep as write",
                        concept_text, existing_path,
                    )
                    kept.append(concept)

            elif decision == "defer":
                # Borderline band OR high-cosine with disagreeing names (fix #1):
                # the ternary judge reads both bodies and decides duplicate /
                # distinct / contradicts — never a mechanical patch, never a
                # silent new note.
                logger.info(
                    "COLLISION: '%s' ~ '%s' → dedup judge (score=%.3f, names %s)",
                    concept_text, existing_path, score,
                    "agree" if names_agree else "disagree",
                )
                deferred_concepts.append({
                    "concept": concept,
                    "inbox_file": inbox_file,
                    "top_match": top,
                    "score": score,
                })

            else:  # "keep" — below τ_low, normal distillation
                kept.append(concept)

        if kept:
            modified_batches.append({"inbox_file": inbox_file, "concepts": kept})

    # Persist borderline concepts in the deferred store
    if deferred_concepts:
        deferred_op_dicts = [
            _deferred_op_dict(fsm, d, "collision_deferred") for d in deferred_concepts
        ]
        fsm._defer_ops(
            deferred_op_dicts,
            {
                (d["concept"].get("name", str(i)) if isinstance(d["concept"], dict) else str(i)):
                f"borderline_similarity score={d['score']:.3f}"
                for i, d in enumerate(deferred_concepts)
            },
            phase="COLLISION",
        )

    # Producer: hand each borderline pair to the leashed dedup sub-agent so it
    # can run concurrently while the Injector keeps writing its other batches.
    # The candidate match is a pre-existing (committed) vault note, so the
    # sub-agent's append-only patch never races the Injector's new-note writes;
    # the per-path lease covers the rare same-note overlap.
    if deferred_concepts and fsm.work_queue is not None:
        from silica.kernel.workqueue import WorkItem, batch_dedup_items
        dedup_items = []
        for d in deferred_concepts:
            concept = d["concept"]
            match = d.get("top_match", {})
            candidate_path = match.get("path", "")
            if not candidate_path:
                continue
            name = concept.get("name", "") if isinstance(concept, dict) else str(concept)
            excerpt = concept.get("inbox_excerpt", "") if isinstance(concept, dict) else ""
            dedup_items.append(WorkItem(
                kind="dedup",
                target_path=candidate_path,
                context={
                    "concept": name,
                    "excerpt": excerpt,
                    "candidate": match.get("name", candidate_path),
                    "score": d.get("score"),
                    "inbox_file": d.get("inbox_file", fsm._current_source_file),
                    "hub": fsm.hub,
                    # C2 verdict routing: lets the dedup capability clean up
                    # (or author the distinct spoke from) the twin bundle.
                    "content_hash": fsm._current_content_hash,
                    "target_dir": fsm.target_dir,
                },
                reason=f"borderline_similarity score={d.get('score', 0):.3f}",
            ))
        # Concepts hitting the same candidate note become ONE judge call.
        for wi in batch_dedup_items(dedup_items):
            try:
                fsm.work_queue.enqueue(wi)
            except Exception as _qe:
                logger.debug("COLLISION: failed to enqueue dedup item: %s", _qe)

    # Store pre-routed ops for merging in VALIDATE (Phase 5)
    fsm.context[f"chunk_{idx}_collision_ops"] = pre_routed_ops

    # Capture the idempotency hash BEFORE mutating the chunk.
    # COLLISION re-routes concepts based on what is currently in the vault,
    # which changes between a partial run and its resume (done chunks have
    # already written their notes).  Hashing the pre-COLLISION chunk means
    # the key is stable across runs with the same source input.
    import json as _json
    fsm.context[f"chunk_{idx}_input_hash"] = hashlib.sha256(
        _json.dumps(chunk, sort_keys=True).encode()
    ).hexdigest()

    # Replace chunk with filtered version (remove patched/deferred concepts)
    fsm._chunks[idx] = {
        "schema_version": chunk.get("schema_version", 1),
        "batches": modified_batches,
    }

    fsm._progress_note(
        fsm._chunk_task_id("collision", idx), "collision", "done",
        output_ref=f"{len(pre_routed_ops)} patch-routed, {len(deferred_concepts)} deferred",
    )
