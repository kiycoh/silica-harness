# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Graph & relatedness tools — indexes, search, linking, and the vault audit.

Embedding and co-occurrence index refresh, semantic search, autolink/backlink
passes, the vis.js graph export, and the structural vault report.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.tools import tool
from silica.tools.atomic import EmptyArgs

logger = logging.getLogger(__name__)


from silica.kernel.recall.paths import in_folder as _in_folder  # canonical folder-scope predicate


class GraphExportArgs(BaseModel):
    output_path: str = Field(
        default="graph.html",
        description="Filesystem path for the output HTML file (e.g. 'graph.html' or '/tmp/vault_graph.html')",
    )
    folder: str = Field(
        default="",
        description="Vault-relative folder to restrict scope (empty = entire vault)",
    )
    title: str = Field(
        default="Vault Graph",
        description="Title shown in the visualization header",
    )
    knn_k: int = Field(
        default=6,
        description="semantic overlay: nearest neighbours per note (SIMILAR edge density)",
    )

@tool(GraphExportArgs, cls="composed")
def silica_graph_export(output_path: str = "graph.html", folder: str = "",
                        title: str = "Vault Graph", knn_k: int = 6) -> dict[str, Any]:
    """Self-contained interactive HTML graph of the vault: the wikilink graph
    (Louvain-clustered; ghost nodes mark unresolved links) plus a toggleable
    embedding k-NN overlay that places link-orphans next to their semantic
    neighbours. Opens in any browser. Visualization only — for an actionable
    structural audit use silica_vault_report.
    """
    from silica.ui.web.graph_view import export_graph

    # Best-effort: refresh the co-occurrence index so clusters get named labels
    # (incremental — skips already-indexed notes). Naming degrades to "Cluster N"
    # if this fails; the graph still renders. ponytail: full-vault refresh, scope
    # to changed notes only if it gets slow on big vaults.
    try:
        silica_cooccurrence_refresh(folder=folder)
    except Exception as exc:
        logger.warning("silica_graph_export: cooccurrence refresh skipped (%s)", exc)

    return export_graph(output_path=output_path, folder=folder, title=title, knn_k=knn_k)


class MindmapArgs(BaseModel):
    note_path: str = Field(description="Vault-relative path of the note to root the map on")
    force: bool = Field(default=False, description="Overwrite an existing maps/<stem>.canvas (defaults to no-clobber)")

@tool(MindmapArgs, cls="composed")
def silica_mindmap(note_path: str, force: bool = False) -> dict[str, Any]:
    """Radial mind-map rooted on one note, written to maps/<stem>.canvas
    (editable in Obsidian). Deterministic: BFS over wikilinks plus the latent
    relatedness leg, radial wedges by community. No-clobber: an existing map
    needs force=True. Flat whole-vault network: silica_graph_export.
    """
    from pathlib import Path

    from silica.config import CONFIG
    from silica.kernel.recall.mindmap import (
        build_mapview,
        gather_materials,
        mapview_to_canvas,
        note_resolver,
    )

    # Accept a path OR a title (the GUI input and casual CLI use give titles).
    root = note_resolver()(note_path)
    if root is None:
        return {"error": f"'{note_path}' not found in the vault graph."}

    stem = Path(root).stem
    vault = CONFIG.vault_path or "."
    out = Path(vault) / "maps" / f"{stem}.canvas"

    # No-clobber: exists + not force → refuse. Diffing the generated map
    # against a user-rearranged one is a feature for the day someone asks;
    # here we simply never clobber.
    if out.exists() and not force:
        return {"skipped": str(out), "reason": "exists", "hint": "re-run with force=True to regenerate"}

    materials = gather_materials(root, latent_k=CONFIG.mindmap_latent_k)
    mv = build_mapview(
        root, materials, max_nodes=CONFIG.mindmap_max_nodes, hops=CONFIG.mindmap_hops
    )
    if len(mv.nodes) <= 1:
        return {"error": f"'{root}' has no neighbours to map (isolated in the graph)."}

    import orjson
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(mapview_to_canvas(mv), option=orjson.OPT_INDENT_2))
    logger.info("silica_mindmap: wrote %s — %d nodes, %d edges", out, len(mv.nodes), len(mv.edges))
    return {"path": str(out), "nodes": len(mv.nodes), "edges": len(mv.edges)}


class AutolinkArgs(BaseModel):
    note_paths: list[str] | None = Field(default=None, description="List of vault-relative paths to autolink")
    note_path: str = Field(default="", description="Vault-relative path of the note to autolink (legacy single-file)")
    use_candidates: bool = Field(default=True, description="Use embedding candidates to focus autolinking (requires index)")

@tool(AutolinkArgs, cls="composed", collapse="eager")
def silica_autolink(note_paths: list[str] | None = None, note_path: str = "", use_candidates: bool = True) -> dict[str, Any]:
    """Scan the given notes for mentions of existing vault titles and wrap them
    as wikilinks. Skips frontmatter, code, math, headings, already-linked text;
    only links titles that exist (graph-safe).

    ONE call consumes the ENTIRE `note_paths` list; `notes_linked` below
    `notes_scanned` is normal — there is no rest, do NOT re-call to finish.
    Reverse direction (links TO new notes from older neighbours):
    silica_backlink. Vault-wide pass that finds its own candidates:
    silica_curate.
    """
    from silica.kernel.link.autolink import build_title_index

    paths = note_paths or []
    if note_path and note_path not in paths:
        paths.append(note_path)

    if not paths:
        return {"error": "No note paths provided."}

    try:
        all_refs = DRIVER.list_files()
    except Exception as e:
        return {"error": f"Failed to list vault files: {e}"}

    title_index = build_title_index(all_refs)
    
    store = None
    embedder = None
    cooccur_store = None
    if use_candidates:
        try:
            from silica.agent.providers import get_embedder
            from silica.config import CONFIG
            from silica.kernel.recall.embed import get_store
            store = get_store()
            if len(store) > 0:
                embedder = get_embedder(CONFIG)
        except Exception:
            pass
        # The co-occurrence leg is embedder-free: load it independently so
        # candidates survive (focused) even when the embedder is down.
        try:
            from silica.config import CONFIG
            from silica.kernel.recall.cooccurrence import get_cooccur_store
            cooccur_store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
            if len(cooccur_store) == 0:
                cooccur_store = None
        except Exception:
            cooccur_store = None

    total_added = 0
    linked = 0
    skipped: list[str] = []
    write_errors: list[str] = []

    for path in paths:
        try:
            nc = DRIVER.read_note(path)
        except Exception as e:
            skipped.append(f"{path}: unreadable ({type(e).__name__})")
            continue

        body = nc.content or ""
        if not body.strip():
            skipped.append(f"{path}: empty")
            continue

        candidates: list[str] | None = None
        if use_candidates and (cooccur_store is not None or (store is not None and embedder is not None)):
            query_vec = None
            if store is not None and embedder is not None:
                try:
                    query_vec = embedder.embed([body[:800]])[0]
                except Exception:
                    query_vec = None
            try:
                from silica.kernel.recall.relatedness import related_notes_for_query
                related = related_notes_for_query(
                    query_vec=query_vec,
                    query_text=body,
                    embed_store=store,
                    cooccur_store=cooccur_store,
                    k=20,
                )
                # Only narrow to candidates when the facade actually proposed
                # some; an empty list would suppress linking, so leave it None
                # to fall back to the full title_index scan.
                if related:
                    candidates = [r.name for r in related]
            except Exception:
                pass  # fall back to full title_index scan

        try:
            added = DRIVER.autolink_note(
                path,
                candidates=candidates if candidates is not None else title_index,
                title_index=title_index,  # already built above; else rebuilt per note
            )
            if added:
                total_added += len(added)
                linked += 1
        except Exception as e:
            write_errors.append(f"{path}: {e}")

    # `notes_scanned` is the anti-relaunch signal: the whole list is consumed in
    # one call, so a low `notes_linked` means "nothing left to link", not
    # "call me again with the rest".
    result: dict[str, Any] = {
        "notes_scanned": len(paths),
        "notes_linked": linked,
        "total_links_added": total_added,
    }
    if skipped:
        result["skipped"] = skipped
    if write_errors:
        result["write_errors"] = write_errors
    return result


class BacklinkArgs(BaseModel):
    new_titles: list[str] = Field(description="Titles of notes just created in this run")
    neighbourhood: list[str] = Field(description="Vault-relative paths of candidate notes to scan")

@tool(BacklinkArgs, cls="composed", collapse="eager")
def silica_backlink(new_titles: list[str], neighbourhood: list[str]) -> dict[str, Any]:
    """Inject wikilinks to newly-created notes into pre-existing neighbouring notes.

    For each note in `neighbourhood`, wraps mentions of any title in `new_titles`
    with a wikilink — the reverse of silica_autolink. Skips frontmatter, code,
    math, and already-linked spans. Returns {path: [titles_added]}.
    """
    from silica.kernel.link.autolink import backlink_pass, build_title_index

    try:
        all_refs = DRIVER.list_files()
    except Exception as e:
        return {"error": f"Failed to list vault files: {e}"}

    title_index = build_title_index(all_refs)
    added_map = backlink_pass(new_titles, title_index=title_index, neighbourhood=neighbourhood)
    total = sum(len(v) for v in added_map.values())
    return {"added": total, "notes_modified": len(added_map), "details": added_map}


def _stale_entry(stale_map: dict[str, str], r) -> dict:
    """{"stale": level} for a flagged result, {} otherwise. Memory-lane
    results are another vault (ADR-0019): never annotated, like cluster."""
    if r.origin == "memory" or not stale_map:
        return {}
    from silica.kernel.code import codedocs

    lvl = codedocs.peek_level(stale_map, r.path)
    return {"stale": lvl} if lvl else {}


def _peek_stale() -> dict[str, str]:
    """The stale peek map for the active vault; {} on any failure (spec §5)."""
    try:
        from silica.config import CONFIG
        from silica.kernel.code import codedocs

        return codedocs.peek(CONFIG.vault_path)
    except Exception:
        return {}


# First-stage pool for _facade_search: over-fetch so dropping staging entries
# still leaves k notes. 20 is the rerank pool the gate is calibrated on.
_NOTE_POOL = 20


def _facade_search(text: str, k: int, memory: bool = True) -> dict[str, Any]:
    """Fused embeddings + co-occurrence search for a fresh text, then reranked.

    Shared core of silica_semantic_search, now routed
    through perception.facade_retrieve — the same retrieval path perceive()
    and the memory eval use. Returns ``{"results": [{path, name, score}, ...]}``
    or ``{"error": ...}`` when no index is available at all. The two legs
    abstain independently: an empty embedding index (or an offline embedder)
    still serves co-occurrence results, and vice versa — mirroring how
    perception and the run substrate consume the facade. (COLLISION does not:
    it routes on the plain cosine search, ADR-0030.)
    """
    from silica.kernel.recall.perception import facade_retrieve

    from silica.kernel.recall.paths import is_inbox_path

    # Staging outranks the notes distilled from it: a raw lecture repeats every
    # term its notes split up, so it wins first-stage retrieval and then eats the
    # top-k. Measured on the ML vault: asking whether a concept had a note, the
    # inbox copy of the ASKING document answered its own question at score 1.0,
    # three times. The inbox stays indexed (staging is source material and
    # `silica_recall` still reaches it) — it just stops crowding the ranked list
    # this tool returns, so over-fetch, drop staging, then cut to k.
    from silica.agent.providers import get_reranker
    from silica.config import CONFIG
    from silica.kernel.recall.rerank import rerank_related

    # Filter BEFORE the cross-encoder, not after: reranking the wide pool and
    # then dropping staging pays for documents that get thrown away, and on a
    # local GPU a 20-document batch OOMs outright. This way the reranker still
    # sees exactly k candidates, and they are all notes.
    results, _query_vec = facade_retrieve(text, k=max(k, _NOTE_POOL), use_rerank=False,
                                          use_memory=memory)
    if results is None:
        return {"error": "No index available. Run silica_embed_refresh or silica_cooccurrence_refresh first."}
    # is_inbox_path answers for the ACTIVE vault's staging roots; a memory-lane
    # note whose vault happens to use the same folder name is not this vault's
    # staging, and dropping it here silently shrank the guest lane (ADR-0019).
    notes = [r for r in results
             if r.origin == "memory" or not is_inbox_path(r.path or "")]
    # Never answer empty because everything relevant was staged: if the vault has
    # nothing else to say, staging IS the answer.
    rr: dict = {"reranked": False}
    results = (notes or results)[:k]
    reranker = get_reranker(CONFIG)
    if reranker:
        results = rerank_related(reranker, text, results, k=k, stats=rr)
    stale_map = _peek_stale()
    return {
        # Which scale `score` is on. The cross-encoder abstains on pools it
        # cannot read, and the surviving first-stage cosines live around 0.03
        # where a rerank probability lives around 0.99 — same field, two scales.
        # Without this a caller reads 0.03 as "nothing here" when it means
        # "not scored". Never threshold across calls; compare within one.
        "reranked": bool(rr.get("reranked", False)),
        "results": [
            {
                "path": r.path,
                "name": r.name,
                "score": round(r.score, 4),
                # Origin marker (ADR-0019): only when the note is NOT in the
                # active vault, so single-vault payloads stay unchanged.
                **({"origin": "memory"} if r.origin == "memory" else {}),
                **_stale_entry(stale_map, r),
            }
            for r in results
        ]
    }


class SemanticSearchArgs(BaseModel):
    query: str = Field(description="Free-form query text to embed and search against the vault index")
    k: int = Field(default=5, description="Number of results to return")
    memory: bool = Field(default=True, description="Include the personal-memory lane (ADR-0019). Pass false for questions scoped to THIS vault — repo and code questions — so an unrelated personal vault cannot occupy result slots.")

@tool(SemanticSearchArgs, cls="composed")
def silica_semantic_search(query: str, k: int = 5, memory: bool = True) -> dict[str, Any]:
    """Find vault notes by MEANING (embeddings + co-occurrence fused, reranked).

    Use for "what do I have about X" when the exact wording is unknown;
    `query` can be a phrase or a whole paragraph. A leg that is down (empty
    index, embedder offline) degrades to the survivor. For literal text use
    silica_search_context; when the text IS an existing note, prefer
    silica_related. Returns at most k results, best first; verify with
    silica_read_note before acting.

    `score` is comparable only WITHIN one call: `reranked: true` means
    cross-encoder relevance, `false` means a raw fusion cosine (a different,
    much smaller scale). A low score never means "absent" — to decide whether
    a note exists, use silica_exists or silica_search, not a threshold here.
    """
    return {"query": query, **_facade_search(query, k=k, memory=memory)}


class RecallArgs(BaseModel):
    query: str = Field(description="The question or topic to recall memory for")
    k: int = Field(default=15, description="Maximum number of notes contributing to the context")
    memory: bool = Field(default=True, description="Include the personal-memory lane and its facts (ADR-0019). Pass false for questions scoped to THIS vault — repo and code questions — so an unrelated personal vault cannot occupy context slots.")
    vault: str = Field(default="", description="Peek: path of another Silica vault to answer from (see silica_vaults). Read-only; the session's vault does not change.")


@tool(RecallArgs, cls="composed")
def silica_recall(query: str, k: int = 15, memory: bool = True, vault: str = "") -> dict[str, Any]:
    """Assemble an answer-ready memory context for a question: fused retrieval,
    each note's query-densest window under a rank/evidence/date header,
    recalled personal facts first. Use when ANSWERING from vault memory
    INSTEAD of stitching searches and reads yourself; for a bare ranked list
    use silica_semantic_search. Answer from `context`; re-read only the notes
    named in `partial`, the rest arrived whole. Paths under `memory` live in
    the personal-memory vault; re-read them, and a peek's `partial`, with
    silica_read_note(name, vault=<memory_vault> or the same `vault`).
    """
    import datetime

    from silica.kernel.recall.perception import perceive

    from silica.kernel.code import codedocs

    peek = _peek_target(vault)
    if isinstance(peek, dict):  # a refusal, already shaped as the reply
        return {"query": query, **peek}
    p = perceive(query, now=datetime.date.today().isoformat(), k=k,
                 use_memory=memory, vault=peek)
    # Staleness is the ACTIVE vault's code lane; a peeked vault's paths would
    # only ever match it by coincidence.
    stale_map = {} if peek else _peek_stale()
    flagged = {b.path: lvl for b in p.blocks
               if (lvl := codedocs.peek_level(stale_map, b.path))}
    # render(windowed=True) emits b.excerpt, so a note whose window IS its body
    # was delivered complete — measured: 3 of 9 calls in a chat were re-reads of
    # notes recall had already handed over whole.
    out = {"query": query, "context": p.render(stale=stale_map or None),
           "notes": [b.path for b in p.blocks],
           # Active-vault and peeked notes: the caller can re-read both (the
           # latter with vault=). Memory-lane notes stay out (ADR-0032) — they
           # are listed under `memory` with the vault to read them from, and
           # "recall names notes that read_note denies" is how the doctor's
           # worst report was born.
           "partial": [b.path for b in p.blocks
                       if b.origin != "memory" and b.excerpt.strip() != b.body.strip()],
           "facts": len(p.fact_hits)}
    mem_paths = [b.path for b in p.blocks if b.origin == "memory"]
    if mem_paths:
        out["memory"] = mem_paths
        from silica.kernel.recall.memory_lane import memory_vault

        mv = memory_vault()
        if mv is not None:
            out["memory_vault"] = str(mv)  # silica_read_note(name, vault=this) opens them
    if flagged:
        out["stale"] = flagged
    if peek:
        out["vault"] = peek
        from silica.kernel.recall.vault_registry import coverage

        cov = coverage(Path(peek))
        if cov["level"] != "indexed":
            # An empty answer from a cold index reads as "that vault knows
            # nothing about this"; what it means is that nobody indexed it.
            out["coverage"] = cov["level"]
            out["hint"] = (
                "no recall index yet: open the vault once (SILICA_VAULT=<path> silica) "
                "and run /embed and /cooccur"
                if cov["level"] == "cold" else
                "only the lexical index exists (silica_vaults can score it); run /embed "
                "and /cooccur in that vault for recall to see it")
    return out


def _peek_target(vault: str) -> str | dict | None:
    """Resolve `vault=`: None for a plain call (empty, or the active vault
    itself), the resolved path for a peek, or the error reply for a folder
    Silica never adopted (no vault.yaml — `capture.find_vault`'s own test)."""
    if not (vault or "").strip():
        return None
    from silica.config import CONFIG
    from silica.kernel.recall.vault_registry import resolve_known

    try:
        target = resolve_known(vault)
    except ValueError as e:
        return {"error": str(e), "hint": "silica_vaults lists the vaults this machine knows",
                "context": "", "notes": [], "partial": [], "facts": 0}
    active = (getattr(CONFIG, "vault_path", "") or "").strip()
    if active and Path(active).resolve() == target:
        return None
    return str(target)


class VaultsArgs(BaseModel):
    query: str = Field(default="", description="Question to score each vault against; empty = the plain list")
    k: int = Field(default=3, description="Top titles per vault")


@tool(VaultsArgs, cls="atomic")
def silica_vaults(query: str = "", k: int = 3) -> dict[str, Any]:
    """The Silica vaults this machine knows (active, personal memory, adopted
    Obsidian vaults): name, path, brief, write_dir, coverage. With `query`,
    each row adds `top` titles and `rerank`, to tell WHICH vault to read with
    silica_recall(query, vault=<path>). `home` lists the vaults that hold the
    answer ([] = none of them; null = no calibrated reranker, judge by `top`).
    Rows are in relevance order only when `ranked` is true. `coverage: "cold"`
    = never indexed, not "no hits". Never switches or fuses vaults.
    """
    from silica.config import CONFIG
    from silica.kernel.recall import vault_registry

    out = vault_registry.route(query, k=k)
    active = next((r["path"] for r in out["vaults"] if r["active"]),
                  (getattr(CONFIG, "vault_path", "") or ""))
    return {"query": query, "active": active, **out}


class TimelineArgs(BaseModel):
    start: str = Field(default="", description="Inclusive ISO start date (empty = unbounded)")
    end: str = Field(default="", description="Inclusive ISO end date (empty = unbounded)")
    limit: int = Field(default=50, description="Maximum rows; on overflow the most recent are kept")


@tool(TimelineArgs, cls="composed")
def silica_timeline(start: str = "", end: str = "", limit: int = 50) -> dict[str, Any]:
    """Chronological index of the vault's dated notes, oldest first — for
    ORDERING and TIME questions ("when did X happen", "most recent Z"). Reads
    `date` frontmatter; undated notes are excluded. Order here, then
    silica_read_note the stem for detail. Content-based recall: silica_recall.
    """
    from pathlib import Path

    from silica.config import CONFIG
    from silica.kernel.write.timeline import timeline

    vault = Path(getattr(CONFIG, "vault_path", "") or "").expanduser()
    if not vault.is_dir():
        return {"error": "No vault configured."}
    t = timeline(vault, start=start, end=end, limit=limit)
    lines = [f"{date}  -> {label} ({stem}.md)" for date, label, stem in t["rows"]]
    if t["dropped"]:
        lines.append(f"...and {t['dropped']} more dated notes before this range")
    return {"timeline": "\n".join(lines),
            "total_dated": t["total_dated"], "dropped": t["dropped"]}


class RelatedArgs(BaseModel):
    note: str = Field(description="Note name (wikilink-style) or vault-relative path to find related notes for")
    k: int = Field(default=5, description="Number of results to return")
    memory: bool = Field(default=True, description="Include the personal-memory lane (ADR-0019). Pass false for questions scoped to THIS vault — repo and code questions — so an unrelated personal vault cannot occupy result slots.")

@tool(RelatedArgs, cls="composed")
def silica_related(note: str, k: int = 5, memory: bool = True) -> dict[str, Any]:
    """Given an EXISTING note (by name or path), the notes most related to it —
    embeddings + co-occurrence fused into one ranked shortlist, then reranked.

    Use for "what's related to note X" INSTEAD of reading X and
    keyword-searching; for free-form text use silica_semantic_search. Each
    result carries `evidence` (which metric proposed it, with native scores),
    `cluster`, and `distance` (wikilink hops; null = unreachable). High score +
    null/large distance = a missing link worth creating; distance 1 = already
    linked. Verify with silica_read_note before acting.

    `score` follows the silica_semantic_search contract: comparable only
    WITHIN one call — `reranked: true` means cross-encoder relevance, `false`
    means first-stage fusion mass (a much smaller scale). `legs` names which
    retrieval legs actually ranked; an absent leg is not agreement.
    """
    from silica.config import CONFIG
    from silica.driver import DRIVER
    from silica.kernel.recall.cooccurrence import (
        CooccurStore, cooccur_key, get_cooccur_store,
    )
    from silica.kernel.recall.embed import get_store
    from silica.kernel.recall.relatedness import related_notes

    # Resolve name-or-path to the canonical vault path (any backend), then reduce to
    # the store keyspace via cooccur_key (strip .md, posix, CASE-PRESERVED). This is
    # the single source of truth for both index keyspaces: it makes the query hit the
    # stored vectors/nodes AND lets related_notes exclude the query itself (blocking
    # a raw ".md" path would let the note resurface among its own results). Never
    # _norm_path here — its lowercasing misses the case-preserving stored keys.
    try:
        query_path = DRIVER.read_note(note).ref.path
        resolved = bool(query_path)  # a backend returning '' must not pass as resolved
    except Exception:
        resolved = False
    if not resolved:
        query_path = note  # unresolved: treat the input itself as a path
    note_path = query_path if resolved else ""  # pre-cooccur_key form, for file_refs_of
    query_path = cooccur_key(query_path)

    embed_store = get_store()
    cooccur_store: CooccurStore | None
    try:
        cooccur_store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
        if len(cooccur_store) == 0:
            cooccur_store = None
    except Exception:
        cooccur_store = None
    if len(embed_store) == 0 and cooccur_store is None:
        return {"note": note, "error": "No index available. Run silica_embed_refresh or silica_cooccurrence_refresh first."}

    mem_embed = mem_cooccur = None
    if memory:
        from silica.kernel.recall.memory_lane import memory_stores

        mem_embed, mem_cooccur = memory_stores()  # ADR-0019 second recall lane
    results = related_notes(
        query_path,
        embed_store=embed_store,
        cooccur_store=cooccur_store,
        memory_embed_store=mem_embed,
        memory_cooccur_store=mem_cooccur,
        k=k,
    )
    # Same score contract as silica_semantic_search (ADR-0032): reorder-only
    # within the fused top-k (membership stays first-stage, retrieval-gates 2a),
    # query = the seed's bare title — the one measured rerank query shape. The
    # raw RRF term 1/(60+rank) the payload used to expose as `score` cannot
    # separate candidates by construction (measured 2026-08-25: six results
    # inside [0.0159, 0.0164]).
    rr: dict[str, Any] = {"reranked": False}
    if results:
        from silica.agent.providers import get_reranker
        from silica.kernel.recall.rerank import link_query, rerank_related

        reranker = get_reranker(CONFIG)
        if reranker:
            results = rerank_related(reranker, link_query(query_path), results,
                                     k=k, stats=rr)
    # Cluster membership from the cached ctx (last Louvain run; {} when cold):
    # tells the caller whether a candidate sits in the query's own knowledge
    # area or across a cluster boundary. Memory-lane notes are another vault —
    # never annotated.
    from silica.kernel.recall.graph_export import cluster_hub_of, graph_distances, load_cluster_ctx

    gctx_map = (load_cluster_ctx() or {}).get("ctx") or {}
    # Structural distance: wikilink hops from the query to each result — the
    # per-pair coherence read. High fused score + null (unreachable) or large
    # distance = a missing link worth creating; distance 1 = already linked.
    # Omitted entirely when the wikilink graph is unavailable.
    dists = graph_distances(query_path)
    stale_map = _peek_stale()
    out: dict[str, Any] = {
        "note": note,
        "reranked": bool(rr.get("reranked", False)),
        # Which legs actually ranked: an absent leg (cold index, lane off) is
        # otherwise indistinguishable from unanimous agreement — 2026-08-25 the
        # embed index was empty and six cooccur-only results read as consensus.
        "legs": {
            "embed": len(embed_store) > 0,
            "cooccur": cooccur_store is not None,
            "memory": mem_embed is not None or mem_cooccur is not None,
        },
        "results": [
            {
                "path": r.path,
                "name": r.name,
                "score": round(r.score, 4),
                "evidence": r.evidence,
                **({"embed": round(r.embed_score, 3)}
                   if r.embed_score is not None else {}),
                **({"cooccur": int(round(r.cooccur_weight))}
                   if r.cooccur_weight is not None else {}),
                **(
                    {"cluster": hub}
                    if r.origin != "memory" and (hub := cluster_hub_of(gctx_map, r.path))
                    else {}
                ),
                **(
                    {"distance": dists.get(r.path)}
                    if dists is not None and r.origin != "memory"
                    else {}
                ),
                **({"origin": "memory"} if r.origin == "memory" else {}),
                **_stale_entry(stale_map, r),
            }
            for r in results
        ],
    }
    if note_path:
        try:
            fr = DRIVER.file_refs_of(note_path)
            files = {kind: v for kind, v in fr.items() if v and kind != "unresolved"}
            if files:
                # Structural section, not a retrieval leg: what the seed note
                # references on disk (embeds, documents:). Never enters fusion
                # (ADR-0018 boundary); omitted when empty to save tokens.
                out["files"] = files
        except Exception:
            pass  # a backend without file refs must not fail the related call
    if not results:
        # Empty is ambiguous — say why so the caller acts instead of guessing.
        if not resolved:
            out["hint"] = f"note '{note}' did not resolve to a vault note — check the name/path."
        elif len(embed_store) == 0:
            out["hint"] = "embedding index empty — co-occurrence only. Run silica_embed_refresh for semantic neighbors."
    return out


class ConceptsArgs(BaseModel):
    term: str = Field(default="", description="A single word/concept to look up in the vault's co-occurrence graph")
    note: str = Field(default="", description="An existing note (name or path): return ITS concepts instead of a term's neighbourhood")
    k: int = Field(default=10, description="Number of neighbouring concepts and containing notes to return")

@tool(ConceptsArgs, cls="composed")
def silica_concepts(term: str = "", note: str = "", k: int = 10) -> dict[str, Any]:
    """The vault's concepts (deterministic co-occurrence graph, embedder-free).

    `term=`: canonical label, centrality, top co-occurring concepts, top notes
    — for terminology decisions and picking wikilink targets BEFORE coining a
    synonym. Single words only; for a phrase query its most distinctive word.
    `note=`: that note's top concepts by weight — "what is this about" without
    reading the body. For related NOTES use silica_related or
    silica_semantic_search.
    """
    from silica.config import CONFIG
    from silica.kernel.recall.cooccurrence import get_cooccur_store
    from silica.kernel.text.text import stem_word

    try:
        store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
    except Exception as e:
        return {"error": f"co-occurrence store unavailable ({e}) — run silica_cooccurrence_refresh"}
    if len(store) == 0:
        return {"error": "co-occurrence index empty. Run silica_cooccurrence_refresh first."}

    if note:
        # The note's own concept vector is already stored per note by the index —
        # no extraction, no LLM. note_nodes() applies cooccur_key to the path itself.
        from silica.driver import DRIVER

        try:
            path = DRIVER.read_note(note).ref.path or note
        except Exception:
            path = note
        nodes = store.note_nodes(path)
        top = sorted(nodes.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        out: dict[str, Any] = {
            "note": note,
            "concepts": [{"concept": store.node_label(s), "weight": c} for s, c in top],
        }
        if not top:
            out["hint"] = (
                f"'{note}' has no entry in the co-occurrence index — check the name/path, "
                "or run silica_cooccurrence_refresh if the note is new."
            )
        return out

    if not term.strip():
        return {"error": "pass either term= (concept neighbourhood) or note= (a note's concepts)."}

    stem = stem_word(term.strip().lower(), lang=store.lang)
    neighbors = store.neighbors(term, k=k)

    # Concept -> notes inverted lookup, ranked by contribution count. Read the
    # cached stem postings directly (O(df log df)) instead of an O(N) vault scan.
    posting = store.stem_postings().get(stem, {})
    notes = sorted(posting.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    out = {
        "term": term,
        "concept": store.node_label(stem),
        "centrality": round(sum(store.adjacency().get(stem, {}).values()), 1),
        "neighbors": neighbors,
        "notes": [{"path": p, "count": c} for p, c in notes],
    }
    if not neighbors and not notes:
        out["hint"] = (
            f"'{term}' is not a concept node in the co-occurrence graph — "
            "concepts are single content words; try another word, or run "
            "silica_cooccurrence_refresh if the index is stale."
        )
    return out


class EmbedRefreshArgs(BaseModel):
    folder: str = Field(default="", description="Vault-relative folder to restrict indexing (empty = entire vault)")
    force: bool = Field(default=False, description="Re-embed all notes, even if already indexed")

@tool(EmbedRefreshArgs, cls="composed", collapse="eager")
def silica_embed_refresh(folder: str = "", force: bool = False) -> dict[str, Any]:
    """Build or refresh the vault embedding index.

    Powers silica_semantic_search, silica_related, and silica_dedup — run it
    first if those report an empty index. Incremental: skips notes already
    indexed (unless force=True). Call after bulk writes to keep it fresh.
    """
    from silica.agent.providers import get_embedder
    from silica.config import CONFIG
    from silica.kernel.recall.embed import build_index

    try:
        all_refs = DRIVER.list_files(folder or None)
    except Exception as e:
        return {"error": f"Failed to list vault files: {e}"}

    from silica.kernel.text.media import strip_images
    notes: list[tuple[str, str, str]] = []
    errors: list[str] = []
    for ref in all_refs:
        path = ref.path or ref.name
        name = ref.name or path
        try:
            nc = DRIVER.read_note(path)
            body = strip_images(nc.content or "")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        # Strip .md extension for index key
        idx_path = path.removesuffix(".md")
        notes.append((idx_path, name, body))

    if not notes:
        return {"error": "No notes found to index", "read_errors": errors}

    try:
        embedder = get_embedder(CONFIG)
        # prune drops every indexed path missing from `notes`, so `notes` may
        # only be trusted as the live set when EVERY listed note was read. A
        # transient read failure (non-UTF-8 byte, permission blip, Obsidian
        # mid-write) would otherwise evict a note that is still on disk, and no
        # later incremental refresh re-adds what it never read.
        store = build_index(embedder, notes, force=force, prune=not errors, folder=folder)
    except Exception as e:
        return {"error": f"Index build failed: {e}", "read_errors": errors}

    out = {
        "indexed": len(store),
        "total_notes": len(notes),
        "read_errors": errors,
        "index_path": str(store._path),
    }
    # First whole-vault build also seeds the lexical index: the sweep maintains
    # stores that exist and never builds one, and until 2026-09-01 no vault on
    # the field had a lexical.json because it was a separate opt-in, so the
    # cheap stage of silica_vaults (1.4 MB for 872 notes) never existed where
    # the 18 MB embed store did. Whole vault only: a folder-scoped first build
    # would leave a partial index the sweep then keeps partial.
    from silica.kernel.recall.paths import index_dir
    if not folder and not (index_dir() / "lexical.json").is_file():
        out["lexical"] = silica_lexical_refresh().get("indexed")
    return out


class CooccurrenceRefreshArgs(BaseModel):
    folder: str = Field(default="", description="Vault-relative folder to restrict indexing (empty = entire vault)")
    force: bool = Field(default=False, description="Re-process all notes, even if already indexed")

@tool(CooccurrenceRefreshArgs, cls="composed", collapse="eager")
def silica_cooccurrence_refresh(folder: str = "", force: bool = False) -> dict[str, Any]:
    """Build or refresh the co-occurrence index: a deterministic concept graph
    from note text, works without the embedder. Powers cluster naming and
    silica_vault_report signals. Incremental (force=True to redo). Seed once
    on an existing vault; writes keep it fresh afterwards.
    """
    from silica.config import CONFIG
    from silica.kernel.recall.cooccurrence import _index_path, build_index, get_cooccur_store

    try:
        all_refs = DRIVER.list_files(folder or None)
    except Exception as e:
        return {"error": f"Failed to list vault files: {e}"}

    notes: list[tuple[str, str, str]] = []
    errors: list[str] = []
    for ref in all_refs:
        path = ref.path or ref.name
        name = ref.name or path
        try:
            # Pass RAW content: build_contribution strips frontmatter + media itself.
            body = DRIVER.read_note(path).content or ""
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        idx_path = path.removesuffix(".md")
        notes.append((idx_path, name, body))

    if not notes:
        return {"error": "No notes found to index", "read_errors": errors}

    store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
    try:
        # prune=not errors: `notes` is the live set only when every listed note
        # was actually read — see silica_embed_refresh. A note skipped by a
        # transient read failure must not be un-edged and deleted.
        # refreeze rides on force: `/cooccur --force` is the deliberate rebuild
        # (and the doctor remedy for a wrong-frozen store language) — it
        # re-processes every note, so re-detecting store.lang here is safe.
        # A plain incremental /cooccur skips already-indexed notes and must NOT
        # refreeze: flipping the language without re-stemming existing
        # contributions would mix stemmers across node keys.
        # prune: drop nodes for notes deleted out-of-band. save=False: one
        # flush at the end after the prune.
        build_index(
            notes, store=store, lang=CONFIG.cooccurrence_lang,
            force=force, refreeze=force, save=False, prune=not errors, folder=folder,
        )
    except Exception as e:
        return {"error": f"Index build failed: {e}", "read_errors": errors}

    # A refresh that indexed nothing must not rewrite the index. Every graph
    # export calls this, the file is ~9.5 MB on a 700-note vault, and an
    # unchanged rewrite still moves the mtime - which is what vault_version()
    # digests, so the GUI's "vault changed" chip fired after every build of the
    # graph it had just drawn. force still writes, and so does an absent file:
    # skipping there would leave the vault with no index and nothing to say so.
    if force or store.is_dirty() or not _index_path().exists():
        store.save()

    return {
        "indexed": len(store),
        "total_notes": len(notes),
        "read_errors": errors,
        "index_path": str(store._path),
    }


class LexicalRefreshArgs(BaseModel):
    folder: str = Field(default="", description="Vault-relative folder to restrict indexing (empty = entire vault)")
    force: bool = Field(default=False, description="Rebuild the scanned folder's slice from empty (the whole index when folder is empty)")

@tool(LexicalRefreshArgs, cls="composed", collapse="eager")
def silica_lexical_refresh(folder: str = "", force: bool = False) -> dict[str, Any]:
    """Build or refresh the lexical (BM25/fuzzy) index over note title+body —
    strong on rare tokens, proper nouns, dates. Seeds the optional use_lexical
    retrieval leg; run once on an existing vault, writes keep it fresh
    afterwards.
    """
    from silica.kernel.recall.lexical import get_lexical_store

    try:
        all_refs = DRIVER.list_files(folder or None)
    except Exception as e:
        return {"error": f"Failed to list vault files: {e}"}

    notes: list[tuple[str, str, str]] = []
    errors: list[str] = []
    for ref in all_refs:
        path = ref.path or ref.name
        name = ref.name or path
        try:
            # RAW content (matches the write-hook + cooccur): keeps a bulk-seeded
            # index identical to an incrementally-maintained one.
            body = DRIVER.read_note(path).content or ""
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        idx_path = path.removesuffix(".md")
        notes.append((idx_path, name, body))

    if not notes:
        return {"error": "No notes found to index", "read_errors": errors}

    store = get_lexical_store()
    if force:
        for p in list(store.paths()):
            if _in_folder(p, folder):
                store.remove(p)
    for idx_path, name, body in notes:
        store.upsert(idx_path, name, body)

    # GC: drop indexed notes no longer present in the (folder-scoped) vault.
    # Scoped to the full LISTING, not the notes read this run: a note whose read
    # failed transiently is still on disk, and evicting it here would silently
    # cost recall forever (no later incremental refresh re-adds what it skipped).
    current_paths = {(ref.path or ref.name).removesuffix(".md") for ref in all_refs}
    stale_paths = [
        p for p in store.paths()
        if _in_folder(p, folder) and p not in current_paths
    ]
    for p in stale_paths:
        store.remove(p)
    store.save()

    return {
        "indexed": len(store),
        "total_notes": len(notes),
        "read_errors": errors,
        "index_path": str(store._path),
    }


def _covering_stem(path: Path) -> str:
    """Stem of the note that would cover `path`, "" when only its own can.

    A tabular profile is named for its shard FAMILY, so the 12 members of a
    converted family each read as unconverted under a plain stem match. This
    is the exact inverse of `convert._family_stem` (strip the trailing
    counter), not a fuzzy match — and it is restricted to data files because
    families exist nowhere else, which bounds a wrong answer here to
    suppressing one hint about one CSV.
    """
    from silica.sources.convert import TABULAR_EXTS

    if not path.suffix.lower().endswith(TABULAR_EXTS):
        return ""
    stem = re.sub(r"[\s_.\-0-9]+$", "", path.stem)
    return stem if len(stem) >= 3 and stem != path.stem else ""


class VaultReportArgs(BaseModel):
    folder: str = Field(default="", description="Vault-relative folder to scope (empty = whole vault)")
    top_k: int = Field(default=10, description="How many god-nodes / bridges to surface")
    with_embeddings: bool = Field(default=False, description="Also propose missing links via the embedding index")
    with_cooccurrence: bool = Field(default=False, description="Also compute the co-occurrence vs wikilink delta (autolink candidates, stale links, missing hubs) — embedder-free")
    seed_ledger: bool = Field(default=True, description="Persist a run (TaskLedger+ProgressLedger) pre-seeded with remediation tasks")

@tool(VaultReportArgs, cls="composed")
def silica_vault_report(
    folder: str = "",
    top_k: int = 10,
    with_embeddings: bool = False,
    with_cooccurrence: bool = False,
    seed_ledger: bool = True,
) -> dict[str, Any]:
    """Deterministic structural audit — the entry point for /graph and vault
    health. Computes god-nodes, surprising cross-cluster connections, orphans,
    dangling links, clusters; writes GRAPH_REPORT.md and (seed_ledger=True)
    seeds a remediation run for silica_ledger_next. Tiers: auto (execute
    without confirmation), propose (ask first), escalate (IssueCards, human
    judgment). Visual graph: silica_graph_export. Straight to executable
    maintenance: silica_curate.
    """
    import orjson
    from pathlib import Path

    from silica.config import CONFIG
    from silica.kernel.report.graph_report import compute_report, to_digest, to_facts, write_report
    from silica.kernel.analyst_plan import build_task_plan
    from silica.kernel.progress import IssueCard, Run

    # 1. Build report (on-demand /graph: full analytics — god_nodes/bridges/cohesion)
    report = compute_report(
        folder=folder, top_k=top_k, analytics=True,
        with_embeddings=with_embeddings, with_cooccurrence=with_cooccurrence,
    )

    # Warm the cluster-ctx cache so silica_related/build_substrate can annotate
    # candidates with their community without a nucleate run having happened.
    # Whole-vault only: a folder-scoped map would clobber the global one.
    if not folder:
        try:
            from silica.kernel.recall.graph_export import ctx_from_report, save_cluster_ctx

            save_cluster_ctx(
                [report.totals.get("notes", 0), report.totals.get("links", 0)],
                ctx_from_report(report),
            )
        except Exception as exc:
            logger.debug("silica_vault_report: cluster ctx warm skipped (%s)", exc)

    # 2. Determine output path
    vault_path = getattr(CONFIG, "vault_path", None) or ""
    if vault_path:
        report_path = str(Path(vault_path) / "GRAPH_REPORT.md")
    else:
        report_path = "GRAPH_REPORT.md"

    paths = write_report(report, report_path)

    result: dict[str, Any] = {
        "digest": to_digest(report),
        "report_md": paths["path_md"],
    }

    # Material waiting in the inbox is part of the vault's state: an audit of
    # an "empty" vault that holds four PDFs must say so, or the caller reports
    # "nothing to do" over a folder full of work (observed 2026-08-15).
    try:
        from silica.tools.atomic import _unconverted_under
        from silica.kernel.vault_manifest import active_inbox_dir

        inbox = active_inbox_dir()
        pending = _unconverted_under(inbox) if inbox else []
        if pending:
            result["inbox_pending"] = {
                "total": len(pending),
                "files": pending[:20],
                "hint": "not yet in the vault — suggest /nucleate <path> (or the folder) to bring them in",
            }
    except Exception as exc:
        logger.debug("silica_vault_report: inbox pending scan skipped (%s)", exc)

    # The inbox is not the only place unread material sits: a student's
    # slide/*.pdf never entered Inbox/, and the audit read as "nothing to do"
    # over a folder of unconverted documents (observed 2026-08-15). Documents
    # whose converted .md already exists (inbox or done/) are not re-reported.
    try:
        if vault_path:
            from silica.kernel.vault_manifest import active_done_dir
            from silica.onboarding.wizard import unindexable_docs

            root = Path(vault_path)
            # rglob, not glob: the archive mirrors the inbox tree since
            # 2026-08-23, so a flat scan saw only what had been archived from
            # the inbox root and re-reported every nested book as unread.
            converted_stems = {
                p.stem for p in (root / active_done_dir()).rglob("*.md")
            }
            try:
                from silica.driver import DRIVER as _drv

                converted_stems |= {
                    Path(r.path).stem for r in _drv.list_inbox_files()
                    if r.path.endswith(".md")
                }
            except Exception:
                pass
            docs = [
                p.relative_to(root).as_posix()
                for p in unindexable_docs(root)
                if p.stem not in converted_stems
                and (_covering_stem(p) or p.stem) not in converted_stems
            ]
            if docs:
                result["unconverted"] = {
                    "total": len(docs),
                    "files": docs[:20],
                    "hint": "not readable by the index — suggest /nucleate <path> "
                            "(or /convert) to bring them in",
                }
    except Exception as exc:
        logger.debug("silica_vault_report: unconverted scan skipped (%s)", exc)

    if not seed_ledger:
        return result

    # 3. Build plan and seed ledger
    plan = build_task_plan(report)

    # Name what was queued: the chat layer asks consent on these tasks, and a
    # bare count ("2 proposed fixes") forces a blind yes/no (observed 2026-08-15).
    result["plan_preview"] = [
        {"tier": c.tier, "capability": c.capability_name, "reason": c.reason}
        for c in plan.auto + plan.propose
    ]

    run = Run.new(
        mode="analyst",
        user_request=f"audit {folder or 'vault'}",
        checkpoints=plan.checkpoints,
        inputs={"scope": folder or "vault"},
        facts=to_facts(report),
    )
    payloads_dir = run.payloads_dir

    # Seed tasks from auto + propose (propose carries needs_confirmation flag)
    for candidate in plan.auto + plan.propose:
        task = run.progress.add_task(candidate.capability_name)
        # Write payload to disk
        payload = dict(candidate.payload)
        payload["_reason"] = candidate.reason
        if candidate.tier == "propose":
            payload["needs_confirmation"] = True
        payload_path = str(payloads_dir / f"{task.id}.json")
        Path(payload_path).write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        task.input_ref = payload_path

    # Escalate items → IssueCards
    for i, candidate in enumerate(plan.escalate):
        card = IssueCard(
            task_id=f"issue_{i}",
            question=candidate.reason,
            # Per-candidate when the rule knows what it is asking; the dangling-link
            # wording is only the fallback, not every escalation's question.
            options=candidate.options or [
                {"label": "create_note", "description": "Create a new note with this title"},
                {"label": "rename_existing", "description": "Rename an existing note to match"},
                {"label": "ignore", "description": "Leave the broken link as-is"},
            ],
        )
        run.progress.issues.append(card)

    run.save()

    result["run_id"] = run.run_id
    result["auto"] = len(plan.auto)
    result["propose"] = len(plan.propose)
    result["issues"] = len(plan.escalate)

    return result


@tool(EmptyArgs, cls="composed")
def silica_health() -> dict[str, Any]:
    """Retrieval + write-path health check, live: `fusion` (masked-wikilink
    recovery — low recall or embed_coverage < 1.0 means related/semantic
    search is degraded; refresh with silica_embed_refresh /
    silica_cooccurrence_refresh and re-run) and `integrity` (differential lint
    across the write-path transforms — under 1.0 the pipeline CORRUPTS note
    bodies and writes should stop; null means the vault held no notes and
    nothing was measured). Full-vault sweep, on demand, not a
    per-write gate. Structural audit of content: silica_vault_report.
    """
    from pathlib import Path

    from silica.config import CONFIG
    from silica.kernel.recall.cooccurrence import get_cooccur_store
    from silica.kernel.recall.embed import get_store
    from silica.kernel.link.health import fusion_probe, integrity_probe

    vault = Path(getattr(CONFIG, "vault_path", "") or "").expanduser()
    if not vault.is_dir():
        return {"error": "No vault configured."}

    try:
        store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
    except Exception as e:
        store = None
        fusion: dict[str, Any] = {"error": f"co-occurrence store unavailable ({e}) — run silica_cooccurrence_refresh"}
    if store is not None:
        fusion = fusion_probe(vault, store, embed_store=get_store())

    return {"fusion": fusion, "integrity": integrity_probe(vault)}
