# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Pipeline tools — the mechanical injector stages as system tools.

Recon → payload → sanitize → validate → bulk-write → lint, plus the
deferred-ops retry path. The per-stage tools are registered internal=True:
the InjectorFSM drives them programmatically and they are hidden from the
main agent's default toolset (the full run lives in silica.tools.runners,
exposed as silica_run_injector). Only the deferred retry is agent-facing.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.tools import tool
from silica.kernel.write.ops import OpType
from silica.kernel.write.ops_io import load_ops, dump_ops

logger = logging.getLogger(__name__)


def _link_recovered_writes(
    ops: list, target_dir: str, hub: str | None, source_path: str = ""
) -> None:
    """Give anneal-recovered writes the graph edges the FSM's AUTOLINK and
    HUB_UPDATE states would have added — the deferred retry path bypasses both,
    so recovered notes otherwise land as orphans with zero edges and no MOC
    membership (audit finding 2).

    Best-effort, mirroring the FSM's non-fatal stance: both passes only ADD
    links; neither can break a valid note.
    """
    import os
    from silica.kernel.link.autolink import build_title_index
    from silica.kernel.write.moc import hub_desc, merge_moc_section, moc_heading

    hub_name = (hub or "").strip("[]")
    hub_l = hub_name.lower()
    written = [
        op for op in ops
        if op.op in (OpType.write, OpType.overwrite) and op.touched_ref()
        and os.path.splitext(os.path.basename(op.touched_ref()))[0].lower() != hub_l
    ]
    if not written:
        return

    # Inline autolink (what the FSM's AUTOLINK state does per chunk).
    try:
        title_index = build_title_index(DRIVER.list_files(target_dir or ""))
    except Exception as e:
        logger.warning("anneal: title-index build failed, recovered notes stay unlinked: %s", e)
        return
    for op in written:
        try:
            DRIVER.autolink_note(
                op.touched_ref(), candidates=title_index, title_index=title_index
            )
        except Exception as e:
            logger.debug("anneal: autolink skipped '%s' (non-fatal): %s", op.touched_ref(), e)

    # Hub-MOC membership (what the FSM's HUB_UPDATE state does per chunk):
    # same heading/merge helpers, so recovered bullets coalesce with the
    # section an in-flight chunk of the same source already created.
    if not hub_name:
        return
    from silica.kernel.write.moc import moc_target
    hub_path = moc_target(hub_name, target_dir)
    if not hub_path:  # staging note, never a MOC target
        return
    try:
        hub_note = DRIVER.read_note(hub_path)
    except Exception as e:
        logger.warning("anneal: hub '%s' not readable, MOC membership skipped: %s", hub_path, e)
        return
    try:
        entries = [
            (os.path.splitext(os.path.basename(op.touched_ref()))[0],
             hub_desc(op.snippet or op.content or ""))
            for op in written
        ]
        source_name = os.path.splitext(os.path.basename(source_path))[0] or "deferred"
        sample = hub_note.content + " ".join(d for _, d in entries[:3])
        heading = moc_heading(source_name, sample)
        note_lines = [f"- [[{n}]] — {d}" if d else f"- [[{n}]]" for n, d in entries]
        DRIVER.overwrite(hub_path, merge_moc_section(hub_note.content, heading, note_lines))
        logger.info(
            "anneal: %d recovered note(s) autolinked and added to hub '%s' MOC",
            len(written), hub_name,
        )
    except Exception as e:
        logger.warning("anneal: hub MOC update failed (non-fatal): %s", e)


def _recon_embedder():
    """Pool ranker for recon; None (=> mined-rank fallback) when unavailable.

    Module-level seam so tests can disable the network embedder (see conftest)
    and keep recon deterministic; production uses the real embedder.
    """
    try:
        from silica.agent.providers import get_embedder
        from silica.config import CONFIG
        return get_embedder(CONFIG)
    except Exception:
        return None


def _same_note(ref_a, ref_b) -> bool:
    """Path-safe comparison between two NoteRefs — handles slashes, casing, and .md suffix."""
    import os
    def norm(r):
        p = r.path or r.name
        return os.path.normcase(p.replace("\\", "/").removesuffix(".md").strip("/"))
    return norm(ref_a) == norm(ref_b)


def _prefers(candidate, incumbent) -> bool:
    """True when `candidate` is the vault original and `incumbent` its mirror copy.

    Only under safe mode is a note ever present twice; everywhere else this
    answers False and grouping keeps its first hit exactly as before.
    """
    from silica.kernel.vault_manifest import active_write_dir, within
    from silica.onboarding.adopt import SAFE_WRITE_DIR

    if active_write_dir() != SAFE_WRITE_DIR:
        return False
    a, b = candidate.path or "", incumbent.path or ""
    return bool(a and b) and within(b, SAFE_WRITE_DIR) and not within(a, SAFE_WRITE_DIR)


class ReconArgs(BaseModel):
    inbox_file: str = Field(description="Path to the inbox file to analyze")
    limit: int = Field(default=0, description="Limit for concept extraction")

@tool(ReconArgs, cls="composed", internal=True)
def silica_recon(inbox_file: str, limit: int = 0) -> dict[str, Any]:
    """Mechanical extraction of concepts from an inbox file and searching for collisions in the vault."""
    from silica.kernel.text.recon import (
        collision_priority, is_title_match, mentions_whole_word, rank_hits,
    )
    from silica.kernel.text.keyphrase import extract_keyphrases
    from silica.config import CONFIG

    try:
        nc = DRIVER.read_note(inbox_file)
    except RuntimeError:
        return {"error": f"File not found: {inbox_file}"}

    embedder = _recon_embedder()
    cands = extract_keyphrases(
        nc.content,
        lang=CONFIG.cooccurrence_lang, embedder=embedder,
    )

    concepts = [c.phrase for c in cands]
    if not concepts:
        return {"file": inbox_file, "collisions": [], "new_concepts": []}

    collisions = []
    new_concepts = []
    
    batch = DRIVER.search_context_batch(concepts)   # one eval instead of N
    for c in concepts:
        hits = batch.get(c, [])
        if not hits:
            new_concepts.append(c)
            continue
            
        # Group hits by ref
        from silica.kernel.recall.paths import is_inbox_path
        grouped = {}
        for h in hits:
            # Casefolded: the archive is `Done` since 2026-08-23 while a vault
            # that predates the rename keeps its `done`, and an archived source
            # registered as a collision target dooms every downstream op.
            _p = (h.ref.path or '').replace('\\', '/').casefold()
            if '/done/' in _p or _p.startswith('done/'):
                continue
            # Inbox notes are staging, never collision targets: registering one
            # as vault_collision dooms every downstream op (patch-the-inbox is
            # forbidden, patch-the-right-note mismatches the expected collision).
            if h.ref.path and is_inbox_path(h.ref.path):
                continue
            if _same_note(h.ref, nc.ref):
                continue
            # Whole-word only (see mentions_whole_word): the driver matched a
            # substring, which is its contract, not this one's. ponytail: the
            # driver caps materialized hits at 20 per note BEFORE this filter,
            # so a note whose first 20 matching lines are all fragments hides a
            # later whole-word line; reopen if a recon run misses a collision
            # that `silica_search_context` shows on line 21 or beyond.
            if not mentions_whole_word(c, h.snippet):
                continue

            name = h.ref.name
            if name not in grouped:
                grouped[name] = {"ref": h.ref, "count": 0}
            elif _prefers(h.ref, grouped[name]["ref"]):
                # Safe mode puts a preview copy of a note beside the note. They
                # group under one name, so WHICH path answered was down to scan
                # order — and the model was shown its own unmerged draft as the
                # vault's state. The original always wins; the copy exists to be
                # pasted over it, not to stand in for it.
                grouped[name]["ref"] = h.ref
            grouped[name]["count"] += 1
            
        if not grouped:
            new_concepts.append(c)
            continue
            
        raw_hits = []
        for name, data in grouped.items():
            in_t = is_title_match(c, name)
            raw_hits.append({
                "path": data["ref"].path or data["ref"].name,
                "count": data["count"],
                "in_title": in_t
            })
            
        ranked = rank_hits(raw_hits)
        collisions.append({
            "name": c,
            "total_hits": sum(h["count"] for h in raw_hits),
            "best_match": "title" if ranked[0]["in_title"] else "body",
            "hits": ranked
        })
        
    collisions.sort(key=collision_priority)
    new_concepts.sort()
    
    return {
        "file": inbox_file,
        "collisions": collisions,
        "new_concepts": new_concepts,
    }


class PayloadArgs(BaseModel):
    recon_report_path: str = Field(description="Path to the recon report JSON file")
    max_concepts: int = Field(default=7, description="Maximum concepts per batch")
    max_bytes: int = Field(default=80 * 1024, description="Maximum bytes (JSON size) per chunk")

@tool(PayloadArgs, cls="composed", internal=True)
def silica_payload(recon_report_path: str, max_concepts: int = 7, max_bytes: int = 80 * 1024) -> dict[str, Any]:
    """Assembles payloads for the Distiller by pre-extracting snippets from the vault."""
    import orjson
    from silica.kernel.text.payload import build_payload
    from silica.kernel.partition import partition_by_concepts
    
    try:
        with open(recon_report_path, 'rb') as f:
            recon_reports = orjson.loads(f.read())
    except Exception as e:
        return {"error": f"Failed to read recon report: {e}"}
        
    # We use a default window of 450 chars
    payload = build_payload(recon_reports, window=450)
    
    # C4/S3.1: Always run partition_by_concepts if we have constraints
    if max_concepts > 0 or max_bytes > 0:
        chunks = partition_by_concepts(payload, max_concepts, max_bytes)
        return {"chunks": chunks}
        
    return {"payload": payload}


class SanitizeArgs(BaseModel):
    distiller_output_path: str = Field(description="Path to the raw distiller output JSON file")
    verbatim_source: str | None = Field(
        default=None,
        description="The chunk's own inbox text; anchors verbatim-body escape repair per-site")

@tool(SanitizeArgs, cls="composed", internal=True)
def silica_sanitize(distiller_output_path: str,
                    verbatim_source: str | None = None) -> dict[str, Any]:
    """Validates and sanitizes the JSON returned by Distiller workers."""
    from silica.kernel.text.sanitize import TruncatedArray, parse_json, normalize_ops

    try:
        with open(distiller_output_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
    except Exception as e:
        return {"error": f"Failed to read distiller output: {e}"}

    salvaged = None
    try:
        parsed_obj, was_clean = parse_json(raw_content, strict=False)
    except TruncatedArray as t:
        # max_tokens cut the payload: apply the clean leading ops instead of
        # losing the whole chunk, and say so — the loss must reach the caller.
        parsed_obj, was_clean = t.ops, False
        salvaged = {"recovered_ops": len(t.ops), "lost_chars": len(t.tail)}
        logger.warning(
            "sanitize: distiller output truncated — recovered %d leading ops, "
            "%d chars lost (%s)",
            len(t.ops), len(t.tail), distiller_output_path,
        )
    except Exception as e:
        return {"error": f"JSON Parse Error: {e}"}

    # Normalize op content: strip .md from wikilinks, etc.
    if isinstance(parsed_obj, list):
        parsed_obj = normalize_ops(parsed_obj, verbatim_source=verbatim_source)
    elif isinstance(parsed_obj, dict) and "updates" in parsed_obj:
        parsed_obj["updates"] = normalize_ops(parsed_obj["updates"],
                                              verbatim_source=verbatim_source)

    # Axis enforcement (Layer 2): demote ops whose linked_axis is not in main_thematic_axes.
    # Only activates when the distiller actually emitted axes — graceful degradation otherwise.
    if isinstance(parsed_obj, dict):
        axes = {a.strip().lower() for a in parsed_obj.get("main_thematic_axes", []) if a}
        if axes:
            for op in parsed_obj.get("updates", []):
                if isinstance(op, dict) and op.get("op") in ("write", "patch"):
                    la = (op.get("linked_axis") or "").strip().lower()
                    if la and la not in axes:
                        op["op"] = "skip"
                        op["reason"] = f"unlinked_axis '{op.get('linked_axis')}' not in main_thematic_axes"

    result = {
        "success": True,
        "parsed": parsed_obj,
        "was_clean": was_clean
    }
    if salvaged:
        result["salvaged"] = salvaged
    return result


class ValidateOpsArgs(BaseModel):
    ops_json_path: str = Field(description="Path to the consolidated operations JSON file to validate")
    payload_paths: list[str] = Field(default_factory=list, description="Paths to the original payload JSON files")
    target_dir: str = Field(default="", description="Target folder in the vault")
    hub: str = Field(default="", description="Hub note name")
    future_ref_whitelist: list[str] = Field(default_factory=list, description="Optional whitelist of future reference note names")
    profile: str | None = Field(default=None, description="Per-run distill profile (None = process-global resolution)")

@tool(ValidateOpsArgs, cls="composed", internal=True)
def silica_validate_ops(
    ops_json_path: str,
    payload_paths: list[str] | None = None,
    target_dir: str = "",
    hub: str = "",
    future_ref_whitelist: list[str] | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Pre-write gate: checks structural validity and applies rejection threshold (10%).

    C4: After validation, OVERWRITES ops_json_path with the coerced + deduped
    validated ops. Snapshot and bulk_write MUST read from the same ops_json_path
    after this call — never from a pre-validation snapshot.
    """
    import orjson
    from silica.kernel.write.validate import validate_operations

    if payload_paths is None:
        payload_paths = []

    try:
        ops = load_ops(ops_json_path)
    except Exception as e:
        return {"error": f"Failed to load operations: {e}"}

    payloads = []
    for path in payload_paths:
        try:
            with open(path, 'rb') as f:
                payloads.append(orjson.loads(f.read()))
        except Exception as e:
            return {"error": f"Failed to load payload {path}: {e}"}

    cleared_parents: list[dict] = []
    cleared_links: list[dict] = []
    ungrounded: list[dict] = []
    normalized: dict[str, int] = {}
    validated_ops, rejected_ops = validate_operations(
        ops,
        payloads,
        target_dir,
        hub=hub,
        cleared_parents_out=cleared_parents,
        future_ref_whitelist=future_ref_whitelist,
        cleared_links_out=cleared_links,
        ungrounded_out=ungrounded,
        profile=profile,
        normalized_out=normalized,
    )

    total = len(ops)
    rejected_count = len(rejected_ops)
    # C4 denominator: skip ops excluded from rejection rate
    actionable = sum(1 for o in ops if o.op != OpType.skip)
    rejection_rate = rejected_count / actionable if actionable > 0 else 0.0

    # C4: Always overwrite ops_json_path with validated (coerced + deduped) ops —
    # even when rejection_rate exceeds the old 10% threshold.  Policy (abort vs.
    # continue) is the FSM's responsibility; the tool is a pure filter.
    try:
        dump_ops(ops_json_path, validated_ops)
    except Exception as e:
        return {"error": f"Failed to persist validated ops: {e}"}

    return {
        "success": True,
        "total": total,
        "validated_count": len(validated_ops),
        "rejected_count": rejected_count,
        "rejection_rate": rejection_rate,
        "validated_ops": [o.model_dump() for o in validated_ops],
        "rejected_ops": [r.model_dump() for r in rejected_ops],
        "cleared_parents": cleared_parents,
        "cleared_links": cleared_links,
        "ungrounded": ungrounded,
        # Malformed-but-unambiguous ops repaired rather than rejected, per
        # pattern. The reject total says an op failed; this says which prompt
        # rule the model keeps missing.
        "normalized": normalized,
    }


class BulkWriteArgs(BaseModel):
    ops_json_path: str = Field(description="Path to the validated operations JSON file")

@tool(BulkWriteArgs, cls="composed", collapse="eager", internal=True)
def silica_bulk_write(ops_json_path: str) -> dict[str, Any]:
    """Applies write/patch/overwrite/delete operations in batch in the vault."""
    from silica.kernel.write.bulk import execute_operations

    try:
        ops = load_ops(ops_json_path)
    except Exception as e:
        return {"error": f"Failed to load operations: {e}"}

    res = execute_operations(ops)
    return res.model_dump()


class LintArgs(BaseModel):
    note_name: str = Field(description="Name of the note to lint")
    op_type: str = Field(default="", description="Operation type (write/patch/overwrite) for conditional checks")
    hub: str = Field(default="", description="Hub note name for wikilink validation")

@tool(LintArgs, cls="composed", internal=True)
def silica_lint(note_name: str, op_type: str = "", hub: str = "") -> dict[str, Any]:
    """Post-write gate: executes the OFM linter to find structural regressions."""
    from silica.kernel.link.linter import validate_note

    errors, warnings = validate_note(note_name, hub=hub or None, op_type=op_type or None)

    return {
        "success": len(errors) == 0,
        "note": note_name,
        "errors": errors,
        "warnings": warnings,
    }


class DeferredRetryArgs(BaseModel):
    content_hash: str = Field(description="Content hash of the deferred bundle to retry (from silica_deferred_list)")

@tool(DeferredRetryArgs, cls="composed", collapse="eager")
def silica_deferred_retry(content_hash: str) -> dict[str, Any]:
    """Retry writing a deferred op bundle: re-validates against the current vault,
    snapshots, writes the ops that now pass, and updates the bundle.

    - Ops that pass validation are written immediately.
    - Ops that still fail remain in the deferred store.
    - If the bundle is fully cleared, it is removed from the deferred store.
    """
    from silica.kernel.recall.deferred import get_deferred_store
    from silica.kernel.write.validate import validate_operations
    from silica.kernel.write.ops_io import parse_ops
    from silica.tools.wrapped import build_txn
    from silica.kernel.write.bulk import execute_operations

    store = get_deferred_store()
    bundle = store.get(content_hash)
    if not bundle:
        return {"error": f"No deferred bundle found for hash {content_hash[:8]}…"}

    rejected_raw = bundle.get("rejected_ops", [])
    target_dir = bundle.get("target_dir", "")
    hub = bundle.get("hub")

    try:
        ops = parse_ops(rejected_raw)
    except Exception as e:
        return {"error": f"Failed to parse deferred ops: {e}"}

    # Re-validate against the bundle's ORIGINAL payloads (persisted at defer
    # time) so grounding/heading/collision checks run with the same evidence
    # that rejected the ops — an empty list here used to admit them on strictly
    # weaker validation (audit finding 2). Old bundles without payloads keep
    # the previous behavior.
    payloads = bundle.get("payloads") or []
    validated, still_rejected = validate_operations(ops, payloads, target_dir, hub=hub)

    if not validated:
        return {
            "success": False,
            "message": "All deferred ops still rejected by the validator",
            "rejected": [
                {"path": r.op.path, "heading": r.op.heading, "reason": r.reason}
                for r in still_rejected
            ],
            "still_deferred": len(still_rejected),
        }

    try:
        # Snapshot before writing for rollback safety
        txn = build_txn(validated)

        result = execute_operations(validated)
        if not result.ok:
            from silica.tools.wrapped import silica_restore
            silica_restore(txn_id=txn.id, inverses=[i.model_dump() for i in txn.inverses])
            failures = [f.model_dump() for f in result.failed]
            return {"error": f"Deferred retry write failed: {failures}"}
    except Exception as e:
        return {"error": f"Deferred retry failed: {e}"}

    # Recovered writes bypassed the FSM's AUTOLINK/HUB_UPDATE — give them edges.
    _link_recovered_writes(validated, target_dir, hub, bundle.get("source_path", ""))
    # ...and they bypassed CLEANUP, which is where the journal is flushed and
    # provenance is appended. The boundary anneal runs in the FSM's `finally`,
    # after both. Measured 2026-08-18 on a 3-paper library gate: 5 of 94 notes
    # sat on disk with no inverse and no record, so `/revert` walked past them
    # and `check_renucleate` reported the source as never nucleated.
    _record_recovered_writes(txn, validated, content_hash, bundle)

    # Update or clear the deferred store
    if still_rejected:
        store.put(
            content_hash=content_hash,
            source_path=bundle.get("source_path", ""),
            target_dir=target_dir,
            hub=hub,
            rejected_ops=[r.op.model_dump() for r in still_rejected],
            rejection_reasons={
                (r.op.path or r.op.heading or "?"): r.reason for r in still_rejected
            },
            phase="RETRY",
            payloads=payloads,  # keep grounding evidence for the next retry
        )
    else:
        store.remove(content_hash)

    return {
        "success": True,
        "written": len(validated),
        "still_deferred": len(still_rejected),
        "bundle_cleared": len(still_rejected) == 0,
    }


def _record_recovered_writes(txn, validated, content_hash: str, bundle: dict) -> None:
    """Journal + provenance for ops recovered outside the FSM's CLEANUP.

    Joins the ambient run when there is one. The boundary anneal fires inside
    the FSM's `finally`, so its writes belong to the run the user just started:
    a fresh journal run carries a later `started_at`, `last_active_run` orders
    by that, and `/revert` therefore undid the handful of recovered notes and
    left the whole nucleation on disk. Only a STANDALONE `silica_deferred_retry`
    — no run to join — opens its own revertible unit. Best-effort throughout:
    the notes are already on disk, and failing here must not undo them.
    """
    import hashlib

    from silica.agent.commit import _current_ledger_run, _current_undo_run
    from silica.config import CONFIG
    from silica.driver import DRIVER

    try:
        from silica.kernel.write.undo_journal import get_undo_journal

        journal = get_undo_journal()
        run_id = _current_undo_run.get() or journal.start_run(
            source="anneal", vault=CONFIG.vault_path.strip() or None)
        for inv in txn.inverses:
            try:
                post = DRIVER.read_note(inv.path).content
                post_hash = hashlib.sha256((post or "").encode("utf-8")).hexdigest()
            except Exception:
                post_hash = None
            journal.record(run_id, inv, post_hash)
    except Exception as exc:
        logger.debug("anneal: journal record failed (non-fatal): %s", exc)
        run_id = ""

    try:
        import os as _os

        from silica.kernel.write.provenance import append_record, is_deriving_op

        source = _os.path.basename(bundle.get("source_path", "") or "")
        notes = sorted({
            (op.path or "").removesuffix(".md")
            for op in validated
            if is_deriving_op(op.op) and op.path
        })
        if source and content_hash and notes:
            # The LEDGER's run id, which is not the journal's: every reader of
            # this field (Coordinator._sweep_dangling_links) matches on
            # `fsm.progress.run_id`, so stamping the journal's uuid4 here wrote
            # a record nothing could ever find. A standalone retry has no run to
            # name and stays "anneal". A separate record either way — the ledger
            # is append-only history, never a rewrite of what CLEANUP recorded.
            append_record(source, content_hash,
                          _current_ledger_run.get() or "anneal", notes)
    except Exception as exc:
        logger.debug("anneal: provenance append failed (non-fatal): %s", exc)


class SubmitRepairedOpsArgs(BaseModel):
    content_hash: str = Field(
        description="Hash of the deferred bundle being repaired (given in the repair task)")
    ops: list[dict] = Field(
        description="Corrected ops in the standard op schema. Bodies are plain "
                    "strings: real line breaks, single backslashes.")


# collapse stays "lazy" (the default) on purpose: the verdict IS the feedback,
# and an eager stub would erase the rejection reasons from the very history the
# steer loop iterates on.
@tool(SubmitRepairedOpsArgs, cls="composed", internal=True)
def submit_repaired_ops(content_hash: str, ops: list[dict]) -> dict[str, Any]:
    """Validate and write repaired ops for a deferred bundle. Ops that pass the
    validator are written to the vault immediately and leave the bundle; ops
    that fail come back under "rejected" with the exact validator reason.
    Correct the rejected ops and resubmit them; never resubmit an op listed
    under "written" or "renamed" (renamed = written, but under a heading no
    parked op carries, so the original stays deferred).
    """
    from silica.kernel.recall.deferred import get_deferred_store
    from silica.kernel.text.sanitize import normalize_ops
    from silica.kernel.write.bulk import execute_operations
    from silica.kernel.write.ops_io import parse_ops
    from silica.kernel.write.validate import validate_operations
    from silica.tools.wrapped import build_txn

    store = get_deferred_store()
    bundle = store.get(content_hash)
    if not bundle:
        return {"error": f"No deferred bundle for hash {content_hash[:8]}…"}
    if not ops:
        return {"error": "ops is empty — submit the corrected ops"}
    target_dir = bundle.get("target_dir", "")
    hub = bundle.get("hub")
    payloads = bundle.get("payloads") or []
    # Same post-processing the main distill path gets (silica_sanitize →
    # normalize_ops): tool-arg transport guarantees well-formed JSON, not
    # well-escaped content — over-escaped LaTeX (`\\{a_c\\}`, `\\dots`) still
    # needs the per-site collapse, anchored on the bundle's own excerpts.
    anchor = "\n".join(
        c.get("inbox_excerpt") or ""
        for p in payloads if isinstance(p, dict)
        for b in p.get("batches", [])
        for c in b.get("concepts", [])
    ) or None
    fixed = parse_ops(normalize_ops(ops, verbatim_source=anchor))
    fixed = [op for op in fixed if op.op != OpType.skip]
    if not fixed:
        return {"error": "no usable ops after parsing — check the op schema"}
    # The bundle's ORIGINAL payloads, same as silica_deferred_retry (audit
    # finding 2): an empty list re-admits the fix on strictly weaker validation
    # — measured live, a hallucinated op with zero grounded facts sailed
    # through and landed in the vault.
    validated, still = validate_operations(fixed, payloads, target_dir, hub=hub)
    rejected = [{"path": r.op.path, "heading": r.op.heading, "reason": r.reason}
                for r in still]
    if not validated:
        return {"written": [], "rejected": rejected,
                "remaining": len([o for o in bundle.get("rejected_ops", [])
                                  if isinstance(o, dict)])}
    txn = build_txn(validated)
    bulk = execute_operations(validated)
    if not bulk.ok:
        from silica.tools.wrapped import silica_restore
        silica_restore(txn_id=txn.id, inverses=[i.model_dump() for i in txn.inverses])
        return {"error": "write failed; nothing was committed this call"}
    # Same bookkeeping as the mechanical retry: these writes bypassed the FSM's
    # AUTOLINK/HUB_UPDATE and CLEANUP. The old steer commit path skipped BOTH
    # calls, so steered notes landed with no edges, no undo inverse and no
    # provenance record — /revert walked past them (the 2026-08-18 defect,
    # fixed for retry only).
    _link_recovered_writes(validated, target_dir, hub, bundle.get("source_path", ""))
    _record_recovered_writes(txn, validated, content_hash, bundle)
    # written ops are dropped from the bundle by heading match only —
    # an op the model renamed stays parked and re-anneals (writes are idempotent
    # via block_present), which is the safe direction.
    renamed = [op.heading for op in validated
               if not store.remove_op(content_hash, op.heading)]
    result: dict[str, Any] = {
        "written": [op.heading for op in validated],
        "rejected": rejected,
        "remaining": len([o for o in
                          (store.get(content_hash) or {}).get("rejected_ops", [])
                          if isinstance(o, dict)]),
    }
    if renamed:
        # Named to the model, or it reads the unchanged `remaining` as a
        # failed write and resubmits the same op until the cap.
        result["renamed"] = renamed
    return result


class AnnealArgs(BaseModel):
    steer: bool = Field(
        default=False,
        description="After the mechanical pass, hand each bundle's still-failing ops to the escalation model (a bounded repair loop per bundle)",
    )
    limit: int = Field(default=0, description="Max bundles to process (0 = all)")

@tool(AnnealArgs, cls="composed", collapse="eager")
def silica_anneal(steer: bool = False, limit: int = 0) -> dict[str, Any]:
    """Boundary annealing: sweep EVERY deferred bundle through the mechanical
    retry (re-validate against the current vault, write what now passes), then
    with steer=True hand each bundle's still-failing ops to the escalation
    model in a bounded repair loop per bundle, steered by the per-op
    ``rejection_reason`` stamps and the validator's live verdicts.
    """
    from silica.kernel.recall.deferred import get_deferred_store

    store = get_deferred_store()
    bundles = store.list_all()
    if limit:
        bundles = bundles[:limit]
    swept: list[dict[str, Any]] = []
    for b in bundles:
        h = b["content_hash"]
        res = silica_deferred_retry(h)
        row: dict[str, Any] = {
            "content_hash": h[:8],
            "written": res.get("written", 0),
            "still_deferred": res.get("still_deferred", 0),
            "cleared": res.get("bundle_cleared", False),
        }
        if res.get("error"):
            row["error"] = res["error"]
        if steer and row["still_deferred"]:
            row["steer"] = _steer_bundle(h)
        swept.append(row)
    return {
        "bundles": len(swept),
        "written": sum(r["written"] for r in swept)
        + sum(r.get("steer", {}).get("written", 0) for r in swept),
        "still_deferred": sum(r["still_deferred"] for r in swept)
        - sum(r.get("steer", {}).get("written", 0) for r in swept),
        "results": swept,
    }


def _steer_bundle(content_hash: str) -> dict[str, Any]:
    """A bounded repair loop over a bundle's still-failing ops.

    Spec: docs/specs/anneal-steer-loop.md; ADR-0031 carries the decision
    against the one-shot and the rejected alternatives (forced single tool
    call; generalizing the seam to every FSM state). The one-shot asked the
    escalation model for a perfect JSON array in free text and threw the
    validator's verdicts away: measured 2026-08-23, 1 of 24 ops recovered,
    with the prose plan eating the parse and a stray ===SILICA-BODY=== marker
    landing in a body as the next run's rejection. Now the model corrects ops
    by calling submit_repaired_ops, whose result carries the per-op verdicts
    (PDDL-INSTRUCT: the verdict is the feedback), and iterates inside the same
    turn. Ops leave the bundle only by being written, so written is the op
    count before minus after, and partial progress survives a dead loop.
    """
    import os

    import orjson as _orjson

    from silica.agent.constraints import AgentConstraints
    from silica.agent.loop import run_agent
    from silica.config import CONFIG
    from silica.kernel.recall.deferred import get_deferred_store

    store = get_deferred_store()
    bundle = store.get(content_hash)
    if not bundle:
        return {"status": "gone"}
    ops = [o for o in bundle.get("rejected_ops", []) if isinstance(o, dict)]
    if not ops:
        return {"status": "empty"}
    target_dir = bundle.get("target_dir", "")
    hub = bundle.get("hub")
    payloads = bundle.get("payloads") or []
    # The heading gate downstream only admits headings named in the payloads,
    # but the model never saw that list — it re-conceptualized freely and lost
    # the whole retry to mechanical rejections (17 of 55 deferrals on the
    # 2026-08-05 run were RETRY-phase 'Heading not present in payload concepts').
    allowed_headings = sorted({
        c.get("name")
        for p in payloads if isinstance(p, dict)
        for b in p.get("batches", [])
        for c in b.get("concepts", [])
        if c.get("name")
    })
    file_reasons = bundle.get("rejection_reasons", {})
    feedback = [
        {
            "op": o,
            "rejected_because": o.get("rejection_reason")
            or file_reasons.get(o.get("path") or o.get("heading") or "?", "unknown"),
        }
        for o in ops
    ]
    hub_line = f"\nHUB: {hub}" if hub else ""
    headings_line = (
        "\nALLOWED HEADINGS (every op's \"heading\" MUST be one of these, "
        "verbatim — never invent a new one):\n"
        + "\n".join(f"- {h}" for h in allowed_headings) + "\n"
    ) if allowed_headings else ""
    prompt = (
        "You are repairing note-write operations that a validation gate rejected.\n"
        f"TARGET_DIR: {target_dir}{hub_line}{headings_line}\n"
        "Each op below is echoed with the exact reason it was rejected. Fix ONLY\n"
        "what the reason requires — keep the content otherwise identical.\n"
        "Submit corrections by calling submit_repaired_ops with content_hash\n"
        f'"{content_hash}" and the corrected ops in the same op schema. Bodies\n'
        "are plain strings: real line breaks, single backslashes, never\n"
        "double-escaped.\n"
        "The result lists per-op verdicts: ops under \"written\" are done and\n"
        "must never be resubmitted; correct the ops under \"rejected\" and\n"
        "resubmit only those. When nothing is left, or a remaining op is\n"
        "unfixable, reply with a one-line summary instead of a call.\n\nREJECTED OPS:\n"
        + _orjson.dumps(feedback, option=_orjson.OPT_INDENT_2).decode()
    )
    before = len(ops)
    error = None
    summary = ""
    try:
        summary = run_agent(
            [{"role": "user", "content": prompt}],
            # Reproduces get_provider(role="escalation")'s fallback chain: the
            # field is already ensure_prefix-ed to the litellm string call_llm
            # resolves endpoints from, unset falls back to the router model.
            model=CONFIG.distill_escalation_model or CONFIG.model,
            constraints=AgentConstraints(
                tools=("submit_repaired_ops",),
                # One iteration = one LLM call, and a run that hits the cap
                # pays one extra tool-less landing call, so 4 caps the spend
                # at 5 calls per bundle (vs 1 for the old one-shot) — and only
                # for bundles that already failed the mechanical pass.
                max_iterations=int(os.getenv("ANNEAL_STEER_ITERATIONS", "4")),
            ),
        )
    except Exception as e:
        # Endpoint failures (broken tool-calling included) land here; whatever
        # earlier iterations already wrote is on disk and counted below, the
        # rest of the bundle stays parked and /anneal --steer is rerunnable.
        error = str(e)[:200]
    # Same isinstance filter as `before`: a corrupt non-dict entry must not
    # skew the delta (a negative `written` would even read as committed,
    # since any non-zero int is truthy).
    after = len([
        o for o in (store.get(content_hash) or {}).get("rejected_ops", [])
        if isinstance(o, dict)
    ])
    written = before - after
    row: dict[str, Any] = {
        "status": "error" if error else ("committed" if written else "no_fix"),
        "written": written,
        "still_rejected": after,
    }
    # The model's closing report ("op X is unfixable because...") is the only
    # trace of WHY ops stayed deferred; "(silica: ...)" sentinels mean there
    # was no real summary (cancelled, or a cap landing that said nothing).
    if summary and not summary.startswith("(silica:"):
        row["summary"] = summary[:300]
    if error:
        row["error"] = error
    return row
