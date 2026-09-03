# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Vault write executor — the single authorised path for applying Ops.

Every write in Silica (FSM bulk write, deferred retry, interactive patch)
funnels through here. The per-op-type logic lives in small helpers; `execute_one`
is the shared single-op entry point and `execute_operations` is the batch loop
that aggregates results into a BulkResult.

Keeping one executor means any future change to how a write reaches disk
(e.g. atomic file writes at the DRIVER level) is inherited by every caller.

Post-write verify (falsifiable gate): every successful write/overwrite/patch/
delete dispatch is followed by a read-back through the DRIVER (_verify_landed /
_verify_deleted) that is compared against the body actually composed for the
DRIVER call — never raw op.content. `success: True` used to mean only "the
DRIVER call didn't raise"; a mismatch here raises VerifyMismatchError, which
the existing exception handling in execute_one/execute_operations already
turns into a failed op — no new failure machinery, just a stricter definition
of success.

VerifyMismatchError is a distinct type (not a plain ValueError) because it
carries a different fact than every other execute_one failure: the DRIVER
call already happened and something landed on disk (possibly corrupted).
Every other failure (missing params, a driver error raised before/without a
successful write) still means "nothing landed" — commit_note_atomic relies on
this distinction to decide whether a revert is needed.
"""
from __future__ import annotations

from silica.driver import DRIVER
from silica.kernel.write import templates
from silica.kernel.write.merge import three_way_merge
from silica.kernel.write.notetype import stamp_type
from silica.kernel.write.ops import Op, OpType, FailedOp, BulkResult


class VerifyMismatchError(RuntimeError):
    """Raised by _verify_landed/_verify_deleted: the write/delete already hit
    the DRIVER and the post-write read-back proves it didn't land as intended
    (or, for delete, didn't land at all). Unlike other execute_one failures,
    this means disk state may have changed — see commit_note_atomic."""


def _verify_landed(op: Op, path: str, intended: str | None) -> str | None:
    """Falsifiable gate: re-read from the DRIVER and compare. None = ok, str = error.

    `intended` is the final body composed by the caller (post frontmatter/merge/
    patch enrichment — write, overwrite AND patch all pass their exact composed
    body) — never raw op.content or op.snippet — so this is a check against
    exactly what was handed to the DRIVER.

    STOPGAP (deliberate, ceiling below): comparison strips edge whitespace
    before comparing. The tolerance dates from the deleted cli backend, whose
    read channel stripped stdout edges; fs reads bytes verbatim and ws returns
    Obsidian's cachedRead. Interior content still compares byte-exact, so real
    corruption (e.g. the 2026-06-30 backslash-doubling bug) is still caught.
    Ceiling: verify ws edge-faithfulness live, then go byte-exact — do not
    widen the tolerance further.
    """
    try:
        landed = DRIVER.read_note(path).content or ""
    except RuntimeError as e:
        return f"post-write verify: read-back failed: {e}"
    # The DRIVER stamps the OKF `type` when absent (notetype.stamp_type at the
    # create/overwrite seam), so a body composed without one lands with one.
    # Normalising BOTH sides through that same pure, idempotent function keeps
    # the comparison byte-exact without assuming which side already carries the
    # field — a real altered payload still differs, and one that changed a
    # derivation signal differs in the stamp too.
    if intended is not None and stamp_type(path, landed).strip() != stamp_type(path, intended).strip():
        return "post-write verify: content mismatch (backend altered payload)"
    return None


def _is_note_absent(e: Exception) -> bool:
    """True only for a backend's genuine "note not found" read failure.

    fs_backend's shape is verified: "File not found: {path}". ws_backend's is
    NOT — it re-raises the Obsidian plugin's reply verbatim, the plugin is not
    in this tree, and the only evidence for the wording is a test stub that
    raises the string the test itself chose. A reply like "Note not found: X"
    would miss this test, and the two callers want opposite biases when it does:
    _execute_overwrite fails safe (the op defers), while _verify_deleted reads
    the miss as "inconclusive" and reverts a delete that actually succeeded.
    Widening the substring would trade the second bug for the first, so the
    durable fix is to type the not-found at the ws boundary (raise a dedicated
    exception in ws_backend.read_note) and test the type here, once the live
    plugin's exact reply is confirmed.

    A dead read channel (bridge down, socket error) also raises RuntimeError but
    with a different shape and must NOT be read as "the note isn't there".
    """
    msg = str(e).lower()
    return isinstance(e, RuntimeError) and "file" in msg and "not found" in msg


def _verify_deleted(path: str) -> str | None:
    """Falsifiable gate for delete: read-back must now fail (existence negated).

    Only a genuine "note not found" style RuntimeError counts as confirmation.
    """
    try:
        DRIVER.read_note(path)
    except RuntimeError as e:
        if _is_note_absent(e):
            return None
        return f"post-write verify: delete check inconclusive: {e}"
    return "post-write verify: note still present after delete"


def render_write(op: Op, snippet: str | None = None) -> str:
    """The note a write op lands as, before any driver call.

    Shared with the expand worker, which re-authors a landed note in place
    (overwrite) and must produce byte-for-byte what a fresh write would.
    """
    body = op.snippet or "" if snippet is None else snippet
    fields = templates.prepare_fields(
        title=op.title or op.heading,
        body=body,
        hub=op.hub,
        tags=op.tags,
        related=op.related,
        parent=op.parent,
    )
    content = templates.render_note(templates.resolve_template(), fields)
    content = templates.ensure_system_floor(content)
    content = templates.stamp_sources(content, op.source_basename)
    from silica.kernel.write.provenance import citation_of
    content = templates.stamp_citation(content, citation_of(op.source_basename))
    if op.review:
        # Soft-gate verdict, on the note itself: the one place the reader
        # opens. The run ledger has the same row, but nobody opens a ledger
        # after a "Success" (the 2026-08-20 coverage-line finding).
        content = templates.upsert_props(content, {"review": op.review})
    if op.section:
        # Its own key, not a `Lezione 11.md#Margine` entry in `sources:`: four
        # readers (note_authored_by, already_distilled, citation_of, the MOC)
        # compare sources entries to a basename by equality.
        content = templates.upsert_props(content, {"section": op.section})
    return content


def _execute_write(op: Op, path: str) -> dict:
    """Create a new note from the resolved template. Requires heading + hub."""
    heading = op.heading
    snippet = op.snippet or ""
    hub = op.hub

    if not heading or not hub:
        raise ValueError("Missing 'heading' or 'hub' parameter for write operation")

    # Claim stamp under the H1. Only when there is a body to stamp: prepending
    # to an empty snippet would make it truthy and swallow prepare_fields'
    # "(to be expanded)" placeholder. op.snippet stays untouched so the MOC
    # bullet (hub_desc) still reads the first line of real prose.
    if op.valid_from and snippet.strip():
        from silica.kernel.write.contested import stamp
        rendered = stamp(valid_from=op.valid_from)
        if rendered:
            snippet = f"{rendered}\n\n{snippet.lstrip()}"

    content = render_write(op, snippet)
    DRIVER.create(path, content)
    err = _verify_landed(op, path, content)
    if err:
        raise VerifyMismatchError(err)
    return {"path": path, "op": "write", "success": True}


def _execute_patch(op: Op, path: str) -> dict:
    """Append a distilled snippet to an existing note. Requires heading + snippet + source_basename."""
    heading = op.heading
    snippet = op.snippet
    source_basename = op.source_basename

    if not heading or not snippet or not source_basename:
        raise ValueError(
            "Missing 'heading', 'snippet', or 'source_basename' for patch operation"
        )

    from silica.kernel.vault_manifest import seed_mirror_copy
    seed_mirror_copy(path)
    try:
        nc = DRIVER.read_note(path)
    except RuntimeError as e:
        # Preserve the historical message so callers/tests can match on it.
        raise ValueError(f"Cannot patch; {e}") from e

    # Idempotent re-injection: skip the append when this note already carries
    # this source's content (deterministic, no LLM). Two signals:
    #   1. the exact provenance block is present (a prior PATCH from this source);
    #   2. the provenance ledger records this source as the note's author (a
    #      prior WRITE lays down no block, so #1 can't see it) — this is the
    #      re-ingest case: an edited source re-distills unchanged concepts that
    #      collision-match their own prior notes and would each gain a redundant
    #      "Additional notes: <heading> (from <source>)" block.
    # Either way the hub-link repair must still land, or the post-write lint
    # fails this op on every retry (2026-07-17 nucleate run: notes patched by
    # an interrupted run could never be re-patched).
    # read_records memoizes the parsed ledger on (path, mtime, size), so this
    # stays one parse per run rather than one per patch op.
    from silica.kernel.write.provenance import note_authored_by
    # The ledger leg needs the note to still SAY the heading: the ledger is
    # never pruned (revert rewinds bodies, not records), so on its own it kept
    # claiming authorship after the section was hand-deleted or the note
    # rewritten, and the enrichment was skipped as a "duplicate" in silence.
    already_present = templates.block_present(nc.content, heading, source_basename) or (
        bool(source_basename)
        and heading.casefold() in nc.content.casefold()
        and note_authored_by(path, source_basename)
    )
    if already_present:
        # Same lint floor as the real-patch path below: a safe-mode skip lands
        # on a freshly-seeded mirror copy of a human note (no `AI` key), and an
        # unstamped copy fails the chunk LINT gate on every later touch.
        # `partial`: the agent contributed a hub link, not the body — the note
        # stays human-tier in a contest (contested.reliability_tier).
        repaired = templates.ensure_ai_flag(
            templates.ensure_hub_link(nc.content, op.hub), value="partial")
        if repaired != nc.content:
            DRIVER.overwrite(path, repaired)
        return {"path": path, "op": "patch", "success": True, "skipped": "duplicate"}

    new_content = templates.patch_snippet(
        heading=heading,
        snippet=snippet,
        source_basename=source_basename,
        hub=op.hub,
        existing_content=nc.content,
        valid_from=op.valid_from,
    )
    if op.contested_by:
        from silica.kernel.write.contested import mark_contested
        new_content = mark_contested(new_content, op.contested_by)
    # A patch appends one section to a note that was the user's: `partial`
    # says so, and the note keeps its human reliability tier. A note born
    # agent-authored already carries `AI: true` and is left untouched.
    new_content = templates.ensure_ai_flag(new_content, value="partial")
    new_content = templates.stamp_sources(new_content, source_basename)
    from silica.kernel.write.provenance import citation_of
    new_content = templates.stamp_citation(new_content, citation_of(source_basename))
    DRIVER.overwrite(path, new_content)
    err = _verify_landed(op, path, new_content)
    if err:
        raise VerifyMismatchError(err)
    return {"path": path, "op": "patch", "success": True}


def _execute_overwrite(op: Op, path: str) -> dict:
    """Rewrite a whole note. Requires content.

    When the op carries base_content (the note as it was at op-build time) and
    the note on disk has changed since, the incoming content is written with a
    conflict callout prepended instead of stomping silently (ADR-0007 soft
    failure; charter UC6).
    """
    content = op.content
    if content is None:
        raise ValueError("Missing 'content' for overwrite operation")

    had_conflict = False
    prior: str | None = None
    try:
        prior = DRIVER.read_note(path).content
    except Exception as e:
        # Only a genuine "not found" means "no prior note". Every other read
        # failure (ws bridge down, OSError, a decode error) must not be read as
        # absent: prior=None disarms the conflict gate AND makes
        # ensure_system_floor build a minimal frontmatter block instead of
        # re-injecting the user's tags/aliases — the note is stomped on a read
        # channel fault. Let it propagate so the op fails and is deferred.
        if not _is_note_absent(e):
            raise
    if op.base_content is not None:
        content, had_conflict = three_way_merge(op.base_content, prior, content)

    content = templates.ensure_system_floor(content, prior=prior)
    DRIVER.overwrite(path, content)
    err = _verify_landed(op, path, content)
    if err:
        raise VerifyMismatchError(err)
    result = {"path": path, "op": "overwrite", "success": True}
    if had_conflict:
        result["conflict"] = True
    return result


def _execute_delete(op: Op, path: str) -> dict:
    """Delete a note."""
    DRIVER.delete(path)
    err = _verify_deleted(path)
    if err:
        raise VerifyMismatchError(err)
    return {"path": path, "op": "delete", "success": True}


_DISPATCH = {
    OpType.write: _execute_write,
    OpType.patch: _execute_patch,
    OpType.overwrite: _execute_overwrite,
    OpType.delete: _execute_delete,
}


def execute_one(op: Op) -> dict:
    """Execute a single Op and return its success-result dict.

    Shared single-op entry point reused by execute_operations and by the
    interactive silica_patch_note tool. Raises ValueError on missing required
    params, a missing path, or an unknown op type — callers decide how to record
    the failure. Skip ops are a no-op success.
    """
    op_type = op.op

    if op_type == OpType.skip:
        return {"op": "skip", "success": True}

    path = op.touched_ref()
    if not path:
        raise ValueError("Missing 'path' parameter")

    handler = _DISPATCH.get(op_type)
    if handler is None:
        raise ValueError(f"Unknown operation type: {op_type}")

    return handler(op, path)


def execute_operations(ops: list[Op]) -> BulkResult:
    """Apply a batch of Ops, aggregating outcomes into a BulkResult.

    Each op is executed via execute_one; failures become FailedOp entries
    without aborting the batch. The return shape (ok/failed/results/total/
    successful) is the contract consumed by silica_bulk_write and the FSM.
    """
    results: list[dict] = []
    failed_ops: list[FailedOp] = []
    success_count = 0

    for idx, op in enumerate(ops):
        op_type = op.op
        path = op.touched_ref() or ""
        try:
            res = execute_one(op)
        except Exception as e:
            failed_ops.append(
                FailedOp(index=idx, path=path, op=op_type.value, error=str(e))
            )
            failure: dict = {"index": idx, "success": False, "error": str(e)}
            if path:
                failure["path"] = path
            results.append(failure)
            continue

        success_count += 1
        results.append({"index": idx, **res})

    return BulkResult(
        ok=len(failed_ops) == 0,
        failed=failed_ops,
        results=results,
        total=len(ops),
        successful=success_count,
    )
