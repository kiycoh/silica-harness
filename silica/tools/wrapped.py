# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Wrapped tools — L0 atomics with domain invariants (Golden Rules) baked in.

From SILICA.md §4.4:
  Wrapped tools enforce invariants in the toolset, not in the system prompt.
  - silica_move always updates wikilinks (graph-safe).
  - silica_delete refuses to delete if it loses density.

C3 rollback strategy (ADR-0009):
  - write ops   → InverseOp(delete_created, path)
  - patch/overwrite ops → InverseOp(restore_version, path, version=N)
  - Txn.inverses: list[InverseOp] replaces the ad-hoc created_paths field.

C3 clarification: silica_snapshot DOES NOT leak _txn_obj through the tool
registry. The orchestrator holds the Txn directly (it calls snapshot
programmatically and receives the return value). The tool is kept for CLI
discoverability only — the FSM bypasses it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.driver.base import NoteRef, Txn
from silica.kernel.write.ops import InverseOp, InverseOpKind, OpType, Op
from silica.kernel.write.ops_io import parse_ops, load_ops
from silica.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# silica_move
# ---------------------------------------------------------------------------

class MoveArgs(BaseModel):
    ref: str = Field(description="Name or path of the note to move")
    to: str = Field(description="Destination path")

@tool(MoveArgs, cls="wrapped", collapse="eager")
def silica_move(ref: str, to: str) -> dict[str, Any]:
    """Move/rename a note safely. Obsidian updates all wikilinks (graph-safe)."""
    try:
        DRIVER.move(ref, to)
        return {"success": True, "moved": ref, "to": to}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# silica_delete
# ---------------------------------------------------------------------------

class DeleteArgs(BaseModel):
    ref: str = Field(description="Name or path of the note to delete")
    confirm: bool = Field(default=False, description="Explicit confirmation for density loss")

@tool(DeleteArgs, cls="wrapped", collapse="eager")
def silica_delete(ref: str, confirm: bool = False) -> dict[str, Any]:
    """Delete a note. Requires confirmation if density is lost."""
    if not confirm:
        return {"error": "Anti-deletion policy: must pass confirm=True to acknowledge no density is lost."}

    try:
        DRIVER.delete(ref)
        return {"success": True, "deleted": ref}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# build_txn — internal helper (not a tool)
# ---------------------------------------------------------------------------

def build_txn(ops_data: list[Op] | list[dict]) -> Txn:
    """Build a Txn with InverseOp entries before WRITE executes.

    Rollback strategies (C3):
      write   → delete_created(path)         — note didn't exist; undo = delete
      patch / overwrite → restore_version(path, prior_content=<full body>)
                          — note existed; undo = overwrite with saved content.
                          prior_content is the primary rollback path; version is
                          kept as a best-effort hint for backends that support it.
      delete  → recreate_deleted(path, prior_content=<full body>)
    """
    from silica.kernel.vault_manifest import seed_mirror_copy

    ops = parse_ops(ops_data)
    patch_refs: list[NoteRef] = []
    prior_contents: dict[str, str | None] = {}
    seeded_here: set[str] = set()
    inverses: list[InverseOp] = []

    for op in ops:
        op_type = op.op
        path = op.touched_ref()
        if not path or op_type == OpType.skip:
            continue

        if op_type == OpType.write:
            # write's contract is "path MUST NOT exist", but nothing enforces it
            # at the FS boundary (DRIVER.create overwrites verbatim). If the path
            # already holds a note, undo must RESTORE it, not delete it — else
            # /revert turns an accidental clobber into data loss. Snapshot the
            # prior body and pick the inverse accordingly.
            try:
                prior = DRIVER.read_note(path).content
            except Exception:
                prior = None
            if prior:
                inverses.append(InverseOp(
                    kind=InverseOpKind.restore_version, path=path, prior_content=prior,
                ))
            else:
                inverses.append(InverseOp(kind=InverseOpKind.delete_created, path=path))
        elif op_type in (OpType.patch, OpType.overwrite):
            name = path.rsplit("/", 1)[-1].removesuffix(".md")
            ref = NoteRef(name=name, path=path)
            patch_refs.append(ref)
            # Read current content now (before WRITE) for content-based rollback.
            # This is more reliable than history:restore whose version numbering
            # shifts after each new write (position 1 becomes position 2, etc.).
            try:
                nc = DRIVER.read_note(ref)
                prior_contents[path] = nc.content
            except Exception as e:
                # Safe mode: the patch lands on the note's mirror copy, which
                # _execute_patch seeds at WRITE time — after this snapshot and
                # after the pre-write graph. Seed it here instead. Born later,
                # the copy gave every same-folder [[link]] a nearer target
                # between the two graph snapshots, and the original's lost
                # backlinks read as vandalism to the regression gate; born now,
                # both worlds contain it and the diff is clean. It also gives
                # the rollback a body to restore instead of skipping the op.
                seed_mirror_copy(path)
                try:
                    prior_contents[path] = DRIVER.read_note(ref).content
                    seeded_here.add(path)
                except Exception:
                    logger.warning("build_txn: could not read prior content for %s: %s", path, e)
                    prior_contents[path] = None
        elif op_type == OpType.delete:
            name = path.rsplit("/", 1)[-1].removesuffix(".md")
            ref = NoteRef(name=name, path=path)
            try:
                nc = DRIVER.read_note(ref)
                prior_content = nc.content
            except Exception as e:
                # Can't snapshot the body we're about to delete → /revert will be
                # unable to recreate it. Surface it now, not silently at revert.
                logger.warning(
                    "build_txn: cannot snapshot %s before delete; /revert won't "
                    "recreate it: %s", path, e)
                prior_content = None
            inverses.append(InverseOp(
                kind=InverseOpKind.recreate_deleted,
                path=path,
                prior_content=prior_content
            ))

    # Txn id comes from the driver; rollback is content-based (prior_content).
    base_txn = DRIVER.snapshot_versions(patch_refs)

    for ref in patch_refs:
        key = ref.path or ref.name
        if key in seeded_here:
            # A copy this txn brought into being is undone by removing it, not
            # by restoring a body the vault never had before the chunk. The
            # created_paths below picks it up too, so the graph gate grants it
            # the same forward-reference exemption as any other new note.
            inverses.append(InverseOp(kind=InverseOpKind.delete_created, path=ref.path))
            continue
        inverses.append(InverseOp(
            kind=InverseOpKind.restore_version,
            path=ref.path,
            prior_content=prior_contents.get(key),
        ))

    created_paths = [
        inv.path for inv in inverses
        if inv.kind == InverseOpKind.delete_created
    ]
    txn = Txn(
        id=base_txn.id,
        refs=patch_refs,
        created_paths=created_paths,
        inverses=inverses,
    )
    return txn


# ---------------------------------------------------------------------------
# silica_snapshot
# ---------------------------------------------------------------------------

class SnapshotArgs(BaseModel):
    ops_json_path: str = Field(description="Path to validated operations JSON to snapshot before writing")

@tool(SnapshotArgs, cls="wrapped", collapse="eager", internal=True)
def silica_snapshot(ops_json_path: str) -> dict[str, Any]:
    """Snapshot the current state of notes before they are modified.

    Builds InverseOp entries (C3):
      - write ops   → delete_created(path)     — rollback by deleting the new note
      - patch / overwrite ops → restore_version(path, N) — rollback via history:restore

    The orchestrator holds the returned Txn object directly.
    The tool result is JSON-serialisable (no _txn_obj leak per addendum note).
    """
    try:
        ops = load_ops(ops_json_path)
    except Exception as e:
        return {"error": f"Failed to load operations for snapshot: {e}"}

    try:
        txn = build_txn(ops)
    except Exception as e:
        return {"error": f"Snapshot failed: {e}"}

    return {
        "success": True,
        "txn_id": txn.id,
        "refs": [r.name for r in txn.refs],
        "created_paths": txn.created_paths,
        "inverses": txn.inverses_serialized,
        # _txn_obj intentionally absent — orchestrator calls build_txn() directly
    }


# ---------------------------------------------------------------------------
# silica_restore (real tool, usable by YAML recipe engine at S3.3)
# ---------------------------------------------------------------------------

class RestoreArgs(BaseModel):
    txn_id: str = Field(description="Transaction ID to restore (for audit log only)")
    inverses: list[dict] = Field(description="InverseOp list from silica_snapshot result")

@tool(RestoreArgs, cls="wrapped", collapse="eager", internal=True)
def silica_restore(txn_id: str, inverses: list[dict]) -> dict[str, Any]:
    """Apply InverseOp list to rollback a transaction.

    Accepts the 'inverses' list produced by silica_snapshot — fully
    JSON-serialisable, no hidden Python objects.
    """
    errors: list[str] = []
    applied: list[str] = []

    for raw in inverses:
        try:
            inv = InverseOp(**raw)
        except Exception as e:
            errors.append(f"Invalid InverseOp {raw}: {e}")
            continue

        path = inv.path
        try:
            if inv.kind == InverseOpKind.delete_created:
                try:
                    DRIVER.delete(path)
                    applied.append(f"deleted_created:{path}")
                except Exception as e:
                    err_str = str(e).lower()
                    if "not found" in err_str or "no such file" in err_str:
                        applied.append(f"deleted_created:{path} (already_absent)")
                    else:
                        raise

            elif inv.kind == InverseOpKind.restore_version:
                if inv.prior_content is not None:
                    # Overwrite with captured content (reliable across backends).
                    DRIVER.overwrite(path, inv.prior_content)
                    applied.append(f"restored_content:{path}")
                else:
                    logger.warning("restore_version: no prior_content for %s — skipped", path)

            elif inv.kind == InverseOpKind.recreate_deleted:
                if inv.prior_content is not None:
                    # upsert: a re-run restore (partial rollback retry) finds the
                    # note already recreated — restoring content is still correct.
                    DRIVER.upsert(path, inv.prior_content)
                    applied.append(f"recreated_deleted:{path}")
                else:
                    errors.append(f"recreate_deleted missing prior_content for {path}")

            elif inv.kind == InverseOpKind.move_back:
                # Undo a move: send the note from where it landed back to origin.
                # if a new note now occupies `path`, DRIVER.move raises
                # and we route it to errors — same collision stance as the FSM's
                # in-run rollback; no silent overwrite.
                if inv.to_path:
                    DRIVER.move(inv.to_path, path)
                    applied.append(f"moved_back:{inv.to_path}->{path}")
                else:
                    errors.append(f"move_back missing to_path for {path}")

        except Exception as e:
            errors.append(f"Inverse op {inv.kind} on {path} failed: {e}")
            logger.error("Rollback error: %s", e)

    return {
        "success": not errors,
        "txn_id": txn_id,
        "applied": applied,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# silica_cleanup (real tool for S3.3 YAML recipe)
# ---------------------------------------------------------------------------

class CleanupArgs(BaseModel):
    inbox_file: str = Field(description="Vault-relative path of the inbox file to archive")
    done_dir: str = Field(default="", description="Destination folder; empty resolves this vault's archive root")

@tool(CleanupArgs, cls="wrapped", collapse="eager", internal=True)
def silica_cleanup(inbox_file: str, done_dir: str = "") -> dict[str, Any]:
    """Archive the inbox file under Done/ after successful pipeline completion.

    C5: Only callable from DONE state — the orchestrator enforces this.

    A source that sits OUTSIDE the write boundary is never moved: under safe
    mode the run is a preview the user merges by pasting, and consuming the
    source would mutate the one tree the boundary exists to protect. It stays
    in the inbox, which is also where a re-run needs to find it.

    The archive keeps the source's inbox folder structure — see
    `archive_path_for`, which owns that rule. `done_dir` overrides the resolved
    root; empty asks `active_done_dir`, which keeps a pre-2026-08-23 `done/`
    where it is rather than renaming it.
    """
    from silica.kernel.vault_manifest import archive_path_for, active_write_dir, within

    write_root = active_write_dir()
    if write_root and not within(inbox_file, write_root):
        return {"success": True, "skipped": inbox_file,
                "reason": f"outside the write boundary '{write_root}/'"}
    target = archive_path_for(inbox_file, done_dir)
    # The fs backend creates the destination parents itself; the Obsidian one
    # forwards to a plugin that does not, and every nested source now needs a
    # subfolder under the archive that has never existed. Make it here, once,
    # for both — an empty folder is inert if the move then fails.
    try:
        from silica.config import CONFIG

        vault = (getattr(CONFIG, "vault_path", "") or "").strip()
        if vault and "/" in target:
            (Path(vault) / target).parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.debug("CLEANUP: archive folder pre-create skipped (%s)", exc)
    try:
        DRIVER.move(inbox_file, target)
        return {"success": True, "moved": inbox_file, "to": target}
    except Exception as e:
        return {"error": str(e)}
