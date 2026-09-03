# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Canonical Op schema for the Silica pipeline (ADR-0007 / Addendum C1).

This is the single source of truth imported by sanitize, validate, snapshot,
bulk, and lint. No module defines its own op structure locally.

Key invariant: touched_ref() returns op.path — NEVER a field named 'name'
(which does not exist). This closes B1 at the root.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class OpType(str, Enum):
    write = "write"          # create new note (path MUST NOT exist)
    patch = "patch"          # enrich existing note (path MUST exist)
    overwrite = "overwrite"  # rewrite whole note, preserve identity/history
    delete = "delete"        # only via wrapped tool + confirm
    skip = "skip"            # explicit no-op (excluded from gate denominator)
    move = "move"            # relocate note (organize pipeline); DRIVER.move() updates wikilinks


class Op(BaseModel):
    op: OpType
    heading: str                        # concept name; provenance key in payload
    source_basename: str                # inbox filename (basename) this op derives from
    path: str | None = None             # vault-relative path; required for write/patch/overwrite/delete
    title: str | None = None            # clean note filename stem when heading is structurally compound
    snippet: str = ""                   # distilled body (write / patch)
    hub: str | None = None              # [[Hub]] link required for write ops
    content: str | None = None          # full body (overwrite only)
    base_content: str | None = None     # overwrite only: note content at op-build time; enables 3-way conflict detection
    tags: list[str] | None = None
    related: list[str] | None = None
    concepts: list[str] | None = None  # #9: normalized concept phrases for the co-occurrence graph
    reason: str | None = None           # skip reason / rejection note
    review: str | None = None           # soft-gate verdict: the op landed anyway, this says what to look at (validate.py)
    section: str | None = None          # source heading the note comes from (outline lane); frontmatter `section:`
    linked_axis: str | None = None      # thematic axis this concept belongs to (Layer 2)
    parent: str | None = None           # specific parent note (≠ run hub); None → falls back to hub
    contested_by: str | None = None     # patch only: contradiction ref → mark_contested on the target
    valid_from: str | None = None       # write/patch: source event clock, stamped on the claim; None → no stamp
    from_path: str | None = None        # move op: current vault-relative path
    to_path: str | None = None          # move op: destination vault-relative path

    @model_validator(mode="after")
    def validate_path_required(self) -> Op:
        if self.op in (OpType.write, OpType.patch, OpType.overwrite, OpType.delete):
            if not self.path:
                raise ValueError(f"path required for op '{self.op.value}'")
        if self.op == OpType.move:
            if not self.from_path or not self.to_path:
                raise ValueError("move op requires both 'from_path' and 'to_path'")
        # Trust boundary for model-supplied ops (2026-07-08 merge-run corruption):
        # 1. Note paths must carry the .md extension — the read channel appends
        #    it but the write channel resolves verbatim, so a bare path makes
        #    getFileByPath miss and the cli fallback fabricates a phantom
        #    extensionless file next to the real note.
        # 2. NUL bytes can never occur in markdown; they arrive as model-emitted
        #    backslash-u0000 JSON escapes and poison the write channel (subprocess
        #    rejects NUL in argv). Strip them here so every entry point inherits it.
        for f in ("path", "from_path", "to_path"):
            v = getattr(self, f)
            if v and not v.endswith(".md"):
                setattr(self, f, v + ".md")
        for f in ("content", "snippet", "base_content"):
            v = getattr(self, f)
            if v and "\x00" in v:
                setattr(self, f, v.replace("\x00", ""))
        return self

    def touched_ref(self) -> str | None:
        """The vault path touched by this op.

        This is the ONLY authorised way for lint/snapshot to derive the
        note reference from an op. Using any other field (e.g. 'name') is a
        violation — 'name' does not exist on Op (closes B1).

        For move ops, returns from_path (the note being relocated) so that
        lint/snapshot can locate the note before the move executes.
        """
        if self.op in (OpType.write, OpType.patch, OpType.overwrite, OpType.delete):
            return self.path
        if self.op == OpType.move:
            return self.from_path
        return None

    def __getitem__(self, item: str) -> Any:
        try:
            val = getattr(self, item)
            if isinstance(val, Enum):
                return val.value
            return val
        except AttributeError:
            raise KeyError(item)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            val = getattr(self, item)
            if isinstance(val, Enum):
                return val.value
            return val
        except AttributeError:
            return default


# ---------------------------------------------------------------------------
# Rollback inverse ops (ADR-0009 / Addendum C3)
# ---------------------------------------------------------------------------

class InverseOpKind(str, Enum):
    delete_created = "delete_created"       # undo a write: delete the note that was created
    restore_version = "restore_version"     # undo a patch: history:restore to prior version
    recreate_deleted = "recreate_deleted"   # undo a delete: recreate with prior content
    move_back = "move_back"                 # undo a move: DRIVER.move(to_path, from_path)


class InverseOp(BaseModel):
    kind: InverseOpKind
    path: str
    version: int | None = None            # for restore_version
    prior_content: str | None = None      # for recreate_deleted
    to_path: str | None = None            # for move_back: where the note was moved to


class FailedOp(BaseModel):
    index: int
    path: str
    op: str | None = None
    error: str

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class BulkResult(BaseModel):
    ok: bool
    failed: list[FailedOp]
    results: list[dict]
    total: int
    successful: int

    @property
    def success(self) -> bool:
        return self.ok

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        d = super().model_dump(*args, **kwargs)
        d["success"] = self.ok
        return d

    def __getitem__(self, item: str) -> Any:
        if item == "success":
            return self.ok
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        if item == "success":
            return self.ok
        return getattr(self, item, default)


class EphemeralFact(BaseModel):
    """Personal, time-bound fact routed to the episodic store, not to notes."""
    key: str
    text: str


class DistillerOutput(BaseModel):
    main_thematic_axes: list[str] = Field(default_factory=list)
    updates: list[Op]
    ephemerals: list[EphemeralFact] = Field(default_factory=list)


class StructureOp(BaseModel):
    """Wire schema for the two-pass structure call: Op minus every body field,
    so constrained decoding cannot put prose inside a JSON string. Decode-only —
    the stitched dicts are consumed through Op/parse_ops as usual. No move
    fields: the distiller never plans moves (organize pipeline does)."""
    op: OpType
    heading: str
    source_basename: str
    path: str | None = None
    title: str | None = None
    hub: str | None = None
    tags: list[str] | None = None
    related: list[str] | None = None
    concepts: list[str] | None = None
    reason: str | None = None
    linked_axis: str | None = None
    parent: str | None = None
    contested_by: str | None = None
    valid_from: str | None = None


class DistillerStructure(BaseModel):
    main_thematic_axes: list[str] = Field(default_factory=list)
    updates: list[StructureOp]
    ephemerals: list[EphemeralFact] = Field(default_factory=list)


