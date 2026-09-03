# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Code-lane agent tools — silica_document (ADR-0012), silica_code_pack.

silica_document — stage a skeleton stub from a source file (ADR-0012).

Thin agent-facing wrapper over the code SourceAdapter (ADR-0014): guards,
sanitization and stub assembly live in silica/sources/code.py. Writes ONLY
to Inbox/ — RBAC inbox-write, never the vault. No LLM call here: the
curation pipeline refines Inbox stubs.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from silica.tools import tool


class DocumentArgs(BaseModel):
    path: str = Field(description="Repo-relative path to the source file to document")


@tool(DocumentArgs, cls="composed")
def silica_document(path: str) -> dict:
    """Extract a shallow AST skeleton from a source code file and stage it as a
    documentation stub in Inbox/ (never directly in the vault). Sets
    documents:/code_ref frontmatter for staleness tracking; source-derived text
    is sanitized and fenced. Nucleate the stub afterwards with silica_run_injector."""
    from silica.driver import DRIVER
    from silica.sources.code import CODE

    try:
        item = CODE.read(path)
        item.meta["stage_to_inbox"] = True  # RBAC inbox-write, never the vault
        stub = CODE.to_stub(item)
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    DRIVER.upsert(stub.note_path, stub.body)  # re-running on the same file refreshes the stub
    return {
        "status": "ok",
        "note_path": stub.note_path,
        "code_ref": item.meta.get("code_ref", ""),
        "skeleton": item.meta.get("language") is not None,
    }


class CodePackArgs(BaseModel):
    target: str = Field(
        description="Repo-relative source path, optionally narrowed with "
                    "'#Class' or '#Class.member'"
    )
    budget_chars: int = Field(
        default=24000,
        description="Character budget for the whole pack. The target is always "
                    "served, sections fill what is left.",
    )
    sections: list[str] = Field(
        default_factory=list,
        description="Which sections to emit besides the target: any of "
                    "'hierarchy', 'neighborhood', 'external', 'importers'. "
                    "Empty = all. Pass ['importers'] on a second pack in the "
                    "same package so the neighbourhood outline is not repaid.",
    )


def _stamp_code_pack_use() -> None:
    """Kill-gate evidence for codepack (kill by 2026-10-28 if unused): one
    timestamp line per invocation, whatever the surface reached it (MCP or a
    summoned chat turn), so the kill-date check reads usage data instead of
    chat silence. Grep ~/.silica/index/*/codepack_usage.log at the date.
    Best-effort: never fails the tool."""
    try:
        import datetime as _dt

        from silica.config import CONFIG
        from silica.kernel.recall.paths import index_dir_for

        p = index_dir_for(CONFIG.vault_path) / "codepack_usage.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(_dt.datetime.now().isoformat(timespec="seconds") + "\n")
    except Exception:
        pass


@tool(CodePackArgs, cls="composed")
def silica_code_pack(target: str, budget_chars: int = 24000,
                     sections: list[str] | None = None) -> dict:
    """Deterministic context pack for one source file inside a character
    budget: the target plus its supertypes, extenders, the visible signatures
    it actually names, external dependencies, and importers. A closure, not a
    search — same repo state, same bytes. Use before rewriting or porting a
    file, instead of ten greps.

    `target` is repo-relative, optionally narrowed with '#Class' or
    '#Class.member'. `target_mode`: "verbatim" = whole file; "symbol" = that
    declaration whole, rest as outline; "outline" = signatures only (over
    budget) — check `truncated` before treating the target as complete.
    `dropped`: `note: ...` entries are degrades (not fetchable); other entries
    are `<section>: <label>` items that did not fit and can be requested by
    label. Section counts are true repo-wide totals.
    """
    from silica.config import CONFIG
    from silica.kernel.code import codepack

    vault = str(getattr(CONFIG, "vault_path", "") or "").strip()
    if not vault:
        return {"status": "error", "message": "no vault configured"}
    _stamp_code_pack_use()
    try:
        pack = codepack.code_pack(vault, target, budget_chars, sections=sections or None)
    except (ValueError, OSError) as e:
        # OSError: loading the code graph can write its store, and an
        # unwritable store is a tool-level error, not a crash of the caller.
        return {"status": "error", "message": str(e)}
    return {"status": "ok", **pack}


class ImpactArgs(BaseModel):
    range_spec: str = Field(
        default="",
        description=(
            "Git range or ref (e.g. 'HEAD~3..HEAD', 'main..HEAD', one SHA = "
            "that ref vs the working tree). Empty = uncommitted changes vs HEAD."
        ),
    )


_IMPACT_CAP = 100  # entries; a diff wider than this is a rewrite, not a change


@tool(ImpactArgs, cls="composed")
def silica_impact(range_spec: str = "") -> dict:
    """Which notes a code change touches: changed source files (working tree
    vs HEAD by default, or a git range) classified cosmetic/structural, each
    with the notes documenting it and the notes of its 1-hop import
    neighbors, sorted structural-first by fan-in. The blast-radius call to
    make before and after editing code the vault documents. Deterministic,
    no LLM; a vault outside a git repo answers status "no_repo"."""
    from silica.config import CONFIG
    from silica.kernel.code.codegraph import compute_impact

    vault = str(getattr(CONFIG, "vault_path", "") or "").strip()
    if not vault:
        return {"status": "error", "message": "no vault configured"}
    try:
        entries = compute_impact(vault, range_spec.strip() or None)
    except (ValueError, OSError) as e:
        # OSError for the same reason as code_pack: the graph load may write
        # its derived store, and an unwritable store is a tool-level error.
        return {"status": "error", "message": str(e)}
    if entries is None:
        return {
            "status": "no_repo",
            "message": "vault is not inside a git repository; the code lane is off",
        }
    return {
        "status": "ok",
        "total": len(entries),
        "entries": [
            {
                "path": e.path,
                "change_level": e.change_level,
                "details": list(e.details),
                "fan_in": e.fan_in,
                "notes": list(e.notes),
                "neighbor_notes": list(e.neighbor_notes),
            }
            for e in entries[:_IMPACT_CAP]
        ],
        # Never a silent cap (same discipline as the tabular lane).
        "truncated": len(entries) > _IMPACT_CAP,
    }
