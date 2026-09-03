# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""MOC (map-of-content) section helpers.

Shared by the FSM's HUB_UPDATE state (router/states/write.py) and the
deferred-retry recovery path (tools/pipeline.py): the retry path must give
recovered notes the same hub-MOC membership the FSM gives in-flight ones, and
tools cannot import router states without inverting the tools←router layering.
"""
from __future__ import annotations

import re


def resolve_note_path(name: str) -> str | None:
    """Real vault-relative path of an existing note by name, searched vault-wide.

    Mirrors validate._resolve_parent's `search_names` check. Returns None when
    the note doesn't exist yet (caller falls back to `target_dir/<name>.md`).
    """
    from silica.driver import DRIVER

    name_l = name.lower()
    try:
        matches = [r for r in DRIVER.search_names(name) if r.name.lower() == name_l and r.path]
    except Exception:
        return None
    if not matches:
        return None
    # Deterministic pick on duplicate names: shallowest path, then lexical.
    return sorted(matches, key=lambda r: (r.path.count("/"), r.path.lower()))[0].path


def moc_target(name: str, target_dir: str) -> str:
    """Where a hub/parent note's MOC block must be written; "" ⇒ nowhere.

    Two corrections in one, because they compose:

    - `target_dir/<name>.md` was assumed, but a hub defaults to the BASENAME of
      target_dir and the common vault convention puts that note BESIDE its
      folder (`ML/Apprendimento supervisionato.md` for the folder of the same
      name) rather than inside it. Assuming the layout sent every hub read to a
      path the vault never had, and MOC membership was skipped for whole runs.
    - Resolution answers with the REAL note, which under safe mode is exactly
      the file the run must not touch. The landing is rebased onto the mirror
      copy; `seed_mirror_copy` gives that copy the original's content first, so
      the block lands on top of what the note already said.
    """
    from silica.kernel.recall.paths import is_inbox_path
    from silica.kernel.vault_manifest import in_write_dir, seed_mirror_copy

    resolved = resolve_note_path(name)
    # A source file shares its title with the notes distilled from it, so name
    # resolution can answer with the very lecture being ingested. The inbox is
    # staging and never a MOC target: "" rather than the target_dir fallback,
    # which would name a path nothing will ever write and log a miss that reads
    # like a failure. Under safe mode it also stopped a mirror copy of the
    # source being seeded beside the notes.
    if resolved and is_inbox_path(resolved):
        return ""
    fallback = f"{(target_dir or '').rstrip('/')}/{name}.md".replace("//", "/")
    landing = in_write_dir(resolved or fallback)
    seed_mirror_copy(landing)
    return landing


# Emitted until 2026-09. Recognised forever by merge_moc_section: a hub that
# already carries this spelling must keep merging into it, or every later chunk
# of the same source opens a second section. Never emitted again.
LEGACY_MOC_PREFIX = "## Da: "


def moc_heading(source_name: str, sample: str = "") -> str:
    """MOC section heading: '## From: {name}'.

    Vault strings are UI copy and go out in English. The language-detected
    Italian spelling this replaces flipped between chunks of ONE source when
    the sample changed (2026-09-02: `## Da:` in the hub, `## From:` in a
    spoke), so the same source ended up with two section headings. `sample`
    is kept so callers need not change.
    """
    return f"## From: {source_name}"


def merge_moc_section(content: str, heading: str, note_lines: list[str]) -> str:
    """Append note_lines to an existing MOC section or create a new one.

    When the same source file produces multiple chunks, each chunk calls
    HUB_UPDATE.  Rather than duplicating the heading, new links are appended
    inside the existing section.
    """
    if heading.startswith("## From: "):
        legacy = LEGACY_MOC_PREFIX + heading[len("## From: "):]
        if legacy + "\n" in content or legacy + "\r\n" in content:
            heading = legacy  # keep appending into the pre-2026-09 section
    if heading + "\n" in content or heading + "\r\n" in content:
        # Append new links just before the next same-level heading or end of file.
        pattern = re.compile(re.escape(heading) + r'(.*?)(?=\n##\s|\Z)', re.DOTALL)
        def _append(m: re.Match) -> str:
            return m.group(0).rstrip() + "\n" + "\n".join(note_lines) + "\n"
        return pattern.sub(_append, content, count=1)
    moc_block = f"\n{heading}\n\n" + "\n".join(note_lines) + "\n"
    return content.rstrip() + "\n" + moc_block


def _safe_cut(line: str, cap: int) -> str:
    """Largest whitespace cut <= cap that leaves every $...$ and [[...]] span
    closed, ellipsis appended; "" when no safe point exists (better a bare
    `- [[Name]]` bullet than broken markup)."""
    safe = 0
    in_math = False
    depth = 0
    for i, ch in enumerate(line[:cap + 1]):
        if ch == "$":
            in_math = not in_math
        elif line.startswith("[[", i):
            depth += 1
        elif line.startswith("]]", i) and depth:
            depth -= 1
        elif ch.isspace() and not in_math and depth == 0:
            safe = i
    out = line[:safe].rstrip(" \t,;:.-")
    return out + "…" if out else ""


def hub_desc(snippet: str, cap: int = 120) -> str:
    """First real prose line of a body for a hub bullet — cleaned, and capped
    at a SAFE boundary.

    Guards the MOC bullet from garbage like `> [!NOTE] Documento originale: ...`
    when the distiller opens a body with a fabricated callout (audit finding 3).

    The cap used to be a raw `[:cap]` slice and the edge strip ate one side of
    a bold pair: the 2026-08-21 hub carried bullets cut mid-word, mid-LaTeX
    ("$\\boldsymbol{A}^{\\mathsf{T}} \\in \\mathbb{") and mid-link, plus
    unpaired "Error rate**:" bolds and bare "$$" descriptions. Bold markers now
    strip pairwise, letterless lines (display-math fences, rules) are skipped,
    and the cut lands on whitespace outside any $...$ / [[...]] span.
    """
    fence = False
    for raw in (snippet or "").splitlines():
        stripped = raw.strip()
        # Display-math fences: everything between $$ pairs is TeX innards,
        # never a description ("- [[ha bias]] — $$ \operatorname{bias}(...").
        odd = stripped.count("$$") % 2 == 1
        if fence:
            fence = not odd
            continue
        if stripped.startswith("$$"):
            fence = odd
            continue
        line = re.sub(r'^\s*>+\s*', '', stripped)             # blockquote
        line = re.sub(r'^\[![^\]]+\][-+]?\s*', '', line)      # callout tag
        line = re.sub(r'^#{1,6}\s*', '', line)                # heading
        line = re.sub(r'^[-*+]\s+', '', line)                 # list bullet
        line = line.replace("**", "").strip().strip('*_`').strip()
        if not re.search(r'[A-Za-zÀ-ÿ]{3,}', line):
            continue  # bare $$ / horizontal rules / lone symbols: not prose
        return line if len(line) <= cap else _safe_cut(line, cap)
    return ""
