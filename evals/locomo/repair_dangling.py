#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Selective dangling-link repair for the LoCoMo read-path A/B.

Isolates the "navigator hits a wall" mechanism on a frozen FSM vault without
touching the original (which stays the A/B control). Two moves, decided by the
occurrence-vs-notes curve (see pick_n.py):

  * MATERIALIZE recurring targets (folded refs >= N): create one hub note per
    target, body = backlinks to the source notes that mention it. Turns a
    dead-end into a junction. N=3 because that is the last tier where each note
    added removes >= N walls (>=3:1), above the 1:1/2:1 tail that would just
    feed the dominant over-fragmentation defect.
  * STRIP everything else (prose fragments + refs < N): unwrap [[x]] -> x so the
    navigator sees plain text instead of a broken link.

Result: zero dangling, +19 notes (N=3), walls removed concentrated on traffic.
Run:  uv run python evals/locomo/repair_dangling.py [SRC] [DST] [N]
"""
from __future__ import annotations

import collections
import re
import shutil
import sys
from pathlib import Path

WIKI_FULL = re.compile(r"\[\[([^\]]+)\]\]")  # whole [[...]] span, rewritable


def target_of(inner: str) -> str:
    """[[target|alias]] / [[target#heading]] -> target basename, lowercased key."""
    tgt = inner.split("|", 1)[0].split("#", 1)[0].strip().removesuffix(".md")
    return tgt.rsplit("/", 1)[-1]  # basename: mirrors graph_export ghost naming


def display_of(inner: str) -> str:
    """Text to keep when unwrapping: alias if present, else the target text."""
    left, _, alias = inner.partition("|")
    return (alias or left).split("#", 1)[0].strip() or left.strip()


def is_prose(key: str) -> bool:
    return len(key.split()) >= 3


def stems_of(vault: Path) -> set[str]:
    return {p.stem.lower() for p in vault.rglob("*.md")}


def scan_dangling(vault: Path, resolvable: set[str]):
    """Return (folded_counter, occurrences_by_folded_key->list[source_path])."""
    counts: collections.Counter[str] = collections.Counter()
    sources: dict[str, list[Path]] = collections.defaultdict(list)
    for p in vault.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in WIKI_FULL.finditer(text):
            key = target_of(m.group(1)).lower()
            if key and key not in resolvable:
                counts[key] += 1
                sources[key].append(p)
    return counts, sources


def repair(src: Path, dst: Path, n: int) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    resolvable = stems_of(dst)
    counts, sources = scan_dangling(dst, resolvable)

    # --- decide the two sets ---
    materialize = {k for k, c in counts.items() if c >= n and not is_prose(k)}

    # dominant surface casing per materialized key (for the hub filename)
    surface: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in dst.rglob("*.md"):
        for m in WIKI_FULL.finditer(p.read_text(encoding="utf-8", errors="replace")):
            key = target_of(m.group(1)).lower()
            if key in materialize:
                surface[key][target_of(m.group(1))] += 1

    # --- create hubs (added to resolvable BEFORE the strip pass) ---
    created = 0
    for key in sorted(materialize):
        name = surface[key].most_common(1)[0][0]
        src_notes = sorted({p for p in sources[key]}, key=lambda p: p.name)
        folder = collections.Counter(p.parent for p in src_notes).most_common(1)[0][0]
        rel = "\n".join(f'  - "[[{p.stem}]]"' for p in src_notes)
        body = "\n".join(f"- [[{p.stem}]]" for p in src_notes)
        (folder / f"{name}.md").write_text(
            f"---\ntags:\n  - hub\nrelated:\n{rel}\n---\n"
            f"# {name}\n\nReferences to **{name}** across the vault:\n{body}\n",
            encoding="utf-8",
        )
        resolvable.add(name.lower())
        created += 1

    # --- single resolve-or-unwrap pass over the copy ---
    stripped = 0
    for p in dst.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="replace")

        def rewrite(m: re.Match) -> str:
            nonlocal stripped
            inner = m.group(1)
            if target_of(inner).lower() in resolvable:
                return m.group(0)  # resolves now -> keep the link
            stripped += 1
            return display_of(inner)  # dangling -> unwrap to plain text

        new = WIKI_FULL.sub(rewrite, text)
        if new != text:
            p.write_text(new, encoding="utf-8")

    # --- runnable check: the copy must have zero dangling now ---
    after, _ = scan_dangling(dst, stems_of(dst))
    assert not after, f"repair left {sum(after.values())} dangling: {dict(after)}"

    total = sum(counts.values())
    print(f"src {src}  ->  dst {dst}   (N={n})")
    print(f"  dangling before: {total} occ / {len(counts)} folded targets")
    print(f"  materialized:    {created} hubs (refs>={n}, non-prose)")
    print(f"  stripped:        {stripped} link occurrences unwrapped")
    print("  dangling after:  0  (asserted)")
    print(f"  notes: {len(list(src.rglob('*.md')))} -> {len(list(dst.rglob('*.md')))}")
    return created


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench/locomo_e2e_fsm")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("bench/locomo_e2e_fsm_repaired")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    repair(src, dst, n)
