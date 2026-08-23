# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Co-write transactions for the coupling variable (V3): which notes each
nucleate run wrote together, read from the run manifests.

Lives here and not in silica.kernel.progress, which owns the run layout,
because progress transitively imports silica.agent (via episodic) and the
L1 report may not. The runs root is derived from the same ~/.silica home the
rest of the kernel uses; only the "runs" segment is named twice.
"""
from __future__ import annotations

import logging
from pathlib import Path

import orjson

logger = logging.getLogger(__name__)


def runs_root() -> Path:
    # Function, not constant: tests monkeypatch it.
    from silica.kernel.recall.paths import _SILICA_HOME

    return _SILICA_HOME / "runs"


def cowrite_transactions(vault: str, *, cap: int = 30) -> tuple[list[set[str]], int]:
    """(note sets written together per run, runs dropped for size).

    One transaction per manifest whose ledger names `vault` (the runs root is
    machine-wide, and most manifests on a developer box come from pytest's
    isolated vaults). Paths are manifest paths (vault-relative, no `.md`).
    Runs with more than `cap` notes are dropped and COUNTED, not silently
    capped: a 456-note batch ingest says nothing about which of its notes
    belong together. Runs with fewer than two notes carry no pair and are not
    counted as dropped.
    """
    out: list[set[str]] = []
    dropped = 0
    root = runs_root()
    if not root.exists():
        return out, dropped
    for d in sorted(root.iterdir()):
        ledger = d / "ledger.json"
        manifest = d / "manifest.json"
        if not (d.is_dir() and ledger.exists() and manifest.exists()):
            continue
        try:
            if str(orjson.loads(ledger.read_bytes()).get("vault") or "") != vault:
                continue
            entries = orjson.loads(manifest.read_bytes()).get("entries") or []
        except Exception:
            continue  # a torn run file is one run's memory, not the vault's
        paths = {str(e.get("path") or "") for e in entries if e.get("path")}
        if len(paths) < 2:
            continue
        if len(paths) > cap:
            dropped += 1
            continue
        out.append(paths)
    return out, dropped


def coupling_transactions(
    vault: str,
    sources_of: dict[str, list[str]],
    ids: set[str] | None = None,
    *,
    cap: int = 30,
) -> tuple[list[set[str]], int]:
    """Every transaction the coupling variable (V3) reads, in one assembly.

    Two kinds, because a vault records "written together" two ways: a
    frontmatter `sources:` entry (the notes that cite one source) and a run
    manifest (the notes one nucleate wrote). Extracted here rather than left
    inline in compute_report because the graph viewer draws the same pairs,
    and two assemblies of the same evidence would drift the moment one of them
    learned about a third kind.

    `sources_of` and `ids` are in the GRAPH keyspace (`.md`); manifest paths
    are not, so they are matched back through `ids`. `ids` None means keep the
    manifest paths as they are, which is what a caller with no graph wants.
    """
    by_source: dict[str, set[str]] = {}
    for nid, srcs in sources_of.items():
        for src in srcs:
            by_source.setdefault(src, set()).add(nid)
    transactions = [t for t in by_source.values() if len(t) > 1]

    runs, dropped = cowrite_transactions(vault, cap=cap)
    if ids is None:
        transactions += [set(run) for run in runs]
    else:
        gid_by_key = {i.removesuffix(".md"): i for i in ids}
        transactions += [
            {gid_by_key[p] for p in run if p in gid_by_key} for run in runs
        ]
    return transactions, dropped
