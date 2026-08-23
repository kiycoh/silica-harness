# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Recall-outcome weight store — LoCoMo eval-only, phase 1 of `improve`.

Records, per vault, which notes contributed to a context that answered a
question correctly. `_fuse` (relatedness.py) folds these into RRF as an
optional extra ranking leg when the caller opts in; every other caller
defaults to abstaining, so this file is inert unless `bump` has been called
for the active vault.

ponytail: no schema, no db, no quarantine on corrupt JSON (unlike
EpisodicStore) — this is a small derived cache, not user data; on corruption
it silently resets to empty rather than sidelining the file.
"""
from __future__ import annotations

import json
import logging

from silica.kernel.recall.cooccurrence import cooccur_key
from silica.kernel.recall.paths import atomic_write_bytes, index_dir

logger = logging.getLogger(__name__)


def _store_path():
    return index_dir() / "recall_weights.json"


def _load() -> dict[str, float]:
    try:
        data = json.loads(_store_path().read_text(encoding="utf-8"))
    except Exception:
        return {}
    # Any shape-corruption (non-dict, or a non-numeric value from a manual
    # edit / partial rewrite) resets to empty, so ranking() can never raise —
    # the docstring's "silently resets to empty" contract, actually enforced.
    if not isinstance(data, dict) or not all(
            isinstance(v, (int, float)) for v in data.values()):
        return {}
    return data


def bump(paths: list[str]) -> None:
    """Increment the weight of each note in `paths` by 1, once each.

    Normalizes every path to the canonical (no-`.md`, case-preserved) key via
    `cooccur_key` before deduping and incrementing, so the same note reached
    via two different tool calls in one turn (e.g. silica_recall then
    silica_read_note) counts once. Best-effort: an I/O failure logs and
    returns rather than raising — a persisted weight is a nice-to-have, not
    something a judged question should ever fail over.
    """
    if not paths:
        return
    keys = {cooccur_key(p) for p in paths}
    try:
        weights = _load()
        for key in keys:
            weights[key] = weights.get(key, 0.0) + 1.0
        atomic_write_bytes(
            _store_path(), json.dumps(weights, ensure_ascii=False).encode("utf-8"))
    except Exception as e:
        logger.warning("recall_weights.bump: write failed (%s); weights not persisted", e)


def ranking() -> list[tuple[str, float]] | None:
    """Weighted paths, best-first, for the RRF `recall_rank` leg.

    None when the store is empty/missing/corrupt — `_fuse` treats this as an
    abstaining leg, the same idiom as `cooc_rank`.
    """
    weights = _load()
    if not weights:
        return None
    return sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
