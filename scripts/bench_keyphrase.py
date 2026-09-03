# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Keyphrase extraction bench on a real vault: link-title recall and precision.

Gold for a note = the titles it links to with `[[...]]`: what the author judged
worth a note of its own is the best label of "concept" a vault carries for free,
and the matcher is the stem-level one the golden eval uses. Both extraction
modes are measured because they fail differently: with the embedder the pool's
composition matters, without it the pool's own order is the rank.

This is the harness that decided the yake replacement (ADR-0034): 246 notes,
yake .259/.159 fallback and .314/.173 with the embedder; the in-house miner
.256/.157 and .309/.171. Run it before and after touching
kernel/text/candidates.py or keyphrase.py, on the same vault, and compare.

Same vault, 250 notes on 2026-09-02, fallback mode: .2552/.1572 before the
name-hygiene screens of `candidates.is_fragment` and the markup-seed casing
fix, .2567/.1593 after. The screens delete titles, so precision was the
metric at risk and it moved the right way (9 notes gained recall, 3 lost).

Usage:
    uv run python scripts/bench_keyphrase.py <vault_dir> [--embedder] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from silica.kernel.text import language  # noqa: E402
from silica.kernel.text.keyphrase import extract_keyphrases  # noqa: E402
from silica.kernel.text.overlay import overlay_for_lang  # noqa: E402
from tests.golden.test_eval_keyphrase import concept_recalled  # noqa: E402

_LINK = re.compile(r"\[\[([^\[\]|#]+)(?:[#|][^\]]*)?\]\]")
_FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.S)
MIN_LINKS, MIN_WORDS = 3, 120


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("vault", type=Path)
    ap.add_argument("--embedder", action="store_true",
                    help="rank with the configured embedder (default: deterministic fallback)")
    ap.add_argument("--out", type=Path, help="write per-note rows and the summary as JSON")
    args = ap.parse_args()

    embedder = None
    if args.embedder:
        from silica.agent.providers import get_embedder
        from silica.config import CONFIG
        embedder = get_embedder(CONFIG)
        embedder.embed(["probe"])  # a dead endpoint fails here, not silently per note

    rows = []
    for path in sorted(args.vault.rglob("*.md")):
        if any(part.startswith(".") for part in path.parts):
            continue
        body = _FRONTMATTER.sub("", path.read_text(encoding="utf-8", errors="replace"), count=1)
        gold = sorted({m.group(1).strip() for m in _LINK.finditer(body)
                       if 2 <= len(m.group(1).strip()) <= 60})
        if len(gold) < MIN_LINKS or len(body.split()) < MIN_WORDS:
            continue
        lang = language.resolve("auto", body)
        t0 = time.perf_counter()
        cands = extract_keyphrases(body, overlay=overlay_for_lang(lang), lang=lang, embedder=embedder)
        phrases = [c.phrase for c in cands]
        hits = sum(concept_recalled(g, phrases, lang) for g in gold)
        precise = sum(any(concept_recalled(g, [p], lang) for g in gold) for p in phrases)
        rows.append({
            "note": str(path.relative_to(args.vault)), "lang": lang, "gold": gold,
            "phrases": phrases, "recall": hits / len(gold),
            "precision": precise / len(phrases) if phrases else 0.0,
            "secs": time.perf_counter() - t0,
        })
    if not rows:
        print("no note with enough links and words under", args.vault)
        return 1
    n = len(rows)
    summary = {
        "mode": "embedder" if embedder else "fallback", "notes": n,
        "mean_recall": round(sum(r["recall"] for r in rows) / n, 4),
        "mean_precision": round(sum(r["precision"] for r in rows) / n, 4),
        "total_secs": round(sum(r["secs"] for r in rows), 1),
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
