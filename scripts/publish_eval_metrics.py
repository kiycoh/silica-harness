#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Copy the eval runs the README cites out of bench/ and into evals/.

    uv run python scripts/publish_eval_metrics.py

bench/ is a scratch tree: it is gitignored, it holds whole vaults beside the
metrics, and a run that lands there is one laptop away from being unciteable.
The README quotes numbers, so the file behind each number has to be in the
repository. This moves the few that are cited and nothing else.

Two shapes come out, because the runs differ in what is worth keeping:

- The LoCoMo runs carry a `questions` array of full model answers, 210-290 KB
  each. That array is a transcript, not evidence: nobody re-reads 150 answers
  to check an accuracy, and eight of them would put 2 MB of model prose in a
  git history that has to be cloned forever. Headline, config, provenance and
  the by-type split survive; the transcript does not.
- The FActScore runs are already small and their per-note scores ARE the
  evidence (the interesting row is which notes scored zero), so they copy whole.

The substrate A/B collapses to one file rather than seven: the claim it backs
is the shape of a table, and seven near-identical run files would each have to
be opened to read one row of it.

Re-runnable and idempotent: it reads bench/ and rewrites evals/, so a fresh run
republishes by pointing SOURCES at the new file.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
EVALS = ROOT / "evals"

# The two runs the README's LoCoMo rows quote, headline and abstention both.
# NOT named metrics.*.json: .gitignore reserves that pattern for the scratch
# output a run drops next to itself, the same split golden already makes between
# its ignored metrics.json and its committed baseline.json.
HEADLINE = {
    "c26.json": "final_c26.metrics.json",
    "c47.json": "final_c47.metrics.json",
}

# (conversation, arm, lossy?, source file). "lossy" is the variable under test:
# whether the write path is allowed to rewrite the source into its own prose.
SUBSTRATE = [
    ("conv-26", "extractive", False, "ab_extractive.agent.metrics.json"),
    ("conv-26", "hybrid", False, "ab_hybrid.agent.metrics.json"),
    ("conv-26", "verbatim", False, "ab_c26_verbatim_agent.metrics.json"),
    ("conv-26", "distill", True, "ab_c26_distill_agent.metrics.json"),
    ("conv-47", "verbatim", False, "v1_c47_verbatim.agent.metrics.json"),
    ("conv-47", "extractive", False, "v1_c47_extractive.agent.metrics.json"),
    ("conv-47", "distill", True, "v1_c47_distill.agent.metrics.json"),
]

FACTSCORE = {
    "verbatim.c26.json": "factscore_verbatim.c26.json",
    "distill.c26.json": "factscore_distill.c26.json",
    # The corrected run. The superseded one scored entity notes against a single
    # attributed session instead of the whole conversation and read 0.669; it
    # stays in bench/ rather than shipping a number that was a harness artifact.
    "extractive.c26.json": "factscore_extractive_fullconv.c26.json",
}


def _load(name: str) -> dict:
    return json.loads((BENCH / name).read_text())


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1024:.1f} KB")


def publish_headline() -> None:
    for out, src in HEADLINE.items():
        d = _load(src)
        d.pop("questions", None)
        _write(EVALS / "locomo" / out, d)


def publish_substrate() -> None:
    rows = []
    for conv, arm, lossy, src in SUBSTRATE:
        d = _load(src)
        m = d["metrics"]
        rows.append({
            "conversation": conv,
            "arm": arm,
            "lossy_write_path": lossy,
            "ingest_mode": d["config"].get("ingest_mode", "flat"),
            "overall_accuracy": m["overall_accuracy"],
            "answerable_n": m["answerable_n"],
            "abstention_accuracy": m["abstention_accuracy"],
            "abstention_n": m["abstention_n"],
            # The README claims the lossy arm is last in EVERY category on both
            # conversations, which is a stronger claim than the headline and is
            # only checkable if the split ships with it.
            "by_type": {k: v.get("accuracy") for k, v in m.get("by_type", {}).items()},
            # The two oldest c26 arms ran before `_shared.provenance` existed,
            # so they carry no run_id and no git_sha. Saying so beats leaving a
            # reader to assume every row is equally traceable.
            "provenance": d.get("provenance") or {
                "generated_at": d.get("generated_at"),
                "note": "predates evals/_shared.provenance (added 2026-07-22): "
                        "no run_id, no git_sha",
            },
            "source": f"bench/{src}",
        })
    _write(EVALS / "locomo" / "substrate.json", {
        "benchmark": "locomo",
        "question": "does a write path that rewrites the source into its own "
                    "prose answer better than one that keeps the source's words",
        "answer_mode": "agent",
        "answer_model": "openrouter/deepseek/deepseek-v4-flash",
        "judge_model": "openrouter/deepseek/deepseek-v4-flash",
        "caveats": [
            "n=2 conversations, one run each, OpenRouter routing unpinned: no CI.",
            "Arm ranking among the three non-lossy arms does NOT replicate: "
            "extractive leads on conv-26 and verbatim leads on conv-47, and the "
            "swing within one arm across conversations is larger than the gap "
            "between arms. Only the lossy-vs-non-lossy split holds on both.",
            "The distill arm is a flat slice-distill with no autolink or dedup, "
            "so it is a weaker pipeline as well as a lossy one.",
        ],
        "arms": rows,
    })


def publish_factscore() -> None:
    for out, src in FACTSCORE.items():
        dst = EVALS / "factscore" / out
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BENCH / src, dst)
        print(f"{dst.relative_to(ROOT)}: {dst.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    publish_headline()
    publish_substrate()
    publish_factscore()
