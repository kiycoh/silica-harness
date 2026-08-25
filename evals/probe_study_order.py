# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""G6 gate: does prerequisite-first block order ground explanations better?

Design (offline-signals-map §3, G6 verdict row): self-labeling question set
from the vault's own RefD edges ("Explain {dependent}"), gated on the FIRED
subset — questions whose top-10 already contains a prerequisite of a rendered
block; on the rest the lever is a no-op by construction, and averaging them
in would dilute a real effect below detection (the W2/W6 lesson, inverted).

The two arms share ONE retrieval: the study arm is `_study_order` applied to
a copy of the same blocks, so membership and excerpts are bit-identical and
the only variables are block order + builds-on tokens — the joint lever as
it ships. Answers via evals.oracle.cached_text (greedy, cached: a re-run
repays neither tokens nor noise).

Primary metric, deterministic: prereq coverage — the answer mentions (whole
word / phrase) at least one prerequisite that was IN the rendered context.
Paired McNemar. Guardrail: abstention/empty-answer rate must not rise in the
study arm (didactic order may not pay by refusing to answer).

Usage:
  uv run python -m evals.probe_study_order --vault ~/Documents/Obsidian/test \
      --fired 150 --out bench/study_order_probe.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ABSTAIN_RE = re.compile(r"do not have|don't have|no information|not covered",
                         re.IGNORECASE)

_SYSTEM = (
    "You answer strictly from the user's notes, which follow as context "
    "blocks. Explain the asked concept clearly for someone learning it, "
    "grounding the explanation in what the notes actually say. If the notes "
    "do not cover it, say so briefly."
)


def _name_of(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".md")


def _mentions(answer: str, name: str) -> bool:
    """Whole-word / whole-phrase match, case-insensitive. Multi-token names
    match as a phrase (any whitespace between tokens): the title_key
    stopword-collision lesson says never match on a name's subset."""
    toks = [re.escape(t) for t in name.split() if t]
    if not toks:
        return False
    return re.search(r"\b" + r"\s+".join(toks) + r"\b", answer,
                     re.IGNORECASE) is not None


def _mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def build_cases(pm: dict[str, list[str]], k: int, fired_target: int,
                now: str) -> list[dict]:
    """Scan dependents alphabetically (reproducible, no sampling bias toward
    hubs) until `fired_target` questions fire. One retrieval per case, shared
    by both arms."""
    from silica.kernel.recall.perception import _study_order, perceive

    cases: list[dict] = []
    for dep in sorted(pm):
        if len(cases) >= fired_target:
            break
        name = _name_of(dep)
        if len(name) < 4:
            continue  # junk-match risk in the coverage regex
        query = f"Explain {name}"
        base = perceive(query, now=now, k=k, with_facts=False)
        present = {b.path for b in base.blocks}
        # prereqs that are IN context: the coverage claim is about grounding
        # in material the model was actually shown, identical across arms.
        in_ctx = sorted({p for b in base.blocks for p in pm.get(b.path, [])
                         if p in present})
        in_ctx = [p for p in in_ctx if len(_name_of(p)) >= 4 and p != dep]
        if not in_ctx:
            continue
        study_blocks = _study_order(copy.deepcopy(base.blocks))
        if [b.path for b in study_blocks] == [b.path for b in base.blocks] \
                and not any(b.builds_on for b in study_blocks):
            continue  # fired but inert: nothing for the answer model to see
        study = copy.copy(base)
        study.blocks = study_blocks
        cases.append({
            "dependent": dep, "query": query,
            "prereqs_in_ctx": in_ctx,
            "ctx_base": base.render(),
            "ctx_study": study.render(),
        })
    return cases


def _answer(model: str, query: str, context: str) -> str:
    from evals.oracle import cached_text

    user = f"Notes:\n{context}\n\nTask: {query}"
    return (cached_text(model, [{"role": "system", "content": _SYSTEM},
                                {"role": "user", "content": user}],
                        max_tokens=512, temperature=0.0) or "").strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--fired", type=int, default=150,
                    help="stop scanning once this many fired cases are built")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--now", default="2026-08-25")
    ap.add_argument("--model", default="",
                    help="answer model; default CONFIG.model. ox-alpha returns "
                         "persistent empty/truncated streams on these long "
                         "prompts (probed 2026-08-25), deepseek-v4-flash is "
                         "the repo's stable probe model")
    ap.add_argument("--out", default="bench/study_order_probe.json")
    args = ap.parse_args(argv)

    from evals.longmemeval.runner import bind_vault
    from silica.config import CONFIG

    bind_vault(Path(args.vault).expanduser())
    from silica.kernel.report.learner import prerequisites_map

    pm = prerequisites_map()
    print(f"prereq edges: {sum(len(v) for v in pm.values())} "
          f"dependents: {len(pm)}")
    cases = build_cases(pm, args.k, args.fired, args.now)
    print(f"fired cases: {len(cases)}")
    if not cases:
        print("gate VACUOUS: nothing fired")
        return 1

    model = args.model or CONFIG.model

    def _run(case: dict) -> dict:
        row = dict(case)
        for arm in ("base", "study"):
            try:
                row[f"ans_{arm}"] = _answer(model, case["query"],
                                            case[f"ctx_{arm}"])
                row[f"err_{arm}"] = None
            except Exception as e:
                row[f"ans_{arm}"], row[f"err_{arm}"] = "", f"{type(e).__name__}: {e}"
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_run, cases))

    # <20 chars = a truncated stream, not an answer (ox-alpha probe above):
    # a 1-char reply must count as a failed row, never as "covered nothing".
    paired = [r for r in rows if not r["err_base"] and not r["err_study"]
              and len(r["ans_base"]) >= 20 and len(r["ans_study"]) >= 20]
    for r in paired:
        names = [_name_of(p) for p in r["prereqs_in_ctx"]]
        r["cov_base"] = any(_mentions(r["ans_base"], n) for n in names)
        r["cov_study"] = any(_mentions(r["ans_study"], n) for n in names)
    b = sum(1 for r in paired if r["cov_base"] and not r["cov_study"])
    c = sum(1 for r in paired if r["cov_study"] and not r["cov_base"])
    cov_a = sum(r["cov_base"] for r in paired)
    cov_b = sum(r["cov_study"] for r in paired)
    abst_a = sum(1 for r in paired if _ABSTAIN_RE.search(r["ans_base"]))
    abst_b = sum(1 for r in paired if _ABSTAIN_RE.search(r["ans_study"]))
    n = len(paired)
    p = _mcnemar_p(b, c)
    print(f"\npaired n={n} (errors excluded: {len(rows) - n})")
    print(f"prereq coverage: base {cov_a}/{n} ({cov_a/n:.3f}) -> "
          f"study {cov_b}/{n} ({cov_b/n:.3f})")
    print(f"flips: base-only {b}, study-only {c}   McNemar p={p:.4f}")
    print(f"guardrail abstention: base {abst_a} vs study {abst_b}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"vault": args.vault, "k": args.k, "model": model,
                   "fired_target": args.fired, "now": args.now},
        "metrics": {"n": n, "cov_base": cov_a, "cov_study": cov_b,
                    "flips_base_only": b, "flips_study_only": c,
                    "mcnemar_p": p, "abstain_base": abst_a,
                    "abstain_study": abst_b},
        "rows": rows,
    }, indent=1), encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
