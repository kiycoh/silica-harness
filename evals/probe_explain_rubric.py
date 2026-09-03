# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Does a four-axis rubric carry any information about what the vault can EXPOSE?

Lineage: `docs/repos/get-it` scores a STUDENT on four axes (memory,
comprehension, structure, application) from their study transcript, and clamps
the scores monotone non-decreasing so a learner never regresses. We keep the
four axes and the two anti-inflation clauses; we invert the subject and drop
the clamp:

  get-it        "how well does the reader know this concept?"
  here          "how well can the VAULT expose this fact to a reader?"

The clamp goes because it is a motivational device for a learner. A vault
regresses for real: delete the worked example and the application axis drops.

Unit of scoring is the NOTE, not the concept. Two reasons. Notes are Silica's
atom and already carry a wikilink degree, which makes gate 4 free; and a
concept-level rubric presupposes the typed concept graph that does not exist
yet, so it would measure the rubric and the missing graph at once.

WHAT WOULD MAKE THIS WORTHLESS (pre-registered, before the run, so a bad result
cannot be re-read as a good one afterwards):

  G1 VARIANCE       a score every note shares carries no information.
                    >= 3 of 4 axes need stdev >= 10 points on 0-100.
  G2 INDEPENDENCE   four names for one axis is a complexity tax.
                    every pairwise |spearman| between axes < 0.85.
  G3 REPRODUCIBLE   the run-1 signal of the code-why probe REVERSED on
                    replication. So arm R replicates arm A up front, with the
                    batch composition and the note order SHUFFLED — that tests
                    context contamination and position bias, which is what
                    actually moves, rather than decoder determinism, which
                    temperature 0 fakes. Per-axis mean|A-R| < stdev(A)/2.
  G4 NOT-A-PROXY    if the structure axis just re-reads the wikilink graph,
                    read the graph — it is free and already computed.
                    |spearman(structure, degree)| < 0.60.

  H  (harness, not a product gate) arm T re-scores each note with its BODY
     STRIPPED to the title. If title-only notes score like full notes the
     judge is not reading the note and every number above is noise.
     mean(T) must sit >= 20 points below mean(A) on >= 3 axes.

All four gates must pass to earn the next probe (does the score predict the
quality of an explanation built from those notes). Any failure kills the axis
or the rubric, here, cheaply, before a line of product code exists.

    uv run python -m evals.probe_explain_rubric --vault ~/Documents/Obsidian/test

History: the comprehension axis is adjacent to the WHY stratum killed twice on
2026-07-25 (`docs/spec-code-why.md`). That kill was ceiling-forced by a
greppable docs/ tree, which does not apply to a vault, but it is the reason
this probe exists at all instead of a feature branch.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from pydantic import BaseModel
from scipy.stats import spearmanr

AXES = ("memory", "comprehension", "structure", "application")

# --- Gates. Declared here, before the run. Not tuned afterwards. -----------
GATE_MIN_STDEV = 10.0        # G1: per-axis spread on 0-100
GATE_MIN_AXES_WITH_SPREAD = 3
GATE_MAX_AXIS_RHO = 0.85     # G2: pairwise correlation between axes
GATE_REPRO_RATIO = 0.5       # G3: mean|A-R| must stay under stdev(A) * this
GATE_MAX_DEGREE_RHO = 0.60   # G4: structure vs wikilink degree
HARNESS_MIN_TITLE_DROP = 20.0  # H: mean(A) - mean(T)

BATCH = 8            # notes per judge call
BODY_CAP = 6000      # chars of note body sent per note; counted when it bites


# ---------------------------------------------------------------------------
# The rubric
# ---------------------------------------------------------------------------

RUBRIC_SYSTEM = """You score how well a single note lets a reader be TAUGHT the \
fact it carries. You are not scoring the reader, and you are not scoring whether \
the note is well written. You score what a teacher could do with this note and \
nothing else in front of them.

Four independent 0-100 axes per note.

  MEMORY - is the fact STATED, plainly and atomically?
    Can a reader quote one span of this note and have the claim itself? High:
    the claim is explicit and self-contained. Low: the fact is implied,
    scattered across the note, or buried in a narrative that has to be
    reassembled before it can be repeated.

  COMPREHENSION - is there a layer that RE-SAYS the fact, or only the source's
    own shape?
    High: the note re-expresses the idea in different words than the source
    would have used, or says why it holds, or names what it rules out. Low: the
    note is a faithful paraphrase of the material it came from and adds no
    second angle on it. A correct, well-cited restatement with no re-saying
    scores LOW on this axis. This is the axis a distiller does not fill by
    being accurate.

  STRUCTURE - does the note SITUATE the fact among other facts, in its own
    prose?
    High: names what must be understood first, what it contrasts with, what it
    is a special case of, what it causes. Low: an isolated statement. Judge the
    PROSE only: a wikilink with no sentence explaining the relation is not
    situation, it is an address.

  APPLICATION - is there a CONCRETE case?
    High: a worked example, a real instance, a boundary case, a number with
    units, a "this is what it looks like when it fails". Low: the general
    statement only.

RULES
1. Quantity does not entitle a score. A long note that says one thing five
   times scores as if it said it once.
2. No evidence for an axis means 0 on that axis. Do not award a floor for
   effort, length, or good writing. Most notes will genuinely be 0 on
   application; that is information, not a miss on your part.
3. The four axes are independent. A note can be 90 memory and 0 comprehension
   (a crisp statement, never re-said) or 0 memory and 70 application (a story
   that shows the thing without ever stating it). Do not let one axis pull the
   others.
4. Score the note in whatever language it is written in. Never translate, never
   penalise a language.
5. `missing` names the single axis whose absence costs a teacher the most on
   this note, in one short clause, in the note's own language. Empty if the
   note is already exposable on every axis.

Return one JSON object matching the schema. No prose outside it."""


class NoteScore(BaseModel):
    index: int
    memory: int
    comprehension: int
    structure: int
    application: int
    missing: str = ""


class RubricResult(BaseModel):
    scores: list[NoteScore]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_notes(vault: Path, store, n: int, seed: str) -> list[dict]:
    """`n` notes stratified by wikilink degree, deterministic under `seed`.

    Stratified because gate 4 asks whether the structure axis is a restatement
    of degree: drawing uniformly from a vault whose degree distribution has a
    long thin tail would leave that correlation undefined at the top end, and an
    undefined correlation passes the gate for the wrong reason.
    """
    from silica.kernel.link.health import iter_notes, wikilink_graph

    adj = wikilink_graph(vault, store)
    rows: list[dict] = []
    for p in iter_notes(vault):
        key = p.relative_to(vault).with_suffix("").as_posix()
        body = p.read_text(encoding="utf-8", errors="replace")
        if not body.strip():
            continue
        rows.append({
            "key": key,
            "title": p.stem,
            "body": body,
            "chars": len(body),
            "degree": len(adj.get(key, ())),
        })
    if not rows:
        return []

    rows.sort(key=lambda r: (r["degree"], r["key"]))
    rng = random.Random(seed)
    # Three degree strata, equal draw from each: isolated / typical / hub.
    third = max(1, len(rows) // 3)
    strata = [rows[:third], rows[third:2 * third], rows[2 * third:]]
    per = max(1, n // 3)
    out: list[dict] = []
    for band in strata:
        out.extend(rng.sample(band, min(per, len(band))))
    # Top up from whatever is left if a stratum ran short of `per`.
    taken = {r["key"] for r in out}
    remaining = [r for r in rows if r["key"] not in taken]
    rng.shuffle(remaining)
    out.extend(remaining[: max(0, n - len(out))])
    out.sort(key=lambda r: r["key"])
    return out[:n]


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def _render(note: dict, *, title_only: bool) -> str:
    if title_only:
        return f"TITLE: {note['title']}\nBODY: (empty)"
    body = note["body"][:BODY_CAP]
    return f"TITLE: {note['title']}\nBODY:\n{body}"


def score_batch(notes: list[dict], model: str, *, title_only: bool = False) -> dict[str, dict]:
    """One judge call over `notes`; {key: {axis: int, ...}}. Missing rows drop.

    A batch that fails to parse returns nothing rather than a zeroed row: a
    zero is a real score on this rubric ("no evidence"), so faking one would
    push every gate the wrong way and look like a finding.
    """
    from silica.agent.llm import call_llm
    from silica.kernel.text.sanitize import parse_json

    blocks = "\n\n".join(
        f"===== NOTE {i} =====\n{_render(nt, title_only=title_only)}"
        for i, nt in enumerate(notes)
    )
    resp = call_llm(
        model=model,
        messages=[
            {"role": "system", "content": RUBRIC_SYSTEM},
            {"role": "user", "content": f"{blocks}\n\nScore every note. JSON only."},
        ],
        response_format=RubricResult,
        temperature=0.0,
    )
    parsed, _ = parse_json(resp.text or "", strict=False)
    rows = parsed.get("scores") if isinstance(parsed, dict) else None
    out: dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(notes)):
            continue
        try:
            out[notes[idx]["key"]] = {
                **{a: max(0, min(100, int(row[a]))) for a in AXES},
                "missing": str(row.get("missing", ""))[:200],
            }
        except (KeyError, TypeError, ValueError):
            continue
    return out


def run_arm(notes: list[dict], model: str, *, shuffle_seed: str | None = None,
            title_only: bool = False) -> dict[str, dict]:
    """Score every note, batched.

    `shuffle_seed` re-packs the batches and re-orders the notes inside them.
    That, not a second call at the same temperature, is what a replication has
    to vary: the failure mode being tested is one note's score moving because
    of the notes it was judged next to.
    """
    order = list(notes)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    scores: dict[str, dict] = {}
    for i in range(0, len(order), BATCH):
        batch = order[i:i + BATCH]
        try:
            scores.update(score_batch(batch, model, title_only=title_only))
        except Exception as exc:  # one dead batch must not lose the run
            print(f"   batch {i // BATCH} failed ({type(exc).__name__}: {exc}) — skipped")
    return scores


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _col(scores: dict[str, dict], keys: list[str], axis: str) -> list[float]:
    return [float(scores[k][axis]) for k in keys]


def _rho(xs: list[float], ys: list[float]) -> float:
    """Spearman, 0.0 when either side is constant (rho is undefined there).

    Undefined must read as "no evidence of redundancy", never as nan silently
    comparing False against every threshold.
    """
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0
    rho = float(spearmanr(xs, ys).statistic)
    return 0.0 if rho != rho else rho


def gates(arm_a: dict, arm_r: dict, arm_t: dict, notes: list[dict]) -> dict:
    both = sorted(set(arm_a) & set(arm_r))
    keys_a = sorted(arm_a)
    degree = {n["key"]: float(n["degree"]) for n in notes}

    stdev = {a: (statistics.stdev(_col(arm_a, keys_a, a)) if len(keys_a) > 1 else 0.0)
             for a in AXES}
    mean_a = {a: statistics.fmean(_col(arm_a, keys_a, a)) if keys_a else 0.0 for a in AXES}
    spread_axes = [a for a in AXES if stdev[a] >= GATE_MIN_STDEV]

    pairs = {}
    for i, x in enumerate(AXES):
        for y in AXES[i + 1:]:
            pairs[f"{x}~{y}"] = _rho(_col(arm_a, keys_a, x), _col(arm_a, keys_a, y))

    repro = {}
    for a in AXES:
        deltas = [abs(arm_a[k][a] - arm_r[k][a]) for k in both]
        mad = statistics.fmean(deltas) if deltas else 0.0
        repro[a] = {"mean_abs_delta": mad,
                    "budget": stdev[a] * GATE_REPRO_RATIO,
                    "ok": mad < stdev[a] * GATE_REPRO_RATIO}

    deg_rho = {a: _rho(_col(arm_a, keys_a, a), [degree[k] for k in keys_a]) for a in AXES}

    keys_t = sorted(set(arm_a) & set(arm_t))
    drops = {a: (statistics.fmean(_col(arm_a, keys_t, a)) -
                 statistics.fmean(_col(arm_t, keys_t, a))) if keys_t else 0.0
             for a in AXES}

    g1 = len(spread_axes) >= GATE_MIN_AXES_WITH_SPREAD
    g2 = all(abs(v) < GATE_MAX_AXIS_RHO for v in pairs.values())
    g3 = all(repro[a]["ok"] for a in AXES)
    g4 = abs(deg_rho["structure"]) < GATE_MAX_DEGREE_RHO
    harness = sum(1 for a in AXES if drops[a] >= HARNESS_MIN_TITLE_DROP) >= 3

    return {
        "n_scored": len(keys_a), "n_replicated": len(both), "n_title_only": len(keys_t),
        "mean": mean_a, "stdev": stdev, "axes_with_spread": spread_axes,
        "axis_correlations": pairs, "reproducibility": repro,
        "degree_correlation": deg_rho, "title_only_drop": drops,
        "G1_variance": g1, "G2_independence": g2, "G3_reproducible": g3,
        "G4_not_a_proxy": g4, "H_judge_reads_the_note": harness,
        "verdict": ("PASS" if (g1 and g2 and g3 and g4 and harness)
                    else "HARNESS BUG" if not harness else "KILL"),
    }


# ---------------------------------------------------------------------------

def run(vault: Path, *, n: int, model: str) -> dict:
    from evals._shared import provenance, warn_unpinned_provider
    from evals.golden.runner import _open_stores, vault_digest
    from silica.config import CONFIG

    warn_unpinned_provider(model, CONFIG.openrouter_provider)
    store, _embed = _open_stores(vault)
    digest, total = vault_digest(vault)
    notes = sample_notes(vault, store, n, seed=digest)
    truncated = sum(1 for nt in notes if nt["chars"] > BODY_CAP)
    print(f"vault {vault}  ({total} notes, {digest[:19]}…)  judge {model}")
    print(f"sampled {len(notes)} notes, degree "
          f"{min(nt['degree'] for nt in notes)}–{max(nt['degree'] for nt in notes)}"
          f"{f', {truncated} bodies truncated at {BODY_CAP} chars' if truncated else ''}")

    print("\n1. ARMS")
    arm_a = run_arm(notes, model)
    print(f"   A  full note                     {len(arm_a)}/{len(notes)} scored")
    arm_r = run_arm(notes, model, shuffle_seed=f"{digest}:R")
    print(f"   R  replication, batches shuffled {len(arm_r)}/{len(notes)} scored")
    arm_t = run_arm(notes, model, title_only=True)
    print(f"   T  title only  (harness check)   {len(arm_t)}/{len(notes)} scored")

    g = gates(arm_a, arm_r, arm_t, notes)

    print("\n2. GATES")
    print("   G1 variance      " + "  ".join(
        f"{a[:5]} σ{g['stdev'][a]:5.1f} µ{g['mean'][a]:5.1f}" for a in AXES))
    print(f"      {len(g['axes_with_spread'])}/4 axes with σ >= {GATE_MIN_STDEV} "
          f"(need {GATE_MIN_AXES_WITH_SPREAD}) → {'PASS' if g['G1_variance'] else 'FAIL'}")
    worst = max(g["axis_correlations"].items(), key=lambda kv: abs(kv[1]))
    print(f"   G2 independence  worst pair {worst[0]} rho {worst[1]:+.3f} "
          f"(< {GATE_MAX_AXIS_RHO}) → {'PASS' if g['G2_independence'] else 'FAIL'}")
    for a in AXES:
        r = g["reproducibility"][a]
        print(f"   G3 repro {a:<14} mean|A-R| {r['mean_abs_delta']:5.2f} "
              f"budget {r['budget']:5.2f} {'ok' if r['ok'] else 'NOISE'}")
    print(f"      → {'PASS' if g['G3_reproducible'] else 'FAIL'}")
    print(f"   G4 not-a-proxy   structure~degree rho {g['degree_correlation']['structure']:+.3f} "
          f"(< {GATE_MAX_DEGREE_RHO}) → {'PASS' if g['G4_not_a_proxy'] else 'FAIL'}")
    print("   H  judge reads   drop A-T " + "  ".join(
        f"{a[:5]} {g['title_only_drop'][a]:+5.1f}" for a in AXES)
        + f" → {'ok' if g['H_judge_reads_the_note'] else 'HARNESS BUG'}")
    print(f"\n   VERDICT: {g['verdict']}")

    if g["verdict"] == "PASS":
        weakest = min(AXES, key=lambda a: g["mean"][a])
        print(f"   weakest axis of this vault: {weakest} (µ {g['mean'][weakest]:.1f})")

    return {
        "provenance": provenance(vault),
        "vault": {"path": str(vault), "digest": digest, "notes": total},
        "config": {"n": len(notes), "batch": BATCH, "model": model,
                   "body_cap": BODY_CAP, "bodies_truncated": truncated,
                   "provider_pin": CONFIG.openrouter_provider},
        "gates": g,
        "arms": {"A": arm_a, "R": arm_r, "T": arm_t},
        "notes": [{k: nt[k] for k in ("key", "degree", "chars")} for nt in notes],
    }


def main(argv=None) -> int:
    from evals.golden.runner import resolve_vault
    from silica.config import CONFIG

    ap = argparse.ArgumentParser(prog="python -m evals.probe_explain_rubric")
    ap.add_argument("--vault")
    ap.add_argument("--n", type=int, default=60, help="notes to score (default 60)")
    ap.add_argument("--model", default=None, help="judge model (default CONFIG.model)")
    ap.add_argument("--json", default="bench/explain_rubric.json")
    args = ap.parse_args(argv)

    vault = resolve_vault(args.vault)
    try:
        res = run(vault, n=args.n, model=args.model or CONFIG.model)
    finally:
        import silica.driver
        silica.driver._driver = None
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
