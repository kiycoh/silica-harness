# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""COLLISION routing as selective prediction: the risk-coverage curve of the
scalar it thresholds, with duplicates that exist.

COLLISION (router/states/collision.py) auto-decides an incoming concept when a
cosine clears tau_high (patch into the existing note) or falls under tau_low
(keep as a new note) and defers the band to the dedup judge. The two constants
were set by hand, and the scalar they test is the cosine of the RRF winner,
which is not the cosine-best note in 57% of cases (ADR-0029). The selective
prediction literature evaluates such a policy on its risk-coverage curve
(Kamath et al. 2020 arXiv 2006.09462; Phillips et al. 2026 arXiv 2603.21172;
Zhu et al. 2026 arXiv 2605.18792): coverage = fraction auto-decided, risk =
error rate among those, the judge being the abstention that costs a call but
not a mistake.

Labels. Every non-inbox note is an incoming concept whose human-kept
neighbours are distinct: a mechanical patch is a FALSE MERGE. The duplicates
are SYNTHETIC: the analyst-labeled TRUE_DUPS of the golden harness were
merged out of the vault, so for a seeded sample of notes the worker model
writes the second note a different student would write (same concept, other
wording, every other one in the other language, the two shapes TRUE_DUPS had)
and that note is the incoming concept whose only correct target is its
source; a keep is a LEAK. Both classes are embedded the way COLLISION embeds
an incoming concept (`_note_text(name, excerpt)`, no folder prefix) and routed
through the production call (`related_notes_for_query`), so the only thing
synthetic is the duplicate's prose, and that is stated next to every number.

Scalars, all routed by the real `route_concept` with the real `_names_agree`
and hub set:

  rrf_winner   shipped: the facade winner's cosine, target = that winner
  cosine_top   the cosine-best candidate's cosine, target = it. This is also
               the routing of a slate (Extract-Define-Canonicalize, arXiv
               2404.03868: decide on several candidates plus an explicit
               none-of-the-above): any candidate over tau_low means the
               cosine-best one is, so a slate changes what the judge reads,
               never which notes reach it. `slate_ceiling` is how often the
               duplicate is in the top-5 at all, the recall the judge can
               never exceed (OBLIQ-Bench, arXiv 2605.06235: verification is
               reliable once the document is surfaced; surfacing is the hard part).

The convex-combination fused score of probe_fusion_function is NOT a scalar
here: its TMM normalisation divides by the query's best score, so the
top candidate's embed term is 1.0 by construction and the fused score of a
winner is >= alpha whatever its cosine. Bounded, not comparable across queries.

Per scalar: the operating point of the shipped constants, the coverage
reachable at risk 0 / 1% / 2% anywhere on the (tau_high, tau_low) grid, the
AURC of the grid's Pareto frontier, and the AUC of the scalar as a separator
of duplicates from distinct notes.

  uv run python -m evals.probe_collision_risk_coverage --vault ~/Documents/Obsidian/test
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from evals.golden.probe_dedup import _hub_keys
from evals.probe_graph_variables import _auc, _stores

TOP_N = 5        # COLLISION's own k
N_POSITIVES = 60
MIN_BODY = 400   # a source short enough to have no claims makes a vacuous duplicate
EXCERPT_CHARS = 600
SEED = 42
GRID = tuple(round(0.30 + 0.01 * i, 2) for i in range(70))   # 0.30 .. 0.99
RISK_TARGETS = (0.0, 0.01, 0.02)
OTHER_LANGUAGE = {"italian": "English", "english": "Italian"}


@dataclass
class Incoming:
    key: str
    title: str
    excerpt: str
    dups: frozenset[str]            # store keys that are a correct patch target
    exclude: frozenset[str]         # a vault note standing in as incoming is not its own target
    synthetic: bool
    vec: list[float] | None = None
    # scalar -> (target key or None, score). None target = the cold path,
    # where COLLISION keeps the concept without consulting a threshold.
    scalars: dict[str, tuple[str | None, float]] = field(default_factory=dict)
    slate: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# incoming concepts: the vault's notes, and a synthetic duplicate per sample
# ---------------------------------------------------------------------------

def _bodies(vault: Path, keys: list[str]) -> dict[str, str]:
    from silica.kernel.write import frontmatter

    out = {}
    for k in keys:
        try:
            _d, _r, body = frontmatter.split((vault / f"{k}.md").read_text(encoding="utf-8"))
        except OSError:
            continue
        out[k] = " ".join(body.split())
    return out


def sample_sources(keys: list[str], bodies: dict[str, str], *, n: int = N_POSITIVES,
                   seed: int = SEED) -> list[str]:
    """Round-robin over top-level folders so one big domain cannot be the
    whole positive set; deterministic for a given key list."""
    rng = random.Random(seed)
    by_domain: dict[str, list[str]] = {}
    for k in keys:
        if len(bodies.get(k, "")) >= MIN_BODY:
            by_domain.setdefault(k.split("/")[0], []).append(k)
    for group in by_domain.values():
        rng.shuffle(group)
    picked: list[str] = []
    while len(picked) < n and any(by_domain.values()):
        for domain in sorted(by_domain):
            if by_domain[domain] and len(picked) < n:
                picked.append(by_domain[domain].pop())
    return picked


def _synth_prompt(title: str, body: str, *, language: str | None) -> str:
    where = (f"in {language}, translating the concept's name the way a student would "
             f"(keep established technical terms)") if language else "in the same language as the source"
    return (
        "You are simulating a second student who writes a note on the SAME concept as the "
        "source note below without having seen it. Write that note " + where + ".\n"
        "Rules: same concept and the same claims as the source; different wording throughout; "
        "the title must name the same concept but must not copy the source title.\n"
        "Return STRICT JSON with exactly these keys: "
        '{"title": "<the new title>", "excerpt": "<350 to 600 characters of note text, '
        'plain prose or short bullets, no headings>"}\n\n'
        f"SOURCE TITLE: {title}\nSOURCE NOTE:\n{body[:2500]}\n"
    )


def synthesize(sources: list[str], bodies: dict[str, str], *, lang: str) -> list[Incoming]:
    """One synthetic duplicate per source, cached by prompt (evals/.oracle), so
    a rerun costs nothing and routes the same prose."""
    from evals.oracle import cached_text
    from silica.config import CONFIG
    from silica.kernel.text.sanitize import parse_json

    other = OTHER_LANGUAGE.get(lang, "English")
    out: list[Incoming] = []
    for i, key in enumerate(sources):
        title = key.split("/")[-1]
        print(f"synthesize {i + 1}/{len(sources)} {title!r}", file=sys.stderr, flush=True)
        prompt = _synth_prompt(title, bodies[key], language=other if i % 2 else None)
        # reasoning=False: the same budget arithmetic as the dedup judge. A
        # hybrid model spends max_tokens on its trace and the JSON arrives cut
        # (finish=length), which here would silently drop the positive.
        raw = cached_text(CONFIG.model, [{"role": "user", "content": prompt}],
                          max_tokens=1024, temperature=0.0, reasoning=False)
        try:
            parsed, _ = parse_json(raw, strict=False)
        except Exception:
            parsed = None
        if not isinstance(parsed, dict) or not str(parsed.get("excerpt", "")).strip():
            continue
        out.append(Incoming(
            key=f"synthetic/{key}", title=str(parsed.get("title") or title).strip(),
            excerpt=" ".join(str(parsed["excerpt"]).split()), dups=frozenset({key}),
            exclude=frozenset(), synthetic=True,
        ))
    return out


def vault_incoming(keys: list[str], bodies: dict[str, str]) -> list[Incoming]:
    return [
        Incoming(key=k, title=k.split("/")[-1], excerpt=bodies[k][:EXCERPT_CHARS], dups=frozenset(),
                 exclude=frozenset({k, k + ".md"}), synthetic=False)
        for k in keys if k in bodies
    ]


def embed_all(records: list[Incoming], *, batch: int = 32) -> None:
    """The COLLISION query vector: `_note_text(name, excerpt)`, no folder."""
    from silica.agent.providers import get_embedder_or_none
    from silica.config import CONFIG
    from silica.kernel.recall.embed import _note_text

    embedder = get_embedder_or_none(CONFIG, "probe_collision_risk_coverage")
    if embedder is None:
        raise SystemExit("no embedder: COLLISION cannot route and this probe cannot measure")
    texts = [_note_text(r.title, r.excerpt) for r in records]
    for start in range(0, len(texts), batch):
        print(f"embed {start}/{len(texts)}", file=sys.stderr, flush=True)
        vecs = embedder.embed(texts[start:start + batch])
        if len(vecs) != len(texts[start:start + batch]):
            raise RuntimeError("embedder returned a ragged batch; refusing to mispair vectors")
        for rec, vec in zip(records[start:start + batch], vecs):
            rec.vec = vec


def route_all(records: list[Incoming], es, store) -> None:
    """Fill the scalars through the production entry point COLLISION calls."""
    from silica.kernel.recall.paths import is_inbox_path
    from silica.kernel.recall.relatedness import related_notes_for_query

    for rec in records:
        fused = related_notes_for_query(
            query_vec=rec.vec, query_text=f"{rec.title}\n{rec.excerpt}",
            embed_store=es, cooccur_store=store, k=TOP_N, exclude=set(rec.exclude),
        )
        best = next((c for c in fused if c.embed_score is not None and not is_inbox_path(c.path)), None)
        rec.scalars["rrf_winner"] = (best.path, best.embed_score) if best else (None, 0.0)

        hits = es.cosine_top_k(rec.vec, k=TOP_N + len(rec.exclude) + 8, exclude=set(rec.exclude))
        hits = [h for h in hits if not is_inbox_path(h["path"])][:TOP_N]
        rec.slate = tuple(h["path"] for h in hits)
        rec.scalars["cosine_top"] = (hits[0]["path"], float(hits[0]["score"])) if hits else (None, 0.0)


# ---------------------------------------------------------------------------
# policy evaluation (the real router, a different scalar)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Routed:
    """One incoming concept as `route_concept` sees it under one scalar: the
    score, and the two predicates that depend only on the target. Computed
    once per record, because the grid below asks the same question 2485
    times and `_names_agree` folds two titles each time it is asked."""

    score: float
    names_agree: bool
    is_hub: bool
    target_is_dup: bool
    has_dup: bool
    cold: bool          # no candidate at all: COLLISION keeps without a threshold


def routed(records: list[Incoming], scalar: str, hubs: set[str]) -> list[Routed]:
    from silica.router.states.collision import _names_agree

    out = []
    for rec in records:
        target, score = rec.scalars[scalar]
        if target is None:
            out.append(Routed(0.0, False, False, False, bool(rec.dups), True))
            continue
        out.append(Routed(
            score=score,
            names_agree=_names_agree(rec.title, target.split("/")[-1]),
            is_hub=target.removesuffix(".md") in hubs,
            target_is_dup=target.removesuffix(".md") in rec.dups,
            has_dup=bool(rec.dups),
            cold=False,
        ))
    return out


def evaluate(rows: list[Routed], *, tau_high: float, tau_low: float) -> dict:
    from silica.router.states.collision import route_concept

    auto = false_merge = leak = defer = 0
    for r in rows:
        decision = "keep" if r.cold else route_concept(
            r.score, names_agree=r.names_agree, is_hub=r.is_hub, tau_high=tau_high, tau_low=tau_low)
        if decision == "defer":
            defer += 1
            continue
        auto += 1
        if decision == "patch" and not r.target_is_dup:
            false_merge += 1
        elif decision == "keep" and r.has_dup:
            leak += 1
    n = len(rows)
    return {
        "tau_high": tau_high, "tau_low": tau_low,
        "coverage": round(auto / n, 4) if n else 0.0,
        "risk": round((false_merge + leak) / auto, 4) if auto else 0.0,
        "false_merges": false_merge, "leaks": leak, "deferred": defer, "auto": auto,
    }


def frontier(points: list[dict]) -> list[dict]:
    """Pareto frontier of (coverage up, risk down): a point survives when no
    other point covers at least as much at strictly lower risk."""
    best: dict[float, dict] = {}
    for p in points:
        cur = best.get(p["coverage"])
        if cur is None or p["risk"] < cur["risk"]:
            best[p["coverage"]] = p
    out: list[dict] = []
    floor = float("inf")
    for cov in sorted(best, reverse=True):
        if best[cov]["risk"] < floor:
            floor = best[cov]["risk"]
            out.append(best[cov])
    return sorted(out, key=lambda p: p["coverage"])


def aurc(front: list[dict]) -> float:
    """Area under risk(coverage) on the frontier, trapezoidal, from coverage 0
    (risk 0 by convention: deferring everything makes no mistake) to the
    frontier's last point."""
    area = 0.0
    prev_c, prev_r = 0.0, 0.0
    for p in front:
        area += (p["coverage"] - prev_c) * (p["risk"] + prev_r) / 2.0
        prev_c, prev_r = p["coverage"], p["risk"]
    return round(area, 5)


def _quantiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    xs = sorted(xs)
    at = lambda q: round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)
    return {"min": round(xs[0], 3), "p10": at(0.10), "p25": at(0.25), "p50": at(0.50), "p75": at(0.75)}


def target_agreement(records: list[Incoming], a: str, b: str) -> dict:
    """Paired on the duplicates: does scalar `a` hand the judge the source
    more often than scalar `b`? Exact McNemar on the discordant concepts."""
    from scipy.stats import binomtest

    def hit(rec: Incoming, scalar: str) -> bool:
        t = rec.scalars[scalar][0]
        return bool(t) and t.removesuffix(".md") in rec.dups

    dups = [r for r in records if r.dups]
    a_only = sum(1 for r in dups if hit(r, a) and not hit(r, b))
    b_only = sum(1 for r in dups if hit(r, b) and not hit(r, a))
    n = a_only + b_only
    return {"a": a, "b": b, "a_only": a_only, "b_only": b_only,
            "mcnemar_p": round(binomtest(a_only, n, 0.5).pvalue, 4) if n else 1.0}


def false_merges_at(records: list[Incoming], scalar: str, hubs: set[str], tau_high: float,
                    tau_low: float) -> list[dict]:
    from silica.router.states.collision import _names_agree, route_concept

    out = []
    for rec in records:
        target, score = rec.scalars[scalar]
        if target is None or rec.dups:
            continue
        d = route_concept(score, names_agree=_names_agree(rec.title, target.split("/")[-1]),
                          is_hub=target.removesuffix(".md") in hubs, tau_high=tau_high, tau_low=tau_low)
        if d == "patch":
            out.append({"incoming": rec.key, "target": target, "cosine": round(score, 3)})
    return out


def curve(records: list[Incoming], scalar: str, hubs: set[str], shipped: tuple[float, float]) -> dict:
    rows = routed(records, scalar, hubs)
    grid = [evaluate(rows, tau_high=th, tau_low=tl) for th in GRID for tl in GRID if tl <= th]
    front = frontier(grid)
    reach = {}
    for r in RISK_TARGETS:
        ok = [p for p in front if p["risk"] <= r]
        reach[str(r)] = max(ok, key=lambda p: p["coverage"]) if ok else None
    pos = [rec.scalars[scalar][1] for rec in records if rec.dups]
    neg = [rec.scalars[scalar][1] for rec in records if not rec.dups]
    return {
        "operating_point": evaluate(rows, tau_high=shipped[0], tau_low=shipped[1]),
        "shipped_false_merges": false_merges_at(records, scalar, hubs, shipped[0], shipped[1]),
        "aurc": aurc(front),
        "auc_dup_vs_distinct": _auc(pos, neg),
        "duplicate_score_quantiles": _quantiles(pos),
        "distinct_score_quantiles": _quantiles(neg),
        "coverage_at_risk": reach,
        "frontier": front,
        "target_is_dup": round(sum(1 for rec in records if rec.dups and rec.scalars[scalar][0]
                                   and rec.scalars[scalar][0].removesuffix(".md") in rec.dups)
                               / max(1, len(pos)), 4),
    }


# ---------------------------------------------------------------------------
# judge A/B: one candidate (production) vs a slate with an explicit none
# ---------------------------------------------------------------------------

SLATE_BODY_CHARS = 1500   # five bodies in one prompt; the single path hands the judge 8000
JUDGE_BAND_MAX = 60       # per class; the band is where the judge acts, so it is all that is judged


def _slate_prompt(rec: Incoming, candidates: list[tuple[str, str]]) -> str:
    n = len(candidates)
    head = (
        "You are a deduplication judge in a knowledge-base pipeline. An INCOMING CONCEPT "
        "(a name plus an excerpt distilled from new material) is compared against the "
        f"{n} CANDIDATE NOTES below, the vault's closest notes to it.\n"
        "Decide whether the incoming concept is the SAME concept as exactly one candidate "
        "(a true duplicate: the same topic under a different name, paraphrase, abbreviation "
        "or translation), CONTRADICTS one candidate (the same subject, conflicting claims), "
        "or is DISTINCT from all of them.\n"
        "Candidates are ranked by similarity of wording, which is not evidence of sameness: "
        "judge the claims, never the layout. Answer 0 (none of the above) is a normal "
        "answer, not a failure.\n"
        "Return STRICT JSON with exactly these keys: "
        '{"verdict": "duplicate" | "contradicts" | "distinct", '
        '"candidate": <1-based index of the matching candidate, 0 when distinct>, '
        '"rationale": "<one sentence>"}\n'
    )
    blocks = [f"---\nCANDIDATE {i} ({name}):\n{body[:SLATE_BODY_CHARS]}\n"
              for i, (name, body) in enumerate(candidates, 1)]
    return head + "\n" + "\n".join(blocks) + f"\n---\nINCOMING CONCEPT: {rec.title}\nEXCERPT:\n{rec.excerpt}\n"


def score_single(rec: Incoming, target: str | None, verdict: str) -> bool:
    """Production outcome for one judged concept. A duplicate whose handed
    candidate is not its source cannot end well: "distinct" writes a second
    note, "duplicate" merges into the wrong one."""
    if rec.dups:
        return target is not None and target.removesuffix(".md") in rec.dups and verdict == "duplicate"
    return verdict != "duplicate"


def score_slate(rec: Incoming, slate: tuple[str, ...], verdict: str, picked: int) -> bool:
    if rec.dups:
        return (verdict == "duplicate" and 1 <= picked <= len(slate)
                and slate[picked - 1].removesuffix(".md") in rec.dups)
    return verdict != "duplicate"


def judge_ab(records: list[Incoming], bodies: dict[str, str], *, tau_low: float,
             scalar: str = "cosine_top", seed: int = SEED) -> dict:
    """The band COLLISION defers (cosine-top over tau_low), judged two ways:
    the production judge on one candidate (the `scalar` target: the cosine-top
    once routing moves there, the RRF winner as shipped), and a slate of the
    cosine top-5 with an explicit none. Paired on the same concepts; McNemar
    on the discordant ones. The single arm calls the real `_decide_dedup`,
    uncached, so it judges exactly what ships."""
    from evals.oracle import cached_text
    from evals.paired_stats import paired
    from silica.capabilities.dedup import _decide_dedup
    from silica.config import CONFIG
    from silica.kernel.text.sanitize import parse_json

    rng = random.Random(seed)
    band = [r for r in records if r.scalars["cosine_top"][1] >= tau_low and r.scalars["cosine_top"][0]]
    pos = [r for r in band if r.dups][:JUDGE_BAND_MAX]
    neg = [r for r in band if not r.dups]
    rng.shuffle(neg)
    neg = neg[:max(len(pos), 1)]
    def slate_verdict(rec: Incoming, keys: tuple[str, ...]) -> tuple[str, int]:
        cands = [(k.split("/")[-1], bodies.get(k.removesuffix(".md"), "")) for k in keys]
        raw = cached_text(CONFIG.worker_model or CONFIG.model,
                          [{"role": "user", "content": _slate_prompt(rec, cands)}],
                          max_tokens=512, temperature=0.0, reasoning=False)
        try:
            parsed, _ = parse_json(raw, strict=False)
        except Exception:
            parsed = None
        if not isinstance(parsed, dict):
            parsed = {}
        try:
            picked = int(parsed.get("candidate") or 0)
        except (TypeError, ValueError):
            picked = 0
        return str(parsed.get("verdict", "distinct")), picked

    rows = []
    for i, rec in enumerate(pos + neg):
        print(f"judge {i + 1}/{len(pos) + len(neg)} {rec.title!r}", file=sys.stderr, flush=True)
        target, score = rec.scalars[scalar]
        single = _decide_dedup(
            CONFIG, concept=rec.title, excerpt=rec.excerpt,
            candidate_name=(target or "").split("/")[-1],
            candidate_body=bodies.get((target or "").removesuffix(".md"), "")[:8000],
            score=score, full_score=score,
        ) if target else None
        # Three arms, so the slate's FORM and its WORDING are told apart: the
        # production prompt on one candidate, the slate prompt on that same
        # one candidate (an explicit none, "ranked by wording"), the slate
        # prompt on the cosine top-5.
        one = (target,) if target else ()
        nil_verdict, nil_pick = slate_verdict(rec, one) if one else ("distinct", 0)
        sl_verdict, sl_pick = slate_verdict(rec, rec.slate)
        rows.append({
            "key": rec.key, "duplicate": bool(rec.dups), "score": round(score, 3),
            "unreachable_single": bool(rec.dups and not (target and target.removesuffix(".md") in rec.dups)),
            "single_ok": score_single(rec, target, single.verdict if single else "distinct"),
            "single_nil_ok": score_slate(rec, one, nil_verdict, nil_pick),
            "slate_ok": score_slate(rec, rec.slate, sl_verdict, sl_pick),
            "single_verdict": single.verdict if single else None,
            "single_nil_verdict": nil_verdict, "slate_verdict": sl_verdict, "slate_pick": sl_pick,
        })

    def doc(field: str) -> dict:
        return {"questions": [{"question_id": r["key"], "correct": r[field]} for r in rows]}

    def arm(name: str, subset: list[dict]) -> dict:
        n_pos = sum(1 for r in subset if r["duplicate"])
        return {
            "accuracy": round(sum(r[f"{name}_ok"] for r in subset) / max(1, len(subset)), 4),
            "dup_recall": round(sum(r[f"{name}_ok"] for r in subset if r["duplicate"]) / max(1, n_pos), 4),
            "false_duplicate": sum(1 for r in subset if not r["duplicate"] and r[f"{name}_verdict"] == "duplicate"),
            "n": len(subset), "duplicates": n_pos,
        }

    arms = ("single", "single_nil", "slate")
    n_pos = sum(1 for r in rows if r["duplicate"])
    return {
        "judged": len(rows), "duplicates": n_pos, "distinct": len(rows) - n_pos, "single_scalar": scalar,
        "band_tau_low": tau_low,
        "unreachable_for_single": sum(1 for r in rows if r["unreachable_single"]),
        "arms": {a: arm(a, rows) for a in arms},
        # the sub-band the shipped tau_low keeps: what lowering it would hand the judge
        "arms_under_shipped_tau_low": {a: arm(a, [r for r in rows if r["score"] < CONFIG.sim_threshold_low])
                                       for a in arms},
        "paired": {
            "slate_vs_single": paired(doc("slate_ok"), doc("single_ok")),
            "single_nil_vs_single": paired(doc("single_nil_ok"), doc("single_ok")),
            "slate_vs_single_nil": paired(doc("slate_ok"), doc("single_nil_ok")),
        },
        "rows": rows,
    }


def run(vault: Path, *, verbose: bool = True, judge: bool = False) -> dict:
    from silica.config import CONFIG
    from silica.kernel.link.health import iter_notes
    from silica.kernel.recall.paths import is_inbox_path

    store, es = _stores(vault)
    if es is None:
        raise SystemExit("no embedding index: COLLISION never routes on this vault")
    CONFIG.cooccur_bm25 = True
    hubs = _hub_keys(vault)
    shipped = (CONFIG.sim_threshold_high, CONFIG.sim_threshold_low)

    keys = [p.relative_to(vault).with_suffix("").as_posix() for p in iter_notes(vault)]
    keys = [k for k in keys if es.get_vec(k) is not None and not is_inbox_path(k)]
    bodies = _bodies(vault, keys)
    sources = sample_sources(keys, bodies)
    t0 = time.perf_counter()
    positives = synthesize(sources, bodies, lang=store.lang)
    t_synth = time.perf_counter() - t0
    records = vault_incoming(keys, bodies) + positives

    t0 = time.perf_counter()
    embed_all(records)
    t_embed = time.perf_counter() - t0
    route_all(records, es, store)

    out: dict = {
        "incoming": len(records), "vault_notes": len(records) - len(positives),
        "synthetic_duplicates": len(positives), "sources_sampled": len(sources),
        "hubs": len(hubs), "store_lang": store.lang,
        "shipped": {"tau_high": shipped[0], "tau_low": shipped[1]},
        "seconds": {"synthesize": round(t_synth, 1), "embed": round(t_embed, 1)},
        "slate_ceiling": round(sum(1 for r in positives if any(
            s.removesuffix(".md") in r.dups for s in r.slate)) / max(1, len(positives)), 4),
        "scalars": {scalar: curve(records, scalar, hubs, shipped) for scalar in ("rrf_winner", "cosine_top")},
    }
    out["target_agreement"] = target_agreement(records, "cosine_top", "rrf_winner")
    if judge:
        # Judge the band the improved routing would defer: the 1%-risk point of
        # the cosine-top curve when one exists, else the shipped tau_low.
        point = out["scalars"]["cosine_top"]["coverage_at_risk"].get("0.01")
        out["judge_ab"] = judge_ab(records, bodies, tau_low=point["tau_low"] if point else shipped[1])
    if verbose:
        _print(out)
    return out


def _print(out: dict) -> None:
    print(f"\nincoming {out['incoming']} = {out['vault_notes']} vault notes + {out['synthetic_duplicates']} "
          f"synthetic duplicates ({out['sources_sampled']} sampled, lang {out['store_lang']}); {out['hubs']} hubs; "
          f"shipped tau_high {out['shipped']['tau_high']} tau_low {out['shipped']['tau_low']}; "
          f"synth {out['seconds']['synthesize']}s embed {out['seconds']['embed']}s")
    print(f"slate ceiling: duplicate in the cosine top-{TOP_N} for {out['slate_ceiling']:.1%} of duplicates")
    ta = out["target_agreement"]
    print(f"target is the duplicate: {ta['a']} only {ta['a_only']}, {ta['b']} only {ta['b_only']}, "
          f"McNemar p {ta['mcnemar_p']}")
    for scalar, s in out["scalars"].items():
        op = s["operating_point"]
        print(f"\n{scalar}: target is the duplicate {s['target_is_dup']:.1%}; AUC dup-vs-distinct "
              f"{s['auc_dup_vs_distinct']:.3f}; AURC {s['aurc']:.5f}")
        print(f"  duplicate score quantiles {s['duplicate_score_quantiles']}; distinct {s['distinct_score_quantiles']}")
        for fm in s["shipped_false_merges"]:
            print(f"  shipped false merge: {fm['incoming']!r} -> {fm['target']!r} cos {fm['cosine']}")
        print(f"  shipped point: coverage {op['coverage']:.3f} risk {op['risk']:.4f} "
              f"(false merges {op['false_merges']}, leaks {op['leaks']}, deferred {op['deferred']})")
        for r, p in s["coverage_at_risk"].items():
            if p is None:
                print(f"  risk <= {float(r):.0%}: unreachable")
            else:
                print(f"  risk <= {float(r):.0%}: coverage {p['coverage']:.3f} at tau_high {p['tau_high']} "
                      f"tau_low {p['tau_low']} (false merges {p['false_merges']}, leaks {p['leaks']}, "
                      f"deferred {p['deferred']})")
    if "judge_ab" in out:
        j = out["judge_ab"]
        print(f"\njudge A/B on the band cosine-top >= {j['band_tau_low']} ({j['judged']} concepts: "
              f"{j['duplicates']} duplicates, {j['distinct']} distinct; {j['unreachable_for_single']} "
              f"duplicates whose {j['single_scalar']} target is not the source)")
        for name, a in j["arms"].items():
            u = j["arms_under_shipped_tau_low"][name]
            print(f"  {name:11s} accuracy {a['accuracy']:.3f}, duplicate recall {a['dup_recall']:.3f}, "
                  f"false duplicates {a['false_duplicate']}/{a['n'] - a['duplicates']}; under the shipped "
                  f"tau_low ({u['n']} concepts, {u['duplicates']} duplicates): accuracy {u['accuracy']:.3f}, "
                  f"duplicate recall {u['dup_recall']:.3f}, false duplicates {u['false_duplicate']}")
        for name, st in j["paired"].items():
            print(f"  {name}: McNemar p {st['mcnemar_p']} discordant {st['discordant']} delta ci95 {st['delta_ci95']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vault", required=True)
    ap.add_argument("--record", help="write the results JSON here")
    ap.add_argument("--judge", action="store_true",
                    help="also run the judge A/B on the deferred band (network, worker model)")
    args = ap.parse_args(argv)
    out = run(Path(args.vault).expanduser(), judge=args.judge)
    if args.record:
        Path(args.record).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
