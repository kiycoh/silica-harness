# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Can today's `/explain` anchor its claims to verbatim spans in the notes it cites?

Gate for lever C of `docs/spec-explain.md` (get-it idea 6: an anchor is the
verbatim tail of the sentence that introduces the concept, and it must occur
once). Today `/explain` (`silica/cli.py:1355`) cites a whole note: "the source
is somewhere in this file". This probe asks, after the fact, for the literal
span behind every claim and checks it with a substring. No judge, no embedding.

WHAT THIS MEASURES, AND WHAT IT DOES NOT

  It measures the ANCHORING of an exposition to the notes it declares. It does
  not measure retrieval quality and it does not exercise the agent loop:
  retrieval runs here through `facade_retrieve` (the same seam the chat tools
  use) and generation is one `call_llm` with the `/explain` contract. Going
  through `run_agent` is where three harness defects came from on the code-why
  probe of 2026-07-25 (arm B silently identical to arm A). The metric is about
  claims against cited bodies, so the choreography of tool calls cannot move it.

  Concepts are note TITLES sampled from the vault. That is the favourable case:
  the source note is certain to be retrievable. A failure here is therefore a
  floor, not a worst case.

THE DEFECT IN THE PRE-REGISTERED METRIC, DECLARED BEFORE THE RUN

  `spans_verified / spans_claimed` conflates INVENTED with SYNTHESISED. A claim
  that faithfully summarises three sentences has no single verbatim span, and
  would count as a failure while nothing is wrong. So the anchoring pass may
  return an empty span with a reason, and the reasons split the failures:

    synthesis  the claim compresses several sentences or several notes  LEGITIMATE
    absent     the note does not state it                              THE TARGET
    general    world knowledge, not from any note                      THE TARGET

  The pre-registered ratio stays the gate (moving it after a run is moving the
  goalposts). `grounded_rate` is reported beside it as the interpretation aid:
  every claim counts against it except the legitimate synthesis declines, so a
  claim that cites nothing (`uncited`) or cites a note nobody read
  (`unknown_note`) is an unanchored claim there. If the two disagree, the
  reasons say which to believe.

THE COMPARISON IS NORMALISED, AND WHY (found by the n=3 smoke, before the run)

  The spec's check is `ok = span in note_body`, character for character. On the
  smoke run 5 of 5 spans on one note were that note's prose exactly, with the
  `**` emphasis pairs and the blockquote `> ` dropped by the copy, and the
  strict check called all five invented citations. Strict `in` measures markup
  fidelity, not grounding, and would have reported a false C JUSTIFIED.

  So the check runs on `normalize()` (content characters, one space, no
  emphasis, no list or quote lead) and the strict ratio is printed beside it:
  their gap is the markup tax lever C would pay on real Obsidian notes.

GATES (pre-registered, before the run)

  C-JUSTIFIED   spans_verified / spans_claimed  < 0.85  → lever C is justified:
                more than one claim in seven is not anchorable to anything
                literal in the notes it declares.
  C-DEAD        the same ratio                  > 0.95  → span anchoring is
                ceremony, note-level citation was enough, lever C dies and F1
                leaves the spec.
  between       inconclusive. n is claims, not concepts, so raise n before
                reading anything into it.

  H1 (harness) declined / all claims <= 0.40. A model that declines its way out
     of the task drives the ratio to 1.0 and kills the lever for the wrong
     reason.
  H2 (harness) cross-note hit rate of verified spans < 0.10, checked in pure
     code: a span that also occurs verbatim in a DIFFERENT note of the same pack
     is boilerplate (frontmatter, a `## Related` heading, a template line), not
     evidence. If this bites, the substring check is not discriminating notes
     and every number above is inflated.

    OPENROUTER_PROVIDER=DeepInfra uv run python -m evals.probe_explain_spans \\
      --vault ~/Documents/Obsidian/test --n 20

Sibling: `evals/probe_explain_rubric.py` (whether the vault can expose at all;
KILL on its G2 on 2026-07-26, with `application` 0 on 27 of 60 notes).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from pydantic import BaseModel

# --- Gates. Declared here, before the run. Not tuned afterwards. ------------
GATE_JUSTIFIED = 0.85     # below: lever C earns its keep
GATE_DEAD = 0.95          # above: lever C is ceremony
GATE_MAX_DECLINED = 0.40  # H1: the model routing around the task
GATE_MAX_CROSS = 0.10     # H2: spans that do not discriminate notes

K = 6                # notes retrieved per concept, as /explain's "top matches"
BODY_CAP = 4000      # chars of note body sent per note
SPAN_MIN = 30        # get-it's span floor; reported, not gated
SPAN_MAX = 120

DECLINE_REASONS = ("synthesis", "absent", "general")


# ---------------------------------------------------------------------------
# Comparison form. Found by the n=3 smoke, fixed before the real run.
# ---------------------------------------------------------------------------

_LINE_LEAD = re.compile(r"^[ \t]*(?:>+[ \t]*)*(?:[-*+][ \t]+|\d+[.)][ \t]+|#{1,6}[ \t]+)?")
_WIKI = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_INLINE = re.compile(r"\*\*|__|~~|==|`+|\*|_")


def normalize(text: str) -> str:
    """Content characters only: no emphasis markers, no list/quote lead, one space.

    The strict `ok = span in note_body` of the spec measures markup fidelity,
    not grounding. Measured on the n=3 smoke: 5 of 5 spans on one note were the
    note's prose character for character with the `**` pairs and the blockquote
    `> ` dropped, and the strict check called every one of them an invented
    citation. That would have reported a false C JUSTIFIED.

    Removing markers cannot turn one sentence into a different one, so the
    anti-fabrication guarantee survives the loosening; H2 (a span must not also
    occur in another note of the pack) is the control that says the looser form
    still discriminates notes. The strict ratio stays in the output beside the
    normalised one, because the gap between them is the cost lever C would pay
    on real Obsidian notes.
    """
    lines = []
    for line in text.split("\n"):
        line = _LINE_LEAD.sub("", line, count=1)
        line = _WIKI.sub(lambda m: m.group(2) or m.group(1), line)
        lines.append(_INLINE.sub("", line))
    return " ".join(" ".join(lines).split())


def _nbody(entry: dict) -> str:
    """Normalised body, memoised on the entry (one pass per note, not per claim)."""
    n = entry.get("nbody")
    if n is None:
        n = entry["nbody"] = normalize(entry["body"])
    return n


# ---------------------------------------------------------------------------
# Prompts. The first is today's /explain contract, verbatim in substance.
# ---------------------------------------------------------------------------

# A paraphrase, not the template itself: the shipped one tells an agent to search
# the vault and call tools, which is not the setting here (the notes are handed
# over directly). The clauses that can move the measurement are copied verbatim
# instead, and `test_probe_explain_spans.py` fails if the product changes them
# and this copy does not follow, so a run cannot silently measure the old prompt.
ATTRIBUTION_CLAUSE = "Attribute a claim to a note only if that note states it."

EXPOSE_SYSTEM = f"""You explain a concept grounded in a personal knowledge vault, \
for a practitioner: clear, correct, minimal jargon.

You are given the notes a search over that vault returned. Explain the concept \
in prose, citing every note you drew on as a [[wikilink]] using its exact title. \
If the notes have nothing relevant, say so plainly: do not silently answer from \
general knowledge alone.

{ATTRIBUTION_CLAUSE} A note that merely sits near the topic is not a source for \
it, and a point no note supports goes in its own sentence, marked as not coming \
from the vault.

Write the explanation and nothing else."""

ANCHOR_SYSTEM = """You are given an EXPLANATION that was written from a set of \
NOTES, and the notes themselves. Split the explanation into its factual claims \
and anchor each one to the note it came from.

For every claim, `span` is the passage OF THE NOTE that supports it, copied \
CHARACTER FOR CHARACTER from the note body: not corrected, not re-punctuated, \
not stitched together from two places. It is the note's own wording, not a copy \
of the claim, so a claim the note states in different words is still anchorable \
to the note's sentence. Aim for 30 to 120 characters.

If no single passage of one note supports the claim, set `span` to "" and give \
`reason`:

  synthesis  the claim compresses several sentences, or several notes
  absent     that note does not support the claim
  general    the claim is world knowledge, not from any note

An empty span with an honest reason is a CORRECT answer and costs nothing. \
Inventing a span that is not in the note body is the one thing that is wrong \
here: the span will be checked mechanically against the note.

One row per claim, in the order the claims appear in the explanation. Do not \
add claims the explanation does not make. `note` is the note title exactly as \
the explanation cited it, without brackets, or "" if the claim cites no note.

Return one JSON object matching the schema. No prose outside it."""


class Claim(BaseModel):
    text: str
    note: str = ""
    span: str = ""
    reason: str = ""


class AnchorResult(BaseModel):
    claims: list[Claim]


# ---------------------------------------------------------------------------
# Vault bodies
# ---------------------------------------------------------------------------

def load_bodies(vault: Path) -> dict[str, dict]:
    """{lookup key: {key, title, body}} for every note, frontmatter stripped.

    Frontmatter goes because a span verified against `tags: [x]` is not
    evidence, and because it is the shape most likely to repeat verbatim across
    notes (which H2 would then flag as a harness failure rather than as the
    templating artifact it is). Keys are indexed by relpath with and without
    the suffix, and by lowercased title, because a wikilink cites the title
    while retrieval returns a path.
    """
    from silica.kernel.write import frontmatter
    from silica.kernel.link.health import iter_notes

    out: dict[str, dict] = {}
    for p in iter_notes(vault):
        rel = p.relative_to(vault)
        *_, body = frontmatter.split(p.read_text(encoding="utf-8", errors="replace"))
        if not body.strip():
            continue
        entry = {"key": rel.with_suffix("").as_posix(), "title": p.stem, "body": body}
        for k in (rel.as_posix(), rel.with_suffix("").as_posix(), p.stem.lower()):
            out.setdefault(k, entry)
    return out


def resolve(cited: str, bodies: dict[str, dict]) -> dict | None:
    """A note as the explanation cited it → its entry, or None.

    Handles `[[Title]]`, `[[Title|alias]]`, `path/Title.md` and bare titles.
    """
    s = (cited or "").strip().strip("[]").split("|")[0].strip()
    if not s:
        return None
    for cand in (s, s.removesuffix(".md"), s.lower(), s.removesuffix(".md").lower(),
                 s.rsplit("/", 1)[-1].removesuffix(".md").lower()):
        hit = bodies.get(cand)
        if hit is not None:
            return hit
    return None


# ---------------------------------------------------------------------------
# One concept, end to end
# ---------------------------------------------------------------------------

def retrieve_pack(concept: str, bodies: dict[str, dict]) -> tuple[list[dict], bool]:
    """The notes `/explain` would read for `concept`, and whether the embed leg
    was live. Memory-lane hits are dropped: their paths are relative to another
    vault, so their bodies are not in `bodies` and could not be verified."""
    from silica.kernel.recall.perception import facade_retrieve

    results, query_vec = facade_retrieve(concept, k=K)
    pack: list[dict] = []
    for r in results or []:
        if r.origin != "vault":
            continue
        entry = bodies.get(r.path) or bodies.get(r.path.removesuffix(".md"))
        if entry is not None and entry not in pack:
            pack.append(entry)
    return pack, query_vec is not None


def _blocks(pack: list[dict]) -> str:
    return "\n\n".join(
        f"===== NOTE: {nt['title']} =====\n{nt['body'][:BODY_CAP]}" for nt in pack)


def expose(concept: str, pack: list[dict], model: str) -> str:
    from silica.agent.llm import call_llm

    resp = call_llm(
        model=model,
        messages=[
            {"role": "system", "content": EXPOSE_SYSTEM},
            {"role": "user", "content": f"CONCEPT: {concept}\n\n{_blocks(pack)}"},
        ],
        temperature=0.0,
    )
    return (resp.text or "").strip()


def anchor(explanation: str, pack: list[dict], model: str) -> list[dict]:
    """The claims of `explanation` with their spans. [] when the call is unusable.

    An unparseable batch returns nothing rather than a row per claim with an
    empty span: a decline is a real answer on this instrument, so faking one
    would push H1 towards a harness failure and look like a finding.
    """
    from silica.agent.llm import call_llm
    from silica.kernel.text.sanitize import parse_json

    resp = call_llm(
        model=model,
        messages=[
            {"role": "system", "content": ANCHOR_SYSTEM},
            {"role": "user", "content": f"===== EXPLANATION =====\n{explanation}\n\n"
                                        f"{_blocks(pack)}\n\nAnchor every claim. JSON only."},
        ],
        response_format=AnchorResult,
        temperature=0.0,
    )
    try:
        parsed, _ = parse_json(resp.text or "", strict=False)
    except Exception:
        return []
    rows = parsed.get("claims") if isinstance(parsed, dict) else None
    return [r for r in (rows or []) if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# Verification: pure, no LLM, no embedding
# ---------------------------------------------------------------------------

BUCKETS = ("verified", "invented", "declined", "uncited", "unknown_note")


def classify(rows: list[dict], pack: list[dict], bodies: dict[str, dict]) -> list[dict]:
    """One verdict per claim. Buckets, checked in this order:

      declined      no span offered, with a reason
      uncited       a span, but the claim names no note at all
      unknown_note  the note it names is not one of the notes that were read
      verified      the span occurs in that note's body (normalised comparison)
      invented      a span was offered and is not there

    The decline test comes FIRST on purpose. Ordering it after the note tests
    (as the first draft did) sent every decline that also named no note into
    `uncited`: on the n=20 run that was 14 of 22 declines, it zeroed the
    `synthesis` count and so dropped the secondary `grounded_rate` from 0.874 to
    0.848, across the 0.85 line, for a bucketing reason and not a measured one.

    `strict` marks a verified span that also matches character for character,
    so the gap between the two counts is the markup tax. `cross_note` marks a
    verified span that also occurs in a different note of the same pack:
    verbatim without discriminating (H2).
    """
    out: list[dict] = []
    for row in rows:
        cited = str(row.get("note", ""))
        span = str(row.get("span", "") or "")
        reason = str(row.get("reason", "") or "").strip().lower()
        entry = resolve(cited, bodies)
        rec = {"note": cited, "span": span[:200], "reason": reason,
               "chars": len(span), "cross_note": False, "strict": False}
        nspan = normalize(span)
        if not span.strip():
            rec["bucket"] = "declined"
            if reason not in DECLINE_REASONS:
                rec["reason"] = "unstated"
        elif not cited.strip().strip("[]"):
            rec["bucket"] = "uncited"
        elif entry is None or entry not in pack:
            rec["bucket"] = "unknown_note"
        elif not nspan:
            # A span of nothing but markup normalises to "", and "" is a
            # substring of everything: it must never verify.
            rec["bucket"] = "invented"
        else:
            hits = _nbody(entry).count(nspan)
            if hits:
                rec["bucket"] = "verified"
                rec["occurrences"] = hits
                rec["strict"] = span in entry["body"]
                rec["short"] = len(span) < SPAN_MIN
                rec["long"] = len(span) > SPAN_MAX
                rec["cross_note"] = any(
                    other is not entry and nspan in _nbody(other) for other in pack)
            else:
                rec["bucket"] = "invented"
        out.append(rec)
    return out


def gates(claims: list[dict]) -> dict:
    n = {b: sum(1 for c in claims if c["bucket"] == b) for b in BUCKETS}
    reasons = {r: sum(1 for c in claims if c["bucket"] == "declined" and c["reason"] == r)
               for r in (*DECLINE_REASONS, "unstated")}

    claimed = n["verified"] + n["invented"]
    ratio = (n["verified"] / claimed) if claimed else 0.0
    strict_hits = sum(1 for c in claims if c.get("strict"))
    strict_ratio = (strict_hits / claimed) if claimed else 0.0

    # Secondary: every claim counts against it except the legitimate synthesis
    # declines. An uncited or mis-cited claim is an unanchored claim too.
    total = sum(n.values())
    grounded_den = total - reasons["synthesis"]
    grounded = (n["verified"] / grounded_den) if grounded_den else 0.0

    declined_rate = (n["declined"] / total) if total else 0.0
    cross = sum(1 for c in claims if c.get("cross_note"))
    cross_rate = (cross / n["verified"]) if n["verified"] else 0.0

    h1 = declined_rate <= GATE_MAX_DECLINED
    h2 = cross_rate < GATE_MAX_CROSS
    if not (h1 and h2):
        verdict = "HARNESS"
    elif not claimed:
        verdict = "NO DATA"
    elif ratio < GATE_JUSTIFIED:
        verdict = "C JUSTIFIED"
    elif ratio > GATE_DEAD:
        verdict = "C DEAD"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "n_claims": total, "buckets": n, "decline_reasons": reasons,
        "spans_claimed": claimed, "spans_verified": n["verified"],
        "verified_ratio": ratio, "strict_ratio": strict_ratio,
        "strict_verified": strict_hits, "grounded_rate": grounded,
        "declined_rate": declined_rate, "cross_note_hits": cross,
        "cross_note_rate": cross_rate,
        "span_chars_median": statistics.median(
            [c["chars"] for c in claims if c["bucket"] == "verified"]) if n["verified"] else 0,
        "short_spans": sum(1 for c in claims if c.get("short")),
        "long_spans": sum(1 for c in claims if c.get("long")),
        "ambiguous_spans": sum(1 for c in claims if c.get("occurrences", 0) > 1),
        "H1_not_declining_out": h1, "H2_spans_discriminate": h2,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------

def run(vault: Path, *, n: int, model: str) -> dict:
    from evals._shared import assert_reranker_live, provenance, warn_unpinned_provider
    from evals.golden.runner import _open_stores, vault_digest
    from evals.probe_explain_rubric import sample_notes
    from silica.config import CONFIG

    warn_unpinned_provider(model, CONFIG.openrouter_provider)
    store, _embed = _open_stores(vault)
    assert_reranker_live(CONFIG)
    digest, total = vault_digest(vault)
    bodies = load_bodies(vault)
    concepts = [nt["title"] for nt in sample_notes(vault, store, n, seed=digest)]
    print(f"vault {vault}  ({total} notes, {digest[:19]}…)  model {model}")
    print(f"{len(concepts)} concepts (note titles), k={K} notes retrieved each")

    print("\n1. EXPOSITIONS")
    rows: list[dict] = []
    per_concept: list[dict] = []
    embed_live = None
    for i, concept in enumerate(concepts):
        try:
            pack, live = retrieve_pack(concept, bodies)
            embed_live = live if embed_live is None else (embed_live and live)
            if not pack:
                print(f"   {i:>3} {concept[:52]:<52} no notes retrieved — skipped")
                per_concept.append({"concept": concept, "pack": 0, "claims": 0})
                continue
            text = expose(concept, pack, model)
            claims = classify(anchor(text, pack, model), pack, bodies)
        except Exception as exc:  # one dead concept must not lose the run
            print(f"   {i:>3} {concept[:52]:<52} FAILED ({type(exc).__name__}: {exc})")
            per_concept.append({"concept": concept, "pack": 0, "claims": 0,
                                "error": f"{type(exc).__name__}: {exc}"})
            continue
        for c in claims:
            c["concept"] = concept
        rows.extend(claims)
        ok = sum(1 for c in claims if c["bucket"] == "verified")
        print(f"   {i:>3} {concept[:52]:<52} {len(pack)} notes  "
              f"{len(claims):>2} claims  {ok:>2} verified")
        per_concept.append({"concept": concept, "pack": len(pack), "claims": len(claims),
                            "verified": ok, "chars": len(text)})

    g = gates(rows)
    b, r = g["buckets"], g["decline_reasons"]

    print(f"\n   embed leg: {'live' if embed_live else 'OFF (co-occurrence only)'}")
    print("\n2. CLAIMS")
    print("   " + "   ".join(f"{k.replace('_', ' ')} {b[k]:>3}" for k in BUCKETS)
          + f"   ({g['n_claims']} total)")
    print("   declines: " + ("  ".join(f"{k} {v}" for k, v in r.items() if v) or "none"))
    print(f"   verified spans: median {g['span_chars_median']:.0f} chars, "
          f"{g['short_spans']} under {SPAN_MIN}, {g['long_spans']} over {SPAN_MAX}, "
          f"{g['ambiguous_spans']} non-unique")

    print("\n3. GATES")
    print(f"   primary  spans_verified/spans_claimed  {g['verified_ratio']:.3f}  "
          f"({g['spans_verified']}/{g['spans_claimed']})")
    print(f"            < {GATE_JUSTIFIED} justifies lever C, > {GATE_DEAD} kills it")
    print(f"   markup tax: strict `span in body` would read {g['strict_ratio']:.3f} "
          f"({g['strict_verified']}/{g['spans_claimed']})")
    print(f"   secondary grounded_rate (drops synthesis declines) "
          f"{g['grounded_rate']:.3f}")
    print(f"   H1 declining out   {g['declined_rate']:.3f} (<= {GATE_MAX_DECLINED}) → "
          f"{'ok' if g['H1_not_declining_out'] else 'HARNESS BUG'}")
    print(f"   H2 spans separate  cross-note {g['cross_note_rate']:.3f} "
          f"({g['cross_note_hits']} spans, < {GATE_MAX_CROSS}) → "
          f"{'ok' if g['H2_spans_discriminate'] else 'HARNESS BUG'}")
    print(f"\n   VERDICT: {g['verdict']}")

    return {
        "provenance": provenance(vault),
        "vault": {"path": str(vault), "digest": digest, "notes": total},
        "config": {"n": len(concepts), "k": K, "model": model, "body_cap": BODY_CAP,
                   "provider_pin": CONFIG.openrouter_provider, "embed_leg": embed_live},
        "gates": g,
        "concepts": per_concept,
        "claims": rows,
    }


def main(argv=None) -> int:
    from evals.golden.runner import resolve_vault
    from silica.config import CONFIG

    ap = argparse.ArgumentParser(prog="python -m evals.probe_explain_spans")
    ap.add_argument("--vault")
    ap.add_argument("--n", type=int, default=20,
                    help="concepts to expose (default 20; the unit of measure is "
                         "the claim, and each concept yields several)")
    ap.add_argument("--model", default=None, help="model (default CONFIG.model)")
    ap.add_argument("--json", default="bench/explain_spans.json")
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
