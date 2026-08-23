# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""W6 premise + gate — at spoke-creation time, is the concept already discussed
inside another note's BODY, and is that body a plausible home? (spec §10)

Throwaway instrument, NOT product code. Zero-LLM, zero-API: the co-occurrence
stem postings as the candidate index and the driver's own `mentions_in` /
`build_title_trie` for the phrase confirmation, so the probe measures the same
two primitives the lever consults.

Replay shape: every note in the vault is treated in turn as the INCOMING spoke
and the rest of the vault as what already exists. That over-counts nothing the
lever would not see (the lookup is body text, which does not depend on the
spoke's own note existing), and it under-counts within-run ordering (a spoke
written at position 40 also sees spokes 1-39).

The premise this must establish before any code ships, given W2 died on a false
premise:

  1. FIRE RATE — how many spoke titles have a confirmed body mention somewhere
     else in the vault. Near zero closes the lever.
  2. NEW INFORMATION — of those, how many the shipped C3 band-2 `near_titles`
     gate would NOT already have deferred. A lever that only re-flags what
     validate.py already defers is inert.
  3. STRANDING — of those, how many mentions are NOT already wikilinked to the
     spoke. A mention that is already a link means the graph connects the two
     and nothing is stranded, so those fires are the false-positive class.
  4. GENERIC OVER-FIRING — the df distribution of the titles that fire, which
     is what the W3 specificity floor has to separate. Reported as `1/df` of the
     title's RAREST stem (its most discriminative token) plus the confirmed-hit
     count, so the floor can be chosen from data instead of taste.

  uv run python -m evals.probe_w6_mention_home --vault ~/Documents/Obsidian/test
  uv run python -m evals.probe_w6_mention_home --vault <v> --limit 200 --out w6.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_CAND_CAP = 200   # bodies read per title; truncation is REPORTED, never silent
_SAMPLE = 40      # fires dumped for the manual audit
_FLOORS = (3, 5, 10, 20, 40)   # df ceilings for the W3 specificity sweep
_AUDIT_FLOOR = 10              # the ceiling whose routed cases get dumped


def _sample(keys: list[str], limit: int | None) -> list[str]:
    """Deterministic even-stride subsample (probe_w2 convention)."""
    if not limit or limit >= len(keys):
        return keys
    step = len(keys) / limit
    return [keys[int(i * step)] for i in range(limit)]


def _title_stems(title: str, lang: str) -> set[str]:
    from silica.kernel.recall.cooccurrence import tokenize

    return {
        stem
        for sentence in tokenize(title, stem_lang=lang, stopword_lang=lang)
        for stem, _surface in sentence
    }


def _headings(body: str) -> list[str]:
    """Heading texts of a body (markers stripped)."""
    import re

    return [m.group(1).strip() for m in re.finditer(r"^#{1,6}\s+(.+)$", body, re.M)]


def _link_targets(body: str) -> set[str]:
    """Lower-cased basenames of every wikilink target in `body`."""
    from silica.kernel.link.ast import extract_links

    out = set()
    for t in extract_links(body):
        name = t.split("|")[0].split("#")[0].strip().replace("\\", "/")
        out.add(name.rsplit("/", 1)[-1].removesuffix(".md").lower())
    return out


# The title trie the driver used for its mention index (deleted from the product
# 2026-08-23 with `mentions_of`, which had no caller). Kept here verbatim so the
# probe still measures what it measured: a title matches when it occurs as a
# substring starting at a word boundary.
_TITLE = "\x00"


def _is_word_char(c: str) -> bool:
    return ("a" <= c <= "z") or ("0" <= c <= "9")


def build_title_trie(title_lowers) -> dict:
    root: dict = {}
    for title_lower in title_lowers:
        if len(title_lower) < 2:
            continue
        node = root
        for ch in title_lower:
            node = node.setdefault(ch, {})
        node[_TITLE] = title_lower
    return root


def mentions_in(content_lower: str, trie: dict) -> set[str]:
    found: set[str] = set()
    n = len(content_lower)
    for i in range(n):
        if not _is_word_char(content_lower[i]):
            continue
        if i and _is_word_char(content_lower[i - 1]):
            continue
        node = trie
        j = i
        while j < n:
            nxt = node.get(content_lower[j])
            if nxt is None:
                break
            node = nxt
            title = node.get(_TITLE)
            if title is not None:
                found.add(title)
            j += 1
    return found


def run(vault: Path, *, limit: int | None = None, verbose: bool = False) -> dict:
    from silica.kernel.link.health import iter_notes
    from silica.kernel.recall.cooccurrence import cooccur_key
    from silica.kernel.recall.paths import is_inbox_path
    from silica.kernel.text.title import NEAR_BAND, near_titles, title_key
    from evals.golden.runner import _open_stores

    store, _embed = _open_stores(vault)
    postings = store.stem_postings()
    if not postings:
        raise SystemExit("co-occurrence index absent for this vault — nothing to measure")
    lang = getattr(store, "lang", None) or "english"
    n_notes = len(store)

    # One pass over the corpus: key -> (title, folder, body). Bodies are held in
    # memory (a vault is tens of MB) so each candidate is read once, not once
    # per firing title.
    bodies: dict[str, str] = {}
    titles: dict[str, str] = {}
    folders: dict[str, str] = {}
    for p in iter_notes(vault):
        rel = p.relative_to(vault).as_posix()
        if is_inbox_path(rel):
            continue
        key = cooccur_key(rel)
        try:
            bodies[key] = p.read_text(encoding="utf-8")
        except Exception:
            continue
        titles[key] = p.stem
        folders[key] = rel.rsplit("/", 1)[0] if "/" in rel else ""

    by_folder: dict[str, list[str]] = {}
    for key, folder in folders.items():
        by_folder.setdefault(folder, []).append(titles[key])

    keys = _sample(sorted(k for k in titles if k in postings or True), limit)

    stats = {
        "notes_indexed": n_notes,
        "notes_replayed": 0,
        "no_stems": 0,             # title is all stopwords/punctuation
        "no_candidates": 0,        # empty postings intersection
        "truncated": 0,            # candidate set over the read cap
        "fired": 0,                # >=1 confirmed body mention elsewhere
        "fired_new": 0,            # ... and near_titles would NOT have deferred it
        "fired_new_stranded": 0,   # ... and at least one mention is unlinked
        "already_linked_only": 0,  # fires whose every mention is already a wikilink
    }
    fires: list[dict] = []
    floor_sweep: dict[int, dict] = {}
    floor_cases: list[dict] = []
    heading_homes: list[dict] = []
    dense_homes: list[dict] = []
    hit_counts: list[int] = []
    spec_fired: list[float] = []
    spec_quiet: list[float] = []

    for key in keys:
        title = titles[key]
        stems = _title_stems(title, lang)
        stats["notes_replayed"] += 1
        if not stems:
            stats["no_stems"] += 1
            continue
        dfs = {s: len(postings.get(s, {})) for s in stems}
        min_df = min(dfs.values())
        specificity = 1.0 / min_df if min_df else 1.0

        cand: set[str] | None = None
        for s in stems:
            plist = set(postings.get(s, {}))
            cand = plist if cand is None else (cand & plist)
            if not cand:
                break
        cand = {c for c in (cand or set()) if c != key and c in bodies}
        if not cand:
            stats["no_candidates"] += 1
            spec_quiet.append(specificity)
            continue

        ordered = sorted(cand)
        if len(ordered) > _CAND_CAP:
            stats["truncated"] += 1
            ordered = ordered[:_CAND_CAP]

        trie = build_title_trie([title.lower()])
        hits: list[tuple[str, bool, str]] = []   # (cand_key, already_linked, line)
        tkey = title_key(title)
        for ck in ordered:
            body = bodies[ck]
            if not mentions_in(body.lower(), trie):
                continue
            linked = title.lower() in _link_targets(body)
            line = next(
                (ln.strip() for ln in body.splitlines() if title.lower() in ln.lower()),
                "",
            )
            hits.append((ck, linked, line[:200]))
            # Arm B: the concept is a SECTION of the candidate, not a passing
            # prose mention — the shape that actually duplicates content.
            heads = [h for h in _headings(body) if title_key(h) == tkey]
            if heads:
                heading_homes.append({"title": title, "cand": ck, "heading": heads[0]})
            # Arm C: mentioned repeatedly in one body — "discussed", not "named".
            if body.lower().count(title.lower()) >= 3:
                dense_homes.append({"title": title, "cand": ck,
                                    "n": body.lower().count(title.lower())})

        if not hits:
            spec_quiet.append(specificity)
            continue

        stats["fired"] += 1
        hit_counts.append(len(hits))
        spec_fired.append(specificity)
        # W3 floor sweep: at each df ceiling, does the lever still fire, and are
        # the surviving fires the stranded (unlinked) ones or the redundant ones?
        for ceil in _FLOORS:
            if min_df <= ceil:
                b = floor_sweep.setdefault(ceil, {"titles": 0, "stranded": 0, "hits": 0})
                b["titles"] += 1
                b["hits"] += len(hits)
                if any(not linked for _c, linked, _l in hits):
                    b["stranded"] += 1
                if ceil == _AUDIT_FLOOR:
                    floor_cases.append({
                        "title": title, "min_df": min_df, "hits": len(hits),
                        "homes": [{"cand": c, "linked": lk, "line": ln}
                                  for c, lk, ln in hits[:2]],
                    })

        # Baseline: would the shipped C3 band-2 gate already defer this write?
        siblings = [t for t in by_folder.get(folders[key], []) if t != title]
        near = near_titles(title, siblings, band=NEAR_BAND)
        if near:
            continue
        stats["fired_new"] += 1
        stranded = [h for h in hits if not h[1]]
        if stranded:
            stats["fired_new_stranded"] += 1
        else:
            stats["already_linked_only"] += 1
        fires.append({
            "title": title,
            "key": key,
            "min_df": min_df,
            "specificity": round(specificity, 5),
            "hits": len(hits),
            "stranded": len(stranded),
            "homes": [
                {"cand": ck, "linked": linked, "line": line}
                for ck, linked, line in (stranded or hits)[:3]
            ],
        })

    def _pct(a: int, b: int) -> float:
        return round(100.0 * a / b, 2) if b else 0.0

    rep = {
        **stats,
        "fire_rate_pct": _pct(stats["fired"], stats["notes_replayed"]),
        "new_rate_pct": _pct(stats["fired_new"], stats["notes_replayed"]),
        "stranded_rate_pct": _pct(stats["fired_new_stranded"], stats["notes_replayed"]),
        "hits_mean": round(sum(hit_counts) / len(hit_counts), 2) if hit_counts else 0.0,
        "hits_p90": sorted(hit_counts)[int(0.9 * (len(hit_counts) - 1))] if hit_counts else 0,
        "hits_max": max(hit_counts) if hit_counts else 0,
        "min_df_fired_mean": round(sum(1 / s for s in spec_fired) / len(spec_fired), 1) if spec_fired else 0.0,
        "min_df_quiet_mean": round(sum(1 / s for s in spec_quiet) / len(spec_quiet), 1) if spec_quiet else 0.0,
        "heading_home_pairs": len(heading_homes),
        "heading_home_titles": len({h["title"] for h in heading_homes}),
        "heading_home_rate_pct": _pct(len({h["title"] for h in heading_homes}), stats["notes_replayed"]),
        "dense_home_pairs": len(dense_homes),
        "dense_home_titles": len({h["title"] for h in dense_homes}),
        "dense_home_rate_pct": _pct(len({h["title"] for h in dense_homes}), stats["notes_replayed"]),
        "heading_home_sample": heading_homes[:_SAMPLE],
        "dense_home_sample": sorted(dense_homes, key=lambda h: -h["n"])[:_SAMPLE],
        "floor_sweep": {
            str(c): {**v, "rate_pct": _pct(v["titles"], stats["notes_replayed"])}
            for c, v in sorted(floor_sweep.items())
        },
        "floor_cases": floor_cases,
        "audit_sample": sorted(fires, key=lambda f: -f["hits"])[:_SAMPLE],
    }

    if verbose:
        print(f"corpus: {n_notes} indexed notes, {stats['notes_replayed']} replayed, lang={lang}")
        print(f"no stems: {stats['no_stems']}   no candidates: {stats['no_candidates']}"
              f"   truncated at {_CAND_CAP}: {stats['truncated']}")
        print(f"FIRE      {stats['fired']} ({rep['fire_rate_pct']}%)")
        print(f"  NEW     {stats['fired_new']} ({rep['new_rate_pct']}%)  "
              f"[near_titles would already defer the rest]")
        print(f"  STRANDED{stats['fired_new_stranded']} ({rep['stranded_rate_pct']}%)  "
              f"already-linked-only: {stats['already_linked_only']}")
        print(f"hits/title: mean {rep['hits_mean']}  p90 {rep['hits_p90']}  max {rep['hits_max']}")
        print(f"rarest-stem df: fired mean {rep['min_df_fired_mean']}  "
              f"quiet mean {rep['min_df_quiet_mean']}")
        print(f"ARM B heading-home: {rep['heading_home_titles']} titles "
              f"({rep['heading_home_rate_pct']}%), {rep['heading_home_pairs']} pairs")
        print(f"ARM C dense-home (>=3 mentions in one body): {rep['dense_home_titles']} titles "
              f"({rep['dense_home_rate_pct']}%), {rep['dense_home_pairs']} pairs")
        print("\n-- W3 floor sweep (df of the title's rarest stem <= ceiling) --")
        for c, v in rep["floor_sweep"].items():
            print(f"  df<={c:>3}: {v['titles']:>4} titles ({v['rate_pct']}%)  "
                  f"stranded {v['stranded']}  mean hits {round(v['hits']/max(v['titles'],1),2)}")
        print(f"\n-- routed cases at df<={_AUDIT_FLOOR} (manual audit, first 25) --")
        for f in rep["floor_cases"][:25]:
            print(f"  {f['title']!r} df={f['min_df']} hits={f['hits']}")
            for h in f["homes"][:1]:
                print(f"      {'LINKED' if h['linked'] else 'stranded'} {h['cand']}: {h['line'][:110]}")
        print("\n-- heading homes (concept is a SECTION of another note) --")
        for h in rep["heading_home_sample"][:20]:
            print(f"  {h['title']!r}  in  {h['cand']}   heading={h['heading']!r}")
        print("\n-- top fires by hit count --")
        for f in rep["audit_sample"][:15]:
            print(f"  {f['title']!r}  hits={f['hits']} stranded={f['stranded']} "
                  f"min_df={f['min_df']}")
            for h in f["homes"][:1]:
                print(f"      {h['cand']}: {h['line'][:120]}")
    return rep


def main() -> int:
    from evals.golden.runner import resolve_vault, vault_digest

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vault")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    vault = resolve_vault(args.vault)
    digest, notes = vault_digest(vault)
    rep = run(vault, limit=args.limit, verbose=not args.quiet)
    doc = {
        "probe": "w6_mention_home",
        "corpus": {"path": str(vault), "digest": digest, "notes": notes},
        "report": rep,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
