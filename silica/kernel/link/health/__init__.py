# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""L1 Health — the golden harness's gated metrics, callable on the live vault.

The two GATED probes of the golden coherence harness (evals/golden),
homed here so the agent can run them in-session via the silica_health tool.
One definition, two consumers: the harness runner (regression gate) and the
tool (diagnostic) both call these.

``fusion_probe`` — masked-pair recovery through the FULL relatedness facade.
For each human body-wikilink pair (A, B) that stays >2 hops apart once the
link is masked: does ``related_notes()`` — the real fused ranking (RRF over
embed + cooccur + note_edges, with abstention) — surface the counterpart in
its top-k? Tier-adaptive with no tier code: for indexed notes the embed leg
is a pure index lookup + cosine (no API call); absent or empty, the leg
abstains per the facade contract and the probe measures cooccur+edges fusion.
``legs`` reports what was actually live; ``embed_coverage`` (fraction of
evaluated notes with a stored vector) exposes stale or key-mismatched
embedding indexes that would otherwise read as a recall drop. Masking caveat:
the wikilink's surface text stays in the note body, so recall is an
optimistic ceiling — a regression/trend number, not an absolute claim.

``integrity_probe`` — differential corruption on write-path transforms. An
absolute lint of the human vault measures the human, not the pipeline (those
totals are informational). The GATED form is differential: run each real note
body through the pipeline's body-transforming functions and require
``violations(after) ⊆ violations(before)`` — zero NEW violations. Gated at
exactly 1.0; a single introduced violation fails. Transforms under test:
  T1  frontmatter split → dump round-trip (contract = violation-set
      inclusion, NOT byte equality — dump re-dumps yaml + lstrips body).
  T2  autolink insertion (exercises _build_skip_mask on real bodies).
  T3  fs backend write → read (the channel of the historical double-escaping
      LaTeX bug) — a scratch tempdir, never the real vault.
  T4  sanitize.normalize_ops (distiller post-processing).
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

from silica.kernel.write import frontmatter
from silica.kernel.link.autolink import autolink, build_title_index
from silica.kernel.link.health import lint
from silica.kernel.text.sanitize import normalize_ops

# Facade depth measured; matches the k of every production related_notes surface.
# The metric name pins it — change both together or not at all.
K = 10

# (!?) embed marker, target, optional #anchor, optional |alias — targets only.
_WIKILINK = re.compile(r"(!?)\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")

_EMPTY_FUSION = {
    "pairs_evaluated": 0,
    "recall_at_10": 0.0,
    "mrr": 0.0,
    "embed_coverage": 0.0,
    "legs": "",
}


def iter_notes(vault: Path) -> list[Path]:
    """Sorted ``*.md`` under the vault, excluding dot-directories.

    Single source of truth for the harness digest and every probe, so a run
    can never measure a note set the digest didn't hash.
    """
    return sorted(
        p for p in vault.rglob("*.md")
        if not any(part.startswith(".") for part in p.relative_to(vault).parts)
    )


def pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def wikilink_graph(vault: Path, store) -> dict[str, set[str]]:
    """Human body-wikilink adjacency {key: {keys}}, keyed by store paths.

    Ambiguous basenames (one stem, several notes) are dropped — unresolvable,
    the same ceiling probe_links accepts. Embeds (![[...]]) carry no prose link.
    """
    keys = set(store.paths())
    by_stem: dict[str, str | None] = {}
    for k in keys:
        stem = k.split("/")[-1]
        by_stem[stem] = None if stem in by_stem else k

    adj: dict[str, set[str]] = {}
    for p in iter_notes(vault):
        src = p.relative_to(vault).with_suffix("").as_posix()
        if src not in keys:
            continue
        _data, _raw, body = frontmatter.split(p.read_text(encoding="utf-8"))
        for m in _WIKILINK.finditer(body):
            if m.group(1):  # embed — skip
                continue
            target = m.group(2).strip().replace("\\", "/").split("/")[-1]
            dst = by_stem.get(target)
            if dst and dst != src:
                adj.setdefault(src, set()).add(dst)
                adj.setdefault(dst, set()).add(src)
    return adj


def eligible_pairs(adj: dict[str, set[str]]) -> list[tuple[str, str]]:
    """Unordered wikilinked pairs that stay >2 hops apart once masked.

    A shared neighbour leaves a 2-hop path (via the hub) after masking, so
    the pair is not a fair recovery target — excluded, per the real filter.
    """
    eligible: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for a, nbrs in adj.items():
        for b in nbrs:
            p = pair(a, b)
            if p in seen:
                continue
            seen.add(p)
            if (adj.get(p[0], set()) - {p[1]}) & (adj.get(p[1], set()) - {p[0]}):
                continue
            eligible.append(p)
    return eligible


def fusion_probe(vault: Path, store, *, embed_store=None, k: int = K, verbose: bool = False) -> dict:
    from silica.kernel.link import correlate
    from silica.kernel.recall.relatedness import related_notes

    es = embed_store if (embed_store is not None and len(embed_store)) else None
    legs = ("embed+" if es is not None else "") + ("cooccur+edges" if len(store) else "")

    if len(store) == 0:
        return {**_EMPTY_FUSION, "legs": legs}

    # Self-contained: derive note_edges from the current contributions
    # (idempotent — probe order in the runner must not matter).
    correlate.recompute_all_edges(store)

    eligible = eligible_pairs(wikilink_graph(vault, store))
    if not eligible:
        return {**_EMPTY_FUSION, "legs": legs}

    # One facade call per unique endpoint, not per pair — pairs share notes.
    endpoints = sorted({e for pr in eligible for e in pr})
    topk = {
        key: [r.path for r in related_notes(key, embed_store=es, cooccur_store=store, k=k)]
        for key in endpoints
    }

    covered = 0
    if es is not None:
        for key in endpoints:  # mirror _embed_ranking's exact-then-stripped lookup
            if es.get_vec(key) is not None or es.get_vec(key.removesuffix(".md")) is not None:
                covered += 1

    hits = 0
    rr_sum = 0.0
    for a, b in eligible:
        ranks = []
        if b in topk[a]:
            ranks.append(topk[a].index(b) + 1)
        if a in topk[b]:
            ranks.append(topk[b].index(a) + 1)
        if ranks:  # recovered from either direction, best rank feeds MRR
            hits += 1
            rr_sum += 1.0 / min(ranks)

    n = len(eligible)
    res = {
        "pairs_evaluated": n,
        "recall_at_10": round(hits / n, 4),
        "mrr": round(rr_sum / n, 4),
        "embed_coverage": round(covered / len(endpoints), 4) if es is not None else 0.0,
        "legs": legs,
    }
    if verbose:
        print(f"\nfusion[{legs}]: recall@{k} {hits}/{n} = {res['recall_at_10']:.1%}, "
              f"mrr {res['mrr']:.3f}, embed coverage {res['embed_coverage']:.1%}")
    return res


# ---------------------------------------------------------------------------
# Integrity — differential lint across the write-path transforms
# ---------------------------------------------------------------------------

def _t1_frontmatter(text, data, body, stem) -> dict:
    if not isinstance(data, dict):
        return {}  # no frontmatter / YAML error — nothing to round-trip
    return lint.new_violations(text, frontmatter.dump(data, body), stem)


def _t2_autolink(body, title_index, stem) -> dict:
    low = body.casefold()
    cands = [t for t in title_index if t.casefold() in low]
    new_body, _added = autolink(body, title_index, candidates=cands, self_title=stem)
    return lint.new_violations(body, new_body, stem)


def _t3_fs_roundtrip(backend, rel, text, stem) -> dict:
    ref = backend.create(rel, text)
    roundtrip = backend.read_note(ref).content
    return lint.new_violations(text, roundtrip, stem)


def _t4_sanitize(body, stem) -> dict:
    res = normalize_ops([{"content": body}])
    after = res[0].get("content", body) if res else body
    return lint.new_violations(body, after, stem)


def integrity_probe(vault: Path, *, verbose: bool = False) -> dict:
    from silica.driver.fs_backend import ObsidianFSBackend  # lazy: kernel stays driver-free at import

    all_md = iter_notes(vault)
    title_index = build_title_index([p.stem for p in all_md])

    notes = 0
    clean = 0
    vault_structural = 0
    vault_style = 0
    notes_with_structural = 0

    with tempfile.TemporaryDirectory() as scratch:
        backend = ObsidianFSBackend(vault_path=scratch)
        for p in all_md:
            text = p.read_text(encoding="utf-8")
            stem = p.stem
            rel = p.relative_to(vault).as_posix()
            data, _raw, body = frontmatter.split(text)

            # absolute lint (informational)
            structural, style = lint.totals(lint.scan(text, stem))
            vault_structural += structural
            vault_style += style
            if structural:
                notes_with_structural += 1

            # differential across the 4 transforms
            introduced = {
                "T1-frontmatter": _t1_frontmatter(text, data, body, stem),
                "T2-autolink": _t2_autolink(body, title_index, stem),
                "T3-fs-roundtrip": _t3_fs_roundtrip(backend, rel, text, stem),
                "T4-sanitize": _t4_sanitize(body, stem),
            }
            notes += 1
            if any(introduced.values()):
                if verbose:
                    for transform, viols in introduced.items():
                        for name, cnt in viols.items():
                            print(f"  NEW {rel} [{transform}] {name} +{cnt}")
            else:
                clean += 1

    # None, not 1.0, over zero notes: a walk that reached nothing (wrong vault,
    # every folder ignored) measured nothing, and 1.0 is the exact value the
    # golden gate passes on. compare() fails a None; doctor-style readers show it.
    rate = round(clean / notes, 4) if notes else None
    if verbose:
        shown = "not measured" if rate is None else f"{rate:.3f}"
        print(f"\nintegrity: rate {clean}/{notes} = {shown} | "
              f"vault structural={vault_structural} style={vault_style} "
              f"notes_with_structural={notes_with_structural}")

    return {
        "rate": rate,
        "notes": notes,
        "vault_structural_violations": vault_structural,
        "vault_style_flags": vault_style,
        "vault_notes_with_structural": notes_with_structural,
    }
