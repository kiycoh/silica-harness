# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Curator composer — project graph_report findings into a typed CurationPlan.

The vault already has every remediation *mechanism* (/dedup, /refine, orphan
repair, autolink) but they are all *pull*: a vault is only curated when the
user remembers to. This module is the *initiative* half — a pure projection
from the L1 VaultReport onto a plan of typed items the dispatch layer
(silica.tools.curate) then enqueues on the existing capability seam. No new
power: every item maps onto a WorkItem kind (or the mechanical autolink path)
that already exists.

Finding → item (spec-hermes-coherence §5):
    strong autolink candidate  → autolink  (mechanical, no LLM, direct commit)
    orphan                     → orphan    WorkItem
    high-similarity pair       → dedup     WorkItem (ternary verdict incl.
                                            contradicts → contested sweep)
    oversized / lean note      → refine    WorkItem

Pure & kernel-legal: reads only graph_report dataclasses, no I/O, no
router/capabilities import (import-linter boundary). It is the curator's twin
of kernel.analyst_plan — that one seeds the analyst ledger, this one drives the
background policy.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from silica.kernel.recall.graph_export import is_vault_artifact as _is_vault_artifact
from silica.kernel.report.graph_report import VaultReport

# The four item kinds the curator can emit. "autolink" is the mechanical,
# LLM-free direct commit; the rest are WorkItem kinds on the capability seam.
Kind = str  # "autolink" | "orphan" | "dedup" | "refine"

# Closed set of emittable kinds — single source of truth for CurationPlan.filtered
# validation, so a typo like "dedups" is rejected loudly instead of silently
# filtering the plan to empty.
VALID_KINDS = frozenset({"autolink", "orphan", "dedup", "refine"})


def _norm_note_path(path: str) -> str:
    """Forward-slash, `.md`-suffixed note path for suffix matching.

    Local to the kernel (curator.py can't import silica.agent.bounds across the
    import-linter boundary); named apart from bounds._norm_path to avoid confusion.
    """
    p = path.strip().replace("\\", "/")
    return p if p.endswith(".md") else p + ".md"

# Silica's own vault-root artifacts (log.md, GRAPH_REPORT.md) must never be
# planned for work — --apply would LLM-rewrite the journal or the report on
# every vault with >=1 nucleate. Same predicate now also keeps them out of the
# graph itself (silica.kernel.recall.graph_export.is_vault_artifact), so orphan/hub
# metrics no longer self-pollute from the report's own wikilinks.


@dataclass
class CurationItem:
    kind: Kind
    target: str            # primary note id — graph node id form, carries `.md`
    partner: str = ""      # dedup/autolink: the other note in the pair
    score: float = 0.0     # similarity / co-occurrence weight, when available
    reason: str = ""       # human-readable provenance


@dataclass
class CurationPlan:
    items: list[CurationItem] = field(default_factory=list)
    # Findings the composer deliberately does NOT enqueue, with the reason: a
    # refine on a load-bearing note (V4). Kept on the plan rather than dropped
    # so a dry-run can print what was held back; never applied.
    vetoed: list[CurationItem] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def is_empty(self) -> bool:
        return not self.items

    def by_kind(self, kind: Kind) -> list[CurationItem]:
        return [i for i in self.items if i.kind == kind]

    def counts(self) -> dict[str, int]:
        """Item count per kind, omitting kinds with zero items."""
        return dict(Counter(i.kind for i in self.items))

    def filtered(
        self,
        kinds: list[str] | None = None,
        targets: list[str] | None = None,
    ) -> "CurationPlan":
        """Return a new plan keeping items that satisfy both predicates.

        - `kinds`: keep items whose kind is in `kinds` (case-insensitive).
          Validated against VALID_KINDS — an unknown kind raises ValueError
          rather than silently filtering to empty. Empty/None ⇒ all kinds.
        - `targets`: keep items where a requested path suffix-matches
          `item.target` OR `item.partner` on segment boundaries (a request `r`
          matches path `p` when `p == r` or `p.endswith("/" + r)`, both
          normalized to forward-slash + `.md`). Empty/None ⇒ all targets.
        - Both predicates AND; both empty ⇒ identity. Pure, no mutation.
        """
        kset: set[str] | None = None
        if kinds:
            kset = {k.strip().lower() for k in kinds}
            unknown = kset - VALID_KINDS
            if unknown:
                raise ValueError(
                    "unknown curation kind(s): "
                    + ", ".join(sorted(unknown))
                    + "; valid kinds: "
                    + ", ".join(sorted(VALID_KINDS))
                )
        reqs = [_norm_note_path(t) for t in (targets or []) if t.strip()]

        def keep(item: CurationItem) -> bool:
            if kset is not None and item.kind not in kset:
                return False
            if reqs:
                # A bare stem `x.md` suffix-matches every folder's x.md
                # (intended convenience); the escape hatch is a folder-qualified
                # path (Concepts/x.md), which the segment-boundary rule narrows.
                # `ambiguous_targets` reports when a bare stem hit >1 folder, so
                # --apply can never silently rewrite a note the caller never named.
                paths = [_norm_note_path(item.target)]
                if item.partner:
                    paths.append(_norm_note_path(item.partner))
                if not any(
                    p == r or p.endswith("/" + r) for p in paths for r in reqs
                ):
                    return False
            return True

        return CurationPlan(items=[i for i in self.items if keep(i)], vetoed=list(self.vetoed))

    def ambiguous_targets(
        self,
        targets: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """Bare-stem requests that match items in more than one folder.

        `filtered` matches a request against any path segment-suffix, so a bare
        `x.md` selects EVERY folder's `x.md`. That is convenient for a dry-run
        and dangerous for --apply, which would rewrite notes the caller never
        named. Maps each such request to the sorted distinct paths it hits (>=2
        by definition); a folder-qualified request never appears. Pure, and
        deliberately advisory: "curate every x.md" stays expressible.
        """
        out: dict[str, list[str]] = {}
        for t in (targets or []):
            if not t.strip():
                continue
            req = _norm_note_path(t)
            if "/" in req:
                continue                      # folder-qualified: unambiguous by construction
            hits: set[str] = set()
            for item in self.items:
                for raw in (item.target, item.partner):
                    if not raw:
                        continue
                    p = _norm_note_path(raw)
                    if p == req or p.endswith("/" + req):
                        hits.add(p)
            if len(hits) > 1:
                out[t] = sorted(hits)
        return out


def compose_curation_plan(report: VaultReport) -> CurationPlan:
    """Project a VaultReport into a deterministic plan of typed curation items.

    Deterministic and side-effect-free: the same report always yields the same
    plan. Duplicate dedup pairs (a pair present in both the confirmed and the
    borderline band, in either orientation) collapse to a single item.
    """
    items: list[CurationItem] = []

    # 1. Strong autolink candidates → mechanical direct commit.
    #    "Strong" == corroborated by a directly shared concept (INFERRED, per
    #    analyst_plan.classify_autolink). Associative-only pairs (no shared
    #    concept) are AMBIGUOUS — a human decides, so the curator leaves them.
    for cand in report.autolink_candidates:
        if _is_vault_artifact(cand.source) or _is_vault_artifact(cand.target):
            continue
        if cand.shared:
            items.append(CurationItem(
                kind="autolink",
                target=cand.source,
                partner=cand.target,
                score=cand.weight,
                reason="co-occurrence: " + ", ".join(cand.shared),
            ))

    # 1b. Predicted (V1) and coupled (V3) pairs, only when a second, independent
    #     evidence corroborates: the structural or transactional signal says
    #     "near", the co-occurrence store says "about the same thing". One
    #     signal alone stays a report row for a human.
    seen_auto: set[tuple[str, str]] = {
        (min(i.target, i.partner), max(i.target, i.partner)) for i in items
    }
    for sl in report.structural_links:
        if _is_vault_artifact(sl.source) or _is_vault_artifact(sl.target) or not sl.shared:
            continue
        key = (min(sl.source, sl.target), max(sl.source, sl.target))
        if key in seen_auto:
            continue
        seen_auto.add(key)
        items.append(CurationItem(
            kind="autolink", target=sl.source, partner=sl.target, score=sl.score,
            reason="structural: via " + ", ".join(sl.common[:3])
                   + "; co-occurrence: " + ", ".join(sl.shared),
        ))
    for cp in report.coupled_pairs:
        if _is_vault_artifact(cp.source) or _is_vault_artifact(cp.target) or not cp.shared:
            continue
        key = (min(cp.source, cp.target), max(cp.source, cp.target))
        if key in seen_auto:
            continue
        seen_auto.add(key)
        items.append(CurationItem(
            kind="autolink", target=cp.source, partner=cp.target, score=cp.score,
            reason=f"coupling: w={cp.score}; co-occurrence: " + ", ".join(cp.shared),
        ))

    # 2. Orphans (in-degree 0) → orphan-connector WorkItem.
    for orphan in report.orphans:
        if _is_vault_artifact(orphan):
            continue
        items.append(CurationItem(
            kind="orphan",
            target=orphan,
            reason="orphan (no inbound links)",
        ))

    # 3. High-similarity pairs → dedup WorkItem. Confirmed (>= tau_high) first,
    #    then the borderline band; the dedup worker itself returns the ternary
    #    verdict (duplicate / distinct / contradicts), so feeding both bands
    #    also yields the contested-notes sweep for free.
    seen_pairs: set[tuple[str, str]] = set()
    for dp in list(report.confirmed_duplicate_pairs) + list(report.duplicate_pairs):
        if _is_vault_artifact(dp.source) or _is_vault_artifact(dp.target):
            continue
        lo, hi = sorted((dp.source, dp.target))
        key = (lo, hi)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        items.append(CurationItem(
            kind="dedup",
            target=dp.source,
            partner=dp.target,
            score=dp.score,
            reason=f"similarity {dp.score:.3f}",
        ))

    # 4. Oversized / lean notes → refine WorkItem. reformat_notes is the
    #    report's "Stylistic Refinement" bucket; lean_notes fold in per the spec
    #    row "oversized / lean → refine". Misfiled (V5) and sprawling (V7) rows
    #    deliberately do NOT feed this: judged 2026-08-22 on two vaults, the
    #    misfiled list was filed right 13 of 14 times and the split candidates
    #    were no more splittable than random notes (ADR-0027). They stay
    #    report rows a person reads.
    # Matched on the normalised path (the report mixes graph ids with `.md`
    # and store keys without), emitted under the first id seen so a caller's
    # own ids come back unchanged.
    reasons: dict[str, list[str]] = {}
    raw_of: dict[str, str] = {}

    def note_reason(raw: str, why: str) -> None:
        norm = _norm_note_path(raw)
        raw_of.setdefault(norm, raw)
        reasons.setdefault(norm, []).append(why)

    for note in list(report.reformat_notes) + list(report.lean_notes):
        note_reason(note, "needs stylistic refinement")
    # V4 veto: the refine worker rewrites a body and guarantees nothing about
    # its wikilinks, and a cut vertex is the one note whose links hold areas
    # together. Listed, not enqueued.
    load_bearing = {_norm_note_path(a) for a in report.articulation}
    vetoed: list[CurationItem] = []
    for norm, why in reasons.items():
        note = raw_of[norm]
        if _is_vault_artifact(note):
            continue
        item = CurationItem(kind="refine", target=note, reason="; ".join(why))
        if norm in load_bearing:
            item.reason = "load-bearing (cut vertex): refine by hand; " + item.reason
            vetoed.append(item)
        else:
            items.append(item)

    return CurationPlan(items=items, vetoed=vetoed)
