# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Output renderers for a VaultReport: markdown, facts, digest, files.

Read-only over the report — no graph computation, no signal logic.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import orjson

from silica.kernel.report.graph_report.models import VaultReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------------

_MEMBERS_CAP = 25  # max members shown per cluster in markdown
_LIST_CAP = 30     # max bullet/row items for long lists (orphans, dangling, …)
_MIN_CLUSTER = 2   # size-1 clusters are noise — summarised, never listed


def append_energy_point(series_path: Path, record: dict,
                        prev: float | None, prev_terms: dict | None) -> bool:
    """Append one energy point to the series unless nothing moved.

    `prev`/`prev_terms` are the head file's last state: equal value and terms
    mean a re-render of an unchanged vault, which must not grow the series.
    A missing series file always takes the point (the trend has to start
    somewhere). One line per write; a torn tail is skipped by readers the way
    quiz.jsonl's is. Returns True when a line landed.
    """
    if series_path.is_file() and prev == record["value"] and prev_terms == record["terms"]:
        return False
    with series_path.open("ab") as fh:
        fh.write(orjson.dumps(record) + b"\n")
    return True


def _short(p: str | None) -> str:
    return p.rsplit("/", 1)[-1].removesuffix(".md") if p else "—"


_TOTAL_LABELS = {
    "dangling_links": "Broken links (point nowhere)",
    "missing_links": "Missing links (proposed)",
    "duplicate_pairs": "Related pairs (borderline — link, not merge)",
    "confirmed_duplicates": "Likely duplicates (merge candidates)",
    "autolink_candidates": "Autolink candidates",
    "integration_deficits": "Integration deficit (rich text, few links)",
    "lean_notes": "Thin notes (enrich?)",
    "reformat_notes": "Notes to reformat",
    "orphans": "Orphans (no incoming links)",
    "structural_gaps": "Structural gaps (disconnected areas)",
}


def _fold(add, kind: str, title: str, items: list, fmt) -> None:
    """Wrap a bulleted list in a collapsed OFM callout (`> [!kind]- title`).

    Every line is `>`-prefixed so it renders inside the callout and the
    `[[wikilinks]]` survive — an HTML <details> fold would swallow them.
    `fmt(item)` returns each bullet's text; the list is capped at `_LIST_CAP`.
    Trailing blank line separates this callout from the next block.

    The overflow line always points at GRAPH_REPORT.json: every folded list is
    a VaultReport field, so the full list is always there.
    """
    add(f"> [!{kind}]- {title}")
    for it in items[:_LIST_CAP]:
        add(f"> - {fmt(it)}")
    if len(items) > _LIST_CAP:
        add(f"> - _… +{len(items) - _LIST_CAP} more (see GRAPH_REPORT.json)_")
    add("")


def to_markdown(r: VaultReport, title: str = "Silica Vault Report") -> str:
    """Render a VaultReport as OFM-friendly, human-readable markdown.

    Long lists fold into collapsed callouts (`[!kind]-`) so the note opens
    compact; tables (Totals, God Nodes, Bridges) stay flat since tables render
    poorly inside callouts. Singleton clusters are summarised, lists capped;
    the full data lives in the sibling GRAPH_REPORT.json.
    """
    lines: list[str] = []
    add = lines.append

    add(f"# {title}")
    add(f"_Generated: {r.generated_at}_")
    if r.scope:
        add(f"_Scope: `{r.scope}`_")
    add("")

    # cluster_id -> hub name, to label god-nodes & bridges by area (not by number)
    hub_of = {c.cluster_id: _short(c.hub) for c in r.clusters}
    linked = sorted(
        (c for c in r.clusters if c.size >= _MIN_CLUSTER),
        key=lambda c: c.size, reverse=True,
    )
    singletons = sum(1 for c in r.clusters if c.size < _MIN_CLUSTER)
    t = r.totals

    # Summary (prose — the part a human actually reads)
    add("## Summary")
    add(
        f"This vault holds **{t.get('notes', 0)} notes** connected by "
        f"**{t.get('links', 0)} links**, forming **{len(linked)} linked areas** "
        f"plus {singletons} standalone notes."
    )
    if linked:
        top = ", ".join(f"[[{_short(c.hub)}]] ({c.size})" for c in linked[:5])
        add(f"Largest areas: {top}.")
    if r.discourse_state:
        add(f"Discourse shape: **{r.discourse_state}**.")
    add("")
    health = []
    if t.get("components", 0) > 1:
        health.append(
            f"{t['components']} disconnected islands (no reading path between them)"
        )
    if t.get("orphans"):
        health.append(f"{t['orphans']} orphans (no incoming links)")
    if t.get("dangling_links"):
        health.append(f"{t['dangling_links']} broken links (point to notes that don't exist)")
    if t.get("contested"):
        health.append(f"{t['contested']} contested notes (unresolved contradictions)")
    if t.get("source_drift"):
        health.append(f"{t['source_drift']} notes derived from a superseded source version")
    if health:
        add("> [!warning] Health")
        for h in health:
            add(f"> - {h}")
        add("")
    else:
        add("> [!success] Health")
        add("> No orphans, broken links, or contradictions.")
        add("")
    fixes = [
        f"{t[k]} {word}"
        for k, word in (
            ("autolink_candidates", "autolink candidates"),
            ("lean_notes", "notes to enrich"),
            ("confirmed_duplicates", "likely duplicates to merge"),
            ("duplicate_pairs", "borderline-related pairs"),
            ("reformat_notes", "notes to reformat"),
        )
        if t.get(k)
    ]
    if fixes:
        add("> [!tip] Suggestions ready (not applied)")
        add("> " + " · ".join(fixes) + ".")
        add("> Nothing changes until you approve — review below, then ask Silica to apply.")
        add("")

    # Totals
    add("## Totals")
    add("| Metric | Count |")
    add("|---|---|")
    for k, v in r.totals.items():
        add(f"| {_TOTAL_LABELS.get(k, k.replace('_', ' ').capitalize())} | {v} |")
    add("")

    # E(vault) — lattice-energy thermometer with its per-term decomposition
    # (spec-harness-promotion §3). Lower is more coherent; the six signed
    # contributions sum to the total, so ΔE between two reports decomposes
    # per term. Every term is an existing VaultReport field.
    # Local import, and it has to stay local: vault_energy imports
    # graph_report.models at module level, and graph_report/__init__ imports
    # this module — a module-level import here deadlocks whenever vault_energy
    # is the first of the two to be imported.
    from silica.kernel.report.vault_energy import vault_energy

    e = vault_energy(r)
    add("## Energy")
    add(f"**E(vault): {e.total:+.2f}** — lower is more coherent (thermometer, not a target).")
    add("")
    add("| Term | Contribution |")
    add("|---|---|")
    for term in ("cohesion", "orphans", "dangling", "gaps", "deficits", "contested"):
        add(f"| {term} | {getattr(e, term):+.2f} |")
    add("")

    # Temporal layer — what the vault records about its own history. Counts only:
    # this is a reading of the bi-temporal layer, not a score, and deliberately
    # not folded into E (see TemporalStat).
    if r.temporal and r.temporal.notes_scanned:
        tp = r.temporal
        add("## Temporal Layer")
        add("| Signal | Value |")
        add("|---|---|")
        for tier, label in ((3, "human"), (2, "grounded"), (1, "distilled")):
            add(f"| Notes at tier {tier} ({label}) | {tp.by_tier.get(tier, 0)} |")
        add(f"| Notes with a `## Superseded` section | {tp.superseded_sections} |")
        add(f"| Notes merged away (`superseded_by`) | {tp.superseded_notes} |")
        add(f"| Notes carrying a claim stamp | {tp.stamped} / {tp.notes_scanned} |")
        if tp.oldest_valid_from:
            add(f"| Earliest `valid_from` | {tp.oldest_valid_from} |")
        add("")

    # God nodes (PageRank dropped — it reads 0.0 at this scale; degree is the signal)
    add("## God Nodes (High-Degree Hubs)")
    if r.god_nodes:
        # Betweenness rides alongside degree: a hub with high betweenness is also
        # a bottleneck (its removal fragments the discourse), not just popular.
        add("| Note | Area | Links | In | Out | Between |")
        add("|---|---|---|---|---|---|")
        for n in r.god_nodes:
            area = hub_of.get(n.cluster, f"#{n.cluster}")
            add(f"| [[{n.label}]] | {area} | {n.degree} | {n.in_degree} | {n.out_degree} | {n.betweenness} |")
    else:
        add("_No connected notes found._")
    add("")

    # Surprising bridges
    add("## Surprising Cross-Cluster Connections")
    add("_Links joining two otherwise-separate areas — often the most interesting._")
    if r.bridges:
        add("| Source | Target | Areas joined | Surprise |")
        add("|---|---|---|---|")
        for b in r.bridges:
            sa = hub_of.get(b.source_cluster, f"#{b.source_cluster}")
            ta = hub_of.get(b.target_cluster, f"#{b.target_cluster}")
            add(f"| [[{_short(b.source)}]] | [[{_short(b.target)}]] | {sa} ↔ {ta} | {b.weight} |")
    else:
        add("_No cross-cluster bridges found._")
    add("")

    # Structural gaps — the mirror of bridges: areas that should connect but don't
    add("## Structural Gaps (Disconnected Knowledge Areas)")
    add("_Well-formed areas with no links between them — candidate bridges to build._")
    if r.structural_gaps:
        add("| Area A | Area B | Links | Gap score |")
        add("|---|---|---|---|")
        for g in r.structural_gaps:
            area_a = hub_of.get(g.cluster_a, f"#{g.cluster_a}")
            area_b = hub_of.get(g.cluster_b, f"#{g.cluster_b}")
            add(f"| {area_a} | {area_b} | {g.inter_edges} | {g.gap_score} |")
    else:
        add("_No disconnected areas (or too few clusters to compare)._")
    add("")

    # Clusters (named by hub, singletons collapsed, biggest first)
    add("## Clusters (Knowledge Areas)")
    if linked:
        for c in linked:
            add(f"> [!abstract]- [[{_short(c.hub)}]] — {c.size} notes · cohesion {c.cohesion}")
            member_links = ", ".join(f"[[{_short(m)}]]" for m in c.members[:_MEMBERS_CAP])
            if len(c.members) > _MEMBERS_CAP:
                member_links += f" … (+{len(c.members) - _MEMBERS_CAP} more)"
            add(f"> {member_links}")
            add("")
    else:
        add("_No linked clusters detected (vault has no resolved wikilinks)._")
        add("")
    if singletons:
        add(f"_Plus {singletons} standalone notes with no internal links (full list in GRAPH_REPORT.json)._")
        add("")

    # Orphans
    add("## Orphans (No Incoming Links)")
    if r.orphans:
        _fold(add, "warning", f"{len(r.orphans)} orphans", r.orphans,
              lambda o: f"[[{_short(o)}]]")
    else:
        add("_No orphans._")
        add("")

    # Dangling links — targets are unresolved (inline code, not wikilinks)
    add("## Dangling Links (Unresolved Wikilinks)")
    if r.dangling:
        _fold(add, "bug", f"{len(r.dangling)} broken links", r.dangling,
              lambda d: f"`{d['target']}` — {d['refs']}×")
    else:
        add("_No unresolved wikilinks._")
        add("")

    # Contested claims — authoritative, kept visible until a human resolves them
    if r.contested:
        add("## Contested Claims (Unresolved Contradictions)")
        _fold(add, "danger", f"{len(r.contested)} contested", r.contested,
              lambda c: f"[[{_short(c.path)}]] ↮ {'; '.join(c.refs) if c.refs else '—'}")

    # Source drift — authoritative, from <vault>/provenance.json: notes still
    # carrying claims from a source version that has since been re-nucleated
    if r.source_drift:
        add("## Source Drift (Notes From a Superseded Source Version)")
        _fold(add, "warning", f"{len(r.source_drift)} drifted notes", r.source_drift,
              lambda d: f"[[{_short(d.note)}]] — derived from a superseded version of {d.source}")

    # Missing links (proposed)
    if r.missing_links:
        add("## Proposed Missing Links _(embedding candidates — not authoritative)_")
        add("| Source | Target | Cosine | d_prev | Novelty |")
        add("|---|---|---|---|---|")
        for ml in r.missing_links[:_LIST_CAP]:
            novelty = "🔴 novel" if ml.d_prev == 0 or ml.d_prev >= 3 else "🟡 likely"
            d_str = str(ml.d_prev) if ml.d_prev > 0 else "∞"
            add(f"| [[{_short(ml.source)}]] | [[{_short(ml.target)}]] | {ml.cosine} | {d_str} | {novelty} |")
        add("")

    # Likely duplicates (≥ τ_high — genuine merge candidates)
    if r.confirmed_duplicate_pairs:
        add(f"### Likely Duplicates ({len(r.confirmed_duplicate_pairs)}) _(≥ τ_high — review for merge)_")
        _fold(add, "warning", "Merge candidates", r.confirmed_duplicate_pairs,
              lambda dp: f"[[{_short(dp.source)}]] vs [[{_short(dp.target)}]] (score {dp.score:.3f})")

    # Borderline-related pairs (τ_low..τ_high — topically close, link not merge)
    if r.duplicate_pairs:
        add(f"### Related Pairs ({len(r.duplicate_pairs)}) _(borderline similarity — link, don't merge)_")
        _fold(add, "note", "Borderline pairs", r.duplicate_pairs,
              lambda dp: f"[[{_short(dp.source)}]] vs [[{_short(dp.target)}]] (score {dp.score:.3f})")

    # Co-occurrence delta (proposed, embedder-free)
    if r.autolink_candidates:
        add("\n## Autolink Candidates _(co-occurrence − wikilink — not authoritative)_")
        add("| Source | Target | Via | Weight | Hubs | Shared Concepts |")
        add("|---|---|---|---|---|---|")
        for cand in r.autolink_candidates:
            shared = ", ".join(cand.shared) if cand.shared else "_(associative)_"
            add(f"| [[{_short(cand.source)}]] | [[{_short(cand.target)}]] | {cand.provenance} | {cand.weight} | {cand.convergence} | {shared} |")

    if r.stale_links:
        add("\n## Stale Links _(wikilink − co-occurrence — review)_")
        _fold(add, "note", f"{len(r.stale_links)} stale links", r.stale_links,
              lambda s: f"[[{_short(s.source)}]] ↔ [[{_short(s.target)}]] _(linked, no shared concepts)_")

    if r.missing_hubs:
        add("\n## Missing Hubs _(central concepts with no hub note)_")
        add("| Concept | Centrality |")
        add("|---|---|")
        for miss in r.missing_hubs:
            add(f"| {miss.concept} | {miss.centrality} |")

    if r.integration_deficits:
        add("\n## Integration Deficit _(concept-rich, weakly linked — not authoritative)_")
        add("| Note | Concepts | Links | Score |")
        add("|---|---|---|---|")
        for idf in r.integration_deficits:
            add(f"| [[{_short(idf.path)}]] | {idf.concepts} | {idf.degree} | {idf.score} |")

    if r.code_coverage:
        cc = r.code_coverage
        add("\n## Code Coverage _(codegraph — supported files documented by a note)_")
        add(f"**{cc.documented}/{cc.total}** supported source files are documented.")
        if cc.undocumented:
            _fold(add, "todo", f"{len(cc.undocumented)} undocumented files (by fan-in)",
                  cc.undocumented, lambda u: f"`{u[0]}` — {u[1]} importer(s)")
        else:
            add("")

    # Seven inter-note variables (spec 2026-08-22-graph-variables-design).
    # Each section appears only when the signal fired, like the proposed
    # sections above it: an empty heading would read as "measured, clean".
    if r.load_bearing:
        add("\n## Load-Bearing Notes _(cut vertices first, then high betweenness at low degree)_")
        add("| Note | Links | Between | Core | Cut vertex | Surprise |")
        add("|---|---|---|---|---|---|")
        for lb in r.load_bearing:
            add(f"| [[{_short(lb.path)}]] | {lb.degree} | {lb.betweenness} | {lb.coreness} | "
                f"{'yes' if lb.articulation else ''} | {lb.surprise} |")

    if r.structural_links:
        add("\n## Predicted Links _(Adamic-Adar over shared neighbours — not authoritative)_")
        add("| Source | Target | Score | Via | Shared concepts |")
        add("|---|---|---|---|---|")
        for sl in r.structural_links:
            via = ", ".join(_short(c) for c in sl.common[:3])
            add(f"| [[{_short(sl.source)}]] | [[{_short(sl.target)}]] | {sl.score} | {via} | "
                f"{', '.join(sl.shared) if sl.shared else '—'} |")

    if r.coupled_pairs:
        add("\n## Coupled Notes _(share sources or were written together, unlinked — not authoritative)_")
        add("| Source | Target | Coupling | Shared concepts |")
        add("|---|---|---|---|")
        for cp in r.coupled_pairs:
            add(f"| [[{_short(cp.source)}]] | [[{_short(cp.target)}]] | {cp.score} | "
                f"{', '.join(cp.shared) if cp.shared else '—'} |")

    if r.prerequisites:
        add("\n## Prerequisite Chains _(RefD: read the first before the second — not authoritative)_")
        add("| Read first | Then | RefD |")
        add("|---|---|---|")
        for pe in r.prerequisites[:_MEMBERS_CAP]:
            add(f"| [[{_short(pe.prereq)}]] | [[{_short(pe.dependent)}]] | {pe.refd} |")
        if len(r.prerequisites) > _MEMBERS_CAP:
            add(f"_… (+{len(r.prerequisites) - _MEMBERS_CAP} more in GRAPH_REPORT.json)_")

    if r.misfiled:
        add("\n## Misfiled Notes _(linked into one area, read like another — not authoritative)_")
        add("| Note | Links | Dissonance |")
        add("|---|---|---|")
        for mf in r.misfiled:
            add(f"| [[{_short(mf.path)}]] | {mf.degree} | {mf.dissonance} |")

    if r.bursting_concepts:
        add("\n## Bursting Concepts _(over-represented in the latest writing window)_")
        add("| Concept | z | Recent notes | All notes |")
        add("|---|---|---|---|")
        for bc in r.bursting_concepts:
            add(f"| {bc.concept} | {bc.z} | {bc.recent} | {bc.total} |")

    if r.sprawling:
        add("\n## Sprawling Notes _(broad and flat concept spread — split candidates)_")
        add("| Note | Concepts | Entropy (bits) | Flatness |")
        add("|---|---|---|---|")
        for sp in r.sprawling:
            add(f"| [[{_short(sp.path)}]] | {sp.concepts} | {sp.entropy} | {sp.flatness} |")

    if r.attention_candidates:
        add("\n## Attention Candidates _(recall misses × idle × weakly-linked, not authoritative)_")
        add("| Note | Idle (days) | Links | Wrong/Asked | Score |")
        add("|---|---|---|---|---|")
        for ac in r.attention_candidates:
            asked = f"{ac.misses}/{ac.attempts}" if ac.attempts else "-"
            add(f"| [[{_short(ac.path)}]] | {ac.days_idle} | {ac.degree} | {asked} | {ac.score} |")

    if r.lean_notes:
        add(f"\n### Lean Notes (Enrichment Candidates) ({len(r.lean_notes)})")
        _fold(add, "todo", "Enrichment candidates", r.lean_notes,
              lambda n: f"[[{_short(n)}]]")

    if r.reformat_notes:
        add(f"\n### Reformat Notes (Stylistic Refinement) ({len(r.reformat_notes)})")
        _fold(add, "todo", "Stylistic refinements", r.reformat_notes,
              lambda n: f"[[{_short(n)}]]")

    return "\n".join(lines)


def to_facts(report: VaultReport) -> dict:
    """Compact, stable subset for TaskLedger.facts (write-once, digest-friendly)."""
    return {
        "scope": report.scope,
        "totals": dict(report.totals),
        "god_nodes": [n.id for n in report.god_nodes],
        "top_bridges": [[b.source, b.target] for b in report.bridges[:5]],
        "orphan_count": report.totals.get("orphans", 0),
        "dangling_top": report.dangling[:5],
    }


def to_digest(report: VaultReport, *, max_items: int = 8) -> str:
    """Compact summary targeting < 500 tokens."""
    lines: list[str] = []
    t = report.totals
    header = (
        f"VAULT AUDIT  scope={report.scope or 'all'}  "
        f"notes={t.get('notes',0)}  links={t.get('links',0)}  "
        f"clusters={t.get('clusters',0)}  orphans={t.get('orphans',0)}  "
        f"unresolved={t.get('unresolved',0)}"
    )
    if report.discourse_state:
        header += f"  shape={report.discourse_state}"
    lines.append(header)
    lines.append("─" * 36)

    def row(label: str, items, fmt) -> None:
        """`LABEL  a, b, c` capped at max_items. Empty list ⇒ no line at all.

        The overflow count is never silent: a truncated row that looked complete
        would read as "this is all of them".
        """
        if not items:
            return
        body = ", ".join(fmt(it) for it in items[:max_items])
        if len(items) > max_items:
            body += f" (+{len(items) - max_items} more)"
        lines.append(f"{label}  {body}")

    def pair(dp) -> str:
        return f"{_short(dp.source)}↔{_short(dp.target)}(cos={dp.score})"

    # bet=, wrong= and coh= appear only when the signal exists: a popular hub and
    # a bottleneck whose removal fragments the discourse are different readings,
    # and a printed zero would read as "measured, came out flat".
    row("TOP HUBS", report.god_nodes,
        lambda n: f"{n.label}(deg={n.degree}"
                  + (f",bet={n.betweenness}" if n.betweenness else "") + ")")
    row("BRIDGES", report.bridges,
        lambda b: f"{_short(b.source)}↔{_short(b.target)}(w={b.weight})")
    row("GAPS", report.structural_gaps,
        lambda g: f"{_short(g.hub_a)}↮{_short(g.hub_b)}(links={g.inter_edges})")
    row("ORPHANS", report.orphans, _short)
    row("DANGLING", report.dangling, lambda d: f"{d['target']}(×{d['refs']})")
    row("CONTESTED", report.contested, lambda c: _short(c.path))
    row("SOURCE DRIFT", report.source_drift, lambda d: f"{_short(d.note)}←{d.source}")
    row("ATTENTION", report.attention_candidates,
        lambda a: f"{_short(a.path)}(idle={a.days_idle}d,deg={a.degree}"
                  + (f",wrong={a.misses}/{a.attempts}" if a.attempts else "") + ")")
    row("CLUSTERS", report.clusters,
        lambda c: f"C{c.cluster_id}(n={c.size},hub={_short(c.hub) if c.hub else '-'}"
                  + (f",coh={c.cohesion}" if c.cohesion else "") + ")")
    row("MISSING HUBS", report.missing_hubs, lambda h: f"{h.concept}(cent={h.centrality})")
    row("INTEGRATION DEFICIT", report.integration_deficits,
        lambda i: f"{_short(i.path)}(concepts={i.concepts},deg={i.degree})")
    row("PROPOSED", report.missing_links,
        lambda m: f"{_short(m.source)}→{_short(m.target)}(cos={m.cosine},d={m.d_prev})")
    row("DUPS", report.confirmed_duplicate_pairs, pair)
    row("RELATED", report.duplicate_pairs, pair)
    row("LOAD-BEARING", report.load_bearing,
        lambda lb: f"{_short(lb.path)}(deg={lb.degree},core={lb.coreness}"
                   + (",cut" if lb.articulation else "") + ")")
    row("PREDICTED", report.structural_links,
        lambda sl: f"{_short(sl.source)}↔{_short(sl.target)}(aa={sl.score})")
    row("COUPLED", report.coupled_pairs,
        lambda cp: f"{_short(cp.source)}↔{_short(cp.target)}(w={cp.score})")
    row("PREREQ", report.prerequisites,
        lambda pe: f"{_short(pe.prereq)}→{_short(pe.dependent)}({pe.refd})")
    row("MISFILED", report.misfiled, lambda mf: f"{_short(mf.path)}(d={mf.dissonance})")
    row("BURSTING", report.bursting_concepts, lambda bc: f"{bc.concept}(z={bc.z})")
    row("SPRAWLING", report.sprawling,
        lambda sp: f"{_short(sp.path)}(concepts={sp.concepts},H={sp.entropy})")

    return "\n".join(lines)


def write_report(report: VaultReport, output_path: str) -> dict:
    """Write GRAPH_REPORT.md and report.json. Returns {path_md, path_json}."""
    out_md = Path(output_path)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(to_markdown(report), encoding="utf-8")

    out_json = out_md.with_suffix(".json")
    # NON_STR_KEYS: TemporalStat.by_tier is int-keyed; JSON has no int keys, so
    # they land as "3"/"2"/"1". The file is write-only (humans + tests), never
    # round-tripped, so the coercion costs nothing.
    out_json.write_bytes(
        orjson.dumps(
            dataclasses.asdict(report),
            option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS,
        )
    )

    # Persist E(vault) for /status (spec-harness-promotion §3). Whole-vault
    # reports only: a folder-scoped E is not comparable and would corrupt the
    # delta. `prev` carries the prior value so /status can show the delta
    # without a second file. Best-effort: never fails the report write.
    if not report.scope:
        try:
            import datetime as _dt

            from silica.config import CONFIG
            from silica.kernel.report.vault_energy import vault_energy

            vault = getattr(CONFIG, "vault_path", None)
            if vault:
                energy_path = Path(vault) / ".silica" / "energy.json"
                prev: float | None = None
                prev_terms: dict | None = None
                if energy_path.is_file():
                    _old = orjson.loads(energy_path.read_bytes())
                    prev = _old.get("value")
                    prev_terms = _old.get("terms")
                e = vault_energy(report)
                # The six contributions sum to the total, so persisting them is what
                # makes ΔE ATTRIBUTABLE ("orphans fell, cohesion held") instead of a
                # bare number that moved. Storing only `value` made the decomposition
                # the docstring promises unobservable across runs.
                record: dict = {
                    "value": e.total,
                    "terms": {t: getattr(e, t) for t in
                              ("cohesion", "orphans", "dangling", "gaps", "deficits", "contested")},
                    "at": _dt.datetime.now().isoformat(timespec="seconds"),
                }
                payload = dict(record)
                if prev is not None:
                    payload["prev"] = prev
                if prev_terms:
                    payload["prev_terms"] = prev_terms
                energy_path.parent.mkdir(parents=True, exist_ok=True)
                energy_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
                # The trend lives beside the head: energy.jsonl gets one line
                # per actual movement, so re-rendering an unchanged vault
                # never grows it. The series only exists from the day the
                # append starts, which is why it ships before any UI reads it.
                append_energy_point(
                    energy_path.with_suffix(".jsonl"), record, prev, prev_terms)
        except Exception as exc:
            logger.debug("graph_report: energy.json persist skipped (%s)", exc)

    logger.info(
        "graph_report: wrote %s and %s",
        out_md,
        out_json,
    )
    return {"path_md": str(out_md.resolve()), "path_json": str(out_json.resolve())}
