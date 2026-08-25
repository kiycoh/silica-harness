# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Vault map — a compact semantic self-model of the corpus for recall at session start.

CoALA: consolidates the persistent co-occurrence index + the folder structure
into a short Markdown block, injected into working memory at startup, so the
agent starts oriented instead of rediscovering the vault via tools.

Deterministic, zero LLM. Best-effort: any sub-block that fails is omitted;
an empty vault or cooccur index → None (the caller injects nothing).
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silica.kernel.recall.cooccurrence import CooccurStore

logger = logging.getLogger(__name__)

_warned: set[str] = set()


def _warn_once(kind: str, msg: str, *args) -> None:
    """Report a degraded map once per kind, then at DEBUG.

    Every block below fails open by design, and the whole builder returns None
    on failure — which `_inject_vault_map` cannot distinguish from an empty
    vault, so it injects nothing and the session silently starts without its
    self-model. This is the one line that says so. Once per kind and not per
    call because the GUI rebuilds the map every turn.
    """
    if kind in _warned:
        logger.debug(msg, *args)
        return
    _warned.add(kind)
    logger.warning(msg, *args)


def build_vault_map(
    *,
    store: "CooccurStore | None" = None,
    max_folders: int = 8,
    max_clusters: int = 8,
    max_vocab: int = 15,
    max_hubs: int = 8,
    max_contested: int = 8,
    log_tail: int = 5,
) -> str | None:
    try:
        from silica.config import CONFIG
        from silica.kernel.recall.cooccurrence import get_cooccur_store

        store = store if store is not None else get_cooccur_store(lang=CONFIG.cooccurrence_lang)
        if len(store) == 0:
            return None

        lines: list[str] = ["## Vault map  (corpus snapshot at session start)"]
        skipped: list[str] = []

        # A single vault pass: refs feeds both the folders block and the
        # contested scan (one list_files call, not two).
        refs: list = []
        try:
            from silica.driver import DRIVER

            refs = DRIVER.list_files()
        except Exception as e:  # best-effort
            skipped.append("list_files")
            logger.debug("build_vault_map: list_files failed: %s", e)

        # Note count + top folders
        try:
            if refs:
                folder_counts: Counter[str] = Counter(
                    (r.path.rsplit("/", 1)[0] if "/" in r.path else "(root)")
                    for r in refs
                    if getattr(r, "path", "")
                )
                lines.append(f"- Notes: {len(refs)} in {len(folder_counts)} folders")
                top = folder_counts.most_common(max_folders)
                if top:
                    lines.append(
                        "- Top folders: "
                        + ", ".join(f"{f} ({c})" for f, c in top)
                    )
        except Exception as e:  # best-effort
            skipped.append("folders")
            logger.debug("build_vault_map: folders block skipped: %s", e)

        # Contested notes (spec-hermes-coherence §1 leftover): frontmatter
        # `contested: true`, same scan pattern as graph_report/compute.py
        # but via props_of (frontmatter-only, no body) — embedder-free,
        # kernel-only. No line emitted if N == 0.
        try:
            from silica.driver import DRIVER

            contested_names: list[str] = []
            dated_count = 0  # same props pass feeds the silica_timeline hint below
            for ref in refs:
                try:
                    props = DRIVER.props_of(ref)
                except Exception:
                    continue
                if props and props.get("date"):
                    dated_count += 1
                if props and props.get("contested"):
                    contested_names.append(
                        ref.path.rsplit("/", 1)[-1].removesuffix(".md")
                    )
            if contested_names:
                shown = ", ".join(f"[[{n}]]" for n in contested_names[:max_contested])
                extra = len(contested_names) - max_contested
                if extra > 0:
                    shown += f" … +{extra}"
                lines.append(f"⚠ {len(contested_names)} contested notes: {shown}")
            if dated_count:
                lines.append(
                    f"- Timeline: {dated_count} dated notes — "
                    "silica_timeline for chronology/ordering questions"
                )
        except Exception as e:  # best-effort
            skipped.append("contested")
            logger.debug("build_vault_map: contested block skipped: %s", e)

        # Top clusters (Louvain over the concept graph; each community is
        # labelled by its highest-weight stems — community_labels must NOT be
        # used here: it wants communities of note paths, not of stems).
        try:
            from networkx.algorithms.community import louvain_communities

            G = store.to_networkx()
            if G.number_of_nodes():
                deg = dict(G.degree(weight="weight"))
                communities = sorted(
                    louvain_communities(G, seed=42), key=len, reverse=True
                )
                cluster_labels: list[str] = []
                for members in communities[:max_clusters]:
                    top_nodes = sorted(
                        members, key=lambda n: deg.get(n, 0.0), reverse=True
                    )[:2]
                    label = " · ".join(store.node_label(n) for n in top_nodes)
                    if label:
                        cluster_labels.append(label)
                if cluster_labels:
                    lines.append(
                        "- Top clusters: " + ", ".join(cluster_labels)
                    )
        except Exception as e:  # networkx missing or empty graph → skip
            skipped.append("cluster")
            logger.debug("build_vault_map: cluster block skipped: %s", e)

        # Core vocabulary
        try:
            stems = store.top_stems(max_vocab)
            if stems:
                lines.append("- Core vocabulary: " + ", ".join(stems))
        except Exception as e:
            skipped.append("vocabulary")
            logger.debug("build_vault_map: vocabulary block skipped: %s", e)

        # Hub notes — proxy: notes that touch the most distinct concepts
        try:
            ranked = sorted(
                store.paths(),
                key=lambda p: len(store.note_nodes(p)),
                reverse=True,
            )[:max_hubs]
            hub_names = [p.rsplit("/", 1)[-1].removesuffix(".md") for p in ranked]
            if hub_names:
                lines.append(
                    "- Hub notes: " + ", ".join(f"[[{h}]]" for h in hub_names)
                )
        except Exception as e:
            skipped.append("hub")
            logger.debug("build_vault_map: hub block skipped: %s", e)

        # Usage salience (graft G2): which facts recur across RUNS, from the
        # episodic heads' run sets. This is the map's only usage-derived block
        # — PEEK's content ablation (2605.19932 App. B) puts every purely
        # structural filling in the +0.7..+5.7% band while the gains live in
        # "what past work proved useful", and run recurrence is the offline
        # proxy Silica already records for exactly that.
        try:
            from silica.kernel.recall.episodic import EpisodicStore

            heads = EpisodicStore().live_facts()
            recur = sorted((f for f in heads if len(f.runs) >= 2),
                           key=lambda f: (-len(f.runs), f.key))[:6]
            if recur:
                lines.append(
                    "- Recurring facts (by runs): "
                    + ", ".join(f"{f.key} (x{len(f.runs)})" for f in recur)
                )
        except Exception as e:
            skipped.append("recurring")
            logger.debug("build_vault_map: recurring block skipped: %s", e)

        # Tail of log.md — the agent sees what happened recently without
        # having to open the run JSON (Task 2: human-readable append-only journal).
        try:
            from silica.kernel.recall.run_log import tail_log

            recent = tail_log(log_tail)
            if recent:
                lines.append("- Recent log:")
                lines.extend(f"  {ln}" for ln in recent)
        except Exception as e:
            skipped.append("log")
            logger.debug("build_vault_map: log block skipped: %s", e)

        # Only the header → nothing useful: behave like an empty vault. The
        # cooccur index is non-empty here (checked above), so this is a
        # degraded build, not an empty vault: it always has a cause in
        # `skipped`, and the warning below names it.
        if len(lines) == 1:
            _warn_once(
                "empty",
                "vault map came back empty over a non-empty index (%s failed) — "
                "the session starts with no self-model; run `silica doctor`",
                ", ".join(skipped) or "no block succeeded",
            )
            return None
        if skipped:
            _warn_once(
                "partial",
                "vault map is partial, %d block(s) skipped (%s) — run at DEBUG "
                "for the cause",
                len(skipped), ", ".join(skipped),
            )
        return "\n".join(lines)

    except Exception as e:  # the map must never break the session
        _warn_once(
            "failed",
            "vault map failed (%s) — the session starts with no self-model; "
            "run `silica doctor`", e,
        )
        logger.debug("build_vault_map: failed (non-fatal)", exc_info=True)
        return None
