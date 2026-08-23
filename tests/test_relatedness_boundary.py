"""Architectural boundary for the relatedness facade.

kernel/relatedness.py is the single place where the two PROPOSE-legs
(embeddings, co-occurrence) are fused into a note ranking. Relatedness
*queries* must go through it; direct imports of the legs are reserved for:

  - index maintenance (build/refresh/delete entries),
  - pairwise cosine math that does not fit rank fusion (dedup windows,
    salience gates, missing-link thresholds),
  - constructing stores to inject INTO the facade.

This test pins the set of modules allowed to import the legs directly. A new
direct-import site fails here and forces an explicit decision: route through
the facade, or extend the allowlist with a justification.
"""
from pathlib import Path

import re

SILICA_ROOT = Path(__file__).resolve().parent.parent / "silica"

_LEG_IMPORT_RE = re.compile(
    r"from silica\.kernel\.recall\.(embed|cooccurrence) import|"
    r"import silica\.kernel\.recall\.(embed|cooccurrence)\b|"
    # `from silica.kernel.recall import cooccurrence` (any name position,
    # aliased or not) is functionally the same direct leg import — without this
    # alternative it slipped past the regex, defeating the test's documented
    # purpose of forcing an explicit facade-or-allowlist decision. `\b` keeps
    # non-leg names like `embed_signals` from matching.
    r"from silica\.kernel\.recall import [^\n]*\b(embed|cooccurrence)\b"
)

# The legs themselves are out of scope — they ARE the implementation.
LEGS = {"kernel/recall/embed.py", "kernel/recall/cooccurrence.py"}

# module (relative to silica/) → why direct leg access is legitimate
ALLOWED = {
    "kernel/recall/relatedness.py":        "the facade itself",
    "kernel/recall/recall_weights.py":     "constructs recall-outcome weight store to inject into the facade; needs cooccur_key for keyspace normalization",
    "kernel/recall/run_substrate.py":      "constructs stores to inject into the facade",
    "kernel/recall/perception.py":         "constructs stores to inject into the facade (facade_retrieve, the shared fresh-query wiring)",
    "kernel/report/graph_report/embed_signals.py": "pairwise cosine (missing links, dup pairs)",
    "kernel/report/graph_report/cooccur_delta.py": "co-occurrence delta + cosine-band filter + store injection",
    "kernel/recall/graph_export.py":       "cluster labels via CooccurStore.community_labels, not relatedness ranking",
    "kernel/recall/mindmap.py":            "constructs stores to inject into the facade (mindmap latent leg)",
    "kernel/recall/memory_lane.py":        "constructs the memory-lane store pair to inject into the facade (ADR-0019), never ranks",
    "kernel/recall/vault_map.py":          "session-start vault map via CooccurStore (to_networkx/node_label, top_stems), not relatedness ranking",
    "kernel/recall/sync.py":               "index maintenance (invocation-time sweep: build/refresh/prune entries), never ranks",
    "kernel/organize/classify.py":           "L1 tokenizer/concept matching, not relatedness ranking",
    "kernel/text/keyphrase.py":          "pairwise cosine (candidate phrase vs document theme) for concept reranking, not note ranking",
    "kernel/recall/episodic.py":           "pairwise cosine on its own fact vecs (episodic recall scoring), not note relatedness ranking",
    "router/organize_fsm.py":       "L1 co-occurrence classification, not relatedness ranking",
    "router/orchestrator.py":       "co-occurrence index freshness hook (build_index)",
    "router/states/collision.py":   "constructs stores to inject into the facade",
    "router/states/setup.py":       "pairwise cosine (salience theme gate)",
    "kernel/residue.py":            "pairwise cosine (residue theme filter, same gate as salience), not relatedness ranking",
    "router/states/write.py":       "incremental embed index refresh after writes",
    "router/states/finalize.py":    "embed index cleanup on rollback",
    "driver/fs_backend.py":         "embed vector cleanup on note delete/rename (index maintenance), not relatedness ranking",
    "tools/graph.py":               "index refresh tools + raw semantic search by design",
    "tools/runners.py":             "pairwise cosine dedup windows (silica_dedup)",
    "tools/curate.py":              "constructs stores to inject into the facade (orphan candidates)",
    "onboarding/checks.py":         "metadata-only read via the public frozen_lang accessor (doctor language check), no store construction, not relatedness ranking",
    "kernel/report/graph_report/compute.py": "store-file stat only (_index_path, for the report memo's freshness key), no store construction, not relatedness ranking",
    "kernel/report/learner.py":     "concept-centrality read via CooccurStore.note_nodes/adjacency (unexplored-pool ranking), not relatedness ranking",
    "kernel/report/structure.py":   "keyspace normalization (cooccur_key) plus paths/note_nodes/node_label for the burst window (V6), same reads as learner.py; constructs no ranking",
}


def test_leg_imports_are_allowlisted():
    offenders = []
    for path in SILICA_ROOT.rglob("*.py"):
        rel = path.relative_to(SILICA_ROOT).as_posix()
        if rel in ALLOWED or rel in LEGS:
            continue
        if _LEG_IMPORT_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert not offenders, (
        "Direct kernel.embed/kernel.cooccurrence import outside the allowlist: "
        f"{offenders}. Route relatedness queries through kernel/relatedness.py, "
        "or extend ALLOWED in this test with a justification."
    )


def test_allowlist_has_no_stale_entries():
    stale = [
        rel for rel in ALLOWED
        if not (SILICA_ROOT / rel).exists()
        or not _LEG_IMPORT_RE.search((SILICA_ROOT / rel).read_text(encoding="utf-8"))
    ]
    assert not stale, (
        f"Allowlist entries no longer import the legs directly: {stale}. "
        "Remove them so the boundary stays tight."
    )
