# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Deterministic co-occurrence graph.

L1 kernel: no LLM, no API, NO embedder dependency. Turns note prose into a
weighted concept graph via a sliding window with decaying weights.

This module is a CANDIDATE GENERATOR (a second PROPOSE-layer alongside
embeddings). It is never authoritative about vault structure; that role
belongs to graph_diff / the driver. Fusion with embeddings lives in a
separate relatedness facade — NOT here.
"""
from __future__ import annotations

import math
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import orjson

from silica.kernel.text import language
from silica.kernel.recall.paths import DiskSynced, in_folder, quarantine

# --- algorithm constants -------------------------------------
NARRATIVE_WEIGHT = 3
GAP = 4               # effective look-back of GAP - 1 = 3 tokens
MIN_TOKEN_LEN = 3
EVIDENCE = "cooccur"


def _index_path() -> Path:
    # Function, not constant: resolves per current vault; tests monkeypatch it.
    from silica.kernel.recall import paths

    return paths.index_file("cooccurrence")


def _index_path_for(vault: str) -> Path:
    """Store path for an explicit `vault`, independent of the global CONFIG
    singleton. Backs `frozen_lang` below — a diagnostic comparing a
    *specific* vault's detected language against *that same vault's* frozen
    store must not silently fall back to whatever vault CONFIG currently
    points at (that would be a false cross-vault mismatch on a vault switch
    or a fresh `SilicaConfig()`). Tests monkeypatch this directly."""
    from silica.kernel.recall import paths

    return paths.index_dir_for(vault) / "cooccurrence.json"


def frozen_lang(vault: str) -> str | None:
    """Public, read-only accessor: the `lang` field frozen into `vault`'s
    persisted co-occurrence store, if one exists on disk.

    This is the only supported way for code outside this module to learn a
    store's frozen language — it owns the on-disk schema (the `lang` key
    inside the store's orjson document) so callers never hand-parse it.
    Never instantiates/builds/mutates a `CooccurStore` (a pure diagnostic
    read, not part of the store's mutation API). `None` when no store file
    exists yet for this vault, or on any read/parse error (degrade, never
    raise).
    """
    try:
        path = _index_path_for(vault)
        if not path.is_file():
            return None
        data = orjson.loads(path.read_bytes())
        lang = data.get("lang")
        return lang if isinstance(lang, str) and lang else None
    except Exception:
        return None

def tokenize(
    text: str,
    stem_lang: str = "english",
    stopword_lang: str | None = None,
) -> list[list[tuple[str, str]]]:
    """Sentences of (stem, surface) pairs — thin delegate to the kernel/text
    seam (C1), kept as public API of this module.

    `stem_lang` is the STORE's frozen stemming language — one stemmer per
    store, since node keys are stemmed tokens and a per-note stemmer would
    split cross-language shared terms. `stopword_lang` is per-NOTE: `None`
    detects it from `text`; pass an explicit language to pin it (e.g.
    matching a short label against store node keys, where detection on a
    2-4 word sample is noise).
    """
    from silica.kernel.text import text as text_seam

    return text_seam.tokens(text, lang=stem_lang, stopword_lang=stopword_lang)


def cooccur_key(path: str) -> str:
    """Canonical key for the co-occurrence keyspace — single source of truth.

    Producers store the stripped path ('notes/foo'); consumers pass graph node
    ids that carry '.md' ('notes/foo.md'). Normalising here, at the store's own
    boundary, means the two can no longer diverge into ghost keys. Idempotent.
    """
    return (path or "").replace("\\", "/").removesuffix(".md")


def build_contribution(
    name: str,
    body: str,
    lang: str = "english",
    concepts: list[str] | None = None,
    strip_fences: bool = False,
) -> dict[str, Any]:
    """Turn a note's text into its per-note co-occurrence contribution.

    Strips frontmatter + media, tokenizes per sentence, then for each token
    links it back to the previous `GAP - 1` tokens with a decaying weight:
        distance 1 -> 3 (narrative), distance 2 -> 2, distance 3 -> 1 (gap scan)
    Edges are directed earlier->later. Weights accumulate.

    `concepts` (#9, Marwitz et al. 2026): optional LLM-extracted, normalized
    concept phrases. Each is appended as its own sentence so its stems become
    nodes and its words co-occur through the same window logic — reinforcing
    LLM-validated concepts above rule-based body noise. `None`/`[]` leaves the
    contribution byte-identical to a body-only build (graceful degradation).
    """
    from silica.kernel.text.text import clean_body, is_drawing_note

    # Excalidraw drawings carry no prose — skip entirely so their element-id /
    # SVG soup never becomes nodes (empty contribution = the note indexes clean).
    if is_drawing_note(body):
        return {"nodes": {}, "edges": []}

    # strip_fences: prose vaults treat code blocks as noise; code vaults keep
    # their identifiers as graph signal (C1 fork ⚑). Keyed on the manifest
    # `sources` by the caller. Math and images never become nodes.
    text = f"{name}\n\n{clean_body(body, fences=strip_fences)}"
    if concepts:
        concept_sentences = ". ".join(c.strip() for c in concepts if c and c.strip())
        if concept_sentences:
            text = f"{text}\n\n{concept_sentences}."

    node_surface_counts: dict[str, dict[str, int]] = {}
    edge_acc: dict[tuple[str, str], float] = {}

    for sentence in tokenize(text, stem_lang=lang):
        stems = [stem for (stem, _s) in sentence]
        for i, (stem, surface) in enumerate(sentence):
            node_surface_counts.setdefault(stem, {})
            node_surface_counts[stem][surface] = node_surface_counts[stem].get(surface, 0) + 1
            # link back up to GAP - 1 tokens with decaying weight (3, 2, 1)
            for dist in range(1, GAP):
                j = i - dist
                if j < 0:
                    break
                src = stems[j]
                if src == stem:
                    continue
                weight = float(NARRATIVE_WEIGHT + 1 - dist)  # 3, 2, 1
                if weight <= 0:
                    continue
                edge_acc[(src, stem)] = edge_acc.get((src, stem), 0.0) + weight

    nodes = {
        stem: {
            "label": max(surfaces.items(), key=lambda kv: kv[1])[0],
            "count": sum(surfaces.values()),
        }
        for stem, surfaces in node_surface_counts.items()
    }
    edges = [[f, t, w] for (f, t), w in edge_acc.items()]
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Cached accessor (the seam — Fix 3, twin of embed.get_store)
# ---------------------------------------------------------------------------

_STORE_CACHE: dict[str, "CooccurStore"] = {}


def get_cooccur_store(lang: str = "english") -> "CooccurStore":
    """Return the shared CooccurStore for the current vault's index.

    Process-lifetime singleton keyed by resolved index path (twin of
    ``embed.get_store``). ``lang`` only seeds an empty store; a loaded store
    keeps the language frozen on disk. Use ``clear()`` in tests.
    """
    from silica.kernel.recall.paths import path_keyed_singleton
    store = path_keyed_singleton(_STORE_CACHE, str(_index_path()), lambda: CooccurStore(lang=lang))
    store.sync_from_disk()
    return store


def clear() -> None:
    """Drop all cached co-occurrence stores (test isolation; /vault switch)."""
    _STORE_CACHE.clear()


class CooccurStore(DiskSynced):
    """orjson-backed store of per-note co-occurrence contributions.

    The global graph is a lazy, cached aggregation over the per-note
    contributions (mirrors EmbedStore's lazy search matrix). Storing per-note
    contributions makes refresh a dict replacement — no fragile weight
    subtraction.
    """

    def __init__(self, path: Path | None = None, lang: str = "english"):
        self._path = path if path is not None else _index_path()
        self._notes: dict[str, dict[str, Any]] = {}
        self.lang = lang
        # lazy aggregated graph caches (scope=None only)
        self._adj: dict[str, dict[str, float]] | None = None
        self._labels: dict[str, str] | None = None
        # scoped aggregate + in-scope path caches: a folder-scoped report calls
        # _aggregate(scope)/paths_in_scope(scope) once per note with the SAME
        # scope, which was an O(N) rebuild each time -> O(N^2). Keyed by scope,
        # cleared with the rest on any mutation.
        self._scoped_agg_cache: dict[str, tuple[dict[str, dict[str, float]], dict[str, str]]] = {}
        self._scope_paths_cache: dict[str, list[str]] = {}
        # per-path note_nodes() derived-dict cache (mirrors the _adj cache)
        self._note_nodes_cache: dict[str, dict[str, int]] = {}
        # stem -> {path: count} inverted index cache (mirrors the _adj cache)
        self._stem_postings: dict[str, dict[str, int]] | None = None
        # ({path: total stem count}, mean) for the BM25 length term
        self._doc_lengths: tuple[dict[str, int], float] | None = None
        # note-to-note edges (CORRELATE, ADR-0029): a memo of
        # correlate.compute_edges over the contributions, dropped with every
        # other derived cache. Never persisted - see note_adjacency().
        self._note_adj: dict[str, dict[str, float]] | None = None
        # DiskSynced bookkeeping: keys upserted / deleted since the last sync
        # with the file. The lock guards only the sync/save skeleton; this
        # store's mutators were never thread-safe and that is unchanged here.
        self._lock = threading.RLock()
        self._dirty: set[str] = set()
        self._gone: set[str] = set()
        self._load()

    # --- caches ---
    def _invalidate(self) -> None:
        self._adj = None
        self._labels = None
        self._scoped_agg_cache = {}
        self._scope_paths_cache = {}
        self._note_nodes_cache = {}
        self._stem_postings = None
        self._doc_lengths = None
        self._note_adj = None

    # --- I/O ---
    def _read_disk(self) -> dict[str, Any]:
        src = self._path
        # No legacy soft-migration: inheriting the global index copied old-schema
        # keys forward, and since build_index never GCs they survived as orphans
        # poisoning aggregations. Per-vault keying is stable; a fresh store loads
        # empty and /cooccur rebuilds it clean.
        if not src.exists():
            return {}
        try:
            return orjson.loads(src.read_bytes())
        except Exception:
            # Derived index: quarantine for doctor visibility, then
            # rebuild from empty (/cooccur restores it).
            quarantine(src)
            return {}

    def _take_disk(self, data: dict[str, Any]) -> None:
        notes = data.get("notes", {})
        # Files written before ADR-0029 carry a "note_edges" section. It is
        # read by nobody and never written again: the edges are derived from
        # the contributions on demand (note_adjacency), so a stale section is
        # simply ignored rather than migrated.
        # The frozen language follows the file unless this process has
        # already stemmed contributions under its own: mixing them is the
        # cross-language node-splitting build_index guards against.
        if not self._dirty:
            self.lang = data.get("lang", self.lang)
        mine = {k: self._notes[k] for k in self._dirty if k in self._notes}
        for k in self._gone:
            notes.pop(k, None)
        notes.update(mine)
        self._notes = notes
        self._invalidate()

    def _snapshot(self) -> tuple[str, dict[str, Any]]:
        # Contributions are replaced wholesale, so one level of copy is a
        # consistent view.
        return (self.lang, dict(self._notes))

    def _serialize(self, snapshot) -> bytes:
        lang, notes = snapshot
        # No OPT_INDENT_2: machine-only derived index, pretty-printing is pure
        # I/O tax (Fix 2A). orjson defaults to compact output.
        return orjson.dumps({"version": 1, "lang": lang, "notes": notes})

    def _dirty_sets(self) -> tuple[set, ...]:
        return (self._dirty, self._gone)

    # --- mutation ---
    def upsert_note(self, path: str, contribution: dict[str, Any]) -> None:
        key = cooccur_key(path)
        self._notes[key] = contribution
        self._dirty.add(key)
        self._gone.discard(key)
        self._invalidate()

    def delete_note(self, path: str) -> None:
        key = cooccur_key(path)
        self._notes.pop(key, None)
        self._gone.add(key)
        self._dirty.discard(key)
        self._invalidate()

    # --- note edges (derived at read; CORRELATE, ADR-0029) ---
    def note_adjacency(self) -> dict[str, dict[str, float]]:
        """Symmetric {path: {neighbour: jaccard}} over the whole store, memoized.

        A pure function of the contributions, so it lives in the same cache
        family as `_stem_postings` and `doc_lengths` and is dropped by
        `_invalidate()` on any mutation or reload. Not persisted: the section
        this replaces needed prune-on-delete, orphan-prune-on-load and a
        dirty-edge overlay in the disk sync, and every one of those was a seam
        where the file and the contributions could disagree. Recomputing is
        well under a second on 709 notes (correlate.compute_edges), paid once
        per vault mutation by the first reader.
        """
        if self._note_adj is None:
            from silica.kernel.link.correlate import compute_edges

            self._note_adj = compute_edges(
                {path: self.note_nodes(path) for path in self._notes},
            )
        return self._note_adj

    def note_edges_for(self, path: str) -> dict[str, float]:
        """Derived neighbours of `path` -> score, both directions.

        O(deg) per call off the memoized adjacency, so a path walk that asks
        per visited node (mindmap.reading_path) stays O(V+E). Fresh dict per
        call, like note_nodes: callers may mutate their result.
        """
        return dict(self.note_adjacency().get(cooccur_key(path), {}))

    # --- lookup ---
    def paths(self) -> list[str]:
        return list(self._notes.keys())

    def __len__(self) -> int:
        return len(self._notes)

    def note_nodes(self, path: str) -> dict[str, int]:
        """Return {stem: count} for one note's contribution ({} if absent).

        Public read access used by the relatedness facade to build the
        concept->notes inverted index (granularity reconciliation lives there,
        not here). The derived dict is computed once per path and cached
        (invalidated via `_invalidate()`, alongside `_adj`/`_labels`); each
        call still returns a fresh copy so callers may mutate their result
        without corrupting the cache.
        """
        key = cooccur_key(path)
        cached = self._note_nodes_cache.get(key)
        if cached is None:
            contrib = self._notes.get(key)
            cached = (
                {
                    stem: int(meta.get("count", 1))
                    for stem, meta in contrib.get("nodes", {}).items()
                }
                if contrib else {}
            )
            self._note_nodes_cache[key] = cached
        return dict(cached)

    def stem_postings(self) -> dict[str, dict[str, int]]:
        """Inverted index stem -> {path: count}, lazily built from stored notes and
        invalidated with the other derived caches. Gives df (len of a posting) and
        the candidate set (union of query-stem postings) without an all-notes scan."""
        if self._stem_postings is None:
            from silica.kernel.recall.paths import build_postings
            self._stem_postings = build_postings({p: self.note_nodes(p) for p in self._notes})
        return self._stem_postings

    def doc_lengths(self) -> tuple[dict[str, int], float]:
        """({path: total stem count}, mean length) for the BM25 length term.

        Derived like stem_postings, from the same cached note_nodes, and dropped
        by the same _invalidate() — so a write (upsert_note/delete_note) refreshes
        avgdl instead of leaving a stale one behind. Living ON the store also
        keeps two stores from sharing lengths: the memory lane passes a DIFFERENT
        store to the same ranking seam, and borrowing the active vault's avgdl
        there would be a silent cross-vault bug.
        """
        if self._doc_lengths is None:
            lens = {p: sum(self.note_nodes(p).values()) for p in self._notes}
            self._doc_lengths = (lens, (sum(lens.values()) / len(lens)) if lens else 1.0)
        return self._doc_lengths

    def top_stems(self, n: int = 20) -> list[str]:
        """Top-n stem nodes by total weight across all notes, as display labels.

        Pure aggregation over stored contributions (embedder-free) — backs the
        '## Vault vocabulary' substrate section.
        """
        totals: dict[str, float] = {}
        for path in self.paths():
            for stem, weight in self.note_nodes(path).items():
                totals[stem] = totals.get(stem, 0.0) + weight
        if not totals:
            return []
        _, labels = self._aggregate()
        ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [labels.get(stem, stem) for stem, _ in ranked]

    # --- aggregation (lazy, scope=None cached) ---
    def _aggregate(self, scope: str | None = None) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
        """Build undirected adjacency {stem: {neighbor: weight}} + label map.

        scope, if given, restricts to notes whose path == scope or starts with
        scope + "/" (folder scoping, context-level filtering).
        Both scope=None and scoped results are cached (per-scope), cleared on
        any mutation via _invalidate().
        """
        if scope is None and self._adj is not None and self._labels is not None:
            return self._adj, self._labels
        if scope is not None and scope in self._scoped_agg_cache:
            return self._scoped_agg_cache[scope]

        adj: dict[str, dict[str, float]] = {}
        label_counts: dict[str, dict[str, int]] = {}
        for path, contrib in self._notes.items():
            if not in_folder(path, scope):
                continue
            for stem, meta in contrib.get("nodes", {}).items():
                lc = label_counts.setdefault(stem, {})
                lc[meta["label"]] = lc.get(meta["label"], 0) + int(meta.get("count", 1))
            for f, t, w in contrib.get("edges", []):
                adj.setdefault(f, {})[t] = adj.setdefault(f, {}).get(t, 0.0) + w
                adj.setdefault(t, {})[f] = adj.setdefault(t, {}).get(f, 0.0) + w
        labels = {
            stem: max(surfaces.items(), key=lambda kv: kv[1])[0]
            for stem, surfaces in label_counts.items()
        }
        if scope is None:
            self._adj, self._labels = adj, labels
        else:
            self._scoped_agg_cache[scope] = (adj, labels)
        return adj, labels

    def paths_in_scope(self, scope: str | None) -> list[str]:
        """In-scope note paths (cached per scope). scope=None returns all paths.

        Lets scope-repeating consumers (a folder-scoped report ranking every
        note) avoid re-filtering all paths per call.
        """
        if not scope:
            return self.paths()
        cached = self._scope_paths_cache.get(scope)
        if cached is None:
            cached = [p for p in self._notes if in_folder(p, scope)]
            self._scope_paths_cache[scope] = cached
        return list(cached)

    def node_label(self, stem: str) -> str:
        _adj, labels = self._aggregate()
        return labels.get(stem, stem)

    def community_labels(
        self,
        communities: list[set[str]],
        *,
        terms: int = 2,
    ) -> dict[int, str]:
        """Name each community using class-based TF-IDF over the concept index.

        Returns {community_index: label_string}. Communities whose member notes
        are all absent from the store are omitted (the caller falls back to its
        own "Cluster N" label).  Pure read-only — never calls save() and never
        mutates self._notes.

        Algorithm (BERTopic-style c-TF-IDF):
          1. For each community i, build tf_i = sum of note_nodes(path) over
             all member paths (absent paths contribute nothing).
          2. Drop communities with an empty tf_i; they are excluded from N and df.
          3. N = number of non-empty communities.
          4. df[stem] = number of non-empty communities containing that stem.
          5. score(stem, i) = tf_i[stem] * log(1 + N / df[stem]).
          6. Rank by (-score, stem) — score desc, stem asc as tie-break. Take top
             `terms` stems; resolve each to its surface via node_label(stem).
          7. Join surfaces with " · ".
        """
        if not communities:
            return {}

        # Step 1 — build per-community term-frequency vectors
        tf_list: list[dict[str, int]] = []
        valid_indices: list[int] = []
        for i, members in enumerate(communities):
            tf: Counter[str] = Counter()
            for path in members:
                tf.update(self.note_nodes(path))
            if tf:
                tf_list.append(tf)
                valid_indices.append(i)

        if not tf_list:
            return {}

        # Steps 3–4 — N and document frequency
        N = len(tf_list)
        df = Counter(stem for tf in tf_list for stem in tf)

        # Steps 5–7 — score, rank, surface, label
        result: dict[int, str] = {}
        ranked_by_i: dict[int, list[str]] = {}
        for orig_i, counts in zip(valid_indices, tf_list):
            scored = [
                (stem, count * math.log(1 + N / df[stem]))
                for stem, count in counts.items()
            ]
            ranked = sorted(scored, key=lambda kv: (-kv[1], kv[0]))
            ranked_by_i[orig_i] = [stem for stem, _score in ranked]
            top_stems = ranked_by_i[orig_i][:terms]
            surfaces = [self.node_label(stem) for stem in top_stems]
            result[orig_i] = " · ".join(surfaces)

        # Step 8 — dedup: two communities sharing top stems came out with the
        # SAME label ("memory · agents" twice, 2026-08-15), indistinguishable
        # in every legend. Extend colliding labels with their next-ranked stem
        # until unique (or the ranking runs dry — then the tie stands).
        seen: dict[str, int] = {}
        for orig_i in sorted(result):
            label = result[orig_i]
            if label not in seen:
                seen[label] = orig_i
                continue
            depth = terms
            while label in seen and depth < len(ranked_by_i[orig_i]):
                depth += 1
                stems = ranked_by_i[orig_i][:depth]
                label = " · ".join(self.node_label(s) for s in stems)
            if label not in seen:
                result[orig_i] = label
                seen[label] = orig_i

        return result

    def adjacency(self, scope: str | None = None) -> dict[str, dict[str, float]]:
        """Cached undirected adjacency {stem: {neighbour: weight}}, no NetworkX.

        The same aggregated data `to_networkx()` wraps in a graph object,
        exposed directly for callers that only need a neighbour-weight lookup
        (e.g. profile expansion) rather than graph algorithms — skips paying
        to construct a graph object just to immediately walk it.
        """
        adj, _labels = self._aggregate(scope=scope)
        return adj

    def to_networkx(self, scope: str | None = None):
        """Return an undirected weighted nx.Graph for consumers (centrality,
        delta-vs-wikilink in graph_report). Built from aggregated adjacency.

        For a plain neighbour-weight lookup instead of graph algorithms, use
        `adjacency()` — it skips the per-call graph construction below.
        """
        import networkx as nx

        adj, _labels = self._aggregate(scope=scope)
        G = nx.Graph()
        for stem, nbrs in adj.items():
            G.add_node(stem)
            for nb, w in nbrs.items():
                # undirected: add once; adjacency already symmetric
                if not G.has_edge(stem, nb):
                    G.add_edge(stem, nb, weight=w)
        return G

    def neighbors(self, concept: str, k: int = 10, scope: str | None = None) -> list[dict[str, Any]]:
        """Top-k co-occurring concepts for `concept`. Returns [] if absent.

        Never raises and never touches the embedder — this is the stable leg of
        the relatedness fusion (works with LM Studio down).
        """
        concept = concept.strip()
        if not concept:
            return []
        from silica.kernel.text.text import stem_word
        stem = stem_word(concept.lower(), lang=self.lang)  # 'auto' falls back inside the seam
        adj, labels = self._aggregate(scope=scope)
        nbrs = adj.get(stem)
        if not nbrs:
            return []
        ranked = sorted(nbrs.items(), key=lambda kv: (-kv[1], kv[0]))
        return [
            {"concept": labels.get(nb_stem, nb_stem), "weight": w, "evidence": EVIDENCE}
            for nb_stem, w in ranked[:k]
        ]


def _strip_fences_for_active_vault() -> bool:
    """True ⇒ drop ```code``` blocks before tokenizing (prose vault noise).

    Keyed on the active vault's manifest `sources` (ADR-0014): a vault that
    declares `code` keeps code-block identifiers as graph signal; a prose-only
    vault treats them as noise. Degrades to False (legacy: keep fences) on any
    failure, so a missing/broken manifest never changes today's behavior.
    """
    from silica.kernel import vault_manifest

    try:
        return "code" not in vault_manifest.get_active_manifest().sources
    except Exception:
        return False


def build_index(
    notes: list[tuple[str, str, str]],
    *,
    store: CooccurStore | None = None,
    lang: str | None = None,
    force: bool = False,
    refreeze: bool = False,
    concepts_by_path: dict[str, list[str]] | None = None,
    save: bool = True,
    prune: bool = False,
    folder: str = "",
) -> CooccurStore:
    """Build/refresh the co-occurrence store from (path, name, body) tuples.

    Mirrors embed.build_index. Incremental: skips notes already present unless
    `force`. Returns the store. ``save=False`` (Fix A) defers persistence to a
    single end-of-run flush.

    `force` and `refreeze` are DIFFERENT axes. `force` is per-note replacement
    semantics ("re-process notes already present, replacing their prior
    contribution — never inflate"); the post-write freshness hook uses it for
    every incremental batch, so it must never touch the frozen language.
    `refreeze` is the store-level language axis: only a deliberate rebuild
    (/cooccur --force, the doctor remedy for a wrong-frozen store) passes
    refreeze=True to re-detect `store.lang` from the batch sample.

    `concepts_by_path` (#9): optional map of note path -> LLM-extracted concept
    phrases, forwarded into build_contribution to reinforce those concepts.

    `prune`: if True, `notes` is the AUTHORITATIVE live set for `folder` — drop
    (and un-edge) nodes under `folder` whose note is absent from it (deleted
    out-of-band). Off by default: incremental callers pass a partial `notes`,
    so pruning against it would delete the unlisted rest. `folder` scopes it.
    """
    if store is None:
        store = get_cooccur_store(lang=lang or "english")
    requested = lang or store.lang
    # Sticky freeze: store.lang is frozen at FIRST build. An "auto" request
    # against an already-populated store with a concrete frozen language must
    # NOT re-detect from just this (incremental, possibly foreign-language)
    # batch — that would flip the stemmer under already-stemmed node keys
    # (cross-language node-splitting). Re-detection is reserved for: the
    # first build (store has no notes yet) and explicit `refreeze=True`
    # rebuilds. NOT keyed on `force`: the write hook passes force=True with
    # replacement (not rebuild) intent on every batch.
    if requested == "auto" and not refreeze and store.paths() and store.lang != "auto":
        use_lang = store.lang
    else:
        use_lang = language.resolve(requested, "\n".join(b for _p, _n, b in notes[:50]))
    store.lang = use_lang  # freeze resolved language (no 'auto' persisted)
    cmap = concepts_by_path or {}
    strip_fences = _strip_fences_for_active_vault()
    for path, name, body in notes:
        if not force and path in store._notes:  # O(1); paths() rebuilds a full list per iter
            continue
        store.upsert_note(
            path,
            build_contribution(
                name, body, lang=use_lang, concepts=cmap.get(path), strip_fences=strip_fences
            ),
        )
    if prune:
        from silica.kernel.recall.paths import in_folder
        live = {path for path, _, _ in notes}
        for p in [p for p in store.paths() if p not in live and in_folder(p, folder)]:
            store.delete_note(p)
    if save:
        store.save()
    return store


def refresh_note(
    path: str,
    name: str,
    body: str,
    *,
    store: CooccurStore | None = None,
    lang: str | None = None,
) -> CooccurStore:
    """Re-build a single note's contribution and persist (freshness hook).

    Replacement, not accumulation: the note's prior contribution is overwritten,
    so weights never inflate on re-processing.
    """
    if store is None:
        store = get_cooccur_store(lang=lang or "english")
    requested = lang or store.lang
    # Sticky freeze — see build_index's matching comment. refresh_note has no
    # `force`; it is always incremental (a single note), so re-detection is
    # reserved for the first build (store has no notes yet).
    if requested == "auto" and store.paths() and store.lang != "auto":
        use_lang = store.lang
    else:
        use_lang = language.resolve(requested, body)
    store.lang = use_lang  # freeze resolved language (no 'auto' persisted)
    store.upsert_note(
        path,
        build_contribution(name, body, lang=use_lang, strip_fences=_strip_fences_for_active_vault()),
    )
    store.save()
    return store
