# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Lexical retrieval leg — hand-written in-memory postings BM25 + fuzzy match.

Silica's semantic stack (embeddings + cross-encoder + co-occurrence) is weak on
rare tokens, proper nouns, and dates — the exact queries a lexical index nails.
IWE uses BM25 (no Tantivy); we mirror that hand-rolled, matching the codebase
idiom (the co-occurrence and embedding indexes are also hand-managed) rather
than adding a dependency.

ponytail: an in-memory term -> {path: tf} postings map (maintained in
upsert/remove, rebuilt on load — never persisted) gives O(1) df and O(union)
candidates instead of an O(docs) scan per query; swap for a real index
(Tantivy/whoosh) only if the corpus outgrows memory. Fused into RRF by RANK, so
its unbounded BM25 scores never need to be comparable to cosine (spec 1.2).
"""
from __future__ import annotations

import math
import threading
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import orjson

from silica.kernel.recall.paths import DiskSynced

_BM25_K1 = 1.5
_BM25_B = 0.75
_FUZZY_MIN = 0.82  # SequenceMatcher ratio floor for a fuzzy title/key hit


def _index_path() -> Path:
    from silica.kernel.recall import paths
    return paths.index_file("lexical")


def _tokens(text: str) -> list[str]:
    """Tokens for lexical matching — reuse the C1 text seam, surface (unstemmed)
    so proper nouns and dates match verbatim."""
    from silica.kernel.text.text import tokens
    from silica.config import CONFIG
    out: list[str] = []
    for sentence in tokens(text, lang=CONFIG.cooccurrence_lang, stem=False):
        out.extend(surface for _stem, surface in sentence)
    return out


class LexicalStore(DiskSynced):
    def __init__(self, path: Path | None = None):
        self._path = path if path is not None else _index_path()
        self._docs: dict[str, dict[str, int]] = {}   # path -> {term: tf}
        self._len: dict[str, int] = {}               # path -> doc length
        self._name: dict[str, str] = {}              # path -> title/key for fuzzy
        self._postings: dict[str, dict[str, int]] = {}   # DERIVED: term -> {path: tf}
        self._name_lower: dict[str, str] = {}            # DERIVED: path -> name.lower()
        # DiskSynced bookkeeping; the lock guards only the sync/save skeleton,
        # this store's mutators were never thread-safe and that is unchanged.
        self._lock = threading.RLock()
        self._dirty: set[str] = set()
        self._gone: set[str] = set()

    def __len__(self) -> int:
        return len(self._docs)

    def _unindex(self, path: str) -> None:
        """Drop `path` from every posting list of its current terms."""
        for t in self._docs.get(path, {}):
            d = self._postings.get(t)
            if d:
                d.pop(path, None)
                if not d:
                    self._postings.pop(t, None)

    def _reindex(self) -> None:
        """Rebuild the derived postings/name_lower indexes from _docs/_name."""
        from silica.kernel.recall.paths import build_postings
        self._postings = build_postings(self._docs)
        self._name_lower = {path: name.lower() for path, name in self._name.items()}

    def upsert(self, path: str, name: str, body: str) -> None:
        if path in self._docs:
            self._unindex(path)
        toks = _tokens(f"{name}\n{body}")
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        self._docs[path] = tf
        self._len[path] = len(toks)
        self._name[path] = name
        for term, f in tf.items():
            self._postings.setdefault(term, {})[path] = f
        self._name_lower[path] = name.lower()
        self._dirty.add(path)
        self._gone.discard(path)

    def remove(self, path: str) -> None:
        self._unindex(path)
        self._docs.pop(path, None)
        self._len.pop(path, None)
        self._name.pop(path, None)
        self._name_lower.pop(path, None)
        self._gone.add(path)
        self._dirty.discard(path)

    def paths(self) -> list[str]:
        return list(self._docs)

    def query_idf(self, terms: set[str]) -> dict[str, float]:
        """BM25 idf per KNOWN term, for the query-density window scan
        (rerank.best_window_spans). Terms with df=0 are omitted, not given
        the formula's maximum: the postings are C1-tokenized (stopwords
        dropped), the window scan is not, so df=0 cannot distinguish "rare
        in this corpus" from "a stopword the index never stores" — and the
        maximum handed 'what' the same weight as 'fundraiser' (probed on
        conv-26, 2026-08-25). Omitted terms keep the scan's neutral 1.0 via
        w.get, which orders known-rare > unknown > known-common. Empty
        store -> {}, the same abstention shape as rank()."""
        n = len(self._docs)
        if not n:
            return {}
        out: dict[str, float] = {}
        for t in terms:
            df = len(self._postings.get(t, {}))
            if df:
                out[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        return out

    def rank(self, query: str, *, k: int = 25) -> list[tuple[str, float]]:
        if not self._docs:
            return []
        q_terms = _tokens(query)
        n = len(self._docs)
        avgdl = (sum(self._len.values()) / n) if n else 0.0
        q_term_set = set(q_terms)
        df: dict[str, int] = {term: len(self._postings.get(term, {})) for term in q_term_set}
        candidates: set[str] = set().union(
            *(self._postings.get(t, {}).keys() for t in q_term_set)
        )

        bm25: dict[str, float] = {}
        for path in candidates:
            tf = self._docs[path]
            dl = self._len[path] or 1
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f or df.get(term, 0) == 0:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                denom = f + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / (avgdl or 1))
                score += idf * (f * (_BM25_K1 + 1)) / denom
            if score > 0.0:
                bm25[path] = score
        bm25_ranked = sorted(bm25.items(), key=lambda kv: (-kv[1], kv[0]))

        # Fuzzy leg: quick-ratio upper bounds reject before the full O(L^2)
        # ratio() — both are documented upper bounds on ratio(), so this can
        # never drop a real hit. Must scan ALL names, not just BM25 candidates.
        ql = query.strip().lower()
        fuzzy: dict[str, float] = {}
        for path, name_lower in self._name_lower.items():
            sm = SequenceMatcher(None, ql, name_lower)
            if sm.real_quick_ratio() < _FUZZY_MIN or sm.quick_ratio() < _FUZZY_MIN:
                continue
            r = sm.ratio()
            if r >= _FUZZY_MIN:
                fuzzy[path] = r
        fuzzy_ranked = sorted(fuzzy.items(), key=lambda kv: (-kv[1], kv[0]))

        # Fuse the two lexical rankings by rank (RRF-style, local constant).
        fused: dict[str, float] = {}
        for ranking in (bm25_ranked, fuzzy_ranked):
            for rank, (path, _s) in enumerate(ranking):
                fused[path] = fused.get(path, 0.0) + 1.0 / (60 + rank + 1)
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def _read_disk(self) -> dict[str, Any]:
        try:
            if self._path.is_file():
                return orjson.loads(self._path.read_bytes())
        except Exception:
            # Derived index: quarantine for doctor visibility, then
            # reset to empty (a rebuild repopulates it).
            from silica.kernel.recall.paths import quarantine
            quarantine(self._path)
        return {}

    def _take_disk(self, data: dict[str, Any]) -> None:
        docs = {p: dict(tf) for p, tf in data.get("docs", {}).items()}
        lens = dict(data.get("len", {}))
        names = dict(data.get("name", {}))
        for p in self._gone:
            docs.pop(p, None)
            lens.pop(p, None)
            names.pop(p, None)
        for p in self._dirty:
            if p in self._docs:
                docs[p], lens[p], names[p] = self._docs[p], self._len[p], self._name[p]
        self._docs, self._len, self._name = docs, lens, names
        self._reindex()

    def _snapshot(self) -> tuple[dict, dict, dict]:
        return (dict(self._docs), dict(self._len), dict(self._name))

    def _serialize(self, snapshot) -> bytes:
        docs, lens, names = snapshot
        return orjson.dumps({"docs": docs, "len": lens, "name": names})

    def _dirty_sets(self) -> tuple[set[str], set[str]]:
        return (self._dirty, self._gone)

    @classmethod
    def load(cls, path: Path | None = None) -> "LexicalStore":
        store = cls(path)
        store._load()
        return store


_STORE_CACHE: dict[str, "LexicalStore"] = {}


def get_lexical_store() -> "LexicalStore":
    from silica.kernel.recall.paths import path_keyed_singleton
    store = path_keyed_singleton(_STORE_CACHE, str(_index_path()), LexicalStore.load)
    store.sync_from_disk()
    return store


def clear() -> None:
    """Drop all cached stores (test isolation; frees memory on /vault switch)."""
    _STORE_CACHE.clear()
