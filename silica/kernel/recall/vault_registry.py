# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""The vaults this machine knows, and how well each answers one question.

Discovery reads what already exists instead of keeping a registry: the active
vault, the personal-memory vault, every vault a Silica process has served on
this machine (the sweep leaves `vault.json` in the vault's index namespace,
because the namespace key is a one-way digest), and Obsidian's own vault list;
the last two filtered to the folders Silica has adopted (a `vault.yaml`, the
same test `capture.find_vault` applies).

Scoring is two-stage. Each vault's own indexes NOMINATE a pool of notes
(lexical top-pool where a lexical.json exists, embedding top-pool where
embeddings do); the cross-encoder reranker, when one is configured, scores the
pool and that best score is what orders the rows, because it is the one number
comparable between vaults. Bench (scripts/bench_vault_router.py, 2026-09-01,
4 indexed vaults, 42 queries with a known home + 10 with none, bge-reranker-
v2-m3): the reranker's best score routed 42/42 at every pool size; top cosine
30/42, cosine spread 28/42, a two-term lexical hit count 22/42 (an earlier
4-query probe had already seen a union count inflated 250/270 by one
corpus-wide token). So those two numbers never order anything and stay out of
the rows unless a caller asks for diagnostics; a row the reranker did not
score says so (`scored: false`) and sorts last. The pool is wider than the
titles shown (`k`) because the reranker can only promote a note stage one put
forward. `home` is the abstention: the vaults whose best score clears a floor
measured on the same bench, under the same model only. Stores load through
the same cache the peek uses, so scoring a vault pre-pays reading it. Routing,
not fusion (ADR-0019 keeps recall at two lanes): the caller peeks at ONE vault.
"""
from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

from silica.kernel.recall.rerank import _WINDOW_CHARS, best_window

# How deep the lexical probe counts. `hits` is a count, not a ranking, so the
# pool only has to be larger than any number worth reporting exactly.
_PROBE_POOL = 50

# Stage-one pool per stage (lexical, embed), independent of the titles shown.
# Bench 2026-09-01: nominee recall of the labelled note 0.929 at 3, 0.952 at 6
# and flat through 25, routing 42/42 at every size, and the reranked top
# title matched the label less often as the pool widened (0.786 at 3 and 6,
# 0.714 at 25). Six is the knee: reopen if a vault's nominee recall drops
# below 0.9 on the bench.
_NOMINEE_POOL = 6

# Abstention floor for `home`, a logit of THIS cross-encoder (prefix-matched
# against the served model's name, "-Q8_0" and friends included). Bench
# 2026-09-01, pool 6: the balanced optimum was -2.06 (40/42 homed kept, 10/10
# homeless refused), but the costs are not symmetric: a refused home loses
# the answer, an admitted stranger costs one 0.06 s peek. -2.5 keeps 41/42
# (the one below, -5.97, is a stage-one miss no floor can fix), admits 1/10
# homeless (a query sharing the token "test" with a vault of hypothesis-test
# notes, -2.23) and names a second home for 2/42. Scores also move ~0.03
# between batch sizes, so a floor must not sit on a measured point. Any other
# model gets no floor; re-run the bench before trusting this on another
# vault set.
_HOME_FLOOR = -2.5
_HOME_FLOOR_MODEL = "bge-reranker-v2-m3"

# The slice of a nominee the reranker reads: rerank's own document budget, so a
# vault is judged on the text a peek would score, and a change there moves both.
_RERANK_WINDOW = _WINDOW_CHARS


def _obsidian_json() -> Path:
    """Obsidian's vault registry for this OS; the first candidate when none exists."""
    home = Path.home()
    if sys.platform == "darwin":
        cands = [home / "Library/Application Support/obsidian/obsidian.json"]
    elif os.name == "nt":
        cands = [Path(os.environ.get("APPDATA") or (home / "AppData/Roaming")) / "obsidian/obsidian.json"]
    else:
        cands = [home / ".config/obsidian/obsidian.json",
                 home / ".var/app/md.obsidian.Obsidian/config/obsidian/obsidian.json",
                 home / "snap/obsidian/current/.config/obsidian/obsidian.json"]
    for c in cands:
        if c.is_file():
            return c
    return cands[0]


def obsidian_vaults() -> list[Path]:
    try:
        data = json.loads(_obsidian_json().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # No Obsidian on this machine, or its registry mid-rewrite: discovery
        # degrades to the vaults config names, which is still a correct list.
        return []
    out: list[Path] = []
    for entry in (data.get("vaults") or {}).values():
        p = (entry or {}).get("path")
        if p:
            out.append(Path(p))
    return out


def is_adopted(path: Path) -> bool:
    """A folder Silica has been pointed at: `declare_write_dir` writes the
    manifest on first launch, so this is the same test `capture.find_vault` uses."""
    return (Path(path) / "vault.yaml").is_file()


def served_vaults() -> list[Path]:
    """Vaults some Silica process has swept on this machine, from the
    `vault.json` markers in ~/.silica/index/*/ (newest marker first)."""
    from silica.kernel.recall.paths import _SILICA_HOME

    out: list[tuple[float, Path]] = []
    try:
        markers = list((_SILICA_HOME / "index").glob("*/vault.json"))
    except OSError:
        return []
    for m in markers:
        try:
            out.append((m.stat().st_mtime, Path(json.loads(m.read_text(encoding="utf-8"))["path"])))
        except (OSError, ValueError, KeyError, TypeError):
            continue  # a torn or foreign marker names nothing; the other sources still do
    return [p for _t, p in sorted(out, reverse=True)]


def known_vaults() -> list[Path]:
    """Resolved vault paths, deduplicated: active first, then the memory vault,
    then adopted vaults served here (newest first), then Obsidian's adopted
    vaults in registry order."""
    from silica.config import CONFIG
    from silica.kernel.recall.memory_lane import memory_vault

    seen: set[Path] = set()
    out: list[Path] = []

    def add(p: str | Path) -> None:
        try:
            r = Path(p).expanduser().resolve()
        except OSError:
            return
        if r in seen or not r.is_dir():
            return
        seen.add(r)
        out.append(r)

    active = (getattr(CONFIG, "vault_path", "") or "").strip()
    if active:
        add(active)
    mem = memory_vault()  # None when it coincides with the active vault or is absent
    if mem is not None:
        add(mem)
    for p in served_vaults() + obsidian_vaults():
        if is_adopted(p):
            add(p)
    return out


def resolve_known(vault: str) -> Path:
    """The resolved path of a vault a session may READ: adopted (`vault.yaml`)
    or configured (the active vault, the memory vault, which carries no
    manifest). ValueError otherwise, worded for the model that passed it, and
    the one shared rule behind `silica_recall(vault=)` and `silica_read_note(vault=)`."""
    target = Path(vault).expanduser().resolve()
    if is_adopted(target) or target in known_vaults():
        return target
    raise ValueError(f"not a Silica vault (no vault.yaml): {target}; "
                     "silica_vaults lists the vaults this machine knows")


_LEXICAL_CACHE: dict = {}


def lexical_for(vault: Path):
    """A vault's lexical store as it lies on disk, or None when absent/empty.
    Cached per index path and re-read on another process's write (DiskSynced)."""
    from silica.kernel.recall.lexical import LexicalStore
    from silica.kernel.recall.paths import index_dir_for, path_keyed_singleton

    p = index_dir_for(str(vault)) / "lexical.json"
    if not p.is_file():
        return None
    store = path_keyed_singleton(_LEXICAL_CACHE, str(p), lambda: LexicalStore.load(p))
    store.sync_from_disk()
    return store if len(store) else None


def coverage(vault: Path) -> dict:
    """Which recall legs a vault's index namespace can serve.

    `level`: "indexed" (embed or cooccur present: silica_recall will answer),
    "lexical-only" (silica_vaults can score it, recall cannot see it), "cold".
    """
    from silica.kernel.recall.paths import index_dir_for

    idx = index_dir_for(str(vault))

    def present(name: str) -> bool:
        # Size, not a load: "{}" is an empty store, and an 18MB embeddings file
        # must not be parsed just to say it exists.
        try:
            return (idx / name).stat().st_size > 2
        except OSError:
            return False

    embed, cooccur = present("embeddings.json"), present("cooccurrence.json")
    lex = lexical_for(vault)
    level = "indexed" if (embed or cooccur) else ("lexical-only" if lex is not None else "cold")
    return {"level": level, "embed": embed, "cooccur": cooccur, "lexical": lex is not None,
            "notes": len(lex) if lex is not None else None}


def _query_vec(query: str):
    """The query embedding, or None when the embedder is unreachable."""
    from silica.agent.providers import EMBED_ERRORS, get_embedder
    from silica.config import CONFIG

    try:
        return get_embedder(CONFIG).embed([query])[0]
    except EMBED_ERRORS:
        # Offline embedder: embed-nominated vaults put nothing forward. Coverage
        # (the index exists) stays true and the lexical stage still runs, so
        # the scoreboard degrades to a list instead of failing the call.
        return None


def _embed_store(vault: Path):
    """A vault's embed store via the lane module, which owns the rule "the active
    vault's store is the process singleton, any other is read as it lies",
    and is the one place allowed to construct leg stores (test_relatedness_boundary)."""
    from silica.kernel.recall.memory_lane import embed_store_for

    return embed_store_for(vault)


def _reranker():
    """The configured cross-encoder, or None. `get_reranker` only constructs a
    client; every failure surfaces later as `scores() is None`."""
    from silica.agent.providers import get_reranker
    from silica.config import CONFIG

    return get_reranker(CONFIG)


def reranker_name(reranker) -> str | None:
    """The model a reranker's scores are on the scale of (a served endpoint's
    model, or the primary of a fallback pair). Reported with every ranked
    reply because a rerank score is a logit of ONE cross-encoder: the same
    number means something else under another model."""
    if reranker is None:
        return None
    primary = getattr(reranker, "primary", None)
    return getattr(primary if primary is not None else reranker, "model", None) or None


def _body(vault: Path, path: str) -> str:
    from silica.kernel.write import frontmatter

    try:
        _data, _raw, body = frontmatter.split(
            (vault / (path + ".md")).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return ""
    return body or ""


def nominate(vault: Path, query: str, *, pool: int = _NOMINEE_POOL, query_vec=None) -> dict:
    """Stage one: what a vault's own indexes put forward for `query`.

    ``lexical`` = up to `pool` ``(path, name)`` in BM25 order; ``embed`` = up to
    `pool` ``(path, name, cosine)`` in cosine order; ``stages`` names the ones
    that ran; ``hits`` = notes matching two distinct query terms (lexical index
    only); ``best`` = top cosine; ``notes`` = the store's size when one loaded.
    ``query_vec`` is a zero-arg callable returning the query embedding (or
    None), so a scoreboard embeds the question once for every vault.
    """
    out: dict = {"stages": [], "lexical": [], "embed": [], "hits": None, "best": None, "notes": None}
    lex = lexical_for(vault)
    if lex is not None:
        out["stages"].append("lexical")
        out["hits"] = lex.match_count(query)
        out["notes"] = len(lex)
        out["lexical"] = [(p, p.rsplit("/", 1)[-1]) for p, _score in lex.rank(query, k=pool)]
    es = _embed_store(vault)
    if es is not None:
        if out["notes"] is None:
            out["notes"] = len(es)
        vec = query_vec() if query_vec is not None else _query_vec(query)
        if vec is not None:
            from silica.kernel.recall.relatedness import _rank_embeddings_from_vec

            ranked = _rank_embeddings_from_vec(es, vec, k=pool, exclude=set()) or []
            if ranked:
                out["stages"].append("embed")
                out["best"] = round(float(ranked[0][2]), 3)
                out["embed"] = [(p, n, float(s)) for p, n, s in ranked]
    return out


def pool_union(nominated: dict, pool: int) -> list[tuple[str, str]]:
    """The nominees the reranker sees: lexical top-`pool`, then embed top-`pool`
    not already present. Lexical first because its order carries rare tokens
    and names, which is what a title-shaped question is made of."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for p, n in nominated["lexical"][:pool]:
        if p not in seen:
            seen.add(p)
            out.append((p, n))
    for p, n, _cos in nominated["embed"][:pool]:
        if p not in seen:
            seen.add(p)
            out.append((p, n))
    return out


def rerank_nominees(vault: Path, query: str, nominees: list[tuple[str, str]], reranker) -> list[float] | None:
    """Stage two: one cross-encoder score per nominee, or None when the reranker
    abstained (down, timed out, malformed reply): the provider already fails
    open with None, so the row stays unscored instead of half-scored."""
    if not nominees:
        return None
    docs = [f"{n}\n{best_window(_body(vault, p), query, _RERANK_WINDOW)}" for p, n in nominees]
    scores = reranker.scores(query, docs)
    if scores is None or len(scores) != len(nominees):
        return None
    return [float(s) for s in scores]


def probe(vault: Path, query: str, k: int = 3, query_vec=None, reranker=None, *,
          pool: int = _NOMINEE_POOL) -> dict:
    """How a vault answers `query`: stage one nominates a pool, stage two scores it.

    ``probe`` names the stages that nominated ("lexical", "embed",
    "lexical+embed", None); ``top`` = up to `k` titles, in rerank order once a
    reranker scored the pool, else in nomination order; ``rerank`` = the best
    cross-encoder score over the pool and ``scored`` whether one was obtained.
    ``hits``/``best``/``notes`` pass through from `nominate` (indicative only,
    see the module docstring).
    """
    nominated = nominate(vault, query, pool=pool, query_vec=query_vec)
    nominees = pool_union(nominated, pool)
    row = {
        "probe": "+".join(nominated["stages"]) or None,
        "hits": nominated["hits"], "best": nominated["best"], "notes": nominated["notes"],
        "rerank": None, "scored": False,
        "top": [n for _p, n in nominees[:k]],
    }
    if reranker is not None and nominees:
        scores = rerank_nominees(vault, query, nominees, reranker)
        if scores is not None:
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            row["rerank"] = round(scores[order[0]], 3)
            row["scored"] = True
            row["top"] = [nominees[i][1] for i in order[:k]]
    return row


def _brief(vault: Path) -> str:
    from silica.kernel.recall.paths import index_dir_for

    try:
        got = json.loads((index_dir_for(str(vault)) / "vault_brief.json").read_text(encoding="utf-8"))
        return str(got.get("text") or "")
    except (OSError, ValueError):
        return ""  # no GUI ever wrote one; name, manifest and coverage still describe the vault


def describe(vault: Path, query: str = "", k: int = 3, query_vec=None, reranker=None) -> dict:
    from silica.config import CONFIG
    from silica.kernel.recall.memory_lane import memory_vault
    from silica.kernel.vault_manifest import load_manifest

    v = Path(vault).resolve()
    active = (getattr(CONFIG, "vault_path", "") or "").strip()
    manifest = load_manifest(v)
    cov = coverage(v)
    row = {
        "name": v.name,
        "path": str(v),
        "active": bool(active) and Path(active).resolve() == v,
        "memory": memory_vault() == v,
        "write_dir": manifest.write_dir or "",
        "language": manifest.cooccurrence_lang or "",
        "brief": _brief(v),
        "coverage": cov["level"],
        "notes": cov["notes"],
    }
    if query.strip():
        got = probe(v, query, k, query_vec=query_vec, reranker=reranker)
        counted = got.pop("notes")
        if row["notes"] is None:
            row["notes"] = counted  # the probe loaded a store; coverage alone never does
        row.update(got)
    return row


_DIAGNOSTIC_FIELDS = ("hits", "best")


def scoreboard(query: str = "", k: int = 3, *, diagnostics: bool = False) -> list[dict]:
    """One row per known vault. With a query and a reranker that answered, the
    rows are in rerank order (`scored` rows first, best score first, a row it
    could not score or that nominated nothing goes last); otherwise they keep
    discovery order, active vault first. ``hits`` and ``best`` are kept only
    for ``diagnostics`` callers (the bench): measured to mislead, they must not
    sit next to `top` in what a model reads."""
    once = functools.lru_cache(maxsize=1)(lambda: _query_vec(query))
    rr = _reranker() if query.strip() else None
    rows = [describe(v, query, k, query_vec=once, reranker=rr) for v in known_vaults()]
    if any(r.get("scored") for r in rows):
        rows.sort(key=lambda r: (not r["scored"], -(r["rerank"] or 0.0), not r["active"]))
    if not diagnostics:
        for r in rows:
            for f in _DIAGNOSTIC_FIELDS:
                r.pop(f, None)
    return rows


def home_of(rows: list[dict], reranker: str | None) -> list[str] | None:
    """The vaults whose best rerank score clears the calibrated floor, in row
    order. ``[]`` says no known vault holds the answer; ``None`` says the rule
    cannot speak: no reranker, one the floor was not measured on, or a verdict
    that would be incomplete because an indexed vault ran no stage (embedder
    down) or nominated notes the reranker then failed to score. A pool that
    came back empty everywhere is a verdict: ``[]``."""
    if not reranker or not reranker.startswith(_HOME_FLOOR_MODEL):
        return None
    for r in rows:
        if r.get("coverage") == "cold":
            continue
        if r.get("probe") is None or (r.get("top") and not r.get("scored")):
            return None
    return [r["path"] for r in rows if r.get("scored") and r["rerank"] >= _HOME_FLOOR]


def route(query: str = "", k: int = 3) -> dict:
    """The scoreboard as the tool returns it: ``ranked`` (at least one row was
    scored, so position means relevance), ``reranker`` (the model those scores
    are on the scale of, None without one), ``home`` (see `home_of`) and the
    rows."""
    rr = _reranker() if query.strip() else None
    rows = scoreboard(query, k)
    name = reranker_name(rr)
    return {
        "reranker": name,
        "ranked": any(r.get("scored") for r in rows),
        "home": home_of(rows, name),
        "vaults": rows,
    }


def clear() -> None:
    """Drop cached lexical stores (test isolation)."""
    _LEXICAL_CACHE.clear()
