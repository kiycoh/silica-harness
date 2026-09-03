# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Personal-memory recall lane (ADR-0019).

The memory vault (default ~/.silica/vault, override SILICA_MEMORY_VAULT) is a
second, READ-ONLY pair of (embed, cooccur) stores fed to the same RRF fusion
as the active vault's legs. Writes never route here — material enters the
memory vault through its own inbox (UC1, same trust regime as any ingress).

Degenerate case (today's default without repo mode): when the active vault IS
the memory vault, the lane abstains — `memory_vault()` returns None, no store
is loaded twice, and fusion collapses to single-vault behavior bit-identically.

The same machinery serves a *peek* (`perceive(vault=...)`): any adopted vault's
stores, read as they lie on disk, standing in for the active legs. `stores_for`
is that generalisation; `foreign_root` is where a non-active origin's note
bodies are read from. Both stay read-only by construction — nothing here knows
how to write.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silica.kernel.recall.cooccurrence import CooccurStore
    from silica.kernel.recall.embed import EmbedStore

from collections import OrderedDict
from pathlib import Path

from silica.config import CONFIG


def memory_vault() -> Path | None:
    """Resolved memory-vault path, or None when the lane must abstain.

    Abstains when the memory vault coincides with the active vault (resolved
    path equality — nested vaults are out of scope, ADR-0019) or does not
    exist on disk yet.
    """
    raw = (getattr(CONFIG, "memory_vault", "") or "").strip()
    # Default mirrors cli.default_user_vault (kernel must not import cli).
    mem = (Path(raw).expanduser() if raw else Path.home() / ".silica" / "vault").resolve()
    active = (getattr(CONFIG, "vault_path", "") or "").strip()
    if active and Path(active).resolve() == mem:
        return None
    if not mem.is_dir():
        return None
    return mem


# Store cache keyed by index dir (twin of embed._STORE_CACHE), least recently
# used first. Bounded because `silica_vaults` loads one embed store per known
# vault (14 + 18 + 5 + 1 MB on this machine, 2026-09-01) and a peek adds the
# co-occurrence store (108 MB on the repo vault): unbounded, every vault the
# machine knows ends up resident in a process that serves one. Four keeps the
# memory vault and the last three peeked vaults warm; the fifth re-reads from
# disk. ponytail: an index built mid-session by another process is not picked
# up until eviction or restart; add an mtime check if that ever bites.
_CACHE: OrderedDict[str, tuple[EmbedStore | None, CooccurStore | None]] = OrderedDict()
_CACHE_MAX = 4


def foreign_root(origin: str) -> Path | None:
    """The folder a non-active `origin` resolves note paths in: the memory
    vault for "memory", the peeked vault for an absolute path (`perceive`'s
    `vault=`), None for "vault". "memory" is None as well while the lane
    abstains — callers treat None as unreadable, never as the active vault,
    because the active vault is read through the driver, not by path."""
    if origin == "memory":
        return memory_vault()
    if origin and origin != "vault":
        return Path(origin)
    return None


def memory_stores():
    """Memory-lane ``(embed_store, cooccur_store)`` for the fusion facade.

    Each leg is None when its index is absent/empty (the facade then abstains
    on that leg); ``(None, None)`` when the whole lane abstains. Never raises.
    """
    mem = memory_vault()
    if mem is None:
        return None, None
    return stores_for(mem)


def embed_store_for(vault: str | Path):
    """One vault's embed store, or None when absent/empty. The ACTIVE vault's is
    the process singleton (one copy in memory, and the copy this process's
    writes land in); any other vault's is the cached read-as-it-lies store.
    Embed only, on purpose: the scoreboard that calls this must not pay for a
    co-occurrence store (108 MB on the repo vault) to score one question."""
    from silica.config import CONFIG

    v = Path(vault).resolve()
    active = (getattr(CONFIG, "vault_path", "") or "").strip()
    if active and Path(active).resolve() == v:
        from silica.kernel.recall.embed import get_store

        es = get_store()
        return es if len(es) else None
    es, _cooccur = stores_for(v)
    return es


def stores_for(vault: str | Path):
    """A foreign vault's ``(embed_store, cooccur_store)``, read as its index
    lies on disk — the memory lane and a peek share it. Each leg is None when
    its index file is absent or empty. Never raises: a corrupt foreign index
    must cost that lane, not the active vault's answer.
    """
    try:
        from silica.kernel.recall import paths
        from silica.kernel.recall.cooccurrence import CooccurStore
        from silica.kernel.recall.embed import EmbedStore

        idx = paths.index_dir_for(str(vault))
        key = str(idx)
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
        else:
            # Explicit exists() guard: the store constructors soft-migrate from
            # the LEGACY global index when their file is missing — right for
            # the active vault, wrong for this lane (it would resurrect some
            # old vault's data as "memory"). A missing index ⇒ the leg abstains.
            ep = idx / "embeddings.json"
            cp = idx / "cooccurrence.json"
            hit = (
                EmbedStore(path=ep) if ep.is_file() else None,
                CooccurStore(path=cp) if cp.is_file() else None,
            )
            _CACHE[key] = hit
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
        es, cs = hit
        return (
            es if es is not None and len(es) else None,
            cs if cs is not None and len(cs) else None,
        )
    except Exception:
        return None, None


def clear() -> None:
    """Drop cached memory stores (test isolation)."""
    _CACHE.clear()
