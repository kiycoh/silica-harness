# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""WorkQueue — the producer/consumer channel between the Injector and sub-agents.

The Injector (router) *produces* WorkItems as it commits batches; a pool of
leashed sub-agents *consumes* them concurrently on the worker model.  The router
never blocks on a sub-agent: it enqueues fire-and-forget and the Coordinator
drains/joins the queue only at the end of the run.

Concurrency safety has two layers:
  1. Temporal disjointness — items reference only notes from already-committed
     batches (or pre-existing vault notes), so the router is never mid-write on a
     sub-agent's target.
  2. Per-path lease — `path_lease()` serialises writes to the same note across
     sub-agents (and any lease-aware writer).  Both the Injector and the pool run
     as threads in one process, so an in-process lock registry is sufficient.

WorkItem kinds are owned by the capability registry — see
silica/capabilities/__init__.py (CAPABILITIES), the single dispatch table for
background work. tests/test_capability_seam.py enforces that every kind
produced anywhere has a registered capability.
"""
from __future__ import annotations

import hashlib
import logging
import os
import queue
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import orjson

try:  # POSIX only; Windows falls back to the in-process lock alone.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-path advisory lease (shared across the Injector, the sub-agent pool, and
# any *other process* pointed at the same vault — e.g. two MCP clients)
# ---------------------------------------------------------------------------

_LEASES: dict[str, threading.Lock] = {}
_LEASES_GUARD = threading.Lock()


def _lease_key(path: str) -> str:
    return (path or "").replace("\\", "/").removesuffix(".md").lower()


def _lock_file(key: str) -> Path | None:
    """Vault-scoped lock file for `key`; None when fcntl is unavailable.

    Lives under the per-vault index namespace so every process resolving the
    same vault agrees on the same file; the filename is a hash of the key so
    deep paths and separators never leak into the filesystem.
    """
    if fcntl is None:
        return None
    from silica.kernel.recall.paths import index_dir

    d = index_dir() / "locks"
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".lock")


@contextmanager
def path_lease(path: str) -> Iterator[None]:
    """Serialise writes to a single vault note across threads *and processes*.

    Two layers: an in-process ``threading.Lock`` (fast path, and it makes the
    OS lock contention-free between threads of one process) plus an advisory
    ``flock`` on a vault-scoped lock file so separate processes writing the
    same vault (two MCP clients, MCP + a running pipeline) cannot interleave.

    Advisory: only protects writers that opt in by acquiring the same lease.
    Lock-file bookkeeping is best-effort — if it fails we log and fall back to
    in-process-only rather than break the write (the threading lock still holds
    for same-process writers). ``commit_ops`` acquires multiple leases in sorted
    order, so cross-process acquisition order is consistent and deadlock-free.
    """
    key = _lease_key(path)
    with _LEASES_GUARD:
        lock = _LEASES.setdefault(key, threading.Lock())
    lock.acquire()
    fd: int | None = None
    try:
        try:
            lf = _lock_file(key)
            if lf is not None:
                fd = os.open(str(lf), os.O_CREAT | os.O_RDWR, 0o644)
                fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as e:
            logger.warning("path_lease: cross-process lock unavailable for %s (%s); "
                           "in-process lock only", key, e)
            if fd is not None:
                os.close(fd)
                fd = None
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        lock.release()


# ---------------------------------------------------------------------------
# WorkItem
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    kind: str                       # "dedup" | "refine" | "orphan"
    target_path: str                # vault-rel note the sub-agent may write
    context: dict[str, Any] = field(default_factory=dict)
    reason: str = ""                # the FSM warning that produced this item
    id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "pending"         # pending | done | failed | rejected
    result: dict[str, Any] | None = None
    # Runtime-only: not serialised. Set by caller to request early exit.
    cancel_token: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
        compare=False,
        hash=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_path": self.target_path,
            "context": self.context,
            "reason": self.reason,
            "id": self.id,
            "status": self.status,
            "result": self.result,
        }


# ---------------------------------------------------------------------------
# WorkQueue
# ---------------------------------------------------------------------------

_SENTINEL = object()  # poison pill — injected by close() to unblock consumers


class WorkQueue:
    """Thread-safe producer/consumer queue with optional disk persistence."""

    def __init__(self, run_dir: str | Path | None = None):
        self._q: queue.Queue = queue.Queue()
        self._items: list[WorkItem] = []        # every item ever enqueued (feeds summary/inspection)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._pending: int = 0                  # items enqueued but not yet complete
        # Tolerate a non-path run_dir (e.g. a mocked progress object): persistence
        # is best-effort, so fall back to no-disk rather than raising.
        try:
            self._run_dir = Path(run_dir) if isinstance(run_dir, (str, Path)) else None
        except (TypeError, ValueError):
            self._run_dir = None

    # --- producer side ----------------------------------------------------

    def enqueue(self, item: WorkItem) -> None:
        """Add an item. No-op after close() (producers have finished)."""
        if self._closed.is_set():
            return
        with self._lock:
            self._items.append(item)
            self._pending += 1
        self._q.put(item)
        self._persist()

    def close(self) -> None:
        """Signal that no further items will be produced.

        Injects a sentinel into the queue so blocking consumers wake up and
        drain cleanly.  The sentinel cascades: each consumer that receives it
        rebroadcasts it before returning, waking the next consumer.
        """
        self._closed.set()
        self._q.put(_SENTINEL)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    # --- consumer side ----------------------------------------------------

    def claim(self, *, timeout: float | None = None) -> WorkItem | None:
        """Pop the next item.

        With ``timeout=None`` (default): blocks at OS level until an item
        arrives or the queue is closed (sentinel-based — no polling).
        Returns None only when the queue has been closed and is empty.

        With ``timeout=<float>``: legacy mode — returns None if no item
        arrives within that many seconds (backward-compatible with tests).
        """
        if timeout is None:
            item = self._q.get()
        else:
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                return None

        if item is _SENTINEL:
            self._q.put(_SENTINEL)  # rebroadcast so the next consumer wakes up
            return None
        return item

    def complete(self, item: WorkItem, status: str, result: dict[str, Any] | None = None) -> None:
        """Mark an item finished and record the outcome."""
        with self._lock:
            item.status = status
            item.result = result
            self._pending -= 1
        self._persist()

    # --- draining ---------------------------------------------------------

    def drained(self) -> bool:
        """True when producers are done and every queued item is consumed."""
        with self._lock:
            return self._closed.is_set() and self._pending == 0

    # --- inspection / persistence ----------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def items(self) -> list[WorkItem]:
        with self._lock:
            return list(self._items)

    def summary(self) -> dict[str, int]:
        """Counts by status, for the final run report."""
        with self._lock:
            return dict(Counter(it.status for it in self._items))

    def _persist(self) -> None:
        # Atomic (tmp + rename): a crash mid-dump must not leave the one
        # post-mortem artifact torn.
        # Write-only debug dump — workqueue.json has no reader/from_dict, so
        # this is post-mortem inspection only; a loader arrives with resume,
        # if resume is ever built (until then, no resume path exists).
        if not self._run_dir:
            return
        try:
            from silica.kernel.recall.paths import atomic_write_bytes

            self._run_dir.mkdir(parents=True, exist_ok=True)
            # The rename stays INSIDE the lock: snapshotting under it and
            # replacing outside lets a thread that snapshotted an older list
            # win the race to os.replace and clobber a fresher dump with a
            # stale one — atomic per write, lost update across writes.
            with self._lock:
                payload = [it.to_dict() for it in self._items]
                atomic_write_bytes(
                    self._run_dir / "workqueue.json",
                    orjson.dumps(payload, option=orjson.OPT_INDENT_2),
                )
        except Exception:
            # Persistence is best-effort; never break the pipeline over it.
            pass


_BATCH_CONCEPT_KEYS = ("concept", "excerpt", "score", "full_score", "title_score", "inbox_file")
# Fixed cap bounds the batch prompt; raise it the day real families outgrow it.
_MAX_FAMILY_BATCH = 8


def batch_dedup_items(items: list[WorkItem]) -> list[WorkItem]:
    """Collapse dedup WorkItems that share a candidate note into family batches.

    Grouping key is ``target_path``: COLLISION's borderline concepts and
    /curate's union-find duplicate families both converge there by
    construction. The candidate body is the bulk of the judge's prompt, so a
    family of N concepts judged per-item repeats it N times; one batch item is
    one judge call (the dedup capability fans the verdicts back out from
    ``context["concepts"]``). Singletons and non-dedup items pass through
    untouched, so callers can apply this unconditionally before dispatch.

    Lives kernel-side (not in the dedup capability) so both producers — the
    router's COLLISION state and /curate — reach it without importing the
    capabilities package, which would cycle through the P9 peer boundary.
    """
    out: list[WorkItem] = []
    groups: dict[str, list[WorkItem]] = {}
    for it in items:
        if it.kind == "dedup" and it.target_path and not it.context.get("concepts"):
            groups.setdefault(it.target_path, []).append(it)
        else:
            out.append(it)
    for path, group in groups.items():
        for i in range(0, len(group), _MAX_FAMILY_BATCH):
            chunk = group[i:i + _MAX_FAMILY_BATCH]
            if len(chunk) == 1:
                out.append(chunk[0])
                continue
            shared = {
                k: v for k, v in chunk[0].context.items()
                if k not in _BATCH_CONCEPT_KEYS
            }
            shared["concepts"] = [
                {k: it.context[k] for k in _BATCH_CONCEPT_KEYS if k in it.context}
                for it in chunk
            ]
            out.append(WorkItem(
                kind="dedup",
                target_path=path,
                context=shared,
                reason=f"dedup_family n={len(chunk)} → {path.rsplit('/', 1)[-1]}",
            ))
    return out
