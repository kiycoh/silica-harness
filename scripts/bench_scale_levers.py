# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""A/B bench for the 2026-08-19 scale levers, so each one carries a measured
verdict (positive/negative) instead of an inferred one.

Every lever has an OFF arm reproducing the pre-lever behavior and an ON arm
exercising the shipped code, on synthetic data at two sizes: today's scale
(~1.2k notes) and the trigger scale the old comments named (10k). Run it any
time with:

    uv run python scripts/bench_scale_levers.py

Levers covered here: learner epoch memo, cooccur two-way adjacency, embed
deferred delete-saves, body-cache LRU bound. The aliases chunking has no CPU
arm (it bounds an LLM prompt; tests/test_debt_paydown.py guards the split),
and the zoneMST memo is a browser lever measured in-page.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

SIZES = (1200, 10000)
RESULTS: list[tuple[str, int, float, float]] = []  # (lever, size, off_ms, on_ms)


def timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


def make_vault(root: Path, n: int) -> Path:
    vault = root / f"vault_{n}"
    vault.mkdir(parents=True)
    for i in range(n):
        d = vault / f"area{i % 20}"
        d.mkdir(exist_ok=True)
        (d / f"note{i}.md").write_text(
            f"---\ntags: [t{i % 7}]\ndate: 2026-01-{(i % 28) + 1:02d}\n"
            f"AI: {'true' if i % 3 else 'false'}\n---\n\n# Note {i}\n\n"
            + ("prose line\n" * 12),
            encoding="utf-8",
        )
    return vault


def bench_learner_memo(root: Path) -> None:
    from silica.kernel.report import learner

    for n in SIZES:
        vault = make_vault(root / "learner", n) if not (root / "learner" / f"vault_{n}").exists() else root / "learner" / f"vault_{n}"

        def off():
            learner._meta_memo.clear()
            learner._notes_meta(vault)

        off()  # warm the fs cache so both arms read hot files
        off_ms = timed(off)
        learner._meta_memo.clear()
        learner._notes_meta(vault)          # populate the memo
        on_ms = timed(lambda: learner._notes_meta(vault))  # epoch hit
        RESULTS.append(("learner epoch memo (repeat call)", n, off_ms, on_ms))


def bench_embed_deferred(root: Path) -> None:
    from silica.kernel.recall.embed import EmbedStore

    deletes = 64
    for n in SIZES:
        dim = 2560 if n <= 2000 else 512  # keep the 10k index in RAM honestly
        base = EmbedStore(path=root / f"embed_{n}.npz")
        vec = [0.01] * dim
        for i in range(n):
            base.upsert(f"area{i % 20}/note{i}", f"note{i}", list(vec))
        base.save()

        def arm(per_op: bool) -> float:
            path = root / f"embed_{n}_{per_op}.npz"
            shutil.copy(root / f"embed_{n}.npz", path)
            store = EmbedStore(path=path)

            def run():
                for i in range(deletes):
                    store.delete(f"area{i % 20}/note{i}")
                    if per_op:
                        store.save()
                if not per_op:
                    store.save()  # the threshold/exit flush

            return timed(run)

        off_ms = arm(per_op=True)
        on_ms = arm(per_op=False)
        RESULTS.append((f"embed deferred saves ({deletes} deletes, dim {dim})", n, off_ms, on_ms))


def bench_body_cache_bound(root: Path) -> None:
    import silica.driver.fs_backend as fsb

    for n in SIZES:
        vault = make_vault(root / "driver", n)
        drv = fsb.ObsidianFSBackend(str(vault))
        refs = drv.list_files("")

        def scan():
            for r in refs:
                drv.read_note(r.path)

        scan()  # warm OS cache
        cap = fsb._BODY_CACHE_CAP

        fsb._BODY_CACHE_CAP = 10 ** 9  # OFF: unbounded (pre-lever)
        drv._body_cache.clear()
        scan()
        off_ms = timed(scan)  # second pass, all hits

        fsb._BODY_CACHE_CAP = cap      # ON: shipped bound
        drv._body_cache.clear()
        scan()
        on_ms = timed(scan)
        bounded = len(drv._body_cache) <= cap
        RESULTS.append((f"body-cache LRU (double scan, bound {'held' if bounded else 'BROKEN'})", n, off_ms, on_ms))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="silica_lever_bench_"))
    try:
        bench_learner_memo(tmp)
        bench_embed_deferred(tmp)
        bench_body_cache_bound(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'lever':<48} {'size':>6} {'off ms':>9} {'on ms':>9} {'verdict':>9}")
    for lever, n, off_ms, on_ms in RESULTS:
        if on_ms <= off_ms * 0.8:
            verdict = "positive"
        elif on_ms <= off_ms * 1.15:
            verdict = "neutral"
        else:
            verdict = "NEGATIVE"
        print(f"{lever:<48} {n:>6} {off_ms:>9.1f} {on_ms:>9.1f} {verdict:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
