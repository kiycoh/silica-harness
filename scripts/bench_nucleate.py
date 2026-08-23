#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Measure nucleation speed on a sample of a document folder, then extrapolate.

A sampler and not a stopwatch on the whole run: a 257-PDF library is hours of
wall-clock, so "how long would the folder take" has to be answerable in
minutes. PAGES are the extrapolation variable, not file count: a 4-page note
and a 300-page book are one file each and two orders of magnitude of distiller
work apart, and every measured run so far (2026-08-21: 79% of wall-clock in
the distiller) says the distiller is what the estimate is estimating.

    # measure 3 papers spread across the size distribution, then extrapolate
    uv run python scripts/bench_nucleate.py docs/research/papers --sample 3

    # re-extrapolate over any folder from the last measurement — no LLM calls
    uv run python scripts/bench_nucleate.py docs/research/papers --estimate-only

    # the arithmetic's own check
    uv run python scripts/bench_nucleate.py --self-check

The sample is nucleated FOR REAL into the configured vault. A scratch vault
would answer a different question — empty embed index, empty graph gate, no
autolink candidates — and throw the notes away; `/revert` takes a unit back
out. Pass --vault to send it somewhere else.

Reads back what the pipeline already records instead of re-deriving it: the
BUS `work/phase` events give the per-phase split, `SILICA_TOKEN_METER=1` gives
the per-call-site token table at exit, and the run result gives the yield. The
yield columns are not decoration: an "optimization" that halves the time by
writing half the notes has to be visible in the same table as the seconds.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH_OUT = REPO / "bench" / "nucleate_speed.json"

# Folders inside a library that are outputs or tooling, never sources.
SKIP_DIRS = {"mineru_out", "_tools", "done", ".git", "sources"}


# --------------------------------------------------------------------------
# pure arithmetic (self-checked below; no silica imports, no side effects)
# --------------------------------------------------------------------------

def stratified(items: list[dict], k: int) -> list[dict]:
    """`k` items spread across the page distribution, deterministically.

    Not `random.sample`: with 257 papers whose page counts span 4..600, a
    uniform draw of 3 lands in the fat middle and the estimate then misses the
    tail that dominates the total. Evenly-spaced quantiles of the sorted list
    put one probe in the short tail, one in the middle, one in the long tail.
    """
    if k >= len(items):
        return list(items)
    ordered = sorted(items, key=lambda it: (it["pages"], it["path"]))
    step = (len(ordered) - 1) / max(k - 1, 1) if k > 1 else 0
    picked = {round(i * step) for i in range(k)}
    return [ordered[i] for i in sorted(picked)]


def rates(measured: list[dict]) -> dict:
    """Fixed cost per file + marginal cost per page, from the sample.

    `secs = a + b·pages`, not `secs = rate·pages`, because the fixed half is
    most of a small file: the pilot run measured a 2-page paper at 268s — recon,
    payload, salience, residue, curator and finalize all run once per file
    regardless of its length, and a pure ratio charges the whole 268s to two
    pages and then reports 220 hours for a 5,900-page library. Two points fit
    the line; below that there is nothing to separate `a` from `b` and the
    median ratio is returned with `degenerate: True` so the report can say so.

    The band is the per-page ratio's p25/p75 — spread, not fit quality. On the
    theology batch one scanned book ran 4x the per-page rate of the text PDFs
    beside it, and a single number reports that as the typical case.
    """
    usable = [m for m in measured if m["pages"] and m["secs"]]
    if not usable:
        return {}
    per_page = sorted(m["secs"] / m["pages"] for m in usable)
    conv = sorted(m["convert_s"] / m["pages"] for m in usable)
    out = {
        "n": len(usable),
        "s_per_page": statistics.median(per_page),
        "s_per_page_lo": per_page[len(per_page) // 4],
        "s_per_page_hi": per_page[(3 * len(per_page)) // 4],
        "s_per_page_convert": statistics.median(conv) if conv else 0.0,
        "s_per_chunk": statistics.median(
            [m["secs"] / m["chunks"] for m in usable if m.get("chunks")] or [0.0]
        ),
        "chunks_per_page": statistics.median(
            [m["chunks"] / m["pages"] for m in usable if m.get("chunks")] or [0.0]
        ),
        "degenerate": len(usable) < 2,
    }
    if out["degenerate"]:
        out["fixed_s"], out["marginal_s_per_page"] = 0.0, out["s_per_page"]
        return out
    xs = [float(m["pages"]) for m in usable]
    ys = [float(m["secs"]) for m in usable]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else 0.0
    intercept = my - slope * mx
    # A negative half is arithmetic, not a measurement: three noisy points can
    # tilt the line below zero, and a negative fixed cost would then SUBTRACT
    # time for every extra file in the folder.
    out["marginal_s_per_page"] = max(slope, 0.0)
    out["fixed_s"] = max(intercept, 0.0)
    return out


def eta(rate: dict, files_left: int, pages_left: int) -> dict:
    """Wall-clock for what is left, sequential, at the fitted cost."""
    if not rate:
        return {}
    hours = (files_left * rate["fixed_s"] + pages_left * rate["marginal_s_per_page"]) / 3600
    return {
        "files": files_left,
        "pages": pages_left,
        "hours": hours,
        "hours_lo": pages_left * rate["s_per_page_lo"] / 3600,
        "hours_hi": pages_left * rate["s_per_page_hi"] / 3600,
    }


def _fmt_h(h: float) -> str:
    return f"{h * 60:.0f}m" if h < 1 else f"{h:.1f}h"


# --------------------------------------------------------------------------
# census
# --------------------------------------------------------------------------

def page_count(path: Path) -> int:
    """Pages, or a size-derived guess when the file will not open.

    ~40 KB/page is the measured average of this library (567 MB / ~14k pages);
    it only ever covers files pymupdf refuses, which are exactly the ones whose
    conversion cost is least predictable anyway.
    """
    try:
        import pymupdf

        with pymupdf.open(path) as doc:
            return doc.page_count or 1
    except Exception:
        return max(1, round(path.stat().st_size / 40_000))


def census(folder: Path, exts: tuple[str, ...]) -> list[dict]:
    out = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if SKIP_DIRS & set(p.relative_to(folder).parts):
            continue
        out.append({"path": str(p), "pages": page_count(p), "bytes": p.stat().st_size})
    return out


# --------------------------------------------------------------------------
# the measured run
# --------------------------------------------------------------------------

class PhaseClock:
    """Per-phase seconds, from the events the FSM already publishes.

    Sums can exceed wall-clock and that is not a bug: the boundary prefetcher
    runs the next file's distill on another thread (docs/audits/
    2026-08-16-boundary-prefetch-lever.md), so phases genuinely overlap. Read
    the shares as "where the work is", not as a partition of the clock.
    """

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self._open: dict[str, list[float]] = {}

    def __call__(self, event) -> None:
        phase, status = getattr(event, "phase", ""), getattr(event, "status", "")
        if status == "running":
            self._open.setdefault(phase, []).append(time.perf_counter())
            return
        stack = self._open.get(phase) or []
        if not stack:
            return
        started = stack.pop()
        self.totals[phase] = self.totals.get(phase, 0.0) + (time.perf_counter() - started)
        self.calls[phase] = self.calls.get(phase, 0) + 1


class RetryLog(logging.Handler):
    """Counts the retries the LLM layer already logs, per file.

    Not a nicety: in the pilot run one call hit the 130s silent-hang backstop
    (`_LOCAL_LLM_TIMEOUT` in agent/llm.py) and retried, which was ~48% of that
    file's wall-clock. An ETA fitted over a sample that silently contained a
    stalled provider measures the provider's bad afternoon, not the pipeline.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.retries = 0
        self.timeouts = 0
        self.sites: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Transient error" not in msg:
            return
        self.retries += 1
        if "timeout" not in msg.lower():
            return
        self.timeouts += 1
        # Which lane stalled, not just how often: the retry is logged from
        # agent/llm.py for every call site alike, and "the provider is slow"
        # and "the residue lane's 15k-token prompts are slow" call for
        # different fixes. Same walk _meter_site does, from the handler's
        # thread — the failing call is still on this stack.
        frame = sys._getframe()
        while frame:
            f = frame.f_code.co_filename
            if "/silica/" in f and "agent/llm.py" not in f and "logging" not in f:
                site = f"{Path(f).parent.name}/{Path(f).name}:{frame.f_code.co_name}"
                self.sites[site] = self.sites.get(site, 0) + 1
                return
            frame = frame.f_back


def subagent_seconds(sid: str) -> dict[str, float]:
    """Seconds per worker kind (dedup, refine, expand), from the narration.

    The phase clock cannot see these: the curator runs on the Coordinator's own
    thread pool, beside the FSM, and in the pilot run the workers plus residue
    plus conversion were 73% of the wall-clock the phase table did not explain.
    The narration already spans every worker (agent/narration.py span_open
    "subagent"), so this is a reader, not new instrumentation.
    """
    from silica.agent.narration import narration_dir, read_beats

    open_at: dict[str, tuple[float, str]] = {}
    out: dict[str, float] = {}
    for beat in read_beats(narration_dir() / f"{sid}.jsonl"):
        if beat.get("kind") != "subagent" or not beat.get("id"):
            continue
        bid = beat["id"]
        if beat.get("status") == "running":
            open_at[bid] = (beat["ts"], (beat.get("payload") or {}).get("kind", "?"))
        elif bid in open_at:
            started, kind = open_at.pop(bid)
            out[kind] = out.get(kind, 0.0) + (beat["ts"] - started)
    return out


def measure(sample: list[dict], target_dir: str, profile: str) -> list[dict]:
    from silica.agent.bus import BUS
    from silica.agent.narration import NARRATOR
    from silica.cli import _nucleate_prepare
    from silica.kernel.write.undo_journal import get_undo_journal
    from silica.config import CONFIG
    from silica.router.coordinator import Coordinator
    from silica.sources.convert import convert

    clock = PhaseClock()
    BUS.subscribe("work/phase", clock)
    retries = RetryLog()
    llm_log = logging.getLogger("silica.agent.llm")
    llm_log.addHandler(retries)
    llm_log.setLevel(logging.WARNING)
    sid = NARRATOR.ensure_session(driver="bench_nucleate")
    rows = []
    for i, item in enumerate(sample, 1):
        src = item["path"]
        print(f"\n[{i}/{len(sample)}] {Path(src).name} — {item['pages']} pages", flush=True)
        undo_run = get_undo_journal().start_run(
            source="bench_nucleate", vault=CONFIG.vault_path.strip() or None
        )
        before = dict(clock.totals)
        retries_before, timeouts_before = retries.retries, retries.timeouts
        t0 = time.perf_counter()
        try:
            segs = convert(src, target_dir)
        except (ValueError, RuntimeError) as exc:
            print(f"    convert failed: {exc}")
            continue
        t_conv = time.perf_counter() - t0
        mfs = _nucleate_prepare(segs, target_dir, profile or None, undo_run)
        if not mfs:
            # Already distilled, all apparatus, or filed as a draft. Zero
            # seconds for zero work would drag the median toward "instant".
            print("    nothing to distill (already nucleated / apparatus) — excluded")
            continue
        t1 = time.perf_counter()
        result = Coordinator(
            inbox_files=mfs, target_dir=target_dir, hub=None,
            keep_sources=True, distill_profile=profile or None,
        ).run()
        t_nuc = time.perf_counter() - t1
        cov = result.get("coverage") or {}
        rows.append({
            "path": src,
            "pages": item["pages"],
            "segments": len(mfs),
            "convert_s": round(t_conv, 1),
            "nucleate_s": round(t_nuc, 1),
            "secs": round(t_conv + t_nuc, 1),
            "chunks": result.get("committed_chunks", 0),
            "chunks_failed": len(result.get("failed_chunks") or []),
            "notes": result.get("yield_notes", 0),
            "links": result.get("yield_links", 0),
            "deferred": cov.get("deferred_ops", 0),
            "residue": cov.get("residue_facts", 0),
            "status": result.get("final_status", "?"),
            "retries": retries.retries - retries_before,
            "timeouts": retries.timeouts - timeouts_before,
            "phases": {k: round(v - before.get(k, 0.0), 1)
                       for k, v in clock.totals.items() if v - before.get(k, 0.0) > 0.5},
        })
        r = rows[-1]
        print(f"    {r['secs']}s ({r['convert_s']}s convert) · {r['chunks']} chunks · "
              f"{r['notes']} notes · {r['status']}", flush=True)
    BUS.unsubscribe("work/phase", clock)
    llm_log.removeHandler(retries)
    workers = subagent_seconds(sid) if sid else {}
    NARRATOR.close()
    for r in rows:
        r["phase_totals"] = clock.totals
        r["worker_totals"] = workers
        r["stall_sites"] = retries.sites
    return rows


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def report(all_files: list[dict], measured: list[dict], rate: dict, folder: Path) -> None:
    pages_all = sum(f["pages"] for f in all_files)
    done = {m["path"] for m in measured}
    pages_left = sum(f["pages"] for f in all_files if f["path"] not in done)

    print(f"\n{'=' * 78}\n{folder}: {len(all_files)} files · {pages_all:,} pages\n{'=' * 78}")
    if measured:
        print(f"{'file':<40}{'pages':>6}{'conv':>6}{'nucl':>7}{'chunks':>7}"
              f"{'notes':>6}{'defer':>6}{'status':>9}")
        for m in measured:
            print(f"{Path(m['path']).name[:39]:<40}{m['pages']:>6}"
                  f"{m['convert_s']:>6.0f}{m['nucleate_s']:>7.0f}"
                  f"{m['chunks']:>7}{m['notes']:>6}{m['deferred']:>6}{m['status']:>9}")

        phases = measured[-1].get("phase_totals") or {}
        wall = sum(m["secs"] for m in measured)
        if phases:
            accounted = sum(phases.values())
            print("\nwhere the time went (phases overlap: the prefetcher runs ahead)")
            for name, secs in sorted(phases.items(), key=lambda kv: -kv[1])[:8]:
                print(f"  {name:<22}{secs:>9.0f}s {secs / wall:>6.0%}")
            # The phase clock only sees FSM capabilities. Conversion, the
            # curator (dedup/refine), residue and finalize run outside them, and
            # in the pilot run that gap was 70% of the wall-clock — a phase
            # table normalised to its own sum would have hidden exactly the
            # part worth optimising.
            print(f"  {'unaccounted':<22}{wall - accounted:>9.0f}s "
                  f"{(wall - accounted) / wall:>6.0%}  (convert, curator, residue, finalize)")

        workers = measured[-1].get("worker_totals") or {}
        if workers:
            print("\nworker threads beside the FSM (curator lane, 3 concurrent)")
            for kind, secs in sorted(workers.items(), key=lambda kv: -kv[1]):
                print(f"  {kind:<22}{secs:>9.0f}s {secs / wall:>6.0%}")

        stalls = sum(m.get("timeouts", 0) for m in measured)
        if stalls:
            # Stalled seconds, NOT a share of the wall-clock: up to six calls
            # are in flight at once (distill_concurrency 3 + curator pool 3),
            # so two stalls can overlap into one 130s hole. The ratio is the
            # ceiling on what re-timing the backstop could give back.
            print(f"\n! {stalls} call(s) hit the 130s silent-hang backstop and "
                  f"retried: {stalls * 130 / 60:.0f} min of stalled provider time against "
                  f"{wall / 60:.0f} min of wall-clock. Calls overlap, so that is a ceiling "
                  f"on the loss, not a measurement of it — but the work was thrown away.")
            sites = measured[-1].get("stall_sites") or {}
            for site, n in sorted(sites.items(), key=lambda kv: -kv[1]):
                print(f"    {n:>3} in {site}")

    if not rate:
        print("\nnothing measured — no rate, no estimate.")
        return

    if rate.get("degenerate"):
        print("\n! one file measured: its whole fixed cost is charged to its pages.")
        print("  Re-run with --sample 3 or more before trusting the ETA.")
    print(f"\ncost: {rate['fixed_s']:.0f}s per file (fixed) + "
          f"{rate['marginal_s_per_page']:.1f}s per page")
    print(f"      ratio {rate['s_per_page']:.1f} s/page "
          f"(band {rate['s_per_page_lo']:.1f}–{rate['s_per_page_hi']:.1f}), "
          f"conversion {rate['s_per_page_convert']:.1f} s/page")
    if rate.get("s_per_chunk"):
        print(f"      {rate['s_per_chunk']:.0f} s/chunk · "
              f"{rate['chunks_per_page']:.1f} chunks/page")
    e = eta(rate, len(all_files) - len(done), pages_left)
    print(f"\nETA for the remaining {e['files']} files ({pages_left:,} pages), "
          f"sequential: {_fmt_h(e['hours'])}  "
          f"(flat-ratio cross-check {_fmt_h(e['hours_lo'])}–{_fmt_h(e['hours_hi'])})")
    print(f"\nwritten to {BENCH_OUT.relative_to(REPO)}")


# --------------------------------------------------------------------------

def self_check() -> None:
    items = [{"path": f"p{i}", "pages": i} for i in range(1, 101)]
    picked = stratified(items, 3)
    assert [p["pages"] for p in picked] == [1, 51, 100], picked  # tails included
    assert stratified(items, 500) == items
    assert len(stratified(items, 1)) == 1

    m = [
        {"pages": 10, "secs": 100.0, "convert_s": 10.0, "chunks": 5},
        {"pages": 20, "secs": 400.0, "convert_s": 20.0, "chunks": 10},
        {"pages": 30, "secs": 300.0, "convert_s": 30.0, "chunks": 10},
    ]
    r = rates(m)
    assert r["s_per_page"] == 10.0, r          # median of 10, 20, 10
    assert r["s_per_page_hi"] == 20.0, r
    assert r["s_per_page_convert"] == 1.0, r

    # secs = a + b·pages on exact points: the fit must recover a and b, and the
    # ETA must charge the fixed half per FILE, not per page.
    exact = [{"pages": p, "secs": 200.0 + 10.0 * p, "convert_s": 0.0, "chunks": 1}
             for p in (10, 50, 100)]
    f = rates(exact)
    assert abs(f["fixed_s"] - 200.0) < 1e-6, f
    assert abs(f["marginal_s_per_page"] - 10.0) < 1e-6, f
    assert abs(eta(f, 3, 300)["hours"] - (3 * 200 + 3000) / 3600) < 1e-9

    # A downward-sloping sample must not produce a negative fixed cost, which
    # would make every extra file SHORTEN the estimate.
    neg = rates([{"pages": p, "secs": s, "convert_s": 0.0, "chunks": 1}
                 for p, s in ((10, 500.0), (100, 100.0))])
    assert neg["fixed_s"] >= 0 and neg["marginal_s_per_page"] >= 0, neg

    one = rates([{"pages": 2, "secs": 268.0, "convert_s": 15.0, "chunks": 6}])
    assert one["degenerate"] and one["fixed_s"] == 0.0, one

    # A file that produced nothing must not enter the rate as "instant".
    assert rates([{"pages": 5, "secs": 0.0, "convert_s": 0.0, "chunks": 0}]) == {}
    print("self-check ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="document folder to census / sample")
    ap.add_argument("--sample", type=int, default=3, help="files to nucleate (default 3)")
    ap.add_argument("--max-pages", type=int, default=40,
                    help="skip files longer than this when sampling (0 = no cap). "
                         "The census still counts them: the point is a calibration "
                         "that ends in minutes, and the 135-page survey in this "
                         "library is ~400 chunks — hours on its own")
    ap.add_argument("--target", default="", help="vault folder for the notes (required to run)")
    ap.add_argument("--vault", default="", help="vault to nucleate into (default: configured)")
    ap.add_argument("--profile", default="", help="distill profile, e.g. study")
    ap.add_argument("--estimate-only", action="store_true",
                    help="census + ETA from the last measurement; no LLM calls")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return 0
    if not args.folder:
        ap.error("folder is required (or --self-check)")

    # Before importing silica: config captures VAULT_PINNED at import time.
    if args.vault:
        os.environ["SILICA_VAULT"] = str(Path(args.vault).expanduser().resolve())
    os.environ.setdefault("SILICA_TOKEN_METER", "1")

    from silica.cli import _activate_repo_mode, _ensure_servers
    from silica.config import CONFIG
    from silica.kernel.vault_manifest import apply_manifest_to_config
    from silica.sources.convert import CONVERTIBLE_DOC_EXTS

    _activate_repo_mode()
    apply_manifest_to_config()

    folder = Path(args.folder).expanduser().resolve()
    files = census(folder, tuple(CONVERTIBLE_DOC_EXTS))
    if not files:
        print(f"no convertible document under {folder}")
        return 1

    if args.estimate_only:
        prior = json.loads(BENCH_OUT.read_text()) if BENCH_OUT.exists() else {}
        # Refit rather than replay `prior["rate"]`: the measurements are the
        # durable half, the cost model is not, and a stored rate from an older
        # model would be re-quoted forever against a newer report.
        measured = prior.get("measured", [])
        report(files, measured, rates(measured), folder)
        return 0

    if not args.target:
        ap.error("--target is required to run the sample (the notes go somewhere real)")
    _ensure_servers()
    print(f"vault: {CONFIG.vault_path}  target: {args.target}  model: {CONFIG.model}")

    pool = [f for f in files if not args.max_pages or f["pages"] <= args.max_pages]
    if not pool:
        print(f"every file is longer than --max-pages {args.max_pages}")
        return 1
    sample = stratified(pool, args.sample)
    measured = measure(sample, args.target, args.profile)
    rate = rates(measured)
    BENCH_OUT.parent.mkdir(exist_ok=True)
    BENCH_OUT.write_text(json.dumps({
        "folder": str(folder),
        "vault": CONFIG.vault_path,
        "model": CONFIG.model,
        "files": len(files),
        "pages": sum(f["pages"] for f in files),
        "measured": measured,
        "rate": rate,
    }, indent=2))
    report(files, measured, rate, folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
