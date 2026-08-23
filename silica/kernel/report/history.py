"""What the last report measured, so the next one can state what MOVED.

The metrics view could always say what the vault IS. It could never say what
changed, because nothing kept the previous reading: every open recomputed from
scratch and threw the numbers away, which is why the deck's Report panel is a
store before it is a surface.

Same shape as `energy.jsonl` beside it (see graph_report/render.py): an
append-only series with one line per actual movement, so opening the metrics
tab twice on an unchanged vault does not manufacture a second data point.

Only DEPTH-INDEPENDENT signals are kept. `integration_deficits` and the
autolink family are zero unless the co-occurrence leg ran, so a structural
report diffed against a full one would read "137 deficits closed" for work
nobody did. That is the same trap vault_energy documents for E(vault), and the
cheapest way out is to never store the terms that carry it.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

import orjson

logger = logging.getLogger(__name__)

# The signals the Report panel diffs, each a count that means the same thing at
# either depth. Adding one is safe (an older line simply has no value for it and
# reads as "first seen"); renaming one silently breaks every stored line, so
# emit a new key and keep recognising the old.
SIGNALS = (
    "notes",
    "links",
    "unresolved",
    "dangling_links",
    "orphans",
    "lean_notes",
    "contested",
    "structural_gaps",
    "areas",
)


def history_path(vault: str | Path) -> Path:
    return Path(vault) / ".silica" / "report_history.jsonl"


def signals_of(totals: dict, areas: int) -> dict[str, int]:
    """The storable subset of a report's totals.

    `areas` is passed rather than read from `totals["clusters"]` because the app
    counts an area as a cluster holding more than one note, and the strip, the
    landing and this store have to agree on one definition of the word.
    """
    out = {k: int(totals.get(k, 0) or 0) for k in SIGNALS if k != "areas"}
    out["areas"] = int(areas)
    return out


def read_history(vault: str | Path) -> list[dict]:
    """Every stored reading, oldest first. A torn tail is skipped, not raised.

    ponytail: whole-file read. The series gains a line only when a count moves,
    so it is a few thousand lines after years of daily use; seek to the tail if
    it ever passes 1 MB.
    """
    path = history_path(vault)
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        try:
            rec = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue  # a half-written last line: the reading it held is worth less
        if isinstance(rec, dict) and isinstance(rec.get("signals"), dict):
            rows.append(rec)
    return rows


def record_report(vault: str | Path, signals: dict[str, int],
                  at: str | None = None) -> dict | None:
    """Store `signals` if they moved, and return the last DIFFERENT reading.

    "Different" and not "previous line": the head equals the present whenever
    the vault has not changed since the last report, and diffing it against
    itself would blank the panel on every second open. The reading a person
    wants is against the last state that actually differed, which is also the
    only state the series holds, since identical readings are never appended.

    Returns None on the first report of a vault: there is nothing to diff, and
    a delta invented against zero would read as if the vault had just been
    written in one go.
    """
    rows = read_history(vault)
    prev = next((r for r in reversed(rows) if r.get("signals") != signals), None)
    if not rows or rows[-1].get("signals") != signals:
        record = {"at": at or _dt.datetime.now().isoformat(timespec="seconds"),
                  "signals": signals}
        path = history_path(vault)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as fh:
            fh.write(orjson.dumps(record) + b"\n")
    return prev
