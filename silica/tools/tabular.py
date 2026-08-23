# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Tabular query tool — read-only SQL over a data file, via DuckDB.

The BI probe lane. Rows cannot survive the SourceAdapter contract (ADR-0014
turns every source into markdown prose), and no amount of rerank tuning makes
"revenue by region" a top-k similarity problem — so tabular data gets its own
retrieval path instead of being forced through recall. This module is that
path's whole surface: one tool, one dependency, no ETL and no server.

Zero-trust (ADR-0009): the SQL is model-authored, so it is parsed and rejected
unless it is a single SELECT, and the connection is confined to the target
file's own directory before the query runs.
"""
from __future__ import annotations

import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from silica.tools import tool

logger = logging.getLogger(__name__)

# Everything DuckDB binds from a bare path with no extension loaded.
READABLE_EXTS = (".csv", ".tsv", ".txt", ".parquet", ".json", ".ndjson")
# Spreadsheets: one sheet per call, extracted in-process to a temp CSV that is
# then queried like any other. The direct route (DuckDB's `excel` extension)
# needs INSTALL, which the sandbox below blocks by design — the detour keeps
# the whole downstream path (sniffing, confinement, caps) shared instead of
# duplicated per format.
SHEET_EXTS = (".xlsx", ".xlsm", ".xls")
# Text formats are sniffed on the WHOLE file (sample_size=-1), not DuckDB's
# default 20480-row head: a column typed on its head either errors on the first
# odd row past the sample or — worse — bakes a type the data does not keep.
# Parquet/JSON carry their own types, so the knob does not apply to them.
_SNIFFED_EXTS = (".csv", ".tsv", ".txt")

# Rows are capped in bytes as well as in count: `limit` alone never sees that
# 200 rows of a wide table is a six-figure token payload.
_PAYLOAD_BYTE_CAP = 50_000


def _connect(data_dir: Path):
    """A DuckDB connection that can read `data_dir` and nothing else.

    The order is load-bearing and was verified, not assumed: `allowed_directories`
    on its own confines nothing (it read /etc/passwd and wrote via COPY TO). It
    is an allowlist carved out of `enable_external_access=false`, so the access
    flag must be dropped *after* the directory is named, and the configuration
    locked *after* both — otherwise the model's SQL can simply widen it back.
    """
    import duckdb

    con = duckdb.connect()
    con.execute("SET allowed_directories=[?]", [str(data_dir)])
    con.execute("SET enable_external_access=false")
    con.execute("SET lock_configuration=true")
    return con


def _select_only(sql: str) -> str:
    """The one statement in `sql` if it is a SELECT, else ValueError.

    Parsed by DuckDB, not pattern-matched: a keyword blocklist misses `ATTACH`,
    misreads a column literally named "drop", and is exactly the flimsy half of
    a choice where the correct half costs the same one call.
    """
    import duckdb

    try:
        statements = duckdb.extract_statements(sql)
    except Exception as e:
        raise ValueError(f"unparsable SQL: {e}") from e
    if len(statements) != 1:
        raise ValueError(
            f"expected exactly 1 statement, got {len(statements)} — "
            "this tool reads, so it runs one SELECT per call"
        )
    stmt = statements[0]
    if stmt.type != duckdb.StatementType.SELECT:
        raise ValueError(
            f"{stmt.type.name} rejected: silica_query_table is read-only, "
            "the query must be a SELECT (WITH … SELECT is fine)"
        )
    return stmt.query.strip().rstrip(";")


def _bind_source(con, path: Path) -> None:
    """CREATE VIEW t over `path` — read per query, never copied.

    Interpolated because a path cannot be a prepared-statement parameter in
    FROM; the path is resolved and real, and the quote-doubling closes the
    only injection route left.
    """
    quoted = str(path).replace("'", "''")
    if path.suffix.lower() in _SNIFFED_EXTS:
        con.execute(
            f"CREATE VIEW t AS SELECT * FROM read_csv('{quoted}', sample_size=-1)"
        )
    else:
        con.execute(f"CREATE VIEW t AS SELECT * FROM '{quoted}'")


def _pick_sheet(names: list[str], sheet: str) -> str:
    """The requested sheet name, or the only one; ambiguity is a teaching error."""
    if sheet:
        if sheet in names:
            return sheet
        raise ValueError(f"sheet {sheet!r} not found — this workbook has: {', '.join(names)}")
    if len(names) == 1:
        return names[0]
    # Guessing (say, the first sheet) would answer from the wrong table without
    # any signal that it did; the retry costs one call and names the choices.
    raise ValueError(
        "this workbook has several sheets — pass sheet= to pick one of: " + ", ".join(names)
    )


def _sheet_rows(src: Path, sheet: str):
    """(sheet name, iterator of value rows) for a workbook, values not formulas."""
    if src.suffix.lower() == ".xls":
        import xlrd  # base dependency; convert.py reads .xls with it too

        # logfile: xlrd narrates parse trivia straight to stdout otherwise.
        book = xlrd.open_workbook(str(src), logfile=io.StringIO())
        name = _pick_sheet(book.sheet_names(), sheet)
        sh = book.sheet_by_name(name)

        def cell(c):
            # Dates are bare floats in BIFF; only the cell type says so.
            if c.ctype == xlrd.XL_CELL_DATE:
                return xlrd.xldate_as_datetime(c.value, book.datemode)
            return c.value

        return name, ([cell(c) for c in sh.row(r)] for r in range(sh.nrows))
    try:
        from openpyxl import load_workbook
    except ImportError as e:
        raise ValueError(
            "reading .xlsx needs openpyxl: pip install 'silica-harness[bi]'"
        ) from e
    wb = load_workbook(src, read_only=True, data_only=True)
    name = _pick_sheet(wb.sheetnames, sheet)
    return name, (list(r) for r in wb[name].iter_rows(values_only=True))


def _sheet_to_csv(src: Path, sheet: str, workdir: Path) -> tuple[str, Path]:
    """Extract one sheet to a CSV under `workdir`; (sheet name, csv path).

    A CSV round-trip on purpose: datetimes go out as ISO strings and numbers as
    their repr, both of which the full-file sniff recovers unambiguously — and
    typed values that openpyxl/xlrd already resolved (a formula's cached value,
    a BIFF date float) arrive as data, not artifacts.
    """
    import csv

    name, rows = _sheet_rows(src, sheet)
    out = workdir / "sheet.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for row in rows:
            writer.writerow(
                "" if v is None else v.isoformat() if hasattr(v, "isoformat") else v
                for v in row
            )
    return name, out


def _fit_rows(rows: list[list[Any]]) -> list[list[Any]]:
    """The longest prefix of `rows` that serializes under the byte cap."""
    size = 0
    for i, row in enumerate(rows):
        size += len(json.dumps(row, ensure_ascii=False, default=str)) + 2
        if size > _PAYLOAD_BYTE_CAP:
            return rows[:i]
    return rows


class QueryTableArgs(BaseModel):
    path: str = Field(
        description=(
            "Path to the data file to query "
            "(.csv/.tsv/.parquet/.json, or .xlsx/.xls for one sheet at a time)"
        ),
    )
    sql: str = Field(
        description=(
            "A single read-only SELECT. The file is bound to the table name `t` "
            "— e.g. SELECT region, avg(score) FROM t GROUP BY 1. "
            "When the columns are unknown, call with `SUMMARIZE t` first: one "
            "call returns every column's type, min/max, distinct count and "
            "null share, without spending rows."
        ),
    )
    limit: int = Field(
        default=200,
        description="Max rows returned; the reply flags whether it truncated",
    )
    sheet: str = Field(
        default="",
        description=(
            "Excel files only: which sheet to query. Omit for single-sheet "
            "workbooks; a multi-sheet workbook rejects the call and lists its "
            "sheet names."
        ),
    )


@tool(QueryTableArgs, cls="atomic")
def silica_query_table(
    path: str, sql: str, limit: int = 200, sheet: str = ""
) -> dict[str, Any]:
    """Answers a question about a data file (.csv/.tsv/.parquet/.json, Excel
    via sheet=) by running SQL over it — the aggregation path semantic search
    cannot do: sums, group-by, ranking, ranges. Queried in place, bound to
    table `t`. Read-only: one SELECT per call. `SUMMARIZE t` is the cheap
    first call when columns are unknown. Replies carry the bound schema; a
    numeric-looking VARCHAR column holds non-numbers — count what try_cast
    loses before trusting an aggregate.
    """
    try:
        import duckdb  # noqa: F401
    except ImportError as e:
        raise ValueError(
            "the tabular lane needs DuckDB: pip install 'silica-harness[bi]'"
        ) from e

    src = Path(path).expanduser()
    try:
        src = src.resolve(strict=True)
    except OSError as e:
        raise ValueError(f"cannot read {path}: {e}") from e
    if not src.is_file():
        raise ValueError(f"{src} is not a file")
    suffix = src.suffix.lower()
    if suffix not in (*READABLE_EXTS, *SHEET_EXTS):
        raise ValueError(
            f"{suffix or 'no extension'} is not a tabular format — "
            f"expected one of {', '.join((*READABLE_EXTS, *SHEET_EXTS))}"
        )
    if sheet and suffix not in SHEET_EXTS:
        raise ValueError(f"sheet= applies to Excel files only, not {suffix}")

    inner = _select_only(sql)
    sheet_name = ""
    # The temp dir is the confinement boundary for the sheet path: the workbook
    # is read in-process BEFORE any model SQL runs, so the SQL can only see the
    # one extracted sheet.
    with tempfile.TemporaryDirectory() as tmp:
        if suffix in SHEET_EXTS:
            sheet_name, bound = _sheet_to_csv(src, sheet, Path(tmp))
        else:
            bound = src
        con = _connect(bound.parent)
        try:
            _bind_source(con, bound)
            schema = dict(
                (row[0], row[1]) for row in con.execute("DESCRIBE t").fetchall()
            )
            # limit+1 so truncation is observed rather than guessed at.
            rows = con.execute(f"SELECT * FROM ({inner}) LIMIT {int(limit) + 1}").fetchall()
            columns = [d[0] for d in con.description]
        except Exception as e:
            raise ValueError(f"query failed: {type(e).__name__}: {e}") from e
        finally:
            con.close()

    over_limit = len(rows) > limit
    within_limit = [list(r) for r in rows[:limit]]
    kept = _fit_rows(within_limit)
    truncated = over_limit or len(kept) < len(within_limit)
    if over_limit:
        note = f"truncated at limit={limit}; aggregate or raise limit"
    elif truncated:
        note = (
            f"truncated at ~{_PAYLOAD_BYTE_CAP // 1000} KB of rows; "
            "aggregate or select fewer columns instead of scanning"
        )
    else:
        note = ""
    return {
        "path": str(src),
        **({"sheet": sheet_name} if sheet_name else {}),
        # The bound table's schema on every reply: SQL written against guessed
        # column names or types is the lane's main failure mode.
        "schema": schema,
        "columns": columns,
        "rows": kept,
        "row_count": len(kept),
        # Never a silent cap: a truncated answer that reads as complete is how a
        # BI number comes out confidently wrong.
        "truncated": truncated,
        **({"note": note} if note else {}),
    }

# One file per call, so no joins across files. Binding a dict of path→name as
# t1..tn is the design for the day a real question needs two tables at once.
