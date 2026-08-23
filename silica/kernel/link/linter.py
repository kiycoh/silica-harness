# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path
from silica.kernel.link import ofm
from silica.kernel.write import frontmatter
from silica.driver import DRIVER

# Internal wire-format delimiter (`===SILICA-BODY N===`, garbled variants
# included). One leaked verbatim into a committed note on 2026-08-15 — the
# exact corruption class the post-write gate exists to reject, so any line
# shaped like it is a hard error, never a warning.
_SENTINEL_LEAK_RE = re.compile(r"^===\s*SILICA\b.*===\s*$", re.MULTILINE | re.IGNORECASE)


def check_expires_at(data: dict) -> list[str]:
    """Return warnings for `expires_at` frontmatter violations.

    Returns a list with at most one warning string:
    - empty list if the field is absent, None, or a future/today date
    - one-element list if the date is past or unparseable
    """
    raw = (data or {}).get("expires_at")
    if raw is None:
        return []
    try:
        expires = datetime.date.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return [f"expires_at '{raw}' is invalid (expected ISO 8601 date)"]
    if expires < datetime.date.today():
        return [f"Note expired on {raw}"]
    return []


def check_documents_paths(data: dict, repo_root=None) -> list[str]:
    """Return a warning per `documents:` path that no longer exists on disk.

    Non-blocking — same class as a broken wikilink. `repo_root` is the base the
    repo-relative paths resolve against; when None, no check is performed.
    """
    raw = (data or {}).get("documents")
    if not raw or repo_root is None:
        return []
    paths = [raw] if isinstance(raw, str) else list(raw)
    root = Path(repo_root)
    warnings = []
    for rel in paths:
        if not (root / str(rel)).exists():
            warnings.append(f"documents: path '{rel}' no longer exists")
    return warnings


def check_plan_status(data: dict) -> list[str]:
    """Return a warning if `status:` is present but outside the plan enum.

    Non-blocking — absent status returns []. Enum mirrors plans.VALID_STATUS.
    """
    raw = (data or {}).get("status")
    if raw is None:
        return []
    from silica.kernel.plans import VALID_STATUS
    status = str(raw).strip()
    if status not in VALID_STATUS:
        return [f"status '{raw}' is not a valid plan status ({sorted(VALID_STATUS)})"]
    return []


def validate_note(path, hub, op_type=None):
    """Validate a single note.

    Returns (errors, warnings) where errors are hard violations that fail
    the pipeline and warnings are auditable flags that do NOT block.
    """
    errors = []
    warnings = []
    try:
        nc = DRIVER.read_note(path)
        content = nc.content

        if _SENTINEL_LEAK_RE.search(content):
            errors.append("Internal delimiter leaked into note (===SILICA-…=== line)")

        data, _, _ = frontmatter.split(content)
        if data is None:
            errors.append("Missing or invalid frontmatter")
        else:
            warnings += check_expires_at(data)
            # Notes live in the vault, so the vault's validated code-lane root
            # (ADR-0019) is the right base — not a walk-up from the note file.
            from silica.config import CONFIG
            from silica.kernel.recall.paths import repo_root_for
            warnings += check_documents_paths(
                data, repo_root=repo_root_for(getattr(CONFIG, "vault_path", "") or "")
            )
            warnings += check_plan_status(data)

        # hub wikilink: required for spoke write/patch; NOT for hub-index/reformat/merge
        # overwrites, and NOT for the hub note itself (a self-link is meaningless).
        is_hub_note = hub and Path(path).stem.casefold() == hub.casefold()
        if op_type != "overwrite" and hub and not is_hub_note and not ofm.has_wikilink(content, hub):
            errors.append(f"Missing wikilink to [[{hub}]]")

        # atomicity: skip for patch (append) only
        if op_type != "patch":
            m = ofm.metrics(content)
            if m["line_count"] > ofm.LIMITS["max_lines"]:
                warnings.append(f"Note too long ({m['line_count']} lines)")
            if m["char_count"] > ofm.LIMITS["max_chars"]:
                warnings.append(f"Note too large ({m['char_count']} chars)")

        # OFM structural lint (calibrated against golden notes)
        r = ofm.ofm_lint(content, stem=Path(path).stem)
        errors += r["violations"]
        warnings += r["flags"]

    except Exception as e:
        errors.append(f"Read error: {e}")
    return errors, warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="Target folder in the vault")
    parser.add_argument("--operations", help="Path to JSON file containing operations")
    parser.add_argument("--files", nargs="+", help="Specific file paths to validate")
    parser.add_argument("--hub", default=None, help="Hub note name for wikilink validation (optional for dedup)")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format (text or json)")
    args = parser.parse_args()

    if not args.target and not args.operations and not args.files:
        if args.format == "json":
            print(json.dumps({"error": "Either --target, --operations, or --files must be specified."}))
        else:
            print("Error: Either --target, --operations, or --files must be specified.")
        sys.exit(1)

    files_to_check = [] # List of tuples: (path, op_type, per_file_hub)
    if args.files:
        for f in args.files:
            files_to_check.append((f, None, None))
    elif args.operations:
        if not os.path.exists(args.operations):
            if args.format == "json":
                print(json.dumps({"error": f"Operations file {args.operations} does not exist."}))
            else:
                print(f"Error: Operations file {args.operations} does not exist.")
            sys.exit(1)
        try:
            with open(args.operations, 'r', encoding='utf-8') as f:
                ops = json.load(f)
            for op in ops:
                if op.get("op") == "delete":
                    continue
                path = op.get("path")
                if path and path.endswith('.md'):
                    files_to_check.append((path, op.get("op"), op.get("hub")))
        except Exception as e:
            if args.format == "json":
                print(json.dumps({"error": f"Failed to parse operations JSON: {e}"}))
            else:
                print(f"Error: Failed to parse operations JSON: {e}")
            sys.exit(1)
    elif args.target:
        if not os.path.isdir(args.target):
            if args.format == "json":
                print(json.dumps({"error": f"Target directory {args.target} does not exist."}))
            else:
                print(f"Error: Target directory {args.target} does not exist.")
            sys.exit(1)
        for f in os.listdir(args.target):
            if f.endswith('.md'):
                files_to_check.append((os.path.join(args.target, f), None, None))

    if not files_to_check:
        # A zero-file run printed "All files validated successfully.": a typo
        # in --target passed the gate. Nothing scanned is an invocation error.
        msg = f"no .md files to validate under {args.target or args.operations}"
        print(json.dumps({"error": msg}) if args.format == "json" else f"Error: {msg}")
        sys.exit(1)

    error_results = {}
    warning_results = {}
    for path, op_type, per_file_hub in files_to_check:
        effective_hub = args.hub or per_file_hub
        if os.path.exists(path):
            errs, warns = validate_note(path, effective_hub, op_type)
            if errs:
                error_results[os.path.basename(path)] = errs
            if warns:
                warning_results[os.path.basename(path)] = warns
        else:
            error_results[os.path.basename(path)] = ["File does not exist"]
    
    if args.format == "json":
        print(json.dumps({
            "success": not error_results,
            "failed_count": len(error_results),
            "errors": error_results,
            "warning_count": len(warning_results),
            "warnings": warning_results,
        }, indent=2, ensure_ascii=False))
        if error_results:
            sys.exit(1)
    else:
        # Warnings (flags) — always printed, never block
        if warning_results:
            print(f"Warnings for {len(warning_results)} files:")
            for fname, warns in warning_results.items():
                print(f"  ⚠ {fname}:")
                for w in warns:
                    print(f"    · {w}")
            print()  # blank line separator

        # Errors (violations) — block pipeline
        if not error_results:
            print("All files validated successfully.")
        else:
            print(f"Validation failed for {len(error_results)} files:")
            for fname, errs in error_results.items():
                print(f"  ✗ {fname}:")
                for err in errs:
                    print(f"    * {err}")
            sys.exit(1)
