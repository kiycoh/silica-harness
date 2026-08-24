# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`silica update` — keep the install current, whatever installed it.

Two install kinds, two upstreams. An editable git checkout (`uv pip install
-e .`) *is* the running code: updating is a `git pull`, guarded by a post-pull
`compileall` that rolls back to the pre-pull commit on failure — the one
corruption risk is pulling code with a syntax error, which would brick the CLI
on next launch. A wheel install (`uv tool install` / pipx / pip) has PyPI as
its upstream and its own manager owns the venv, so here we only diagnose:
compare against the latest release and name the exact upgrade command.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root, or site-packages for a wheel

PACKAGE = "silica-harness"  # PyPI distribution name ([project].name in pyproject.toml)

# behind_count()'s wheel-side cache: latest released version string, mtime =
# last check. Lives next to the user .env so `uv tool upgrade` (which replaces
# the whole venv) cannot wipe it.
CACHE = Path.home() / ".silica" / "pypi-latest"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _install_kind() -> str:
    """'git', 'uv', 'pipx', or 'pip' — which upgrade path owns this install.

    Wheel managers leave no marker inside site-packages, so the venv's own
    location is the only trace of who owns it: `uv tool` installs under
    .../uv/tools/<name>/..., pipx under a pipx/ segment. Anything else gets
    the pip command, which works in any venv even if less specific.
    """
    if (ROOT / ".git").is_dir():
        return "git"
    parts = ROOT.parts
    if any(parts[i:i + 2] == ("uv", "tools") for i in range(len(parts))):
        return "uv"
    if "pipx" in parts:
        return "pipx"
    return "pip"


def _upgrade_cmd(kind: str) -> str:
    return {
        "uv": f"uv tool upgrade {PACKAGE}",
        "pipx": f"pipx upgrade {PACKAGE}",
    }.get(kind, f"{sys.executable} -m pip install -U {PACKAGE}")


def _pypi_latest(timeout: float = 10.0) -> str:
    """Latest released version on PyPI (info.version excludes prereleases)."""
    import json
    from urllib.request import urlopen

    with urlopen(f"https://pypi.org/pypi/{PACKAGE}/json", timeout=timeout) as r:
        return json.load(r)["info"]["version"]


def _newer(latest: str, installed: str) -> bool:
    """True when `latest` is a strictly newer release than `installed`.

    Numeric-prefix compare, not `packaging` (transitive-only here) and not
    lexical ("0.9.10" must beat "0.9.3"). Parsing stops at the first
    non-numeric part, so a dev suffix ("0.9.4.dev3+g…") compares as its tag
    and garbage compares as () — a bad string can only silence the nudge,
    never nag.
    """
    def ver(s: str) -> tuple:
        out = []
        for part in s.strip().split("."):
            if not part.isdigit():
                break
            out.append(int(part))
        return tuple(out)

    return ver(latest) > ver(installed)


def update(check_only: bool = False) -> int:
    kind = _install_kind()
    if kind != "git":
        from silica import __version__
        try:
            latest = _pypi_latest()
        except (OSError, ValueError):  # no network, or PyPI answered garbage
            print("✗ Could not reach PyPI — check your network.")
            return 1
        if not _newer(latest, __version__):
            print("✓ Already up to date.")
            return 0
        print(f"→ {latest} available (installed: {__version__}).")
        print(f"  Upgrade with: {_upgrade_cmd(kind)}")
        # 1 = "an update exists and this command did not apply it": the venv
        # belongs to uv/pipx/pip, replacing it out from under them corrupts
        # their metadata, so the owner's own command must do it.
        return 0 if check_only else 1

    if shutil.which("git") is None:
        # Same shape as the installer below: every _git caller reads returncode,
        # but an absent binary raises before any of those guards can run.
        print("✗ git is not on PATH — install git, then retry.")
        return 1
    if _git("fetch", "--quiet").returncode != 0:
        print("✗ Fetch failed — check your network.")
        return 1

    counted = _git("rev-list", "--count", "HEAD..@{u}")
    if counted.returncode != 0:
        print("✗ No upstream branch to compare against.")
        print("  Set one with: git branch --set-upstream-to=origin/main")
        return 1
    ahead = counted.stdout.strip()
    if ahead in ("", "0"):
        print("✓ Already up to date.")
        return 0
    print(f"→ {ahead} new commit(s) available.")
    if check_only:  # pure query — a dirty tree is fine, we touch nothing
        return 0
    if _git("status", "--porcelain").stdout.strip():
        # Product stance: abort, don't stash. A shipped checkout is clean;
        # auto-stash only if users routinely edit their install in place.
        print("✗ Uncommitted local changes — commit or stash them, then retry.")
        return 1

    old = _git("rev-parse", "HEAD").stdout.strip()
    changed = _git("diff", "--name-only", "HEAD", "@{u}").stdout
    if _git("merge", "--ff-only", "@{u}").returncode != 0:
        print("✗ Fast-forward not possible (history diverged) — resolve manually.")
        return 1

    # Corruption guard: never keep code that doesn't byte-compile.
    if subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(ROOT / "silica")]
    ).returncode != 0:
        print("✗ Pulled code has a syntax error — rolling back.")
        _git("reset", "--hard", old)
        return 1

    if "pyproject.toml" in changed:
        print("→ Dependencies changed, reinstalling…")
        # uv is not guaranteed: a pip checkout has none on PATH, and a missing
        # binary raises FileNotFoundError instead of returning non-zero — the
        # guidance below would never print and the CLI would die on a traceback.
        uv = shutil.which("uv")
        cmd = [uv, "pip", "install", "-e", "."] if uv else [
            sys.executable, "-m", "pip", "install", "-e", "."]
        hint = "uv pip install -e ." if uv else f"{sys.executable} -m pip install -e ."
        try:
            failed = subprocess.run(cmd, cwd=ROOT).returncode != 0
        except OSError:
            failed = True
        if failed:
            print("✗ Reinstall failed — code is updated but deps may be stale.")
            print(f"  Retry manually: {hint}")
            return 1

    print("✓ Updated. Restart silica to load the new version.")
    return 0


def behind_count() -> int:
    """Updates the install is behind its upstream (0 if unknown/current).

    Git checkout: commits behind the tracking ref. Wheel install: 1 when PyPI
    has a newer release. Either way no network on the read path — a background
    refresh at most once/day keeps the local state roughly fresh, and its
    result shows on the next launch.
    """
    if _install_kind() != "git":
        return _pypi_behind()
    git = ROOT / ".git"
    try:
        fetch_head = git / "FETCH_HEAD"
        stale = not fetch_head.exists() or time.time() - fetch_head.stat().st_mtime > 86_400
        if stale:
            # Fire-and-forget — never blocks startup. Leaves one short-lived
            # zombie per session, reaped when the process exits; accepted.
            subprocess.Popen(
                ["git", "fetch", "--quiet"],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        out = _git("rev-list", "--count", "HEAD..@{u}").stdout.strip()
        return int(out) if out.isdigit() else 0
    except Exception:
        return 0  # offline / detached HEAD / no upstream → no nudge


def _pypi_behind() -> int:
    from silica import __version__
    try:
        stale = not CACHE.exists() or time.time() - CACHE.stat().st_mtime > 86_400
        if stale:
            # A daemon thread, not a Popen like the git side: the fetch is
            # pure Python here, and the thread dies with the process.
            import threading
            threading.Thread(target=_refresh_pypi_cache, daemon=True).start()
        if CACHE.exists() and _newer(CACHE.read_text(), __version__):
            return 1
    except Exception:
        pass  # unreadable cache / thread spawn failure → no nudge, like the git path
    return 0


def _refresh_pypi_cache() -> None:
    try:
        latest = _pypi_latest()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(latest)
        tmp.replace(CACHE)  # atomic: a daemon thread can die mid-write
    except Exception:
        pass  # offline / PyPI down → cache stays stale, retried next launch
