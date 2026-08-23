# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`silica setup <client>` — wire the MCP server into a coding agent's config.

The read side of the vault needs no model and no key, so serving it over MCP is
the shortest path from install to something useful. What stood between the two
was hand-pasting a JSON or TOML block into a file whose location the user has to
look up. This writes that block instead.

Never clobbers: an existing silica entry is left alone (the user may have tuned
it), the file is backed up before any write, and `--dry-run` prints what would
change. A file that does not parse is refused rather than overwritten.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import tomllib
from importlib.resources import files
from pathlib import Path

import yaml
from rich.markup import escape

from silica.ui.console import CONSOLE

# Everything this module prints carries a payload full of square brackets: the
# `[mcp]` extra, TOML table headers, a parser error quoting a `]`. rich reads a
# bracketed word as a style tag and drops it, which silently turned the printed
# install command into `--from silica-harness` — a command that runs and installs
# the wrong thing. Every interpolated value goes through escape(), and the
# config block (which has no styling of its own) prints with markup off.

# What every client is told to run. uvx keeps the server at one command with no
# install step of its own, which is the whole point of the generated block.
MCP_COMMAND = ["uvx", "--from", "silica-harness[mcp]", "silica", "mcp"]

# No SILICA_VAULT in the generated block, deliberately. Claude Code, Codex and
# opencode are CLIs launched from a project and spawn the stdio server with
# that working directory, which is already the answer (`cli.resolve_cwd_vault`),
# so the server serves the project you opened — the Claude Code model, one
# vault per place. Writing the vault that happened to be active at `silica
# setup` time would pin it into every project afterwards, which is what
# `_activate_repo_mode` warns against. A fixed vault is still expressible:
# export SILICA_VAULT, or add the env block by hand to the file this wrote.
# DeepSeek Harness is the one client where that pin is worth considering: its
# `dsh web` process spawns the server once, in its own launch folder, for
# every session it serves (see `_setup_dsh`).

CLIENTS = ("claude", "codex", "opencode", "dsh")

# Codex gives a stdio server `startup_timeout_sec` (default 10) to answer
# `initialize`. A cold `uvx` resolves and installs silica-harness[mcp] first,
# which takes longer than that on a first run, and a server that misses the
# window is simply absent for that session, with one line in a log nobody
# reads.
CODEX_STARTUP_TIMEOUT_SEC = 60


def skill_path() -> Path:
    """The one SKILL.md. It ships inside the package so an installed `silica`
    can copy it into a client's skill root, and the plugin manifests point at
    the same file, so there is nothing to keep in step."""
    return Path(str(files("silica") / "skills" / "silica" / "SKILL.md"))


def install_skill() -> Path:
    """Copy the skill to ~/.agents/skills, the user root both Codex and
    DeepSeek Harness scan, so one copy serves both. Claude Code gets the skill
    from the plugin instead. A copy and not a symlink: the package path moves
    on upgrade and symlinks need privileges on Windows; rerunning setup
    refreshes it."""
    dest = Path.home() / ".agents" / "skills" / "silica" / "SKILL.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(skill_path(), dest)
    return dest


def _default_path(client: str) -> Path:
    home = Path.home()
    if client == "codex":
        return home / ".codex" / "config.toml"
    if client == "dsh":
        # resolveDshHome in the harness: $DSH_HOME, else ~/.dsh. The home-level
        # patch file is the layer applied over every profile.
        return Path(os.environ.get("DSH_HOME") or home / ".dsh") / "cordis.patch.yml"
    return home / ".config" / "opencode" / "opencode.json"


def _backup(path: Path) -> Path:
    """Timestamped copy beside the original, returned for the report."""
    dest = path.with_suffix(path.suffix + f".bak.{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, dest)
    return dest


def _report(path: Path, block: str, dry_run: bool, backup: Path | None) -> int:
    if dry_run:
        CONSOLE.print(f"  [dim]would write to {escape(str(path))}:[/]")
        CONSOLE.print(block, markup=False)
        return 0
    CONSOLE.print(f"  [green]✓[/] wrote {escape(str(path))}")
    if backup:
        CONSOLE.print(f"  [dim]backup: {escape(str(backup))}[/]")
    return 0


def _codex_block() -> str:
    args = ", ".join(f'"{a}"' for a in MCP_COMMAND[1:])
    return (
        "\n[mcp_servers.silica]\n"
        f'command = "{MCP_COMMAND[0]}"\n'
        f"args = [{args}]\n"
        f"startup_timeout_sec = {CODEX_STARTUP_TIMEOUT_SEC}\n"
    )


def _setup_codex(path: Path, dry_run: bool) -> int:
    """Append the server block to ~/.codex/config.toml.

    ponytail: appended as text, not re-serialised. tomllib reads but cannot
    write, and a real TOML writer is a dependency for one block — appending
    also preserves the comments and ordering a round-trip would flatten. The
    ceiling is that it only ever adds a top-level table; if silica ever needs
    to edit an existing entry, that is when tomlkit earns its place.
    """
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing.strip():
        try:
            parsed = tomllib.loads(existing)
        except tomllib.TOMLDecodeError as e:
            CONSOLE.print(f"  [red]✗[/] {escape(str(path))} is not valid TOML ({escape(str(e))}) — not touching it")
            return 1
        if "silica" in parsed.get("mcp_servers", {}):
            CONSOLE.print(f"  [dim]silica is already configured in {escape(str(path))} — nothing to do[/]")
            return 0
    block = _codex_block()
    if dry_run:
        return _report(path, block, True, None)
    backup = _backup(path) if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "" if not existing or existing.endswith("\n") else "\n"
    path.write_text(existing + sep + block, encoding="utf-8")
    return _report(path, block, False, backup)


def _dsh_row() -> dict:
    return {
        "id": "mcp-silica",
        "name": "@deepseek-ai/dsh-mcp-client",
        "config": {
            "serverName": "silica",
            "transport": "stdio",
            "command": MCP_COMMAND[0],
            "args": MCP_COMMAND[1:],
        },
    }


def _setup_dsh(path: Path, dry_run: bool) -> int:
    """Insert the MCP client row into the harness's user patch file.

    DeepSeek Harness composes one `dsh-mcp-client` row per server and reads
    `cordis.patch.yml` as a top-level list of patches, so the row rides an
    `insert` entry appended to that list. Re-serialised through yaml rather
    than appended as text: the file is one YAML document, and text appended
    after a list can land inside the previous entry's indentation. Comments
    do not survive the round trip, which is what the backup is for.

    The harness spawns this child once per process, in the folder `dsh web`
    was launched from, so on this client the vault is that folder.
    """
    patches: list = []
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing.strip():
        try:
            patches = yaml.safe_load(existing) or []
        except yaml.YAMLError as e:
            CONSOLE.print(f"  [red]✗[/] {escape(str(path))} is not valid YAML ({escape(str(e))}) — not touching it")
            return 1
        if not isinstance(patches, list):
            CONSOLE.print(f"  [red]✗[/] {escape(str(path))} is not a list of patches — not touching it")
            return 1
        rows = [r for p in patches if isinstance(p, dict) for r in p.get("insert") or [] if isinstance(r, dict)]
        if any(r.get("id") == "mcp-silica" for r in rows):
            CONSOLE.print(f"  [dim]silica is already configured in {escape(str(path))} — nothing to do[/]")
            return 0
    patches.append({"insert": [_dsh_row()]})
    block = yaml.safe_dump(patches, sort_keys=False, allow_unicode=True)
    if dry_run:
        return _report(path, block, True, None)
    backup = _backup(path) if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(block, encoding="utf-8")
    return _report(path, block, False, backup)


def _setup_opencode(path: Path, dry_run: bool) -> int:
    """Merge the server into opencode.json under `mcp.silica`."""
    data: dict = {}
    if path.exists() and path.read_text(encoding="utf-8").strip():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            CONSOLE.print(f"  [red]✗[/] {escape(str(path))} is not valid JSON ({escape(str(e))}) — not touching it")
            return 1
        if not isinstance(data, dict):
            CONSOLE.print(f"  [red]✗[/] {escape(str(path))} is not a JSON object — not touching it")
            return 1
        if "silica" in data.get("mcp", {}):
            CONSOLE.print(f"  [dim]silica is already configured in {escape(str(path))} — nothing to do[/]")
            return 0
    entry: dict = {"type": "local", "command": MCP_COMMAND, "enabled": True}
    data.setdefault("mcp", {})["silica"] = entry
    block = json.dumps(data, indent=2) + "\n"
    if dry_run:
        return _report(path, block, True, None)
    backup = _backup(path) if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(block, encoding="utf-8")
    return _report(path, block, False, backup)


def _setup_claude(dry_run: bool) -> int:
    """Delegate to `claude mcp add`.

    Claude Code owns its own config format and ships a command for exactly this,
    so writing the file by hand would be a second implementation to keep in sync
    with theirs. When the CLI is absent, printing the command is still the whole
    answer.

    User scope, not the `local` default: the entry names no vault, so the one
    registration serves every project (each resolving its own vault from the cwd
    Claude spawns the server in). Per-project scope would mean re-running this
    in each repo to say the same thing.
    """
    cmd = ["claude", "mcp", "add", "--scope", "user",
           "--transport", "stdio", "silica", "--", *MCP_COMMAND]
    printable = " ".join(cmd)
    if dry_run or not shutil.which("claude"):
        if not dry_run:
            CONSOLE.print("  [yellow]⚠[/] the `claude` CLI is not on PATH — run this yourself:")
        # soft_wrap so rich does not fold the line at the console width: this is
        # a command meant to be copied, and a wrap puts a real newline in the
        # middle of it, so pasting runs a fragment.
        CONSOLE.print(f"  {printable}", markup=False, soft_wrap=True)
        return 0
    result = subprocess.run(cmd)
    if result.returncode != 0:
        CONSOLE.print(f"  [red]✗[/] `{escape(printable)}` failed")
        return result.returncode
    CONSOLE.print("  [green]✓[/] registered with Claude Code (user scope: every project, vault from its folder)")
    CONSOLE.print(
        "  [dim]for the skill and the session hooks too: "
        "claude plugin marketplace add kiycoh/silica-harness && "
        "claude plugin install silica@silica[/]"
    )
    return 0


def run_setup(args: list[str]) -> int:
    """`silica setup <client> [--dry-run] [--config PATH]`."""
    positional = [a for a in args if not a.startswith("-")]
    client = positional[0] if positional else ""
    if client not in CLIENTS:
        CONSOLE.print(f"  Usage: silica setup <{'|'.join(CLIENTS)}> [--dry-run] [--config PATH]", markup=False)
        return 1
    dry_run = "--dry-run" in args
    if client == "claude":
        return _setup_claude(dry_run)
    override = next((a.split("=", 1)[1] for a in args if a.startswith("--config=")), "")
    if not override and "--config" in args:
        i = args.index("--config")
        override = args[i + 1] if i + 1 < len(args) else ""
    path = Path(override).expanduser() if override else _default_path(client)
    writers = {"codex": _setup_codex, "dsh": _setup_dsh, "opencode": _setup_opencode}
    rc = writers[client](path, dry_run)
    # Also on "already configured": rerunning setup is how the skill copy
    # follows a package upgrade.
    if rc == 0 and not dry_run and client in ("codex", "dsh"):
        CONSOLE.print(f"  [green]✓[/] skill installed at {escape(str(install_skill()))}")
    return rc
