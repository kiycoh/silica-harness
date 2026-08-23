# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`silica hook <event>` — the hook-side producer for coding-agent harnesses.

Claude Code, Codex and DeepSeek Harness all run a `hooks.json` command with
the event payload on stdin and add plain stdout to the session as context on
SessionStart (ADR-0025). Plain text rather than the JSON envelope on purpose:
the envelope's field names differ per dialect, while stdout text is what all
three accept on that event.

Nothing here may fail loud. This process runs inside someone else's session,
so every exit is 0 and every problem is silence; `silica.capture` made the
same choice for the end of the session.
"""
from __future__ import annotations

import json
import os
import sys

from silica.capture import find_vault


def session_start(stdin_text: str) -> str:
    """The session's opening line about its vault, or "" when cwd has none."""
    try:
        payload = json.loads(stdin_text or "{}")
        cwd = payload.get("cwd") or os.getcwd()
    except (ValueError, AttributeError):
        return ""  # the payload is the client's; anything unreadable is "no vault"
    vault = find_vault(str(cwd))
    if vault is None:
        return ""
    from silica.ui.mcp import INSTRUCTIONS

    # The path and the loop, nothing counted: a note count needs the excludes
    # the indexes apply (a repo vault here walked 37,897 gitignored fixtures
    # as "notes"), and the count changes nothing the agent does next.
    return f"Silica vault: {vault}.\n{INSTRUCTIONS}\n"


_PRODUCERS = {"SessionStart": session_start}


def run_hook(argv: list[str], stdin_text: str) -> int:
    producer = _PRODUCERS.get(argv[0] if argv else "")
    if producer is None:
        return 0  # an event this version has nothing to say about is not an error
    text = producer(stdin_text)
    if text:
        sys.stdout.write(text)
        sys.stdout.flush()
    return 0
