# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""FlatMarkdown autolinks bare URLs so the terminal can open them.

rich builds its own parser inside `Markdown.__init__` and parses on the spot, so
the subclass re-parses with linkify on. The one call site is the agent answer
(cli.py), which is where a `## Sources (web)` block lands.
"""
from __future__ import annotations

from rich.console import Console

from silica.ui.style import FlatMarkdown

OSC8 = "\x1b]8;"  # the hyperlink escape rich emits around a link token


def _render(markup: str) -> str:
    console = Console(force_terminal=True, width=120, _environ={"TERM_PROGRAM": "iTerm.app"})
    with console.capture() as cap:
        console.print(FlatMarkdown(markup))
    return cap.get()


def test_bare_url_becomes_a_terminal_hyperlink():
    out = _render("1. Chem - https://en.wikipedia.org/wiki/chemistry")
    assert OSC8 in out
    assert "https://en.wikipedia.org/wiki/chemistry" in out
    # The URL is still the displayed text, which is what keeps this honest on a
    # terminal with no OSC 8 support: the escape is dropped, the URL is not.
    assert out.count("https://en.wikipedia.org/wiki/chemistry") >= 2


def test_vault_paths_are_never_mistaken_for_websites():
    # `.md` is Moldova's ccTLD: with linkify's fuzzy matching left on, a note
    # reference in prose renders as a link to http://nota.md.
    out = _render("vedi nota.md e area/nota.md, mail foo@bar.com")
    assert OSC8 not in out
    assert "nota.md" in out and "area/nota.md" in out
