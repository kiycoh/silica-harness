# tests/webassets.py
"""app.js is eight files and app.css nine, and a test that greps one wants
one string.

Both cuts were mechanical -- `cat` of the pieces in load order is byte-identical
to the file that was there before -- so joining them back is not an
approximation of the old text, it IS the old text. That matters for the
assertions that slice (`app[app.index("function renderPins()"):]`): they only
hold if the join keeps the order the browser loads in, which is why the order is
read out of index.html rather than listed here as a second copy that can drift
from it.
"""
from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "silica" / "ui" / "web" / "static"

_INDEX = (WEB / "index.html").read_text(encoding="utf-8")

APP_JS_PARTS = [
    WEB / name
    for name in re.findall(r'<script src="/static/(app-[\w-]+\.js)">', _INDEX)
]
APP_CSS_PARTS = [
    WEB / name
    for name in re.findall(r'<link rel="stylesheet" href="/static/(app-[\w-]+\.css)"', _INDEX)
]


def _join(parts: list[Path], what: str) -> str:
    assert parts, f"index.html loads no {what} — did the split get reverted?"
    return "".join(p.read_text(encoding="utf-8") for p in parts)


def app_js() -> str:
    """The eight cuts of app.js, joined in the order index.html loads them."""
    return _join(APP_JS_PARTS, "app-*.js")


def app_css() -> str:
    """The nine cuts of app.css, joined in the order index.html links them."""
    return _join(APP_CSS_PARTS, "app-*.css")
