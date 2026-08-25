# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""FastAPI backend for the localhost GUI.

Single in-memory session (localhost, one user, no auth). The critical seam is
sync `run_agent` (blocking) -> async SSE: run it in a worker thread and bridge
its callback events onto the event loop with `call_soon_threadsafe`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from silica.agent.constraints import AgentConstraints, chat_tools, web_turn_constraints
from silica.agent.loop import _is_tool_failure, run_agent
from silica.agent.recall_watch import THIN_COVERAGE_HINT, RecallWatch
from silica.config import CONFIG
from silica.kernel.recall.mindmap import note_resolver
from silica.kernel.write import session_changes
from silica.sources.web_research import WebTurn
from silica.ui.web.callback import event_to_json, tool_calls_to_json

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# --- module-level session state (spec: single session) -----------------------
messages: list[dict] = []
current_cancel: threading.Event | None = None  # cancel token of the in-flight turn
current_task: asyncio.Task | None = None  # in-flight worker; owns the busy-gate release
_collapsed: set[int] = set()  # message indices elided by compaction, across turns
_busy = False  # one turn at a time; a second /chat is refused with 409
current_session_id: str | None = None  # narration sid of the live conversation
# The store moved to ~/.silica/narration (docs/specs/narration/spec.md §5);
# legacy web_sessions/*.json are read forever through silica.agent.narration.


# Fresh-session seed, precomputed so /reset ("new chat") is instant instead of
# rebuilding the vault map + token count on the click path (~seconds on a real
# vault). Built at startup, refreshed in the background after each turn (the
# turn may have written notes). (messages, their token count).
_seed: tuple[list[dict], int] | None = None


def _build_seed() -> tuple[list[dict], int]:
    """Compute the fresh-session seed. Never touches the live session state:
    uses the pure token counter so a background rebuild can't clobber the
    context meter of the conversation in progress."""
    global _seed
    from silica.cli import _count_context_tokens, seed_messages

    # The same builder the TUI seeds from, math=True for the MathML renderer.
    msgs = seed_messages(math=True)
    _seed = (msgs, _count_context_tokens(msgs))
    return _seed


def _prewarm_seed() -> None:
    """Refresh the seed off the request path; failures only cost freshness."""

    def work():
        try:
            _build_seed()
        except Exception:
            logger.exception("seed prewarm failed")

    threading.Thread(target=work, daemon=True).start()


def _reset_session() -> None:
    global current_cancel, current_task, _busy, current_session_id
    from silica.agent import narration as _narr
    _narr.NARRATOR.close()   # release the flock; next turn opens a fresh sid
    seed_msgs, seed_tokens = _seed if _seed is not None else _build_seed()
    messages[:] = [dict(m) for m in seed_msgs]  # per-message copy; contents are never mutated
    CONFIG.context_tokens = seed_tokens
    _collapsed.clear()
    current_cancel = None
    current_task = None
    _busy = False
    current_session_id = None  # next turn opens a new file


def _ctx_meter() -> dict:
    """What the composer's context ring reads, in one shape.

    The parts are counted, not apportioned: `_context_breakdown` charges the
    chat envelope once so they sum to the total beside them, and a meter that
    invites you to add up its own segments has to survive that. Three sites emit
    this (the two `done` yields and the transcript headers) and they must not
    drift, since the ring is the one place the user is told how much room is
    left before the history starts being collapsed.
    """
    from silica.agent.compaction import COMPACT_FRACTION
    from silica.cli import _context_breakdown

    return {"context_tokens": CONFIG.context_tokens,
            "max_context_tokens": CONFIG.max_context_tokens,
            "context_parts": _context_breakdown(messages),
            # The one fill level that means anything: past it the loop starts
            # collapsing old read results. Sent rather than mirrored in the
            # client, so the ring cannot go on reassuring at 0.6 after someone
            # moves the threshold in compaction.py.
            "compact_at": COMPACT_FRACTION}


def _capture_own_session() -> None:
    """Flush the live conversation to the capture WAL, if capture is on.

    The server owns the session, so the two moments it can see a conversation
    end are a new chat and its own shutdown. A closed tab is neither; the next
    one of these catches its content (accepted ceiling, spec §10).
    `capture_session` is opt-in and fail-open in itself — a capture bug can
    never break the GUI from here.
    """
    from silica.capture import capture_session

    capture_session(messages, session_id=current_session_id or uuid.uuid4().hex[:12],
                    driver="gui")


def _narrate_user_turn(msg: dict) -> None:
    """Session born at the first user turn (spec §5), then the turn beat.
    Runs on the worker thread; ensure_session is idempotent per turn."""
    global current_session_id
    from silica.agent import narration as _narr
    try:
        current_session_id = _narr.NARRATOR.ensure_session(
            driver="gui", sid=current_session_id)
        _narr.NARRATOR.turn(msg)
    except _narr.SessionBusy as e:   # another process owns it: keep chatting unsaved
        logger.warning("narration unavailable for this turn: %s", e)


def _session_title(msgs: list[dict]) -> str:
    for m in msgs:
        if m.get("role") == "user" and m.get("content"):
            line = str(m["content"]).strip().splitlines()[0]
            return line[:57] + "…" if len(line) > 58 else line
    return "untitled"


def _list_sessions() -> list[dict]:
    """Saved conversations for the current vault, newest first, both stores.
    _save_session is gone with the snapshot model: appending turn beats IS
    the save (spec §5), so a refresh/close can no longer lose history."""
    from silica.agent import narration as _narr
    return _narr.list_sessions(CONFIG.vault_path or "")


import html as _html
import re
from html.parser import HTMLParser
from urllib.parse import quote as _quote
from urllib.parse import urlsplit as _urlsplit

# A whitespace-delimited path-like token: contains "/" or ends in ".md".
_PATHLIKE = re.compile(r"[^\s\[\]]*(?:/[^\s\[\]]*|\.md)")
_WIKILINK = re.compile(r"(!?)\[\[([^\]\[]+)\]\]")  # optional ! marks an embed
_TRAIL = ".,;:!?)"  # sentence punctuation to peel off a bare path token

# Vault attachments the drawer may inline; served only through /asset.
_ASSET_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}

# An SVG is a document, not just a picture: an <img> context runs none of its
# scripts, but /asset is a same-origin URL a note can link and the address bar
# can reach, and a navigated SVG runs them on the GUI's origin. `sandbox` puts
# any such navigation in an opaque origin with scripting off, and default-src
# 'none' stops the file fetching anything; neither affects an <img> load, so
# inline vault images keep rendering. nosniff pins the guessed type, which is
# already correct for every extension above.
_ASSET_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
    "X-Content-Type-Options": "nosniff",
}

# --- OFM (Obsidian-flavored markdown) sugar ----------------------------------
# ==highlight== | #tag (letter-first, so #123 and hex colors stay literal)
_MARK_OR_TAG = re.compile(r"==([^=\n]+)==|(?<![\w#])#([A-Za-z_][\w/-]*)")
_COMMENT = re.compile(r"%%.*?%%", re.S)
_BLOCK_ID = re.compile(r"[ \t]+\^[\w-]+[ \t]*$", re.M)
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
# Inline code span: a backtick run closed by an equal-length run. Parked
# before the %%/^ subs so span contents read as code, not markup.
_CODE_SPAN = re.compile(r"(`+)[\s\S]*?\1")
_PARKED = re.compile(r"\x00(\d+)\x00")
_CALLOUT_HEAD = re.compile(r"\[!(\w+)\][+-]?[ \t]*(.*)")  # first line of a callout quote
_TASK_HEAD = re.compile(r"^\[([ xX])\][ \t]+")  # first inline text of a task list item
_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.S)


def _clean_name(ref: str) -> str:
    """Display name: basename without folders or `.md` (`a/b.md` -> `b`)."""
    return ref.rsplit("/", 1)[-1].removesuffix(".md")


def _anchor(path: str, display: str) -> str:
    return (
        f'<a class="note-link" data-path="{_html.escape(path, quote=True)}">'
        f"{_html.escape(display)}</a>"
    )


def _embed_img(target: str, alias: str) -> str:
    """<img> for a `![[file.png]]` embed; a numeric alias is Obsidian's width.
    ponytail: target is taken vault-root-relative — no shortest-name resolution
    for attachments; index attachment names if that ever bites."""
    src = "/asset?path=" + _quote(target)
    width = f' width="{alias}"' if alias.isdigit() else ""
    stem = target.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    alt = stem if alias.isdigit() or not alias else alias
    return f'<img src="{_html.escape(src, quote=True)}" alt="{_html.escape(alt, quote=True)}"{width}>'


_RAW_IMG_SRC = re.compile(r"""(<img\b[^>]*?\bsrc\s*=\s*)(["'])(.*?)\2""", re.IGNORECASE | re.DOTALL)


def _rewrite_raw_img_src(html: str) -> str:
    """Route a raw-HTML ``<img src="assets/x.png">`` through /asset.

    markdown-it's commonmark preset passes raw HTML through, and the image
    rewrite in _render only reaches markdown-native ``![alt](path)`` tokens. A
    note written for GitHub uses the HTML form instead, so its src arrived
    intact and the browser resolved it against the page origin: the drawer
    404'd on every such image and showed the alt text in a box. Absolute,
    external, anchor and data: URLs pass untouched, same rule as the token
    path."""
    def sub(m: "re.Match[str]") -> str:
        src = m.group(3)
        if not src or src.startswith(("http://", "https://", "data:", "/", "#")):
            return m.group(0)
        return f"{m.group(1)}{m.group(2)}/asset?path={_quote(src)}{m.group(2)}"

    return _RAW_IMG_SRC.sub(sub, html)


# --- raw-HTML allowlist ------------------------------------------------------
# markdown-it's commonmark preset passes raw HTML through verbatim, and a note
# body is untrusted input: it can arrive from a nucleated document written by
# anyone. app.js writes this render with innerHTML, so whatever is emitted here
# executes on the GUI's own origin, which can reach /chat, /note and /settings.
#
# Raw HTML stays SUPPORTED — <br>, <details>, <img> are ordinary Obsidian markup
# and the app already post-processes it (_rewrite_raw_img_src) — but it may not
# execute. The fragment is re-parsed and re-serialized from these allowlists, so
# an unknown tag or attribute is dropped by construction; nothing is ever matched
# against a list of known payloads.
#
# The families are kept whole on purpose: a note that writes `<b>` rather than
# `**b**` (every README authored for GitHub does) must still read as bold, and a
# raw table that keeps `<td>` but loses its `<caption>` is a worse render than no
# allowlist at all. Every tag below is inert markup — none can carry a URL or a
# behavior — so admitting the whole family costs nothing the allowlist protects.
_ALLOWED_TAGS = frozenset({
    "a", "abbr", "b", "blockquote", "br", "caption", "code", "dd", "details",
    "div", "dl", "dt", "em", "figcaption", "figure", "h1", "h2", "h3", "h4",
    "h5", "h6", "hr", "i", "img", "input", "kbd", "li", "mark", "ol", "p",
    "pre", "s", "small", "span", "strong", "sub", "summary", "sup", "table",
    "tbody", "td", "tfoot", "th", "thead", "tr", "u", "ul",
})
# Attributes are allowlisted per tag on top of these two. No `on*` handler and no
# `style` is listed anywhere below, and that omission IS the rule: an event
# handler is a script, and a style declaration cannot be vetted from here.
_ATTRS_ANY = frozenset({"class", "title"})
_ATTRS_BY_TAG = {
    "a": frozenset({"href"}),
    "img": frozenset({"src", "alt", "width", "height"}),
    "input": frozenset({"type", "checked", "disabled"}),
    "details": frozenset({"open"}),
    "ol": frozenset({"start"}),
    "td": frozenset({"colspan", "rowspan"}),
    "th": frozenset({"colspan", "rowspan", "scope"}),
}
_URL_ATTRS = frozenset({"href", "src"})
_VOID_TAGS = frozenset({"br", "hr", "img", "input"})
# script/style bodies are raw text, not prose: HTMLParser hands them to
# handle_data, so they are dropped with the tag instead of landing in the page.
_RAW_TEXT_TAGS = frozenset({"script", "style"})
_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")


def _safe_url(value: str, tag: str) -> str | None:
    """The URL as written, or None when the attribute must be dropped.

    Relative URLs pass (that is how a note names a vault attachment, and how the
    app's own /asset route is reached); http/https pass; a data: URL passes only
    as an image source, which is how a note carries an inline picture. Every
    other scheme is refused — `javascript:` is the whole reason this exists, and
    the check is on the value with control characters and spaces removed,
    because `java\tscript:` is still that scheme to a browser.
    """
    probe = "".join(ch for ch in value if ord(ch) > 0x20)
    if not _URL_SCHEME.match(probe):
        return value
    low = probe.lower()
    if low.startswith(("http://", "https://")):
        return value
    if tag == "img" and low.startswith("data:image/"):
        return value
    return None


class _HtmlAllowlist(HTMLParser):
    """Re-serialize a raw-HTML fragment, keeping only allowlisted markup.

    Anything not allowlisted is escaped rather than emitted, so it shows as text
    instead of silently vanishing (and cannot run either way).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        # Body of an open raw-text element, held rather than dropped: see close().
        self._raw_text: list[str] | None = None

    def close(self) -> None:
        super().close()
        # A `<script>` that never closes makes markdown-it hand the WHOLE rest of
        # the note over as one html_block (its raw-text rule runs to the closing
        # tag or to EOF), so everything after it is prose, not a script body, and
        # dropping it silently emptied the note from that line down. Only a body
        # that was actually terminated by its own tag is a script.
        if self._raw_text:
            self.out.append(_html.escape("".join(self._raw_text), quote=False))
        self._raw_text = None

    def result(self) -> str:
        return "".join(self.out)

    def _emit_tag(self, tag: str, attrs) -> bool:
        if tag not in _ALLOWED_TAGS:
            return False
        # `or ""`: HTMLParser reports a valueless attribute (`<input type>`) as
        # None, and the drawer promises never to 500 on a note body.
        if tag == "input" and (dict(attrs).get("type") or "").strip().lower() != "checkbox":
            return False  # a task checkbox is the only <input> a note renders
        allowed = _ATTRS_ANY | _ATTRS_BY_TAG.get(tag, frozenset())
        parts = [tag]
        for name, value in attrs:
            name = name.lower()
            if name not in allowed:
                continue
            if value is None:
                parts.append(name)
                continue
            if name in _URL_ATTRS:
                value = _safe_url(value, tag)
                if value is None:
                    continue
            parts.append(f'{name}="{_html.escape(value, quote=True)}"')
        self.out.append("<" + " ".join(parts) + ">")
        return True

    def handle_starttag(self, tag, attrs):
        if tag in _RAW_TEXT_TAGS and self._raw_text is None:
            self._raw_text = []
        if not self._emit_tag(tag, attrs):
            self.out.append(_html.escape(self.get_starttag_text() or f"<{tag}>"))

    def handle_startendtag(self, tag, attrs):
        # `<br/>`: the start tag alone, never a stray `</br>`.
        if not self._emit_tag(tag, attrs):
            self.out.append(_html.escape(self.get_starttag_text() or f"<{tag}/>"))

    def handle_endtag(self, tag):
        if tag in _RAW_TEXT_TAGS:
            self._raw_text = None  # closed by its own tag: that body was a script
        if tag in _ALLOWED_TAGS:
            if tag not in _VOID_TAGS:
                self.out.append(f"</{tag}>")
            return
        self.out.append(_html.escape(f"</{tag}>"))

    def handle_data(self, data):
        if self._raw_text is not None:
            self._raw_text.append(data)
            return
        self.out.append(_html.escape(data, quote=False))

    # Comments, doctypes, CDATA sections and processing instructions carry no
    # prose and are how markup gets smuggled past a naive parser: dropped.
    def handle_comment(self, data):
        pass

    def handle_decl(self, decl):
        pass

    def unknown_decl(self, data):
        pass

    def handle_pi(self, data):
        pass


def _sanitize_html(fragment: str) -> str:
    parser = _HtmlAllowlist()
    parser.feed(fragment)
    parser.close()
    return parser.result()


def _sanitize_raw_tokens(tokens) -> None:
    """Allowlist the note's own raw HTML in the token stream, in place.

    Runs on the parser output and nowhere later: every html_block/html_inline
    added after this point is built by this module (mermaid, MathML, callout
    titles, task checkboxes, the linkified prose) and re-sanitizing it would
    strip the MathML the math renderer just produced.
    """
    for tok in tokens:
        if tok.type in ("html_block", "html_inline"):
            tok.content = _sanitize_html(tok.content)
        elif tok.type == "inline" and tok.children:
            _sanitize_raw_tokens(tok.children)


def _linkify_text(text: str, resolve) -> str:
    """Turn resolvable note refs in one plain-text run into `.note-link` anchors.

    Two layers: wikilinks first (explicit `[[...]]` delimiters), then bare
    path-like tokens in the surviving prose. Unresolved wikilinks render like
    resolved ones but tagged `.broken` (no data-path — the click is a no-op);
    unresolved bare paths stay verbatim. `resolve=None` means plain escape.
    Returns an HTML fragment (safe parts escaped).
    """
    if resolve is None:
        return _html.escape(text)

    def link_paths(prose: str) -> str:
        out, pos = [], 0
        for m in _PATHLIKE.finditer(prose):
            out.append(_html.escape(prose[pos:m.start()]))
            tok = m.group(0)
            core = tok.rstrip(_TRAIL)
            tail = tok[len(core):]
            hit = resolve(core)
            if hit:
                out.append(_anchor(hit, _clean_name(core)) + _html.escape(tail))
            else:
                out.append(_html.escape(tok))
            pos = m.end()
        out.append(_html.escape(prose[pos:]))
        return "".join(out)

    out, pos = [], 0
    for m in _WIKILINK.finditer(text):
        out.append(link_paths(text[pos:m.start()]))
        bang, inner = m.group(1), m.group(2)
        target, _, alias = inner.partition("|")
        target, alias = target.strip(), alias.strip()
        # Obsidian subpath (#center alignment hint, #heading anchor): irrelevant
        # to serving a raster attachment, and it would defeat the ext check.
        target = target.split("#", 1)[0].strip()
        if bang and "." + target.rsplit(".", 1)[-1].lower() in _ASSET_EXTS:
            out.append(_embed_img(target, alias))
        else:
            hit = resolve(target)
            display = alias or _clean_name(target)
            if hit:
                out.append(_anchor(hit, display))
            else:
                out.append(f'<a class="note-link broken">{_html.escape(display)}</a>')
        pos = m.end()
    out.append(link_paths(text[pos:]))
    return "".join(out)


def _inline_ofm(text: str, resolve) -> str:
    """OFM inline sugar over one plain-text run: ==highlight== -> <mark>,
    #tag -> chip. Prose between matches still goes through note-ref linking."""
    out, pos = [], 0
    for m in _MARK_OR_TAG.finditer(text):
        out.append(_linkify_text(text[pos:m.start()], resolve))
        if m.group(1) is not None:
            out.append(f"<mark>{_linkify_text(m.group(1), resolve)}</mark>")
        else:
            out.append(f'<span class="tag">#{_html.escape(m.group(2))}</span>')
        pos = m.end()
    out.append(_linkify_text(text[pos:], resolve))
    return "".join(out)


def _ofm_blocks(tokens) -> None:
    """OFM block sugar, rewriting the token stream in place: ```mermaid fences
    become client-rendered <pre class="mermaid">, `> [!kind] title` blockquotes
    become callouts, and `- [ ]` list items become checkbox tasks."""
    from markdown_it.token import Token

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "fence" and tok.info.strip() == "mermaid":
            raw = Token("html_block", "", 0)
            raw.content = f'<pre class="mermaid">{_html.escape(tok.content)}</pre>\n'
            tokens[i] = raw
        elif tok.type == "math_block":
            raw = Token("html_block", "", 0)
            raw.content = f'<div class="math">{_mathml(tok.content, display=True)}</div>\n'
            tokens[i] = raw
        elif tok.type == "blockquote_open":
            j = next((k for k in range(i + 1, len(tokens)) if tokens[k].type == "inline"), None)
            kids = tokens[j].children if j is not None else None
            first = kids[0] if kids else None
            m = _CALLOUT_HEAD.match(first.content) if first is not None and first.type == "text" else None
            if m:
                kind = m.group(1).lower()
                tok.attrJoin("class", f"callout callout-{kind}")
                rest = kids[1:] if kids else []
                if rest and rest[0].type == "softbreak":
                    rest = rest[1:]
                tokens[j].children = rest
                head = Token("html_block", "", 0)
                title = m.group(2).strip() or kind
                head.content = f'<p class="callout-title">{_html.escape(title)}</p>\n'
                tokens.insert(i + 1, head)
                i += 1  # skip the injected title
        elif (
            tok.type == "list_item_open"
            and i + 2 < len(tokens)
            and tokens[i + 1].type == "paragraph_open"
            and tokens[i + 2].type == "inline"
            and tokens[i + 2].children
        ):
            first = tokens[i + 2].children[0]
            m = _TASK_HEAD.match(first.content) if first.type == "text" else None
            if m:
                tok.attrJoin("class", "task")
                first.content = first.content[m.end():]
                box = Token("html_inline", "", 0)
                checked = " checked" if m.group(1) in "xX" else ""
                box.content = f'<input type="checkbox" disabled{checked}> '
                tokens[i + 2].children.insert(0, box)
        i += 1


# A math block is injected AFTER the raw-HTML allowlist has run (the allowlist
# would strip the MathML), so what the converter emits is the last word — and
# `\text{…}` is copied into <mtext> character for character. <mtext> is a MathML
# text integration point: the browser switches back to HTML parsing rules inside
# it, so `$$\text{<img/src=x/onerror=…>}$$` lands a live element with a live
# handler in the page. These are the tags and attributes latex2mathml itself can
# emit; anything else in the tree came from the note. `href` and `style` are
# deliberately absent — `\href{…}` and `\style{…}` put a note-authored URL and a
# note-authored declaration on the element, and neither can be vetted from here.
_MATHML_TAGS = frozenset({
    "math", "menclose", "merror", "mfrac", "mi", "mmultiscripts", "mn", "mo",
    "mover", "mpadded", "mphantom", "mprescripts", "mroot", "mrow", "ms",
    "mspace", "msqrt", "mstyle", "msub", "msubsup", "msup", "mtable", "mtd",
    "mtext", "mtr", "munder", "munderover", "none",
})
_MATHML_ATTRS = frozenset({
    "accent", "accentunder", "border-color", "class", "close", "columnalign", "columnlines",
    "columnspacing", "columnspan", "depth", "display", "displaystyle", "fence",
    "form", "height", "largeop", "linebreak", "linethickness", "lspace",
    "mathbackground", "mathcolor", "mathsize", "mathvariant", "maxsize",
    "minsize", "movablelimits", "notation", "open", "rowalign", "rowlines",
    "rowspacing", "rowspan", "rspace", "scriptlevel", "separator", "separators",
    "stretchy", "symmetric", "voffset", "width", "xmlns",
})


class _MathMLProbe(HTMLParser):
    """Flags any markup in a converted formula that latex2mathml cannot emit.

    A probe, not a rewriter: the converted string is returned untouched or not at
    all, so a formula the browser accepts today keeps rendering byte for byte.
    Comments, declarations and processing instructions count as foreign too —
    the converter emits none, and an unterminated `<!--` swallows the rest of the
    note in the browser while HTMLParser reads it as nothing.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.foreign = False

    def handle_starttag(self, tag, attrs):
        if tag not in _MATHML_TAGS or any(n.lower() not in _MATHML_ATTRS for n, _ in attrs):
            self.foreign = True

    def handle_endtag(self, tag):
        if tag not in _MATHML_TAGS:
            self.foreign = True

    def handle_comment(self, data):
        self.foreign = True

    def handle_decl(self, decl):
        self.foreign = True

    def unknown_decl(self, data):
        self.foreign = True

    def handle_pi(self, data):
        self.foreign = True


def _mathml(tex: str, display: bool) -> str:
    """LaTeX -> MathML, rendered natively by the browser (no client JS/fonts).
    A failed conversion degrades to the escaped source in a code span, and so
    does one carrying markup the converter cannot have produced itself."""
    try:
        from latex2mathml.converter import convert

        out = convert(tex, display="block" if display else "inline")
        probe = _MathMLProbe()
        probe.feed(out)
        probe.close()
        if not probe.foreign:
            return out
    except Exception:
        pass
    fence = "$$" if display else "$"
    return f'<code class="math-err">{_html.escape(fence + tex + fence)}</code>'


def _highlight(code: str, lang: str, _attrs: str) -> str:
    """Pygments fence highlighting; empty string falls back to a plain fence.
    Token colors live in app.css, mapped onto the site palette."""
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import get_lexer_by_name

        lexer = get_lexer_by_name(lang)
    except Exception:  # no/unknown language — markdown-it escapes it plain
        return ""
    return highlight(code, lexer, HtmlFormatter(nowrap=True))


# Inline spans are parked as \x00 sentinels for the length of the sub, the
# same trick mdLite plays client-side; pathological nested backtick runs can
# still fool the span regex, which costs a stripped %% inside them, cosmetic.
def _strip_ofm_meta(text: str) -> str:
    """Strip %%comments%% and trailing ^block-ids, sparing fenced code and
    inline `code spans` where %% and ^ are code (a lone %% in a fence would
    otherwise pair with a prose %% and swallow everything between)."""
    pieces: list[str] = []
    run: list[str] = []
    fence: tuple[str, int] | None = None  # (marker char, marker length)

    def _flush() -> None:
        if run:
            spans: list[str] = []

            def _park(m: "re.Match[str]") -> str:
                spans.append(m.group(0))
                return f"\x00{len(spans) - 1}\x00"

            parked = _CODE_SPAN.sub(_park, "\n".join(run))
            cleaned = _BLOCK_ID.sub("", _COMMENT.sub("", parked))
            pieces.append(
                _PARKED.sub(lambda m: spans[int(m.group(1))], cleaned))
            run.clear()

    for line in text.split("\n"):
        m = _FENCE.match(line)
        if fence is None:
            if m:
                _flush()
                fence = (m.group(1)[0], len(m.group(1)))
                pieces.append(line)
            else:
                run.append(line)
        else:
            pieces.append(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1]:
                fence = None
    _flush()
    return "\n".join(pieces)


def _linkify(text: str, resolve=None) -> str:
    """Render markdown (+ OFM sugar) to HTML, linkifying resolvable note refs
    when `resolve` is given. Works on the markdown-it token stream, so
    `code_inline`/`fence` are separate token types and code is never linkified
    or tag-ified by construction."""
    from markdown_it import MarkdownIt
    from markdown_it.token import Token
    from mdit_py_plugins.dollarmath import dollarmath_plugin

    text = _strip_ofm_meta(text or "")
    md = (
        MarkdownIt(options_update={"highlight": _highlight})
        .enable("table")
        .enable("strikethrough")
        .enable("linkify")
    )
    md.options["linkify"] = True
    # fuzzy_link off or `nota.md` in prose resolves to the Moldovan ccTLD and
    # renders as a link to http://nota.md; fuzzy_email off keeps the scope at
    # "a URL is clickable", not "prose opens a mail client".
    if md.linkify is not None:  # only present with the [linkify] extra
        md.linkify.set({"fuzzy_link": False, "fuzzy_email": False})
    # allow_space=False keeps prose prices ("$5 and $10") out of math
    md.use(dollarmath_plugin, allow_space=False, allow_digits=False)
    tokens = md.parse(text)
    _sanitize_raw_tokens(tokens)  # the note's raw HTML; must precede the injected kind
    _ofm_blocks(tokens)
    for tok in tokens:
        if tok.type == "html_block":
            tok.content = _rewrite_raw_img_src(tok.content)
            continue
        if tok.type != "inline" or not tok.children:
            continue
        new = []
        # Note refs are suppressed inside an anchor. `_PATHLIKE` matches any token
        # with a slash, so the display text of a link (an autolinked URL is its own
        # text) hit the basename fallback in _resolve_in and came back as a note:
        # `https://en.wikipedia.org/wiki/chemistry` rendered as a `.note-link` to
        # `chemistry.md`, nested inside the <a href> the browser then unnests.
        # Whoever writes `[...](url)` has already said where the link points.
        depth = 0
        for child in tok.children:
            if child.type == "link_open":
                depth += 1
            elif child.type == "link_close":
                depth = max(0, depth - 1)
            if child.type == "html_inline":
                child.content = _rewrite_raw_img_src(child.content)
                new.append(child)
                continue
            if child.type == "image":
                # vault-relative image: route through /asset (absolute/external
                # and data: URLs pass untouched)
                src = str(child.attrGet("src") or "")
                if src and not src.startswith(("http://", "https://", "data:", "/")):
                    child.attrSet("src", "/asset?path=" + _quote(src))
                new.append(child)
                continue
            if child.type == "math_inline":
                raw = Token("html_inline", "", 0)
                raw.content = _mathml(child.content, display=False)
                new.append(raw)
                continue
            if child.type != "text":
                new.append(child)
                continue
            frag = _inline_ofm(child.content, None if depth else resolve)
            raw = Token("html_inline", "", 0)
            raw.content = frag
            new.append(raw)
        tok.children = new
    return md.renderer.render(tokens, md.options, {})


def _split_frontmatter(text: str) -> tuple[dict | None, str]:
    """Split a leading YAML frontmatter block. Returns (props, body); props is
    None unless the block parses to a mapping."""
    import yaml

    m = _FRONTMATTER.match(text or "")
    if not m:
        return None, text
    try:
        props = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None, text
    if not isinstance(props, dict):
        return None, text
    return props, text[m.end():]


def _render_frontmatter(props: dict) -> str:
    """Properties box for the note drawer: native <details>, one row per key,
    list values as individual chips."""
    rows = []
    for key, val in props.items():
        vals = val if isinstance(val, (list, tuple)) else [val]
        chips = "".join(
            f'<span class="fm-val">{_html.escape("" if v is None else str(v))}</span>'
            for v in vals
        )
        rows.append(
            f'<div class="fm-row"><span class="fm-key">{_html.escape(str(key))}</span>{chips}</div>'
        )
    return '<details class="fm" open><summary>properties</summary>' + "".join(rows) + "</details>"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _begin_turn() -> bool:
    """Claim the single-turn slot. Sync with no `await` between the test and the
    set, so two racing POSTs can't both pass. Returns False if one's in flight."""
    global _busy
    if _busy:
        return False
    _busy = True
    return True


def _end_turn() -> None:
    """Release the turn slot. Idempotent (a completed turn and its worker's
    done-callback may both call it)."""
    global _busy, current_cancel, current_task
    _busy = False
    current_cancel = None
    current_task = None


def _sweep_if_orphaned() -> None:
    """Free a gate claimed for a turn whose `run_turn` never ran — the client
    dropped between POST and the SSE body's first `__anext__`, so nothing else
    releases it. Runs after the response closes; a no-op once a worker exists."""
    if _busy and (current_task is None or current_task.done()):
        _end_turn()


async def run_turn(text: str) -> AsyncIterator[dict]:
    """One agent turn as a stream of transport-neutral wire dicts.

    Yields `event_to_json(...)` dicts as the agent streams, then exactly one
    terminal dict: `{"type": "done", ...}` or `{"type": "error", ...}`. Owns the
    whole turn lifecycle (session append, sync→async queue bridge, cancel token,
    context compaction, save). Both `--gui` (SSE) and `connect` (WS) consume this
    — the framing is the transport's job, not this core's.

    Gate lifecycle: the slot is freed on normal end/error at once; on abandonment
    (the consumer stops iterating — a dropped SSE/WS client) the worker keeps
    running, so we signal cancel and defer the release to the worker's exit, so
    no second turn overlaps a zombie still mutating `messages`.
    """
    from silica.cli import _compact_context, _expand_web_turn, _update_context_tokens

    global _busy, current_cancel, current_task, _collapsed
    if not _busy:  # direct callers (tests, future WS) that didn't pre-claim
        _busy = True
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    current_cancel = threading.Event()  # module-level so /stop can see it
    task: asyncio.Task | None = None

    def cb(ev):  # runs in the agent/LLM worker thread
        data = event_to_json(ev)
        if data is not None:
            loop.call_soon_threadsafe(q.put_nowait, data)

    # InjectorFSM phase transitions arrive on BUS, not through the agent
    # callback: they are emitted from inside the tool, several layers below the
    # loop that owns `cb`. Subscribed per turn and dropped in the finally, so a
    # turn never receives another turn's phases and nothing accumulates across
    # turns. Publishing happens on the FSM's thread, hence the same
    # call_soon_threadsafe hop `cb` uses.
    def on_phase(ev):
        try:
            cb(ev)
        except RuntimeError:
            pass  # loop already closed: the turn is gone, the event is moot

    from silica.agent.bus import BUS
    BUS.subscribe("work/phase", on_phase)

    try:
        # A slash command follows the REPL's dispatch order (silica/cli.py): the
        # direct handler first — synchronous, no LLM round-trip — then the
        # workflow expansion. Asking the same handler the REPL asks is the point:
        # the hand-kept list of "web commands" that used to gate this drifted, and
        # /lexical /wiki /graph /map /find /vault fell through it into an error.
        agent_msg = text
        # /web comes first: it is neither direct nor a workflow expansion but an
        # agent turn with web-only tools. A usage error raises ValueError, which
        # the except below turns into the single error event.
        web = _expand_web_turn(text, messages) if text.startswith("/") else None
        if web is not None:
            agent_msg = web[1]
        elif text.startswith("/"):
            from silica.cli import _expand_workflow_shortcut, _handle_direct_shortcut
            from silica.ui.console import CONSOLE

            def _run_slash():
                # Both dispatchers print their result to CONSOLE and both can do
                # real work, so both run under the capture and off the loop
                # thread. The expansion is not a pure string builder: /fetch,
                # /web-search and /convert do the whole job inside it and return
                # "" to say the REPL has nothing left for the agent. Reading
                # that "" as "not available" was reporting failure for work that
                # had already written notes to disk, with the success line going
                # to the server's stdout where no browser user can see it.
                with CONSOLE.capture() as capture:
                    handled = _handle_direct_shortcut(text, messages)
                    expanded = None if handled else _expand_workflow_shortcut(text)
                return handled or expanded == "", expanded, capture.get()

            handled, expanded, captured_out = await asyncio.to_thread(_run_slash)

            if handled:
                # Appended only once the verdict is in: a False falls through to
                # the agent below, which appends the expanded turn itself.
                messages.append({"role": "user", "content": text, "origin": "cli"})
                _narrate_user_turn(messages[-1])
                out = captured_out.strip()
                answer = f"```text\n{out}\n```" if out else "```text\n(done)\n```"
                messages.append({"role": "assistant", "content": answer})
                from silica.agent import narration as _narr
                _narr.NARRATOR.turn(messages[-1])

                # Yield a fake agent turn with the direct result
                yield {
                    "type": "done",
                    "answer": answer,
                    "html": _linkify(answer, note_resolver()),
                    **_ctx_meter(),
                }
                return

            if expanded is None:
                yield {"type": "error", "error": f"'{text}' not available in this session"}
                return
            agent_msg = expanded

        msg = {"role": "user", "content": agent_msg}
        if text.startswith("/"):
            msg["origin"] = "cli"
        messages.append(msg)
        _narrate_user_turn(msg)

        # Both wrappers forward every event to `cb` untouched: WebTurn records the
        # trace the citations are built from, RecallWatch counts recall misses for
        # the thin-coverage hint.
        watch = WebTurn(web[0], cb) if web else RecallWatch(cb)

        sentinel = object()
        task = asyncio.create_task(
            asyncio.to_thread(
                run_agent, messages, CONFIG.model, watch,
                cancel_token=current_cancel,
                # Same chat_tools cut as the CLI REPL (the tool block is the
                # biggest per-iteration cost); interactive keeps the GUI's live
                # stream and keeps the turn off the worker-slot cap. Slash
                # directives carry origin="cli", so excluded tools a command
                # names are summoned back exactly as in the terminal.
                constraints=(
                    web_turn_constraints() if web
                    else AgentConstraints(tools=chat_tools(messages), interactive=True)
                ),
            )
        )
        current_task = task
        task.add_done_callback(lambda t: q.put_nowait(sentinel))

        while True:
            item = await q.get()
            if item is sentinel:
                break
            yield item

        answer = await task  # re-raises if run_agent failed
        # isinstance, not `if web:` — the two are the same condition (watch is
        # built from it one screen up) but only this one says which half of the
        # union carries `attribute`.
        if isinstance(watch, WebTurn):
            # Before _linkify and before the compaction sweep: the Sources block
            # belongs to what the user sees AND to what the history carries.
            answer = watch.attribute(answer, messages)
        elif watch.web_answer:
            from silica.sources.web_research import relay_sources

            answer = relay_sources(answer, messages)
        # Final-assistant turn beat, post-attribution (see loop.py).
        if messages and messages[-1].get("role") == "assistant":
            from silica.agent import narration as _narr
            _narr.NARRATOR.turn(messages[-1])
        _update_context_tokens(messages)
        _collapsed = _compact_context(messages, _collapsed)
        # note_resolver reads the DRIVER graph — with the ws backend installed
        # (silica connect) a driver call on the loop thread deadlocks (`_rpc`
        # blocks the very loop that must send the frame), so render off-loop.
        html = await asyncio.to_thread(lambda: _linkify(answer, note_resolver()))
        done = {
            "type": "done",
            "answer": answer,
            "html": html,
            **_ctx_meter(),
        }
        if isinstance(watch, RecallWatch) and watch.thin:
            done["hint"] = THIN_COVERAGE_HINT  # muted line under the answer
        yield done
    except Exception as exc:  # never leave the UI stuck on the spinner
        logger.exception("web turn failed")
        yield {"type": "error", "error": str(exc)}
    finally:
        BUS.unsubscribe("work/phase", on_phase)
        _prewarm_seed()  # the turn may have written notes — refresh the new-chat seed
        if task is not None and not task.done():
            current_cancel.set()  # abandonment: stop the zombie...
            task.add_done_callback(lambda t: _end_turn())  # ...free the gate when it exits
        else:
            _end_turn()  # normal / error / early-return: free now


def _turn_response(text: str) -> StreamingResponse:
    """One agent turn as an SSE stream. Caller must claim the slot via
    `_begin_turn()` first; `_sweep_if_orphaned` frees it if the body never runs."""

    async def gen():
        async for item in run_turn(text):
            yield _sse(item)

    return StreamingResponse(
        gen(), media_type="text/event-stream", background=BackgroundTask(_sweep_if_orphaned)
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Host the Obsidian bridge for the GUI session — the plugin dials in and
    the driver hot-swaps to ws (falls back on drop). No-op without the
    [connect] extra or when the vault has no .obsidian/."""
    from silica.ui.connect import maybe_start_bridge

    bridge = None
    try:
        bridge = await maybe_start_bridge()
    except Exception:
        logger.exception("bridge auto-start failed")  # the GUI must not die for it
    yield
    if bridge is not None:
        await bridge.stop()


app = FastAPI(lifespan=_lifespan)

# NOT GZipMiddleware. It would also wrap /chat, which is text/event-stream, and
# Starlette's gzip responder writes each chunk into a zlib compressor that
# buffers until it has enough to emit — so the SSE frames that make a turn
# stream would be withheld and the transcript would arrive in bursts. /graph
# compresses itself instead, at the one route where the payload is large enough
# to matter (see graph()).

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _authority(value: str) -> tuple[str, str]:
    """`host[:port]` split into (host, port), with the loopback spellings folded
    into one host — the browser repeats whatever the user typed, and the server
    was reached on the loopback interface either way."""
    raw = value.strip().lower()
    host, _, port = raw.rpartition(":")
    if not host or "]" in port:  # no port at all, or a bare IPv6 literal
        host, port = raw, ""
    return ("loopback" if host in _LOOPBACK_HOSTS else host), port


def _require_same_origin(request: Request) -> None:
    """Refuse a state-changing request another origin made on the user's behalf.

    The GUI has no auth (localhost, one user), so it answers with the browser's
    ambient authority, and a multipart POST is CORS-safelisted: it crosses
    origins with no preflight. Any page the user happened to visit could
    therefore fetch() an upload into the vault Inbox and run a whole agent turn
    with the write tools. A browser always sends Origin on a cross-origin POST,
    so checking it closes that while a client that sends no Origin at all (curl,
    the tests) keeps working. Every state-changing route must carry this.
    """
    site = (request.headers.get("sec-fetch-site") or "").lower()
    if site and site not in ("same-origin", "none"):
        raise HTTPException(403, "cross-origin request refused")
    origin = request.headers.get("origin")
    if origin is None:
        return
    parsed = _urlsplit(origin)
    if parsed.scheme not in ("http", "https") or _authority(parsed.netloc) != _authority(
        request.headers.get("host", "")
    ):
        raise HTTPException(403, "cross-origin request refused")


_SAME_ORIGIN = [Depends(_require_same_origin)]


@app.post("/chat", dependencies=_SAME_ORIGIN)
async def chat(payload: dict):
    if not _begin_turn():
        raise HTTPException(status_code=409, detail="a turn is already in progress")
    return _turn_response(payload.get("text", ""))


@app.get("/supported_types")
def supported_types():
    """Extensions the nucleate picker offers — drives the `+` button's `accept`."""
    from silica.sources.registry import supported_nucleate_extensions

    return {"extensions": supported_nucleate_extensions()}


@app.get("/commands")
def list_commands():
    """Commands the web GUI's fuzzy picker offers — everything the chat turn can
    actually dispatch. `repl_only` ones are terminal-session business and would
    only answer 'not available in this session' if offered here."""
    from silica.ui.commands import COMMANDS

    return [
        {"name": c.name, "summary": c.summary, "usage": c.usage}
        for c in COMMANDS
        if not c.repl_only
    ]


# A scanned book is the honest ceiling for one drop; past that the file is not
# something the nucleate lane can chew, and the Inbox is a vault folder, not a
# dump. The body is streamed in chunks rather than read whole, so an oversized
# upload is refused after one chunk instead of after materialising it in memory.
_UPLOAD_MAX_BYTES = 256 * 1024 * 1024
_UPLOAD_MAX_FILES = 32
_UPLOAD_CHUNK = 1024 * 1024


async def _write_upload(f: UploadFile, dest: Path) -> None:
    """Stream one upload to `dest`, refusing it past the cap."""
    written = 0
    with dest.open("wb") as fh:
        while chunk := await f.read(_UPLOAD_CHUNK):
            written += len(chunk)
            if written > _UPLOAD_MAX_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)  # no half file left in the vault
                raise HTTPException(
                    413,
                    f"{dest.name} is over the {_UPLOAD_MAX_BYTES // (1024 * 1024)} MB upload limit",
                )
            fh.write(chunk)


async def _stage_uploads(files: list[UploadFile]) -> tuple[list[str], list[str]]:
    """Write uploads to Inbox and mechanically stage them, mirroring the inline
    half of `/nucleate` (silica/cli.py): PDFs convert to markdown, code/notebooks
    become skeleton stubs, prose stays as-is. Returns (ready, stubs): markdown
    notes ready for the injector/reading, and code stub note paths already
    written to the vault. The semantic step (nucleate? summarize?) is the agent's,
    driven by the user's message — see `_compose_nucleate_turn`.

    convert() shells out to mineru (can be minutes on a book) and stage() reads
    whole files, so both run in a worker thread: on the loop thread they blocked
    /stop for the whole conversion, leaving a visible Stop button that could not
    be served.
    """
    from silica.kernel.vault_manifest import get_active_manifest
    from silica.sources.convert import convert
    from silica.sources.registry import adapter_for, stage

    if len(files) > _UPLOAD_MAX_FILES:
        raise HTTPException(413, f"at most {_UPLOAD_MAX_FILES} files per drop")
    inbox = Path(CONFIG.vault_path or ".") / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    enabled = get_active_manifest().sources
    ready: list[str] = []
    stubs: list[str] = []
    for f in files:
        # `.name` is the containment: a client-chosen filename never contributes
        # a directory, and one that is nothing but directories ("..") gets the
        # fallback rather than resolving to the Inbox itself.
        dest = inbox / (Path(f.filename or "").name or "dropped")
        await _write_upload(f, dest)
        rel = f"Inbox/{dest.name}"
        adapter = adapter_for(rel, enabled=enabled)
        if adapter is None:  # no source claims it → converter fallback (PDF today)
            try:
                ready.extend(await asyncio.to_thread(convert, rel))
            except ValueError as exc:
                logger.warning("nucleate: skipped %s: %s", dest.name, exc)
            continue
        result = await asyncio.to_thread(stage, adapter, rel)
        if result["status"] == "distill":       # prose → injector re-reads it
            ready.append(rel)
        elif result["status"] == "ok":            # code/notebook → stub written
            stubs.append(result["note_path"])
        else:
            logger.warning("nucleate: %s: %s", dest.name, result.get("message", ""))
    return ready, stubs


def _compose_nucleate_turn(text: str, ready: list[str], stubs: list[str]) -> str:
    """The agent turn for a batch of attached files: the user's instruction plus
    a factual manifest of what got staged. Empty instruction defaults to nucleate."""
    lines: list[str] = []
    if ready:
        lines.append("Markdown staged in Inbox, ready to nucleate or read:")
        lines += [f"- {p}" for p in ready]
    if stubs:
        lines.append("Code skeleton stubs already staged in the vault:")
        lines += [f"- {p}" for p in stubs]
    manifest = "\n".join(lines) if lines else "(no files could be staged)"
    base = text.strip() or (
        "Nucleate the attached file(s) into an appropriate folder; "
        "ask me if the target is unclear."
    )
    return f"{base}\n\n---\nAttached files:\n{manifest}"


@app.post("/nucleate", dependencies=_SAME_ORIGIN)
async def nucleate(files: list[UploadFile] = File(...), text: str = Form("")):
    if not _begin_turn():
        raise HTTPException(status_code=409, detail="a turn is already in progress")
    try:
        ready, stubs = await _stage_uploads(files)
    except Exception:
        _end_turn()  # release the slot the staging never got to use
        raise
    return _turn_response(_compose_nucleate_turn(text, ready, stubs))


# How many rows of an uncapped list the metrics view receives. The report caps
# its own ranked lists at top_k; orphans and dangling are exhaustive, so they get
# cut here — and the true length always rides along in `totals`, so a cut list
# can never read as "this is all of them".
#
# 12, not 60: at 60 the orphans and dangling cards ran to 60 rows each and the
# dashboard became two long lists with charts above them (8.5k px on a 686-note
# vault). A card samples; GRAPH_REPORT.md is where the full list lives.
_METRICS_ROWS = 12

# Degree-distribution buckets. Doubling widths, not equal ones: a wikilink graph
# is heavy-tailed, so linear bins put ~everything in the first two and stretch a
# hundred empty bins under the hubs. The first three degrees stay their own bin
# because 0 (isolated), 1 (a leaf) and 2 mean different things about a note.
_DEGREE_BINS = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, None))


def _degree_histogram(degree_map: dict[str, int]) -> list[dict]:
    """Bucket every note's resolved-link degree. Trailing empty buckets are
    dropped so the axis ends where the vault does; interior empties stay, since
    a hole in the middle of the distribution is itself the reading."""
    out = []
    for lo, hi in _DEGREE_BINS:
        n = sum(1 for d in degree_map.values() if d >= lo and (hi is None or d <= hi))
        label = str(lo) if hi == lo else (f"{lo}+" if hi is None else f"{lo}-{hi}")
        out.append({"label": label, "count": n, "lo": lo})
    while len(out) > 1 and out[-1]["count"] == 0:
        out.pop()
    return out


# How many stored readings the trend band draws. The series gains a line only
# when a count actually moves, so 60 is years of daily use on a vault that keeps
# changing -- and past that a sparkline 80px wide is drawing more points than it
# has pixels, which is a smear, not a trend.
_HISTORY_POINTS = 60


def _metrics_history(vault: str | None) -> list[dict]:
    """The stored readings the trend band draws, oldest last.

    Best-effort like the delta beside it: a vault that is read-only or unbound
    still gets its metrics, and losing the trend costs a band, not the view.
    """
    if not vault:
        return []
    try:
        from silica.kernel.report.history import read_history

        return read_history(vault)[-_HISTORY_POINTS:]
    except Exception as exc:
        logger.debug("metrics: history unreadable (%s)", exc)
        return []


def _area_of(report) -> dict[str, int]:
    """Note id -> area id, indexed under both keyspaces the callers arrive in.

    `clusters[].members` are graph ids, which carry `.md`; the store keyspace
    drops it, and the V7 rows (sprawling) are computed there -- `/metrics`
    already re-adds the suffix on its way out for exactly this reason. Indexing
    both spellings once is cheaper than making every caller remember which one
    it holds, and a lookup that silently missed would not raise, it would report
    an area as clean.
    """
    out: dict[str, int] = {}
    for c in report.clusters:
        if c.size <= 1:
            continue  # a singleton is its own area; see inter_cluster's cut
        for m in c.members:
            out[m] = c.cluster_id
            if m.endswith(".md"):
                out[m[:-3]] = c.cluster_id
            else:
                out[m + ".md"] = c.cluster_id
    return out


def _signal_areas(report) -> dict[str, dict[str, int]]:
    """How many notes each worklist signal puts in each area.

    Counted over the report's FULL lists, never the twelve rows `/metrics` ships:
    a lens coloured from the slice would paint the top-12's areas hot and every
    other area clean, which is a confident statement about the wrong population.
    That is the same trap `dangling_hist` exists to avoid for the tail reading.

    `dangling` and `gaps` are absent by construction and not by omission: a
    dangling target has no note to place, and a gap is already a fact about a
    PAIR of areas, so tallying either into one area would invent a location.
    """
    area = _area_of(report)
    lists = {
        "lean": report.lean_notes,
        "orphans": report.orphans,
        "attention": [a.path for a in report.attention_candidates],
        "deficits": [d.path for d in report.integration_deficits],
        "contested": [c.path for c in report.contested],
        "drift": [d.note for d in report.source_drift],
        "sprawling": [x.path for x in report.sprawling],
    }
    out: dict[str, dict[str, int]] = {}
    for key, paths in lists.items():
        tally: dict[str, int] = {}
        for p in paths:
            cid = area.get(p)
            if cid is None:
                continue  # a note outside every multi-note area has no column
            tally[str(cid)] = tally.get(str(cid), 0) + 1
        if tally:
            out[key] = tally
    return out


def _lean_limit() -> int:
    """The character count under which a note is called lean.

    Read from the linter rather than restated here: the pane prints it beside
    every row, and a copy would keep saying 600 for a year after the limit moved.
    """
    try:
        from silica.kernel.link.ofm import LIMITS

        return int(LIMITS["lean_chars"])
    except Exception as exc:
        logger.debug("metrics: lean limit unreadable (%s)", exc)
        return 0  # the pane drops the comparison rather than inventing a bound


def _area_matrix(report) -> dict | None:
    """The area x area coupling grid, in the shape `/shape` already ships.

    Same payload shape on purpose: one client renderer draws both, so the two
    surfaces cannot drift into disagreeing about what a coupling is. The
    diagonal carries intra-area linked pairs, which is cohesion's numerator, and
    the renderer prints cohesion there instead.

    None rather than an empty grid below two areas: a 1x1 matrix is a cell
    saying nothing, and None is what lets the card say so in words.
    """
    if not report.inter_cluster:
        return None  # analytics did not run; see VaultReport.inter_cluster
    areas = [c for c in report.clusters if c.size > 1]
    if len(areas) < 2:
        return None
    areas.sort(key=lambda c: (-c.size, c.cluster_id))
    ids = [c.cluster_id for c in areas]
    short = lambda nid: (nid or "").rsplit("/", 1)[-1]  # noqa: E731
    cell = report.inter_cluster
    at = lambda a, b: cell.get(f"{min(a, b)}|{max(a, b)}", 0)  # noqa: E731
    return {
        "areas": [
            {"id": c.cluster_id, "label": short(c.hub) or f"#{c.cluster_id}",
             "path": c.hub or "", "size": c.size, "cohesion": c.cohesion,
             "intra": at(c.cluster_id, c.cluster_id)}
            for c in areas
        ],
        "matrix": [[at(a, b) for b in ids] for a in ids],
    }


def _shape_reading(adj: dict, deg: dict, areas: list[dict], label_of: dict, stops: int = 24) -> dict[str, Any]:
    """A reading order over the vault, derived rather than authored.

    Areas biggest first, and inside one the hub then its best-connected
    neighbours, breadth-first. Not a ranking of what matters: it is the order
    that keeps each next note adjacent to something already read, which is the
    only property a reading path can actually promise from link structure.

    Capped, because a path with 795 stops is the file tree with extra words.
    """
    out: list[dict] = []
    seen: set[str] = set()
    # Every area's hub, reserved. Hubs link to each other, so without this the
    # second area's hub gets picked up as a neighbour of the first, and then its
    # own section is skipped as already-seen — silently dropping the area, and
    # the biggest ones first, since those are exactly the well-connected hubs.
    hubs = {a["path"] for a in areas if a["path"]}
    covered = 0
    for a in areas:
        hub = a["path"]
        if not hub:
            continue
        if len(out) + 1 > stops:
            break
        covered += 1
        seen.add(hub)
        out.append({"path": hub, "label": label_of.get(hub, hub), "area": a["label"],
                    "why": f"hub of {a['label']} — {a['size']} notes, the densest point of the area"})
        # Two neighbours per area: enough to show what the hub opens onto,
        # few enough that the biggest area cannot eat the whole path.
        near = sorted((n for n in adj.get(hub, ()) if n not in seen and n not in hubs),
                      key=lambda n: -deg.get(n, 0))[:2]
        for n in near:
            if len(out) >= stops:
                break
            seen.add(n)
            out.append({"path": n, "label": label_of.get(n, n), "area": a["label"],
                        "why": f"linked from the hub, {deg.get(n, 0)} links of its own"})
    return {"stops": out, "areas_covered": covered, "areas_total": len(areas)}


@app.get("/vault_version")
def vault_version():
    """A digest the explore surfaces poll to learn the vault moved under them.

    Deliberately NOT the BUS/SSE the turn stream uses: that bus is in-process,
    so it carries this browser's own agent and nothing else — and the writes
    this exists for are the ones from OUTSIDE it (Obsidian, `silica nucleate`
    in a terminal, a second window). A poll is the only signal that sees them
    without a resident watcher, which the charter rules out anyway.
    """
    from silica.kernel.recall.sync import vault_version as _version

    return {"version": _version()}


@app.get("/shape")
def shape():
    """Three read-only views over ONE graph build: containment, area coupling,
    and a derived reading order.

    They share an endpoint because they share the expensive part — build_graph_data
    plus community detection — and splitting them into three routes would pay it
    three times for surfaces a reader flips between.
    """
    from silica.kernel.recall.graph_export import build_graph_data, detect_communities

    try:
        nodes, edges = build_graph_data(folder="")
        communities = detect_communities(nodes, edges)
    except Exception as exc:
        logger.warning("shape: graph build failed (%s)", exc)
        return {"error": str(exc)}

    real = [n for n in nodes if n.get("type") != "ghost"]
    group_of = {n["id"]: n.get("group", -1) for n in real}
    label_of = {n["id"]: n.get("label") or n["id"].rsplit("/", 1)[-1] for n in real}

    adj: dict[str, set[str]] = {}
    deg: dict[str, int] = {}
    intra: dict[int, int] = {}
    inter: dict[tuple[int, int], int] = {}
    # Unordered pairs, counted once. A wikilink is directed and a mutual pair is
    # two edges, so counting edges made cohesion exceed 1 on any small area whose
    # notes link both ways — a 2-note area with a mutual link scored 2.0 on a
    # ratio bounded in [0, 1]. `compute_report` walks `G_und.edges()` for exactly
    # this reason; deduping here is what puts the two on one currency. The
    # off-diagonal therefore counts LINKED PAIRS of notes, not wikilinks.
    seen_pairs: set[tuple[str, str]] = set()
    for e in edges:
        if e.get("type") != "EXTRACTED":
            continue
        a, b = e.get("from"), e.get("to")
        ga, gb = group_of.get(a), group_of.get(b)
        if ga is None or gb is None or a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
        if ga == gb:
            if ga >= 0:
                intra[ga] = intra.get(ga, 0) + 1
        elif ga >= 0 and gb >= 0:
            inter[(min(ga, gb), max(ga, gb))] = inter.get((min(ga, gb), max(ga, gb)), 0) + 1

    sizes: dict[int, int] = {}
    for g in group_of.values():
        if g >= 0:
            sizes[g] = sizes.get(g, 0) + 1

    # Multi-note communities only, biggest first. A singleton is its own
    # community: on the matrix it is a row and column of zeroes with a perfect
    # diagonal, which is 65 rows of noise around the 26 that carry the vault.
    ids = sorted((g for g, s in sizes.items() if s > 1), key=lambda g: (-sizes[g], g))
    hub_of: dict[int, str] = {}
    for g in ids:
        members = [n for n in real if group_of.get(n["id"]) == g]
        best = max(members, key=lambda n: deg.get(n["id"], 0), default=None)
        if best:
            hub_of[g] = best["id"]

    def cohesion(g: int) -> float:
        s = sizes[g]
        possible = s * (s - 1) / 2
        return round(intra.get(g, 0) / possible, 4) if possible else 0.0

    areas = [
        {"id": g, "label": label_of.get(hub_of.get(g, ""), f"#{g}"), "path": hub_of.get(g, ""),
         "size": sizes[g], "cohesion": cohesion(g), "intra": intra.get(g, 0)}
        for g in ids
    ]
    matrix = [[(intra.get(a, 0) if a == b else inter.get((min(a, b), max(a, b)), 0))
               for b in ids] for a in ids]

    return {
        "areas": areas,
        "matrix": matrix,
        # Every real note, for the containment view. Three fields, not the whole
        # node: the treemap needs a path, an area and a weight, and shipping the
        # colour/font/title the canvas uses would triple the payload for nothing.
        "notes": [{"path": n["id"], "size": n.get("size") or 1, "area": group_of.get(n["id"], -1)}
                  for n in real],
        "reading": _shape_reading(adj, deg, areas, label_of),
        "totals": {"notes": len(real), "areas": len(ids),
                   "singletons": sum(1 for s in sizes.values() if s <= 1)},
    }


def _write_sessions(report) -> dict | None:
    """The days claims were written, crossed against the areas that received them.

    Not a chronology of the vault: the vault's clock is per-claim (`valid_from`
    stamps), and only a nucleated note carries one. So this measures what WROTE
    the vault, not when the vault's subjects happened.

    A session x area matrix, not a time axis. Measured on a real vault the dates
    collapse onto 9 days inside a 2-month window with one straggler two years
    back; on a linear date axis that straggler takes 90% of the width and the
    nine days that hold 99% of the work land in a smear. The matrix drops the
    duration and keeps what varies -- which areas recur across sessions.

    Areas are the multi-note communities only. A singleton is its own community,
    so counting them would make every session look perfectly focused by
    construction. Areas never written into are counted rather than listed: the
    reading here is coverage, and 19 empty columns bury the 7 carrying the work.

    A stem that resolves to more than one note (the vault has forked pairs
    sharing a subpath under `write_dir`) is attributed to no area and counted as
    ambiguous, because guessing one of the two would silently move a mark into
    the wrong column.
    """
    from silica.kernel.write.timeline import timeline

    vault = Path(CONFIG.vault_path or "").expanduser()
    if not vault.is_dir():
        return None
    rows = timeline(vault, limit=10**6)["rows"]
    if not rows:
        return None

    areas = [c for c in report.clusters if c.size > 1]
    # Stem -> the areas claiming it. A set, so a genuine fork shows up as >1
    # rather than as whichever member the iteration happened to reach last.
    by_stem: dict[str, set[int]] = {}
    for c in areas:
        for m in c.members:
            by_stem.setdefault(m.rsplit("/", 1)[-1].removesuffix(".md"), set()).add(c.cluster_id)
    hub_path = {c.cluster_id: (c.hub or "") for c in areas}

    days: dict[str, dict[str, int]] = {}
    touched: dict[int, int] = {}
    ambiguous = unplaced = 0
    for date, _label, stem in rows:
        hit = by_stem.get(stem)
        if not hit:
            unplaced += 1
            continue
        if len(hit) > 1:
            ambiguous += 1
            continue
        cid = next(iter(hit))
        cells = days.setdefault(date, {})
        cells[str(cid)] = cells.get(str(cid), 0) + 1
        touched[cid] = touched.get(cid, 0) + 1

    if not days:
        return None
    # Busiest area first: the columns are read left to right, and the areas the
    # writing actually lands in are the ones worth seeing without scrolling.
    ordered = sorted(touched.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "areas": [
            {"id": str(cid), "label": (hub_path.get(cid, "") or f"#{cid}").rsplit("/", 1)[-1]
             .removesuffix(".md"), "path": hub_path.get(cid, ""), "total": n}
            for cid, n in ordered
        ],
        "days": [
            {"date": d, "notes": sum(cells.values()), "cells": cells}
            for d, cells in sorted(days.items())
        ],
        "areas_total": len(areas),
        "untouched": len(areas) - len(touched),
        # The notes with no claim clock at all. Named, never dropped: on a vault
        # written mostly by hand this is the overwhelming majority, and a matrix
        # that omits it reads as "the whole vault, over 9 days".
        "undated": max((report.totals or {}).get("notes", 0) - len(rows), 0),
        "ambiguous": ambiguous,
        "unplaced": unplaced,
    }


@app.get("/calendar")
def calendar(start: str = "", days: int = 7):
    """The 4-axis agenda days for the calendar tab — one endpoint, one build
    (the /shape pattern), plus the one reading only a calendar can carry.

    `bursting` (V6) is what the last fortnight of WRITING turned out to be
    about. It is here and not on explore because it is the only one of the
    seven variables with no position in the graph at all: its axis is time,
    which is this tab's axis. Read from the cheap pass (~0.3 s) rather than
    from the report's co-occurrence depth (9 s cold) — both call the same
    `signals.burst` over the same inputs, and tests/test_graph_variables.py
    holds them equal.
    """
    from silica.kernel.report.structure import bursting
    from silica.tools.events import silica_agenda

    out = silica_agenda(start=start or "today", days=max(1, min(90, days)))
    try:
        out["bursting"] = bursting()
    except Exception:
        # A reading, never the tab: an index that cannot answer costs the strip
        # and leaves every scheduled day exactly where it was.
        logger.debug("calendar: burst unavailable", exc_info=True)
    return out


@app.post("/reminders", dependencies=_SAME_ORIGIN)
def reminders_poll():
    """The front-end poll IS the reminder tick: compute due, advance the
    high-water marks, return the list. POST, not GET — the poll mutates the
    sidecar, and a cacheable GET risks a stale 200 swallowing a delivery.
    At-most-once: the mark advances on delivery; REPL and GUI share the
    sidecar, so whichever surface polls first delivers."""
    import datetime as _dt
    from pathlib import Path as _Path

    from silica.config import CONFIG
    from silica.kernel.calendar.model import scan_events
    from silica.kernel.calendar.reminders import (
        advance_marks, delivery_lock, due_reminders, load_marks, save_marks,
    )

    vault = _Path(CONFIG.vault_path)
    events = scan_events(vault)
    with delivery_lock(vault):
        marks = load_marks(vault)
        due = due_reminders(events, marks, _dt.datetime.now())
        if due:
            save_marks(vault, advance_marks(marks, due))
    return {"due": [{"stem": r["stem"], "title": r["title"],
                     "start": r["start"].isoformat(sep=" "), "late": r["late"]}
                    for r in due]}


def _report_head(report, payload: dict, elapsed_s: float, started) -> dict:
    """The run head of the Report panel, plus the reading it can only get from a store.

    Everything else the metrics view shows is derived from THIS report. The one
    thing it cannot derive is what moved, so this is where the report is filed
    and the last different one is handed back.

    Best-effort by construction: a vault that is read-only, unbound or on a full
    disk still gets its metrics. Losing the delta costs a section of one panel;
    failing the call costs the whole view.
    """
    import datetime as _dt

    from silica.kernel.report.history import record_report, signals_of

    areas = len([c for c in report.clusters if c.size > 1])
    # compute_report memoises per vault epoch, so a second open of the tab costs
    # microseconds and "0.0s" in the run head would read as "the audit is free"
    # rather than "you are looking at the one from two minutes ago". The duration
    # is stated only when this call actually computed it; when it did not, the
    # head's own timestamp is the honest answer and the panel drops the figure.
    #
    # Compared against the instant before the call rather than against a
    # tolerance: `generated_at` is stamped inside compute_report, so a fresh
    # report's is strictly after `started` and a memo hit's is strictly before.
    # No threshold to tune and no wrong answer on a fast vault.
    fresh = True
    try:
        made = _dt.datetime.fromisoformat(report.generated_at)
        fresh = made >= started
    except (TypeError, ValueError):
        pass  # an unparseable stamp is not a reason to drop the duration
    head = {
        "at": report.generated_at,
        "elapsed_s": round(elapsed_s, 2) if fresh else None,
        "notes": (report.totals or {}).get("notes", 0),
        "depth": payload["depth"],
        "signals": {},
        "previous": None,
        "since": None,
    }
    try:
        signals = signals_of(report.totals or {}, areas)
        head["signals"] = signals
        if CONFIG.vault_path:
            prev = record_report(CONFIG.vault_path, signals)
            if prev:
                head["previous"] = prev.get("signals")
                head["since"] = prev.get("at")
    except Exception as exc:
        logger.debug("metrics: report history skipped (%s)", exc)
    return head


@app.get("/metrics")
def metrics(proposals: bool = False):
    """Everything the L1 graph report measures, as JSON for the metrics tab.

    Two depths, because the co-occurrence leg costs an order of magnitude more
    than the rest and, unlike the rest, grows with the square of the vault
    (_compute_cooccur_delta ranks every note against every other):

      default          — analytics + embeddings (~2s on a 686-note vault).
                         `depth: "structural"`.
      ?proposals=1     — adds the co-occurrence delta (autolink candidates,
                         stale links, missing hubs, integration deficits).
                         ~7s on the same vault. `depth: "full"`.

    The depth rides in the payload because E(vault) is only comparable across
    reports built at the same depth (see vault_energy): its `deficits` term is
    zero without the co-occurrence leg, and on a real vault that term dominates.
    The client labels the number rather than letting two different E's look alike.
    """
    import datetime as _datetime
    from collections import Counter

    from silica.kernel.report.graph_report import compute_report
    from silica.kernel.report.vault_energy import vault_energy

    t0, started = time.perf_counter(), _datetime.datetime.now(_datetime.timezone.utc)
    try:
        report = compute_report(
            analytics=True, with_embeddings=True, with_cooccurrence=proposals, top_k=20,
        )
    except Exception as exc:
        logger.warning("metrics: report failed (%s)", exc)
        return {"error": str(exc)}
    elapsed_s = time.perf_counter() - t0

    e = vault_energy(report)
    short = lambda nid: (nid or "").rsplit("/", 1)[-1]  # noqa: E731
    # An area is named by its hub note's *name*: the full store path is a folder
    # tree, and in a table cell it wraps to three lines and says nothing extra.
    label = {c.cluster_id: (short(c.hub) or f"#{c.cluster_id}") for c in report.clusters}
    size = {c.cluster_id: c.size for c in report.clusters}

    payload = {
        "path": CONFIG.vault_path or "",
        "generated_at": report.generated_at,
        "depth": "full" if proposals else "structural",
        "totals": report.totals,
        "discourse_state": report.discourse_state,
        "energy": {
            "total": round(e.total, 2),
            # Ordered as E is composed: the one negative (bond-forming) term
            # first, then the entropic costs. `deficits` is dropped rather than
            # printed as 0.00 when the leg that measures it never ran — a zero
            # would read as "measured, came out flat". It contributes 0.0 either
            # way, so the terms still sum to `total`.
            "terms": [
                {"name": "cohesion", "value": round(e.cohesion, 2)},
                {"name": "orphans", "value": round(e.orphans, 2)},
                {"name": "dangling", "value": round(e.dangling, 2)},
                {"name": "gaps", "value": round(e.gaps, 2)},
                *([{"name": "deficits", "value": round(e.deficits, 2)}] if proposals else []),
                {"name": "contested", "value": round(e.contested, 2)},
            ],
        },
        "degree_histogram": _degree_histogram(report.degree_map),
        "clusters": [
            {"id": c.cluster_id, "size": c.size, "hub": short(c.hub), "path": c.hub,
             "cohesion": c.cohesion}
            for c in sorted(report.clusters, key=lambda c: -c.size)
        ],
        "hubs": [
            {"label": n.label, "path": n.id, "area": label.get(n.cluster, f"#{n.cluster}"),
             "degree": n.degree, "in": n.in_degree, "out": n.out_degree,
             "betweenness": n.betweenness}
            for n in report.god_nodes
        ],
        "bridges": [
            {"source": short(b.source), "target": short(b.target),
             "source_path": b.source, "target_path": b.target, "weight": b.weight}
            for b in report.bridges
        ],
        # The two area sizes ride along because they *are* the ranking:
        # gap_score = size_a * size_b / (1 + inter_edges). gap_density is left
        # out — on a real vault it reads 99.7-100% on every row, and a column
        # that never varies cannot explain the order it is sitting in.
        "gaps": [
            {"a": short(g.hub_a), "b": short(g.hub_b), "a_path": g.hub_a, "b_path": g.hub_b,
             "inter_edges": g.inter_edges, "size_a": size.get(g.cluster_a, 0),
             "size_b": size.get(g.cluster_b, 0)}
            for g in report.structural_gaps
        ],
        "orphans": [{"label": short(p), "path": p} for p in report.orphans[:_METRICS_ROWS]],
        # A target is a row you act on, so it carries what it needs to be judged:
        # who asks for it. Three names and a count, because at 372px of evidence
        # pane the fourth name is what starts wrapping.
        "dangling": [
            {"target": d["target"], "refs": d["refs"],
             "from": [short(src) for src in d.get("sources", [])[:3]],
             "from_more": max(0, len(d.get("sources", [])) - 3)}
            for d in report.dangling[:_METRICS_ROWS]
        ],
        # The tail, over EVERY target and not the twelve above. The shape IS the
        # reading -- a handful of targets carry most of the references and the
        # rest are asked for once each, which is the difference between "write
        # twenty notes" and "stub four hundred" -- and it cannot be seen from a
        # top-12 slice, so it is summarised here rather than shipped as rows.
        "dangling_hist": [
            {"refs": refs, "targets": n}
            for refs, n in sorted(Counter(d["refs"] for d in report.dangling).items())
        ],
        "dangling_top_refs": sum(d["refs"] for d in report.dangling[:20]),
        "contested": [
            {"label": short(c.path), "path": c.path, "refs": c.refs} for c in report.contested
        ],
        "source_drift": [
            {"label": short(d.note), "path": d.note, "source": d.source}
            for d in report.source_drift[:_METRICS_ROWS]
        ],
        "attention": [
            {"label": short(a.path), "path": a.path, "days_idle": a.days_idle,
             "degree": a.degree, "misses": a.misses, "attempts": a.attempts,
             "score": round(a.score, 2)}
            for a in report.attention_candidates
        ],
        "deficits": [
            {"label": short(d.path), "path": d.path, "concepts": d.concepts,
             "degree": d.degree, "score": round(d.score, 2)}
            for d in report.integration_deficits
        ],
        # Confirmed first — those are the merge candidates; the borderline band
        # is only "link, don't merge". Neither list is capped by the report, and
        # on a real vault they run to the hundreds, so the slice happens here.
        "duplicates": ([
            {"a": short(d.source), "b": short(d.target), "a_path": d.source,
             "b_path": d.target, "score": d.score, "confirmed": True}
            for d in report.confirmed_duplicate_pairs
        ] + [
            {"a": short(d.source), "b": short(d.target), "a_path": d.source,
             "b_path": d.target, "score": d.score, "confirmed": False}
            for d in report.duplicate_pairs
        ])[:_METRICS_ROWS],
        # Sliced like every other uncapped list: the report caps the
        # co-occurrence leg at top_k, but the import-derived candidates
        # _compute_code_signals appends are exhaustive — 13k pairs on a
        # 400-note vault, which is a 4 MB payload and a card 390,000 px tall.
        # The true count rides in `totals`, so the cut list can't read as all.
        "autolinks": [
            {"a": short(a.source), "b": short(a.target), "a_path": a.source,
             "b_path": a.target, "weight": a.weight, "shared": a.shared[:4]}
            for a in report.autolink_candidates[:_METRICS_ROWS]
        ],
        "stale_links": [
            {"a": short(s.source), "b": short(s.target), "a_path": s.source, "b_path": s.target}
            for s in report.stale_links
        ],
        "missing_hubs": [
            {"concept": h.concept, "centrality": round(h.centrality, 3)}
            for h in report.missing_hubs
        ],
        # `chars` and the limit it is under, together: a bare "412" is a number
        # the reader has to be told the meaning of, and the pane cannot say
        # "under 600" for a row without knowing what 600 is.
        "lean_notes": [
            {"label": short(p), "path": p, "chars": report.lean_chars.get(p, 0)}
            for p in report.lean_notes[:_METRICS_ROWS]
        ],
        "lean_limit": _lean_limit(),
        # V7 and V6 (spec 2026-08-22). Both ride the co-occurrence depth, so
        # both are absent at structural depth rather than shipped empty: an
        # empty list would read as "measured, found nothing".
        "sprawling": [
            # `path` is the store keyspace; the row needs a graph id to open a
            # note, and every sprawling note is in the graph by construction.
            {"label": short(x.path), "path": x.path + ".md", "concepts": x.concepts,
             "entropy": round(x.entropy, 2), "flatness": round(x.flatness, 3)}
            for x in report.sprawling
        ],
        "bursting": [
            {"concept": b.concept, "z": round(b.z, 2), "recent": b.recent, "total": b.total}
            for b in report.bursting_concepts
        ],
        "temporal": (
            {
                "notes_scanned": report.temporal.notes_scanned,
                "by_tier": {str(k): v for k, v in report.temporal.by_tier.items()},
                "stamped": report.temporal.stamped,
                "superseded_notes": report.temporal.superseded_notes,
                "superseded_sections": report.temporal.superseded_sections,
                "oldest_valid_from": report.temporal.oldest_valid_from,
            }
            if report.temporal and report.temporal.notes_scanned
            else None
        ),
        "sessions": _write_sessions(report),
        # The three readings the charts need that no single report can hold.
        # history is the only one that outlives a run: it is what turns four
        # tiles from a state into a direction.
        "history": _metrics_history(CONFIG.vault_path),
        "area_matrix": _area_matrix(report),
        "signal_areas": _signal_areas(report),
        "code_coverage": (
            {
                "documented": report.code_coverage.documented,
                "total": report.code_coverage.total,
                "undocumented": [
                    {"path": p, "fan_in": f}
                    for p, f in report.code_coverage.undocumented[:_METRICS_ROWS]
                ],
            }
            if report.code_coverage and report.code_coverage.total
            else None
        ),
    }
    payload["report"] = _report_head(report, payload, elapsed_s, started)
    return payload


@app.get("/graph")
def graph(request: Request):
    """The explore iframe's whole document, force-graph bundles inlined.

    That inlining is what makes this route ~6 MB, and it used to ship
    uncompressed with no validator at all, so every visit to the explore tab
    refetched every byte. Two cheap fixes, in this order:

      * an ETag over the payload, so a revisit costs a 304 and nothing else.
        Content-addressed rather than time-based, because the document changes
        when the vault does and no clock knows when that was.
      * gzip when the client asks for it, which measured 6,229,645 -> 766,371
        bytes on a 1,199-note vault. Done here rather than with GZipMiddleware,
        which would also wrap the SSE turn stream and stall it.

    no-cache, not no-store: the browser must revalidate (the vault changes under
    it) but is allowed to keep the bytes and take the 304.
    """
    import gzip as _gzip
    import hashlib
    import tempfile

    from silica.tools import TOOLS

    # Per-request directory, not a fixed name in /tmp: the route is a sync def,
    # so two overlapping requests ran in the threadpool and shared one file — and
    # a world-writable path anyone can pre-create as a symlink is not somewhere
    # to write a document we then serve back.
    try:
        with tempfile.TemporaryDirectory(prefix="silica-graph-") as tmp:
            out = Path(tmp) / "graph.html"
            TOOLS["silica_graph_export"].run(output_path=str(out), folder="")
            body = out.read_text(encoding="utf-8").encode("utf-8")
    except Exception as exc:
        return HTMLResponse(
            f"<p style='font-family:monospace'>graph unavailable: {_html.escape(str(exc))}</p>"
        )

    etag = '"' + hashlib.blake2b(body, digest_size=8).hexdigest() + '"'
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    if "gzip" in request.headers.get("accept-encoding", "").lower():
        body = _gzip.compress(body, compresslevel=6)
        headers["Content-Encoding"] = "gzip"
        headers["Vary"] = "Accept-Encoding"
    return Response(body, media_type="text/html; charset=utf-8", headers=headers)


@app.get("/map")
def mindmap(note: str = ""):
    """Static-SVG radial map rooted on `note` — ephemeral, in-session (not written).

    Consumes the same precomputed positions as the .canvas serializer, so the two
    surfaces cannot diverge. Empty/unknown note degrades to a message, like /graph.
    """
    from silica.config import CONFIG
    from silica.kernel.recall.mindmap import (
        build_mapview,
        gather_materials,
        note_resolver,
        render_map_svg,
    )

    if not note.strip():
        return HTMLResponse("<p style='font-family:monospace;color:#8a93a3'>enter a note: /map?note=…</p>")
    try:
        # Accept a title or a path — the input field usually gives a title.
        root = note_resolver()(note)
        if root is None:
            return HTMLResponse(
                f"<p style='font-family:monospace;color:#8a93a3'>"
                f"'{_html.escape(note)}' not found in vault.</p>"
            )
        materials = gather_materials(root, latent_k=CONFIG.mindmap_latent_k)
        mv = build_mapview(
            root, materials, max_nodes=CONFIG.mindmap_max_nodes, hops=CONFIG.mindmap_hops
        )
        if len(mv.nodes) <= 1:
            return HTMLResponse(
                f"<p style='font-family:monospace;color:#8a93a3'>'{_html.escape(root)}' has no "
                "neighbors to map (isolated in the graph).</p>"
            )
        return HTMLResponse(render_map_svg(mv, title=f"map · {root}"))
    except Exception as exc:
        # The exception text quotes the caller's `note`, so it is untrusted here
        # too — this response loads same-origin in the explore iframe.
        return HTMLResponse(
            f"<p style='font-family:monospace'>map unavailable: {_html.escape(str(exc))}</p>"
        )


@app.get("/path")
def prereq_path(note: str = ""):
    """The prerequisite ladder around one note (V2, RefD): read-order space.

    The fifth explore surface exists because the other four cannot show
    DIRECTION. graph and map lay notes out by how they connect, folders by
    where they are filed, areas by how two groups couple: all four are
    undirected, and "what do I read before this" has no answer in any of them.

    Empty `note` returns the landing instead of an error, like /map: the notes
    that root the biggest ladders, which is the one list that cannot be got
    from the file tree.
    """
    from silica.kernel.recall.mindmap import note_resolver
    from silica.kernel.report.structure import ladder, structure_map

    try:
        m = structure_map()
    except Exception as exc:
        logger.warning("path: structure map failed (%s)", exc)
        return {"error": str(exc)}
    if not m.prereq:
        return {
            "picks": [],
            # The one thing that makes this surface empty, said as the surface
            # itself rather than as a blank pane: RefD needs the co-occurrence
            # index, and a vault that never built one has no reading order to
            # show, not a reading order of length zero.
            "hint": "no reading order yet: the prerequisite direction is derived from the "
                    "co-occurrence index. Run /report to build it.",
        }

    if not note.strip():
        # Ranked by how much of a ladder each note actually roots. Computing the
        # real ladder for every candidate would be O(V) walks, so the rank is
        # the cheap proxy (own prerequisites + own dependents) and the top slice
        # is then measured for real, which is what the count on the row says.
        rough = sorted(
            m.prereq.keys() | m.unlocks.keys(),
            key=lambda k: (-(len(m.prereq.get(k, ())) + len(m.unlocks.get(k, ()))), k),
        )[:40]
        picks: list[dict[str, Any]] = []
        for k in rough:
            l = ladder(k + ".md")
            if len(l["nodes"]) < 2:
                continue
            picks.append({
                "name": _clean_name(k), "path": k + ".md",
                "notes": len(l["nodes"]),
                "before": sum(1 for n in l["nodes"] if n["depth"] < 0),
                "after": sum(1 for n in l["nodes"] if n["depth"] > 0),
            })
        # A note with notes on BOTH sides is the instructive root: it is the one
        # whose ladder shows an order rather than a fan. Ranking by size alone
        # put the vault's biggest hub first every time, and its ladder is one
        # rung of 59 dependents, which is a list with a title.
        picks.sort(key=lambda r: (-min(r["before"], r["after"]), -r["notes"], r["name"]))
        return {"picks": picks[:12], "hint": ""}

    # `ladder` resolves its own root and hands the graph id back, so the two
    # keyspaces (graph ids carry `.md`, RefD works without it) are bridged once,
    # inside the kernel, and never here.
    resolve = note_resolver()
    l = ladder(resolve(note) or note)
    canon = l["root"]
    if not canon:
        return {"error": f"'{note}' not found in vault."}
    if not l["nodes"]:
        return {
            "root": _row(canon), "levels": [], "edges": [], "cycles": 0, "truncated": False,
            "hint": "nothing reads before or after this note: RefD found no direction "
                    "between it and its neighbours.",
        }
    by_depth: dict[int, list[dict]] = {}
    for n in l["nodes"]:
        by_depth.setdefault(n["depth"], []).append(
            {**_row(n["path"]), "cyclic": n["cyclic"], "root": n["path"] == canon}
        )
    levels = [
        {"depth": d, "notes": sorted(by_depth[d], key=lambda r: r["name"])}
        for d in sorted(by_depth)
    ]
    return {
        "root": _row(canon), "levels": levels, "edges": l["edges"],
        "cycles": l["cycles"], "truncated": l["truncated"], "hint": "",
    }


@app.get("/find")
def find(q: str = "", k: int = 5):
    """Direct semantic-search panel: calls the tool straight, same pattern as /graph and /map."""
    from silica.tools import TOOLS

    q = q.strip()
    if not q:
        return HTMLResponse("<p style='font-family:monospace;color:#8a93a3'>usage: /find &lt;query&gt; [--k=N]</p>")
    try:
        parsed = json.loads(TOOLS["silica_semantic_search"].run(query=q, k=k))
    except Exception as exc:
        return HTMLResponse(
            f"<p style='font-family:monospace'>find unavailable: {_html.escape(str(exc))}</p>"
        )
    if "error" in parsed:
        return HTMLResponse(f"<p style='font-family:monospace;color:#8a93a3'>{_html.escape(parsed['error'])}</p>")
    results = parsed.get("results", [])
    if not results:
        return HTMLResponse(f"<p style='font-family:monospace;color:#8a93a3'>no results for '{_html.escape(q)}'.</p>")
    rows = []
    for r in results:
        p = r.get("path") or r.get("name") or "?"
        rows.append(
            f'<div class="find-result">{_anchor(p, _clean_name(p))}'
            f'<span class="find-score">{r.get("score", 0.0):.3f}</span></div>'
        )
    return HTMLResponse("".join(rows))


@app.get("/note")
def note(path: str = ""):
    """Read-only rendered note for the drawer. Graceful on miss (never 500).

    Only keys present in the vault index resolve, so an out-of-vault `path`
    falls through to the graceful message — path traversal is closed for free.
    """
    from silica.driver import get_driver
    from silica.driver.base import NoteRef

    resolve = note_resolver()
    canon = resolve(path)
    if not canon:
        return {"title": path, "html": "<p>note not found in vault.</p>"}
    try:
        content = get_driver().read_note(NoteRef(name=_clean_name(canon), path=canon)).content
    except Exception:
        return {"title": _clean_name(canon), "html": "<p>note unreadable.</p>"}
    props, body = _split_frontmatter(content)
    html = _linkify(body, resolve)
    if props:
        html = _render_frontmatter(props) + html
    return {"title": _clean_name(canon), "html": html}


# --- what this session changed (GET /changes, /changes/diff) ------------------
# The ledger and the tally are the driver's (silica.kernel.write.session_changes),
# shared with the REPL's /changes. All that is added here is a display name and
# the line-by-line rendering, which only a drawer needs.

_DIFF_CONTEXT = 3
# A hard line cap, tail dropped with a count — a decided constant, not a
# deferral: past a few hundred lines a diff stops being reviewable in a
# drawer and the note itself is one click away.
_MAX_DIFF_LINES = 800
# difflib opens every diff with a hunk header, but a gap marker only *means*
# something when lines were skipped above it — which is not the case when the
# first hunk starts at the top of the file (or at 0, for a create or a delete).
_HUNK_AT_TOP = re.compile(r"^@@ -[01](?:,\d+)? \+[01](?:,\d+)? @@")


def _change_rows() -> list[dict]:
    return [{**r, "name": _clean_name(r["path"])} for r in session_changes.rows()]


@app.get("/changes")
def changes():
    """Every note this session has changed, oldest first."""
    return _change_rows()


@app.get("/changes/diff")
def changes_diff(path: str = ""):
    """One note's diff as flat rows: `-` removed, `+` added, ` ` context, `@` gap."""
    import difflib

    base = session_changes.snapshot().get(path)
    if base is None:
        # No baseline: this session never touched the note, so there is no diff
        # to read and the caller should show the note itself. Distinct from a
        # note that WAS touched and is now byte-identical again, which has a
        # baseline and an empty line list — the drawer says so in its own words,
        # and a write card must not silently degrade one into the other.
        return {"path": path, "name": _clean_name(path), "kind": "unchanged",
                "baseline": False, "lines": []}
    before, after = base.before or "", session_changes.current_text(path)
    added, removed = session_changes.tally(before, after or "")
    rows: list[dict] = []
    diff = difflib.unified_diff(before.splitlines(), (after or "").splitlines(),
                                lineterm="", n=_DIFF_CONTEXT)
    for i, ln in enumerate(diff):
        if i < 2:
            continue  # the ---/+++ file headers difflib always emits first
        if ln.startswith("@@"):
            if not rows and _HUNK_AT_TOP.match(ln):
                continue  # nothing was skipped above the first line
            rows.append({"op": "@", "text": ""})
            continue
        rows.append({"op": ln[:1] or " ", "text": ln[1:]})
    return {
        "path": path,
        "name": _clean_name(path),
        "kind": session_changes.kind(base.before, after, base.origin, bool(added or removed)),
        "from": base.origin,
        "baseline": True,
        "added": added,
        "removed": removed,
        "lines": rows[:_MAX_DIFF_LINES],
        "clipped": max(0, len(rows) - _MAX_DIFF_LINES),
    }


# --- context explorer (GET /context) -----------------------------------------
# One blocking call, all of it deterministic and LLM-free. Measured on a
# 718-note vault, warm: related 0.01s, concepts(note=) 0.06s, outline/links/
# unresolved 0.00s. The first call in a fresh process pays ~0.9s to load the
# co-occurrence store — once, in a long-lived server. So: no progressive fill,
# no client-side hybrid, one endpoint that returns the whole drawer.

# Below this a note reads faster whole than as an extract, so the snippets
# section is dropped rather than duplicating the reader.
_SNIPPET_MIN_BODY = 700
_SNIPPET_CUT = 140
_H2 = re.compile(r"^##\s+(.+?)\s*$", re.M)
# Lines that say nothing as a one-line extract: headings, lists, quotes,
# callouts, tables, fences, images.
_NOT_PROSE = re.compile(r"^\s*(?:[-*+>#|]|\d+[.)]\s|```|~~~|!\[|\[!)")
_SENTENCE_END = re.compile(r"(?<=[.!?])(?:\s|$)")
# A related note this far away (or unreachable) is a link that does not exist
# yet; distance 1 means it is already linked and belongs under Related instead.
_SUGGEST_MIN_DIST = 3


# A wikilink SPLIT INTO ITS PARTS, which is what the two readers below need and
# what `_WIKILINK` (line 173, the linkifier's) deliberately does not do: that one
# keeps `Target|alias` whole because it re-emits the link, while these two have
# to resolve the target and display the alias separately. Naming them apart is
# not style — the first version of this reused `_WIKILINK` and shadowed the
# linkifier, and every note in the app rendered without its links.
# `[[Target#anchor|alias]]`, and either half can be absent.
_WIKI_REF = re.compile(r"!?\[\[([^\]|#]*)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
# Only the PAIRED marks, and only the unambiguous ones. A lone `*` or `_` is
# left alone on purpose: stripping those turns snake_case into snakecase and a
# multiplication sign into nothing, which is a worse read than the emphasis
# marker it would have removed.
_MD_PAIR = re.compile(r"(\*\*|__|==|~~|`)(.+?)\1", re.S)


def _plain(text: str) -> str:
    """Inline markup out, the words it wrapped left in.

    The extract is prose, not source. This ran nowhere while the snippets lived
    in the note drawer, one scroll under the reader that showed the same
    sentence properly rendered; they are the first thing the work panel says
    about a note now, and `==Un **Database** o Base di dati` at the top of a
    panel reads as a broken row rather than as emphasis. Block-level markup
    needs no handling here: _NOT_PROSE already refuses those lines.
    """
    out = _WIKI_REF.sub(lambda m: (m.group(2) or m.group(1) or "").strip(), text)
    out = _MD_LINK.sub(lambda m: m.group(1).strip(), out)
    # Pairs NEST (`==A **b** c==`) and sub() does not rescan what it inserted,
    # so one pass leaves the inner marks standing. Three passes, because three
    # is one more than the deepest nesting anyone writes by hand: highlight
    # around bold around code.
    for _ in range(3):
        stripped = _MD_PAIR.sub(lambda m: m.group(2), out)
        if stripped == out:
            break
        out = stripped
    return out


def _fm_ref(value: str) -> tuple[str, str]:
    """A frontmatter `related:` entry as (what to resolve, what to show)."""
    v = str(value).strip()
    m = _WIKI_REF.fullmatch(v)
    if not m:
        return v, _clean_name(v)
    target = (m.group(1) or "").strip()
    return target, (m.group(2) or "").strip() or _clean_name(target)


def _lead_prose(chunk: str) -> str:
    """The first run of plain prose in a chunk, as one line."""
    run: list[str] = []
    for line in chunk.splitlines():
        if not line.strip() or _NOT_PROSE.match(line):
            if run:
                break
            continue
        run.append(line.strip())
    return " ".join(run)


def _first_sentence(text: str, limit: int = _SNIPPET_CUT) -> str:
    """First sentence, trimmed to `limit` on a word boundary.
    ponytail: regex sentence split, so `e.g.` cuts early — a snippet, not a quote."""
    text = " ".join(text.split())
    if not text:
        return ""
    m = _SENTENCE_END.search(text)
    out = text[: m.start()] if m else text
    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0] + "…"
    return out


def _key_snippets(body: str) -> list[dict]:
    """First sentence of the body plus the first sentence of each `##` section,
    at most three — a probe into the note, not a second reader."""
    if len(body) < _SNIPPET_MIN_BODY:
        return []
    parts: list[tuple[str, str]] = []
    head, last = "", 0
    for m in _H2.finditer(body):
        parts.append((head, body[last:m.start()]))
        head, last = m.group(1), m.end()
    parts.append((head, body[last:]))

    out: list[dict] = []
    for heading, chunk in parts:
        # _plain BEFORE _first_sentence, not after: the cut lands mid-sentence
        # by design, so a `==highlight==` whose closing mark falls past the
        # limit would otherwise keep its opening one and nothing to close it.
        text = _first_sentence(_plain(_lead_prose(chunk)))
        if text:
            out.append({"heading": _plain(heading), "text": text})
        if len(out) == 3:
            break
    return out


def _row(path: str) -> dict:
    return {"name": _clean_name(path), "path": path}


def _note_concepts(target: str, k: int = 18) -> list[dict]:
    from silica.tools.graph import silica_concepts

    try:
        return silica_concepts(note=target, k=k).get("concepts") or []
    except Exception:
        logger.debug("context: concepts failed for %s", target, exc_info=True)
        return []


def _unresolved_links() -> list:
    from silica.driver import get_driver

    try:
        return get_driver().unresolved()
    except Exception:
        logger.debug("context: unresolved() failed", exc_info=True)
        return []


def _ghost_context(name: str) -> dict:
    """An unresolved wikilink as a subject of its own.

    Today a ghost node carries path "" (graph_export.py), so clicking one used to
    post an empty path and open nothing. It has no body to read and no reader
    mode — what it does have is a name, the notes that invoke it, and their
    merged concepts, which is exactly the material for deciding whether to write
    it.
    """
    from collections import Counter

    stem = _clean_name(name).lower()
    invokers = sorted({
        link.source.path for link in _unresolved_links()
        if _clean_name(link.target).lower() == stem and link.source.path
    })
    merged: Counter = Counter()
    for src in invokers[:12]:  # decided cap: a 12-note cloud is already dense
        for c in _note_concepts(src, k=12):
            merged[c["concept"]] += c.get("weight", 1)
    return {
        "title": _clean_name(name),
        "path": "",
        "ghost": True,
        "snippets": [],
        "concepts": [{"concept": c, "weight": w} for c, w in merged.most_common(18)],
        "related": {"frontmatter": [], "outgoing": [], "backlinks": [_row(p) for p in invokers]},
        "suggested": [],
        "hint": "",
    }


def _suggested(canon: str, related: list[dict], linked: set[str], resolve) -> list[dict]:
    """How this note SHOULD be connected, in two flavours.

    - ghost: a wikilink leaving this note whose target does not exist. The note
      already claims the connection; only the file is missing.
    - note: a computed relative that scores high and sits far away (or
      unreachable) in the wikilink graph — "a missing link worth creating", per
      silica_related's own docstring. distance 1 is already linked, so it is
      Related's business, not this section's.

    Structural GAPs stay out on purpose: they are hub-to-hub by construction, so
    the section would be empty on every note that is not a hub.
    """
    out: list[dict] = [
        {"name": _clean_name(link.target), "path": "", "kind": "ghost",
         "why": "linked from here, never written"}
        for link in _unresolved_links()
        if link.source.path == canon
    ]
    for r in related:
        dist = r.get("distance")
        # The recall stores key on cooccur_key (path minus .md), the wikilink
        # graph on the full path — resolve back, or `linked` never matches and
        # the click target is a path the drawer cannot open.
        rpath = resolve(r["path"]) or r["path"]
        if rpath in linked or rpath == canon or (dist is not None and dist < _SUGGEST_MIN_DIST):
            continue
        out.append({
            "name": r.get("name") or _clean_name(rpath), "path": rpath, "kind": "note",
            "why": ("unreachable" if dist is None else f"{dist} hops away")
                   + f" · score {r.get('score', 0):.2f}",
            # The same score the sentence above states in words. The explore
            # panel draws it as a meter, and parsing it back out of `why` is how
            # a display string becomes an accidental API: reword the sentence
            # and the bar goes blank.
            "score": float(r.get("score", 0) or 0),
        })
        if len(out) >= 8:
            break
    return out


def _note_structure(canon: str) -> dict:
    """The seven variables for one note, named for a reader rather than a paper.

    Rows, not scalars: the panel prints what each number MEANS, so the naming
    happens here where the thresholds can be justified, not in JS where they
    would be three magic numbers in a template. A note the graph has never seen
    (written this second, index not rebuilt) returns {} and the panel omits the
    section rather than printing six zeroes.
    """
    from silica.kernel.report.structure import note_structure

    try:
        st = note_structure(canon)
    except Exception:
        logger.debug("context: structure failed for %s", canon, exc_info=True)
        return {}
    if not st or not st.get("in_graph"):
        return {}
    return {
        "coreness": st["coreness"],
        "articulation": st["articulation"],
        "strands": st["strands"],
        # Rounded to whole percent HERE: the raw pct-rank difference carries
        # four decimals of sampling noise from a betweenness taken at 400
        # pivots, and printing them would claim a precision the estimate has not
        # got.
        "surprise": round(st["surprise"] * 100),
        "dissonance": (None if st["dissonance"] is None
                       else round(st["dissonance"] * 100)),
        # prerequisites/unlocks are NOT here: they are the +-1 rungs of the
        # ladder below, and the panel draws that instead. Two spellings of one
        # relation in one payload is the drift this file's tests exist to stop.
    }


def _note_ladder(canon: str, *, near: int = 2, per_rung: int = 3) -> dict:
    """Where this note sits in a reading order, not just who touches it.

    The two flat lists this replaced could not say the one thing the DAG knows:
    that a prerequisite has a prerequisite. `ladder` walks three hops each way,
    and the drawer draws the near rungs at `near`; everything past that is a
    count and a hand-off to the Path surface, which is the one place in the app
    that draws a whole ladder. Same reason `per_rung` exists: a 310px column
    that prints all 21 dependents of a hub buries every section under it.

    Empty dict, never a rung of one: a note with no direction around it has no
    reading order, and a section holding only the note you are already on is a
    heading with nothing under it.
    """
    from silica.kernel.report.structure import ladder as build_ladder

    try:
        lad = build_ladder(canon)
    except Exception:
        logger.debug("context: ladder failed for %s", canon, exc_info=True)
        return {}
    if not lad["nodes"]:
        return {}
    by_depth: dict[int, list[str]] = {}
    for n in lad["nodes"]:
        by_depth.setdefault(n["depth"], []).append(n["path"])
    rungs, further = [], 0
    for d in sorted(by_depth):
        paths = sorted(by_depth[d], key=_clean_name)
        # Past the near rungs the drawer says how many rather than which: the
        # names are the Path surface's job and it has the width for them.
        if abs(d) > near:
            further += len(paths)
            continue
        rungs.append({
            "depth": d,
            "notes": [_row(q) for q in paths[:per_rung]],
            "hidden": max(0, len(paths) - per_rung),
        })
    return {"rungs": rungs, "further": further, "root": lad["root"],
            "cycles": lad["cycles"], "truncated": lad["truncated"]}


@app.get("/context")
def context(path: str = "", name: str = "", ghost: bool = False):
    """Everything deterministic the vault knows about one note, in one call.

    Sections: key snippets (what it says), concepts (what it is about), related
    (how it IS connected), suggested (how it SHOULD be). Zero LLM calls — every
    number here is index lookup, so the drawer is a read, not a turn. Graceful
    on miss, like /note: never 500.
    """
    from silica.driver import get_driver
    from silica.driver.base import NoteRef
    from silica.tools.graph import silica_related

    if ghost or (not path and name):
        return _ghost_context(name or path)

    resolve = note_resolver()
    canon = resolve(path)
    if not canon:
        return {"title": path, "path": path, "ghost": False, "error": "note not found in vault."}

    driver = get_driver()
    try:
        content = driver.read_note(NoteRef(name=_clean_name(canon), path=canon)).content
    except Exception:
        content = ""
    props, body = _split_frontmatter(content)

    def _refs(fn) -> list[dict]:
        # Resolved only. DRIVER.links() also returns a synthesised ref for every
        # UNRESOLVED wikilink (path "<Target>.md", a file that does not exist),
        # and listing those here would put a dead row under "how it IS
        # connected" — they belong under suggested, as links worth writing.
        try:
            return [_row(p) for r in fn(canon) if (p := resolve(r.path or ""))]
        except Exception:
            logger.debug("context: %s failed for %s", fn.__name__, canon, exc_info=True)
            return []

    outgoing = _refs(driver.links)
    backlinks = _refs(driver.backlinks)

    # frontmatter `related:` is a hand-written claim, so it is shown as written
    # and resolved only for the click target; an unresolvable entry still lists.
    # The hand writing it is usually Obsidian's, which writes `- "[[B]]"` — and
    # neither resolve() nor _clean_name() sees through the brackets, so every
    # such entry used to come back named "[[B]]" with an empty path: a dead row
    # under a heading that promises a connection. _fm_ref splits the two jobs.
    fm_raw = (props or {}).get("related") or []
    if not isinstance(fm_raw, (list, tuple)):
        fm_raw = [fm_raw]
    frontmatter = []
    for v in fm_raw:
        if not v:
            continue
        target, display = _fm_ref(str(v))
        frontmatter.append({"name": display, "path": resolve(target) or ""})

    try:
        rel_out = silica_related(note=canon, k=12)
    except Exception:
        logger.debug("context: related failed for %s", canon, exc_info=True)
        rel_out = {}
    rel = rel_out.get("results") or []

    linked = {r["path"] for r in outgoing} | {r["path"] for r in backlinks}
    # Whether the semantic leg contributed is READ OFF the ranking's own
    # provenance — every result names the metric that proposed it (embed:0.83,
    # cooccur:w9, edge:0.57). Asking the embed store directly would reach past
    # the relatedness facade for a fact the facade already reports.
    embed_ran = any(
        str(e).startswith("embed:") for r in rel for e in (r.get("evidence") or [])
    )

    return {
        "title": _clean_name(canon),
        "path": canon,
        "ghost": False,
        "snippets": _key_snippets(body),
        "concepts": _note_concepts(canon),
        "related": {"frontmatter": frontmatter, "outgoing": outgoing, "backlinks": backlinks},
        "suggested": _suggested(canon, rel, linked, resolve),
        "structure": _note_structure(canon),
        "ladder": _note_ladder(canon),
        # Without embeddings `related` ranks on co-occurrence alone, and the
        # section looks thin for a reason the reader cannot see from here.
        # silica_related's own hint wins when it has one — it knows more about
        # why it came back empty than an inference from the evidence can.
        "hint": rel_out.get("hint") or ("" if embed_ran else
                "no embedding index — relatedness is co-occurrence only "
                "(run /embed to add the semantic half)"),
    }


@app.get("/concept")
def concept(term: str = "", k: int = 20):
    """The notes that carry one concept — the click target of the context
    drawer's cloud, which lights them all in the graph at once. Paths are
    resolved back to graph keys so the ids match the nodes the viewer holds."""
    from silica.tools.graph import silica_concepts

    resolve = note_resolver()
    try:
        res = silica_concepts(term=term, k=k)
    except Exception as exc:
        logger.debug("concept: lookup failed for %s", term, exc_info=True)
        return {"term": term, "notes": [], "error": str(exc)}
    return {
        "term": term,
        "concept": res.get("concept") or term,
        "notes": [
            p for n in (res.get("notes") or [])
            if (p := resolve(n.get("path", "")) or n.get("path", ""))
        ],
    }


@app.get("/asset")
def asset(path: str = ""):
    """Vault-relative attachment for the note drawer, `<img>`-only by contract.
    Extension whitelist + resolved-inside-the-vault check close traversal.

    `![[img.png]]` embeds name an attachment by basename even when the file
    lives in an attachments subfolder, so an exact-path miss falls back to a
    first-match basename search under the vault (Obsidian's shortest-path rule
    minus the nearest-to-note tiebreak). rglob stays inside root, so traversal
    is still closed on the fallback path.
    ponytail: per-request rglob on the miss case; build a basename index if a
    large vault makes it slow."""
    if not path or not CONFIG.vault_path:
        raise HTTPException(status_code=404)
    if Path(path).suffix.lower() not in _ASSET_EXTS:
        raise HTTPException(status_code=404)
    root = Path(CONFIG.vault_path).resolve()
    target = (root / path).resolve()
    found: Path | None = target
    if not (target.is_relative_to(root) and target.is_file()):
        found = next((p for p in root.rglob(Path(path).name) if p.is_file()), None)
    if found is None or not found.is_relative_to(root) or found.suffix.lower() not in _ASSET_EXTS:
        raise HTTPException(status_code=404)
    return FileResponse(found, headers=_ASSET_HEADERS)


@app.get("/vault_info")
def vault_info():
    """Sidebar data: vault stats + file tree, from the same builders as the
    graph view so the numbers can't disagree between the two surfaces."""
    from silica.kernel.recall.graph_export import build_graph_data, detect_communities
    from silica.ui.web.graph_view import render_tree

    try:
        nodes, edges = build_graph_data(folder="")
        communities = detect_communities(nodes, edges)
    except Exception as exc:
        return {"error": str(exc)}
    return {
        # `path` so the header label follows a `/vault <dir>` switch mid-session:
        # this endpoint already re-runs after every turn, the boot-time header
        # never did.
        "path": CONFIG.vault_path or "",
        "notes": sum(1 for n in nodes if n.get("type") != "ghost"),
        "links": sum(1 for e in edges if e.get("type") == "EXTRACTED"),
        # STRUCTURAL clusters — Louvain on the wikilinks. The semantic partition
        # is a separate count and is not summed into this one (ADR-0023); the
        # sidebar tile says so in its tooltip, and the graph's HUD counts the
        # zones itself. Nothing here computes the semantic layer: it needs the
        # k-NN edges this endpoint has no reason to build.
        "clusters": len(communities),
        "unresolved": sum(1 for n in nodes if n.get("type") == "ghost"),
        # actions=True: the rail's tree carries a pin per note. The graph
        # frame renders the same tree without one, because a pin there would
        # point at a rail that document does not have.
        "tree": render_tree(nodes, actions=True),
        "hubs": _top_hubs(nodes, edges),
        # What the vault is ABOUT, as opposed to how big it is. The chat's
        # landing states this in one line, and it must be a fact rather than a
        # generated sentence: these are the co-occurrence topic labels of the
        # largest structural communities, the same ones the graph HUD keys its
        # colours on. Singletons are dropped — a one-note community names
        # itself, which says nothing about the corpus.
        "topics": [
            {"label": c.label, "size": c.size}
            for c in communities if c.size > 1
        ][:6],
    }


def _top_hubs(nodes: list[dict], edges: list[dict], top_n: int = 24) -> list[dict]:
    """Best-connected notes by resolved-link degree — the map view's landing
    picker (a radial map must be rooted on one note, so 'most central' is the
    sensible entry point). Ghost/unlinked nodes are skipped."""
    from collections import Counter

    deg: Counter = Counter()
    for e in edges:
        if e.get("type") == "EXTRACTED":
            deg[e.get("from")] += 1
            deg[e.get("to")] += 1
    hubs = [
        {"name": n.get("label") or (n.get("path") or "").rsplit("/", 1)[-1],
         "path": n["path"], "degree": deg[n["id"]]}
        for n in nodes
        if n.get("type") != "ghost" and n.get("path") and deg[n["id"]] > 0
    ]
    hubs.sort(key=lambda h: (-h["degree"], h["name"].lower()))
    return hubs[:top_n]


# --- the vault brief ---------------------------------------------------------
# Two readings of "what is this folder", stacked on the chat's landing. The
# counted one is always true and always free, and is rendered from /vault_info
# in the browser. This is the other one: a sentence a model writes over those
# same counts, which is a convenience and not a fact, so it is a separate
# request the landing can render without and a toggle can switch off.
#
# It caches into the vault's index dir rather than the vault: Silica's own
# bookkeeping never lands in a folder the user reads.

def _brief_path() -> Path:
    from silica.kernel.recall import paths

    return paths.index_dir() / "vault_brief.json"


@app.get("/vault_brief")
def vault_brief(refresh: int = 0):
    """One sentence about what the vault holds, written by the worker model.

    `stamp` is the corpus shape the sentence was written against, so a vault
    that grew gets a new sentence and a vault that only sat there replays the
    one on disk. A failure here is never an error on the landing: the counted
    line above it already answered the question.
    """
    if not CONFIG.vault_brief:
        return {"enabled": False, "text": ""}

    from silica.kernel.recall.graph_export import build_graph_data, detect_communities

    try:
        nodes, edges = build_graph_data(folder="")
        communities = detect_communities(nodes, edges)
    except Exception as exc:
        return {"enabled": True, "text": "", "error": str(exc)}

    notes = sum(1 for n in nodes if n.get("type") != "ghost")
    topics = [c.label for c in communities if c.size > 1][:8]
    stamp = f"{notes}|" + "|".join(topics)

    cache = _brief_path()
    if not refresh:
        try:
            got = json.loads(cache.read_text(encoding="utf-8"))
            if got.get("stamp") == stamp and got.get("text"):
                return {"enabled": True, "text": got["text"], "cached": True}
        except Exception:
            pass  # no cache, unreadable cache, or a stamp from an older shape

    hubs = [h["name"] for h in _top_hubs(nodes, edges, top_n=12)]
    text = _write_brief(notes, topics, hubs)
    if not text:
        return {"enabled": True, "text": ""}
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"stamp": stamp, "text": text}), encoding="utf-8")
    except Exception:
        pass  # an uncacheable brief is still a usable brief
    return {"enabled": True, "text": text}


def _write_brief(notes: int, topics: list[str], hubs: list[str]) -> str:
    """Ask the worker model for one sentence over evidence it cannot embellish.

    It is handed labels and note names, never note bodies: the sentence has to
    say what the collection is about, and a corpus of 1,200 notes has no
    summary that fits in a landing line anyway. Trimmed hard on the way out —
    a model asked for one sentence returns three often enough that the landing
    cannot be built on trusting it.
    """
    if not topics and not hubs:
        return ""
    from silica.agent.providers import get_provider

    prompt = (
        "Below are the topic labels of the largest link clusters in a personal "
        "markdown vault, and the names of its best-connected notes.\n\n"
        f"Clusters: {', '.join(topics) or 'none'}\n"
        f"Best-connected notes: {', '.join(hubs) or 'none'}\n"
        f"Total notes: {notes}\n\n"
        "Write ONE sentence, at most 24 words, naming what this collection is "
        "about. Describe only what the labels above support. No preamble, no "
        "quotes, no note counts, no adjectives of praise. Reply with the "
        "sentence and nothing else."
    )
    try:
        provider = get_provider(CONFIG, role="worker")
        reply = provider.call_llm(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            response_schema=None,
            max_tokens=200,
        )
        out = (reply.text or "").strip().strip('"').split("\n")[0].strip()
    except Exception as exc:
        logger.info("vault brief unavailable: %s", exc)
        return ""
    return out[:240]


@app.get("/messages")
def get_messages():
    resolve = note_resolver()
    # A call whose result carries an error must not replay as a tick: the one
    # place the user checks whether a write landed is this transcript.
    failed = {
        m["tool_call_id"] for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id") and _is_tool_failure(m.get("content"))
    }
    # Tool results, which the loop below skips over: a nucleate run's outcome
    # (notes, links, which chunks died and where) exists nowhere else, so
    # without this a reloaded chat could only say the injector had run.
    results = {
        m["tool_call_id"]: m.get("content") or ""
        for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    data = []
    for m in messages:
        if m.get("role") not in ("user", "assistant"):
            continue
        tools = tool_calls_to_json(m, failed, results) if m["role"] == "assistant" else []
        content = m.get("content") or ""
        # The thinking that produced this step, kept out of the wire by _to_wire.
        # Plain text, not rendered: it is a trace, and the live block shows it raw.
        thinking = m.get("silica_reasoning") or "" if m["role"] == "assistant" else ""
        if not content and not tools and not thinking:
            continue
        data.append({"role": m["role"], "content": content, "tools": tools,
                     "thinking": thinking,
                     "html": _linkify(content, resolve) if content else ""})
    # Vault label + context usage ride headers so the body stays a plain list.
    # The breakdown goes as JSON in one header rather than three: it is one
    # reading of one number, and splitting it invites a client to read two of
    # the three and draw a ring that does not close.
    meter = _ctx_meter()
    return JSONResponse(data, headers={
        "X-Silica-Vault": CONFIG.vault_path or "",
        "X-Silica-Context-Tokens": str(meter["context_tokens"]),
        "X-Silica-Max-Context-Tokens": str(meter["max_context_tokens"]),
        "X-Silica-Context-Parts": json.dumps(meter["context_parts"]),
        "X-Silica-Compact-At": str(meter["compact_at"]),
    })


@app.get("/narration")
def narration_replay(from_seq: int = 0, sid: str = ""):
    """Replay from a cursor — record zero by default: a joining client that
    replays everything cannot see a span whose opening it missed (ticket 08)."""
    from silica.agent import narration as _narr
    target = sid or current_session_id
    if not target or not str(target).isalnum():
        return JSONResponse({"sid": None, "beats": []})
    path = _narr.narration_dir() / f"{target}.jsonl"
    return JSONResponse({"sid": target,
                         "beats": list(_narr.read_beats(path, from_seq=from_seq))})


# The GUI's own uvicorn server, so an endless response can ask whether the
# process is leaving. None under TestClient and in the suite, where nothing is
# shutting down and every stream is read to its end by its caller.
_SERVER = None
_SSE_POLL_S = 0.25   # four wakeups a second per open tab, against a 1s deadline


def _stopping() -> bool:
    return _SERVER is not None and bool(_SERVER.should_exit)


@app.get("/narration/sse")
async def narration_sse(request: Request, from_seq: int = 0):
    """Live beats. The SSE `id:` field is the seq, so the browser's own
    Last-Event-ID header is the reconnect cursor — no custom client state."""
    from silica.agent import narration as _narr
    from silica.agent.bus import BUS

    last = request.headers.get("last-event-id")
    cursor = int(last) if last and last.isdigit() else from_seq
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def on_beat(rec: dict) -> None:
        try:
            loop.call_soon_threadsafe(q.put_nowait, rec)
        except RuntimeError:
            pass   # loop closed: the client is gone, the beat is moot

    BUS.subscribe(_narr.BEAT_TOPIC, on_beat)

    async def gen():
        try:
            sid = current_session_id
            if sid:
                path = _narr.narration_dir() / f"{sid}.jsonl"
                for rec in _narr.read_beats(path, from_seq=cursor):
                    yield (f"event: beats\nid: {rec['seq']}\n"
                           f"data: {json.dumps({'sid': sid, 'beats': [rec]}, default=str)}\n\n")
            while True:
                # Not a plain `await q.get()`: this body never returns on its
                # own, and uvicorn's shutdown waits for every open response, so
                # one live tab held Ctrl+C forever (measured 2026-08-23: 0.5s to
                # exit with no stream, never with one). The poll lets the stream
                # end itself while the server is still waiting politely, which
                # is what keeps the exit free of the 40-line ASGI traceback that
                # a cancelled connection task prints.
                try:
                    rec = await asyncio.wait_for(q.get(), _SSE_POLL_S)
                except asyncio.TimeoutError:
                    if _stopping():
                        return
                    continue
                yield (f"event: beats\nid: {rec.get('seq')}\n"
                       f"data: {json.dumps({'sid': rec.get('sid'), 'beats': [rec]}, default=str)}\n\n")
        finally:
            BUS.unsubscribe(_narr.BEAT_TOPIC, on_beat)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/sessions")
def list_sessions():
    # Current id rides a header so the body stays a plain list (matches /messages).
    return JSONResponse(_list_sessions(), headers={"X-Silica-Session": current_session_id or ""})


@app.post("/session/load", dependencies=_SAME_ORIGIN)
def load_session(payload: dict):
    global current_session_id, _collapsed
    if _busy:
        raise HTTPException(status_code=409, detail="a turn is already in progress")
    from silica.cli import _update_context_tokens
    from silica.agent import narration as _narr

    sid = str(payload.get("id", ""))
    if not sid.isalnum():  # ids are hex — blocks path traversal
        raise HTTPException(status_code=404, detail="no such session")
    replayed = _narr.load_session_messages(sid, CONFIG.vault_path or "")
    if replayed is None:
        raise HTTPException(status_code=404, detail="no such session")
    if (_narr.narration_dir() / f"{sid}.jsonl").exists():
        try:
            _narr.NARRATOR.resume(sid)   # continue appending to the same account
        except _narr.SessionBusy as e:
            raise HTTPException(status_code=409, detail=str(e)) from None
        current_session_id = sid
    else:
        # Legacy snapshot: continue as a NEW narration session seeded with its
        # turns — emit new, recognise legacy (ticket 05).
        _narr.NARRATOR.close()
        current_session_id = _narr.NARRATOR.ensure_session(driver="gui")
        for m in replayed:
            _narr.NARRATOR.turn(m)
    messages[:] = replayed
    _collapsed = set()
    _update_context_tokens(messages)
    return {"ok": True}


@app.post("/reset", dependencies=_SAME_ORIGIN)
def reset():
    _capture_own_session()
    _reset_session()
    return {"ok": True, "vault": CONFIG.vault_path}


@app.post("/stop", dependencies=_SAME_ORIGIN)
def stop():
    if current_cancel is not None:
        current_cancel.set()
        from silica.agent import narration as _narr
        _narr.NARRATOR.cancel(driver="gui", target=None, scope="turn")
    return {"ok": True}


@app.get("/health")
def health(all: bool = False):
    """The doctor's findings — non-ok by default, everything with `?all=1`.

    A server the user forgot to start degrades recall silently here: the
    embedder/reranker warnings go to the launching terminal's stderr, which the
    browser never shows. Same checks as `silica doctor`, so the two surfaces
    cannot disagree; ok rows are dropped for the sidebar notice, which is for
    what needs fixing, and kept for the settings panel's diagnostics and for a
    bug report, which needs the passing rows just as much.
    """
    from silica.onboarding.checks import run_checks

    # "session capture" tells the user to edit .claude/settings.json — a
    # Claude-Code-integration concern the browser can do nothing about. It
    # stays out of the sidebar notices; ?all=1 (diagnostics) keeps it.
    return [
        {"name": r.name, "status": r.status, "detail": r.detail, "hint": r.hint}
        for r in run_checks(CONFIG)
        if all or (r.status != "ok" and r.name != "session capture")
    ]


# 60s of 16 kHz mono 16-bit PCM is 1.92 MB; the cap is generous enough for the
# recorder's own ceiling and still refuses a body that was never a clip.
_STT_MAX_BYTES = 8 * 1024 * 1024


@app.get("/stt")
def stt_status():
    """Whether dictation can work — asked before the browser requests the mic.

    A probe, not a config flag: stt_base_url has a default, so "configured" and
    "listening" are different questions and only the second one is useful. Shares
    ensure_local_servers' readiness check, which knows that llama.cpp-family
    servers answer 503 while they load and that an open port therefore lies.
    """
    from silica.onboarding.serve import ready

    url = CONFIG.stt_base_url
    if not url:
        return {"ok": False, "url": "", "detail": "SILICA_STT_BASE_URL is empty"}
    if ready(url):
        return {"ok": True, "url": url, "detail": ""}
    return {"ok": False, "url": url, "detail": f"nothing is answering at {url}"}


@app.post("/stt", dependencies=_SAME_ORIGIN)
async def stt(audio: UploadFile = File(...)):
    """Proxy one recorded clip to the transcription endpoint.

    The browser sends 16 kHz mono WAV. MediaRecorder can only produce webm/opus,
    and whisper.cpp's server reads WAV unless it was built with ffmpeg, so the
    conversion happens in app.js, where it costs no dependency on either side.
    """
    import httpx

    if not CONFIG.stt_base_url:
        raise HTTPException(503, "no transcription endpoint configured")
    clip = await audio.read()
    if not clip:
        raise HTTPException(400, "empty recording")
    if len(clip) > _STT_MAX_BYTES:
        raise HTTPException(413, f"recording over {_STT_MAX_BYTES // (1024 * 1024)} MB")
    form = {"model": CONFIG.stt_model, "response_format": "json"}
    if CONFIG.stt_lang:
        form["language"] = CONFIG.stt_lang
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{CONFIG.stt_base_url.rstrip('/')}/audio/transcriptions",
                files={"file": ("clip.wav", clip, "audio/wav")},
                data=form,
                headers={"Authorization": f"Bearer {CONFIG.stt_api_key}"},
            )
    except Exception as exc:
        raise HTTPException(502, f"transcription endpoint unreachable: {exc}") from exc
    if resp.status_code != 200:
        raise HTTPException(502, f"transcription failed ({resp.status_code}): {resp.text[:200]}")
    try:
        text = (resp.json().get("text") or "").strip()
    except Exception as exc:
        raise HTTPException(502, f"transcription endpoint returned no JSON: {exc}") from exc
    return {"text": text}


@app.get("/config")
def get_config():
    """Session config for the header panel: the active model (read-only — Silica
    has no runtime model-switch op, so this mirrors the TUI's display-only
    /model) plus the one live toggle the web surfaces, thinking (/thinking)."""
    from silica.agent.providers import model_limits

    window = 0
    if CONFIG.model:
        window, _ = model_limits(CONFIG.provider, CONFIG.model)
    return {
        "model": CONFIG.model or "",
        # Empty means "the chat model does the worker's job too" (every call site
        # falls back to CONFIG.model), and the reader decides what to do with it.
        "worker_model": CONFIG.worker_model or "",
        "provider": CONFIG.provider or "",
        "context_window": window or 0,
        "show_thinking": CONFIG.show_thinking,
    }


# --- settings panel ----------------------------------------------------------
# The write half of /config is absorbed here: `thinking` is a persisted row like
# any other now, not a session-only flip. /config stays as the header chip's
# cheap read — GET /settings probes four endpoints, which is seconds the header
# label must not wait for.


@app.get("/settings")
def get_settings():
    """Every admitted row: value, where the value came from, whether it is
    locked, and its suggestions. Plus what About needs, so opening the panel is
    one round trip."""
    from silica import __version__
    from silica.onboarding.wizard import resolve_env_path
    from silica.ui.web import settings as st
    from silica.update import behind_count

    return {
        "env_path": st.short_path(resolve_env_path()),
        "busy": _busy,
        "sections": st.read_sections(),
        "version": __version__,
        "behind": behind_count(),
        "issues_url": "https://github.com/kiycoh/silica-harness/issues",
    }


def _reject_if_busy_or_locked(key: str) -> None:
    """One rule for every write: nothing lands while a turn is running.

    Deliberately not a per-row list of what a turn reads — that list would rot
    at the first new tool and no test would catch it. The cost is waiting for a
    response to finish.
    """
    from silica.ui.web import settings as st

    if _busy:
        raise HTTPException(status_code=409, detail="a response is running")
    if st.locked(key):
        raise HTTPException(status_code=409, detail=f"defined in the environment ({key})")


@app.post("/settings", dependencies=_SAME_ORIGIN)
def set_setting(payload: dict):
    """Apply one row: live in CONFIG, persisted in the .env that wins at boot."""
    from silica.ui.web import settings as st

    key = str(payload.get("key", ""))
    _reject_if_busy_or_locked(key)
    if key == st.VAULT_KEY or key in st.EMBED_KEYS:
        raise HTTPException(status_code=400, detail="this row goes through /settings/confirm")
    result = st.apply(key, payload.get("value", ""))
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/settings/confirm", dependencies=_SAME_ORIGIN)
async def confirm_setting(payload: dict):
    """The two rows that need a sequence, not an assignment: switching the vault
    and swapping the embedding model.

    Both are long enough to block, so they run off the event loop. The embedding
    swap does not repair itself: `sweep()` decides what to re-embed from mtimes,
    which a model change never touches, so stale vectors would be compared in an
    incompatible space in silence until something forces a full re-index.
    """
    from silica.ui.web import settings as st

    key = str(payload.get("key", ""))
    value = str(payload.get("value", ""))
    _reject_if_busy_or_locked(key)
    if key == st.VAULT_KEY:
        return await asyncio.to_thread(_apply_vault_switch, value)
    if key in st.EMBED_KEYS:
        return await asyncio.to_thread(_apply_embedding_swap, key, value)
    raise HTTPException(status_code=400, detail="this row does not need confirming")


def _apply_vault_switch(path: str) -> dict:
    from silica.cli import switch_vault
    from silica.ui.web import settings as st

    switched = switch_vault(path)
    if switched.error:
        return {"ok": False, "error": switched.error}
    # The Changes list describes paths in the vault we just left.
    session_changes.clear()
    # The resolved absolute path, not what was typed: that is what the next boot
    # must read back, and what the caches were just rebuilt for.
    result = st.apply(st.VAULT_KEY, switched.vault)
    # The fresh-session seed carries the old vault's map until it is rebuilt.
    _prewarm_seed()
    notes = []
    if switched.write_dir:
        notes.append(f"writes confined to {switched.write_dir}/")
    if switched.invalid_write_dir:
        notes.append("vault.yaml declares an invalid write_dir — every write will be rejected")
    if switched.repo_warning:
        notes.append(switched.repo_warning)
    if switched.language_drift:
        notes.append(
            f"language {switched.language}, co-occurrence store frozen "
            f"{switched.store_language} — rebuild it with /cooccur --force"
        )
    return {**result, "vault": switched.vault, "notes": notes}


def _apply_embedding_swap(key: str, value: str) -> dict:
    from silica.tools import TOOLS
    from silica.ui.web import settings as st

    result = st.apply(key, value)
    if not result["ok"]:
        return result
    raw = TOOLS["silica_embed_refresh"].run(folder="", force=True)
    try:
        report = json.loads(raw)
    except (TypeError, ValueError):
        report = {"result": str(raw)[:200]}
    return {**result, "reindex": report}


@app.get("/bug_report")
def get_bug_report():
    """The diagnostic block a bug report attaches. Built server-side on purpose:
    in the browser the API keys sit in the panel's own fields, and a public issue
    is exactly the wrong place for one."""
    from silica.ui.web import settings as st

    return st.bug_report()


@app.get("/endpoints")
def get_endpoints():
    from silica.ui.web import settings as st

    return st.endpoint_status()


@app.post("/endpoints/start", dependencies=_SAME_ORIGIN)
async def start_endpoint(payload: dict):
    """Start one local endpoint from the command its own .env key names. Loading
    a model takes tens of seconds, so this waits on a worker thread."""
    from silica.ui.web import settings as st

    return await asyncio.to_thread(st.start_endpoint, str(payload.get("label", "")))


@app.get("/")
def index():
    # Cache-bust the three churning assets by content hash: StaticFiles sets no
    # Cache-Control, so browsers serve them stale from heuristic freshness
    # (edited JS never reaches the page). A content-keyed URL can't be stale.
    # The big vendored bundles keep their long-lived cache — only these churn.
    import hashlib

    from silica.config import CONFIG

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    # work.js belongs here as much as the other two: it owns the whole node
    # panel and churns with it, and without a version an edit to it reaches
    # nobody who has already loaded the page once.
    for asset in ("app.js", "app.css", "work.js"):
        ver = hashlib.sha256((STATIC_DIR / asset).read_bytes()).hexdigest()[:8]
        html = html.replace(f"/static/{asset}", f"/static/{asset}?v={ver}")
    # The preference, not the resolution: "auto" is a question only the browser
    # can answer, and the inline script in <head> answers it before first paint.
    # Stamping it server-side is what keeps a light session from flashing dark.
    pref = CONFIG.theme if CONFIG.theme in ("auto", "dark", "light") else "auto"
    html = html.replace('data-theme-pref="auto"', f'data-theme-pref="{pref}"')
    return HTMLResponse(html)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def serve(port: int = 8765) -> None:
    """Apply config, open the browser on startup, then block on uvicorn."""
    import uvicorn

    from silica.ui.banner import print_banner
    from silica.ui.console import CONSOLE

    _reset_session()

    print_banner()
    CONSOLE.print(f"  [dim]GUI live at[/] [cyan]http://127.0.0.1:{port}[/]\n")

    global _SERVER
    # Built here rather than through uvicorn.run(), which keeps the Server to
    # itself: /narration/sse polls should_exit to end itself on the way out.
    # timeout_graceful_shutdown is the backstop under it, for the one stream
    # that does not poll - an in-flight /chat turn, which carries its own cancel
    # path and would otherwise hold Ctrl+C for the length of the turn.
    _SERVER = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, timeout_graceful_shutdown=1))
    try:
        try:
            _SERVER.run()
        except KeyboardInterrupt:
            # uvicorn re-raises the signal it captured, AFTER it has shut down
            # cleanly, and `silica` is installed as cli:main - so the module's
            # own __main__ guard never runs and nothing above here catches it.
            # Without this a tidy Ctrl+C ends in a traceback that reads as a
            # crash. The shutdown already happened; there is nothing to salvage.
            pass
    finally:
        _capture_own_session()  # last chance: this conversation ends with the server
