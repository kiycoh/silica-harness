# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`web_fetch` — read one URL and return its text. No third party in the path.

ADR-0015 staged acquisition: Silica may fetch on request but never decides what
enters the vault. This module returns text to its caller (the /web-search
research loop, or /fetch); nothing here writes to the vault.

`web_fetch` is `sensitive=True` (ADR-0009): the main agent's default toolset
excludes it, so it is reachable only where it is named explicitly in
AgentConstraints, or called directly by a command.

Direct httpx plus stdlib html.parser, no Jina and no trafilatura: a third-party
reader puts every fetched URL in front of someone else, which contradicts the
local-first posture. The price of dropping it is that SSRF becomes ours, and
`_validated` is that price paid in full.
"""
from __future__ import annotations

import ipaddress
import json
import re
import shutil
import socket
import subprocess
import tempfile
import time
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlencode, urlsplit

import httpx
from pydantic import BaseModel

from silica.tools import tool

# ~7.5k tokens of a 60k default context budget. A LinkedIn guest page is 374 KB
# raw; without this ceiling one fetch eats the window.
_MAX_CHARS = 30_000
_MAX_REDIRECTS = 3
_HTTP_TIMEOUT = 30
# _MAX_CHARS bounds what reaches the model; this bounds what reaches memory,
# which nothing did before: a buffered GET materialises the whole body (and then
# a whole str) before any ceiling of ours runs, so one hostile or merely enormous
# URL OOMs the process while `_truncate` waits its turn. ~270x the char ceiling,
# so no real page meets it.
_MAX_BODY_BYTES = 8 * 1024 * 1024
# `timeout` is per socket operation, not a deadline: an endless stream dripping
# one byte before each timeout window never trips it and never ends, so the body
# gets its own wall clock.
_BODY_DEADLINE = 120
_YT_DOMAINS: tuple[str, ...] = ("youtube.com", "youtu.be")
# Languages only: `--sub-lang` does not choose between auto-generated and
# uploaded subs (that is `--write-auto-sub` / `--write-sub`, and we ask for
# both). `en.*` already matches bare `en`, so no separate entry is needed.
# Which of the downloaded files wins is alphabetical, not preferential:
# `sorted(glob("*.vtt"))[0]`, where `sub.en-GB.vtt` beats `sub.en.vtt` because
# `-` (0x2D) sorts before `.` (0x2E).
_YT_SUB_LANGS = "en.*,it.*"
_YT_TIMEOUT = 120
_WP_DOMAINS: tuple[str, ...] = ("wikipedia.org",)
_WP_PREFIX = "/wiki/"
_WP_API_PATH = "/w/api.php"
_WP_MATH_MAXLEN = 3
# Wikimedia's user-agent policy asks for a descriptive agent with a contact URL
# and throttles generic browser strings, so the API branch does not send
# _HEADERS. Their own docs call the browser-string case "discouraged".
_WP_HEADERS = {
    "User-Agent": "silica-harness (https://github.com/kiycoh/silica-harness)",
    "Accept": "application/json",
}
# A bare httpx user agent collects more 403s than a browser string does, and
# the 401/403/429 branch below is how we surface the ones that remain.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
}
_TEXT_TYPES = ("text/", "application/xhtml", "application/xml", "application/json")

_SCHEMES = ("http", "https")


def host_matches(url: str, *domains: str) -> bool:
    """True when `url` is http(s), carries no userinfo, and its host is one of
    `domains` or a subdomain of one.

    Anchored on a leading dot, so `x.com.evil.test` (a substring match would
    pass) and `x.com@evil.test` (userinfo disguise) both come back False.
    Malformed ports only raise when `.port` is touched, so touch it.
    """
    try:
        parts = urlsplit(url)
        _ = parts.port
    except ValueError:
        return False
    if parts.scheme.lower() not in _SCHEMES:
        return False
    if parts.username is not None or parts.password is not None:
        return False
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return False
    return any(
        host == d or host.endswith("." + d) for d in (x.lower() for x in domains)
    )


def _validated(url: str) -> None:
    """Fail closed on anything we should not open a socket to.

    Rejects non-HTTP schemes, embedded credentials, and any hostname that
    resolves to a non-global address. `ipaddress.is_global` is the single
    primitive that covers loopback, RFC1918, link-local (including
    169.254.169.254, the cloud metadata endpoint), CGNAT and unique-local IPv6.
    Every resolved address must pass: one global answer cannot launder a
    private sibling.

    ponytail: residual TOCTOU. httpx resolves the hostname again after this
    check, so a hostile DNS server could rebind between the two. Closing it
    means pinning the resolved IP into the transport and setting the Host
    header by hand; build that only if Silica ever fetches attacker-supplied
    URLs unattended. Today they come from Tavily results or from the user.
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as e:
        raise ValueError(f"malformed URL {url!r}: {e}") from e
    if parts.scheme.lower() not in _SCHEMES:
        raise ValueError(f"refusing non-HTTP URL: {url!r}")
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"refusing URL with embedded credentials: {url!r}")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError(f"URL has no host: {url!r}")
    default_port = 443 if parts.scheme.lower() == "https" else 80
    try:
        infos = socket.getaddrinfo(host, port or default_port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve {host!r}: {e}") from e
    if not infos:
        # An empty answer means we checked nothing, so the loop below approves
        # everything. The one function whose job is failing closed must not have
        # a branch that fails open.
        raise ValueError(f"cannot resolve {host!r}: no addresses returned")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ValueError(f"refusing non-global address {ip} for host {host!r}")


_SKIP_TAGS = frozenset({
    "script", "style", "noscript", "template", "svg", "iframe",
    "nav", "header", "footer", "form", "aside",
})
# `pre` is handled on its own, not here: it fences instead of breaking.
_BREAK_TAGS = frozenset({
    "p", "div", "br", "li", "tr", "td", "th", "section", "article",
    "blockquote", "title", "h1", "h2", "h3", "h4", "h5", "h6",
})
_FENCE = "```"


class _TextExtractor(HTMLParser):
    """Collect visible text, skipping boilerplate containers.

    ponytail: stdlib html.parser, not trafilatura. Measured on four real pages
    both cut raw HTML by 10x to 16x and trafilatura is only 1.05x to 1.5x
    tighter; lxml plus trafilatura is a heavy transitive tree for roughly 30%
    fewer boilerplate tokens. Revisit if that boilerplate measurably pollutes
    nucleated notes.

    Second ceiling: html.parser treats `<script>` and `<style>` as CDATA, so an
    unclosed or truncated one swallows every byte after it with no error and
    no truncation marker. A real HTML5 tokenizer (lxml/html5lib) recovers from
    unclosed CDATA where html.parser cannot; that recovery is the concrete
    reason to pay for that dependency, if this ever bites on real pages.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._last_alt = ""
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "pre":
            self._fence()
        elif tag == "img":
            self._image_alt(attrs)
        elif tag in _BREAK_TAGS:
            self.parts.append("\n")

    def _image_alt(self, attrs) -> None:
        """Emit an image's alt text, marked as an image.

        This parser reads no attributes anywhere else, so until now a fetched
        page arrived with its figures simply absent. On a technical page the alt
        is frequently the only description of a diagram that exists in the
        markup, which makes it the difference between a figure being in the vault
        and not.

        Marked `[image: ...]`, not inlined as prose the way omniparse inlines it:
        an alt is a caption, and a caption indistinguishable from body text is a
        small lie in a note somebody later quotes. Adjacent-equal dedup drops the
        icon rows that repeat one alt down a page; requiring a letter drops the
        `alt="***"` spacer. Decorative one-word alts still ride through, which
        the marker makes cheap: a reader skips a bracketed line.
        """
        if self._skip:
            return
        alt = " ".join((dict(attrs).get("alt") or "").split())
        if not alt or not any(c.isalpha() for c in alt) or alt == self._last_alt:
            return
        self._last_alt = alt
        self.parts.append(f"\n[image: {alt}]\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            # clamped: real pages ship stray close tags, and a negative counter
            # would swallow everything after one
            self._skip = max(0, self._skip - 1)
        elif tag == "pre":
            self._fence()
        elif tag in _BREAK_TAGS:
            self.parts.append("\n")

    def _fence(self) -> None:
        """Open or close a markdown code fence around a `<pre>` block.

        Suppressed inside a skipped container: a `<pre>` in a nav or an aside
        contributes no text, and a lone unpaired fence would turn the whole rest
        of the page into one code block.
        """
        if not self._skip:
            self.parts.append(f"\n{_FENCE}\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _extract_text(html: str) -> str:
    """HTML to readable plain text: drop boilerplate, collapse whitespace.

    Whitespace collapsing stops inside a fence. A `<pre>` block that keeps its
    fence but loses its indentation is half a job: the page's code arrives as
    prose either way. Lines between fences keep their leading whitespace.

    ponytail: fence state is tracked by counting our own emitted ``` lines, so
    a page whose *prose* contains a bare ``` line (a markdown document served as
    HTML) flips it. The alternative is threading a nesting depth out of the
    parser for a case that inverts one block's formatting, not the note.
    """
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines: list[str] = []
    in_code = False
    just_opened = False
    for raw in "".join(parser.parts).splitlines():
        if raw.strip() == _FENCE:
            if in_code and lines and not lines[-1]:
                # the source newline before `</pre>`, which every real page has;
                # a blank line is never significant at the end of a code block
                lines.pop()
            in_code = not in_code
            just_opened = in_code
            lines.append(_FENCE)
            continue
        if just_opened and not raw.strip():
            # browsers drop the newline right after `<pre>`, and real HTML
            # nearly always has one; keeping it opens every block with a blank
            just_opened = False
            continue
        just_opened = False
        line = raw.rstrip() if in_code else " ".join(raw.split())
        # keep at most one blank line between blocks, and none at the top;
        # inside a fence every line is significant, blanks included
        if line or in_code or (lines and lines[-1]):
            lines.append(line)
    if in_code:
        # truncated page, or a stray ``` in the prose: never hand the vault an
        # open fence, which renders every line after it as code
        lines.append(_FENCE)
    return "\n".join(lines).strip()


def _truncate(text: str, limit: int = _MAX_CHARS) -> str:
    """Hard ceiling with a visible marker, so the model knows it saw a prefix."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    # An odd fence count means the cut landed inside a code block. Close it, or
    # the truncation marker (and, in the vault, nothing else) renders as code.
    if sum(1 for ln in cut.splitlines() if ln.strip() == _FENCE) % 2:
        cut = f"{cut}\n{_FENCE}"
    return f"{cut}\n\n[truncated at {limit} characters]"


_MAX_TITLE = 120
# Separators real pages put between a page title and their own site name.
_TITLE_SEPS = (" | ", " - ", " – ", " — ", " :: ", " · ", " » ")


class _TitleExtractor(HTMLParser):
    """`<title>`, `og:title`, `twitter:title` and `og:site_name`, nothing else.

    Structured harvest, because the alternative in use was positional: take the
    first line of the extracted text and hope it is the title. That hope fails
    on every page whose text path is not the rendered HTML (the Wikipedia API
    extract opens with the lead sentence, a YouTube transcript with the first
    thing said) and on every page that has no `<title>` at all.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.og_title = ""
        self.twitter_title = ""
        self.site_name = ""
        self._in_title = False
        self._seen_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title" and not self._seen_title:
            # first one wins: `<svg><title>` is an accessibility label, and it
            # comes after the real one in the head
            self._in_title = True
            self._seen_title = True
        elif tag == "meta":
            a = dict(attrs)
            # `property` is the OG spec, `name` is what half the web ships
            # instead; twitter cards only ever use `name`
            key = (a.get("property") or a.get("name") or "").lower()
            content = (a.get("content") or "").strip()
            if not content:
                return
            if key == "og:title" and not self.og_title:
                self.og_title = content
            elif key == "twitter:title" and not self.twitter_title:
                self.twitter_title = content
            elif key == "og:site_name" and not self.site_name:
                self.site_name = content

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


def _strip_site_suffix(title: str, site_name: str) -> str:
    """Drop a trailing `<separator><site name>` when the page names its own site.

    Only a suffix we can positively identify is removed. Splitting on the
    separator alone would eat the back half of "The Pragmatic Programmer - 20th
    Anniversary Edition", and a truncated title is no better for the reranker
    than a polluted one.

    ponytail: a page that omits `og:site_name` keeps its suffix. If polluted
    titles show up in the vault, derive the candidate site name from the host,
    never from the separator position.
    """
    if not site_name:
        return title
    for sep in _TITLE_SEPS:
        tail = sep + site_name
        if title.endswith(tail):
            return title[: -len(tail)].strip() or title
    return title


def page_title(html: str) -> str:
    """The page's own title: `og:title`, else `twitter:title`, else `<title>`.

    OG first because it is authored for sharing and so usually arrives without
    the site-name suffix that `<title>` carries for SEO. Empty string when the
    page declares no title at all, which is the caller's cue to fall back.
    """
    parser = _TitleExtractor()
    parser.feed(html)
    parser.close()
    site_name = " ".join(parser.site_name.split())
    for candidate in (parser.og_title, parser.twitter_title, parser.title):
        title = " ".join(candidate.split())
        if title:
            return _strip_site_suffix(title, site_name)[:_MAX_TITLE]
    return ""


def _render(url: str, text: str) -> str:
    """Header line plus body.

    `Source:` carries the final URL after redirects, so a citation points at
    what was actually read, and web_research can lift it out of the tool trace.
    """
    return f"Source: {url}\n\n{text}"


def _raise_for_status(resp: httpx.Response, url: str) -> None:
    """401, 403 and 429 are the failures a direct fetcher actually meets, and
    they mean different things to the caller. Distinct messages, not one
    generic HTTPStatusError."""
    if resp.status_code in (401, 403):
        raise ValueError(
            f"{resp.status_code} at {url}: the site refuses unauthenticated "
            "reads (bot wall or paywall). Try a different source."
        )
    if resp.status_code == 429:
        raise ValueError(
            f"429 at {url}: rate limited. Back off, or use a different source."
        )
    resp.raise_for_status()


def content_type(resp: httpx.Response) -> str:
    """The bare media type, lowercased, without parameters."""
    return resp.headers.get("content-type", "").split(";")[0].strip().lower()


def _refuse_binary(resp: httpx.Response, url: str) -> None:
    """Decided on the headers, before a byte of the body is spent: a PDF or a
    video is refused either way, and refusing it after buffering it is the
    expensive way to reach the same answer."""
    ctype = content_type(resp)
    if ctype and not ctype.startswith(_TEXT_TYPES):
        raise ValueError(f"refusing to read {ctype} content at {url}")


def _decoded_headers(headers: httpx.Headers) -> httpx.Headers:
    """The response headers minus the ones that describe the wire body.

    `iter_bytes` hands back DECODED bytes, so carrying `Content-Encoding: gzip`
    onto the rebuilt response makes httpx decompress an already-decompressed
    body — every gzip site (i.e. most of the web) died on `.text` with
    "Error -3 while decompressing data: incorrect header check". Content-Length
    describes the compressed body too, and is equally a lie once decoded.
    """
    out = httpx.Headers(headers)
    for name in ("content-encoding", "content-length"):
        if name in out:
            del out[name]
    return out


def _read_capped(resp: httpx.Response, url: str) -> bytes:
    """Accumulate the body against a size ceiling and a wall clock, aborting on
    either. Content-Length is not consulted: it is the server's claim, and a
    chunked response makes none."""
    chunks: list[bytes] = []
    total = 0
    deadline = time.monotonic() + _BODY_DEADLINE
    for chunk in resp.iter_bytes():
        total += len(chunk)
        if total > _MAX_BODY_BYTES:
            raise ValueError(
                f"body at {url} passed {_MAX_BODY_BYTES} bytes; refusing to read on"
            )
        if time.monotonic() > deadline:
            raise ValueError(
                f"body at {url} still arriving after {_BODY_DEADLINE}s; giving up"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _fetch(url: str, headers: dict[str, str] = _HEADERS) -> tuple[httpx.Response, str]:
    """GET with redirects followed by hand, revalidating every hop, and a body
    read under a ceiling instead of buffered whole.

    Open WebUI validates the first URL and then hands it to a client that
    follows redirects itself, so a perfectly global URL can 302 into link-local
    space. Following them here closes that.

    Streamed, so the two cheap refusals (wrong content type, oversized body)
    happen while the body is still on the wire. The response handed back carries
    the bytes we accepted, so callers keep reading `.text` / `.json()`.
    """
    for _ in range(_MAX_REDIRECTS + 1):
        _validated(url)
        with httpx.stream(
            "GET", url, follow_redirects=False, timeout=_HTTP_TIMEOUT, headers=headers
        ) as resp:
            if resp.is_redirect and resp.next_request is not None:
                url = str(resp.next_request.url)
                continue
            _raise_for_status(resp, url)
            _refuse_binary(resp, url)
            body = _read_capped(resp, url)
            return (
                httpx.Response(
                    resp.status_code,
                    headers=_decoded_headers(resp.headers),
                    content=body,
                    request=resp.request,
                ),
                url,
            )
    raise ValueError(f"more than {_MAX_REDIRECTS} redirects, giving up at {url}")


def _strip_mathml(extract: str) -> str:
    """Drop the exploded MathML that `explaintext` leaves beside every formula.

    TextExtracts strips tags from the rendered `<math>` element, so a formula
    arrives twice: first its presentation MathML, one glyph per indented line
    (`P` / `R` / `(` / `E` / `)`), then the readable
    `{\\displaystyle PR(E).}`. Measured on PageRank, Transformer and Maxwell's
    equations the glyph lines are 40-47% of the extract, and those articles run
    60k to 122k chars against a 30k ceiling: the junk is being truncated in
    ahead of the prose.

    Indented AND at most three characters is the whole rule. Prose survives it
    (Wikipedia indents inline fragments by one space, MathML by two or more),
    and so do the indented pseudocode blocks, which a plain drop-every-indented
    -line rule ate. ponytail: about a dozen multi-glyph operator labels per
    article ("masked tokens", "otherwise") ride through. A longer cut buys 0.6%
    and starts eating real lines, so it stays at three.
    """
    return "\n".join(
        line
        for line in extract.splitlines()
        if not (line.startswith("  ") and len(line.strip()) <= _WP_MATH_MAXLEN)
    )


def _wikipedia_extract(url: str) -> tuple[str, str] | None:
    """Article plaintext and canonical title, from the MediaWiki API instead of
    the rendered page.

    Wikipedia is the one site where _extract_text is measurably poor. On
    /wiki/PageRank the HTML path returns 30k chars that stop 14 sections in and
    spend their tail on the footer navbox, while `prop=extracts` returns the
    whole 60k-char article and nothing else: no nav, no image captions, no
    navbox. Keyless, and the host carries the language, so it.wikipedia.org
    answers in Italian. Prose comes out identical to the HTML path minus the
    `[6]` reference markers; display math arrives doubled and
    `_strip_mathml` keeps the readable half.

    None means "not a plain article", and the caller reads the HTML instead:
    a query string (`?action=history`), an empty title, or a page the API has
    no extract for (Special:Random, a red link). Errors are not swallowed --
    a 429 from the API raises like any other fetch.

    The title comes back from the response, not from the path: with
    `redirects=1` the API answers /wiki/Page_rank with "PageRank", and that
    canonical form is what the note should be called.
    """
    parts = urlsplit(url)
    if parts.query or not parts.path.startswith(_WP_PREFIX):
        return None
    title = unquote(parts.path[len(_WP_PREFIX):])
    if not title:
        return None
    query = urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",  # pages as a list, not keyed by pageid
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "wiki",  # `== Heading ==`, pinned: the default
            "redirects": "1",           # /wiki/Page_rank -> PageRank
            "titles": title,
        }
    )
    api = f"{parts.scheme}://{parts.hostname}{_WP_API_PATH}?{query}"
    resp, _ = _fetch(api, headers=_WP_HEADERS)
    pages = resp.json().get("query", {}).get("pages") or [{}]
    extract = _strip_mathml(pages[0].get("extract", "")).strip()
    if not extract:
        return None
    return extract, " ".join(str(pages[0].get("title", "")).split())[:_MAX_TITLE]


class WebFetchArgs(BaseModel):
    url: str


class Page(NamedTuple):
    """What one fetch yielded: the rendered text, and the page's own title.

    `title` is "" when the source declares none, which is the caller's cue to
    fall back. It is deliberately not folded into `text`: the `Source: <url>`
    header is a parsed contract with web_research, and a second header line
    would move the body at every consumer that hardcodes that shape.
    """

    text: str
    title: str


def fetch_page(url: str) -> Page:
    """`web_fetch` plus the title, for callers that name a note after the page.

    The tool wrapper below drops the title; nothing else about the text differs,
    so both paths stay one code path.
    """
    if host_matches(url, *_YT_DOMAINS):
        return _youtube_transcript(url)
    if host_matches(url, *_WP_DOMAINS):
        # Cites the requested URL, not the redirect target: it is what resolves
        # in a browser, and the anchor in /wiki/PageRank#History survives it.
        article = _wikipedia_extract(url)
        if article:
            extract, title = article
            return Page(_render(url, _truncate(extract)), title)
    resp, final_url = _fetch(url)
    # the type gate already ran in _fetch, on the headers, before the body
    ctype = content_type(resp)
    # no charset sniffing, httpx already decoded from the header.
    body = resp.text
    is_html = "html" in ctype or not ctype
    text = _extract_text(body) if is_html else body
    # plain text and JSON carry no title element; "" says so rather than
    # promoting their first line to one
    return Page(_render(final_url, _truncate(text)), page_title(body) if is_html else "")


@tool(WebFetchArgs, cls="atomic", sensitive=True)
def web_fetch(url: str) -> str:
    """Read one web page and return its text, boilerplate stripped and
    truncated. Call this on a promising search result instead of guessing from
    its snippet. The first line is `Source: <url>`, which is what to cite."""
    return fetch_page(url).text


_VTT_TAG_RE = re.compile(r"<[^>]*>")
_VTT_NOISE = ("WEBVTT", "Kind:", "Language:", "NOTE ", "STYLE", "REGION")
# `00:01:02.500 --> 00:01:04.000`, hours optional (whisper.cpp omits them).
_VTT_CUE_RE = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)


def _cue_times(line: str) -> tuple[float, float] | None:
    """(start, end) in seconds for a VTT cue-timing line, else None."""
    m = _VTT_CUE_RE.search(line)
    if not m:
        return None
    h1, m1, s1, ms1, h2, m2, s2, ms2 = m.groups()
    start = int(h1 or 0) * 3600 + int(m1) * 60 + int(s1) + int(ms1.ljust(3, "0")) / 1000
    end = int(h2 or 0) * 3600 + int(m2) * 60 + int(s2) + int(ms2.ljust(3, "0")) / 1000
    return start, end


def vtt_to_text(vtt: str, paragraph_gap_s: float = 0.0) -> str:
    """VTT to plain lines: drop cue timings and inline markup, and collapse the
    rolling duplication auto-subs produce (each cue repeats the line before it).

    ponytail: adjacent-equal dedup, not a diff of overlapping cues. It clears
    the common rolling case; upgrade to longest-common-suffix trimming only if
    real transcripts come out visibly doubled.

    `paragraph_gap_s` > 0 inserts a blank line wherever the speaker paused for
    at least that long, which matters for a transcript headed into the vault: a
    text with no blank line anywhere is ONE paragraph, `_split_by_size` leaves an
    oversized paragraph whole, and a two-hour talk would land as a single inbox
    note whose concepts RECON caps at 40. A pause is the one paragraph boundary
    a transcript actually carries. 0 (the default, and the YouTube path) keeps
    the original single-newline join.
    """
    lines: list[str] = []
    prev_end: float | None = None
    for raw in vtt.splitlines():
        s = raw.strip()
        if not s or s.isdigit() or s.startswith(_VTT_NOISE):
            continue
        if "-->" in s:
            times = _cue_times(s)
            if paragraph_gap_s and times and prev_end is not None:
                if times[0] - prev_end >= paragraph_gap_s and lines and lines[-1]:
                    lines.append("")
            if times:
                prev_end = times[1]
            continue
        s = unescape(_VTT_TAG_RE.sub("", s)).strip()
        if s and (not lines or lines[-1] != s):
            lines.append(s)
    return "\n".join(lines)


def _youtube_transcript(url: str) -> Page:
    """Subtitles via yt-dlp, keyless.

    There is no shortcut worth trying: the watch page does carry
    `captionTracks`, but every `baseUrl` now returns HTTP 200 with 0 bytes in
    every format, because timedtext is gated behind a PO token. yt-dlp handles
    the token and player-client dance.

    Auto-subs are ASR output: transcription errors, no speaker labels. Anything
    nucleated from them inherits that noise, so the sources/ leaf matters more
    here than usual.

    ponytail: no installer and no doctor for one optional binary. A clear
    prescription at call time beats a health-check subsystem. If stale-venv
    shims start confusing users, the upgrade is Agent-Reach's probe taxonomy
    (missing / broken / timeout / error), not an installer.
    """
    exe = shutil.which("yt-dlp")
    if not exe:
        raise ValueError(
            "reading YouTube needs yt-dlp on PATH: install it with "
            '`python -m pip install -U "yt-dlp[default]"`. The watch page '
            "itself is a JavaScript shell with no transcript in the HTML."
        )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sub"
        proc = subprocess.run(
            [
                exe, "--skip-download", "--write-auto-sub", "--write-sub",
                "--write-info-json",
                "--sub-lang", _YT_SUB_LANGS, "--sub-format", "vtt",
                "--no-playlist", "--playlist-items", "1", "-o", str(out), "--", url,
            ],
            capture_output=True,
            text=True,
            timeout=_YT_TIMEOUT,
        )
        files = sorted(Path(tmp).glob("*.vtt"))
        if not files:
            tail = (proc.stderr or proc.stdout or "").strip()[-300:]
            raise ValueError(
                f"yt-dlp found no subtitles for {url}" + (f": {tail}" if tail else "")
            )
        text = vtt_to_text(files[0].read_text(encoding="utf-8", errors="replace"))
        title = _yt_info_title(Path(tmp))
    return Page(_render(url, _truncate(text)), title)


def _yt_info_title(tmp: Path) -> str:
    """The video's title from the sidecar `--write-info-json` writes.

    One extra flag on a subprocess we already run, and it is the only real title
    a transcript has: the alternative is the first thing said in the video.
    Best effort by design -- a missing or malformed sidecar costs the note its
    title, not the transcript.
    """
    for info in sorted(tmp.glob("*.info.json")):
        try:
            data = json.loads(info.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        title = " ".join(str(data.get("title") or "").split())
        if title:
            return title[:_MAX_TITLE]
    return ""
