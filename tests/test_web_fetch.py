"""web_fetch: URL guards, HTML extraction, the fetch loop, the YouTube branch.

No real network (httpx.stream and socket.getaddrinfo are monkeypatched) and no
real subprocess (subprocess.run is monkeypatched).
"""
from __future__ import annotations

import json

import pytest

from silica.sources import web_fetch as wf


# --- host_matches -----------------------------------------------------------

def test_host_matches_exact_and_subdomain():
    assert wf.host_matches("https://youtube.com/watch?v=a", "youtube.com")
    assert wf.host_matches("https://www.youtube.com/watch?v=a", "youtube.com")
    assert wf.host_matches("https://m.youtube.com/watch?v=a", "youtube.com")
    assert wf.host_matches("https://youtu.be/a", "youtube.com", "youtu.be")


def test_host_matches_is_case_and_trailing_dot_insensitive():
    assert wf.host_matches("https://YouTube.COM./watch", "youtube.com")


def test_host_matches_rejects_suffix_lookalike():
    # substring matching would pass this one
    assert not wf.host_matches("https://youtube.com.evil.test/a", "youtube.com")


def test_host_matches_rejects_userinfo_disguise():
    # urlsplit reads the real host as evil.test
    assert not wf.host_matches("https://youtube.com@evil.test/", "youtube.com")


def test_host_matches_rejects_non_http_scheme():
    assert not wf.host_matches("file:///etc/passwd", "etc")
    assert not wf.host_matches("ftp://youtube.com/a", "youtube.com")


def test_host_matches_rejects_malformed_port():
    assert not wf.host_matches("https://youtube.com:notaport/a", "youtube.com")


def test_host_matches_no_domains_is_false():
    assert not wf.host_matches("https://youtube.com/a")


# --- _validated -------------------------------------------------------------

def _resolves_to(monkeypatch, *ips: str):
    """Pin getaddrinfo so no test ever hits DNS."""
    def fake(host, port, *a, **kw):
        return [(2, 1, 6, "", (ip, port)) for ip in ips]
    monkeypatch.setattr(wf.socket, "getaddrinfo", fake)


def test_validated_accepts_a_global_address(monkeypatch):
    _resolves_to(monkeypatch, "93.184.216.34")
    wf._validated("https://example.com/a")  # must not raise


def test_validated_rejects_loopback(monkeypatch):
    _resolves_to(monkeypatch, "127.0.0.1")
    with pytest.raises(ValueError, match="non-global"):
        wf._validated("https://localhost.evil.test/")


def test_validated_rejects_cloud_metadata_endpoint(monkeypatch):
    _resolves_to(monkeypatch, "169.254.169.254")
    with pytest.raises(ValueError, match="non-global"):
        wf._validated("https://metadata.evil.test/")


def test_validated_rejects_rfc1918(monkeypatch):
    _resolves_to(monkeypatch, "10.0.0.5")
    with pytest.raises(ValueError, match="non-global"):
        wf._validated("https://intranet.evil.test/")


def test_validated_rejects_when_any_resolved_address_is_private(monkeypatch):
    # one global answer must not launder a private one
    _resolves_to(monkeypatch, "93.184.216.34", "192.168.1.1")
    with pytest.raises(ValueError, match="non-global"):
        wf._validated("https://split.evil.test/")


def test_validated_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="non-HTTP"):
        wf._validated("file:///etc/passwd")


def test_validated_rejects_embedded_credentials():
    with pytest.raises(ValueError, match="credentials"):
        wf._validated("https://user:pw@example.com/")


def test_validated_rejects_hostless_url():
    with pytest.raises(ValueError, match="no host"):
        wf._validated("http:///path")


def test_validated_rejects_garbage():
    # no scheme at all, so the scheme check fires before the host check
    with pytest.raises(ValueError, match="non-HTTP"):
        wf._validated("not a url")


def test_validated_reports_dns_failure(monkeypatch):
    def boom(*a, **kw):
        raise wf.socket.gaierror("nope")
    monkeypatch.setattr(wf.socket, "getaddrinfo", boom)
    with pytest.raises(ValueError, match="cannot resolve"):
        wf._validated("https://nx.evil.test/")


def test_validated_fails_closed_on_an_empty_resolution(monkeypatch):
    """No exception but no addresses either: the per-address loop then checks
    nothing and approves the host. Fail closed instead."""
    monkeypatch.setattr(wf.socket, "getaddrinfo", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="cannot resolve"):
        wf._validated("https://silent.evil.test/")


# --- extraction -------------------------------------------------------------

_PAGE = """<html><head><title>Real Title</title>
<style>body{color:red}</style></head>
<body>
  <nav>Home Login Signup</nav>
  <header>Site banner</header>
  <article><p>First paragraph.</p><p>Second   paragraph &amp; more.</p></article>
  <form><input name="q"></form>
  <footer>Copyright notice</footer>
  <script>var tracking = 1;</script>
</body></html>"""


def test_extract_keeps_prose_and_title():
    out = wf._extract_text(_PAGE)
    assert "Real Title" in out
    assert "First paragraph." in out
    assert "Second paragraph & more." in out  # entities decoded, runs collapsed


def test_extract_drops_boilerplate_tags():
    out = wf._extract_text(_PAGE)
    for noise in ("color:red", "Home Login", "Site banner", "Copyright notice",
                  "var tracking"):
        assert noise not in out


def test_extract_separates_block_elements():
    # open and close both emit a break, so blocks land one blank line apart
    assert wf._extract_text("<p>one</p><p>two</p>") == "one\n\ntwo"


def test_extract_collapses_blank_runs():
    assert wf._extract_text("<p>a</p><div></div><div></div><div></div><p>b</p>") == "a\n\nb"


def test_extract_survives_unbalanced_tags():
    # a stray close tag must not drive the skip counter negative and swallow
    # the rest of the page
    out = wf._extract_text("</script><p>visible</p>")
    assert "visible" in out


def test_extract_of_empty_html_is_empty():
    assert wf._extract_text("") == ""


# --- truncation -------------------------------------------------------------

def test_truncate_is_a_noop_under_the_limit():
    assert wf._truncate("short", limit=100) == "short"


def test_truncate_marks_the_cut():
    out = wf._truncate("x" * 500, limit=100)
    assert out.startswith("x" * 100)
    assert "truncated at 100 characters" in out


# --- the fetch loop ---------------------------------------------------------

import httpx
from types import SimpleNamespace

from silica.tools import TOOLS


class _Resp:
    """Minimal stand-in for a streamed httpx.Response.

    `_fetch` streams, so the stand-in is a context manager whose body arrives
    through `iter_bytes()`; `text` is what those bytes decode back to.
    """

    request = None

    def __init__(self, status=200, text="", ctype="text/html", location=None,
                 payload=None, extra_headers=None):
        self.status_code = status
        self.text = text
        self.headers = {"content-type": ctype} if ctype else {}
        # A real server's headers describe the WIRE body; iter_bytes hands back
        # the decoded one. Tests that need that mismatch pass it in here.
        self.headers.update(extra_headers or {})
        self.is_redirect = location is not None
        self.next_request = SimpleNamespace(url=location) if location else None
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_bytes(self):
        if self._payload is not None:
            yield json.dumps(self._payload).encode()
        else:
            yield self.text.encode()

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}", request=None, response=None
            )


def _serve(monkeypatch, *responses, allow_all_dns=True):
    """Queue responses for successive httpx.stream calls; record requested URLs."""
    seen: list[str] = []
    queue = list(responses)

    def fake_stream(method, url, **kw):
        seen.append(url)
        assert method == "GET"
        assert kw.get("follow_redirects") is False, "redirects must be manual"
        return queue.pop(0)

    monkeypatch.setattr(wf.httpx, "stream", fake_stream)
    if allow_all_dns:
        monkeypatch.setattr(
            wf.socket, "getaddrinfo",
            lambda host, port, *a, **kw: [(2, 1, 6, "", ("93.184.216.34", port))],
        )
    return seen


def test_web_fetch_registered_and_sensitive():
    assert "web_fetch" in TOOLS
    assert TOOLS["web_fetch"].sensitive is True


def test_fetch_drops_the_wire_encoding_headers_from_the_rebuilt_response(monkeypatch):
    """`iter_bytes()` yields DECODED bytes, so the rebuilt response must not keep
    the header saying they are still gzipped — httpx would try to decompress the
    plain body and every gzip site (most of the web) died on `.text` with
    "Error -3 while decompressing data: incorrect header check"."""
    _serve(monkeypatch, _Resp(
        text="<html><body><p>Hello world.</p></body></html>",
        extra_headers={"content-encoding": "gzip", "content-length": "31"},
    ))
    resp, _url = wf._fetch("https://example.com/a")
    assert "content-encoding" not in resp.headers
    # httpx restamps content-length from the body it was handed, so the stale
    # wire length ("31") must not survive either.
    assert resp.headers["content-length"] == str(len(resp.content))
    assert "Hello world." in resp.text  # the read httpx used to raise on


def test_web_fetch_returns_extracted_text_under_a_source_header(monkeypatch):
    _serve(monkeypatch, _Resp(text="<html><body><p>Hello world.</p></body></html>"))
    out = wf.web_fetch("https://example.com/a")
    assert out.splitlines()[0] == "Source: https://example.com/a"
    assert "Hello world." in out


def test_web_fetch_follows_redirects_and_reports_the_final_url(monkeypatch):
    seen = _serve(
        monkeypatch,
        _Resp(status=302, location="https://example.com/final"),
        _Resp(text="<p>arrived</p>"),
    )
    out = wf.web_fetch("https://example.com/start")
    assert seen == ["https://example.com/start", "https://example.com/final"]
    assert out.splitlines()[0] == "Source: https://example.com/final"


def test_web_fetch_revalidates_every_redirect_hop(monkeypatch):
    """A global first hop must not launder a redirect into link-local space."""
    seen: list[str] = []

    def fake_stream(method, url, **kw):
        seen.append(url)
        return _Resp(status=302, location="http://169.254.169.254/latest/meta-data/")

    def fake_dns(host, port, *a, **kw):
        ip = "169.254.169.254" if host == "169.254.169.254" else "93.184.216.34"
        return [(2, 1, 6, "", (ip, port))]

    monkeypatch.setattr(wf.httpx, "stream", fake_stream)
    monkeypatch.setattr(wf.socket, "getaddrinfo", fake_dns)

    with pytest.raises(ValueError, match="non-global"):
        wf.web_fetch("https://example.com/redirector")
    assert seen == ["https://example.com/redirector"]  # second hop never issued


def test_web_fetch_caps_the_redirect_chain(monkeypatch):
    hop = _Resp(status=302, location="https://example.com/next")
    _serve(monkeypatch, *[hop] * (wf._MAX_REDIRECTS + 1))
    with pytest.raises(ValueError, match="redirects"):
        wf.web_fetch("https://example.com/loop")


def test_web_fetch_spends_the_whole_redirect_budget(monkeypatch):
    """The cap's lower bound. `http -> https -> www -> canonical` is an ordinary
    chain, so following one hop fewer than advertised is a real regression, and
    only the upper bound was pinned: `range(_MAX_REDIRECTS)` instead of
    `range(_MAX_REDIRECTS + 1)` passed the whole suite."""
    hops = [
        _Resp(status=302, location=f"https://example.com/h{i}")
        for i in range(wf._MAX_REDIRECTS)
    ]
    seen = _serve(monkeypatch, *hops, _Resp(text="<p>arrived</p>"))
    out = wf.web_fetch("https://example.com/start")
    assert len(seen) == wf._MAX_REDIRECTS + 1
    assert "arrived" in out
    assert out.splitlines()[0] == f"Source: https://example.com/h{wf._MAX_REDIRECTS - 1}"


def test_web_fetch_403_says_bot_wall(monkeypatch):
    _serve(monkeypatch, _Resp(status=403))
    with pytest.raises(ValueError, match="403"):
        wf.web_fetch("https://example.com/paywalled")


def test_web_fetch_429_says_rate_limited(monkeypatch):
    _serve(monkeypatch, _Resp(status=429))
    with pytest.raises(ValueError, match="rate limited"):
        wf.web_fetch("https://example.com/busy")


def test_web_fetch_500_still_raises(monkeypatch):
    _serve(monkeypatch, _Resp(status=500))
    with pytest.raises(httpx.HTTPStatusError):
        wf.web_fetch("https://example.com/broken")


def test_web_fetch_refuses_binary_content(monkeypatch):
    _serve(monkeypatch, _Resp(ctype="application/pdf", text="%PDF-1.7"))
    with pytest.raises(ValueError, match="application/pdf"):
        wf.web_fetch("https://example.com/paper.pdf")


def test_web_fetch_passes_plain_text_through_unparsed(monkeypatch):
    _serve(monkeypatch, _Resp(ctype="text/plain", text="a <b> c"))
    out = wf.web_fetch("https://example.com/robots.txt")
    assert "a <b> c" in out  # not run through the HTML parser


def test_web_fetch_truncates_long_pages(monkeypatch):
    _serve(monkeypatch, _Resp(text="<p>" + ("word " * 40_000) + "</p>"))
    out = wf.web_fetch("https://example.com/long")
    assert "[truncated at" in out
    assert len(out) < wf._MAX_CHARS + 200


def test_web_fetch_rejects_a_private_target_before_any_request(monkeypatch):
    called = {"n": 0}

    def fake_stream(method, url, **kw):
        called["n"] += 1
        return _Resp()

    monkeypatch.setattr(wf.httpx, "stream", fake_stream)
    monkeypatch.setattr(
        wf.socket, "getaddrinfo",
        lambda host, port, *a, **kw: [(2, 1, 6, "", ("127.0.0.1", port))],
    )
    with pytest.raises(ValueError, match="non-global"):
        wf.web_fetch("http://localhost:8080/admin")
    assert called["n"] == 0


# --- YouTube ----------------------------------------------------------------

from pathlib import Path

_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000
hello <00:00:01.000><c>world</c>

00:00:02.000 --> 00:00:04.000
hello world

00:00:04.000 --> 00:00:06.000
second line &amp; more
"""


def test_vtt_to_text_strips_timings_markup_and_rolling_duplicates():
    assert wf.vtt_to_text(_VTT).splitlines() == [
        "hello world",
        "second line & more",
    ]


def test_youtube_without_ytdlp_prescribes_the_install(monkeypatch):
    monkeypatch.setattr(wf.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="yt-dlp"):
        wf.web_fetch("https://www.youtube.com/watch?v=abc")


def test_youtube_never_takes_the_http_path(monkeypatch):
    monkeypatch.setattr(wf.shutil, "which", lambda name: None)

    def boom(*a, **kw):
        raise AssertionError("no HTTP request must run for a YouTube URL")

    monkeypatch.setattr(wf.httpx, "stream", boom)
    with pytest.raises(ValueError, match="yt-dlp"):
        wf.web_fetch("https://youtu.be/abc")


def _fake_ytdlp(monkeypatch, *, writes=True, stderr=""):
    monkeypatch.setattr(wf.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    def fake_run(argv, **kw):
        if writes:
            out = Path(argv[argv.index("-o") + 1])
            out.with_suffix(".en.vtt").write_text(_VTT, encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr=stderr)

    monkeypatch.setattr(wf.subprocess, "run", fake_run)


def test_youtube_returns_the_transcript(monkeypatch):
    _fake_ytdlp(monkeypatch)
    out = wf.web_fetch("https://www.youtube.com/watch?v=abc")
    assert out.splitlines()[0] == "Source: https://www.youtube.com/watch?v=abc"
    assert "second line & more" in out


def test_youtube_channel_or_playlist_url_is_bounded_to_one_item(monkeypatch):
    """A channel (`/@someone`) or playlist URL passes `host_matches` exactly
    like a watch URL. `--no-playlist` only suppresses expansion when the URL
    is a video *inside* a playlist, not when the URL *is* the collection, so
    `--playlist-items 1` must be present to keep yt-dlp bounded to a single
    item instead of enumerating the whole channel/playlist."""
    monkeypatch.setattr(wf.shutil, "which", lambda name: "/usr/bin/yt-dlp")
    captured = {}

    def fake_run(argv, **kw):
        captured["argv"] = argv
        out = Path(argv[argv.index("-o") + 1])
        out.with_suffix(".en.vtt").write_text(_VTT, encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(wf.subprocess, "run", fake_run)
    wf.web_fetch("https://www.youtube.com/@someone")
    argv = captured["argv"]
    assert "--playlist-items" in argv
    assert argv[argv.index("--playlist-items") + 1] == "1"
    # The `--` end-of-options separator is the whole basis of the ruling that
    # this branch needs no _validated() call of its own: without it a URL
    # beginning with `-` is read by yt-dlp as a flag, not as the single
    # positional it is bounded to. A security invariant with no test is not one.
    assert argv[-2] == "--"
    assert argv[-1] == "https://www.youtube.com/@someone"


def test_youtube_without_subtitles_reports_the_stderr_tail(monkeypatch):
    _fake_ytdlp(monkeypatch, writes=False, stderr="ERROR: no subtitles available")
    with pytest.raises(ValueError, match="no subtitles available"):
        wf.web_fetch("https://youtu.be/abc")


def test_youtube_lookalike_domain_takes_the_http_path(monkeypatch):
    _serve(monkeypatch, _Resp(text="<p>not youtube</p>"))
    out = wf.web_fetch("https://youtube.com.evil.test/watch?v=abc")
    assert "not youtube" in out


def test_youtube_userinfo_on_a_real_host_does_not_reach_ytdlp(monkeypatch):
    """`urlsplit` already resolves `youtube.com@evil.test` to host `evil.test`,
    so that disguise never needed the userinfo guard: it fails the domain
    check regardless. The guard earns its keep on the opposite shape, where
    `.hostname` genuinely IS youtube.com but userinfo is riding along
    (`x@youtube.com`). The YouTube branch shells out to yt-dlp with no
    `_validated()` call of its own, so `host_matches` is the only gate; without
    the guard this URL would route straight to `subprocess.run`."""
    monkeypatch.setattr(wf.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    def boom(*a, **kw):
        raise AssertionError("subprocess.run must not run: host_matches should reject userinfo")

    monkeypatch.setattr(wf.subprocess, "run", boom)
    with pytest.raises(ValueError, match="credentials"):
        wf.web_fetch("https://x@youtube.com/watch?v=abc")


# --- Wikipedia ---------------------------------------------------------------


def _wp_page(**page) -> dict:
    return {"batchcomplete": True, "query": {"pages": [page]}}


def test_wikipedia_reads_the_api_instead_of_the_rendered_page(monkeypatch):
    seen = _serve(
        monkeypatch,
        _Resp(ctype="application/json", payload=_wp_page(title="PageRank",
                                                         extract="== History ==\nProse.")),
    )
    out = wf.web_fetch("https://en.wikipedia.org/wiki/PageRank")
    assert seen[0].startswith("https://en.wikipedia.org/w/api.php?")
    assert "titles=PageRank" in seen[0]
    assert "explaintext=1" in seen[0] and "redirects=1" in seen[0]
    # The citation stays the human-resolvable article URL, not the API call.
    assert out.splitlines()[0] == "Source: https://en.wikipedia.org/wiki/PageRank"
    assert "Prose." in out


def test_wikipedia_titles_the_page_from_the_api_not_its_lead_sentence(monkeypatch):
    """The API extract opens with the lead paragraph, so the positional guess
    named these notes "PageRank is an algorithm used by Google Search to ran".
    The canonical title comes back in the response, past `redirects=1`."""
    _serve(
        monkeypatch,
        _Resp(ctype="application/json", payload=_wp_page(
            title="PageRank",
            extract="PageRank is an algorithm used by Google Search to rank web pages.",
        )),
    )
    page = wf.fetch_page("https://en.wikipedia.org/wiki/Page_rank")
    assert page.title == "PageRank"
    assert page.text.startswith("Source: https://en.wikipedia.org/wiki/Page_rank")


def test_wikipedia_title_is_percent_decoded_before_it_is_a_parameter(monkeypatch):
    """`/wiki/Cura%C3%A7ao` must reach the API as the title `Curaçao`, urlencoded
    once. Passing the raw path through double-encodes it into a missing page."""
    seen = _serve(
        monkeypatch,
        _Resp(payload=_wp_page(title="Curaçao", extract="An island.")),
    )
    out = wf.web_fetch("https://en.wikipedia.org/wiki/Cura%C3%A7ao")
    assert "titles=Cura%C3%A7ao" in seen[0]
    assert "An island." in out


def test_wikipedia_language_comes_from_the_host(monkeypatch):
    seen = _serve(monkeypatch, _Resp(payload=_wp_page(extract="Un'isola.")))
    wf.web_fetch("https://it.wikipedia.org/wiki/Cura%C3%A7ao")
    assert seen[0].startswith("https://it.wikipedia.org/w/api.php?")


def test_wikipedia_sends_a_descriptive_user_agent_not_a_browser_string(monkeypatch):
    """Wikimedia's UA policy throttles generic browser strings, and `_HEADERS`
    is one. Only this branch opts out, so pin it."""
    sent: list[dict] = []

    def fake_stream(method, url, **kw):
        sent.append(kw.get("headers") or {})
        return _Resp(payload=_wp_page(extract="Prose."))

    monkeypatch.setattr(wf.httpx, "stream", fake_stream)
    monkeypatch.setattr(
        wf.socket, "getaddrinfo",
        lambda host, port, *a, **kw: [(2, 1, 6, "", ("93.184.216.34", port))],
    )
    wf.web_fetch("https://en.wikipedia.org/wiki/PageRank")
    assert "silica-harness" in sent[0]["User-Agent"]
    assert "Mozilla" not in sent[0]["User-Agent"]


def test_wikipedia_missing_page_falls_back_to_the_rendered_page(monkeypatch):
    """A red link or a Special: page has no extract. The article HTML is still
    worth reading, so the branch declines instead of failing the fetch."""
    seen = _serve(
        monkeypatch,
        _Resp(payload=_wp_page(title="Nope", missing=True)),
        _Resp(text="<html><body><p>Rendered instead.</p></body></html>"),
    )
    out = wf.web_fetch("https://en.wikipedia.org/wiki/Nope")
    assert "/w/api.php?" in seen[0]
    assert seen[1] == "https://en.wikipedia.org/wiki/Nope"
    assert "Rendered instead." in out


def test_wikipedia_non_article_url_never_calls_the_api(monkeypatch):
    """`?action=history`, `/w/index.php?title=`, and the bare host are not
    articles; `prop=extracts` would answer about the wrong thing or not at
    all."""
    for url in (
        "https://en.wikipedia.org/wiki/PageRank?action=history",
        "https://en.wikipedia.org/w/index.php?title=PageRank&oldid=1",
        "https://en.wikipedia.org/",
    ):
        seen = _serve(monkeypatch, _Resp(text="<p>page</p>"))
        out = wf.web_fetch(url)
        assert seen == [url], url
        assert "page" in out


def test_wikipedia_api_rate_limit_is_not_swallowed(monkeypatch):
    """The fallback covers "not an article", not "the API said no": a 429 that
    silently became an HTML fetch would hide that we are being throttled."""
    _serve(monkeypatch, _Resp(status=429))
    with pytest.raises(ValueError, match="rate limited"):
        wf.web_fetch("https://en.wikipedia.org/wiki/PageRank")


# Shape taken verbatim from the live /wiki/PageRank extract: the presentation
# MathML one glyph per line at indent 2+, then the LaTeX, and around it prose
# that Wikipedia indents by exactly one space.
_WP_MATH = """The PageRank of E is denoted by
\x20\x20
\x20\x20\x20\x20
\x20\x20\x20\x20\x20\x20\x20\x20P
\x20\x20\x20\x20\x20\x20\x20\x20R
\x20\x20\x20\x20\x20\x20\x20\x20(
\x20\x20\x20\x20\x20\x20\x20\x20E
\x20\x20\x20\x20\x20\x20\x20\x20)
\x20\x20\x20\x20{\\displaystyle PR(E).}
 are the pages under consideration,
\x20\x20\x20\x20\x20\x20\x20\x20z[t] <- encoder.embedding(t) + positional(t)
"""


def test_wikipedia_extract_drops_exploded_mathml_and_keeps_the_latex(monkeypatch):
    """`explaintext` ships each formula twice: unreadable glyph-per-line MathML
    and then the LaTeX. Measured 40-47% of a math article, truncated in ahead of
    its prose. Prose and indented pseudocode must survive the cut."""
    _serve(monkeypatch, _Resp(ctype="application/json",
                              payload=_wp_page(title="PageRank", extract=_WP_MATH)))
    out = wf.web_fetch("https://en.wikipedia.org/wiki/PageRank")

    assert "{\\displaystyle PR(E).}" in out
    assert "are the pages under consideration," in out
    assert "z[t] <- encoder.embedding(t) + positional(t)" in out  # long, so kept
    assert "\nP\n" not in out and "\n(\n" not in out
    assert not [ln for ln in out.splitlines() if ln.strip() in ("P", "R", "(", ")", "E")]


# --- the `Source: ` seam, producer to consumer -------------------------------


def test_a_real_fetch_result_yields_a_citation_and_a_title(monkeypatch):
    """`_render`'s `Source: <url>` header is a contract with web_research:
    `_collect_sources` lifts the citation out of it (at a magic offset) and
    `_first_line_of` skips past it to probe readability. Every other test on
    both sides hardcodes its own copy of that shape, so changing `_render` would
    leave the suite green while every fetched citation disappears. Run the real
    producer into the real consumers once, so the seam has somewhere to break.

    The title now rides beside the text instead of being read out of it, so
    assert both halves of the Page the consumer receives."""
    from silica.sources.web_research import _collect_sources, _first_line_of

    _serve(
        monkeypatch,
        _Resp(text="<html><title>Real Title</title><body><p>Prose.</p></body></html>"),
    )
    page = wf.fetch_page("https://example.com/article")

    assert _collect_sources([page.text]) == [
        ("https://example.com/article", "https://example.com/article")
    ]
    assert _first_line_of(page.text) == "Real Title"
    assert page.title == "Real Title"


# --- E1: structured title harvest -------------------------------------------


def test_og_title_beats_the_title_element():
    """og:title is authored for sharing, so it arrives without the SEO suffix."""
    html = """<html><head>
      <title>How to Grind Coffee | Bean Weekly</title>
      <meta property="og:title" content="How to Grind Coffee">
    </head><body><p>x</p></body></html>"""
    assert wf.page_title(html) == "How to Grind Coffee"


def test_twitter_title_is_the_second_choice():
    html = """<html><head><title>t</title>
      <meta name="twitter:title" content="Card Title"></head></html>"""
    assert wf.page_title(html) == "Card Title"


def test_og_title_shipped_as_name_is_still_read():
    """Half the web puts the OG key in `name=` instead of `property=`."""
    html = '<html><head><meta name="og:title" content="OG via name"></head></html>'
    assert wf.page_title(html) == "OG via name"


def test_declared_site_name_suffix_is_stripped():
    html = """<html><head>
      <title>Attention Is All You Need — arXiv</title>
      <meta property="og:site_name" content="arXiv"></head></html>"""
    assert wf.page_title(html) == "Attention Is All You Need"


def test_a_suffix_that_is_not_the_site_name_survives():
    """No guessing which half of a separated title is the site: without a
    declared site name the whole title stands."""
    html = "<html><head><title>The Pragmatic Programmer - 20th Anniversary</title></head></html>"
    assert wf.page_title(html) == "The Pragmatic Programmer - 20th Anniversary"


def test_site_name_only_strips_from_the_end():
    html = """<html><head><title>arXiv - Attention Is All You Need</title>
      <meta property="og:site_name" content="arXiv"></head></html>"""
    assert wf.page_title(html) == "arXiv - Attention Is All You Need"


def test_a_doubled_site_name_is_stripped_once():
    html = """<html><head><title>Bean Weekly | Bean Weekly</title>
      <meta property="og:site_name" content="Bean Weekly"></head></html>"""
    assert wf.page_title(html) == "Bean Weekly"


def test_a_title_that_is_nothing_but_the_suffix_is_kept_whole():
    """Stripping would leave an empty title, which is worse than a redundant
    one: the note would fall back to naming itself after a lead sentence."""
    html = """<html><head><title> | Bean Weekly</title>
      <meta property="og:site_name" content="Bean Weekly"></head></html>"""
    assert wf.page_title(html) == "| Bean Weekly"


def test_svg_title_does_not_win_over_the_head_title():
    html = ("<html><head><title>Head Title</title></head>"
            "<body><svg><title>icon label</title></svg></body></html>")
    assert wf.page_title(html) == "Head Title"


def test_page_with_no_title_yields_empty_string():
    assert wf.page_title("<html><body><p>orphan prose</p></body></html>") == ""


def test_title_whitespace_is_collapsed_and_capped():
    html = "<html><head><title>\n  spread   out\n</title></head></html>"
    assert wf.page_title(html) == "spread out"
    long = f"<html><head><title>{'x' * 300}</title></head></html>"
    assert len(wf.page_title(long)) == wf._MAX_TITLE


def test_plain_text_response_carries_no_title(monkeypatch):
    """A .txt or JSON body has no title element; promoting its first line to one
    is exactly the positional guess this replaces."""
    _serve(monkeypatch, _Resp(text="just some prose", ctype="text/plain"))
    page = wf.fetch_page("https://example.com/notes.txt")
    assert page.title == ""
    assert "just some prose" in page.text


# --- E2: code fences --------------------------------------------------------


def test_pre_block_is_fenced_and_keeps_its_indentation():
    html = """<body><p>Before.</p>
<pre><code>def f():
    return 1
</code></pre>
<p>After.</p></body>"""
    out = wf._extract_text(html)
    assert "```\ndef f():\n    return 1\n```" in out
    assert "Before." in out and "After." in out


def test_fenced_code_keeps_blank_lines_but_prose_still_collapses():
    html = "<pre>a\n\nb</pre><p>c    d</p>"
    out = wf._extract_text(html)
    assert "```\na\n\nb\n```" in out
    assert "c d" in out


def test_pre_inside_skipped_container_emits_no_fence():
    """An unpaired fence would render the whole rest of the page as code."""
    out = wf._extract_text("<nav><pre>menu</pre></nav><p>real</p>")
    assert "```" not in out
    assert out == "real"


def test_unclosed_pre_is_fenced_shut():
    out = wf._extract_text("<body><pre>truncated code")
    assert out.count("```") == 2
    assert out.endswith("```")


def test_truncation_closes_a_fence_it_cut_open():
    text = "intro\n```\n" + "x" * 200
    out = wf._truncate(text, limit=40)
    assert out.count("```") == 2
    assert "truncated at 40 characters" in out


def test_truncation_leaves_balanced_fences_alone():
    out = wf._truncate("```\na\n```\n" + "y" * 200, limit=40)
    assert out.count("```") == 2


# --- image alt text ---------------------------------------------------------


def test_image_alt_text_survives_marked_as_an_image():
    """On a technical page the alt is often the only description of a diagram
    that exists in the markup, and this parser read no attributes at all."""
    out = wf._extract_text(
        '<p>See below.</p><img src="a.png" alt="Transformer block diagram">'
        "<p>As shown.</p>"
    )
    assert "[image: Transformer block diagram]" in out
    assert "See below." in out and "As shown." in out


def test_self_closing_image_is_read_too():
    out = wf._extract_text('<img src="a.png" alt="XHTML self closed" />')
    assert "[image: XHTML self closed]" in out


def test_image_without_alt_adds_nothing():
    assert wf._extract_text('<p>a</p><img src="x.png"><p>b</p>') == "a\n\nb"
    assert wf._extract_text('<p>a</p><img src="x.png" alt="  "><p>b</p>') == "a\n\nb"


def test_alt_with_no_letters_is_dropped():
    """`alt="***"` and `alt="—"` are spacers, not captions."""
    assert "image:" not in wf._extract_text('<img alt="***"><img alt="—">')


def test_repeated_alt_is_emitted_once():
    """Icon rows repeat one alt down a page; each is a line of pure noise."""
    html = "".join('<li><img src="i.png" alt="bullet icon">item</li>' for _ in range(5))
    assert wf._extract_text(html).count("[image: bullet icon]") == 1


def test_alt_returns_after_a_different_alt_interrupts():
    """Adjacent-equal dedup, not seen-once: the same diagram legitimately appears
    twice on a long page, separated by other content."""
    out = wf._extract_text(
        '<img alt="figure one"><p>prose</p><img alt="figure two">'
        '<p>prose</p><img alt="figure one">'
    )
    assert out.count("[image: figure one]") == 2


def test_alt_inside_skipped_boilerplate_is_dropped():
    """A nav is full of icons; none of them describe the page."""
    out = wf._extract_text(
        '<nav><img alt="home icon"><img alt="search icon"></nav>'
        '<article><img alt="real figure">body</article>'
    )
    assert "home icon" not in out and "search icon" not in out
    assert "[image: real figure]" in out


def test_alt_does_not_leak_into_the_title_harvest():
    """`page_title` must still name the page, not its first figure."""
    html = ('<html><head><title>Real Title</title></head>'
            '<body><img alt="hero banner"><p>x</p></body></html>')
    assert wf.page_title(html) == "Real Title"
