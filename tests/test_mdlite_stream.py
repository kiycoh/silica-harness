# tests/test_mdlite_stream.py
"""mdLite() must terminate on every prefix of a streamed answer.

A table header arrives one delta before its delimiter row, and mdLite() re-parses
the whole segment on each delta. A half-arrived block used to consume no line and
spin the parser forever, until the tab threw RangeError and the SSE reader died
with it, truncating the answer.
"""
import json
import re
import shutil
import subprocess

import pytest

from tests.link_cases import URL_CASES

from tests.webassets import app_js

STREAMED = (
    "### Panoramica\n\n"
    "| Settimana | Date | Cosa studi |\n"
    "|:---------:|:----:|-----------|\n"
    "| **1** | 3/8 | Fondamenti |\n\n"
    "- una lista\n"
    "```\nfence\n```\n"
    # A URL arrives one character at a time too, and trimUrl() peels trailing
    # punctuation in a loop — every prefix of it has to terminate as well.
    "fonte https://en.wikipedia.org/wiki/A_(b), e [x](https://e.com).\n"
    "chiusura.\n"
)


def _md_lite_source() -> str:
    m = re.search(r"^function mdLite\(src\) \{.*?^\}", app_js(), re.S | re.M)
    assert m, "mdLite() not found in app.js"
    return m.group(0)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run app.js")
def test_every_streaming_prefix_terminates(tmp_path):
    script = tmp_path / "sweep.js"
    script.write_text(
        _md_lite_source()
        + "\nconst src = JSON.parse(process.argv[2]);\n"
        + "for (let n = 0; n <= src.length; n++) mdLite(src.slice(0, n));\n"
        + "console.log(mdLite(src).includes('<table>') ? 'TABLE' : 'NO-TABLE');\n"
    )
    out = subprocess.run(
        ["node", str(script), json.dumps(STREAMED)],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "TABLE"  # the finished table still renders as one


def _render(tmp_path, sources: list[str]) -> list[str]:
    """mdLite() over each source, under node."""
    script = tmp_path / "render.js"
    script.write_text(
        _md_lite_source()
        + "\nfor (const s of JSON.parse(process.argv[2])) console.log(JSON.stringify(mdLite(s)));\n"
    )
    out = subprocess.run(
        ["node", str(script), json.dumps(sources)],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    return [json.loads(line) for line in out.stdout.splitlines()]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run app.js")
def test_url_contract_matches_the_server_render(tmp_path):
    """The same corpus test_gui_web.py runs against `_linkify`. A tool-split turn
    keeps its live segments until reload, so what mdLite does to a URL is what the
    user sees — the two renders answer to one contract."""
    htmls = _render(tmp_path, [md for md, _p, _a in URL_CASES])
    for (md, present, absent), html in zip(URL_CASES, htmls):
        for frag in present:
            assert frag in html, (md, html)
        for frag in absent:
            assert frag not in html, (md, html)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run app.js")
def test_unsafe_scheme_never_becomes_a_live_href(tmp_path):
    """mdLite builds anchors by hand and gets none of markdown-it's validateLink,
    so a model-authored `javascript:` used to land as a live anchor in the app's
    own origin. Whitelist: the text stays, the anchor does not."""
    cases = [
        "[x](javascript:alert%281%29)",
        "[x](JaVaScRiPt:alert%281%29)",
        "[x](data:text/html,<script>1</script>)",
        "[x](vbscript:msgbox)",
    ]
    for html in _render(tmp_path, cases):
        assert "<a href" not in html, html
        assert ">x</a>" not in html


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run app.js")
def test_markdown_link_is_not_double_wrapped_by_the_url_pass(tmp_path):
    """Both link forms share one pass: a bare-URL match running before the
    markdown form eats the target inside `](…)`, running after it re-matches the
    URL already in href="…" and nests the anchors."""
    (html,) = _render(tmp_path, ["vedi [testo](https://e.com/a) qui"])
    assert html.count("<a ") == 1, html
    assert '<a href="https://e.com/a">testo</a>' in html
