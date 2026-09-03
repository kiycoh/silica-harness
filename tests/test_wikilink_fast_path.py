# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`extract_links` may skip the CommonMark parse, but only when it can prove
the parse would have found nothing.

`extract_links` was 95% of every vault index rebuild (measured 2026-08-24:
3.65s of build_graph_data's 3.86s over 395 notes), all of it a full
markdown-it parse run to reach text tokens that then get regexed anyway. The
skip is worth having; being wrong about it is not, because a missed link reads
downstream as a *removed* edge and the graph regression gate rolls the chunk
back.

So these tests are the proof, not a smoke check. They come in three parts:

  - the skip happens at all (sabotage the parser, assert nothing calls it);
  - every construct that CAN produce a link still reaches the parser (same
    sabotage, assert it IS called) — this is what fails if the marker scan is
    ever widened past what it can prove;
  - the two paths agree, target for target, over the adversarial corpus
    below.
"""
from __future__ import annotations

import textwrap

import pytest

from silica.kernel.link import ast as ast_mod
from silica.kernel.link.ast import extract_links


class _ParserReached(Exception):
    """Raised by the sabotaged parser so a test can assert it was reached."""


@pytest.fixture
def sabotaged_parser(monkeypatch):
    """Make `_MD.parse` explode, so 'was the parse skipped' is observable."""
    def boom(_src):
        raise _ParserReached
    monkeypatch.setattr(ast_mod._MD, "parse", boom)


# Prose that cannot contain a link under any CommonMark reading. Each carries
# characters that are *near* a marker without being one, so a sloppy scan
# fails here rather than in production.
NO_LINK_AT_ALL = [
    "Plain prose with no links at all.\n",
    "# Heading\n\nA paragraph, *emphasis*, **strong**, and a list:\n\n- one\n- two\n",
    "Array syntax x[0] and y[1] are not links.\n",
    "A closing bracket ] alone, and an opening [ alone.\n",
    "An ampersand & on its own, plus &amp; and &nbsp; entities.\n",
    "A less-than < and a bare <em> tag, plus a > quote marker.\n",
    "A backslash \\ and an escaped \\* asterisk.\n",
    "```\nfenced code with x[0] and & and <\n```\n",
    "    indented code with x[0]\n",
    "Inline `code[0]` and a URL written bare: https://example.com/a\n",
    "Colons: key: value, and a lone colon at the end:\n",
]
# `d[k]: int` is deliberately NOT here: the reference-definition marker is
# unanchored, so a bracket-then-colon anywhere sends the note to the parser.
# That is the conservative direction, and the differential corpus below still
# proves the answer comes out the same.


# Every construct that CAN reach the walk in `extract_links`, with the target
# the AST path produces for it today. Pinned by measurement, not by reading
# the CommonMark spec: `HTTP://` in caps survives the http/https/mailto filter
# because that filter is case-sensitive, and a link reference definition
# produces a link with no `](` anywhere in the source.
LINK_PRODUCING = [
    pytest.param("See [[Alpha]] here.\n", ["Alpha"], id="literal-wikilink"),
    pytest.param("An embed ![[Beta]] here.\n", ["Beta"], id="embed-wikilink"),
    pytest.param("See [x](Gamma.md) here.\n", ["Gamma.md"], id="inline-destination"),
    pytest.param("An image ![alt](Delta.md)\n", ["Delta.md"], id="image-destination"),
    pytest.param("See [x][b].\n\n[b]: Epsilon.md\n", ["Epsilon.md"], id="reference-link"),
    pytest.param("See [b] here.\n\n[b]: Zeta.md\n", ["Zeta.md"], id="shortcut-reference"),
    pytest.param("See [b].\n\n   [b]: Eta.md\n", ["Eta.md"], id="indented-reference"),
    # A definition inside a blockquote is still document-global, and the line
    # it sits on starts with '>' rather than whitespace. Found by mutation
    # testing the marker scan 2026-08-24; the line-anchored version this
    # replaced returned [] here.
    pytest.param("> [b]: Eta2.md\n\nSee [b].\n", ["Eta2.md"], id="blockquote-reference"),
    pytest.param("See \\[\\[Theta\\]\\] here.\n", ["Theta"], id="backslash-escape"),
    pytest.param("See &#91;&#91;Iota&#93;&#93; here.\n", ["Iota"], id="decimal-entity"),
    pytest.param("See &#x5B;&#x5B;Kappa&#x5D;&#x5D; here.\n", ["Kappa"], id="hex-entity"),
    pytest.param("See &lbrack;&lbrack;Lambda&rbrack;&rbrack;.\n", ["Lambda"], id="lbrack-entity"),
    pytest.param("See &lsqb;&lsqb;Mu&rsqb;&rsqb;.\n", ["Mu"], id="lsqb-entity"),
    pytest.param("<ftp://host/Nu>\n", ["ftp://host/Nu"], id="non-web-autolink"),
    # A scheme that merely ENDS in a web one. Catches a marker scan whose
    # web exclusion looks only at the characters before the colon.
    pytest.param("<xhttp://host/Nu2>\n", ["xhttp://host/Nu2"], id="near-web-autolink"),
    pytest.param("See [x](HTTP://Xi) here.\n", ["HTTP://Xi"], id="uppercase-scheme"),
    pytest.param("<HTTP://Omicron>\n", ["HTTP://Omicron"], id="uppercase-autolink"),
    pytest.param("    See [b].\n\n    [b]: Pi.md\n", ["Pi.md"], id="reference-behind-dedent"),
    pytest.param("Keeps ADR-0001 intact.\n", ["ADR-0001"], id="adr-prose-reference"),
]


# The full corpus the two paths must agree on: the no-link prose, every
# link-producing construct, and the cases where the AST deliberately finds
# nothing (code regions, HTML blocks, emphasis splitting a target in two).
DIFFERENTIAL_CORPUS = NO_LINK_AT_ALL + [c.values[0] for c in LINK_PRODUCING] + [
    "See `[[Alpha]]` here.\n",
    "```\n[[Alpha]]\n```\n",
    "```python\nx = \"[[Alpha]]\"\n```\n",
    "~~~\n[[Alpha]]\n~~~\n",
    "text\n\n    [[Alpha]]\n",
    "text\n\n\t[[Alpha]]\n",
    "<div>\n[[Alpha]]\n</div>\n",
    "a <b>[[Alpha]]</b> b\n",
    "See [[Alpha *b* Gamma]] here.\n",
    "See [[Alpha **b** Gamma]] here.\n",
    "See [[Alpha_b_Gamma]] here.\n",
    "See [[Alpha\nBeta]] here.\n",
    "See [[#Head]] and [[^blk]].\n",
    "See [[Alpha|shown]] and [[Alpha#Sec]].\n",
    "See [[O''Brien]].\n",
    "See [[A\\B]].\n",
    "![alt](Rho.png)\n",
    "![alt](Sigma.PNG)\n",
    "<http://example.com/a>\n",
    "See [x](https://example.com/a) and [y](mailto:a@b.c).\n",
    "See [x](#section) here.\n",
    "| a |\n|---|\n| [[Alpha]] |\n",
    "text[^1]\n\n[^1]: [[Alpha]]\n",
    "[[[Alpha]]](Beta.md)\n",
    "See [[A&amp;B]] here.\n",
    "See [[A\\|B]] here.\n",
    "- item\n\n    [[Alpha]]\n",
    "- a\n  - b\n    [b]: Deep.md\n\nSee [b].\n",
    "- a\n  - b\n    - c\n      [b]: Deeper.md\n\nSee [b].\n",
    "> - a\n>   - b\n>     [b]: Quoted.md\n\nSee [b].\n",
    "Python typing in prose: d[k]: int is not a definition.\n",
    "> [!note] callout\n> with [[Alpha]]\n",
    "---\ntitle: \"[[Alpha]]\"\n---\n\nBody with [[Beta]].\n",
    "See [[Alpha]] twice: [[Alpha]].\n",
    "",
    "\n\n\n",
]


def test_a_note_with_no_link_marker_never_reaches_the_parser(sabotaged_parser):
    for content in NO_LINK_AT_ALL:
        assert extract_links(content) == [], content


@pytest.mark.parametrize("content,expected", LINK_PRODUCING)
def test_every_link_producing_construct_still_reaches_the_parser(
    content, expected, sabotaged_parser
):
    # The value assertion lives in the differential test below; here the only
    # question is whether the marker scan let the parse happen. A guard that
    # grows too confident returns [] instead of raising, and this is what
    # catches it.
    with pytest.raises(_ParserReached):
        extract_links(content)


@pytest.mark.parametrize("content,expected", LINK_PRODUCING)
def test_every_link_producing_construct_keeps_its_target(content, expected):
    assert extract_links(content) == expected


@pytest.mark.parametrize("content", DIFFERENTIAL_CORPUS)
def test_the_fast_path_agrees_with_the_parser_target_for_target(content):
    assert extract_links(content) == ast_mod._extract_links_ast(textwrap.dedent(content))


def test_the_fast_path_agrees_with_the_parser_on_the_repo_corpus():
    """The adversarial corpus above is hand-written; this one is not.

    Runs both paths over every tracked Markdown file in the repo — the same
    text the index rebuild feeds `extract_links` — so a divergence the
    fixtures did not imagine still fails here.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    notes = sorted(p for p in repo.glob("*.md")) + sorted(
        p for p in (repo / "tests").rglob("*.md")
    )
    assert notes, "no Markdown in the repo to differential-test against"
    for path in notes:
        content = path.read_text(encoding="utf-8", errors="replace")
        assert extract_links(content) == ast_mod._extract_links_ast(
            textwrap.dedent(content)
        ), path
