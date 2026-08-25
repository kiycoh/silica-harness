# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

from __future__ import annotations

import re
import textwrap
from markdown_it import MarkdownIt

_MD = MarkdownIt()  # stateless parser, shared by every parse below

NON_MD_EXTENSIONS = (
    '.png', '.jpg', '.jpeg', '.pdf', '.webp', '.svg', '.gif', '.mp4', '.zip', '.html', '.css'
)

# Wikilink target extraction: captures the target of [[Target]], [[Target|alias]]
# and [[Target#anchor]] (everything before the first | or #). The shared regex
# for quick target scans; extract_links below is the full AST-aware version.
WIKILINK_TARGET_RE = re.compile(r"\[\[([^\]|#]+)")


# Every marker that can make `_extract_links_ast` return anything. It reads
# targets from two places only — a text token containing `[[…]]`, and the
# `[[href]]` it injects for a link_open/image token — and each of those needs
# one of these six shapes in the source:
#
#   [[                       a literal wikilink
#   ](  not web              an inline destination that survives the
#                            http/https/mailto filter in the walk
#   [label]:                 a link reference definition, which turns a bare
#                            [label] elsewhere into a link with no `](` in
#                            the source at all
#   \[                       a backslash escape decoding to '['
#   &#91; &#x5B;             an entity decoding to '['
#   &lbrack; &lsqb;
#   <scheme:  not web        an autolink whose href the same filter keeps
#
# Nothing else in CommonMark puts a '[' into a text token or an href into the
# walk, so no marker means no link and the parse can be skipped. The web
# exclusions are case-sensitive on purpose: so is the filter they model, which
# is why `[x](HTTP://y)` really does yield a target today.
#
# Worth the care because this is the hot loop of every index rebuild: measured
# 2026-08-24, `extract_links` was 3.65s of build_graph_data's 3.86s over 395
# notes, all of it markdown-it, and the rebuild is per-process — every
# invocation pays it again. With the scan: 0.80s, and 58% of extract_links'
# wall clock gone across 2500 files / 23.3 MB of real vaults, with zero
# disagreements against the parser over the same corpus.
#
# The two directions are not symmetric. A marker too many costs one parse; a
# marker too few drops a link, which reads downstream as a *removed* edge and
# rolls the chunk back at the graph gate. So anything unproven belongs on the
# parser side of this scan, not this side — and every alternative here is
# mutation-tested (tests/test_wikilink_fast_path.py), because a scan that is
# merely nearly right looks exactly like one that is right.
_LINK_MARKER_RE = re.compile(
    r"\[\["
    r"|\]\((?!https?://|mailto:)"
    # Unanchored on purpose. A definition is document-global wherever it sits,
    # and inside a blockquote its line starts with '>', not whitespace — the
    # line-anchored version of this alternative returned [] for
    # "> [b]: Target.md" while the parser returned the target (found by
    # mutation-testing the scan, 2026-08-24). Matching a stray "[x]:" in prose
    # costs one parse; missing one costs an edge.
    r"|\[[^\]\n]*\]:"
    r"|\\\["
    r"|&(?:#0*91|#[xX]0*5[bB]|lbrack|lsqb);"
    # The lookbehinds carry the '<' so a scheme merely *ending* in a web one
    # (<xhttp://…>, a real target) is not mistaken for the web scheme itself.
    r"|<[A-Za-z][A-Za-z0-9+.\-]{1,31}:(?<!<http:)(?<!<https:)(?<!<mailto:)"
)


def extract_links(content: str) -> list[str]:
    """Extract clean wikilinks (both [[target]] and ![[target]]).

    Answers from a marker scan when that scan can prove the CommonMark parse
    would have found nothing, and from the parse itself otherwise. The two are
    the same function to every caller; see `_LINK_MARKER_RE` for why they have
    to be, and what happens if they ever stop being.
    """
    # Scanned after dedent so there is only ever one string in play — the one
    # the parser would have seen. No marker is anchored to a column, so the
    # order is not load-bearing (verified over 564 real notes: dedent flips no
    # verdict); keeping it this way just means a reader never has to check.
    content = textwrap.dedent(content)
    if not _LINK_MARKER_RE.search(content):
        return []
    return _extract_links_ast(content)


def _extract_links_ast(content: str) -> list[str]:
    """`extract_links` without the marker scan, on already-dedented content.

    Split out so the equivalence between the two paths is a thing the tests
    can assert directly (tests/test_wikilink_fast_path.py) instead of a claim.
    """
    tokens = _MD.parse(content)

    text_pieces: list[str] = []

    def walk(toks: list) -> None:
        for t in toks:
            if t.type == "inline":
                if t.children:
                    walk(t.children)
            elif t.type == "text":
                text_pieces.append(t.content)
            elif t.type == "image":
                src = t.attrs.get("src")
                if src and not (src.startswith("http://") or src.startswith("https://") or src.startswith("mailto:")):
                    text_pieces.append(f"[[{src}]]")
            elif t.type == "link_open":
                href = t.attrs.get("href")
                if href and not (href.startswith("http://") or href.startswith("https://") or href.startswith("mailto:") or href.startswith("#")):
                    text_pieces.append(f"[[{href}]]")

    walk(tokens)

    cleaned = []
    for text in text_pieces:
        # Match [[target]] links (allowing characters like # and ^)
        raw_targets = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]', text)
        for t in raw_targets:
            t = t.strip().replace("''", "'")
            if not t:
                continue
            # [[#Heading]] / [[^block]] point INSIDE the note that carries
            # them: they are not note links, and no resolver downstream can
            # ever match one, so each landed in the backends' unresolved set
            # and read to the graph gate as a newly introduced dangling link —
            # 2 of the 6 chunk rollbacks in the 2026-08-21 machine_learning run
            # were an intra-note anchor. link/sweep.py's regex already refuses
            # to match them; this is the outlier that did not.
            if t.startswith('#') or t.startswith('^'):
                continue
            # Split off heading/section part
            t = t.split('#', 1)[0].strip()
            if not t:
                continue
            if t.lower().endswith(NON_MD_EXTENSIONS):
                continue
            if t not in cleaned:
                cleaned.append(t)
    return cleaned


_FRONTMATTER_BLOCK_RE = re.compile(r"\A\s*---\r?\n.*?\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_HEADING_LINE_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t].*$", re.MULTILINE)


def extract_links_typed(content: str) -> dict[str, bool]:
    """`extract_links` targets, each flagged scaffold (True) or prose (False).

    Scaffold is a link the vault's structure wrote rather than its prose: a
    frontmatter property (`parent note`, `related:`) or a heading line
    (`## [[Concept]]`, a hub's spoke list). Measured 2026-08-23 on a 709-note
    human vault: 257 of 1059 linked pairs exist only as scaffold (24%). The
    flag rides on the graph edge so a consumer can tell the two apart; none
    switches by default, because on that vault Adamic-Adar scored AUC 0.806
    with scaffold and 0.750 without, and dropping it split the graph from 96
    components into 178 (ADR-0029). A target that occurs in prose anywhere is
    prose, whatever else mentions it. Same targets, same order as
    `extract_links`.
    """
    all_targets = extract_links(content)
    if not all_targets:
        return {}
    m = _FRONTMATTER_BLOCK_RE.match(content)
    body = content[m.end():] if m else content
    prose = set(extract_links(_HEADING_LINE_RE.sub("", body)))
    return {t: t not in prose for t in all_targets}


def parse_headings(body: str) -> list[dict]:
    """Parse headings from the body using AST, ignoring code blocks."""
    body = textwrap.dedent(body)

    tokens = _MD.parse(body)

    lines = body.splitlines(keepends=True)
    line_offsets = []
    current_offset = 0
    for line in lines:
        line_offsets.append(current_offset)
        current_offset += len(line)

    headings = []
    for idx, t in enumerate(tokens):
        if t.type == "heading_open":
            level = int(t.tag[1])
            # Find next inline token to get heading text
            next_t = tokens[idx + 1]
            text = next_t.content
            
            line_idx = t.map[0] if t.map else 0
            pos = line_offsets[line_idx] if line_idx < len(line_offsets) else len(body)
            headings.append({"level": level, "text": text, "pos": pos})

    return headings


def _balanced(body: str) -> list[str]:
    """Check for unbalanced OFM structural delimiters (fence-aware) using AST."""
    body = textwrap.dedent(body)
    issues = []
    # If there's an odd number of ``` in the raw body, code fence is unclosed
    if body.count("```") % 2:
        issues.append("unclosed code fence")


    tokens = _MD.parse(body)

    text_pieces: list[str] = []

    def walk(toks: list) -> None:
        for t in toks:
            if t.type == "inline":
                if t.children:
                    walk(t.children)
            elif t.type == "text":
                text_pieces.append(t.content)

    walk(tokens)

    combined_text = "".join(text_pieces)

    if combined_text.count("$$") % 2:
        issues.append("unbalanced $$ block")
    if combined_text.count("==") % 2:
        issues.append("unbalanced == highlight")
    if combined_text.count("[[") != combined_text.count("]]"):
        issues.append("unbalanced [[wikilink]]")

    return issues


def extract_callouts(body: str) -> list[str]:
    """Extract Obsidian callout types (e.g. 'note', 'tip') from blockquotes."""
    body = textwrap.dedent(body)

    tokens = _MD.parse(body)

    callout_types = []
    for idx, t in enumerate(tokens):
        if t.type == "blockquote_open":
            # Search for the first inline token inside the blockquote
            for k in range(idx + 1, len(tokens)):
                if tokens[k].type == "blockquote_close":
                    break
                if tokens[k].type == "inline":
                    content = tokens[k].content
                    match = re.match(r'^\[!([A-Za-z]+)\]', content)
                    if match:
                        callout_types.append(match.group(1))
                    break
    return callout_types


def get_non_code_text(body: str) -> str:
    """Extract all text tokens from body, ignoring code blocks/fences/inline-code."""
    body = textwrap.dedent(body)

    tokens = _MD.parse(body)
    text_pieces = []
    def walk(toks: list) -> None:
        for t in toks:
            if t.type == "inline":
                if t.children:
                    walk(t.children)
            elif t.type == "text":
                text_pieces.append(t.content)
    walk(tokens)
    return "".join(text_pieces)


def resolve_relative(target: str, source_path: str) -> str | None:
    """Vault-rooted form of a `./`/`../` link target, joined against the source
    note's folder; None when the walk escapes the vault root (no note can match,
    and a stripped guess would resolve to the wrong file).

    Any other target returns unchanged: bare names and vault-rooted paths
    already resolve in both backends, and rewriting them here would change
    measured resolution behavior for no defect. Resolution-side only — the one
    link GENERATOR stays extract_links (ADR-0029); before this helper existed
    the suffix matcher saw "/./x.md", matched nothing, and one INDEX.md carried
    332 ghost targets (2026-08-25).
    """
    if not (target.startswith(("./", "../")) or "/./" in target or "/../" in target):
        return target
    import posixpath

    base = source_path.rsplit("/", 1)[0] if "/" in source_path else ""
    joined = posixpath.normpath(posixpath.join(base, target))
    if joined == ".." or joined.startswith("../"):
        return None
    return joined
