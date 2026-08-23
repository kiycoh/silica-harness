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


def extract_links(content: str) -> list[str]:
    """Extract clean wikilinks (both [[target]] and ![[target]]) using AST parsing."""
    content = textwrap.dedent(content)

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
