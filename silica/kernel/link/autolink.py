# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Deterministic wikilink injector for touched notes (Phase 4).

Rule (from the plan):
  "embeddings PROPOSE, graph DISPOSES"
  - `candidates`  — optional list of titles prioritized by embedding similarity.
  - `title_index` — authoritative list of titles that exist in the vault graph.
  A link is emitted ONLY when the title exists in `title_index`.  If `candidates`
  is given, only titles in candidates∩title_index are considered, which keeps
  the autolink pass focused and fast.

Skip regions (never modified):
  - YAML frontmatter  (--- block at the very top of the note, LF or CRLF)
  - Fenced code       (``` or ~~~ blocks, incl. unclosed-to-EOF) and indented code
  - Inline code       (`...`)
  - LaTeX math        ($...$  and  $$...$$)
  - Bare URLs, markdown links/images, inline #tags, HTML tags/comments
  - Existing wikilinks ([[...]]) and heading lines

Disambiguation rule:
  If `title_index` contains two entries that differ only in path but share the
  same display name, the caller must deduplicate them before passing — this
  function works on display names only and will happily link an ambiguous title.
  Use `build_title_index` (below) to get a pre-disambiguated index from the
  driver.

Idempotency: calling autolink twice on the same body is a no-op — any already-
linked title is in a skip region on the second pass.
"""
from __future__ import annotations

import functools
import re
from typing import Sequence

# ---------------------------------------------------------------------------
# Skip-region detection
# ---------------------------------------------------------------------------

# Matches the YAML frontmatter at the very top of a note (OFM convention).
# \r?\n so Windows (CRLF) user notes touched by backlink_pass are protected too
# — otherwise the whole frontmatter fails to match and becomes linkable.
_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---[ \t]*\r?\n?", re.DOTALL)

# Inline code (`...`)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# Display math ($$...$$) — must come before single-$ match
_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)

# Inline math ($...$) — single-line only
_INLINE_MATH_RE = re.compile(r"\$[^$\n]+\$")

# Bare URLs — never link a word inside https://example.com/Neural-Networks
_URL_RE = re.compile(r"https?://[^\s<>()\[\]]+")

# Inline Obsidian tags (#tag, #nested/tag) — preceded by start/whitespace so
# C# and heading markers ("# Title") don't match. Linking would kill the tag.
_INLINE_TAG_RE = re.compile(r"(?<!\S)#[A-Za-z_][\w/-]*")

# Existing wikilinks [[...]]
_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")

# Markdown links and images: [text](href) / ![alt](href). Protects both the
# link/alt text and the href. Autolink-only — rename REWRITES these hrefs, so
# this never goes in the BASE set.
_MD_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\([^)\n]*\)")

# HTML tags with their attributes (<img alt="Neural Networks" ...>)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")

# HTML comments
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Heading lines (# ... at line start)
_HEADING_RE = re.compile(r"^#{1,6} .+$", re.MULTILINE)


# Blockquote / callout markers opening a line: "> ", ">> ", "> > ".
_QUOTE_PREFIX_RE = re.compile(r"^[ \t]*(?:>[ \t]*)+")


def _block_skip_spans(text: str) -> list[tuple[int, int]]:
    """Char spans for line-based code regions: fenced code (``` / ~~~, including
    an unclosed fence that runs to EOF) and indented (4-space/tab) code blocks.

    Line-scanned, not regex: sequential-pairing regexes can't survive an
    unbalanced fence marker (audit finding 6), and an indented code block is
    defined by a preceding blank line (CommonMark) — neither is a clean regex.

    A blockquote/callout prefix is stripped before the fence test. Without it a
    fence inside a callout (`> ```python`) was not a fence, so its language tag
    got wikilinked into `> ```[[Python]]` and — worse — the missed opener flipped
    fence parity for the rest of the note, which is how a URL two lines up came
    back as `…Article.[[HTML]]`. Measured on a real vault note.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    fence_open_at: int | None = None
    prev_blank = True          # start of doc can begin an indented code block
    in_indented = False
    for line in text.splitlines(keepends=True):
        end = pos + len(line)
        bare = _QUOTE_PREFIX_RE.sub("", line).lstrip(" \t")
        is_fence = bare.startswith(("```", "~~~"))
        is_blank = line.strip() == ""

        if fence_open_at is not None:          # inside a fence
            if is_fence:                       # closing delimiter
                spans.append((fence_open_at, end))
                fence_open_at = None
            prev_blank, in_indented, pos = False, False, end
            continue
        if is_fence:                           # opening delimiter
            fence_open_at = pos
            prev_blank, in_indented, pos = False, False, end
            continue

        indented = (line.startswith(("    ", "\t")) and not is_blank)
        if indented and (prev_blank or in_indented):
            spans.append((pos, end))
            in_indented = True
        elif not is_blank:                     # blank lines may sit inside a block
            in_indented = False
        prev_blank, pos = is_blank, end

    if fence_open_at is not None:              # unclosed fence → mask to EOF
        spans.append((fence_open_at, len(text)))
    return spans


# Shared skip-region idiom (kernel/rename.py reuses it via build_skip_mask).
# BASE = regions both callers protect; FULL adds regions only autolink skips.
# rename REWRITES wikilinks, headings and markdown-link hrefs, so those live in
# FULL, never BASE. Fenced/indented code is always masked (see build_skip_mask).
SKIP_PATTERNS_BASE = (
    _FRONTMATTER_RE,
    _INLINE_CODE_RE,
    _DISPLAY_MATH_RE,
    _INLINE_MATH_RE,
    _URL_RE,
    _INLINE_TAG_RE,
)
SKIP_PATTERNS_FULL = SKIP_PATTERNS_BASE + (
    _WIKILINK_RE,
    _MD_LINK_RE,
    _HTML_TAG_RE,
    _HTML_COMMENT_RE,
    _HEADING_RE,
)


def build_skip_mask(text: str, patterns=SKIP_PATTERNS_FULL) -> list[bool]:
    """Return a per-character boolean mask: True = inside a skip region.

    Applies `patterns` plus the always-on line-based code regions
    (`_block_skip_spans`): fenced and indented code protect both callers.
    """
    mask = [False] * len(text)
    for pattern in patterns:
        for m in pattern.finditer(text):
            for i in range(m.start(), m.end()):
                mask[i] = True
    for start, end in _block_skip_spans(text):
        for i in range(start, end):
            mask[i] = True
    return mask


def _build_skip_mask(text: str) -> list[bool]:
    return build_skip_mask(text, SKIP_PATTERNS_FULL)


# ---------------------------------------------------------------------------
# Main autolink function
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+")


@functools.lru_cache(maxsize=8)
def _word_buckets(title_index: tuple[str, ...]) -> dict[str, list[str]]:
    """first-word (lowercase) -> index titles containing that word.

    A title can only contain a candidate as whole words if it contains the
    candidate's first word as a whole word, so the containment test below walks
    one bucket instead of the index. Keyed on the tuple (not id(): the index is
    rebuilt per chunk from cached refs and an id can be reused after GC).
    """
    buckets: dict[str, list[str]] = {}
    for t in title_index:
        for w in set(_WORD_RE.findall(t.lower())):
            buckets.setdefault(w, []).append(t)
    return buckets


def _containing_titles(title_index: tuple[str, ...], work_lower: frozenset[str]) -> list[str]:
    """Non-candidate index titles that contain a candidate as whole words.

    Was O(|index| x |candidates|) regex searches per note (705 titles x 183
    candidates = 129k on the 709-note vault, which is why the cosine gate saved
    only 20% of autolink's 67 ms/note: the gate shrank the candidates and this
    scan still walked the index for each of them). Same predicate, same order
    as the index, only the titles in a candidate's word bucket are tested.
    """
    buckets = _word_buckets(title_index)
    hit: set[str] = set()
    for w in work_lower:
        first = _WORD_RE.search(w)
        if first is None:
            continue
        rx = re.compile(r"(?<!\w)" + re.escape(w) + r"(?!\w)")
        for t in buckets.get(first.group(0), ()):
            tl = t.lower()
            if tl not in work_lower and len(t) > len(w) and rx.search(tl):
                hit.add(t)
    return [t for t in title_index if t in hit]


def autolink(
    body: str,
    title_index: Sequence[str],
    candidates: Sequence[str] | None = None,
    self_title: str | None = None,
    aliases: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Wrap the first occurrence of each vault title in `body` with a wikilink.

    Args:
        body:        The full note text (including frontmatter if present).
        title_index: All vault titles that may be linked (pre-disambiguated).
        candidates:  Optional prioritized subset from embeddings.  If given,
                     only titles in candidates∩title_index are processed.
        self_title:  The title/basename of the note being processed.  When
                     provided, this title is excluded from linking — a note must
                     never contain a wikilink to itself.
        aliases:     Optional surface->canonical map from `build_alias_map`.  An
                     alias is a second spelling of a title that is already in the
                     work set, scanned exactly like the title and emitted as
                     `[[Canonical|surface]]` so the graph keeps one node.

    Returns:
        (new_body, added_links) — modified body and list of linked titles.
        Added links are canonical titles, never alias surfaces.

    Guarantees:
        - Never modifies text inside skip regions.
        - Only links titles that already exist in `title_index` (graph-safe).
        - At most one wikilink per title per call (first occurrence only).
        - Never creates a self-referential wikilink (self_title excluded).
        - Idempotent: running twice produces the same result.
    """
    if not body or not title_index:
        return body, []

    # Determine which titles to consider
    if candidates is not None:
        title_set = {t.lower(): t for t in title_index}
        work_titles = [t for t in candidates if t.lower() in title_set]
        # Canonicalise to the title_index spelling
        work_titles = [title_set[t.lower()] for t in work_titles]
    else:
        work_titles = list(title_index)

    # Exclude the note's own title to prevent self-referential wikilinks
    if self_title:
        _self_lower = self_title.lower()
        work_titles = [t for t in work_titles if t.lower() != _self_lower]

    if not work_titles:
        return body, []

    # A frontmatter alias joins the work set as its own surface, mapped back to
    # the title it belongs to. It rides through the rest of this function as an
    # ordinary string — the skip mask, the shadow pass and longest-first all
    # operate on surfaces, not on titles — and only the emitted target differs.
    # An alias whose note is not in the work set is dropped here rather than in
    # build_alias_map: the candidates gate narrows per note, the alias map is
    # vault-wide.
    target_of: dict[str, str] = {t: t for t in work_titles}
    if aliases:
        by_lower = {t.lower(): t for t in work_titles}
        for surface, canonical in aliases.items():
            owner = by_lower.get(canonical.lower())
            if owner is not None and surface.lower() not in by_lower:
                target_of[surface] = owner

    # Sort longest-first: prevents short titles from shadowing longer ones
    # ("Deep Learning" before "Learning")
    work_titles = sorted(target_of, key=len, reverse=True)

    # A longer vault title OWNS its mentions even when it is not a candidate:
    # "statistica descrittiva" is an occurrence of [[Statistica descrittiva]],
    # never of [[Statistica]]. Longest-first only arbitrates among candidates,
    # so narrowing the set made the generic link WORSE (285-note A/B:
    # [[Statistica]] false positives rose 15x → 16x from T=0.00 to T=0.40 as
    # the specific title fell below threshold). Mask every occurrence of a
    # longer index title that contains a candidate as a whole word.
    shadow_res: list[re.Pattern] = []
    if candidates is not None:
        work_lower = {t.lower() for t in work_titles}
        shadow_res = [
            re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.IGNORECASE)
            for t in _containing_titles(tuple(title_index), frozenset(work_lower))
        ]

    # Pre-scan: collect titles that are already wikilinked in the body.
    # A note should have at most one [[title]] link — if it's already there,
    # skip that title entirely regardless of where in the body it appears.
    # Path-qualified targets ([[topics/Python]]) also register their basename
    # ("python") so we don't add a second, redundant [[Python]] (audit §3).
    from silica.kernel.link.ast import WIKILINK_TARGET_RE
    existing_links: set[str] = set()
    for _m in WIKILINK_TARGET_RE.findall(body):
        t = _m.strip().lower()
        if t:
            existing_links.add(t)
            existing_links.add(t.rsplit("/", 1)[-1])

    added: list[str] = []
    current = body

    def _mask_of(text: str) -> list[bool]:
        m = _build_skip_mask(text)
        for rx in shadow_res:
            for sm in rx.finditer(text):
                for i in range(sm.start(), sm.end()):
                    m[i] = True
        return m

    # Skip mask built once; rebuilt only after an actual substitution shifts
    # positions. When most titles don't match (full-index fallback), this is the
    # difference between one mask build and one-per-title (audit §4).
    mask = _mask_of(current)

    for surface in work_titles:
        title = target_of[surface]   # the link target; == surface unless aliased

        if len(surface) < 2:
            continue  # single-character titles are too noisy

        if title.lower() in existing_links:
            continue  # already linked elsewhere in the note — skip

        # Build case-insensitive whole-word pattern.
        # `(?!\.\w)`: a title followed by a dot-and-letter is the stem of a
        # FILENAME, not prose — "Inbox excerpt: Lezione 9.md" was becoming
        # "[[Lezione 9]].md", a link with its extension hanging outside the
        # brackets (5 notes of one real run). A sentence-final "Lezione 9." is
        # unaffected: the dot there is followed by a space.
        escaped = re.escape(surface)
        pattern = re.compile(
            r"(?<!\[)(?<!\w)" + escaped + r"(?!\w)(?!\])(?!\.\w)",
            re.IGNORECASE,
        )

        # Find the first match that is NOT inside a skip region
        match = None
        for m in pattern.finditer(current):
            if not any(mask[i] for i in range(m.start(), m.end())):
                match = m
                break

        if match is None:
            continue

        # Preserve the body's casing as an alias when it differs from the
        # canonical title ([[Neural Networks|neural networks]]) — otherwise the
        # canonical form rewrites mid-sentence prose (audit §3).
        matched_text = current[match.start() : match.end()]
        # Inside a table row the alias pipe is a column separator, so an alias
        # link silently adds a column to that row (health integrity_probe, 5
        # notes). `\|` is the GFM/Obsidian escape and is what the vault's
        # hand-written tables already use.
        line_start = current.rfind("\n", 0, match.start()) + 1
        sep = "\\|" if current[line_start:].lstrip().startswith("|") else "|"
        link = f"[[{title}]]" if matched_text == title else f"[[{title}{sep}{matched_text}]]"
        current = current[: match.start()] + link + current[match.end() :]
        added.append(title)
        existing_links.add(title.lower())  # prevent duplicates within this call
        mask = _mask_of(current)           # positions shifted — rebuild

    return current, added


# ---------------------------------------------------------------------------
# Reverse-link pass — inject links to newly created notes into pre-existing ones
# ---------------------------------------------------------------------------

def backlink_pass(
    new_titles: list[str],
    *,
    title_index: list[str],
    neighbourhood: list[str],
) -> dict[str, list[str]]:
    """For each note in `neighbourhood`, autolink only the `new_titles`.

    Runs `autolink(body, title_index, candidates=new_titles)` on every neighbour,
    wrapping mentions of newly-created notes with wikilinks in pre-existing content.
    Returns {path: titles_added}. Inherits all autolink() guarantees (graph-safe,
    skip-region aware, idempotent). Best-effort: per-note failures are logged and
    skipped.
    """
    import os as _os
    from silica.driver import DRIVER
    from silica.kernel.recall.paths import is_inbox_path
    from silica.kernel.vault_manifest import active_write_dir, within

    # This path rewrites pre-existing notes without passing validate_operations,
    # so it enforces the vault write boundary itself: on a vault that reads a
    # whole source tree, a backlink must never edit its README or its docs.
    write_root = active_write_dir()

    result: dict[str, list[str]] = {}
    for path in neighbourhood:
        if write_root and not within(path, write_root):
            continue
        # The inbox is staging, never a write target — and the guard above
        # cannot stand in for this one: a vault with no vault.yaml has no write
        # dir, so `write_root` is "" and `within` returns True for everything.
        # The neighbourhood sweep excludes source leaves but not staging chunks,
        # so wikilinks were being injected into the very text CLEANUP later
        # copies into the verbatim source leaf, and "verbatim" stopped holding.
        if is_inbox_path(path):
            continue
        try:
            nc = DRIVER.read_note(path)
            body = nc.content or ""
            if not body.strip():
                continue
            stem = _os.path.splitext(_os.path.basename(path))[0]
            new_body, added = autolink(body, title_index, candidates=new_titles, self_title=stem)
            if added:
                DRIVER.overwrite(path, new_body)
                result[path] = added
                import logging as _l
                _l.getLogger(__name__).info("BACKLINK: %s ← %s", path, added)
        except Exception as _e:
            import logging as _l
            _l.getLogger(__name__).debug("BACKLINK: skipped '%s' (non-fatal): %s", path, _e)
    return result


# ---------------------------------------------------------------------------
# Title index helpers
# ---------------------------------------------------------------------------

def build_title_index(refs: list) -> list[str]:
    """Build a disambiguated title list from driver NoteRef objects.

    Drops any title that appears more than once (basename conflict) — such
    titles cannot be safely linked without an explicit path qualifier. The count
    is case-insensitive to match autolink's IGNORECASE matching: `Foo` and `foo`
    are ambiguous together and both dropped (audit §3).

    Args:
        refs: list of NoteRef objects with `.name` attribute.

    Returns:
        Sorted list of unique, unambiguous display names.
    """
    from collections import Counter

    lower_counts: Counter[str] = Counter()
    first_casing: dict[str, str] = {}
    for ref in refs:
        name = ref if isinstance(ref, str) else (getattr(ref, "name", None) or "")
        if name:
            lower_counts[name.lower()] += 1
            first_casing.setdefault(name.lower(), name)

    return sorted(
        first_casing[lc] for lc, count in lower_counts.items() if count == 1
    )


def build_alias_map(
    pairs, title_index: Sequence[str]
) -> dict[str, str]:
    """Build the alias surface -> canonical title map from frontmatter aliases.

    Args:
        pairs:       iterable of (title, aliases) as harvested from the vault
                     (see DRIVER.alias_index / frontmatter.aliases_of).
        title_index: the disambiguated title list the links will target.

    An alias is dropped when it collides with a real note title, when two notes
    claim it, or when its own note is missing from `title_index` — the same
    ambiguity rule build_title_index applies to titles, so an ambiguous note
    cannot come back through the alias door. Surfaces are lowercased: autolink
    matches case-insensitively and preserves the body's own casing in the pipe.
    """
    index = {t.lower(): t for t in title_index}
    claims: dict[str, set[str]] = {}

    for title, alias_list in pairs:
        canonical = index.get(str(title or "").lower())
        if canonical is None:
            continue
        for alias in alias_list or []:
            surface = str(alias).strip().lower()
            # `surface in index`: a note titled "AI" outranks another note's
            # "AI" alias, always. `len < 2` mirrors autolink's own noise floor.
            if len(surface) < 2 or surface in index:
                continue
            claims.setdefault(surface, set()).add(canonical)

    return {
        surface: next(iter(owners))
        for surface, owners in claims.items()
        if len(owners) == 1
    }
