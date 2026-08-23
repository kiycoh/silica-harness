"""Phase 4 tests — autolink: deterministic wikilink injector."""
from __future__ import annotations

import pytest
from silica.kernel.link.autolink import autolink, build_alias_map, build_title_index


# ---------------------------------------------------------------------------
# Basic linking
# ---------------------------------------------------------------------------

def test_autolink_adds_wikilink_for_matching_title():
    body = "Neural Networks are powerful tools."
    new_body, added = autolink(body, ["Neural Networks"])
    assert "[[Neural Networks]]" in new_body
    assert "Neural Networks" in added


def test_autolink_case_insensitive():
    body = "We study neural networks."
    new_body, added = autolink(body, ["Neural Networks"])
    # Casing differs from the canonical title → alias-preserving link so the
    # body prose keeps its own casing (audit §3).
    assert "[[Neural Networks|neural networks]]" in new_body
    assert "Neural Networks" in added


def test_autolink_links_first_occurrence_only():
    body = "Neural Networks are great. Neural Networks are fun."
    new_body, added = autolink(body, ["Neural Networks"])
    assert new_body.count("[[Neural Networks]]") == 1
    assert "Neural Networks" in added


def test_autolink_noncandidate_longer_title_shadows_its_words():
    """A mention owned by a LONGER vault title must not be linked as the
    shorter title it contains, even when the longer title is not a candidate.
    Longest-first ordering only protected among candidates, so narrowing the
    candidate set (embedding threshold) made the generic link WORSE: on the
    285-note A/B, [[Statistica]] false positives rose 15x → 16x going from
    T=0.00 to T=0.40 because [[Statistica descrittiva]] fell out of the set."""
    body = "La statistica descrittiva studia i dati raccolti."
    new_body, added = autolink(
        body,
        ["Statistica", "Statistica descrittiva"],
        candidates=["Statistica"],
    )
    assert added == []
    assert new_body == body


def test_autolink_shadowed_word_still_links_standalone_occurrence():
    """Shadowing masks only the longer title's own occurrences: a standalone
    mention of the shorter candidate elsewhere in the body must still link."""
    body = "La statistica descrittiva studia i dati. La statistica è ampia."
    new_body, added = autolink(
        body,
        ["Statistica", "Statistica descrittiva"],
        candidates=["Statistica"],
    )
    assert added == ["Statistica"]
    assert "La [[Statistica|statistica]] è ampia." in new_body
    assert "[[Statistica|statistica]] descrittiva" not in new_body


def test_autolink_no_match_returns_unchanged():
    body = "This note talks about attention mechanisms."
    new_body, added = autolink(body, ["Transformers"])
    assert new_body == body
    assert added == []


def test_autolink_multiple_titles():
    body = "Neural Networks and Backpropagation are key concepts."
    new_body, added = autolink(body, ["Neural Networks", "Backpropagation"])
    assert "[[Neural Networks]]" in new_body
    assert "[[Backpropagation]]" in new_body
    assert len(added) == 2


# ---------------------------------------------------------------------------
# Skip regions
# ---------------------------------------------------------------------------

def test_autolink_skips_frontmatter():
    body = "---\ntitle: Neural Networks\ntags: [AI]\n---\nNeural Networks are great."
    new_body, added = autolink(body, ["Neural Networks"])
    # The frontmatter title should NOT be linked, but the body occurrence should
    assert "[[Neural Networks]]" in new_body
    lines = new_body.split("\n")
    # Frontmatter lines should be unchanged
    assert lines[1] == "title: Neural Networks"


def test_autolink_skips_fenced_code():
    body = "Study Neural Networks.\n```python\n# Neural Networks example\npass\n```"
    new_body, added = autolink(body, ["Neural Networks"])
    # Only the first occurrence (before the code block) should be linked
    assert new_body.count("[[Neural Networks]]") == 1
    assert "# Neural Networks example" in new_body  # code unchanged


def test_autolink_never_links_fence_info_string_even_when_unbalanced():
    # An unbalanced fence elsewhere used to shift the sequential-pairing mask and
    # leave a later info string exposed: ```python → ```[[Python]] (audit finding 5).
    body = (
        "See ```unclosed fence with no partner\n\n"
        "Later:\n```python\nx = 1\n```\n"
    )
    new_body, added = autolink(body, ["Python"])
    assert "```python" in new_body          # info string intact
    assert "[[Python]]" not in new_body     # fence delimiter never linked


def test_autolink_skips_inline_code():
    body = "The `Neural Networks` module. Neural Networks are great."
    new_body, added = autolink(body, ["Neural Networks"])
    # Inline code should be skipped; plain text occurrence should be linked
    assert "`Neural Networks`" in new_body  # inline code unchanged
    assert "[[Neural Networks]]" in new_body


def test_autolink_skips_existing_wikilinks():
    body = "See [[Neural Networks]] for details. Neural Networks matter."
    new_body, added = autolink(body, ["Neural Networks"])
    # Already has [[Neural Networks]] — should not add a second one
    # (the plain-text occurrence after it is the second, not first)
    # Since [[Neural Networks]] is in a skip region, the plain text is the first non-skip match
    # But we still want idempotency: no double-link
    assert new_body.count("[[Neural Networks]]") >= 1
    # Added list should be empty since no NEW link was created in skip-free text
    # (the first occurrence is inside [[...]] which is skipped)
    assert added == []


def test_autolink_skips_math_display_block():
    body = "$$\nNeural Networks equation\n$$\nNeural Networks are great."
    new_body, added = autolink(body, ["Neural Networks"])
    assert "[[Neural Networks]]" in new_body
    assert new_body.count("[[Neural Networks]]") == 1


def test_autolink_skips_heading_lines():
    body = "# Neural Networks\n\nNeural Networks are powerful."
    new_body, added = autolink(body, ["Neural Networks"])
    # Heading line should be unchanged
    assert new_body.startswith("# Neural Networks\n")
    # Body paragraph should be linked
    assert "[[Neural Networks]]" in new_body


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_autolink_idempotent():
    body = "Neural Networks are powerful tools for learning."
    new_body1, added1 = autolink(body, ["Neural Networks"])
    new_body2, added2 = autolink(new_body1, ["Neural Networks"])
    assert new_body1 == new_body2
    assert added2 == []  # second pass adds nothing


# ---------------------------------------------------------------------------
# Candidates (embedding-prioritized subset)
# ---------------------------------------------------------------------------

def test_autolink_candidates_restricts_linking():
    body = "Neural Networks and Backpropagation are important."
    # candidates only has Neural Networks → only that gets linked
    new_body, added = autolink(
        body,
        title_index=["Neural Networks", "Backpropagation"],
        candidates=["Neural Networks"],
    )
    assert "[[Neural Networks]]" in new_body
    assert "[[Backpropagation]]" not in new_body
    assert added == ["Neural Networks"]


def test_autolink_candidates_empty_list_no_links():
    body = "Neural Networks are great."
    new_body, added = autolink(body, ["Neural Networks"], candidates=[])
    # Empty candidates → no titles to process
    assert new_body == body
    assert added == []


# ---------------------------------------------------------------------------
# Word-boundary matching
# ---------------------------------------------------------------------------

def test_autolink_whole_word_only():
    """'Net' should not match inside 'Network'."""
    body = "Neural Networks is not just a Net."
    new_body, added = autolink(body, ["Net"])
    # 'Net' appears as a whole word → should be linked
    assert "[[Net]]" in new_body
    # 'Networks' should NOT become '[[Net]]works'
    assert "[[Net]]works" not in new_body


def test_autolink_does_not_link_single_char_title():
    body = "The A in AI stands for artificial."
    new_body, added = autolink(body, ["A"])
    assert new_body == body
    assert added == []


# ---------------------------------------------------------------------------
# Longest-first ordering
# ---------------------------------------------------------------------------

def test_autolink_longer_title_takes_precedence():
    """'Deep Learning' should be linked as a unit, not 'Learning' separately."""
    body = "Deep Learning is a subset of Machine Learning."
    new_body, added = autolink(body, ["Deep Learning", "Learning"])
    assert "[[Deep Learning]]" in new_body
    # The standalone 'Learning' in 'Machine Learning' may or may not be linked
    # — the important thing is Deep Learning is handled as a unit


# ---------------------------------------------------------------------------
# Empty / edge inputs
# ---------------------------------------------------------------------------

def test_autolink_empty_body():
    new_body, added = autolink("", ["Neural Networks"])
    assert new_body == ""
    assert added == []


def test_autolink_empty_title_index():
    body = "Neural Networks are great."
    new_body, added = autolink(body, [])
    assert new_body == body
    assert added == []


# ---------------------------------------------------------------------------
# build_title_index — disambiguation
# ---------------------------------------------------------------------------

def test_build_title_index_deduplicates():
    """Two refs with the same name → dropped (ambiguous)."""
    from unittest.mock import MagicMock

    ref_a = MagicMock()
    ref_a.name = "Neural Networks"
    ref_b = MagicMock()
    ref_b.name = "Neural Networks"  # duplicate
    ref_c = MagicMock()
    ref_c.name = "Backpropagation"

    index = build_title_index([ref_a, ref_b, ref_c])
    assert "Neural Networks" not in index
    assert "Backpropagation" in index


def test_build_title_index_unique_titles_kept():
    from unittest.mock import MagicMock

    refs = []
    for name in ("A", "B", "C"):
        r = MagicMock()
        r.name = name
        refs.append(r)

    index = build_title_index(refs)
    assert sorted(index) == ["A", "B", "C"]


def test_build_title_index_sorted():
    from unittest.mock import MagicMock

    refs = []
    for name in ("Zig", "Alpha", "Middle"):
        r = MagicMock()
        r.name = name
        refs.append(r)

    index = build_title_index(refs)
    assert index == sorted(index)


# ---------------------------------------------------------------------------
# Regression tests for structural bugs (reported post Phase 4)
# ---------------------------------------------------------------------------

def test_autolink_no_self_link():
    """A note must never wikilink to itself (self_title excluded)."""
    body = "DDS è un middleware. Il Data Distribution Service è usato in ROS."
    new_body, added = autolink(body, ["DDS", "ROS"], self_title="DDS")
    assert "[[DDS]]" not in new_body, "self-link must not be emitted"
    assert "[[ROS]]" in new_body, "other titles must still be linked"
    assert "DDS" not in added
    assert "ROS" in added


def test_autolink_self_link_case_insensitive():
    """Self-title exclusion is case-insensitive."""
    body = "HAL layers abstract the hardware. See also Linux."
    new_body, added = autolink(body, ["HAL", "Linux"], self_title="hal")
    assert "[[HAL]]" not in new_body
    assert "[[Linux]]" in new_body


def test_autolink_self_link_not_excluded_when_none():
    """When self_title is None (default), no exclusion is applied."""
    body = "PWM controls duty cycle."
    new_body, added = autolink(body, ["PWM"])
    # Without self_title, the title IS linked (previous behavior preserved)
    assert "[[PWM]]" in new_body


# ---------------------------------------------------------------------------
# Content-corruption regressions (audit 2026-07-23 §2 — incomplete skip mask)
# Each case ran `autolink()` real and produced corruption before the fix.
# ---------------------------------------------------------------------------

def test_autolink_skips_bare_url():
    body = "See https://example.com/page for more."
    new_body, added = autolink(body, ["example", "page"])
    assert new_body == body
    assert added == []


def test_autolink_skips_markdown_link_text():
    body = "Read [intro to Neural Networks](http://u) today."
    new_body, added = autolink(body, ["Neural Networks"])
    assert new_body == body  # link text is not prose to link


def test_autolink_skips_url_inside_markdown_link():
    body = "Read [the docs](https://x.com/page) now."
    new_body, added = autolink(body, ["page"])
    assert new_body == body


def test_autolink_does_not_kill_inline_tag():
    body = "This note is about #Python and its ecosystem."
    new_body, added = autolink(body, ["Python"])
    assert "#Python" in new_body            # tag survives intact
    assert "#[[Python]]" not in new_body    # never rewritten into the tag


def test_autolink_skips_indented_code_block():
    body = "Prose here.\n\n    import Python\n    Python.run()\n\nMore prose."
    new_body, added = autolink(body, ["Python"])
    assert "    import Python" in new_body
    assert "[[Python]]" not in new_body


def test_autolink_skips_unclosed_fence_to_eof():
    # A fence opened and never closed leaves its body exposed to matching
    # unless masked to EOF (audit finding 6).
    body = "intro\n```python\nresult = Python.run()\n"
    new_body, added = autolink(body, ["Python"])
    assert new_body == body
    assert added == []


def test_autolink_skips_crlf_frontmatter():
    # Windows line endings must not defeat frontmatter masking — backlink_pass
    # rewrites pre-existing USER notes, which may be CRLF (audit finding 7).
    body = "---\r\ntitle: Neural Networks\r\ntags: [ai]\r\n---\r\nNeural Networks rock."
    new_body, added = autolink(body, ["Neural Networks"])
    assert "title: Neural Networks" in new_body            # frontmatter untouched
    assert "[[Neural Networks]]" in new_body               # body occurrence linked
    assert new_body.count("[[Neural Networks]]") == 1


def test_autolink_skips_html_attribute():
    body = 'Diagram: <img alt="Neural Networks flow" src="x.png"> below.'
    new_body, added = autolink(body, ["Neural Networks"])
    assert new_body == body


# ---------------------------------------------------------------------------
# Link-coherence regressions (audit 2026-07-23 §3)
# ---------------------------------------------------------------------------

def test_autolink_preserves_body_casing_as_alias():
    body = "we love neural networks a lot"
    new_body, added = autolink(body, ["Neural Networks"])
    assert "[[Neural Networks|neural networks]]" in new_body
    # And it stays idempotent through the alias
    again, added2 = autolink(new_body, ["Neural Networks"])
    assert again == new_body
    assert added2 == []


def test_autolink_exact_casing_uses_plain_link():
    body = "Neural Networks are powerful."
    new_body, added = autolink(body, ["Neural Networks"])
    assert "[[Neural Networks]]" in new_body
    assert "|" not in new_body  # no needless alias when casing matches


def test_autolink_path_qualified_link_blocks_duplicate():
    # [[topics/Python]] already links the note — do not add a bare [[Python]].
    body = "See [[topics/Python]]. Python is great."
    new_body, added = autolink(body, ["Python"])
    assert new_body.count("[[") == 1
    assert added == []


def test_build_title_index_case_insensitive_dedup():
    # Foo and foo are ambiguous under IGNORECASE matching → both dropped.
    index = build_title_index(["Foo", "foo", "Bar"])
    assert index == ["Bar"]


def test_autolink_escapes_the_alias_pipe_inside_a_table_row():
    """An alias link in a table cell must not add a column.

    The pipe in `[[Title|alias]]` is a column separator inside a table row, so a
    cased alias silently widened the row (health.integrity_probe caught it on 5
    vault notes). `\\|` is the GFM/Obsidian escape, and the vault's hand-written
    tables already use it. Asserted through the linter, which is the instrument
    that has to agree.
    """
    from silica.kernel.link.health import lint

    body = "| Concept | Note |\n| --- | --- |\n| neural networks | see above |\n"
    before = lint.totals(lint.scan(body, "Stem"))
    new_body, added = autolink(body, ["Neural Networks"])

    assert added == ["Neural Networks"]
    assert "[[Neural Networks\\|neural networks]]" in new_body
    # the transform introduced no structural violation: same column count
    assert lint.totals(lint.scan(new_body, "Stem")) == before

    # outside a table the alias pipe stays bare (escaping there would render).
    prose, _ = autolink("I like neural networks a lot.", ["Neural Networks"])
    assert "[[Neural Networks|neural networks]]" in prose


# ---------------------------------------------------------------------------
# Frontmatter aliases — a second surface for a title, one node in the graph
# ---------------------------------------------------------------------------

def test_autolink_links_an_alias_to_its_canonical_title():
    aliases = build_alias_map([("Artificial Intelligence", ["AI"])], ["Artificial Intelligence"])
    body = "The AI winter ended."
    new_body, added = autolink(body, ["Artificial Intelligence"], aliases=aliases)
    # the target is the note, the visible text is the body's own word
    assert "[[Artificial Intelligence|AI]]" in new_body
    assert added == ["Artificial Intelligence"]


def test_autolink_alias_is_idempotent_and_does_not_double_link():
    aliases = build_alias_map([("Artificial Intelligence", ["AI"])], ["Artificial Intelligence"])
    body = "AI matters. Artificial Intelligence matters more."
    once, added = autolink(body, ["Artificial Intelligence"], aliases=aliases)
    twice, again = autolink(once, ["Artificial Intelligence"], aliases=aliases)
    # one link for the note, whichever surface came first — not one per surface
    assert once.count("[[Artificial Intelligence") == 1
    assert added == ["Artificial Intelligence"]
    assert twice == once and again == []


def test_autolink_alias_respects_the_candidate_gate():
    aliases = build_alias_map([("Artificial Intelligence", ["AI"])], ["Artificial Intelligence"])
    body = "The AI winter ended."
    # the note is not a candidate for this body → neither is its alias
    new_body, added = autolink(body, ["Artificial Intelligence"], candidates=[], aliases=aliases)
    assert new_body == body and added == []


def test_autolink_alias_excluded_on_its_own_note():
    aliases = build_alias_map([("Artificial Intelligence", ["AI"])], ["Artificial Intelligence"])
    body = "AI is the subject of this very note."
    new_body, added = autolink(
        body, ["Artificial Intelligence"], self_title="Artificial Intelligence", aliases=aliases
    )
    assert new_body == body and added == []


def test_build_alias_map_drops_alias_claimed_by_two_notes():
    m = build_alias_map(
        [("Artificial Intelligence", ["AI"]), ("Adobe Illustrator", ["AI"])],
        ["Artificial Intelligence", "Adobe Illustrator"],
    )
    assert m == {}


def test_build_alias_map_drops_alias_colliding_with_a_real_title():
    # a note titled "AI" outranks another note's "AI" alias, always
    m = build_alias_map([("Artificial Intelligence", ["ai"])], ["Artificial Intelligence", "AI"])
    assert m == {}


def test_build_alias_map_drops_alias_of_a_note_absent_from_the_index():
    # build_title_index drops ambiguous titles; the alias door must stay shut too
    m = build_alias_map([("Ambiguous", ["Amb"])], ["Artificial Intelligence"])
    assert m == {}


def test_build_alias_map_skips_single_char_and_blank_surfaces():
    m = build_alias_map([("Artificial Intelligence", ["A", " ", "AI"])], ["Artificial Intelligence"])
    assert m == {"ai": "Artificial Intelligence"}


def test_aliases_of_reads_the_obsidian_spellings():
    from silica.kernel.write.frontmatter import aliases_of

    assert aliases_of("---\naliases: [AI, ANI]\n---\n\nbody\n") == ["AI", "ANI"]
    assert aliases_of("---\naliases:\n  - AI\n  - ANI\n---\n\nbody\n") == ["AI", "ANI"]
    assert aliases_of("---\nalias: AI, ANI\n---\n\nbody\n") == ["AI", "ANI"]
    assert aliases_of("---\ntags: [x]\n---\n\nbody\n") == []
    assert aliases_of("no frontmatter here") == []
    assert aliases_of("---\naliases: [\n---\n\nbroken yaml\n") == []


def test_containing_titles_matches_the_brute_force_predicate():
    """The bucketed scan must equal the index-wide regex it replaced, on a
    vocabulary built to produce shared words, punctuation and case noise."""
    import random
    import re as _re
    from silica.kernel.link.autolink import _containing_titles

    rng = random.Random(7)
    words = ["Statistica", "descrittiva", "rete", "neurale", "Deep", "learning", "K", "mean", "A-B", "tipo"]
    index = sorted({" ".join(rng.sample(words, rng.randint(1, 3))) for _ in range(120)})
    cands = frozenset(t.lower() for t in rng.sample(index, 25))
    brute = [
        t for t in index
        if t.lower() not in cands
        and any(len(t) > len(w) and _re.search(r"(?<!\w)" + _re.escape(w) + r"(?!\w)", t.lower()) for w in cands)
    ]
    assert _containing_titles(tuple(index), cands) == brute
    assert brute, "fixture must exercise at least one shadow"
