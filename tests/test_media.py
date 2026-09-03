"""Unit tests for silica.kernel.text.media — strip_images and section images."""
from __future__ import annotations

import textwrap

from silica.kernel.text.media import (
    strip_images,
    images_for_section,
    append_section_images,
)


# ---------------------------------------------------------------------------
# strip_images — Obsidian-flavor embeds
# ---------------------------------------------------------------------------

class TestStripOFMImages:
    def test_bare_jpg(self):
        text = "Before\n![[images/a1e8022c.jpg]]\nAfter"
        result = strip_images(text)
        assert "![[" not in result
        assert "Before" in result
        assert "After" in result

    def test_bare_png(self):
        assert "![[" not in strip_images("![[screenshot.png]]")

    def test_bare_gif(self):
        assert "![[" not in strip_images("![[anim.gif]]")

    def test_bare_webp(self):
        assert "![[" not in strip_images("![[photo.webp]]")

    def test_bare_svg(self):
        assert "![[" not in strip_images("![[icon.svg]]")

    def test_size_hint(self):
        """![[file.png|300]] — Obsidian size hint must be stripped too."""
        result = strip_images("![[diagram.png|300]]")
        assert "![[" not in result

    def test_size_hint_wh(self):
        result = strip_images("![[diagram.png|200x400]]")
        assert "![[" not in result

    def test_path_with_subdirs(self):
        raw = "![[attachments/2024/screenshot.jpeg]]"
        assert strip_images(raw).strip() == ""

    def test_uppercase_extension(self):
        raw = "![[Photo.JPG]]"
        assert "![[" not in strip_images(raw)

    def test_multiple_embeds(self):
        raw = "Title\n![[a.jpg]]\nText\n![[b.png|100]]\nEnd"
        result = strip_images(raw)
        assert "![[" not in result
        assert "Title" in result
        assert "Text" in result
        assert "End" in result


# ---------------------------------------------------------------------------
# strip_images — Standard Markdown images
# ---------------------------------------------------------------------------

class TestStripMarkdownImages:
    def test_empty_alt(self):
        raw = "![](images/a1e8022c.jpg)"
        assert "![]" not in strip_images(raw)
        assert strip_images(raw).strip() == ""

    def test_with_alt_text(self):
        raw = "![Figure 1](images/fig1.png)"
        result = strip_images(raw)
        assert "!["not in result

    def test_remote_url(self):
        raw = "![logo](https://example.com/logo.png)"
        assert strip_images(raw).strip() == ""

    def test_mixed_with_text(self):
        raw = "Some text.\n![](img.jpg)\nMore text."
        result = strip_images(raw)
        assert "Some text." in result
        assert "More text." in result
        assert "!["  not in result

    def test_inline_in_paragraph(self):
        raw = "See the ![diagram](diag.png) below for details."
        result = strip_images(raw)
        assert "See the" in result
        assert "below for details." in result
        assert "![" not in result


# ---------------------------------------------------------------------------
# strip_images — things that must NOT be stripped
# ---------------------------------------------------------------------------

class TestStripPreservation:
    def test_wikilink_untouched(self):
        raw = "[[NeuralNetwork]] is connected to [[Backprop]]."
        assert strip_images(raw) == raw

    def test_plain_text_untouched(self):
        raw = "The quick brown fox jumps over the lazy dog."
        assert strip_images(raw) == raw

    def test_hyperlink_not_image(self):
        """[text](url) without leading ! must not be stripped."""
        raw = "[See docs](https://example.com/docs)"
        assert strip_images(raw) == raw

    def test_markdown_bold_untouched(self):
        raw = "**Bold text** and *italic*."
        assert strip_images(raw) == raw

    def test_empty_string(self):
        assert strip_images("") == ""

    def test_blank_line_collapse(self):
        """Multiple blank lines left by removed embeds are collapsed to one."""
        raw = "Line A\n\n![[a.jpg]]\n\n![[b.png]]\n\nLine B"
        result = strip_images(raw)
        assert "Line A" in result
        assert "Line B" in result
        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# images_for_section — section-scoped image embeds for a concept
# ---------------------------------------------------------------------------

_SRC = textwrap.dedent("""\
    # Doc

    ## Architecture

    The system uses a pipeline.

    ![[images/arch.png]]

    ## Storage

    Uses LanceDB.

    ![Figure 2](images/db.png)
    ![remote](https://example.com/logo.png)
""")


class TestImagesForSection:
    def test_ofm_embed_normalized_to_basename(self):
        assert images_for_section(_SRC, "Architecture") == ["![[arch.png]]"]

    def test_markdown_image_normalized_skips_remote(self):
        # db.png is local → kept; the https logo → dropped (would point nowhere).
        assert images_for_section(_SRC, "Storage") == ["![[db.png]]"]

    def test_no_matching_heading_returns_empty(self):
        assert images_for_section(_SRC, "Nonexistent") == []

    def test_concept_substring_of_heading_matches(self):
        src = "## Message provenance and tracking\n![[p.png]]\n"
        assert images_for_section(src, "Message provenance") == ["![[p.png]]"]

    def test_dedup_same_image_twice(self):
        src = "## S\n![[a.png]]\ntext\n![[images/a.png]]\n"
        assert images_for_section(src, "S") == ["![[a.png]]"]

    def test_no_images_in_section(self):
        assert images_for_section("## S\nplain text only\n", "S") == []


class TestAppendSectionImages:
    def test_appends_after_snippet(self):
        out = append_section_images("distilled body.", _SRC, "Architecture")
        assert out.startswith("distilled body.")
        assert out.rstrip().endswith("![[arch.png]]")

    def test_noop_when_no_section_images(self):
        assert append_section_images("body", _SRC, "Nonexistent") == "body"

    def test_handles_empty_snippet(self):
        out = append_section_images("", _SRC, "Architecture")
        assert out.strip() == "![[arch.png]]"
