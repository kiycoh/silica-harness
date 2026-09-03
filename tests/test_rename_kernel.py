"""Table-driven tests for silica.kernel.link.rename.rewrite_links.

Cases covered (one section per numbered spec requirement):
  1. Name-based wikilink, basename UNCHANGED (pure folder move) — not rewritten
  2. Name-based wikilink, basename CHANGED (rename) — rewritten; aliases preserved
  3. Heading / block suffixes preserved verbatim
  4. Path-based wikilinks — rewritten by full-path match (case-insensitive)
  5. Markdown links — relative hrefs rewritten; %20 / angle-bracket forms handled
  6. Skip regions — fenced code, inline code, frontmatter, display math, inline math
  7. Case-insensitive matching; whole-target-only guard (no partial matches)
  Extras:
  - rewrite_name_links=False — name-based skip, path-based still runs
  - n (count) return value
  - embed links (![[...]])
"""
from __future__ import annotations

from silica.kernel.link.rename import rewrite_links


# ---------------------------------------------------------------------------
# 1. Pure folder move — basename unchanged — name links NOT rewritten
# ---------------------------------------------------------------------------

class TestFolderMove:
    """old_path and new_path share the same basename → name links untouched."""

    def test_simple_wikilink_unchanged(self):
        content = "See [[Note]] for details."
        new, n = rewrite_links(content, "A/Note.md", "B/Note.md")
        assert new == content
        assert n == 0

    def test_wikilink_with_alias_unchanged(self):
        content = "Check [[Note|the note]] for info."
        new, n = rewrite_links(content, "A/Note.md", "B/Note.md")
        assert new == content
        assert n == 0

    def test_embed_unchanged(self):
        content = "![[Note]] is embedded."
        new, n = rewrite_links(content, "A/Note.md", "B/Note.md")
        assert new == content
        assert n == 0

    def test_multiple_occurrences_unchanged(self):
        content = "[[Note]] and again [[Note|alias]]."
        new, n = rewrite_links(content, "A/Note.md", "B/Note.md")
        assert new == content
        assert n == 0


# ---------------------------------------------------------------------------
# 2. Rename (basename changed) — name-based wikilinks rewritten
# ---------------------------------------------------------------------------

class TestRename:
    """Basename changes → [[Old]] → [[New]], alias preserved."""

    def test_simple_rename(self):
        content = "See [[Old]] for details."
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == "See [[New]] for details."
        assert n == 1

    def test_alias_preserved(self):
        """[[Old|alias]] → [[New|alias]] — alias stays, target changes."""
        content = "See [[Old|the old note]] for details."
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == "See [[New|the old note]] for details."
        assert n == 1

    def test_no_alias_added(self):
        """[[Old]] → [[New]], NOT [[New|Old]] — Obsidian behavior."""
        content = "[[Old]]"
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == "[[New]]"
        assert n == 1

    def test_embed_rename(self):
        """![[Old]] → ![[New]]"""
        content = "![[Old]]"
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == "![[New]]"
        assert n == 1

    def test_multiple_occurrences_all_rewritten(self):
        content = "[[Old]] and [[Old|alias]] and ![[Old]]."
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == "[[New]] and [[New|alias]] and ![[New]]."
        assert n == 3

    def test_case_insensitive_match_canonical_casing(self):
        """[[old note]] (lowercase) matches 'Old Note'; rewritten to new canonical casing."""
        content = "See [[old note]] here."
        new, n = rewrite_links(content, "A/Old Note.md", "A/New Note.md")
        assert new == "See [[New Note]] here."
        assert n == 1

    def test_case_insensitive_with_alias(self):
        content = "See [[OLD NOTE|my alias]] here."
        new, n = rewrite_links(content, "A/Old Note.md", "A/New Note.md")
        assert new == "See [[New Note|my alias]] here."
        assert n == 1


# ---------------------------------------------------------------------------
# 3. Heading / block suffixes preserved verbatim
# ---------------------------------------------------------------------------

class TestSuffixes:
    """#heading and ^blockid suffixes survive the rename."""

    def test_heading_suffix(self):
        content = "[[Old#Section]]"
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == "[[New#Section]]"
        assert n == 1

    def test_block_suffix(self):
        content = "[[Old^blockid]]"
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == "[[New^blockid]]"
        assert n == 1

    def test_heading_suffix_with_alias(self):
        content = "[[Old#Section|see here]]"
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == "[[New#Section|see here]]"
        assert n == 1

    def test_block_suffix_with_alias(self):
        content = "[[Old^id|label]]"
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == "[[New^id|label]]"
        assert n == 1

    def test_heading_on_folder_move_unchanged(self):
        """[[Note#Section]] unchanged when basename not changing."""
        content = "[[Note#Section]]"
        new, n = rewrite_links(content, "A/Note.md", "B/Note.md")
        assert new == content
        assert n == 0


# ---------------------------------------------------------------------------
# 4. Path-based wikilinks — always rewritten (even on pure folder move)
# ---------------------------------------------------------------------------

class TestPathBasedWikilinks:
    """[[Folder/Old]] and [[Folder/Old.md]] match by full path."""

    def test_path_link_no_extension(self):
        content = "See [[Folder/Old]] here."
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == "See [[Folder/New]] here."
        assert n == 1

    def test_path_link_with_extension(self):
        content = "See [[Folder/Old.md]] here."
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == "See [[Folder/New.md]] here."
        assert n == 1

    def test_path_link_extension_preserved_on_move(self):
        """Pure folder move: [[A/Note]] → [[B/Note]]"""
        content = "[[A/Note]]"
        new, n = rewrite_links(content, "A/Note.md", "B/Note.md")
        assert new == "[[B/Note]]"
        assert n == 1

    def test_path_link_with_md_extension_preserved_on_move(self):
        content = "[[A/Note.md]]"
        new, n = rewrite_links(content, "A/Note.md", "B/Note.md")
        assert new == "[[B/Note.md]]"
        assert n == 1

    def test_path_link_case_insensitive(self):
        content = "[[folder/old]]"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == "[[Folder/New]]"
        assert n == 1

    def test_path_link_case_insensitive_with_extension(self):
        content = "[[FOLDER/OLD.MD]]"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == "[[Folder/New.md]]"
        assert n == 1

    def test_path_link_with_alias(self):
        content = "[[Folder/Old|my label]]"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == "[[Folder/New|my label]]"
        assert n == 1

    def test_path_link_heading_suffix(self):
        content = "[[Folder/Old#Section]]"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == "[[Folder/New#Section]]"
        assert n == 1

    def test_path_link_rewrite_name_links_false_still_rewritten(self):
        """Path-based links are always rewritten regardless of rewrite_name_links."""
        content = "[[Folder/Old]]"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md",
                                rewrite_name_links=False)
        assert new == "[[Folder/New]]"
        assert n == 1


# ---------------------------------------------------------------------------
# 5. Markdown links
# ---------------------------------------------------------------------------

class TestMarkdownLinks:
    """[text](href) — relative hrefs rewritten; http/anchor skipped."""

    def test_simple_markdown_link(self):
        content = "[text](Folder/Old.md)"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == "[text](Folder/New.md)"
        assert n == 1

    def test_markdown_link_no_extension_skipped(self):
        """Markdown links without .md extension pointing to exact old_path (minus ext)
        should NOT be rewritten — we only match exact vault-relative path."""
        content = "[text](Folder/Old)"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        # "Folder/Old" does not match "Folder/Old.md" exactly
        assert new == content
        assert n == 0

    def test_markdown_link_http_skipped(self):
        content = "[text](https://example.com/Old.md)"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == content
        assert n == 0

    def test_markdown_link_anchor_skipped(self):
        content = "[text](#section)"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == content
        assert n == 0

    def test_markdown_link_mailto_skipped(self):
        content = "[me](mailto:test@example.com)"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == content
        assert n == 0

    def test_markdown_link_percent_encoded(self):
        """[text](Folder/Old%20Note.md) → [text](Folder/New%20Note.md)"""
        content = "[text](Folder/Old%20Note.md)"
        new, n = rewrite_links(content, "Folder/Old Note.md", "Folder/New Note.md")
        assert new == "[text](Folder/New%20Note.md)"
        assert n == 1

    def test_markdown_link_angle_bracket(self):
        """[text](<Folder/Old Note.md>) → [text](<Folder/New Note.md>)"""
        content = "[text](<Folder/Old Note.md>)"
        new, n = rewrite_links(content, "Folder/Old Note.md", "Folder/New Note.md")
        assert new == "[text](<Folder/New Note.md>)"
        assert n == 1

    def test_markdown_link_case_insensitive(self):
        content = "[text](folder/old.md)"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == "[text](Folder/New.md)"
        assert n == 1

    def test_markdown_link_folder_move(self):
        content = "[text](A/Note.md)"
        new, n = rewrite_links(content, "A/Note.md", "B/Note.md")
        assert new == "[text](B/Note.md)"
        assert n == 1

    def test_markdown_link_with_fragment(self):
        """[t](Folder/Old.md#Section) → [t](Folder/New.md#Section)"""
        content = "[t](Folder/Old.md#Section)"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == "[t](Folder/New.md#Section)"
        assert n == 1

    def test_markdown_link_fragment_percent_encoded(self):
        """Fragment preserved when href uses %20 encoding."""
        content = "[t](Folder/Old%20Note.md#Sec)"
        new, n = rewrite_links(content, "Folder/Old Note.md", "Folder/New Note.md")
        assert new == "[t](Folder/New%20Note.md#Sec)"
        assert n == 1

    def test_markdown_link_fragment_angle_bracket(self):
        """Angle-bracket form with fragment: [t](<Folder/Old Note.md#Sec>)"""
        content = "[t](<Folder/Old Note.md#Sec>)"
        new, n = rewrite_links(content, "Folder/Old Note.md", "Folder/New Note.md")
        assert new == "[t](<Folder/New Note.md#Sec>)"
        assert n == 1

    def test_markdown_pure_anchor_never_touched(self):
        """[t](#sec) — pure anchor, no path component — must never be rewritten."""
        content = "[t](#sec)"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == content
        assert n == 0

    def test_markdown_link_fragment_mismatch_untouched(self):
        """[t](Other.md#sec) — path does not match old_path — untouched."""
        content = "[t](Other.md#sec)"
        new, n = rewrite_links(content, "Folder/Old.md", "Folder/New.md")
        assert new == content
        assert n == 0


# ---------------------------------------------------------------------------
# 6. Skip regions — links inside these must never be rewritten
# ---------------------------------------------------------------------------

class TestSkipRegions:

    def test_fenced_code_triple_backtick(self):
        content = "```\n[[Old]]\n```"
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == content
        assert n == 0

    def test_fenced_code_tilde(self):
        content = "~~~\n[[Old]]\n~~~"
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == content
        assert n == 0

    def test_inline_code(self):
        content = "Try `[[Old]]` for code."
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == content
        assert n == 0

    def test_frontmatter_wikilink_properties_rewritten(self):
        """Wikilink-valued frontmatter properties ARE rewritten (A19) — Silica
        stores parent note:/related:/hub: as wikilinks and Obsidian rewrites
        link-type properties on rename. Both the property and the body update."""
        content = "---\nlinks: [[Old]]\n---\n\nSee [[Old]] here."
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert "links: [[New]]" in new
        assert "See [[New]] here." in new
        assert n == 2

    def test_frontmatter_silica_shaped_properties(self):
        """The real producer shape: quoted wikilinks in parent note:/related:."""
        content = (
            '---\nparent note: "[[Old]]"\nrelated:\n  - "[[Old]]"\n---\n\nbody\n'
        )
        new, n = rewrite_links(content, "Hubs/Old.md", "Hubs/New.md")
        assert 'parent note: "[[New]]"' in new
        assert '- "[[New]]"' in new
        assert n == 2

    def test_display_math(self):
        content = "$$[[Old]]$$"
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == content
        assert n == 0

    def test_inline_math(self):
        content = "Here $[[Old]]$ is math."
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == content
        assert n == 0

    def test_outside_skip_region_still_rewritten(self):
        """Link outside fenced code is still rewritten."""
        content = "[[Old]]\n```\n[[Old]]\n```\n[[Old]]"
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == "[[New]]\n```\n[[Old]]\n```\n[[New]]"
        assert n == 2

    def test_markdown_link_in_code_block_skipped(self):
        content = "```\n[text](A/Old.md)\n```"
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert new == content
        assert n == 0


# ---------------------------------------------------------------------------
# 7. Case-insensitive matching; whole-target-only guard
# ---------------------------------------------------------------------------

class TestWholeTargetGuard:
    """[[Older]] must NOT be touched when renaming [[Old]]."""

    def test_longer_target_not_touched(self):
        content = "[[Older]] is not [[Old]]."
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert "[[Older]]" in new
        assert n == 1

    def test_prefix_not_touched(self):
        content = "[[OldVersion]] stays, [[Old]] goes."
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert "[[OldVersion]]" in new
        assert "[[New]]" in new
        assert n == 1

    def test_suffix_not_touched(self):
        content = "[[MyOld]] stays, [[Old]] goes."
        new, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert "[[MyOld]]" in new
        assert "[[New]]" in new
        assert n == 1


# ---------------------------------------------------------------------------
# rewrite_name_links=False
# ---------------------------------------------------------------------------

class TestRewriteNameLinksFalse:
    """With rewrite_name_links=False, name-based matches are skipped but
    path-based links are still rewritten."""

    def test_name_link_skipped(self):
        content = "See [[Old]] here."
        new, n = rewrite_links(content, "A/Old.md", "A/New.md",
                                rewrite_name_links=False)
        assert new == content
        assert n == 0

    def test_path_link_still_rewritten(self):
        content = "[[A/Old]] is path-based."
        new, n = rewrite_links(content, "A/Old.md", "A/New.md",
                                rewrite_name_links=False)
        assert new == "[[A/New]] is path-based."
        assert n == 1

    def test_markdown_link_still_rewritten(self):
        content = "[text](A/Old.md)"
        new, n = rewrite_links(content, "A/Old.md", "A/New.md",
                                rewrite_name_links=False)
        assert new == "[text](A/New.md)"
        assert n == 1


# ---------------------------------------------------------------------------
# Count (n) accuracy
# ---------------------------------------------------------------------------

class TestCount:

    def test_zero_matches(self):
        content = "No links here."
        _, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert n == 0

    def test_exact_count(self):
        content = "[[Old]] [[Old]] [[Old]]"
        _, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert n == 3

    def test_mixed_types_counted(self):
        content = "[[Old]] [[A/Old]] [t](A/Old.md)"
        _, n = rewrite_links(content, "A/Old.md", "A/New.md")
        assert n == 3

    def test_empty_content(self):
        new, n = rewrite_links("", "A/Old.md", "A/New.md")
        assert new == ""
        assert n == 0
