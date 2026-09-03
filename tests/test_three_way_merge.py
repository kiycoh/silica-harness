"""Tests for the 3-way merge conflict callout (Tier 2, Item 7).

Terminology:
    base     — content at snapshot time (what we expected to be there)
    current  — content on disk right now when the write lands
    incoming — what the op wants to write
"""
from silica.kernel.write.merge import (
    CONFLICT_CALLOUT_HEADER,
    LEGACY_CONFLICT_CALLOUT_HEADER,
    detect_conflict,
    inject_conflict_callout,
    three_way_merge,
)


class TestDetectConflict:
    def test_no_conflict_when_identical(self):
        assert detect_conflict("hello", "hello") is False

    def test_conflict_when_content_changed(self):
        assert detect_conflict("hello", "hello world") is True

    def test_no_conflict_when_base_is_none(self):
        """No base = new note; can't detect a concurrent modification."""
        assert detect_conflict(None, "anything") is False

    def test_no_conflict_when_current_is_none(self):
        """Current is None = note was deleted; treat as no baseline conflict."""
        assert detect_conflict("base", None) is False

    def test_conflict_when_both_non_empty_and_different(self):
        assert detect_conflict("v1 content", "v2 content") is True

    def test_whitespace_difference_is_conflict(self):
        assert detect_conflict("hello\n", "hello\n\n") is True


class TestInjectCallout:
    def test_callout_header_present(self):
        result = inject_conflict_callout("# My Note\n\nBody text.")
        assert CONFLICT_CALLOUT_HEADER in result

    def test_callout_prepended_before_content(self):
        result = inject_conflict_callout("Body.")
        idx_callout = result.index(CONFLICT_CALLOUT_HEADER)
        idx_body = result.index("Body.")
        assert idx_callout < idx_body

    def test_callout_is_valid_obsidian_callout(self):
        result = inject_conflict_callout("content")
        # Every line of the callout block must start with >
        callout_lines = [
            ln for ln in result.splitlines()
            if ln.startswith("> ")
        ]
        assert len(callout_lines) >= 1

    def test_callout_does_not_duplicate_on_repeated_call(self):
        """Idempotency guard — calling twice should not double the callout."""
        once = inject_conflict_callout("content")
        twice = inject_conflict_callout(once)
        assert twice.count(CONFLICT_CALLOUT_HEADER) == 1

    def test_legacy_italian_callout_is_not_double_stamped(self):
        """A note stamped before the header was translated must still match."""
        existing = (
            f"{LEGACY_CONFLICT_CALLOUT_HEADER}\n"
            "> This note was modified concurrently.\n\n"
            "body"
        )
        assert inject_conflict_callout(existing) == existing


class TestThreeWayMerge:
    def test_no_conflict_returns_incoming_unchanged(self):
        merged, had_conflict = three_way_merge(
            base="original",
            current="original",
            incoming="# Updated\n\nNew body.",
        )
        assert had_conflict is False
        assert merged == "# Updated\n\nNew body."

    def test_conflict_injects_callout(self):
        merged, had_conflict = three_way_merge(
            base="v1",
            current="v2 (concurrent edit)",
            incoming="# My Patch\n\nBody.",
        )
        assert had_conflict is True
        assert CONFLICT_CALLOUT_HEADER in merged

    def test_conflict_incoming_content_still_present(self):
        """The callout is prepended — the incoming content must not be lost."""
        merged, _ = three_way_merge(
            base="v1",
            current="v2",
            incoming="# Critical Section\n\nImportant body.",
        )
        assert "# Critical Section" in merged
        assert "Important body." in merged

    def test_no_base_no_conflict(self):
        merged, had_conflict = three_way_merge(
            base=None,
            current="anything",
            incoming="# New Note\n\nContent.",
        )
        assert had_conflict is False
        assert merged == "# New Note\n\nContent."

    def test_no_current_no_conflict(self):
        merged, had_conflict = three_way_merge(
            base="v1",
            current=None,
            incoming="# Fresh Start\n\nContent.",
        )
        assert had_conflict is False
        assert merged == "# Fresh Start\n\nContent."

    def test_callout_format_is_danger_type(self):
        merged, _ = three_way_merge(
            base="original",
            current="modified",
            incoming="body",
        )
        assert "[!danger]" in merged
        assert "Semantic Conflict" in merged
