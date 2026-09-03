"""Graph-safe move() for ObsidianFSBackend — TDD suite (Task 2).

Covers:
  1. Folder move (basename unchanged): name-links untouched, path-links rewritten.
  2. Rename: all link flavours rewritten; backlinks + unresolved counts correct.
  3. Ambiguity guard: two Dup.md notes; rewrite only when resolution matches.
  4. Unresolved promotion: [[Target]] becomes resolved after rename → Target.md.
  5. Move → undo round-trip: referrer files byte-identical after reverse move.
  6. graph_snapshot non-regression: counts consistent after pure folder move.
  7. Failure path: referrer write failure sets _needs_reindex, index rebuilds.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from silica.driver import fs_backend
from silica.driver.fs_backend import ObsidianFSBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(vault: Path) -> ObsidianFSBackend:
    """Return a fresh FS backend for *vault*, forcing a clean initial index."""
    b = ObsidianFSBackend(str(vault))
    b._ensure_index()
    return b


def _write(vault: Path, rel: str, content: str) -> Path:
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Test 1 — folder move (basename unchanged)
# ---------------------------------------------------------------------------

class TestFolderMove:
    """Moving a note from one folder to another without changing its name."""

    @pytest.fixture
    def vault(self, tmp_path: Path) -> Path:
        v = tmp_path / "vault"
        v.mkdir()
        # The note being moved: starts in folder A
        _write(v, "A/Note.md", "# Note\n\nSome content.\n")
        # Referrer 1: name-based link — should NOT be rewritten (basename unchanged)
        _write(v, "Ref_Name.md", "# Ref\n\nSee [[Note]] for details.\n")
        # Referrer 2: path-based link pointing at the CURRENT location A/Note
        # — MUST be rewritten to [[B/Note]] after the move
        _write(v, "Ref_Path.md", "# Ref Path\n\nSee [[A/Note]] for details.\n")
        # Pre-create the destination folder
        (v / "B").mkdir()
        return v

        # Name-based link in the other referrer is untouched (checked separately)

    def test_name_link_untouched_while_path_link_rewritten(self, vault: Path) -> None:
        """Simultaneously verify: path referrer rewritten, name referrer unchanged."""
        b = _make_backend(vault)
        b.move("A/Note.md", "B/Note.md")

        name_content = (vault / "Ref_Name.md").read_text(encoding="utf-8")
        path_content = (vault / "Ref_Path.md").read_text(encoding="utf-8")

        assert "[[Note]]" in name_content          # name-based link untouched
        assert "[[B/Note]]" in path_content        # path-based link updated
        assert "[[A/Note]]" not in path_content    # old path-based link gone

    def test_backlinks_correct_after_folder_move(self, vault: Path) -> None:
        b = _make_backend(vault)
        b.move("A/Note.md", "B/Note.md")

        backlinks = {r.path for r in b.backlinks("B/Note.md")}
        # Ref_Name has [[Note]] (name-based, still resolves to B/Note after move)
        # So B/Note.md should still have backlinks
        assert "Ref_Name.md" in backlinks

    def test_unresolved_does_not_grow(self, vault: Path) -> None:
        b = _make_backend(vault)
        unres_before = len(b.unresolved())
        b.move("A/Note.md", "B/Note.md")
        unres_after = len(b.unresolved())
        assert unres_after <= unres_before


# ---------------------------------------------------------------------------
# Test 2 — rename (basename changed)
# ---------------------------------------------------------------------------

class TestRename:
    """Renaming a note (new basename); all link flavours must be rewritten."""

    @pytest.fixture
    def vault(self, tmp_path: Path) -> Path:
        v = tmp_path / "vault"
        v.mkdir()
        _write(v, "Old.md", "# Old\n\nSome content.\n")
        # Various link flavours all pointing at Old.md
        _write(v, "Ref.md", (
            "[[Old]]\n"
            "[[Old|my alias]]\n"
            "![[Old]]\n"
            "[[Old#Section]]\n"
        ))
        return v

    def test_all_link_flavours_rewritten(self, vault: Path) -> None:
        b = _make_backend(vault)
        b.move("Old.md", "New.md")

        content = (vault / "Ref.md").read_text(encoding="utf-8")
        assert "[[New]]" in content
        assert "[[New|my alias]]" in content
        assert "![[New]]" in content
        assert "[[New#Section]]" in content
        # Old name must be gone
        assert "[[Old]]" not in content
        assert "[[Old|" not in content
        assert "![[Old]]" not in content
        assert "[[Old#" not in content

    def test_backlinks_updated(self, vault: Path) -> None:
        b = _make_backend(vault)
        b.move("Old.md", "New.md")

        backlinks_new = {r.path for r in b.backlinks("New.md")}
        assert "Ref.md" in backlinks_new

        # Old path should have no backlinks
        backlinks_old = b.backlinks("Old.md")
        assert backlinks_old == []

    def test_zero_new_unresolved(self, vault: Path) -> None:
        b = _make_backend(vault)
        unres_before = {(s, t) for s, t in b._unresolved_links}
        b.move("Old.md", "New.md")
        unres_after = {(s, t) for s, t in b._unresolved_links}
        # No new unresolved entries should have appeared from the referrer
        new_unresolved = unres_after - unres_before
        assert new_unresolved == set()

    def test_old_note_not_in_index(self, vault: Path) -> None:
        b = _make_backend(vault)
        b.move("Old.md", "New.md")
        assert "Old.md" not in b._notes
        assert "New.md" in b._notes


# ---------------------------------------------------------------------------
# Test 3 — ambiguity guard
# ---------------------------------------------------------------------------

class TestAmbiguityGuard:
    """Two notes with the same basename in different folders."""

    @pytest.fixture
    def vault(self, tmp_path: Path) -> Path:
        v = tmp_path / "vault"
        (v / "folder_a").mkdir(parents=True)
        (v / "folder_b").mkdir(parents=True)
        _write(v, "folder_a/Dup.md", "# Dup A\n\nContent A.\n")
        _write(v, "folder_b/Dup.md", "# Dup B\n\nContent B.\n")
        # Referrer in folder_a: [[Dup]] resolves to folder_a/Dup (same dir priority)
        _write(v, "folder_a/Ref.md", "See [[Dup]] for details.\n")
        return v

    def test_referrer_unchanged_when_resolves_elsewhere(self, vault: Path) -> None:
        """Moving folder_b/Dup should NOT rewrite folder_a/Ref, whose [[Dup]] resolves to folder_a/Dup."""
        b = _make_backend(vault)
        original_content = (vault / "folder_a" / "Ref.md").read_text(encoding="utf-8")
        b.move("folder_b/Dup.md", "folder_b/Dup_moved.md")

        content = (vault / "folder_a" / "Ref.md").read_text(encoding="utf-8")
        assert content == original_content, (
            "Referrer was incorrectly rewritten; its [[Dup]] resolves to folder_a/Dup, not folder_b/Dup."
        )

    def test_referrer_rewritten_when_resolves_to_moved_note(self, vault: Path) -> None:
        """Moving folder_a/Dup (the one folder_a/Ref resolves to) MUST rewrite the referrer."""
        b = _make_backend(vault)
        b.move("folder_a/Dup.md", "folder_a/Dup_renamed.md")

        content = (vault / "folder_a" / "Ref.md").read_text(encoding="utf-8")
        # After rename, [[Dup]] link should have been updated to [[Dup_renamed]]
        assert "[[Dup_renamed]]" in content
        assert "[[Dup]]" not in content


# ---------------------------------------------------------------------------
# Test 4 — unresolved promotion
# ---------------------------------------------------------------------------

class TestUnresolvedPromotion:
    """Renaming a note resolves previously-unresolvable wikilinks."""

    @pytest.fixture
    def vault(self, tmp_path: Path) -> Path:
        v = tmp_path / "vault"
        v.mkdir()
        # Note with a dangling link [[Target]] — no Target.md exists yet
        _write(v, "Pointer.md", "See [[Target]] for details.\n")
        # A note that will be renamed to Target.md
        _write(v, "Other.md", "# Other content\n")
        return v

    def test_link_unresolved_before_rename(self, vault: Path) -> None:
        b = _make_backend(vault)
        unres_targets = {t for _, t in b._unresolved_links}
        assert "Target" in unres_targets

    def test_link_resolves_after_rename(self, vault: Path) -> None:
        b = _make_backend(vault)
        b.move("Other.md", "Target.md")

        # [[Target]] in Pointer.md should now be a resolved graph edge
        assert "Pointer.md" in b._graph
        successors = set(b._graph.successors("Pointer.md"))
        assert "Target.md" in successors, (
            "Expected Pointer.md → Target.md edge after rename; unresolved not promoted."
        )

    def test_link_removed_from_unresolved_after_rename(self, vault: Path) -> None:
        b = _make_backend(vault)
        b.move("Other.md", "Target.md")

        # The (Pointer.md, Target) pair must no longer appear in _unresolved_links
        unres_pairs = {(s, t) for s, t in b._unresolved_links if s == "Pointer.md"}
        assert ("Pointer.md", "Target") not in unres_pairs

    def test_links_api_shows_resolved_after_rename(self, vault: Path) -> None:
        b = _make_backend(vault)
        b.move("Other.md", "Target.md")

        link_names = {r.name for r in b.links("Pointer.md")}
        assert "Target" in link_names


# ---------------------------------------------------------------------------
# Test 5 — move → undo round-trip
# ---------------------------------------------------------------------------

class TestMoveUndoRoundTrip:
    """move() followed by the reverse move returns referrers to original byte content."""

    @pytest.fixture
    def vault(self, tmp_path: Path) -> Path:
        v = tmp_path / "vault"
        v.mkdir()
        _write(v, "Original.md", "# Original\n\nContent.\n")
        _write(v, "Ref.md", "[[Original]]\n![[Original]]\n[[Original|alias]]\n")
        return v

    def test_round_trip_byte_identical(self, vault: Path) -> None:
        b = _make_backend(vault)
        original_ref_content = (vault / "Ref.md").read_text(encoding="utf-8")

        # Forward: move Original → Renamed
        b.move("Original.md", "Renamed.md")
        after_forward = (vault / "Ref.md").read_text(encoding="utf-8")
        # Sanity: referrer was actually rewritten
        assert "[[Renamed]]" in after_forward

        # Reverse: move Renamed → Original (undo)
        b.move("Renamed.md", "Original.md")
        after_undo = (vault / "Ref.md").read_text(encoding="utf-8")

        assert after_undo == original_ref_content, (
            "Referrer content after undo does not match original.\n"
            f"Expected:\n{original_ref_content!r}\nGot:\n{after_undo!r}"
        )


# ---------------------------------------------------------------------------
# Test 6 — graph_snapshot non-regression after folder move
# ---------------------------------------------------------------------------

class TestGraphSnapshotNonRegression:
    """A pure folder move must not corrupt orphan sets or link counts."""

    @pytest.fixture
    def vault(self, tmp_path: Path) -> Path:
        v = tmp_path / "vault"
        (v / "A").mkdir(parents=True)
        # Hub is linked by Child; Orphan has no links
        _write(v, "Hub.md", "# Hub\n")
        _write(v, "Child.md", "See [[Hub]] for info.\n")
        _write(v, "Orphan.md", "# Orphan\n\nNo links here.\n")
        return v

    def test_snapshot_consistent_after_folder_move(self, vault: Path) -> None:
        b = _make_backend(vault)
        snap_before = b.graph_snapshot()

        # Move Hub into a subfolder (basename unchanged)
        b.move("Hub.md", "A/Hub.md")
        snap_after = b.graph_snapshot()

        # Orphan and Child are unchanged — their orphan membership must be stable
        orphan_names_before = {o.name for o in snap_before.orphans}
        orphan_names_after = {o.name for o in snap_after.orphans}
        assert orphan_names_before == orphan_names_after

        # Child links to Hub so it's not an orphan; Orphan and Child (no in-links) should be orphans
        # After move Hub is still linked by Child, so Child is still not an orphan
        # Hub moved — Child.md still has [[Hub]] which resolves via name to A/Hub.md
        assert "Orphan" in orphan_names_after
        assert "Child" in orphan_names_after  # no note links to Child

        # Total note count unchanged
        assert len(snap_before.link_counts) == len(snap_after.link_counts)

        # Hub's backlink count: Child still links to it
        hub_key_after = next(
            (k for k in snap_after.backlink_counts if k.lower().endswith("hub")), None
        )
        assert hub_key_after is not None
        assert snap_after.backlink_counts[hub_key_after] == 1

        # unresolved set must not grow
        assert len(snap_after.unresolved) <= len(snap_before.unresolved)


# ---------------------------------------------------------------------------
# Test 7 — failure path: the referrer write raises mid-move
# ---------------------------------------------------------------------------

class TestMoveFailurePath:
    """If a referrer write fails mid-move, _needs_reindex must be set so the
    next read API call triggers a clean full rebuild — no permanently torn state.
    """

    @pytest.fixture
    def vault(self, tmp_path: Path) -> Path:
        v = tmp_path / "vault"
        v.mkdir()
        _write(v, "Source.md", "# Source\n\nContent.\n")
        _write(v, "Referrer.md", "See [[Source]] for details.\n")
        return v

    def test_move_raises_on_referrer_write_failure(
        self, vault: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """move() must propagate the exception when a referrer write fails."""
        b = _make_backend(vault)
        referrer_path = vault / "Referrer.md"
        original_write = fs_backend.atomic_write_bytes

        def _failing_write(path, data):
            if path == referrer_path:
                raise OSError("Simulated disk write failure")
            return original_write(path, data)

        monkeypatch.setattr(fs_backend, "atomic_write_bytes", _failing_write)

        with pytest.raises(OSError, match="Simulated disk write failure"):
            b.move("Source.md", "Renamed.md")

    def test_needs_reindex_set_after_referrer_write_failure(
        self, vault: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a mid-move failure, _needs_reindex must be True."""
        b = _make_backend(vault)
        referrer_path = vault / "Referrer.md"
        original_write = fs_backend.atomic_write_bytes

        def _failing_write(path, data):
            if path == referrer_path:
                raise OSError("Simulated disk write failure")
            return original_write(path, data)

        monkeypatch.setattr(fs_backend, "atomic_write_bytes", _failing_write)

        try:
            b.move("Source.md", "Renamed.md")
        except OSError:
            pass

        assert b._needs_reindex is True, (
            "Backend must set _needs_reindex=True after a mid-move failure so "
            "the next _ensure_index() triggers a clean full rebuild."
        )

    def test_index_consistent_after_rebuild_following_failure(
        self, vault: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After the failure-triggered rebuild, the index reflects actual disk state.

        The physical rename (src → dst) already succeeded before the write
        failure, so the NEW path must be in the index and the OLD path must not.
        """
        b = _make_backend(vault)
        referrer_path = vault / "Referrer.md"
        original_write = fs_backend.atomic_write_bytes

        def _failing_write(path, data):
            if path == referrer_path:
                raise OSError("Simulated disk write failure")
            return original_write(path, data)

        monkeypatch.setattr(fs_backend, "atomic_write_bytes", _failing_write)

        try:
            b.move("Source.md", "Renamed.md")
        except OSError:
            pass

        # Remove the monkeypatch so _ensure_index() can run without interference
        monkeypatch.undo()

        # Trigger rebuild via a read API
        files = b.list_files()
        file_paths = [f.path for f in files]

        # The physical rename succeeded: Renamed.md exists on disk, Source.md does not
        assert (vault / "Renamed.md").exists(), "Physical rename must have succeeded"
        assert not (vault / "Source.md").exists(), "Old file must be gone after rename"

        # Index must reflect actual disk state after rebuild
        assert "Renamed.md" in file_paths, "New path must be in index after rebuild"
        assert "Source.md" not in file_paths, "Old path must not be in index after rebuild"


# ---------------------------------------------------------------------------
# Test 8 — move into inbox: _patch_index indexes inbox paths like any other
# ---------------------------------------------------------------------------

class TestMoveIntoInbox:
    """Moving a note into the configured inbox keeps it indexed.

    _rebuild_index walks the inbox like any other folder, so the incremental
    patch path must do the same — the two disagreeing is what strands an index
    entry the next full rebuild would contradict, in either direction.
    """

    @pytest.fixture
    def vault(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        from silica.config import CONFIG
        monkeypatch.setattr(CONFIG, "inbox_dir", "Inbox")
        v = tmp_path / "vault"
        v.mkdir()
        (v / "Inbox").mkdir()
        _write(v, "Note.md", "# Note\n\nContent.\n")
        _write(v, "Ref.md", "See [[Note]] for details.\n")
        return v

    def test_moved_note_stays_indexed(self, vault: Path) -> None:
        b = _make_backend(vault)
        b.move("Note.md", "Inbox/Note.md")
        assert "Inbox/Note.md" in b._notes
        assert "Inbox/Note.md" in b._graph
        # The link in Ref.md follows the note into the inbox instead of dangling.
        assert b._resolve_target("Note").path == "Inbox/Note.md"

    def test_patched_index_matches_full_rebuild(self, vault: Path) -> None:
        b = _make_backend(vault)
        b.move("Note.md", "Inbox/Note.md")
        fresh = _make_backend(vault)
        assert set(b._notes) == set(fresh._notes)
        assert set(b._graph.nodes) == set(fresh._graph.nodes)
        assert b._unresolved_links == fresh._unresolved_links
