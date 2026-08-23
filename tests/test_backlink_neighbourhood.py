# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""BACKLINK phase — the neighbourhood of a title created in THIS process.

The phase exists to wrap mentions of a newly written note inside pre-existing
bodies. Its neighbourhood used to come from `DRIVER.mentions_of(title)`, which
structurally cannot answer for a fresh title: the mention index is keyed by the
titles that were in the trie when each body was last indexed, and `_patch_index`
re-scans only the note just written. So the phase saw an empty neighbourhood and
did nothing, for every title it was built for.
"""
from __future__ import annotations


def _driver():
    """The driver with its graph index already built — the state a real run is
    in by the time WRITE creates a note (validate/collision have all read it).

    It matters: with a never-touched index `create()` takes the full-rebuild
    branch, the new file is already on disk when the trie is built, and the
    mention index comes out complete. Only the incremental branch — the one a
    real run takes — is blind, so a test that skips this is measuring the
    fixture instead of the phase.
    """
    import silica.driver

    silica.driver.DRIVER.list_files()
    return silica.driver.DRIVER


def test_neighbourhood_finds_a_pre_existing_body_mentioning_a_new_title(tmp_vault):
    from silica.router.states.linking import _backlink_neighbourhood

    tmp_vault.note(
        "Attention.md",
        "# Attention\n\nSparse attention is a cheaper variant used in long-context models.\n",
    )
    tmp_vault.note("Unrelated.md", "# Unrelated\n\nNothing to see here.\n")
    drv = _driver()
    drv.create("Sparse attention.md", "# Sparse attention\n\nA cheaper attention variant.\n")

    import os

    hood = _backlink_neighbourhood(["Sparse attention"], {os.path.abspath("Sparse attention.md")})
    assert hood == ["Attention.md"]


def test_neighbourhood_excludes_notes_this_chunk_touched(tmp_vault):
    """A note written by this chunk is not a pre-existing neighbour, even when
    its body mentions another new title (autolink already handled it)."""
    import os

    from silica.router.states.linking import _backlink_neighbourhood

    tmp_vault.note("Attention.md", "# Attention\n\nSparse attention shows up here.\n")
    drv = _driver()
    drv.create("Sparse attention.md", "# Sparse attention\n\nbody\n")

    touched = {os.path.abspath("Attention.md"), os.path.abspath("Sparse attention.md")}
    assert _backlink_neighbourhood(["Sparse attention"], touched) == []


def test_neighbourhood_is_empty_without_new_titles(tmp_vault):
    """No titles, no vault sweep."""
    from silica.router.states.linking import _backlink_neighbourhood

    tmp_vault.note("Attention.md", "# Attention\n\nbody\n")
    assert _backlink_neighbourhood([], set()) == []


def test_neighbourhood_dedupes_across_titles(tmp_vault):
    """One body mentioning two new titles is one neighbour, once."""
    from silica.router.states.linking import _backlink_neighbourhood

    tmp_vault.note(
        "Attention.md",
        "# Attention\n\nSparse attention and linear attention both cut the quadratic cost.\n",
    )
    drv = _driver()
    drv.create("Sparse attention.md", "# Sparse attention\n\nbody\n")
    drv.create("Linear attention.md", "# Linear attention\n\nbody\n")

    hood = _backlink_neighbourhood(["Sparse attention", "Linear attention"], set())
    assert hood.count("Attention.md") == 1


def test_backlink_pass_wraps_the_mention_in_the_pre_existing_note(tmp_vault):
    """End to end over the two real functions: the phase now actually links.

    Before the neighbourhood fix this produced no edit at all, because the
    neighbourhood it iterates was empty.
    """
    from silica.kernel.link.autolink import backlink_pass, build_title_index
    from silica.router.states.linking import _backlink_neighbourhood

    home = tmp_vault.note(
        "Attention.md",
        "# Attention\n\nSparse attention is a cheaper variant used in long-context models.\n",
    )
    drv = _driver()
    drv.create("Sparse attention.md", "# Sparse attention\n\nA cheaper attention variant.\n")

    hood = _backlink_neighbourhood(["Sparse attention"], set())
    added = backlink_pass(
        ["Sparse attention"],
        title_index=build_title_index(list(drv.list_files())),
        neighbourhood=hood,
    )

    assert added == {"Attention.md": ["Sparse attention"]}
    assert "[[Sparse attention]]" in tmp_vault.read(home)


def test_backlink_never_writes_into_the_inbox(tmp_vault):
    """The inbox is staging, never a write target.

    backlink_pass enforces the vault write boundary itself (it bypasses
    validate_operations), but its only guard was `within(path, write_root)` —
    and on a vault with no vault.yaml `active_write_dir()` is "", so `within`
    returns True for everything. The neighbourhood sweep excludes source leaves
    and nothing else, so wikilinks were injected into the very staging text
    CLEANUP later copies verbatim into the source leaf.
    """
    from silica.kernel.link.autolink import backlink_pass, build_title_index

    staging = (
        "# Method\n\nSparse attention is a cheaper variant used in long-context "
        "models, and the section explains why.\n"
    )
    path = tmp_vault.note("Inbox/paper/03-method.md", staging)
    kept = tmp_vault.note(
        "Notes/Attention.md",
        "# Attention\n\nSparse attention is a cheaper variant used in long-context models.\n",
    )
    drv = _driver()
    drv.create("Sparse attention.md", "# Sparse attention\n\nA cheaper attention variant.\n")

    index = build_title_index(list(drv.list_files()))
    added = backlink_pass(
        ["Sparse attention"], title_index=index,
        neighbourhood=["Inbox/paper/03-method.md", "Notes/Attention.md"],
    )

    assert "Inbox/paper/03-method.md" not in added
    assert tmp_vault.read(path) == staging, "the staging chunk was rewritten"
    # the non-inbox neighbour is still linked: the guard is a filter, not an off switch
    assert "[[Sparse attention]]" in tmp_vault.read(kept)
