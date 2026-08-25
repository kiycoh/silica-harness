# SPDX-License-Identifier: AGPL-3.0-or-later

"""Relative markdown link targets resolve against the source note.

The single link generator already emits markdown-link targets (link/ast.py
injects a synthetic wikilink per non-web href), but `_resolve_target` never
normalized `./` and `../`, so every such link died as a ghost node: measured
2026-08-25, one INDEX.md carried 332 dead `./`-prefixed targets, and the
distance signal under-reported on the whole docs/ tree. Resolution is the one
place this may live — a second parser anywhere would break ADR-0029's "one
generator".
"""
from __future__ import annotations

from silica.driver.fs_backend import ObsidianFSBackend
from silica.kernel.link.ast import resolve_relative


# ---------------------------------------------------------------------------
# the pure helper (shared by both backends)
# ---------------------------------------------------------------------------

def test_resolve_relative_joins_and_normalizes():
    assert resolve_relative("./sub/b.md", "docs/a.md") == "docs/sub/b.md"
    assert resolve_relative("../top.md", "docs/sub/b.md") == "docs/top.md"
    assert resolve_relative("a/./b.md", "docs/x.md") == "docs/a/b.md"


def test_resolve_relative_leaves_absolute_style_targets_alone():
    # Vault-rooted and bare-name targets already resolve today; touching them
    # would change measured behavior for no defect.
    assert resolve_relative("docs/adr/0029.md", "docs/a.md") == "docs/adr/0029.md"
    assert resolve_relative("0029-foo.md", "docs/adr/0030.md") == "0029-foo.md"


def test_resolve_relative_refuses_targets_escaping_the_vault():
    # A target climbing above the vault root cannot name a note; unresolved is
    # the honest answer, never a stripped guess.
    assert resolve_relative("../../outside.md", "docs/a.md") is None
    assert resolve_relative("../escape.md", "top.md") is None


# ---------------------------------------------------------------------------
# fs backend integration: ghosts become edges
# ---------------------------------------------------------------------------

def _vault(tmp_path):
    (tmp_path / "docs" / "sub").mkdir(parents=True)
    (tmp_path / "top.md").write_text("# top", encoding="utf-8")
    (tmp_path / "docs" / "sub" / "b.md").write_text("# b", encoding="utf-8")
    (tmp_path / "docs" / "a.md").write_text(
        "see [b](./sub/b.md) and [top](../top.md) and [gone](../../outside.md)",
        encoding="utf-8",
    )
    backend = ObsidianFSBackend(vault_path=str(tmp_path))
    backend._rebuild_index()
    return backend


def test_relative_markdown_targets_resolve(tmp_path):
    backend = _vault(tmp_path)
    b = backend._resolve_target("./sub/b.md", "docs/a.md")
    top = backend._resolve_target("../top.md", "docs/a.md")
    assert b is not None and b.path == "docs/sub/b.md"
    assert top is not None and top.path == "top.md"
    # The index built from the same resolver: the two links are edges, not ghosts.
    unresolved_targets = {t for (_s, t) in backend._unresolved_links}
    assert "./sub/b.md" not in unresolved_targets
    assert "../top.md" not in unresolved_targets


def test_escaping_target_stays_unresolved(tmp_path):
    backend = _vault(tmp_path)
    assert backend._resolve_target("../../outside.md", "docs/a.md") is None
    assert ("docs/a.md", "../../outside.md") in backend._unresolved_links
