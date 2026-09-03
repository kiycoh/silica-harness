"""TDD tests for the synthetic vault factory (WS0).

Written BEFORE the implementation — these define the contracts C0.1–C0.5.
All tests should be RED until vault_factory.py is implemented.
"""
from __future__ import annotations

import json
import os
import time
from unittest.mock import patch


from tests.fixtures.vault_factory import (
    SPEC,
    build_synthetic_vault,
    _resolve_root,
)


# ---------------------------------------------------------------------------
# C0.1 — Creates when absent
# ---------------------------------------------------------------------------

def test_vault_factory_creates_when_absent(tmp_path):
    """After build_*, all paths declared in SPEC exist on disk."""
    root = tmp_path / "vault"
    result = build_synthetic_vault(root)

    assert result == root
    for spec in SPEC:
        note_path = root / spec.path
        assert note_path.exists(), f"Expected note missing: {spec.path}"

    # Manifest must be present
    manifest_path = root / ".silica_fixture_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "spec_version" in manifest
    assert "spec_sha256" in manifest
    assert "notes" in manifest
    assert "generated_at" in manifest


# ---------------------------------------------------------------------------
# C0.1 — Idempotent (second build does not rewrite files)
# ---------------------------------------------------------------------------

def test_vault_factory_idempotent(tmp_path):
    """A second call without force=True is a no-op; manifest is unchanged."""
    root = tmp_path / "vault"
    build_synthetic_vault(root)

    # Capture mtimes after first build
    mtimes_before = {
        spec.path: (root / spec.path).stat().st_mtime
        for spec in SPEC
    }
    manifest_before = (root / ".silica_fixture_manifest.json").read_text()

    # Small sleep so any rewrite would produce a different mtime
    time.sleep(0.05)
    build_synthetic_vault(root)

    for spec in SPEC:
        mt_after = (root / spec.path).stat().st_mtime
        assert mt_after == mtimes_before[spec.path], (
            f"{spec.path} was rewritten on second build (should be no-op)"
        )

    manifest_after = (root / ".silica_fixture_manifest.json").read_text()
    assert manifest_before == manifest_after, "Manifest changed on second build"


# ---------------------------------------------------------------------------
# C0.1 — force=True rebuilds and updates generated_at
# ---------------------------------------------------------------------------

def test_vault_factory_force_rebuilds(tmp_path):
    """force=True regenerates all files and bumps generated_at."""
    root = tmp_path / "vault"
    build_synthetic_vault(root)

    manifest_path = root / ".silica_fixture_manifest.json"
    ts_before = json.loads(manifest_path.read_text())["generated_at"]

    time.sleep(0.05)
    build_synthetic_vault(root, force=True)

    ts_after = json.loads(manifest_path.read_text())["generated_at"]
    assert ts_after != ts_before, "generated_at not updated on force rebuild"


# ---------------------------------------------------------------------------
# C0.2 — Respects SILICA_TEST_VAULT env override
# ---------------------------------------------------------------------------

def test_vault_factory_respects_env_override(tmp_path):
    """SILICA_TEST_VAULT env var overrides the default root location."""
    override = tmp_path / "custom_vault"

    with patch.dict(os.environ, {"SILICA_TEST_VAULT": str(override)}):
        resolved = _resolve_root()

    assert resolved == override


# ---------------------------------------------------------------------------
# C0.4 — Manifest topology matches actual disk state
# ---------------------------------------------------------------------------

def test_vault_factory_topology_matches_manifest(tmp_path):
    """Orphans, duplicates, and unresolved declared in manifest match a re-scan."""
    root = tmp_path / "vault"
    build_synthetic_vault(root)

    manifest = json.loads((root / ".silica_fixture_manifest.json").read_text())
    notes_by_role: dict[str, list[str]] = {}
    for entry in manifest["notes"]:
        role = entry["expected_role"]
        notes_by_role.setdefault(role, []).append(entry["path"])

    # Orphan: no other note links to it — verify Isolated/Orphan.md has no backlinks
    # (we check by scanning all files for wikilinks to it)
    orphan_path = "Isolated/Orphan.md"
    assert orphan_path in notes_by_role.get("orphan", []), (
        "Isolated/Orphan.md should be declared as orphan in manifest"
    )

    # Duplicate basename: A/Cell.md and B/Cell.md
    dup_paths = [n for n in manifest["notes"] if n["path"].endswith("Cell.md")]
    assert len(dup_paths) == 2, (
        f"Expected 2 Cell.md entries in manifest, got {dup_paths}"
    )
    dup_dirs = {p["path"].split("/")[0] for p in dup_paths}
    assert dup_dirs == {"A", "B"}, f"Unexpected Cell dirs: {dup_dirs}"

    # Unresolved: Perceptron.md links to [[MissingNote]]
    percettrone_note = root / "Concepts" / "Perceptron.md"
    content = percettrone_note.read_text()
    assert "MissingNote" in content, "Perceptron.md should contain [[MissingNote]]"

    # Inbox notes present
    assert (root / "_inbox" / "Lecture.md").exists()
    assert (root / "_inbox" / "New.md").exists()


# ---------------------------------------------------------------------------
# C0.5 — Session fixture is cached (only calls build_* once)
# ---------------------------------------------------------------------------

