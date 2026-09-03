# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`.silicaignore` — per-vault extension of NOISE_DIRS."""

from silica.kernel.recall.paths import NOISE_DIRS, SILICAIGNORE_REL, ignore_matcher
from silica.onboarding.adopt import seed_silicaignore


def test_builtins_apply_without_a_file(tmp_path):
    skip = ignore_matcher(tmp_path)
    assert skip("node_modules")
    assert not skip("notes")


def test_names_globs_and_comments(tmp_path):
    (tmp_path / SILICAIGNORE_REL).write_text(
        "# a comment\n"
        "\n"
        "archive/\n"
        "*.egg-info\n"
        "drafts  # trailing comment\n",
        encoding="utf-8",
    )
    skip = ignore_matcher(tmp_path)
    assert skip("archive")            # trailing slash tolerated
    assert skip("silica.egg-info")    # glob
    assert skip("drafts")             # inline comment stripped
    assert skip("__pycache__")        # built-ins still there
    assert not skip("a comment")
    assert not skip("notes")


def test_seed_writes_builtins_commented_out(tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    seeded = seed_silicaignore(tmp_path)
    assert seeded == tmp_path / SILICAIGNORE_REL

    text = seeded.read_text(encoding="utf-8")
    assert all(f"# {d}\n" in text for d in NOISE_DIRS)
    # Commented out ⇒ the seeded file parses to exactly the built-ins.
    names = ignore_matcher(tmp_path)
    assert names("node_modules") and not names("src")

    # Never overwrites, and prose vaults stay file-free.
    assert seed_silicaignore(tmp_path) is None
    prose = tmp_path / "notes"
    prose.mkdir()
    (prose / "a.md").write_text("hi", encoding="utf-8")
    assert seed_silicaignore(prose) is None
    assert not (prose / SILICAIGNORE_REL).exists()


def test_index_walk_honours_it(tmp_path, monkeypatch):
    from silica.driver.fs_backend import ObsidianFSBackend

    (tmp_path / "keep.md").write_text("keep", encoding="utf-8")
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "old.md").write_text("old", encoding="utf-8")
    (tmp_path / SILICAIGNORE_REL).write_text("archive\n", encoding="utf-8")

    backend = ObsidianFSBackend(str(tmp_path))
    backend._rebuild_index()
    assert set(backend._notes) == {"keep.md"}
