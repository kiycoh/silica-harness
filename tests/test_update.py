# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`silica update` against throwaway git repos.

Each test builds a bare remote + an "install" clone tracking origin/main, then
monkeypatches ``silica.update.ROOT`` at the install so update() operates on it.
The rollback test is the one that matters: a pulled syntax error must never
survive on disk.
"""

from __future__ import annotations

import subprocess

import silica.update as upd


def _run(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _make_install(tmp_path):
    """A clone whose origin/main tracks a shared bare remote."""
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(remote)],
        check=True, capture_output=True,
    )
    seed = tmp_path / "seed"
    seed.mkdir()
    _run(seed, "init", "-b", "main")
    _run(seed, "config", "user.email", "t@t")
    _run(seed, "config", "user.name", "t")
    _run(seed, "remote", "add", "origin", str(remote))
    (seed / "silica").mkdir()
    (seed / "silica" / "__init__.py").write_text("x = 1\n")
    _run(seed, "add", "-A")
    _run(seed, "commit", "-m", "init")
    _run(seed, "push", "-u", "origin", "main")

    install = tmp_path / "install"
    _run(tmp_path, "clone", str(remote), str(install))
    _run(install, "config", "user.email", "t@t")
    _run(install, "config", "user.name", "t")
    return install


def _push(tmp_path, name, rel, content):
    """Land a new commit on the shared remote via a throwaway clone."""
    clone = tmp_path / name
    _run(tmp_path, "clone", str(tmp_path / "remote.git"), str(clone))
    _run(clone, "config", "user.email", "t@t")
    _run(clone, "config", "user.name", "t")
    p = clone / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    _run(clone, "add", "-A")
    _run(clone, "commit", "-m", "change")
    _run(clone, "push", "origin", "main")


def _head(cwd):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True
    ).stdout.strip()


def test_already_up_to_date(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(upd, "ROOT", _make_install(tmp_path))
    assert upd.update() == 0
    assert "up to date" in capsys.readouterr().out.lower()


def test_dirty_tree_aborts(tmp_path, monkeypatch, capsys):
    install = _make_install(tmp_path)
    _push(tmp_path, "other", "silica/feature.py", "y = 2\n")  # an update exists
    (install / "silica" / "__init__.py").write_text("x = 999\n")  # local edit
    monkeypatch.setattr(upd, "ROOT", install)
    assert upd.update() == 1
    assert "uncommitted" in capsys.readouterr().out.lower()


def test_check_ignores_dirty_tree(tmp_path, monkeypatch, capsys):
    install = _make_install(tmp_path)
    _push(tmp_path, "other", "silica/feature.py", "y = 2\n")
    (install / "silica" / "__init__.py").write_text("x = 999\n")
    monkeypatch.setattr(upd, "ROOT", install)
    assert upd.update(check_only=True) == 0  # pure query, dirty tree irrelevant
    out = capsys.readouterr().out.lower()
    assert "available" in out and "uncommitted" not in out


def test_pulls_and_updates(tmp_path, monkeypatch, capsys):
    install = _make_install(tmp_path)
    _push(tmp_path, "other", "silica/feature.py", "y = 2\n")
    monkeypatch.setattr(upd, "ROOT", install)
    assert upd.update() == 0
    assert (install / "silica" / "feature.py").exists()
    assert "updated" in capsys.readouterr().out.lower()


def test_rolls_back_on_syntax_error(tmp_path, monkeypatch, capsys):
    install = _make_install(tmp_path)
    old = _head(install)
    _push(tmp_path, "other", "silica/broken.py", "def (\n")  # not valid Python
    monkeypatch.setattr(upd, "ROOT", install)
    assert upd.update() == 1
    assert _head(install) == old  # rolled back to pre-pull commit
    assert "syntax error" in capsys.readouterr().out.lower()


# --- wheel installs (no .git): PyPI is the upstream --------------------------

def _wheel_root(tmp_path, monkeypatch, *parts):
    """A ROOT with no .git, nested under path segments naming the installer."""
    root = tmp_path.joinpath(*parts, "site-packages")
    root.mkdir(parents=True)
    monkeypatch.setattr(upd, "ROOT", root)
    return root


def test_wheel_uv_tool_names_uv_upgrade(tmp_path, monkeypatch, capsys):
    import silica
    _wheel_root(tmp_path, monkeypatch, "uv", "tools", "silica-harness")
    monkeypatch.setattr(silica, "__version__", "1.0.0")
    monkeypatch.setattr(upd, "_pypi_latest", lambda: "999.0.0")
    assert upd.update() == 1
    out = capsys.readouterr().out
    assert "999.0.0" in out
    assert "uv tool upgrade silica-harness" in out


def test_wheel_pipx_names_pipx_upgrade(tmp_path, monkeypatch, capsys):
    import silica
    _wheel_root(tmp_path, monkeypatch, "pipx", "venvs", "silica-harness")
    monkeypatch.setattr(silica, "__version__", "1.0.0")
    monkeypatch.setattr(upd, "_pypi_latest", lambda: "999.0.0")
    assert upd.update() == 1
    assert "pipx upgrade silica-harness" in capsys.readouterr().out


def test_wheel_plain_pip_names_pip_upgrade(tmp_path, monkeypatch, capsys):
    import silica
    _wheel_root(tmp_path, monkeypatch, "venv", "lib")
    monkeypatch.setattr(silica, "__version__", "1.0.0")
    monkeypatch.setattr(upd, "_pypi_latest", lambda: "999.0.0")
    assert upd.update() == 1
    assert "pip install -U silica-harness" in capsys.readouterr().out


def test_wheel_up_to_date(tmp_path, monkeypatch, capsys):
    import silica
    _wheel_root(tmp_path, monkeypatch, "uv", "tools", "silica-harness")
    monkeypatch.setattr(silica, "__version__", "1.0.0")
    monkeypatch.setattr(upd, "_pypi_latest", lambda: "1.0.0")
    assert upd.update() == 0
    assert "up to date" in capsys.readouterr().out.lower()


def test_wheel_check_only_reports_without_failing(tmp_path, monkeypatch, capsys):
    import silica
    _wheel_root(tmp_path, monkeypatch, "uv", "tools", "silica-harness")
    monkeypatch.setattr(silica, "__version__", "1.0.0")
    monkeypatch.setattr(upd, "_pypi_latest", lambda: "999.0.0")
    assert upd.update(check_only=True) == 0
    assert "999.0.0" in capsys.readouterr().out


def test_wheel_pypi_unreachable(tmp_path, monkeypatch, capsys):
    def _boom():
        raise OSError("no network")
    _wheel_root(tmp_path, monkeypatch, "uv", "tools", "silica-harness")
    monkeypatch.setattr(upd, "_pypi_latest", _boom)
    assert upd.update() == 1
    assert "pypi" in capsys.readouterr().out.lower()


def test_pypi_version_compare_is_numeric():
    assert upd._newer("0.9.10", "0.9.3")
    assert not upd._newer("0.9.3", "0.9.10")
    assert not upd._newer("garbage", "0.1.0")
    assert not upd._newer("0.9.4", "0.9.4.dev3+g123abc")  # dev tree at the tag


def test_behind_count_wheel_uses_cache(tmp_path, monkeypatch):
    import silica
    _wheel_root(tmp_path, monkeypatch, "uv", "tools", "silica-harness")
    monkeypatch.setattr(silica, "__version__", "1.0.0")
    cache = tmp_path / "pypi-latest"
    monkeypatch.setattr(upd, "CACHE", cache)
    cache.write_text("2.0.0")  # fresh mtime: no background refresh fires
    assert upd.behind_count() == 1
    cache.write_text("1.0.0")
    assert upd.behind_count() == 0


def test_behind_count_wheel_no_cache_is_quiet(tmp_path, monkeypatch):
    _wheel_root(tmp_path, monkeypatch, "uv", "tools", "silica-harness")
    monkeypatch.setattr(upd, "CACHE", tmp_path / "absent")
    monkeypatch.setattr(upd, "_refresh_pypi_cache", lambda: None)  # no network in tests
    assert upd.behind_count() == 0


def test_refresh_pypi_cache_writes_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(upd, "CACHE", tmp_path / "cache" / "pypi-latest")
    monkeypatch.setattr(upd, "_pypi_latest", lambda: "3.1.4")
    upd._refresh_pypi_cache()
    assert upd.CACHE.read_text() == "3.1.4"
    assert not upd.CACHE.with_suffix(".tmp").exists()
