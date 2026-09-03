import io
import subprocess
from pathlib import Path

import pytest

from silica.cli import _activate_repo_mode, default_user_vault, resolve_cwd_vault
from silica.config import CONFIG


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def _obsidian(path: Path) -> None:
    (path / ".obsidian").mkdir(parents=True, exist_ok=True)


@pytest.fixture
def vault_env(monkeypatch):
    """Restore CONFIG.vault_path, and keep the exported-pin off by default."""
    monkeypatch.setattr("silica.config.VAULT_PINNED", False)
    orig = CONFIG.vault_path
    yield
    CONFIG.vault_path = orig


def test_cwd_beats_a_configured_vault(tmp_path, monkeypatch, vault_env):
    # The headline rule: launching silica in a folder curates THAT folder, even
    # when a .env still carries a SILICA_VAULT pointing somewhere else.
    monkeypatch.chdir(tmp_path)
    CONFIG.vault_path = str(tmp_path / "elsewhere")
    _activate_repo_mode()
    assert Path(CONFIG.vault_path).resolve() == tmp_path.resolve()


def test_exported_vault_pin_beats_cwd(tmp_path, monkeypatch, vault_env):
    # `SILICA_VAULT=x silica`, an MCP server env block or a cron unit: a
    # deliberate per-invocation pin still outranks whatever cwd happens to be.
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("silica.config.VAULT_PINNED", True)
    CONFIG.vault_path = str(pinned)
    _activate_repo_mode()
    assert Path(CONFIG.vault_path).resolve() == pinned.resolve()


def test_cwd_code_repo_adopts_root_and_declares_write_dir(tmp_path, monkeypatch, vault_env):
    # A source tree: the vault is the repo itself (reads see the whole tree), and
    # the write boundary is declared rather than guessed.
    _init_repo(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    monkeypatch.chdir(tmp_path)
    CONFIG.vault_path = ""
    _activate_repo_mode()
    assert Path(CONFIG.vault_path).resolve() == tmp_path.resolve()
    assert (tmp_path / "vault.yaml").read_text(encoding="utf-8") == "write_dir: docs/silica\n"


def test_cwd_prose_folder_is_adopted_whole_and_written_in_place(tmp_path, monkeypatch, vault_env):
    # A folder of notes is adopted whole and left alone: safe mode is off by
    # default, so nothing is declared and no folder is invented.
    (tmp_path / "nota.md").write_text("# nota")
    monkeypatch.chdir(tmp_path)
    CONFIG.vault_path = ""
    _activate_repo_mode()
    assert Path(CONFIG.vault_path).resolve() == tmp_path.resolve()
    assert not (tmp_path / "vault.yaml").exists()
    assert not (tmp_path / "silica").exists()
    assert not (tmp_path / "docs").exists()


def test_configured_vault_file_is_refused(tmp_path, monkeypatch, vault_env):
    # Pointing SILICA_VAULT at a file used to crash with NotADirectoryError while
    # trying to mkdir <file>/docs/silica. Reached only via the fallback, so cwd
    # must be a non-vault place ($HOME here).
    target = tmp_path / "note.md"
    target.write_text("# not a vault")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(tmp_path)
    CONFIG.vault_path = str(target)
    _activate_repo_mode()
    assert CONFIG.vault_path == str(target)  # left alone, not mangled


def test_cwd_obsidian_vault_verbatim(tmp_path, monkeypatch, vault_env):
    # An Obsidian vault (even git-tracked) is adopted exactly — no docs/silica.
    _init_repo(tmp_path)
    _obsidian(tmp_path)
    monkeypatch.chdir(tmp_path)
    CONFIG.vault_path = ""
    _activate_repo_mode()
    assert Path(CONFIG.vault_path).resolve() == tmp_path.resolve()
    assert not (tmp_path / "docs" / "silica").exists()


def test_cwd_subdirectory_resolves_to_the_repo_root(tmp_path):
    _init_repo(tmp_path)
    sub = tmp_path / "src" / "deep"
    sub.mkdir(parents=True)
    assert Path(resolve_cwd_vault(sub)).resolve() == tmp_path.resolve()


def test_plain_folder_outside_any_repo_is_itself_the_vault(tmp_path):
    assert Path(resolve_cwd_vault(tmp_path)).resolve() == tmp_path.resolve()


def test_home_itself_is_never_the_vault(tmp_path):
    # Curating everything you own is not a vault decision — fall back instead.
    assert resolve_cwd_vault(tmp_path, home=tmp_path) is None


def test_filesystem_root_is_never_the_vault(tmp_path):
    # No shell starts there, but a GUI client can spawn a stdio MCP server with
    # cwd `/`, and indexing the whole disk is never what that meant.
    assert resolve_cwd_vault(Path(Path.cwd().anchor), home=tmp_path) is None


def test_a_pre_write_dir_layout_migrates_on_launch(tmp_path, monkeypatch, vault_env):
    # The vault Silica opens is the repo you launched in, whatever it finds
    # underneath. A vault from before the write boundary keeps its notes and
    # gets the whole repo back as read scope, in one launch and with no prompt.
    _init_repo(tmp_path)
    (tmp_path / "docs" / "silica").mkdir(parents=True)
    (tmp_path / "docs" / "silica" / "nota.md").write_text("# nota", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    CONFIG.vault_path = ""
    _activate_repo_mode()
    assert Path(CONFIG.vault_path).resolve() == tmp_path.resolve()
    assert (tmp_path / "vault.yaml").read_text(encoding="utf-8") == "write_dir: docs/silica\n"


def test_declared_manifest_root_wins_over_its_write_dir(tmp_path, monkeypatch, vault_env):
    _init_repo(tmp_path)
    (tmp_path / "vault.yaml").write_text("write_dir: docs/silica\n", encoding="utf-8")
    (tmp_path / "docs" / "silica").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    CONFIG.vault_path = ""
    _activate_repo_mode()
    assert Path(CONFIG.vault_path).resolve() == tmp_path.resolve()


def test_adoption_never_prompts(tmp_path, monkeypatch, vault_env):
    # `silica mcp` runs with stdin bound to the MCP client: a prompt there would
    # eat the first JSON-RPC message and then die on EOF. cwd decides silently.
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO())
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("prompted for adoption"))
    CONFIG.vault_path = ""
    _activate_repo_mode()
    assert Path(CONFIG.vault_path).resolve() == tmp_path.resolve()


def test_home_vault_when_nothing_else_applies(tmp_path, monkeypatch, vault_env):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(tmp_path)
    CONFIG.vault_path = ""
    _activate_repo_mode()
    assert Path(CONFIG.vault_path).resolve() == (tmp_path / ".silica" / "vault").resolve()


def test_default_user_vault_under_home(tmp_path):
    assert default_user_vault(home=tmp_path) == tmp_path / ".silica" / "vault"


def test_system_directories_are_never_the_vault(tmp_path):
    # A root process with cwd /tmp adopted it and wrote /tmp/vault.yaml
    # (2026-09-02), which the session hook then found above every pytest tmp
    # dir. A shell can start in /tmp or /usr; a vault cannot be one of them.
    # Exact match only: /tmp/<x> is somebody's folder (the test above).
    import tempfile
    for d in ("/tmp", "/usr", "/etc", "/var", "/home", tempfile.gettempdir()):
        assert resolve_cwd_vault(d, home=tmp_path) is None, d
