"""Settings panel — the seams that fail silently if they break.

Ponytail: one check per contract from docs/specs/settings-panel.md §11. The
panel's own rendering is not tested here (that is a browser's job); what is
tested is everything that decides WHERE a value lands, whether it is allowed to
land at all, and whether a secret can escape.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from silica.config import CONFIG  # noqa: E402
from silica.onboarding.wizard import merge_env, resolve_env_path  # noqa: E402
from silica.ui.web import settings as st  # noqa: E402


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """Point every write at a throwaway .env — never the developer's own."""
    path = tmp_path / ".env"
    monkeypatch.setattr("silica.onboarding.wizard.resolve_env_path", lambda: path)
    return path


@pytest.fixture
def client(tmp_vault, tmp_path, monkeypatch):
    from silica.ui.web import server

    monkeypatch.setattr(server, "_busy", False)
    return TestClient(server.app), server


def test_merge_env_writes_many_keys_and_keeps_the_rest(env_file):
    existing = "# a comment\nSILICA_MODEL=old\nUNRELATED=keepme\n"
    out = merge_env(existing, {"SILICA_MODEL": "new", "SILICA_PROVIDER": "openrouter"})
    assert "# a comment" in out
    assert "UNRELATED=keepme" in out
    assert "SILICA_MODEL=new" in out
    assert "SILICA_MODEL=old" not in out
    assert "SILICA_PROVIDER=openrouter" in out


def test_env_path_is_the_user_file_even_inside_a_repo(tmp_path, monkeypatch):
    """The file a write lands in is the one that would win at the next boot, and
    config.py layers only the user-level file. A .env in the working directory
    or at the repo root must not attract the write: nothing reads it back."""
    repo = tmp_path / "repo"
    sub = repo / "sub"
    sub.mkdir(parents=True)
    user_env = tmp_path / "home" / ".silica" / ".env"
    monkeypatch.setattr("silica.onboarding.wizard.USER_ENV", user_env)
    monkeypatch.chdir(sub)
    (repo / ".env").write_text("X=1")
    (sub / ".env").write_text("X=2")

    assert resolve_env_path() == user_env


def test_a_write_is_refused_while_a_turn_runs(client, env_file, monkeypatch):
    c, server = client
    monkeypatch.setattr(server, "_busy", True)
    r = c.post("/settings", json={"key": "SILICA_INBOX_DIR", "value": "Nope"})
    assert r.status_code == 409
    assert not env_file.exists()  # refused means nothing was written


def test_an_exported_key_is_locked_and_unwritable(client, env_file, monkeypatch):
    c, _server = client
    monkeypatch.setattr(st, "SHELL_ENV", frozenset({"SILICA_INBOX_DIR"}))
    rows = {
        row["key"]: row
        for section in st.read_sections(probe=False)
        for row in section["rows"]
    }
    assert rows["SILICA_INBOX_DIR"]["locked"] is True
    assert rows["SILICA_INBOX_DIR"]["origin"] == "env"
    assert rows["SILICA_GIT_COMMIT"]["locked"] is False

    r = c.post("/settings", json={"key": "SILICA_INBOX_DIR", "value": "Nope"})
    assert r.status_code == 409
    assert not env_file.exists()


def test_changing_the_provider_writes_its_whole_group_at_once(env_file, monkeypatch):
    """No intermediate state where the provider is new and the model is old:
    both move in one merge_env, so there is no instant to observe them apart."""
    writes = []
    real_write = st.write_env
    monkeypatch.setattr(st, "write_env", lambda u: (writes.append(dict(u)), real_write(u))[1])
    monkeypatch.setattr(CONFIG, "model", "qwen3-32b")
    monkeypatch.setattr(CONFIG, "_provider", "lmstudio")
    # apply() makes the keys live by writing os.environ; declare them here so
    # monkeypatch restores the real environment after the test.
    for key in ("SILICA_MODEL", "SILICA_PROVIDER", "SILICA_PROVIDER_BASE_URL"):
        monkeypatch.setenv(key, "")

    result = st.apply("SILICA_PROVIDER", "openrouter")

    assert result["ok"], result
    assert len(writes) == 1, "the group must be one write, not three"
    assert set(writes[0]) == {"SILICA_MODEL", "SILICA_PROVIDER", "SILICA_PROVIDER_BASE_URL"}
    assert CONFIG.provider == "openrouter"
    assert CONFIG.model.startswith("openrouter/")
    text = env_file.read_text()
    assert "SILICA_PROVIDER=openrouter" in text
    assert "SILICA_MODEL=openrouter/" in text


def test_switch_vault_rebuilds_the_driver_on_the_new_path(tmp_vault, tmp_path):
    """The nine steps as one call: a live process holds a driver, an overlay, a
    manifest and vault-scoped caches, and they all still answer for the old
    folder until they are reset."""
    from silica.cli import switch_vault
    from silica.driver import get_driver

    other = tmp_path / "other-vault"
    other.mkdir()
    get_driver()  # build one against the old vault first

    result = switch_vault(str(other))

    assert result.error is None
    assert result.vault == str(other.resolve())
    assert CONFIG.vault_path == str(other.resolve())
    assert get_driver().vault_path == other.resolve()


def test_switch_vault_refuses_a_path_that_is_not_a_directory(tmp_vault, tmp_path):
    from silica.cli import switch_vault

    file_path = tmp_path / "notes.md"
    file_path.write_text("x")
    before = CONFIG.vault_path

    result = switch_vault(str(file_path))

    assert result.error and "not a directory" in result.error
    assert CONFIG.vault_path == before  # nothing moved


def test_safe_mode_round_trips_through_vault_yaml_not_the_env(tmp_vault, env_file):
    """The one row whose store is the manifest. Off then on re-derives the
    boundary from the vault's content, so there is no remembered value to rot."""
    from pathlib import Path

    from silica.kernel.vault_manifest import reset_manifest_cache
    from silica.ui.web import settings as st

    root = Path(CONFIG.vault_path)
    (root / "vault.yaml").write_text("write_dir: silica\n", encoding="utf-8")
    reset_manifest_cache()
    row = next(r for rows in st.sections().values() for r in rows
               if r.key == st.SAFE_MODE_KEY)
    assert st._value(row) == "true"

    assert st.apply(st.SAFE_MODE_KEY, "false")["ok"] is True
    assert st._value(row) == "false"
    assert 'write_dir: ""' in (root / "vault.yaml").read_text(encoding="utf-8")

    result = st.apply(st.SAFE_MODE_KEY, "true")
    assert result["values"] == {"safe_mode": "true", "write_dir": "silica"}
    assert st._value(row) == "true"
    assert not env_file.exists()  # never an env var


def test_the_bug_payload_carries_no_api_key(tmp_vault, monkeypatch):
    """The report is built server-side precisely so that nothing which can see a
    key ever builds it — a public issue is the wrong place for one."""
    secret = "sk-or-v1-THIS-MUST-NEVER-APPEAR"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setattr(CONFIG, "provider_api_key", secret)
    monkeypatch.setattr(CONFIG, "tavily_api_key", secret)

    payload = st.bug_report()["payload"]

    assert secret not in payload
    assert "silica" in payload  # it still says something useful


def test_vault_switch_is_live_only_and_never_persists(client, env_file, tmp_path):
    """A SILICA_VAULT line in the .env is ignored at boot (config.load_user_env),
    so persisting the switch would write a value nothing reads back and the
    panel would show a "file" origin for a vault the next launch will not use."""
    _, server = client
    other = tmp_path / "other-vault"
    other.mkdir()
    out = server._apply_vault_switch(str(other))
    assert out["ok"] and out["vault"] == str(other.resolve())
    assert CONFIG.vault_path == str(other.resolve())
    assert not env_file.exists() or "SILICA_VAULT" not in env_file.read_text()


def test_vault_origin_is_never_the_file():
    assert st._origin(st.VAULT_KEY, {st.VAULT_KEY}) == "derived"
