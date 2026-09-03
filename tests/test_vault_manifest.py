"""vault.yaml manifest (ADR-0014): declared capabilities, not vault types."""
import subprocess

import pytest

from silica.config import CONFIG
from silica.kernel.vault_manifest import (
    apply_manifest_to_config,
    get_active_manifest,
    load_manifest,
    reset_manifest_cache,
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    reset_manifest_cache()
    yield
    reset_manifest_cache()


def test_defaults_prose_only_outside_git(tmp_path):
    m = load_manifest(tmp_path)
    assert m.sources == ("prose",)
    assert m.cooccurrence_lang is None


def test_defaults_include_code_inside_git(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    vault = tmp_path / ".silica"
    vault.mkdir()
    assert load_manifest(vault).sources == ("prose", "code", "notebook")


def test_manifest_file_overrides_defaults(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "sources: [prose]\ncooccurrence_lang: italian\n",
        encoding="utf-8",
    )
    m = load_manifest(tmp_path)
    assert m.sources == ("prose",)
    assert m.cooccurrence_lang == "italian"


def test_malformed_manifest_degrades_to_defaults(tmp_path):
    (tmp_path / "vault.yaml").write_text("sources: 42\n", encoding="utf-8")
    assert load_manifest(tmp_path).sources == ("prose",)


def test_episodic_keys_absent_means_no_schema(tmp_path):
    (tmp_path / "vault.yaml").write_text("conventions:\n  max_tags: 5\n",
                                         encoding="utf-8")
    assert load_manifest(tmp_path).conventions.episodic_keys is None


def test_episodic_keys_empty_block_gets_defaults(tmp_path):
    (tmp_path / "vault.yaml").write_text("conventions:\n  episodic_keys: {}\n",
                                         encoding="utf-8")
    ks = load_manifest(tmp_path).conventions.episodic_keys
    assert ks is not None
    assert ks.prefixes == ("user", "assistant")
    assert ks.default_prefix == "user"
    assert ks.max_depth == 3


def test_episodic_keys_custom_values_parsed(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n"
        "  episodic_keys:\n"
        "    prefixes: [user, assistant, team]\n"
        "    default_prefix: team\n"
        "    max_depth: 2\n",
        encoding="utf-8",
    )
    ks = load_manifest(tmp_path).conventions.episodic_keys
    assert ks.prefixes == ("user", "assistant", "team")
    assert ks.default_prefix == "team"
    assert ks.max_depth == 2


def test_episodic_keys_malformed_block_means_no_schema(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  episodic_keys: [user]\n", encoding="utf-8")
    assert load_manifest(tmp_path).conventions.episodic_keys is None


def test_episodic_keys_bad_fields_fall_back_to_defaults(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n"
        "  episodic_keys:\n"
        "    prefixes: 42\n"
        "    default_prefix: [x]\n"
        "    max_depth: -1\n",
        encoding="utf-8",
    )
    ks = load_manifest(tmp_path).conventions.episodic_keys
    assert ks.prefixes == ("user", "assistant")
    assert ks.default_prefix == "user"
    assert ks.max_depth == 3
    (tmp_path / "vault.yaml").write_text(":\n  - not yaml mapping [", encoding="utf-8")
    assert load_manifest(tmp_path).sources == ("prose",)


def test_get_active_manifest_caches_until_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    assert get_active_manifest().sources == ("prose",)
    (tmp_path / "vault.yaml").write_text("sources: [prose, code]\n", encoding="utf-8")
    assert get_active_manifest().sources == ("prose",)  # cached
    reset_manifest_cache()
    assert get_active_manifest().sources == ("prose", "code")


def test_apply_manifest_defaults_to_auto_without_manifest(tmp_path, monkeypatch):
    """Regression: vault with no vault.yaml at all must land on the config-level
    "auto" default (per-store detection), not the dead "english" fallback."""
    monkeypatch.delenv("SILICA_COOCCURRENCE_LANG", raising=False)
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(CONFIG, "cooccurrence_lang", "english")
    apply_manifest_to_config()
    assert CONFIG.cooccurrence_lang == "auto"


def test_apply_manifest_defaults_to_auto_when_manifest_omits_field(tmp_path, monkeypatch):
    """Regression: vault.yaml present but without `cooccurrence_lang:` must also
    land on "auto", not "english"."""
    (tmp_path / "vault.yaml").write_text("sources: [prose]\n", encoding="utf-8")
    monkeypatch.delenv("SILICA_COOCCURRENCE_LANG", raising=False)
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(CONFIG, "cooccurrence_lang", "english")
    apply_manifest_to_config()
    assert CONFIG.cooccurrence_lang == "auto"


def test_apply_manifest_env_var_wins_over_auto_default(tmp_path, monkeypatch):
    """Env var precedence must survive the bug fix: it still wins even when the
    manifest declares nothing."""
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(CONFIG, "cooccurrence_lang", "french")
    monkeypatch.setenv("SILICA_COOCCURRENCE_LANG", "french")
    apply_manifest_to_config()
    assert CONFIG.cooccurrence_lang == "french"


def test_apply_manifest_env_wins(tmp_path, monkeypatch):
    (tmp_path / "vault.yaml").write_text(
        "cooccurrence_lang: italian\n", encoding="utf-8"
    )
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    monkeypatch.setattr(CONFIG, "cooccurrence_lang", "english")

    monkeypatch.delenv("SILICA_COOCCURRENCE_LANG", raising=False)
    apply_manifest_to_config()
    assert CONFIG.cooccurrence_lang == "italian"

    reset_manifest_cache()
    monkeypatch.setattr(CONFIG, "cooccurrence_lang", "french")
    monkeypatch.setenv("SILICA_COOCCURRENCE_LANG", "french")
    apply_manifest_to_config()
    assert CONFIG.cooccurrence_lang == "french"  # env precedence


def test_apply_manifest_clears_lang_on_switch_to_plain_vault(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "vault.yaml").write_text(
        "cooccurrence_lang: italian\n", encoding="utf-8"
    )
    monkeypatch.delenv("SILICA_COOCCURRENCE_LANG", raising=False)
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path / "a"))
    monkeypatch.setattr(CONFIG, "cooccurrence_lang", "english")
    apply_manifest_to_config()
    assert CONFIG.cooccurrence_lang == "italian"

    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path / "b"))
    reset_manifest_cache()
    apply_manifest_to_config()
    assert CONFIG.cooccurrence_lang == "auto"  # not leaked from vault a; "auto" is the real default


def test_nucleate_gating_sources_prose_only(tmp_path, monkeypatch, capsys):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    vault = tmp_path / ".silica"
    vault.mkdir()
    (vault / "vault.yaml").write_text("sources: [prose]\n", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    from silica.driver import fs_backend
    import silica.driver as driver_mod
    monkeypatch.setattr(driver_mod, "DRIVER", fs_backend.ObsidianFSBackend(str(vault)))
    reset_manifest_cache()

    from silica.cli import _expand_workflow_shortcut

    msg = _expand_workflow_shortcut("/nucleate m.py")
    assert msg == ""  # handled: skipped, nothing for the agent
    assert not (vault / "Inbox" / "m.md").exists()  # code source disabled
    assert "Skipped" in capsys.readouterr().out
