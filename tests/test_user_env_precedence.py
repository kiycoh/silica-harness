# SPDX-License-Identifier: AGPL-3.0-or-later

"""~/.silica/.env outranks any .env a third-party loader injected.

Measured 2026-09-02: litellm/__init__.py calls load_dotenv() at import and
find_dotenv walks up from the venv to the repo checkout, so `<repo>/.env`
(a retired model pin) landed in os.environ BEFORE silica.config loaded
~/.silica/.env with override=False, and the REPL ran on a 404 model while
`python -c "from silica.config import CONFIG"` said the right one. A key the
shell exported is a pin and still wins; a key that appeared behind silica's
back between package import and config import does not.
"""
from __future__ import annotations

import os

import silica
from silica.config import load_user_env


def test_foreign_loader_value_loses_to_the_user_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("SILICA_MODEL=from-user-env\n", encoding="utf-8")
    monkeypatch.setattr(silica, "SHELL_ENV", frozenset())  # not exported by the shell
    monkeypatch.setenv("SILICA_MODEL", "injected-by-litellm")
    load_user_env(env)
    assert os.environ["SILICA_MODEL"] == "from-user-env"


def test_shell_export_still_outranks_the_user_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("SILICA_MODEL=from-user-env\n", encoding="utf-8")
    monkeypatch.setattr(silica, "SHELL_ENV", frozenset({"SILICA_MODEL"}))
    monkeypatch.setenv("SILICA_MODEL", "exported-pin")
    load_user_env(env)
    assert os.environ["SILICA_MODEL"] == "exported-pin"


def test_missing_user_env_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICA_MODEL", "whatever")
    load_user_env(tmp_path / "absent.env")
    assert os.environ["SILICA_MODEL"] == "whatever"


def test_vault_in_the_user_env_is_ignored_with_a_warning(tmp_path, monkeypatch, caplog):
    """The vault is a per-launch fact (silica curates the folder it starts
    in); a SILICA_VAULT line in ~/.silica/.env served every entry point that
    never ran the CLI bootstrap (benches, probes) a vault the user was not in.
    Only an export pins."""
    import logging
    env = tmp_path / ".env"
    env.write_text("SILICA_VAULT=/from/user/env\nSILICA_MODEL=m\n", encoding="utf-8")
    monkeypatch.setattr(silica, "SHELL_ENV", frozenset())
    monkeypatch.delenv("SILICA_VAULT", raising=False)
    with caplog.at_level(logging.WARNING, logger="silica.config"):
        load_user_env(env)
    assert "SILICA_VAULT" not in os.environ
    assert os.environ["SILICA_MODEL"] == "m"
    assert any("SILICA_VAULT" in r.getMessage() and "export" in r.getMessage()
               for r in caplog.records)


def test_exported_vault_is_untouched_by_the_user_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("SILICA_VAULT=/from/user/env\n", encoding="utf-8")
    monkeypatch.setattr(silica, "SHELL_ENV", frozenset({"SILICA_VAULT"}))
    monkeypatch.setenv("SILICA_VAULT", "/exported")
    load_user_env(env)
    assert os.environ["SILICA_VAULT"] == "/exported"
