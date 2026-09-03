"""Tests for silica.onboarding.wizard — .env merge and interactive flow."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _fresh_setup_env(monkeypatch):
    # Importing silica.config preloads ~/.silica/.env (SILICA_MODEL included)
    # into os.environ. These tests simulate fresh setups, where no chat model is
    # configured yet; the keep-current shortcut has its own test.
    monkeypatch.delenv("SILICA_MODEL", raising=False)


class TestMergeEnv:
    def test_updates_managed_key_in_place(self):
        from silica.onboarding.wizard import merge_env
        out = merge_env("SILICA_MODEL=old\nFOO=bar\n", {"SILICA_MODEL": "new"})
        assert "SILICA_MODEL=new" in out
        assert "SILICA_MODEL=old" not in out

    def test_preserves_unknown_lines_and_comments(self):
        from silica.onboarding.wizard import merge_env
        existing = "# my comment\nFOO=bar\n\nexport OTHER=keep\n"
        out = merge_env(existing, {"SILICA_MODEL": "m"})
        assert "# my comment" in out
        assert "FOO=bar" in out
        assert "export OTHER=keep" in out

    def test_appends_new_keys(self):
        from silica.onboarding.wizard import merge_env
        out = merge_env("FOO=bar\n", {"SILICA_BACKEND": "fs"})
        assert out.splitlines()[-1] == "SILICA_BACKEND=fs"

    def test_empty_existing(self):
        from silica.onboarding.wizard import merge_env
        out = merge_env("", {"SILICA_MODEL": "m"})
        assert out == "SILICA_MODEL=m\n"

    def test_matches_export_prefix(self):
        from silica.onboarding.wizard import merge_env
        out = merge_env("export SILICA_MODEL=old\n", {"SILICA_MODEL": "new"})
        assert "SILICA_MODEL=new" in out
        assert "old" not in out

    def test_uncomments_and_fills_commented_key(self):
        from silica.onboarding.wizard import merge_env
        out = merge_env("# SILICA_MODEL=example\n", {"SILICA_MODEL": "m"})
        assert "SILICA_MODEL=m" in out
        assert "# SILICA_MODEL" not in out

    def test_leaves_unrelated_commented_lines_verbatim(self):
        from silica.onboarding.wizard import merge_env
        out = merge_env("# SILICA_MODEL=example\n# SILICA_BACKEND=fs\n", {"SILICA_MODEL": "m"})
        assert "SILICA_MODEL=m" in out
        assert "# SILICA_BACKEND=fs" in out  # unrelated key stays commented

    def test_appends_key_absent_from_commented_template(self):
        from silica.onboarding.wizard import merge_env
        out = merge_env("# SILICA_MODEL=example\n", {"GROQ_API_KEY": "gsk"})
        assert "# SILICA_MODEL=example" in out  # untouched
        assert out.splitlines()[-1] == "GROQ_API_KEY=gsk"

    def test_duplicate_commented_key_sets_only_first(self):
        from silica.onboarding.wizard import merge_env
        out = merge_env("# SILICA_MODEL=a\n# SILICA_MODEL=b\n", {"SILICA_MODEL": "m"})
        lines = out.splitlines()
        assert lines[0] == "SILICA_MODEL=m"    # first uncommented + set
        assert lines[1] == "# SILICA_MODEL=b"  # later duplicate stays a comment


class TestFindEnvExample:
    def test_finds_packaged_copy_without_repo_root(self):
        # The seeding fix: a real pip/uv-tool install has no repo root, so the
        # wizard must locate silica/.env.example shipped inside the package.
        # Guards both the symlink and the parents[1] candidate.
        from silica.onboarding.wizard import _find_env_example
        found = _find_env_example(None)
        assert found is not None and found.is_file(), found
        assert found.name == ".env.example"
        assert found.read_text(encoding="utf-8").lstrip().startswith("#")

    def test_repo_root_copy_wins_when_present(self, tmp_path):
        from silica.onboarding.wizard import _find_env_example
        (tmp_path / ".env.example").write_text("# repo-root seed\n", encoding="utf-8")
        assert _find_env_example(tmp_path) == tmp_path / ".env.example"


class TestRunWizard:
    @pytest.fixture(autouse=True)
    def _no_live_endpoint_probe(self, monkeypatch):
        # Never touch a real LM Studio / local endpoint from the suite. Tests that
        # exercise autodetect re-monkeypatch this to a fixed list (later wins).
        # Rerank-extra detection is pinned False so the reranker question is
        # deterministic regardless of what this machine has installed.
        import silica.onboarding.wizard as wizard
        monkeypatch.setattr(wizard, "endpoint_model_ids", lambda url: [])
        monkeypatch.setattr(wizard, "_rerank_extra_present", lambda: False)

    def _scripted(self, answers):
        it = iter(answers)
        return lambda prompt: next(it)

    def test_full_lmstudio_flow_writes_env(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"
        env_path.write_text("# keep me\nFOO=bar\n")

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        # os.environ.update inside the wizard must not leak into the test env
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # setup mode → essential
            str(vault),    # vault path (no repo detected)
            "",            # force language? → Enter, follow source
            "",            # provider → default lmstudio
            "test-model",  # model id
            "n",           # high-value gate → skip embeddings/reranker
            "",            # write .env → default y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        content = env_path.read_text()
        assert "# keep me" in content
        assert "FOO=bar" in content
        assert f"SILICA_VAULT={vault}" in content
        assert "SILICA_PROVIDER=lmstudio" in content
        assert "SILICA_MODEL=test-model" in content
        assert "OPENROUTER_API_KEY" not in content

    def test_vault_path_defaults_to_the_launch_folder(self, monkeypatch, tmp_path):
        """Outside a git repo the vault question had no default at all, so the
        promised "Enter accepts the shown default" looped forever on the first
        step. The folder you launched `silica init` in is the default vault,
        same rule as `silica` itself."""
        import silica.onboarding.wizard as wizard

        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "a.md").write_text("# a")
        env_path = tmp_path / ".env"

        monkeypatch.chdir(notes)
        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # setup mode → essential
            "",            # vault path → Enter accepts the launch folder
            "",            # force language? → Enter, follow source
            "",            # provider → default lmstudio
            "test-model",  # model id
            "n",           # high-value gate → skip embeddings/reranker
            "",            # write .env → default y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert f"SILICA_VAULT={notes}" in env_path.read_text()

    def test_configured_model_kept_with_one_enter(self, monkeypatch, tmp_path):
        # A model already configured (the global ~/.silica/.env) survives a
        # per-vault re-run: Enter keeps it, no provider/model/key re-asking.
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"
        env_path.write_text("SILICA_MODEL=openrouter/foo/bar\nSILICA_PROVIDER=openrouter\n")

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        env = dict(os.environ)
        env["SILICA_MODEL"] = "openrouter/foo/bar"
        env["SILICA_PROVIDER"] = "openrouter"
        monkeypatch.setattr(wizard.os, "environ", env)

        answers = [
            "",          # setup mode → essential
            str(vault),  # vault path
            "",          # force language? → Enter
            "",          # keep configured chat model? → default y
            "n",         # high-value gate → skip
            "",          # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        content = env_path.read_text()
        assert "SILICA_MODEL=openrouter/foo/bar" in content
        assert "SILICA_PROVIDER=openrouter" in content

    def test_custom_provider_flow_writes_base_url_and_key(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",                             # setup mode → essential
            str(vault),                     # vault path
            "",                             # force language? → Enter
            "custom",                       # provider
            "http://localhost:8000/v1",     # base URL
            "",                             # API key → default dummy-key
            "qwen3",                        # model id
            "n",                            # high-value gate → skip
            "",                             # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        content = env_path.read_text()
        assert "SILICA_PROVIDER=custom" in content
        assert "SILICA_PROVIDER_BASE_URL=http://localhost:8000/v1" in content
        assert "SILICA_PROVIDER_API_KEY=dummy-key" in content
        assert "SILICA_MODEL=qwen3" in content

    def test_ollama_flow_writes_provider_and_model(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))
        # Don't probe a live Ollama; offer a pick-list deterministically.
        monkeypatch.setattr(wizard, "_ollama_installed_models", lambda: ["llama3.2:3b"])

        answers = [
            "",              # setup mode → essential
            str(vault),      # vault path
            "",              # force language? → Enter
            "ollama",        # provider
            "llama3.2:3b",   # model id
            "n",             # high-value gate → skip
            "",              # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        content = env_path.read_text()
        assert "SILICA_PROVIDER=ollama" in content
        assert "SILICA_MODEL=llama3.2:3b" in content

    def test_ollama_flow_accepts_installed_default_on_enter(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))
        monkeypatch.setattr(wizard, "_ollama_installed_models", lambda: ["mistral:7b"])

        answers = [
            "",          # setup mode → essential
            str(vault),  # vault path
            "",          # force language? → Enter
            "ollama",    # provider
            "",          # model id → Enter accepts first installed default
            "n",         # high-value gate → skip
            "",          # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert "SILICA_MODEL=mistral:7b" in env_path.read_text()

    def test_hosted_groq_flow_writes_key_env(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))
        # No network in tests: an unreachable /models falls back to the
        # curated list, which is the path this flow asserts.
        monkeypatch.setattr(wizard, "endpoint_model_ids", lambda base, api_key="": [])

        answers = [
            "",             # setup mode → essential
            str(vault),     # vault path
            "",             # force language? → Enter
            "groq",         # provider
            "gsk_test",     # Groq API key (asked first: it feeds /models)
            "",             # model → default groq/llama-3.3-70b-versatile
            "n",            # high-value gate → skip
            "",             # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        content = env_path.read_text()
        assert "SILICA_PROVIDER=groq" in content
        assert "SILICA_MODEL=groq/llama-3.3-70b-versatile" in content
        assert "GROQ_API_KEY=gsk_test" in content

    def test_declining_write_aborts(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)

        answers = ["", str(vault), "", "", "test-model", "n", "n"]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 1
        assert not env_path.exists()

    def test_repo_mode_skips_vault_key(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        (tmp_path / "docs" / "silica").mkdir(parents=True)
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # setup mode → essential
            "",            # repo mode? → default y
            "",            # force language? → Enter, follow source
            "",            # provider → lmstudio
            "test-model",  # model
            "n",           # high-value gate → skip
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        # Repo mode must not *set* SILICA_VAULT. A commented `# SILICA_VAULT=`
        # line seeded from .env.example is fine; an active one is not.
        active = [l for l in env_path.read_text().splitlines() if not l.lstrip().startswith("#")]
        assert not any(l.startswith("SILICA_VAULT=") for l in active)

    def test_repo_mode_writes_vault_manifest(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # setup mode → essential
            "",            # repo mode? → default y
            "",            # force language? → Enter, follow source
            "",            # provider → lmstudio
            "test-model",  # model
            "n",           # high-value gate → skip
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        manifest = tmp_path / "vault.yaml"
        assert manifest.is_file()
        text = manifest.read_text(encoding="utf-8")
        assert "sources:" in text and "code" in text
        assert "conventions" not in text and "language" not in text

    def test_repo_mode_writes_forced_language_into_manifest(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # setup mode → essential
            "",            # repo mode? → default y
            "Italian",     # force language? → explicit answer
            "",            # provider → lmstudio
            "test-model",  # model
            "n",           # high-value gate → skip
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        manifest = tmp_path / "vault.yaml"
        text = manifest.read_text(encoding="utf-8")
        assert "conventions:\n  language: Italian" in text

    def test_forced_language_roundtrips_through_load_manifest(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard
        from silica.kernel.vault_manifest import load_manifest

        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = ["", "", "Italian", "", "test-model", "n", ""]
        wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        manifest = load_manifest(str(tmp_path))
        assert manifest.conventions.language == "Italian"

    def test_enter_leaves_language_none_after_load_manifest(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard
        from silica.kernel.vault_manifest import load_manifest

        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = ["", "", "", "", "test-model", "n", ""]
        wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        manifest = load_manifest(str(tmp_path))
        assert manifest.conventions.language is None

    def test_repo_mode_declares_the_root_over_a_legacy_layout(self, monkeypatch, tmp_path):
        # A vault from before the write boundary: the wizard adopts the repo root
        # like any other, names the folder the notes already live in, and leaves
        # the old inner manifest alone.
        import silica.onboarding.wizard as wizard

        (tmp_path / "docs" / "silica").mkdir(parents=True)
        (tmp_path / "docs" / "silica" / "vault.yaml").write_text("sources: [prose]\n", encoding="utf-8")
        (tmp_path / "docs" / "silica" / "nota.md").write_text("# nota", encoding="utf-8")
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = ["", "", "", "", "test-model", "n", ""]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert "write_dir: docs/silica" in (tmp_path / "vault.yaml").read_text(encoding="utf-8")
        assert (tmp_path / "docs" / "silica" / "vault.yaml").read_text(encoding="utf-8") == "sources: [prose]\n"

    def test_eof_mid_wizard_aborts_cleanly(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        env_path = tmp_path / ".env"
        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)

        vault = tmp_path / "vault"
        vault.mkdir()
        # Input exhausts after the vault answer → simulates Ctrl+D mid-wizard
        rc = wizard.run_wizard(input_fn=self._scripted(["", str(vault)]), env_path=env_path)

        assert rc == 1
        assert not env_path.exists()

    def test_non_repo_mode_writes_forced_language_into_manifest(self, monkeypatch, tmp_path):
        """The design's language question is unscoped to repo mode: an explicit-path
        vault with no vault.yaml yet must also be asked, and an explicit answer writes
        a minimal manifest pinning both cooccurrence_lang (stemmer) and the conventions
        language (distiller) — no sources, unlike repo mode."""
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # setup mode → essential
            str(vault),    # vault path (no repo detected)
            "Italian",     # force language? → explicit answer
            "",            # provider → lmstudio
            "test-model",  # model
            "n",           # high-value gate → skip
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        manifest = vault / "vault.yaml"
        assert manifest.is_file()
        assert manifest.read_text(encoding="utf-8") == (
            "cooccurrence_lang: italian\nconventions:\n  language: Italian\n"
        )
        # both axes resolve through load_manifest
        from silica.kernel.vault_manifest import load_manifest

        m = load_manifest(str(vault))
        assert m.cooccurrence_lang == "italian"
        assert m.conventions.language == "Italian"

    def test_non_repo_mode_enter_writes_no_manifest(self, monkeypatch, tmp_path):
        """Enter on the language question must write nothing at all — no
        vault.yaml — mirroring repo mode's 'no conventions block' behavior for
        the case where nothing else was due to be written for this vault."""
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # setup mode → essential
            str(vault),    # vault path (no repo detected)
            "",            # force language? → Enter, follow source
            "",            # provider → lmstudio
            "test-model",  # model
            "n",           # high-value gate → skip
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert not (vault / "vault.yaml").exists()

    def test_repo_mode_yes_like_language_answer_not_written_as_bool(self, monkeypatch, tmp_path):
        """Finding 5 (final multilingua review): an unvalidated "yes" answer to the
        language question parses as a YAML boolean, which `_parse_conventions`
        folds to None — the user believes they forced a language but silently
        didn't. "yes" must be rejected and treated like Enter (no language forced)."""
        import silica.onboarding.wizard as wizard

        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # setup mode → essential
            "",            # repo mode? → default y
            "yes",         # force language? → looks like consent, not a language
            "",            # provider → lmstudio
            "test-model",  # model
            "n",           # high-value gate → skip
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        manifest = tmp_path / "vault.yaml"
        text = manifest.read_text(encoding="utf-8")
        assert "language" not in text

    def test_repo_mode_colon_language_answer_does_not_corrupt_manifest(self, monkeypatch, tmp_path):
        """Finding 5: a colon-containing free-text answer, embedded raw into YAML,
        breaks the whole manifest (repo mode's sources would silently
        degrade to defaults on next load). The answer must be validated before
        it ever reaches the file."""
        import silica.onboarding.wizard as wizard
        from silica.kernel.vault_manifest import load_manifest

        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: tmp_path)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",                   # setup mode → essential
            "",                   # repo mode? → default y
            "English: British",  # force language? → colon would corrupt raw YAML
            "",                   # provider → lmstudio
            "test-model",         # model
            "n",                  # high-value gate → skip
            "",                   # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        manifest = load_manifest(str(tmp_path))
        # sources are written unconditionally in repo mode — an invalid
        # language answer must never degrade the whole manifest to defaults.
        assert "code" in manifest.sources

    def test_non_repo_mode_preserves_existing_manifest_no_question_asked(self, monkeypatch, tmp_path):
        """An existing vault.yaml must never be touched — and the question must not
        even be asked (proven by NOT including an extra scripted answer for it; the
        wizard would abort on EOF if it asked and this test would fail rc != 0)."""
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "vault.yaml").write_text("sources: [prose]\n", encoding="utf-8")
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        answers = [
            "",            # setup mode → essential
            str(vault),    # vault path (no repo detected)
            # NO language answer — vault.yaml already exists, must not be asked
            "",            # provider → lmstudio
            "test-model",  # model
            "n",           # high-value gate → skip
            "",            # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert (vault / "vault.yaml").read_text(encoding="utf-8") == "sources: [prose]\n"


    def test_lmstudio_autodetect_offers_probed_default(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))
        monkeypatch.setattr(wizard, "endpoint_model_ids", lambda url: ["qwen3-30b"])

        answers = [
            "",          # setup mode → essential
            str(vault),  # vault path
            "",          # force language? → Enter
            "",          # provider → lmstudio
            "",          # model id → Enter accepts probed default
            "n",         # high-value gate → skip
            "",          # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert "SILICA_MODEL=qwen3-30b" in env_path.read_text()

    def test_lmstudio_falls_back_to_freetext_when_probe_empty(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))
        # autouse fixture already stubs endpoint_model_ids → []

        answers = [
            "",              # setup mode → essential
            str(vault),      # vault path
            "",              # force language? → Enter
            "",              # provider → lmstudio
            "typed-model",   # model id → must be typed (no probed default)
            "n",             # high-value gate → skip
            "",              # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert "SILICA_MODEL=typed-model" in env_path.read_text()

    def test_embeddings_reuse_single_confirm_local(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))
        monkeypatch.setattr(
            wizard, "endpoint_model_ids", lambda url: ["qwen3-30b", "nomic-embed-text"]
        )

        answers = [
            "",          # setup mode → essential
            str(vault),  # vault path
            "",          # force language? → Enter
            "",          # provider → lmstudio
            "",          # model → Enter accepts probed qwen3-30b
            "",          # high-value gate → default y
            "",          # configure embeddings? → default y
            "",          # "use nomic-embed-text at ...?" → default y
            "",          # autostart command → Enter accepts the lmstudio default
            "",          # in-process reranker? → default n
            "",          # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        content = env_path.read_text()
        assert "SILICA_EMBEDDING_MODEL=nomic-embed-text" in content
        assert "SILICA_EMBEDDING_BASE_URL=http://localhost:1234/v1" in content
        assert "SILICA_EMBEDDING_API_KEY=" in content

    def test_embeddings_no_candidate_prefills_local_base_url(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))
        # A chat model is listed but nothing with "embed" → no candidate.
        monkeypatch.setattr(wizard, "endpoint_model_ids", lambda url: ["qwen3-30b"])

        answers = [
            "",          # setup mode → essential
            str(vault),  # vault path
            "",          # force language? → Enter
            "",          # provider → lmstudio
            "",          # model → Enter accepts qwen3-30b
            "",          # high-value gate → default y
            "y",         # configure embeddings?
            "",          # embedding model → Enter accepts default
            "",          # embedding base URL → Enter accepts pre-filled local
            "",          # embedding API key → Enter accepts default
            "",          # autostart command → Enter accepts the lmstudio default
            "",          # in-process reranker? → default n
            "",          # write → y
        ]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        assert "SILICA_EMBEDDING_BASE_URL=http://localhost:1234/v1" in env_path.read_text()

    def test_fresh_env_seeds_from_example(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        example = tmp_path / "example.env"
        example.write_text(
            "# SILICA_MODEL=example-default\n"
            "# SILICA_PROVIDER=openrouter\n"
            "# SILICA_BACKEND=fs\n"
            "# SILICA_SIM_THRESHOLD_HIGH=0.85\n"
        )
        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"  # does not exist → seed from example

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))
        monkeypatch.setattr(wizard, "_find_env_example", lambda *a: example)

        answers = ["", str(vault), "", "", "test-model", "n", ""]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        content = env_path.read_text()
        assert "SILICA_MODEL=test-model" in content       # collected → uncommented
        assert "SILICA_PROVIDER=lmstudio" in content
        assert "# SILICA_BACKEND=fs" in content            # untouched knob stays commented
        assert "# SILICA_SIM_THRESHOLD_HIGH=0.85" in content

    def test_find_env_example_none_falls_back_to_minimal(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))
        monkeypatch.setattr(wizard, "_find_env_example", lambda *a: None)

        answers = ["", str(vault), "", "", "test-model", "n", ""]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        content = env_path.read_text()
        assert "SILICA_MODEL=test-model" in content
        assert "SIM_THRESHOLD" not in content  # no scaffolding when no example found

    def test_existing_env_ignores_example_and_keeps_unrelated_comment(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        vault = tmp_path / "vault"
        vault.mkdir()
        env_path = tmp_path / ".env"
        env_path.write_text("# SILICA_SIM_THRESHOLD_HIGH=0.85\nFOO=bar\n")

        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))

        def _boom(*a):
            raise AssertionError("_find_env_example must not be consulted when .env exists")

        monkeypatch.setattr(wizard, "_find_env_example", _boom)

        answers = ["", str(vault), "", "", "test-model", "n", ""]
        rc = wizard.run_wizard(input_fn=self._scripted(answers), env_path=env_path)

        assert rc == 0
        content = env_path.read_text()
        assert "# SILICA_SIM_THRESHOLD_HIGH=0.85" in content  # unrelated, still commented
        assert "FOO=bar" in content
        assert "SILICA_MODEL=test-model" in content


class TestEndpointModelIds:
    def test_returns_ids_from_openai_models_payload(self, monkeypatch):
        import silica.onboarding.wizard as wizard

        class _Resp:
            @staticmethod
            def json():
                return {"data": [{"id": "a"}, {"id": "b"}, {"no_id": 1}]}

        monkeypatch.setattr("httpx.get", lambda url, headers=None, timeout=0: _Resp())
        assert wizard.endpoint_model_ids("http://x/v1") == ["a", "b"]

    def test_returns_empty_on_error(self, monkeypatch):
        import silica.onboarding.wizard as wizard

        def _raise(url, timeout=0):
            raise OSError("down")

        monkeypatch.setattr("httpx.get", _raise)
        assert wizard.endpoint_model_ids("http://x/v1") == []


class TestPick:
    """Model pick-list: a number selects, `other` asks, free text passes through."""

    def _scripted(self, answers):
        it = iter(answers)
        return lambda prompt: next(it)

    def test_number_selects_that_option(self):
        from silica.onboarding.wizard import _pick
        got = _pick(self._scripted(["2"]), "Model", ["a", "b", "c"])
        assert got == "b"

    def test_enter_accepts_first_option(self):
        from silica.onboarding.wizard import _pick
        assert _pick(self._scripted([""]), "Model", ["a", "b"]) == "a"

    def test_other_slot_asks_for_an_id(self):
        from silica.onboarding.wizard import _pick
        # 3 = the `other` slot after two options; then the id, empty once first.
        got = _pick(self._scripted(["3", "", "vendor/mine"]), "Model", ["a", "b"])
        assert got == "vendor/mine"

    def test_free_text_bypasses_the_list(self):
        from silica.onboarding.wizard import _pick
        assert _pick(self._scripted(["vendor/typed"]), "Model", ["a"]) == "vendor/typed"

    def test_default_none_lets_enter_answer_nothing(self):
        from silica.onboarding.wizard import _pick
        got = _pick(self._scripted([""]), "Model", ["a"], default=None, required=False)
        assert got == ""

    def test_empty_options_falls_back_to_a_plain_ask(self):
        from silica.onboarding.wizard import _pick
        assert _pick(self._scripted([""]), "Model", [], default="d") == "d"


class TestWizardModes:
    """Spec 2026-07-23: essential/advanced modes, high-value gate, reranker
    step, back navigation, next-steps block."""

    @pytest.fixture(autouse=True)
    def _stub(self, monkeypatch, tmp_path):
        import silica.onboarding.wizard as wizard

        monkeypatch.setattr(wizard, "endpoint_model_ids", lambda url: [])
        monkeypatch.setattr(wizard, "_rerank_extra_present", lambda: False)
        monkeypatch.setattr(wizard.gitstate, "find_repo_root", lambda p: None)
        monkeypatch.setattr(wizard, "run_checks", lambda cfg: [])
        monkeypatch.setattr(wizard.os, "environ", dict(os.environ))
        self.wizard = wizard
        self.vault = tmp_path / "vault"
        self.vault.mkdir()
        self.env_path = tmp_path / ".env"

    def _run(self, answers, **kw):
        it = iter(answers)
        return self.wizard.run_wizard(
            input_fn=lambda p: next(it), env_path=self.env_path, **kw
        )

    def _active(self):
        return [
            l for l in self.env_path.read_text().splitlines()
            if l and not l.lstrip().startswith("#")
        ]

    def test_mode_enter_defaults_to_essential(self):
        # Exact answer count proves no advanced question is asked.
        rc = self._run(["", str(self.vault), "", "", "m", "n", ""])
        assert rc == 0
        assert "SILICA_MODEL=m" in self.env_path.read_text()

    def test_essential_gate_n_writes_no_embedding_or_rerank_keys(self):
        rc = self._run(["", str(self.vault), "", "", "m", "n", ""])
        assert rc == 0
        active = self._active()
        assert not any(l.startswith("SILICA_EMBEDDING") for l in active)
        assert not any(l.startswith("SILICA_RERANK") for l in active)

    def test_essential_gate_y_embeddings_autodetect_still_works(self, monkeypatch):
        monkeypatch.setattr(
            self.wizard, "endpoint_model_ids",
            lambda url: ["qwen3-30b", "nomic-embed-text"],
        )
        answers = [
            "",              # mode → essential
            str(self.vault), # vault path
            "",              # language
            "",              # provider → lmstudio
            "",              # model → probed qwen3-30b
            "",              # high-value gate → default y
            "",              # configure embeddings? → y
            "",              # use candidate? → y
            "",              # autostart command → Enter accepts the lmstudio default
            "",              # in-process reranker? → default n
            "",              # write → y
        ]
        rc = self._run(answers)
        assert rc == 0
        content = self.env_path.read_text()
        assert "SILICA_EMBEDDING_MODEL=nomic-embed-text" in content

    def test_advanced_mode_writes_curated_keys(self):
        answers = [
            "advanced",       # mode
            str(self.vault),  # vault
            "",               # language
            "",               # provider → lmstudio
            "m",              # model
            "skip",           # embeddings (asked directly, no gate)
            "",               # in-process reranker → n
            "worker-x",       # worker model
            "y",              # git auto-commit → auto
            "tvly-key",       # tavily key
            "docling",        # pdf provider
            "",               # OCR languages → default
            "",               # external reranker → default n
            "",               # write → y
        ]
        rc = self._run(answers)
        assert rc == 0
        content = self.env_path.read_text()
        assert "SILICA_WORKER_MODEL=worker-x" in content
        assert "SILICA_GIT_COMMIT=auto" in content
        assert "SILICA_TAVILY_API_KEY=tvly-key" in content
        assert "SILICA_PDF_PROVIDER=docling" in content
        assert "SILICA_PDF_OCR_LANG=en,it,fr,de,es" in content

    def test_advanced_enter_through_writes_no_advanced_keys(self):
        answers = [
            "advanced", str(self.vault), "", "", "m",
            "skip",  # embeddings
            "",      # reranker → n
            "",      # worker → inherit
            "",      # git → leave off
            "",      # tavily → skip
            "",      # pdf → mineru default
            "",      # external reranker → n
            "",      # write → y
        ]
        rc = self._run(answers)
        assert rc == 0
        active = self._active()
        for prefix in (
            "SILICA_WORKER_MODEL", "SILICA_GIT_COMMIT", "SILICA_TAVILY_API_KEY",
            "SILICA_PDF_", "SILICA_RERANK_",
        ):
            assert not any(l.startswith(prefix) for l in active), prefix

    def test_advanced_external_reranker_writes_all_three_keys(self):
        answers = [
            "advanced", str(self.vault), "", "", "m",
            "skip", "",       # embeddings, in-process reranker
            "", "", "", "",   # worker, git, tavily, pdf
            "y",              # external reranker → yes
            "", "", "", "",   # URL, model, key, autostart command → defaults
            "",               # write
        ]
        rc = self._run(answers)
        assert rc == 0
        content = self.env_path.read_text()
        assert "SILICA_RERANK_BASE_URL=http://localhost:1235/v1" in content
        assert "SILICA_RERANK_MODEL=bge-reranker-v2-m3" in content
        assert "SILICA_RERANK_API_KEY=lm-studio" in content

    def test_advanced_flag_skips_mode_question(self):
        # No mode answer: first prompt is already the vault question.
        answers = [
            str(self.vault), "", "", "m",
            "skip", "", "worker-x", "", "", "", "", "",
        ]
        rc = self._run(answers, advanced=True)
        assert rc == 0
        assert "SILICA_WORKER_MODEL=worker-x" in self.env_path.read_text()

    def test_back_from_provider_rewinds_to_vault(self, tmp_path):
        vault_b = tmp_path / "vault-b"
        vault_b.mkdir()
        answers = [
            "",               # mode → essential
            str(self.vault),  # vault → first answer
            "",               # language
            "back",           # provider prompt → go back to vault step
            str(vault_b),     # vault → second answer
            "",               # language (asked again)
            "",               # provider → lmstudio
            "m",              # model
            "n",              # gate → skip to write
            "",               # write → y
        ]
        rc = self._run(answers)
        assert rc == 0
        lines = self.env_path.read_text().splitlines()
        assert f"SILICA_VAULT={vault_b}" in lines
        assert f"SILICA_VAULT={self.vault}" not in lines

    def test_back_on_first_step_reprompts(self):
        rc = self._run(["back", "", str(self.vault), "", "", "m", "n", ""])
        assert rc == 0

    def test_rerank_extra_present_skips_prompt(self, monkeypatch):
        monkeypatch.setattr(self.wizard, "_rerank_extra_present", lambda: True)
        # No reranker answer in the script: if the wizard asked, the write
        # answer would be consumed and the run would abort on EOF.
        answers = ["", str(self.vault), "", "", "m", "", "skip", ""]
        rc = self._run(answers)
        assert rc == 0
        assert not any(l.startswith("SILICA_RERANK") for l in self._active())

    def test_rerank_absent_yes_prints_install_command(self, capsys):
        answers = ["", str(self.vault), "", "", "m", "", "skip", "y", ""]
        rc = self._run(answers)
        assert rc == 0
        assert "silica-harness[rerank]" in capsys.readouterr().out
        assert not any(l.startswith("SILICA_RERANK") for l in self._active())

    def test_next_steps_block_printed(self, capsys):
        rc = self._run(["", str(self.vault), "", "", "m", "n", ""])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Next steps" in out
        assert "silica init" in out


class TestAskSecret:
    def test_secret_default_is_masked_in_prompt(self):
        from silica.onboarding.wizard import _ask

        prompts: list[str] = []

        def fake_input(p: str) -> str:
            prompts.append(p)
            return ""

        result = _ask(fake_input, "OpenRouter API key", "sk-or-verysecret-a1b2", secret=True)
        assert result == "sk-or-verysecret-a1b2"   # empty input keeps the default
        assert "verysecret" not in prompts[0]       # secret never echoed
        assert "a1b2" in prompts[0]                 # last-4 hint shown


class TestUnindexableDocs:
    def test_finds_docs_the_markdown_index_cannot_read(self, tmp_path):
        from silica.onboarding.wizard import unindexable_docs

        (tmp_path / "notes.md").write_text("indexed", encoding="utf-8")
        (tmp_path / "cover.png").write_bytes(b"not a document")
        (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")
        sub = tmp_path / "Inbox"
        sub.mkdir()
        (sub / "book.EPUB").write_bytes(b"epub")

        found = {p.name for p in unindexable_docs(tmp_path)}
        assert found == {"paper.pdf", "book.EPUB"}   # recursive, case-insensitive

    def test_attachments_are_not_offered_as_documents(self, tmp_path):
        """The converter also takes images, decks and media, but an image sitting
        in a vault is an attachment. Counting them would greet someone who pastes
        screenshots with "N documents to convert" — and `<inbox>/Images` is where
        Silica puts the figures it extracted itself, so it would offer to
        re-ingest its own output."""
        from silica.onboarding.wizard import unindexable_docs

        images = tmp_path / "Inbox" / "Images"
        images.mkdir(parents=True)
        for name in ("paper-fig1.png", "pasted-20260813.jpg", "diagram.tiff"):
            (images / name).write_bytes(b"img")
        (tmp_path / "standup.mp4").write_bytes(b"vid")
        (tmp_path / "voice-memo.mp3").write_bytes(b"aud")
        (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "deck.pptx").write_bytes(b"pptx")

        found = {p.name for p in unindexable_docs(tmp_path)}
        assert found == {"report.pdf", "deck.pptx"}

    def test_every_offered_extension_is_one_convert_accepts(self):
        """The narrow list must stay a subset of what the converter dispatches on,
        or onboarding would offer a conversion that immediately errors."""
        from silica.sources.convert import CONVERTIBLE_DOC_EXTS, DOC_EXTS

        assert set(CONVERTIBLE_DOC_EXTS) <= set(DOC_EXTS)

    def test_walk_is_bounded(self, tmp_path):
        from silica.onboarding.wizard import unindexable_docs

        for i in range(30):
            (tmp_path / f"f{i:02d}.pdf").write_bytes(b"%PDF-1.4")
        assert unindexable_docs(tmp_path, cap=10) != unindexable_docs(tmp_path)
