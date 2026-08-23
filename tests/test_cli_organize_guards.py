# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Guards on the strings a model is allowed to turn into filesystem paths.

Every value pinned here reaches the driver: the auto-target folder pick is
composed straight into `<folder>/<title>.md`, and a taxonomy folder becomes an
`os.path.join(vault, folder)` for `DRIVER.move`. Both are generated from vault
content (an ingested document, note titles), so they are the injection channel,
not trusted config.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


# --- auto-target folder pick ------------------------------------------------

class _Reply:
    def __init__(self, text: str) -> None:
        self.text = text


def _pick_with(monkeypatch, reply: str, source_rel: str = "Inbox/01-src.md"):
    import silica.agent.llm as llm
    from silica.cli import _pick_target_folder

    monkeypatch.setattr(llm, "call_llm", lambda *a, **k: _Reply(reply))
    return _pick_target_folder([source_rel])


def test_folder_pick_rejects_a_parent_traversal(tmp_vault, monkeypatch):
    """The prompt carries 1500 chars of the ingested document, so the reply is
    attacker-controlled; `../..` must not compose into a write path."""
    tmp_vault.note("Concepts/existing.md", "census non-empty\n")
    tmp_vault.note("Inbox/01-src.md", "body\n")

    with pytest.raises(ValueError):
        _pick_with(monkeypatch, "../../tmp")


def test_folder_pick_rejects_an_absolute_path(tmp_vault, monkeypatch):
    tmp_vault.note("Concepts/existing.md", "census non-empty\n")
    tmp_vault.note("Inbox/01-src.md", "body\n")

    with pytest.raises(ValueError):
        _pick_with(monkeypatch, "/etc/silica")


def test_folder_pick_rejects_a_backslash_traversal(tmp_vault, monkeypatch):
    """A Windows-style separator is normalised before the check, not after."""
    tmp_vault.note("Concepts/existing.md", "census non-empty\n")
    tmp_vault.note("Inbox/01-src.md", "body\n")

    with pytest.raises(ValueError):
        _pick_with(monkeypatch, "..\\..\\tmp")


def test_folder_pick_returns_a_vault_relative_folder(tmp_vault, monkeypatch):
    """The ordinary pick still works, and an absolute pick that happens to sit
    inside the vault is relativized rather than rejected."""
    tmp_vault.note("Concepts/existing.md", "census non-empty\n")
    tmp_vault.note("Inbox/01-src.md", "body\n")

    assert _pick_with(monkeypatch, "Concepts/AI") == "Concepts/AI"

    from silica.config import CONFIG
    assert _pick_with(monkeypatch, f"{CONFIG.vault_path}/Concepts/AI") == "Concepts/AI"


# --- taxonomy folder validation ---------------------------------------------

def test_folder_rule_rejects_a_traversal():
    from silica.kernel.organize.taxonomy import FolderRule

    with pytest.raises(Exception):
        FolderRule(folder="../../tmp", themes=["x"])


def test_folder_rule_rejects_an_absolute_path_and_a_backslash_traversal():
    from silica.kernel.organize.taxonomy import FolderRule

    with pytest.raises(Exception):
        FolderRule(folder="/etc/silica", themes=["x"])
    with pytest.raises(Exception):
        FolderRule(folder="..\\..\\tmp", themes=["x"])


def test_taxonomy_rejects_an_escaping_uncategorized_and_scope():
    from silica.kernel.organize.taxonomy import Taxonomy

    with pytest.raises(Exception):
        Taxonomy.from_dict({"uncategorized": "../../tmp"})
    # scope is PREPENDED to every rule folder, so it escapes on their behalf.
    with pytest.raises(Exception):
        Taxonomy.from_dict({"scope": "../..", "rules": [{"folder": "AI"}]})


def test_taxonomy_still_accepts_ordinary_folders():
    from silica.kernel.organize.taxonomy import Taxonomy

    t = Taxonomy.from_dict({
        "scope": "Research Notes",
        "uncategorized": "Uncategorized",
        "rules": [{"folder": "DeepSeek", "themes": ["moe"], "keywords": ["MoE"]}],
    })
    assert t.rules[0].folder == "Research Notes/DeepSeek"
    assert t.uncategorized == "Research Notes/Uncategorized"


# --- keyword scoring --------------------------------------------------------

def _score(title_rel: str, keywords: list[str]):
    from silica.kernel.organize.classify import _score_note_against_rules
    from silica.kernel.organize.taxonomy import FolderRule, Taxonomy

    tax = Taxonomy(
        rules=[FolderRule(folder="Concepts/AI", themes=[], keywords=keywords)],
        uncategorized="Uncategorized",
    )
    return _score_note_against_rules(title_rel, {}, tax, {})


def test_keyword_does_not_fire_on_a_substring_collision():
    """"ai" inside "Sustainability" used to score 0.4 — two such collisions
    cleared tau_high and the note was moved without reaching the arbiter."""
    folder, score, evidence, _themes = _score("Notes/Sustainability.md", ["ai", "ml"])
    assert folder == "Uncategorized"
    assert evidence == "uncategorized"
    assert score == 0.0


def test_keyword_still_fires_on_a_whole_word():
    folder, score, evidence, _themes = _score("Notes/AI Agents.md", ["ai"])
    assert folder == "Concepts/AI"
    assert evidence == "keyword"
    assert score == pytest.approx(0.4)


def test_multi_word_keyword_still_matches_as_a_phrase():
    _folder, _score_, evidence, _themes = _score(
        "Notes/Machine Learning Basics.md", ["machine learning"]
    )
    assert evidence == "keyword"


# --- `silica import` bootstraps its own vault -------------------------------

def test_import_activates_the_vault_before_reading_it(tmp_path, monkeypatch):
    """Subcommand dispatch runs before main()'s setup: without an explicit
    activation the envelopes land in whatever vault the config file names."""
    import silica.capture as capture_mod
    from silica import cli
    from silica.config import CONFIG

    monkeypatch.setattr(CONFIG, "vault_path", "")
    vault = tmp_path / "vault"
    order: list[str] = []

    def _activate():
        order.append("activate")
        CONFIG.vault_path = str(vault)

    monkeypatch.setattr(cli, "_activate_repo_mode", _activate)

    seen: dict = {}

    def _run_import(target, vault_arg):
        seen["target"] = target
        seen["vault"] = vault_arg
        return 3, 0

    monkeypatch.setattr(capture_mod, "run_import", _run_import)

    assert cli._dispatch_subcommand(["import", "export.zip"]) == 0
    assert order == ["activate"]
    assert seen["vault"] == str(vault)


# --- the batch hook the REPL renderer owns ----------------------------------

def test_web_search_reuses_the_repl_renderer(tmp_vault, monkeypatch):
    """A throwaway renderer would claim the module-global batch hook and never
    give it back, so /refine and /dedup lost their panel for the session."""
    import silica.sources.web_research as wr
    import silica.ui.renderer as renderer_mod
    from silica import cli

    monkeypatch.setattr(wr, "web_research", lambda *a, **k: "Sources/found.md")
    repl = renderer_mod.make_progress_callback()
    try:
        assert cli._expand_workflow_shortcut("/web-search agents", progress=repl) == ""
        assert renderer_mod._batch_run_hook == repl.__call__
    finally:
        repl.close()


def test_web_search_closes_a_renderer_it_owns(tmp_vault, monkeypatch):
    import silica.sources.web_research as wr
    import silica.ui.renderer as renderer_mod
    from silica import cli

    monkeypatch.setattr(wr, "web_research", lambda *a, **k: "Sources/found.md")
    renderer_mod._batch_run_hook = None
    assert cli._expand_workflow_shortcut("/web-search agents") == ""
    assert renderer_mod._batch_run_hook is None


# --- the injector failure line names the phase that failed ------------------

def test_injector_failure_names_the_failed_phase(monkeypatch):
    from silica.agent.events import ToolCompleteEvent
    from silica.config import CONFIG
    from silica.ui.console import CONSOLE
    from silica.ui.renderer import make_progress_callback

    monkeypatch.setattr(CONSOLE, "_force_terminal", True)
    monkeypatch.setattr(CONFIG, "tool_progress", "all")
    cb = make_progress_callback()
    try:
        cb._injector_call_id = "c1"
        cb._inject_inbox_label = "Inbox"
        cb._inject_file_count = 1
        # Ordered by FIRST appearance: cleanup ran (and succeeded) after recon blew up.
        cb._pipeline_phases = [
            {"phase": "recon", "status": "failed"},
            {"phase": "cleanup", "status": "done"},
        ]
        with CONSOLE.capture() as cap:
            cb(ToolCompleteEvent(
                name="silica_run_injector", args={}, call_id="c1",
                result="{}", duration_s=1.0, iteration=1,
            ))
        out = cap.get()
    finally:
        cb.close()
    assert "failed at recon" in out
    assert "failed at cleanup" not in out


# --- `silica update` without uv on PATH -------------------------------------

def _fake_git(diff_out: str):
    def _git(*args):
        out = ""
        if args[:1] == ("rev-list",):
            out = "2"
        elif args[:1] == ("rev-parse",):
            out = "deadbeef"
        elif args[:1] == ("diff",):
            out = diff_out
        return subprocess.CompletedProcess(list(args), 0, out, "")
    return _git


def _stub_install(tmp_path, monkeypatch, diff_out="pyproject.toml\n"):
    import silica.update as upd

    root = tmp_path / "install"
    (root / ".git").mkdir(parents=True)
    (root / "silica").mkdir()
    monkeypatch.setattr(upd, "ROOT", root)
    monkeypatch.setattr(upd, "_git", _fake_git(diff_out))
    return upd


def test_update_without_uv_reports_instead_of_crashing(tmp_path, monkeypatch, capsys):
    """A missing binary raises FileNotFoundError instead of returning non-zero,
    so the guarded branch never ran and the retry guidance never printed."""
    upd = _stub_install(tmp_path, monkeypatch)
    monkeypatch.setattr(
        upd.shutil, "which", lambda name: "/usr/bin/git" if name == "git" else None
    )

    seen: dict = {}

    def _run(cmd, **kw):
        if "compileall" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        seen["cmd"] = list(cmd)
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(upd.subprocess, "run", _run)

    assert upd.update() == 1
    out = capsys.readouterr().out
    assert "Retry manually" in out
    assert "uv pip install" not in out
    assert seen["cmd"][:3] == [sys.executable, "-m", "pip"]


def test_update_prefers_uv_when_it_is_on_path(tmp_path, monkeypatch, capsys):
    upd = _stub_install(tmp_path, monkeypatch)
    monkeypatch.setattr(
        upd.shutil, "which",
        lambda name: {"git": "/usr/bin/git", "uv": "/usr/local/bin/uv"}.get(name),
    )

    seen: dict = {}

    def _run(cmd, **kw):
        if "compileall" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        seen["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(upd.subprocess, "run", _run)

    assert upd.update() == 0
    assert seen["cmd"][:2] == ["/usr/local/bin/uv", "pip"]


def test_update_without_git_says_so(tmp_path, monkeypatch, capsys):
    upd = _stub_install(tmp_path, monkeypatch)
    monkeypatch.setattr(upd.shutil, "which", lambda name: None)

    assert upd.update() == 1
    assert "git is not on PATH" in capsys.readouterr().out


def test_folder_pick_prompt_says_when_the_vault_has_no_folders(tmp_vault, monkeypatch):
    """A fresh vault (inbox only) used to render "Existing folders:" followed by
    nothing. 2026-08-23 run: the model spent 473 tokens of reasoning on "the
    user says 'Existing folders:' but then nothing follows" before guessing.
    An empty census is an answer, and the prompt has to state it."""
    import silica.agent.llm as llm
    from silica.cli import _pick_target_folder

    captured: dict = {}

    def fake(model, messages, **kw):
        captured["prompt"] = messages[0]["content"]
        return _Reply("Digital Libraries")

    monkeypatch.setattr(llm, "call_llm", fake)
    tmp_vault.note("Inbox/14-digital-libraries/01-src.md", "body\n")

    assert _pick_target_folder(["Inbox/14-digital-libraries/01-src.md"]) == "Digital Libraries"
    prompt = captured["prompt"]
    assert "Existing folders:\n\n" not in prompt
    assert "no folders yet" in prompt


def test_folder_pick_prompt_lists_existing_folders(tmp_vault, monkeypatch):
    import silica.agent.llm as llm
    from silica.cli import _pick_target_folder

    captured: dict = {}

    def fake(model, messages, **kw):
        captured["prompt"] = messages[0]["content"]
        return _Reply("Concepts")

    monkeypatch.setattr(llm, "call_llm", fake)
    tmp_vault.note("Concepts/existing.md", "census non-empty\n")
    tmp_vault.note("Inbox/01-src.md", "body\n")

    assert _pick_target_folder(["Inbox/01-src.md"]) == "Concepts"
    assert "- Concepts" in captured["prompt"]
    assert "no folders yet" not in captured["prompt"]
