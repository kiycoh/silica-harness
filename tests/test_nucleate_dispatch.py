"""/nucleate — one verb, extension dispatch (spec D2).

md/.txt → Coordinator FSM dispatched inline ("" sentinel); target folder from
--target= or a single folder-pick LLM call (agent-message fallback if the pick
fails); code → skeleton stub staged inline, "" sentinel."""
import subprocess
from pathlib import Path

import pytest

from silica.cli import _expand_workflow_shortcut
from silica.config import CONFIG


@pytest.fixture(autouse=True)
def _reset_manifest_cache():
    from silica.kernel.vault_manifest import reset_manifest_cache
    reset_manifest_cache()
    yield
    reset_manifest_cache()


@pytest.fixture
def stub_coordinator(monkeypatch):
    """Record Coordinator ctor kwargs; skip the real FSM."""
    calls: list[dict] = []

    class _FakeCoordinator:
        def __init__(self, **kw):
            calls.append(kw)

        def run(self):
            return {"final_status": "Success"}

    import silica.router.coordinator as coord_mod
    monkeypatch.setattr(coord_mod, "Coordinator", _FakeCoordinator)
    return calls


def test_supported_nucleate_extensions_covers_every_lane():
    # The GUI "+" picker derives its accept= list from this; every nucleate lane
    # (prose, code, notebook, pdf) must be represented or the picker hides files
    # the server would actually accept.
    from silica.kernel.code.codeast import BARE_LANGUAGES, EXTENSION_MAP
    from silica.sources.registry import supported_nucleate_extensions

    exts = set(supported_nucleate_extensions())
    assert {".md", ".txt", ".ipynb", ".pdf"} <= exts  # prose / notebook / pdf lanes
    symbol_bearing = {e for e, lang in EXTENSION_MAP.items() if lang not in BARE_LANGUAGES}
    assert symbol_bearing <= exts                      # every symbol-bearing code language
    # bare languages are graph-only (presence, co-change): not a nucleate lane,
    # so the picker must not advertise them
    assert not {".toml", ".html", ".css"} & exts
    assert all(e.startswith(".") for e in exts)        # accept= wants dotted extensions


def test_code_adapter_matches_new_languages_not_bare():
    from silica.sources.code import CodeAdapter

    adapter = CodeAdapter()
    assert adapter.matches("src/App.java")
    assert adapter.matches("src/main.c")
    assert adapter.matches("include/x.hpp")
    for bare in ("pyproject.toml", "site/index.html", "site/style.css"):
        assert not adapter.matches(bare)


def test_nucleate_md_with_target_dispatches_fsm_directly(stub_coordinator):
    msg = _expand_workflow_shortcut("/nucleate Inbox/a.md --target=Concepts/AI")
    assert msg == ""  # handled inline — no agent turn
    assert stub_coordinator == [
        {"inbox_files": ["Inbox/a.md"], "target_dir": "Concepts/AI", "hub": None,
         "keep_sources": True, "seen_override": None, "distill_profile": None}
    ]


def test_nucleate_md_missing_target_uses_folder_pick(stub_coordinator, monkeypatch):
    import silica.cli as cli_mod
    monkeypatch.setattr(cli_mod, "_pick_target_folder", lambda files: "Concepts/AI")
    msg = _expand_workflow_shortcut("/nucleate Inbox/a.md")
    assert msg == ""
    assert stub_coordinator[0]["target_dir"] == "Concepts/AI"
    assert stub_coordinator[0]["inbox_files"] == ["Inbox/a.md"]


def test_nucleate_folder_pick_failure_falls_back_to_agent(monkeypatch):
    import silica.cli as cli_mod

    def boom(files):
        raise ValueError("no llm")

    monkeypatch.setattr(cli_mod, "_pick_target_folder", boom)
    msg = _expand_workflow_shortcut("/nucleate Inbox/a.md")
    assert msg is not None and "silica_run_injector" in msg
    assert "Inbox/a.md" in msg
    # the agent must pick the folder, not receive a preset one
    assert "target_dir=<chosen folder>" in msg
    assert "most relevant existing vault folder" in msg


def test_nucleate_no_resolvable_files_falls_back_to_agent():
    # A dropped --folder= (starts with '-', so the flag parser skips it) used to
    # hard-error "requires at least one file". Now the raw line goes to the agent
    # to infer intent instead of rejecting it.
    msg = _expand_workflow_shortcut("/nucleate --folder=Inbox/x --target=Concepts")
    assert msg is not None
    assert not msg.startswith("Error:")
    assert "silica_run_injector" in msg
    assert "--folder=Inbox/x" in msg  # the raw input is echoed for the agent


def test_unknown_slash_command_falls_through_to_agent():
    from silica.cli import _handle_slash_command
    # Known meta command → handled deterministically (True).
    assert _handle_slash_command("/model", []) is True
    # Unknown command → None so the REPL hands the raw line to the agent
    # instead of printing "Unknown command".
    assert _handle_slash_command("/ingest --folder=x --target=y", []) is None


def test_inject_shortcut_is_retired():
    assert _expand_workflow_shortcut("/inject Inbox/a.md --target=C") is None


def test_plain_prose_with_apostrophe_is_not_hijacked():
    # An Italian contraction ("L'hub") is a single unmatched shlex quote char.
    # Non-slash input must skip shlex entirely, not get replaced by the
    # "unbalanced quotes" error message.
    msg = _expand_workflow_shortcut("L'hub machine learning quali 5 argomenti fondamentali riporta?")
    assert msg is None


def test_slash_command_unbalanced_quotes_still_errors():
    msg = _expand_workflow_shortcut('/nucleate "Inbox/no closing quote.pdf')
    assert msg == 'Error: unbalanced quotes in command. Wrap paths with spaces in "...".'


def test_run_injector_converts_a_pdf_itself(repo_vault, monkeypatch):
    """The agent tool used to answer "ask the user to run /convert" — told, in a
    REPL, to the person who had just asked the agent to do it. /nucleate has
    always converted inline; the tool now does the same."""
    import silica.tools.runners as runners_mod
    from silica.tools import TOOLS

    seen: dict = {}

    class _Fake:
        def __init__(self, **kw):
            seen.update(kw)
            self.fsm = type("F", (), {"progress": type("P", (), {"run_id": "r1"})()})()

        def run(self):
            return {"final_status": "done"}

    monkeypatch.setattr("silica.router.coordinator.Coordinator", _Fake)
    monkeypatch.setattr("silica.sources.convert.convert",
                        lambda target, dest_dir="": ["Concepts/paper-01.md"])
    assert runners_mod  # module imported for the patch target above

    out = TOOLS["silica_run_injector"].fn(inbox_files=["Inbox/paper.pdf"],
                                          target_dir="Concepts")

    assert "error" not in out
    assert seen["inbox_files"] == ["Concepts/paper-01.md"]


def test_run_injector_names_a_file_no_converter_handles(repo_vault, monkeypatch):
    from silica.tools import TOOLS

    def _boom(target, dest_dir=""):
        raise ValueError("unsupported file type .zip")

    monkeypatch.setattr("silica.sources.convert.convert", _boom)

    out = TOOLS["silica_run_injector"].fn(inbox_files=["Inbox/corpus.zip"],
                                          target_dir="Concepts")
    assert "corpus.zip" in out["error"] and ".zip" in out["error"]


def test_run_injector_asks_for_a_target_before_transcoding(repo_vault, monkeypatch):
    """Converting a 200 MB scan and then bailing on a missing target_dir burns
    minutes for nothing."""
    from silica.tools import TOOLS

    def _never(target, dest_dir=""):
        raise AssertionError("converted before the target check")

    monkeypatch.setattr("silica.sources.convert.convert", _never)

    out = TOOLS["silica_run_injector"].fn(inbox_files=["Inbox/paper.pdf"], target_dir="")
    assert "target_dir" in out["error"]


def test_run_injector_rejects_unknown_type_without_convert_hint(repo_vault):
    from silica.tools import TOOLS

    out = TOOLS["silica_run_injector"].fn(inbox_files=["Inbox/data.csv"], target_dir="Concepts")
    assert "error" in out
    assert "data.csv" in out["error"] and "/convert" not in out["error"]


@pytest.fixture
def repo_vault(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "m.py").write_text("def hi():\n    return 1\n", encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    vault = tmp_path / ".silica"
    vault.mkdir()
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(CONFIG, "inbox_dir", "Inbox")
    from silica.driver import fs_backend
    import silica.driver as driver_mod
    backend = fs_backend.ObsidianFSBackend(str(vault))
    monkeypatch.setattr(driver_mod, "DRIVER", backend)
    # Also install it behind the proxy: modules that bind `DRIVER` at import time
    # (kernel.undo_journal) never see the name-level patch above.
    driver_mod.set_driver(backend)
    yield tmp_path, vault
    driver_mod.set_driver(None)


def test_nucleate_code_stages_stub_and_returns_sentinel(repo_vault):
    root, vault = repo_vault
    msg = _expand_workflow_shortcut("/nucleate m.py")
    assert msg == ""  # fully handled inline, nothing for the agent
    stub = vault / root.name / "m.md"
    assert stub.is_file()
    text = stub.read_text(encoding="utf-8")
    assert "def hi()" in text and "return 1" not in text


def test_nucleate_mixed_batch_stages_code_and_dispatches_md(repo_vault, stub_coordinator):
    root, vault = repo_vault
    msg = _expand_workflow_shortcut("/nucleate m.py Inbox/note.md --target=Concepts")
    assert msg == ""
    assert stub_coordinator[0]["inbox_files"] == ["Inbox/note.md"]  # md → FSM
    # code file NOT forwarded (staged inline)
    assert (vault / root.name / "m.md").is_file()


def test_nucleate_folder_of_code_stages_a_stub_per_file(repo_vault):
    """A folder of source files (no .md in sight) is ingestible: it expands to
    the git-listed files under it, one skeleton stub each."""
    root, vault = repo_vault
    pkg = root / "controller"
    pkg.mkdir()
    (pkg / "Api.java").write_text("class Api { void get() {} }\n", encoding="utf-8")
    (pkg / "Web.java").write_text("class Web { void post() {} }\n", encoding="utf-8")
    (pkg / "notes.txt").write_text("scratch\n", encoding="utf-8")  # no code lane
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "java"], cwd=root, check=True)

    msg = _expand_workflow_shortcut("/nucleate controller")
    assert msg == ""  # handled inline, nothing punted to the agent
    # destination folder is named after the nucleated source folder, not Inbox
    assert (vault / "controller" / "Api.md").is_file()
    assert (vault / "controller" / "Web.md").is_file()


def test_nucleate_folder_of_notes_resolves_without_the_agent(repo_vault, stub_coordinator):
    """A folder of notes expands here, deterministically.

    `expand_folder` reads a git-backed census of the code lane, so it is blind to
    a plain Obsidian vault, and the note index used to skip the inbox — the
    folder reached the agent as a bare name with no listing behind it, and the
    model answered with filenames it had invented.
    """
    _, vault = repo_vault
    folder = vault / "Inbox" / "machine_learning"
    folder.mkdir(parents=True)
    for n in (1, 2, 10):
        (folder / f"Lezione {n}.md").write_text(f"lezione {n}\n", encoding="utf-8")
    (folder / "slides.pdf").write_bytes(b"%PDF-1.4\n")  # unconverted: not a note

    msg = _expand_workflow_shortcut(
        "/nucleate Inbox/machine_learning --target=Concepts/AI"
    )
    assert msg == ""  # handled inline — nothing punted to the agent
    assert stub_coordinator[0]["inbox_files"] == [
        "Inbox/machine_learning/Lezione 1.md",
        "Inbox/machine_learning/Lezione 2.md",
        "Inbox/machine_learning/Lezione 10.md",
    ]


def test_nucleate_absolute_folder_path_resolves_too(repo_vault, stub_coordinator):
    """Users paste the absolute vault path; every listing below is vault-relative."""
    _, vault = repo_vault
    folder = vault / "Inbox" / "ml"
    folder.mkdir(parents=True)
    (folder / "a.md").write_text("a\n", encoding="utf-8")

    msg = _expand_workflow_shortcut(f'/nucleate "{folder}" --target=Concepts/AI')
    assert msg == ""
    assert stub_coordinator[0]["inbox_files"] == ["Inbox/ml/a.md"]


def test_nucleate_empty_folder_still_falls_back_to_the_agent(repo_vault):
    """No listing found is not the same as a listing of nothing: the raw line
    still goes to the agent, which may read an intent the flag parser could not."""
    _, vault = repo_vault
    (vault / "Inbox" / "vuota").mkdir(parents=True)

    msg = _expand_workflow_shortcut("/nucleate Inbox/vuota --target=Concepts/AI")
    assert msg is not None and "silica_run_injector" in msg


def test_nucleate_run_is_revertable(repo_vault):
    """The terminal lane skips the FSM, which is where journalling lived — so
    without its own run these writes were the only ones /revert could not see."""
    from silica.kernel.write.undo_journal import get_undo_journal, revert_run

    root, vault = repo_vault
    assert _expand_workflow_shortcut("/nucleate m.py") == ""
    note = vault / root.name / "m.md"
    assert note.is_file()

    run_id = get_undo_journal().last_active_run(vault=str(vault))
    assert run_id, "nucleate must open an undo run for this vault"
    res = revert_run(run_id)
    assert res["reverted"] == [f"{root.name}/m.md"]
    assert not note.exists()  # the note did not exist before → undo deletes it


def test_nucleate_revert_restores_a_refreshed_note(repo_vault):
    """Re-nucleating an existing note must undo to its prior body, not delete it."""
    from silica.kernel.write.undo_journal import get_undo_journal, revert_run

    root, vault = repo_vault
    _expand_workflow_shortcut("/nucleate m.py")
    note = vault / root.name / "m.md"
    first = note.read_text(encoding="utf-8")

    (root / "m.py").write_text("def hi():\n    return 2\n\ndef bye():\n    pass\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "second"], cwd=root, check=True)
    _expand_workflow_shortcut("/nucleate m.py")
    assert "def bye()" in note.read_text(encoding="utf-8")

    revert_run(get_undo_journal().last_active_run(vault=str(vault)))
    assert note.read_text(encoding="utf-8") == first


def test_expand_folder_ignores_untracked_noise_and_escapes(repo_vault):
    from silica.sources.registry import expand_folder

    root, _ = repo_vault
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.js").write_text("x\n", encoding="utf-8")
    (root / ".gitignore").write_text("node_modules/\n", encoding="utf-8")

    assert expand_folder("node_modules") == []   # git-ignored → never staged
    assert expand_folder("/etc") == []           # outside the repo
    assert expand_folder("m.py") == []           # a file, not a folder
    assert expand_folder("nope") == []           # missing


def test_silica_files_lists_code_under_a_folder(repo_vault):
    """A code folder holds no .md: the old listing read as empty and the agent
    concluded there was nothing to nucleate."""
    from silica.tools import TOOLS

    root, _ = repo_vault
    pkg = root / "controller"
    pkg.mkdir()
    (pkg / "Api.java").write_text("class Api {}\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "java"], cwd=root, check=True)

    out = TOOLS["silica_files"].fn(folder="controller")
    assert out["files"] == [] and out["total"] == 0
    assert out["code"] == ["controller/Api.java"] and out["code_total"] == 1
    # a bare call stays a vault listing — no repo dump into the context window
    assert "code" not in TOOLS["silica_files"].fn()


def test_nucleate_unsupported_extension_is_skipped(repo_vault, capsys):
    root, vault = repo_vault
    msg = _expand_workflow_shortcut("/nucleate data.csv")
    assert msg == ""  # handled, nothing for the agent
    assert not (vault / "Inbox" / "data.md").exists()
    out = capsys.readouterr().out
    assert "data.csv" in out and "Skipped" in out  # warning is part of the contract


def test_nucleate_folder_and_connective_words_falls_back_to_agent(repo_vault):
    # "Inbox/lacascia in lacascia/" — a folder plus the connective word "in".
    # None of the three tokens resolves to an ingestible file (no extension), but
    # the intent is clear, so the raw line goes to the agent instead of silently
    # doing nothing.
    root, vault = repo_vault
    msg = _expand_workflow_shortcut("/nucleate Inbox/lacascia in lacascia/")
    assert msg is not None and msg != ""   # not the silent "handled" sentinel
    assert not msg.startswith("Error:")
    assert "silica_run_injector" in msg


def test_nucleate_pdf_converts_and_forwards_converted_md(repo_vault, monkeypatch, stub_coordinator):
    """No adapter claims .pdf → convert() runs and the CONVERTED .md is what
    the FSM is told to re-read (not the .pdf)."""
    import silica.sources.convert as conv_mod

    monkeypatch.setattr(conv_mod, "convert", lambda f, dest_dir="": ["Inbox/paper.md"])
    msg = _expand_workflow_shortcut("/nucleate paper.pdf --target=Concepts/AI")
    assert msg == ""
    assert stub_coordinator[0]["inbox_files"] == ["Inbox/paper.md"]  # converted .md, not the .pdf


def test_nucleate_pdf_converter_error_is_caught(repo_vault, monkeypatch, capsys):
    import silica.sources.convert as conv_mod

    def boom(f, dest_dir=""):
        raise ValueError("mineru not installed")

    monkeypatch.setattr(conv_mod, "convert", boom)
    msg = _expand_workflow_shortcut("/nucleate paper.pdf --target=Concepts/AI")
    assert msg == ""  # nothing to run; batch did not crash
    assert "mineru not installed" in capsys.readouterr().out


def test_convert_command_returns_sentinel_and_reports(repo_vault, monkeypatch, capsys):
    import silica.sources.convert as conv_mod

    monkeypatch.setattr(conv_mod, "convert", lambda f, dest_dir="": ["Inbox/paper.md"])
    msg = _expand_workflow_shortcut("/convert paper.pdf")
    assert msg == ""  # fully handled inline
    assert "Converted" in capsys.readouterr().out


def test_convert_command_no_files_errors():
    msg = _expand_workflow_shortcut("/convert --target=X")
    assert msg is not None and msg.startswith("Error:")


# ---------------------------------------------------------------------------
# Re-nucleate-of-modified-source warning (spec-hermes-coherence §3): a file
# about to be staged whose basename is already registered in
# .silica/provenance.json under a DIFFERENT sha256 means notes derived from
# it may now be stale.
# ---------------------------------------------------------------------------

def test_nucleate_renucleate_of_modified_source_warns(repo_vault, capsys, stub_coordinator):
    root, vault = repo_vault
    inbox = vault / "Inbox"
    inbox.mkdir(exist_ok=True)
    (inbox / "lezione.md").write_text("v2 content", encoding="utf-8")

    from silica.kernel.write.provenance import append_record
    append_record("lezione.md", "old-sha-not-matching", "run1", ["Concepts/A", "Concepts/B"])

    msg = _expand_workflow_shortcut("/nucleate Inbox/lezione.md --target=Concepts/AI")
    assert msg == ""

    out = capsys.readouterr().out
    assert "re-nucleate of a modified source" in out
    assert "2 note" in out


def test_nucleate_same_sha_no_warning(repo_vault, capsys, stub_coordinator):
    root, vault = repo_vault
    inbox = vault / "Inbox"
    inbox.mkdir(exist_ok=True)
    (inbox / "lezione.md").write_text("same content", encoding="utf-8")

    from silica.kernel.write.provenance import append_record, content_sha256
    sha = content_sha256("Inbox/lezione.md")
    append_record("lezione.md", sha, "run1", ["Concepts/A"])

    msg = _expand_workflow_shortcut("/nucleate Inbox/lezione.md --target=Concepts/AI")
    assert msg is not None

    out = capsys.readouterr().out
    assert "re-nucleate of a modified source" not in out


def test_nucleate_no_prior_provenance_no_warning(repo_vault, capsys, stub_coordinator):
    root, vault = repo_vault
    inbox = vault / "Inbox"
    inbox.mkdir(exist_ok=True)
    (inbox / "fresh.md").write_text("brand new", encoding="utf-8")

    msg = _expand_workflow_shortcut("/nucleate Inbox/fresh.md --target=Concepts/AI")
    assert msg is not None

    out = capsys.readouterr().out
    assert "re-nucleate of a modified source" not in out


def test_nucleate_missing_target_still_warns_on_renucleate(repo_vault, capsys, monkeypatch):
    """Auto-target (no --target) is a valid invocation — the provenance
    drift warning must still print on the way to the agent fallback."""
    import silica.cli as cli_mod
    root, vault = repo_vault
    inbox = vault / "Inbox"
    inbox.mkdir(exist_ok=True)
    (inbox / "lezione.md").write_text("v2 content", encoding="utf-8")

    from silica.kernel.write.provenance import append_record
    append_record("lezione.md", "old-sha-not-matching", "run1", ["Concepts/A", "Concepts/B"])

    monkeypatch.setattr(cli_mod, "_pick_target_folder",
                        lambda files: (_ for _ in ()).throw(ValueError("no llm")))
    msg = _expand_workflow_shortcut("/nucleate Inbox/lezione.md")

    assert msg is not None and "silica_run_injector" in msg
    out = capsys.readouterr().out
    assert "re-nucleate of a modified source" in out


def test_settings_sets_and_shows_vault_yaml(repo_vault, capsys):
    from silica.config import CONFIG
    from silica.kernel.vault_manifest import get_active_manifest, reset_manifest_cache

    msg = _expand_workflow_shortcut("/settings conventions.language italian")
    assert msg == ""
    assert "language" in (Path(CONFIG.vault_path) / "vault.yaml").read_text()
    assert get_active_manifest().conventions.language == "italian"  # cache reset

    msg = _expand_workflow_shortcut("/settings")
    assert msg == ""
    assert "italian" in capsys.readouterr().out

    assert _expand_workflow_shortcut("/settings bogus.key x").startswith("Error:")
    reset_manifest_cache()


def test_run_injector_projects_outcomes_not_raw_context(repo_vault, monkeypatch):
    """Agent boundary gets outcomes only: no payload/recon (planned concepts once
    read as 'created notes'), per-chunk failures not last-error-wins."""
    import silica.router.coordinator as coord_mod

    raw = {
        "final_status": "failed",
        "committed_chunks": 0,
        "failed_chunks": [{"chunk": f"f0_c{i}", "error": "boom"} for i in range(6)],
        "error": "Critical failure delegating batch 5: boom",
        "payload": {"chunks": ["planned concepts must not leak"]},
        "recon": {"concepts": ["Stimatore media campionaria"]},
        "subagents": {},
    }

    class _Fake:
        def __init__(self, **kw):
            self.fsm = type("F", (), {"progress": type("P", (), {"run_id": "r1"})()})()

        def run(self):
            return raw

    monkeypatch.setattr(coord_mod, "Coordinator", _Fake)
    from silica.tools import TOOLS

    out = TOOLS["silica_run_injector"].fn(inbox_files=["Inbox/a.md"], target_dir="C")
    assert out["final_status"] == "failed"
    assert out["chunks_failed"] == 6 and out["chunks_committed"] == 0
    assert len(out["failed_chunks"]) == 6
    assert "payload" not in out and "recon" not in out
    assert out["run_id"] == "r1"


# --- auto-target: cold-start and inbox self-pick -----------------------------
#
# Both defects were measured on a 74-PDF research library whose first
# `/nucleate <book>.pdf` produced zero notes in ~10 minutes: the folder census
# saw no .md anywhere, the model echoed the inbox subfolder back, and VALIDATE
# then rejected all 31 ops with "contains forbidden inbox segment".

def test_auto_target_refuses_a_pick_inside_the_inbox(tmp_vault, monkeypatch):
    """A pick under the inbox is rejected at the source, not 20 LLM calls later
    at VALIDATE, where it costs a whole run and writes nothing."""
    import silica.agent.llm as llm
    from silica.cli import _pick_target_folder

    class _R:
        text = "Inbox/Book-of-Enoch"

    monkeypatch.setattr(llm, "call_llm", lambda *a, **k: _R())
    tmp_vault.note("Concepts/existing.md", "census non-empty\n")
    tmp_vault.note("Inbox/Book-of-Enoch/01-enoch.md", "body\n")

    with pytest.raises(ValueError):
        _pick_target_folder(["Inbox/Book-of-Enoch/01-enoch.md"])


def test_folder_census_offers_folders_that_hold_no_markdown(tmp_vault, monkeypatch):
    """A researcher's library is folders of PDFs. Censusing only .md parents
    hides the whole taxonomy and leaves the model nothing to pick."""
    import silica.agent.llm as llm
    from silica.cli import _pick_target_folder

    seen: dict = {}

    class _R:
        text = "02-apocrifi"

    def _spy(model, messages, **kw):
        seen["prompt"] = messages[0]["content"]
        return _R()

    monkeypatch.setattr(llm, "call_llm", _spy)
    tmp_vault.note("02-apocrifi/Book-of-Enoch.pdf", "%PDF-1.4\n")
    tmp_vault.note("04-massoneria/Mackey.pdf", "%PDF-1.4\n")
    tmp_vault.note("Inbox/01-enoch.md", "body\n")

    assert _pick_target_folder(["Inbox/01-enoch.md"]) == "02-apocrifi"
    assert "02-apocrifi" in seen["prompt"]
    assert "04-massoneria" in seen["prompt"]
    assert "- Inbox" not in seen["prompt"]


# --- folder batch: idempotent re-runs, no re-OCR ----------------------------
#
# A research library is 74 scanned books at minutes of OCR + minutes of LLM
# each: the batch is days of wall clock, so it WILL be interrupted. Re-running
# `/nucleate <folder>` must resume — skip books already in done/, reuse
# segments already converted, drop segments already distilled — instead of
# paying the whole corpus again. All three identities are already on disk
# (frontmatter `source_file`, the done/ archive, provenance.json); these pin
# that the batch reads them.

def _library(tmp_vault, monkeypatch):

    import silica.cli as cli_mod

    tmp_vault.note("vault.yaml", "write_dir: silica\n")
    tmp_vault.note("04-massoneria/mackey.pdf", "%PDF mackey")
    tmp_vault.note("04-massoneria/memphis.pdf", "%PDF memphis")
    tmp_vault.note("04-massoneria/_foto/set-a/p1.jpg", "img")
    tmp_vault.note("04-massoneria/_foto/set-a/p2.jpg", "img")
    from silica.kernel.vault_manifest import reset_manifest_cache
    reset_manifest_cache()

    converted: list[str] = []

    def _fake_convert(target, dest_dir=""):
        converted.append(target)
        stem = Path(target).stem
        rel = f"silica/Inbox/{stem}/01-{stem}.md"
        tmp_vault.note(rel, f"# {stem}\n\nbody of {stem}\n")
        return [rel]

    monkeypatch.setattr("silica.sources.convert.convert", _fake_convert)
    monkeypatch.setattr(cli_mod, "_pick_target_folder",
                        lambda files: "silica/04-massoneria")
    return converted


def test_folder_batch_converts_documents_and_leaves_page_photos_alone(
        tmp_vault, monkeypatch, stub_coordinator):
    converted = _library(tmp_vault, monkeypatch)

    msg = _expand_workflow_shortcut("/nucleate 04-massoneria")

    assert msg == ""
    assert sorted(converted) == ["04-massoneria/mackey.pdf",
                                 "04-massoneria/memphis.pdf"]
    # One Coordinator run per book — the pipeline unit. Book B converts while
    # book A distills, so their segments must never share a run.
    assert [c["inbox_files"] for c in stub_coordinator] == [
        ["silica/Inbox/mackey/01-mackey.md"],
        ["silica/Inbox/memphis/01-memphis.md"],
    ]


def test_an_already_ingested_book_is_not_reconverted(
        tmp_vault, monkeypatch, stub_coordinator):

    from silica.config import CONFIG

    converted = _library(tmp_vault, monkeypatch)
    abs_pdf = str(Path(CONFIG.vault_path) / "04-massoneria/mackey.pdf")
    tmp_vault.note("silica/done/01-mackey.md",
                   f'---\nsource_file: "{abs_pdf}"\n---\n\nbody\n')

    _expand_workflow_shortcut("/nucleate 04-massoneria")

    assert converted == ["04-massoneria/memphis.pdf"]
    assert stub_coordinator[0]["inbox_files"] == [
        "silica/Inbox/memphis/01-memphis.md"]


def test_interrupted_conversion_segments_are_reused_not_reocred(
        tmp_vault, monkeypatch, stub_coordinator):

    from silica.config import CONFIG

    converted = _library(tmp_vault, monkeypatch)
    abs_pdf = str(Path(CONFIG.vault_path) / "04-massoneria/mackey.pdf")
    tmp_vault.note("silica/Inbox/mackey/01-mackey.md",
                   f'---\nsource_file: "{abs_pdf}"\n---\n\nseg one\n')
    tmp_vault.note("silica/Inbox/mackey/02-mackey.md",
                   f'---\nsource_file: "{abs_pdf}"\n---\n\nseg two\n')

    _expand_workflow_shortcut("/nucleate 04-massoneria")

    assert converted == ["04-massoneria/memphis.pdf"]
    # Reused segments are a ready unit and dispatch first (their distill is
    # what the conversion of the next book overlaps with).
    assert [c["inbox_files"] for c in stub_coordinator] == [
        ["silica/Inbox/mackey/01-mackey.md", "silica/Inbox/mackey/02-mackey.md"],
        ["silica/Inbox/memphis/01-memphis.md"],
    ]


def test_an_unchanged_already_distilled_segment_is_dropped(
        tmp_vault, monkeypatch, stub_coordinator):
    from silica.kernel.write.provenance import append_record, content_sha256

    tmp_vault.note("vault.yaml", "write_dir: silica\n")
    from silica.kernel.vault_manifest import reset_manifest_cache
    reset_manifest_cache()
    tmp_vault.note("silica/Inbox/b/01-x.md", "segment one\n")
    tmp_vault.note("silica/Inbox/b/02-y.md", "segment two\n")
    append_record("01-x.md", content_sha256("silica/Inbox/b/01-x.md"),
                  "prior-run", ["silica/T/X"])

    _expand_workflow_shortcut(
        "/nucleate silica/Inbox/b/01-x.md silica/Inbox/b/02-y.md --target=T")

    assert stub_coordinator[0]["inbox_files"] == ["silica/Inbox/b/02-y.md"]


def test_a_zero_yield_record_does_not_freeze_the_failure(
        tmp_vault, monkeypatch, stub_coordinator):
    """A prior run that produced no notes (everything deferred) must not make
    the skip permanent — the segment gets another chance."""
    from silica.kernel.write.provenance import append_record, content_sha256

    tmp_vault.note("vault.yaml", "write_dir: silica\n")
    from silica.kernel.vault_manifest import reset_manifest_cache
    reset_manifest_cache()
    tmp_vault.note("silica/Inbox/b/01-x.md", "segment one\n")
    append_record("01-x.md", content_sha256("silica/Inbox/b/01-x.md"),
                  "prior-run", [])

    _expand_workflow_shortcut("/nucleate silica/Inbox/b/01-x.md --target=T")

    assert stub_coordinator[0]["inbox_files"] == ["silica/Inbox/b/01-x.md"]


def test_a_fully_ingested_batch_ends_without_a_run(
        tmp_vault, monkeypatch, stub_coordinator):

    from silica.config import CONFIG

    converted = _library(tmp_vault, monkeypatch)
    for stem in ("mackey", "memphis"):
        abs_pdf = str(Path(CONFIG.vault_path) / f"04-massoneria/{stem}.pdf")
        tmp_vault.note(f"silica/done/01-{stem}.md",
                       f'---\nsource_file: "{abs_pdf}"\n---\n\nbody\n')

    msg = _expand_workflow_shortcut("/nucleate 04-massoneria")

    assert msg == ""            # nothing left is an answer, not an agent turn
    assert converted == []
    assert stub_coordinator == []


def test_next_book_converts_while_the_previous_distills(
        tmp_vault, monkeypatch):
    """The ~2x lever: conversion is local OCR, distillation is network LLM —
    they use different resources and used to run strictly in sequence. The
    conversion of book B must complete DURING book A's distill, not after."""
    import threading

    import silica.cli as cli_mod
    import silica.router.coordinator as coord_mod

    tmp_vault.note("vault.yaml", "write_dir: silica\n")
    tmp_vault.note("04-massoneria/alpha.pdf", "%PDF alpha")
    tmp_vault.note("04-massoneria/beta.pdf", "%PDF beta")
    from silica.kernel.vault_manifest import reset_manifest_cache
    reset_manifest_cache()

    timeline: list[str] = []
    first_dispatch_started = threading.Event()
    overlap_seen = threading.Event()

    def _fake_convert(target, dest_dir=""):
        stem = Path(target).stem
        if stem == "beta":
            # Sequential pipeline: this wait times out (beta converts before
            # any dispatch). Overlapped: alpha's distill is already running.
            if first_dispatch_started.wait(5):
                overlap_seen.set()
        timeline.append(f"convert:{stem}")
        rel = f"silica/Inbox/{stem}/01-{stem}.md"
        tmp_vault.note(rel, f"body of {stem}\n")
        return [rel]

    class _SlowCoordinator:
        def __init__(self, **kw):
            self.files = kw["inbox_files"]

        def run(self):
            timeline.append(f"dispatch:{Path(self.files[0]).stem}")
            first_dispatch_started.set()
            return {"final_status": "Success"}

    monkeypatch.setattr("silica.sources.convert.convert", _fake_convert)
    monkeypatch.setattr(coord_mod, "Coordinator", _SlowCoordinator)
    monkeypatch.setattr(cli_mod, "_pick_target_folder",
                        lambda files: "silica/04-massoneria")

    _expand_workflow_shortcut("/nucleate 04-massoneria")

    assert overlap_seen.is_set(), f"no overlap — timeline: {timeline}"
    assert timeline == ["convert:alpha", "dispatch:01-alpha",
                        "convert:beta", "dispatch:01-beta"]


def test_keep_sources_is_on_by_default_and_can_be_turned_off(stub_coordinator):
    """The verbatim leaf in sources/ is what makes a note's source reachable at
    all (reliability_tier reads it), and sources/ is retrieval-invisible, so
    keeping it costs nothing but disk. --no-keep-sources is the way back."""
    _expand_workflow_shortcut("/nucleate Inbox/a.md --target=Concepts/AI")
    assert stub_coordinator[-1]["keep_sources"] is True

    _expand_workflow_shortcut("/nucleate Inbox/a.md --target=Concepts/AI --no-keep-sources")
    assert stub_coordinator[-1]["keep_sources"] is False

    # the old explicit flag still parses, for anything that scripts it
    _expand_workflow_shortcut("/nucleate Inbox/a.md --target=Concepts/AI --keep-sources")
    assert stub_coordinator[-1]["keep_sources"] is True
