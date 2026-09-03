# SPDX-License-Identifier: AGPL-3.0-or-later
"""Nucleation forms (docs/specs/nucleation-forms.md): ingress stamps first.

A converted media file is a transcript by construction; a /fetch note is a
clip by construction. Both stamp `form:` into frontmatter so downstream
profile resolution never has to guess what the ingress lane already knew.
"""

from pathlib import Path


class TestSniffCallShape:
    def test_sniff_is_deterministic_and_thinks_none(self):
        # One-word classifier: thinking is billed against the 512-token budget
        # and an overrun returns "", which silently picks the fallback lens.
        from unittest.mock import patch as _patch
        from types import SimpleNamespace as _NS
        import silica.kernel.forms as _forms

        _forms._sniff_memo.clear()
        with _patch("silica.kernel.forms.call_llm",
                    return_value=_NS(text="study")) as llm:
            assert _forms.sniff_form("some source text") == "study"
        assert llm.call_args.kwargs["temperature"] == 0
        assert llm.call_args.kwargs["reasoning"] is False


class TestIngressStamps:
    def test_converted_media_is_stamped_form_transcript(self, tmp_path):
        from silica.sources.convert import _provenance_fm

        src = tmp_path / "interview.mp3"
        src.write_bytes(b"\x00")
        fm = _provenance_fm(src)

        assert "form: transcript\n" in fm

    def test_converted_video_is_stamped_form_transcript(self, tmp_path):
        from silica.sources.convert import _provenance_fm

        src = tmp_path / "episode.mp4"
        src.write_bytes(b"\x00")
        fm = _provenance_fm(src)

        assert "form: transcript\n" in fm

    def test_converted_document_carries_no_form_stamp(self, tmp_path):
        """A PDF's form is not knowable at ingress — and not by the sniff
        either, so an unstamped conversion goes to the vault fallback
        (TestConvertedDocumentsAreNotSniffed)."""
        from silica.sources.convert import _provenance_fm

        src = tmp_path / "paper.pdf"
        src.write_bytes(b"%PDF-1.4")
        fm = _provenance_fm(src)

        assert "form:" not in fm

    def test_fetch_note_is_stamped_form_clip(self, tmp_vault, monkeypatch):
        import silica.sources.web_fetch as wf
        from silica.config import CONFIG
        from silica.kernel.write import frontmatter
        from silica.sources.web_research import fetch_to_inbox

        monkeypatch.setattr(
            wf, "fetch_page",
            lambda url: wf.Page("Source: https://x.test\n\nSome article body.", "A Title"),
        )
        note_rel = fetch_to_inbox("https://x.test/post")

        text = (Path(CONFIG.vault_path) / note_rel).read_text(encoding="utf-8")
        data, _, _ = frontmatter.split(text)
        assert data is not None and data.get("form") == "clip"


class TestFormResolution:
    """The ladder: stamp > sniff > vault fallback > default. Verdict always
    carries its origin so the run header can print it."""

    def test_stamped_form_reads_the_frontmatter_key(self):
        from silica.kernel.forms import stamped_form

        text = "---\nform: transcript\nsource_file: \"/x/a.mp3\"\n---\n\nbody\n"
        assert stamped_form(text) == "transcript"

    def test_stamped_form_rejects_unknown_values(self):
        from silica.kernel.forms import stamped_form

        assert stamped_form("---\nform: essay\n---\n\nbody\n") == ""
        assert stamped_form("no frontmatter here\n") == ""

    def test_profile_map_gives_draft_no_lens(self):
        from silica.kernel.forms import profile_for

        assert profile_for("study") == "default"
        assert profile_for("transcript") == "transcript"
        assert profile_for("clip") == "clip"
        assert profile_for("draft") == ""

    def test_resolve_prefers_the_stamp_and_never_sniffs_stamped_text(self, monkeypatch):
        import silica.kernel.forms as forms

        calls = []
        monkeypatch.setattr(forms, "sniff_form", lambda text: calls.append(1) or "clip")
        got = forms.resolve("---\nform: transcript\n---\n\nbody\n")

        assert (got.form, got.profile, got.origin) == ("transcript", "transcript", "stamp")
        assert calls == []

    def test_resolve_sniffs_unstamped_text(self, monkeypatch):
        import silica.kernel.forms as forms

        monkeypatch.setattr(forms, "sniff_form", lambda text: "clip")
        got = forms.resolve("clipped article body\n")

        assert (got.form, got.profile, got.origin) == ("clip", "clip", "sniff")

    def test_resolve_falls_back_when_the_sniff_is_unsure(self, tmp_vault, monkeypatch):
        import silica.kernel.forms as forms

        monkeypatch.setattr(forms, "sniff_form", lambda text: "")
        got = forms.resolve("ambiguous body\n")

        assert (got.form, got.profile, got.origin) == ("", "default", "default")

    def test_resolve_fallback_names_the_vault_profile(self, tmp_vault, monkeypatch):
        import silica.kernel.forms as forms
        import silica.kernel.prep_delegation as prep

        monkeypatch.setattr(forms, "sniff_form", lambda text: "")
        monkeypatch.setattr(prep, "active_distill_profile", lambda: "extractive")
        got = forms.resolve("ambiguous body\n")

        assert (got.form, got.profile, got.origin) == ("", "extractive", "fallback")

    def test_resolve_can_skip_the_sniff(self, monkeypatch):
        import silica.kernel.forms as forms

        def _boom(text):
            raise AssertionError("sniff must not run")

        monkeypatch.setattr(forms, "sniff_form", _boom)
        got = forms.resolve("body\n", allow_sniff=False)
        assert got.origin in ("fallback", "default")


class TestSniffCall:
    def test_sniff_parses_a_noisy_reply(self, monkeypatch):
        import silica.kernel.forms as forms

        class _R:
            text = "Transcript.\nBecause it records a call."

        monkeypatch.setattr(forms, "call_llm", lambda *a, **k: _R())
        assert forms.sniff_form("typed call notes\n") == "transcript"

    def test_sniff_degrades_to_empty_on_error_or_unsure(self, monkeypatch):
        import silica.kernel.forms as forms

        class _R:
            text = "unsure"

        monkeypatch.setattr(forms, "call_llm", lambda *a, **k: _R())
        assert forms.sniff_form("body\n") == ""

        def _raise(*a, **k):
            raise RuntimeError("endpoint down")

        monkeypatch.setattr(forms, "call_llm", _raise)
        assert forms.sniff_form("body\n") == ""


class TestPerFilePlumbing:
    """The resolved profile is pinned per file at PAYLOAD and read at both
    distill sites; the run-level arg (--profile, /promote) wins over the pin."""

    def _stub_fsm(self, pinned=None, run_level=None):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        fsm = SimpleNamespace()
        fsm._chunks = [{"schema_version": 1,
                        "batches": [{"inbox_file": "in.md",
                                     "concepts": [{"name": "c0"}]}]}]
        fsm._chunk_flat_to_fi_ci = {0: (0, 0)}
        fsm._current_file_idx = 0
        fsm._current_chunk_idx = 0
        fsm._file_chunks = {}
        fsm.inbox_file = "in.md"
        fsm.context = dict(pinned or {})
        fsm.target_dir = "Notes"
        fsm.hub = None
        fsm.distill_profile = run_level
        fsm.manifest = MagicMock()
        fsm.manifest.titles.return_value = []
        fsm.progress = MagicMock()
        fsm.progress.digest.return_value = "digest"
        fsm.progress.started_at = "2026-08-16T00:00:00"
        fsm._chunk_task_id = lambda cap, idx=None: f"f0_c0_{cap}"
        fsm._prefetcher = None
        return fsm

    def test_file_profile_prefers_the_run_level_arg(self):
        from silica.router.states.distill import _file_profile

        fsm = self._stub_fsm(pinned={"file_0_profile": "transcript"},
                             run_level="extractive")
        assert _file_profile(fsm, 0) == "extractive"

    def test_file_profile_reads_the_payload_pin(self):
        from silica.router.states.distill import _file_profile

        fsm = self._stub_fsm(pinned={"file_0_profile": "transcript"})
        assert _file_profile(fsm, 0) == "transcript"

    def test_file_profile_defaults_to_none_when_nothing_is_pinned(self):
        from silica.router.states.distill import _file_profile

        fsm = self._stub_fsm()
        assert _file_profile(fsm, 0) is None

    def test_distill_inputs_carry_the_pinned_profile(self):
        from silica.router.states.distill import _distill_inputs

        fsm = self._stub_fsm(pinned={"file_0_profile": "clip"})
        kw = _distill_inputs(fsm, 0)
        assert kw["profile"] == "clip"


class TestPayloadPin:
    def test_pin_resolves_and_stores_profile_form_and_origin(self, tmp_vault, monkeypatch):
        from silica.router.states.setup import _pin_file_profile

        tmp_vault.note("Inbox/talk.md", "---\nform: transcript\n---\n\ncall notes\n")
        fsm = TestPerFilePlumbing()._stub_fsm()
        _pin_file_profile(fsm, 0, "Inbox/talk.md")

        assert fsm.context["file_0_profile"] == "transcript"
        assert fsm.context["file_0_form"] == "transcript"
        assert fsm.context["file_0_form_origin"] == "stamp"

    def test_pin_is_skipped_when_the_run_names_a_profile(self, tmp_vault, monkeypatch):
        import silica.kernel.forms as forms
        from silica.router.states.setup import _pin_file_profile

        def _boom(text, **kw):
            raise AssertionError("resolution must not run under a run-level profile")

        monkeypatch.setattr(forms, "resolve", _boom)
        tmp_vault.note("Inbox/x.md", "body\n")
        fsm = TestPerFilePlumbing()._stub_fsm(run_level="promotion")
        _pin_file_profile(fsm, 0, "Inbox/x.md")

        assert "file_0_profile" not in fsm.context

    def test_a_draft_verdict_inside_the_fsm_degrades_to_the_fallback(self, tmp_vault, monkeypatch):
        """Filing happens at dispatch; a draft that reaches the FSM anyway
        (direct tool call) distills under the vault fallback, verdict kept."""
        import silica.kernel.forms as forms
        from silica.router.states.setup import _pin_file_profile

        monkeypatch.setattr(forms, "sniff_form", lambda text: "draft")
        tmp_vault.note("Inbox/vo.md", "rough VO pass, ignore the ums\n")
        fsm = TestPerFilePlumbing()._stub_fsm()
        _pin_file_profile(fsm, 0, "Inbox/vo.md")

        assert fsm.context["file_0_profile"] == "default"
        assert fsm.context["file_0_form"] == "draft"
        assert fsm.context["file_0_form_origin"] == "sniff"


class TestLenses:
    """The clip lens exists and carries the two-voice rule; the transcript
    lens carries the voice rule. Mechanical presence checks, same spirit as
    the spec's register gate."""

    def test_clip_profile_splices_without_falling_back_to_default(self):
        from silica.kernel.prep_delegation import _splice_lens

        body = "A\n{LENS_RUBRIC}\nB\n{LENS_QUALITY}\nC"
        out = _splice_lens(body, "clip")

        assert "{LENS_" not in out
        # the defining rule: source voice and owner voice never merge
        assert "two voices" in out.lower()
        assert "first person" in out.lower()

    def test_clip_lens_keeps_owner_commentary_as_prose_not_links(self):
        from silica.kernel.prep_delegation import _splice_lens

        out = _splice_lens("{LENS_QUALITY}", "clip")
        assert "wikilink" in out.lower()  # the demotion ban is stated

    def test_transcript_lens_carries_the_voice_rule(self):
        from silica.kernel.prep_delegation import _splice_lens

        out = _splice_lens("{LENS_QUALITY}", "transcript")
        low = out.lower()
        assert "first person" in low
        assert '"the user"' in low or "'the user'" in low  # named as banned


class TestDraftFiling:
    """A draft is filed, not distilled: one note, decent title, form: draft
    frontmatter, body intact, /revert coverage, source archived."""

    def _dispatch(self, tmp_vault, monkeypatch, line):
        import silica.router.coordinator as rc

        calls: dict = {}

        class _FakeCoordinator:
            def __init__(self, **kw):
                calls.update(kw)

            def run(self):
                return {"final_status": "done"}

        monkeypatch.setattr(rc, "Coordinator", _FakeCoordinator)
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))  # not a tty
        from silica.cli import _expand_workflow_shortcut

        return _expand_workflow_shortcut(line), calls

    def test_a_stamped_draft_is_filed_headless_and_skips_the_fsm(self, tmp_vault, monkeypatch):
        from pathlib import Path

        from silica.config import CONFIG
        from silica.kernel.write import frontmatter

        body = "rough VO pass, ignore the ums\n\nthe number that convinced me was five months\n"
        tmp_vault.note("Inbox/vo-ep180.md", f"---\nform: draft\n---\n\n{body}")

        out, calls = self._dispatch(
            tmp_vault, monkeypatch, "/nucleate Inbox/vo-ep180.md --target=Videos"
        )

        assert out == ""
        assert calls == {}  # nothing went to the distiller
        filed = Path(CONFIG.vault_path) / "Videos" / "vo-ep180.md"
        assert filed.is_file()
        data, _, note_body = frontmatter.split(filed.read_text(encoding="utf-8"))
        assert data is not None and data.get("form") == "draft"
        assert "five months" in note_body  # body substantially intact
        # the source is archived, not left to be re-filed on the next run
        assert not (Path(CONFIG.vault_path) / "Inbox" / "vo-ep180.md").exists()

    def test_the_title_comes_from_the_first_heading_never_a_fragment(self, tmp_vault, monkeypatch):
        from pathlib import Path

        from silica.config import CONFIG

        tmp_vault.note(
            "Inbox/dump.md",
            "---\nform: draft\n---\n\n# Local still loses on thumbnails\n\nbody text\n",
        )
        out, calls = self._dispatch(
            tmp_vault, monkeypatch, "/nucleate Inbox/dump.md --target=Videos"
        )

        assert (Path(CONFIG.vault_path) / "Videos" / "Local still loses on thumbnails.md").is_file()

    def test_the_filed_note_is_revert_covered(self, tmp_vault, monkeypatch):
        from pathlib import Path

        from silica.config import CONFIG
        from silica.kernel.write.undo_journal import get_undo_journal, revert_run

        tmp_vault.note("Inbox/vo.md", "---\nform: draft\n---\n\ndraft body here\n")
        self._dispatch(tmp_vault, monkeypatch, "/nucleate Inbox/vo.md --target=Videos")

        filed = Path(CONFIG.vault_path) / "Videos" / "vo.md"
        assert filed.is_file()
        store = get_undo_journal()
        run_id = store.last_active_run()
        assert run_id is not None
        revert_run(run_id, store=store)
        assert not filed.exists()

    def test_a_sniffed_draft_files_too(self, tmp_vault, monkeypatch):
        from pathlib import Path

        import silica.kernel.forms as forms
        from silica.config import CONFIG

        monkeypatch.setattr(forms, "sniff_form", lambda text: "draft")
        tmp_vault.note("Inbox/loose.md", "unstamped working notes\n")
        out, calls = self._dispatch(
            tmp_vault, monkeypatch, "/nucleate Inbox/loose.md --target=Ideas"
        )

        assert calls == {}
        assert (Path(CONFIG.vault_path) / "Ideas" / "loose.md").is_file()

    def test_non_draft_files_still_reach_the_fsm(self, tmp_vault, monkeypatch):
        import silica.kernel.forms as forms

        monkeypatch.setattr(forms, "sniff_form", lambda text: "study")
        tmp_vault.note("Inbox/notes.md", "lecture notes\n")
        out, calls = self._dispatch(
            tmp_vault, monkeypatch, "/nucleate Inbox/notes.md --target=Concepts"
        )

        assert calls.get("inbox_files") == ["Inbox/notes.md"]


def test_sniff_is_memoized_per_content(monkeypatch):
    """Dispatch and PAYLOAD both resolve the same file; only one call is paid."""
    import silica.kernel.forms as forms

    forms._sniff_memo.clear()
    calls = []

    class _R:
        text = "clip"

    monkeypatch.setattr(forms, "call_llm", lambda *a, **k: calls.append(1) or _R())
    assert forms.sniff_form("same body\n") == "clip"
    assert forms.sniff_form("same body\n") == "clip"
    assert len(calls) == 1


class TestVisibility:
    """The form verdict is never silent: it prints at dispatch and the
    declared residue reaches the run report."""

    def test_the_dispatch_announces_the_form_verdict(self, tmp_vault, monkeypatch):
        import silica.cli as cli
        import silica.kernel.forms as forms
        import silica.router.coordinator as rc

        printed = []
        monkeypatch.setattr(cli.CONSOLE, "print", lambda *a, **k: printed.append(" ".join(map(str, a))))
        monkeypatch.setattr(forms, "sniff_form", lambda text: "clip")

        class _Fake:
            def __init__(self, **kw): ...
            def run(self):
                return {"final_status": "done"}

        monkeypatch.setattr(rc, "Coordinator", _Fake)
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
        tmp_vault.note("Inbox/art.md", "clipped body\n")

        cli._expand_workflow_shortcut("/nucleate Inbox/art.md --target=Research")

        assert any("clip" in p and "sniff" in p for p in printed)

    def test_declared_residue_reaches_the_files_summary(self, tmp_vault, monkeypatch):
        import types

        import silica.router.states.finalize as finalize

        fsm = types.SimpleNamespace(
            manifest=types.SimpleNamespace(entries=[]),
            context={"declared_residue": {"a.md": ["fact one", "fact two"]}},
            progress=types.SimpleNamespace(run_id="r1"),
            _file_content_hashes=[],
        )
        finalize._log_nucleate_completion(fsm, 0, "Inbox/a.md")

        summary = fsm.context["files_summary"][0]
        assert summary["residue"] == 2


class TestProfileFlag:
    """`/nucleate --profile=<name>` is the explicit override: it reaches the
    Coordinator and short-circuits the whole form resolution, filing included."""

    def _dispatch(self, monkeypatch, line):
        import silica.router.coordinator as rc

        calls: dict = {}

        class _Fake:
            def __init__(self, **kw):
                calls.update(kw)

            def run(self):
                return {"final_status": "done"}

        monkeypatch.setattr(rc, "Coordinator", _Fake)
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
        from silica.cli import _expand_workflow_shortcut

        _expand_workflow_shortcut(line)
        return calls

    def test_the_flag_reaches_the_coordinator(self, tmp_vault, monkeypatch):
        tmp_vault.note("Inbox/x.md", "body\n")
        calls = self._dispatch(
            monkeypatch, "/nucleate Inbox/x.md --target=Concepts --profile=extractive"
        )
        assert calls.get("distill_profile") == "extractive"

    def test_the_flag_short_circuits_resolution_and_filing(self, tmp_vault, monkeypatch):
        import silica.kernel.forms as forms

        def _boom(text, **kw):
            raise AssertionError("resolution must not run under --profile")

        monkeypatch.setattr(forms, "resolve", _boom)
        tmp_vault.note("Inbox/vo.md", "---\nform: draft\n---\n\ndraft body\n")
        calls = self._dispatch(
            monkeypatch, "/nucleate Inbox/vo.md --target=Videos --profile=transcript"
        )
        # even a draft-stamped file distills under the explicit profile
        assert calls.get("inbox_files") == ["Inbox/vo.md"]
        assert calls.get("distill_profile") == "transcript"


class TestWizardFallbackProposal:
    """Onboarding proposes a vault fallback profile only on a skewed, sizable
    stamped distribution — mechanical census, zero LLM calls."""

    def _stamped_vault(self, tmp_path, counts: dict):
        v = tmp_path / f"wv{sum(counts.values())}{len(counts)}"
        v.mkdir()
        i = 0
        for form, n in counts.items():
            for _ in range(n):
                fm = f"---\nform: {form}\n---\n" if form else "---\nx: 1\n---\n"
                (v / f"n{i}.md").write_text(f"{fm}\nbody\n", encoding="utf-8")
                i += 1
        return v

    def test_a_skewed_stamped_vault_proposes_its_profile(self, tmp_path):
        from silica.onboarding.wizard import propose_form_fallback

        v = self._stamped_vault(tmp_path, {"transcript": 12, "clip": 2})
        assert propose_form_fallback(v) == "transcript"

    def test_an_unstamped_or_small_vault_proposes_nothing(self, tmp_path):
        from silica.onboarding.wizard import propose_form_fallback

        assert propose_form_fallback(self._stamped_vault(tmp_path, {"": 30})) is None
        assert propose_form_fallback(
            self._stamped_vault(tmp_path, {"transcript": 5})
        ) is None  # sample too small

    def test_a_mixed_vault_proposes_nothing(self, tmp_path):
        from silica.onboarding.wizard import propose_form_fallback

        v = self._stamped_vault(tmp_path, {"transcript": 6, "clip": 6})
        assert propose_form_fallback(v) is None

    def test_a_draft_dominant_vault_proposes_nothing(self, tmp_path):
        """draft has no lens: it can never be a fallback profile."""
        from silica.onboarding.wizard import propose_form_fallback

        v = self._stamped_vault(tmp_path, {"draft": 12})
        assert propose_form_fallback(v) is None

    def test_the_offer_writes_only_on_yes(self, tmp_path):
        import yaml

        from silica.onboarding.wizard import _offer_form_fallback

        v = self._stamped_vault(tmp_path, {"transcript": 12})
        _offer_form_fallback(lambda prompt: "n", v)
        assert not (v / "vault.yaml").exists()

        _offer_form_fallback(lambda prompt: "y", v)
        data = yaml.safe_load((v / "vault.yaml").read_text(encoding="utf-8"))
        assert data["conventions"]["distill_profile"] == "transcript"


def test_a_txt_draft_is_read_and_filed_too(tmp_vault, monkeypatch):
    """DRIVER.read_note only reads notes; a .txt source must still be
    classifiable at dispatch (the ep180 case is a .txt)."""
    from pathlib import Path

    import silica.kernel.forms as forms
    import silica.router.coordinator as rc
    from silica.config import CONFIG

    calls: dict = {}

    class _Fake:
        def __init__(self, **kw):
            calls.update(kw)

        def run(self):
            return {"final_status": "done"}

    monkeypatch.setattr(rc, "Coordinator", _Fake)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    monkeypatch.setattr(forms, "sniff_form", lambda text: "draft")
    tmp_vault.note("Inbox/vo-pass.txt", "rough VO pass for ep 180\n")

    from silica.cli import _expand_workflow_shortcut

    _expand_workflow_shortcut("/nucleate Inbox/vo-pass.txt --target=Videos")

    assert calls == {}  # filed, not distilled
    assert (Path(CONFIG.vault_path) / "Videos" / "vo-pass.md").is_file()


def test_a_txt_draft_files_under_auto_target_too(tmp_vault, monkeypatch):
    """_pick_target_folder must read a .txt source: with no --target the ep180
    case died there and fell back to distillation."""
    from pathlib import Path

    import silica.agent.llm as llm
    import silica.kernel.forms as forms
    import silica.router.coordinator as rc
    from silica.config import CONFIG

    calls: dict = {}

    class _Fake:
        def __init__(self, **kw):
            calls.update(kw)

        def run(self):
            return {"final_status": "done"}

    class _R:
        text = "Videos"

    monkeypatch.setattr(rc, "Coordinator", _Fake)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    monkeypatch.setattr(forms, "sniff_form", lambda text: "draft")
    monkeypatch.setattr(llm, "call_llm", lambda *a, **k: _R())
    tmp_vault.note("Concepts/existing.md", "so the folder census is non-empty\n")
    tmp_vault.note("Inbox/vo-pass.txt", "rough VO pass for ep 180\n")

    from silica.cli import _expand_workflow_shortcut

    _expand_workflow_shortcut("/nucleate Inbox/vo-pass.txt")

    assert calls == {}  # filed, not distilled
    assert (Path(CONFIG.vault_path) / "Videos" / "vo-pass.md").is_file()


class TestConvertedSourcesAreNeverDrafts:
    """A file with `source_file:` provenance was produced by convert() — it is a
    converted document, not the owner's working material. On a real library the
    sniffer read OCR'd Coptic transliteration as "draft" and the filing lane
    copied raw book segments into the user's source folder; the stamped copies
    then re-filed themselves on every later run."""

    _SEG = ('---\nsource_title: "Pistis Sophia"\n'
            'source_file: "/lib/Pistis-Sophia.pdf"\n---\n\n'
            "15. OCR'd coptic transliteration\n")

    def test_a_sniffed_draft_verdict_is_vetoed_by_provenance(self, monkeypatch):
        import silica.kernel.forms as forms

        monkeypatch.setattr(forms, "sniff_form", lambda text: "draft")
        got = forms.resolve(self._SEG)

        assert got.form != "draft"

    def test_a_stamped_draft_is_vetoed_by_provenance_too(self):
        """The stamp is Silica's own earlier misfire persisted to disk — the
        amplification vector. Provenance outranks it."""
        import silica.kernel.forms as forms

        stamped = self._SEG.replace("---\n\n", 'form: "draft"\n---\n\n', 1)
        got = forms.resolve(stamped, allow_sniff=False)

        assert got.form != "draft"

    def test_a_transcript_stamp_keeps_working_with_provenance(self):
        """Media conversions legitimately carry BOTH source_file and
        form: transcript — the veto is draft-only."""
        import silica.kernel.forms as forms

        text = ('---\nform: transcript\nsource_file: "/lib/ep.mp3"\n---\n\nwords\n')
        got = forms.resolve(text, allow_sniff=False)

        assert got.form == "transcript" and got.origin == "stamp"


class TestConvertedDocumentsAreNotSniffed:
    """An unstamped converted file came from a document, and the ingress lane
    already stamps the two forms it knows (media -> transcript, /fetch ->
    clip). Sniffing it anyway asks the model to guess a fact provenance
    already carries — and it guesses badly: on eight OCR'd book segments from
    the theology library the sniff returned transcript twice and draft twice,
    and one file answered clip, clip, transcript across three identical calls.
    """

    _SEG = ('---\nsource_title: "The Teachings of Zoroaster"\n'
            'source_file: "/lib/17_Kapadia_Teachings-of-Zoroaster.pdf"\n---\n\n'
            "## THE VISION OF ARDA-VIRAF\n\nThey say that, once upon a time...\n")

    def test_a_converted_document_never_pays_a_sniff(self, monkeypatch):
        import silica.kernel.forms as forms

        calls = []
        monkeypatch.setattr(
            forms, "sniff_form", lambda text: calls.append(text) or "transcript"
        )
        got = forms.resolve(self._SEG)

        assert calls == []
        assert got.form == "" and got.origin in ("default", "fallback")

    def test_an_unconverted_file_is_still_sniffed(self, monkeypatch):
        """The sniff is for what actually is ambiguous: a bare .md/.txt drop."""
        import silica.kernel.forms as forms

        monkeypatch.setattr(forms, "sniff_form", lambda text: "transcript")
        got = forms.resolve("standup, 9am. Ale: the index is rebuilt.\n")

        assert got.form == "transcript" and got.origin == "sniff"


def test_a_filed_draft_lands_inside_the_write_dir(tmp_vault, monkeypatch):
    """The filing wrote to the picked folder verbatim: a bare-folder pick put
    the draft in the user's source tree, beside their own files."""
    from pathlib import Path

    import silica.kernel.forms as forms
    import silica.router.coordinator as rc
    from silica.config import CONFIG
    from silica.kernel.vault_manifest import reset_manifest_cache

    tmp_vault.note("vault.yaml", "write_dir: silica\n")
    reset_manifest_cache()
    tmp_vault.note("Inbox/vo-pass.md", "rough VO pass for ep 180\n")

    class _NoFSM:
        def __init__(self, **kw):
            raise AssertionError("draft must not reach the FSM")

    monkeypatch.setattr(rc, "Coordinator", _NoFSM)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    monkeypatch.setattr(forms, "sniff_form", lambda text: "draft")

    from silica.cli import _expand_workflow_shortcut

    _expand_workflow_shortcut("/nucleate Inbox/vo-pass.md --target=Videos")

    assert (Path(CONFIG.vault_path) / "silica" / "Videos" / "vo-pass.md").is_file()
    assert not (Path(CONFIG.vault_path) / "Videos").exists()


class TestFileDraftsParallelResolve:
    def test_order_and_kept_preserved_with_pool(self):
        # The pool resolves upfront; the printing loop must consume verdicts
        # in input order and keep non-drafts exactly as the serial loop did.
        from unittest.mock import patch as _patch
        import silica.cli as cli
        import silica.kernel.forms as forms

        files = [f"Inbox/f{i}.md" for i in range(5)]
        def fake_resolve(text, **kw):
            return forms.Form("study", "default", "sniff")
        with _patch.object(forms, "read_source_text", side_effect=lambda f: f), \
             _patch.object(forms, "resolve", side_effect=fake_resolve):
            kept = cli._file_drafts(list(files), "Target", None)
        assert kept == files

    def test_resolve_failure_keeps_the_file(self):
        from unittest.mock import patch as _patch
        import silica.cli as cli
        import silica.kernel.forms as forms

        with _patch.object(forms, "read_source_text",
                           side_effect=RuntimeError("boom")):
            kept = cli._file_drafts(["Inbox/x.md"], "Target", None)
        assert kept == ["Inbox/x.md"]
