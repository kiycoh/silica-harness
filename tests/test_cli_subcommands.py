"""Tests for the `silica doctor` / `silica init` subcommand dispatch."""
from __future__ import annotations



class TestDispatchSubcommand:
    def test_no_subcommand_returns_none(self):
        from silica.cli import _dispatch_subcommand
        assert _dispatch_subcommand([]) is None
        assert _dispatch_subcommand(["--something"]) is None

    def test_doctor_ok_returns_zero(self, monkeypatch):
        import silica.onboarding.checks as checks
        from silica.cli import _dispatch_subcommand
        ok = checks.CheckResult("chat model", "ok", "x")
        monkeypatch.setattr(checks, "run_checks", lambda cfg: [ok])
        monkeypatch.setattr(checks, "render_report", lambda results: None)
        assert _dispatch_subcommand(["doctor"]) == 0

    def test_doctor_failure_returns_one(self, monkeypatch):
        import silica.onboarding.checks as checks
        from silica.cli import _dispatch_subcommand
        bad = checks.CheckResult("vault", "fail", "missing", "run `silica init`")
        monkeypatch.setattr(checks, "run_checks", lambda cfg: [bad])
        monkeypatch.setattr(checks, "render_report", lambda results: None)
        assert _dispatch_subcommand(["doctor"]) == 1

    def test_doctor_hold_returns_two(self, monkeypatch):
        """Nothing failed, but a row needs reading: exit 2, so a script can
        tell that from a clean run without parsing the table."""
        import silica.onboarding.checks as checks
        from silica.cli import _dispatch_subcommand
        hold = checks.CheckResult("rerank", "warn", "unreachable")
        monkeypatch.setattr(checks, "run_checks", lambda cfg: [hold])
        monkeypatch.setattr(checks, "render_report", lambda results: None)
        assert _dispatch_subcommand(["doctor"]) == 2

    def test_doctor_json_parses_and_mirrors_the_exit_code(self, monkeypatch, capsys):
        """stdout is the payload: nothing else may print on it."""
        import json

        import silica.cli as cli
        import silica.onboarding.checks as checks
        from silica.cli import _dispatch_subcommand

        bad = checks.CheckResult("vault", "fail", "missing", "run `silica init`")
        monkeypatch.setattr(checks, "run_checks", lambda cfg: [bad])
        # The autostart prints to the console; under --json that has to be
        # redirected away from stdout, which is what this asserts.
        monkeypatch.setattr(cli, "_ensure_servers", lambda: print("starting llama-server…"))

        code = _dispatch_subcommand(["doctor", "--json"])

        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert code == 1
        assert payload["results"] == [
            {"name": "vault", "status": "fail", "detail": "missing", "hint": "run `silica init`"}
        ]

    def test_live_probe_is_a_row_in_both_outcomes(self, monkeypatch, capsys):
        """A passing --live run must be distinguishable from --live never run,
        and a failing one must show WHY the exit code is 1 in the report itself."""
        import json

        import silica.cli as cli
        import silica.onboarding.checks as checks
        from silica.cli import _dispatch_subcommand

        monkeypatch.setattr(checks, "run_checks",
                            lambda cfg: [checks.CheckResult("chat model", "ok", "x")])
        for replied, status, code in ((True, "ok", 0), (False, "fail", 1)):
            monkeypatch.setattr(cli, "_doctor_live_probe", lambda r=replied: r)
            assert _dispatch_subcommand(["doctor", "--live", "--json"]) == code
            rows = {r["name"]: r["status"]
                    for r in json.loads(capsys.readouterr().out)["results"]}
            assert rows["live probe"] == status

    def test_init_delegates_to_wizard(self, monkeypatch):
        import silica.onboarding.wizard as wizard_mod
        from silica.cli import _dispatch_subcommand
        seen = {}

        def fake(advanced=False):
            seen["advanced"] = advanced
            return 0

        monkeypatch.setattr(wizard_mod, "run_wizard", fake)
        assert _dispatch_subcommand(["init"]) == 0
        assert seen["advanced"] is False

    def test_init_advanced_flag_forwarded(self, monkeypatch):
        import silica.onboarding.wizard as wizard_mod
        from silica.cli import _dispatch_subcommand
        seen = {}

        def fake(advanced=False):
            seen["advanced"] = advanced
            return 0

        monkeypatch.setattr(wizard_mod, "run_wizard", fake)
        assert _dispatch_subcommand(["init", "--advanced"]) == 0
        assert seen["advanced"] is True


class TestAutolaunchWizard:
    def _spy_env(self, monkeypatch, *, configured, tty, wizard_done):
        import os
        import sys
        import silica.cli as cli
        import silica.onboarding.wizard as wizard_mod

        calls = {"wizard": 0, "execve": 0}
        monkeypatch.setattr(cli, "_model_configured", lambda: configured)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: tty)
        if wizard_done:
            monkeypatch.setenv("SILICA_WIZARD_DONE", "1")
        else:
            monkeypatch.delenv("SILICA_WIZARD_DONE", raising=False)

        def _run():
            calls["wizard"] += 1
            return 0

        def _execve(*a):
            calls["execve"] += 1  # spy: do not actually replace the process

        monkeypatch.setattr(wizard_mod, "run_wizard", _run)
        monkeypatch.setattr(os, "execve", _execve)
        return calls

    def test_fires_when_unconfigured_and_tty(self, monkeypatch):
        from silica.cli import _autolaunch_wizard_if_unconfigured
        calls = self._spy_env(monkeypatch, configured=False, tty=True, wizard_done=False)
        _autolaunch_wizard_if_unconfigured()
        assert calls["wizard"] == 1
        assert calls["execve"] == 1

    def test_skipped_when_wizard_done(self, monkeypatch):
        from silica.cli import _autolaunch_wizard_if_unconfigured
        calls = self._spy_env(monkeypatch, configured=False, tty=True, wizard_done=True)
        _autolaunch_wizard_if_unconfigured()
        assert calls["wizard"] == 0
        assert calls["execve"] == 0

    def test_skipped_when_not_tty(self, monkeypatch):
        from silica.cli import _autolaunch_wizard_if_unconfigured
        calls = self._spy_env(monkeypatch, configured=False, tty=False, wizard_done=False)
        _autolaunch_wizard_if_unconfigured()
        assert calls["wizard"] == 0
        assert calls["execve"] == 0

    def test_noop_when_already_configured(self, monkeypatch):
        from silica.cli import _autolaunch_wizard_if_unconfigured
        calls = self._spy_env(monkeypatch, configured=True, tty=True, wizard_done=False)
        _autolaunch_wizard_if_unconfigured()
        assert calls["wizard"] == 0
        assert calls["execve"] == 0

    def test_no_reexec_when_wizard_aborts(self, monkeypatch):
        import os
        import sys
        import silica.cli as cli
        from silica.cli import _autolaunch_wizard_if_unconfigured
        import silica.onboarding.wizard as wizard_mod

        execve_calls = []
        monkeypatch.setattr(cli, "_model_configured", lambda: False)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.delenv("SILICA_WIZARD_DONE", raising=False)
        monkeypatch.setattr(wizard_mod, "run_wizard", lambda: 1)  # aborted
        monkeypatch.setattr(os, "execve", lambda *a: execve_calls.append(a))

        _autolaunch_wizard_if_unconfigured()
        assert execve_calls == []  # aborted wizard → no re-exec
