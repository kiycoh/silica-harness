# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""A scan that saw zero inputs is not a pass.

Three gates answered "clean" over nothing: the integrity probe returned rate
1.0 over zero notes, and 1.0 is the exact-one value the golden runner gates
on; the OKF census read a missing or empty vault as a conformant bundle; the
linter CLI printed "All files validated successfully" over an empty target
folder. A path typo, an over-eager .silicaignore or a vault never opened would
have passed all three. An empty scan now reads "not measured" where a
diagnostic consumes it and fails where a gate does.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from silica.config import SilicaConfig


def _cfg(vault_path: str) -> SilicaConfig:
    cfg = SilicaConfig()
    cfg.vault_path = vault_path
    return cfg


class TestTheIntegrityProbe:
    def test_zero_notes_is_not_measured(self, tmp_path):
        from silica.kernel.link.health import integrity_probe

        res = integrity_probe(tmp_path)

        assert res["notes"] == 0
        assert res["rate"] is None

    def test_the_golden_gate_fails_an_unmeasured_rate(self):
        from evals.golden import runner

        base = {"metrics": {"integrity.rate": 1.0}, "config": {}}
        doc = {"metrics": {"integrity.rate": None}, "config": {}}

        fails = runner.compare(base, doc)

        assert len(fails) == 1
        assert "zero notes" in fails[0]

    def test_the_golden_gate_still_passes_a_measured_one(self):
        from evals.golden import runner

        doc = {"metrics": {"integrity.rate": 1.0}, "config": {}}
        assert runner.compare(doc, doc) == []

    def test_the_runner_refuses_a_vault_with_no_notes(self, tmp_path, capsys):
        """The root guard: one refusal before any probe runs, so no probe's
        zero reaches compare() or gets frozen as a baseline."""
        from evals.golden import runner

        with pytest.raises(SystemExit) as exc:
            runner.resolve_vault(str(tmp_path))

        assert exc.value.code == 2
        assert "0 notes" in capsys.readouterr().out

    def test_the_table_renders_an_unmeasured_rate(self, capsys):
        """print_table formats every metric as a float: None crashed the run
        at the one moment the table had something new to say."""
        from evals.golden import runner

        doc = {
            "vault": {"path": "/v", "notes": 0, "digest": "sha256:x"},
            "tier": "cheap",
            "config": {"cooccur_store": "s", "cooccur_lang": "en",
                       "embedding_model": "m"},
            "metrics": {"integrity.rate": None, "integrity.notes": 0},
        }
        runner.print_table(doc, None)
        out = capsys.readouterr().out
        assert "integrity.rate" in out


class TestTheOkfCensus:
    def test_counts_what_it_saw(self, tmp_path):
        from silica.kernel.write.notetype import okf_conformance

        (tmp_path / "a.md").write_text("---\ntype: Note\n---\n\nB\n", encoding="utf-8")
        census = okf_conformance(tmp_path)

        assert census.scanned == 1
        assert census.violations == []

    def test_refuses_a_missing_vault(self, tmp_path):
        from silica.kernel.write.notetype import okf_conformance

        with pytest.raises(NotADirectoryError):
            okf_conformance(tmp_path / "nope")

    def test_doctor_row_over_zero_notes_is_unknown(self, tmp_path):
        from silica.onboarding.checks import check_okf

        result = check_okf(_cfg(str(tmp_path)))

        assert result.status == "unknown"
        assert "no notes" in result.detail

    def test_doctor_row_over_a_missing_vault_is_unknown(self, tmp_path):
        """The vault row already fails on the path; this row must simply not
        claim a bundle it never walked."""
        from silica.onboarding.checks import check_okf

        result = check_okf(_cfg(str(tmp_path / "nope")))

        assert result.status == "unknown"


class TestTheLinterCli:
    def test_refuses_an_empty_target(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, "-m", "silica.kernel.link.linter", "--target", str(tmp_path)],
            capture_output=True, text=True, timeout=120,
        )

        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "no .md files" in proc.stdout + proc.stderr


class TestTheLanguageRow:
    def test_over_zero_notes_is_unknown(self, tmp_path):
        """"no notes yet" read as ok: the same empty walk the OKF row now
        refuses to call a bundle."""
        from silica.onboarding.checks import check_language

        result = check_language(_cfg(str(tmp_path)))

        assert result.status == "unknown"
        assert "no notes" in result.detail
