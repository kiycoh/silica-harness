# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Tool layer of the learner model: richer graded entries in, picker rows out."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

import silica.tools.atomic  # noqa: F401 — @tool registration is import-driven
from silica.tools import TOOLS


@pytest.fixture
def log(tmp_path, monkeypatch):
    import silica.kernel.report.quiz as quiz

    p = tmp_path / "quiz.jsonl"
    monkeypatch.setattr(quiz, "log_path", lambda: p)
    return p


def test_record_quiz_passes_concepts_question_and_anchor(tmp_vault, log):
    out = TOOLS["silica_record_quiz"].fn(results=[
        {"path": "Math/LA.md", "correct": True,
         "concepts": ["eigenvector"], "q": "Define an eigenvector.", "anchor": "#Definitions"},
    ])
    assert out["recorded"] == 1
    line = json.loads(log.read_text().splitlines()[0])
    assert line["concepts"] == ["eigenvector"]
    assert line["q"] == "Define an eigenvector."
    assert line["anchor"] == "#Definitions"


def test_record_quiz_schema_accepts_and_defaults_the_new_fields():
    from silica.tools.atomic import QuizResult

    r = QuizResult(path="A.md", correct=True)  # legacy call shape still validates
    assert r.concepts == [] and r.q == "" and r.anchor == ""


def test_review_queue_replaces_weak_notes(tmp_vault, log):
    assert "silica_weak_notes" not in TOOLS
    tool = TOOLS["silica_review_queue"]
    assert tool.cls == "atomic"

    tmp_vault.note("Old.md", "---\ndate: 2024-01-01\n---\nlong forgotten\n")
    tmp_vault.note("Robot.md", "---\nAI: true\n---\nnever learned\n")
    rows = tool.fn(limit=4)
    whys = {r["path"]: r["why"] for r in rows}
    assert whys["Old.md"] == "due"
    assert whys["Robot.md"] == "unexplored"


def test_review_queue_target_mode_scopes_and_reports_known(tmp_vault, log):
    # Dated off the clock, not frozen: R = exp(-dt / 90 days), so a literal
    # date crosses the 0.9 line about ten days after someone typed it and the
    # test fails on a calendar, not on a regression.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tmp_vault.note("Area/Fresh.md", f"---\ndate: {today}\n---\njust written\n")
    tmp_vault.note("Elsewhere/Old.md", "---\ndate: 2024-01-01\n---\nout of scope\n")
    rows = TOOLS["silica_review_queue"].fn(target="Area/")
    assert [r["path"] for r in rows] == ["Area/Fresh.md"]
    assert rows[0]["R"] > 0.9
