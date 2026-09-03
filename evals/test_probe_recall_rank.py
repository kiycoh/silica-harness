# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""probe_recall_rank end-to-end on a synthetic LME-shaped fixture.

Offline: co-occurrence leg only (--no-embed --no-rerank), so no server is
needed and the ranking is deterministic. What must hold is the measurement
arithmetic — the curve is monotone, recall@k matches the histogram, and the
free-cut point is the earliest N that already reaches recall@k.
"""
from __future__ import annotations

import json


def _inst(qid: str, question: str, gold_session: str, n_sessions: int) -> dict:
    """One LME instance: `gold_session` is the only haystack session whose turns
    mention the question's subject; the rest is filler."""
    sids = [f"s{i}" for i in range(n_sessions)]
    sessions = []
    for sid in sids:
        if sid == gold_session:
            turns = [{"role": "user", "content": f"my {question} is Tuesday evening"},
                     {"role": "assistant", "content": f"noted, {question} on Tuesday"}]
        else:
            turns = [{"role": "user", "content": "unrelated chatter about groceries"},
                     {"role": "assistant", "content": "acknowledged the grocery list"}]
        sessions.append(turns)
    return {"question_id": qid, "question_type": "single-session-user",
            "question": f"when is my {question}?", "answer": "Tuesday evening",
            "question_date": "2026-05-01",
            "haystack_session_ids": sids,
            "haystack_dates": ["2026-01-01"] * n_sessions,
            "haystack_sessions": sessions,
            "answer_session_ids": [gold_session]}


def _inst_crowded(qid: str, subject: str, gold_session: str, n_sessions: int) -> dict:
    """Like `_inst`, but EVERY session mentions the subject, so the cooccur leg
    returns a real ranking instead of a single match. A one-match haystack makes
    the cap curve flat and measures nothing (see the vacuity test below)."""
    inst = _inst(qid, subject, gold_session, n_sessions)
    for sid, turns in zip(inst["haystack_session_ids"], inst["haystack_sessions"]):
        if sid == gold_session:
            continue
        turns[0]["content"] = f"is my {subject} still on this week"
        turns[1]["content"] = f"your {subject} schedule is unchanged"
    return inst


def test_probe_reports_ranks_curve_and_free_cut(tmp_path, capsys):
    from evals.probe_recall_rank import main

    data = [_inst_crowded("q1", "yoga class", "s2", 6),
            _inst_crowded("q2", "dentist appointment", "s4", 6)]
    data_path = tmp_path / "lme.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "rep.json"

    rc = main(["--data", str(data_path), "--run-root", str(tmp_path / "vaults"),
               "--k", "5", "--no-embed", "--no-rerank", "--out", str(out)])
    assert rc == 0

    doc = json.loads(out.read_text(encoding="utf-8"))
    rep = doc["report"]
    assert rep["n_questions"] == 2
    assert rep["sessions_per_vault"] == {"mean": 6.0, "min": 6, "max": 6}
    # Crowded haystack: more than one session competes, so the curve can move.
    assert all(len(q["cum_chars"]) > 1 for q in doc["questions"])

    # Every gold that was retrieved owns exactly one rank; the histogram must
    # account for precisely those hits, no double counting.
    assert sum(rep["gold_rank_histogram"].values()) == \
        sum(len(q["gold_ranks"]) for q in doc["questions"])

    curve = rep["curve"]
    assert [c["n"] for c in curve] == [1, 2, 3, 4, 5]
    # Recall is cumulative and cost grows with N: both monotone non-decreasing.
    assert all(b["recall"] >= a["recall"] for a, b in zip(curve, curve[1:]))
    assert all(b["mean_chars"] >= a["mean_chars"] for a, b in zip(curve, curve[1:]))
    assert curve[-1]["mean_chars"] > curve[0]["mean_chars"]  # cost really grows

    # free_cut is the FIRST N reaching recall@k.
    cut = rep["free_cut"]
    assert next(c["recall"] for c in curve if c["n"] == cut) == rep["recall_at_k"]
    assert all(c["recall"] < rep["recall_at_k"] for c in curve if c["n"] < cut)
    if cut < curve[-1]["n"]:
        assert rep["free_cut_tokens_saved"] > 0

    printed = capsys.readouterr().out
    assert "gold rank histogram" in printed
    assert "free cut at N=" in printed
    # The cap-vs-k caveat must never be silent when --verify-k was not passed.
    assert any("not a change to DEFAULT_K" in n for n in rep["notes"])


def test_single_match_haystack_is_reported_as_vacuous(tmp_path):
    """When only the gold session matches, gold is rank 1 everywhere and the
    curve is flat — a cap looks free because nothing was ever ranked. The probe
    must say so rather than report free_cut=1 as a finding."""
    from evals.probe_recall_rank import main

    data = [_inst("q1", "yoga class", "s2", 6)]
    data_path = tmp_path / "lme.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "rep.json"

    main(["--data", str(data_path), "--run-root", str(tmp_path / "v"),
          "--k", "5", "--no-embed", "--no-rerank", "--out", str(out)])
    rep = json.loads(out.read_text(encoding="utf-8"))["report"]

    assert rep["gold_rank_histogram"] == {"1": 1}
    assert rep["free_cut"] == 1
    assert rep["free_cut_tokens_saved"] == 0  # one block retrieved, nothing to cut
    assert any("cannot discriminate a cap" in n for n in rep["notes"])


def test_verify_k_arm_measures_the_real_pipeline(tmp_path):
    """--verify-k re-runs perceive(k=N); its recall is recorded per question and
    reported next to cap@N, because stage 1 truncates before the rerank."""
    from evals.probe_recall_rank import main

    data = [_inst("q1", "yoga class", "s2", 6)]
    data_path = tmp_path / "lme.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "rep.json"

    main(["--data", str(data_path), "--run-root", str(tmp_path / "v"),
          "--k", "5", "--verify-k", "2", "--no-embed", "--no-rerank",
          "--out", str(out)])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["questions"][0]["verified_recall"]["2"] in (0.0, 1.0)
    assert any("k=2: true recall" in n for n in doc["report"]["notes"])


def test_sample_is_stratified_by_type_and_seed_stable():
    """longmemeval_s is grouped by type on disk (measured: the first 100
    answerable questions cover 2 of 6 types), so a head slice is biased."""
    from evals.probe_recall_rank import sample

    data = ([{"question_id": f"a{i}", "question_type": "temporal"} for i in range(60)]
            + [{"question_id": f"b{i}", "question_type": "multi"} for i in range(30)]
            + [{"question_id": f"c{i}", "question_type": "pref"} for i in range(10)])

    got = sample(data, 20, seed="0")
    assert len(got) == 20
    counts = {t: sum(1 for q in got if q["question_type"] == t)
              for t in ("temporal", "multi", "pref")}
    assert counts == {"temporal": 12, "multi": 6, "pref": 2}  # proportional
    # A head slice would have been all-temporal; stratification is the point.
    assert sample(data, 20, seed="0") == got                    # seed-stable
    assert sample(data, 20, seed="1") != got                    # seed actually moves it
    # Oversized request degrades to the whole corpus, no duplicates.
    everything = sample(data, 500, seed="0")
    assert len(everything) == len(data) == len({q["question_id"] for q in everything})


def test_abstention_and_goldless_questions_are_excluded(tmp_path):
    """Abstention golds are synthetic markers absent from the haystack: a rank
    is undefined for them, not a 0.0 miss (runner.aggregate's rule)."""
    from evals.probe_recall_rank import main

    keep = _inst("q1", "yoga class", "s2", 4)
    abst = _inst("q2_abs", "dentist appointment", "s1", 4)
    goldless = _inst("q3", "book club", "s1", 4)
    goldless["answer_session_ids"] = []
    data_path = tmp_path / "lme.json"
    data_path.write_text(json.dumps([keep, abst, goldless]), encoding="utf-8")
    out = tmp_path / "rep.json"

    main(["--data", str(data_path), "--run-root", str(tmp_path / "v"),
          "--k", "3", "--no-embed", "--no-rerank", "--out", str(out)])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert [q["question_id"] for q in doc["questions"]] == ["q1"]
