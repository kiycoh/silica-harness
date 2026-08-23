# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Lane A (survey-provenance spec): the `distinct` verdict leaves a written,
typed, parseable relation trace on the committed spoke.

Builder + parser are a templates.py canonical pair (same single-source-of-truth
idiom as provenance_header); `_route_distinct` is the only emitter. Emit only
the canonical form — there is no legacy form to recognize.
"""

from unittest.mock import patch

from silica.capabilities.dedup import DedupDecision, _route_distinct
from silica.config import SilicaConfig
from silica.kernel.workqueue import WorkItem
from silica.kernel.write.ops import OpType
from silica.kernel.write.templates import (
    has_related_trace,
    parse_related_traces,
    related_trace,
)


# ---------------------------------------------------------------------------
# Builder + parser round-trip (the templates.py pair)
# ---------------------------------------------------------------------------


def test_builder_emits_canonical_form():
    line = related_trace("Gradient Descent", "near but genuinely different")
    assert line == (
        "> [!info] Related: [[Gradient Descent]] "
        "(judged distinct: near but genuinely different)"
    )


def test_builder_collapses_rationale_to_one_line():
    line = related_trace("X", "first line\nsecond line")
    assert "\n" not in line
    assert parse_related_traces(line) == [("X", "first line second line")]


def test_builder_truncates_rationale_to_200_chars():
    line = related_trace("X", "r" * 500)
    [(candidate, rationale)] = parse_related_traces(line)
    assert candidate == "X"
    assert len(rationale) == 200


def test_builder_empty_rationale_omits_colon_segment():
    line = related_trace("X", "")
    assert line == "> [!info] Related: [[X]] (judged distinct)"
    assert parse_related_traces(line) == [("X", "")]


def test_builder_strips_brackets_from_candidate():
    # A candidate title carrying wikilink brackets must not corrupt the link.
    line = related_trace("Bad ]] Title [[", "r")
    assert parse_related_traces(line) == [("Bad Title", "r")]


def test_parser_ignores_plain_related_lines_and_other_callouts():
    body = "\n".join([
        "Intro paragraph.",
        "",
        related_trace("Alpha", "why alpha"),
        "Related: [[Beta]]",
        "> [!info] Unrelated callout mentioning [[Gamma]]",
        "> [!warning] Related: [[Delta]] (judged distinct: wrong callout type)",
    ])
    assert parse_related_traces(body) == [("Alpha", "why alpha")]


def test_has_related_trace_matches_candidate():
    body = "Some text.\n\n" + related_trace("Alpha", "why")
    assert has_related_trace(body, "Alpha")
    assert not has_related_trace(body, "Beta")


# ---------------------------------------------------------------------------
# _route_distinct wiring (pipeline path only)
# ---------------------------------------------------------------------------


def _pipeline_ctx(**over):
    ctx = {
        "concept": "Discesa del gradiente",
        "excerpt": "Variante mini-batch con momentum applicata al training.",
        "candidate": "Gradient Descent",
        "inbox_file": "Inbox/ml.md",
        "hub": "Concepts",
        "target_dir": "Concepts",
    }
    ctx.update(over)
    return ctx


def _item():
    return WorkItem(
        kind="dedup",
        target_path="Concepts/Gradient Descent.md",
        context={},
        reason="test",
    )


def _decision(**over):
    kw = dict(
        verdict="distinct",
        rationale="near but genuinely different",
        title="Discesa del gradiente",
        body="Il metodo iterativo che segue il gradiente negativo.",
    )
    kw.update(over)
    return DedupDecision(**kw)


def _committed_op(ctx, decision):
    with patch(
        "silica.capabilities.dedup.commit_ops",
        return_value={"status": "committed", "committed": 1},
    ) as commit:
        res = _route_distinct(_item(), ctx, decision, SilicaConfig())
    assert res["status"] == "committed"
    ops = commit.call_args.args[0]
    assert len(ops) == 1
    return ops[0]


def test_distinct_pipeline_spoke_carries_exactly_one_trace():
    op = _committed_op(_pipeline_ctx(), _decision())
    assert op.op == OpType.write
    assert parse_related_traces(op.snippet) == [
        ("Gradient Descent", "near but genuinely different")
    ]


def test_trace_replaces_the_old_untyped_related_line():
    op = _committed_op(_pipeline_ctx(), _decision())
    assert not any(
        line.startswith("Related: [[")
        for line in op.snippet.splitlines()
    ), "the untyped 'Related: [[..]]' line must not be emitted anymore"


def test_trace_not_duplicated_when_body_already_carries_one():
    prior = "Testo del giudice.\n\n" + related_trace("Gradient Descent", "prior run")
    op = _committed_op(_pipeline_ctx(), _decision(body=prior))
    assert len(parse_related_traces(op.snippet)) == 1


def test_inline_wikilink_still_gets_typed_trace():
    # An inline [[link]] is not a judged relation: the typed trace is added
    # anyway (this is the behavior change vs the old plain-line guard).
    body = "Vedi [[Gradient Descent]] per il contesto generale."
    op = _committed_op(_pipeline_ctx(), _decision(body=body))
    assert parse_related_traces(op.snippet) == [
        ("Gradient Descent", "near but genuinely different")
    ]


def test_mechanical_fallback_carries_trace_too():
    # No judge-authored title/body: the mechanical excerpt write still leaves
    # the typed trace.
    op = _committed_op(_pipeline_ctx(), _decision(title="", body=""))
    assert "Variante mini-batch con momentum" in op.snippet
    assert parse_related_traces(op.snippet) == [
        ("Gradient Descent", "near but genuinely different")
    ]


def test_ad_hoc_pair_writes_nothing():
    # No target_dir: historical contract, distinct means no write.
    with patch("silica.capabilities.dedup.commit_ops") as commit:
        res = _route_distinct(
            _item(), _pipeline_ctx(target_dir=""), _decision(), SilicaConfig()
        )
    assert res["status"] == "no_merge"
    assert commit.call_count == 0


# ---------------------------------------------------------------------------
# An unparseable judge reply is not a verdict: the spoke stays linked to the
# candidate, but no "judged distinct" record may be written for it.
# ---------------------------------------------------------------------------


def test_unjudged_link_is_not_a_trace():
    from silica.kernel.write.templates import related_unjudged

    line = related_unjudged("Gradient Descent")
    assert "[[Gradient Descent]]" in line
    assert "judged" not in line
    assert "unparseable" not in line
    assert parse_related_traces(line) == []
    assert not has_related_trace(line, "Gradient Descent")


def test_unjudged_decision_links_without_claiming_a_verdict():
    """2026-08-23 run: `rdf.md` carried "(judged distinct: unparseable decision)"
    after the judge's JSON was cut at max_tokens. The pair was never judged;
    the survey parser must not read a verdict off it, and the internal
    diagnostic must not reach vault copy."""
    op = _committed_op(
        _pipeline_ctx(),
        _decision(title="", body="", rationale="unparseable decision", judged=False),
    )
    assert "[[Gradient Descent]]" in op.snippet
    assert parse_related_traces(op.snippet) == []
    assert "unparseable" not in op.snippet
    assert "judged" not in op.snippet
