"""Structured steering feedback (per-op verdicts) + partial-rejection steer arc.

Paper-aligned (PDDL-INSTRUCT, arXiv:2509.13351): detailed per-op validator
feedback with the previous output echoed beats a flat reasons blob, and the
steer loop must also cover partial rejections, re-delegating only the
rejected concepts while validated ops are carried forward.
"""
import json
from unittest.mock import patch

from silica.kernel.prep_delegation import render_steer_feedback
from silica.router.orchestrator import InjectorFSM, InjectorState
from silica.router.recipe_parser import load_recipe

# Loaded before any test patches builtins.open (same guard as test_fsm.py).
_RECIPE = load_recipe("injector")


def _rejection(heading="Beta", path="bad/Beta.md", reason="Path 'bad/Beta.md' not in target folder", **op_extra):
    return {
        "op": {"op": "write", "heading": heading, "path": path,
               "source_basename": "test.md", "snippet": "some body", **op_extra},
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# render_steer_feedback
# ---------------------------------------------------------------------------

def test_render_has_header_and_per_op_verdict():
    msg = render_steer_feedback([_rejection()], attempt=1, max_attempts=2)
    assert msg.startswith("## STEERING CORRECTION (attempt 1/2)")
    assert '[write] "Beta"' in msg
    assert "Path 'bad/Beta.md' not in target folder" in msg
    # previous op is echoed so the model sees what it emitted
    assert '"heading": "Beta"' in msg


def test_render_truncates_long_snippet_in_echo():
    long_body = "x" * 5000
    msg = render_steer_feedback([_rejection(snippet=long_body)], attempt=1, max_attempts=2)
    assert long_body not in msg
    assert "truncated" in msg
    # reason stays verbatim even when the echo is truncated
    assert "not in target folder" in msg


def test_render_lists_accepted_ops_as_do_not_reemit():
    accepted = [{"op": "write", "heading": "Alpha", "path": "TargetDir/Alpha.md"}]
    msg = render_steer_feedback([_rejection()], attempt=1, max_attempts=2,
                                accepted=accepted, partial=True)
    assert "TargetDir/Alpha.md" in msg
    assert "do NOT re-emit" in msg
    # partial scope: payload was filtered to the rejected concepts
    assert "ONLY the concepts" in msg


def test_render_full_scope_asks_for_full_regeneration():
    msg = render_steer_feedback([_rejection()], attempt=2, max_attempts=2, partial=False)
    assert "attempt 2/2" in msg
    assert "ONLY the concepts" not in msg


def test_render_survives_entry_without_op():
    # Legacy/degenerate rejection entries carry only a reason.
    msg = render_steer_feedback([{"reason": "bad path"}], attempt=1, max_attempts=2)
    assert "bad path" in msg


def test_render_includes_ungrounded_spans_as_advisory():
    ungrounded = [{"heading": "Alpha", "path": "TargetDir/Alpha.md",
                   "spans": ["a fact nowhere in the payload"]}]
    msg = render_steer_feedback([_rejection()], attempt=1, max_attempts=2,
                                ungrounded=ungrounded)
    assert "Grounding warnings" in msg
    assert "a fact nowhere in the payload" in msg
    # advisory, not a verdict: accepted stays accepted
    assert "not rejected" in msg


def test_distiller_prompt_carries_contrastive_examples():
    from silica.kernel.prep_delegation import render_prompt
    with patch("silica.kernel.vault_manifest.get_active_manifest") as mm:
        mm.return_value.conventions.language = "English"
        mm.return_value.conventions.max_tags = 3
        prompt = render_prompt(target="TargetDir", hub="Hub")
    assert "### Contrastive Examples" in prompt
    assert "Invented heading" in prompt
    assert "Placeholder body" in prompt
    assert "Descriptive meta-body" in prompt


# ---------------------------------------------------------------------------
# VALIDATE: partial-rejection steer arc
# ---------------------------------------------------------------------------

_CHUNK = {
    "schema_version": 1,
    "batches": [{
        "inbox_file": "Inbox/test.md",
        "concepts": [
            {"name": "Alpha", "inbox_excerpt": "alpha facts"},
            {"name": "Beta", "inbox_excerpt": "beta facts"},
        ],
    }],
}

_VALIDATED_ALPHA = {"op": "write", "heading": "Alpha", "path": "TargetDir/Alpha.md",
                    "source_basename": "test.md", "snippet": "alpha body"}


def _fsm_at_validate(parsed=None):
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._chunks = [_CHUNK]
    fsm._current_chunk_idx = 0
    fsm.context.setdefault("chunk", {})["sanitized"] = {"parsed": parsed if parsed is not None else []}
    fsm.context["source_content_hash"] = ""  # deferred store: skipped
    fsm.state = InjectorState.VALIDATE
    return fsm


@patch("silica.router.orchestrator.silica_validate_ops")
def test_partial_rejection_steers_only_rejected_concepts(mock_validate):
    mock_validate.return_value = {
        "success": True, "rejection_rate": 0.5, "total": 2,
        "validated_count": 1, "rejected_count": 1,
        "validated_ops": [_VALIDATED_ALPHA],
        "rejected_ops": [_rejection(heading="Beta")],
        "ungrounded": [{"heading": "Alpha", "path": "TargetDir/Alpha.md",
                        "spans": ["ungrounded claim"]}],
    }
    fsm = _fsm_at_validate()
    fsm.step()

    assert fsm.state == InjectorState.DELEGATE
    assert fsm.context["chunk_0_steer_attempts"] == 1
    # retry payload holds ONLY the rejected concept
    retry = fsm.context["chunk_0_retry_payload"]
    names = [c["name"] for b in retry["batches"] for c in b["concepts"]]
    assert names == ["Beta"]
    # validated ops carried forward for the merge on the next VALIDATE pass
    assert fsm.context["chunk_0_carry_ops"] == [_VALIDATED_ALPHA]
    # structured feedback: header, verdict, accepted list
    steer = fsm.context["chunk_0_steer_context"]
    assert steer.startswith("## STEERING CORRECTION")
    assert "Beta" in steer and "not in target folder" in steer
    assert "TargetDir/Alpha.md" in steer
    # ungrounded spans on accepted ops ride along as advisory feedback
    assert "ungrounded claim" in steer


@patch("silica.router.orchestrator.silica_validate_ops")
def test_partial_rejection_specialized_lanes_do_not_steer(mock_validate):
    # near-title → dedup judge; short snippet → expand worker: no steer arc.
    mock_validate.return_value = {
        "success": True, "rejection_rate": 0.5, "total": 3,
        "validated_count": 1, "rejected_count": 2,
        "validated_ops": [_VALIDATED_ALPHA],
        "rejected_ops": [
            _rejection(heading="Beta", reason="near_title candidate='Betas' path='TargetDir/Betas.md' — deferred for dedup review"),
            _rejection(heading="Beta", reason="snippet too short (0 < 100 chars) — would write a placeholder note, deferred for retry"),
        ],
    }
    fsm = _fsm_at_validate()
    fsm.step()

    assert fsm.state == InjectorState.SNAPSHOT
    assert "chunk_0_retry_payload" not in fsm.context
    assert fsm.context.get("chunk_0_steer_attempts", 0) == 0


@patch("silica.router.orchestrator.silica_validate_ops")
def test_partial_steer_respects_budget(mock_validate):
    mock_validate.return_value = {
        "success": True, "rejection_rate": 0.5, "total": 2,
        "validated_count": 1, "rejected_count": 1,
        "validated_ops": [_VALIDATED_ALPHA],
        "rejected_ops": [_rejection(heading="Beta")],
    }
    fsm = _fsm_at_validate()
    fsm.context["chunk_0_steer_attempts"] = 2  # budget already exhausted
    fsm.step()

    assert fsm.state == InjectorState.SNAPSHOT
    assert "chunk_0_retry_payload" not in fsm.context


@patch("silica.router.orchestrator.silica_validate_ops")
def test_carry_ops_merge_into_next_validate_pass(mock_validate):
    mock_validate.return_value = {
        "success": True, "rejection_rate": 0.0, "total": 2,
        "validated_count": 2, "rejected_count": 0,
        "validated_ops": [], "rejected_ops": [],
    }
    retry_op = {"op": "write", "heading": "Beta", "path": "TargetDir/Beta.md",
                "source_basename": "test.md", "snippet": "beta body"}
    fsm = _fsm_at_validate(parsed={"updates": [retry_op]})
    fsm.context["chunk_0_carry_ops"] = [_VALIDATED_ALPHA]
    fsm.step()

    ops_path = mock_validate.call_args.args[0]
    with open(ops_path, encoding="utf-8") as f:
        ops = json.load(f)
    headings = [o["heading"] for o in ops]
    # carry ops re-enter the gate FIRST (dedup keeps them ahead of re-emits)
    assert headings == ["Alpha", "Beta"]
    assert "chunk_0_carry_ops" not in fsm.context


# ---------------------------------------------------------------------------
# DELEGATE: retry payload consumption
# ---------------------------------------------------------------------------

@patch("silica.router.states.distill.run_distiller")
def test_delegate_consumes_retry_payload_and_keeps_hash(mock_distiller):
    mock_distiller.return_value = {"updates": []}
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._chunks = [_CHUNK]
    fsm._current_chunk_idx = 0
    retry = {"schema_version": 1, "batches": [{
        "inbox_file": "Inbox/test.md",
        "concepts": [{"name": "Beta", "inbox_excerpt": "beta facts"}],
    }]}
    fsm.context["chunk_0_retry_payload"] = retry
    fsm.context["chunk_0_hash"] = "hash-from-first-pass"
    fsm.context["chunk_0_steer_context"] = "## STEERING CORRECTION (attempt 1/2)\n..."
    fsm.state = InjectorState.DELEGATE
    fsm.step()

    assert mock_distiller.call_args.kwargs["payload"] == retry
    assert mock_distiller.call_args.kwargs["steer_context"].startswith("## STEERING CORRECTION")
    assert "chunk_0_retry_payload" not in fsm.context
    # knowledge-block checkpoint stays keyed to the original input hash
    assert fsm.context["chunk_0_hash"] == "hash-from-first-pass"
    assert fsm.state == InjectorState.SANITIZE
