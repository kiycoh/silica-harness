"""Expand capability — in-run retry for «snippet too short» rejections.

The gate (MIN_WRITE_SNIPPET_CHARS) rejects distiller write ops with a thin
body. Deterministic re-validation can never clear them (silica_deferred_retry
only re-validates), so without this worker every such op waits for a manual
rewrite. The expand worker re-prompts the LLM with the concept's inbox excerpt
(max 2 attempts), commits through the same gate, and cleans the deferred twin
on success — after 2 short attempts the op stays in the deferred store, final.
"""
from __future__ import annotations

from unittest.mock import patch

from silica.config import CONFIG
from silica.kernel.write.ops import OpType
from silica.kernel.workqueue import WorkItem
from silica.capabilities.expand import run_expand, MAX_EXPAND_ATTEMPTS

_GOOD_BODY = "La matrice diagonale ha $D_{i,j}=0$ per $i \\neq j$. " * 9
_SHORT_BODY = "troppo corto"


def _item(**ctx_overrides) -> WorkItem:
    context = {
        "op": {
            "op": "write",
            "heading": "Matrici diagonali",
            "path": "Corso/Matrici diagonali.md",
            "title": "Matrici diagonali",
            "snippet": "",
            "source_basename": "lez.md",
            "tags": ["matrici"],
            "related": ["[[Norma vettoriale]]"],
        },
        "excerpt": "[[Matrici diagonali]]: $$D_{i,j} = 0$$ per $i \\neq j$ …",
        "reason": "snippet too short (0 < 100 chars) — would write a placeholder note, deferred for retry",
        "hub": "Corso",
        "target_dir": "Corso",
        "inbox_file": "Inbox/lez.md",
        "content_hash": "hash9",
    }
    context.update(ctx_overrides)
    return WorkItem(kind="expand", target_path="Corso/Matrici diagonali.md", context=context)


def test_expand_commits_authored_body_with_op_metadata():
    with patch("silica.capabilities.expand._author_body", return_value=_GOOD_BODY) as author, \
         patch("silica.capabilities.expand.commit_ops",
               return_value={"status": "committed", "committed": 1}) as commit, \
         patch("silica.kernel.recall.deferred.get_deferred_store") as store:
        res = run_expand(_item(), CONFIG)

    assert res["status"] == "committed"
    assert author.call_count == 1
    ops_arg = commit.call_args.args[0]
    assert len(ops_arg) == 1
    op = ops_arg[0]
    assert op.op == OpType.write
    assert op.path == "Corso/Matrici diagonali.md"
    assert op.snippet == _GOOD_BODY.strip()
    # The distiller's metadata survives — only the body was missing.
    assert op.tags == ["matrici"]
    assert op.related == ["[[Norma vettoriale]]"]
    assert op.title == "Matrici diagonali"
    bounds = commit.call_args.kwargs["bounds"]
    assert bounds.name == "expand"
    assert bounds.allowed_ops == frozenset({OpType.write})
    # Verified commit → the parked twin leaves the deferred bundle.
    store.return_value.remove_op.assert_called_once_with("hash9", "Matrici diagonali")


def test_expand_retries_once_then_commits():
    with patch("silica.capabilities.expand._author_body",
               side_effect=[_SHORT_BODY, _GOOD_BODY]) as author, \
         patch("silica.capabilities.expand.commit_ops",
               return_value={"status": "committed", "committed": 1}), \
         patch("silica.kernel.recall.deferred.get_deferred_store"):
        res = run_expand(_item(), CONFIG)

    assert res["status"] == "committed"
    assert author.call_count == 2
    # The second prompt carries corrective feedback about the short body.
    assert "too short" in (author.call_args.kwargs.get("feedback") or "")


def test_expand_gives_up_after_max_attempts():
    with patch("silica.capabilities.expand._author_body", return_value=_SHORT_BODY) as author, \
         patch("silica.capabilities.expand.commit_ops") as commit, \
         patch("silica.kernel.recall.deferred.get_deferred_store") as store:
        res = run_expand(_item(), CONFIG)

    assert res["status"] == "still_short"
    assert author.call_count == MAX_EXPAND_ATTEMPTS == 2
    commit.assert_not_called()
    store.return_value.remove_op.assert_not_called()


def test_expand_skips_without_excerpt():
    with patch("silica.capabilities.expand._author_body") as author:
        res = run_expand(_item(excerpt="   "), CONFIG)
    assert res["status"] == "skipped"
    author.assert_not_called()


def test_expand_failed_commit_keeps_deferred_twin():
    with patch("silica.capabilities.expand._author_body", return_value=_GOOD_BODY), \
         patch("silica.capabilities.expand.commit_ops",
               return_value={"status": "rolled_back", "committed": 0}), \
         patch("silica.kernel.recall.deferred.get_deferred_store") as store:
        res = run_expand(_item(), CONFIG)

    assert res["status"] == "rolled_back"
    store.return_value.remove_op.assert_not_called()


def test_expand_is_registered_capability():
    from silica.capabilities import CAPABILITIES
    assert CAPABILITIES["expand"] is run_expand


# ---------------------------------------------------------------------------
# VALIDATE → expand enqueue (partial rejections only; full rejections steer)
# ---------------------------------------------------------------------------

def _rejected_short(heading: str = "Matrici diagonali") -> dict:
    return {
        "op": {"op": "write", "heading": heading, "path": f"Corso/{heading}.md",
               "snippet": "", "source_basename": "lez.md"},
        "reason": "snippet too short (0 < 100 chars) — would write a placeholder note, deferred for retry",
    }


class _FSM:
    hub = "Corso"
    target_dir = "Corso"
    inbox_file = "Inbox/lez.md"
    _current_source_file = "Inbox/lez.md"
    _current_content_hash = "hash9"
    _current_chunk_idx = 0
    progress = None

    def __init__(self):
        from silica.kernel.workqueue import WorkQueue
        self.work_queue = WorkQueue()
        self._chunks = [{
            "batches": [{
                "inbox_file": "Inbox/lez.md",
                "concepts": [{
                    "name": "Matrici diagonali",
                    "inbox_excerpt": "[[Matrici diagonali]]: $$D_{i,j}=0$$",
                }],
            }],
        }]


def test_short_snippet_rejection_enqueues_expand_workitem():
    from silica.router.states.distill import _enqueue_short_snippet_expands

    fsm = _FSM()
    rejected = [
        _rejected_short(),
        {"op": {"op": "write", "heading": "X", "path": "Corso/X.md"},
         "reason": "Heading 'X' not present in payload concepts"},
    ]
    _enqueue_short_snippet_expands(fsm, rejected)

    items = fsm.work_queue.items()
    assert len(items) == 1, "only short-snippet rejections become expand work"
    it = items[0]
    assert it.kind == "expand"
    assert it.target_path == "Corso/Matrici diagonali.md"
    assert it.context["op"]["heading"] == "Matrici diagonali"
    assert "D_{i,j}" in it.context["excerpt"]
    assert it.context["content_hash"] == "hash9"
    assert it.context["target_dir"] == "Corso"
    assert it.context["hub"] == "Corso"


def test_enqueue_expand_noop_without_queue():
    from silica.router.states.distill import _enqueue_short_snippet_expands

    fsm = _FSM()
    fsm.work_queue = None
    _enqueue_short_snippet_expands(fsm, [_rejected_short()])  # must not raise
