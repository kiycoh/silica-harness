# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""FSM bookkeeping guards: checkpoint ids and per-file defer attribution.

Both are silent-corruption bugs: nothing raises when a checkpoint is filed
under an id nobody reads, or when a file's deferred work is attributed to the
previous file — the run just does the wrong thing on the NEXT run.
"""
from unittest.mock import patch


from silica.router.orchestrator import InjectorFSM, InjectorState
from silica.router.states import distill as d
from silica.router.states import setup as s


# ----------------------------------------------------------------------
# VALIDATE checkpoint id  ⟷  DELEGATE resume check
# ----------------------------------------------------------------------

_OP = {"type": "write", "path": "Concepts/Alpha.md", "heading": "Alpha",
       "content": "Alpha is a thing."}

_VALIDATE_OK = {
    "success": True,
    "validated_count": 1,
    "rejected_count": 0,
    "validated_ops": [_OP],
    "rejected_ops": [],
    "rejection_rate": 0.0,
}


def _validated_fsm(chunk_hash: str = "deadbeef") -> InjectorFSM:
    """An FSM parked on the second file's first chunk, ready for VALIDATE."""
    fsm = InjectorFSM(inbox_files=["Inbox/a.md", "Inbox/b.md"], target_dir="Concepts")
    fsm._chunks = [{"batches": []}, {"batches": []}]
    fsm._chunk_flat_to_fi_ci = {0: (0, 0), 1: (1, 0)}
    fsm._current_chunk_idx = 1
    fsm._current_file_idx = 1
    fsm._file_chunks = {
        0: {"source_file": "Inbox/a.md", "chunks": [fsm._chunks[0]]},
        1: {"source_file": "Inbox/b.md", "chunks": [fsm._chunks[1]]},
    }
    fsm.context["chunk_1_hash"] = chunk_hash
    fsm.context["chunk_1_input_hash"] = chunk_hash
    fsm._chunk_ctx["sanitized"] = {"parsed": {"updates": [_OP]}}
    fsm.state = InjectorState.VALIDATE
    return fsm


def test_validate_checkpoint_is_filed_under_the_id_the_readers_use():
    fsm = _validated_fsm()

    with patch.object(d.orch, "silica_validate_ops", return_value=_VALIDATE_OK):
        d.handle_validate(fsm)

    task_id = fsm._chunk_task_id("validate")
    assert task_id == "f1_c0_validate"
    assert fsm.progress.is_checkpoint_done(task_id, "deadbeef") == fsm._chunk_ctx["ops_path"]

    # The legacy "chunk_<flat idx>_validate" id must not be created at all: it
    # both hid the checkpoint and left the real task running forever.
    ids = [t.id for t in fsm.progress.tasks]
    assert "chunk_1_validate" not in ids
    assert [t.status for t in fsm.progress.tasks if t.id == task_id] == ["done"]


def test_resume_skips_the_distiller_for_an_unchanged_chunk():
    """The checkpoint VALIDATE writes is what DELEGATE reads to skip a chunk."""
    fsm = _validated_fsm()

    with patch.object(d.orch, "silica_validate_ops", return_value=_VALIDATE_OK):
        d.handle_validate(fsm)

    # A resume re-enters DELEGATE for the same chunk with the same input hash.
    fsm.state = InjectorState.DELEGATE
    fsm._chunk_ctx.pop("ops_path", None)
    with patch.object(d, "run_distiller",
                      side_effect=AssertionError("distiller re-ran a checkpointed chunk")):
        d.handle_delegate(fsm)

    assert fsm.state == InjectorState.SNAPSHOT
    assert fsm._chunk_ctx["ops_path"]


# ----------------------------------------------------------------------
# Pre-chunk defers are attributed to the file being assembled
# ----------------------------------------------------------------------

def _payload_fsm() -> InjectorFSM:
    """An FSM entering PAYLOAD for file 1 with file 0's chunk still under the cursor."""
    fsm = InjectorFSM(inbox_files=["Inbox/a.md", "Inbox/b.md"], target_dir="Concepts")
    fsm._file_content_hashes = ["hash-of-a", "hash-of-b"]
    fsm._chunks = [{"batches": []}]
    fsm._chunk_flat_to_fi_ci = {0: (0, 0)}
    fsm._file_chunks = {0: {"source_file": "Inbox/a.md", "chunks": [fsm._chunks[0]]}}
    fsm._current_chunk_idx = 0          # still file 0's chunk
    fsm._current_file_idx = 1
    fsm.context["recon"] = [{"concepts": []}]
    fsm.context["payload"] = {"chunks": [{"batches": [{"inbox_file": "Inbox/a.md"}]}]}
    fsm.context["vault_graph_ctx"] = {}
    fsm.state = InjectorState.PAYLOAD
    return fsm


def test_second_file_diversions_are_deferred_under_their_own_file():
    """The novelty gate defers before the chunk cursor moves onto its own file.

    _defer_ops derives content_hash and source_path from the cursor, so file 2's
    diverted concepts used to land in file 1's bundle.
    """
    from silica.kernel.recall.deferred import get_deferred_store

    fsm = _payload_fsm()
    seen: dict[str, str] = {}

    def _gate(gate_fsm, raw_payload, *a, **kw):
        # Exactly the values the real gate reads for its defer + WorkItem.
        seen["content_hash"] = gate_fsm._current_content_hash
        seen["source_file"] = gate_fsm._current_source_file
        gate_fsm._defer_ops(
            [{"op": "write", "heading": "Dup", "path": "Concepts/Dup.md",
              "snippet": "x", "hub": gate_fsm.hub}],
            {"Dup": "novelty_gate score=0.960"},
            phase="NOVELTY",
        )
        return {"schema_version": 1, "batches": []}, 1

    res = {"chunks": [{"schema_version": 1, "batches": [
        {"inbox_file": "Inbox/b.md", "concepts": [{"name": "Dup", "excerpt": "x"}]}]}]}
    with patch.object(s.orch, "silica_payload", return_value=res), \
         patch.object(s, "novelty_gate", _gate):
        s.handle_payload(fsm)

    assert seen == {"content_hash": "hash-of-b", "source_file": "Inbox/b.md"}

    store = get_deferred_store()
    assert store.get("hash-of-a") is None
    bundle = store.get("hash-of-b")
    assert bundle is not None
    assert bundle["source_path"] == "Inbox/b.md"
    # The previous file's payload must not ride along as this file's grounding.
    assert all(b.get("inbox_file") != "Inbox/a.md"
               for p in bundle.get("payloads", []) for b in p.get("batches", []))


def test_payload_reassembles_a_file_whose_chunks_never_attached():
    """`_file_chunks[fi]` is claimed before assembly, so the warm fast-path must
    key on the chunk list, not on the key's presence — otherwise a file that
    failed mid-assembly would fast-path with zero chunks on the next attempt."""
    fsm = _payload_fsm()
    fsm._file_chunks[1] = {"source_file": "Inbox/b.md", "chunks": []}

    res = {"chunks": [{"schema_version": 1, "batches": [
        {"inbox_file": "Inbox/b.md", "concepts": [{"name": "Beta", "excerpt": "y"}]}]}]}
    with patch.object(s.orch, "silica_payload", return_value=res):
        s.handle_payload(fsm)

    assert fsm._file_chunks[1]["chunks"], "PAYLOAD fast-pathed on an empty chunk list"
    assert fsm._current_chunk_idx == 1
    assert fsm._chunk_flat_to_fi_ci[1] == (1, 0)


def test_warm_next_file_reprepares_a_file_whose_chunks_never_attached():
    fsm = _payload_fsm()
    fsm._file_chunks[1] = {"source_file": "Inbox/b.md", "chunks": []}
    fsm._current_file_idx = 0

    with patch.object(s.orch, "silica_recon", return_value={"error": "boom"}) as recon:
        assert s.warm_next_file(fsm) is False
    assert recon.called, "warm stood down on a file it had only claimed, not attached"
