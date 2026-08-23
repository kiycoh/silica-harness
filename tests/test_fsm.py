import pytest
from unittest.mock import patch, MagicMock, call
from silica.router import states
from silica.router.orchestrator import InjectorFSM, InjectorState
from silica.router.recipe_parser import load_recipe
from silica.tools import TOOLS


@pytest.fixture(autouse=True)
def _historical_snippet_floor(monkeypatch):
    # Predates the 100→400 write-floor raise; short fixtures here exercise
    # routing/coercion, not the length gate — pin their original floor.
    monkeypatch.setenv("SILICA_MIN_WRITE_SNIPPET_CHARS", "100")


# Loaded before any test patches builtins.open: tests that mock open() would
# otherwise break the (fail-fast) recipe load inside InjectorFSM.__init__.
_RECIPE = load_recipe("injector")

def test_injector_fsm_initialization():
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    assert fsm.state.name == "INIT"


def test_injector_fsm_hub_inheritance():
    # If hub is None and target_dir is provided, it should inherit from target_dir's basename
    fsm = InjectorFSM("Inbox/test.md", "Deep Learning/Concepts")
    assert fsm.hub == "Concepts"

    # If hub is provided explicitly, it should preserve it
    fsm_explicit = InjectorFSM("Inbox/test.md", "Deep Learning/Concepts", hub="MyExplicitHub")
    assert fsm_explicit.hub == "MyExplicitHub"


def test_validate_operations_hub_inheritance():
    from silica.kernel.write.validate import validate_operations
    ops = [
        {"op": "write", "path": "Deep Learning/Concepts/Neural Network.md", "heading": "Neural Network", "source_basename": "inbox.md", "snippet": "corpo " * 20}
    ]
    # We patch path_exists to return False so that the write operation is validated as a creation
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=RuntimeError("File not found")):
        validated, rejected = validate_operations(ops, [], "Deep Learning/Concepts")
    assert len(rejected) == 0
    # Returns 2: 1 for the auto-generated Hub note and 1 for the Neural Network spoke note
    assert len(validated) == 2
    assert validated[0]["heading"] == "Concepts"
    assert validated[0]["op"] == "write"
    assert validated[1]["heading"] == "Neural Network"
    assert validated[1]["hub"] == "Concepts"


def test_validate_operations_hub_coercion():
    from silica.kernel.write.validate import validate_operations
    ops = [
        {"op": "write", "path": "Deep Learning/Concepts/Neural Network.md", "heading": "Neural Network", "hub": "import_from_inbox", "source_basename": "inbox.md", "snippet": "corpo " * 20}
    ]
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=RuntimeError("File not found")):
        validated, rejected = validate_operations(ops, [], "Deep Learning/Concepts", hub="CustomConceptsHub")
    assert len(rejected) == 0
    assert len(validated) == 2
    assert validated[0]["heading"] == "CustomConceptsHub"
    assert validated[0]["hub"] == "CustomConceptsHub"
    assert validated[1]["heading"] == "Neural Network"
    assert validated[1]["hub"] == "CustomConceptsHub"


def test_validate_operations_auto_creates_missing_hub():
    from silica.kernel.write.validate import validate_operations

    # Case A: Hub note doesn't exist anywhere in the vault
    ops_missing = [
        {"op": "write", "path": "Deep Learning/Concepts/Neural Network.md", "heading": "Neural Network", "hub": "Concepts", "source_basename": "inbox.md", "snippet": "corpo " * 20}
    ]
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=RuntimeError("File not found")):
        validated, _ = validate_operations(ops_missing, [], "Deep Learning/Concepts")
    assert len(validated) == 2
    assert validated[0]["heading"] == "Concepts"
    assert validated[0]["path"] == "Deep Learning/Concepts/Concepts.md"

    # Case B: Hub note already exists in the vault
    ops_exists = [
        {"op": "write", "path": "Deep Learning/Concepts/Neural Network.md", "heading": "Neural Network", "hub": "Concepts", "source_basename": "inbox.md", "snippet": "corpo " * 20}
    ]
    # Here read_note succeeds for the hub "Concepts" but fails for the spoke note
    def mock_read_note(ref):
        if ref == "Concepts":
            return MagicMock()
        raise RuntimeError("File not found")

    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=mock_read_note):
        validated, _ = validate_operations(ops_exists, [], "Deep Learning/Concepts")
    # Only 1 because Hub note already exists
    assert len(validated) == 1
    assert validated[0]["heading"] == "Neural Network"

    # Case C: Hub note is already being created by another write operation in the list
    ops_already_creating = [
        {"op": "write", "path": "Deep Learning/Concepts/Concepts.md", "heading": "Concepts", "hub": "Concepts", "source_basename": "inbox.md", "snippet": "corpo " * 20},
        {"op": "write", "path": "Deep Learning/Concepts/Neural Network.md", "heading": "Neural Network", "hub": "Concepts", "source_basename": "inbox.md", "snippet": "corpo " * 20}
    ]
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=RuntimeError("File not found")):
        validated, _ = validate_operations(ops_already_creating, [], "Deep Learning/Concepts")
    # 2 because the explicit creation is preserved, and no duplicate is injected
    assert len(validated) == 2




def test_silica_run_injector_is_registered():
    assert "silica_run_injector" in TOOLS
    tool = TOOLS["silica_run_injector"]
    assert tool.cls == "composed"

# Pin k=1: this asserts the inline single-chunk delegate contract, which is the
# k=1 path. At the default k=3 the prefetcher fires an extra lookahead call and
# call_args (last call) would no longer be this chunk's.
@patch("silica.router.states.distill.orch.CONFIG.distill_concurrency", 1)
@patch("silica.router.states.distill.run_distiller")
def test_fsm_delegate_single_chunk(mock_run_distiller):
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._chunks = [
        {"chunk_id": 0, "concepts": ["a"]},
        {"chunk_id": 1, "concepts": ["b"]}
    ]
    fsm._current_chunk_idx = 0
    fsm.state = InjectorState.DELEGATE

    mock_run_distiller.return_value = {"updates": [{"op": "write", "path": "notes/note1.md", "heading": "Note 1"}]}

    with patch.object(fsm, "_make_tmp", return_value="temp_chunk_path.json") as mock_make_tmp:
        fsm.step()
        
        call_kwargs = mock_run_distiller.call_args.kwargs
        assert call_kwargs["payload"] == {"chunk_id": 0, "concepts": ["a"]}
        assert call_kwargs["target"] == "TargetDir"
        assert call_kwargs["hub"] == "TargetDir"
        # ledger_digest is injected by Phase 2 — just assert it's a string or None
        assert isinstance(call_kwargs.get("ledger_digest"), (str, type(None)))
        # F2a: live ingest resolves relative dates against the run's start date.
        assert call_kwargs["session_date"] == fsm.progress.started_at[:10]
        mock_make_tmp.assert_called_once_with({"updates": [{"op": "write", "path": "notes/note1.md", "heading": "Note 1"}]})
        assert fsm.state == InjectorState.SANITIZE
        assert fsm.context["chunk"]["distiller_output_path"] == "temp_chunk_path.json"


# Pin k=1: asserts the inline delegate's session_date + a single read_note; at
# the default k=3 the prefetch snapshot reads the doc a second time.
@patch("silica.router.states.distill.orch.CONFIG.distill_concurrency", 1)
@patch("silica.router.states.distill.run_distiller")
def test_fsm_delegate_dated_doc_anchors_session_date(mock_run_distiller):
    """A source doc with frontmatter `date:` anchors the distiller's
    relative-date resolution to the doc's own day, not the ingest day."""
    from silica.driver.base import NoteContent, NoteRef

    fsm = InjectorFSM("Inbox/journal.md", "TargetDir")
    fsm._chunks = [{"chunk_id": 0, "concepts": ["a"]}]
    fsm._file_chunks = {0: {"source_file": "Inbox/journal.md", "chunks": fsm._chunks}}
    fsm._chunk_flat_to_fi_ci = {0: (0, 0)}
    fsm._current_chunk_idx = 0
    fsm.state = InjectorState.DELEGATE

    mock_run_distiller.return_value = {"updates": [{"op": "write", "path": "notes/n.md", "heading": "N"}]}
    doc = NoteContent(
        ref=NoteRef(name="journal", path="Inbox/journal.md"),
        content=(
            "---\ndate: 2024-03-12\ntags: [journal]\n---\n"
            "Yesterday we froze the corpus.\n"
            "date: 9999-01-01 in the body must not match\n"
        ),
    )
    with patch.object(fsm, "_make_tmp", return_value="tmp.json"), \
         patch("silica.router.orchestrator.DRIVER.read_note", return_value=doc) as mock_read:
        fsm.step()

    mock_read.assert_called_once_with("Inbox/journal.md")
    assert mock_run_distiller.call_args.kwargs["session_date"] == "2024-03-12"


@patch("silica.router.orchestrator.silica_recon")
@patch("silica.router.orchestrator.silica_payload")
@patch("silica.router.states.distill.run_distiller")
@patch("silica.router.orchestrator.silica_sanitize")
@patch("silica.router.orchestrator.silica_validate_ops")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.tools.wrapped.silica_snapshot")
@patch("silica.router.orchestrator.silica_lint")
@patch("silica.tools.wrapped.silica_cleanup")
@patch("silica.kernel.recall.embed.EmbedStore")
def test_fsm_multi_chunk_loop(
    mock_embed_store, mock_cleanup, mock_lint, mock_snapshot, mock_driver,
    mock_validate, mock_sanitize, mock_run_distiller, mock_payload, mock_recon
):
    # Setup mock payload with 2 chunks
    mock_embed_store.return_value.__len__ = lambda _: 0  # empty index → COLLISION skips early
    mock_recon.return_value = {"success": True}
    mock_payload.return_value = {
        "chunks": [
            {"chunk_id": 0, "concepts": ["a"]},
            {"chunk_id": 1, "concepts": ["b"]}
        ]
    }
    mock_run_distiller.return_value = {"updates": [{"op": "write", "path": "notes/NoteA.md"}]}
    mock_sanitize.return_value = {"parsed": []}
    mock_validate.return_value = {"success": True, "rejection_rate": 0.0, "validated_count": 1, "rejected_count": 0}
    mock_snapshot.return_value = {"txn_id": "txn_123", "inverses": []}
    mock_lint.return_value = {"success": True}
    mock_cleanup.return_value = {"success": True}

    # Setup graph mocks
    pre_graph = MagicMock()
    post_graph = MagicMock()
    mock_driver.graph_snapshot.return_value = post_graph

    # Initialize FSM
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    
    # Track states visited
    states_visited = []
    original_step = fsm.step
    def step_wrapper():
        states_visited.append(fsm.state)
        original_step()
    fsm.step = step_wrapper

    with patch("silica.kernel.graph_diff.check_graph_regression", return_value=(True, [])):
        res = fsm.run()

    # Check that it reached DONE state
    assert fsm.state == InjectorState.DONE
    assert res.get("final_status") == "Success"

    # Verify that the sequence of states visited loops back to DELEGATE
    expected_sequence = [
        # First chunk cycle
        InjectorState.RECON,
        InjectorState.PAYLOAD,
        InjectorState.SALIENCE,   # Phase 2.05 (best-effort, embedder unavailable → skip)
        InjectorState.COLLISION,  # Phase 5 (best-effort, no index → skip)
        InjectorState.DELEGATE,
        InjectorState.SANITIZE,
        InjectorState.VALIDATE,
        InjectorState.SNAPSHOT,
        InjectorState.WRITE,
        InjectorState.HUB_UPDATE,
        InjectorState.AUTOLINK,   # Phase 4
        InjectorState.BACKLINK,   # Phase 4.5 (best-effort)
        InjectorState.LINT,
        InjectorState.CLEANUP,
        # Second chunk cycle (SALIENCE does not re-run)
        InjectorState.COLLISION,  # Phase 5 (best-effort, no index → skip)
        InjectorState.DELEGATE,
        InjectorState.SANITIZE,
        InjectorState.VALIDATE,
        InjectorState.SNAPSHOT,
        InjectorState.WRITE,
        InjectorState.HUB_UPDATE,
        InjectorState.AUTOLINK,   # Phase 4
        InjectorState.BACKLINK,   # Phase 4.5 (best-effort)
        InjectorState.LINT,
        InjectorState.CLEANUP,
    ]
    assert states_visited == expected_sequence

    # Verify run_distiller was called twice, once for each chunk
    assert mock_run_distiller.call_count == 2
    # Phase 2: run_distiller now receives ledger_digest kwarg — check payloads only
    payloads_seen = [c.kwargs["payload"] for c in mock_run_distiller.call_args_list]
    assert any({"chunk_id": 0, "concepts": ["a"]}.items() <= p.items() for p in payloads_seen)
    assert any({"chunk_id": 1, "concepts": ["b"]}.items() <= p.items() for p in payloads_seen)

    # Verify cleanup (file move) was only called once (on the final chunk completion)
    mock_cleanup.assert_called_once_with("Inbox/test.md")


def test_fsm_recipe_configuration():
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    assert fsm._recipe is not None
    assert fsm._recipe["name"] == "injector"
    
    # Check gates configuration
    assert fsm._get_recipe_gate("rejection_rate_max", 0.05) == 0.10
    assert fsm._get_recipe_gate("graph_regression", "allow") == "forbid_new_orphans"

    # Check phases configuration
    payload_conf = fsm._get_recipe_phase("payload")
    assert payload_conf.get("partition_if_over") == 7


@patch("silica.router.orchestrator.silica_validate_ops")
def test_fsm_gate_all_rejected_steers_then_defers(mock_validate):
    # Phase 6: when ALL ops are rejected, VALIDATE steers back to DELEGATE (max 2
    # attempts). After the budget is exhausted, it goes to CLEANUP with "no_ops".
    all_rejected = {
        "success": True,
        "rejection_rate": 1.0,
        "total": 2,
        "validated_count": 0,
        "rejected_count": 2,
        "rejected_ops": [{"reason": "bad path"}],
        "validated_ops": [],
    }
    mock_validate.return_value = all_rejected

    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._chunks = [{"schema_version": 1, "batches": []}]
    fsm._current_chunk_idx = 0
    # Pin the historical two in-flight retries; the default is now 1 (the second
    # recovery moved to the boundary anneal) — covered separately below.
    fsm._recipe = {"gates": {"max_steer_attempts": 2}}
    fsm.context.setdefault("chunk", {})["sanitized"] = {"parsed": []}
    fsm.state = InjectorState.VALIDATE

    # First call: steer attempt 1 → go to DELEGATE
    fsm.step()
    assert fsm.state == InjectorState.DELEGATE
    assert fsm.context.get("chunk_0_steer_attempts") == 1
    assert fsm.context.get("chunk_0_steer_context") is not None

    # Put FSM back at VALIDATE and call again (simulating steer attempt 2)
    fsm.context.setdefault("chunk", {})["sanitized"] = {"parsed": []}
    fsm.state = InjectorState.VALIDATE
    fsm.step()
    assert fsm.state == InjectorState.DELEGATE
    assert fsm.context.get("chunk_0_steer_attempts") == 2

    # Third call: budget exhausted → CLEANUP with "no_ops"
    fsm.context.setdefault("chunk", {})["sanitized"] = {"parsed": []}
    fsm.state = InjectorState.VALIDATE
    fsm.step()
    assert fsm.state == InjectorState.CLEANUP
    assert fsm.context.get("final_status") == "no_ops"


@patch("silica.router.orchestrator.silica_validate_ops")
def test_fsm_default_steer_is_one_then_defers(mock_validate):
    # Default max_steer_attempts is now 1: one in-flight steer, then the ops are
    # left for the boundary anneal instead of a second in-flight re-delegation.
    mock_validate.return_value = {
        "success": True, "rejection_rate": 1.0, "total": 1,
        "validated_count": 0, "rejected_count": 1,
        "rejected_ops": [{"reason": "bad path"}], "validated_ops": [],
    }
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._chunks = [{"schema_version": 1, "batches": []}]
    fsm._current_chunk_idx = 0
    fsm.context.setdefault("chunk", {})["sanitized"] = {"parsed": []}
    fsm.state = InjectorState.VALIDATE

    fsm.step()  # attempt 1 → DELEGATE
    assert fsm.state == InjectorState.DELEGATE
    assert fsm.context.get("chunk_0_steer_attempts") == 1

    fsm.context.setdefault("chunk", {})["sanitized"] = {"parsed": []}
    fsm.state = InjectorState.VALIDATE
    fsm.step()  # budget (1) exhausted → CLEANUP
    assert fsm.state == InjectorState.CLEANUP
    assert fsm.context.get("final_status") == "no_ops"


@patch("silica.router.orchestrator.silica_validate_ops")
def test_fsm_gate_partial_rejection_continues(mock_validate):
    # When some ops pass and some are rejected, VALIDATE must continue the
    # pipeline (SNAPSHOT) and log a warning — it does NOT abort.
    mock_validate.return_value = {
        "success": True,
        "rejection_rate": 0.44,   # > 10% but partial
        "total": 9,
        "validated_count": 5,
        "rejected_count": 4,
        "rejected_ops": [],       # empty so deferred store is not called
        "validated_ops": [],
    }

    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.context["payload"] = {"payload": {"chunk_id": 0}}
    fsm.context.setdefault("chunk", {})["sanitized"] = {"parsed": []}
    fsm.context["source_content_hash"] = ""  # no deferred store call
    fsm.state = InjectorState.VALIDATE

    fsm.step()

    # Pipeline advances past VALIDATE (to SNAPSHOT)
    assert fsm.state == InjectorState.SNAPSHOT
    assert "abort_reason" not in fsm.context.get("chunk", {})


@patch("silica.router.orchestrator.silica_lint")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.tools.wrapped.silica_restore")
@patch("silica.router.recipe_parser.load_recipe", new=lambda *a, **k: _RECIPE)
@patch("builtins.open")
def test_fsm_graph_regression_orphan_is_warning(mock_open, mock_restore, mock_driver, mock_lint):
    """Orphan-only regression errors must emit a WARNING and not trigger ROLLBACK."""
    mock_open.return_value.__enter__.return_value.read.return_value = '[]'
    mock_lint.return_value = {"success": True}

    pre_graph = MagicMock()
    post_graph = MagicMock()
    mock_driver.graph_snapshot.return_value = post_graph

    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.context.setdefault("chunk", {})["ops_path"] = "dummy_ops.json"
    fsm._pre_graph = pre_graph
    fsm._txn = MagicMock()
    fsm._txn.created_paths = ["notes/NoteA.md"]
    fsm.context.setdefault("chunk", {})["snapshot"] = {
        "txn_id": "txn_123",
        "inverses": [],
    }

    fsm.state = InjectorState.LINT

    orphan_error = "Unplanned orphans introduced: done/lezione_1.md"
    with patch("silica.kernel.graph_diff.check_graph_regression", return_value=(False, [orphan_error])) as mock_check:
        fsm.step()
        mock_check.assert_called_once_with(pre_graph, post_graph, ["notes/NoteA.md"], frozenset(), frozenset())
        # Orphan-only: pipeline must NOT roll back
        assert fsm.state != InjectorState.ROLLBACK, "Orphan-only errors must not trigger ROLLBACK"


@patch("silica.router.orchestrator.silica_lint")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.tools.wrapped.silica_restore")
@patch("silica.router.recipe_parser.load_recipe", new=lambda *a, **k: _RECIPE)
@patch("builtins.open")
def test_fsm_graph_regression_gate_rollback(mock_open, mock_restore, mock_driver, mock_lint):
    """Blocking regression errors (broken backlinks, unresolved links) still trigger ROLLBACK."""
    mock_open.return_value.__enter__.return_value.read.return_value = '[]'
    mock_lint.return_value = {"success": True}

    pre_graph = MagicMock()
    post_graph = MagicMock()
    mock_driver.graph_snapshot.return_value = post_graph

    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.context.setdefault("chunk", {})["ops_path"] = "dummy_ops.json"
    fsm._pre_graph = pre_graph

    # self._txn is the single source of truth for rollback inverses (C3).
    inverses = [{"op": "delete", "path": "notes/NoteA.md"}]
    from silica.driver.base import Txn
    fsm._txn = Txn(id="txn_123", created_paths=["notes/NoteA.md"], inverses=inverses)
    fsm.context.setdefault("chunk", {})["snapshot"] = {
        "txn_id": "txn_123",
        "inverses": inverses,
    }

    fsm.state = InjectorState.LINT

    blocking_error = "Broken backlinks detected for 'NoteA': decreased from 2 to 0"
    with patch("silica.kernel.graph_diff.check_graph_regression", return_value=(False, [blocking_error])) as mock_check:
        fsm.step()
        # patched_paths carries every txn-touched path (post-WRITE phases
        # included), not only op-level patches — see txn_touched_paths.
        mock_check.assert_called_once_with(pre_graph, post_graph, ["notes/NoteA.md"],
                                           frozenset(), frozenset({"notes/NoteA.md"}))
        assert fsm.state == InjectorState.ROLLBACK
        assert "Graph regression gate failed: Broken backlinks" in fsm.context["chunk"]["abort_reason"]

    mock_restore.return_value = {"success": True}
    fsm.step()
    mock_restore.assert_called_once_with(txn_id="txn_123", inverses=inverses)

    assert fsm.state == InjectorState.DONE
    assert fsm.context["final_status"] == "failed"  # zero commits: honest verdict, not "partial"


def _lint_gate_fsm(op_path: str):
    """LINT-state FSM with one committed patch op on op_path (graph gate green)."""
    from silica.kernel.write.ops import Op, OpType
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.context.setdefault("chunk", {})["ops_path"] = "dummy_ops.json"
    fsm._pre_graph = MagicMock()
    fsm._txn = MagicMock()
    fsm._txn.created_paths = []
    fsm.state = InjectorState.LINT
    op = Op(op=OpType.patch, path=op_path, heading="H",
            source_basename="s.md", snippet="x", hub="Hub")
    return fsm, [op]


@patch("silica.router.orchestrator.silica_lint")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.router.recipe_parser.load_recipe", new=lambda *a, **k: _RECIPE)
def test_fsm_lint_gate_tolerates_per_op_baseline(mock_driver, mock_lint):
    """The gate must apply the same diff-aware tolerance as per-op lint
    (commit_note_atomic baseline): a pre-existing violation in a patched user
    note is not this chunk's regression. Run 880b9aa9: f1_c0/f1_c1 aborted —
    discarding their committed ops — on an inherited "frontmatter 'AI'
    missing" the WRITE phase had correctly waved through."""
    inherited = "frontmatter 'AI' missing or not boolean"
    mock_lint.return_value = {"success": False, "errors": [inherited]}
    mock_driver.graph_snapshot.return_value = MagicMock()

    fsm, ops = _lint_gate_fsm("Topics/Legacy.md")
    fsm.context["chunk"]["lint_baseline"] = {"Topics/Legacy.md": [inherited]}
    with patch("silica.router.orchestrator.load_ops", return_value=ops), \
         patch("silica.kernel.graph_diff.check_graph_regression", return_value=(True, [])):
        fsm.step()

    assert fsm.state != InjectorState.ROLLBACK


@patch("silica.router.orchestrator.silica_lint")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.router.recipe_parser.load_recipe", new=lambda *a, **k: _RECIPE)
def test_fsm_lint_gate_still_blocks_new_violations(mock_driver, mock_lint):
    """Control for the baseline tolerance: a violation NOT in the baseline
    (introduced by this chunk, e.g. by autolink/hub mutations after per-op
    lint) must still abort to ROLLBACK."""
    mock_lint.return_value = {"success": False, "errors": ["unknown callout type [!x]"]}
    mock_driver.graph_snapshot.return_value = MagicMock()

    fsm, ops = _lint_gate_fsm("Topics/Legacy.md")
    fsm.context["chunk"]["lint_baseline"] = {
        "Topics/Legacy.md": ["frontmatter 'AI' missing or not boolean"]
    }
    with patch("silica.router.orchestrator.load_ops", return_value=ops):
        fsm.step()

    assert fsm.state == InjectorState.ROLLBACK
    assert "unknown callout type" in fsm.context["chunk"]["abort_reason"]


@patch("silica.router.orchestrator.silica_recon")
@patch("silica.router.orchestrator.silica_payload")
@patch("silica.router.states.distill.run_distiller")
@patch("silica.router.orchestrator.silica_sanitize")
@patch("silica.router.orchestrator.silica_validate_ops")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.tools.wrapped.silica_snapshot")
@patch("silica.router.orchestrator.silica_lint")
@patch("silica.tools.wrapped.silica_cleanup")
def test_fsm_recipe_end_to_end_flow(
    mock_cleanup, mock_lint, mock_snapshot, mock_driver,
    mock_validate, mock_sanitize, mock_run_distiller, mock_payload, mock_recon
):
    # Setup mocks
    mock_recon.return_value = {"success": True}
    mock_payload.return_value = {"payload": {"chunk_id": 0}}
    mock_run_distiller.return_value = {"updates": []}
    mock_sanitize.return_value = {"parsed": []}
    mock_validate.return_value = {"success": True, "rejection_rate": 0.0, "validated_count": 1, "rejected_count": 0}
    mock_snapshot.return_value = {"txn_id": "txn_123", "inverses": []}
    mock_lint.return_value = {"success": True}
    mock_cleanup.return_value = {"success": True}

    # Setup graph mocks
    pre_graph = MagicMock()
    post_graph = MagicMock()
    mock_driver.graph_snapshot.return_value = post_graph

    # Initialize FSM (loads actual injector.yaml recipe)
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    
    # Track states visited
    states_visited = []
    original_step = fsm.step
    def step_wrapper():
        states_visited.append(fsm.state)
        original_step()
    fsm.step = step_wrapper

    with patch("silica.kernel.graph_diff.check_graph_regression", return_value=(True, [])):
        res = fsm.run()

    # Check that it reached DONE state
    assert fsm.state == InjectorState.DONE
    assert res.get("final_status") == "Success"

    # Verify the sequence of states visited exactly matches the injector.yaml phases
    expected_sequence = [
        InjectorState.RECON,
        InjectorState.PAYLOAD,
        InjectorState.SALIENCE,   # Phase 2.05 (best-effort, embedder unavailable → skip)
        InjectorState.COLLISION,  # Phase 5 (best-effort, empty index → skip)
        InjectorState.DELEGATE,
        InjectorState.SANITIZE,
        InjectorState.VALIDATE,
        InjectorState.SNAPSHOT,
        InjectorState.WRITE,
        InjectorState.HUB_UPDATE,
        InjectorState.AUTOLINK,   # Phase 4
        InjectorState.BACKLINK,   # Phase 4.5 (best-effort)
        InjectorState.LINT,
        InjectorState.CLEANUP,
    ]
    assert states_visited == expected_sequence


def test_fsm_already_nucleated():
    # We patch get_ledger to return a mock ledger where is_committed is True
    with patch("silica.kernel.write.ledger.get_ledger") as mock_get_ledger:
        mock_ledger = MagicMock()
        mock_ledger.is_committed.return_value = True
        mock_get_ledger.return_value = mock_ledger

        fsm = InjectorFSM("Inbox/already_processed.md", "TargetDir")
        res = fsm.run()

        # Should short-circuit pre-RECON
        assert fsm.state == InjectorState.INIT
        assert res.get("final_status") == "already_nucleated"
        # The mock is called with (source_canonical, content_hash=...) — canonical drops .md, lowercased
        call_args = mock_ledger.is_committed.call_args
        assert call_args[0][0] == "already_processed", (
            f"Expected canonical key 'already_processed', got {call_args[0][0]!r}"
        )
        # content_hash kwarg must be present (may be empty str if file not found)
        assert "content_hash" in call_args[1]


# ---------------------------------------------------------------------------
# Phase 5 — Collision routing tests
# ---------------------------------------------------------------------------

def _make_fsm_at_collision(chunk_concepts: list, tmp_path=None) -> InjectorFSM:
    """Helper: build an FSM positioned at COLLISION with given chunk."""
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._chunks = [{"schema_version": 1, "batches": [
        {"inbox_file": "Inbox/test.md", "concepts": chunk_concepts}
    ]}]
    fsm._current_chunk_idx = 0
    fsm.state = InjectorState.COLLISION
    fsm.context["source_content_hash"] = "abc123"
    return fsm


def test_collision_high_similarity_routes_to_patch():
    """score ≥ τ_high → pre-routed patch op stored in context, concept removed from chunk."""
    fsm = _make_fsm_at_collision([{"name": "Neural Networks", "inbox_excerpt": "A neural net intro."}])

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5  # non-empty index
    mock_store.cosine_top_k.return_value = [{"path": "DL/Neural Networks.md", "name": "Neural Networks", "score": 0.92}]

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.router.orchestrator.DRIVER") as mock_driver, \
         patch("silica.kernel.recall.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder):
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.65
        mock_driver.read_note.return_value = MagicMock()  # graph confirms node exists

        fsm.step()

    assert fsm.state == InjectorState.DELEGATE
    collision_ops = fsm.context.get("chunk_0_collision_ops", [])
    assert len(collision_ops) == 1
    assert collision_ops[0]["op"] == "patch"
    assert collision_ops[0]["path"] == "DL/Neural Networks.md"
    # Concept removed from modified chunk
    remaining = fsm._chunks[0].get("batches", [])
    assert remaining == [] or all(len(b.get("concepts", [])) == 0 for b in remaining)


def test_collision_low_similarity_keeps_for_distillation():
    """score ≤ τ_low → concept kept in chunk for normal distillation, no patch ops."""
    fsm = _make_fsm_at_collision([{"name": "Quantum Entanglement", "inbox_excerpt": "Something unrelated."}])

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5
    mock_store.cosine_top_k.return_value = [{"path": "Physics/Photons.md", "name": "Photons", "score": 0.40}]

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.kernel.recall.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder):
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.65

        fsm.step()

    assert fsm.state == InjectorState.DELEGATE
    collision_ops = fsm.context.get("chunk_0_collision_ops", [])
    assert collision_ops == []
    # Concept still in chunk
    concepts_left = [
        c
        for b in fsm._chunks[0].get("batches", [])
        for c in b.get("concepts", [])
    ]
    assert len(concepts_left) == 1
    assert concepts_left[0]["name"] == "Quantum Entanglement"


def test_collision_embeds_concept_with_excerpt():
    """Regression (Error 3): the COLLISION query must embed name+excerpt, not the
    bare acronym.

    Embedding "MEM" alone scores spuriously high against unrelated short-acronym
    notes (e.g. "RAM (Random Access Memory)") because the index stores rich
    title+body vectors; the bare-name query is incomparable. The excerpt — already
    in hand — anchors the concept in its real semantic neighbourhood.
    """
    fsm = _make_fsm_at_collision([
        {"name": "MEM", "inbox_excerpt": "Agentic memory mechanism for evolving context."}
    ])

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5
    mock_store.cosine_top_k.return_value = [
        {"path": "AI/RAM (Random Access Memory).md", "name": "RAM (Random Access Memory)", "score": 0.40}
    ]

    embedded_texts: list[str] = []

    def _capture(texts):
        embedded_texts.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    mock_embedder = MagicMock()
    mock_embedder.embed.side_effect = _capture

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.kernel.recall.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder):
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.65

        fsm.step()

    assert embedded_texts, "embedder was never called"
    assert any("Agentic memory mechanism" in t for t in embedded_texts), \
        f"excerpt missing from query embedding text: {embedded_texts!r}"
    assert any("MEM" in t for t in embedded_texts)


def test_collision_high_score_name_mismatch_deferred_to_judge():
    """High cosine but disagreeing names must NOT mechanically patch AND must NOT
    be written as a silent new note (fix #1) — it is deferred to the ternary
    dedup judge, which reads both bodies.

    A surface-name duplicate ("SVDD" vs "Support Vector Data Description") and a
    domain collision ("MEMORY" vs "RAM (Random Access Memory)") are identical from
    names+cosine alone; only the judge can tell them apart. The old contract
    demoted these to the distiller as new notes, starving the judge of exactly the
    pairs it exists to resolve.
    """
    fsm = _make_fsm_at_collision([
        {"name": "MEMORY", "inbox_excerpt": "Agentic memory mechanisms for evolving context."}
    ])

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5
    mock_store.cosine_top_k.return_value = [
        {"path": "Agenti Autonomi/RAM (Random Access Memory).md",
         "name": "RAM (Random Access Memory)", "score": 0.88}
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    mock_deferred = MagicMock()

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.kernel.recall.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder), \
         patch("silica.kernel.recall.deferred.get_deferred_store", return_value=mock_deferred):
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.65
        fsm.step()

    # No mechanical patch — the name disagreement forbids the fast path.
    assert fsm.context.get("chunk_0_collision_ops", []) == []
    # Concept removed from the chunk — deferred to the judge, not written as new.
    kept = [c for b in fsm._chunks[0].get("batches", []) for c in b.get("concepts", [])]
    assert not any(c["name"] == "MEMORY" for c in kept)
    # It landed in the deferred store (the ternary judge's queue).
    mock_deferred.put.assert_called_once()


def test_collision_high_score_acronym_match_still_patches():
    """High cosine WITH agreeing names (concept == note acronym) still auto-patches.

    Positive control: GPT → "Modelli Linguistici Generativi (GPT)" is a true dedup
    target, so the mechanical pre-route must remain.
    """
    fsm = _make_fsm_at_collision([
        {"name": "GPT", "inbox_excerpt": "Generative pretrained transformer language model."}
    ])

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5
    mock_store.cosine_top_k.return_value = [
        {"path": "Agenti Autonomi/Modelli Linguistici Generativi (GPT).md",
         "name": "Modelli Linguistici Generativi (GPT)", "score": 0.88}
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.router.orchestrator.DRIVER") as mock_driver, \
         patch("silica.kernel.recall.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder):
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.65
        mock_driver.read_note.return_value = MagicMock()
        fsm.step()

    ops = fsm.context.get("chunk_0_collision_ops", [])
    assert len(ops) == 1
    assert ops[0]["op"] == "patch"
    assert ops[0]["path"] == "Agenti Autonomi/Modelli Linguistici Generativi (GPT).md"


def test_collision_borderline_deferred():
    """τ_low < score < τ_high → concept deferred, removed from chunk, deferred store called."""
    fsm = _make_fsm_at_collision([{"name": "Backprop", "inbox_excerpt": "Backpropagation intro."}])

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5
    mock_store.cosine_top_k.return_value = [{"path": "DL/Gradient Descent.md", "name": "Gradient Descent", "score": 0.75}]

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    mock_deferred = MagicMock()

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.kernel.recall.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder), \
         patch("silica.kernel.recall.deferred.get_deferred_store", return_value=mock_deferred):
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.65

        fsm.step()

    assert fsm.state == InjectorState.DELEGATE
    collision_ops = fsm.context.get("chunk_0_collision_ops", [])
    assert collision_ops == []
    # Concept removed from chunk (deferred)
    concepts_left = [
        c
        for b in fsm._chunks[0].get("batches", [])
        for c in b.get("concepts", [])
    ]
    assert concepts_left == []
    # Deferred store was called
    mock_deferred.put.assert_called_once()
    put_kwargs = mock_deferred.put.call_args[1]
    assert put_kwargs["content_hash"] == "abc123"
    assert len(put_kwargs["rejected_ops"]) == 1


def test_collision_borderline_defers_rematerializable_op_and_tags_workitem():
    """C2: the deferred bundle carries the full op — excerpt as snippet, a real
    write path, candidate+score in the reason — never an op='skip'/path=None
    stub; and the dedup WorkItem carries content_hash + target_dir so the
    verdict routing can clean up (or author from) the twin bundle."""
    from silica.kernel.workqueue import WorkQueue

    fsm = _make_fsm_at_collision([{"name": "Backprop", "inbox_excerpt": "Backpropagation intro."}])
    fsm.work_queue = WorkQueue()

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5
    mock_store.cosine_top_k.return_value = [{"path": "DL/Gradient Descent.md", "name": "Gradient Descent", "score": 0.75}]

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    mock_deferred = MagicMock()
    mock_deferred.get.return_value = None

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.kernel.recall.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder), \
         patch("silica.kernel.recall.deferred.get_deferred_store", return_value=mock_deferred):
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.65

        fsm.step()

    put_kwargs = mock_deferred.put.call_args[1]
    op = put_kwargs["rejected_ops"][0]
    assert op["op"] == "write"
    assert op["path"] == "TargetDir/Backprop.md"
    assert op["snippet"] == "Backpropagation intro."
    assert "Gradient Descent" in op["reason"] and "0.75" in op["reason"]

    items = fsm.work_queue.items()
    assert len(items) == 1
    ctx = items[0].context
    assert ctx["content_hash"] == "abc123"
    assert ctx["target_dir"] == "TargetDir"


def test_collision_does_not_consult_the_fused_facade():
    """The candidate is the cosine-best note (ADR-0030); the co-occurrence leg
    is never consulted, so a note only that leg would propose cannot become a
    candidate, and an empty cosine result keeps the concept for distillation."""
    fsm = _make_fsm_at_collision([{"name": "Graph Theory", "inbox_excerpt": "Nodes and edges."}])

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5  # non-empty index
    mock_store.cosine_top_k.return_value = []

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.kernel.recall.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder), \
         patch("silica.kernel.recall.relatedness.related_notes_for_query") as facade:
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.65

        fsm.step()

    facade.assert_not_called()
    assert fsm.state == InjectorState.DELEGATE
    assert fsm.context.get("chunk_0_collision_ops", []) == []
    concepts_left = [
        c
        for b in fsm._chunks[0].get("batches", [])
        for c in b.get("concepts", [])
    ]
    assert len(concepts_left) == 1
    assert concepts_left[0]["name"] == "Graph Theory"


def test_collision_inbox_candidate_is_filtered():
    """An Inbox note surfacing as top collision candidate must never become a
    patch target — validate rejects every Inbox path, so routing one guarantees
    a rejected op (real incident: 2026-07-17 nucleate run, Lezione 1↔2 and the
    SVM book cross-patching). The concept flows to normal distillation."""
    fsm = _make_fsm_at_collision([{"name": "Support Vector Machines", "inbox_excerpt": "SVM intro."}])

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5  # non-empty index
    mock_store.cosine_top_k.return_value = [
        {"path": "Inbox/svm-book/01-intro.md", "name": "Support Vector Machines", "score": 0.92},
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.router.orchestrator.DRIVER") as mock_driver, \
         patch("silica.kernel.recall.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder):
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.65
        mock_driver.read_note.return_value = MagicMock()  # graph would confirm the node

        fsm.step()

    assert fsm.state == InjectorState.DELEGATE
    assert fsm.context.get("chunk_0_collision_ops", []) == []
    concepts_left = [
        c
        for b in fsm._chunks[0].get("batches", [])
        for c in b.get("concepts", [])
    ]
    assert [c["name"] for c in concepts_left] == ["Support Vector Machines"]


def test_collision_empty_index_skips_transparently():
    """Empty embedding index → COLLISION is a no-op, chunk flows unchanged."""
    concepts = [{"name": "Test Concept"}, {"name": "Another Concept"}]
    fsm = _make_fsm_at_collision(concepts)

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 0  # empty index

    with patch("silica.kernel.recall.embed.EmbedStore", return_value=mock_store):
        fsm.step()

    assert fsm.state == InjectorState.DELEGATE
    assert fsm.context.get("chunk_0_collision_ops", []) == []
    # Chunk is UNCHANGED (not modified by collision)
    concepts_left = [
        c
        for b in fsm._chunks[0].get("batches", [])
        for c in b.get("concepts", [])
    ]
    assert len(concepts_left) == 2


@patch("silica.agent.llm.call_llm")
@patch("silica.agent.providers.get_provider")
def test_worker_read_only(mock_get_provider, mock_call_llm):
    # Test 1: Verify that run_distiller calls call_llm with tools=None (single-shot variant)
    from silica.kernel.prep_delegation import run_distiller
    
    mock_get_provider.side_effect = Exception("Mocked provider failure")
    
    mock_response = MagicMock()
    mock_response.text = '{"updates": []}'
    mock_call_llm.return_value = mock_response

    payload = {"schema_version": 1, "batches": []}
    run_distiller(payload, target="TargetDir", hub="Hub")

    mock_call_llm.assert_called_once()
    _, kwargs = mock_call_llm.call_args
    assert kwargs.get("tools") is None

    # Test 2: Verify that every builtin worker profile allowlists only atomic
    # read-only tools (the profile.tools tuple is the single enforcement seam:
    # AgentConstraints receives it verbatim in run_worker).
    from silica.capabilities.profiles_builtin import READER, ROUTER
    import silica.tools.atomic  # noqa: F401 — populates TOOLS via @tool decorators
    from silica.tools import TOOLS

    mutation_tools = {
        "silica_run_injector",
        "silica_bulk_write",
        "silica_move",
        "silica_delete",
        "silica_snapshot",
        "silica_restore",
        "silica_cleanup",
    }
    for profile in (READER, ROUTER):
        for name in profile.tools:
            assert name in TOOLS, f"profile '{profile.name}' lists unknown tool '{name}'"
            assert TOOLS[name].cls == "atomic", f"profile '{profile.name}' exposes non-atomic tool '{name}'"
            assert name not in mutation_tools, f"profile '{profile.name}' exposes mutation tool '{name}'"


@patch("silica.router.orchestrator.silica_validate_ops")
def test_fsm_short_circuit_no_ops(mock_validate):
    # Setup mock validation with 0 validated and 0 rejected (all skip/no-op)
    mock_validate.return_value = {
        "success": True,
        "validated_count": 0,
        "rejected_count": 0,
        "rejection_rate": 0.0,
        "total": 5,
    }

    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.context["payload"] = {"payload": {"chunk_id": 0}}
    fsm.context.setdefault("chunk", {})["sanitized"] = {"parsed": []}
    fsm.state = InjectorState.VALIDATE

    with patch.object(fsm, "_make_tmp", return_value="dummy_ops.json"):
        fsm.step()

    # Verify transition directly to CLEANUP and final_status set to "no_ops"
    assert fsm.state == InjectorState.CLEANUP
    assert fsm.context["final_status"] == "no_ops"


@patch("silica.router.orchestrator.silica_recon")
@patch("silica.router.orchestrator.silica_payload")
@patch("silica.router.states.distill.run_distiller")
@patch("silica.router.orchestrator.silica_sanitize")
@patch("silica.router.orchestrator.silica_validate_ops")
@patch("silica.router.orchestrator.DRIVER")
@patch("silica.tools.wrapped.silica_snapshot")
@patch("silica.tools.wrapped.silica_restore")
@patch("silica.tools.wrapped.silica_cleanup")
def test_fsm_create_settle_timeout_rollback(
    mock_cleanup, mock_restore, mock_snapshot, mock_driver,
    mock_validate, mock_sanitize, mock_run_distiller, mock_payload, mock_recon
):
    mock_recon.return_value = {"success": True}
    mock_payload.return_value = {"payload": {"chunk_id": 0}}
    mock_run_distiller.return_value = {"updates": []}
    mock_sanitize.return_value = {"parsed": [{"op": "write", "path": "test.md", "heading": "Test", "hub": "Hub", "source_basename": "test_fsm_settle.md"}]}
    
    import json
    def side_effect_validate(ops_json_path, **kwargs):
        with open(ops_json_path, 'w', encoding='utf-8') as f:
            json.dump([{"op": "write", "path": "test.md", "heading": "Test", "hub": "Hub", "source_basename": "test_fsm_settle.md"}], f)
        return {
            "success": True, "rejection_rate": 0.0, "validated_count": 1, "rejected_count": 0,
        }
    mock_validate.side_effect = side_effect_validate
    
    mock_snapshot.return_value = {
        "txn_id": "txn_123",
        "inverses": [{"kind": "delete_created", "path": "test.md"}]
    }
    mock_restore.return_value = {"success": True}
    
    pre_graph = MagicMock()
    mock_driver.graph_snapshot.return_value = pre_graph
    
    fsm = InjectorFSM("Inbox/test_fsm_settle.md", "TargetDir")
    
    # We patch silica.kernel.write.bulk.DRIVER.create to raise a driver write failure
    with patch("silica.kernel.write.bulk.DRIVER.create", side_effect=RuntimeError("Settle timeout mock error")):
        with patch("silica.kernel.graph_diff.check_graph_regression", return_value=(True, [])):
            res = fsm.run()
        
    # Chunk-level containment: single-chunk run concludes as partial (not ERROR):
    # the settle-timeout defers the ops but the chunk still reaches CLEANUP, so
    # committed_chunks=1 and "partial" is the honest verdict (vs "failed" = zero commits)
    assert fsm.state == InjectorState.DONE
    assert res.get("final_status") == "partial"

    # Check that cleanup is not called (inbox not moved — chunk had a failure)
    mock_cleanup.assert_not_called()
    # Check that restore (rollback) was not called globally since the failed note write was self-atomic
    mock_restore.assert_not_called()


# ---------------------------------------------------------------------------
# WRITE partial-failure containment (Fase A → B)
# ---------------------------------------------------------------------------

@patch("silica.kernel.recall.deferred.get_deferred_store")
@patch("silica.kernel.write.atomic_write.commit_note_atomic")
def test_handle_write_partial_failure_defers_and_continues(
    mock_commit, mock_get_store, tmp_path
):
    """Partial write failure: committed ops survive, failed ops land in deferred store,
    FSM continues to HUB_UPDATE (no rollback), has_partial_failure is set."""
    import json
    from silica.kernel.write.atomic_write import NoteCommitResult
    from silica.kernel.recall.deferred import DeferredStore
    from silica.kernel.write.ops import InverseOp, InverseOpKind

    ops_data = [
        {"op": "write", "path": "TargetDir/A.md", "heading": "A", "hub": "TargetDir",
         "source_basename": "test.md", "snippet": "a"},
        {"op": "write", "path": "TargetDir/B.md", "heading": "B", "hub": "TargetDir",
         "source_basename": "test.md", "snippet": "b"},
    ]
    ops_file = tmp_path / "ops.json"
    ops_file.write_text(json.dumps(ops_data))

    # Op A committed, op B failed (lint failure)
    def commit_side_effect(op, hub=None, lint=True):
        if "A.md" in (op.path or ""):
            inv = InverseOp(kind=InverseOpKind.delete_created, path=op.path or "")
            return NoteCommitResult(ok=True, path=op.path or "", op=op.op.value,
                                    inverses=[inv])
        return NoteCommitResult(ok=False, path=op.path or "", op=op.op.value,
                                error="Settle timeout", reverted=True)
    mock_commit.side_effect = commit_side_effect

    deferred_store = DeferredStore(tmp_path / "deferred")
    mock_get_store.return_value = deferred_store

    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.state = InjectorState.WRITE
    fsm.context["source_content_hash"] = "abc123hash"
    fsm.context["chunk"] = {
        "ops_path": str(ops_file),
        "txn_id": "txn_test",
        "snapshot": {"txn_id": "txn_test", "inverses": []},
    }

    fsm.step()

    # FSM advances to HUB_UPDATE — no rollback
    assert fsm.state == InjectorState.HUB_UPDATE
    assert fsm.context.get("has_partial_failure") is True

    # Op B deferred under the source content hash
    bundle = deferred_store.get("abc123hash")
    assert bundle is not None
    failed_paths = {o.get("path") for o in bundle["rejected_ops"]}
    assert "TargetDir/B.md" in failed_paths
    assert "Settle timeout" in bundle["rejection_reasons"].get("TargetDir/B.md", "")


@patch("silica.kernel.write.atomic_write.commit_note_atomic")
def test_handle_write_all_fail_defers_and_continues(mock_commit, tmp_path):
    """When ALL ops fail lint/write, they are deferred and FSM continues (no rollback).

    This is the new per-note atomic behavior: bulk_write_atomic never raises;
    every failure is self-reverted and deferred. The FSM advances to HUB_UPDATE
    with has_partial_failure set.
    """
    import json
    from silica.kernel.write.atomic_write import NoteCommitResult

    ops_data = [
        {"op": "write", "path": "TargetDir/A.md", "heading": "A", "hub": "TargetDir",
         "source_basename": "test.md", "snippet": "a"},
    ]
    ops_file = tmp_path / "ops.json"
    ops_file.write_text(json.dumps(ops_data))

    mock_commit.return_value = NoteCommitResult(
        ok=False, path="TargetDir/A.md", op="write",
        error="fatal lint failure", reverted=True,
    )

    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.state = InjectorState.WRITE
    fsm.context["source_content_hash"] = "xyz"
    fsm.context["chunk"] = {
        "ops_path": str(ops_file),
        "txn_id": "txn_x",
        "snapshot": {"txn_id": "txn_x", "inverses": []},
    }

    # All-fail: _handle_write defers and continues (no raise, no rollback).
    fsm.step()

    assert fsm.state == InjectorState.HUB_UPDATE
    assert fsm.context.get("has_partial_failure") is True
    write_ctx = fsm.context.get("write", {})
    assert write_ctx.get("successful") == 0
    assert len(write_ctx.get("failed", [])) == 1


def test_hub_inverse_appears_in_chunk_ctx_snapshot(tmp_path):
    """After HUB_UPDATE, the hub rollback inverse must land in the txn's authoritative
    inverses list (the single source of truth read by ROLLBACK) — not in a stale dict."""
    import json, os
    from unittest.mock import patch, MagicMock
    from silica.router.orchestrator import InjectorFSM

    ops_path = str(tmp_path / "ops.json")
    with open(ops_path, "w") as f:
        json.dump([{
            "op": "write", "path": "TargetDir/Note.md",
            "heading": "Note", "source_basename": "test.md",
            "content": "# Note\n", "snippet": "Note snippet",
            "hub": None, "parent": None,
        }], f)

    fsm = InjectorFSM("Inbox/test.md", "TargetDir", hub="Hub")
    # _chunk_ctx is a property returning self.context.setdefault("chunk", {}),
    # so seed the per-chunk state via context["chunk"] directly.
    fsm.context["chunk"] = {
        "ops_path": ops_path,
        "snapshot": {"txn_id": "txn-001", "inverses": [], "created_paths": []},
        "snapshot_domain": [],
    }
    fsm._current_chunk_idx = 0
    fsm._chunk_flat_to_fi_ci = {0: (0, 0)}
    fsm._file_chunks = {0: {"source_file": "Inbox/test.md", "chunks": [{}]}}

    mock_hub_note = MagicMock()
    mock_hub_note.content = "# Hub\n\nExisting content\n"

    from silica.driver.base import Txn
    fsm._txn = Txn(id="txn-001", created_paths=[], inverses=[])

    with patch("silica.router.orchestrator.DRIVER") as mock_driver, \
         patch("silica.router.orchestrator.time") as mock_time:
        mock_driver.read_note.return_value = mock_hub_note
        mock_driver.overwrite.return_value = None
        mock_time.monotonic.side_effect = [0.0, 10.0]
        mock_time.sleep.return_value = None
        states.write.handle_hub_update(fsm)

    txn_inverses = fsm._txn.inverses_serialized
    assert any(
        inv.get("path", "").endswith("Hub.md") for inv in txn_inverses
    ), f"Hub inverse missing from txn.inverses; got: {txn_inverses}"
    # The dual-write to _chunk_ctx['snapshot']['inverses'] is gone — the txn is
    # now the single source of truth, so the snapshot dict stays as SNAPSHOT left it.
    assert fsm._chunk_ctx["snapshot"]["inverses"] == [], \
        "Hub inverse must not be dual-written into _chunk_ctx['snapshot']['inverses']"
    # Also confirm it did NOT land in stale context['snapshot'].
    assert "snapshot" not in fsm.context or "inverses" not in fsm.context.get("snapshot", {}), \
        "Hub inverse was written to stale context['snapshot']"


def test_hub_update_writes_parent_at_its_real_vault_path(tmp_path):
    """An existing parent note in another folder must be patched where it lives,
    not at target_dir/<parent>.md (which would read as 'File not found')."""
    import json
    from unittest.mock import patch, MagicMock
    from silica.router.orchestrator import InjectorFSM
    from silica.driver.base import NoteRef, Txn

    ops_path = str(tmp_path / "ops.json")
    with open(ops_path, "w") as f:
        json.dump([{
            "op": "write", "path": "testing/SomeNote.md",
            "heading": "SomeNote", "source_basename": "test.md",
            "content": "# SomeNote\n", "snippet": "Some snippet",
            "hub": None, "parent": "lezione_7",
        }], f)

    fsm = InjectorFSM("Inbox/test.md", "testing", hub="Hub")
    fsm.context["chunk"] = {
        "ops_path": ops_path,
        "snapshot": {"txn_id": "txn-001", "inverses": [], "created_paths": []},
        "snapshot_domain": [],
    }
    fsm._current_chunk_idx = 0
    fsm._chunk_flat_to_fi_ci = {0: (0, 0)}
    fsm._file_chunks = {0: {"source_file": "Inbox/test.md", "chunks": [{}]}}
    fsm._txn = Txn(id="txn-001", created_paths=[], inverses=[])

    note = MagicMock()
    note.content = "# lezione_7\n\nExisting\n"

    # Two names for the one proxy: the FSM reads/overwrites through
    # orchestrator.DRIVER, name resolution lives in kernel/write/moc.py and
    # reaches the driver directly (kernel must not import router).
    with patch("silica.router.orchestrator.DRIVER") as mock_driver, \
         patch("silica.driver.DRIVER", mock_driver), \
         patch("silica.router.orchestrator.time") as mock_time:
        mock_driver.read_note.return_value = note
        mock_driver.overwrite.return_value = None
        # lezione_7 lives in 'Lezioni/', NOT under target_dir 'testing/'.
        mock_driver.search_names.return_value = [NoteRef(name="lezione_7", path="Lezioni/lezione_7.md")]
        mock_time.monotonic.side_effect = [0.0, 10.0, 20.0, 30.0]
        mock_time.sleep.return_value = None
        states.write.handle_hub_update(fsm)

    overwritten_paths = [c.args[0] for c in mock_driver.overwrite.call_args_list]
    assert "Lezioni/lezione_7.md" in overwritten_paths, \
        f"parent not patched at its real path; overwrote: {overwritten_paths}"
    assert "testing/lezione_7.md" not in overwritten_paths



@patch("silica.router.states.distill.orch.CONFIG.distill_concurrency", 1)
@patch("silica.router.states.distill.run_distiller")
def test_fsm_seen_override_reaches_episodic_capture(mock_run_distiller):
    """seen_override (bench knob) replaces the ingest day in capture_from_distill."""
    fsm = InjectorFSM("Inbox/test.md", "TargetDir", seen_override="2023-05-08")
    fsm._chunks = [{"chunk_id": 0, "concepts": ["a"]}]
    fsm._current_chunk_idx = 0
    fsm.state = InjectorState.DELEGATE
    mock_run_distiller.return_value = {"updates": []}

    captured = {}

    def _rec(result, *, run_id, seen, **kw):
        captured["run_id"] = run_id
        captured["seen"] = seen

    with patch.object(fsm, "_make_tmp", return_value="tmp.json"), \
         patch("silica.kernel.recall.episodic.capture_from_distill", side_effect=_rec):
        fsm.step()

    assert captured["seen"] == "2023-05-08"
    assert captured["run_id"] == fsm.progress.run_id


@patch("silica.router.states.distill.orch.CONFIG.distill_concurrency", 1)
@patch("silica.router.states.distill.run_distiller")
def test_fsm_seen_default_is_ingest_day(mock_run_distiller):
    """Without the override the product behavior is unchanged: ingest day."""
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._chunks = [{"chunk_id": 0, "concepts": ["a"]}]
    fsm._current_chunk_idx = 0
    fsm.state = InjectorState.DELEGATE
    mock_run_distiller.return_value = {"updates": []}

    captured = {}

    def _rec(result, *, run_id, seen, **kw):
        captured["seen"] = seen

    with patch.object(fsm, "_make_tmp", return_value="tmp.json"), \
         patch("silica.kernel.recall.episodic.capture_from_distill", side_effect=_rec):
        fsm.step()

    assert captured["seen"] == fsm.progress.started_at[:10]


@patch("silica.router.states.distill.orch.CONFIG.distill_concurrency", 1)
@patch("silica.router.states.distill.run_distiller")
def test_fsm_can_run_without_feeding_the_episodic_store(mock_run_distiller):
    """/promote distills a RENDER of episodic memory. Capturing that back into
    the store nests the chain inside itself: measured on a real run, one
    promotion turned the head into a summary of its own history."""
    fsm = InjectorFSM("Inbox/test.md", "TargetDir", episodic_capture=False)
    fsm._chunks = [{"chunk_id": 0, "concepts": ["a"]}]
    fsm._current_chunk_idx = 0
    fsm.state = InjectorState.DELEGATE
    mock_run_distiller.return_value = {"updates": []}

    calls = []

    with patch.object(fsm, "_make_tmp", return_value="tmp.json"), \
         patch("silica.kernel.recall.episodic.capture_from_distill",
               side_effect=lambda *a, **kw: calls.append(kw)):
        fsm.step()

    assert calls == []


def test_coordinator_forwards_seen_override():
    from silica.router.coordinator import Coordinator

    coord = Coordinator(inbox_files=["Inbox/test.md"], target_dir="TargetDir",
                        seen_override="2023-05-08")
    assert coord.fsm.seen_override == "2023-05-08"


def test_coordinator_forwards_the_episodic_capture_switch():
    from silica.router.coordinator import Coordinator

    coord = Coordinator(inbox_files=["Inbox/test.md"], target_dir="TargetDir",
                        episodic_capture=False)
    assert coord.fsm.episodic_capture is False


def test_best_effort_states_from_recipe():
    # A26: salience/collision/autolink/backlink are best_effort in the
    # recipe, so an unhandled failure in any of them must skip, not abort.
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    assert fsm._best_effort_states == {
        InjectorState.SALIENCE, InjectorState.COLLISION,
        InjectorState.AUTOLINK, InjectorState.BACKLINK,
    }


def test_best_effort_failure_skips_to_next_phase():
    # A26: a raising best-effort handler advances to the next phase instead of
    # routing to ERROR (which for post-write phases would strand a live txn).
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.state = InjectorState.SALIENCE

    def _boom():
        raise RuntimeError("salience blew up")

    def _stop():
        fsm.state = InjectorState.DONE

    fsm._HANDLERS[InjectorState.SALIENCE] = _boom
    fsm._HANDLERS[InjectorState.COLLISION] = _stop  # next in sequence after salience

    fsm._run_loop()

    assert fsm.state == InjectorState.DONE  # reached COLLISION, not ERROR
    assert fsm.context.get("error") == "salience blew up"


def test_recon_steps_straight_to_payload():
    """CROSSDEDUP is gone (2026-08-09 A/B: it deleted 44% of a book's concepts,
    only 60% of them real duplicates). RECON hands the file to PAYLOAD, and the
    state no longer exists on the enum."""
    fsm = InjectorFSM("Inbox/a.md", "TargetDir", inbox_files=["Inbox/a.md", "Inbox/b.md"])
    fsm.state = InjectorState.RECON
    fsm.context["recon"] = [{"file": "Inbox/a.md", "new_concepts": ["PIL"], "collisions": []}]

    with patch("silica.router.orchestrator.states.setup.handle_recon",
               side_effect=lambda f: f._transition_success()):
        fsm.step()

    assert fsm.state == InjectorState.PAYLOAD
    assert not hasattr(InjectorState, "CROSSDEDUP")


def test_collision_routes_on_the_cosine_best_note_not_the_fused_winner():
    """Routing thresholds are cosine thresholds, so the note they are tested on
    is the cosine-best one. Measured 2026-08-23 on 60 duplicates (ADR-0030):
    the cosine-top is the duplicate 98% of the time, the RRF winner 87%, and a
    duplicate handed to the judge under the wrong candidate is lost either
    way. Here the fused facade would put a 0.70 note first (kept, under
    tau_low) while the cosine-best note sits at 0.80 (deferred)."""
    from silica.kernel.recall.relatedness import RelatedNote
    from silica.kernel.workqueue import WorkQueue

    fsm = _make_fsm_at_collision([{"name": "ML estimator", "inbox_excerpt": "Maximises the likelihood."}])
    fsm.work_queue = WorkQueue()

    mock_store = MagicMock()
    mock_store.__len__ = lambda _: 5
    mock_store.cosine_top_k.return_value = [
        {"path": "Stats/Stimatore di massima verosimiglianza.md", "name": "Stimatore di massima verosimiglianza", "score": 0.80},
        {"path": "Stats/Funzione di verosimiglianza.md", "name": "Funzione di verosimiglianza", "score": 0.70},
    ]
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    fused_winner = [RelatedNote(
        path="Stats/Funzione di verosimiglianza.md", name="Funzione di verosimiglianza",
        score=0.03, evidence=["embed:0.70", "cooccur:w9"], embed_score=0.70, cooccur_weight=9.0,
    )]
    mock_deferred = MagicMock()
    mock_deferred.get.return_value = None

    with patch("silica.router.orchestrator.CONFIG") as mock_cfg, \
         patch("silica.kernel.recall.embed.EmbedStore", return_value=mock_store), \
         patch("silica.agent.providers.get_embedder", return_value=mock_embedder), \
         patch("silica.kernel.recall.relatedness.related_notes_for_query", return_value=fused_winner), \
         patch("silica.kernel.recall.deferred.get_deferred_store", return_value=mock_deferred):
        mock_cfg.sim_threshold_high = 0.85
        mock_cfg.sim_threshold_low = 0.75
        fsm.step()

    items = fsm.work_queue.items()
    assert len(items) == 1, "the cosine-best note at 0.80 is in the band and must reach the judge"
    assert items[0].target_path == "Stats/Stimatore di massima verosimiglianza.md"
    assert items[0].context["score"] == 0.80
