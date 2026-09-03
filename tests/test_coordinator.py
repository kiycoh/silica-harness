"""Concurrency tests for the Coordinator producer/consumer wiring.

Uses a fake FSM (to avoid LLM/vault) and a fake sub-agent so we exercise the
queue drain/join logic deterministically.
"""
import time
from types import SimpleNamespace
from unittest.mock import patch

from silica.config import SilicaConfig
from silica.router.coordinator import Coordinator
from silica.kernel.workqueue import WorkItem


class _FakeFSM:
    """Produces N dedup items during run(), like the Injector committing batches."""

    def __init__(self, n_items, *, per_item_delay=0.0):
        self.work_queue = None
        self.progress = SimpleNamespace(run_dir=None)
        self._n = n_items
        self._delay = per_item_delay

    def run(self):
        # Mirror the real FSM: only produce when a queue has been wired in.
        for i in range(self._n):
            if self.work_queue is not None:
                self.work_queue.enqueue(WorkItem(kind="dedup", target_path=f"N{i}.md"))
            if self._delay:
                time.sleep(self._delay)
        return {"final_status": "ok"}


class _FakeAgent:
    def __init__(self, *a, **k):
        pass

    def handle(self, item):
        time.sleep(0.01)  # simulate worker-model latency
        return {"status": "committed"}


def _coordinator_with(fake_fsm, config):
    import threading
    coord = object.__new__(Coordinator)
    coord.config = config
    coord.fsm = fake_fsm
    coord._stop = threading.Event()  # mirrors Coordinator.__init__
    return coord


def test_coordinator_drains_all_items():
    cfg = SilicaConfig()
    cfg.subagent_max_concurrent = 3
    coord = _coordinator_with(_FakeFSM(8, per_item_delay=0.005), cfg)

    with patch("silica.agent.subagent.BoundedSubAgent", _FakeAgent):
        result = coord.run()

    # Every produced item was consumed and committed — none lost on join.
    assert result["subagents"] == {"committed": 8}
    assert result["final_status"] == "ok"


def test_coordinator_handles_empty_production():
    cfg = SilicaConfig()
    coord = _coordinator_with(_FakeFSM(0), cfg)
    with patch("silica.agent.subagent.BoundedSubAgent", _FakeAgent):
        result = coord.run()
    assert result["subagents"] == {}


# --- end-of-run orphan resolution ------------------------------------------

class _FakeFSMWithWarnings:
    """Produces nothing during run, but leaves two orphan warnings behind."""

    def __init__(self):
        self.work_queue = None
        self.warning_ledger = None
        self.target_dir = "Concepts"
        self.hub = "Concepts"
        self.progress = SimpleNamespace(run_dir=None)

    def run(self):
        # The Coordinator wires warning_ledger before calling run().
        self.warning_ledger.add("Concepts/Lonely.md", "orphan", "orphaned")
        self.warning_ledger.add("Concepts/Connected.md", "orphan", "orphaned")
        return {"final_status": "ok"}


def test_current_orphans_uses_driver_graph_not_full_report(monkeypatch):
    """Fix B: orphan status comes from the driver graph (in-degree), not a full
    compute_report — no Louvain/PageRank needed for "who is orphaned". At 10k
    notes that is ~3.8s (report) vs a sub-ms graph scan, fired up to 2x/run.
    """
    import silica.driver as drv
    from silica.driver.base import NoteRef
    from silica.agent.bounds import _norm_path

    coord = _coordinator_with(SimpleNamespace(target_dir="Concepts"), SilicaConfig())
    monkeypatch.setattr(
        drv, "DRIVER",
        SimpleNamespace(orphans=lambda: [NoteRef(name="Lonely", path="Concepts/Lonely.md")]),
    )
    # The expensive report path must NOT run for an orphan check.
    import silica.kernel.report.graph_report as gr

    def _boom(*a, **k):
        raise AssertionError("compute_report must not run for the orphan check")

    monkeypatch.setattr(gr, "compute_report", _boom)

    assert coord._current_orphans() == {_norm_path("Concepts/Lonely.md")}


def test_coordinator_enqueues_only_residual_orphans_and_reverifies():
    cfg = SilicaConfig()
    coord = _coordinator_with(_FakeFSMWithWarnings(), cfg)

    # After the run, only "Concepts/Lonely" is still orphaned (Connected got linked
    # by a later chunk / backlink). Re-verify finds zero still orphaned post-repair.
    current_calls = [
        {"concepts/lonely"},   # _enqueue_orphan_repairs → residual computation
        set(),                 # _reverify_orphans → none still orphaned
    ]

    def fake_current():
        return current_calls.pop(0) if current_calls else set()

    with patch("silica.agent.subagent.BoundedSubAgent", _FakeAgent), \
         patch.object(type(coord), "_current_orphans", side_effect=fake_current), \
         patch.object(type(coord), "_orphan_candidates", return_value=[{"name": "X", "path": "Concepts/X"}]):
        result = coord.run()

    ow = result["orphan_warnings"]
    assert ow["warned"] == 2
    assert ow["residual"] == 1          # only Lonely still orphaned
    assert ow["enqueued"] == 1
    assert ow["residual_after"] == 0    # re-verify: repaired


def test_orphan_repairs_are_capped_per_run():
    """Each repair is a worker LLM call carrying an 8000-char body: a run that
    warned on a whole folder must not turn CLEANUP into an unbounded spend.
    Past the cap, orphans stay visible in the ledger — just not repaired now."""
    from silica.router.coordinator import MAX_ORPHAN_REPAIRS_PER_RUN

    class _ManyWarnings(_FakeFSMWithWarnings):
        def run(self):
            for i in range(MAX_ORPHAN_REPAIRS_PER_RUN + 5):
                self.warning_ledger.add(f"Concepts/O{i}.md", "orphan", "orphaned")
            return {"final_status": "ok"}

    coord = _coordinator_with(_ManyWarnings(), SilicaConfig())
    with patch("silica.agent.subagent.BoundedSubAgent", _FakeAgent), \
         patch.object(type(coord), "_current_orphans", return_value=set()), \
         patch.object(type(coord), "_orphan_candidates", return_value=[{"name": "X", "path": "Concepts/X"}]):
        result = coord.run()

    ow = result["orphan_warnings"]
    assert ow["residual"] == MAX_ORPHAN_REPAIRS_PER_RUN + 5   # visibility intact
    assert ow["enqueued"] == MAX_ORPHAN_REPAIRS_PER_RUN       # spend bounded


# --- interrupt regression ---------------------------------------------------
# Regression: KI mid-drain left the Live display active (garbled terminal) and
# worker threads alive (blocking Ctrl+C).  Now _stop is set on BaseException,
# _consume bails on the flag, and the renderer's Live is torn down in cli.py.


class _SlowAgent:
    """Simulate a worker that blocks for a while on each item."""

    def __init__(self, *a, **k):
        pass

    def handle(self, item):
        time.sleep(0.05)
        return {"status": "committed"}


class _KIFakeFSM:
    """Produces two items then raises KeyboardInterrupt — mimics Ctrl+C mid-run."""

    def __init__(self):
        self.work_queue = None
        self.warning_ledger = None
        self.progress = SimpleNamespace(run_dir=None)

    def run(self):
        for i in range(2):
            if self.work_queue is not None:
                self.work_queue.enqueue(WorkItem(kind="dedup", target_path=f"N{i}.md"))
        raise KeyboardInterrupt


def test_interrupt_mid_drain_stops_workers_and_renderer():
    """KeyboardInterrupt from the producer must:

    1. Set _stop so consumers exit without completing queued items.
    2. Not leave any worker threads alive after the coordinator unwinds.
    3. Leave the renderer in a closeable state (Live == None after close()).
    """
    import threading
    from silica.ui.renderer import make_progress_callback

    cfg = SilicaConfig()
    cfg.subagent_max_concurrent = 2

    coord = _coordinator_with(_KIFakeFSM(), cfg)

    renderer = make_progress_callback()

    import concurrent.futures

    original_executor_init = concurrent.futures.ThreadPoolExecutor.__init__

    # Track created worker threads so we can assert they exit after interrupt.
    captured_pool: list[concurrent.futures.ThreadPoolExecutor] = []

    def patched_init(self, *args, **kwargs):
        original_executor_init(self, *args, **kwargs)
        captured_pool.append(self)

    with patch("silica.agent.subagent.BoundedSubAgent", _SlowAgent), \
         patch.object(concurrent.futures.ThreadPoolExecutor, "__init__", patched_init):
        try:
            coord.run()
        except KeyboardInterrupt:
            pass  # expected

    # 1. _stop must be set — consumers won't block on new claims.
    assert coord._stop.is_set(), "_stop must be set after KI so consumers exit"

    # 2. All worker threads must have exited (pool shut down without wait).
    #    Give threads a moment to clean up after shutdown(wait=False).
    if captured_pool:
        pool = captured_pool[0]
        pool.shutdown(wait=True)  # idempotent; ensures threads joined for assertion
    alive_workers = [
        t for t in threading.enumerate()
        if t.name.startswith("subagent") and t.is_alive()
    ]
    assert not alive_workers, f"Worker threads still alive after KI: {alive_workers}"

    # 3. Renderer can be closed cleanly (Live is stopped, no AttributeError).
    renderer.close()
    assert renderer._live is None, "renderer._live must be None after close()"


def test_consumer_threads_resolve_the_run_id_lazily_not_at_thread_entry():
    """The pool is submitted BEFORE fsm.run(), so _undo_run_id is still None when
    a worker thread starts. Snapshotting it at entry pins None forever and every
    expand/dedup write skips the journal — read it per item instead."""

    fsm = _FakeFSM(0)
    fsm._undo_run_id = None  # not opened yet, exactly as at pool submit time
    coord = _coordinator_with(fsm, SilicaConfig())

    seen = {}

    def _fake_consume(wq, agent, stop=None, undo_run=None, **narration_kw):
        fsm._undo_run_id = "run-xyz"          # the FSM opens its run meanwhile
        seen["at_item"] = undo_run() if undo_run else None

    with patch("silica.agent.subagent.consume", _fake_consume):
        coord._consume(None)

    assert seen["at_item"] == "run-xyz"


def test_consume_stamps_the_run_id_per_item(tmp_vault):
    """The seam itself: consume() must set the contextvar commit_ops reads."""
    from silica.agent.commit import _current_undo_run
    from silica.agent.subagent import consume
    from silica.kernel.workqueue import WorkQueue, WorkItem

    wq = WorkQueue()
    wq.enqueue(WorkItem(kind="dedup", target_path="N.md"))
    wq.close()

    seen = []

    class _Agent:
        def handle(self, item):
            seen.append(_current_undo_run.get())
            return {"status": "committed"}

    consume(wq, _Agent(), None, undo_run=lambda: "run-abc")
    assert seen == ["run-abc"]
