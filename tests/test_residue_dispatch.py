"""Residue verification dispatch (verification-based residue, 2026-08-17).

Concurrency shape: decompose is dispatched when a file's chunks are attached
(it rides the whole file's distillation), evidence is gathered on the MAIN
thread at the last chunk's WRITE (snapshot: no race with autolink edits),
judge batches run as parallel futures, and the gate assembles the verdicts.
Every degrade falls back to the inline path or to [] (fail-open).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from silica.router.states import finalize as fz

# Bound at import time, BEFORE the autouse network guard in conftest stubs
# the module attributes for every other test.
real_decompose_dispatch = fz.maybe_dispatch_residue_decompose
real_check_dispatch = fz.maybe_dispatch_residue_check
real_residue_facts = fz.residue_facts
real_verify_now = fz._verify_now


def _driver_stub(content="Source text stating a fact."):
    return SimpleNamespace(read_note=lambda p: SimpleNamespace(content=content))


def _residue_fsm(ci=1, n_chunks=2, form=None, failed_task=None):
    fsm = SimpleNamespace()
    fsm._current_chunk_idx = ci
    fsm._chunk_flat_to_fi_ci = {i: (0, i) for i in range(n_chunks)}
    fsm._file_chunks = {0: {"source_file": "Inbox/in.md",
                            "chunks": [{} for _ in range(n_chunks)]}}
    fsm.inbox_file = "Inbox/in.md"
    fsm.context = {}
    if form:
        fsm.context["file_0_form"] = form
    fsm.progress = MagicMock()
    fsm.progress.tasks = (
        [SimpleNamespace(id=failed_task, status="failed")] if failed_task else []
    )
    fsm.progress.inputs = {}
    return fsm


def _fake_embedder(dim: int = 8):
    """A deterministic stand-in for the embedding server.

    Without it this file measured whether a local embedder happened to be
    listening: the dispatch path embeds, so it passed on a machine running
    llama-server on :1234 and failed in CI, where the connection error escapes
    into the seam's catch-all. The vectors are identical on purpose, so the
    theme filter keeps every fact and the guards under test are the only thing
    that can decide the outcome.
    """
    return SimpleNamespace(embed=lambda texts: [[1.0] + [0.0] * (dim - 1)
                                                for _ in texts])


def _shutdown(fsm):
    pool = getattr(fsm, "_residue_executor", None)
    if pool is not None:
        pool.shutdown(wait=False)


class TestDecomposeDispatch:
    def test_stores_future_per_file(self):
        fsm = _residue_fsm()
        with patch("silica.driver.DRIVER", _driver_stub()), \
             patch("silica.kernel.residue.decompose_facts",
                   return_value=["f1"]) as dec:
            real_decompose_dispatch(fsm, 0, "Inbox/in.md")
            assert fsm._residue_decompose[0].result(timeout=5) == ["f1"]
        dec.assert_called_once()
        _shutdown(fsm)

    def test_skips_draft_and_empty_source_and_is_idempotent(self):
        fsm = _residue_fsm(form="draft")
        with patch("silica.driver.DRIVER", _driver_stub()):
            real_decompose_dispatch(fsm, 0, "Inbox/in.md")
        assert not getattr(fsm, "_residue_decompose", None)

        fsm2 = _residue_fsm()
        with patch("silica.driver.DRIVER", _driver_stub(content="  ")):
            real_decompose_dispatch(fsm2, 0, "Inbox/in.md")
        assert not getattr(fsm2, "_residue_decompose", None)

        fsm3 = _residue_fsm()
        with patch("silica.driver.DRIVER", _driver_stub()), \
             patch("silica.kernel.residue.decompose_facts", return_value=["f"]) as dec:
            real_decompose_dispatch(fsm3, 0, "Inbox/in.md")
            real_decompose_dispatch(fsm3, 0, "Inbox/in.md")
        dec.assert_called_once()
        _shutdown(fsm3)


class TestCheckDispatch:
    def _decomposed(self, fsm, facts):
        with patch("silica.driver.DRIVER", _driver_stub()), \
             patch("silica.kernel.residue.decompose_facts", return_value=facts):
            real_decompose_dispatch(fsm, 0, "Inbox/in.md")
            fsm._residue_decompose[0].result(timeout=5)

    def test_gathers_evidence_and_submits_judge_futures(self):
        fsm = _residue_fsm()
        self._decomposed(fsm, ["fact a", "fact b"])
        with patch("silica.agent.providers.get_embedder_or_none",
                   return_value=object()), \
             patch("silica.kernel.recall.embed.get_store", return_value=object()), \
             patch("silica.driver.DRIVER", _driver_stub()), \
             patch("silica.kernel.residue.filter_on_theme",
                   return_value=(["fact a", "fact b"], [[1.0], [1.0]], 0)), \
             patch("silica.kernel.residue.gather_evidence",
                   return_value=["ea", "eb"]) as gev, \
             patch("silica.kernel.residue.judge_covered",
                   return_value=[True, False]):
            real_check_dispatch(fsm)
            fi, facts, futures = fsm._residue_future
            assert fi == 0 and facts == ["fact a", "fact b"]
            assert fsm.context["file_0_residue_meta"] == {"total": 2, "off_theme": 0}
            verdicts = [v for f in futures for v in f.result(timeout=5)]
        gev.assert_called_once()  # evidence snapshotted on the main thread
        assert gev.call_args.kwargs.get("vecs") == [[1.0], [1.0]]
        assert verdicts == [True, False]
        _shutdown(fsm)

    def test_decompose_not_done_leaves_gate_inline(self):
        fsm = _residue_fsm()
        import threading
        gate = threading.Event()
        with patch("silica.driver.DRIVER", _driver_stub()), \
             patch("silica.kernel.residue.decompose_facts",
                   side_effect=lambda s, **kw: gate.wait(5) or ["f"]):
            real_decompose_dispatch(fsm, 0, "Inbox/in.md")
            real_check_dispatch(fsm)
            gate.set()
        assert getattr(fsm, "_residue_future", None) is None
        assert getattr(fsm, "_residue_ready", None) is None
        _shutdown(fsm)

    def test_decompose_failure_becomes_ready_skip(self):
        fsm = _residue_fsm()
        self._decomposed(fsm, None)
        with patch("silica.driver.DRIVER", _driver_stub()):
            real_check_dispatch(fsm)
        fi, res = fsm._residue_ready
        assert fi == 0 and res["missing"] == [] and res.get("skipped")
        _shutdown(fsm)

    def test_guards_mid_file_and_draft(self):
        for fsm in (_residue_fsm(ci=0),
                    _residue_fsm(form="draft")):
            self._decomposed(fsm, ["f"]) if not fsm.context.get("file_0_form") \
                else None
            with patch("silica.driver.DRIVER", _driver_stub()):
                real_check_dispatch(fsm)
            assert getattr(fsm, "_residue_future", None) is None
            assert getattr(fsm, "_residue_ready", None) is None
            _shutdown(fsm)

    def test_failed_file_still_dispatches(self):
        # A failed chunk no longer refuses verification: the deferred store
        # is the rolled-back content's only recovery channel (run 262e6847).
        fsm = _residue_fsm(failed_task="f0_c0_write")
        self._decomposed(fsm, ["f"])
        with patch("silica.driver.DRIVER", _driver_stub()), \
             patch("silica.agent.providers.get_embedder_or_none",
                   lambda *a, **k: _fake_embedder()), \
             patch("silica.kernel.residue.judge_covered", return_value=[True]):
            real_check_dispatch(fsm)
            fut = getattr(fsm, "_residue_future", None)
            if fut is not None:
                [f.result(timeout=5) for f in fut[2]]
        ready = getattr(fsm, "_residue_ready", None)
        assert (getattr(fsm, "_residue_future", None) is not None
                or ready is not None)
        # Getting past the guard by crashing is not getting past the guard: the
        # catch-all leaves its own marker now, and a test that accepted it would
        # go on passing through any breakage in the dispatch below.
        assert ready is None or "pre-dispatch failed" not in ready[1].get("skipped", "")
        _shutdown(fsm)


class TestResidueFactsConsumption:
    def test_assembles_judge_futures_and_stashes_stats(self):
        from concurrent.futures import ThreadPoolExecutor
        fsm = _residue_fsm()
        pool = ThreadPoolExecutor(max_workers=1)
        futures = [pool.submit(lambda: [True, False, None])]
        fsm._residue_future = (0, ["a", "b", "c"], futures)
        missing = real_residue_facts(fsm, 0, "Inbox/in.md")
        assert missing == ["b"]
        stats = fsm.context["file_0_residue_stats"]
        assert stats["total"] == 3 and stats["judged"] == 2 and stats["failures"] == 1
        assert fsm._residue_future is None
        pool.shutdown(wait=False)

    def test_consumes_ready_result(self):
        fsm = _residue_fsm()
        fsm._residue_ready = (0, {"missing": [], "total": 0, "judged": 0,
                                  "failures": 0, "skipped": "decompose failed"})
        assert real_residue_facts(fsm, 0, "Inbox/in.md") == []
        assert fsm.context["file_0_residue_stats"]["skipped"] == "decompose failed"

    def test_inline_fallback_runs_full_verification(self):
        fsm = _residue_fsm()
        with patch.object(fz, "_verify_now", real_verify_now), \
             patch("silica.driver.DRIVER", _driver_stub()), \
             patch("silica.agent.providers.get_embedder_or_none",
                   return_value=object()), \
             patch("silica.kernel.recall.embed.get_store", return_value=object()), \
             patch("silica.kernel.residue.verify_missing",
                   return_value={"missing": ["m"], "total": 2,
                                 "judged": 2, "failures": 0}) as vm:
            missing = real_residue_facts(fsm, 0, "Inbox/in.md")
        assert missing == ["m"]
        assert vm.call_args.kwargs.get("facts") is None  # decomposes itself
        assert fsm.context["file_0_residue_stats"]["total"] == 2

    def test_judge_future_failure_degrades_to_empty(self):
        from concurrent.futures import ThreadPoolExecutor
        fsm = _residue_fsm()
        pool = ThreadPoolExecutor(max_workers=1)

        def boom():
            raise RuntimeError("judge died")
        fsm._residue_future = (0, ["a"], [pool.submit(boom)])
        assert real_residue_facts(fsm, 0, "Inbox/in.md") == []
        pool.shutdown(wait=False)


class TestLaneGate:
    """SILICA_RESIDUE_CHECK: auto (default) skips the outline lane, whose
    coverage pass already answers "what did we drop" per source heading in
    the same call; measured 2026-09-02 the lane cost 61% of a lecture's
    tokens and every spot-checked declaration was a false positive."""

    def _outline_fsm(self):
        fsm = _residue_fsm(ci=0, n_chunks=1)
        fsm._file_chunks[0]["chunks"] = [{"lane": "outline", "source_text": "x"}]
        return fsm

    def test_auto_skips_decompose_on_outline_lane(self, monkeypatch):
        monkeypatch.setattr("silica.config.CONFIG.residue_check", "auto")
        fsm = self._outline_fsm()
        with patch("silica.driver.DRIVER", _driver_stub()), \
             patch("silica.kernel.residue.decompose_facts") as dec:
            real_decompose_dispatch(fsm, 0, "Inbox/in.md")
        assert not getattr(fsm, "_residue_decompose", None)
        dec.assert_not_called()

    def test_auto_keeps_keyphrase_lane(self, monkeypatch):
        monkeypatch.setattr("silica.config.CONFIG.residue_check", "auto")
        fsm = _residue_fsm()
        with patch("silica.driver.DRIVER", _driver_stub()), \
             patch("silica.kernel.residue.decompose_facts", return_value=["f"]):
            real_decompose_dispatch(fsm, 0, "Inbox/in.md")
            assert fsm._residue_decompose[0].result(timeout=5) == ["f"]
        _shutdown(fsm)

    def test_on_overrides_outline_lane(self, monkeypatch):
        monkeypatch.setattr("silica.config.CONFIG.residue_check", "on")
        fsm = self._outline_fsm()
        with patch("silica.driver.DRIVER", _driver_stub()), \
             patch("silica.kernel.residue.decompose_facts", return_value=["f"]):
            real_decompose_dispatch(fsm, 0, "Inbox/in.md")
            assert fsm._residue_decompose[0].result(timeout=5) == ["f"]
        _shutdown(fsm)

    def test_off_skips_every_lane(self, monkeypatch):
        monkeypatch.setattr("silica.config.CONFIG.residue_check", "off")
        fsm = _residue_fsm()
        with patch("silica.driver.DRIVER", _driver_stub()), \
             patch("silica.kernel.residue.decompose_facts") as dec:
            real_decompose_dispatch(fsm, 0, "Inbox/in.md")
        assert not getattr(fsm, "_residue_decompose", None)
        dec.assert_not_called()

    def test_gate_never_verifies_inline_on_a_skipped_lane(self, monkeypatch):
        # residue_facts falls back to the synchronous verification when no
        # future is pending: without this guard the skip at PAYLOAD would
        # only move the same 60k tokens to CLEANUP.
        monkeypatch.setattr("silica.config.CONFIG.residue_check", "auto")
        fsm = self._outline_fsm()
        with patch.object(fz, "_verify_now") as vn:
            assert real_residue_facts(fsm, 0, "Inbox/in.md") == []
        vn.assert_not_called()
        assert fsm.context["file_0_residue_stats"]["skipped"] == "outline lane"

    def test_decompose_receives_the_file_language(self, monkeypatch):
        # English-only prompts decomposed an Italian lecture into English
        # facts; the lexical evidence window then shared no words with the
        # notes and the judge saw the wrong paragraphs (2026-09-02).
        monkeypatch.setattr("silica.config.CONFIG.residue_check", "auto")
        fsm = _residue_fsm()
        fsm.context["file_0_language"] = "Italian"
        with patch("silica.driver.DRIVER", _driver_stub()), \
             patch("silica.kernel.residue.decompose_facts", return_value=["f"]) as dec:
            real_decompose_dispatch(fsm, 0, "Inbox/in.md")
            fsm._residue_decompose[0].result(timeout=5)
        assert dec.call_args.kwargs["language"] == "Italian"
        _shutdown(fsm)
