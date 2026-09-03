# SPDX-License-Identifier: AGPL-3.0-or-later

"""A run states what it did NOT cover, next to what it did.

Every number below was already recorded — the run report, log.md, the deferred
bundle — and none of them reached the line the user reads. A nucleation that
covered part of a lecture announced the same "Success" as one that covered all
of it, which is how a 159-note ingest got read as 20 concepts out of 80.
"""
from __future__ import annotations

from types import SimpleNamespace


from silica.router.coordinator import Coordinator

LONG = ("Il percettrone separa due classi con un iperpiano appreso per "
        "correzione dell'errore sui campioni mal classificati. " * 4)


def _park(monkeypatch, tmp_path):
    from silica.kernel.recall import deferred

    monkeypatch.setattr(deferred, "_store_dir", lambda: tmp_path / "deferred")
    deferred._stores.clear()
    return deferred.get_deferred_store()


def _coord(fsm):
    c = object.__new__(Coordinator)
    c.fsm = fsm
    return c


def _fsm(*, residue=None, hashes=(), recovered=0):
    return SimpleNamespace(
        progress=SimpleNamespace(inputs={"residue": residue or {}}),
        _file_content_hashes=list(hashes),
        _annealed_ops=recovered,
    )


def test_uncovered_facts_reach_the_result(tmp_path, monkeypatch):
    _park(monkeypatch, tmp_path)
    fsm = _fsm(residue={"f0": {"missing": ["a", "b", "c"], "total": 30}})

    result: dict = {}
    _coord(fsm)._coverage_summary(result)

    assert result["coverage"]["residue_facts"] == 3


def test_ops_still_parked_are_counted_after_the_anneal(tmp_path, monkeypatch):
    store = _park(monkeypatch, tmp_path)
    store.put("aaa1", "inbox/a.md", "Reti", None,
              [{"op": "write", "heading": "Stub", "source_basename": "a.md",
                "path": "Reti/Stub.md", "snippet": "corto"}],
              rejection_reasons={"Reti/Stub.md": "snippet too short"},
              phase="VALIDATE")
    fsm = _fsm(hashes=["aaa1"])

    result: dict = {}
    _coord(fsm)._coverage_summary(result)

    assert result["coverage"]["deferred_ops"] == 1


def test_another_run_s_bundles_are_not_this_run_s_debt(tmp_path, monkeypatch):
    """The store is per-vault, not per-run: only this run's sources count."""
    store = _park(monkeypatch, tmp_path)
    store.put("other", "inbox/z.md", "Reti", None,
              [{"op": "write", "heading": "Z", "source_basename": "z.md",
                "path": "Reti/Z.md", "snippet": "corto"}],
              rejection_reasons={"Reti/Z.md": "snippet too short"},
              phase="VALIDATE")
    fsm = _fsm(hashes=["mine"])

    result: dict = {}
    _coord(fsm)._coverage_summary(result)

    assert result.get("coverage") is None


def test_recovered_ops_are_reported_too(tmp_path, monkeypatch):
    """The anneal's recovery is the good half of the same honesty."""
    _park(monkeypatch, tmp_path)
    fsm = _fsm(recovered=23)

    result: dict = {}
    _coord(fsm)._coverage_summary(result)

    assert result["coverage"]["recovered_ops"] == 23


def test_a_fully_covered_run_says_nothing(tmp_path, monkeypatch):
    _park(monkeypatch, tmp_path)

    result: dict = {}
    _coord(_fsm())._coverage_summary(result)

    assert "coverage" not in result


def test_the_summary_never_breaks_a_run(tmp_path, monkeypatch):
    """Same fail-open contract as every other end-of-run pass."""
    _park(monkeypatch, tmp_path)
    broken = SimpleNamespace()  # no progress, no hashes

    result: dict = {}
    _coord(broken)._coverage_summary(result)

    assert "coverage" not in result
