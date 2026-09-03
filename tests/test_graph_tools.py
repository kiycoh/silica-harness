"""Tests for silica_graph_path, silica_graph_explain, silica_ledger_next, silica_ledger_update.

All tests are isolated from the live driver. graph_export.build_graph_data is
monkeypatched where needed; ProgressLedger uses a temporary _RUNS_DIR.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(nid: str, group: int = 0, node_type: str = "note") -> dict:
    return {"id": nid, "label": nid, "group": group, "type": node_type}


def _edge(eid: str, src: str, dst: str, etype: str = "EXTRACTED") -> dict:
    return {"id": eid, "from": src, "to": dst, "type": etype}


def _make_graph():
    """A → B → C (cluster 0),  D ← C (cross-cluster bridge to cluster 1)."""
    nodes = [
        _node("A", group=0), _node("B", group=0), _node("C", group=0),
        _node("D", group=1),
    ]
    edges = [
        _edge("e0", "A", "B"),
        _edge("e1", "B", "C"),
        _edge("e2", "C", "D"),
    ]
    return nodes, edges


# ---------------------------------------------------------------------------
# silica_graph_path
# ---------------------------------------------------------------------------

class TestGraphPath:
    def test_direct_path(self):
        from silica.tools.atomic import silica_graph_path
        nodes, edges = _make_graph()
        with patch("silica.kernel.recall.graph_export.build_graph_data", return_value=(nodes, edges)):
            res = silica_graph_path("A", "D")
        assert "error" not in res
        assert res["length"] == 3
        assert res["paths"][0][0] == "A"
        assert res["paths"][0][-1] == "D"

    def test_same_node(self):
        from silica.tools.atomic import silica_graph_path
        nodes, edges = _make_graph()
        with patch("silica.kernel.recall.graph_export.build_graph_data", return_value=(nodes, edges)):
            res = silica_graph_path("A", "A")
        assert "error" not in res
        assert res["length"] == 0

    def test_no_path(self):
        from silica.tools.atomic import silica_graph_path
        # Isolated node E with no edges
        nodes = [_node("A"), _node("E")]
        edges = []
        with patch("silica.kernel.recall.graph_export.build_graph_data", return_value=(nodes, edges)):
            res = silica_graph_path("A", "E")
        assert "error" in res

    def test_unknown_source(self):
        from silica.tools.atomic import silica_graph_path
        nodes, edges = _make_graph()
        with patch("silica.kernel.recall.graph_export.build_graph_data", return_value=(nodes, edges)):
            res = silica_graph_path("DOES_NOT_EXIST", "B")
        assert "error" in res

    def test_unknown_target(self):
        from silica.tools.atomic import silica_graph_path
        nodes, edges = _make_graph()
        with patch("silica.kernel.recall.graph_export.build_graph_data", return_value=(nodes, edges)):
            res = silica_graph_path("A", "DOES_NOT_EXIST")
        assert "error" in res

    def test_multiple_paths(self):
        from silica.tools.atomic import silica_graph_path
        # A ↔ B ↔ C and A ↔ C (two paths A→C)
        nodes = [_node("A"), _node("B"), _node("C")]
        edges = [_edge("e0", "A", "B"), _edge("e1", "B", "C"), _edge("e2", "A", "C")]
        with patch("silica.kernel.recall.graph_export.build_graph_data", return_value=(nodes, edges)):
            res = silica_graph_path("A", "C", max_paths=3)
        assert "error" not in res
        assert len(res["paths"]) >= 1


# ---------------------------------------------------------------------------
# silica_ledger_next / silica_ledger_update (round-trip)
# ---------------------------------------------------------------------------

class TestLedgerSteering:
    def test_round_trip(self, tmp_path, monkeypatch):
        """Full cycle: vault_report seeds tasks → next → update("done") → done."""
        import silica.kernel.progress as prog_mod
        monkeypatch.setattr(prog_mod, "_RUNS_DIR", tmp_path / "runs")

        from silica.kernel.progress import ProgressLedger
        from silica.tools.atomic import silica_ledger_next, silica_ledger_update
        import orjson

        # Manually create a run with one pending task
        progress = ProgressLedger.new(mode="analyst", inputs={})
        run_id = progress.run_id
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True)
        payloads_dir = run_dir / "payloads"
        payloads_dir.mkdir()

        task = progress.add_task("silica_autolink")
        payload = {"note_path": "Notes/Test", "_reason": "test"}
        payload_path = str(payloads_dir / f"{task.id}.json")
        Path(payload_path).write_bytes(orjson.dumps(payload))
        task.input_ref = payload_path
        progress.save()

        # silica_ledger_next returns the pending task
        result = silica_ledger_next(run_id)
        assert "error" not in result
        assert result["remaining"] == 0
        assert result["tasks"][0]["capability"] == "silica_autolink"
        assert result["tasks"][0]["payload"]["note_path"] == "Notes/Test"
        task_id = result["tasks"][0]["task_id"]

        # silica_ledger_update marks it done
        upd = silica_ledger_update(run_id, task_id, "done")
        assert upd.get("ok") is True
        assert "digest" in upd

        # silica_ledger_next now returns done
        result2 = silica_ledger_next(run_id)
        assert result2.get("done") is True

    def test_ledger_next_unknown_run(self, tmp_path, monkeypatch):
        import silica.kernel.progress as prog_mod
        monkeypatch.setattr(prog_mod, "_RUNS_DIR", tmp_path / "runs")
        from silica.tools.atomic import silica_ledger_next
        res = silica_ledger_next("nonexistent_run_id")
        assert "error" in res

    def test_ledger_update_unknown_task(self, tmp_path, monkeypatch):
        import silica.kernel.progress as prog_mod
        monkeypatch.setattr(prog_mod, "_RUNS_DIR", tmp_path / "runs")
        from silica.kernel.progress import ProgressLedger
        from silica.tools.atomic import silica_ledger_update

        p = ProgressLedger.new(mode="test")
        p.save()
        res = silica_ledger_update(p.run_id, "bad_task_id", "done")
        assert "error" in res

    def test_needs_confirmation_flag_propagated(self, tmp_path, monkeypatch):
        """propose tasks have needs_confirmation=True in their payload."""
        import silica.kernel.progress as prog_mod
        monkeypatch.setattr(prog_mod, "_RUNS_DIR", tmp_path / "runs")

        from silica.kernel.progress import ProgressLedger
        from silica.tools.atomic import silica_ledger_next
        import orjson

        p = ProgressLedger.new(mode="test")
        run_id = p.run_id
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True)
        payloads_dir = run_dir / "payloads"
        payloads_dir.mkdir()

        t = p.add_task("silica_autolink")
        payload = {"note_path": "X", "needs_confirmation": True, "_reason": "proposed"}
        pp = str(payloads_dir / f"{t.id}.json")
        Path(pp).write_bytes(orjson.dumps(payload))
        t.input_ref = pp
        p.save()

        res = silica_ledger_next(run_id)
        assert res["tasks"][0]["needs_confirmation"] is True

    def test_next_drains_the_whole_ready_frontier(self, tmp_path, monkeypatch):
        """One call hands out every runnable task, not one per round trip.

        Serving them one at a time cost 3 LLM round trips per task and read to
        the agent as "this tool only takes one item".
        """
        import silica.kernel.progress as prog_mod
        monkeypatch.setattr(prog_mod, "_RUNS_DIR", tmp_path / "runs")

        from silica.kernel.progress import ProgressLedger
        from silica.tools.atomic import silica_ledger_next
        import orjson

        p = ProgressLedger.new(mode="test")
        run_dir = tmp_path / "runs" / p.run_id
        (run_dir / "payloads").mkdir(parents=True)
        for i in range(5):
            t = p.add_task("silica_autolink")
            pp = run_dir / "payloads" / f"{t.id}.json"
            pp.write_bytes(orjson.dumps({"note_paths": [f"N{i}.md"]}))
            t.input_ref = str(pp)
        # ...except one gated behind a still-pending task: not part of the frontier.
        gated = p.add_task("silica_autolink", depends_on=[p.tasks[0].id])
        p.save()

        res = silica_ledger_next(p.run_id)
        assert len(res["tasks"]) == 5
        assert res["remaining"] == 0
        assert gated.id not in {t["task_id"] for t in res["tasks"]}

    def test_next_caps_a_fat_drain_and_says_what_is_left(self, tmp_path, monkeypatch):
        """The byte budget bounds one drain; `remaining` is how the agent knows."""
        import silica.kernel.progress as prog_mod
        monkeypatch.setattr(prog_mod, "_RUNS_DIR", tmp_path / "runs")

        from silica.kernel.progress import ProgressLedger
        from silica.tools.atomic import silica_ledger_next, LEDGER_DRAIN_BYTES
        import orjson

        p = ProgressLedger.new(mode="test")
        run_dir = tmp_path / "runs" / p.run_id
        (run_dir / "payloads").mkdir(parents=True)
        fat = ["Folder/" + "x" * 60 + f"{i}.md" for i in range(60)]  # ~4KB per payload
        for _ in range(10):
            t = p.add_task("silica_autolink")
            pp = run_dir / "payloads" / f"{t.id}.json"
            pp.write_bytes(orjson.dumps({"note_paths": fat}))
            t.input_ref = str(pp)
        p.save()

        res = silica_ledger_next(p.run_id)
        assert 0 < len(res["tasks"]) < 10
        assert res["remaining"] == 10 - len(res["tasks"])
        assert len(orjson.dumps(res)) < LEDGER_DRAIN_BYTES * 2


# ---------------------------------------------------------------------------
# Tests for partition_by_file invariant
# ---------------------------------------------------------------------------

class TestPartitionByFile:
    def test_no_cross_file_chunks(self):
        from silica.kernel.partition import partition_by_file

        payload = {
            "schema_version": 1,
            "batches": [
                {"inbox_file": "Inbox/A.md", "concepts": [{"name": f"c{i}"} for i in range(10)]},
                {"inbox_file": "Inbox/B.md", "concepts": [{"name": f"d{i}"} for i in range(5)]},
            ],
        }
        groups = partition_by_file(payload, max_concepts=4)
        # Each group must belong to exactly one source file
        for group in groups:
            sf = group["source_file"]
            for chunk in group["chunks"]:
                for batch in chunk.get("batches", []):
                    assert batch["inbox_file"] == sf, "Cross-file chunk detected!"

    def test_concept_counts_preserved(self):
        from silica.kernel.partition import partition_by_file

        payload = {
            "schema_version": 1,
            "batches": [
                {"inbox_file": "A.md", "concepts": [{"name": str(i)} for i in range(7)]},
                {"inbox_file": "B.md", "concepts": [{"name": str(i)} for i in range(3)]},
            ],
        }
        groups = partition_by_file(payload, max_concepts=4)
        for group in groups:
            sf = group["source_file"]
            expected = 7 if sf == "A.md" else 3
            actual = sum(
                len(b.get("concepts", []))
                for chunk in group["chunks"]
                for b in chunk.get("batches", [])
            )
            assert actual == expected

    def test_source_file_tag_on_chunks(self):
        from silica.kernel.partition import partition_by_file

        payload = {
            "schema_version": 1,
            "batches": [
                {"inbox_file": "X.md", "concepts": [{"name": "a"}, {"name": "b"}]},
            ],
        }
        groups = partition_by_file(payload, max_concepts=10)
        for group in groups:
            for chunk in group["chunks"]:
                assert chunk.get("source_file") == "X.md"


class TestGraphExportAutoCooccur:
    """silica_graph_export refreshes the co-occurrence index first (best-effort)."""

    def test_refreshes_cooccurrence_before_export(self, monkeypatch):
        import silica.tools.graph as gmod
        import silica.ui.web.graph_view as gx

        calls = []
        monkeypatch.setattr(gmod, "silica_cooccurrence_refresh",
                            lambda folder="": calls.append(("refresh", folder)))
        monkeypatch.setattr(gx, "export_graph",
                            lambda **kw: calls.append(("export", kw["folder"])) or {"ok": True})

        result = gmod.silica_graph_export(folder="sub")

        assert result == {"ok": True}
        assert calls == [("refresh", "sub"), ("export", "sub")]  # refresh first

    def test_refresh_failure_is_best_effort(self, monkeypatch):
        import silica.tools.graph as gmod
        import silica.ui.web.graph_view as gx

        def boom(folder=""):
            raise RuntimeError("index locked")
        monkeypatch.setattr(gmod, "silica_cooccurrence_refresh", boom)
        monkeypatch.setattr(gx, "export_graph", lambda **kw: {"ok": True})

        # Must not raise — naming degrades to "Cluster N", graph still renders.
        assert gmod.silica_graph_export() == {"ok": True}


def test_report_names_the_fixes_it_queues(tmp_vault, tmp_path, monkeypatch):
    """The chat layer asked "apply the 2 proposed fixes?" without ever saying
    which — the tool result must name tier + capability + reason for every
    queued task, so consent is informed instead of blind."""
    import silica.kernel.progress as prog_mod
    monkeypatch.setattr(prog_mod, "_RUNS_DIR", tmp_path / "runs")

    tmp_vault.note("A.md", "# A\n\nalpha beta gamma.")
    tmp_vault.note("B.md", "# B\n\ndelta epsilon zeta.")

    from silica.tools.graph import silica_vault_report
    res = silica_vault_report()

    assert "plan_preview" in res
    for t in res["plan_preview"]:
        assert t["tier"] in ("auto", "propose")
        assert t["capability"]
        assert t["reason"]


def test_report_names_unconverted_documents_anywhere_in_the_vault(tmp_vault, tmp_path, monkeypatch):
    """/report only scanned Inbox/ for pending files: a student's slide/*.pdf
    was invisible and the audit read as "nothing to do" over a folder of
    unread material. Already-converted documents (their .md sits in done/ or
    the inbox) are not re-reported."""
    import silica.kernel.progress as prog_mod
    from pathlib import Path
    from silica.config import CONFIG

    monkeypatch.setattr(prog_mod, "_RUNS_DIR", tmp_path / "runs")
    tmp_vault.note("A.md", "# A\n\nalpha.")
    root = Path(CONFIG.vault_path)
    (root / "vault.yaml").write_text("write_dir: silica\n", encoding="utf-8")
    from silica.kernel.vault_manifest import reset_manifest_cache
    reset_manifest_cache()
    (root / "slide").mkdir()
    (root / "slide" / "deck.pdf").write_bytes(b"%PDF-1.4")
    (root / "letture").mkdir()
    (root / "letture" / "paper.pdf").write_bytes(b"%PDF-1.4")
    tmp_vault.note("silica/done/paper.md", "# paper (converted)")

    from silica.tools.graph import silica_vault_report
    res = silica_vault_report()

    pend = res.get("unconverted") or {}
    assert "slide/deck.pdf" in pend.get("files", [])
    assert "letture/paper.pdf" not in pend.get("files", [])
