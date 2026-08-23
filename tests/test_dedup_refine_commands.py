"""Tests for the ad-hoc /dedup and /refine commands and run_subagent_batch."""
import json
from unittest.mock import patch, MagicMock

from silica.agent.subagent import run_subagent_batch, BoundedSubAgent
from silica.kernel.workqueue import WorkItem


# --- run_subagent_batch ----------------------------------------------------

def test_run_subagent_batch_aggregates_outcomes():
    items = [WorkItem(kind="dedup", target_path=f"N{i}.md") for i in range(4)]
    with patch.object(BoundedSubAgent, "handle", lambda self, it: {"status": "committed"}):
        res = run_subagent_batch(items, max_workers=2)
    assert res["items"] == 4
    assert res["summary"] == {"committed": 4}
    assert len(res["results"]) == 4


def test_run_subagent_batch_empty():
    assert run_subagent_batch([])["items"] == 0


# --- silica_dedup ----------------------------------------------------------

class _BatchFromTopK:
    """cosine_top_k_batch expressed through the fake's own cosine_top_k.

    The scan calls the batch API (one matmul for the whole scope); these fixtures
    stay single-source on the per-note ranking they were written to express.
    """

    def cosine_top_k_batch(self, keys, k=5, *, exclude_self=True, block=256):
        return {
            p: self.cosine_top_k(self.get_vec(p), k, {p} if exclude_self else set())
            for p in keys if self.get_vec(p) is not None
        }

    def sync_from_disk(self):
        """The accessor re-reads the index when another process wrote it
        (paths.DiskSynced); a fake has no file, so there is never anything
        to re-read."""
        return False


class _FakeStore(_BatchFromTopK):
    def __len__(self):
        return 2

    def paths(self):
        return ["Concepts/A", "Concepts/B"]

    def get_vec(self, p):
        return [1.0, 0.0]

    def get_title_vec(self, p):
        return None  # simulates pre-title_vec index entry

    def cosine_top_k(self, vec, k=5, exclude=None):
        exclude = exclude or set()
        cand = next(x for x in ["Concepts/A", "Concepts/B"] if x not in exclude)
        return [{"path": cand, "score": 0.80, "name": cand}]  # borderline: τ_low .75 < .80 < τ_high .85


def _read_note(path):
    bodies = {"Concepts/A": "short", "Concepts/B": "a much longer note body " * 20}
    return MagicMock(content=bodies.get(path, ""))


def test_silica_dedup_builds_pair_targeting_larger_note():
    from silica.tools.composed import silica_dedup
    with patch("silica.kernel.recall.embed.EmbedStore", _FakeStore), \
         patch("silica.driver.DRIVER.read_note", side_effect=_read_note), \
         patch("silica.agent.subagent.run_subagent_batch", return_value={"items": 1, "summary": {"committed": 1}, "results": []}) as batch:
        res = silica_dedup(folder="Concepts")

    items = batch.call_args.args[0]
    assert len(items) == 1
    # The larger note (B) is the merge target; the smaller (A) is the source.
    assert items[0].target_path == "Concepts/B"
    assert items[0].context["concept"] == "A"
    assert res["pairs_found"] == 1


def test_silica_dedup_requires_index():
    from silica.tools.composed import silica_dedup

    class _Empty(_FakeStore):
        def __len__(self):
            return 0

    with patch("silica.kernel.recall.embed.EmbedStore", _Empty):
        res = silica_dedup(folder="X")
    assert "error" in res


# --- silica_refine_batch / silica_enrich_batch --------------------------------

def test_silica_refine_batch_requires_paths():
    from silica.tools.composed import silica_refine_batch
    res = silica_refine_batch(note_paths=[])
    assert "error" in res


def test_silica_refine_batch_enqueues_items():
    from silica.tools.composed import silica_refine_batch
    paths = ["Notes/x.md", "Notes/y.md"]
    with patch("silica.agent.subagent.run_subagent_batch", return_value={"items": 2, "summary": {"committed": 2}, "results": []}) as batch:
        res = silica_refine_batch(note_paths=paths)
    items = batch.call_args.args[0]
    assert len(items) == 2
    assert all(it.kind == "refine" for it in items)
    assert res["notes"] == 2


def test_silica_enrich_batch_requires_paths():
    from silica.tools.composed import silica_enrich_batch
    res = silica_enrich_batch(note_paths=[])
    assert "error" in res


def test_silica_enrich_batch_enqueues_items():
    from silica.tools.composed import silica_enrich_batch
    paths = ["Notes/lean.md"]
    with patch("silica.agent.subagent.run_subagent_batch", return_value={"items": 1, "summary": {"committed": 1}, "results": []}) as batch:
        res = silica_enrich_batch(note_paths=paths)
    items = batch.call_args.args[0]
    assert len(items) == 1
    assert items[0].kind == "enrich"
    assert res["notes"] == 1


# --- CLI wiring ------------------------------------------------------------

def test_cli_dedup_seeds_ledger():
    """/dedup scans for pairs then seeds a resumable, agent-driven ledger
    (like /refine), instead of blocking on a synchronous silica_dedup call."""
    from silica import cli

    pairs = [{"source": "Concepts/A", "target": "Concepts/B", "score": 0.8}]
    with patch("silica.tools.runners._scan_dedup_pairs", return_value=(pairs, None)) as scan, \
         patch("silica.cli._seed_batch_ledger", return_value="ledger msg") as seed:
        msg = cli._expand_workflow_shortcut("/dedup Concepts/ML")

    scan.assert_called_once_with("Concepts/ML")
    cap, payloads = seed.call_args.args
    assert cap == "silica_dedup_pairs"
    assert payloads == [{"pairs": pairs}]
    assert seed.call_args.kwargs["kind"] == "dedup"
    assert msg == "ledger msg"


def test_cli_dedup_no_pairs_handled_inline():
    """An empty scan is fully handled inline — the agent gets nothing to do."""
    from silica import cli
    with patch("silica.tools.runners._scan_dedup_pairs", return_value=([], None)):
        msg = cli._expand_workflow_shortcut("/dedup")
    assert msg == ""


def test_cli_refine_scopes_with_in_folder():
    """/refine Foo must not leak into a sibling FooBar/ folder (prefix bug)."""
    from silica import cli
    from silica.driver.base import NoteRef

    refs = [NoteRef(name="a", path="Foo/a.md"), NoteRef(name="b", path="FooBar/b.md")]
    captured = {}

    def fake_seed(cap, payloads, *, kind, label):
        captured["cap"] = cap
        captured["paths"] = [p for pl in payloads for p in pl["note_paths"]]
        return "ok"

    with patch("silica.driver.DRIVER.list_files", return_value=refs), \
         patch("silica.cli._seed_batch_ledger", side_effect=fake_seed):
        cli._expand_workflow_shortcut("/refine Foo")

    assert captured["cap"] == "silica_refine_batch"
    assert captured["paths"] == ["Foo/a.md"]  # FooBar/b.md excluded by _in_folder


# --- Regression: k=1 horizon bug -------------------------------------------

class _ThreeNoteStore(_BatchFromTopK):
    """Simulates a vault where:
      - A→B scores 0.90 (above τ_high=0.85 → must be skipped, NOT stop the loop)
      - A→C scores 0.80 (borderline [τ_low=0.75, τ_high=0.85] → must be found)
    With k=1 only B was returned, causing the dedup scan to miss C entirely.
    With k>=2 both B and C are returned; B is skipped, C is captured.
    """

    def __len__(self):
        return 3

    def paths(self):
        return ["Folder/A", "Folder/B", "Folder/C"]

    def get_vec(self, p):
        return [1.0, 0.0]

    def get_title_vec(self, p):
        return None  # simulates pre-title_vec index entry

    def cosine_top_k(self, vec, k=5, exclude=None):
        exclude = exclude or set()
        # Return results in descending score order, honouring `k` and `exclude`.
        all_results = [
            {"path": "Folder/B", "score": 0.90, "name": "B"},  # above τ_high
            {"path": "Folder/C", "score": 0.80, "name": "C"},  # borderline
            {"path": "Folder/A", "score": 0.30, "name": "A"},  # below τ_low
        ]
        return [r for r in all_results if r["path"] not in exclude][:k]


def _read_three(path):
    return MagicMock(content="body " * 10)


def test_dedup_secondary_borderline_found_with_expanded_k():
    """Regression for k=1 horizon bug.

    When A's primary match (B, score=0.90) is above τ_high, the old k=1 code
    would discard B and move on without visiting C (score=0.80, borderline).
    With the multi-match loop (k=dedup_scan_k=5), C is correctly captured.

    The store also returns B→C as borderline (both excluded from k=1 scan),
    so the new code legitimately finds 2 pairs: A↔C and B↔C.
    The key invariant: no pair should have B as a borderline partner of A
    (A→B is 0.90, above τ_high, and must be suppressed).
    """
    from silica.tools.composed import silica_dedup

    with patch("silica.kernel.recall.embed.EmbedStore", _ThreeNoteStore), \
         patch("silica.driver.DRIVER.read_note", side_effect=_read_three), \
         patch("silica.agent.subagent.run_subagent_batch",
               return_value={"items": 1, "summary": {"committed": 1}, "results": []}) as batch:
        res = silica_dedup(folder="Folder")

    # At least one borderline pair must be found (A↔C or B↔C).
    assert res["pairs_found"] >= 1

    # No item should represent the A↔B pair (score 0.90, above τ_high).
    items = batch.call_args.args[0]
    for item in items:
        pair = {item.target_path, item.context["inbox_file"]}
        assert pair != {"Folder/A", "Folder/B"}, "A↔B is above τ_high and must be excluded"


# --- Title-similarity gate ---------------------------------------------------

class _TitleGateStore(_BatchFromTopK):
    """Simulates a vault where 'ROS' and 'JSON in ROS 2' have:
      - full-note score = 0.40 (below τ_low=0.75 → normally excluded)
      - title_vec cosine  = 0.85 (above sim_title_threshold=0.80 → admitted)
    """

    def __len__(self):
        return 2

    def paths(self):
        return ["Robotica/ROS", "Robotica/JSON in ROS 2"]

    def get_vec(self, p):
        return [1.0, 0.0]

    def get_title_vec(self, p):
        # Both notes have nearly identical title vectors
        return [0.9, 0.1]

    def cosine_top_k(self, vec, k=5, exclude=None):
        exclude = exclude or set()
        all_results = [
            {"path": "Robotica/ROS",           "score": 0.40, "name": "ROS"},
            {"path": "Robotica/JSON in ROS 2", "score": 0.40, "name": "JSON in ROS 2"},
        ]
        return [r for r in all_results if r["path"] not in exclude][:k]


def _read_ros(path):
    return MagicMock(content="body of note about ROS " * 5)


def test_dedup_title_gate_promotes_low_fullscore_pair():
    """Title gate: pair with full_score < τ_low but title_score ≥ sim_title_threshold
    must be admitted, regardless of the body-level score.

    This is the exact scenario for "ROS" / "JSON in ROS 2": the bodies are
    topically distinct (ROS overview vs JSON serialization in ROS 2) so the
    full-note cosine falls below τ_low, but the titles are semantically linked.
    """
    from silica.tools.composed import silica_dedup

    with patch("silica.kernel.recall.embed.EmbedStore", _TitleGateStore), \
         patch("silica.driver.DRIVER.read_note", side_effect=_read_ros), \
         patch("silica.agent.subagent.run_subagent_batch",
               return_value={"items": 1, "summary": {"committed": 1}, "results": []}) as batch:
        res = silica_dedup(folder="Robotica")

    assert res["pairs_found"] >= 1, "Title gate must admit the ROS / JSON in ROS 2 pair"

    items = batch.call_args.args[0]
    pair_paths = {items[0].target_path, items[0].context["inbox_file"]}
    assert "Robotica/ROS" in pair_paths
    assert "Robotica/JSON in ROS 2" in pair_paths

    # effective_score must be the title_score (0.85), not the full score (0.40)
    assert items[0].context["score"] > 0.80
    assert items[0].context["title_score"] >= 0.80
