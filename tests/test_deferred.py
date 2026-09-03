"""Tests for the DeferredStore and the VALIDATE partial-write gate."""
import pytest

from silica.kernel.recall.deferred import DeferredStore


# ---------------------------------------------------------------------------
# DeferredStore unit tests
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return DeferredStore(path=tmp_path / "deferred")


def test_deferred_put_and_get(store):
    store.put(
        content_hash="abc123",
        source_path="inbox/lezione_15.md",
        target_dir="Agenti Autonomi",
        hub="Agenti Autonomi",
        rejected_ops=[{"op": "write", "path": "Agenti Autonomi/MCU.md"}],
        rejection_reasons={"Agenti Autonomi/MCU.md": "too generic"},
    )
    bundle = store.get("abc123")
    assert bundle is not None
    assert bundle["source_path"] == "inbox/lezione_15.md"
    assert bundle["target_dir"] == "Agenti Autonomi"
    assert len(bundle["rejected_ops"]) == 1
    assert bundle["rejection_reasons"]["Agenti Autonomi/MCU.md"] == "too generic"


def test_put_persists_payloads_deduplicated(store):
    """Payloads (the ops' original validation evidence, audit finding 2) are
    stored with the bundle; the existing+new merge across chunks/re-runs must
    not re-append the same payload verbatim."""
    p = {"batches": [{"inbox_file": "inbox/a.md", "concepts": [{"name": "X"}]}]}
    store.put(
        "h1", "inbox/a.md", "Dir", None,
        [{"op": "write", "path": "Dir/X.md", "heading": "X"}],
        payloads=[p, p, {**p}],  # duplicates by content, not identity
    )
    bundle = store.get("h1")
    assert bundle["payloads"] == [p]


def test_put_without_payloads_stores_empty_list(store):
    store.put("h2", "inbox/a.md", "Dir", None, [{"op": "write", "path": "Dir/X.md"}])
    assert store.get("h2")["payloads"] == []


def test_defer_ops_persists_current_chunk_payload(tmp_path, monkeypatch):
    """The FSM funnel (_defer_ops) snapshots the current chunk's payload into
    the bundle, so every defer site gets retry-time grounding parity for free."""
    from silica.kernel.recall import deferred
    from silica.router.orchestrator import InjectorFSM

    monkeypatch.setattr(deferred, "_store_dir", lambda: tmp_path / "deferred")
    deferred._stores.clear()

    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._file_content_hashes = ["hash-defer"]  # _current_content_hash property source
    chunk_payload = {"batches": [{"inbox_file": "Inbox/test.md",
                                  "concepts": [{"name": "C", "inbox_excerpt": "testo"}]}]}
    fsm._chunks = [chunk_payload]
    fsm._current_chunk_idx = 0

    assert fsm._defer_ops(
        [{"op": "write", "path": "TargetDir/C.md", "heading": "C", "snippet": "corpo"}],
        {"TargetDir/C.md": "snippet too short"},
        phase="VALIDATE",
    )
    bundle = deferred.get_deferred_store().get("hash-defer")
    assert bundle["payloads"] == [chunk_payload]


def test_put_stamps_per_op_rejection_reason_and_phase(store):
    """Each op self-describes why and where it was deferred — forensics must
    not need the file-level dict join (path-keyed and heading-keyed alike)."""
    original = {"op": "patch", "path": "Dir/A.md", "reason": "distiller rationale"}
    store.put(
        "h", "inbox/a.md", "Dir", None,
        [original, {"op": "write", "heading": "Concept B"}],
        rejection_reasons={
            "Dir/A.md": "lint failed: X",
            "Concept B": "borderline_similarity score=0.71",
        },
        phase="VALIDATE",
    )
    ops = store.get("h")["rejected_ops"]
    assert ops[0]["rejection_reason"] == "lint failed: X"
    assert ops[0]["reason"] == "distiller rationale"  # distiller rationale untouched
    assert ops[1]["rejection_reason"] == "borderline_similarity score=0.71"
    assert all(o["rejection_phase"] == "VALIDATE" for o in ops)
    assert "rejection_reason" not in original  # caller's dict not mutated


def test_merge_keeps_first_phase_and_refreshes_reason(store):
    """_defer_ops merges existing + new: merged ops keep the gate that first
    deferred them; a re-deferred op picks up the fresh rejection reason."""
    store.put("h", "inbox/a.md", "Dir", None,
              [{"op": "write", "path": "Dir/A.md"}],
              rejection_reasons={"Dir/A.md": "borderline score=0.71"},
              phase="COLLISION")
    prev = store.get("h")
    store.put("h", "inbox/a.md", "Dir", None,
              prev["rejected_ops"] + [{"op": "write", "path": "Dir/B.md"}],
              rejection_reasons={**prev["rejection_reasons"], "Dir/B.md": "lint failed"},
              phase="WRITE")
    ops = {o["path"]: o for o in store.get("h")["rejected_ops"]}
    assert ops["Dir/A.md"]["rejection_phase"] == "COLLISION"
    assert ops["Dir/A.md"]["rejection_reason"] == "borderline score=0.71"
    assert ops["Dir/B.md"]["rejection_phase"] == "WRITE"
    assert ops["Dir/B.md"]["rejection_reason"] == "lint failed"


def test_deferred_get_missing(store):
    assert store.get("nonexistent") is None


def test_deferred_put_overwrites(store):
    store.put("abc123", "inbox/a.md", "Dir", None, [{"op": "write", "path": "Dir/A.md"}])
    store.put("abc123", "inbox/a.md", "Dir", None, [{"op": "write", "path": "Dir/B.md"}])
    bundle = store.get("abc123")
    assert bundle["rejected_ops"][0]["path"] == "Dir/B.md"


def test_deferred_put_dedups_same_op(store):
    """Merging existing + new re-rejections of the same (path, heading) must
    collapse to one, keeping the latest content — not grow every run."""
    store.put("h", "inbox/a.md", "Dir", None,
              [{"op": "write", "path": "Dir/A.md", "heading": "A", "reason": "v1"}])
    existing = store.get("h")["rejected_ops"]
    store.put("h", "inbox/a.md", "Dir", None,
              existing + [{"op": "write", "path": "Dir/A.md", "heading": "A", "reason": "v2"}])
    ops = store.get("h")["rejected_ops"]
    assert len(ops) == 1
    assert ops[0]["reason"] == "v2"


def test_deferred_sweeps_expired_on_open(tmp_path):
    """A bundle older than the TTL is unlinked when the store re-opens; a fresh
    one (and one with no timestamp) survives."""
    import orjson, time
    from silica.kernel.recall.deferred import DeferredStore, _DEFERRED_TTL_SECONDS

    d = tmp_path / "deferred"
    store = DeferredStore(path=d)
    store.put("fresh", "inbox/a.md", "Dir", None, [{"op": "write"}])
    # Backdate one bundle past the TTL; leave another with no timestamp.
    (d / "old.json").write_bytes(orjson.dumps(
        {"content_hash": "old", "timestamp": time.time() - _DEFERRED_TTL_SECONDS - 1}))
    (d / "nots.json").write_bytes(orjson.dumps({"content_hash": "nots"}))

    reopened = DeferredStore(path=d)  # sweep runs in __init__
    hashes = {i["content_hash"] for i in reopened.list_all()}
    assert hashes == {"fresh", "nots"}


def test_deferred_list_all(store):
    store.put("hash1", "inbox/a.md", "Dir", None, [{"op": "write", "path": "Dir/A.md"}])
    store.put("hash2", "inbox/b.md", "Dir2", "Hub2", [{"op": "write"}, {"op": "patch"}])
    items = store.list_all()
    assert len(items) == 2
    hashes = {i["content_hash"] for i in items}
    assert "hash1" in hashes
    assert "hash2" in hashes
    by_hash = {i["content_hash"]: i for i in items}
    assert by_hash["hash2"]["rejected_count"] == 2


def test_deferred_remove(store):
    store.put("abc123", "inbox/a.md", "Dir", None, [])
    assert store.remove("abc123") is True
    assert store.get("abc123") is None


def test_deferred_remove_missing(store):
    assert store.remove("nonexistent") is False


def test_deferred_list_empty(store):
    assert store.list_all() == []


# ---------------------------------------------------------------------------
# VALIDATE gate: partial-write behaviour
# ---------------------------------------------------------------------------

def test_validate_returns_validated_and_rejected_lists(tmp_path):
    """validate_operations always returns (validated, rejected) lists — never raises."""
    from silica.kernel.write.ops import Op, OpType
    from silica.kernel.write.validate import validate_operations

    op_a = Op(op=OpType.write, path="Dir/GPU.md", heading="GPU", source_basename="lezione.md")
    op_b = Op(op=OpType.write, path="Dir/MCU.md", heading="MCU", source_basename="lezione.md")

    # No payloads → heading check is skipped; path check will fail because
    # Dir/GPU.md and Dir/MCU.md don't exist in the real vault. Both will
    # be validated (write to non-existent path is valid) or rejected by path
    # rules depending on target_dir. Either way it must return lists.
    validated, rejected = validate_operations([op_a, op_b], [], target_dir=str(tmp_path))
    assert isinstance(validated, list)
    assert isinstance(rejected, list)
    assert len(validated) + len(rejected) == 2


def test_deferred_store_populated_on_partial_rejection(tmp_path):
    """When some ops are rejected, deferred store must receive the rejected ops."""
    from silica.kernel.recall.deferred import DeferredStore

    store = DeferredStore(path=tmp_path / "deferred")

    rejected_ops_raw = [
        {"op": {"op": "write", "path": "Dir/MCU.md", "heading": "MCU"}, "reason": "too generic"}
    ]

    deferred_ops = [
        r.get("op", r) if isinstance(r, dict) and "op" in r else r
        for r in rejected_ops_raw
    ]
    rejection_reasons = {
        (r.get("op", {}).get("path") or r.get("op", {}).get("heading") or "?"): r.get("reason", "")
        for r in rejected_ops_raw if isinstance(r, dict)
    }

    store.put(
        content_hash="testhash",
        source_path="inbox/lezione_15.md",
        target_dir="Dir",
        hub="Dir",
        rejected_ops=deferred_ops,
        rejection_reasons=rejection_reasons,
    )

    bundle = store.get("testhash")
    assert bundle is not None
    assert bundle["rejected_ops"][0]["path"] == "Dir/MCU.md"
    assert bundle["rejection_reasons"]["Dir/MCU.md"] == "too generic"


def test_defer_ops_accumulates_across_phases(tmp_path, monkeypatch):
    """_defer_ops must MERGE into the bundle, not overwrite it: COLLISION,
    VALIDATE and WRITE all key on the same source content_hash, so a later
    phase (or chunk) deferring ops must not clobber an earlier phase's ops."""
    import silica.kernel.recall.deferred as deferred_mod
    from silica.router.orchestrator import InjectorFSM

    # conftest's _isolate_deferred_store already points the default store at tmp.
    fsm = InjectorFSM("Inbox/lez.md", "TargetDir", hub="Hub")
    fsm.context["source_content_hash"] = "shared-hash"

    # COLLISION defers one op... (with a body: empty-payload ops are filtered out)
    assert fsm._defer_ops(
        [{"op": "write", "path": "TargetDir/A.md", "heading": "A",
          "source_basename": "lez.md", "snippet": "body A"}],
        {"A": "borderline"},
        phase="COLLISION",
    )
    # ...then VALIDATE defers another for the SAME content hash.
    assert fsm._defer_ops(
        [{"op": "write", "path": "TargetDir/B.md", "heading": "B",
          "source_basename": "lez.md", "snippet": "body B"}],
        {"TargetDir/B.md": "too generic"},
        phase="VALIDATE",
    )

    bundle = deferred_mod.get_deferred_store().get("shared-hash")
    headings = {o.get("heading") for o in bundle["rejected_ops"]}
    assert headings == {"A", "B"}, f"VALIDATE clobbered COLLISION's deferred op: {headings}"
    assert bundle["rejection_reasons"] == {"A": "borderline", "TargetDir/B.md": "too generic"}


def test_defer_ops_skips_without_content_hash(tmp_path, monkeypatch):
    """No content_hash → nothing persisted, returns False (no crash)."""
    import silica.kernel.recall.deferred as deferred_mod
    from silica.router.orchestrator import InjectorFSM

    fsm = InjectorFSM("Inbox/lez.md", "TargetDir")
    # No source_content_hash set and no per-file hashes → empty hash.
    assert fsm._defer_ops([{"op": "skip", "heading": "X"}], {}, phase="VALIDATE") is False
    assert deferred_mod.get_deferred_store().list_all() == []


def test_empty_payload_ops_are_not_deferred():
    """An op with no body re-fails identically on every verbatim retry — it must
    never land in the deferred store (0fe49a8a bundle: 7/11 ops were dead weight)."""
    from silica.router.orchestrator import InjectorFSM
    r = InjectorFSM._retryable
    assert not r({"op": "skip", "path": "x.md"})
    assert not r({"op": "write", "path": "x.md", "snippet": ""})
    assert not r({"op": "patch", "path": "x.md", "snippet": "  "})
    assert not r({"op": "overwrite", "path": "x.md", "content": None})
    assert r({"op": "write", "path": "x.md", "snippet": "real body"})
    assert r({"op": "overwrite", "path": "x.md", "content": "real body"})
    assert r({"op": "delete", "path": "x.md"})  # no body needed — retriable


def test_defer_ops_drops_a_bundle_aimed_at_a_different_target(tmp_path, monkeypatch):
    """Merging is right across the phases and chunks of one run. Across runs
    that retargeted the file it is not: a first /nucleate that mistargeted a
    book left 41 ops pointing into the inbox, and the successful re-run reported
    "34 new, 42 deferred" — reading, to the user, as half the book dropped. The
    stale ops are unretryable too: their paths are the ones VALIDATE rejects."""
    from silica.kernel.recall import deferred
    from silica.router.orchestrator import InjectorFSM

    monkeypatch.setattr(deferred, "_store_dir", lambda: tmp_path / "deferred")
    deferred._stores.clear()
    store = deferred.get_deferred_store()
    store.put("h-same", "Inbox/enoch.md", "Inbox/Book-of-Enoch", None,
              [{"op": "write", "path": "Inbox/Book-of-Enoch/Uriel.md", "heading": "Uriel"}])

    fsm = InjectorFSM("Inbox/enoch.md", "Concepts/Apocrypha")
    fsm._file_content_hashes = ["h-same"]
    fsm._chunks = []
    fsm._current_chunk_idx = 0
    fsm._defer_ops(
        [{"op": "write", "path": "Concepts/Apocrypha/Raphael.md", "heading": "Raphael",
          "snippet": "body"}],
        {"Concepts/Apocrypha/Raphael.md": "lint failed"},
        phase="VALIDATE",
    )

    ops = store.get("h-same")["rejected_ops"]
    assert [o["path"] for o in ops] == ["Concepts/Apocrypha/Raphael.md"]


def test_defer_ops_still_accumulates_within_one_target(tmp_path, monkeypatch):
    from silica.kernel.recall import deferred
    from silica.router.orchestrator import InjectorFSM

    monkeypatch.setattr(deferred, "_store_dir", lambda: tmp_path / "deferred")
    deferred._stores.clear()
    store = deferred.get_deferred_store()
    store.put("h-acc", "Inbox/enoch.md", "Concepts/Apocrypha", None,
              [{"op": "write", "path": "Concepts/Apocrypha/Uriel.md", "heading": "Uriel",
                "snippet": "body"}])

    fsm = InjectorFSM("Inbox/enoch.md", "Concepts/Apocrypha")
    fsm._file_content_hashes = ["h-acc"]
    fsm._chunks = []
    fsm._current_chunk_idx = 0
    fsm._defer_ops(
        [{"op": "write", "path": "Concepts/Apocrypha/Raphael.md", "heading": "Raphael",
          "snippet": "body"}],
        {}, phase="WRITE",
    )

    assert len(store.get("h-acc")["rejected_ops"]) == 2


# ---------------------------------------------------------------------------
# Residue facts in the bundle (verification-based residue, 2026-08-17):
# the invariant lives in the STORE so retry/anneal call sites that rewrite
# bundles with explicit field lists cannot destroy the facts.
# ---------------------------------------------------------------------------

def _ops():
    return [{"op": "write", "path": "Dir/X.md", "heading": "X"}]


def test_put_preserves_existing_residue_facts(store):
    store.put_residue_facts("h1", "inbox/a.md", "Dir", None, ["fact one"])
    # a later defer site rewrites the bundle knowing nothing of the field
    store.put("h1", "inbox/a.md", "Dir", None, _ops())
    b = store.get("h1")
    assert b["residue_facts"] == ["fact one"]
    assert len(b["rejected_ops"]) == 1


def test_put_residue_facts_merges_and_keeps_ops(store):
    store.put("h1", "inbox/a.md", "Dir", None, _ops())
    store.put_residue_facts("h1", "inbox/a.md", "Dir", None, ["fact one"])
    store.put_residue_facts("h1", "inbox/a.md", "Dir", None,
                            ["fact one", "fact two"])
    b = store.get("h1")
    assert b["residue_facts"] == ["fact one", "fact two"]  # deduped merge
    assert len(b["rejected_ops"]) == 1


def test_put_residue_facts_creates_residue_only_bundle(store):
    store.put_residue_facts("h2", "inbox/b.md", "Dir", None, ["only fact"])
    b = store.get("h2")
    assert b["residue_facts"] == ["only fact"]
    assert b["rejected_ops"] == []


def test_remove_demotes_to_residue_only_bundle(store):
    store.put("h1", "inbox/a.md", "Dir", None, _ops())
    store.put_residue_facts("h1", "inbox/a.md", "Dir", None, ["fact one"])
    assert store.remove("h1") is True  # ops cleared...
    b = store.get("h1")
    assert b is not None and b["residue_facts"] == ["fact one"]  # ...facts kept
    assert b["rejected_ops"] == []


def test_remove_without_residue_deletes(store):
    store.put("h1", "inbox/a.md", "Dir", None, _ops())
    assert store.remove("h1") is True
    assert store.get("h1") is None


def test_purge_deletes_even_with_residue(store):
    store.put_residue_facts("h1", "inbox/a.md", "Dir", None, ["fact one"])
    assert store.purge("h1") is True
    assert store.get("h1") is None


def test_list_all_reports_residue_count(store):
    store.put("h1", "inbox/a.md", "Dir", None, _ops())
    store.put_residue_facts("h1", "inbox/a.md", "Dir", None, ["f1", "f2"])
    rows = store.list_all()
    assert rows[0]["rejected_count"] == 1
    assert rows[0]["residue_count"] == 2
