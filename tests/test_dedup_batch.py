"""Family-batched ternary dedup: grouping, batch parsing, fan-out routing."""
import pytest
from unittest.mock import patch

from silica.capabilities.dedup import DedupDecision, _parse_batch, run_dedup
from silica.config import SilicaConfig
from silica.kernel.workqueue import WorkItem, batch_dedup_items


@pytest.fixture(autouse=True)
def _historical_snippet_floor(monkeypatch):
    # Predates the 100→400 write-floor raise; short fixtures here exercise
    # routing/coercion, not the length gate — pin their original floor.
    monkeypatch.setenv("SILICA_MIN_WRITE_SNIPPET_CHARS", "100")


def _item(target: str, concept: str, **ctx) -> WorkItem:
    return WorkItem(
        kind="dedup",
        target_path=target,
        context={"concept": concept, "excerpt": f"about {concept}", "score": 0.7, **ctx},
        reason="test",
    )


# ---------------------------------------------------------------------------
# batch_dedup_items
# ---------------------------------------------------------------------------

def test_batch_groups_same_candidate_passes_singletons():
    items = [
        _item("Dir/Hub.md", "A", hub="Hub", target_dir="Dir"),
        _item("Dir/Hub.md", "B", hub="Hub", target_dir="Dir"),
        _item("Dir/Other.md", "C"),
        WorkItem(kind="refine", target_path="Dir/X.md", context={}, reason="r"),
    ]
    out = batch_dedup_items(items)
    kinds = sorted((it.kind, it.target_path) for it in out)
    assert len(out) == 3
    batch = next(it for it in out if it.context.get("concepts"))
    assert batch.target_path == "Dir/Hub.md"
    assert [c["concept"] for c in batch.context["concepts"]] == ["A", "B"]
    # shared keys hoisted, per-concept keys not duplicated at top level
    assert batch.context["hub"] == "Hub"
    assert "concept" not in batch.context
    # singleton and non-dedup pass through untouched (same objects)
    assert items[2] in out and items[3] in out
    assert ("refine", "Dir/X.md") in kinds


def test_batch_caps_family_size():
    items = [_item("Dir/Hub.md", f"C{i}") for i in range(10)]
    out = batch_dedup_items(items)
    sizes = sorted(len(it.context["concepts"]) for it in out)
    assert sizes == [2, 8]


def test_batch_items_already_batched_pass_through():
    b = batch_dedup_items([_item("Dir/H.md", "A"), _item("Dir/H.md", "B")])
    assert batch_dedup_items(b) == b


# ---------------------------------------------------------------------------
# _parse_batch
# ---------------------------------------------------------------------------

def test_parse_batch_pads_truncates_and_legacy():
    raw = (
        '{"decisions": ['
        '{"verdict": "duplicate", "addition": "new info"},'
        '{"is_duplicate": true, "rationale": "legacy"}'
        "]}"
    )
    out = _parse_batch(raw, 3)
    assert [d.verdict for d in out] == ["duplicate", "duplicate", "distinct"]
    assert out[2].rationale == "missing from batch response"
    # truncation: extra entries beyond n are dropped
    assert len(_parse_batch(raw, 1)) == 1
    # garbage → all conservative defaults, never a merge
    assert all(d.verdict == "distinct" for d in _parse_batch("not json", 2))


# ---------------------------------------------------------------------------
# wire schema selection
# ---------------------------------------------------------------------------

def test_judge_schema_omits_authoring_fields_unless_spoke_requested():
    """Constrained decoding fills every key the schema declares, so title/body on
    the judge-only path made the worker re-author the note it was judging —
    ~4x the output tokens and a truncated (silently unparseable) verdict.
    """
    from types import SimpleNamespace

    from silica.capabilities.dedup import (
        DedupBatchDecision, DedupVerdict, DedupVerdictBatch,
        _decide_dedup, _decide_dedup_batch,
    )

    seen = []

    class _Provider:
        def call_llm(self, *, messages, tools, response_schema, max_tokens, **kw):
            seen.append(response_schema)
            return SimpleNamespace(text='{"verdict": "distinct", "rationale": "r"}')

    cfg = SilicaConfig()
    common = {"candidate_name": "N", "candidate_body": "body"}
    with patch("silica.agent.providers.get_provider", return_value=_Provider()):
        for spoke in (False, True):
            _decide_dedup(cfg, concept="C", excerpt="e", author_spoke=spoke, **common)
            _decide_dedup_batch(cfg, concepts=[{"concept": "C", "excerpt": "e"}],
                                author_spoke=spoke, **common)

    assert seen == [DedupVerdict, DedupVerdictBatch, DedupDecision, DedupBatchDecision]
    assert "body" not in DedupVerdict.model_fields
    assert "body" in DedupDecision.model_fields


# ---------------------------------------------------------------------------
# run_dedup on a batch item — real commit path
# ---------------------------------------------------------------------------

LONG = "Il protocollo MQTT usa un broker centrale per il routing dei messaggi " \
       "publish/subscribe tra client distribuiti su reti inaffidabili. " * 2


def test_run_batch_routes_each_verdict(tmp_vault):
    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    mqtt_abs = tmp_vault.note("Reti/MQTT.md", "# MQTT\n\nProtocollo pub/sub leggero.\n")
    vault_root = mqtt_abs.removesuffix("Reti/MQTT.md")

    [batch] = batch_dedup_items([
        _item("Reti/MQTT.md", "MQTT QoS", hub="Reti", target_dir="Reti",
              excerpt=LONG, candidate="MQTT", inbox_file="Inbox/reti.md"),
        _item("Reti/MQTT.md", "Broker MQTT", hub="Reti", target_dir="Reti",
              excerpt=LONG, candidate="MQTT", inbox_file="Inbox/reti.md"),
    ])
    decisions = [
        DedupDecision(verdict="duplicate", rationale="same thing", addition=LONG),
        DedupDecision(verdict="distinct", rationale="own topic",
                      title="Broker MQTT", body=LONG + "\n\nVedi [[MQTT]]."),
    ]
    with patch("silica.capabilities.dedup._decide_dedup_batch", return_value=decisions):
        res = run_dedup(batch, SilicaConfig())

    assert res["batch"] == 2
    assert [r["status"] for r in res["results"]] == ["committed", "committed"]
    assert res["status"] == "committed"
    assert "broker centrale" in tmp_vault.read(mqtt_abs)
    assert "Vedi [[MQTT]]" in tmp_vault.read(vault_root + res["results"][1]["spoke_path"])
    assert "followups" not in res  # authored spoke → no mechanical refine


def test_run_batch_mechanical_spoke_emits_followups(tmp_vault):
    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    mqtt_abs = tmp_vault.note("Reti/MQTT.md", "# MQTT\n\nProtocollo pub/sub leggero.\n")
    vault_root = mqtt_abs.removesuffix("Reti/MQTT.md")

    [batch] = batch_dedup_items([
        _item("Reti/MQTT.md", "QoS livelli", hub="Reti", target_dir="Reti",
              excerpt=LONG, candidate="MQTT", inbox_file="Inbox/reti.md"),
        _item("Reti/MQTT.md", "Retained msg", hub="Reti", target_dir="Reti",
              excerpt=LONG, candidate="MQTT", inbox_file="Inbox/reti.md"),
    ])
    decisions = [  # no title/body → mechanical spoke → refine follow-up each
        DedupDecision(verdict="distinct", rationale="d1"),
        DedupDecision(verdict="distinct", rationale="d2"),
    ]
    with patch("silica.capabilities.dedup._decide_dedup_batch", return_value=decisions):
        res = run_dedup(batch, SilicaConfig())

    assert [f["kind"] for f in res.get("followups", [])] == ["refine", "refine"]
    assert "[[MQTT]]" in tmp_vault.read(vault_root + res["results"][0]["spoke_path"])


# ---------------------------------------------------------------------------
# engine: followups list dispatch
# ---------------------------------------------------------------------------

def test_handle_dispatches_followups_list():
    from silica.agent.subagent import BoundedSubAgent

    seen = []

    def cap(item, config):
        return {"status": "ok", "followups": [
            {"kind": "refine", "target_path": "a.md"},
            {"kind": "refine", "target_path": "b.md"},
            {"kind": "unknown", "target_path": "c.md"},  # not in registry → skipped
        ]}

    def refine(item, config):
        seen.append(item.target_path)
        return {"status": "done"}

    agent = BoundedSubAgent(SilicaConfig(), capabilities={"x": cap, "refine": refine})
    res = agent.handle(WorkItem(kind="x", target_path="t.md", context={}, reason="r"))
    assert seen == ["a.md", "b.md"]
    assert [f["status"] for f in res["followups"]] == ["done", "done"]


def test_handle_single_followup_unchanged():
    from silica.agent.subagent import BoundedSubAgent

    def cap(item, config):
        return {"status": "ok", "followup": {"kind": "refine", "target_path": "a.md"}}

    def refine(item, config):
        return {"status": "done"}

    agent = BoundedSubAgent(SilicaConfig(), capabilities={"x": cap, "refine": refine})
    res = agent.handle(WorkItem(kind="x", target_path="t.md", context={}, reason="r"))
    assert res["followup"]["status"] == "done"
