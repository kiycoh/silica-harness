"""silica_anneal: mechanical sweep of all deferred bundles + escalation steer."""
from types import SimpleNamespace

import orjson

LONG = (
    "Il pattern publish/subscribe disaccoppia produttori e consumatori tramite "
    "un broker che smista i messaggi per topic su reti inaffidabili. " * 4
)


def _park(monkeypatch, tmp_path):
    """Point the deferred store at a temp dir and return it."""
    from silica.kernel.recall import deferred

    monkeypatch.setattr(deferred, "_store_dir", lambda: tmp_path / "deferred")
    deferred._stores.clear()
    return deferred.get_deferred_store()


# --- steer = a bounded run_agent loop (docs/specs/anneal-steer-loop.md) -------
# The LLM seam for steer is the loop's call_llm, same stub point as the other
# run_agent suites, NOT get_provider: steer no longer owns a provider call.

def _submit_resp(content_hash, ops, call_id="t1"):
    """One assistant turn calling submit_repaired_ops with `ops`."""
    args = {"content_hash": content_hash, "ops": ops}
    return SimpleNamespace(
        assistant_message={"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {
                "name": "submit_repaired_ops",
                "arguments": orjson.dumps(args).decode()}}]},
        tool_calls=[SimpleNamespace(id=call_id, name="submit_repaired_ops",
                                    args=args)],
        text=None, reasoning=None, usage={})


def _final_resp(text="done"):
    return SimpleNamespace(
        assistant_message={"role": "assistant", "content": text},
        tool_calls=[], text=text, reasoning=None, usage={})


def _steer_llm(monkeypatch, responses):
    """Script the loop's LLM; returns per-call snapshots of the history."""
    calls = []
    it = iter(responses)

    def fake(model, messages, **kw):
        calls.append([dict(m) for m in messages])
        return next(it)

    monkeypatch.setattr("silica.agent.loop.call_llm", fake)
    return calls


def test_anneal_sweeps_all_bundles(tmp_vault, tmp_path, monkeypatch):
    from silica.tools.pipeline import silica_anneal

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    # Bundle 1: write op that passes validation now → written, bundle cleared.
    store.put(
        "aaa1", "inbox/a.md", "Reti", None,
        [{"op": "write", "heading": "PubSub", "source_basename": "a.md",
          "path": "Reti/PubSub.md", "title": "PubSub", "snippet": LONG}],
        rejection_reasons={"Reti/PubSub.md": "lint failed (stale)"},
        phase="VALIDATE",
    )
    # Bundle 2: op still failing (snippet under the 100-char gate).
    store.put(
        "bbb2", "inbox/b.md", "Reti", None,
        [{"op": "write", "heading": "Stub", "source_basename": "b.md",
          "path": "Reti/Stub.md", "title": "Stub", "snippet": "troppo corto"}],
        rejection_reasons={"Reti/Stub.md": "snippet too short"},
        phase="VALIDATE",
    )

    res = silica_anneal()

    assert res["bundles"] == 2
    assert res["written"] == 1
    assert res["still_deferred"] == 1
    assert store.get("aaa1") is None          # cleared
    assert store.get("bbb2") is not None      # still parked


def test_anneal_steer_fixes_with_stamped_reason(tmp_vault, tmp_path, monkeypatch):
    from silica.tools import pipeline

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    store.put(
        "ccc3", "inbox/c.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "c.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": "corto"}],
        rejection_reasons={"Reti/Broker.md": "snippet too short"},
        phase="VALIDATE",
    )

    calls = _steer_llm(monkeypatch, [
        _submit_resp("ccc3", [{
            "op": "write", "heading": "Broker", "source_basename": "c.md",
            "path": "Reti/Broker.md", "title": "Broker", "snippet": LONG}]),
        _final_resp(),
    ])

    res = pipeline.silica_anneal(steer=True)

    [row] = res["results"]
    assert row["steer"]["status"] == "committed", row
    assert res["written"] == 1
    assert store.get("ccc3") is None  # written op removed → bundle gone
    # the stamped per-op reason reached the repair task
    assert "snippet too short" in calls[0][0]["content"]


def test_anneal_steer_iterates_on_the_validator_verdict(tmp_vault, tmp_path, monkeypatch):
    # The loop's reason to exist: the one-shot path threw the validator's
    # verdict away (1/24 ops recovered on the 2026-08-23 run). A fix rejected
    # by submit_repaired_ops must come back to the model as the tool result,
    # and the corrected resubmission must land in the SAME steer turn.
    from silica.tools import pipeline

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    store.put(
        "kkkb", "inbox/c.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "c.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": "corto"}],
        rejection_reasons={"Reti/Broker.md": "snippet too short"},
        phase="VALIDATE",
    )

    still_short = {"op": "write", "heading": "Broker", "source_basename": "c.md",
                   "path": "Reti/Broker.md", "title": "Broker", "snippet": "ancora corto"}
    fixed = dict(still_short, snippet=LONG)
    calls = _steer_llm(monkeypatch, [
        _submit_resp("kkkb", [still_short], call_id="t1"),
        _submit_resp("kkkb", [fixed], call_id="t2"),
        _final_resp(),
    ])

    res = pipeline.silica_anneal(steer=True)

    [row] = res["results"]
    assert row["steer"]["status"] == "committed", row
    assert row["steer"]["written"] == 1
    assert store.get("kkkb") is None
    # the verdict of submit #1 was in the history the model saw before #2
    tool_msgs = [m for m in calls[1] if m.get("role") == "tool"]
    assert tool_msgs, "no tool result reached the model"
    assert '"rejected"' in tool_msgs[-1]["content"]
    assert '"reason"' in tool_msgs[-1]["content"]


def test_anneal_recovered_write_is_autolinked_not_orphan(tmp_vault, tmp_path, monkeypatch):
    # The deferred path bypasses the FSM's AUTOLINK and HUB_UPDATE — recovered
    # notes used to land with zero edges and no MOC membership (audit finding
    # 2). They must get inline links AND a hub-MOC bullet now.
    from silica.tools.pipeline import silica_anneal

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    tmp_vault.note("Reti/Broker.md", "# Broker\n\nInstradatore di messaggi.\n")
    store = _park(monkeypatch, tmp_path)
    body = ("Il topic raggruppa i messaggi per argomento; il Broker li smista ai "
            "sottoscrittori interessati mantenendo il disaccoppiamento. " * 4)
    store.put(
        "ddd4", "inbox/d.md", "Reti", "Reti",
        [{"op": "write", "heading": "Topic", "source_basename": "d.md",
          "path": "Reti/Topic.md", "title": "Topic", "snippet": body}],
        phase="VALIDATE",
    )

    res = silica_anneal()
    assert res["written"] == 1

    from silica.driver import DRIVER
    content = DRIVER.read_note("Reti/Topic.md").content
    assert "[[Broker]]" in content  # inline edge to an existing sibling

    hub = DRIVER.read_note("Reti/Reti.md").content
    assert "- [[Topic]]" in hub  # MOC membership, same as the FSM path
    assert "## Da: d" in hub or "## From: d" in hub  # language-aware section


def test_anneal_retry_keeps_grounding_parity_with_persisted_payloads(tmp_vault, tmp_path, monkeypatch):
    # Finding 2 core: the retry used to re-validate with EMPTY payloads, so ops
    # rejected on payload-grounded checks (unknown heading, collision paths)
    # passed on strictly weaker validation. With the bundle's original payloads
    # persisted, the same checks run again and the op stays deferred.
    from silica.tools.pipeline import silica_deferred_retry

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    payloads = [{"batches": [{"inbox_file": "inbox/d.md", "concepts": [
        {"name": "Topic", "inbox_excerpt": "solo Topic è definito qui"},
    ]}]}]
    store.put(
        "eee5", "inbox/d.md", "Reti", "Reti",
        [{"op": "write", "heading": "Ghost", "source_basename": "d.md",
          "path": "Reti/Ghost.md", "title": "Ghost", "snippet": LONG}],
        phase="VALIDATE",
        payloads=payloads,
    )

    res = silica_deferred_retry("eee5")
    assert res.get("success") is False
    assert any("not present in payload" in r["reason"] for r in res["rejected"])
    bundle = store.get("eee5")
    assert bundle is not None                      # still parked
    assert bundle.get("payloads") == payloads      # evidence survives the re-put


def test_anneal_steer_validates_on_the_same_evidence_that_rejected(tmp_vault, tmp_path, monkeypatch):
    # Finding 2, steer edition: _steer_bundle used to validate the escalation
    # model's "fix" with EMPTY payloads, so a hallucinated op sailed through the
    # weaker gate. Measured live: a promotion bundle came back as an invented
    # encyclopedia note (in Danish) with zero facts from the source. The fix
    # must pass the bundle's persisted payloads, same as silica_deferred_retry.
    from silica.tools import pipeline

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    payloads = [{"batches": [{"inbox_file": "inbox/c.md", "concepts": [
        {"name": "Broker", "inbox_excerpt": "solo Broker è definito qui"},
    ]}]}]
    store.put(
        "ggg7", "inbox/c.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "c.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": "corto"}],
        rejection_reasons={"Reti/Broker.md": "snippet too short"},
        phase="VALIDATE",
        payloads=payloads,
    )

    # the model pivots to a concept the evidence never grounded, then gives up
    calls = _steer_llm(monkeypatch, [
        _submit_resp("ggg7", [{
            "op": "write", "heading": "Ghost", "source_basename": "c.md",
            "path": "Reti/Ghost.md", "title": "Ghost", "snippet": LONG}]),
        _final_resp("unfixable"),
    ])

    res = pipeline.silica_anneal(steer=True)

    [row] = res["results"]
    assert row["steer"]["status"] == "no_fix", row
    assert res["written"] == 0
    assert store.get("ggg7") is not None  # still parked, never written
    import pytest as _pytest
    from silica.driver import DRIVER
    with _pytest.raises(Exception):
        DRIVER.read_note("Reti/Ghost.md")
    # the gate's verdict was fed back, not swallowed
    tool_msgs = [m for m in calls[1] if m.get("role") == "tool"]
    assert any("not present in payload" in m["content"] for m in tool_msgs)
    # ...and the model's closing report survives as the row's summary: it is
    # the only trace of WHY the ops stayed deferred.
    assert row["steer"]["summary"] == "unfixable"


def test_anneal_steer_healthy_latex_survives_the_tool_transport(tmp_vault, tmp_path, monkeypatch):
    """Successor of the Body-Appendix test. The appendix existed because the
    one-shot steer turn was free text and JSON-escape decoding corrupted
    bodies ("\\top" → TAB). With ops travelling as tool arguments the
    transport marker is gone; what must hold instead is that a healthy
    single-backslash body in the args reaches the vault verbatim."""
    from silica.tools import pipeline

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    body = LONG + "\nVincolo: $\\top \\neq \\frac{1}{2}$."
    payloads = [{"batches": [{"inbox_file": "inbox/e.md", "concepts": [
        {"name": "Broker", "inbox_excerpt": "vincolo $\\top \\neq \\frac{1}{2}$"},
    ]}]}]
    store.put(
        "hhh8", "inbox/e.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "e.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": "corto"}],
        rejection_reasons={"Reti/Broker.md": "snippet too short"},
        phase="VALIDATE",
        payloads=payloads,
    )

    calls = _steer_llm(monkeypatch, [
        _submit_resp("hhh8", [{
            "op": "write", "heading": "Broker", "source_basename": "e.md",
            "path": "Reti/Broker.md", "title": "Broker", "snippet": body}]),
        _final_resp(),
    ])

    res = pipeline.silica_anneal(steer=True)

    # the old transport protocol must be gone from the prompt
    assert "===SILICA-BODY" not in calls[0][0]["content"]
    assert res["written"] == 1
    from silica.driver import DRIVER
    note = DRIVER.read_note("Reti/Broker.md").content
    assert "$\\top \\neq \\frac{1}{2}$" in note
    assert "\t" not in note


def test_anneal_steer_prompt_lists_allowed_headings(tmp_vault, tmp_path, monkeypatch):
    # The heading gate only admits headings named in the payloads, but the
    # steer model never saw that list — it re-conceptualized freely and lost
    # the whole retry to mechanical rejections (17 of 55 deferrals on the
    # 2026-08-05 run). The prompt must carry the allowed names.
    from silica.tools import pipeline

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    payloads = [{"batches": [{"inbox_file": "inbox/c.md", "concepts": [
        {"name": "Broker", "inbox_excerpt": "il Broker smista i messaggi"},
        {"name": "Topic", "inbox_excerpt": "il Topic raggruppa per argomento"},
    ]}]}]
    store.put(
        "iii9", "inbox/c.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "c.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": "corto"}],
        rejection_reasons={"Reti/Broker.md": "snippet too short"},
        phase="VALIDATE",
        payloads=payloads,
    )

    calls = _steer_llm(monkeypatch, [
        _submit_resp("iii9", [{
            "op": "write", "heading": "Broker", "source_basename": "c.md",
            "path": "Reti/Broker.md", "title": "Broker", "snippet": LONG}]),
        _final_resp(),
    ])

    res = pipeline.silica_anneal(steer=True)

    prompt = calls[0][0]["content"]
    assert "ALLOWED HEADINGS" in prompt
    assert "- Broker" in prompt and "- Topic" in prompt
    assert res["written"] == 1


def test_anneal_steer_output_gets_the_sanitize_repairs(tmp_vault, tmp_path, monkeypatch):
    # The steer path used to feed the model's JSON straight to parse_ops,
    # skipping normalize_ops entirely — over-escaped LaTeX (`\\top`, `\\{`)
    # landed verbatim in the vault (8 committed notes, 2026-08-05). Tool-arg
    # transport does not change this: the over-escape is content-level, so the
    # bundle's own excerpts still anchor the per-site collapse.
    from silica.tools import pipeline

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    payloads = [{"batches": [{"inbox_file": "inbox/c.md", "concepts": [
        {"name": "Broker", "inbox_excerpt": "vincolo $\\top$ e insieme $\\{a\\}$"},
    ]}]}]
    store.put(
        "jjja", "inbox/c.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "c.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": "corto"}],
        rejection_reasons={"Reti/Broker.md": "snippet too short"},
        phase="VALIDATE",
        payloads=payloads,
    )

    _steer_llm(monkeypatch, [
        _submit_resp("jjja", [{  # body over-escaped by the model
            "op": "write", "heading": "Broker", "source_basename": "c.md",
            "path": "Reti/Broker.md", "title": "Broker",
            "snippet": LONG + " Vincolo: $\\\\top$ su $\\\\{a\\\\}$."}]),
        _final_resp(),
    ])

    res = pipeline.silica_anneal(steer=True)
    assert res["written"] == 1

    from silica.driver import DRIVER
    note = DRIVER.read_note("Reti/Broker.md").content
    assert "$\\top$" in note and "$\\{a\\}$" in note
    assert "\\\\top" not in note and "\\\\{" not in note


def test_anneal_retry_without_payloads_keeps_legacy_behavior(tmp_vault, tmp_path, monkeypatch):
    # Old bundles (pre-schema) carry no payloads: retry still validates
    # payload-free, so they are not bricked by the schema addition.
    from silica.tools.pipeline import silica_deferred_retry

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    store.put(
        "fff6", "inbox/d.md", "Reti", "Reti",
        [{"op": "write", "heading": "Ghost", "source_basename": "d.md",
          "path": "Reti/Ghost.md", "title": "Ghost", "snippet": LONG}],
        phase="VALIDATE",
    )

    res = silica_deferred_retry("fff6")
    assert res.get("success") is True and res["written"] == 1


# --- recovered writes must be revertible and traceable (2026-08-18) ----------
# The boundary anneal runs in the FSM's `finally`, after CLEANUP has flushed
# the journal and closed the manifest. Measured on a 3-paper library gate:
# 5 of 94 notes existed on disk with no undo inverse and no provenance record —
# `/revert` walked past them and `check_renucleate` could not see them.

def test_recovered_writes_land_in_the_undo_journal(tmp_vault, tmp_path, monkeypatch):
    from silica.kernel.write.undo_journal import get_undo_journal
    from silica.tools.pipeline import silica_anneal

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    store.put(
        "ddd4", "inbox/d.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "d.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": LONG}],
        rejection_reasons={"Reti/Broker.md": "lint failed (stale)"},
        phase="VALIDATE",
    )

    assert silica_anneal()["written"] == 1

    journal = get_undo_journal()
    run_id = journal.last_active_run()
    assert run_id, "the anneal opened no journal run"
    assert "Reti/Broker.md" in {inv.path for inv, _ in journal.inverses_for(run_id)}


def test_recovered_writes_are_appended_to_provenance(tmp_vault, tmp_path, monkeypatch):
    from silica.kernel.write.provenance import read_records
    from silica.tools.pipeline import silica_anneal

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    store.put(
        "eee5", "inbox/e.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "e.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": LONG}],
        rejection_reasons={"Reti/Broker.md": "lint failed (stale)"},
        phase="VALIDATE",
    )

    assert silica_anneal()["written"] == 1

    recovered = [r for r in read_records() if r.get("sha256") == "eee5"]
    assert recovered, "no provenance record for the recovered source"
    assert any("Reti/Broker" in n for r in recovered for n in r["notes"])
    assert recovered[0]["source"] == "e.md"


# --- who owns the recovered writes (2026-08-19) -----------------------------

def _recovery_args():
    from types import SimpleNamespace

    from silica.kernel.write.ops import InverseOp, InverseOpKind, Op, OpType

    txn = SimpleNamespace(inverses=[
        InverseOp(kind=InverseOpKind.delete_created, path="Reti/PubSub.md")])
    ops = [Op(op=OpType.write, heading="PubSub", source_basename="a.md",
              path="Reti/PubSub.md", snippet=LONG)]
    return txn, ops, {"source_path": "Inbox/a.md"}


class _FakeJournal:
    def __init__(self):
        self.started, self.recorded = [], []

    def start_run(self, **kw):
        self.started.append(kw)
        return "fresh-anneal-run"

    def record(self, run_id, inverse, post_hash):
        self.recorded.append(run_id)


def test_the_boundary_anneal_rides_the_run_it_fires_inside(tmp_vault, monkeypatch):
    """The anneal runs in the FSM's `finally`, so a journal run of its own gets
    a LATER started_at — and `last_active_run` orders by that. /revert therefore
    undid the handful of recovered notes and left the whole nucleation on disk.
    The ledger id is separate and must be the FSM's progress run: that is what
    Coordinator._sweep_dangling_links matches on."""
    from silica.agent import commit as commit_mod
    from silica.kernel.write import undo_journal
    from silica.kernel.write.provenance import read_records
    from silica.tools import pipeline

    journal = _FakeJournal()
    monkeypatch.setattr(undo_journal, "get_undo_journal", lambda: journal)

    txn, ops, bundle = _recovery_args()
    undo_tok = commit_mod._current_undo_run.set("fsm-undo-run")
    ledger_tok = commit_mod._current_ledger_run.set("fsm-progress-run")
    try:
        pipeline._record_recovered_writes(txn, ops, "sha-x", bundle)
    finally:
        commit_mod._current_ledger_run.reset(ledger_tok)
        commit_mod._current_undo_run.reset(undo_tok)

    assert journal.started == [], "opened a journal run that outranks the FSM's"
    assert journal.recorded == ["fsm-undo-run"]
    rec = [r for r in read_records() if r.get("sha256") == "sha-x"]
    assert rec and rec[0]["run_id"] == "fsm-progress-run"


def test_a_standalone_retry_still_opens_its_own_revertible_unit(tmp_vault, monkeypatch):
    from silica.kernel.write import undo_journal
    from silica.kernel.write.provenance import read_records
    from silica.tools import pipeline

    journal = _FakeJournal()
    monkeypatch.setattr(undo_journal, "get_undo_journal", lambda: journal)

    txn, ops, bundle = _recovery_args()
    pipeline._record_recovered_writes(txn, ops, "sha-y", bundle)

    assert journal.started and journal.started[0]["source"] == "anneal"
    assert journal.recorded == ["fresh-anneal-run"]
    rec = [r for r in read_records() if r.get("sha256") == "sha-y"]
    assert rec and rec[0]["run_id"] == "anneal"


# --- steer-loop seam guarantees (docs/specs/anneal-steer-loop.md) ------------

def test_submit_repaired_ops_is_internal_and_keeps_its_verdict(tmp_vault):
    # internal: reachable only when AgentConstraints names it, never from chat.
    # lazy collapse: the verdict IS the feedback — an eager stub would erase
    # the rejection reasons from the very history the model iterates on.
    from silica.tools import TOOLS

    t = TOOLS["submit_repaired_ops"]
    assert t.internal is True
    assert t.collapse == "lazy"


def test_anneal_steer_commits_are_journaled_and_provenanced(tmp_vault, tmp_path, monkeypatch):
    # The old steer commit path called NEITHER _link_recovered_writes NOR
    # _record_recovered_writes: steered notes were invisible to /revert and to
    # provenance (the same 2026-08-18 defect the mechanical retry already
    # fixed). The tool path must do both.
    from silica.kernel.write.provenance import read_records
    from silica.kernel.write.undo_journal import get_undo_journal
    from silica.tools import pipeline

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    store.put(
        "mmmc", "inbox/c.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "c.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": "corto"}],
        rejection_reasons={"Reti/Broker.md": "snippet too short"},
        phase="VALIDATE",
    )

    _steer_llm(monkeypatch, [
        _submit_resp("mmmc", [{
            "op": "write", "heading": "Broker", "source_basename": "c.md",
            "path": "Reti/Broker.md", "title": "Broker", "snippet": LONG}]),
        _final_resp(),
    ])

    assert pipeline.silica_anneal(steer=True)["written"] == 1

    journal = get_undo_journal()
    run_id = journal.last_active_run()
    assert run_id, "the steer commit opened no journal run"
    assert "Reti/Broker.md" in {inv.path for inv, _ in journal.inverses_for(run_id)}

    recovered = [r for r in read_records() if r.get("sha256") == "mmmc"]
    assert recovered, "no provenance record for the steered source"
    assert recovered[0]["source"] == "c.md"


def test_anneal_steer_reports_a_dead_loop_and_keeps_the_bundle(tmp_vault, tmp_path, monkeypatch):
    # An escalation endpoint with broken tool-calling (the Ollama seam) must
    # surface as an error row, never as a silent no_fix, and must not consume
    # the bundle: /anneal --steer is rerunnable once the endpoint is fixed.
    from silica.tools import pipeline

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    store.put(
        "nnnd", "inbox/c.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "c.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": "corto"}],
        rejection_reasons={"Reti/Broker.md": "snippet too short"},
        phase="VALIDATE",
    )

    def boom(model, messages, **kw):
        raise RuntimeError("tool-calling unsupported")

    monkeypatch.setattr("silica.agent.loop.call_llm", boom)

    res = pipeline.silica_anneal(steer=True)

    [row] = res["results"]
    assert row["steer"]["status"] == "error", row
    assert "tool-calling unsupported" in row["steer"]["error"]
    assert row["steer"]["written"] == 0
    assert store.get("nnnd") is not None  # still parked, rerunnable


def test_anneal_steer_cost_is_bounded_by_the_iteration_cap(tmp_vault, tmp_path, monkeypatch):
    # The economic promise of the loop: a model that NEVER converges (garbage
    # resubmits, rejection verdicts every round — no error, so the convergence
    # guard stays silent) costs exactly max_iterations tool rounds plus ONE
    # tool-less landing call, then the sweep moves on with the bundle intact.
    from silica.tools import pipeline

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    store.put(
        "oooe", "inbox/c.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "c.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": "corto"}],
        rejection_reasons={"Reti/Broker.md": "snippet too short"},
        phase="VALIDATE",
    )
    monkeypatch.setenv("ANNEAL_STEER_ITERATIONS", "3")

    tools_per_call = []

    def relentless(model, messages, **kw):
        tools_per_call.append(kw.get("tools"))
        return _submit_resp("oooe", [{
            "op": "write", "heading": "Broker", "source_basename": "c.md",
            "path": "Reti/Broker.md", "title": "Broker", "snippet": "ancora corto"}],
            call_id=f"t{len(tools_per_call)}")

    monkeypatch.setattr("silica.agent.loop.call_llm", relentless)

    res = pipeline.silica_anneal(steer=True)

    [row] = res["results"]
    assert row["steer"]["status"] == "no_fix", row
    assert row["steer"]["written"] == 0
    assert store.get("oooe") is not None      # bundle intact, re-annealable
    assert len(tools_per_call) == 4           # 3 tool rounds + 1 landing
    assert all(t for t in tools_per_call[:3])  # tools offered while in budget
    assert tools_per_call[-1] is None         # landing call: tools removed
    # the landing produced no real text, so no summary is fabricated
    assert "summary" not in row["steer"]


def test_submit_repaired_ops_names_a_renamed_write_instead_of_looping(tmp_vault, tmp_path, monkeypatch):
    # A payload-less bundle has no heading gate, so a model that renames a
    # heading writes a REAL note the bundle cannot account for: remove_op
    # matches by heading and the parked original stays (idempotent re-anneal,
    # the safe direction). The verdict must NAME the rename, or the model
    # reads the unchanged `remaining` as a failed write and resubmits the
    # same op until the cap.
    import json

    from silica.tools import TOOLS

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    store.put(
        "pppf", "inbox/c.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "c.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": "corto"}],
        rejection_reasons={"Reti/Broker.md": "snippet too short"},
        phase="VALIDATE",
    )

    res = json.loads(TOOLS["submit_repaired_ops"].run(
        content_hash="pppf",
        ops=[{"op": "write", "heading": "Mediatore", "source_basename": "c.md",
              "path": "Reti/Mediatore.md", "title": "Mediatore", "snippet": LONG}]))

    assert res["written"] == ["Mediatore"]
    assert res["renamed"] == ["Mediatore"]
    assert res["remaining"] == 1              # the parked original is untouched
    assert store.get("pppf") is not None
    from silica.driver import DRIVER
    assert "Mediatore" in DRIVER.read_note("Reti/Mediatore.md").content
