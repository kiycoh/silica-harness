"""Tests for capability dispatch (silica/agent/subagent.py) and the per-capability
behaviours (silica/capabilities/*).

Dispatch is a keyed lookup: ``BoundedSubAgent.handle()`` selects the capability
registered under ``item.kind`` and runs it. Each behaviour is a plain
``run(item, config) -> dict`` function in its own module, and its LLM-decision
seam is a module-level function the tests patch directly.
"""
from unittest.mock import patch, MagicMock

from silica.agent.subagent import BoundedSubAgent
from silica.capabilities.dedup import run_dedup, DedupDecision
from silica.capabilities.refine import run_refine
from silica.capabilities.enrich import run_enrich
from silica.capabilities.orphan import run_orphan, OrphanLinkDecision
from silica.capabilities._base import NoteContent
from silica.config import SilicaConfig
from silica.kernel.write.ops import OpType
from silica.kernel.workqueue import WorkItem

CONFIG = SilicaConfig()


# --- dispatch --------------------------------------------------------------

def test_handle_dispatches_to_capability_by_kind():
    """handle() routes to the capability registered under item.kind."""
    seen = {}

    def fake_run(item, config):
        seen["called"] = item.kind
        return {"status": "ok"}

    agent = BoundedSubAgent(CONFIG, capabilities={"mystery": fake_run})
    res = agent.handle(WorkItem(kind="mystery", target_path="X.md"))
    assert res == {"status": "ok"}
    assert seen["called"] == "mystery"


def test_handle_skips_unknown_kind():
    agent = BoundedSubAgent(CONFIG, capabilities={})
    res = agent.handle(WorkItem(kind="nope", target_path="X.md"))
    assert res["status"] == "skipped"


def test_handle_catches_capability_errors():
    def boom(item, config):
        raise RuntimeError("kaboom")

    agent = BoundedSubAgent(CONFIG, capabilities={"boom": boom})
    res = agent.handle(WorkItem(kind="boom", target_path="X.md"))
    assert res["status"] == "error"
    assert "kaboom" in res["error"]


# --- dedup behaviour -------------------------------------------------------

def _item():
    return WorkItem(
        kind="dedup",
        target_path="Concepts/Gradient Descent.md",
        context={
            "concept": "Discesa del gradiente",
            "excerpt": "Variante mini-batch con momentum.",
            "candidate": "Gradient Descent",
            "inbox_file": "Inbox/ml.md",
            "hub": "Concepts",
        },
        reason="borderline_similarity score=0.78",
    )


def test_dedup_merge_builds_single_patch_under_leash():
    decision = DedupDecision(verdict="duplicate", rationale="same concept", addition="### Momentum\nNew info.")

    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="existing body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = run_dedup(_item(), CONFIG)

    assert res["status"] == "committed"
    # The merge itself: exactly one patch op + a dedup leash on the candidate.
    # It is call [0] because a duplicate verdict now also records the absorbed
    # concept's spelling as an alias, in a second, separately-leashed write.
    merge = commit.call_args_list[0]
    ops_arg = merge.args[0]
    assert len(ops_arg) == 1
    assert ops_arg[0].op == OpType.patch
    assert ops_arg[0].path == "Concepts/Gradient Descent.md"
    bounds = merge.kwargs["bounds"]
    assert bounds.name == "dedup"
    assert OpType.patch in bounds.allowed_ops and OpType.overwrite not in bounds.allowed_ops

    # The alias write never travels on the merge's leash: its own envelope
    # permits overwrite, and only of the note that absorbed the concept.
    alias = commit.call_args_list[1]
    assert alias.args[0][0].op == OpType.overwrite
    assert alias.kwargs["bounds"].name == "dedup_alias"
    assert alias.kwargs["bounds"].allowed_ops == frozenset({OpType.overwrite})


def test_dedup_no_merge_when_not_duplicate():
    decision = DedupDecision(verdict="distinct", rationale="different topics")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops") as commit:
        res = run_dedup(_item(), CONFIG)
    assert res["status"] == "no_merge"
    commit.assert_not_called()


def test_dedup_no_merge_when_addition_empty():
    decision = DedupDecision(verdict="duplicate", rationale="dup but nothing new", addition="   ")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops") as commit:
        res = run_dedup(_item(), CONFIG)
    assert res["status"] == "no_merge"
    commit.assert_not_called()


def test_dedup_contradicts_builds_contested_patch():
    """Third verdict: the conflicting claim lands as ONE contested patch op —
    warning callout in the snippet, contested_by set for the frontmatter mark.

    The note is agent-written so the contest genuinely stays open: §6.1-bis
    files the claim instead when the note strictly outranks it, which the next
    test covers. Every mocked read returns this body, the source leaf included,
    so the incoming side ranks grounded and nothing is suppressed.
    """
    decision = DedupDecision(
        verdict="contradicts",
        rationale="conflicting dosage",
        addition="Il dosaggio raccomandato è 50mg/die.",
    )
    agent_note = "---\nAI: true\n---\n\nexisting body"
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content=agent_note)), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = run_dedup(_item(), CONFIG)

    assert res["status"] == "committed"
    assert res["verdict"] == "contradicts"
    ops_arg = commit.call_args.args[0]
    assert len(ops_arg) == 1
    op = ops_arg[0]
    assert op.op == OpType.patch
    assert op.path == "Concepts/Gradient Descent.md"
    assert op.contested_by == "source: ml.md"
    assert op.snippet.startswith("> [!warning]")
    assert "50mg/die" in op.snippet
    # Same leash as the merge path: the model never escalates beyond a patch.
    bounds = commit.call_args.kwargs["bounds"]
    assert bounds.name == "dedup"


def test_dedup_contradicts_files_a_claim_the_note_outranks():
    """§6.1-bis: the op shape of a suppressed contest, and its leash.

    An overwrite rather than a patch because the block goes INSIDE `## Superseded`
    and patches land above it by construction; `base_content` is what makes a
    concurrent edit collide instead of being stomped.
    """
    from silica.kernel.write.ops import OpType

    decision = DedupDecision(
        verdict="contradicts",
        rationale="conflicting dosage",
        addition="Il dosaggio raccomandato è 50mg/die.",
    )
    # Human note, and no source leaf on disk: the incoming claim is distilled
    # and undated, so nothing suggests it is the fresher side.
    def _read(ref, *a, **k):
        path = getattr(ref, "path", ref)
        if "sources/" in str(path):
            raise FileNotFoundError(path)
        return MagicMock(content="# Gradient Descent\n\nexisting body")

    with patch("silica.driver.DRIVER.read_note", side_effect=_read), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops",
               return_value={"status": "committed", "committed": 1}) as commit:
        res = run_dedup(_item(), CONFIG)

    assert res["status"] == "committed"
    assert res["superseded"] is True
    op = commit.call_args.args[0][0]
    assert op.op == OpType.overwrite
    assert op.path == "Concepts/Gradient Descent.md"
    assert op.base_content == "# Gradient Descent\n\nexisting body"
    assert "## Superseded" in op.content
    assert "50mg/die" in op.content.split("## Superseded", 1)[1]
    assert "existing body" in op.content.split("## Superseded", 1)[0]
    bounds = commit.call_args.kwargs["bounds"]
    assert bounds.name == "dedup_supersede"
    assert bounds.allowed_ops == frozenset({OpType.overwrite})


def test_dedup_contradicts_without_claim_is_no_merge():
    """A contradiction verdict with no quoted claim is unactionable — never write."""
    decision = DedupDecision(verdict="contradicts", rationale="conflict", addition="  ")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops") as commit:
        res = run_dedup(_item(), CONFIG)
    assert res["status"] == "no_merge"
    commit.assert_not_called()


def _seed_twin_bundle(content_hash: str = "hash1"):
    """Seed the (conftest-isolated) deferred store with the borderline bundle
    COLLISION would have written for _item()'s concept plus one sibling op."""
    from silica.kernel.recall.deferred import get_deferred_store

    store = get_deferred_store()
    store.put(
        content_hash, "Inbox/ml.md", "Concepts", "Concepts",
        [
            {"op": "write", "heading": "Discesa del gradiente",
             "path": "Concepts/Discesa del gradiente.md",
             "snippet": "Variante mini-batch con momentum.",
             "reason": "collision_deferred score=0.780 candidate=Gradient Descent"},
            {"op": "write", "heading": "Altra cosa",
             "path": "Concepts/Altra cosa.md", "snippet": "corpo",
             "reason": "collision_deferred score=0.700 candidate=X"},
        ],
        {"Discesa del gradiente": "borderline_similarity score=0.780",
         "Altra cosa": "borderline_similarity score=0.700"},
    )
    return store


def _item_with_hash(content_hash: str = "hash1"):
    item = _item()
    item.context["content_hash"] = content_hash
    item.context["target_dir"] = "Concepts"
    return item


def test_dedup_duplicate_commit_cleans_twin_bundle():
    """C2 verdict routing: a committed merge removes the concept's op from the
    deferred twin bundle; sibling ops survive."""
    store = _seed_twin_bundle()
    decision = DedupDecision(verdict="duplicate", rationale="same", addition="### New\ninfo.")

    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops", return_value={"status": "committed", "committed": 1}):
        res = run_dedup(_item_with_hash(), CONFIG)

    assert res["status"] == "committed"
    bundle = store.get("hash1")
    headings = [o["heading"] for o in bundle["rejected_ops"]]
    assert headings == ["Altra cosa"]


def test_dedup_failed_commit_keeps_twin_bundle():
    """Bundle cleaned only on verified commit: a rolled-back merge must leave
    the deferred op in place — the op degrades, it is never lost."""
    store = _seed_twin_bundle()
    decision = DedupDecision(verdict="duplicate", rationale="same", addition="### New\ninfo.")

    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops", return_value={"status": "rolled_back", "committed": 0}):
        run_dedup(_item_with_hash(), CONFIG)

    headings = {o["heading"] for o in store.get("hash1")["rejected_ops"]}
    assert "Discesa del gradiente" in headings


def test_dedup_distinct_authors_wikilinked_spoke():
    """C2 fork (giudice+autore): pipeline distinct — the verdict call also
    authored the spoke; it is committed as ONE write op under write-only
    bounds, wikilinked to the candidate, and the twin bundle is cleaned."""
    store = _seed_twin_bundle()
    decision = DedupDecision(
        verdict="distinct", rationale="related but distinct",
        title="Discesa del gradiente",
        body="La variante mini-batch aggiorna i pesi per sottoinsiemi.\n\nVedi [[Gradient Descent]].",
    )

    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = run_dedup(_item_with_hash(), CONFIG)

    assert res["status"] == "committed"
    assert res["verdict"] == "distinct"
    ops_arg = commit.call_args.args[0]
    assert len(ops_arg) == 1
    op = ops_arg[0]
    assert op.op == OpType.write
    assert op.path == "Concepts/Discesa del gradiente.md"
    assert "[[Gradient Descent]]" in op.snippet
    bounds = commit.call_args.kwargs["bounds"]
    assert OpType.write in bounds.allowed_ops
    assert OpType.patch not in bounds.allowed_ops and OpType.overwrite not in bounds.allowed_ops
    headings = [o["heading"] for o in store.get("hash1")["rejected_ops"]]
    assert headings == ["Altra cosa"], "twin bundle op must be routed away"


def test_dedup_distinct_authoring_failure_falls_back_mechanical():
    """Authoring failed (no title/body) → the excerpt lands verbatim with a
    provenance block and the candidate wikilink, a refine follow-up is
    proposed, and the bundle is cleaned — the op degrades, it is never lost."""
    store = _seed_twin_bundle()
    decision = DedupDecision(verdict="distinct", rationale="related")  # nothing authored

    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = run_dedup(_item_with_hash(), CONFIG)

    assert res["status"] == "committed"
    op = commit.call_args.args[0][0]
    assert op.op == OpType.write
    assert "Variante mini-batch con momentum." in op.snippet  # excerpt verbatim
    assert "(from ml.md)" in op.snippet                          # provenance
    assert "[[Gradient Descent]]" in op.snippet                # born linked
    assert res["followup"]["kind"] == "refine"                 # ADR-0001 deferred refine
    assert res["followup"]["target_path"] == op.path
    headings = [o["heading"] for o in store.get("hash1")["rejected_ops"]]
    assert headings == ["Altra cosa"]


def test_dedup_distinct_failed_spoke_commit_keeps_bundle_and_skips_refine():
    """A rolled-back spoke write must leave the parked op in the bundle and
    propose no follow-up refine."""
    store = _seed_twin_bundle()
    decision = DedupDecision(verdict="distinct", rationale="related")

    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops", return_value={"status": "rolled_back", "committed": 0}):
        res = run_dedup(_item_with_hash(), CONFIG)

    assert "followup" not in res
    headings = {o["heading"] for o in store.get("hash1")["rejected_ops"]}
    assert "Discesa del gradiente" in headings


def test_handle_dispatches_followup_through_registry():
    """The engine — not the capability — runs a proposed follow-up (P9: workers
    are peers), and only one hop deep: a follow-up's follow-up is ignored."""
    seen = []

    def primary(item, config):
        return {"status": "committed",
                "followup": {"kind": "polish", "target_path": "Dir/Spoke.md",
                             "context": {"hub": "H"}}}

    def polish(item, config):
        seen.append(item)
        return {"status": "committed",
                "followup": {"kind": "polish", "target_path": "loop.md"}}

    agent = BoundedSubAgent(CONFIG, capabilities={"primary": primary, "polish": polish})
    res = agent.handle(WorkItem(kind="primary", target_path="X.md"))

    assert len(seen) == 1, "exactly one follow-up hop"
    assert seen[0].target_path == "Dir/Spoke.md"
    assert seen[0].context == {"hub": "H"}
    assert res["followup"]["status"] == "committed"


def test_dedup_distinct_without_target_dir_stays_no_merge():
    """Ad-hoc /dedup pairs (two existing notes, no target_dir in context) keep
    today's contract: distinct → no write, no spoke."""
    decision = DedupDecision(verdict="distinct", rationale="different topics")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops") as commit:
        res = run_dedup(_item(), CONFIG)
    assert res["status"] == "no_merge"
    commit.assert_not_called()


def _decide_with_raw(raw_text: str):
    """Run _decide_dedup with a provider whose response is `raw_text`."""
    from silica.capabilities.dedup import _decide_dedup

    provider = MagicMock()
    provider.call_llm.return_value = MagicMock(text=raw_text)
    with patch("silica.agent.providers.get_provider", return_value=provider):
        return _decide_dedup(
            CONFIG, concept="X", excerpt="e", candidate_name="Y", candidate_body="b",
        )


def test_decide_dedup_unparseable_defaults_to_distinct():
    """Conservative default: garbage output must never become a contradicts/merge."""
    decision = _decide_with_raw("non-JSON garbage")
    assert decision.verdict == "distinct"


def test_decide_dedup_unknown_verdict_defaults_to_distinct():
    decision = _decide_with_raw('{"verdict": "maybe", "rationale": "", "addition": "x"}')
    assert decision.verdict == "distinct"


def test_decide_dedup_parses_authored_spoke_fields():
    decision = _decide_with_raw(
        '{"verdict": "distinct", "rationale": "r", "addition": "", '
        '"title": "Spoke", "body": "corpo [[X]]"}'
    )
    assert decision.title == "Spoke"
    assert decision.body == "corpo [[X]]"


def test_decide_dedup_parses_contradicts():
    decision = _decide_with_raw(
        '{"verdict": "contradicts", "rationale": "r", "addition": "claim"}'
    )
    assert decision.verdict == "contradicts"
    assert decision.addition == "claim"


def test_decide_dedup_legacy_is_duplicate_still_maps():
    """A model answering with the old binary schema degrades gracefully."""
    assert _decide_with_raw('{"is_duplicate": true, "addition": "x"}').verdict == "duplicate"
    assert _decide_with_raw('{"is_duplicate": false}').verdict == "distinct"


def test_unreadable_candidate_is_skipped():
    with patch("silica.driver.DRIVER.read_note", side_effect=RuntimeError("missing")):
        res = run_dedup(_item(), CONFIG)
    assert res["status"] == "skipped"


# --- refine behaviour ------------------------------------------------------

def _refine_item():
    return WorkItem(kind="refine", target_path="Notes/Target.md", context={"hub": "Concepts"})


def test_refine_builds_overwrite_under_refiner_bounds():
    refined = NoteContent(content="# Target\n\n> [!note]\nBody with [[Link]].")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="old body [[Link]]")), \
         patch("silica.capabilities.refine._refine_note", return_value=refined), \
         patch("silica.capabilities._base.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = run_refine(_refine_item(), CONFIG)
    assert res["status"] == "committed"
    ops_arg = commit.call_args.args[0]
    assert ops_arg[0].op == OpType.overwrite
    bounds = commit.call_args.kwargs["bounds"]
    assert bounds.name == "refiner"
    assert bounds.content_guard is not None  # anti-info-loss enforced


def test_refine_skips_empty_note():
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="   ")):
        res = run_refine(_refine_item(), CONFIG)
    assert res["status"] == "skipped"


# --- orphan connector behaviour --------------------------------------------

def _orphan_item():
    return WorkItem(
        kind="orphan",
        target_path="Notes/Lonely.md",
        context={"candidates": [
            {"name": "Gradient Descent", "path": "Concepts/Gradient Descent"},
            {"name": "Backprop", "path": "Concepts/Backprop"},
        ]},
        reason="residual_orphan",
    )


def _run_orphan_capturing(item, decision):
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="orphan body")), \
         patch("silica.capabilities.orphan._decide_links", return_value=decision), \
         patch("silica.capabilities.orphan.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = run_orphan(item, CONFIG)
    return res, commit


def test_orphan_links_only_to_offered_candidates():
    # Model returns one valid candidate + one hallucinated name.
    decision = OrphanLinkDecision(links=["Gradient Descent", "Made Up Note"], rationale="related")
    res, commit = _run_orphan_capturing(_orphan_item(), decision)
    assert res["status"] == "committed"
    ops = commit.call_args.args[0]
    # Hallucinated target filtered out: one op, for the one offered candidate.
    assert len(ops) == 1
    assert ops[0].op == OpType.patch
    assert commit.call_args.kwargs["bounds"].name == "orphan"


def test_orphan_patches_the_neighbour_not_the_orphan():
    """The link must land in the neighbour and point AT the orphan.

    An orphan is a note with in-degree 0, so a link written into the orphan
    itself only adds out-degree: it leaves the note orphaned and the `orphans`
    term of E(vault) unmoved. This is the regression the 2026-08-14 flip fixes.
    """
    decision = OrphanLinkDecision(links=["Gradient Descent"], rationale="related")
    _, commit = _run_orphan_capturing(_orphan_item(), decision)
    op = commit.call_args.args[0][0]
    assert op.path == "Concepts/Gradient Descent.md", "patch must target the neighbour"
    assert op.path != "Notes/Lonely.md"
    assert "[[Lonely]]" in op.snippet, "the link must point back at the orphan"
    assert "Gradient Descent" not in op.snippet


def test_orphan_completes_the_md_suffix_on_candidate_paths():
    """Candidate paths arrive in the relatedness keyspace (cooccur_key, no '.md').

    The driver reads either spelling, but the op path is what gets WRITTEN, so a
    bare key would patch a file that does not exist.
    """
    item = WorkItem(
        kind="orphan",
        target_path="Notes/Lonely.md",
        context={"candidates": [
            {"name": "Bare", "path": "Concepts/Bare"},          # keyspace form
            {"name": "Suffixed", "path": "Concepts/Suffixed.md"},  # already a path
        ]},
        reason="residual_orphan",
    )
    decision = OrphanLinkDecision(links=["Bare", "Suffixed"], rationale="both")
    _, commit = _run_orphan_capturing(item, decision)
    paths = [o.path for o in commit.call_args.args[0]]
    assert paths == ["Concepts/Bare.md", "Concepts/Suffixed.md"]


def test_orphan_provenance_key_is_per_orphan():
    """A popular neighbour is the best candidate for many orphans, and the patch
    executor skips a re-append keyed by (heading, source_basename). A flat
    "orphan" key would let the first orphan land and silently drop the rest."""
    def key_for(orphan_path):
        item = WorkItem(
            kind="orphan",
            target_path=orphan_path,
            context={"candidates": [{"name": "Hub", "path": "Concepts/Hub.md"}]},
            reason="residual_orphan",
        )
        _, commit = _run_orphan_capturing(
            item, OrphanLinkDecision(links=["Hub"], rationale="r")
        )
        return commit.call_args.args[0][0].source_basename

    assert key_for("Notes/First.md") != key_for("Notes/Second.md")


def test_orphan_skips_candidates_outside_the_write_boundary():
    """Same boundary backlink_pass enforces: this rewrites pre-existing notes
    anywhere in the vault, so on a vault that reads a whole source tree it must
    never edit a note outside write_dir."""
    item = WorkItem(
        kind="orphan",
        target_path="docs/silica/Lonely.md",
        context={"candidates": [{"name": "Readme", "path": "README.md"}]},
        reason="residual_orphan",
    )
    decision = OrphanLinkDecision(links=["Readme"], rationale="related")
    with patch("silica.capabilities.orphan.active_write_dir", return_value="docs/silica"):
        res, commit = _run_orphan_capturing(item, decision)
    assert res["status"] == "no_link"
    commit.assert_not_called()


def test_orphan_no_link_when_model_picks_nothing_valid():
    decision = OrphanLinkDecision(links=["Nonexistent"], rationale="nothing fits")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.orphan._decide_links", return_value=decision), \
         patch("silica.capabilities.orphan.commit_ops") as commit:
        res = run_orphan(_orphan_item(), CONFIG)
    assert res["status"] == "no_link"
    commit.assert_not_called()


def test_orphan_no_candidates():
    res = run_orphan(WorkItem(kind="orphan", target_path="X.md", context={"candidates": []}), CONFIG)
    assert res["status"] == "no_candidates"


def test_orphan_hub_is_none_when_context_has_no_hub():
    """When context has no hub key, hub must be None (not basename of target_path)."""
    import silica.capabilities.orphan as orphan_module
    from silica.agent.bounds import orphan_bounds as real_orphan_bounds

    item = WorkItem(
        kind="orphan",
        target_path="notes/MyNote.md",
        context={"candidates": [{"name": "Other", "path": "notes/Other.md"}]},
        reason="test",
    )

    captured_hubs = []

    def capture_orphan_bounds(neighbours, *, orphan_title, hub):
        captured_hubs.append(hub)
        return real_orphan_bounds(neighbours, orphan_title=orphan_title, hub=hub)

    with patch.object(orphan_module, "orphan_bounds", side_effect=capture_orphan_bounds), \
         patch("silica.capabilities.orphan.commit_ops", return_value={"status": "no_ops"}), \
         patch("silica.capabilities.orphan._decide_links", return_value=OrphanLinkDecision(links=["Other"], rationale="test")), \
         patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="# MyNote\n")):
        run_orphan(item, CONFIG)

    assert captured_hubs == [None], f"Expected hub=None when context has no 'hub' key, got {captured_hubs}"


# --- provenance attribution of work items (2026-08-18) ----------------------

def test_item_provenance_reads_a_batched_dedup_context():
    """batch_dedup_items moves `inbox_file` into the per-concept entries (it is
    in _BATCH_CONCEPT_KEYS), so the shared context of a family batch has no
    source at all. Shape copied verbatim from a real run's workqueue.json."""
    from types import SimpleNamespace
    from silica.agent.subagent import _item_provenance

    item = SimpleNamespace(context={
        "candidate": "Single Module",
        "concepts": [
            {"concept": "partitions", "excerpt": "…",
             "inbox_file": "docs/silica/Inbox/physics_0607100v2/2-vi-conclusions.md"},
            {"concept": "modularity", "excerpt": "…",
             "inbox_file": "docs/silica/Inbox/physics_0607100v2/2-vi-conclusions.md"},
        ],
        "content_hash": "5f7e4c65",
        "hub": "Papers",
        "run_id": "run-77",
        "target_dir": "docs/silica/Papers",
    })
    assert _item_provenance(item) == ("2-vi-conclusions.md", "5f7e4c65", "run-77")


def test_item_provenance_is_none_without_a_source():
    from types import SimpleNamespace
    from silica.agent.subagent import _item_provenance

    assert _item_provenance(SimpleNamespace(context={"content_hash": "x"})) is None
    assert _item_provenance(SimpleNamespace(context={})) is None


def test_dedup_items_carry_the_chunk_s_own_file_not_file_zero():
    """`fsm.inbox_file` is pinned to inbox_files[0] for the whole run while
    `_current_content_hash` resolves per chunk, so on a folder nucleation every
    worker's notes were filed under file 0 paired with another file's sha —
    source 0 ends up with records at two sha values, which check_renucleate
    reads as "modified" and drifted_notes reads as derived from a superseded
    version."""
    from types import SimpleNamespace

    from silica.agent.subagent import _item_provenance
    from silica.router.states.distill import _enqueue_near_title_dedups

    class _Q:
        def __init__(self): self.items = []
        def enqueue(self, item): self.items.append(item)

    q = _Q()
    fsm = SimpleNamespace(
        work_queue=q,
        inbox_files=["Inbox/a.md", "Inbox/b.md", "Inbox/c.md"],
        inbox_file="Inbox/a.md",          # pinned to files[0], forever
        _current_source_file="Inbox/c.md",  # the chunk actually being distilled
        _current_content_hash="sha-of-c",
        hub="Hub", target_dir="Target",
        progress=SimpleNamespace(run_id="run-9"),
    )
    _enqueue_near_title_dedups(fsm, [{
        "op": {"heading": "Beta", "snippet": "beta body"},
        "reason": "near_title candidate='Betas' path='Target/Betas.md' "
                  "ratio=0.90 — deferred for dedup review",
    }])

    assert len(q.items) == 1
    assert _item_provenance(q.items[0]) == ("c.md", "sha-of-c", "run-9")


def test_expand_items_carry_the_same_triple():
    from types import SimpleNamespace

    from silica.agent.subagent import _item_provenance
    from silica.router.states.distill import _enqueue_short_snippet_expands

    class _Q:
        def __init__(self): self.items = []
        def enqueue(self, item): self.items.append(item)

    q = _Q()
    fsm = SimpleNamespace(
        work_queue=q, inbox_file="Inbox/a.md", _current_source_file="Inbox/c.md",
        _current_content_hash="sha-of-c", hub="Hub", target_dir="Target",
        progress=SimpleNamespace(run_id="run-9"),
        _chunks=[{"batches": [{"concepts": [{"name": "Beta", "inbox_excerpt": "…"}]}]}],
        _current_chunk_idx=0,
    )
    _enqueue_short_snippet_expands(fsm, [
        {"op": {"heading": "Beta", "path": "Target/Beta.md"},
         "reason": "snippet too short (12 < 100)"},
    ])

    assert len(q.items) == 1
    assert _item_provenance(q.items[0]) == ("c.md", "sha-of-c", "run-9")


def _decide_capturing_call(raw_text: str) -> dict:
    """Run _decide_dedup and return the kwargs the provider was called with."""
    from silica.capabilities.dedup import _decide_dedup

    provider = MagicMock()
    provider.call_llm.return_value = MagicMock(text=raw_text, finish_reason="stop")
    with patch("silica.agent.providers.get_provider", return_value=provider):
        _decide_dedup(CONFIG, concept="X", excerpt="e", candidate_name="Y", candidate_body="b")
    return provider.call_llm.call_args.kwargs


def test_decide_dedup_switches_thinking_off():
    """The judge's output is a few hundred tokens of JSON; with thinking on, a
    hybrid model spends the 2048 budget on its trace and the JSON arrives cut
    (finish=length, "Unterminated string", 2026-08-23 run). The trace had
    concluded "duplicate"; the parse fallback then wrote "distinct"."""
    assert _decide_capturing_call('{"verdict": "distinct", "rationale": "r"}')["reasoning"] is False


def test_decide_dedup_batch_switches_thinking_off():
    from silica.capabilities.dedup import _decide_dedup_batch

    provider = MagicMock()
    provider.call_llm.return_value = MagicMock(text='{"decisions": []}', finish_reason="stop")
    with patch("silica.agent.providers.get_provider", return_value=provider):
        _decide_dedup_batch(
            CONFIG, concepts=[{"concept": "X", "excerpt": "e"}],
            candidate_name="Y", candidate_body="b",
        )
    assert provider.call_llm.call_args.kwargs["reasoning"] is False


def test_decide_dedup_unparseable_is_not_a_judgement():
    """Garbage output routes conservatively (distinct, no merge) but must say
    so: the verdict is a default, and downstream must not record it as judged."""
    decision = _decide_with_raw("non-JSON garbage")
    assert decision.verdict == "distinct"
    assert decision.judged is False


def test_decide_dedup_parsed_verdict_is_judged():
    assert _decide_with_raw('{"verdict": "duplicate", "rationale": "same"}').judged is True


def test_judged_flag_stays_off_the_wire_schema():
    """`judged` is the framework's field: constrained decoding must not ask
    the model to fill it, so neither wire schema may declare it."""
    from silica.capabilities.dedup import DedupDecision, DedupVerdict, DedupVerdictBatch

    assert "judged" not in DedupVerdict.model_json_schema()["properties"]
    assert "judged" not in DedupDecision.model_json_schema()["properties"]
    nested = DedupVerdictBatch.model_json_schema()["$defs"]["DedupVerdict"]["properties"]
    assert "judged" not in nested
