"""kernel/title — one identity for note titles (C3).

Five call sites used to hold five divergent normalizations (one not even
case-insensitive) and the write path never compared a new title against the
vault — that is how the four «Machine Learning» umbrella notes were born.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _historical_snippet_floor(monkeypatch):
    # Predates the 100→400 write-floor raise; short fixtures here exercise
    # routing/coercion, not the length gate — pin their original floor.
    monkeypatch.setenv("SILICA_MIN_WRITE_SNIPPET_CHARS", "100")


# ---------------------------------------------------------------------------
# title_key — the equivalence key
# ---------------------------------------------------------------------------

def test_title_key_folds_case_and_parenthetical_suffix():
    from silica.kernel.text.title import title_key

    # The real fixture: 4 umbrella notes that must share one key.
    assert (
        title_key("Machine Learning")
        == title_key("machine learning")
        == title_key("Machine Learning (9 CFU)")
        == title_key("Machine learning — Corso 9 CFU")
    )


def test_title_key_folds_punctuation_and_plurals():
    from silica.kernel.text.title import title_key

    assert title_key("Reti Neurali", lang="italian") == title_key("rete neurale", lang="italian")
    assert title_key("K-Means: clustering") == title_key("k means clustering")


def test_title_key_distinct_concepts_stay_distinct():
    from silica.kernel.text.title import title_key

    assert title_key("Machine Learning") != title_key("ML per la statistica")
    assert title_key("Descriptor") != title_key("Description")


# ---------------------------------------------------------------------------
# near_titles — the fuzzy band under key-equality
# ---------------------------------------------------------------------------

def test_near_titles_catches_descriptor_description():
    from silica.kernel.text.title import near_titles

    hits = near_titles("Description", ["Descriptor", "Statistica"])
    assert [t for (t, _r) in hits] == ["Descriptor"]


def test_near_titles_excludes_key_equal_and_unrelated():
    from silica.kernel.text.title import near_titles

    # key-equal is coercion territory, not review; unrelated stays out
    hits = near_titles(
        "Machine Learning (9 CFU)",
        ["Machine Learning", "ML per la statistica", "Analisi Matematica"],
    )
    assert hits == []


# ---------------------------------------------------------------------------
# Convergence — _names_agree and recon.normalize cross the same identity
# ---------------------------------------------------------------------------

def test_names_agree_converges_on_title_key():
    from silica.router.states.collision import _names_agree

    # Cosmetic variants of the same title now agree (they did NOT before C3:
    # one of the two local normalizations was not even case-insensitive).
    assert _names_agree("Machine Learning (9 CFU)", "Machine Learning")
    assert _names_agree("machine learning", "Machine Learning")
    # The acronym leg survives untouched.
    assert _names_agree("GPT", "Generative Pretrained Transformer (GPT)")
    # Domain collisions still disagree.
    assert not _names_agree("MEMORY", "RAM (Random Access Memory)")


def test_recon_normalize_strips_leading_articles():
    from silica.kernel.text.recon import normalize

    assert normalize("della matrice Hessiana") == "matrice Hessiana"
    assert normalize("la discesa del gradiente") == "discesa del gradiente"
    assert normalize("the Hessian matrix") == "Hessian matrix"
    # Not a leading article → untouched (existing garbage-strip behavior kept).
    assert normalize("  ...matrice Hessiana") == "matrice Hessiana"
    assert normalize("Support Vector Machines") == "Support Vector Machines"


# ---------------------------------------------------------------------------
# The gate at write — validate_operations finally sees the vault
# ---------------------------------------------------------------------------

def _write_op(heading: str, path: str) -> dict:
    return {"op": "write", "path": path, "heading": heading,
            "source_basename": "lez.md",
            "snippet": f"corpo di {heading} " + "lorem " * 20}


def test_write_gate_coerces_key_equal_title_to_patch(tmp_vault):
    """«Machine Learning (9 CFU)» must patch the existing «Machine Learning»,
    not become the fourth umbrella note."""
    from silica.kernel.write.validate import validate_operations

    tmp_vault.note("Corso/Machine Learning.md", "# ML\n\ncorpo")
    validated, rejected = validate_operations(
        [_write_op("Machine Learning (9 CFU)", "Corso/Machine Learning (9 CFU).md")],
        [], "Corso",
    )
    assert rejected == []
    coerced = [o for o in validated if o.heading == "Machine Learning (9 CFU)"]
    assert len(coerced) == 1
    assert coerced[0].op.value == "patch"
    assert coerced[0].path == "Corso/Machine Learning.md"


def test_write_gate_flags_fuzzy_near_title(tmp_vault):
    """Fuzzy band → the op lands flagged (`review:`) and the dedup judge rules
    on the note once it exists; never a hard block, never a silent write."""
    from silica.kernel.write.validate import validate_operations

    tmp_vault.note("Corso/Descriptor.md", "# Descriptor\n\ncorpo")
    validated, rejected = validate_operations(
        [_write_op("Description", "Corso/Description.md")], [], "Corso",
    )
    assert rejected == []
    [op] = [o for o in validated if o.heading == "Description"]
    assert "near_title" in op.review and "Descriptor" in op.review


def test_near_title_rejection_enqueues_dedup_workitem():
    """The fuzzy band reuses C2: the deferred op ALSO becomes a live dedup
    WorkItem (content_hash + target_dir aboard) so the judge routes it —
    retry stays the exception, not the drain."""
    from silica.router.states.distill import _enqueue_near_title_dedups
    from silica.kernel.workqueue import WorkQueue

    class _FSM:
        work_queue = WorkQueue()
        hub = "Corso"
        target_dir = "Corso"
        inbox_file = "Inbox/lez.md"
        _current_source_file = "Inbox/lez.md"
        _current_content_hash = "hash9"
        progress = None

    rejected = [
        {"op": {"op": "write", "heading": "Description", "path": "Corso/Description.md",
                "snippet": "corpo", "source_basename": "lez.md"},
         "reason": "near_title candidate='Descriptor' path='Corso/Descriptor.md' "
                   "ratio=0.89 — deferred for dedup review"},
        {"op": {"op": "write", "heading": "X", "path": "Corso/X.md"},
         "reason": "Heading 'X' not present in payload concepts"},
    ]
    fsm = _FSM()
    _enqueue_near_title_dedups(fsm, rejected)

    items = fsm.work_queue.items()
    assert len(items) == 1, "only near_title rejections become dedup work"
    it = items[0]
    assert it.kind == "dedup"
    assert it.target_path == "Corso/Descriptor.md"
    assert it.context["concept"] == "Description"
    assert it.context["excerpt"] == "corpo"
    assert it.context["candidate"] == "Descriptor"
    assert it.context["content_hash"] == "hash9"
    assert it.context["target_dir"] == "Corso"


def test_write_gate_lets_unrelated_titles_through(tmp_vault):
    """Below the band the write flows untouched — legitimate atomic spokes
    never pass through the guillotine."""
    from silica.kernel.write.validate import validate_operations

    tmp_vault.note("Corso/Analisi Matematica.md", "# AM\n\ncorpo")
    validated, rejected = validate_operations(
        [_write_op("Machine Learning", "Corso/Machine Learning.md")], [], "Corso",
    )
    assert rejected == []
    kept = [o for o in validated if o.heading == "Machine Learning"]
    assert len(kept) == 1 and kept[0].op.value == "write"
