# tests/test_provenance_dedup.py
from silica.config import CONFIG
from silica.kernel.write.templates import provenance_header, block_present
from silica.kernel.write.ops import Op, OpType
from silica.kernel.write.bulk import execute_one
from silica.kernel.write.provenance import append_record, note_authored_by


def test_provenance_helpers():
    hdr = provenance_header("Async IO", "meeting.md")
    assert hdr == "## Additional notes: Async IO (from meeting.md)"
    body = f"seed\n\n{hdr}\n\nfacts\n"
    assert block_present(body, "Async IO", "meeting.md") is True
    assert block_present(body, "Async IO", "other.md") is False
    assert block_present("seed only", "Async IO", "meeting.md") is False


def test_block_present_still_matches_the_legacy_italian_header():
    """The header is the idempotency key for a patch block. A vault written
    before the header was translated still holds blocks in the old spelling,
    and failing to match them appends a second copy of content already there."""
    legacy = "seed\n\n## Note aggiuntive — Async IO (da meeting.md)\n\nfacts\n"
    assert block_present(legacy, "Async IO", "meeting.md") is True
    assert block_present(legacy, "Async IO", "other.md") is False


def test_double_patch_is_idempotent(tmp_vault):
    target = tmp_vault.note("Topics/AsyncIO.md", "---\n---\nseed\n")
    op = Op(op=OpType.patch, heading="Async IO", source_basename="meeting.md",
            path=target, snippet="first fact", hub="Hub")

    execute_one(op)
    after_first = tmp_vault.read(target)
    res = execute_one(op)                          # same op again

    assert res.get("skipped") == "duplicate"
    assert tmp_vault.read(target) == after_first   # no second block appended
    assert after_first.count("## Additional notes: Async IO (from meeting.md)") == 1


def test_patch_skipped_when_source_already_authored_note(tmp_vault):
    """Re-ingest idempotency: a source must not re-append its own concepts into
    a note it already authored. The note was WRITTEN on the first ingest (no
    provenance block), so block_present can't catch it — the provenance ledger
    (this source -> this note) does. Real incident: re-ingesting an edited
    lecture re-patched every unchanged concept into its own prior note."""
    target = tmp_vault.note("Concepts/Machine Learning.md",
                            "---\n---\n# Machine Learning\n\nseed body\n")
    append_record("lezione_1.md", "sha-v1", "run-1",
                  ["Concepts/Machine Learning"], vault_path=CONFIG.vault_path)

    assert note_authored_by(target, "lezione_1.md", vault_path=CONFIG.vault_path)

    op = Op(op=OpType.patch, heading="Machine Learning",
            source_basename="lezione_1.md", path=target,
            snippet="re-distilled excerpt", hub="Hub")
    res = execute_one(op)

    assert res.get("skipped") == "duplicate"
    assert "## Additional notes" not in tmp_vault.read(target)


def test_patch_proceeds_for_a_different_source(tmp_vault):
    """A DIFFERENT source enriching the same note is a legit cross-source
    patch, not a re-ingest — it must still land."""
    target = tmp_vault.note("Concepts/Machine Learning.md", "---\nAI: true\n---\nseed\n")
    append_record("lezione_1.md", "sha-v1", "run-1",
                  ["Concepts/Machine Learning"], vault_path=CONFIG.vault_path)

    op = Op(op=OpType.patch, heading="Machine Learning",
            source_basename="lezione_9.md", path=target,   # other source
            snippet="new fact from a different lecture", hub="Hub")
    res = execute_one(op)

    assert res.get("skipped") is None
    assert "## Additional notes: Machine Learning (from lezione_9.md)" in tmp_vault.read(target)


def test_duplicate_skip_stamps_ai_flag(tmp_vault):
    """The duplicate-skip path must leave the note lint-clean like the real
    patch path does (which stamps `AI: partial` via ensure_ai_flag: the body
    stays the user's). In safe mode the skip lands on a freshly-seeded mirror
    copy of a human note that has no `AI` key, so the copy stayed lint-dirty
    forever and the chunk LINT gate aborted whole chunks on it (run 880b9aa9:
    f1_c0/f1_c1 both died with "frontmatter 'AI' missing or not boolean" on
    notes no op had changed)."""
    target = tmp_vault.note(  # legacy header on purpose: pre-translation vault
        "Topics/AsyncIO.md",
        "---\ntags:\n  - async\n---\nseed\n\n[[Hub]]\n\n"
        "## Note aggiuntive — Async IO (da meeting.md)\n\nfacts\n",
    )
    op = Op(op=OpType.patch, heading="Async IO", source_basename="meeting.md",
            path=target, snippet="first fact", hub="Hub")

    res = execute_one(op)

    assert res.get("skipped") == "duplicate"
    assert "AI: partial" in tmp_vault.read(target)


def test_duplicate_block_still_repairs_hub_link(tmp_vault):
    """A note holding the provenance block but NOT the hub link (state left by
    an interrupted run, or a pre-injection silica version) must gain the link
    on re-patch — otherwise the post-write lint fails the op on every retry
    (real incident: 2026-07-17 nucleate run, Claude Shannon.md)."""
    target = tmp_vault.note(  # legacy header on purpose: pre-translation vault
        "Topics/AsyncIO.md",
        "---\nAI: true\n---\nseed\n\n## Note aggiuntive — Async IO (da meeting.md)\n\nfacts\n",
    )
    op = Op(op=OpType.patch, heading="Async IO", source_basename="meeting.md",
            path=target, snippet="first fact", hub="Hub")

    res = execute_one(op)

    assert res.get("skipped") == "duplicate"
    after = tmp_vault.read(target)
    assert '[[Hub]]' in after                       # link repaired
    assert after.count("## Note aggiuntive") == 1   # snippet still skipped


def test_patch_proceeds_when_the_ledger_claims_a_note_that_lost_the_content(tmp_vault):
    """The ledger says this source authored the note, but the section it wrote
    is gone (hand-deleted, note rewritten, mirror discarded on a re-run). The
    ledger alone used to win and the enrichment was skipped in silence as a
    "duplicate" (open seam since 2026-08-04). The heading is the cheapest proof
    the content is still there: absent, the patch lands."""
    target = tmp_vault.note("Concepts/Machine Learning.md",
                            "---\nAI: true\n---\n# Something else\n\nrewritten body\n")
    append_record("lezione_1.md", "sha-v1", "run-1",
                  ["Concepts/Machine Learning"], vault_path=CONFIG.vault_path)
    assert note_authored_by(target, "lezione_1.md", vault_path=CONFIG.vault_path)

    op = Op(op=OpType.patch, heading="Machine Learning",
            source_basename="lezione_1.md", path=target,
            snippet="re-distilled excerpt", hub="Hub")
    res = execute_one(op)

    assert res.get("skipped") is None
    assert "## Additional notes: Machine Learning (from lezione_1.md)" in tmp_vault.read(target)
