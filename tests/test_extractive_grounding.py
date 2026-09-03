# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Extractive invariant: every body content-line must be a verbatim span of the
source transcript. Enforced (reject/retry) under the `extractive` distill
profile so 'non-lossy' is a checked property, not a prompt hope."""

from silica.kernel.write.provenance import nonextractive_lines
from silica.kernel.write.validate import validate_operations

# Long enough that a body copied from it clears MIN_WRITE_SNIPPET_CHARS (100).
_EXCERPT = ("Elena: I finally signed up for the beginners pottery class at the "
            "community center downtown, it starts on the twentieth of May and "
            "runs every Tuesday evening with Mr. Alvarez, not my sister.")


def _payload(excerpt: str):
    return [{"batches": [{"inbox_file": "/inbox/session_1.md", "concepts": [
        {"name": "pottery class", "inbox_excerpt": excerpt, "vault_collision": None},
    ]}]}]


def _write_op(body: str):
    return {"op": "write", "path": "mem/Elena's pottery class.md",
            "heading": "pottery class", "source_basename": "session_1.md",
            "snippet": body}


def test_verbatim_lines_pass():
    src = ("Elena: I signed up for the pottery class at the community center.\n"
           "Sam: That's great!")
    body = "Elena: I signed up for the pottery class at the community center."
    assert nonextractive_lines(body, src) == []


def test_paraphrase_is_flagged():
    src = "Elena: I finally signed up for the pottery class at the community center!"
    body = "Elena enrolled in a ceramics course at the local rec hall."
    assert nonextractive_lines(body, src)  # reworded prose -> not a copied span


def test_markers_and_wikilinks_stripped():
    src = "Sam said he is switching to a new job in September at a fintech startup."
    body = ("- [[Sam]] said he is switching to a new job in September "
            "at a fintech startup.")
    # A bullet the model prepends and an autolink-shaped wikilink are structure,
    # not drift: the residual prose is still a verbatim span.
    assert nonextractive_lines(body, src) == []


def test_apostrophe_and_whitespace_normalized():
    src = "Elena: I don't teach the advanced   course, my sister does."
    body = "Elena: I don’t teach the advanced course, my sister does."  # curly + collapsed ws
    assert nonextractive_lines(body, src) == []


def test_headings_and_blank_lines_ignored():
    src = "Sam: The itinerary starts in Kyoto and ends in Osaka after five days."
    body = ("## Sam's trip\n\n"
            "Sam: The itinerary starts in Kyoto and ends in Osaka after five days.")
    assert nonextractive_lines(body, src) == []


# --- enforcement wired into validate_operations (gated) ---------------------

def test_verbatim_body_passes_under_enforce(tmp_vault, monkeypatch):
    monkeypatch.setenv("SILICA_MIN_WRITE_SNIPPET_CHARS", "1")  # test extractivity, not length
    monkeypatch.setenv("SILICA_EXTRACTIVE_ENFORCE", "1")
    verbatim = ("Elena: I finally signed up for the beginners pottery class at the "
                "community center downtown, it starts on the twentieth of May and "
                "runs every Tuesday evening with Mr. Alvarez, not my sister.")
    validated, rejected = validate_operations(
        [_write_op(verbatim)], _payload(_EXCERPT), "mem")
    assert rejected == []
    assert any(o.path == "mem/Elena's pottery class.md" for o in validated)


def test_paraphrase_rejected_under_enforce_but_passes_without(tmp_vault, monkeypatch):
    paraphrase = ("Elena enrolled in a beginners ceramics course at the local "
                  "recreation hall downtown, with the first session on May "
                  "twentieth held every Tuesday in the evening, taught by Alvarez.")
    monkeypatch.setenv("SILICA_MIN_WRITE_SNIPPET_CHARS", "1")  # test extractivity, not length
    # Gate ON -> the reworded body is rejected (routed to defer/steer).
    monkeypatch.setenv("SILICA_EXTRACTIVE_ENFORCE", "1")
    _, rejected_on = validate_operations(
        [_write_op(paraphrase)], _payload(_EXCERPT), "mem")
    assert any("extractive" in r.reason for r in rejected_on)
    # Gate OFF (default) -> the default distiller may paraphrase, so it must pass.
    monkeypatch.delenv("SILICA_EXTRACTIVE_ENFORCE", raising=False)
    validated_off, rejected_off = validate_operations(
        [_write_op(paraphrase)], _payload(_EXCERPT), "mem")
    assert not any("extractive" in r.reason for r in rejected_off)
    assert any(o.path == "mem/Elena's pottery class.md" for o in validated_off)


def test_profile_alone_enables_enforcement(tmp_vault, monkeypatch):
    """The extractive profile IS the contract: selecting it (env or vault
    conventions) enforces the verbatim invariant without a second env var."""
    monkeypatch.setenv("SILICA_MIN_WRITE_SNIPPET_CHARS", "1")  # test extractivity, not length
    monkeypatch.delenv("SILICA_EXTRACTIVE_ENFORCE", raising=False)
    monkeypatch.setenv("SILICA_DISTILL_PROFILE", "extractive")
    paraphrase = ("Elena enrolled in a beginners ceramics course at the local "
                  "recreation hall downtown, taught by Alvarez himself.")
    _, rejected = validate_operations(
        [_write_op(paraphrase)], _payload(_EXCERPT), "mem")
    assert any("extractive" in r.reason for r in rejected)


def test_short_verbatim_floor_is_env_lowerable(tmp_vault, monkeypatch):
    # A durable fact copied verbatim can be a short turn (<100 chars). The prose
    # placeholder floor would defer it; the extractive arm lowers it via env.
    short = "Elena: I signed up for the pottery class at the rec center."  # ~59 chars
    excerpt = short + "\nSam: Nice, when does it start?"
    monkeypatch.delenv("SILICA_MIN_WRITE_SNIPPET_CHARS", raising=False)
    validated, _ = validate_operations([_write_op(short)], _payload(excerpt), "mem")
    [op] = [o for o in validated if o.path == "mem/Elena's pottery class.md"]
    assert "snippet too short" in (op.review or "")  # default floor lands it flagged
    monkeypatch.setenv("SILICA_MIN_WRITE_SNIPPET_CHARS", "40")
    validated, _ = validate_operations([_write_op(short)], _payload(excerpt), "mem")
    [op] = [o for o in validated if o.path == "mem/Elena's pottery class.md"]
    assert op.review is None  # lowered floor admits it clean


# ---------------------------------------------------------------------------
# Keyed attribution (OKF §5.1): which line came from which source
# ---------------------------------------------------------------------------

# A real span of _EXCERPT — the invariant is verbatim selection, so a
# paraphrase here would test nothing the extractive gate would ever see.
_VERBATIM = "signed up for the beginners pottery class at the community center downtown"

def test_a_grounded_line_carries_the_leaf_id_as_its_label():
    from silica.kernel.write.provenance import attribute_lines, footnote_label

    label = footnote_label("session_1.md")
    assert label == "session_1"
    note = f"---\ntype: Note\n---\n\n# Pottery\n\n{_VERBATIM}\n"
    out = attribute_lines(note, _EXCERPT, label)
    assert f"{_VERBATIM}[^session_1]" in out
    assert out.startswith("---\ntype: Note\n---\n")   # frontmatter untouched
    assert "# Pottery[^session_1]" not in out          # headings are not claims


def test_a_line_the_source_does_not_contain_stays_unmarked():
    from silica.kernel.write.provenance import attribute_lines

    note = "# Pottery\n\nElena took up competitive freediving last winter.\n"
    assert attribute_lines(note, _EXCERPT, "session_1") == note


def test_attribution_is_idempotent():
    from silica.kernel.write.provenance import attribute_lines

    note = f"# Pottery\n\n{_VERBATIM}\n"
    once = attribute_lines(note, _EXCERPT, "session_1")
    assert attribute_lines(once, _EXCERPT, "session_1") == once
    assert once.count("[^session_1]") == 1


def test_a_second_source_adds_its_own_label_to_its_own_line():
    """The point of keying: three transcripts, three labels, one per claim."""
    from silica.kernel.write.provenance import attribute_lines

    other = "Elena also adopted a tabby cat she named Biscotto in June."
    second = "adopted a tabby cat she named Biscotto in June"
    note = f"# Elena\n\n{_VERBATIM}\n{second}\n"
    out = attribute_lines(attribute_lines(note, _EXCERPT, "session_1"), other, "session_2")
    assert f"{_VERBATIM}[^session_1]" in out
    assert f"{second}[^session_2]" in out
    assert "[^session_2]" not in out.split("\n")[2]   # not on the pottery line


def test_sources_and_superseded_blocks_are_left_alone():
    from silica.kernel.write.provenance import attribute_lines

    note = f"# Pottery\n\n{_VERBATIM}\n\n## Sources\n[[session_1]]\n"
    out = attribute_lines(note, _EXCERPT, "session_1")
    assert out.endswith("## Sources\n[[session_1]]\n")
    assert f"{_VERBATIM}[^session_1]" in out


def test_fenced_code_is_never_marked():
    """A marker inside a fence would corrupt the code it attributes."""
    from silica.kernel.write.provenance import attribute_lines

    src = "run the command uv run pytest --maxfail=1 to reproduce the failure"
    note = "# How\n\n```bash\nuv run pytest --maxfail=1 to reproduce the failure\n```\n"
    assert attribute_lines(note, src, "session_1") == note


def test_quoted_verbatim_span_passes():
    """A selected span wrapped in quotation marks is still a selected span.

    Found by the W1 audit on `bench/ab_extractive/conv-26`: the same line was
    rejected quoted and accepted unquoted, so quoting a source span — the
    canonical extractive act — cost the whole op. The marker strip already
    covers the block-level form (`>`); this is its inline twin.
    """
    src = "Caroline: Yeah, Mel! Life's all about creating memories. Can't wait for the trip!"
    body = '"Life\'s all about creating memories. Can\'t wait for the trip!"'
    assert nonextractive_lines(body, src) == []


def test_quoted_paraphrase_is_still_flagged():
    """Stripping the quotes must not weaken the test it wraps."""
    src = "Caroline: Yeah, Mel! Life's all about creating memories."
    body = '"Caroline said that making memories is what matters most in life."'
    assert nonextractive_lines(body, src)


# --- Structure is not a claim (W1 audit, 2026-08-18) -------------------------
# The audit found the gate strips structural MARKERS but then judges the
# structural TEXT as content. Framework-generated scaffolding — MOC index
# lines, section headings, link footers — is authored by construction and can
# never be a source span, so it made every block carrying it unpassable: 105 of
# 148 rejections on the audited corpus were structure, not rewritten claims.


def test_moc_index_line_judges_only_the_quote():
    """`- [[Note]] — <span>` is an index line: the label is a link to another
    note, the payload is the quote. Only the quote is claim content."""
    src = "Caroline: Being a mom is awesome. I'm creating a library for when I have kids."
    body = "- [[Caroline's future library]] — Caroline: Being a mom is awesome."
    assert nonextractive_lines(body, src) == []


def test_moc_index_line_with_paraphrased_payload_still_flagged():
    src = "Caroline: Being a mom is awesome. I'm creating a library for when I have kids."
    body = "- [[Caroline's future library]] — Caroline plans to build a reading room."
    assert nonextractive_lines(body, src)


def test_long_authored_heading_ignored():
    """`test_headings_and_blank_lines_ignored` already declares headings are
    structure; it only passed because its heading was under the length floor."""
    src = "Caroline: We had a blast last year at the Pride fest with supportive friends."
    body = ("## Additional Pride fest details\n\n"
            "Caroline: We had a blast last year at the Pride fest with supportive friends.")
    assert nonextractive_lines(body, src) == []


def test_link_only_footer_ignored():
    """A line whose prose is just a label around wikilinks is a link footer."""
    src = "Melanie: The picnic was lovely and the weather held up all afternoon."
    body = ("Melanie: The picnic was lovely and the weather held up all afternoon.\n"
            "Correlati: [[Caroline]]")
    assert nonextractive_lines(body, src) == []


def test_inline_wikilink_inside_a_quote_still_judged():
    """Unwrapping an inline autolink must keep judging the sentence: that text
    WAS in the source, so it stays claim content."""
    src = "Melanie: I finally finished the painting I started last spring."
    body = "Melanie: I never started that [[painting]] at all last spring."
    assert nonextractive_lines(body, src)


def test_body_of_only_structure_is_not_declared_extractive():
    """Exempting structure must never mean "nothing was checked, so it passes".

    Skipping heading lines buys the precision, but a body made ONLY of them
    reaches the end with zero lines verified, and `[]` here reads as "fully
    extractive". Under this profile the shape gate is skipped too, so nothing
    else would catch it: the gate's own contract is a declared hole, never
    silent loss. When nothing was verifiable, the structure is judged after
    all.
    """
    src = "Caroline: I love painting. Melanie: The picnic was lovely and warm."
    body = ("## Caroline moved to Berlin last spring\n"
            "## Melanie quit her job back in March\n")
    assert nonextractive_lines(body, src)


def test_structure_stays_exempt_when_the_block_has_real_content():
    src = "Caroline: I love painting and the picnic was lovely and warm."
    body = ("## Additional Pride fest details\n\n"
            "Caroline: I love painting and the picnic was lovely and warm.")
    assert nonextractive_lines(body, src) == []
