"""Precision gate at write — a note body under MIN_WRITE_SNIPPET_CHARS is
deferred, never written as a «(to be expanded)» placeholder.

Real fixture: run 5d0a3350 (2026-07-04) — the distiller returned whole chunks
of write ops with snippet="" despite full inbox excerpts in the payload;
validate passed them and execute_write filled the vault with placeholders
(«Matrici diagonali», «Matrice dei Vettori», «Variabile casuale continua»).
"""
from __future__ import annotations

from silica.kernel.write.validate import (
    MIN_WRITE_SNIPPET_CHARS,
    meta_description_reason,
    min_write_snippet_chars,
    validate_operations,
)

# The audit's real fixture (run 1f8fb488, Classificazione.md): the whole body
# announces what the source section contains instead of teaching it.
_META_BODY = (
    "Task: classificazione supervisionata per prevedere etichette di classe "
    "mancanti da dati di addestramento. Include definizione formale e esempio "
    "di applicazione pratica."
)


def test_meta_description_flags_the_audit_body():
    reason = meta_description_reason(_META_BODY)
    assert reason and "meta-description" in reason


def test_meta_description_flags_patch_style_opener():
    assert meta_description_reason("Estende la sezione su Spark SQL con nuove funzioni.")
    assert meta_description_reason("La sezione descrive i tre teoremi fondamentali.")


def test_meta_description_spares_real_content():
    # Prose that delivers material — no announcement shape.
    assert meta_description_reason(
        "La normalizzazione riscala le feature in un intervallo comune. "
        "Ad esempio, min-max scaling mappa i valori in [0,1]."
    ) is None
    # A marker inside a body with content evidence (code fence) is spared.
    assert meta_description_reason(
        "Include definizione formale:\n```python\ndef f(x):\n    return x\n```"
    ) is None
    # Long multi-paragraph bodies are content by construction.
    assert meta_description_reason("Include definizione ed esempio. " + "parola " * 150) is None


def test_meta_shape_gate_rejects_long_meta_write(tmp_vault, monkeypatch):
    # A body long enough to clear the length floor but still an announcement.
    monkeypatch.delenv("SILICA_META_SHAPE_CHECK", raising=False)
    body = _META_BODY + " Vengono inoltre trattati i criteri di valutazione e le metriche principali usate nei problemi reali di classificazione, con note sulle differenze rispetto alla regressione e sul ruolo dei dati etichettati nella fase di addestramento del modello."
    assert len(body) >= MIN_WRITE_SNIPPET_CHARS
    validated, rejected = validate_operations(
        [_write_op("Classificazione", body)], [], "Corso",
    )
    assert validated == []
    assert "meta-description" in rejected[0].reason


def test_meta_shape_gate_env_kill_switch(tmp_vault, monkeypatch):
    monkeypatch.setenv("SILICA_META_SHAPE_CHECK", "0")
    body = _META_BODY + " Vengono inoltre trattati i criteri di valutazione e le metriche principali usate nei problemi reali di classificazione, con note sulle differenze rispetto alla regressione e sul ruolo dei dati etichettati nella fase di addestramento del modello."
    validated, rejected = validate_operations(
        [_write_op("Classificazione", body)], [], "Corso",
    )
    assert rejected == []
    assert any(o.heading == "Classificazione" for o in validated)


def test_effective_floor_defaults_to_constant_and_honors_env(monkeypatch):
    monkeypatch.delenv("SILICA_MIN_WRITE_SNIPPET_CHARS", raising=False)
    assert min_write_snippet_chars() == MIN_WRITE_SNIPPET_CHARS
    monkeypatch.setenv("SILICA_MIN_WRITE_SNIPPET_CHARS", "42")
    assert min_write_snippet_chars() == 42


def _write_op(heading: str, snippet: str) -> dict:
    return {
        "op": "write",
        "path": f"Corso/{heading}.md",
        "heading": heading,
        "source_basename": "lez.md",
        "snippet": snippet,
    }


def test_empty_snippet_write_is_rejected(tmp_vault):
    validated, rejected = validate_operations(
        [_write_op("Matrici diagonali", "")], [], "Corso",
    )
    assert validated == []
    assert len(rejected) == 1
    assert "too short" in rejected[0].reason


def test_short_snippet_write_lands_flagged(tmp_vault):
    # Thin but real: the note lands with `review:` and the expand worker
    # re-authors it in place (soft gate, 2026-09-02). Only "" stays hard.
    validated, rejected = validate_operations(
        [_write_op("Matrici diagonali", "x" * (MIN_WRITE_SNIPPET_CHARS - 1))],
        [], "Corso",
    )
    assert rejected == []
    [op] = [o for o in validated if o.heading == "Matrici diagonali"]
    assert "too short" in op.review


def test_sufficient_snippet_write_passes(tmp_vault):
    validated, rejected = validate_operations(
        [_write_op("Matrici diagonali", "x" * MIN_WRITE_SNIPPET_CHARS)],
        [], "Corso",
    )
    assert rejected == []
    assert any(o.heading == "Matrici diagonali" for o in validated)


def test_whitespace_padding_does_not_beat_the_gate(tmp_vault):
    padded = ("x" * 40) + " " * MIN_WRITE_SNIPPET_CHARS
    validated, _ = validate_operations(
        [_write_op("Matrici diagonali", padded)], [], "Corso",
    )
    [op] = [o for o in validated if o.heading == "Matrici diagonali"]
    assert "too short (40 <" in op.review


def test_near_title_survives_on_an_empty_body_rejection(tmp_vault):
    """An empty body is still hard, but the fuzzy-title flag it also earned
    rides on the rejected op, so the retry knows a near-duplicate exists."""
    tmp_vault.note("Corso/Descriptor.md", "# Descriptor\n\ncorpo")
    validated, rejected = validate_operations(
        [_write_op("Description", "")], [], "Corso",
    )
    assert not any(o.heading == "Description" for o in validated)
    assert "too short" in rejected[0].reason
    assert "near_title" in rejected[0].op.review


def test_patch_ops_are_not_gated(tmp_vault):
    """Patch appends to an existing note — a short addendum is legitimate."""
    tmp_vault.note("Corso/Norma.md", "# Norma\n\ncorpo")
    op = {
        "op": "patch",
        "path": "Corso/Norma.md",
        "heading": "Norma",
        "source_basename": "lez.md",
        "snippet": "breve aggiunta",
    }
    validated, rejected = validate_operations([op], [], "Corso")
    assert rejected == []
    assert any(o.heading == "Norma" for o in validated)


def _payload(inbox_file, concepts):
    return [{"batches": [{"inbox_file": inbox_file, "concepts": concepts}]}]


def test_empty_snippet_with_empty_excerpt_is_skipped_not_rejected(tmp_vault):
    """Concept only *mentioned* (empty inbox_excerpt) → forward-reference, not a
    rejection. Skipping keeps a whole chunk of such stubs from driving the
    rejection rate to 100% and aborting the run."""
    payload = _payload("Inbox/lez.md", [{"name": "Machine translation", "inbox_excerpt": ""}])
    validated, rejected = validate_operations(
        [_write_op("Machine translation", "")], payload, "Corso",
    )
    assert validated == []
    assert rejected == []  # neither written nor rejected — dropped as a forward-ref


def test_empty_snippet_with_full_excerpt_still_rejected(tmp_vault):
    """Regression 5d0a3350: the excerpt HAD content but the distiller dropped the
    body → must reject so the expand arc retries, never silently skip."""
    payload = _payload("Inbox/lez.md", [{"name": "Machine translation", "inbox_excerpt": "x" * 500}])
    validated, rejected = validate_operations(
        [_write_op("Machine translation", "")], payload, "Corso",
    )
    assert validated == []
    assert len(rejected) == 1 and "too short" in rejected[0].reason
