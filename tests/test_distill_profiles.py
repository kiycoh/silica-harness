# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Distill profiles: per-vault lens selection for the distiller prompt.

The distiller prompt splits into a fixed contract (distiller_prompt.txt,
validator-aligned) and a profile-provided lens (rubric / quality / examples
fragments under capabilities/prompts/profiles/<name>/). The `default`
profile must reproduce the pre-split prompt bit-identically; other profiles
swap the lens without touching the contract sections the validator enforces.

Selection precedence: SILICA_DISTILL_PROFILE env > conventions.distill_profile
> "default". Unknown names fall back to default (soft, warn-only).
"""
from pathlib import Path
from unittest import mock


from silica.kernel import prep_delegation
from silica.kernel.vault_manifest import (
    VaultConventions,
    VaultManifest,
    load_manifest,
)

PROFILES_DIR = (
    Path(prep_delegation.__file__).resolve().parent.parent
    / "capabilities" / "prompts" / "profiles"
)

PLACEHOLDERS = (
    "{LENS_RUBRIC}", "{LENS_QUALITY}", "{LENS_EXAMPLES}",
    "{TARGET}", "{HUB_NAME}", "{LANGUAGE}", "{MAX_TAGS}",
    "{SESSION_DATE}", "{CAPTURE_RULES}",
)

# Text that only exists in the academic (default) lens.
DEFAULT_LENS_MARKERS = ("Adam Optimizer", "scholarly for academic material")
# Contract sections that every profile must keep verbatim.
CONTRACT_MARKERS = (
    "## Anti-Hallucination Guardrails",
    "## Coverage Guardrail",
    "===SILICA-BODY",
    "## Ephemeral Facts (episodic routing)",
    # Key discipline (event-suffix drift guard): the key names the attribute,
    # never the change — with contrastive examples in every profile's render.
    "the key names the ATTRIBUTE",
    "aspiration_reinforced",
)


def _render(profile: str | None = None, **kw) -> str:
    conv = VaultConventions() if profile is None else VaultConventions(
        distill_profile=profile
    )
    m = VaultManifest(sources=("prose",), conventions=conv)
    with mock.patch(
        "silica.kernel.vault_manifest.get_active_manifest", return_value=m
    ):
        return prep_delegation.render_prompt(
            "Target/Dir", hub="HubNote", language="English",
            session_date="2026-07-18", **kw,
        )


# ---------------------------------------------------------------------------
# default profile
# ---------------------------------------------------------------------------

def test_default_render_resolves_every_placeholder(monkeypatch):
    monkeypatch.delenv("SILICA_DISTILL_PROFILE", raising=False)
    out = _render()
    for ph in PLACEHOLDERS:
        assert ph not in out, f"unresolved placeholder {ph}"


def test_default_render_keeps_academic_lens_and_contract(monkeypatch):
    monkeypatch.delenv("SILICA_DISTILL_PROFILE", raising=False)
    out = _render()
    for marker in DEFAULT_LENS_MARKERS + CONTRACT_MARKERS:
        assert marker in out


# ---------------------------------------------------------------------------
# profile selection
# ---------------------------------------------------------------------------

def test_transcript_profile_swaps_lens_keeps_contract(monkeypatch):
    monkeypatch.delenv("SILICA_DISTILL_PROFILE", raising=False)
    out = _render(profile="transcript")
    for marker in DEFAULT_LENS_MARKERS:
        assert marker not in out, f"academic lens leaked into transcript: {marker}"
    for marker in CONTRACT_MARKERS:
        assert marker in out, f"contract section missing: {marker}"
    for ph in PLACEHOLDERS:
        assert ph not in out, f"unresolved placeholder {ph}"


def test_env_var_overrides_manifest_profile(monkeypatch):
    monkeypatch.setenv("SILICA_DISTILL_PROFILE", "transcript")
    out = _render()  # manifest says default
    for marker in DEFAULT_LENS_MARKERS:
        assert marker not in out


def test_unknown_profile_falls_back_to_default(monkeypatch, caplog):
    monkeypatch.delenv("SILICA_DISTILL_PROFILE", raising=False)
    with caplog.at_level("WARNING"):
        out = _render(profile="does-not-exist")
    for marker in DEFAULT_LENS_MARKERS:
        assert marker in out
    assert any("does-not-exist" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# vault-local profiles (<vault>/.silica/profiles/<name>/)
# ---------------------------------------------------------------------------

def _bind_vault(monkeypatch, vault: Path):
    from silica.config import CONFIG
    monkeypatch.setattr(CONFIG, "vault_path", str(vault), raising=False)


def test_vault_local_profile_wins_and_partial_falls_back(monkeypatch, tmp_path):
    monkeypatch.delenv("SILICA_DISTILL_PROFILE", raising=False)
    prof = tmp_path / ".silica" / "profiles" / "custom"
    prof.mkdir(parents=True)
    (prof / "rubric.md").write_text(
        "## Decision Rubric\nCUSTOM-VAULT-LENS rubric.\n", encoding="utf-8"
    )
    _bind_vault(monkeypatch, tmp_path)
    out = _render(profile="custom")
    assert "CUSTOM-VAULT-LENS" in out
    # missing quality/examples fall back to the bundled default lens
    assert "Adam Optimizer" in out
    for ph in PLACEHOLDERS:
        assert ph not in out


def test_vault_local_fragment_shadows_bundled_profile(monkeypatch, tmp_path):
    monkeypatch.delenv("SILICA_DISTILL_PROFILE", raising=False)
    prof = tmp_path / ".silica" / "profiles" / "transcript"
    prof.mkdir(parents=True)
    (prof / "rubric.md").write_text(
        "## Decision Rubric\nVAULT-SHADOWED transcript rubric.\n", encoding="utf-8"
    )
    _bind_vault(monkeypatch, tmp_path)
    out = _render(profile="transcript")
    assert "VAULT-SHADOWED" in out
    # unshadowed fragments still come from the bundled transcript lens
    assert "Attribution is mandatory" in out


def test_profile_name_with_traversal_is_rejected(monkeypatch, tmp_path, caplog):
    # without a name guard, "<vault>/.silica/profiles/../evil" WOULD resolve here
    evil = tmp_path / ".silica" / "evil"
    evil.mkdir(parents=True)
    (evil / "rubric.md").write_text("EVIL-ESCAPED-LENS\n", encoding="utf-8")
    monkeypatch.delenv("SILICA_DISTILL_PROFILE", raising=False)
    _bind_vault(monkeypatch, tmp_path)
    with caplog.at_level("WARNING"):
        out = _render(profile="../evil")
    assert "EVIL-ESCAPED-LENS" not in out
    for marker in DEFAULT_LENS_MARKERS:
        assert marker in out
    assert any("../evil" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# manifest parsing
# ---------------------------------------------------------------------------

def test_conventions_parse_distill_profile(tmp_path):
    (tmp_path / "vault.yaml").write_text(
        "conventions:\n  distill_profile: transcript\n", encoding="utf-8"
    )
    m = load_manifest(tmp_path)
    assert m.conventions.distill_profile == "transcript"


def test_conventions_distill_profile_defaults_empty(tmp_path):
    m = load_manifest(tmp_path)
    assert m.conventions.distill_profile == ""


# ---------------------------------------------------------------------------
# bundled profiles are complete
# ---------------------------------------------------------------------------

def test_every_bundled_profile_has_all_fragments():
    profiles = [p for p in PROFILES_DIR.iterdir() if p.is_dir()]
    assert {p.name for p in profiles} >= {"default", "transcript"}
    for p in profiles:
        for frag in ("rubric.md", "quality.md", "examples.md"):
            assert (p / frag).is_file(), f"{p.name} missing {frag}"
