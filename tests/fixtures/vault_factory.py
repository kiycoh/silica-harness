"""Synthetic vault factory — deterministic, idempotent test fixture.

Generates a fixed-topology Obsidian vault for CI-reproducible graph tests.
Satisfies contracts C0.1–C0.5 from the WS0 spec.

Default location: tests/fixtures/synthetic_vault/
Override: SILICA_TEST_VAULT env var
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# NoteSpec — the unit of vault topology
# ---------------------------------------------------------------------------

@dataclass
class NoteSpec:
    path: str                         # vault-relative path (e.g. "Hub/Concepts.md")
    frontmatter: dict = field(default_factory=dict)
    body: str = ""
    expected_role: str = "normal"     # orphan, hub, spoke, dup-basename, lean, mono, inbox, …


# ---------------------------------------------------------------------------
# SPEC — source of truth for the synthetic vault topology
# ---------------------------------------------------------------------------

SPEC: list[NoteSpec] = [
    # Hub
    NoteSpec(
        path="Hub/Concepts.md",
        frontmatter={"tags": ["concepts"], "AI": True},
        body=(
            "# Concepts\n\n"
            "This note is the central hub.\n\n"
            "- [[Backpropagation]]\n"
            "- [[Gradient]]\n"
            "- [[Perceptron]]\n"
            "- [[A/Cell]]\n"
            "- [[B/Cell]]\n"
        ),
        expected_role="hub",
    ),
    # Spoke — resolved links
    NoteSpec(
        path="Concepts/Backpropagation.md",
        frontmatter={"tags": ["concepts"], "AI": True},
        body=(
            "# Backpropagation\n\n"
            "Optimization algorithm.\n\n"
            "## Relations\n\n"
            "- [[Hub/Concepts]]\n"
            "- [[Gradient]]\n"
        ),
        expected_role="spoke",
    ),
    NoteSpec(
        path="Concepts/Gradient.md",
        frontmatter={"tags": ["concepts"], "AI": True},
        body=(
            "# Gradient\n\n"
            "Partial derivative in a multidimensional space.\n\n"
            "## Relations\n\n"
            "- [[Hub/Concepts]]\n"
        ),
        expected_role="spoke",
    ),
    # Spoke — has 1 unresolved link
    NoteSpec(
        path="Concepts/Perceptron.md",
        frontmatter={"tags": ["concepts"], "AI": True},
        body=(
            "# Perceptron\n\n"
            "Artificial neuron model.\n\n"
            "## Relations\n\n"
            "- [[Hub/Concepts]]\n"
            "- [[MissingNote]]\n"
        ),
        expected_role="spoke-unresolved",
    ),
    # Orphan — no incoming links
    NoteSpec(
        path="Isolated/Orphan.md",
        frontmatter={"tags": ["isolated"]},
        body="# Orphan\n\nThis note does not have backlinks.",
        expected_role="orphan",
    ),
    # Duplicate basename #1
    NoteSpec(
        path="A/Cell.md",
        frontmatter={"tags": ["biology"], "AI": True},
        body=(
            "# Cell (A)\n\n"
            "Fundamental unit of life.\n\n"
            "- [[Hub/Concepts]]\n"
        ),
        expected_role="dup-basename",
    ),
    # Duplicate basename #2
    NoteSpec(
        path="B/Cell.md",
        frontmatter={"tags": ["biology"], "AI": True},
        body=(
            "# Cell (B)\n\n"
            "Variant in folder B.\n\n"
            "- [[Isolated/Orphan]]\n"
        ),
        expected_role="dup-basename",
    ),
    # Lean / empty — triage → enrich
    NoteSpec(
        path="Lean/Empty.md",
        frontmatter={"tags": ["lean"]},
        body="# Empty\n\n",
        expected_role="lean-empty",
    ),
    # Lean / stub — < 600 chars body, triage → enrich
    NoteSpec(
        path="Lean/Stub.md",
        frontmatter={"tags": ["lean"]},
        body=(
            "# Stub\n\n"
            "Very short note.\n\n"
            "- [[Hub/Concepts]]\n"
        ),
        expected_role="lean-stub",
    ),
    # Monolith — over-limit, ≥2 H2, triage → decouple
    NoteSpec(
        path="Mono/Monolith.md",
        frontmatter={"tags": ["mono"]},
        body=(
            "# Monolith\n\n"
            + ("Very long text. " * 60) + "\n\n"
            "## Section One\n\n"
            + ("Content of the first section. " * 40) + "\n\n"
            "## Section Two\n\n"
            + ("Content of the second section. " * 40) + "\n\n"
            "## Section Three\n\n"
            + ("Content of the third section. " * 40) + "\n"
        ),
        expected_role="mono",
    ),
    # Bad frontmatter — inline CSV tags
    NoteSpec(
        path="BadMeta/InlineTag.md",
        frontmatter={"tags": "biology, cell, mitosis"},  # string instead of list
        body=(
            "# InlineTag\n\n"
            "Note with non-conforming frontmatter (tags as CSV string).\n"
        ),
        expected_role="bad-meta",
    ),
    # Inbox — collides with Backpropagation
    NoteSpec(
        path="_inbox/Lecture.md",
        frontmatter={},
        body=(
            "# Lecture on Backpropagation\n\n"
            "Notes from the lecture. Topics: Backpropagation, gradient, \n"
            "gradient descent, optimization.\n"
        ),
        expected_role="inbox-collision",
    ),
    # Inbox — new concept, no collision
    NoteSpec(
        path="_inbox/New.md",
        frontmatter={},
        body=(
            "# Transformers\n\n"
            "Transformer architecture: attention mechanism, self-attention, \n"
            "multi-head attention, encoder-decoder.\n"
        ),
        expected_role="inbox-new",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_ROOT = Path(__file__).parent / "synthetic_vault"
_MANIFEST_NAME = ".silica_fixture_manifest.json"
_SPEC_VERSION = "1"


def _canonical(path: str) -> str:
    """Vault-relative canonical key: strip .md, normalize slashes, lowercase."""
    p = path.replace("\\", "/").strip("/")
    if p.endswith(".md"):
        p = p[:-3]
    return p.lower()


def _spec_sha256() -> str:
    """SHA-256 of the serialised SPEC (used for idempotency check)."""
    content = json.dumps(
        [
            {
                "path": s.path,
                "frontmatter": s.frontmatter,
                "body": s.body,
                "expected_role": s.expected_role,
            }
            for s in SPEC
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _resolve_root() -> Path:
    """Return the vault root: env override or default within the repo."""
    env = os.environ.get("SILICA_TEST_VAULT")
    if env:
        return Path(env)
    return _DEFAULT_ROOT


def _render_note(spec: NoteSpec) -> str:
    """Render a NoteSpec to its full markdown text."""
    import yaml  # PyYAML — already a dev dependency via uv
    parts = []
    if spec.frontmatter:
        parts.append("---")
        parts.append(yaml.dump(spec.frontmatter, allow_unicode=True, default_flow_style=False).rstrip())
        parts.append("---")
        parts.append("")
    parts.append(spec.body.rstrip())
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_synthetic_vault(root: Path, force: bool = False) -> Path:
    """Create the synthetic vault at *root*.

    - If *root* does not exist or has no manifest, creates it from scratch.
    - If the manifest spec_sha256 matches the current SPEC, does nothing (idempotent).
    - If *force* is True, always regenerates.

    Returns the root Path.
    """
    root = Path(root)
    manifest_path = root / _MANIFEST_NAME
    current_sha = _spec_sha256()

    # Idempotency check
    if not force and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("spec_sha256") == current_sha:
                return root  # nothing to do
        except Exception:
            pass  # corrupt manifest → regenerate

    # Write notes
    root.mkdir(parents=True, exist_ok=True)
    for spec in SPEC:
        note_path = root / spec.path
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(_render_note(spec), encoding="utf-8")

    # Write manifest
    manifest = {
        "spec_version": _SPEC_VERSION,
        "spec_sha256": current_sha,
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "notes": [
            {
                "path": s.path,
                "canonical": _canonical(s.path),
                "expected_role": s.expected_role,
            }
            for s in SPEC
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return root
