"""Prospective link check in validate_operations (section 4).

Rules under test:
- write ops are exempt: their wikilinks may be forward references.
- patch/overwrite ops: every wikilink in snippet/content must resolve in the
  current vault OR point to a note being created by a write op in the same batch.
- Links already in the vault (search_names returns a match) are allowed.
"""
from unittest.mock import patch

from silica.driver.base import NoteContent, NoteRef
from silica.kernel.write.validate import MIN_WRITE_SNIPPET_CHARS, validate_operations

# Write ops must clear the precision gate; the padding keeps fixtures short.
_PAD = " lorem" * (MIN_WRITE_SNIPPET_CHARS // 6 + 1)


def _note(name: str, path: str) -> NoteRef:
    return NoteRef(name=name, path=path)


def _existing_read(path: str) -> NoteContent:
    return NoteContent(ref=NoteRef(name=path, path=path), content="# Note")


# ---------------------------------------------------------------------------
# Helpers for composing mocks
# ---------------------------------------------------------------------------

class _ReadSideEffect:
    """DRIVER.read_note mock: succeeds for known paths, raises for others."""

    def __init__(self, known: set[str]):
        self._known = known

    def __call__(self, ref):
        key = ref if isinstance(ref, str) else ref.path
        if any(k in str(key) for k in self._known):
            return _existing_read(str(key))
        raise RuntimeError(f"not found: {key}")


# ---------------------------------------------------------------------------
# Test: write ops are exempt from the link check
# ---------------------------------------------------------------------------

def test_write_op_forward_reference_neutralized():
    """write ops with unresolvable links are kept; the links become forward-refs."""
    ops = [
        {
            "op": "write",
            "path": "Concepts/Neural Network.md",
            "heading": "Neural Network",
            "source_basename": "inbox.md",
            "snippet": "See [[Vanishing Gradient]] and [[Backpropagation]]." + _PAD,
        }
    ]
    read_mock = _ReadSideEffect(known=set())  # nothing exists
    cleared_links: list[dict] = []
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.write.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(
            ops, [], "Concepts", cleared_links_out=cleared_links
        )

    assert not rejected
    assert any(op.get("heading") == "Neural Network" for op in validated)
    recorded = {c["cleared_link"] for c in cleared_links}
    assert {"Vanishing Gradient", "Backpropagation"} <= recorded


# ---------------------------------------------------------------------------
# Test: patch op — link resolved in vault
# ---------------------------------------------------------------------------

def test_patch_op_link_resolved_in_vault():
    """patch op whose snippet links to an existing vault note must be accepted."""
    existing_ref = _note("Quantum Computing", "Concepts/Quantum Computing.md")
    read_mock = _ReadSideEffect(known={"Classical Computing", "Quantum Computing"})

    ops = [
        {
            "op": "patch",
            "path": "Concepts/Classical Computing.md",
            "heading": "Classical Computing",
            "source_basename": "inbox.md",
            "snippet": "Compare with [[Quantum Computing]].",
        }
    ]
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.write.validate.DRIVER.search_names", return_value=[existing_ref]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    assert not rejected
    assert any(op["heading"] == "Classical Computing" for op in validated)


# ---------------------------------------------------------------------------
# Test: patch op — broken link rejected
# ---------------------------------------------------------------------------

def test_patch_op_broken_link_neutralized():
    """patch op with a wikilink not in vault/batch is kept; link becomes a forward-ref."""
    read_mock = _ReadSideEffect(known={"Existing Note"})

    ops = [
        {
            "op": "patch",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            "snippet": "Links to [[Ghost Note That Does Not Exist]].",
        }
    ]
    cleared_links: list[dict] = []
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.write.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(
            ops, [], "Concepts", cleared_links_out=cleared_links
        )

    assert not rejected
    op = next(op for op in validated if op.heading == "Existing Note")
    assert "[[Ghost Note That Does Not Exist]]" in op.snippet
    assert any(c["cleared_link"] == "Ghost Note That Does Not Exist" for c in cleared_links)


# ---------------------------------------------------------------------------
# Test: patch op — link resolved by write op in same batch
# ---------------------------------------------------------------------------

def test_patch_op_link_resolved_by_same_batch_write():
    """patch op linking to a note being created in the same batch must be accepted."""
    read_mock = _ReadSideEffect(known={"Existing Note"})

    ops = [
        {
            "op": "write",
            "path": "Concepts/New Concept.md",
            "heading": "New Concept",
            "source_basename": "inbox.md",
            "snippet": "A new concept." + _PAD,
        },
        {
            "op": "patch",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            "snippet": "See also [[New Concept]].",
        },
    ]
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.write.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    assert not rejected
    assert any(op["heading"] == "Existing Note" for op in validated)
    assert any(op["heading"] == "New Concept" for op in validated)


# ---------------------------------------------------------------------------
# Test: overwrite op — broken link rejected
# ---------------------------------------------------------------------------

def test_overwrite_op_broken_link_neutralized():
    """overwrite ops obey the same neutralize-and-register rule as patch ops."""
    read_mock = _ReadSideEffect(known={"Existing Note"})

    ops = [
        {
            "op": "overwrite",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            "content": "Full rewrite with [[Ghost Link]].",
        }
    ]
    cleared_links: list[dict] = []
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.write.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(
            ops, [], "Concepts", cleared_links_out=cleared_links
        )

    assert not rejected
    assert any(c["cleared_link"] == "Ghost Link" for c in cleared_links)


# ---------------------------------------------------------------------------
# Test: patch op with no snippet — skips the check
# ---------------------------------------------------------------------------

def test_patch_op_no_snippet_skips_check():
    """patch op with empty snippet must not be rejected by the link check."""
    read_mock = _ReadSideEffect(known={"Existing Note"})

    ops = [
        {
            "op": "patch",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            "snippet": "",
        }
    ]
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.write.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    assert not rejected
    assert any(op["heading"] == "Existing Note" for op in validated)


# ---------------------------------------------------------------------------
# Test: auto-created hub note is in batch_created_names → link to it resolves
# ---------------------------------------------------------------------------

def test_patch_op_link_to_auto_hub_resolves():
    """patch op linking to the auto-created hub note for the target_dir must be accepted."""
    read_mock = _ReadSideEffect(known={"Existing Note"})  # hub does not exist yet

    ops = [
        {
            "op": "patch",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            # hub name == basename of target_dir == "Concepts"
            "snippet": "Parent: [[Concepts]].",
        }
    ]
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.write.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    # hub auto-op is a write whose path includes "Concepts.md",
    # so "concepts" enters batch_created_names and resolves the link.
    assert not rejected


# ---------------------------------------------------------------------------
# Test: write ops with links in future_ref_whitelist are accepted
# ---------------------------------------------------------------------------

def test_write_op_whitelist_link_accepted():
    """write ops with links in future_ref_whitelist must be accepted."""
    ops = [
        {
            "op": "write",
            "path": "Concepts/Neural Network.md",
            "heading": "Neural Network",
            "source_basename": "inbox.md",
            "snippet": "See [[Vanishing Gradient]] and [[Backpropagation]]." + _PAD,
        }
    ]
    read_mock = _ReadSideEffect(known=set())
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.write.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(
            ops, [], "Concepts", future_ref_whitelist=["Vanishing Gradient", "Backpropagation"]
        )

    assert not rejected
    assert any(op.get("heading") == "Neural Network" for op in validated)


# ---------------------------------------------------------------------------
# Test: deduplication happens before coercion, avoiding incorrect reject
# ---------------------------------------------------------------------------

def test_unresolved_inline_links_neutralized_not_rejected():
    """Paper-internal wikilinks with no note must be kept as forward-refs, not rejected.

    Reproduces the 83%-rejection incident: a dense paper coerced to patch ops whose
    bodies cross-reference paper-internal terms (MEM, GEPA, A-MEM) that have no note.
    Policy: neutralize + register — keep the op, keep the dangling link, record it.
    """
    read_mock = _ReadSideEffect(known={"RAM (Random Access Memory)", "MACE"})

    ops = [
        {
            "op": "patch",
            "path": "AI/MACE.md",
            "heading": "ACE",
            "source_basename": "inbox.md",
            "snippet": "ACE builds on [[MEM]] and [[GEPA]].",
        },
        {
            "op": "patch",
            "path": "AI/RAM (Random Access Memory).md",
            "heading": "MEM",
            "source_basename": "inbox.md",
            "snippet": "MEM relates to [[A-MEM]].",
        },
    ]
    cleared_links: list[dict] = []
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.write.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(
            ops, [], "AI", cleared_links_out=cleared_links
        )

    # No op is rejected — both survive.
    assert not rejected
    headings = {op.heading for op in validated if op.op != "write"}
    assert {"ACE", "MEM"} <= headings
    # The dangling links are kept verbatim in the op bodies.
    ace_op = next(op for op in validated if op.heading == "ACE")
    assert "[[MEM]]" in ace_op.snippet and "[[GEPA]]" in ace_op.snippet
    # Every unresolved link is recorded as a forward-ref.
    recorded = {c["cleared_link"] for c in cleared_links}
    assert {"MEM", "GEPA", "A-MEM"} <= recorded


def test_patch_link_to_same_batch_patch_sibling_resolves():
    """A link to a sibling that is itself a patch op (not a write) must resolve.

    Error 2: batch_created_names previously counted only write ops, so a cross-ref
    between two colliding (patched) siblings was wrongly treated as unresolved.
    """
    read_mock = _ReadSideEffect(known={"Alpha", "Beta"})

    ops = [
        {
            "op": "patch",
            "path": "Concepts/Alpha.md",
            "heading": "Alpha",
            "source_basename": "inbox.md",
            "snippet": "See [[Beta]].",
        },
        {
            "op": "patch",
            "path": "Concepts/Beta.md",
            "heading": "Beta",
            "source_basename": "inbox.md",
            "snippet": "A sibling.",
        },
    ]
    cleared_links: list[dict] = []
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.write.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(
            ops, [], "Concepts", cleared_links_out=cleared_links
        )

    assert not rejected
    # [[Beta]] resolves via the sibling patch op → not recorded as a forward-ref.
    assert not any(c["cleared_link"].lower() == "beta" for c in cleared_links)


def test_deduplication_before_coercion_prevents_unnecessary_rejection():
    """Duplicate ops for an existing file should not fail validation after coercion.
    
    If two write/patch ops to the same existing path are processed:
    1. Deduplication skips the redundant op, keeping the richest one as 'write' (if write/overwrite was in group).
    2. Coercion converts it to 'patch' because the path exists.
    3. It is validated successfully rather than rejected as a duplicate write error.
    """
    ops = [
        {
            "op": "write",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            "snippet": "A short snippet.",
        },
        {
            "op": "write",
            "path": "Concepts/Existing Note.md",
            "heading": "Existing Note",
            "source_basename": "inbox.md",
            "snippet": "A much longer and richer snippet with more content.",
        }
    ]
    # Path exists in vault
    read_mock = _ReadSideEffect(known={"Existing Note"})
    
    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.write.validate.DRIVER.search_names", return_value=[]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    assert not rejected
    # The list contains the auto-created hub note operation first, then the deduplicated note operation.
    assert len(validated) == 2
    assert validated[0].heading == "Concepts"
    assert validated[1].op == "patch"
    assert "much longer" in validated[1].snippet


def test_dedup_collapses_paths_differing_only_by_case():
    """Two ops whose paths differ only by case are ONE file on a case-insensitive
    filesystem (macOS, Windows): committing both silently overwrites the first
    note with the second, and every op still reports ok."""
    ops = [
        {"op": "write", "path": "Concepts/Percettrone.md", "heading": "Percettrone",
         "source_basename": "inbox.md", "snippet": "short" + _PAD},
        {"op": "write", "path": "Concepts/percettrone.md", "heading": "percettrone",
         "source_basename": "inbox.md", "snippet": "the richer variant wins" + _PAD},
    ]
    read_mock = _ReadSideEffect(known=set())  # empty vault: writes stay writes

    with patch("silica.kernel.write.validate.DRIVER.read_note", side_effect=read_mock), \
         patch("silica.kernel.write.validate.DRIVER.search_names", return_value=[]), \
         patch("silica.kernel.write.validate.DRIVER.list_files", return_value=[]):
        validated, rejected = validate_operations(ops, [], "Concepts")

    touching = [o for o in validated if (o.touched_ref() or "").lower() == "concepts/percettrone.md"]
    assert len(touching) == 1, "case-variant paths must collapse to one op"
    assert "richer variant" in (touching[0].snippet or "")
