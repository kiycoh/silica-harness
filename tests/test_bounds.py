"""Tests for CapabilityBounds (silica/agent/bounds.py)."""
from silica.agent.bounds import (
    dedup_bounds,
    refiner_bounds,
    orphan_bounds,
    make_no_info_loss_guard,
    _wikilinks,
)
from silica.kernel.write.ops import Op, OpType


def _op(op_type, path, *, heading="H", content=None, snippet=""):
    return Op(
        op=op_type,
        heading=heading,
        source_basename="inbox.md",
        path=path,
        content=content,
        snippet=snippet,
    )


# --- dedup bounds -----------------------------------------------------------

def test_dedup_bounds_allows_patch_on_larger():
    bounds = dedup_bounds("Concepts/Big Note.md")
    ops = [_op(OpType.patch, "Concepts/Big Note.md")]
    kept, rejected = bounds.enforce(ops)
    assert len(kept) == 1
    assert not rejected


def test_dedup_bounds_rejects_overwrite_and_delete_and_write():
    bounds = dedup_bounds("Concepts/Big Note.md")
    ops = [
        _op(OpType.overwrite, "Concepts/Big Note.md", content="x"),
        _op(OpType.delete, "Concepts/Big Note.md"),
        _op(OpType.write, "Concepts/New Note.md"),
    ]
    kept, rejected = bounds.enforce(ops)
    assert kept == []
    assert len(rejected) == 3
    assert all("not permitted" in r["reason"] for r in rejected)


def test_dedup_bounds_rejects_patch_on_other_note():
    bounds = dedup_bounds("Concepts/Big Note.md")
    ops = [_op(OpType.patch, "Concepts/Small Note.md")]
    kept, rejected = bounds.enforce(ops)
    assert kept == []
    assert "outside bounds" in rejected[0]["reason"]


def test_dedup_bounds_never_touches_hub():
    bounds = dedup_bounds("Concepts/Big Note.md", hub="Concepts/Big Note.md")
    # Even though it is the "larger" path, being the hub makes it forbidden.
    ops = [_op(OpType.patch, "Concepts/Big Note.md")]
    kept, rejected = bounds.enforce(ops)
    assert kept == []
    assert "outside bounds" in rejected[0]["reason"]


# --- refiner bounds ---------------------------------------------------------

def test_refiner_bounds_allows_lossless_overwrite():
    original = "# Note\n\nSee [[Alpha]] and [[Beta]].\n" + ("body " * 100)
    bounds = refiner_bounds("Notes/Target.md")
    new = "# Note\n\n> [!note]\nSee [[Alpha]] and [[Beta]].\n" + ("body " * 100)
    ops = [_op(OpType.overwrite, "Notes/Target.md", content=new)]
    kept, rejected = bounds.enforce(ops, read_note=lambda p: original)
    assert len(kept) == 1
    assert not rejected


def test_refiner_bounds_rejects_dropped_wikilink():
    original = "See [[Alpha]] and [[Beta]]." + ("x" * 200)
    bounds = refiner_bounds("Notes/Target.md")
    new = "See [[Alpha]] only." + ("x" * 200)  # dropped [[Beta]]
    ops = [_op(OpType.overwrite, "Notes/Target.md", content=new)]
    kept, rejected = bounds.enforce(ops, read_note=lambda p: original)
    assert kept == []
    assert "dropped wikilink" in rejected[0]["reason"]


def test_refiner_bounds_rejects_shrink():
    original = "[[Alpha]]\n" + ("content " * 100)
    bounds = refiner_bounds("Notes/Target.md")
    new = "[[Alpha]]\nshort"
    ops = [_op(OpType.overwrite, "Notes/Target.md", content=new)]
    kept, rejected = bounds.enforce(ops, read_note=lambda p: original)
    assert kept == []
    assert "shrank" in rejected[0]["reason"]


# --- orphan bounds ----------------------------------------------------------

def test_orphan_bounds_allows_patch_into_neighbour_linking_the_orphan():
    bounds = orphan_bounds(["Notes/Neighbor.md"], orphan_title="Orphan")
    op = _op(OpType.patch, "Notes/Neighbor.md", snippet="## Related\n\n- [[Orphan]]\n")
    kept, rejected = bounds.enforce([op])
    assert len(kept) == 1 and not rejected


def test_orphan_bounds_rejects_patch_without_link():
    bounds = orphan_bounds(["Notes/Neighbor.md"], orphan_title="Orphan")
    op = _op(OpType.patch, "Notes/Neighbor.md", snippet="## Related\n\n(no links here)\n")
    kept, rejected = bounds.enforce([op])
    assert kept == []
    assert "no wikilink" in rejected[0]["reason"]


def test_orphan_bounds_rejects_a_link_to_anything_but_the_orphan():
    """A wikilink is not enough: only one pointing AT the orphan lowers its in-degree."""
    bounds = orphan_bounds(["Notes/Neighbor.md"], orphan_title="Orphan")
    op = _op(OpType.patch, "Notes/Neighbor.md", snippet="## Related\n\n- [[Something Else]]\n")
    kept, rejected = bounds.enforce([op])
    assert kept == []
    assert "no wikilink to 'Orphan'" in rejected[0]["reason"]


def test_orphan_bounds_rejects_patching_the_orphan_itself():
    """The regression this fix exists for: patching the orphan only adds out-degree,
    so it must fall outside the envelope even when it does add the link."""
    bounds = orphan_bounds(["Notes/Neighbor.md"], orphan_title="Orphan")
    op = _op(OpType.patch, "Notes/Orphan.md", snippet="## Related\n\n- [[Neighbor]]\n")
    kept, rejected = bounds.enforce([op])
    assert kept == []


def test_orphan_bounds_rejects_overwrite_and_unoffered_targets():
    bounds = orphan_bounds(["Notes/Neighbor.md"], orphan_title="Orphan")
    ops = [
        _op(OpType.overwrite, "Notes/Neighbor.md", content="[[Orphan]]"),
        _op(OpType.patch, "Notes/Other.md", snippet="[[Orphan]]"),
    ]
    kept, rejected = bounds.enforce(ops)
    assert kept == []
    assert len(rejected) == 2


# --- skip + helpers --------------------------------------------------------

def test_skip_ops_always_pass():
    bounds = dedup_bounds("Concepts/Big Note.md")
    skip = Op(op=OpType.skip, heading="H", source_basename="inbox.md", reason="noop")
    kept, rejected = bounds.enforce([skip])
    assert kept == [skip]
    assert not rejected


def test_wikilinks_extraction_handles_alias_and_anchor():
    links = _wikilinks("[[Alpha|alias]] [[Beta#section]] [[Gamma]]")
    assert links == {"alpha", "beta", "gamma"}


def test_no_info_loss_guard_direct():
    guard = make_no_info_loss_guard(floor_ratio=0.85)
    op = Op(op=OpType.overwrite, heading="H", source_basename="i.md",
            path="N.md", content="[[A]] kept content here")
    assert guard(op, "[[A]] kept content here") is None
    assert "dropped wikilink" in guard(op, "[[A]] [[B]] original longer text here")


def test_dedup_bounds_blocks_hub_by_bare_name():
    """Hub protection must work even when hub is a bare name without folder prefix.

    This tests dedup_bounds because its target_predicate matches the hub path
    (both resolve to the same note), so only forbidden_paths can block it.
    The bare hub name 'Concepts' must match the vault-relative 'notes/Concepts.md'.
    """
    # dedup_bounds: target IS notes/Concepts.md, hub is bare name 'Concepts'
    bounds = dedup_bounds("notes/Concepts.md", hub="Concepts")
    hub_op = _op(OpType.patch, "notes/Concepts.md", snippet="some addition")
    kept, rejected = bounds.enforce([hub_op], read_note=lambda p: "# Concepts\n")
    assert len(kept) == 0, "Op targeting hub must be rejected even when hub is a bare name"
    assert len(rejected) == 1


def test_bare_hub_does_not_block_collateral_note():
    """A bare hub name must NOT block a different note that merely shares the same stem.

    Scenario: hub="Foo", target="notes/Bar.md".  An op on "notes/Bar.md" is the
    legitimate repair target — it must pass hub protection even though its directory
    also contains notes whose basename could match other bare forbidden entries.
    The fix: basename expansion is only applied to bare forbidden entries so that
    "notes/Bar.md" is not spuriously blocked because _norm_path(basename) != "foo".
    """
    # dedup_bounds with target=notes/Bar.md, hub bare name "Foo"
    bounds = dedup_bounds("notes/Bar.md", hub="Foo")

    # Op on the actual target (notes/Bar.md) must be allowed — "Bar" != "Foo"
    target_op = _op(OpType.patch, "notes/Bar.md", snippet="[[SomeLink]]")
    kept, rejected = bounds.enforce([target_op], read_note=lambda p: "# Bar\n")
    assert len(kept) == 1, (
        "notes/Bar.md must NOT be blocked by hub='Foo'; "
        f"rejected: {[r['reason'] for r in rejected]}"
    )
    assert len(rejected) == 0

    # Op on the actual hub note (notes/Foo.md) must still be blocked
    hub_op = _op(OpType.patch, "notes/Foo.md", snippet="[[SomeLink]]")
    kept2, rejected2 = bounds.enforce([hub_op], read_note=lambda p: "# Foo\n")
    assert len(kept2) == 0, "notes/Foo.md must be blocked because it matches bare hub 'Foo'"
    assert len(rejected2) == 1


def test_orphan_bounds_allows_repair_when_no_hub():
    """When hub=None, orphan repair must not be blocked."""
    bounds = orphan_bounds(["notes/Neighbor.md"], orphan_title="Orphan", hub=None)
    patch_op = _op(OpType.patch, "notes/Neighbor.md", snippet="[[Orphan]]")
    kept, rejected = bounds.enforce([patch_op], read_note=lambda p: "# Neighbor\n")
    assert len(kept) == 1
    assert len(rejected) == 0
