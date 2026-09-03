from silica.kernel.recall.assembly import Unit, fill_budget


def test_seeds_never_trimmed_even_over_budget():
    seeds = [Unit(path="a", text="x" * 5000, is_seed=True, rank=0)]
    kept, trunc = fill_budget(seeds, [], budget=3000)
    assert [u.path for u in kept] == ["a"]        # protect-seeds invariant
    assert trunc.dropped == []


def test_periphery_fills_by_rank_then_reports_drops():
    seeds = [Unit(path="s", text="x" * 1000, is_seed=True, rank=0)]
    periphery = [
        Unit(path="p1", text="y" * 800, is_seed=False, rank=0),
        Unit(path="p2", text="z" * 800, is_seed=False, rank=1),
        Unit(path="p3", text="w" * 800, is_seed=False, rank=2),
    ]
    kept, trunc = fill_budget(seeds, periphery, budget=2600)  # room for s + p1 + p2
    assert [u.path for u in kept] == ["s", "p1", "p2"]
    assert trunc.dropped == ["p3"]
    assert trunc.kept == 3


from silica.kernel.recall.assembly import relevel_headers, squash


def test_relevel_shifts_headings_capped_at_six():
    body = "# Title\n\ntext\n\n## Sub\n"
    assert relevel_headers(body, 1) == "## Title\n\ntext\n\n### Sub\n"
    assert relevel_headers("###### Deep\n", 2) == "###### Deep\n"  # capped
    assert relevel_headers(body, 0) == body


def test_squash_groups_two_units_under_same_hub():
    units = [
        Unit(path="spoke-a", text="# A\naaa", is_seed=True, rank=0),
        Unit(path="spoke-b", text="# B\nbbb", is_seed=True, rank=1),
    ]
    hub_of = {"spoke-a": "Hub", "spoke-b": "Hub"}
    crumb = {"spoke-a": "Hub > A", "spoke-b": "Hub > B"}
    blocks = squash(units, hub_of, crumb)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.hub == "Hub"
    assert b.members == ["spoke-a", "spoke-b"]
    assert b.text.startswith("Hub > A")  # breadcrumb prefix
    assert "# Hub" in b.text          # hub header present
    assert "## A" in b.text and "## B" in b.text  # members re-leveled under it


def test_single_seed_is_not_squashed():
    units = [Unit(path="only", text="# Solo\nx", is_seed=True, rank=0)]
    blocks = squash(units, {"only": "Hub"}, {"only": "Hub > Solo"})
    assert len(blocks) == 1
    assert blocks[0].hub is None            # degenerate: not a squash
    assert blocks[0].members == ["only"]
    assert blocks[0].text.startswith("Hub > Solo")  # breadcrumb prefix


from silica.kernel.recall.assembly import Neighbors, Caps, assemble


def _fixture():
    # hub <- spoke1, spoke2 (both parent=hub); spoke1 -> related1; spoke1 edge e1
    bodies = {
        "hub": "# Hub\nhub body",
        "spoke1": "# Spoke1\ns1 body",
        "spoke2": "# Spoke2\ns2 body",
        "related1": "# Related1\nr1 body",
        "e1": "# E1\ne1 body",
    }
    nbrs = {
        "spoke1": Neighbors(parent="hub", children=[], related=["related1"], edges=["e1"]),
        "spoke2": Neighbors(parent="hub", children=[], related=[], edges=[]),
        "hub": Neighbors(parent=None, children=["spoke1", "spoke2"], related=[], edges=[]),
        "related1": Neighbors(parent=None, children=[], related=[], edges=[]),
        "e1": Neighbors(parent=None, children=[], related=[], edges=[]),
    }
    return (lambda p: nbrs.get(p, Neighbors(None, [], [], [])),
            lambda p: bodies.get(p, ""))


def test_direction_zero_skips_that_direction():
    neighbors_of, body_of = _fixture()
    caps = Caps(parent=1, children=0, related=0, edges=0)
    res = assemble(["spoke1"], neighbors_of=neighbors_of, body_of=body_of, caps=caps)
    seen = {p for b in res.blocks for p in b.members}
    assert "spoke1" in seen and "hub" in seen   # parent hop taken
    assert "related1" not in seen and "e1" not in seen  # skipped directions


def test_two_seeds_sharing_hub_squash_into_one_block():
    neighbors_of, body_of = _fixture()
    caps = Caps(parent=1, children=0, related=0, edges=0)
    res = assemble(["spoke1", "spoke2"], neighbors_of=neighbors_of,
                   body_of=body_of, caps=caps)
    squashed = [b for b in res.blocks if b.hub == "hub"]
    assert len(squashed) == 1
    assert set(squashed[0].members) >= {"spoke1", "spoke2"}


def test_empty_seeds_returns_empty():
    neighbors_of, body_of = _fixture()
    res = assemble([], neighbors_of=neighbors_of, body_of=body_of)
    assert res.blocks == []


def test_a_periphery_neighbour_never_leads_a_seed_in_a_hub_block():
    """Seed rank and periphery rank are separate spaces both starting at 0.
    Comparing them raw let periphery rank 0 sort before seed rank 1 and become
    members[0] — the member downstream attribution reads. Seeds first, always."""
    units = [
        Unit(path="seed-late", text="seed body", is_seed=True, rank=1),
        Unit(path="peri-early", text="peri body", is_seed=False, rank=0),
    ]
    hub_of = {"seed-late": "Hub", "peri-early": "Hub"}
    blocks = squash(units, hub_of, {})
    assert len(blocks) == 1
    assert blocks[0].members[0] == "seed-late"


def test_a_periphery_only_block_never_outranks_a_seed_block():
    units = [
        Unit(path="seed", text="seed body", is_seed=True, rank=3),
        Unit(path="peri", text="peri body", is_seed=False, rank=0),
    ]
    blocks = squash(units, {"seed": None, "peri": None}, {})
    assert [b.members[0] for b in blocks] == ["seed", "peri"]
