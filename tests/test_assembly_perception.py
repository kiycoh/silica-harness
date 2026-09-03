"""perceive(assemble=True) folds neighbours into the seed blocks; default off
leaves the block list byte-identical."""
from silica.kernel.recall import perception
from silica.kernel.recall.perception import NoteBlock


def test_assemble_off_is_identical(monkeypatch):
    base = [NoteBlock(path="a", date="", evidence="embed:0.9", body="# A\nx",
                      excerpt="# A\nx")]

    def fake_perceive_core(*a, **k):
        return list(base)

    # Assemble=False must not call the assembler at all.
    monkeypatch.setattr(perception, "_assemble_blocks",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    blocks = perception._maybe_assemble(list(base), assemble=False, query="q")
    assert blocks == base


def test_assemble_on_calls_assembler(monkeypatch):
    base = [NoteBlock(path="a", date="", evidence="", body="# A\nx", excerpt="# A\nx")]
    called = {}

    def spy(blocks, query):
        called["yes"] = True
        return blocks

    monkeypatch.setattr(perception, "_assemble_blocks", spy)
    out = perception._maybe_assemble(list(base), assemble=True, query="q")
    assert called.get("yes") is True
    assert out == base


def test_seed_body_reused_not_refetched(monkeypatch):
    """A memory-lane seed whose body is already on the NoteBlock must survive
    assemble=True — _assemble_blocks reuses by_path[seed].body, never re-reads
    it via _assembly_body (which would read the vault with the wrong origin)."""
    from silica.kernel.recall import assembly
    base = [NoteBlock(path="mem/x", date="d", evidence="e",
                      body="REAL MEMORY BODY", excerpt="REAL MEMORY BODY")]
    # A single seed with no neighbours -> one lone assembled block.
    monkeypatch.setattr(perception, "_driver_neighbors",
                        lambda p: assembly.Neighbors(None, [], [], []))
    # If the seed were re-read through _assembly_body it would get this sentinel
    # (or "" in the real memory-lane bug); it must NOT appear.
    monkeypatch.setattr(perception, "_assembly_body",
                        lambda p: "WRONG_REFETCH")
    out = perception._assemble_blocks(list(base), "q")
    assert len(out) == 1
    assert "REAL MEMORY BODY" in out[0].body
    assert "WRONG_REFETCH" not in out[0].body
