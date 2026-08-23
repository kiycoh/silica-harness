import pytest
from silica.driver.fs_backend import ObsidianFSBackend

@pytest.fixture
def temp_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    
    # Create notes
    # A links to B and MissingNote
    (vault / "A.md").write_text("# A\n\n[[B]] and [[MissingNote]]", encoding="utf-8")
    # B links to C
    (vault / "B.md").write_text("# B\n\n[[C]]", encoding="utf-8")
    # C is a leaf note
    (vault / "C.md").write_text("# C\n\nNo links here", encoding="utf-8")
    # D is an orphan (no incoming/outgoing links)
    (vault / "D.md").write_text("# D\n\nOrphan note", encoding="utf-8")
    
    return vault

def test_fs_backend_graph_basic(temp_vault):
    backend = ObsidianFSBackend(str(temp_vault))
    
    # Assert orphans
    # D has no incoming links. A has no incoming links either!
    # A, D are orphans
    orphans = backend.orphans()
    orphan_names = {o.name for o in orphans}
    assert orphan_names == {"A", "D"}
    
    # Assert links (outgoing) from A
    a_links = backend.links("A.md")
    a_link_names = {l.name for l in a_links}
    assert "B" in a_link_names
    assert "MissingNote" in a_link_names
    
    # Assert links (outgoing) from B
    b_links = backend.links("B.md")
    assert [l.name for l in b_links] == ["C"]
    
    # Assert links (outgoing) from C
    c_links = backend.links("C.md")
    assert c_links == []
    
    # Assert backlinks (incoming) to B
    b_backlinks = backend.backlinks("B.md")
    assert [bl.name for bl in b_backlinks] == ["A"]
    
    # Assert backlinks (incoming) to C
    c_backlinks = backend.backlinks("C.md")
    assert [bl.name for bl in c_backlinks] == ["B"]
    
    # Assert unresolved links
    unres = backend.unresolved()
    assert len(unres) == 1
    assert unres[0].source.name == "A"
    assert unres[0].target == "MissingNote"
    
    # Assert full snapshot
    snap = backend.graph_snapshot()
    assert {o.name for o in snap.orphans} == {"A", "D"}
    assert len(snap.unresolved) == 1
    assert snap.unresolved[0].target == "MissingNote"
    assert snap.link_counts["A"] == 2
    assert snap.link_counts["B"] == 1
    assert snap.link_counts["C"] == 0
    assert snap.link_counts["D"] == 0
    assert snap.backlink_counts["A"] == 0
    assert snap.backlink_counts["B"] == 1
    assert snap.backlink_counts["C"] == 1
    assert snap.backlink_counts["D"] == 0

def test_fs_backend_graph_snapshot_incremental(temp_vault):
    backend = ObsidianFSBackend(str(temp_vault))
    backend._ensure_index()
    
    # Incremental snapshot for B.md
    b_ref = backend._notes["B.md"]
    snap = backend.graph_snapshot(refs=[b_ref])
    
    # B.md has links B->C and backlink A->B
    # Neighborhood should cover A, B, C (but not D)
    assert "A" in snap.link_counts
    assert "B" in snap.link_counts
    assert "C" in snap.link_counts
    assert "D" not in snap.link_counts
    
    # check backlink counts
    assert snap.backlink_counts["A"] == 0
    assert snap.backlink_counts["B"] == 1
    assert snap.backlink_counts["C"] == 1
    
    # check orphans in neighborhood
    assert {o.name for o in snap.orphans} == {"A"}


def test_fs_backend_duplicate_basename_snapshot(tmp_path):
    """Two notes with the same basename in different folders must produce separate
    entries in link_counts/backlink_counts, keyed by canonical path (not name).

    Before the Option-A fix, both would collapse onto the same dict key and
    one of them would silently be overwritten — producing wrong counts.
    """
    vault = tmp_path / "vault"
    (vault / "folder_a").mkdir(parents=True)
    (vault / "folder_b").mkdir(parents=True)

    # Two notes named "Note" in different folders
    (vault / "folder_a" / "Note.md").write_text("# Note A\n\n[[Hub]]", encoding="utf-8")
    (vault / "folder_b" / "Note.md").write_text("# Note B\n\nNo links here.", encoding="utf-8")
    # Hub is linked by folder_a/Note.md but NOT by folder_b/Note.md
    (vault / "Hub.md").write_text("# Hub\n\nNo links here.", encoding="utf-8")

    backend = ObsidianFSBackend(str(vault))
    snap = backend.graph_snapshot()

    # With path-keyed snapshot there must be two distinct "Note" entries
    note_keys = [k for k in snap.backlink_counts if k.lower().endswith("/note") or k.lower() == "note"]
    assert len(note_keys) == 2, (
        f"Expected 2 distinct Note entries (one per path), got {note_keys}. "
        "Duplicate basenames are collapsing onto a single key — Option-A fix not applied."
    )

    # Hub should have exactly 1 backlink (only folder_a/Note links to it)
    hub_key = next(k for k in snap.backlink_counts if k.lower().endswith("hub"))
    assert snap.backlink_counts[hub_key] == 1

    # folder_b/Note is an orphan (no incoming links), folder_a/Note is also an orphan
    orphan_paths = {o.path for o in snap.orphans}
    assert "folder_a/Note.md" in orphan_paths
    assert "folder_b/Note.md" in orphan_paths


def test_vault_root_artifacts_excluded_from_index(tmp_path):
    """GRAPH_REPORT.md / log.md are Silica's own output, not knowledge notes.

    They must stay out of the driver index so they never reach any metric
    (list_files → embed + cooccurrence, graph).
    A real note in a subfolder named log.md must NOT be excluded.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Real.md").write_text("# Real\n\nabout apples", encoding="utf-8")
    (vault / "GRAPH_REPORT.md").write_text("# Graph report\n\n[[Real]] apples", encoding="utf-8")
    (vault / "log.md").write_text("# Log\n\n[[Real]] apples", encoding="utf-8")
    (vault / "notes").mkdir()
    (vault / "notes" / "log.md").write_text("# genuine log note\n\napples", encoding="utf-8")

    backend = ObsidianFSBackend(str(vault))
    paths = {r.path for r in backend.list_files()}
    assert "GRAPH_REPORT.md" not in paths
    assert "log.md" not in paths
    assert "notes/log.md" in paths  # subfolder note survives
    assert "Real.md" in paths
    # Graph: the artifacts' [[Real]] links must not count as backlinks either.
    assert {r.path for r in backend.backlinks("Real.md")} == set()


def test_links_resolve_every_ref_shape(tmp_path):
    """links()/backlinks() accept a bare name, a vault-relative path, +/- ".md".

    The tools hand the model {name, path} pairs, so it feeds back either one.
    Trusting a ".md" suffix as a graph key made links("Note.md") return [] for
    a note stored at folder/sub/Note.md — a silent empty, not an error.
    """
    vault = tmp_path / "vault"
    (vault / "folder" / "sub").mkdir(parents=True)
    (vault / "folder" / "sub" / "Note.md").write_text("[[Target]]", encoding="utf-8")
    (vault / "Target.md").write_text("# Target", encoding="utf-8")

    backend = ObsidianFSBackend(str(vault))
    for ref in ("Note", "Note.md", "folder/sub/Note", "folder/sub/Note.md"):
        assert [r.name for r in backend.links(ref)] == ["Target"], ref
    for ref in ("Target", "Target.md"):
        assert [r.name for r in backend.backlinks(ref)] == ["Note"], ref

    assert backend.links("Nonexistent") == []


def test_reads_that_iterate_serialize_against_index_mutation(temp_vault):
    """The MCP server dispatches each request on its own thread; a write's
    _patch_index interleaving with a search iterating _notes/_graph raised
    "dictionary changed size during iteration". The index lock is the fix:
    while a mutator holds it, an iterating read must wait, not interleave."""
    import threading

    backend = ObsidianFSBackend(str(temp_vault))
    backend._ensure_index()

    done = threading.Event()

    def _search():
        backend.search_context("links")
        done.set()

    with backend._index_lock:              # simulate a mutator mid-flight
        t = threading.Thread(target=_search, daemon=True)
        t.start()
        assert not done.wait(0.2)          # the read is queued, not interleaved
    assert done.wait(2)                    # lock released -> read completes
    t.join(2)


def test_search_context_caps_per_note_and_batch_agrees(temp_vault):
    """A broad query must not materialize one Hit per matching line ("e" made
    14535 once): per-note cap at the level the tool layer can still rank by,
    and the batch variant stays byte-for-byte with the single one."""
    from silica.driver.fs_backend import _MAX_HITS_PER_NOTE

    lines = "\n".join(f"needle line {i}" for i in range(_MAX_HITS_PER_NOTE + 15))
    (temp_vault / "Dense.md").write_text(f"# Dense\n\n{lines}\n", encoding="utf-8")

    backend = ObsidianFSBackend(str(temp_vault))
    hits = backend.search_context("needle")
    assert len(hits) == _MAX_HITS_PER_NOTE

    batch = backend.search_context_batch(["needle", "line 3"])
    assert [(h.ref.path, h.line) for h in batch["needle"]] == \
        [(h.ref.path, h.line) for h in hits]
    assert [(h.ref.path, h.line) for h in batch["line 3"]] == \
        [(h.ref.path, h.line) for h in backend.search_context("line 3")]


def test_graph_edges_carry_the_scaffold_class(tmp_path):
    """A frontmatter link is a scaffold edge; a prose link is not; two
    spellings of one note resolve to one edge and prose wins (ADR-0029)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Hub.md").write_text("# Hub\n\n## [[Spoke A]]\n\nProse about [[Spoke B]].\n", encoding="utf-8")
    (vault / "Spoke A.md").write_text("---\nparent note: \"[[Hub]]\"\n---\n\n# Spoke A\n\nSee [[hub]] in prose.\n", encoding="utf-8")
    (vault / "Spoke B.md").write_text("---\nparent note: \"[[Hub]]\"\n---\n\n# Spoke B\n\nNo prose link.\n", encoding="utf-8")
    backend = ObsidianFSBackend(str(vault))
    _notes, _unresolved, G = backend.graph_data()
    assert G["Hub.md"]["Spoke A.md"]["scaffold"] is True      # heading line
    assert G["Hub.md"]["Spoke B.md"]["scaffold"] is False     # prose
    assert G["Spoke A.md"]["Hub.md"]["scaffold"] is False     # frontmatter + prose: prose wins
    assert G["Spoke B.md"]["Hub.md"]["scaffold"] is True      # frontmatter only
    # the incremental path keeps the class too
    backend.create("Spoke C.md", "---\nparent note: \"[[Hub]]\"\n---\n\n# Spoke C\n", )
    _notes, _unresolved, G = backend.graph_data()
    assert G["Spoke C.md"]["Hub.md"]["scaffold"] is True
    from silica.kernel.recall.graph_export import build_graph_data
    import silica.driver as drv
    drv.DRIVER  # noqa: B018 - the export reads the configured driver; exercised via the nx graph above
