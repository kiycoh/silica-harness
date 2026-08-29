# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Note↔file reference edges: extraction (link/ast), resolution and inverse
maps (fs_backend), and the silica_file_links tool surface."""

import pytest

from silica.driver.fs_backend import ObsidianFSBackend


# ---------------------------------------------------------------------------
# ast level: the one generator gains a file bucket, note contract untouched
# ---------------------------------------------------------------------------

def test_extract_file_refs_all_syntaxes():
    from silica.kernel.link.ast import extract_file_refs, extract_links

    content = (
        "# N\n\n"
        "![[foto.jpg]] and [[report.pdf]] and ![chart](img/chart.png)\n"
        "[the notebook](analysis.ipynb) and [[B]] and ![[clip.mov]]\n"
        "![remote](https://x.com/pic.png)\n"
    )
    refs = extract_file_refs(content)
    assert set(refs) == {"foto.jpg", "report.pdf", "img/chart.png",
                         "analysis.ipynb", "clip.mov"}
    # note-link contract is bit-identical: files never leak into extract_links
    assert extract_links(content) == ["B"]


def test_file_targets_no_longer_phantom_note_links():
    # Before the extension list grew, [[analysis.ipynb]] / [[clip.mov]] read as
    # dangling NOTE links; they are file refs.
    from silica.kernel.link.ast import extract_file_refs, extract_links

    assert extract_links("[[analysis.ipynb]] [[clip.mov]]") == []
    assert extract_file_refs("[[analysis.ipynb]] [[clip.mov]]") == [
        "analysis.ipynb", "clip.mov"]


def test_extract_file_refs_fast_path_and_normalization():
    from silica.kernel.link.ast import extract_file_refs

    assert extract_file_refs("plain prose, no refs at all") == []
    assert extract_file_refs("![[img.png|300]]") == ["img.png"]      # size hint
    assert extract_file_refs("[[doc.pdf#page=3]]") == ["doc.pdf"]    # anchor


def test_extract_refs_typed_matches_links_typed():
    from silica.kernel.link.ast import extract_links_typed, extract_refs_typed

    content = (
        "---\nrelated:\n  - \"[[X]]\"\n---\n\n"
        "## [[Scaffold]]\n\n![[a.png]]\n\nprose [[Y]]\n"
    )
    typed, files, mentions = extract_refs_typed(content)
    assert typed == extract_links_typed(content)
    assert files == ["a.png"]
    assert mentions == []


# ---------------------------------------------------------------------------
# fs_backend: census, resolution, inverse, incremental upsert
# ---------------------------------------------------------------------------

@pytest.fixture
def asset_vault(tmp_path):
    vault = tmp_path / "vault"
    (vault / "img").mkdir(parents=True)
    (vault / "sub").mkdir()
    (vault / "img" / "foto.jpg").write_bytes(b"\xff")
    (vault / "img" / "chart.png").write_bytes(b"\x89")
    (vault / "sub" / "chart.png").write_bytes(b"\x89")   # ambiguous basename
    (vault / "analysis.ipynb").write_text("{}", encoding="utf-8")
    (vault / "A.md").write_text(
        "# A\n\n![[foto.jpg]] and ![c](img/chart.png)\n"
        "[[analysis.ipynb]]\n![[ghost.png]]\n",
        encoding="utf-8",
    )
    (vault / "sub" / "B.md").write_text("# B\n\n![[chart.png]]", encoding="utf-8")
    (vault / "C.md").write_text(
        "---\ndocuments:\n  - src/mod.py\ncode_ref: abc123\n---\n\n# C\n",
        encoding="utf-8",
    )
    return vault


def test_file_refs_of_resolves_and_keeps_unresolved(asset_vault):
    b = ObsidianFSBackend(str(asset_vault))
    fr = b.file_refs_of("A.md")
    assert set(fr["embeds"]) == {"img/foto.jpg", "img/chart.png", "analysis.ipynb"}
    assert fr["unresolved"] == ["ghost.png"]     # recorded, never dropped
    assert fr["documents"] == []


def test_ambiguous_basename_prefers_same_dir(asset_vault):
    b = ObsidianFSBackend(str(asset_vault))
    assert b.file_refs_of("sub/B.md")["embeds"] == ["sub/chart.png"]


def test_file_backlinks_by_path_and_basename(asset_vault):
    b = ObsidianFSBackend(str(asset_vault))
    assert b.file_backlinks("img/foto.jpg")["embeds"] == ["A.md"]
    assert b.file_backlinks("foto.jpg")["embeds"] == ["A.md"]
    assert b.file_backlinks("sub/chart.png")["embeds"] == ["sub/B.md"]


def test_documents_edges_both_directions(asset_vault):
    b = ObsidianFSBackend(str(asset_vault))
    assert b.file_refs_of("C.md")["documents"] == ["src/mod.py"]
    assert b.file_backlinks("src/mod.py")["documents"] == ["C.md"]
    # basename convenience over the repo-relative keyspace
    assert b.file_backlinks("mod.py")["documents"] == ["C.md"]


def test_note_graph_and_gate_surface_untouched(asset_vault):
    b = ObsidianFSBackend(str(asset_vault))
    # file refs never become note links nor dangling notes
    assert b.links("A.md") == []
    assert b.unresolved() == []


def test_incremental_upsert_replaces_asset_edges(asset_vault):
    b = ObsidianFSBackend(str(asset_vault))
    assert set(b.file_refs_of("A.md")["embeds"]) >= {"img/foto.jpg"}
    # rewrite A without the foto embed; edges must follow, not accumulate
    new_body = "# A\n\n![[chart.png]]\n"
    (asset_vault / "A.md").write_text(new_body, encoding="utf-8")
    b._patch_index("A.md", new_body)
    fr = b.file_refs_of("A.md")
    assert fr["embeds"] == ["img/chart.png"]     # root note: shortest path wins
    assert fr["unresolved"] == []
    assert "A.md" not in b.file_backlinks("img/foto.jpg")["embeds"]


# ---------------------------------------------------------------------------
# tool surface
# ---------------------------------------------------------------------------

def test_silica_file_links_both_directions(asset_vault, monkeypatch):
    backend = ObsidianFSBackend(str(asset_vault))
    monkeypatch.setattr("silica.tools.atomic.DRIVER", backend)
    from silica.tools.atomic import silica_file_links

    out = silica_file_links("A")
    assert out["note"] == "A.md"
    assert set(out["embeds"]) == {"img/foto.jpg", "img/chart.png", "analysis.ipynb"}

    inv = silica_file_links("foto.jpg")
    assert inv["embedded_in"] == ["A.md"]

    doc = silica_file_links("src/mod.py")
    assert doc["documented_by"] == ["C.md"]


def test_tabular_targets_are_file_refs_not_note_links():
    # A profile note names its data file (convert.py TABULAR_EXTS); as note
    # targets these were phantom dangling links, invisible to file_backlinks.
    from silica.kernel.link.ast import extract_file_refs, extract_links

    body = "[[vendite.csv]] [[data/log.tsv]] ![[frame.parquet]]"
    assert extract_links(body) == []
    assert extract_file_refs(body) == ["vendite.csv", "data/log.tsv", "frame.parquet"]


# ---------------------------------------------------------------------------
# backtick path mentions: `data/raw/x.csv` in prose is how humans cite files
# (field test 2026-08-28: 163/175 CSV citations were inline code, 0 wikilinks)
# ---------------------------------------------------------------------------

def test_backtick_path_mentions_extracted():
    from silica.kernel.link.ast import extract_refs_typed

    content = (
        "# N\n\n"
        "see `data/raw/vendite.csv` and `foo bar.csv` and `mod.py`\n"
        "and `https://x.com/a.csv` and prose [[B]]\n"
        "```\nfenced/path.csv\n`inner.csv`\n```\n"
    )
    typed, files, mentions = extract_refs_typed(content)
    assert mentions == ["data/raw/vendite.csv"]   # spaced, non-censused-ext,
    assert files == []                            # url and fenced all excluded
    assert "B" in typed


def test_backtick_mentions_survive_the_marker_fast_path():
    # A note with no [[ ]] and no ]( still reaches the parser when it carries
    # a path-shaped inline cite.
    from silica.kernel.link.ast import extract_refs_typed

    _, _, mentions = extract_refs_typed("only a `data/m.csv` cite, no links")
    assert mentions == ["data/m.csv"]


def test_backtick_mention_resolves_to_edge_and_misses_are_dropped(asset_vault):
    (asset_vault / "D.md").write_text(
        "# D\n\nnumbers live in `img/foto.jpg` and in `ghost.csv`\n",
        encoding="utf-8",
    )
    b = ObsidianFSBackend(str(asset_vault))
    fr = b.file_refs_of("D.md")
    assert fr["embeds"] == ["img/foto.jpg"]
    # A wikilink miss is a defect worth reporting; a casual inline cite that
    # misses is prose. Mentions never land in the unresolved bucket.
    assert fr["unresolved"] == []
    assert "D.md" in b.file_backlinks("img/foto.jpg")["embeds"]
