"""Tests for the silica_cooccurrence_refresh bulk-seed tool.

The embedder-free twin of silica_embed_refresh: seeds the whole-vault
co-occurrence index from note text, with no LM Studio / no network. The
post-write freshness hook then keeps it fresh incrementally.
"""
from __future__ import annotations

import pytest

from silica.kernel.recall.cooccurrence import CooccurStore


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """Isolated fs vault with two seed notes; points the global DRIVER at it."""
    vault_dir = tmp_path / "vault"
    (vault_dir / "Concepts").mkdir(parents=True)
    (vault_dir / "Concepts" / "Neural.md").write_text(
        "---\ntags:\n  - ai\n---\n\n# Neural\n\nneural network architecture\n",
        encoding="utf-8",
    )
    (vault_dir / "Concepts" / "Boats.md").write_text(
        "# Boats\n\nsailing boat harbour\n", encoding="utf-8"
    )
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(vault_dir))
    monkeypatch.setattr("silica.driver._driver", None)
    yield vault_dir
    monkeypatch.setattr("silica.driver._driver", None)


def test_tool_is_registered():
    import silica.tools.composed  # noqa: F401  — importing registers @tool fns
    from silica.tools import TOOLS
    assert "silica_cooccurrence_refresh" in TOOLS
    assert TOOLS["silica_cooccurrence_refresh"].cls == "composed"


def test_refresh_indexes_all_vault_notes(vault):
    from silica.tools.composed import silica_cooccurrence_refresh
    res = silica_cooccurrence_refresh(folder="", force=True)
    assert res["indexed"] == 2
    assert res["total_notes"] == 2
    # the default index path is redirected to tmp by the autouse conftest fixture
    store = CooccurStore()
    paths = store.paths()
    assert any(p.endswith("Neural") for p in paths)
    assert any(p.endswith("Boats") for p in paths)


def test_a_refresh_that_indexed_nothing_leaves_the_file_alone(vault, monkeypatch):
    """The GUI polls `vault_version()`, which digests index mtimes, and every
    graph export calls this tool. Rewriting an unchanged 9.5 MB index still
    moves the mtime, so the strip announced "vault changed" after every build of
    the graph it had just drawn -- with nothing changed."""
    from silica.kernel.recall.cooccurrence import _index_path
    from silica.kernel.recall.sync import disk_stamp, vault_version
    from silica.tools.composed import silica_cooccurrence_refresh

    silica_cooccurrence_refresh(folder="")
    stamp, version = disk_stamp(_index_path()), vault_version()
    assert stamp is not None, "the first refresh has to write the index"

    silica_cooccurrence_refresh(folder="")
    assert disk_stamp(_index_path()) == stamp
    assert vault_version() == version

    # a real change still lands, and force still writes: a gate that only proved
    # "never rewrites" would pass on a tool that never wrote at all
    (vault / "Concepts" / "Third.md").write_text("# Third\n\nsailing harbour\n",
                                                 encoding="utf-8")
    monkeypatch.setattr("silica.driver._driver", None)  # re-read the roster
    silica_cooccurrence_refresh(folder="")
    assert disk_stamp(_index_path()) != stamp
    moved = disk_stamp(_index_path())
    silica_cooccurrence_refresh(folder="", force=True)
    assert disk_stamp(_index_path()) != moved


def test_refresh_builds_real_queryable_contributions(vault):
    from silica.tools.composed import silica_cooccurrence_refresh
    silica_cooccurrence_refresh(force=True)
    store = CooccurStore()
    # "neural network architecture" -> neural<->network co-occurrence edge
    assert store.neighbors("network", k=5)


def test_refresh_works_with_embedder_down(vault, monkeypatch):
    """The stable leg: bulk-seed must succeed even when the embedder is down."""
    import silica.agent.providers as providers

    def _boom(*a, **k):
        raise RuntimeError("LM Studio not running")

    monkeypatch.setattr(providers, "get_embedder", _boom)
    from silica.tools.composed import silica_cooccurrence_refresh
    res = silica_cooccurrence_refresh(force=True)
    assert res.get("indexed") == 2  # embedder never touched


def test_cli_cooccur_command_routes_to_tool():
    """'/cooccur --force' reaches silica_cooccurrence_refresh with force=True."""
    from unittest.mock import patch
    import silica.tools.composed  # noqa: F401 — ensure tool is registered
    from silica.cli import _handle_direct_shortcut
    from silica.tools import TOOLS

    received: dict = {}

    def capture(folder: str = "", force: bool = False):
        received["folder"] = folder
        received["force"] = force
        return {"indexed": 3, "total_notes": 3, "read_errors": 0, "index_path": "/tmp/x"}

    tool = TOOLS["silica_cooccurrence_refresh"]
    orig = tool.fn
    tool.fn = capture
    try:
        with patch("silica.cli.CONSOLE"):
            result = _handle_direct_shortcut("/cooccur --force", [])
    finally:
        tool.fn = orig

    assert result is True
    assert received.get("force") is True


def test_force_rebuild_refreezes_wrong_frozen_language(tmp_path, monkeypatch):
    """Round 2: /cooccur --force is the doctor remedy for a wrong-frozen store.
    The tool must thread refreeze into build_index so a deliberate force
    rebuild re-detects the store language (the sticky freeze that protects the
    write hook must NOT block the user-facing rebuild)."""
    vault_dir = tmp_path / "vault_it"
    (vault_dir / "Note").mkdir(parents=True)
    (vault_dir / "Note" / "Rete.md").write_text(
        "# Rete\n\nLa rete neurale della azienda migliora la produttivita del team.\n",
        encoding="utf-8",
    )
    (vault_dir / "Note" / "Algoritmi.md").write_text(
        "# Algoritmi\n\nGli algoritmi della rete sono ottimizzati per la performance.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(vault_dir))
    monkeypatch.setattr("silica.config.CONFIG.cooccurrence_lang", "auto")
    monkeypatch.setattr("silica.driver._driver", None)

    # Historic-bug state: the store is populated and frozen "english" on an
    # Italian vault (via the same singleton the tool will use).
    from silica.kernel.recall.cooccurrence import build_contribution, get_cooccur_store
    store = get_cooccur_store(lang="english")
    store.lang = "english"
    store.upsert_note("Note/Old", build_contribution("Old", "old english note", lang="english"))
    store.save()

    from silica.tools.composed import silica_cooccurrence_refresh
    res = silica_cooccurrence_refresh(force=True)
    assert "error" not in res
    assert store.lang == "italian"


def test_refresh_empty_vault_returns_error(tmp_path, monkeypatch):
    vault_dir = tmp_path / "empty_vault"
    vault_dir.mkdir()
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(vault_dir))
    monkeypatch.setattr("silica.driver._driver", None)
    from silica.tools.composed import silica_cooccurrence_refresh
    res = silica_cooccurrence_refresh()
    assert "error" in res
