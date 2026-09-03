# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""A whole-vault /embed first build also seeds lexical.json.

Field 2026-09-01: 0 of 8 vaults on the machine had a lexical index (a separate
opt-in), every indexed one had embeddings, so silica_vaults could only nominate
through the expensive stage. The sweep maintains stores that exist and never
builds one, so the first build is where the cheap stage has to be born.
"""
from __future__ import annotations

from silica.kernel.recall.paths import index_dir


class _Emb:
    model = "fake"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _vault(tmp_path, monkeypatch):
    v = tmp_path / "vault"
    v.mkdir()
    (v / "Neural.md").write_text("# Neural\n\nneural network architecture\n", encoding="utf-8")
    (v / "Boats.md").write_text("# Boats\n\nsailing boat harbour\n", encoding="utf-8")
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(v))
    monkeypatch.setattr("silica.driver._driver", None)
    monkeypatch.setattr("silica.agent.providers.get_embedder", lambda cfg: _Emb())
    return v


def test_whole_vault_embed_build_seeds_the_lexical_index(tmp_path, monkeypatch):
    from silica.tools.graph import silica_embed_refresh

    _vault(tmp_path, monkeypatch)
    out = silica_embed_refresh()
    assert out["indexed"] == 2 and out["lexical"] == 2
    assert (index_dir() / "lexical.json").is_file()
    again = silica_embed_refresh()  # the index exists now: the sweep owns it, /embed leaves it alone
    assert "lexical" not in again


def test_a_folder_scoped_build_does_not_seed_a_partial_lexical_index(tmp_path, monkeypatch):
    from silica.tools.graph import silica_embed_refresh

    v = _vault(tmp_path, monkeypatch)
    (v / "Sub").mkdir()
    (v / "Sub" / "Deep.md").write_text("# Deep\n\ndeep sea diving\n", encoding="utf-8")
    out = silica_embed_refresh(folder="Sub")
    assert out["indexed"] == 1 and "lexical" not in out
    assert not (index_dir() / "lexical.json").exists()
