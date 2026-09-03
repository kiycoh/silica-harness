"""Vault-internal config (taxonomy/overlay) lives at the vault root, with a
read-time fallback to the legacy _silica/ namespace."""

from silica.config import CONFIG


def test_default_taxonomy_path_prefers_vault_root(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    from silica.kernel.organize.taxonomy import default_taxonomy_path
    assert default_taxonomy_path() == tmp_path / "taxonomy.yaml"


def test_default_taxonomy_path_falls_back_to_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    legacy = tmp_path / "_silica" / "taxonomy.yaml"
    legacy.parent.mkdir()
    legacy.write_text("folders: {}\n", encoding="utf-8")
    from silica.kernel.organize.taxonomy import default_taxonomy_path
    assert default_taxonomy_path() == legacy


def test_default_taxonomy_root_wins_over_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    (tmp_path / "taxonomy.yaml").write_text("folders: {}\n", encoding="utf-8")
    legacy = tmp_path / "_silica" / "taxonomy.yaml"
    legacy.parent.mkdir()
    legacy.write_text("folders: {}\n", encoding="utf-8")
    from silica.kernel.organize.taxonomy import default_taxonomy_path
    assert default_taxonomy_path() == tmp_path / "taxonomy.yaml"


def test_overlay_prefers_vault_root(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    from silica.kernel.text import overlay as overlay_mod
    overlay_mod.reset_overlay_cache()
    # stopwords list extends the default; "zzzcustomword" is merged in
    (tmp_path / "overlay.yaml").write_text("stopwords:\n  - zzzcustomword\n", encoding="utf-8")
    try:
        ov = overlay_mod.get_active_overlay()
        assert "zzzcustomword" in ov.stopwords
    finally:
        overlay_mod.reset_overlay_cache()


def test_overlay_falls_back_to_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    from silica.kernel.text import overlay as overlay_mod
    overlay_mod.reset_overlay_cache()
    legacy = tmp_path / "_silica" / "overlay.yaml"
    legacy.parent.mkdir()
    legacy.write_text("stopwords:\n  - zzzlegacyword\n", encoding="utf-8")
    try:
        ov = overlay_mod.get_active_overlay()
        assert "zzzlegacyword" in ov.stopwords
    finally:
        overlay_mod.reset_overlay_cache()
