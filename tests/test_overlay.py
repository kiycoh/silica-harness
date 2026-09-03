"""Tests for silica.kernel.text.overlay — DomainOverlay seam."""
from __future__ import annotations

import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Default overlay
# ---------------------------------------------------------------------------

class TestDefaultOverlay:
    def test_english_function_word_is_stopword(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY
        assert "the" in DEFAULT_OVERLAY.stopwords

    def test_structural_term_is_stopword(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY
        assert "chapter" in DEFAULT_OVERLAY.stopwords

    def test_more_structural_terms(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY
        for term in ("lesson", "exercise", "professor", "university", "syllabus", "exam"):
            assert term in DEFAULT_OVERLAY.stopwords, f"expected '{term}' in stopwords"

    def test_heading_pattern_matches(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY
        assert any(p.search("Chapter 3: Introduction") for p in DEFAULT_OVERLAY.noise_patterns)

    def test_heading_pattern_lesson(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY
        assert any(p.search("Lesson 1 Arrays") for p in DEFAULT_OVERLAY.noise_patterns)

    def test_content_concept_not_matched(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY
        text = "Backpropagation"
        assert text.lower() not in DEFAULT_OVERLAY.stopwords
        assert not any(p.search(text) for p in DEFAULT_OVERLAY.noise_patterns)

    def test_stopwords_are_lowercase(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY
        for w in DEFAULT_OVERLAY.stopwords:
            assert w == w.lower(), f"stopword '{w}' not lowercase"

    def test_noise_patterns_are_compiled(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY
        for p in DEFAULT_OVERLAY.noise_patterns:
            assert isinstance(p, re.Pattern)

    def test_numeric_prefix_noise(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY
        assert any(p.search("1. Introduction") for p in DEFAULT_OVERLAY.noise_patterns)

    def test_question_suffix_noise(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY
        assert any(p.search("What is recursion?") for p in DEFAULT_OVERLAY.noise_patterns)

    def test_vs_noise(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY
        assert any(p.search("arrays vs lists") for p in DEFAULT_OVERLAY.noise_patterns)


# ---------------------------------------------------------------------------
# load_overlay — merge behaviour
# ---------------------------------------------------------------------------

class TestLoadOverlayMerge:
    def test_extends_default_true_inherits_english_stopword(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        f = tmp_path / "overlay.yaml"
        f.write_text("extends_default: true\nstopwords: [foo]\nnoise_patterns: []\n")
        ov = load_overlay(f)
        assert "the" in ov.stopwords
        assert "foo" in ov.stopwords

    def test_extends_default_absent_defaults_to_true(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        f = tmp_path / "overlay.yaml"
        f.write_text("stopwords: [bar]\nnoise_patterns: []\n")
        ov = load_overlay(f)
        assert "the" in ov.stopwords
        assert "bar" in ov.stopwords

    def test_extends_default_false_replaces_entirely(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        f = tmp_path / "overlay.yaml"
        f.write_text("extends_default: false\nstopwords: [only_this]\nnoise_patterns: []\n")
        ov = load_overlay(f)
        assert "only_this" in ov.stopwords
        assert "the" not in ov.stopwords

    def test_noise_patterns_merged(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        custom_pattern = r"^CUSTOM:\s"
        f = tmp_path / "overlay.yaml"
        f.write_text(f"extends_default: true\nstopwords: []\nnoise_patterns:\n  - '{custom_pattern}'\n")
        ov = load_overlay(f)
        # custom pattern present
        assert any(p.search("CUSTOM: something") for p in ov.noise_patterns)
        # default patterns still present
        assert len(ov.noise_patterns) > 1

    def test_noise_patterns_replaced_when_false(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        custom_pattern = r"^CUSTOM:\s"
        f = tmp_path / "overlay.yaml"
        f.write_text(f"extends_default: false\nstopwords: []\nnoise_patterns:\n  - '{custom_pattern}'\n")
        ov = load_overlay(f)
        # Only the custom pattern
        assert any(p.search("CUSTOM: something") for p in ov.noise_patterns)
        # Default heading pattern gone
        assert not any(p.search("Chapter 3: Intro") for p in ov.noise_patterns)

    def test_patterns_compiled_ignorecase(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        f = tmp_path / "overlay.yaml"
        f.write_text("extends_default: false\nstopwords: []\nnoise_patterns:\n  - '^hello'\n")
        ov = load_overlay(f)
        assert any(p.search("HELLO world") for p in ov.noise_patterns)

    def test_invalid_regex_raises_valueerror_naming_path(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        f = tmp_path / "bad_overlay.yaml"
        f.write_text("stopwords: []\nnoise_patterns:\n  - '[invalid'\n")
        with pytest.raises(ValueError, match=str(f)):
            load_overlay(f)

    def test_malformed_yaml_raises_valueerror_naming_path(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        f = tmp_path / "broken.yaml"
        f.write_text("stopwords: [unclosed\n")
        with pytest.raises(ValueError, match=str(f)):
            load_overlay(f)


# ---------------------------------------------------------------------------
# get_active_overlay + cache isolation
# ---------------------------------------------------------------------------

class TestGetActiveOverlay:
    def test_default_when_no_file(self, tmp_path: Path, monkeypatch):
        from silica.kernel.text import overlay as ov_mod
        import silica.config
        vault = tmp_path / "vault"
        vault.mkdir()
        monkeypatch.setattr(silica.config.CONFIG, "vault_path", str(vault))
        ov_mod.reset_overlay_cache()
        result = ov_mod.get_active_overlay()
        assert "the" in result.stopwords

    def test_file_is_honored(self, tmp_path: Path, monkeypatch):
        from silica.kernel.text import overlay as ov_mod
        import silica.config
        vault = tmp_path / "vault"
        silica_dir = vault / "_silica"
        silica_dir.mkdir(parents=True)
        overlay_file = silica_dir / "overlay.yaml"
        overlay_file.write_text(
            "extends_default: false\nstopwords: [custom_term]\nnoise_patterns: []\n"
        )
        monkeypatch.setattr(silica.config.CONFIG, "vault_path", str(vault))
        ov_mod.reset_overlay_cache()
        result = ov_mod.get_active_overlay()
        assert "custom_term" in result.stopwords
        assert "the" not in result.stopwords

    def test_result_is_cached(self, tmp_path: Path, monkeypatch):
        from silica.kernel.text import overlay as ov_mod
        import silica.config
        vault = tmp_path / "vault"
        vault.mkdir()
        monkeypatch.setattr(silica.config.CONFIG, "vault_path", str(vault))
        ov_mod.reset_overlay_cache()
        first = ov_mod.get_active_overlay()
        second = ov_mod.get_active_overlay()
        assert first is second

    def test_reset_cache_clears(self, tmp_path: Path, monkeypatch):
        """After reset_overlay_cache, a vault switch is picked up."""
        from silica.kernel.text import overlay as ov_mod
        import silica.config

        # First call: no overlay file → default
        vault = tmp_path / "vault"
        vault.mkdir()
        monkeypatch.setattr(silica.config.CONFIG, "vault_path", str(vault))
        ov_mod.reset_overlay_cache()
        first = ov_mod.get_active_overlay()
        assert "the" in first.stopwords

        # Now place an overlay file with extends_default: false
        silica_dir = vault / "_silica"
        silica_dir.mkdir()
        (silica_dir / "overlay.yaml").write_text(
            "extends_default: false\nstopwords: [reset_term]\nnoise_patterns: []\n"
        )
        # Without reset the cache still returns the old result
        cached = ov_mod.get_active_overlay()
        assert "reset_term" not in cached.stopwords

        # After reset the new file is picked up
        ov_mod.reset_overlay_cache()
        second = ov_mod.get_active_overlay()
        assert "reset_term" in second.stopwords
        assert "the" not in second.stopwords


# ---------------------------------------------------------------------------
# Italian academic example overlay
# ---------------------------------------------------------------------------

class TestItalianExampleOverlay:
    @pytest.fixture
    def it_overlay(self):
        from silica.kernel.text.overlay import load_overlay
        example_path = (
            Path(__file__).resolve().parent.parent
            / "silica"
            / "overlays"
            / "italian.yaml"
        )
        assert example_path.exists(), f"bundled overlay not found: {example_path}"
        return load_overlay(example_path)

    def test_file_loads_without_error(self, it_overlay):
        pass  # fixture loading is the test

    def test_contains_unipa(self, it_overlay):
        assert "unipa" in it_overlay.stopwords

    def test_contains_multi_word_entry(self, it_overlay):
        assert "materiale didattico" in it_overlay.stopwords

    def test_italian_heading_pattern_rejects(self, it_overlay):
        assert any(p.search("Capitolo 3: Reti Neurali") for p in it_overlay.noise_patterns)

    def test_continua_pattern(self, it_overlay):
        assert any(p.search("slide (continua)") for p in it_overlay.noise_patterns)


# ---------------------------------------------------------------------------
# ValueError contracts — malformed overlay shape
# ---------------------------------------------------------------------------

class TestLoadOverlayValueErrorContracts:
    def test_top_level_list_raises_valueerror_naming_file(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        f = tmp_path / "list_overlay.yaml"
        f.write_text("- foo\n- bar\n")
        with pytest.raises(ValueError, match=str(f)):
            load_overlay(f)

    def test_stopwords_scalar_raises_valueerror(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        f = tmp_path / "scalar_sw.yaml"
        f.write_text("stopwords: just_a_scalar\nnoise_patterns: []\n")
        with pytest.raises(ValueError, match="stopwords"):
            load_overlay(f)

    def test_stopwords_list_with_non_string_raises_valueerror(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        f = tmp_path / "bad_sw_item.yaml"
        f.write_text("stopwords:\n  - foo\n  - 123\nnoise_patterns: []\n")
        with pytest.raises(ValueError, match="stopwords"):
            load_overlay(f)

    def test_noise_patterns_scalar_raises_valueerror(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        f = tmp_path / "scalar_np.yaml"
        f.write_text("stopwords: []\nnoise_patterns: just_a_scalar\n")
        with pytest.raises(ValueError, match="noise_patterns"):
            load_overlay(f)

    def test_noise_patterns_list_with_non_string_raises_valueerror(self, tmp_path: Path):
        from silica.kernel.text.overlay import load_overlay
        f = tmp_path / "bad_np_item.yaml"
        f.write_text("stopwords: []\nnoise_patterns:\n  - '^valid'\n  - 456\n")
        with pytest.raises(ValueError, match="noise_patterns"):
            load_overlay(f)


def test_bundled_italian_overlay_present_with_cfu():
    """The Italian overlay ships under silica/overlays and includes cfu + lezione."""
    from pathlib import Path
    import silica.kernel.text.overlay as ov_mod
    from silica.kernel.text.overlay import load_overlay
    p = Path(ov_mod.__file__).resolve().parents[2] / "overlays" / "italian.yaml"
    assert p.exists(), f"bundled overlay missing: {p}"
    o = load_overlay(p)
    assert "cfu" in o.stopwords
    assert "lezione" in o.stopwords


class TestOverlayForLang:
    def setup_method(self):
        from silica.kernel.text.overlay import reset_overlay_cache
        reset_overlay_cache()

    def teardown_method(self):
        from silica.kernel.text.overlay import reset_overlay_cache
        reset_overlay_cache()

    def test_italian_uses_bundled_overlay(self):
        from silica.kernel.text.overlay import overlay_for_lang
        ov = overlay_for_lang("italian")
        assert "lezione" in ov.stopwords  # structural term from the bundle
        assert "di" in ov.stopwords       # IT function word

    def test_english_is_default(self):
        from silica.kernel.text.overlay import overlay_for_lang, DEFAULT_OVERLAY
        assert overlay_for_lang("english") is DEFAULT_OVERLAY

    def test_known_language_without_bundle_gets_function_words(self):
        from silica.kernel.text.overlay import overlay_for_lang, DEFAULT_OVERLAY
        ov = overlay_for_lang("french")  # no bundled silica/overlays/french.yaml
        assert ov is not DEFAULT_OVERLAY
        assert "les" in ov.stopwords or "des" in ov.stopwords  # FR function words

    def test_unsupported_language_falls_back_to_default(self):
        from silica.kernel.text.overlay import overlay_for_lang, DEFAULT_OVERLAY
        assert overlay_for_lang("klingon") is DEFAULT_OVERLAY


class TestLanguageOverlay:
    def test_french_adds_function_words_over_default_structurals(self):
        from silica.kernel.text.overlay import language_overlay, DEFAULT_OVERLAY
        ov = language_overlay("french")
        assert "les" in ov.stopwords
        assert DEFAULT_OVERLAY.stopwords <= ov.stopwords  # extends, never narrows

    def test_unsupported_language_returns_default(self):
        from silica.kernel.text.overlay import language_overlay, DEFAULT_OVERLAY
        assert language_overlay("klingon") is DEFAULT_OVERLAY


class TestArchaicEnglish:
    """A library of 1560-1900 scripture and masonic history distilled into notes
    titled `Saith the Lord of Spirits`, `Spirits hath revealed`, `Surround the
    Lord`. Snowball's English stoplist is modern: every archaic function word
    reads to the keyphrase miner as a content word, and the recurring liturgical formula wins
    on frequency."""

    def test_archaic_function_words_are_stopwords(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY

        for w in ("saith", "hath", "thou", "thee", "thy", "unto", "ye", "doth"):
            assert w in DEFAULT_OVERLAY.stopwords, w

    def test_modern_content_words_are_untouched(self):
        from silica.kernel.text.overlay import DEFAULT_OVERLAY

        for w in ("lord", "spirit", "angel", "covenant", "temple"):
            assert w not in DEFAULT_OVERLAY.stopwords, w
