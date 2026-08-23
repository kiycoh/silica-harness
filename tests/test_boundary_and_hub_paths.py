# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""The write boundary covers the folders Silica creates for its own bookkeeping.

`done/` and `sources/` were built from bare constants, so a vault confined to
`silica/` still got them dropped at its root — and archiving MOVED a source note
the boundary exists to keep untouched.
"""
from __future__ import annotations

import pytest

from silica.kernel.vault_manifest import in_write_dir


class _RecordingDriver:
    """Records move() calls; every other driver call answers empty."""

    def __init__(self, log):
        self._log = log

    def move(self, src, dst):
        self._log.append((src, dst))

    def __getattr__(self, _name):
        return lambda *a, **k: []


@pytest.fixture
def mirror_vault(tmp_path, monkeypatch):
    """A vault declaring `write_dir: silica`, active for the manifest lookups."""
    import silica.kernel.vault_manifest as vm
    from silica.config import CONFIG

    (tmp_path / "vault.yaml").write_text("write_dir: silica\n", encoding="utf-8")
    (tmp_path / "Inbox").mkdir()
    (tmp_path / "Inbox" / "note.md").write_text("body\n", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    vm.reset_manifest_cache()
    yield tmp_path
    # The driver caches the vault it was built against; leaving it bound to a
    # torn-down tmp dir fails the next test that moves a file.
    from silica.driver import reset_driver
    vm.reset_manifest_cache()
    reset_driver()


def test_in_write_dir_composes_and_is_idempotent(mirror_vault):
    assert in_write_dir("done") == "silica/done"
    assert in_write_dir("silica/done") == "silica/done"


def test_archive_skips_a_source_outside_the_boundary(mirror_vault, monkeypatch):
    """The defect: `/nucleate` under safe mode moved the user's real inbox note
    to `<vault>/done/`, outside the mirror it was supposed to stay out of."""
    from silica.tools import wrapped

    moved: list[tuple[str, str]] = []
    # The module global, not an attribute ON the proxy: reading through the
    # proxy builds the real driver against this tmp vault and leaves it cached
    # for every later test.
    monkeypatch.setattr(wrapped, "DRIVER", _RecordingDriver(moved))

    res = wrapped.silica_cleanup("Inbox/note.md")

    assert moved == [], "a source outside the boundary must not be moved"
    assert res["skipped"] == "Inbox/note.md"


def test_archive_of_a_source_inside_the_boundary_lands_inside_it(mirror_vault, monkeypatch):
    from silica.tools import wrapped

    moved: list[tuple[str, str]] = []
    # The module global, not an attribute ON the proxy: reading through the
    # proxy builds the real driver against this tmp vault and leaves it cached
    # for every later test.
    monkeypatch.setattr(wrapped, "DRIVER", _RecordingDriver(moved))

    wrapped.silica_cleanup("silica/Inbox/note.md")

    assert moved == [("silica/Inbox/note.md", "silica/Done/note.md")]


def test_no_boundary_leaves_the_archive_at_the_vault_root(tmp_path, monkeypatch):
    """A vault that declares nothing behaves exactly as before."""
    import silica.kernel.vault_manifest as vm
    from silica.config import CONFIG
    from silica.tools import wrapped

    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    vm.reset_manifest_cache()
    moved: list[tuple[str, str]] = []
    # The module global, not an attribute ON the proxy: reading through the
    # proxy builds the real driver against this tmp vault and leaves it cached
    # for every later test.
    monkeypatch.setattr(wrapped, "DRIVER", _RecordingDriver(moved))

    wrapped.silica_cleanup("Inbox/note.md")

    assert moved == [("Inbox/note.md", "Done/note.md")]
    vm.reset_manifest_cache()


def test_archive_mirrors_the_inbox_tree(tmp_path, monkeypatch):
    """The archive keeps the folder structure the source had in the inbox.

    Flattening onto a basename made the move unrecoverable by hand: `/revert`
    records no inverse for it, so the structure is the only undo the user has.
    """
    import silica.kernel.vault_manifest as vm
    from silica.config import CONFIG
    from silica.tools import wrapped

    (tmp_path / "Inbox" / "Kant" / "Critica").mkdir(parents=True)
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    vm.reset_manifest_cache()
    moved: list[tuple[str, str]] = []
    monkeypatch.setattr(wrapped, "DRIVER", _RecordingDriver(moved))

    wrapped.silica_cleanup("Inbox/Kant/Critica/ch3.md")

    assert moved == [("Inbox/Kant/Critica/ch3.md", "Done/Kant/Critica/ch3.md")]
    vm.reset_manifest_cache()


def test_archive_keeps_a_legacy_lowercase_done_where_it_is(tmp_path, monkeypatch):
    """A vault that already archived under `done/` is never renamed to `Done/`.

    The rename would fork the archive in two on a case-sensitive filesystem,
    leaving half the user's sources under a folder nothing reads any more.
    """
    import silica.kernel.vault_manifest as vm
    from silica.config import CONFIG
    from silica.tools import wrapped

    (tmp_path / "Inbox").mkdir()
    (tmp_path / "done").mkdir()
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    vm.reset_manifest_cache()
    moved: list[tuple[str, str]] = []
    monkeypatch.setattr(wrapped, "DRIVER", _RecordingDriver(moved))

    wrapped.silica_cleanup("Inbox/note.md")

    assert moved == [("Inbox/note.md", "done/note.md")]
    vm.reset_manifest_cache()


def test_hub_resolves_beside_its_folder_not_inside_it(monkeypatch):
    """The defect: `hub` defaults to the basename of target_dir, and the hub note
    conventionally sits BESIDE that folder — so `target_dir/<hub>.md` named a
    path the vault never had and MOC membership was skipped for the whole run."""
    from unittest.mock import MagicMock
    from silica.driver.base import NoteRef
    from silica.kernel.write.moc import moc_target

    driver = MagicMock()
    driver.search_names.return_value = [
        NoteRef(name="Apprendimento supervisionato",
                path="Informatica/ML/Apprendimento supervisionato.md")
    ]
    monkeypatch.setattr("silica.driver.DRIVER", driver)

    assert moc_target(
        "Apprendimento supervisionato", "Informatica/ML/Apprendimento supervisionato"
    ) == "Informatica/ML/Apprendimento supervisionato.md"


def test_hub_falls_back_to_target_dir_when_not_written_yet(monkeypatch):
    from unittest.mock import MagicMock
    from silica.kernel.write.moc import moc_target

    driver = MagicMock()
    driver.search_names.return_value = []
    monkeypatch.setattr("silica.driver.DRIVER", driver)

    assert moc_target("Hub", "testing") == "testing/Hub.md"


def test_converter_spaced_latex_grounds_against_a_tight_rewrite():
    """The defect: a MinerU-converted lecture spaces every LaTeX token, the model
    re-emits the same formula tight, and the verbatim gate called it fabricated."""
    from silica.kernel.write.provenance import ungrounded_spans

    source = (
        "Matrici di scatter, date K classi\n\n$$\n"
        "S _ {B} = \\sum_ {j = 1} ^ {K} \\left(\\boldsymbol {\\mu} _ {j} - "
        "\\boldsymbol {m}\\right) \\left(\\boldsymbol {\\mu} _ {j} - "
        "\\boldsymbol {m}\\right) ^ {T}\n$$\n"
    )
    body = (
        "La matrice di between scatter:\n\n$$\n"
        "S_B = \\sum_{j=1}^{K} \\left(\\boldsymbol{\\mu}_j - \\boldsymbol{m}\\right)"
        " \\left(\\boldsymbol{\\mu}_j - \\boldsymbol{m}\\right)^{T}\n$$\n"
    )

    assert ungrounded_spans(body, source) == []


def test_a_formula_absent_from_the_source_is_still_flagged():
    """The gate must not go blind: normalization is notation-only."""
    from silica.kernel.write.provenance import ungrounded_spans

    source = "Matrici di scatter\n\n$$\nS _ {B} = \\sum_ {j} \\boldsymbol {\\mu} _ {j}\n$$\n"
    body = "Il criterio ottimo:\n\n$$\n\\lambda = \\operatorname{tr}(S_W^{-1} S_B) + \\gamma\n$$\n"

    assert ungrounded_spans(body, source) != []


def test_math_and_code_normalize_differently():
    """Whitespace is noise in LaTeX and meaning in code — one normalizer each."""
    from silica.kernel.write.provenance import _norm_math, _norm_ws

    assert _norm_math("S _ {B} = \\sum_ {j}") == "S_B=\\sum_j"
    assert _norm_ws("def f(x):\n    return x") == "def f(x): return x"


def test_safe_mode_patch_is_rebased_onto_the_mirror(mirror_vault, monkeypatch):
    """The defect: safe mode rejected every patch of an existing note, so an
    ingest could only ever create — 11 of 20 deferred ops in one real run."""
    from silica.kernel.write import validate as V
    from silica.kernel.write.ops import Op, OpType
    from silica.kernel.write.validate import validate_operations

    # Existence answered off the tmp vault: the routing decision is what's under
    # test, not the driver's index.
    class _FsDriver:
        def read_note(self, p):
            f = mirror_vault / str(p)
            if not f.is_file():
                raise RuntimeError(f"File not found: {p}")
            return type("N", (), {"content": f.read_text(encoding="utf-8"), "path": str(p)})()

        def __getattr__(self, _name):
            return lambda *a, **k: []

    monkeypatch.setattr(V, "DRIVER", _FsDriver())

    (mirror_vault / "Matematica").mkdir()
    (mirror_vault / "Matematica" / "PCA.md").write_text("# PCA\n", encoding="utf-8")
    op = Op(op=OpType.patch, path="Matematica/PCA.md", heading="PCA",
            snippet="x" * 400, source_basename="Lezione 9.md")
    payloads = [{"batches": [{"inbox_file": "Inbox/Lezione 9.md", "concepts": [
        {"name": "PCA", "inbox_excerpt": "x" * 400,
         "vault_collision": {"path": "Matematica/PCA.md", "excerpt": "# PCA\n"}}]}]}]

    valid, rejected = validate_operations([op], payloads=payloads, target_dir="silica/Concetti")

    boundary = [r for r in rejected if "write boundary" in r.reason]
    assert not boundary, [r.reason for r in boundary]
    patched = [o for o in valid if o.op == OpType.patch]
    assert [o.path for o in patched] == ["silica/Matematica/PCA.md"]


def test_patch_seeds_the_mirror_copy_from_the_original(mirror_vault):
    """The mirror is pasted over the vault, so a patched copy must start as the
    note itself — otherwise the paste drops everything the note already said."""
    from silica.kernel.vault_manifest import seed_mirror_copy

    original = mirror_vault / "Matematica" / "PCA.md"
    original.parent.mkdir(parents=True)
    original.write_text("# PCA\n\nTesto originale.\n", encoding="utf-8")

    seed_mirror_copy("silica/Matematica/PCA.md")

    assert (mirror_vault / "silica" / "Matematica" / "PCA.md").read_text() == original.read_text()


def test_seeding_never_overwrites_an_existing_mirror_copy(mirror_vault):
    from silica.kernel.vault_manifest import seed_mirror_copy

    (mirror_vault / "Matematica").mkdir()
    (mirror_vault / "Matematica" / "PCA.md").write_text("originale\n", encoding="utf-8")
    copy = mirror_vault / "silica" / "Matematica" / "PCA.md"
    copy.parent.mkdir(parents=True)
    copy.write_text("gia' patchata\n", encoding="utf-8")

    seed_mirror_copy("silica/Matematica/PCA.md")

    assert copy.read_text() == "gia' patchata\n"


def test_collision_prefers_the_original_over_its_mirror_copy(mirror_vault):
    """The defect: original and mirror copy group under one name, so WHICH path
    answered was scan order — the model got shown its own unmerged draft as the
    vault's state (real run: concept 'PCA' collided with the note that same run
    had written minutes earlier)."""
    from silica.driver.base import NoteRef
    from silica.tools.pipeline import _prefers

    original = NoteRef(name="PCA", path="Matematica/PCA.md")
    copy = NoteRef(name="PCA", path="silica/Matematica/PCA.md")

    assert _prefers(original, copy) is True
    assert _prefers(copy, original) is False
    assert _prefers(original, NoteRef(name="PCA", path="Fisica/PCA.md")) is False


def test_collision_preference_is_inert_without_safe_mode(tmp_path, monkeypatch):
    import silica.kernel.vault_manifest as vm
    from silica.config import CONFIG
    from silica.driver.base import NoteRef
    from silica.tools.pipeline import _prefers

    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))
    vm.reset_manifest_cache()

    assert _prefers(NoteRef(name="PCA", path="Matematica/PCA.md"),
                    NoteRef(name="PCA", path="silica/Matematica/PCA.md")) is False
    vm.reset_manifest_cache()


def test_converter_figure_caption_is_stripped_with_its_image():
    """The defect: a real run made a note titled 'Y-axis Range' — body, tables and
    all — out of the vision model's description of a figure."""
    from silica.kernel.text.media import strip_images

    text = (
        "Vediamo se And e Or sono separabili linearmente:\n\n"
        "![](images/3895e05.jpg)\n\n"
        "<details>\n<summary>contour</summary>\n\n"
        "| Region | X-axis Range | Y-axis Range |\n| --- | --- | --- |\n"
        "| Top-Left (00-10) | 0~0.2 | 10~11 |\n</details>\n\n"
        "Il percettrone non risolve lo XOR.\n"
    )
    out = strip_images(text)

    assert "Y-axis Range" not in out
    assert "contour" not in out
    assert "Il percettrone non risolve lo XOR." in out


def test_a_details_block_of_the_authors_survives():
    """Positional rule: a collapsible that follows no image is hand-written."""
    from silica.kernel.text.media import strip_images

    text = "## Dimostrazione\n\n<details>\n<summary>passaggi</summary>\n\nS_B = ...\n</details>\n"

    assert strip_images(text) == text


def test_snippet_echoing_the_title_does_not_duplicate_the_h1():
    """The defect: the template writes `# Title` and the distiller's snippet
    opened with the same heading — 13 of 24 notes in one real run carried it
    twice, the other 11 once."""
    from silica.kernel.write.templates import BUILTIN_TEMPLATE, prepare_fields, render_note

    out = render_note(BUILTIN_TEMPLATE, prepare_fields(
        title="Kernel Trick", body="# Kernel Trick\n\nIl kernel trick e' ...", hub="ML"))

    assert out.count("# Kernel Trick") == 1


def test_the_provenance_stamp_does_not_hide_the_echo():
    from silica.kernel.write.templates import BUILTIN_TEMPLATE, prepare_fields, render_note

    out = render_note(BUILTIN_TEMPLATE, prepare_fields(
        title="XOR", body="<!-- silica: valid_from=2026-08-04 -->\n\n# XOR\n\nLo XOR ...", hub="ML"))

    assert out.count("# XOR") == 1
    assert "valid_from=2026-08-04" in out


def test_a_different_h1_is_content_and_survives():
    from silica.kernel.write.templates import BUILTIN_TEMPLATE, prepare_fields, render_note

    out = render_note(BUILTIN_TEMPLATE, prepare_fields(
        title="XOR", body="# Separabilita' lineare\n\nLo XOR ...", hub="ML"))

    assert "# Separabilita' lineare" in out
    assert "# XOR" in out


def test_a_filename_is_not_autolinked_as_prose():
    """The defect: '[[Lezione 9]].md' — the extension left outside the brackets."""
    from silica.kernel.link.autolink import autolink

    body, added = autolink("Fonti: Lezione 9.md (sezione kernel).", ["Lezione 9"])

    assert body == "Fonti: Lezione 9.md (sezione kernel)."
    assert added == []


def test_a_sentence_final_title_still_links():
    from silica.kernel.link.autolink import autolink

    body, added = autolink("Lo spiega Lezione 9. Poi vediamo altro.", ["Lezione 9"])

    assert body.startswith("Lo spiega [[Lezione 9]].")
    assert added == ["Lezione 9"]


def test_the_real_inbox_stays_forbidden_under_safe_mode(mirror_vault):
    """The defect safe mode introduced: `active_inbox_dir` composes the write
    boundary, so it named `silica/Inbox` and the user's own `Inbox/` stopped
    reading as an inbox — patch ops aimed at inbox notes stopped being rejected
    (two landed as mirror copies in a real run)."""
    from silica.kernel.recall.paths import is_inbox_path

    assert is_inbox_path("Inbox/giovanni_castelli.md") is True
    assert is_inbox_path("silica/Inbox/staged.md") is True
    assert is_inbox_path("Matematica/Statistica/PCA.md") is False


def test_the_hub_moc_lands_on_the_mirror_copy_not_the_real_note(mirror_vault, monkeypatch):
    """The defect resolving the hub by name introduced: the resolved path IS the
    real vault note, so HUB_UPDATE wrote its MOC block straight into it — a real
    run modified two notes safe mode exists to protect."""
    from unittest.mock import MagicMock
    from silica.driver.base import NoteRef
    from silica.kernel.write.moc import moc_target

    body = "# Apprendimento supervisionato\n\nIndice esistente.\n"

    def read_note(p):
        if str(p) == "Informatica/Apprendimento supervisionato.md":
            return type("N", (), {"content": body})()
        raise RuntimeError(f"File not found: {p}")

    driver = MagicMock()
    driver.read_note.side_effect = read_note
    driver.search_names.return_value = [
        NoteRef(name="Apprendimento supervisionato",
                path="Informatica/Apprendimento supervisionato.md")
    ]
    monkeypatch.setattr("silica.driver.DRIVER", driver)

    landing = moc_target("Apprendimento supervisionato", "Informatica/Apprendimento supervisionato")

    assert landing == "silica/Informatica/Apprendimento supervisionato.md"
    # Seeded through the DRIVER, so the block lands on top of what it already said.
    driver.create.assert_called_once_with(landing, body)


def test_the_real_inbox_stays_forbidden_under_safe_mode(mirror_vault):
    """The defect safe mode introduced: `active_inbox_dir` composes the write
    boundary, so it named `silica/Inbox` and the user's own `Inbox/` stopped
    reading as an inbox — patch ops aimed at inbox notes stopped being rejected
    (two landed as mirror copies in a real run)."""
    from silica.kernel.recall.paths import is_inbox_path

    assert is_inbox_path("Inbox/giovanni_castelli.md") is True
    assert is_inbox_path("silica/Inbox/staged.md") is True
    assert is_inbox_path("Matematica/Statistica/PCA.md") is False


def test_punctuation_the_filename_dropped_still_reads_as_an_echo():
    """The title arrives slugified off the filename, so an exact match called
    'Classificazione: approcci …' and 'Classificazione approcci …' two headings."""
    from silica.kernel.write.templates import BUILTIN_TEMPLATE, prepare_fields, render_note

    out = render_note(BUILTIN_TEMPLATE, prepare_fields(
        title="Classificazione approcci discriminativo e generativo",
        body="# Classificazione: approcci discriminativo e generativo\n\nDue approcci.",
        hub="ML"))

    assert len([ln for ln in out.splitlines() if ln.startswith("# ")]) == 1


def test_a_source_file_is_never_a_moc_target(mirror_vault, monkeypatch):
    """The defect: a lecture shares its title with the notes distilled from it,
    so parent resolution answered with the source being ingested — and safe mode
    then seeded a mirror copy of it beside the notes."""
    from unittest.mock import MagicMock
    from silica.driver.base import NoteRef
    from silica.kernel.write.moc import moc_target

    driver = MagicMock()
    driver.search_names.return_value = [
        NoteRef(name="Lezione 8", path="Inbox/machine_learning/Lezione 8.md")
    ]
    monkeypatch.setattr("silica.driver.DRIVER", driver)

    assert moc_target("Lezione 8", "Informatica/ML") == ""
    assert not (mirror_vault / "silica" / "Inbox").exists()


def test_a_fence_inside_a_callout_is_still_code():
    """The defect: `> ```python` was not read as a fence, so autolink wikilinked
    the language tag AND the missed opener flipped fence parity, which is how a
    URL in the same note came back as '…Article.[[HTML]]'."""
    from silica.kernel.link.autolink import autolink

    body = (
        "> [!IMPORTANT] [Rosenblatt](https://example.org/RosenblattArticle.html)\n"
        "> \n"
        "> ```Python\n"
        "> def step(x): return 1 if x > 0 else 0\n"
        "> ```\n"
    )

    out, added = autolink(body, ["HTML", "Python"])

    assert out == body
    assert added == []


def test_a_plain_fence_is_still_masked():
    from silica.kernel.link.autolink import autolink

    body = "```Python\nprint(1)\n```\n"

    assert autolink(body, ["Python"]) == (body, [])


def test_a_staging_note_answers_with_no_moc_target_at_all(mirror_vault, monkeypatch):
    """"" rather than a target_dir fallback: the fallback named a path nothing
    would ever write and logged a miss that read like a failure, once per run."""
    from unittest.mock import MagicMock
    from silica.driver.base import NoteRef
    from silica.kernel.write.moc import moc_target

    driver = MagicMock()
    driver.search_names.return_value = [
        NoteRef(name="Lezione 8", path="Inbox/machine_learning/Lezione 8.md")
    ]
    monkeypatch.setattr("silica.driver.DRIVER", driver)

    assert moc_target("Lezione 8", "Informatica/ML") == ""
    assert not (mirror_vault / "silica" / "Inbox").exists()


def test_graph_context_is_keyed_the_way_its_consumers_look_it_up():
    """The defect: node ids carry `.md`, every consumer strips it before the
    lookup, so the whole graph-context feature was inert — 0 hits in 717 on a
    real vault. Cluster narrowing, hub detection, cross-cluster warnings and the
    distiller's graph_context were all reading an empty answer."""
    from silica.kernel.recall.graph_export import ctx_from_report

    class _Cluster:
        cluster_id = 3
        hub = "Informatica/ML/Machine learning.md"
        members = ["Informatica/ML/Machine learning.md", "Informatica/ML/SVM.md"]

    class _Report:
        clusters = [_Cluster()]
        pagerank_map = {"Psicologia/Attenzione.md": 0.0}

    ctx = ctx_from_report(_Report())

    assert set(ctx) == {"Informatica/ML/Machine learning", "Informatica/ML/SVM",
                        "Psicologia/Attenzione"}
    # `hub` gets the same shape, or the distiller's vocabulary lists "X.md".
    assert ctx["Informatica/ML/SVM"]["hub"] == "Informatica/ML/Machine learning"
    assert ctx["Informatica/ML/Machine learning"]["is_hub"] is True
    # And the consumers' own key building now lands.
    assert "Informatica/ML/SVM.md".removesuffix(".md") in ctx


def test_a_cluster_cache_of_the_old_key_shape_is_discarded(tmp_path, monkeypatch):
    """Without a version the stale cache is indistinguishable from a fresh one,
    and every consumer would keep missing after the fix."""
    import orjson
    import silica.kernel.recall.graph_export as ge

    p = tmp_path / "cluster_ctx.json"
    monkeypatch.setattr(ge, "cluster_ctx_path", lambda: p)

    p.write_bytes(orjson.dumps({"sig": [1, 1], "ctx": {"A/B.md": {"cluster_id": 0}}}))
    assert ge.load_cluster_ctx() is None

    ge.save_cluster_ctx([1, 1], {"A/B": {"cluster_id": 0}})
    assert ge.load_cluster_ctx()["ctx"] == {"A/B": {"cluster_id": 0}}


def test_staging_hubs_are_not_offered_as_vault_vocabulary():
    """The substrate prints these under "reuse these instead of coining
    synonyms". On a vault of converted PDFs, 27 of 102 hub names were chapter
    slugs like `01-an-introduction-to-support-vector-machines`."""
    from silica.router.states.distill import _vocabulary_hubs

    ctx = {
        "Informatica/ML/Machine learning": {
            "hub": "Informatica/ML/Machine learning", "is_hub": True},
        "Inbox/SVM book/01-an-introduction-to-svm": {
            "hub": "Inbox/SVM book/01-an-introduction-to-svm", "is_hub": True},
        "Informatica/ML/SVM": {"hub": "Informatica/ML/Machine learning", "is_hub": False},
    }

    assert _vocabulary_hubs(ctx) == ["Machine learning"]


def test_the_hub_vocabulary_is_capped_like_its_siblings():
    """Unbounded, it could outweigh every other section of the substrate."""
    from silica.kernel.recall.run_substrate import build_substrate

    out = build_substrate({"schema_version": 1, "batches": []}, manifest_titles=[],
                          hub_names=[f"Concetto molto lungo numero {i}" for i in range(200)])

    line = next((ln for ln in (out or "").splitlines() if ln.startswith("Hub notes: ")), "")
    assert len(line) <= len("Hub notes: ") + 600


def test_neighbours_above_excludes_the_query_note_itself(tmp_path, monkeypatch):
    """The self-exclusion passed the raw `.md` path into a store keyed by
    cooccur_key (no `.md`), so it never matched and every note was its own
    closest neighbour (cosine 1.0). Downstream autolink's self_title guard
    happened to hide it — the recall keyspace mismatch class, again."""
    import silica.kernel.recall.embed as embed_mod
    from silica.kernel.recall.embed import EmbedStore
    from silica.kernel.recall.relatedness import neighbours_above

    monkeypatch.setattr(embed_mod, "_index_path", lambda: tmp_path / "emb.json")
    embed_mod.clear()
    es = embed_mod.get_store()
    es.upsert("Concepts/Near", "Near", [1.0, 0.0])
    es.upsert("Concepts/Nota", "Nota", [1.0, 0.0])

    near = neighbours_above("Concepts/Nota.md", 0.30)

    assert near == ["Near"]


def test_the_relevance_gate_abstains_rather_than_guessing(monkeypatch):
    """None (gate unavailable) and [] (nothing close enough) are different
    answers: conflating them either restores the noise or blocks every link."""
    import silica.router.states.linking as L
    from silica.config import CONFIG

    monkeypatch.setattr(CONFIG, "autolink_min_sim", 0.30)
    index = ["Machine learning", "Attenzione", "XOR"]

    monkeypatch.setattr(L, "_relevance_candidates", L._relevance_candidates)
    import silica.kernel.recall.relatedness as R

    monkeypatch.setattr(R, "neighbours_above", lambda _p, _f: None)
    assert L._relevance_candidates(index)("Nota.md") == index

    monkeypatch.setattr(R, "neighbours_above", lambda _p, _f: [])
    assert L._relevance_candidates(index)("Nota.md") == []

    monkeypatch.setattr(R, "neighbours_above", lambda _p, _f: ["Machine learning", "Sconosciuta"])
    assert L._relevance_candidates(index)("Nota.md") == ["Machine learning"]


def test_a_zero_threshold_leaves_the_full_index(monkeypatch):
    """The opt-out: SILICA_AUTOLINK_MIN_SIM=0 is the pre-gate behaviour."""
    import silica.router.states.linking as L
    from silica.config import CONFIG

    monkeypatch.setattr(CONFIG, "autolink_min_sim", 0.0)
    index = ["Machine learning", "Attenzione"]

    assert L._relevance_candidates(index)("Nota.md") == index
