"""convert() — non-.md → .md ingress frontier (PDF, provider-selectable).

Providers are mocked: docling/opendataloader are injected as fake modules and the
mineru subprocess is patched, so no ML models / real PDFs / installs are needed.
"""
import io
import os
import re
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

from silica.config import CONFIG, SilicaConfig
from silica.sources import convert as conv


def _inbox_note(note_rel: str) -> Path:
    return Path(CONFIG.vault_path) / note_rel


# --- dispatch ---------------------------------------------------------------

@pytest.mark.parametrize("target", ["notes.xyz", "noext", "data.zip"])
def test_unknown_extension_raises(target):
    with pytest.raises(ValueError, match="no converter"):
        conv.convert(target)


@pytest.mark.parametrize("ext", conv.DOC_EXTS)
def test_every_doc_ext_reaches_the_converter(ext, tmp_vault):
    """Dispatch accepts the extension — it fails later, on the missing file."""
    with pytest.raises(ValueError, match="file not found"):
        conv.convert(f"ghost{ext}")


def test_surrounding_whitespace_does_not_hide_the_extension(tmp_vault):
    """A quoted path with a stray trailing space is still a PDF, not an unknown type."""
    with pytest.raises(ValueError, match="file not found"):
        conv.convert("  ghost.pdf ")


# --- pymupdf provider (default; real library, no fakes) ---------------------

def _pdf_bytes(pages: list[str], toc: list | None = None) -> bytes:
    """A real one-column PDF, one page per string, optionally with an outline."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text, fontsize=11)
    if toc:
        doc.set_toc(toc)
    return doc.tobytes()


def test_pymupdf_is_the_default_provider(monkeypatch):
    # config.py loads ~/.silica/.env into os.environ at import, so the field's
    # default_factory reads the developer's own pin unless it is cleared here.
    monkeypatch.delenv("SILICA_PDF_PROVIDER", raising=False)
    assert SilicaConfig().pdf_provider == "pymupdf"


def test_pymupdf_converts_a_real_pdf(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "pymupdf")
    (Path(CONFIG.vault_path) / "paper.pdf").write_bytes(_pdf_bytes(["Hello vault"]))

    body = _inbox_note(conv.convert("paper.pdf")[0]).read_text(encoding="utf-8")
    assert "Hello vault" in body


def test_pymupdf_uses_the_embedded_outline_for_headings(tmp_vault, monkeypatch):
    """The PDF's own outline beats font-size guessing (23 headings vs 12 on a
    probe paper). Without it TocHeaders would collapse everything to one."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "pymupdf")
    pdf = _pdf_bytes(
        ["Chapter One body", "Chapter Two body"],
        toc=[[1, "Chapter One", 1], [1, "Chapter Two", 2]],
    )
    (Path(CONFIG.vault_path) / "book.pdf").write_bytes(pdf)

    body = _inbox_note(conv.convert("book.pdf")[0]).read_text(encoding="utf-8")
    assert "# Chapter One" in body and "# Chapter Two" in body


def test_pymupdf_without_outline_still_produces_text(tmp_vault, monkeypatch):
    """No outline → no TocHeaders; the font heuristic must stay in charge."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "pymupdf")
    (Path(CONFIG.vault_path) / "plain.pdf").write_bytes(_pdf_bytes(["Body with no outline"]))

    body = _inbox_note(conv.convert("plain.pdf")[0]).read_text(encoding="utf-8")
    assert "Body with no outline" in body


def test_docx_bypasses_the_pdf_provider_seam(tmp_vault, monkeypatch):
    """mineru/docling/opendataloader only take PDFs, so a non-PDF must reach
    pymupdf whatever SILICA_PDF_PROVIDER says — here mineru, whose subprocess
    would explode if it were called."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "mineru")
    monkeypatch.setattr(conv.subprocess, "run", _never_called)
    calls: list[Path] = []
    monkeypatch.setattr(
        conv, "_via_pymupdf", lambda src, wd: (calls.append(src) or ("# Doc\n\nbody", wd))
    )
    tmp_vault.note("memo.docx", "x")

    conv.convert("memo.docx")
    assert [p.name for p in calls] == ["memo.docx"]


def _never_called(*a, **k):
    raise AssertionError("PDF provider called for a non-PDF input")


def test_converted_note_carries_source_file_provenance(tmp_vault, monkeypatch):
    """The provenance ledger only ever records the inbox note's basename, so the
    converted note's own frontmatter is the one pointer back to the real file."""
    monkeypatch.setattr(conv, "_via_pymupdf", lambda src, wd: ("# Doc\n\nbody", wd))
    tmp_vault.note("memo.docx", "x")

    note = _inbox_note(conv.convert("memo.docx")[0]).read_text(encoding="utf-8")
    head = note.split("\n---\n")[0]
    assert note.startswith('---\nsource_file: "')
    assert str(Path(CONFIG.vault_path) / "memo.docx") in head
    assert "# Doc" in note  # body intact below the block


def test_every_segment_carries_source_file_provenance(tmp_vault, monkeypatch):
    chapter = " ".join(f"w{i}" for i in range(6000))  # ~30k chars, no degenerate runs
    big = f"# One\n\n{chapter}\n\n# Two\n\n{chapter}\n"
    monkeypatch.setattr(conv, "_via_pymupdf", lambda src, wd: (big, wd))
    tmp_vault.note("book.docx", "x")

    paths = conv.convert("book.docx")
    assert len(paths) > 1
    for p in paths:
        assert _inbox_note(p).read_text(encoding="utf-8").startswith('---\nsource_file: "')


# --- _source_date: the document's own creation date → frontmatter `date:` ----

def test_pdf_creation_date_lands_in_frontmatter(tmp_vault, monkeypatch):
    """Rung 2 of the event clock: a dated PDF dates its claims by the document,
    not the run (`source_event_date` reads the converted note's `date:`)."""
    pymupdf = pytest.importorskip("pymupdf")
    monkeypatch.setattr(CONFIG, "pdf_provider", "pymupdf")
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Dated body", fontsize=11)
    doc.set_metadata({"creationDate": "D:20240402093000+02'00'"})
    (Path(CONFIG.vault_path) / "dated.pdf").write_bytes(doc.tobytes())

    note = _inbox_note(conv.convert("dated.pdf")[0]).read_text(encoding="utf-8")
    assert note.startswith("---\ndate: 2024-04-02\n")


def test_undated_pdf_gets_no_date_line(tmp_vault, monkeypatch):
    """No metadata → no `date:` — a missing event clock must stay missing (the
    FSM would stamp it on every claim as valid_from)."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "pymupdf")
    (Path(CONFIG.vault_path) / "plain.pdf").write_bytes(_pdf_bytes(["Undated body"]))

    note = _inbox_note(conv.convert("plain.pdf")[0]).read_text(encoding="utf-8")
    assert note.startswith('---\nsource_file: "')
    assert "\ndate:" not in note.split("\n---\n")[0]


def test_ooxml_dcterms_created_is_read_from_the_zip(tmp_vault):
    import zipfile

    p = Path(CONFIG.vault_path) / "deck.pptx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(
            "docProps/core.xml",
            '<coreProperties xmlns:dcterms="d"><dcterms:created>'
            "2023-11-20T09:00:00Z</dcterms:created></coreProperties>",
        )
    assert conv._source_date(p) == "2023-11-20"


def test_source_date_rejects_garbage_and_absence(tmp_vault):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page()
    doc.set_metadata({"creationDate": "D:00001332"})  # month 13, day 32
    garbage = Path(CONFIG.vault_path) / "garbage.pdf"
    garbage.write_bytes(doc.tobytes())
    assert conv._source_date(garbage) is None

    missing = Path(CONFIG.vault_path) / "ghost.pdf"
    assert conv._source_date(missing) is None


def test_empty_extraction_raises_pointing_at_ocr(tmp_vault, monkeypatch):
    """A scan with no text layer yields nothing; writing an empty inbox note and
    calling it success is the failure mode this guard exists to prevent."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "pymupdf")
    # setitem, not setattr: the PDF path resolves the provider through the
    # registry dict, which holds its own reference to the function.
    monkeypatch.setitem(conv.PDF_PROVIDERS, "pymupdf", lambda src, wd: ("   \n\n", wd))
    tmp_vault.note("scan.pdf", "x")

    with pytest.raises(ValueError, match="no text extracted.*mineru"):
        conv.convert("scan.pdf")
    assert not (Path(CONFIG.vault_path) / CONFIG.inbox_dir / "scan.md").exists()


# --- shared pipeline (exercised via the docling fake) -----------------------
#
# TODO(real-api): the fakes here hand-mirror the docling / opendataloader APIs
# and the mineru CLI. They prove the SHARED pipeline, not the provider wiring —
# if a library renames those, the fakes drift with the bug and stay green. Add a
# real-install smoke test (skipif on import, one tiny bundled PDF) to catch drift.

def test_pdf_rewrites_any_image_link(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "docling")
    tmp_vault.note("paper.pdf", "x")
    _fake_docling(monkeypatch, md="see [ref](https://x.test/a) and ![](a/b/fig.png)")

    body = _inbox_note(conv.convert("paper.pdf")[0]).read_text(encoding="utf-8")
    assert "[ref](https://x.test/a)" in body          # ordinary link survives
    assert "![[paper-fig.png]]" in body                # image link → Obsidian embed


def test_missing_file_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "docling")
    with pytest.raises(ValueError, match="file not found"):
        conv.convert("ghost.pdf")


def test_relative_path_falls_back_to_cwd(tmp_vault, tmp_path, monkeypatch):
    """A PDF in the user's directory, not in the vault, still converts."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "docling")
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")
    monkeypatch.chdir(tmp_path)
    _fake_docling(monkeypatch, md="body")

    assert _inbox_note(conv.convert("paper.pdf")[0]).exists()


# --- docling provider (keeps figures) ---------------------------------------

def _fake_docling(monkeypatch, md="# Title\n\n![](images/fig.png)\n\nbody"):
    """Inject a fake docling whose save_as_markdown writes one image + references it.

    Returns a dict capturing the PdfFormatOption kwargs (``pipeline_options``)
    so tests can assert the precision pins.
    """
    captured: dict = {}

    class _Doc:
        def save_as_markdown(self, path, *, image_mode, artifacts_dir):
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "fig.png").write_bytes(b"\x89PNG fake")
            Path(path).write_text(md, encoding="utf-8")

    class DocumentConverter:
        def __init__(self, **kw):
            pass

        def convert(self, path):
            return types.SimpleNamespace(document=_Doc())

    class PdfPipelineOptions:
        def __init__(self):
            self.table_structure_options = types.SimpleNamespace(
                mode=None, do_cell_matching=None
            )
            self.ocr_options = types.SimpleNamespace(lang=None)

    dc = types.ModuleType("docling.document_converter")
    dc.DocumentConverter = DocumentConverter
    dc.PdfFormatOption = lambda **kw: captured.update(kw)
    base = types.ModuleType("docling.datamodel.base_models")
    base.InputFormat = types.SimpleNamespace(PDF="pdf")
    popts = types.ModuleType("docling.datamodel.pipeline_options")
    popts.PdfPipelineOptions = PdfPipelineOptions
    popts.TableFormerMode = types.SimpleNamespace(ACCURATE="accurate")
    core = types.ModuleType("docling_core.types.doc")
    core.ImageRefMode = types.SimpleNamespace(REFERENCED="referenced")
    fakes = {
        "docling": types.ModuleType("docling"),
        "docling.datamodel": types.ModuleType("docling.datamodel"),
        "docling.datamodel.base_models": base,
        "docling.datamodel.pipeline_options": popts,
        "docling.document_converter": dc,
        "docling_core": types.ModuleType("docling_core"),
        "docling_core.types": types.ModuleType("docling_core.types"),
        "docling_core.types.doc": core,
    }
    for name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return captured


def test_pdf_docling_provider_embeds_extracted_image(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "docling")
    tmp_vault.note("paper.pdf", "x")
    _fake_docling(monkeypatch)

    note_rels = conv.convert("paper.pdf", dest_dir="Concepts/X")
    assert note_rels == [f"{CONFIG.inbox_dir}/paper.md"]  # small PDF → one flat note
    body = _inbox_note(note_rels[0]).read_text(encoding="utf-8")
    assert "![[paper-fig.png]]" in body          # namespaced by source stem
    assert (Path(CONFIG.vault_path) / "Concepts/X/Images/paper-fig.png").is_file()


def test_unreferenced_extracted_image_is_not_copied(tmp_vault, monkeypatch):
    """mineru dumps every crop it detects (477 files for a 200-page book, 19
    referenced) — only images the markdown references may reach the vault."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "docling")
    tmp_vault.note("paper.pdf", "x")
    _fake_docling(monkeypatch, md="# Title\n\nno figures referenced here")

    conv.convert("paper.pdf", dest_dir="Concepts/X")

    assert not (Path(CONFIG.vault_path) / "Concepts/X/Images/fig.png").exists()


def test_docling_precision_pins(tmp_vault, monkeypatch):
    """Max-precision non-generative config is passed explicitly (spec
    2026-07-22): ACCURATE TableFormer, cell matching, 144 dpi figures, OCR
    languages from config."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "docling")
    tmp_vault.note("paper.pdf", "x")
    captured = _fake_docling(monkeypatch)

    conv.convert("paper.pdf")

    opts = captured["pipeline_options"]
    assert opts.do_table_structure is True
    assert opts.table_structure_options.mode == "accurate"
    assert opts.table_structure_options.do_cell_matching is True
    assert opts.images_scale == 2.0
    assert opts.do_ocr is True
    assert opts.ocr_options.lang == CONFIG.pdf_ocr_lang.split(",")


def test_pdf_ocr_lang_env_override_reaches_docling(tmp_vault, monkeypatch):
    monkeypatch.setenv("SILICA_PDF_OCR_LANG", "en, ja")
    monkeypatch.setattr(CONFIG, "pdf_ocr_lang", SilicaConfig().pdf_ocr_lang)
    monkeypatch.setattr(CONFIG, "pdf_provider", "docling")
    tmp_vault.note("paper.pdf", "x")
    captured = _fake_docling(monkeypatch)

    conv.convert("paper.pdf")

    assert captured["pipeline_options"].ocr_options.lang == ["en", "ja"]  # csv split, stripped


def test_docling_missing_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "docling")
    tmp_vault.note("paper.pdf", "x")
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    with pytest.raises(ValueError, match="docling not installed"):
        conv.convert("paper.pdf")


# --- opendataloader provider (Java-backed, keeps figures) -------------------

def _fake_opendataloader(monkeypatch, md="# Title\n\n![](images/fig.png)\n\nbody"):
    """Inject a fake opendataloader_pdf.convert that writes one .md + one image."""
    mod = types.ModuleType("opendataloader_pdf")

    def convert(*, input_path, output_dir, format, image_output, image_dir, **kw):
        # Exactly the struct-tree pin, and never `hybrid` (generative, out of
        # boundary — spec 2026-07-22).
        assert kw == {"use_struct_tree": True}
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{Path(input_path).stem}.md").write_text(md, encoding="utf-8")
        Path(image_dir).mkdir(parents=True, exist_ok=True)
        (Path(image_dir) / "fig.png").write_bytes(b"\x89PNG fake")

    mod.convert = convert
    monkeypatch.setitem(sys.modules, "opendataloader_pdf", mod)


def test_pdf_opendataloader_provider_embeds_extracted_image(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "opendataloader")
    tmp_vault.note("paper.pdf", "x")
    _fake_opendataloader(monkeypatch)

    body = _inbox_note(conv.convert("paper.pdf", dest_dir="Concepts/X")[0]).read_text(encoding="utf-8")
    assert "![[paper-fig.png]]" in body
    assert (Path(CONFIG.vault_path) / "Concepts/X/Images/paper-fig.png").is_file()


def test_opendataloader_missing_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "opendataloader")
    tmp_vault.note("paper.pdf", "x")
    monkeypatch.setitem(sys.modules, "opendataloader_pdf", None)
    with pytest.raises(ValueError, match="opendataloader-pdf not installed"):
        conv.convert("paper.pdf")


def test_opendataloader_no_markdown_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "opendataloader")
    tmp_vault.note("paper.pdf", "x")
    mod = types.ModuleType("opendataloader_pdf")
    mod.convert = lambda **kw: None  # writes nothing
    monkeypatch.setitem(sys.modules, "opendataloader_pdf", mod)
    with pytest.raises(ValueError, match="produced no markdown"):
        conv.convert("paper.pdf")


# --- mineru provider --------------------------------------------------------

def _fake_mineru_run(returncode=0, stderr="", write_md=True):
    def run(cmd, **kw):
        # Full pinned command (spec 2026-07-22): explicit non-generative backend
        # plus today's upstream defaults pinned against drift. No -l (mineru has
        # no latin-script option; default models cover latin).
        assert cmd[0] == "mineru"
        assert cmd[cmd.index("-b"):] == ["-b", "pipeline", "-m", "auto", "-f", "true", "-t", "true"]
        if write_md:
            out = Path(cmd[cmd.index("-o") + 1])
            stem = Path(cmd[cmd.index("-p") + 1]).stem
            d = out / stem / "txt"
            (d / "images").mkdir(parents=True)
            (d / f"{stem}.md").write_text("# M\n\n![](images/h.jpg)\n", encoding="utf-8")
            (d / "images" / "h.jpg").write_bytes(b"img")

        class R:
            pass

        R.returncode, R.stderr, R.stdout = returncode, stderr, ""
        return R()

    return run


def test_pdf_mineru_provider_success(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "mineru")
    tmp_vault.note("paper.pdf", "x")
    monkeypatch.setattr(conv.subprocess, "run", _fake_mineru_run())

    body = _inbox_note(conv.convert("paper.pdf")[0]).read_text(encoding="utf-8")
    assert "![[paper-h.jpg]]" in body
    assert (Path(CONFIG.vault_path) / "Inbox/Images/paper-h.jpg").is_file()


# --- images and OOXML: mineru is the only backend ---------------------------


@pytest.mark.parametrize("ext", conv.IMG_EXTS + conv.OFFICE_EXTS)
def test_image_and_office_route_to_mineru_whatever_the_provider_says(
    ext, tmp_vault, monkeypatch
):
    """pymupdf opens an image and reads no text out of it (measured: a 1653x2339
    render of a text page yields ''), and it does not open pptx/xlsx at all. So
    unlike DOCX these must NOT fall back to pymupdf when the provider is unset
    or set to something else -- they must reach mineru or fail loudly."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "pymupdf")
    monkeypatch.setattr(conv, "_via_pymupdf", _never_called_pymupdf)
    seen: list[str] = []

    def run(cmd, **kw):
        seen.append(cmd[cmd.index("-p") + 1])
        return _fake_mineru_run()(cmd, **kw)

    monkeypatch.setattr(conv.subprocess, "run", run)
    tmp_vault.note(f"asset{ext}", "x")

    body = _inbox_note(conv.convert(f"asset{ext}")[0]).read_text(encoding="utf-8")
    assert [Path(p).name for p in seen] == [f"asset{ext}"]
    assert "# M" in body


def _never_called_pymupdf(*a, **k):
    raise AssertionError("pymupdf called for an input it cannot read text from")


def test_office_output_lands_under_its_own_parse_dir(tmp_vault, monkeypatch):
    """Measured on mineru 3.4.4: a pptx parses into `<stem>/office/<stem>.md`,
    not the `auto/` a pdf gets. The provider's glob is recursive so both work --
    pinned here because a non-recursive "fix" would silently break only OOXML."""
    tmp_vault.note("deck.pptx", "x")

    def run(cmd, **kw):
        out = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-p") + 1]).stem
        d = out / stem / "office"          # the real layout, not `auto/`
        d.mkdir(parents=True)
        (d / f"{stem}.md").write_text("## Slide One\n\n- bullet\n", encoding="utf-8")

        class R:
            returncode, stderr, stdout = 0, "", ""
        return R()

    monkeypatch.setattr(conv.subprocess, "run", run)

    body = _inbox_note(conv.convert("deck.pptx")[0]).read_text(encoding="utf-8")
    assert "## Slide One" in body
    assert "- bullet" in body


def test_unreadable_image_error_does_not_advise_installing_ocr(tmp_vault, monkeypatch):
    """mineru IS the backend here, so the PDF branch's "install [pdf] and set
    SILICA_PDF_PROVIDER=mineru" would be advice the user has already taken."""
    monkeypatch.setattr(conv, "_pdf_via_mineru", lambda src, wd: ("  \n\n", wd))
    tmp_vault.note("cat.png", "x")

    with pytest.raises(ValueError, match="no readable text in cat.png") as e:
        conv.convert("cat.png")
    assert "SILICA_PDF_PROVIDER" not in str(e.value)
    assert not (Path(CONFIG.vault_path) / CONFIG.inbox_dir / "cat.md").exists()


# The one test that exercises the REAL mineru CLI end to end, answering the
# TODO(real-api) in convert.py for the two families where mineru is the only
# backend. Gated twice: it needs the binary AND ~35 s per input, so it stays out
# of the default run. To verify a change to the image/OOXML lanes:
#
#   SILICA_TEST_REAL_MINERU=1 uv run pytest tests/test_convert.py -k real_mineru
_REAL_MINERU = pytest.mark.skipif(
    not (os.getenv("SILICA_TEST_REAL_MINERU") and shutil.which("mineru")),
    reason="set SILICA_TEST_REAL_MINERU=1 and install mineru to run the real CLI",
)


@_REAL_MINERU
def test_real_mineru_ocrs_an_image_with_no_text_layer(tmp_vault):
    """A screenshot or a scan: pixels only, no text layer anywhere in the file."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Silica ingestion probe", fontsize=24)
    page.insert_text((72, 140), "Second line of scanned text.", fontsize=18)
    rendered = pymupdf.open("pdf", doc.tobytes())[0].get_pixmap(dpi=200)
    png = Path(CONFIG.vault_path) / "scan.png"
    png.write_bytes(rendered.tobytes("png"))
    # the premise: no text layer at all, so a non-OCR provider would find nothing
    assert pymupdf.open(png)[0].get_text().strip() == ""

    body = _inbox_note(conv.convert(str(png))[0]).read_text(encoding="utf-8")
    assert "Silica ingestion probe" in body
    assert "Second line of scanned text." in body


@_REAL_MINERU
def test_real_mineru_reads_a_pptx_without_libreoffice(tmp_vault, monkeypatch):
    """mineru parses OOXML natively (python-pptx), so no `soffice` is spawned --
    which matters because a present-but-hung LibreOffice is a real machine
    state. Slide titles must arrive as headings, since that is what the
    segmenter splits on."""
    pptx_mod = pytest.importorskip("pptx")
    deck = pptx_mod.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[1])
    slide.shapes.title.text = "Ingestion Roadmap"
    slide.placeholders[1].text = "Images via mineru\nOffice native"
    path = Path(CONFIG.vault_path) / "deck.pptx"
    deck.save(path)

    def no_soffice(cmd, **kw):
        assert "soffice" not in str(cmd) and "libreoffice" not in str(cmd)
        return _real_run(cmd, **kw)

    _real_run = conv.subprocess.run
    monkeypatch.setattr(conv.subprocess, "run", no_soffice)

    body = _inbox_note(conv.convert(str(path))[0]).read_text(encoding="utf-8")
    assert "## Ingestion Roadmap" in body
    assert "Images via mineru" in body


def test_gui_picker_accepts_the_new_families(tmp_vault):
    """`supported_nucleate_extensions()` is the GUI drop-zone's accept set; it
    unions DOC_EXTS, so a widened converter must show up there with no edit."""
    from silica.sources.registry import supported_nucleate_extensions

    accepted = set(supported_nucleate_extensions())
    assert {".png", ".jpg", ".tiff", ".pptx", ".xlsx"} <= accepted
    assert {".mp3", ".wav", ".mp4", ".mkv"} <= accepted


# --- legacy office: the LibreOffice hop -------------------------------------


def _fake_soffice(monkeypatch, *, writes_pdf=True, returncode=0, stderr="", hang=False,
                  flavour="libreoffice"):
    """Stand in for soffice, asserting the hardening flags on the way through."""
    def run(cmd, **kw):
        assert cmd[0].endswith(("soffice", "libreoffice"))
        # Single dash, NOT `--headless`: Apache OpenOffice does not know the
        # double-dash form, and an option it does not know makes it start in GUI
        # mode and open its first-start wizard.
        assert "-headless" in cmd and "--headless" not in cmd
        # A fresh profile IS a first run, so the wizard must be suppressed too.
        assert "-nofirststartwizard" in cmd
        # the load-bearing flag: without it a conversion blocks on the lock held
        # by the user's own open office window
        assert any(a.startswith("-env:UserInstallation=file://") for a in cmd)
        assert kw.get("timeout") == conv._SOFFICE_TIMEOUT_S
        assert kw.get("stdin") is subprocess.DEVNULL
        if hang:
            raise subprocess.TimeoutExpired(cmd, conv._SOFFICE_TIMEOUT_S)
        if writes_pdf:
            out = Path(cmd[cmd.index("-outdir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "deck.pdf").write_bytes(_pdf_bytes(["Slide text from a .ppt"]))

        class R:
            pass
        R.returncode, R.stderr, R.stdout = returncode, stderr, ""
        return R()

    monkeypatch.setattr(conv.shutil, "which", lambda n: f"/usr/bin/{n}")
    monkeypatch.setattr(conv, "soffice_flavour", lambda exe=None: (flavour, f"{flavour} 1.0"))
    monkeypatch.setattr(conv.subprocess, "run", run)


@pytest.mark.parametrize("ext", conv.LEGACY_OFFICE_EXTS)
def test_legacy_office_goes_through_libreoffice_then_the_pdf_seam(
    ext, tmp_vault, monkeypatch
):
    monkeypatch.setattr(CONFIG, "pdf_provider", "pymupdf")
    _fake_soffice(monkeypatch)
    tmp_vault.note(f"deck{ext}", "x")

    body = _inbox_note(conv.convert(f"deck{ext}")[0]).read_text(encoding="utf-8")
    assert "Slide text from a .ppt" in body


def test_legacy_office_honours_the_configured_pdf_provider(tmp_vault, monkeypatch):
    """The intermediate is a real PDF, so someone who installed mineru for OCR
    must get mineru here too, not a silent downgrade to pymupdf."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "mineru")
    _fake_soffice(monkeypatch)
    seen: list[str] = []
    monkeypatch.setitem(
        conv.PDF_PROVIDERS, "mineru",
        lambda src, wd: (seen.append(src.suffix) or ("# From mineru\n\nbody", wd)),
    )
    tmp_vault.note("old.doc", "x")

    body = _inbox_note(conv.convert("old.doc")[0]).read_text(encoding="utf-8")
    assert seen == [".pdf"]          # it received the converted PDF, not the .doc
    assert "# From mineru" in body


def test_libreoffice_timeout_says_what_to_do(tmp_vault, monkeypatch):
    """The branch that matters most: a present-but-hung LibreOffice. omniparse
    runs this same conversion with no timeout at all, so there it blocks forever
    with no output and no error."""
    _fake_soffice(monkeypatch, hang=True)
    monkeypatch.setattr(conv, "probe_soffice", lambda *a, **k: ("hung", "no CPU burned"))
    tmp_vault.note("old.ppt", "x")

    with pytest.raises(ValueError, match="LibreOffice timed out.*probe: hung"):
        conv.convert("old.ppt")


def test_libreoffice_silent_failure_is_an_error_not_an_empty_note(tmp_vault, monkeypatch):
    """soffice exits 0 and writes nothing on some inputs; a missing PDF has to
    fail here rather than surface as an empty conversion."""
    _fake_soffice(monkeypatch, writes_pdf=False, stderr="Error: source file could not be loaded")
    tmp_vault.note("old.doc", "x")

    with pytest.raises(ValueError, match="could not convert old.doc.*could not be loaded"):
        conv.convert("old.doc")


def test_missing_libreoffice_leads_with_the_free_workaround(tmp_vault, monkeypatch):
    """Re-saving as `.docx` costs nothing and pymupdf reads it in the base
    install, so the 240 MB install must not be the first thing offered."""
    monkeypatch.setattr(conv.shutil, "which", lambda n: None)
    tmp_vault.note("old.doc", "x")

    with pytest.raises(ValueError, match=r"needs LibreOffice.*re-save the file as \.docx"):
        conv.convert("old.doc")


def test_apache_openoffice_is_refused_before_the_subprocess(tmp_vault, monkeypatch):
    """The measured case. AOO 4.1 never implemented `-convert-to` (the string is
    absent from the whole install), and handed an option it does not know it
    starts in GUI mode and opens its first-start wizard. A dialog on the user's
    screen is not something a timeout can undo, so this must never reach
    subprocess.run at all."""
    monkeypatch.setattr(conv.shutil, "which", lambda n: "/usr/bin/soffice")
    monkeypatch.setattr(
        conv, "soffice_flavour", lambda exe=None: ("openoffice", "OpenOffice 4.1.16")
    )
    monkeypatch.setattr(conv.subprocess, "run", _never_called)
    tmp_vault.note("old.ppt", "x")

    with pytest.raises(ValueError, match="OpenOffice 4.1.16 cannot convert old.ppt"):
        conv.convert("old.ppt")


def test_apache_openoffice_error_names_the_right_ooxml_target_per_format(
    tmp_vault, monkeypatch
):
    """Both remaining formats have a way out, and it is a different one each:
    telling someone with a `.doc` to re-save it as `.pptx` is noise."""
    monkeypatch.setattr(conv.shutil, "which", lambda n: "/usr/bin/soffice")
    monkeypatch.setattr(conv, "soffice_flavour", lambda exe=None: ("openoffice", "AOO"))
    tmp_vault.note("old.ppt", "x")
    tmp_vault.note("old.doc", "x")

    with pytest.raises(ValueError, match=r"Re-save the file as \.pptx or PDF"):
        conv.convert("old.ppt")
    with pytest.raises(ValueError, match=r"Re-save the file as \.docx or PDF"):
        conv.convert("old.doc")




def test_probe_reports_missing_when_there_is_no_binary(monkeypatch):
    monkeypatch.setattr(conv.shutil, "which", lambda n: None)
    assert conv.probe_soffice()[0] == "missing"


def test_probe_reports_unsupported_for_apache_openoffice_without_running_it(monkeypatch):
    """A boolean probe would call AOO healthy: it answers `which`, it starts, it
    exits 0. Identifying it must not involve running it."""
    monkeypatch.setattr(conv.shutil, "which", lambda n: "/usr/bin/soffice")
    monkeypatch.setattr(
        conv, "soffice_flavour", lambda exe=None: ("openoffice", "OpenOffice 4.1.16")
    )
    monkeypatch.setattr(conv.subprocess, "run", _never_called)

    status, detail = conv.probe_soffice()
    assert status == "unsupported"
    assert "no headless" in detail and "4.1.16" in detail


def test_probe_reports_hung_on_a_binary_that_never_exits(monkeypatch):
    monkeypatch.setattr(conv.shutil, "which", lambda n: "/usr/bin/soffice")
    monkeypatch.setattr(conv, "soffice_flavour", lambda exe=None: ("libreoffice", "LO"))

    def never_exits(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 1))

    monkeypatch.setattr(conv.subprocess, "run", never_exits)
    status, detail = conv.probe_soffice(timeout_s=1)
    assert status == "hung"
    assert "did not start and exit" in detail


def test_probe_reports_broken_on_a_nonzero_exit(monkeypatch):
    monkeypatch.setattr(conv.shutil, "which", lambda n: "/usr/bin/soffice")
    monkeypatch.setattr(conv, "soffice_flavour", lambda exe=None: ("libreoffice", "LO"))

    class R:
        returncode = 77
        stderr = "javaldx: Could not find a Java Runtime Environment!\nlibreglo.so: cannot open shared object"
        stdout = ""

    monkeypatch.setattr(conv.subprocess, "run", lambda cmd, **kw: R())
    status, detail = conv.probe_soffice()
    assert status == "broken"
    # the javaldx warning is noise AOO prints even on success; it must not be
    # reported as the cause
    assert "cannot open shared object" in detail
    assert "javaldx" not in detail


def test_flavour_is_read_from_bootstraprc_not_from_running_it(tmp_path, monkeypatch):
    program = tmp_path / "program"
    program.mkdir()
    exe = program / "soffice"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    (program / "bootstraprc").write_text(
        "[Bootstrap]\nProductKey=OpenOffice 4.1.16\n", encoding="utf-8"
    )
    monkeypatch.setattr(conv.subprocess, "run", _never_called)

    assert conv.soffice_flavour(str(exe)) == ("openoffice", "OpenOffice 4.1.16")

    (program / "bootstraprc").write_text(
        "[Bootstrap]\nProductKey=LibreOffice 25.8\n", encoding="utf-8"
    )
    assert conv.soffice_flavour(str(exe)) == ("libreoffice", "LibreOffice 25.8")


def test_flavour_is_unknown_when_there_is_no_bootstraprc(tmp_path):
    exe = tmp_path / "soffice"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    assert conv.soffice_flavour(str(exe)) == ("unknown", "")


_HAS_SOFFICE = pytest.mark.skipif(
    not shutil.which("soffice") and not shutil.which("libreoffice"),
    reason="needs an office suite installed to probe",
)


@_HAS_SOFFICE
def test_real_office_probe_answers_within_its_leash():
    """Probe whatever office suite this machine actually has.

    Accepts any real answer on purpose: on the machine this was written on the
    honest one is "unsupported" (Apache OpenOffice 4.1.16), and a test demanding
    "ok" would only pass on a LibreOffice box. What it pins is what the probe must
    never do — exceed its own leash, crash, or open a window.
    """
    import time

    started = time.monotonic()
    status, detail = conv.probe_soffice(timeout_s=8)
    elapsed = time.monotonic() - started

    assert status in {"ok", "unsupported", "hung", "broken"}
    assert detail
    # the leash plus generous slack for process teardown
    assert elapsed < 12, f"probe took {elapsed:.1f}s"


# --- office without an office suite: ODF, RTF, legacy .xls -------------------

_ODF_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<office:document-content '
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0">'
    "<office:body><office:text>"
)
_ODF_TAIL = "</office:text></office:body></office:document-content>"


def _odf_bytes(body: str, ext: str = ".odt") -> bytes:
    """A real ODF file: a ZIP with mimetype + content.xml, like the suites write."""
    import zipfile

    kind = {".odt": "text", ".odp": "presentation", ".ods": "spreadsheet"}[ext]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", f"application/vnd.oasis.opendocument.{kind}")
        z.writestr("content.xml", _ODF_HEAD + body + _ODF_TAIL)
    return buf.getvalue()


def _biff2_bytes(rows: list[list]) -> bytes:
    """A real (tiny) legacy `.xls`, hand-assembled.

    Writing one needs no library: BIFF2 is BOF / one record per cell / EOF, and
    xlrd still parses it. The alternative was `xlwt`, a dependency added purely
    so a test could produce input — and one that has been unmaintained since 2019.
    """
    def rec(op: int, payload: bytes = b"") -> bytes:
        return struct.pack("<HH", op, len(payload)) + payload

    out = [rec(0x0009, struct.pack("<HH", 2, 0x0010))]      # BOF: biff2, worksheet
    for r, cells in enumerate(rows):
        for c, v in enumerate(cells):
            head = struct.pack("<HH", r, c) + b"\x00\x00\x00"   # row, col, attrs
            if isinstance(v, (int, float)):
                out.append(rec(0x0003, head + struct.pack("<d", v)))     # NUMBER
            else:
                b = v.encode("latin-1")
                out.append(rec(0x0004, head + bytes([len(b)]) + b))      # LABEL
    out.append(rec(0x000A))                                  # EOF
    return b"".join(out)


def _no_subprocess(monkeypatch):
    """Nothing in these lanes may shell out — that is the whole point."""
    monkeypatch.setattr(conv.subprocess, "run", _never_called)
    monkeypatch.setattr(conv, "soffice_bin", _never_called)


def test_only_the_two_binary_formats_still_need_an_office_suite():
    """Regression guard on the routing itself: widening this back to ODF/RTF/xls
    should be a failing test, not a 240 MB dependency quietly coming back."""
    assert set(conv.LEGACY_OFFICE_EXTS) == {".doc", ".ppt"}
    assert set(conv._PURE_PY_OFFICE_EXTS) == {".odt", ".odp", ".ods", ".rtf", ".xls"}


def test_odt_is_read_with_no_office_suite_and_no_subprocess(tmp_vault, monkeypatch):
    _no_subprocess(monkeypatch)
    (Path(CONFIG.vault_path) / "notes.odt").write_bytes(_odf_bytes(
        '<text:h text:outline-level="1">Chapter One</text:h>'
        "<text:p>First paragraph.</text:p>"
        '<text:h text:outline-level="3">Deep heading</text:h>'
        "<text:p>Second paragraph.</text:p>"
    ))

    body = _inbox_note(conv.convert("notes.odt")[0]).read_text(encoding="utf-8")
    assert "# Chapter One" in body
    assert "### Deep heading" in body
    assert "First paragraph." in body and "Second paragraph." in body


def test_odf_outline_level_is_clamped_to_markdown_depth(tmp_vault, monkeypatch):
    """ODF outline levels run past 6; `####### x` is not a heading in markdown."""
    _no_subprocess(monkeypatch)
    (Path(CONFIG.vault_path) / "deep.odt").write_bytes(_odf_bytes(
        '<text:h text:outline-level="9">Too deep</text:h>'
    ))

    body = _inbox_note(conv.convert("deep.odt")[0]).read_text(encoding="utf-8")
    assert "###### Too deep" in body
    assert "####### " not in body


def test_odf_text_inside_a_text_box_is_not_emitted_twice(tmp_vault, monkeypatch):
    """A draw:frame puts `text:p` inside a `text:p`. Walking every block without
    dropping the nested ones prints the frame's words a second time."""
    _no_subprocess(monkeypatch)
    (Path(CONFIG.vault_path) / "boxed.odt").write_bytes(_odf_bytes(
        "<text:p><draw:frame><draw:text-box>"
        "<text:p>Caption words</text:p>"
        "</draw:text-box></draw:frame></text:p>"
    ))

    body = _inbox_note(conv.convert("boxed.odt")[0]).read_text(encoding="utf-8")
    assert body.count("Caption words") == 1


def test_ods_rows_stay_rows_and_the_padding_is_dropped(tmp_vault, monkeypatch):
    """A sheet reaches the segmenter as lines, not as one paragraph per row, and
    the empty cells ODF pads every row with do not become trailing pipes."""
    _no_subprocess(monkeypatch)
    row = (
        "<table:table-row>"
        "<table:table-cell><text:p>{a}</text:p></table:table-cell>"
        "<table:table-cell><text:p>{b}</text:p></table:table-cell>"
        '<table:table-cell table:number-columns-repeated="1022"/>'
        "</table:table-row>"
    )
    (Path(CONFIG.vault_path) / "sheet.ods").write_bytes(_odf_bytes(
        "<table:table>"
        + row.format(a="item", b="qty")
        + row.format(a="bolt", b="3")
        + "</table:table>",
        ext=".ods",
    ))

    body = _inbox_note(conv.convert("sheet.ods")[0]).read_text(encoding="utf-8")
    assert "item | qty\nbolt | 3" in body
    assert "| |" not in body and not any(
        ln.endswith("|") for ln in body.splitlines()
    )


def test_a_corrupt_odf_says_so_instead_of_raising_a_zip_error(tmp_vault, monkeypatch):
    _no_subprocess(monkeypatch)
    (Path(CONFIG.vault_path) / "broken.odt").write_bytes(b"not a zip at all")

    with pytest.raises(ValueError, match="not a readable ODF document.*export it as PDF"):
        conv.convert("broken.odt")


def test_rtf_is_read_with_no_office_suite(tmp_vault, monkeypatch):
    _no_subprocess(monkeypatch)
    tmp_vault.note(
        "memo.rtf",
        r"{\rtf1\ansi\deff0 {\fonttbl{\f0 Times;}}\f0\fs24 Hello from RTF.\par}",
    )

    body = _inbox_note(conv.convert("memo.rtf")[0]).read_text(encoding="utf-8")
    assert "Hello from RTF." in body


def test_legacy_xls_is_read_with_no_office_suite(tmp_vault, monkeypatch):
    _no_subprocess(monkeypatch)
    (Path(CONFIG.vault_path) / "stock.xls").write_bytes(
        _biff2_bytes([["item", "qty"], ["bolt", 3.0], ["nut", 4.5]])
    )

    body = _inbox_note(conv.convert("stock.xls")[0]).read_text(encoding="utf-8")
    assert "item | qty" in body
    # xlrd hands back every number as a float: unrepaired, a count of 3 reaches
    # the vault as "3.0", and a fractional value must still keep its decimals.
    assert "bolt | 3" in body and "bolt | 3.0" not in body
    assert "nut | 4.5" in body


def test_xls_sheet_names_become_headings_the_segmenter_can_split_on(
    tmp_vault, monkeypatch
):
    _no_subprocess(monkeypatch)
    (Path(CONFIG.vault_path) / "book.xls").write_bytes(_biff2_bytes([["only", "row"]]))

    body = _inbox_note(conv.convert("book.xls")[0]).read_text(encoding="utf-8")
    assert re.search(r"^## \S", body, re.M)


def test_an_empty_office_document_does_not_advertise_ocr(tmp_vault, monkeypatch):
    """These lanes have no OCR anywhere, so the usual "install mineru" advice
    would send the user to fix a component this path never touches."""
    _no_subprocess(monkeypatch)
    (Path(CONFIG.vault_path) / "blank.odt").write_bytes(_odf_bytes("<text:p/>"))

    with pytest.raises(ValueError, match="no text in blank.odt") as e:
        conv.convert("blank.odt")
    assert "mineru" not in str(e.value)


# --- media: ffmpeg demux + an ASR provider ----------------------------------

_VTT = """WEBVTT

00:00:00.000 --> 00:00:02.000
The first thing to know

00:00:02.100 --> 00:00:04.000
is that the demux is one call.

00:00:19.000 --> 00:00:21.000
After a long pause, a new thought.
"""


def _fake_asr(monkeypatch, vtt=_VTT):
    """Stand in for the transcription server, at the provider seam."""
    from silica.sources.web_fetch import vtt_to_text

    monkeypatch.setattr(
        conv, "_asr_via_endpoint",
        lambda wav: vtt_to_text(vtt, paragraph_gap_s=conv._ASR_PARAGRAPH_GAP_S),
    )
    monkeypatch.setitem(conv.ASR_PROVIDERS, "endpoint", conv._asr_via_endpoint)


def _fake_wav(monkeypatch):
    """Stand in for ffmpeg: write a wav with more than a bare header."""
    def to_wav(src, workdir):
        wav = Path(workdir) / "audio.wav"
        wav.write_bytes(b"RIFF" + b"\0" * 64)
        return wav
    monkeypatch.setattr(conv, "_media_to_wav", to_wav)


@pytest.mark.parametrize("ext", conv.MEDIA_EXTS)
def test_every_media_ext_reaches_the_asr_lane(ext, tmp_vault, monkeypatch):
    _fake_wav(monkeypatch)
    _fake_asr(monkeypatch)
    tmp_vault.note(f"talk{ext}", "x")

    body = _inbox_note(conv.convert(f"talk{ext}")[0]).read_text(encoding="utf-8")
    assert f"# talk" in body
    assert "the demux is one call." in body


def test_transcript_gets_paragraph_breaks_at_pauses(tmp_vault, monkeypatch):
    """A transcript with no blank line anywhere is ONE paragraph, and
    `_split_by_size` leaves an oversized paragraph whole — so a long talk would
    land as a single note whose concepts RECON caps at 40. A pause is the only
    paragraph boundary a transcript carries."""
    _fake_wav(monkeypatch)
    _fake_asr(monkeypatch)
    tmp_vault.note("talk.mp3", "x")

    body = _inbox_note(conv.convert("talk.mp3")[0]).read_text(encoding="utf-8")
    assert "is that the demux is one call.\n\nAfter a long pause" in body


def test_silent_media_raises_instead_of_writing_an_empty_note(tmp_vault, monkeypatch):
    _fake_wav(monkeypatch)
    monkeypatch.setitem(conv.ASR_PROVIDERS, "endpoint", lambda wav: "   \n")
    tmp_vault.note("music.mp3", "x")

    with pytest.raises(ValueError, match="no speech transcribed"):
        conv.convert("music.mp3")
    assert not (Path(CONFIG.vault_path) / CONFIG.inbox_dir / "music.md").exists()


def test_unknown_asr_provider_names_the_known_ones(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "stt_provider", "nope")
    tmp_vault.note("talk.mp3", "x")

    with pytest.raises(ValueError, match="unknown stt_provider.*endpoint"):
        conv.convert("talk.mp3")


def test_media_never_reaches_a_document_provider(tmp_vault, monkeypatch):
    """A .mp4 handed to pymupdf or mineru is a parse of garbage, not an error."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "mineru")
    monkeypatch.setattr(conv, "_via_pymupdf", _never_called_pymupdf)
    monkeypatch.setattr(conv.subprocess, "run", _never_called)
    _fake_wav(monkeypatch)
    _fake_asr(monkeypatch)
    tmp_vault.note("clip.mkv", "x")

    assert conv.convert("clip.mkv")


@pytest.mark.parametrize("base,expected", [
    ("http://h:8080", "http://h:8080/v1"),
    ("http://h:8080/", "http://h:8080/v1"),
    ("http://h:8080/v1", "http://h:8080/v1"),
    ("http://h:8080/v1/", "http://h:8080/v1"),
])
def test_asr_base_url_is_normalised(base, expected):
    """Users paste both shapes; a doubled or missing /v1 is a 404 whose message
    says nothing about the cause."""
    assert conv._asr_base(base) == expected


def test_endpoint_provider_posts_multipart_and_asks_for_vtt(tmp_vault, monkeypatch):
    """VTT rather than the default json: the cue timings are the only thing that
    can become paragraph breaks downstream."""
    seen = {}

    class R:
        status_code, text = 200, _VTT

    def fake_post(url, files=None, data=None, headers=None, timeout=None):
        seen.update(url=url, data=dict(data or {}), name=files["file"][0],
                    payload=files["file"][1].read(), headers=dict(headers or {}))
        return R()

    import httpx
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(CONFIG, "stt_base_url", "http://127.0.0.1:9999")
    monkeypatch.setattr(CONFIG, "stt_lang", "it")
    wav = Path(CONFIG.vault_path) / "a.wav"
    wav.write_bytes(b"RIFF-payload")

    out = conv._asr_via_endpoint(wav)
    assert seen["url"] == "http://127.0.0.1:9999/v1/audio/transcriptions"
    assert seen["data"]["response_format"] == "vtt"
    assert seen["data"]["language"] == "it"
    assert seen["headers"]["Authorization"] == f"Bearer {CONFIG.stt_api_key}"
    assert seen["payload"] == b"RIFF-payload"   # the file really was sent
    assert "the demux is one call." in out
    assert "-->" not in out and "WEBVTT" not in out


def test_endpoint_provider_survives_a_server_that_ignores_response_format(tmp_vault, monkeypatch):
    """Some OpenAI-compatible servers answer json whatever you ask for. Writing
    `{"text": ...}` into the vault would be worse than reading it out."""
    class R:
        status_code = 200
        text = '{"text": "plain json answer"}'

    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: R())
    wav = Path(CONFIG.vault_path) / "a.wav"
    wav.write_bytes(b"x")

    assert conv._asr_via_endpoint(wav) == "plain json answer"


def test_no_server_says_how_to_start_one(tmp_vault, monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", boom)
    wav = Path(CONFIG.vault_path) / "a.wav"
    wav.write_bytes(b"x")

    with pytest.raises(ValueError, match="no transcription server.*whisper-server"):
        conv._asr_via_endpoint(wav)


def test_whispercpp_without_a_model_says_so(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "stt_whispercpp_bin", "")
    monkeypatch.setattr(CONFIG, "stt_whispercpp_model", "")
    monkeypatch.setattr(conv.shutil, "which", lambda n: "/usr/bin/whisper-cli")
    wav = Path(CONFIG.vault_path) / "a.wav"
    wav.write_bytes(b"x")

    with pytest.raises(ValueError, match="whisper.cpp needs a model file"):
        conv._asr_via_whispercpp(wav)


def test_missing_ffmpeg_names_the_install(tmp_vault, monkeypatch):
    monkeypatch.setattr(conv.shutil, "which", lambda n: None)
    with pytest.raises(ValueError, match="needs ffmpeg on PATH"):
        conv._media_to_wav(Path("a.mp3"), Path(CONFIG.vault_path))


_HAS_FFMPEG = pytest.mark.skipif(
    not shutil.which("ffmpeg"), reason="needs the real ffmpeg binary"
)


@_HAS_FFMPEG
def test_real_ffmpeg_demuxes_a_video_to_mono_16k_wav(tmp_vault, tmp_path):
    """The demux half is Silica's own code, so it is verified against the real
    binary rather than a fake: a container with a video stream and a tone must
    come out as 16 kHz mono PCM with actual samples in it."""
    src = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=1:size=128x96:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src)],
        check=True, capture_output=True,
    )

    wav = conv._media_to_wav(src, tmp_path)
    assert wav.is_file() and wav.stat().st_size > 1000
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels,sample_rate,codec_name", "-of", "csv=p=0", str(wav)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert probe.startswith("pcm_s16le,16000,1"), probe


@_HAS_FFMPEG
def test_real_ffmpeg_rejects_a_file_with_no_audio_track(tmp_vault, tmp_path):
    """A silent screen recording must fail loudly, not transcribe to nothing."""
    src = tmp_path / "mute.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=1:size=128x96:rate=10",
         "-c:v", "libx264", str(src)],
        check=True, capture_output=True,
    )

    with pytest.raises(ValueError, match="no audio track|could not read"):
        conv._media_to_wav(src, tmp_path)


@_HAS_FFMPEG
def test_real_ffmpeg_reports_a_file_that_is_not_media(tmp_vault, tmp_path):
    src = tmp_path / "fake.mp3"
    src.write_text("this is not audio", encoding="utf-8")
    with pytest.raises(ValueError, match="could not read fake.mp3"):
        conv._media_to_wav(src, tmp_path)


@_HAS_FFMPEG
def test_a_real_video_becomes_a_real_note_over_a_real_socket(tmp_vault, tmp_path):
    """The whole media lane end to end, with only the ASR *model* substituted.

    Everything here is the real thing: ffmpeg demuxes a real mp4, httpx encodes
    a real multipart body over a real TCP socket, and a stdlib HTTP server
    answers with VTT the way whisper-server does. That leaves exactly one fake —
    the transcription itself — so the multipart encoding, the response parsing,
    the paragraph breaks and the inbox write are all verified rather than
    asserted. The monkeypatched-httpx test above cannot see any of that.
    """
    import http.server
    import threading

    received: dict[str, bytes] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received["path"] = self.path.encode()
            received["ctype"] = (self.headers.get("content-type") or "").encode()
            received["body"] = self.rfile.read(int(self.headers["content-length"]))
            payload = _VTT.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/vtt")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):  # keep pytest output clean
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        src = Path(CONFIG.vault_path) / "keynote.mp4"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc=duration=1:size=128x96:rate=10",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src)],
            check=True, capture_output=True,
        )
        CONFIG.stt_base_url = f"http://127.0.0.1:{server.server_port}"
        CONFIG.stt_provider = "endpoint"
        CONFIG.stt_lang = ""

        notes = conv.convert(str(src))
    finally:
        server.shutdown()
        CONFIG.stt_base_url = SilicaConfig().stt_base_url
        CONFIG.stt_provider = SilicaConfig().stt_provider

    # the server saw a real multipart upload at the OpenAI-compatible path
    assert received["path"] == b"/v1/audio/transcriptions"
    assert received["ctype"].startswith(b"multipart/form-data")
    assert b'name="file"' in received["body"] and b"RIFF" in received["body"]
    assert b'name="response_format"' in received["body"] and b"vtt" in received["body"]

    body = _inbox_note(notes[0]).read_text(encoding="utf-8")
    assert body.startswith('---\n')                     # frontmatter block
    assert 'source_file: "' in body                     # provenance survived
    assert 'form: transcript' in body                   # media ingress stamp
    assert "# keynote" in body
    assert "The first thing to know" in body
    assert "is that the demux is one call.\n\nAfter a long pause" in body
    assert "WEBVTT" not in body and "-->" not in body


def test_same_figure_name_from_two_pdfs_does_not_clobber(tmp_vault, monkeypatch):
    """Providers name figures by page index (`_page_0_Figure_1.jpeg`), which
    repeats across documents. The vault Images/ dir is flat and the embed is by
    basename, so an un-namespaced second PDF would overwrite the first's figure
    AND silently repoint the first note at it."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "opendataloader")
    tmp_vault.note("alpha.pdf", "x")
    tmp_vault.note("beta.pdf", "x")

    _fake_opendataloader(monkeypatch, md="# A\n\n![](images/fig.png)\n")
    a_body = _inbox_note(conv.convert("alpha.pdf")[0]).read_text(encoding="utf-8")
    images = Path(CONFIG.vault_path) / "Inbox/Images"
    (images / "alpha-fig.png").write_bytes(b"ALPHA")   # distinguishable content

    _fake_opendataloader(monkeypatch, md="# B\n\n![](images/fig.png)\n")
    b_body = _inbox_note(conv.convert("beta.pdf")[0]).read_text(encoding="utf-8")

    assert "![[alpha-fig.png]]" in a_body
    assert "![[beta-fig.png]]" in b_body
    assert (images / "alpha-fig.png").read_bytes() == b"ALPHA"  # untouched by beta
    assert (images / "beta-fig.png").is_file()


def test_image_names_are_stable_across_reconversion(tmp_vault, monkeypatch):
    """Re-converting the same PDF must reuse the same image names, or every run
    would leave the previous run's figures behind as orphans."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "opendataloader")
    tmp_vault.note("paper.pdf", "x")
    _fake_opendataloader(monkeypatch)

    conv.convert("paper.pdf")
    conv.convert("paper.pdf")

    images = Path(CONFIG.vault_path) / "Inbox/Images"
    assert [p.name for p in sorted(images.iterdir())] == ["paper-fig.png"]


def test_mineru_missing_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "mineru")
    tmp_vault.note("paper.pdf", "x")

    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(conv.subprocess, "run", boom)
    with pytest.raises(ValueError, match="mineru not installed"):
        conv.convert("paper.pdf")


def test_mineru_nonzero_exit_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "mineru")
    tmp_vault.note("paper.pdf", "x")
    monkeypatch.setattr(
        conv.subprocess, "run", _fake_mineru_run(returncode=1, stderr="kaboom", write_md=False)
    )
    with pytest.raises(ValueError, match="mineru failed"):
        conv.convert("paper.pdf")


def test_unknown_provider_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "bogus")
    tmp_vault.note("paper.pdf", "x")
    with pytest.raises(ValueError, match="unknown pdf_provider"):
        conv.convert("paper.pdf")


def test_respace_prose_fixes_tight_punctuation_outside_math_and_code():
    md = (
        "symmetric,and positive kernel with 10,000 samples\n"
        "$f(x,y)$ and $$\\alpha,\\beta$$ stay\n"
        "```\na,b = 1,2\n```\n"
    )
    fixed = conv._respace_prose(md)
    assert "symmetric, and positive" in fixed   # prose glitch fixed
    assert "10,000" in fixed                     # digits untouched
    assert "$f(x,y)$" in fixed                   # inline math untouched
    assert "a,b = 1,2" in fixed                  # fenced code untouched


# --- book segmentation (split_markdown) -------------------------------------

def test_split_on_headings_splits_chapters():
    # max_chars small enough that no two sections pack together → cuts land
    # exactly on the heading boundaries.
    segs = conv.split_markdown("# Book\n\nintro\n\n## One\n\naaa\n\n## Two\n\nbbb", max_chars=20)
    assert len(segs) == 3
    assert segs[0].startswith("# Book")      # preamble attached to first heading
    assert "## One" in segs[1] and "## Two" in segs[2]


def test_split_ignores_headings_inside_code_fences():
    md = "intro\n\n```\n# not a heading\n```\n\n## Real\n\nbody"
    segs = conv.split_markdown(md, max_chars=40)
    assert len(segs) == 2                     # the fenced '# ...' is not a boundary
    assert "# not a heading" in segs[0]


def test_split_packs_small_sections_together():
    """Real converters flatten everything to ## and emit lone '## Chapter N'
    lines (80-page docling probe: 53 raw segments, some 14 chars) — adjacent
    small sections must coalesce instead of becoming micro-notes."""
    md = "".join(f"## S{i}\n\n{'x' * 50}\n\n" for i in range(10))
    segs = conv.split_markdown(md, max_chars=200)
    assert 1 < len(segs) < 10                  # packed, not one-note-per-heading
    assert all(len(s) <= 200 for s in segs)
    assert segs[0].count("## S") >= 2          # a pack spans multiple headings


def test_split_small_multiheading_doc_packs_to_one():
    md = "# Paper\n\nintro\n\n## Method\n\naaa\n\n## Results\n\nbbb"
    assert conv.split_markdown(md) == [md]     # a paper stays one flat note


def test_split_dimensional_fallback_when_no_headings():
    body = "".join(f"Paragraph {i} of heading-less scanned prose.\n\n" for i in range(200))
    segs = conv.split_markdown(body, max_chars=500)
    assert len(segs) > 1                       # blind body still gets cut into parts
    assert all(len(s) <= 600 for s in segs)    # ≤ max + one paragraph of slack


def test_split_size_caps_an_oversized_heading_section():
    big = "## Huge\n\n" + "".join(f"line {i}\n\n" for i in range(300))
    segs = conv.split_markdown(big, max_chars=400)
    assert len(segs) > 1                        # a giant chapter is split further


def test_split_single_small_section_is_one_segment():
    assert conv.split_markdown("# Paper\n\nshort body") == ["# Paper\n\nshort body"]


def test_pdf_book_splits_into_multiple_inbox_notes(tmp_vault, monkeypatch):
    """A multi-chapter converted PDF becomes N inbox notes under <stem>/,
    numbered and slugged — one RECON unit per chapter, not the whole book."""
    monkeypatch.setattr(CONFIG, "pdf_provider", "docling")
    tmp_vault.note("book.pdf", "%PDF fake")
    # Two ~30k sections: front+Alpha pack into one unit, Beta overflows into
    # the next — a genuinely book-sized doc, not a paper. Varied words, not one
    # repeated char: strip_degenerate_runs would collapse a degenerate run.
    alpha = " ".join(f"alpha{i}" for i in range(4_000))
    beta = " ".join(f"beta{i}" for i in range(4_000))
    _fake_docling(monkeypatch, md=(
        f"# Book\n\nfront\n\n## Alpha\n\n{alpha}\n\n## Beta\n\n{beta}"
    ))

    paths = conv.convert("book.pdf", dest_dir="Concepts/X")

    assert len(paths) == 2
    assert paths[0] == f"{CONFIG.inbox_dir}/book/1-book.md"
    assert paths[1].endswith("2-beta.md")
    assert all((Path(CONFIG.vault_path) / p).is_file() for p in paths)
    assert "## Alpha" in (Path(CONFIG.vault_path) / paths[0]).read_text(encoding="utf-8")
    assert "## Beta" in (Path(CONFIG.vault_path) / paths[1]).read_text(encoding="utf-8")


# --- tabular profile (csv/tsv/parquet → profile note, never rows) ------------

def test_csv_profile_carries_stats_not_the_table(tmp_vault):
    p = Path(CONFIG.vault_path) / "metrics.csv"
    p.write_text("name,value\n" + "\n".join(f"item{i},{i}" for i in range(50)) + "\n")

    [note_rel] = conv.convert(str(p))

    body = _inbox_note(note_rel).read_text(encoding="utf-8")
    assert "silica_query_table" in body           # the note routes reads to SQL
    assert "50 rows" in body
    assert "BIGINT" in body                        # sniffed type, not a guess
    assert "| item0 |" in body                     # sample spans the file:
    assert "| item49 |" in body                    # both ends of the 50 rows
    assert "item40" not in body                    # the table itself stayed on disk


def test_csv_profile_sample_spans_the_file(tmp_vault):
    p = Path(CONFIG.vault_path) / "sorted.csv"
    p.write_text("name,value\n" + "\n".join(f"row{i:03d},{i}" for i in range(100)) + "\n")

    [note_rel] = conv.convert(str(p))

    body = _inbox_note(note_rel).read_text(encoding="utf-8")
    # A sorted export's head is one stratum; the sample must span the file.
    assert "row000" in body and "row099" in body
    assert "row001" not in body                    # not the head slice


def test_csv_profile_lists_low_cardinality_values(tmp_vault):
    rows = []
    for i in range(60):
        # "north" sits in the middle of the domain: SUMMARIZE's min/max already
        # leak the alphabetic extremes (east/west), so only a value that is
        # neither an extreme nor in any sampled row proves the enumeration.
        region = "north" if i in (6, 7) else ("east" if i % 2 == 0 else "west")
        rows.append(f"id{i:03d},{region},{i}")
    p = Path(CONFIG.vault_path) / "sales.csv"
    p.write_text("id,region,value\n" + "\n".join(rows) + "\n")

    [note_rel] = conv.convert(str(p))

    body = _inbox_note(note_rel).read_text(encoding="utf-8")
    assert "north" in body                         # reachable only via enumeration
    assert "east" in body and "west" in body
    assert "id013" not in body                     # high-cardinality: never enumerated


def test_categorical_section_skips_columns_min_max_already_shows(tmp_vault):
    p = Path(CONFIG.vault_path) / "flags.csv"
    p.write_text(
        "dataflow,gender,value\n"
        + "\n".join(f"DF_ONLY,{'F' if i % 2 else 'M'},{i}" for i in range(30))
        + "\n"
    )

    [note_rel] = conv.convert(str(p))

    body = _inbox_note(note_rel).read_text(encoding="utf-8")
    # 1 and 2 distinct values are fully carried by the min/max columns above;
    # enumerating them again is noise (a real SDMX export: 13 sections, 10 of
    # them redundant).
    assert "## Categorical values" not in body
    assert "DF_ONLY" in body                       # still discoverable, via min/max


def test_csv_profile_carries_documents_edge_inside_a_repo(tmp_vault):
    vault = Path(CONFIG.vault_path)
    subprocess.run(["git", "init", "-q"], cwd=vault, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=vault, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=vault, check=True)
    p = vault / "data" / "m.csv"
    p.parent.mkdir()
    p.write_text("a,b\n1,2\n")
    subprocess.run(["git", "add", "-A"], cwd=vault, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=vault, check=True)

    [note_rel] = conv.convert(str(p))

    from silica.kernel.write import frontmatter
    data, _, _ = frontmatter.split(_inbox_note(note_rel).read_text(encoding="utf-8"))
    assert data["documents"] == ["data/m.csv"]     # repo-relative, the graph keyspace
    assert data.get("code_ref")                    # arms /stale for tracked files


def test_csv_profile_outside_a_repo_gets_no_documents_edge(tmp_vault):
    p = Path(CONFIG.vault_path) / "sales.csv"
    p.write_text("a,b\n1,2\n")

    [note_rel] = conv.convert(str(p))

    from silica.kernel.write import frontmatter
    data, _, _ = frontmatter.split(_inbox_note(note_rel).read_text(encoding="utf-8"))
    assert "documents" not in (data or {})


def test_csv_family_one_profile_for_shards(tmp_vault):
    v = Path(CONFIG.vault_path)
    (v / "raw").mkdir()
    (v / "raw" / "vendite_sicilia_01.csv").write_text("region,amount\na,1\nb,2\nc,3\n")
    (v / "raw" / "vendite_sicilia_02.csv").write_text("region,amount\nd,4\ne,5\nf,6\ng,7\n")
    (v / "raw" / "altro.csv").write_text("x,y\n1,2\n")

    [note_rel] = conv.convert(str(v / "raw" / "vendite_sicilia_01.csv"))

    assert note_rel.endswith("/vendite_sicilia.md")   # family stem, not the shard
    body = _inbox_note(note_rel).read_text(encoding="utf-8")
    assert "7 rows" in body                           # union of both shards
    assert "vendite_sicilia_01.csv" in body and "vendite_sicilia_02.csv" in body
    assert "altro.csv" not in body                    # different header: not family

    # Converting another member refreshes the same note; a 12-shard folder
    # must not become 12 near-identical profiles.
    [note_rel2] = conv.convert(str(v / "raw" / "vendite_sicilia_02.csv"))
    assert note_rel2 == note_rel


def test_family_members_are_not_reported_unconverted(tmp_vault):
    from silica.tools.graph import _covering_stem

    v = Path(CONFIG.vault_path)
    (v / "vendite_01.csv").write_text("a,b\n1,2\n")
    (v / "vendite_02.csv").write_text("a,b\n3,4\n")

    [note_rel] = conv.convert(str(v / "vendite_01.csv"))

    # The profile is named for the family, so a plain stem match sees two
    # uncovered files where one note already describes both.
    covering = _covering_stem(v / "vendite_02.csv")
    assert covering == Path(note_rel).stem == "vendite"
    # A lone file is covered by its own note only — never by a sibling's.
    assert _covering_stem(v / "altro.csv") == ""


def test_csv_family_requires_a_counter_suffix(tmp_vault):
    v = Path(CONFIG.vault_path)
    # Byte-identical header but different tables: SDMX exports share one
    # generic column layout across datasets (field test: censpop_lavoro_15piu
    # grouped with censpop_demografia under an invented "censpop" name).
    hdr = "DATAFLOW,REF_AREA,OBS_VALUE\n"
    (v / "censpop_lavoro_01.csv").write_text(hdr + "L,x,1\n")
    (v / "censpop_demografia.csv").write_text(hdr + "D,y,2\n")

    [note_rel] = conv.convert(str(v / "censpop_lavoro_01.csv"))

    assert note_rel.endswith("/censpop_lavoro_01.md")  # no family, own stem
    assert "censpop_demografia" not in _inbox_note(note_rel).read_text(encoding="utf-8")


def test_csv_family_needs_a_shared_name(tmp_vault):
    v = Path(CONFIG.vault_path)
    (v / "a.csv").write_text("x,y\n1,2\n")
    (v / "b.csv").write_text("x,y\n3,4\n")

    [n1] = conv.convert(str(v / "a.csv"))
    [n2] = conv.convert(str(v / "b.csv"))

    # Same header but no meaningful common stem: grouping two unrelated files
    # under an invented name would be worse than two profiles.
    assert n1 != n2
    assert "b.csv" not in _inbox_note(n1).read_text(encoding="utf-8")


def test_profile_survives_a_timezone_aware_column(tmp_vault):
    # DuckDB infers TIMESTAMP WITH TIME ZONE and converting one to a Python
    # object needs pytz, which is in no install here: a real download ledger
    # (ISO timestamps with offsets) crashed the whole convert.
    p = Path(CONFIG.vault_path) / "ledger.csv"
    p.write_text(
        "fetched_at,name\n"
        + "\n".join(f"2026-08-12T13:35:{i:02d}+00:00,f{i}" for i in range(10))
        + "\n"
    )

    [note_rel] = conv.convert(str(p))

    body = _inbox_note(note_rel).read_text(encoding="utf-8")
    assert "2026-08-12" in body                     # rendered, not crashed
    assert "TIMESTAMP" in body                      # type still reported
    # The sample header names the column, not the SQL that read it.
    assert "| fetched_at | name |" in body
    assert "CAST(" not in body


def test_profile_reads_a_non_utf8_csv(tmp_vault):
    # Public European data ships windows-1252 (4 of 26 CSVs in the field
    # vault). DuckDB's reader validates encoding and refuses the file, and
    # its own encoding='latin-1' still rejects the C1 range ISTAT uses for
    # the ellipsis (0x85).
    p = Path(CONFIG.vault_path) / "legenda.csv"
    # Raw bytes: 0xf9 is "ù" in both codecs, 0x85 is the ellipsis in cp1252
    # and the NEL control in latin-1 — the byte that decides the ladder.
    p.write_bytes(
        b"codice;descrizione\nP2;Variazione pi\xf9 che annua\nN;\x85 non disponibile\n"
    )

    [note_rel] = conv.convert(str(p))

    body = _inbox_note(note_rel).read_text(encoding="utf-8")
    assert "Variazione pi\xf9 che annua" in body     # accented byte decoded
    assert "…" in body                          # cp1252 won the ladder
    assert "2 rows" in body


def test_profile_collapses_all_null_columns(tmp_vault):
    p = Path(CONFIG.vault_path) / "notes.csv"
    p.write_text("a,b,NOTE_X,NOTE_Y\n" + "\n".join(f"{i},{i * 2},," for i in range(10)) + "\n")

    [note_rel] = conv.convert(str(p))

    body = _inbox_note(note_rel).read_text(encoding="utf-8")
    # Field test: 15 of an SDMX export's 28 columns were 100% null and each
    # occupied a Columns row and a Sample column.
    assert "2 columns entirely null: NOTE_X, NOTE_Y" in body
    assert "| NOTE_X |" not in body


def test_profile_paths_are_repo_relative_inside_a_repo(tmp_vault):
    vault = Path(CONFIG.vault_path)
    subprocess.run(["git", "init", "-q"], cwd=vault, check=True)
    p = vault / "data" / "m.csv"
    p.parent.mkdir()
    p.write_text("a,b\n1,2\n")

    [note_rel] = conv.convert(str(p))

    body = _inbox_note(note_rel).read_text(encoding="utf-8").split("---", 2)[2]
    assert "`data/m.csv`" in body                     # prose and the query example
    assert str(p) not in body                         # absolute stays out of the body


# --- references-section flagging (2026-08-15) --------------------------------

def test_references_section_flagged_with_continuations():
    # A references section and its heading-less overflow parts all come out
    # flagged; real content before it never packs into the same segment.
    body = "# Paper\n\nintro prose\n\n## Method\n\nmmm\n\n"
    refs = "## References\n\n" + "\n\n".join(f"[{i}] Author {i}. Title." for i in range(40))
    segs = conv._split_markdown_flagged(body + refs, max_chars=300)
    flags = [f for _, f in segs]
    assert flags[0] == ""
    assert any(flags), "references never flagged"
    # flag is a suffix run: once references start, every later part is flagged
    first_ref = flags.index("references")
    assert all(flags[first_ref:])
    assert "## References" in segs[first_ref][0]
    # no mixed segment: content text never lands in a flagged segment
    assert all("## Method" not in s for s, f in segs if f)


def test_references_heading_variants_match():
    for h in ("## References", "# REFERENCES", "## Bibliography",
              "## 7. References", "## **References**", "## Bibliografia"):
        assert conv._section_kind(h + "\n\n[1] X.") == "references", h
    for h in ("## Reference Architectures", "## Methods", "## Cross-references in text"):
        assert conv._section_kind(h + "\n\nbody") == "", h


def test_split_markdown_unchanged_without_references():
    md = "# Paper\n\nintro\n\n## Method\n\naaa\n\n## Results\n\nbbb"
    assert conv.split_markdown(md) == [md]


def test_doc_citation_finds_doi_and_arxiv_in_head(tmp_path):
    md = ("# Paper Title\n\nAuthors here\n\n"
          "doi: 10.20944/preprints202603.0359.v1\n\narXiv:2603.01234v2\n\nAbstract…")
    fake = tmp_path / "x.bin"  # unopenable by pymupdf → metadata skipped, regexes still run
    fake.write_bytes(b"")
    cite = conv._doc_citation(fake, md)
    assert cite["doi"] == "10.20944/preprints202603.0359.v1"
    assert cite["arxiv"] == "2603.01234v2"


def test_doc_citation_ignores_reference_section_dois(tmp_path):
    # A DOI past the first-page window is another paper's DOI.
    md = "# T\n\nintro\n" + "x" * 9000 + "\n10.9999/someone-elses-paper"
    fake = tmp_path / "x.bin"
    fake.write_bytes(b"")
    assert "doi" not in conv._doc_citation(fake, md)


# --- back-matter flagging: contents, venue checklists (2026-08-18) -----------
# A 246-paper library run showed >50% of the distilled volume of a modern ML
# paper is apparatus: the table of contents and the NeurIPS submission
# checklist. The checklist alone produced Code of ethics.md, IRB Approval.md,
# Crowdsourcing.md — boilerplate identical across hundreds of papers.

def test_contents_heading_is_boilerplate():
    for h in ("## Contents", "# Table of Contents", "## TABLE OF CONTENTS",
              "## Indice", "## 1. Contents"):
        assert conv._section_kind(h + "\n\n1 Intro 1\n2 Method 2") == "boilerplate", h


def test_venue_checklist_item_is_boilerplate_by_body():
    # Split per item, so the heading carries no shared marker — the
    # Question/Answer/Justification triple is what identifies it.
    item = (
        "## 5. Open access to data and code\n\n"
        "Question: Does the paper provide open access to the data and code?\n\n"
        "Answer: [Yes]\n\n"
        "Justification: We provide the code link in the abstract.\n\n"
        "Guidelines:\n\n"
        "• The answer NA means that paper does not include experiments.\n"
    )
    assert conv._section_kind(item) == "boilerplate"
    assert conv._section_kind("## NeurIPS Paper Checklist\n\n"
                              "Question: Do the claims match?\n\nAnswer: [NA]\n\n"
                              "Justification: n/a\n") == "boilerplate"


def test_real_content_is_never_boilerplate():
    for s in ("## Method\n\nWe answer: yes, the model works.\n",
              "## Results\n\nJustification: not a checklist, just prose.\n",
              "## Contents of the Memory Store\n\nEach note carries a context.\n",
              "## Answer Generation\n\nAnswer: the decoder emits a token.\n"):
        assert conv._section_kind(s) == "", s


def test_references_still_flagged_as_references():
    assert conv._section_kind("## References\n\n[1] X.") == "references"


def test_boilerplate_segment_carries_its_own_frontmatter_key(tmp_vault, monkeypatch):
    chapter = " ".join(f"w{i}" for i in range(6000))
    big = (f"# Paper\n\n{chapter}\n\n"
           "## Contents\n\n1 Intro 1\n2 Method 2\n\n"
           "## References\n\n[1] Author. Title.\n")
    monkeypatch.setattr(conv, "_via_pymupdf", lambda src, wd: (big, wd))
    tmp_vault.note("paper.docx", "x")

    heads = [_inbox_note(p).read_text(encoding="utf-8").split("\n---\n")[0]
             for p in conv.convert("paper.docx")]
    assert any("boilerplate: true" in h for h in heads)
    assert any("references: true" in h for h in heads)
    # the boilerplate segment is not stamped as references and vice versa
    assert not any("boilerplate: true" in h and "references: true" in h for h in heads)


def test_skippable_chunk_recognizes_both_keys(tmp_vault):
    tmp_vault.note("Inbox/a.md", "---\nreferences: true\n---\n\n[1] X.")
    tmp_vault.note("Inbox/b.md", "---\nboilerplate: true\n---\n\n## Contents")
    tmp_vault.note("Inbox/c.md", "---\ntype: Note\n---\n\nreal content")
    assert conv.is_skippable_chunk("Inbox/a.md")
    assert conv.is_skippable_chunk("Inbox/b.md")
    assert not conv.is_skippable_chunk("Inbox/c.md")


def test_a_paper_that_merely_contains_a_checklist_is_not_boilerplate():
    """The killer false positive: with sparse headings a whole PDF is ONE
    section, and an unwindowed body rule flagged 100% of a-mem-agentic-memory
    (96 KB) as apparatus. The triple has to sit at the head of the section."""
    paper = ("# A-Mem: Agentic Memory for LLM Agents\n\n"
             + "Real prose about memory evolution. " * 400
             + "\n\n## NeurIPS Paper Checklist\n\nQuestion: Do the claims match?\n\n"
               "Answer: [Yes]\n\nJustification: See Section 4.\n")
    assert conv._section_kind(paper) == ""
    # split into its real sections, only the checklist one is flagged
    kinds = [k for _, k in conv._split_markdown_flagged(paper, 100_000)]
    assert kinds[0] == "" and kinds[-1] == "boilerplate", kinds


def test_checklist_guidelines_fragment_is_boilerplate():
    """The Guidelines block carries no Question/Answer/Justification triple, so
    the head rule cannot see it — the template's own wording can."""
    frag = ("## Guidelines:\n\n"
            "• The answer NA means that the paper has no limitation while the "
            "answer No means that the paper has limitations.\n")
    assert conv._section_kind(frag) == "boilerplate"
    assert conv._section_kind("## Results\n\nNA means not applicable here.\n") == ""


def test_a_document_that_is_all_apparatus_is_flagged_even_as_one_segment(tmp_vault, monkeypatch):
    """The single-segment fast path computed the section kind and then dropped
    it, so a standalone bibliography export, a scanned contents page or a lone
    venue checklist — one segment because it fits under _MAX_SEGMENT_CHARS —
    reached /nucleate unflagged and was distilled into exactly the
    venue/journal/ethics notes the flag exists to prevent."""
    refs = "## References\n\n" + "\n".join(
        f"[{i}] Author {i}. A Title {i}. Journal, 2020." for i in range(40))
    monkeypatch.setattr(conv, "_via_pymupdf", lambda src, wd: (refs, wd))
    tmp_vault.note("biblio.docx", "x")

    paths = conv.convert("biblio.docx")
    assert len(paths) == 1
    head = _inbox_note(paths[0]).read_text(encoding="utf-8").split("\n---\n")[0]
    assert "references: true" in head
    assert conv.is_skippable_chunk(paths[0])


def test_a_single_segment_of_real_content_stays_unflagged(tmp_vault, monkeypatch):
    body = "# Paper\n\n" + "Real prose about memory evolution. " * 200
    monkeypatch.setattr(conv, "_via_pymupdf", lambda src, wd: (body, wd))
    tmp_vault.note("paper.docx", "x")

    paths = conv.convert("paper.docx")
    assert len(paths) == 1
    head = _inbox_note(paths[0]).read_text(encoding="utf-8").split("\n---\n")[0]
    assert "references: true" not in head and "boilerplate: true" not in head
    assert not conv.is_skippable_chunk(paths[0])
