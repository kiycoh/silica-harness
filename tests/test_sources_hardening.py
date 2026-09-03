"""Hardening of the document-ingest frontier (`silica/sources/convert.py`).

Everything here is adversarial: the inputs are the ones a hostile file dropped
into a watched vault folder would carry, not the ones a word processor writes.
Four guards, four failure modes they close:

  * a zip member that *declares* gigabytes (`_MAX_ZIP_MEMBER`),
  * XML that carries a DOCTYPE, the entry point for entity expansion,
  * a filename whose stem is a valid glob character class,
  * an uppercase legacy suffix, which used to be told to re-save a Word
    document as a PowerPoint one.

No fixture file on disk is malicious: every archive below is assembled in the
test, so the repo never ships a bomb. Nothing shells out and nothing reaches
the network — `mineru`/`soffice` are stubbed at the seam.
"""
from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pytest

from silica.config import CONFIG
from silica.sources import convert as conv


def _inbox_note(note_rel: str) -> Path:
    return Path(CONFIG.vault_path) / note_rel


_ODF_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    "<office:document-content "
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
    "<office:body><office:text>"
)
_ODF_TAIL = "</office:text></office:body></office:document-content>"


def _odf_zip(content_xml: str, *, extra: dict[str, str] | None = None) -> bytes:
    """An ODF-shaped archive: mimetype + content.xml, like the suites write."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        z.writestr("content.xml", content_xml)
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


def _forge_declared_size(raw: bytes, member: str, size: int) -> bytes:
    """Rewrite one member's declared uncompressed size in the central directory.

    A real 64 MB+ bomb would cost the suite a 64 MB buffer to prove a guard that
    is supposed to fire *before* anything is read. Forging the declaration is
    the stronger test as well as the cheaper one: the member's real bytes stay a
    few hundred, and stay valid, so a guard that measured the inflated data
    instead of trusting the header would let the file through and fail the test.

    Central-directory header layout (APPNOTE 4.3.12): signature at +0, the
    uncompressed size at +24, the filename length at +28, the name at +46.
    """
    b = bytearray(raw)
    want = member.encode()
    i = 0
    while (i := b.find(b"PK\x01\x02", i)) != -1:
        name_len = struct.unpack_from("<H", b, i + 28)[0]
        if bytes(b[i + 46 : i + 46 + name_len]) == want:
            struct.pack_into("<I", b, i + 24, size)
            return bytes(b)
        i += 4
    raise AssertionError(f"no central-directory entry for {member}")


# --- guard 1: a zip member that declares more than the ceiling --------------

def test_a_normal_odf_still_parses(tmp_vault, monkeypatch):
    """The ceiling must not cost the ordinary document anything."""
    monkeypatch.setattr(conv.subprocess, "run", _never_shells_out)
    (Path(CONFIG.vault_path) / "notes.odt").write_bytes(_odf_zip(
        _ODF_HEAD
        + '<text:h text:outline-level="1">Chapter</text:h>'
        + "<text:p>Ordinary body.</text:p>"
        + _ODF_TAIL
    ))

    body = _inbox_note(conv.convert("notes.odt")[0]).read_text(encoding="utf-8")
    assert "# Chapter" in body
    assert "Ordinary body." in body


def test_oversized_member_is_refused_before_it_is_read(tmp_vault, monkeypatch):
    """A 200 KB archive can declare a member that inflates to gigabytes, and
    `z.read()` would have allocated all of it. The declaration is enough to
    refuse: nothing about the member is decompressed."""
    monkeypatch.setattr(conv.subprocess, "run", _never_shells_out)
    raw = _odf_zip(_ODF_HEAD + "<text:p>Harmless.</text:p>" + _ODF_TAIL)
    bomb = Path(CONFIG.vault_path) / "bomb.odt"
    bomb.write_bytes(_forge_declared_size(raw, "content.xml", 900 * 1024 * 1024))

    # The file on disk is smaller than a single page: the only thing oversized
    # here is the number in the header, which is exactly what a bomb is.
    assert bomb.stat().st_size < 4096
    with pytest.raises(ValueError, match="decompression bomb"):
        conv.convert("bomb.odt")


def test_the_refusal_names_the_member_and_the_ceiling(tmp_vault):
    """The user has to be able to tell this from a corrupt file."""
    raw = _forge_declared_size(
        _odf_zip("<a/>"), "content.xml", conv._MAX_ZIP_MEMBER + 1
    )
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with pytest.raises(ValueError, match=r"content\.xml declares more than 64 MB"):
            conv._zip_member(z, "content.xml")


def test_a_member_exactly_at_the_ceiling_is_still_read(monkeypatch):
    """The bound is a ceiling, not a budget: a member the size of the limit is
    read, only one past it is refused. Asserted with the limit lowered to the
    payload rather than the payload raised to 64 MB."""
    payload = "<a/>" * 8
    monkeypatch.setattr(conv, "_MAX_ZIP_MEMBER", len(payload))
    with zipfile.ZipFile(io.BytesIO(_odf_zip(payload))) as z:
        assert conv._zip_member(z, "content.xml") == payload.encode("utf-8")

    monkeypatch.setattr(conv, "_MAX_ZIP_MEMBER", len(payload) - 1)
    with zipfile.ZipFile(io.BytesIO(_odf_zip(payload))) as z:
        with pytest.raises(ValueError, match="decompression bomb"):
            conv._zip_member(z, "content.xml")


def test_source_date_does_not_read_an_oversized_core_xml(tmp_vault):
    """`_source_date` reads `docProps/core.xml` out of an OOXML zip, and it
    swallows every exception (a missing date is never worth failing a
    conversion over). That blanket except is what makes the ceiling load
    bearing here: without it the OOM happens inside the try, before anything
    can be swallowed. The date below is real and readable — only the declared
    size is not — so an unguarded read would return it."""
    raw = _odf_zip("<a/>", extra={"docProps/core.xml": (
        '<coreProperties xmlns:dcterms="d"><dcterms:created>'
        "2023-11-20T09:00:00Z</dcterms:created></coreProperties>"
    )})
    deck = Path(CONFIG.vault_path) / "deck.pptx"
    deck.write_bytes(_forge_declared_size(
        raw, "docProps/core.xml", conv._MAX_ZIP_MEMBER + 1
    ))

    assert conv._source_date(deck) is None


# --- guard 2: XML that declares a DOCTYPE -----------------------------------

# Three nested entities, not the billion the attack is named for: if the guard
# is ever removed this file must fail by parsing the document, not by taking
# the machine down with it.
_LAUGHS = (
    '<?xml version="1.0"?>'
    "<!DOCTYPE lolz ["
    '  <!ENTITY lol "lol">'
    '  <!ENTITY lol1 "&lol;&lol;&lol;">'
    '  <!ENTITY lol2 "&lol1;&lol1;&lol1;">'
    "]>"
    "<office:document-content "
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
    "<office:body><office:text><text:p>&lol2;</text:p></office:text>"
    "</office:body></office:document-content>"
)


def test_a_doctype_is_refused_at_the_prolog():
    """expat calls the handler at the start of the declaration, so the refusal
    lands before a single entity is expanded."""
    with pytest.raises(ValueError, match="DOCTYPE"):
        conv._parse_office_xml(_LAUGHS.encode("utf-8"))


def test_a_doctype_inside_an_odf_document_refuses_the_conversion(
    tmp_vault, monkeypatch
):
    """No office format writes a DOCTYPE, so refusing one costs no real file."""
    monkeypatch.setattr(conv.subprocess, "run", _never_shells_out)
    (Path(CONFIG.vault_path) / "laughs.odt").write_bytes(_odf_zip(_LAUGHS))

    with pytest.raises(ValueError, match="DOCTYPE"):
        conv.convert("laughs.odt")


def test_the_doctype_refusal_is_not_flattened_into_a_generic_error(
    tmp_vault, monkeypatch
):
    """`_via_odf` turns BadZipFile/KeyError/ParseError into "not a readable ODF
    document — re-save it". A DOCTYPE must not land in that bucket: the file is
    perfectly readable and re-saving it would change nothing."""
    monkeypatch.setattr(conv.subprocess, "run", _never_shells_out)
    (Path(CONFIG.vault_path) / "laughs.odt").write_bytes(_odf_zip(_LAUGHS))

    with pytest.raises(ValueError) as e:
        conv.convert("laughs.odt")
    assert "not a readable ODF" not in str(e.value)


def test_xml_without_a_doctype_parses_normally():
    """The guard refuses the prolog, not the parser."""
    root = conv._parse_office_xml(
        (_ODF_HEAD + "<text:p>Body text.</text:p>" + _ODF_TAIL).encode("utf-8")
    )
    assert "".join(root.itertext()) == "Body text."


# --- guard 3: a filename stem that is a valid glob character class ----------

def _fake_mineru():
    """Stand in for the mineru CLI: write the markdown it would have produced,
    under the `<stem>/auto/<stem>.md` layout the real one uses for a PDF."""
    def run(cmd, **kw):
        assert cmd[0] == "mineru"
        out = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-p") + 1]).stem
        d = out / stem / "auto"
        d.mkdir(parents=True)
        (d / f"{stem}.md").write_text("# Recovered\n\nOCR body.\n", encoding="utf-8")

        class R:
            returncode, stderr, stdout = 0, "", ""

        return R()

    return run


@pytest.mark.parametrize("stem", ["Smith [2020]", "Notes *final*", "draft?"])
def test_glob_metacharacters_in_a_stem_survive_the_round_trip(
    stem, tmp_path, monkeypatch
):
    """Brackets are the case that bites: `Smith [2020]` is a character class, so
    unescaped `[2020]` matches one character out of {2,0} and the pattern
    matches nothing at all. The markdown would be thrown away with the tempdir
    after an hour of OCR and the user told mineru produced nothing. `*` and `?`
    happen to match themselves here, and are pinned so a future rewrite of the
    pattern cannot over-escape its way to the same silence."""
    monkeypatch.setattr(conv.subprocess, "run", _fake_mineru())
    src = tmp_path / f"{stem}.pdf"
    src.write_bytes(b"%PDF-1.4\n")

    md, _images = conv._pdf_via_mineru(src, tmp_path / "work")
    assert "OCR body." in md


def test_a_genuinely_empty_mineru_run_still_reports_it(tmp_path, monkeypatch):
    """Escaping the stem must not turn "produced nothing" into a silent pass."""
    def run(cmd, **kw):
        class R:
            returncode, stderr, stdout = 0, "", ""

        return R()

    monkeypatch.setattr(conv.subprocess, "run", run)
    src = tmp_path / "Smith [2020].pdf"
    src.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(ValueError, match="produced no markdown"):
        conv._pdf_via_mineru(src, tmp_path / "work")


# --- guard 4: the legacy suffix is matched case-insensitively ---------------

def _no_office_suite(monkeypatch):
    monkeypatch.setattr(conv, "soffice_bin", lambda: None)


def _apache_openoffice(monkeypatch):
    monkeypatch.setattr(conv, "soffice_bin", lambda: "/usr/bin/soffice")
    monkeypatch.setattr(
        conv, "soffice_flavour",
        lambda exe=None: ("openoffice", "Apache OpenOffice 4.1.16"),
    )


@pytest.mark.parametrize("name", ["Report.doc", "Report.DOC", "Report.Doc"])
def test_a_word_document_is_told_to_re_save_as_docx(name, tmp_path, monkeypatch):
    """Dispatch lowercases the suffix, so `Report.DOC` reaches this branch — and
    was being told to re-save a Word document as `.pptx`."""
    _no_office_suite(monkeypatch)
    with pytest.raises(ValueError) as e:
        conv._legacy_office_to_pdf(tmp_path / name, tmp_path)
    assert ".docx" in str(e.value)
    assert ".pptx" not in str(e.value)


@pytest.mark.parametrize("name", ["Deck.ppt", "Deck.PPT"])
def test_a_presentation_is_told_to_re_save_as_pptx(name, tmp_path, monkeypatch):
    """The other half of the branch: fixing the `.doc` case must not swap the
    advice for the format that was already right."""
    _no_office_suite(monkeypatch)
    with pytest.raises(ValueError) as e:
        conv._legacy_office_to_pdf(tmp_path / name, tmp_path)
    assert ".pptx" in str(e.value)
    assert ".docx" not in str(e.value)


@pytest.mark.parametrize(
    ("name", "want", "wrong"),
    [("Report.DOC", ".docx", ".pptx"), ("Deck.PPT", ".pptx", ".docx")],
)
def test_the_openoffice_refusal_names_the_same_format(
    name, want, wrong, tmp_path, monkeypatch
):
    """The AOO branch quotes the same escape hatch and must not contradict the
    no-suite one — it is reached on the machines that have the wrong `soffice`
    on PATH, which is where the advice matters most."""
    _apache_openoffice(monkeypatch)
    with pytest.raises(ValueError) as e:
        conv._legacy_office_to_pdf(tmp_path / name, tmp_path)
    assert want in str(e.value)
    assert wrong not in str(e.value)


def test_an_uppercase_legacy_suffix_reaches_the_legacy_branch(
    tmp_vault, monkeypatch
):
    """The case-insensitive message is only worth anything if dispatch routes
    `Report.DOC` here at all rather than at a document reader that does not open it."""
    _no_office_suite(monkeypatch)
    (Path(CONFIG.vault_path) / "Report.DOC").write_bytes(b"\xd0\xcf\x11\xe0")

    with pytest.raises(ValueError, match=r"needs LibreOffice.*\.docx"):
        conv.convert("Report.DOC")


def _never_shells_out(*a, **k):
    """These lanes are pure python by design; a subprocess here is the defect."""
    raise AssertionError("no subprocess may be spawned on this path")
