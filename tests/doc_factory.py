"""Hand-written test documents: PDF (text pages, outline, metadata), DOCX, EPUB, FB2.

The suite used to build its fixtures with pymupdf, which left an AGPL package in
the dev environment after the runtime dropped it (2026-08-31). A text-only PDF
is ~40 lines of the 1.4 syntax: one Helvetica font, one content stream per
page, an /Outlines tree when asked, an /Info dictionary when asked. Standard
14 fonts need no embedding and WinAnsi covers the Latin-1 the tests use, so
any reader recovers the text verbatim.
"""
from __future__ import annotations


def _pdf_string(text: str) -> bytes:
    """A PDF literal string: Latin-1 bytes with the three escapes the syntax needs."""
    raw = text.encode("latin-1", "replace")
    return b"(" + raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def pdf_bytes(
    pages: list[str],
    toc: list[tuple[int, str, int]] | None = None,
    creation_date: str | None = None,
    title: str | None = None,
    author: str | None = None,
    font_sizes: list[list[float]] | None = None,
) -> bytes:
    """A real one-column PDF, one page per string, lines separated by "\\n".

    `toc` entries are (level, title, 1-based page), flattened: every entry
    hangs off the document outline (level is recorded for readers that expose
    it, nesting is not built). `creation_date` is the PDF form, e.g.
    "D:20240402093000+02'00'". `font_sizes`, when given, holds one size per
    line of each page (default 11 pt), so a test can stage a heading heuristic.
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    n_pages = len(pages)
    # Object numbers are assigned up front so that pages can point at their
    # parent and the parent at its kids without a second pass.
    catalog_id, pages_id, font_id = 1, 2, 3
    first_page_id = 4
    content_ids = [first_page_id + n_pages + i for i in range(n_pages)]
    next_id = first_page_id + 2 * n_pages
    outlines_id = next_id if toc else None
    if toc:
        next_id += 1 + len(toc)
    info_id = next_id if (creation_date or title or author) else None

    add(b"<< /Type /Catalog /Pages 2 0 R"
        + (f" /Outlines {outlines_id} 0 R /PageMode /UseOutlines".encode() if toc else b"")
        + b" >>")
    kids = " ".join(f"{first_page_id + i} 0 R" for i in range(n_pages))
    add(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for i in range(n_pages):
        add(f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_ids[i]} 0 R >>".encode())
    for i, text in enumerate(pages):
        lines = text.split("\n")
        sizes = (font_sizes[i] if font_sizes else None) or [11.0] * len(lines)
        stream = b"BT\n"
        y = 720.0
        for line, size in zip(lines, sizes):
            stream += f"/F1 {size:g} Tf 1 0 0 1 72 {y:g} Tm ".encode() + _pdf_string(line) + b" Tj\n"
            y -= size * 1.6
        stream += b"ET"
        add(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    if toc:
        first, last = outlines_id + 1, outlines_id + len(toc)
        add(f"<< /Type /Outlines /First {first} 0 R /Last {last} 0 R /Count {len(toc)} >>".encode())
        for k, (level, heading, page) in enumerate(toc):
            oid = outlines_id + 1 + k
            entry = (f"<< /Title ".encode() + _pdf_string(heading)
                     + f" /Parent {outlines_id} 0 R "
                       f"/Dest [{first_page_id + page - 1} 0 R /XYZ 0 792 0]".encode())
            if k > 0:
                entry += f" /Prev {oid - 1} 0 R".encode()
            if k < len(toc) - 1:
                entry += f" /Next {oid + 1} 0 R".encode()
            # /Level is not standard; readers derive nesting from /First. Kept in
            # a private key so a test can still see what it asked for.
            entry += f" /SilicaLevel {level} >>".encode()
            add(entry)
    if info_id:
        info = b"<<"
        if title:
            info += b" /Title " + _pdf_string(title)
        if author:
            info += b" /Author " + _pdf_string(author)
        if creation_date:
            info += b" /CreationDate " + _pdf_string(creation_date)
        add(info + b" >>")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R"
    if info_id:
        trailer += f" /Info {info_id} 0 R"
    out += (trailer + f" >>\nstartxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


# --- DOCX ---------------------------------------------------------------------

_CT = ('<?xml version="1.0" encoding="UTF-8"?>'
       '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
       '<Default Extension="xml" ContentType="application/xml"/>'
       '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
       '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
       '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
       '</Types>')
# A bullet definition: mammoth emits <ul><li> only for paragraphs whose numPr
# resolves to a numbering instance, so a list needs this part and its relationship.
_DOC_RELS = ('<?xml version="1.0" encoding="UTF-8"?>'
             '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
             '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>'
             '</Relationships>')
_NUMBERING = ('<?xml version="1.0" encoding="UTF-8"?>'
              '<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
              '<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/><w:lvlText w:val="\u2022"/></w:lvl></w:abstractNum>'
              '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>'
              '</w:numbering>')
_RELS = ('<?xml version="1.0" encoding="UTF-8"?>'
         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
         '</Relationships>')
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def docx_bytes(
    blocks: list[tuple[str, str]],
    created: str | None = None,
    title: str | None = None,
    creator: str | None = None,
) -> bytes:
    """A minimal WordprocessingML package.

    `blocks` are (kind, text) with kind "h1", "h2", "p" or "li"; the heading
    kinds map to the built-in Heading1/Heading2 style ids mammoth recognises
    without a styles part. `created`/`title`/`creator` land in docProps/core.xml.
    """
    import io
    import zipfile
    from xml.sax.saxutils import escape

    body = []
    for kind, text in blocks:
        style = {"h1": "Heading1", "h2": "Heading2", "li": "ListParagraph"}.get(kind)
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        if kind == "li":
            ppr = '<w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>'
        body.append(f'<w:p>{ppr}<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>')
    document = (f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{_W}"><w:body>'
                + "".join(body) + "</w:body></w:document>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CT)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", _DOC_RELS)
        z.writestr("word/numbering.xml", _NUMBERING)
        if created or title or creator:
            z.writestr("docProps/core.xml", _core_xml(created, title, creator))
    return buf.getvalue()


def _core_xml(created: str | None, title: str | None, creator: str | None) -> str:
    from xml.sax.saxutils import escape
    parts = ['<?xml version="1.0" encoding="UTF-8"?>'
             '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
             'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
             'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">']
    if title:
        parts.append(f"<dc:title>{escape(title)}</dc:title>")
    if creator:
        parts.append(f"<dc:creator>{escape(creator)}</dc:creator>")
    if created:
        parts.append(f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>')
    parts.append("</cp:coreProperties>")
    return "".join(parts)


# --- EPUB ---------------------------------------------------------------------

def epub_bytes(
    chapters: list[tuple[str, str, list[str]]],
    title: str | None = None,
    creator: str | None = None,
    date: str | None = None,
    spine_order: list[int] | None = None,
) -> bytes:
    """An EPUB 2/3 container: mimetype, META-INF/container.xml, one OPF, one
    XHTML file per chapter. `chapters` are (id, heading, paragraphs). The spine
    lists chapters in `spine_order` (indices) when given, so a test can check
    that reading order follows the spine and not the file names.
    """
    import io
    import zipfile
    from xml.sax.saxutils import escape

    order = spine_order if spine_order is not None else list(range(len(chapters)))
    manifest = "".join(
        f'<item id="{cid}" href="{cid}.xhtml" media-type="application/xhtml+xml"/>'
        for cid, _, _ in chapters)
    spine = "".join(f'<itemref idref="{chapters[i][0]}"/>' for i in order)
    meta = ""
    if title:
        meta += f"<dc:title>{escape(title)}</dc:title>"
    if creator:
        meta += f"<dc:creator>{escape(creator)}</dc:creator>"
    if date:
        meta += f"<dc:date>{date}</dc:date>"
    opf = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">'
           f'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="uid">x</dc:identifier>{meta}</metadata>'
           f"<manifest>{manifest}</manifest><spine>{spine}</spine></package>")
    container = ('<?xml version="1.0" encoding="UTF-8"?>'
                 '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                 '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                 "</rootfiles></container>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", container)
        z.writestr("OEBPS/content.opf", opf)
        for cid, heading, paragraphs in chapters:
            ps = "".join(f"<p>{escape(p)}</p>" for p in paragraphs)
            z.writestr(f"OEBPS/{cid}.xhtml",
                       '<?xml version="1.0" encoding="UTF-8"?><html xmlns="http://www.w3.org/1999/xhtml">'
                       f"<head><title>{escape(heading)}</title><style>p{{margin:0}}</style></head>"
                       f"<body><h1>{escape(heading)}</h1>{ps}</body></html>")
    return buf.getvalue()


# --- FB2 ----------------------------------------------------------------------

def fb2_bytes(sections: list[tuple[str, list[str]]], title: str | None = None) -> bytes:
    """A FictionBook 2 document: `sections` are (title, paragraphs)."""
    from xml.sax.saxutils import escape

    body = "".join(
        f"<section><title><p>{escape(t)}</p></title>" + "".join(f"<p>{escape(p)}</p>" for p in ps) + "</section>"
        for t, ps in sections)
    ti = f"<title-info><book-title>{escape(title)}</book-title></title-info>" if title else ""
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
            f"<description>{ti}</description><body>{body}</body></FictionBook>").encode("utf-8")
