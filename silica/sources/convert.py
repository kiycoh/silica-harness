# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Non-`.md` → `.md` conversion — ingress frontier (ADR-0009).

A plain function, not a `SourceAdapter`: `/convert` exposes it and `/nucleate`
calls it as the fallback when no source adapter claims a file. Dispatch is by
extension over `DOC_EXTS`, three families with three different backends:

  * PDF — the selectable provider seam below.
  * DOCX (mammoth → HTML), EPUB (a ZIP of XHTML) and FB2 (XML)
    (`_BASE_TEXT_EXTS`) — read in process through one HTML/XML-to-markdown
    pass, no office suite and no torch. XPS and MOBI, which only MuPDF opened,
    are no longer accepted: declared unsupported rather than half-read.
  * Images (`IMG_EXTS`) and OOXML decks/sheets (`OFFICE_EXTS`) — mineru, always,
    because it is the only backend that opens them at all. Images take mineru's
    OCR pipeline (it wraps them into a one-page PDF); pptx/xlsx take its native
    python-pptx/openpyxl reader, no LibreOffice involved.
  * ODF, RTF and legacy `.xls` (`_PURE_PY_OFFICE_EXTS`) — read in process, with
    no office suite anywhere: an ODF file is a ZIP the standard library opens,
    and striprtf + xlrd together weigh ~200 KB.
  * Legacy binary `.doc`/`.ppt` (`LEGACY_OFFICE_EXTS`) — the only two formats
    left that need LibreOffice. It converts to PDF, which re-enters the provider
    seam above. Hardened rather than trusted: own profile, hard timeout, doctor
    probe (`probe_soffice`).
  * Audio and video (`MEDIA_EXTS`) — transcribed, not parsed: ffmpeg demuxes to
    16 kHz mono wav and `ASR_PROVIDERS[CONFIG.stt_provider]` turns that into
    text. Every provider returns markdown into the same shared tail, so a talk
    gets the same sanitizing, segmentation and provenance a book gets.

For PDF the converter is selectable via `CONFIG.pdf_provider` (ADR-0011):
`pdfium` default (pypdfium2, Google's PDFium, one ~3 MB wheel, no torch and no
JVM, text layer only so no OCR), `mineru` (heavyweight CLI, best fidelity and
the only OCR path, downloads models on first run), `docling` (MIT but pulls
torch + CUDA), `opendataloader` (Apache-2.0, strong on complex tables and
multi-column reading order, needs a JVM). The non-PDF formats bypass the seam:
docling/opendataloader/mineru take a PDF and nothing else. `pypdfium2` is a
base dependency; the other three are binaries the user installs, and Silica
shells out to them exactly as it shells out to soffice, ffmpeg and whisper-cli.
None of them was ever imported here, which is why the `[pdf]` extra that used to
install mineru was dropped on 2026-09-02 (`silica doctor` reports its presence).

`pdfium` replaced `pymupdf4llm` as the default on 2026-08-31 (ADR-0034). PyMuPDF is AGPL
or a paid Artifex licence, and the AGPL is what blocked any proprietary
redistribution of the engine; pymupdf4llm had also started to hard-depend on a
Polyform Noncommercial layout package. Measured on 12 real papers: pdfium reads
1.05x the words in 1/75 of the time (0.4 s against 30 s), and the line-based
heading heuristic below flags the References section on 11 of them where the
font-size guessing of pymupdf4llm flagged 0. What it does not do: extract
figures (mineru does), or read a scan (nothing without OCR does).

Every provider returns `(markdown, images_dir)`; the rest of the pipeline
(sanitize → copy images flat into the vault → rewrite image links to Obsidian
embeds → write the note to the inbox) is shared and provider-agnostic.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Sequence
from glob import escape as glob_escape, glob
from html.parser import HTMLParser
from pathlib import Path

from silica.config import CONFIG
from silica.kernel.text.sanitize import strip_degenerate_runs

logger = logging.getLogger(__name__)


def _reap_with_parent_factory():
    """`preexec_fn` that makes the kernel kill this child with its parent.

    The converters here are long-running and hold real resources — mineru
    keeps the GPU for a minute or more per book. `subprocess.run` reaps them on
    timeout and on error, but nothing reaps them when Silica's own process is
    killed: the child is reparented to init and runs on (observed 2026-08-16 on
    a 75-PDF batch, mineru left holding the card and cleaned by hand).
    PR_SET_PDEATHSIG hands that job to the kernel, which is the only party
    still alive to do it.

    The signal fires when the forking *thread* exits, not the process — safe
    only because every caller below is a blocking `subprocess.run` on that same
    thread, so the thread cannot outlive the child. Do not reuse this for a
    Popen the caller keeps and walks away from.

    ponytail: Linux-only. Elsewhere it stays None and the child still outlives
    a hard kill; the portable fix is a supervisor process, which is a lot of
    machinery for a case a `pkill mineru` already covers.
    """
    if sys.platform != "linux":
        return None
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError as e:  # no glibc (musl images), no prctl binding
        logger.debug("PR_SET_PDEATHSIG unavailable (%s)", e)
        return None
    _PR_SET_PDEATHSIG = 1

    def _reap_with_parent() -> None:
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)

    return _reap_with_parent


_REAP_WITH_PARENT = _reap_with_parent_factory()

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

# Book segmentation — a converted book is one giant markdown, but RECON caps
# concepts PER FILE (keyphrase.MAX_CONCEPTS=40), so a whole book in one inbox
# note loses almost everything. Split on chapter headings, then size-cap each
# section so RECON sees book-sized units. ~40k chars ≈ 10k tokens ≈ ~15 pages:
# raise for fewer/larger files, lower for more granular notes.
_MAX_SEGMENT_CHARS = 40_000
_HEADING_RE = re.compile(r"^#{1,2} \S")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# MinerU knobs — ponytail: module constants. First run downloads models, so the
# timeout is generous; switch to a VLM/hybrid backend or raise the timeout here.
# Measured ~0.9 s/page on CPU (80-page probe): 600s died on an 800-page book.
_MINERU_BACKEND = "pipeline"
_MINERU_TIMEOUT_S = 3600
# Maximum-precision non-generative pins (today's upstream defaults, pinned
# against drift — the upstream default backend already drifted to a VLM):
# -m auto (parse method), -f true (formula parsing), -t true (table parsing).
# No -l: mineru 3.4.4 has no latin-script choice (ch|ch_server|korean|...) and
# the default `ch` OCR models cover latin script.
#
# These flags are written against 3.4.4 and nothing pins that any more: mineru
# stopped being a dependency on 2026-09-02, so the version on PATH is whatever
# the user installed. `_mineru_error` is the place that reads the damage — an
# unknown flag comes back as the error line, not as a silent empty convert.
_MINERU_ARGS = ["-m", "auto", "-f", "true", "-t", "true"]

# stderr triage (see _mineru_error): noise = loguru INFO/DEBUG, uvicorn banner
# lines, tqdm progress bars; error-ish = a line naming an error/exception.
_MINERU_NOISE_RE = re.compile(
    r"\|\s*(?:INFO|DEBUG)\s*\||^(?:INFO|DEBUG|WARNING):|it/s|\d+%\|", re.IGNORECASE
)
_MINERU_ERR_RE = re.compile(r"error|exception|traceback", re.IGNORECASE)
# The one failure whose fix is not in mineru's own message. Its vendored
# pytorchocr imports `six` without declaring it (3.4.4,
# pytorchocr/data/imaug/operators.py), so a pipeline convert dies with a
# ModuleNotFoundError naming a dependency of a dependency. The [pdf] extra used
# to carry the workaround as a `six` line and it was gone by install time; since
# 2026-09-02 there is no extra, so the user meets this after an hour of OCR
# instead. Drop the branch when upstream declares the import.
_MINERU_SIX_RE = re.compile(r"No module named ['\"]six['\"]")


# Text documents read in process with no PDF renderer in the path: DOCX through
# mammoth, EPUB (a ZIP of XHTML) and FB2 (XML) through the standard library.
# `.txt`/`.md` are absent on purpose: ProseAdapter already claims them.
_BASE_TEXT_EXTS = (".docx", ".epub", ".fb2")

# Image formats mineru opens (its own `image_suffixes`). It wraps the image into
# a one-page PDF internally, so a screenshot or a scan takes the same OCR
# pipeline a scanned PDF takes. Both `.tif` and `.tiff` land because mineru
# sniffs content rather than trusting the extension (measured: `.tif` -> "tiff",
# `.jpg` -> "jpeg"). `.heic` is absent: mineru does not list it, and reading it
# at all needs pillow-heif.
IMG_EXTS = (".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp", ".tif", ".tiff")

# OOXML mineru parses NATIVELY, via python-pptx / openpyxl, with no LibreOffice
# anywhere in the path. Output lands in `<stem>/office/<stem>.md` instead of
# `auto/`, which the provider's recursive glob already finds; slide titles come
# out as `##` headings, which is what `_split_on_headings` wants.
#
# `.docx` is deliberately NOT here: mammoth reads it in the base install, so
# routing it through mineru would demand an OCR install for a format that
# already works. The pre-2007 binaries are not here either -- neither mineru nor
# MuPDF opens them (see `_PURE_PY_OFFICE_EXTS` and `LEGACY_OFFICE_EXTS`).
OFFICE_EXTS = (".pptx", ".xlsx")

# Inputs no provider but mineru opens, so the provider seam does not apply.
_MINERU_ONLY_EXTS = (*IMG_EXTS, *OFFICE_EXTS)

# ODF is a ZIP with a `content.xml` inside, so reading it needs no office suite
# and no dependency at all — `zipfile` plus `xml.etree`. RTF and legacy `.xls`
# need one small pure-python parser each (striprtf 15 KB BSD-3, xlrd 192 KB
# BSD). Together that is ~200 KB against the ~240 MB minimum a headless
# LibreOffice install costs on Debian (measured: libreoffice-core 155 MB +
# -common 47 MB + one app), for five of the seven formats that used to demand it.
#
# ponytail: text only. The old `soffice → pdf → MuPDF` path carried embedded
# figures through; these do not. Re-save as PDF when the images are the point.
ODF_EXTS = (".odt", ".odp", ".ods")
_PURE_PY_OFFICE_EXTS = (*ODF_EXTS, ".rtf", ".xls")

# What is left after the above: the two pre-2007 binary formats with no pure
# python reader worth carrying. Apache POI reads both and is Apache-2.0, but it
# is Java — trading a 240 MB office suite for a JVM is not a trade. So these two
# keep the LibreOffice hop, and when it is absent they are refused with a
# message naming the OOXML escape rather than converted badly.
#
# The hop itself is ~15 lines. Everything else here is because `soffice` names
# two different programs, and the wrong one takes the whole terminal hostage.
# Measured on this developer's machine (`/usr/bin/soffice` →
# `/opt/openoffice4/program/soffice`, Apache OpenOffice 4.1.16):
#
#   * `--convert-to` (double dash) is not an AOO option, and the string
#     "convert-to" appears NOWHERE in the 4.1.16 install: AOO never implemented
#     headless conversion. unoconv exists precisely because of that.
#   * Handed an option it does not know, AOO does not fail — it starts in GUI
#     mode and opens its first-start wizard. That is the "hang": 0.36 s of CPU
#     in 120 s of wall clock, waiting on a dialog nobody asked for.
#   * With single-dash flags and `-nofirststartwizard`, the same binary starts
#     and exits in 0.8 s, exit 0, no window.
#
# So: single-dash flags (LibreOffice accepts them too), the wizard suppressed,
# and AOO refused BEFORE the subprocess rather than after — attempting it puts a
# dialog on the user's screen, which no timeout can take back.
LEGACY_OFFICE_EXTS = (".ppt", ".doc")
# Single dash on purpose (see above). `-invisible`/`-nodefault`/`-norestore`
# keep it from opening a blank document or restoring a crashed session;
# `-nolockcheck` stops it consulting a profile lock we deliberately bypass.
_SOFFICE_QUIET = (
    "-headless", "-invisible", "-nofirststartwizard",
    "-nolockcheck", "-nodefault", "-norestore",
)
_SOFFICE_TIMEOUT_S = 300
# The probe only has to prove the binary starts and exits, so it gets a much
# shorter leash than a real conversion. Measured at 0.8 s on a cold profile.
_SOFFICE_PROBE_TIMEOUT_S = 20
# AOO with no JRE prints this to stderr and still exits 0, so it must not be
# mistaken for the cause when a conversion really does fail.
_SOFFICE_NOISE = ("javaldx",)

# Media: transcribed, not parsed. Both families take the same path (ffmpeg
# demuxes any container to 16 kHz mono wav, which is the one input every ASR
# accepts), so video costs nothing beyond `-vn`. The list is what ffmpeg reads
# in a default build, not everything it can be built for.
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma")
VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mpg", ".mpeg")
MEDIA_EXTS = (*AUDIO_EXTS, *VIDEO_EXTS)

# Data files: never prose-converted row by row. A .csv in the inbox becomes a
# PROFILE note — schema, per-column stats, a 5-row sample — that makes the file
# discoverable by recall while the rows stay on disk for `silica_query_table`
# to aggregate in place. json/ndjson stay out: their shape is unknowable up
# front (config? API dump? records?), so profiling them would guess.
TABULAR_EXTS = (".csv", ".tsv", ".parquet")

DOC_EXTS = (
    ".pdf", *_BASE_TEXT_EXTS, *IMG_EXTS, *OFFICE_EXTS, *MEDIA_EXTS,
    *_PURE_PY_OFFICE_EXTS, *LEGACY_OFFICE_EXTS, *TABULAR_EXTS,
)

# The subset that is a *document* someone meant to keep, as opposed to an
# attachment that happens to be convertible. Onboarding offers to convert what
# it finds sitting in a vault, and by that measure an image or a video is almost
# always an attachment: every Obsidian vault carries pasted screenshots, and
# `_copy_images` puts Silica's OWN extracted figures in `<inbox>/Images`. Offering
# to convert those would mean offering to re-ingest our own output.
CONVERTIBLE_DOC_EXTS = (
    ".pdf", *_BASE_TEXT_EXTS, *OFFICE_EXTS, *_PURE_PY_OFFICE_EXTS,
    *LEGACY_OFFICE_EXTS, *TABULAR_EXTS,
)

# ffmpeg knobs. `-nostdin` is not cosmetic: ffmpeg reads stdin by default and
# would swallow the TUI's keystrokes (or block) when run under a prompt.
_FFMPEG_ARGS = ("-nostdin", "-loglevel", "error", "-y")
_ASR_SAMPLE_RATE = "16000"
# Generous: transcribing is minutes of audio, not one page. omniparse's
# equivalent subprocess call has no timeout at all, so a stuck decoder there
# hangs the request forever.
_FFMPEG_TIMEOUT_S = 1800
_ASR_TIMEOUT_S = 7200
# A pause this long ends a paragraph. Transcripts otherwise carry no blank line
# at all, and `_split_by_size` leaves an oversized single paragraph whole.
_ASR_PARAGRAPH_GAP_S = 2.0


def convert(target: str, dest_dir: str = "") -> list[str]:
    """Convert a non-`.md` document into one or more `.md` notes in the inbox.

    Returns the list of created note paths. A small document is a single note; a
    book-sized one is split into chapter/size-bounded segments (see
    ``split_markdown``) so RECON — which caps concepts PER FILE — sees book
    units, not the whole book collapsed into one note. Dispatch by extension
    over ``DOC_EXTS``; anything else → ``ValueError``. Side artifacts (extracted
    figures) go to ``<dest_dir>/Images`` when given, else ``<inbox>/Images``.
    """
    # Strip first: a quoted path with a stray trailing space ("…book.pdf ") has
    # suffix ".pdf " — not in DOC_EXTS — and the rejection then prints ".pdf",
    # a message the user cannot tell from a real unsupported type.
    target = target.strip()
    if Path(target).suffix.lower() not in DOC_EXTS:
        raise ValueError(f"no converter for {Path(target).suffix.lower() or 'this file type'}")
    return _doc_to_md(target, dest_dir)


def _split_on_headings(md: str) -> list[str]:
    """Split markdown at level-1/2 headings (fence-aware). Always ≥1 segment.

    Content before the first heading stays attached to it (no empty lead
    segment). A ``#``/``##`` inside a fenced code block is not a boundary.
    """
    segs: list[str] = []
    cur: list[str] = []
    in_fence = False
    for line in md.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and _HEADING_RE.match(line) and "".join(cur).strip():
            segs.append("".join(cur))
            cur = []
        cur.append(line)
    if "".join(cur).strip():
        segs.append("".join(cur))
    return segs or [md]


def _split_by_size(text: str, max_chars: int) -> list[str]:
    """Greedy split on blank-line (paragraph) boundaries, ≤ max_chars per part.

    A single paragraph larger than max_chars is left whole (its own oversized
    part) rather than cut mid-sentence — vanishingly rare in prose.
    """
    segs: list[str] = []
    cur = ""
    for part in re.split(r"(\n[ \t]*\n)", text):
        if cur and len(cur) + len(part) > max_chars:
            segs.append(cur)
            cur = ""
        cur += part
    if cur.strip():
        segs.append(cur)
    return segs or [text]


# A section whose first heading is a bibliography heading. Reference lists are
# citation metadata, not content to nucleate: one survey's references produced
# 34 venue notes (NeurIPS.md, ICML.md, …) out of 98 total (2026-08-15 run).
_REFERENCES_HEADING_RE = re.compile(
    r"^#{1,6}\s*[\d.\s]*\**\s*(references?|bibliography|works cited|bibliografia|"
    r"riferimenti(?:\s+bibliografici)?)\s*\**\s*$",
    re.IGNORECASE,
)

# A table of contents: page-number scaffolding, and the autolinker wikilinks
# every entry ("2.1 [[Memory for LLM Agents]] . 2").
_CONTENTS_HEADING_RE = re.compile(
    r"^#{1,6}\s*[\d.\s]*\**\s*(table of contents|contents|indice)\s*\**\s*$",
    re.IGNORECASE,
)

# Venue submission checklists (NeurIPS, ICML, ACL) carry a fixed
# Question/Answer/Justification triple per item. Detected on the BODY, not the
# heading: the splitter cuts one segment per item ("## 5. Open access to data
# and code"), so no shared heading marker survives. Demanding both markers is
# what keeps prose that merely says "Answer:" out of it. Measured on
# a-mem-agentic-memory (2026-08-18): the checklist alone produced 14 notes —
# Code of ethics, IRB Approval, Crowdsourcing — identical across every NeurIPS
# paper in the library.
_CHECKLIST_ANSWER_RE = re.compile(
    r"^\s*(?:\*\*)?Answer(?:\*\*)?:\s*\[\s*(?:yes|no|na|n/a)\s*\]",
    re.IGNORECASE | re.MULTILINE,
)
_CHECKLIST_JUSTIFICATION_RE = re.compile(
    r"^\s*(?:\*\*)?Justification(?:\*\*)?:", re.IGNORECASE | re.MULTILINE
)
# How far into a section the checklist triple must appear. One item is heading
# + Question + Answer + Justification, ~300 chars; anything further in is a
# paper that quotes a checklist, not a checklist.
_CHECKLIST_HEAD_CHARS = 700
# The template's own boilerplate, verbatim in every item's Guidelines block of
# every NeurIPS submission. Catches the `## Guidelines:` fragments the
# Question/Answer/Justification rule cannot see, because they carry neither.
_CHECKLIST_GUIDELINES_RE = re.compile(r"answer\s+NA\s+means", re.IGNORECASE)
_CHECKLIST_HEADING_RE = re.compile(
    r"^#{1,6}\s*[\d.\s]*\**\s*(?:neurips|icml|iclr|acl|aaai|cvpr)?\s*"
    r"paper checklist\s*\**\s*$",
    re.IGNORECASE,
)


def is_skippable_chunk(note_rel: str) -> bool:
    """True when a converted inbox chunk is flagged as non-content.

    `boilerplate: true` is the current key; `references: true` is the original
    one and stays recognized forever — inbox segments from earlier runs carry
    it, and re-converting is not something a vault owner should have to do.

    The /nucleate side of the flag written by `_doc_to_md`. Unreadable or
    unflagged files answer False — never blocks ingestion on a read error.
    """
    from silica.driver import DRIVER
    from silica.kernel.write import frontmatter

    try:
        data, _, _ = frontmatter.split(DRIVER.read_note(note_rel).content)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    return bool(data.get("boilerplate") or data.get("references"))


def _section_kind(section: str) -> str:
    """Empty for content, else why this section must not be distilled.

    The kind doubles as the frontmatter key stamped on the segment
    (`references: true` / `boilerplate: true`), so the flag says which class of
    apparatus it is instead of calling a table of contents a bibliography.
    """
    for line in section.splitlines():
        if _HEADING_RE.match(line):
            head = line.strip()
            if _REFERENCES_HEADING_RE.match(head):
                return "references"
            if _CONTENTS_HEADING_RE.match(head) or _CHECKLIST_HEADING_RE.match(head):
                return "boilerplate"
            break
    # HEAD only, never the whole section: with sparse headings a "section" is
    # the entire document, and a paper that merely CONTAINS the checklist at
    # the back would flag itself end to end (measured: a-mem-agentic-memory
    # came out 100% boilerplate on the unwindowed rule). A real checklist item
    # opens with its triple, so the head is where it must be found.
    head = section[:_CHECKLIST_HEAD_CHARS]
    if _CHECKLIST_ANSWER_RE.search(head) and _CHECKLIST_JUSTIFICATION_RE.search(head):
        return "boilerplate"
    if _CHECKLIST_GUIDELINES_RE.search(head):
        return "boilerplate"
    # ponytail: the template's one-off "IMPORTANT, please:" instruction block
    # (282 B, once per paper) is left in. A third regex for a quarter of a
    # kilobyte is not worth owning; widen this if it ever becomes a note.
    return ""


def _split_markdown_flagged(md: str, max_chars: int) -> list[tuple[str, str]]:
    """split_markdown's engine, each segment carrying its `_section_kind`.

    The kind is per SECTION (heading-level), inherited by its size-split
    continuation pieces, and packing never merges across a kind change — so a
    references section and its heading-less overflow parts all come out
    flagged, and never absorb (or get absorbed by) real content.
    """
    pieces: list[tuple[str, str]] = []
    for section in _split_on_headings(md):
        kind = _section_kind(section)
        if len(section) <= max_chars:
            pieces.append((section, kind))
        else:
            pieces.extend((p, kind) for p in _split_by_size(section, max_chars))

    out: list[tuple[str, str]] = []
    cur, cur_kind = "", ""
    for p, kind in pieces:
        if cur and (len(cur) + len(p) > max_chars or kind != cur_kind):
            out.append((cur, cur_kind))
            cur = ""
        if not cur:
            cur_kind = kind
        cur += p
    if cur.strip():
        out.append((cur, cur_kind))
    return out or [(md, "")]


def split_markdown(md: str, max_chars: int = _MAX_SEGMENT_CHARS) -> list[str]:
    """Book-sized markdown → RECON-sized segments: heading-split, packed to size.

    Headings (``#``/``##``) are the cut points; any section still over
    ``max_chars`` is further split on paragraph boundaries — the same
    dimensional fallback that carries a heading-less scan. Adjacent pieces are
    then greedily packed up to ``max_chars``: real converters flatten every
    section to ``##`` and emit lone ``## Chapter N`` lines (verified on an
    80-page docling probe: 53 raw segments, some 14 chars), so raw sections
    over-fragment — packing restores chapter-sized units and absorbs the
    micro-segments. A document smaller than ``max_chars`` packs to a single
    segment. Always returns ≥1 segment.
    """
    return [seg for seg, _ in _split_markdown_flagged(md, max_chars)]


def _segment_slug(segment: str, fallback: str) -> str:
    """Filename slug from the segment's first heading; ``fallback`` if none."""
    for line in segment.splitlines():
        if _HEADING_RE.match(line):
            slug = _SLUG_RE.sub("-", line.lstrip("#").strip().lower()).strip("-")
            if slug:
                return slug[:50]
    return fallback


# mineru drops the space after , ; : between letters ("symmetric,and positive")
# and the glitch flows into RECON concepts and note titles. Letters-only guard
# keeps digits ("10,000") and LaTeX macros ("\alpha,\beta") untouched.
_TIGHT_PUNCT_RE = re.compile(r"(?<=[A-Za-zà-ÿ])([,;:])(?=[A-Za-zà-ÿ])")


def _respace_prose(md: str) -> str:
    """Re-insert the missing space after ,;: in prose — not in code or math.

    ponytail: inline $…$ spans are skipped per line; display-math interiors are
    not tracked ("x,y" → "x, y" renders identically in LaTeX). Glued words with
    no punctuation ("overthe") need a dictionary — out of scope.
    """
    out: list[str] = []
    in_fence = False
    for line in md.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            parts = line.split("$")
            for i in range(0, len(parts), 2):  # even = outside $…$
                parts[i] = _TIGHT_PUNCT_RE.sub(r"\1 ", parts[i])
            line = "$".join(parts)
        out.append(line)
    return "".join(out)


def _doc_to_md(target: str, dest_dir: str) -> list[str]:
    src = _resolve_input(target)
    suffix = src.suffix.lower()
    # Images and OOXML have exactly one backend, so the provider seam does not
    # apply: a text-layer reader opens an image but reads no text out of it
    # (measured on MuPDF: a 1653x2339 render of a text page yields ''), and
    # neither docling nor opendataloader is a path verified here for either family.
    if suffix in TABULAR_EXTS:
        return _convert_tabular(src)
    if suffix in _MINERU_ONLY_EXTS:
        provider = _pdf_via_mineru
    elif suffix in MEDIA_EXTS:
        provider = _via_asr
    elif suffix in ODF_EXTS:
        provider = _via_odf
    elif suffix == ".rtf":
        provider = _via_rtf
    elif suffix == ".xls":
        provider = _via_xls
    elif suffix in LEGACY_OFFICE_EXTS:
        provider = _via_legacy_office
    # The seam is PDF-only — docling/opendataloader/mineru take a PDF and
    # nothing else, so DOCX/EPUB/FB2 each have their own in-process reader.
    elif suffix == ".docx":
        provider = _via_docx
    elif suffix == ".epub":
        provider = _via_epub
    elif suffix == ".fb2":
        provider = _via_fb2
    else:
        name = resolve_pdf_provider(CONFIG.pdf_provider)
        if name not in PDF_PROVIDERS:
            raise ValueError(
                f"unknown pdf_provider {CONFIG.pdf_provider!r} "
                f"(known: {', '.join(PDF_PROVIDERS)})"
            )
        provider = PDF_PROVIDERS[name]
    with tempfile.TemporaryDirectory() as tmp:
        md_text, images_src = provider(src, Path(tmp))
        if not md_text.strip():
            # Silence here would write an empty inbox note and call it success.
            if suffix in MEDIA_EXTS:
                raise ValueError(
                    f"no speech transcribed from {src.name} — the audio track "
                    "carries no recognisable speech (music, silence, or a "
                    "language the model was not given)"
                )
            if suffix in _MINERU_ONLY_EXTS:
                # mineru already ran (it is the only backend for these), so
                # pointing at OCR would be advice the user has already taken.
                raise ValueError(
                    f"no readable text in {src.name} — the OCR pass found none "
                    "(a photo with no writing in it, or an empty document)"
                )
            if suffix in (*_PURE_PY_OFFICE_EXTS, *_BASE_TEXT_EXTS):
                # No OCR anywhere in this path, so advice about OCR would be a
                # red herring: the file really does carry no text.
                raise ValueError(
                    f"no text in {src.name} — the document is empty, or its "
                    "content is entirely images (which this path does not read; "
                    "re-save it as PDF to run those through OCR)"
                )
            # The usual cause is a scan with no text layer, and the default
            # provider reads text layers only, so name the provider that does.
            raise ValueError(
                f"no text extracted from {src.name} — a scanned document needs OCR: "
                "`pip install 'mineru[pipeline]'` and set SILICA_PDF_PROVIDER=mineru"
            )
        # Copy only images the markdown references: mineru dumps every crop it
        # detects (477 files for a 200-page book, 19 referenced) — the rest
        # would land in the vault as orphans.
        referenced = {os.path.basename(m.group(1)) for m in _MD_IMG_RE.finditer(md_text)}
        renamed = _copy_images(                                  # before tmp is cleaned
            images_src, _images_dest(dest_dir), _image_prefix(src), only=referenced
        )
    body = _rewrite_image_links(_respace_prose(strip_degenerate_runs(md_text)), renamed)
    from silica.driver import DRIVER
    from silica.kernel.vault_manifest import active_inbox_dir

    inbox = active_inbox_dir() or "Inbox"
    flagged = _split_markdown_flagged(body, _MAX_SEGMENT_CHARS)
    segments = [seg for seg, _ in flagged]
    # Every segment names the real file it came from: the provenance ledger
    # only ever records the inbox note's basename, so without this the original
    # PDF is untraceable once the inbox note is archived. Plain quoted string,
    # not a link — the pointer must not enter the graph. CLEANUP carries it
    # into the source leaf when the note is later nucleated with keep_sources.
    fm = _provenance_fm(src, body)
    # Apparatus segments carry a frontmatter flag so /nucleate can skip them: a
    # reference list is citation metadata and a venue checklist is submission
    # paperwork, not content — each gets kept as raw material, never distilled
    # into venue/journal/ethics notes.
    def _flag(kind: str) -> str:
        return fm.replace("---\n", f"---\n{kind}: true\n", 1) if kind else fm

    # Single segment (a paper, an article) keeps the flat inbox path — no change
    # in behaviour, no subdir for the common case. Image links are basename
    # embeds (![[fig.png]]) so they resolve from any segment regardless of dir.
    # It is flagged like any other segment: computing the kind and then dropping
    # it meant a standalone bibliography export or a lone checklist page — one
    # segment because it fits under _MAX_SEGMENT_CHARS — was distilled in full.
    if len(segments) == 1:
        note_rel = f"{inbox}/{src.stem}.md"
        DRIVER.upsert(note_rel, _flag(flagged[0][1]) + body.lstrip("\n"))  # re-converting the same source refreshes its inbox note
        return [note_rel]

    width = len(str(len(segments)))
    paths: list[str] = []
    for i, (seg, kind) in enumerate(flagged, 1):
        slug = _segment_slug(seg, "part")
        note_rel = f"{inbox}/{src.stem}/{i:0{width}d}-{slug}.md"
        DRIVER.upsert(note_rel, _flag(kind) + seg.lstrip("\n"))  # re-converting the same source refreshes its segments
        paths.append(note_rel)
    logger.info("PDF %s split into %d inbox segment(s)", src.name, len(segments))
    return paths


# --- providers (each: src pdf, workdir → markdown text, images dir) ---------
#
# TODO(real-api): each provider's third-party call surface is only exercised by
# hand-faked modules in tests/test_convert.py — a library rename would drift the
# fakes and pass silently. Add a real-install smoke test to catch API drift.

def _via_pdfium(src: Path, workdir: Path) -> tuple[str, Path]:
    """Default provider — PDFium's text layer, headings from the outline or a
    line heuristic. No torch, no JVM, one ~3 MB wheel.

    It has no OCR: a scan with no text layer yields nothing, which `_doc_to_md`'s
    empty guard turns into an error naming mineru. Figures are not extracted
    either (the images dir comes back empty): PDFium exposes image objects, but
    turning them into files needs Pillow, which the base install does not carry;
    mineru (installed separately) is the path for a document whose figures are
    the point.
    """
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_c
    except ImportError:
        raise ValueError(
            "pypdfium2 not installed — `pip install 'pypdfium2>=5'`, "
            "or set SILICA_PDF_PROVIDER to mineru/docling/opendataloader"
        ) from None

    pdf = pdfium.PdfDocument(str(src))
    try:
        toc: list[tuple[int, str, int]] = []
        for bookmark in pdf.get_toc():
            title = " ".join((bookmark.get_title() or "").split())
            dest = bookmark.get_dest()
            index = dest.get_index() if dest is not None else None
            if title and index is not None and index >= 0:
                toc.append((bookmark.level, title, index))
        pages: list[list[tuple[str, float]]] = []
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            try:
                pages.append(_pdf_page_lines(textpage, pdfium_c))
            finally:
                textpage.close()
                page.close()
    finally:
        pdf.close()
    images = workdir / "images"
    images.mkdir(parents=True, exist_ok=True)
    return _pdf_markdown(pages, toc), images


def _pdf_page_lines(textpage, pdfium_c) -> list[tuple[str, float]]:
    """(text, font size) per non-blank line of a page.

    One font-size call per line, at the line's first glyph, rather than one per
    character: PDFium's text index is one unit per character, so the offset of
    a line in the page string IS its character index (a surrogate pair shifts it
    by one glyph, which lands on a neighbouring character of the same line).
    """
    text = textpage.get_text_range()
    n = textpage.count_chars()
    lines: list[tuple[str, float]] = []
    pos = 0
    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        stripped = line.strip()
        if stripped:
            first = pos + (len(line) - len(line.lstrip()))
            size = float(pdfium_c.FPDFText_GetFontSize(textpage, first)) if first < n else 0.0
            lines.append((stripped, size))
        pos += len(raw) + 1
    return lines


# Section headings a paper carries whether or not it has an outline. The first
# two groups are English and Italian apparatus names; the numbered form
# ("2 Method", "3.1 Data") is handled by `_NUMBERED_HEADING_RE`.
_APPARATUS_HEADING_RE = re.compile(
    r"(?:\d+(?:\.\d+)*\.?\s+)?(?:abstract|introduction|background|related work|"
    r"methods?|methodology|experiments?|results|evaluation|discussion|conclusions?|"
    r"limitations|acknowledg(?:e)?ments|references|bibliography|appendix(?:\s+[a-z])?|"
    r"introduzione|metodo|metodi|risultati|discussione|conclusioni|ringraziamenti|"
    r"bibliografia|riferimenti bibliografici|appendice|indice|sommario)",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+(?=[^\W\d_])")
_HEADING_MAX_WORDS = 12


def _body_font_size(lines: list[tuple[str, float]]) -> float:
    """The page's running-text size: the commonest size over long lines (a
    heading is short, a caption or footnote is small), else over every line."""
    import statistics

    long = [round(size, 1) for text, size in lines if len(text) >= 40 and size > 0]
    pool = long or [round(size, 1) for _, size in lines if size > 0]
    return statistics.mode(pool) if pool else 0.0


def _heading_level(text: str, size: float, body: float, big_lines: int) -> int:
    """0 for body text, else the markdown heading depth of a PDF line.

    Three signals, in order of trust: a numbered section line ("2.1 Data", depth
    from the numbering; the first component is capped at 99 so a year never
    reads as a section), an apparatus name (References, Appendix), and a short
    line set at least 15% above the body size (a title). The size rule is
    switched off on a page with more than eight such lines: that is a figure's
    axis labels and legend, not eight titles (measured: 40 on one page of an
    entity-linking paper). Nothing below the body size is ever a heading: page
    headers, footnotes and captions are the small text.

    ponytail: no font clustering, no bold detection. mineru does layout; this
    exists so `_split_on_headings` and the References flag keep working on a
    plain paper without it.
    """
    words = text.split()
    if not words or len(words) > _HEADING_MAX_WORDS or not re.search(r"[^\W\d_]", text):
        return 0
    if body and size < body * 0.95:
        return 0
    m = _NUMBERED_HEADING_RE.match(text)
    if m and int(m.group(1).split(".")[0]) <= 99:
        return min(m.group(1).count(".") + 1, 6)
    if _APPARATUS_HEADING_RE.fullmatch(text):
        return 1
    if body and size >= body * 1.15 and big_lines <= 8:
        return 1 if size >= body * 1.4 else 2
    return 0


def _normalise_line(text: str) -> str:
    return re.sub(r"\W+", " ", text).strip().lower()


def _pdf_markdown(pages: list[list[tuple[str, float]]], toc: list[tuple[int, str, int]]) -> str:
    """Page lines → markdown with headings.

    The embedded outline beats any guessing wherever it exists (23 headings vs
    12 on a 19-entry probe paper in the pymupdf4llm era), so with an outline
    each entry marks the line carrying its title on its page, or opens the page
    when the title is not found there (a bookmark pointing mid-column). With no
    outline, `_heading_level` decides per line.
    """
    by_page: dict[int, list[tuple[int, str]]] = {}
    for level, title, index in toc:
        by_page.setdefault(index, []).append((level, title))
    blocks: list[str] = []
    for i, lines in enumerate(pages):
        texts = [t for t, _ in lines]
        marks: dict[int, int] = {}
        lead: list[str] = []
        if toc:
            norm = [_normalise_line(t) for t in texts]
            for level, title in by_page.get(i, []):
                want = _normalise_line(title)
                hit = next((j for j, n in enumerate(norm)
                            if n == want or (want and (n.startswith(want) or want.startswith(n)))
                            and j not in marks), None)
                depth = min(level + 1, 6)
                if hit is None:
                    lead.append(f"{'#' * depth} {title}")
                else:
                    marks[hit] = depth
        else:
            body = _body_font_size(lines)
            big = sum(1 for t, size in lines
                      if body and size >= body * 1.15 and len(t.split()) <= _HEADING_MAX_WORDS)
            for j, (t, size) in enumerate(lines):
                depth = _heading_level(t, size, body, big)
                if depth:
                    marks[j] = depth
        out: list[str] = list(lead)
        for j, t in enumerate(texts):
            out.append(f"\n{'#' * marks[j]} {t}\n" if j in marks else t)
        blocks.append("\n".join(out))
    md = "\n\n".join(b for b in blocks if b.strip())
    return re.sub(r"\n{3,}", "\n\n", md).strip("\n") + ("\n" if md.strip() else "")


def _via_docx(src: Path, workdir: Path) -> tuple[str, Path]:
    """DOCX through mammoth's HTML, then the shared HTML-to-markdown pass.

    mammoth has a markdown writer of its own; it is not used because it escapes
    every period and bracket in prose (`Body text\\.`), and a note is quoted
    later by people. Its HTML inlines images as data URIs, which the shared
    pass writes out as files so the figures reach the vault like a PDF's do.
    """
    try:
        import mammoth
    except ImportError:
        raise ValueError("mammoth not installed — `pip install 'mammoth>=1.9'`") from None

    images = workdir / "images"
    images.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fh:
        html = mammoth.convert_to_html(fh).value
    return _html_to_md(html, images=images, prefix=src.stem), images


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _epub_opf(z: zipfile.ZipFile) -> tuple[ET.Element, str]:
    """The package document and its directory inside the container."""
    import posixpath

    container = _parse_office_xml(_zip_member(z, "META-INF/container.xml"))
    rootfile = next((el for el in container.iter() if _local_name(el.tag) == "rootfile"), None)
    opf_path = rootfile.get("full-path") if rootfile is not None else None
    if not opf_path:
        raise ValueError("EPUB container names no package document")
    return _parse_office_xml(_zip_member(z, opf_path)), posixpath.dirname(opf_path)


def _via_epub(src: Path, workdir: Path) -> tuple[str, Path]:
    """EPUB: the XHTML chapters in spine order, through the shared HTML pass.

    Reading order is the OPF spine, not the zip's member order or the file
    names, which is what the pymupdf reader also honoured. Members are opened
    through the same size guard as every other zip-backed format here.

    ponytail: chapter images are not carried over (an `<img>` pointing into the
    zip is dropped with its alt text kept). Extract them when a real EPUB
    library with figures shows up; the seam is `_html_to_md`'s image hook.
    """
    import posixpath
    from urllib.parse import unquote

    with zipfile.ZipFile(src) as z:
        opf, base = _epub_opf(z)
        manifest = {el.get("id"): el.get("href")
                    for el in opf.iter() if _local_name(el.tag) == "item" and el.get("href")}
        spine = [el.get("idref") for el in opf.iter() if _local_name(el.tag) == "itemref"]
        parts: list[str] = []
        for idref in spine:
            href = manifest.get(idref or "")
            if not href:
                continue
            member = posixpath.normpath(posixpath.join(base, unquote(href))) if base else unquote(href)
            try:
                data = _zip_member(z, member)
            except KeyError:
                continue
            parts.append(_html_to_md(data.decode("utf-8", "replace")))
    images = workdir / "images"
    images.mkdir(parents=True, exist_ok=True)
    return "\n\n".join(part for part in parts if part.strip()), images


def _epub_metadata(src: Path) -> dict[str, str]:
    """Dublin Core title/creator/date from the package document; absent keys
    for absent fields. Never worth failing over, hence the blanket except."""
    out: dict[str, str] = {}
    try:
        with zipfile.ZipFile(src) as z:
            opf, _ = _epub_opf(z)
        for el in opf.iter():
            name = _local_name(el.tag)
            if name in ("title", "creator", "date") and name not in out:
                value = " ".join((el.text or "").split())
                if value:
                    out[name] = value
    except Exception:
        pass
    return out


def _via_fb2(src: Path, workdir: Path) -> tuple[str, Path]:
    """FictionBook 2: `<section>` titles become headings at their nesting depth,
    `<p>` become paragraphs. The standard library reads it; nothing else is
    involved."""
    root = _parse_office_xml(src.read_bytes())
    blocks: list[str] = []

    def text_of(el: ET.Element) -> str:
        return " ".join(" ".join(el.itertext()).split())

    def walk(el: ET.Element, depth: int) -> None:
        for child in el:
            name = _local_name(child.tag)
            if name == "section":
                walk(child, depth + 1)
            elif name == "title":
                t = text_of(child)
                if t:
                    blocks.append(f"{'#' * min(depth, 6)} {t}")
            elif name in ("p", "subtitle", "text-author", "v"):
                t = text_of(child)
                if t:
                    blocks.append(t)
            elif name in ("epigraph", "cite", "poem", "stanza", "annotation"):
                walk(child, depth)

    for body in root.iter():
        if _local_name(body.tag) == "body":
            walk(body, 1)
    images = workdir / "images"
    images.mkdir(parents=True, exist_ok=True)
    return "\n\n".join(blocks), images


# --- HTML → markdown, shared by DOCX and EPUB --------------------------------

_HTML_SKIP = frozenset({"script", "style", "head", "title", "noscript", "svg", "template"})
_HTML_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_HTML_BLOCKS = frozenset({"p", "div", "section", "article", "blockquote", "figure",
                          "figcaption", "aside", "header", "footer", "main", "nav"})


class _HtmlToMarkdown(HTMLParser):
    """The subset of HTML a document converter emits, to the markdown the rest
    of the pipeline splits on: headings, paragraphs, list items, table rows,
    code blocks, images. Inline emphasis is dropped as markup and kept as text.

    ponytail: stdlib html.parser, as in web_fetch. It treats an unclosed
    `<style>` as CDATA to the end of the file; mammoth and EPUB producers close
    their tags, and a truncated chapter is the least of that file's problems.
    """

    def __init__(self, images: Path | None, prefix: str) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []   # (kind, text)
        self._cur: list[str] = []
        self._kind = "p"
        self._skip = 0
        self._pre = 0
        self._images = images
        self._prefix = prefix
        self._n_images = 0

    def _flush(self) -> None:
        text = "".join(self._cur)
        text = text if self._pre else " ".join(text.split())
        if text.strip():
            self.blocks.append((self._kind, text.strip("\n") if self._pre else text))
        self._cur = []
        self._kind = "p"

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _HTML_SKIP:
            self._skip += 1
        elif self._skip:
            return
        elif tag in _HTML_HEADINGS:
            self._flush()
            self._kind = f"h{_HTML_HEADINGS[tag]}"
        elif tag == "li":
            self._flush()
            self._kind = "li"
        elif tag == "pre":
            self._flush()
            self._pre += 1
            self._kind = "pre"
        elif tag == "br":
            self._cur.append("\n" if self._pre else " ")
        elif tag in ("td", "th"):
            if self._cur and not "".join(self._cur).endswith("| "):
                self._cur.append(" | ")
        elif tag == "tr":
            self._flush()
            self._kind = "tr"
        elif tag == "img":
            self._image(dict(attrs))
        elif tag in _HTML_BLOCKS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _HTML_SKIP:
            self._skip = max(0, self._skip - 1)
        elif self._skip:
            return
        elif tag in _HTML_HEADINGS or tag in ("li", "tr") or tag in _HTML_BLOCKS:
            self._flush()
        elif tag == "pre":
            self._flush()
            self._pre = max(0, self._pre - 1)

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._cur.append(data)

    def _image(self, attrs: dict) -> None:
        """A data-URI image (mammoth inlines every DOCX figure that way) is
        written to the images dir and referenced like a PDF figure, so the
        shared tail copies it into the vault. Any other src keeps its alt text
        as a bracketed caption, the same convention web_fetch uses."""
        import base64

        alt = " ".join((attrs.get("alt") or "").split())
        src = attrs.get("src") or ""
        m = re.match(r"data:image/(png|jpe?g|gif|webp|bmp|tiff?);base64,(.+)", src, re.S)
        if m and self._images is not None:
            ext = {"jpeg": "jpg", "tif": "tiff"}.get(m.group(1), m.group(1))
            self._n_images += 1
            name = f"{self._prefix}-{self._n_images}.{ext}"
            try:
                (self._images / name).write_bytes(base64.b64decode(m.group(2)))
            except (ValueError, OSError):
                # A corrupt data URI is a lost figure, not a lost document.
                return
            self._flush()
            self.blocks.append(("p", f"![{alt}]({self._images / name})"))
        elif alt and any(c.isalpha() for c in alt):
            self._flush()
            self.blocks.append(("p", f"[image: {alt}]"))


def _html_to_md(html: str, images: Path | None = None, prefix: str = "img") -> str:
    parser = _HtmlToMarkdown(images, prefix)
    parser.feed(html)
    parser.close()
    parser._flush()
    out: list[str] = []
    prev = ""
    for kind, text in parser.blocks:
        if kind.startswith("h"):
            line = f"{'#' * int(kind[1])} {text}"
        elif kind == "li":
            line = f"- {text}"
        elif kind == "pre":
            line = f"```\n{text}\n```"
        else:
            line = text
        # Consecutive list items and table rows stay one block apart by a single
        # newline: markdown reads a blank line between items as separate lists.
        sep = "\n" if kind in ("li", "tr") and prev == kind else "\n\n"
        out.append((sep if out else "") + line)
        prev = kind
    return "".join(out)


def _pdf_via_docling(src: Path, workdir: Path) -> tuple[str, Path]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import ImageRefMode
    except ImportError:
        raise ValueError(
            "docling not installed — `pip install docling`, "
            "or set SILICA_PDF_PROVIDER to mineru/opendataloader"
        ) from None

    opts = PdfPipelineOptions()
    opts.generate_picture_images = True  # else REFERENCED export emits placeholders
    # Maximum-precision non-generative pins. No do_formula_enrichment /
    # do_code_enrichment: CodeFormula is a generative model, out of boundary.
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    opts.images_scale = 2.0  # extracted figures at 144 dpi instead of 72
    opts.do_ocr = True
    # docling's default language list omits Italian; csv config, split here.
    opts.ocr_options.lang = [s.strip() for s in CONFIG.pdf_ocr_lang.split(",") if s.strip()]
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    doc = converter.convert(str(src)).document
    images = workdir / "images"
    md_path = workdir / f"{src.stem}.md"
    doc.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED, artifacts_dir=images)
    return md_path.read_text(encoding="utf-8", errors="replace"), images


def _pdf_via_opendataloader(src: Path, workdir: Path) -> tuple[str, Path]:
    # Java-backed (JVM per convert), Apache-2.0. Strong on complex tables and
    # multi-column reading order; the wheel bundles the CLI but needs Java 11+.
    try:
        import opendataloader_pdf
    except ImportError:
        raise ValueError(
            "opendataloader-pdf not installed — `pip install opendataloader-pdf` "
            "(needs Java 11+), or set SILICA_PDF_PROVIDER to docling/mineru"
        ) from None

    out = workdir / "out"
    images = workdir / "images"
    # use_struct_tree: when the PDF carries native structure tags, headings and
    # reading order come from the author's own markup. `hybrid` (the only OCR
    # path) is never passed — it is generative, out of boundary — so scanned
    # PDFs yield nothing from this provider; use mineru/docling for those. If
    # the installed wrapper predates the kwarg, the TypeError names it.
    opendataloader_pdf.convert(
        input_path=str(src), output_dir=str(out),
        format="markdown", image_output="external", image_dir=str(images),
        use_struct_tree=True,
    )
    hits = glob(str(out / "**" / "*.md"), recursive=True)
    if not hits:
        raise ValueError("opendataloader produced no markdown")
    return Path(hits[0]).read_text(encoding="utf-8", errors="replace"), images


# --- office without an office suite: ODF, RTF, legacy .xls ------------------

# Every zip-backed office format is attacker-controlled the moment someone drops
# a file into the vault folder, and a 200 KB archive can declare a member that
# inflates to gigabytes. 64 MB is far past any real content.xml (a 500-page .odt
# with tables lands around 3 MB) and far short of a bomb.
_MAX_ZIP_MEMBER = 64 * 1024 * 1024


def _zip_member(z: zipfile.ZipFile, name: str) -> bytes:
    """One member of an office archive, refused if it inflates past the ceiling.

    The declared size is checked first so a bomb costs nothing to refuse, and the
    read is capped anyway because the local header is written by whoever built
    the archive and may simply lie about it.
    """
    if z.getinfo(name).file_size > _MAX_ZIP_MEMBER:
        raise ValueError(
            f"{name} declares more than {_MAX_ZIP_MEMBER // (1024 * 1024)} MB "
            "uncompressed — refusing to read it (decompression bomb)"
        )
    with z.open(name) as f:
        data = f.read(_MAX_ZIP_MEMBER + 1)
    if len(data) > _MAX_ZIP_MEMBER:
        raise ValueError(
            f"{name} inflates past {_MAX_ZIP_MEMBER // (1024 * 1024)} MB — "
            "refusing to read it (decompression bomb)"
        )
    return data


class _NoDoctype(ET.TreeBuilder):
    """TreeBuilder that refuses a DTD.

    xml.etree is documented-vulnerable to entity expansion (billion laughs,
    quadratic blowup) and the stdlib exposes no knob to bound it. Every one of
    those attacks needs entity declarations, which need a DOCTYPE, and no office
    format writes one — so refusing the prolog closes the whole class without a
    third-party parser. expat calls this at the start of the declaration, before
    a single entity is expanded.
    """

    def doctype(self, name, pubid, system) -> None:
        raise ValueError("XML declares a DOCTYPE — refusing to parse it")


def _parse_office_xml(data: bytes) -> ET.Element:
    """Parse XML that came out of a document, with the DTD refused."""
    return ET.fromstring(data, parser=ET.XMLParser(target=_NoDoctype()))


_ODF_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_ODF_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_ODF_P = f"{{{_ODF_TEXT_NS}}}p"
_ODF_H = f"{{{_ODF_TEXT_NS}}}h"
_ODF_LEVEL = f"{{{_ODF_TEXT_NS}}}outline-level"
_ODF_ROW = f"{{{_ODF_TABLE_NS}}}table-row"
_ODF_CELL = f"{{{_ODF_TABLE_NS}}}table-cell"
# The three block elements worth emitting. Everything else (spans, links,
# bookmarks, annotations) is inline and comes along inside `itertext()`.
_ODF_BLOCKS = (_ODF_P, _ODF_H, _ODF_ROW)


def _odf_text(el: ET.Element) -> str:
    """All text under one block, whitespace-normalised.

    ODF encodes runs of spaces as `<text:s/>` and tabs as `<text:tab/>`, which
    carry no text of their own, so collapsing here loses nothing that survived
    the format anyway.
    """
    return " ".join("".join(el.itertext()).split())


def _odf_row(row: ET.Element) -> str:
    """One spreadsheet/table row as `cell | cell | cell`.

    Trailing empties are dropped because ODF pads every row out to the sheet
    width — a two-column `.ods` still declares its rows 1024 cells wide, via
    `table:number-columns-repeated` on one empty cell. Repeats are deliberately
    NOT expanded: doing so is what turns a small sheet into megabytes of pipes.
    """
    cells = [_odf_text(c) for c in row.findall(_ODF_CELL)]
    while cells and not cells[-1]:
        cells.pop()
    return " | ".join(cells)


def _via_odf(src: Path, workdir: Path) -> tuple[str, Path]:
    """`.odt`/`.odp`/`.ods` → markdown, standard library only.

    An ODF file is a ZIP whose `content.xml` already holds the text in document
    order, so the entire "install a 240 MB office suite" hop bought us, for
    text, an XML walk. Headings keep their outline level, table rows keep their
    shape, and nothing runs as a subprocess.
    """
    try:
        with zipfile.ZipFile(src) as z:
            root = _parse_office_xml(_zip_member(z, "content.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as e:
        raise ValueError(
            f"{src.name} is not a readable ODF document ({type(e).__name__}) — "
            "re-save it, or export it as PDF"
        ) from None

    # Blocks in document order, minus any nested inside another block: a text
    # box holds `text:p` inside a `text:p`, and a table cell holds them inside a
    # row, so emitting both levels would print that text twice.
    blocks = [el for el in root.iter() if el.tag in _ODF_BLOCKS]
    nested = {
        id(d) for el in blocks for d in el.iter() if d is not el and d.tag in _ODF_BLOCKS
    }
    out: list[str] = []
    prev = ""
    for el in blocks:
        if id(el) in nested:
            continue
        text = _odf_row(el) if el.tag == _ODF_ROW else _odf_text(el)
        if not text:
            continue
        if el.tag == _ODF_H:
            raw = el.get(_ODF_LEVEL, "1")
            # Clamped: ODF outline levels run past 6, and `####### x` is not a
            # heading in markdown — it renders as literal hashes.
            level = int(raw) if raw.isdigit() and raw != "0" else 1
            text = f"{'#' * min(level, 6)} {text}"
        if out:
            # Consecutive rows are one block, not one paragraph each: a
            # 200-row sheet would otherwise reach the segmenter as 200 paragraphs.
            out.append("\n" if el.tag == prev == _ODF_ROW else "\n\n")
        out.append(text)
        prev = el.tag
    return "".join(out), workdir


def _via_rtf(src: Path, workdir: Path) -> tuple[str, Path]:
    """`.rtf` → text via striprtf (15 KB, BSD-3)."""
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError:
        raise ValueError(
            "striprtf not installed — `pip install striprtf`, "
            "or re-save the file as PDF"
        ) from None
    # RTF is 7-bit ASCII by spec with non-ASCII escaped, so a stray raw byte is
    # a malformed file rather than a different encoding: replace and move on.
    return rtf_to_text(
        src.read_text(encoding="utf-8", errors="replace"), errors="ignore"
    ), workdir


def _xls_cell(cell, datemode: int) -> str:
    """One legacy-spreadsheet cell as text.

    Numbers matter here: xlrd hands back every number as a float, so a quantity
    of 3 reaches the vault as "3.0" unless it is put back. Dates are worse — the
    wire format is a float offset from an epoch that differs between Mac and
    Windows workbooks, which is what `datemode` carries.
    """
    import xlrd

    if cell.ctype == xlrd.XL_CELL_DATE:
        try:
            return xlrd.xldate_as_datetime(cell.value, datemode).isoformat(" ")
        except Exception:                       # a corrupt serial is not fatal
            return str(cell.value)
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return "true" if cell.value else "false"
    if cell.ctype == xlrd.XL_CELL_NUMBER and float(cell.value).is_integer():
        return str(int(cell.value))
    return str(cell.value).strip()


def _via_xls(src: Path, workdir: Path) -> tuple[str, Path]:
    """Legacy `.xls` (BIFF) → markdown via xlrd (192 KB, BSD).

    xlrd 2.x dropped `.xlsx` and kept exactly this: the pre-2007 binary format,
    which is the one nothing else in the base install reads. Each sheet becomes
    an `##` heading so `_split_on_headings` can segment a fat workbook.
    """
    try:
        import xlrd
    except ImportError:
        raise ValueError(
            "xlrd not installed — `pip install xlrd`, or re-save the file as .xlsx"
        ) from None

    try:
        # logfile: xlrd narrates ("*** No CODEPAGE record...") straight to
        # stdout, which in the TUI lands in the middle of the user's session.
        book = xlrd.open_workbook(str(src), logfile=io.StringIO())
    except Exception as e:
        raise ValueError(f"could not read {src.name}: {e}") from None

    out: list[str] = []
    for sheet in book.sheets():
        rows = []
        for r in range(sheet.nrows):
            cells = [_xls_cell(c, book.datemode) for c in sheet.row(r)]
            while cells and not cells[-1]:
                cells.pop()
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            out.append(f"## {sheet.name}\n\n" + "\n".join(rows))
    return "\n\n".join(out), workdir


# --- data files: profile, never rows ----------------------------------------


def _md_cell(value: object) -> str:
    """A value as one markdown table cell: pipe-safe, capped, never None."""
    text = "" if value is None else str(value)
    if len(text) > 80:
        text = text[:77] + "..."
    return text.replace("|", "\\|").replace("\n", " ")


def _md_table(header: list[str], rows: list[list]) -> str:
    """A markdown table — with the separator row that makes it one."""
    lines = [
        "| " + " | ".join(_md_cell(h) for h in header) + " |",
        "|" + "---|" * len(header),
    ]
    lines += ["| " + " | ".join(_md_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


# Rows shown in the profile's sample table, and the distinct-value window in
# which a VARCHAR column counts as categorical and gets its domain enumerated.
# 20 keeps the enumeration a line, not a table dump. The floor is 3 because the
# Columns table already prints min and max: at 1 distinct they are the value,
# at 2 they are BOTH values, so only from 3 up does the list add a token the
# note does not already carry. Measured on an ISTAT SDMX export: 13 candidate
# columns, 10 of them redundant under this rule.
_SAMPLE_ROWS = 5
_CATEGORICAL_MIN_DISTINCT = 3
_CATEGORICAL_MAX_DISTINCT = 20


def _display_path(src: Path) -> str:
    """`src` repo-relative when the vault's repo contains it, absolute else.

    The absolute path is dead weight in the note body: it breaks on any other
    machine and silica_query_table resolves vault-relative first. Outside the
    repo the absolute path is the only true name, so it stays.
    """
    from silica.kernel.recall import paths as _rpaths

    root = _rpaths.repo_root_for(CONFIG.vault_path) if CONFIG.vault_path else None
    if root is not None:
        try:
            return src.resolve().relative_to(Path(root).resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return str(src)


def _profile_md(
    src: Path,
    n_rows: int,
    columns_table: str,
    sample_cols: list[str],
    sample_rows: list[list],
    sample_label: str,
    categorical: Sequence[tuple[str, int, str]] = (),
    members: list[Path] | None = None,
) -> str:
    n_cols = len(sample_cols)
    disp = _display_path(src)
    members = members or []
    family = len(members) > 1
    title = f"{_family_stem(members)} family" if family else src.name
    if family:
        # A shard family: one table split across files (same header). The
        # profile describes the union; the member list is what a reader (and
        # the documents: edge) needs to find the physical files.
        shards = "\n".join(f"- `{_display_path(m)}`" for m in members)
        head = (f"One table in {len(members)} files — {n_rows} rows x "
                f"{n_cols} columns across:\n\n{shards}\n")
    else:
        head = f"Data file: `{disp}` — {n_rows} rows x {n_cols} columns.\n"
    cat_section = ""
    if categorical:
        # The retrieval surface for cell values: a note mentioning a category
        # ("ACME Corp") reaches this table through BM25 on these tokens even
        # when no sampled row and no min/max extreme carries the value.
        lines = "\n".join(f"- {c} ({n} values): {vals}" for c, n, vals in categorical)
        cat_section = f"## Categorical values\n\n{lines}\n\n"
    return (
        f"# {title} data profile\n\n"
        + head + "\n"
        "The rows are NOT in the vault; this note is the file's profile. Answer "
        "questions about the data by querying the file in place:\n"
        f'`silica_query_table(path="{disp}", sql="SELECT ... FROM t")` — '
        "start with `SUMMARIZE t` when unsure.\n\n"
        f"## Columns\n\n{columns_table}\n\n"
        + cat_section
        + f"## Sample ({sample_label})\n\n"
        + _md_table(sample_cols, sample_rows)
        + "\n"
    )


_COUNTER_RE = re.compile(r"[\s_.\-0-9]*")


def _counter_pair(a: str, b: str) -> bool:
    """True when two stems differ only by a counter (digits, separators, dates)."""
    p = os.path.commonprefix([a, b])
    return bool(_COUNTER_RE.fullmatch(a[len(p):]) and _COUNTER_RE.fullmatch(b[len(p):]))


def _tabular_family(src: Path) -> list[Path]:
    """Same-directory siblings holding shards of one table, `src` included.

    Membership needs BOTH gates, name first. Byte-identical header alone is
    not enough — SDMX exports share one generic column layout across
    DIFFERENT datasets, so on the field vault it grouped lavoro_15piu with
    demografia under an invented "censpop". A shard's name differs from its
    siblings only by the counter (censpop_lavoro_15piu_sicilia_01..12), and
    the check is pairwise against `src`: a global common prefix over every
    same-header file collapses to the export tool's shared prefix and proves
    nothing. Name gate first also spares the header read for most siblings.
    # ponytail: one header read per name-matching sibling per convert call,
    # O(n^2) reads when a batch converts one directory of same-named shards —
    # fine at hundreds; cache headers per directory beyond ~5k shards.
    """
    def head(p: Path) -> bytes | None:
        try:
            with p.open("rb") as fh:
                return fh.readline()
        except OSError:
            return None

    mine = head(src)
    if not mine:
        return [src]
    fam = [src]
    for sib in src.parent.glob(f"*{src.suffix}"):
        if (sib != src and sib.is_file()
                and _counter_pair(src.stem, sib.stem) and head(sib) == mine):
            fam.append(sib)
    return sorted(fam)


def _family_stem(members: list[Path] | None) -> str:
    """Shared stem of a shard family, "" when the names do not prove one.

    Two gates, both required. The names must share a >=3-char prefix (under
    that it is noise, not a name), and past that prefix every stem may differ
    only by a counter (digits, separators, dates). The second gate is the
    field-test correction: SDMX exports share one byte-identical generic
    header across DIFFERENT datasets, so header equality alone grouped
    censpop_lavoro_15piu with censpop_demografia under an invented "censpop"
    — "shards of one table" means the filenames differ only by the shard
    counter, and anything more is a different table.
    """
    if not members:
        return ""
    stems = [m.stem for m in members]
    prefix = os.path.commonprefix(stems)
    if any(not re.fullmatch(r"[\s_.\-0-9]*", s[len(prefix):]) for s in stems):
        return ""
    stem = re.sub(r"[\s_.\-0-9]+$", "", prefix)
    return stem if len(stem) >= 3 else ""


def _bind_members(con, members: list[Path]) -> None:
    """CREATE VIEW t over a shard family — one list-read, schema shared.

    Same quoting contract as tools/tabular._bind_source. List order is the
    sorted member order, so the evenly spaced sample walks shard 01 → NN.
    """
    quoted = ", ".join("'" + str(m).replace("'", "''") + "'" for m in members)
    if members[0].suffix.lower() == ".parquet":
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet([{quoted}])")
        return
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_csv([{quoted}], sample_size=-1)")


def _categorical_values(con, stats) -> list[tuple[str, int, str]]:
    """(column, n_distinct, joined domain) for low-cardinality VARCHAR columns.

    The one signal schema + sample structurally miss: a note citing a value
    that lives only in cells ("ACME Corp") has nothing to match on. Enumerating
    every column would serialize the table back into the vault, so only small
    text domains qualify (the published knob: Schema-First Retrieval,
    arXiv:2606.28387). Numeric domains stay out — min/max already brackets
    them and years/flags add no retrievable token.
    """
    out: list[tuple[str, int, str]] = []
    for row in stats:
        col, ctype, approx = row[0], row[1], row[4]
        if ctype != "VARCHAR":
            continue
        try:
            if int(approx) > _CATEGORICAL_MAX_DISTINCT:
                continue
        except (TypeError, ValueError):
            continue  # approx_unique unparsable: a column we cannot gate, skip
        q = str(col).replace('"', '""')
        vals = con.execute(
            f'SELECT DISTINCT "{q}" FROM t WHERE "{q}" IS NOT NULL '
            f"ORDER BY 1 LIMIT {_CATEGORICAL_MAX_DISTINCT + 1}"
        ).fetchall()
        # approx_unique is a sketch; the exact count decides membership. The
        # floor is applied here too — approx can read 3 on a 2-value column.
        if not _CATEGORICAL_MIN_DISTINCT <= len(vals) <= _CATEGORICAL_MAX_DISTINCT:
            continue
        joined = ", ".join(str(v[0]) for v in vals)
        if len(joined) > 300:
            continue  # a free-text column with few rows is not a category set
        out.append((str(col), len(vals), joined))
    return out


def _duckdb_profile(src: Path, members: list[Path] | None = None) -> str:
    """Schema + per-column stats + sample, computed by DuckDB without loading.

    With `members`, one profile over the whole shard family (same header, one
    schema): stats and sample describe the union, which is the table a reader
    actually reasons about."""
    from silica.tools.tabular import _connect, utf8_source

    # Non-utf-8 members are re-encoded into `tmp` and DuckDB is confined there
    # instead: the reader validates encoding and refuses the original outright.
    # A pure-utf-8 file copies nothing and binds in place, as before.
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        files = members or [src]
        bound = [utf8_source(m, tmpdir) for m in files]
        moved = [b.parent == tmpdir for b in bound]
        if any(moved) and not all(moved):
            # `_connect` allows exactly ONE directory, so a mixed family has to
            # be read entirely from the temp dir; the utf-8 members ride along.
            bound = [b if mv else Path(shutil.copy2(m, tmpdir / m.name))
                     for m, b, mv in zip(files, bound, moved)]
        return _duckdb_profile_bound(src, members, bound, _connect(bound[0].parent))


def _duckdb_profile_bound(src: Path, members: list[Path] | None,
                          bound: list[Path], con) -> str:
    """`_duckdb_profile` with the encoding/confinement question already settled."""
    from silica.tools.tabular import _bind_source

    try:
        if members and len(members) > 1:
            _bind_members(con, bound)
        else:
            _bind_source(con, bound[0])
        n_rows = con.execute("SELECT count(*) FROM t").fetchone()[0]
        stats = con.execute(
            "SELECT column_name, column_type, min, max, approx_unique, "
            "null_percentage FROM (SUMMARIZE t)"
        ).fetchall()
        # All-null columns collapse to one line: an SDMX export carries 15/28
        # pure-NOTE_* padding columns, and each cost a Columns row plus a
        # Sample column while adding zero retrievable signal.
        def _all_null(row) -> bool:
            try:
                return float(row[5]) >= 100.0
            except (TypeError, ValueError):
                return False
        dropped = [str(r[0]) for r in stats if _all_null(r)]
        kept = [r for r in stats if not _all_null(r)]
        if not kept or not dropped:  # nothing to collapse, or nothing left
            dropped, kept = [], list(stats)
        # Every sample column CAST to VARCHAR, in SQL. The sample is rendered
        # as markdown text, so the Python-side value is thrown away anyway —
        # and materializing it is what breaks: DuckDB needs pytz to hand a
        # TIMESTAMP WITH TIME ZONE to Python, and no install here has it, so a
        # download ledger with ISO offsets crashed the whole convert. Casting
        # in the engine also spares the exotic types (INTERVAL, nested) the
        # same trip. The declared type stays visible in the Columns table.
        cols_sql = ", ".join(
            'CAST("{0}" AS VARCHAR) AS "{0}"'.format(str(r[0]).replace('"', '""'))
            for r in kept
        )
        if n_rows <= _SAMPLE_ROWS:
            sample_q = f"SELECT {cols_sql} FROM t LIMIT {_SAMPLE_ROWS}"
            label = "all rows"
        else:
            # Evenly spaced, not LIMIT 5: real exports arrive sorted (by date,
            # region, customer) and the head is then a single stratum. Spread
            # indices also stay deterministic, which USING SAMPLE is not —
            # a re-convert must not churn the note. Insertion order is the
            # row_number here; DuckDB preserves it for file reads by default.
            step = (n_rows - 1) / (_SAMPLE_ROWS - 1)
            idx = sorted({round(k * step) + 1 for k in range(_SAMPLE_ROWS)})
            sample_q = (
                f"SELECT {cols_sql} FROM "
                "(SELECT *, row_number() OVER () AS _silica_rn FROM t) "
                f"WHERE _silica_rn IN ({', '.join(map(str, idx))}) "
                "ORDER BY _silica_rn"
            )
            label = f"{len(idx)} of {n_rows} rows, evenly spaced"
        sample = con.execute(sample_q).fetchall()
        sample_cols = [d[0] for d in con.description]
        categorical = _categorical_values(con, kept)
    finally:
        con.close()
    columns_table = _md_table(
        ["column", "type", "min", "max", "distinct", "null %"],
        [list(row) for row in kept],
    )
    if dropped:
        columns_table += (f"\n\n{len(dropped)} columns entirely null: "
                          + ", ".join(dropped))
    return _profile_md(src, n_rows, columns_table, sample_cols,
                       [list(r) for r in sample], label, categorical, members)


def _convert_tabular(src: Path) -> list[str]:
    """Data file → profile note in the inbox. The rows never enter the vault.

    ADR-0014 turns every source into prose, and for a data file the honest
    prose is a *description* of the table, not a serialization of it: column
    names and types are exactly what the agent needs to write a correct
    `silica_query_table` SELECT on the first try, and they embed — rows don't.

    Own write tail, not `_doc_to_md`'s: a profile has no images and is one
    small note by construction, and the family case needs a note NAMED after
    the family — converting shard 02 of 12 must refresh the same note, not
    mint a twelfth near-duplicate.
    """
    members = _tabular_family(src)
    if len(members) > 1 and not _family_stem(members):
        members = [src]
    md = _duckdb_profile(src, members)

    from silica.driver import DRIVER
    from silica.kernel.vault_manifest import active_inbox_dir

    inbox = active_inbox_dir() or "Inbox"
    stem = _family_stem(members) if len(members) > 1 else src.stem
    fm = _provenance_fm(src, md, tabular_members=members)
    note_rel = f"{inbox}/{stem}.md"
    DRIVER.upsert(note_rel, fm + md.lstrip("\n"))  # re-converting any member refreshes it
    return [note_rel]


# --- legacy office: the LibreOffice hop -------------------------------------


def soffice_bin() -> str | None:
    """Path to LibreOffice's headless entry point, or None.

    `soffice` first: on Debian/Ubuntu `libreoffice` is a shell wrapper around it,
    and one less shell in a subprocess that already hangs is worth having.
    """
    return shutil.which("soffice") or shutil.which("libreoffice")


def soffice_flavour(exe: str | None = None) -> tuple[str, str]:
    """("libreoffice" | "openoffice" | "unknown", product string).

    Read from `bootstraprc` beside the resolved binary, which carries
    `ProductKey=OpenOffice 4.1.16` / `ProductKey=LibreOffice 25.x`. A file read
    and no subprocess: the one thing we must not do to identify the suite is
    *run* it, because running Apache OpenOffice with an unknown option is what
    opens the wizard in the first place.
    """
    exe = exe or soffice_bin()
    if not exe:
        return "unknown", ""
    rc = Path(os.path.realpath(exe)).parent / "bootstraprc"
    try:
        for line in rc.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("ProductKey="):
                product = line.split("=", 1)[1].strip()
                low = product.lower()
                if "libreoffice" in low:
                    return "libreoffice", product
                if "openoffice" in low:
                    return "openoffice", product
                return "unknown", product
    except OSError:
        pass
    return "unknown", ""


def _soffice_cmd(exe: str, profile: Path, args: list[str]) -> list[str]:
    """A soffice invocation that opens no window and collides with no session.

    `-env:UserInstallation` is load-bearing twice over. Without it soffice shares
    the user's real profile, so a conversion blocks on the lock held by their
    open Writer window. With it, the profile is new — which is itself what tells
    AOO it is a first run, so `-nofirststartwizard` is not optional here.
    """
    return [
        exe, *_SOFFICE_QUIET, f"-env:UserInstallation=file://{profile}", *args,
    ]


def _soffice_tail(proc) -> str:
    """Last meaningful stderr/stdout line, minus the known noise."""
    lines = [
        ln.strip() for ln in (proc.stderr or proc.stdout or "").splitlines()
        if ln.strip() and not ln.strip().startswith(_SOFFICE_NOISE)
    ]
    return lines[-1][:200] if lines else ""


def probe_soffice(timeout_s: int = _SOFFICE_PROBE_TIMEOUT_S) -> tuple[str, str]:
    """(status, detail): "missing" | "unsupported" | "hung" | "broken" | "ok".

    Agent-Reach's probe taxonomy rather than a boolean, because each answer needs
    a different sentence from the user's point of view: install it, install a
    *different* one, fix a stuck install, read the error.

    "unsupported" is the one this machine taught us. Apache OpenOffice is a
    perfectly working office suite that answers `which` and starts fine, and it
    still cannot do this job, because it never implemented `-convert-to`. A
    boolean probe would have called it healthy and let the conversion open a
    dialog to discover otherwise.

    `-terminate_after_init` is the cheapest thing that proves the binary starts
    AND exits without a window: measured 0.8 s cold on AOO 4.1.16.
    """
    exe = soffice_bin()
    if not exe:
        return "missing", "no soffice/libreoffice on PATH"
    flavour, product = soffice_flavour(exe)
    if flavour == "openoffice":
        return "unsupported", (
            f"{product or 'Apache OpenOffice'} at {exe} has no headless "
            "`-convert-to`; LibreOffice is the build that converts"
        )
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                _soffice_cmd(exe, Path(tmp) / "profile", ["-terminate_after_init"]),
                capture_output=True, text=True, timeout=timeout_s,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return "hung", (
                f"{exe} did not start and exit within {timeout_s}s — it is waiting "
                "on something (a dialog, or another instance's profile lock)"
            )
        except OSError as e:
            return "broken", f"{exe} could not be run: {e}"
    if proc.returncode != 0:
        return "broken", f"{exe} exited {proc.returncode}: {_soffice_tail(proc)}"
    return "ok", product or exe


def _legacy_office_to_pdf(src: Path, workdir: Path) -> Path:
    """`soffice → pdf`, with its own profile and a hard timeout.

    The timeout is the point. omniparse runs the same conversion as
    `subprocess.run(cmd, check=True)` with no timeout at all, which on a machine
    whose LibreOffice hangs (measured, see LEGACY_OFFICE_EXTS) blocks the caller
    forever with no output and no error.
    """
    # Both remaining formats have a free way out, so the errors below lead with
    # it rather than with the 240 MB install: `.doc`/`.ppt` re-saved as
    # `.docx`/`.pptx` are read by mammoth and mineru with nothing extra.
    # Lowercased because dispatch is: `Report.DOC` reaches here and was being
    # told to re-save a Word document as `.pptx`.
    ooxml = ".docx" if src.suffix.lower() == ".doc" else ".pptx"
    exe = soffice_bin()
    if not exe:
        raise ValueError(
            f"converting {src.suffix} needs LibreOffice — re-save the file as "
            f"{ooxml} or PDF (no install needed), or install it "
            "(`apt install libreoffice` / `brew install --cask libreoffice`)"
        )
    flavour, product = soffice_flavour(exe)
    if flavour == "openoffice":
        # Refused up front, not attempted and timed out: AOO answers an unknown
        # option by starting its GUI and opening the first-start wizard, and a
        # dialog on the user's screen is not something a timeout can undo.
        raise ValueError(
            f"{product or 'Apache OpenOffice'} cannot convert {src.name}: it has "
            "no headless `-convert-to` (the option does not exist in the 4.1 "
            f"line). Re-save the file as {ooxml} or PDF, or install LibreOffice "
            "alongside it"
        )
    out = workdir / "soffice"
    out.mkdir(parents=True, exist_ok=True)
    cmd = _soffice_cmd(
        exe, workdir / "profile",
        ["-convert-to", "pdf", "-outdir", str(out), str(src)],
    )
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_SOFFICE_TIMEOUT_S,
            stdin=subprocess.DEVNULL,  # never let it wait on a terminal
            preexec_fn=_REAP_WITH_PARENT,
        )
    except subprocess.TimeoutExpired:
        status, detail = probe_soffice()
        raise ValueError(
            f"LibreOffice timed out after {_SOFFICE_TIMEOUT_S}s converting "
            f"{src.name} (probe: {status} — {detail}). Run `silica doctor`, or "
            "re-save the file as PDF"
        ) from None
    pdfs = sorted(out.glob("*.pdf"))
    if proc.returncode != 0 or not pdfs:
        detail = _soffice_tail(proc) or f"exit {proc.returncode}, no PDF written"
        raise ValueError(f"LibreOffice could not convert {src.name}: {detail}")
    return pdfs[0]


def _via_legacy_office(src: Path, workdir: Path) -> tuple[str, Path]:
    """Legacy/ODF → PDF → whichever PDF provider is configured.

    Deliberately routed back through the seam rather than pinned to the default:
    the intermediate is a real PDF, so a user who installed mineru for OCR gets
    it here too.
    """
    pdf = _legacy_office_to_pdf(src, workdir)
    provider = PDF_PROVIDERS.get(resolve_pdf_provider(CONFIG.pdf_provider))
    if provider is None:
        raise ValueError(
            f"unknown pdf_provider {CONFIG.pdf_provider!r} "
            f"(known: {', '.join(PDF_PROVIDERS)})"
        )
    return provider(pdf, workdir)


# --- media: ffmpeg demux + a speech-to-text provider ------------------------


def _media_to_wav(src: Path, workdir: Path) -> Path:
    """Any container → 16 kHz mono wav, the one input every ASR accepts.

    One ffmpeg call for audio and video alike: `-vn` drops a video stream that
    may not be there, which is cheaper than branching on the extension. Not
    moviepy (omniparse's choice): that is an imageio + numpy tree to run a
    binary Silica already needs on PATH for the YouTube lane.
    """
    exe = shutil.which("ffmpeg")
    if not exe:
        raise ValueError(
            "reading audio/video needs ffmpeg on PATH — install it with your "
            "package manager (`apt install ffmpeg` / `brew install ffmpeg`)"
        )
    wav = workdir / "audio.wav"
    try:
        proc = subprocess.run(
            [exe, *_FFMPEG_ARGS, "-i", str(src), "-vn",
             "-ac", "1", "-ar", _ASR_SAMPLE_RATE, "-f", "wav", str(wav)],
            capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S,
            preexec_fn=_REAP_WITH_PARENT,
        )
    except subprocess.TimeoutExpired:
        raise ValueError(f"ffmpeg timed out after {_FFMPEG_TIMEOUT_S}s on {src.name}") from None
    if proc.returncode != 0 or not wav.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1][:200] if tail else f"exit {proc.returncode}"
        raise ValueError(f"ffmpeg could not read {src.name}: {detail}")
    if wav.stat().st_size <= 44:  # a wav header and no samples
        raise ValueError(f"{src.name} carries no audio track")
    return wav


def _asr_base(url: str) -> str:
    """Normalise a configured base URL to one ending in `/v1`.

    Users paste both shapes, and a doubled or missing `/v1` is a 404 whose
    message says nothing about the cause.
    """
    base = url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _asr_via_endpoint(wav: Path) -> str:
    """OpenAI-compatible `/v1/audio/transcriptions`, asking for VTT.

    VTT rather than the default json: the cue timings are what
    `vtt_to_text(paragraph_gap_s=...)` turns into paragraph breaks, and a
    transcript with no paragraph breaks is one oversized inbox note. The text is
    thrown away either way, only the boundaries survive.
    """
    import httpx

    from silica.sources.web_fetch import vtt_to_text

    url = f"{_asr_base(CONFIG.stt_base_url)}/audio/transcriptions"
    data = {"model": CONFIG.stt_model, "response_format": "vtt"}
    lang = CONFIG.stt_lang.strip()
    # "auto" is the dictation lane's way of asking whisper-server to detect;
    # here that is spelled by omitting the field, which is what this lane has
    # always sent when unset.
    if lang and lang != "auto":
        data["language"] = lang
    try:
        with wav.open("rb") as fh:
            resp = httpx.post(
                url,
                files={"file": (wav.name, fh, "audio/wav")},
                data=data,
                headers={"Authorization": f"Bearer {CONFIG.stt_api_key}"},
                timeout=_ASR_TIMEOUT_S,
            )
    except httpx.HTTPError as e:
        raise ValueError(
            f"no transcription server at {url}: {e}. Start one (whisper.cpp: "
            "`whisper-server -m <model>`), point SILICA_STT_BASE_URL at it, or "
            "set SILICA_STT_PROVIDER=whispercpp"
        ) from None
    if resp.status_code != 200:
        raise ValueError(f"transcription server returned {resp.status_code}: {resp.text[:200]}")
    body = resp.text
    # A server that ignored response_format answers json; read the text out of it
    # rather than writing `{"text": ...}` into the vault.
    if body.lstrip().startswith("{"):
        try:
            body = str(json.loads(body).get("text", ""))
        except ValueError:
            pass
    return vtt_to_text(body, paragraph_gap_s=_ASR_PARAGRAPH_GAP_S)


def _asr_via_whispercpp(wav: Path) -> str:
    """Local whisper.cpp binary, for a machine with no server running."""
    from silica.sources.web_fetch import vtt_to_text

    configured = CONFIG.stt_whispercpp_bin.strip()
    # Upstream renamed `main` to `whisper-cli` in 2024; accept either.
    exe = shutil.which(configured) if configured else (
        shutil.which("whisper-cli") or shutil.which("whisper.cpp")
    )
    if not exe:
        raise ValueError(
            "whisper.cpp not found — set SILICA_STT_WHISPERCPP_BIN to its path, "
            "or use SILICA_STT_PROVIDER=endpoint with a transcription server"
        )
    model = CONFIG.stt_whispercpp_model.strip()
    if not model:
        raise ValueError(
            "whisper.cpp needs a model file — set SILICA_STT_WHISPERCPP_MODEL "
            "(e.g. models/ggml-base.bin); it has no default it can find itself"
        )
    out = wav.with_suffix("")  # whisper.cpp appends .vtt itself
    cmd = [exe, "-m", model, "-f", str(wav), "-ovtt", "-of", str(out), "-np"]
    if CONFIG.stt_lang.strip() and CONFIG.stt_lang.strip() != "auto":
        cmd += ["-l", CONFIG.stt_lang.strip()]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=_ASR_TIMEOUT_S, preexec_fn=_REAP_WITH_PARENT)
    except subprocess.TimeoutExpired:
        raise ValueError(f"whisper.cpp timed out after {_ASR_TIMEOUT_S}s") from None
    vtt = out.with_suffix(".vtt")
    if proc.returncode != 0 or not vtt.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise ValueError(f"whisper.cpp failed: {tail[-1][:200] if tail else proc.returncode}")
    return vtt_to_text(vtt.read_text(encoding="utf-8", errors="replace"),
                       paragraph_gap_s=_ASR_PARAGRAPH_GAP_S)


ASR_PROVIDERS = {
    "endpoint": _asr_via_endpoint,
    "whispercpp": _asr_via_whispercpp,
}


def _via_asr(src: Path, workdir: Path) -> tuple[str, Path]:
    """Media provider, same `(markdown, images_dir)` contract as the rest.

    Returning into the shared tail is the whole point: a transcript then gets
    sanitizing, paragraph segmentation, `source_file` provenance and the inbox
    write for free, and a three-hour talk is split like a book instead of
    landing as one note RECON caps at 40 concepts.
    """
    if CONFIG.stt_provider not in ASR_PROVIDERS:
        raise ValueError(
            f"unknown stt_provider {CONFIG.stt_provider!r} "
            f"(known: {', '.join(ASR_PROVIDERS)})"
        )
    wav = _media_to_wav(src, workdir)
    text = ASR_PROVIDERS[CONFIG.stt_provider](wav)
    images = workdir / "images"
    images.mkdir(parents=True, exist_ok=True)  # none, but the tail expects a dir
    if not text.strip():
        # Empty, NOT a bare heading: the shared empty guard is what turns this
        # into an error, and a heading would make silence look like success.
        return "", images
    # A heading gives the segmenter a cut point and the note a title line; the
    # stem is all the metadata a bare media file carries.
    return f"# {src.stem}\n\n{text}\n", images


def _mineru_error(stderr: str) -> str:
    """One-line, human-readable error from mineru's stderr.

    mineru may write a JSON task blob (with an ``error`` field) or a loguru
    stream. Pull the ``error`` field when present. Otherwise the cause is NOT
    at the head: this mineru version spins up an internal ``mineru-api`` server
    and floods stderr with startup logs + tqdm bars before any work, so
    head-truncating just surfaces "Started local mineru-api ...". Drop
    INFO/progress noise, then return the last error-ish line (else the last
    meaningful line) — a Python traceback puts "XError: msg" last too.

    One cause is named outright rather than relayed: see `_MINERU_SIX_RE`.
    """
    err = stderr.strip()
    if _MINERU_SIX_RE.search(err):
        return "mineru needs `six`, which it does not declare — `pip install six`, then retry"
    try:
        parsed = json.loads(err)
        return str(parsed.get("error") or err[:300])
    except (ValueError, AttributeError):
        pass
    # The task blob is usually EMBEDDED in the final "Error: N task(s) failed"
    # line, where its "error" field sits past any truncation window — pull it
    # straight out (last match = final task).
    fields = re.findall(r'"error":\s*"((?:[^"\\]|\\.)+)"', err)
    if fields:
        return fields[-1][:300]
    lines = [
        ln.strip() for ln in err.splitlines()  # \r-split too: tqdm bars separate
        if ln.strip() and not _MINERU_NOISE_RE.search(ln)
    ]
    if not lines:
        return err[:300]
    hits = [ln for ln in lines if _MINERU_ERR_RE.search(ln)]
    return (hits[-1] if hits else lines[-1])[:300]


def _pdf_via_mineru(src: Path, workdir: Path) -> tuple[str, Path]:
    """The OCR provider, and the only backend for `IMG_EXTS` / `OFFICE_EXTS`.

    Input-agnostic on purpose: the CLI sniffs the file's content and picks its
    own pipeline (`auto/` for pdf and images, `office/` for pptx/xlsx), and the
    recursive glob below finds the markdown under either.
    """
    out = workdir / "out"
    try:
        proc = subprocess.run(
            ["mineru", "-p", str(src), "-o", str(out), "-b", _MINERU_BACKEND, *_MINERU_ARGS],
            capture_output=True, text=True, timeout=_MINERU_TIMEOUT_S,
            preexec_fn=_REAP_WITH_PARENT,  # killing the batch must free the GPU
        )
    except FileNotFoundError:
        raise ValueError(
            "mineru not installed — `pip install 'mineru[pipeline]'`, "
            "or set SILICA_PDF_PROVIDER to docling/opendataloader"
        ) from None
    if proc.returncode != 0:
        raise ValueError(f"mineru failed: {_mineru_error(proc.stderr)}")
    # The stem is the user's filename: `Smith [2020].pdf` is a character class,
    # and an unescaped one matches nothing after an hour of OCR, so the output is
    # thrown away with the tempdir and the user is told mineru produced nothing.
    stem = glob_escape(src.stem)
    hits = glob(str(out / stem / "**" / f"{stem}.md"), recursive=True)
    if not hits:
        raise ValueError("mineru produced no markdown")
    md_path = Path(hits[0])
    return md_path.read_text(encoding="utf-8", errors="replace"), md_path.parent / "images"


PDF_PROVIDERS = {
    "pdfium": _via_pdfium,
    "docling": _pdf_via_docling,
    "mineru": _pdf_via_mineru,
    "opendataloader": _pdf_via_opendataloader,
}

# Names the setting used to take, recognized forever: `SILICA_PDF_PROVIDER=pymupdf`
# sits in ~/.silica/.env files written before 2026-08-31, and a pin must not turn
# into an "unknown provider" error because the default changed under it. Kept out
# of PDF_PROVIDERS so the settings enum and the doctor row list real names only.
PDF_PROVIDER_ALIASES = {"pymupdf": "pdfium"}


def resolve_pdf_provider(name: str) -> str:
    """The canonical provider name for a configured value (aliases mapped,
    whitespace trimmed); an unknown name comes back unchanged for the caller
    to report."""
    name = (name or "").strip()
    return PDF_PROVIDER_ALIASES.get(name, name)


# --- shared helpers ---------------------------------------------------------

def _source_date(src: Path) -> str | None:
    """Creation date the document itself declares, ISO `YYYY-MM-DD`; None when
    it declares none.

    Feeds rung 2 of the event-clock precedence (kernel/provenance
    `source_event_date` reads the converted note's `date:`), so the FSM dates
    claims by the document instead of the run. Metadata the format states, and
    nothing else: no mtime fallback — a download's mtime is the download, not
    the document. Never worth failing a conversion over, hence the blanket
    except.
    """
    import datetime
    try:
        suffix = src.suffix.lower()
        if suffix in (*OFFICE_EXTS, ".docx"):
            # OOXML core properties: <dcterms:created>2024-04-02T09:00:00Z</…>
            with zipfile.ZipFile(src) as z:
                xml = _zip_member(z, "docProps/core.xml").decode("utf-8", "replace")
            m = re.search(r"<dcterms:created[^>]*>(\d{4})-(\d{2})-(\d{2})", xml)
        elif suffix == ".pdf":
            # Info dictionary date, "D:20240402093000+02'00'"
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(src))
            try:
                stamp = pdf.get_metadata_value("CreationDate") or ""
            finally:
                pdf.close()
            m = re.search(r"(\d{4})(\d{2})(\d{2})", stamp)
        elif suffix == ".epub":
            # Dublin Core <dc:date>2019-05-06T00:00:00Z</dc:date>
            m = re.search(r"(\d{4})-(\d{2})-(\d{2})", _epub_metadata(src).get("date", ""))
        else:
            return None
        if not m:
            return None
        # fromisoformat validates ranges: a "D:00000000"-style stamp dies here.
        return datetime.date.fromisoformat("-".join(m.groups())).isoformat()
    except Exception:
        return None


# A DOI or arXiv id stated near the top of the document. Scanned on the first
# few KB only: a references section is full of OTHER papers' DOIs, and the
# document's own identifier sits on page one.
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>\])}]+)")
_ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)


def _doc_citation(src: Path, md_text: str) -> dict[str, str]:
    """Citation fields the document itself declares: doi/arxiv from the first
    page's text, authors/title from the format's own metadata. Absent fields
    are absent keys — never guessed, never worth failing a conversion over."""
    cite: dict[str, str] = {}
    try:
        suffix = src.suffix.lower()
        title = authors = ""
        if suffix == ".pdf":
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(src))
            try:
                title = pdf.get_metadata_value("Title") or ""
                authors = pdf.get_metadata_value("Author") or ""
            finally:
                pdf.close()
        elif suffix == ".docx":
            # OOXML core properties: <dc:title>, <dc:creator>
            import html as _html

            with zipfile.ZipFile(src) as z:
                xml = _zip_member(z, "docProps/core.xml").decode("utf-8", "replace")
            mt = re.search(r"<dc:title[^>]*>(.*?)</dc:title>", xml, re.S)
            mc = re.search(r"<dc:creator[^>]*>(.*?)</dc:creator>", xml, re.S)
            title = _html.unescape(mt.group(1)) if mt else ""
            authors = _html.unescape(mc.group(1)) if mc else ""
        elif suffix == ".epub":
            meta = _epub_metadata(src)
            title, authors = meta.get("title", ""), meta.get("creator", "")
        if title.strip():
            cite["source_title"] = " ".join(title.split())
        if authors.strip():
            cite["authors"] = " ".join(authors.split())
    except Exception:
        pass
    head = md_text[:8000]
    m = _DOI_RE.search(head)
    if m:
        cite["doi"] = m.group(1).rstrip(".,;")
    m = _ARXIV_RE.search(head)
    if m:
        cite["arxiv"] = m.group(1) + (m.group(2) or "")
    return cite


def _provenance_fm(src: Path, md_text: str = "",
                   tabular_members: list[Path] | None = None) -> str:
    """Frontmatter block naming the converted file's real origin (absolute
    path), the document's own creation date when it states one, and the
    citation fields it declares (doi/arxiv/authors/title) — what a researcher
    needs to cite the source without reopening the original file."""
    quoted = str(src).replace("\\", "\\\\").replace('"', '\\"')
    date = _source_date(src)
    lines = [f"date: {date}\n"] if date else []
    # Form stamp (docs/specs/nucleation-forms.md): a converted media file is a
    # transcript by construction, so the profile ladder never has to sniff it.
    # A document's form is not knowable at ingress; it gets no stamp.
    if src.suffix.lower() in MEDIA_EXTS:
        lines.append("form: transcript\n")
    # A tabular profile documents a LIVE file: the rows stay on disk and the
    # note is the pointer, so the note↔file edge must enter the graph, not
    # just this provenance ledger (source_file: is deliberately not indexed).
    # `documents:` is the one keyspace file_backlinks already reads, and it is
    # repo-relative because an absolute path dies on any other machine;
    # code_ref arms /stale for git-tracked files. A file outside the repo gets
    # no edge — the graph's universe is the repo.
    if src.suffix.lower() in TABULAR_EXTS and CONFIG.vault_path:
        from silica.kernel.code import gitstate
        from silica.kernel.recall import paths as _rpaths

        root = _rpaths.repo_root_for(CONFIG.vault_path)
        if root is not None:
            rels: list[str] = []
            for m in (tabular_members or [src]):
                try:
                    rels.append(m.resolve().relative_to(Path(root).resolve()).as_posix())
                except (ValueError, OSError):
                    pass  # outside the repo: provenance only, no graph edge
            if rels:
                quoted_rels = ", ".join(
                    '"' + r.replace("\\", "\\\\").replace('"', '\\"') + '"' for r in rels
                )
                lines.append(f"documents: [{quoted_rels}]\n")
                head = gitstate.head_ref(root)
                if head:
                    lines.append(f"code_ref: {head}\n")
    for key, val in _doc_citation(src, md_text).items():
        v = str(val).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}: "{v}"\n')
    return f'---\n{"".join(lines)}source_file: "{quoted}"\n---\n\n'


def _resolve_input(target: str) -> Path:
    """Absolute as given; relative tried vault-first, then cwd.

    ProseAdapter.read resolves vault-only because its inputs are notes, which
    live in the vault. A file to CONVERT is the opposite: a PDF sits where the
    user is standing (a download dir, a repo), not among the markdown — so cwd
    is a real fallback here, not just the no-vault special case.
    """
    p = Path(target)
    if not p.is_absolute():
        vault = (CONFIG.vault_path or "").strip()
        tries = ([Path(vault) / target] if vault else []) + [Path.cwd() / target]
        p = next((c for c in tries if c.exists()), tries[-1])
    if not p.exists():
        raise ValueError(f"file not found: {target}")
    return p


def _images_dest(dest_dir: str) -> Path:
    from silica.kernel.vault_manifest import active_inbox_dir

    base = dest_dir.strip() or active_inbox_dir() or "Inbox"
    return Path(CONFIG.vault_path) / base / "Images"


def _image_prefix(src: Path) -> str:
    """Per-source namespace for the flat `Images/` dir.

    Derived from the source STEM, not its content: re-converting the same PDF
    must reproduce the same image names, or every run would leave the previous
    run's figures behind as orphans. Same identity the note path already assumes
    (`{inbox}/{src.stem}.md`), so two sources that collide here already collide
    there.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", src.stem).strip("-_")[:40]
    return slug or "doc"


def _copy_images(
    src_dir: Path,
    dest_dir: Path,
    prefix: str,
    only: set[str] | None = None,
) -> dict[str, str]:
    """Copy referenced images into the flat vault dir, namespaced per source.

    Returns `{original basename: copied basename}` for the link rewrite. The
    prefix is load-bearing: providers name figures by page index
    (`_page_0_Figure_1.jpeg`), which repeats across documents, and both the copy
    and the `![[basename]]` embed are flat — an un-namespaced second PDF would
    overwrite the first's figure AND silently repoint the first note's embed at
    it.
    """
    if not src_dir.is_dir():
        return {}
    files = [
        f for f in src_dir.iterdir()
        if f.is_file() and (only is None or f.name in only)
    ]
    if not files:
        return {}
    dest_dir.mkdir(parents=True, exist_ok=True)
    renamed: dict[str, str] = {}
    for f in files:
        name = f"{prefix}-{f.name}"
        shutil.copy2(f, dest_dir / name)
        renamed[f.name] = name
    return renamed


def _rewrite_image_links(md: str, renamed: dict[str, str] | None = None) -> str:
    """`![alt](any/path/x.png)` → `![[x.png]]` (basename, Obsidian embed).

    `renamed` maps the provider's basename to the namespaced one actually
    copied into the vault; an unmapped basename embeds unchanged.
    """
    def repl(m: "re.Match[str]") -> str:
        base = os.path.basename(m.group(1))
        if not base.lower().endswith(_IMG_EXTS):
            return m.group(0)
        return f"![[{(renamed or {}).get(base, base)}]]"

    return _MD_IMG_RE.sub(repl, md)
