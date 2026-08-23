# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Note templates — template_spoke and patch_snippet.

Migrated AS-IS from hermes_common/templates.py, with the bootstrap path
hack removed (no longer needed — this is a proper Python package now).
"""
import datetime
import logging
import os
import re

import yaml

from silica.kernel.write import frontmatter
from silica.kernel.write.frontmatter import clean_tag  # canonical; do not redefine

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")

# Template-scoped fence split: unlike frontmatter.FM_RE (whose trailing \s*
# swallows every blank line after the closing fence), this stops at the first
# newline so the template author's body spacing passes through unchanged.
_TEMPLATE_FM_RE = re.compile(r"^---[ \t]*\n(.*?)\n---[ \t]*\n", re.DOTALL)


def slugify(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]', '', s)
    s = re.sub(r'\s+', ' ', s)  # normalise newlines, tabs, and multiple spaces to a single space
    return s.strip()


def _link_name(name: str) -> str:
    """Bare note name for a wikilink target — strips brackets the distiller may
    already have wrapped around it, so f'[[{name}]]' never becomes '[[[[X]]]]'
    (a quadruple-bracket frontmatter link Obsidian reads as unresolved), and
    un-escapes a markdown-table pipe (``\\|`` → ``|``): the distiller emits
    aliases like ``[[Target\\|Alias]]``, and ``\\|`` is an invalid escape inside
    the double-quoted YAML scalar we wrap related/hub links in — it breaks
    parsing of the WHOLE frontmatter block (real incident: DQL note, run of
    2026-07-22, rejected as 'Missing or invalid frontmatter')."""
    return name.strip().strip("[]").strip().replace("\\|", "|")


def close_unbalanced_fences(text: str) -> str:
    """Append a closing code fence when the ``` count is odd, mirroring the
    lint's ``body.count('```') % 2`` check (ast._balanced). A distilled snippet
    that opens a fence and never closes it fails post-write lint and gets
    deferred — closing it mechanically lands the note instead of losing it.
    ponytail: balances only the top-level fence count; a snippet that nests
    fences pathologically still needs a real fix."""
    if text.count("```") % 2:
        return text.rstrip() + "\n```\n"
    return text


_ECHOED_H1_RE = re.compile(r"^(\s*(?:<!--.*?-->\s*)*)#\s+(?P<h1>[^\n]+)\n+", re.DOTALL)


def _drop_echoed_title(body: str, h1: str) -> str:
    """Drop the body's opening H1 when it just repeats the note title.

    The template writes `# {h1}` itself, and a distiller that opens its snippet
    with the same heading produced the title twice — inconsistently, since only
    the snippets that happened to include one were affected (13 notes of 24 in
    one real run). A LEADING provenance stamp is stepped over, and an H1 that
    says something DIFFERENT is left alone: that is content, not an echo.

    Compared through `slugify`, because the title reaching here came from a
    filename and lost its punctuation on the way: "Classificazione: approcci
    discriminativo e generativo" in the body against "Classificazione approcci
    discriminativo e generativo" as the title is one heading twice, and an exact
    match read it as two.
    """
    m = _ECHOED_H1_RE.match(body)
    if not m or slugify(m.group("h1")).casefold() != slugify(h1).casefold():
        return body
    return m.group(1).lstrip() + body[m.end():]


def template_spoke(heading: str, snippet: str, hub: str, title: str | None = None, tags: list[str] | None = None, related: list[str] | None = None, parent: str | None = None) -> str:
    today = datetime.date.today().isoformat()
    body = snippet.strip() or "(to be expanded)"
    h1 = title or heading  # title wins: filename and H1 stay in sync

    hub_link = _link_name(hub)
    # parent note link — specific parent overrides hub when provided
    if parent:
        parent_link = _link_name(parent)
        parent_note = f'"[[{parent_link}]]"'
        related_items = [f'"[[{parent_link}]]"', f'"[[{hub_link}]]"']
    else:
        parent_note = f'"[[{hub_link}]]"'
        related_items = [f'"[[{hub_link}]]"']

    # related list
    if related:
        for r in related:
            r_link = f'"[[{_link_name(r)}]]"'
            if r_link not in related_items:
                related_items.append(r_link)

    # tags list
    tag_list = []
    if tags:
        for t in tags:
            ct = clean_tag(t)
            if ct and ct not in tag_list:
                tag_list.append(ct)
    else:
        # default tag derived from hub
        ch = clean_tag(hub)
        if ch:
            tag_list.append(ch)

    # Format YAML components
    related_yaml = "\n".join(f"  - {item}" for item in related_items)
    tags_yaml = "\n".join(f"  - {tag}" for tag in tag_list)

    frontmatter = f"""---
parent note: {parent_note}
related:
{related_yaml}
tags:
{tags_yaml}
last modified: {today}
AI: true
---"""

    return f"""{frontmatter}

# {h1}

{body}
"""


# The historical template_spoke layout as a template string. A vault with no
# templates/ dir and no config renders bit-identically to the old code
# (guarded by the golden parity test).
BUILTIN_TEMPLATE = """---
parent note: {{parent}}
related: {{related}}
tags: {{tags}}
last modified: {{date}}
AI: true
---

# {{title}}

{{body}}
"""


def prepare_fields(*, title: str, body: str, hub: str | None = None,
                   tags: list[str] | None = None,
                   related: list[str] | None = None,
                   parent: str | None = None) -> dict:
    """Encode template_spoke's conditional fallbacks once, for every template.

    Substitution in render_note is pure, so ALL conditional behavior lives
    here: parent falls back to the hub, the hub is merged into related
    (deduplicated), tags default to clean_tag(hub) when empty, date is
    today's ISO date. Values come back ready to substitute — wikilinks
    quoted, tags cleaned. Templates never re-implement these rules.

    Also normalizes body: models drift toward emitting frontmatter at the
    top of markdown regardless of instructions, so a leading YAML block is
    stripped with a warning rather than landing inside the rendered note —
    and so is an opening H1 that only repeats the title the template writes
    itself (see `_drop_echoed_title`).
    """
    m = frontmatter.FM_RE.match(body)
    if m:
        logger.warning("prepare_fields: stripped leading YAML block from body")
        body = body[m.end():]
    body = _drop_echoed_title(body, title)

    hub_link = _link_name(hub) if hub else ""
    parent_link = _link_name(parent) if parent else hub_link
    # A note is never its own parent or its own relative: the auto-created hub
    # op carries hub=<its own name>, which stamped every hub note with a
    # self-link that self-attests ("parent note: [[appunti]]" on appunti.md).
    if parent_link == _link_name(title):
        parent_link = ""
    if hub_link == _link_name(title):
        hub_link = ""

    related_items: list[str] = []
    if parent_link:
        related_items.append(f'"[[{parent_link}]]"')
    if hub_link and f'"[[{hub_link}]]"' not in related_items:
        related_items.append(f'"[[{hub_link}]]"')
    for r in related or []:
        r_link = f'"[[{_link_name(r)}]]"'
        if r_link not in related_items:
            related_items.append(r_link)

    tag_list: list[str] = []
    for t in tags or []:
        ct = clean_tag(t)
        if ct and ct not in tag_list:
            tag_list.append(ct)
    if not tag_list and hub_link:
        ch = clean_tag(hub_link)
        if ch:
            tag_list.append(ch)

    return {
        "title": title,
        "body": close_unbalanced_fences(body.strip()) or "(to be expanded)",
        "tags": tag_list,
        "related": related_items,
        "parent": f'"[[{parent_link}]]"' if parent_link else "",
        "hub": f'"[[{hub_link}]]"' if hub_link else "",
        "date": datetime.date.today().isoformat(),
    }


def render_note(template_source: str, fields: dict) -> str:
    """Logic-free {{placeholder}} substitution over a whole-note skeleton.

    Line-aware over the frontmatter block only: a frontmatter line whose
    placeholder resolves empty is dropped, and a list value expands to a
    YAML block sequence at its key (empty list drops the key line). In the
    body, placeholders substitute in place. Unknown placeholders are
    removed with a warning — they never block the write.
    """
    def _lookup(name: str):
        if name not in fields:
            logger.warning("render_note: unknown placeholder {{%s}} — removed", name)
            return None
        return fields[name]

    def _sub_all(text: str) -> str:
        return _PLACEHOLDER_RE.sub(lambda m: str(_lookup(m.group(1)) or ""), text)

    m = _TEMPLATE_FM_RE.match(template_source)
    if not m:
        return _sub_all(template_source)

    out: list[str] = []
    for line in m.group(1).split("\n"):
        ph = _PLACEHOLDER_RE.search(line)
        if not ph:
            out.append(line)
            continue
        val = _lookup(ph.group(1))
        if isinstance(val, list):
            if not val:
                continue
            out.append(line[:ph.start()].rstrip())
            out.extend(f"  - {item}" for item in val)
        elif val is None or str(val) == "":
            continue
        else:
            out.append(_sub_all(line))
    return "---\n" + "\n".join(out) + "\n---\n" + _sub_all(template_source[m.end():])


def _bad_template_name(name: str) -> bool:
    """True if a template name contains path separators, traversal sequences, or drive letters.

    Rejects: path separators ("/", "\\"), parent-dir traversal (".."), and drive-relative
    names (":") which would re-anchor pathlib operations to escape the templates dir.
    """
    return "/" in name or "\\" in name or ".." in name or ":" in name


class TemplateNotFoundError(ValueError):
    """Explicit template name that does not resolve — fails loudly."""


def _read_template(path) -> str | None:
    """Template source, or None when missing or malformed (a file that opens
    a frontmatter fence and never closes it)."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    source = source.replace("\r\n", "\n")
    if source.startswith("---") and not _TEMPLATE_FM_RE.match(source):
        return None
    return source


def resolve_template(name: str | None = None) -> str:
    """Resolution order: explicit name > vault.yaml default_template > built-in.

    An explicit name that is missing/malformed raises TemplateNotFoundError
    listing the available templates; a broken vault default degrades to the
    built-in with a warning — ingestion never stops for a broken template.
    """
    from pathlib import Path

    from silica.config import CONFIG
    from silica.kernel.vault_manifest import get_active_manifest

    conv = get_active_manifest().conventions
    tdir = Path((getattr(CONFIG, "vault_path", "") or "").strip()) / conv.templates_dir
    if name:
        if _bad_template_name(name):
            raise TemplateNotFoundError(
                f"invalid template name {name!r} — names must not contain path separators, '..' or ':'")
        source = _read_template(tdir / f"{name}.md")
        if source is None:
            available = sorted(p.stem for p in tdir.glob("*.md")) if tdir.is_dir() else []
            raise TemplateNotFoundError(
                f"template '{name}' not found or malformed in '{tdir}' — "
                f"available: {', '.join(available) or 'none'}")
        return source
    if conv.default_template:
        if _bad_template_name(conv.default_template):
            logger.warning("invalid vault default template name %r — names must not contain path separators, '..' or ':' — using built-in",
                           conv.default_template)
            return BUILTIN_TEMPLATE
        source = _read_template(tdir / f"{conv.default_template}.md")
        if source is not None:
            return source
        logger.warning("vault default template %r missing or malformed — using built-in",
                       conv.default_template)
    return BUILTIN_TEMPLATE


def ensure_hub_link(content: str, hub: str | None) -> str:
    """Guarantee the hub wikilink in a note's frontmatter `related:` list.

    No-op when hub is falsy or any casing/alias form of the link is already
    present. Callers: patch_snippet (fresh append) and the duplicate-block
    branch of _execute_patch — the repair must land even when the snippet
    itself is skipped, or lint fails the op forever."""
    from silica.kernel.link.ofm import has_wikilink
    if not hub or has_wikilink(content, hub):
        return content
    if content.startswith("---\n"):
        end_idx = content.find("\n---\n", 4)
        if end_idx == -1:
            return content
        if "\nrelated:\n" in content[:end_idx]:
            parts = content.split("\nrelated:\n", 1)
            return parts[0] + f'\nrelated:\n  - "[[{hub}]]"\n' + parts[1]
        return content[:end_idx] + f'\nrelated:\n  - "[[{hub}]]"' + content[end_idx:]
    today = datetime.date.today().isoformat()
    frontmatter = f"""---
parent note: "[[{hub}]]"
related:
  - "[[{hub}]]"
last modified: {today}
AI: true
---
"""
    return frontmatter + content


def patch_snippet(heading: str, snippet: str, source_basename: str, hub: str | None = None, existing_content: str | None = None, valid_from: str | None = None) -> str:
    # No valid_from → no stamp line and the block is byte-identical to what
    # every pre-stamp write produced. Only the FSM supplies one today.
    stamp_line = ""
    if valid_from:
        from silica.kernel.write.contested import stamp
        rendered = stamp(valid_from=valid_from)
        if rendered:
            stamp_line = f"{rendered}\n\n"
    patch_text = f"""

{provenance_header(heading, source_basename)}

{stamp_line}{close_unbalanced_fences(snippet.strip())}
"""
    if existing_content is not None:
        from silica.kernel.write.contested import append_before_superseded
        existing_content = ensure_hub_link(existing_content, hub)
        return append_before_superseded(existing_content, patch_text)

    return patch_text


_AI_KEY_RE = re.compile(r"^AI:\s", re.MULTILINE)


def ensure_ai_flag(content: str, value: str = "true") -> str:
    """Stamp `AI: <value>` into an existing frontmatter block lacking the field.

    patch/overwrite touch user-authored notes that predate the `AI` convention;
    the OFM lint (ofm.py) requires the `AI` key on the *whole* note, so a patch
    to a legacy note would be reverted. The value says how much the agent now
    owns: `true` for a body the agent wrote (overwrite, floor under fresh
    writes), `partial` for a section appended to a note that was the user's —
    the bulk patch path passes it so the note keeps its human reliability tier
    (contested.reliability_tier) instead of decaying on the first touch.
    String-level (no YAML round-trip) so the rest of the user's frontmatter is
    left byte-for-byte intact.

    No-ops when there is no frontmatter (fresh writes carry it via template_spoke)
    or the `AI` key already exists (never overwrites the user's own value, and
    never upgrades an earlier `partial` to `true`).
    """
    if not content.startswith("---\n"):
        return content
    end_idx = content.find("\n---\n", 4)
    if end_idx == -1:
        return content  # unterminated frontmatter — leave for the lint to flag
    if _AI_KEY_RE.search(content[4:end_idx]):
        return content
    return content[:end_idx] + f"\nAI: {value}" + content[end_idx:]


_LAST_MODIFIED_RE = re.compile(r"^last modified:.*$", re.MULTILINE)
_AGENT_KEY_RE = re.compile(r"^agent:.*$", re.MULTILINE)


def _stamp_agent(content: str) -> str:
    """Set/refresh `agent: "<id>"` in the frontmatter head when SILICA_AGENT_ID
    is set — provenance for a vault written by a fleet of agents.

    Last-writer-wins, exactly like `last modified`: the field names who last
    touched the note; git keeps the full authorship history. Unset env → the
    field is never added and any existing one is left intact, so single-user
    writes are byte-for-byte unchanged. The value is quoted and escaped so a
    stray value can never break or inject YAML.
    """
    agent = os.environ.get("SILICA_AGENT_ID", "").strip()
    if not agent or not content.startswith("---\n"):
        return content
    end = content.find("\n---\n", 4)
    if end == -1:
        return content
    val = agent.splitlines()[0].replace("\\", "\\\\").replace('"', '\\"')
    line = f'agent: "{val}"'
    head = content[4:end]
    if _AGENT_KEY_RE.search(head):
        return "---\n" + _AGENT_KEY_RE.sub(line, head, count=1) + content[end:]
    return content[:end] + "\n" + line + content[end:]


# `[ \t]*`, not `[ \t]+`: a top-level sequence is FLUSH LEFT the moment the
# note has been through `frontmatter.dump`, because that is how yaml.safe_dump
# writes one, and any property edit re-emits the block. Requiring indentation
# deleted the key line and left its items orphaned under the previous scalar,
# which is not YAML: 2 of 52 notes unreadable on the 2026-08-23 nucleate batch,
# and `split` returning None there made `stamp_citation` re-append every key.
_DOCS_BLOCK_RE = re.compile(r"^documents:[^\n]*(?:\n[ \t]*-[^\n]*)*\n?", re.MULTILINE)
_CODE_REF_RE = re.compile(r"^code_ref:.*$", re.MULTILINE)


def stamp_documents(content: str, documents: list[str], code_ref: str | None = None) -> str:
    """Splice `documents:` (and `code_ref:`) into the frontmatter head.

    String-level like `_stamp_agent`, deliberately not `prepare_fields`: the
    binding has to land on the `template="none"` branch too, and going through
    the template renderer would demand a `{{documents}}` placeholder in every
    template. An existing block is replaced by the union, prior entries first,
    so patching a note adds a binding instead of dropping the old ones.
    """
    if not documents or not content.startswith("---\n"):
        return content
    end = content.find("\n---\n", 4)
    if end == -1:
        return content  # unterminated frontmatter — leave for the lint to flag
    from silica.kernel.code.codedocs import documents_of

    data, _, _ = frontmatter.split(content)
    merged = list(dict.fromkeys(documents_of(data if isinstance(data, dict) else {})
                                + list(documents)))
    head, tail = _DOCS_BLOCK_RE.sub("", content[4:end]).rstrip("\n"), content[end:]
    lines = "".join('  - "%s"\n' % p.replace("\\", "\\\\").replace('"', '\\"')
                    for p in merged)
    head = f"{head}\ndocuments:\n{lines}".rstrip("\n")
    if code_ref:
        line = f"code_ref: {code_ref}"
        head = (_CODE_REF_RE.sub(line, head, count=1)
                if _CODE_REF_RE.search(head) else f"{head}\n{line}")
    return "---\n" + head + tail


_SOURCES_BLOCK_RE = re.compile(r"^sources:[^\n]*(?:\n[ \t]*-[^\n]*)*\n?", re.MULTILINE)


def stamp_sources(content: str, source_basename: str) -> str:
    """Splice the source basename into a `sources:` frontmatter list.

    Same string-level mechanism and union semantics as `stamp_documents`: a
    note written by one source and patched by two more accumulates all three.
    Entries are plain quoted strings, never wikilinks — a link to the archived
    source would enter the graph (degree-N source nodes, dangling links when
    `done/` is pruned, bare-name resolution colliding with the note itself);
    an inert string carries the provenance without any of that. The content
    hash stays ledger-side (provenance.json) on purpose.
    """
    if not source_basename or not content.startswith("---\n"):
        return content
    end = content.find("\n---\n", 4)
    if end == -1:
        return content  # unterminated frontmatter — leave for the lint to flag
    data, _, _ = frontmatter.split(content)
    prior = data.get("sources") if isinstance(data, dict) else None
    if isinstance(prior, str):
        prior = [prior]
    merged = list(dict.fromkeys(
        [str(s) for s in (prior or []) if s] + [source_basename]
    ))
    head, tail = _SOURCES_BLOCK_RE.sub("", content[4:end]).rstrip("\n"), content[end:]
    lines = "".join('  - "%s"\n' % s.replace("\\", "\\\\").replace('"', '\\"')
                    for s in merged)
    head = f"{head}\nsources:\n{lines}".rstrip("\n")
    return "---\n" + head + tail


def stamp_citation(content: str, cite: dict[str, str]) -> str:
    """Splice the source's citation fields (doi/arxiv/authors/source_title)
    into a note's frontmatter, prefixed `source_` where not already.

    Same string-level mechanism as `stamp_sources`. First writer wins per key:
    a note fed by two sources keeps the first source's identifiers rather than
    silently swapping them. Values are inert quoted strings — like `sources:`,
    citation metadata must not enter the graph.
    """
    if not cite or not content.startswith("---\n"):
        return content
    end = content.find("\n---\n", 4)
    if end == -1:
        return content  # unterminated frontmatter — leave for the lint to flag
    data, _, _ = frontmatter.split(content)
    existing = set(data.keys()) if isinstance(data, dict) else set()
    head, tail = content[4:end].rstrip("\n"), content[end:]
    for key, val in cite.items():
        out_key = key if key.startswith("source_") else f"source_{key}"
        if out_key in existing:
            continue
        v = str(val).replace("\\", "\\\\").replace('"', '\\"')
        head = f'{head}\n{out_key}: "{v}"'
    return "---\n" + head + tail


# Every character YAML scans as a line break. `\n` and `\r` are the obvious
# two; NEL, LINE SEPARATOR and PARAGRAPH SEPARATOR are the ones a threat model
# built on "no newlines" misses, and the parser this vault reads with honours
# all five (verified against the frontmatter round-trip, not from the spec).
_YAML_BREAKS = "\n\r\x85\u2028\u2029"


def _yaml_scalar(value) -> str:
    """`value` as ONE line of YAML, round-trip exact.

    Quotes only what YAML would otherwise misread, so a plain word stays plain
    and the frontmatter a person reads is unchanged. Two ways out of one line
    are closed: `width` is set past any real value, because the emitter folds
    long scalars across lines by default; and anything carrying a line break is
    forced to double quotes, because the single-quoted style writes those as
    REAL lines (`'a\\n\\n  b'`), and the replace branch in `upsert_props` — one
    regex over one line — would later leave the continuation behind as an
    orphan that folds into whatever value replaced it.
    """
    text = str(value)
    style = '"' if any(c in text for c in _YAML_BREAKS) else None
    dumped = yaml.safe_dump({"k": text}, allow_unicode=True, sort_keys=False,
                            width=10 ** 9, default_flow_style=False,
                            default_style=style)
    # `default_style` quotes the key too, so slice on the separator rather than
    # a fixed width: neither `k` nor `"k"` can contain one.
    return dumped.split(": ", 1)[1].rstrip("\n")


def upsert_props(content: str, props: dict[str, str]) -> str:
    """Insert (or replace) scalar caller-supplied keys in the frontmatter block.

    String-level like `stamp_type`: the rest of the block stays byte-for-byte
    intact. Callers run this after `ensure_system_floor`, so a block always
    exists; content without one is returned unchanged. Reserved-key policy is
    the caller's (the tool layer rejects `AI`/`last modified`/`verified`).

    `props` is a model-controlled tool argument, so every pair goes out as one
    line of emitted YAML and is read back before it is kept: a raw value
    carrying a newline would open a SECOND frontmatter key, and a forged
    `verified: [{by: human:…}]` block — or a second `AI:` that re-declares the
    note as not-agent-written — is what `contested.reliability_tier` reads as
    TIER_HUMAN, a model handing itself the trust tier reserved for a person.
    One line also keeps the replace branch below (a one-line regex) honest on
    the next upsert.
    """
    if not props or not content.startswith("---\n"):
        return content
    end = content.find("\n---\n", 4)
    if end == -1:
        return content
    head = content[:end]
    for key, value in props.items():
        scalar = _yaml_scalar(value)
        line = f"{key}: {scalar}"
        # The KEY goes out raw (the replace branch below matches on it), so a
        # blocklist of characters is the wrong shape here: ':' and '\n' are not
        # the only ways out of one pair. A '#' comments the rest of the line
        # away, a leading indicator retypes the node, and NEL/LS/PS are line
        # breaks the parser honours: a key of "#x", one U+2028, then "AI" emits
        # a comment plus a SECOND `AI:` that re-declares the note as not
        # agent-written, which `contested.reliability_tier` reads as TIER_HUMAN.
        # So read the line back instead of enumerating YAML's syntax: it is safe
        # exactly when it parses to the one pair that went in.
        try:
            reparsed = yaml.safe_load(line)
        except yaml.YAMLError:
            reparsed = None
        if any(c in key for c in ":\n\r") or reparsed != {key: str(value)}:
            raise ValueError(
                f"Unsafe frontmatter key {key!r}: a key must be a plain YAML "
                "scalar — no ':', no comment, no line break."
            )
        line_re = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
        if line_re.search(head[4:]):
            # re.sub calls the replacement eagerly, so the default-argument
            # capture the lambda used to carry was pinning values that cannot
            # change before it runs.
            head = head[:4] + line_re.sub(
                lambda _m: f"{key}: {scalar}", head[4:], count=1)
        else:
            head += f"\n{key}: {scalar}"
    return head + content[end:]


def ensure_system_floor(content: str, prior: str | None = None) -> str:
    """String-level floor under every write: `AI: true` + `last modified`
    always land, whatever the model emitted. No YAML round-trip.

    - content has a frontmatter block: ensure_ai_flag, unchanged.
    - content has none but `prior` (the pre-write note) has one: re-inject
      the prior block verbatim on top of the new body — omission means
      "keep the user's metadata", not "delete it" — then ensure AI: true
      and refresh `last modified` to today.
    - no block anywhere: create the minimal one.
    """
    if content.startswith("---\n"):
        return _stamp_agent(ensure_ai_flag(content))
    today = datetime.date.today().isoformat()
    pm = frontmatter.FM_RE.match(prior) if prior else None
    if pm is None:
        return _stamp_agent(f"---\nAI: true\nlast modified: {today}\n---\n\n{content.lstrip(chr(10))}")
    # Rebuild the prior block with canonical bare fences: FM_RE tolerates CRLF
    # and fence-line whitespace, but the splices below assume exactly
    # "---\n...\n---\n".
    block = "---\n" + pm.group(1) + "\n---\n"
    merged = ensure_ai_flag(block + "\n" + content.lstrip("\n"))
    end_idx = merged.find("\n---\n", 4)
    head, tail = merged[:end_idx], merged[end_idx:]
    if _LAST_MODIFIED_RE.search(head):
        head = _LAST_MODIFIED_RE.sub(f"last modified: {today}", head, count=1)
    else:
        head += f"\nlast modified: {today}"
    return _stamp_agent(head + tail)


PROVENANCE_HEADER_PREFIX = "## Additional notes"

# Emitted until 2026-08. Recognized forever: the header is the idempotency key
# for a patch block (block_present), so a vault carrying blocks in the old
# spelling must keep matching or every re-ingest appends a second copy of what
# it already wrote. Never emitted — read side only.
LEGACY_PROVENANCE_HEADER_PREFIX = "## Note aggiuntive"

PROVENANCE_HEADER_PREFIXES = (PROVENANCE_HEADER_PREFIX, LEGACY_PROVENANCE_HEADER_PREFIX)


def provenance_header(heading: str, source_basename: str) -> str:
    """The exact header line patch_snippet emits for a (heading, source) block.

    Single source of truth so the patch executor can detect an already-injected
    block and stay idempotent on re-injection — patch_snippet interpolates this
    rather than repeating the literal, which is what let the two drift apart.
    """
    return f"{PROVENANCE_HEADER_PREFIX}: {heading} (from {source_basename})"


def _legacy_provenance_header(heading: str, source_basename: str) -> str:
    """The pre-2026-08 Italian spelling of the same block header."""
    return f"{LEGACY_PROVENANCE_HEADER_PREFIX} — {heading} (da {source_basename})"


def block_present(existing_content: str | None, heading: str, source_basename: str) -> bool:
    """True if a provenance block for (heading, source_basename) is already present.

    Both spellings count: a note patched before the header was translated still
    holds that source's block, and re-appending it would duplicate the content.
    """
    if not existing_content:
        return False
    return (
        provenance_header(heading, source_basename) in existing_content
        or _legacy_provenance_header(heading, source_basename) in existing_content
    )


# Lane A (survey-provenance spec): the `distinct` dedup verdict persists as one
# canonical, parseable callout line on the committed spoke. Builder and parser
# live together (same single-source-of-truth idiom as provenance_header) so the
# emitter (dedup) and the readers (survey, graph tooling) cannot drift. Emit
# only this form — there is no legacy form to recognize.
_RELATED_TRACE_RE = re.compile(
    r"^> \[!info\] Related: \[\[([^\]]+)\]\] "
    r"\(judged distinct(?:: (.*))?\)\s*$"
)
_RELATED_TRACE_RATIONALE_CHARS = 200


def _related_trace_candidate(candidate: str) -> str:
    """Candidate title made safe for a wikilink: brackets out, spaces folded."""
    return " ".join(candidate.replace("[", " ").replace("]", " ").split())


def related_trace(candidate: str, rationale: str) -> str:
    """The canonical relation-trace line for a judged-distinct pair."""
    cand = _related_trace_candidate(candidate)
    rat = " ".join((rationale or "").split())[:_RELATED_TRACE_RATIONALE_CHARS]
    tail = f": {rat}" if rat else ""
    return f"> [!info] Related: [[{cand}]] (judged distinct{tail})"


def related_unjudged(candidate: str) -> str:
    """The link a spoke keeps when the judge's reply could not be parsed.

    Deliberately NOT the trace form: `parse_related_traces` must read no
    verdict off a pair that was never judged, so this line shares the link
    and the callout type with the trace and nothing else.
    """
    cand = _related_trace_candidate(candidate)
    return f"> [!info] Related: [[{cand}]] (flagged as similar; no verdict recorded)"


def parse_related_traces(body: str) -> list[tuple[str, str]]:
    """(candidate, rationale) for every canonical trace line in *body*."""
    out: list[tuple[str, str]] = []
    for line in (body or "").splitlines():
        m = _RELATED_TRACE_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2) or ""))
    return out


def has_related_trace(body: str, candidate: str) -> bool:
    """True if *body* already carries a trace line targeting *candidate*."""
    cand = _related_trace_candidate(candidate)
    return any(c == cand for c, _ in parse_related_traces(body))
