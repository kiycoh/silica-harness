# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

import re
import unicodedata
import yaml

FM_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

def split(content):
    """Return (data_dict_or_None, raw_fm_or_None, body).

    `data` is None whenever the block cannot be used as properties: a YAML
    error, or valid YAML that is not a mapping (a sequence, a bare scalar).
    Those are the same case for every caller — nothing here can round-trip them
    as a property dict — and `raw` stays non-None either way, so the standard
    `if data is None and raw is not None: leave the note alone` guard covers
    both. Returning the sequence instead crashed the first `data.get()`
    downstream; two call sites had grown their own `isinstance(data, dict)`
    check, which is the invariant belonging here.

    An EMPTY block is a mapping with no keys, not a failure: `{}`, not None.
    """
    m = FM_RE.match(content)
    if not m:
        return None, None, content
    raw = m.group(1)
    body = content[m.end():]
    try:
        data = yaml.safe_load(raw)
    except Exception:
        return None, raw, body
    if data is None:
        return {}, raw, body
    return (data if isinstance(data, dict) else None), raw, body

def aliases_of(content):
    """The frontmatter alias surfaces of a note, as a list of strings.

    Obsidian accepts `aliases:` (list or inline list), the singular `alias:`,
    and a CSV scalar; all three land here as a flat list.

    The `"alias" not in raw` guard is the point of the function: a note without
    the key never reaches the YAML parser, so harvesting the whole vault costs
    one head-of-file regex per note instead of one yaml.safe_load per note.
    Measured at ~12 us/note (2000-note synthetic vault, 5% declaring aliases),
    which is 0.5% of that vault's index build — cheap enough to sit in the scan.
    """
    m = FM_RE.match(content)
    if not m or "alias" not in m.group(1):
        return []
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    raw = data.get('aliases') or data.get('alias')
    if not raw:
        return []
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(',') if s.strip()]
    if not isinstance(raw, (list, tuple, set)):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]

def documents_in(content):
    """The `documents:` paths of a note, as a flat list of strings.

    Same head-guard economics as aliases_of: a note without the key never
    reaches the YAML parser, so the backends can harvest the whole vault in
    their existing body pass. Only the SHAPE lives here — the semantic
    contract (repo-relative, validated at write, staleness vs code_ref) stays
    in kernel/code/codedocs.py, which the driver must not import: its module
    top pulls codeast, and with it tree-sitter.
    """
    m = FM_RE.match(content)
    if not m or "documents" not in m.group(1):
        return []
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    raw = data.get("documents")
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def add_alias(content, alias):
    """Add `alias` to a note's `aliases:`, idempotently. Returns the new content.

    Unchanged when the alias is already declared (case-insensitively), when it
    equals nothing, or when frontmatter is present but unparseable (never
    destroy what we cannot round-trip). A note without frontmatter gains a
    minimal one. The singular `alias:` spelling is folded into `aliases:`, the
    key Obsidian and aliases_of() both read.
    """
    alias = str(alias or '').strip()
    if not alias:
        return content
    data, raw, body = split(content)
    if data is None:
        if raw is not None:          # present but unusable as properties
            return content
        data, body = {}, content
    current = aliases_of(content)
    if alias.lower() in {a.lower() for a in current}:
        return content
    data.pop('alias', None)
    data['aliases'] = current + [alias]
    return dump(data, body)

def clean_tag(t):
    """Canonical tag normalizer (moved from templates.py — single source of truth)."""
    # Strip a leading list-ordinal ("1. ", "2) ") but not a digit fused to a word:
    # require a separator + space so "3d"/"2fa"/"3D-Printing" keep their leading digit.
    t = re.sub(r'^\d+[.\)]\s+', '', str(t))
    t = t.lower()
    # Transliterate accented chars to ASCII (à→a, ì→i) instead of deleting them:
    # on an Italian vault the old strip truncated "scalabilità"→"scalabilit".
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('ascii')
    t = re.sub(r'[^a-z0-9\s-]', '', t)
    t = re.sub(r'[\s_]+', '-', t)
    return t.strip('-')

def _ensure_tag_list(raw):
    """Coerce raw tags value into a list, splitting CSV scalars."""
    if not raw:
        return []
    if isinstance(raw, str):
        # Detect inline-CSV scalar: "a, b, c" → ["a", "b", "c"]
        if "," in raw:
            return [s.strip() for s in raw.split(",") if s.strip()]
        return [raw]
    return list(raw)

def lint_tags(data):
    issues = []
    tags = _ensure_tag_list((data or {}).get('tags'))
    for t in tags:
        ct = clean_tag(t)
        if ct != str(t):
            issues.append(f"tag '{t}' not normalized -> '{ct}'")
        if not ct:
            issues.append(f"tag '{t}' is empty after normalization")
    return issues

def dump(data, body):
    """Re-emit a full note: --- yaml --- + blank line + body."""
    fm = yaml.safe_dump(data, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{fm}\n---\n\n{body.lstrip()}"
