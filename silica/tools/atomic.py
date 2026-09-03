# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Atomic tools — L0 façades, 1:1 on Obsidian CLI commands.

From SILICA.md §4.2:
  Atomic tools are single Obsidian-native operations, 1:1 on a CLI command
  or a pure kernel function. They are the base vocabulary — called by both
  the agent and the pipeline.
"""
from __future__ import annotations

import re
import subprocess
import unicodedata
from collections.abc import Iterable
from functools import lru_cache

from pathlib import Path

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.tools import tool


# ---------------------------------------------------------------------------
# Read / Discovery
# ---------------------------------------------------------------------------

# One cap for every listing that grows with the vault. ponytail: flat cap
# defends the context window (a 1000-note vault ≈ 20k tokens uncapped); no
# paging — narrowing by folder, or acting on a sample, covers the real uses.
_FILES_CAP = 200


class SearchArgs(BaseModel):
    query: str = Field(description="Text to search for in note names in the vault")


# 868 names re-folded on every search, ~30 searches in one agent turn. Measured
# 2026-08-27 on that vault: 0.767 ms per sweep uncached, 0.027 ms cached (439492
# hits against 868 misses over five sweeps), 3.32 -> 2.44 ms end to end, 134 KB
# held for 868 entries. `_fold` is pure, so a renamed note only leaves an entry
# nobody asks for again — there is nothing to invalidate.
#
# maxsize is the whole contract, not a tuning knob: it must stay above the
# vault's note count, because every search touches all of them and then one
# throwaway query key. Undersized it is SLOWER than no cache at all — 256 on 872
# names measured 0.899 ms, +17%, paying hash and eviction to reuse nothing.
# ponytail: 8192 covers the roster plus the query churn up to ~8000 notes. Past
# that the answer is not a bigger number — it is folding once when the index is
# built and keeping the folded name on NoteRef, which needs no maxsize, no
# eviction and no lock.
@lru_cache(maxsize=8192)
def _fold(text: str) -> str:
    """Case and accents folded away: the form two titles are compared in.

    `casefold` alone left 44 of 872 titles on an Italian vault unreachable by
    anyone typing without the accent — "probabilita" answered 0 while
    `Teoria delle probabilità.md` sat in the index (measured 2026-08-27).
    Dropping the combining marks also makes the NFC vault and the NFD one a
    macOS Obsidian writes compare equal, which no amount of casefolding does.
    It deliberately collapses "papa" and "papà": this is the SEARCH surface,
    where the reader picks from the hits. Wikilink resolution is a different
    seam and must keep them apart.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold())
                   if not unicodedata.combining(c))


# Same unbounded shape as search_context, one level down: substring-over-names
# means a short query answers with the vault ("e" measured 575 paths / 41k chars
# on a 719-note vault). A name lookup that returns 500 names has answered
# nothing, so rank by how well the name matches and keep the head.
_SEARCH_CAP = 40


@tool(SearchArgs, cls="atomic")
def silica_search(query: str) -> dict:
    """Search for notes by NAME/title match. Returns the paths of matching notes.

    A note's wikilink name is its filename without the extension. Case and
    accents are ignored, so "probabilita" finds `Probabilità bayesiana`. For
    text inside note bodies use silica_search_context; for meaning-based search
    when you don't know the exact words use silica_semantic_search.

    Returns {paths, matched}: closest name match first, `truncated` when capped,
    `relaxed` when no title carries every word of the query and the hits only
    share some of them.
    """
    q = _fold(query)
    # Words, not the phrase. A title search used to be one substring test against
    # one string, so a query that says more than the title does answered
    # "nothing" with the note sitting right there: "regressione lineare multipla"
    # missed `Regressione lineare.md` that semantic search scores 1.000. Measured
    # 2026-08-27 on an 886-note vault, 22 of 29 searches in one turn came back
    # empty and the study plan built on them declared owned material missing.
    # Words under 3 chars match half the vault, so they only stand in when the
    # query is nothing but short words.
    tokens = [t for t in re.split(r"\W+", q) if len(t) > 2] or [q]

    # One enumeration, scored in place. The phrase and the words are not two
    # passes: a second lookup is a second round-trip on the ws backend (one
    # `list_files` per token) and it can only re-find what this scan already saw.
    scored: list[tuple] = []
    for ref in DRIVER.search_names(""):
        name = _fold(ref.name)
        carried = sum(t in name for t in tokens)
        if not carried:
            continue
        # 0/1/2 read the query literally (exact title, prefix, substring), 3 is
        # every word present but scattered, 4 is only some of them. Tiers 0-3 are
        # facts about the title; 4 is a guess, and merging the two turns
        # `matched: 1` ("this is the note") into `matched: 12` ("pick one") —
        # which is the number the caller reads as coverage. Then shorter name (a
        # 6-char hit on an 8-char name is a better answer than on an 80-char
        # one), then path.
        if q in name:
            tier = 0 if name == q else (1 if name.startswith(q) else 2)
        else:
            tier = 3 if carried == len(tokens) else 4
        scored.append((tier, -carried, len(name), ref.path))

    scored.sort()
    literal = [row for row in scored if row[0] < 4]
    ranked = literal or scored
    out: dict = {"paths": [row[-1] for row in ranked[:_SEARCH_CAP]], "matched": len(ranked)}
    if not ranked:
        # A bare 0 reads as "the vault does not have this". It does not: on the
        # 2026-08-27 run `boosting` had no title but five notes mentioning it,
        # and the plan built on that 0 marked the topic missing. Name the two
        # tools that would have answered, at the one moment the caller is about
        # to conclude otherwise.
        out["hint"] = (
            "no title matches. The term may still be in note bodies "
            "(silica_search_context) or worded differently "
            "(silica_semantic_search)."
        )
    if ranked and not literal:
        out["relaxed"] = (
            f"no title carries every word of '{query}'; these carry some. "
            "Use silica_semantic_search to rank by meaning."
        )
    if len(ranked) > _SEARCH_CAP:
        out["truncated"] = (
            f"{len(ranked)} notes matched; showing the {_SEARCH_CAP} closest name "
            "matches. Narrow the query, or use silica_semantic_search to rank by meaning."
        )

    # Stale flags (spec-stale-triggers §3): read-only peek, so a search never
    # pays the git walk; at worst the first search after a commit has no flags.
    try:
        from silica.kernel.code import codedocs

        m = _stale_map()
        flagged = {p: lvl for p in out["paths"]
                   if (lvl := codedocs.peek_level(m, p))}
        if flagged:
            out["stale"] = flagged
    except Exception:
        pass  # flags are an aid, never a reason a search fails
    return out


# The third stale level, beside codedocs' "cosmetic" and "structural": the
# note was distilled from a source version that has since been re-nucleated.
_STALE_SOURCE = "source"


def _stale_map() -> dict[str, str]:
    """`note_path.md -> level` for every note a reader should doubt.

    Two ledgers, one flag. codedocs.peek answers for code notes after a commit;
    the provenance drift map answers for notes whose source was re-nucleated
    at another version. The code level wins when a note carries both: it is
    the more specific claim. Each leg is guarded on its own, so a ledger that
    cannot be read costs its flags and nothing else.
    """
    from silica.config import CONFIG

    out: dict[str, str] = {}
    try:
        from silica.kernel.write import provenance

        out.update({p: _STALE_SOURCE
                    for p in provenance.drift_map(vault_path=CONFIG.vault_path)})
    except Exception:
        pass  # flags are an aid, never a reason a search fails
    try:
        from silica.kernel.code import codedocs

        out.update(codedocs.peek(CONFIG.vault_path))
    except Exception:
        pass
    return out


class SearchContextArgs(BaseModel):
    query: str = Field(description="Text to search for within the content of vault notes")


# A literal grep over every body is unbounded by nature: one Hit per matching
# LINE, so a short query returns the vault. Measured on a 719-note vault:
# "OSI" → 529 hits / 170k chars in a single payload, "e" → 14535 hits. The
# window is the scarce resource, so rank by hit density (the note that mentions
# the term 40 times is the one meant; the 190 notes mentioning it once are not)
# and keep the top slice. ponytail: density, not the reranker — this is the
# deterministic literal leg, and putting a model in it buys ranking at the cost
# of reproducibility and a "reranker down" failure mode. Meaning-based ranking
# already has a tool: silica_semantic_search.
_CONTEXT_MAX_NOTES = 12
_CONTEXT_LINES_PER_NOTE = 3
_SOURCE_MAX_FILES = 20
_SOURCE_LINES_PER_FILE = 3


def _source_hits(query: str) -> list[dict] | None:
    """Exact-string hits in the repo's source files; None when the vault has
    no code lane or git could not scan (so the caller reports "not scanned",
    never a false absence).

    The driver indexes markdown only, so on a codebase vault a symbol probe
    answered empty while grep found it (2026-09-03, `follow_superseded`, 7
    lines). `git grep` rather than `grep -r`: it honours .gitignore, so the
    346MB of fixtures under `data` never get walked, and `--untracked` keeps
    a file written this session visible. Notes are excluded here because the
    driver already searched them.
    """
    from silica.config import CONFIG
    from silica.kernel.recall.paths import repo_root_for

    root = repo_root_for(getattr(CONFIG, "vault_path", "") or "")
    if root is None:
        return None
    try:
        proc = subprocess.run(
            ["git", "grep", "-n", "-I", "-F", "-i", "--untracked", "-e", query,
             "--", ":!*.md"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None  # git missing or hung: "not scanned" is the honest answer
    if proc.returncode not in (0, 1):  # 1 = no match; anything else is a git error
        return None
    hits: list[dict] = []
    per_file: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        path, _, rest = line.partition(":")
        lineno, _, snippet = rest.partition(":")
        if path not in per_file and len(per_file) >= _SOURCE_MAX_FILES:
            break
        if per_file.get(path, 0) >= _SOURCE_LINES_PER_FILE:
            continue
        per_file[path] = per_file.get(path, 0) + 1
        hits.append({"kind": "source", "path": path, "line": int(lineno),
                     "snippet": snippet.strip()[:200]})
    return hits


@tool(SearchContextArgs, cls="atomic")
def silica_search_context(query: str) -> dict:
    """Search note BODIES for exact text; returns snippets with line numbers.

    Use to find literal mentions of a term. When the exact wording is unknown,
    use silica_semantic_search instead; to match note titles use silica_search.

    Returns {hits, notes_matched}: densest notes first, 3 snippets each at most,
    `truncated` when capped.
    """
    by_note: dict[str, list] = {}
    for h in DRIVER.search_context(query):
        by_note.setdefault(h.ref.path or h.ref.name, []).append(h)

    q = query.casefold()
    ranked = sorted(
        by_note.values(),
        # note_matches is the note's TRUE occurrence count: the backend caps
        # materialized Hits per note, so len(hs) saturates at the cap and would
        # tie every heavy note. 0 means the backend didn't count — fall back.
        # path last: a stable tiebreak, so the same query answers the same way.
        key=lambda hs: (-(hs[0].note_matches or len(hs)),
                        q not in hs[0].ref.name.casefold(), hs[0].ref.path),
    )
    kept = ranked[:_CONTEXT_MAX_NOTES]

    # Stale flags (spec-stale-triggers §3): one peek for the whole call, read-only
    # so search never pays the git walk. `peek()` itself never raises by contract;
    # this try/except guards the imports and honors the soft-failure rule (§5)
    # regardless.
    try:
        from silica.kernel.code import codedocs

        stale_map = _stale_map()
    except Exception:
        stale_map = {}
        codedocs = None  # type: ignore[assignment]  # peek import failed: no flags this call

    out: dict = {
        "hits": [
            {"name": h.ref.name, "path": h.ref.path, "line": h.line,
             "snippet": h.snippet,
             **({"stale": lvl} if stale_map and codedocs
                and (lvl := codedocs.peek_level(stale_map, h.ref.path or ""))
                else {})}
            for hs in kept for h in hs[:_CONTEXT_LINES_PER_NOTE]
        ],
        "notes_matched": len(by_note),
    }
    if len(kept) < len(by_note) or any(len(hs) > _CONTEXT_LINES_PER_NOTE for hs in kept):
        out["truncated"] = (
            f"{len(by_note)} notes matched; showing the {len(kept)} densest, up to "
            f"{_CONTEXT_LINES_PER_NOTE} lines each. Narrow the query, or use "
            "silica_semantic_search to rank by meaning."
        )
    src = _source_hits(query)
    if src is not None:
        out["hits"].extend(src)
        out["scanned"] = ["notes", "source"]
    return out


class ReadNoteArgs(BaseModel):
    name: str = Field(description="Name of the note to read (wikilink style, without file extension)")
    vault: str = Field(default="", description="Read from another Silica vault (the `vault` or `memory_vault` a recall reply named). Read-only.")


@tool(ReadNoteArgs, cls="atomic")
def silica_read_note(name: str, vault: str = "") -> str:
    """Reads the complete content of a note in the vault by name (wikilink-style
    resolution). DO NOT use paths. `vault=<path>` reads from another vault,
    read-only."""
    if vault.strip():
        from silica.config import CONFIG
        from silica.driver import reader_for
        from silica.kernel.recall.vault_registry import resolve_known

        target = resolve_known(vault)  # ValueError names the rule and the fix
        active = (getattr(CONFIG, "vault_path", "") or "").strip()
        if not active or Path(active).resolve() != target:
            nc = reader_for(str(target)).read_note(name)
            body = _with_stale_banner(nc.content, path=nc.ref.path or "", vault=str(target))
            return (f"> [read-only] {target}: this note belongs to another vault; "
                    "silica_write_note and silica_patch_note act on the active vault only\n\n"
                    + body)
    nc = DRIVER.read_note(name)
    return _with_stale_banner(nc.content, path=nc.ref.path or "")


def _with_stale_banner(content: str, path: str = "", vault: str | None = None) -> str:
    """Prefix a note with its staleness warnings, when it has any.

    A wiki note derived from source outlives the source: after a refactor it
    still reads as authoritative while naming files that have moved, and a
    note distilled from a lecture outlives the lecture's next version the same
    way. The `code_ref`/`documents:` frontmatter and the provenance ledger
    always carried the answer, but only the `/stale` report and the graph
    report ever asked — so the reader, the one acting on the note, was the one
    kept in the dark. The code banner is parsed from the content in hand; the
    drift banner is one ledger lookup keyed by `path`.
    """
    banners: list[str] = []
    from silica.config import CONFIG

    # `vault` names the folder the note was read from (a peek); the code lane
    # and the provenance ledger are per vault, so both lookups follow it.
    vault = vault or CONFIG.vault_path
    try:
        from silica.kernel.code import codedocs
        from silica.kernel.write import frontmatter

        data, _, _ = frontmatter.split(content)
        if data and codedocs.documents_of(data):
            warning = codedocs.read_warning(vault, data)
            if warning:
                banners.append(warning)
    except Exception:
        pass  # a banner is an aid, never a reason a read fails
    try:
        if path:
            from silica.kernel.code import codedocs
            from silica.kernel.write import provenance

            source = codedocs.peek_level(
                provenance.drift_map(vault_path=vault), path)
            if source:
                banners.append(
                    f"[stale] source {source} was re-nucleated after this note was "
                    f"written: it derives from the previous version")
    except Exception:
        pass
    if not banners:
        return content
    return "".join(f"> {b}\n\n" for b in banners) + content


class PropsArgs(BaseModel):
    name: str = Field(description="Name of the note to read the frontmatter properties from")

@tool(PropsArgs, cls="atomic")
def silica_props(name: str) -> dict:
    """Reads the frontmatter properties of a note (saves tokens, does not read the body)."""
    return DRIVER.props_of(name)


class OutlineArgs(BaseModel):
    name: str = Field(description="Name of the note to display the heading tree of")

@tool(OutlineArgs, cls="atomic")
def silica_outline(name: str) -> list:
    """Displays the heading tree (H1-H6) of a note."""
    headings = DRIVER.outline(name)
    return [{"level": h.level, "text": h.text} for h in headings]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class LinksArgs(BaseModel):
    name: str = Field(description="Name of the note to list outgoing links from")

@tool(LinksArgs, cls="atomic")
def silica_links(name: str) -> list:
    """Lists outgoing links from a note (connected notes)."""
    refs = DRIVER.links(name)
    return [r.path for r in refs]


class BacklinksArgs(BaseModel):
    name: str = Field(description="Name of the note to list incoming links (backlinks) for")

@tool(BacklinksArgs, cls="atomic")
def silica_backlinks(name: str) -> list:
    """Lists incoming links (backlinks) pointing to a note."""
    refs = DRIVER.backlinks(name)
    return [r.path for r in refs]


class FileLinksArgs(BaseModel):
    target: str = Field(description="A note (name or path), OR a file path/basename (e.g. foto.jpg, analysis.ipynb, src/mod.py)")

@tool(FileLinksArgs, cls="atomic")
def silica_file_links(target: str) -> dict:
    """Note↔file connections, both directions. Given a note: the files it
    references (image/media/notebook embeds + `documents:` frontmatter) as
    {note, embeds, documents, unresolved}. Given a file: the notes referencing
    it as {file, embedded_in, documented_by}. Complements silica_links, which
    sees only note→note wikilinks. Embeds are vault-relative paths;
    `documents:` entries are repo-relative (code lane)."""
    from silica.kernel.link.ast import NON_MD_EXTENSIONS

    t = (target or "").strip()
    # Notes win over files, but a file-shaped target (extension allowlist —
    # a vault note can never end in one) skips the pointless note resolution.
    if t and not t.lower().endswith(NON_MD_EXTENSIONS):
        try:
            note_path = DRIVER.read_note(t).ref.path
        except Exception:
            note_path = ""  # not a note: tolerated, the file direction answers below
        if note_path:
            out: dict = {"note": note_path}
            out.update(DRIVER.file_refs_of(note_path))
            return out
    bl = DRIVER.file_backlinks(t)
    out = {"file": t, "embedded_in": bl.get("embeds", []),
           "documented_by": bl.get("documents", [])}
    if not out["embedded_in"] and not out["documented_by"]:
        out["hint"] = ("No note references this target — or it is neither a "
                       "resolvable note nor a vault file. silica_files lists "
                       "what exists under a folder.")
    return out


class EmptyArgs(BaseModel):
    pass

@tool(EmptyArgs, cls="atomic")
def silica_orphans() -> dict:
    """Lists orphan notes (notes with no incoming links) in the vault.

    Returns {"total": N, "orphans": [path, ...]} capped at 200 entries — a
    neglected vault is mostly orphans, so the count is the answer to "how many"
    and the sample is enough to start linking.
    """
    refs = DRIVER.orphans()
    paths = [r.path for r in refs]
    out: dict = {"total": len(paths), "orphans": paths[:_FILES_CAP]}
    if len(paths) > _FILES_CAP:
        out["truncated"] = True
    return out


@tool(EmptyArgs, cls="atomic")
def silica_unresolved() -> list:
    """Lists unresolved wikilinks in the vault (links pointing to non-existent notes)."""
    links = DRIVER.unresolved()
    return [{"target": l.target} for l in links]


# ---------------------------------------------------------------------------
# List files
# ---------------------------------------------------------------------------

def _natural_key(path: str) -> list:
    """Sort key that reads embedded runs of digits as numbers.

    Plain lexicographic order interleaves "Lezione 10" between 1 and 2, and the
    injector ingests a folder in listing order: lesson 10's concepts would land
    before lesson 2 defines them. re.split alternates non-digit/digit, so the
    element at a given index always has the same type across paths.
    """
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def notes_under(folder: str) -> list[str]:
    """Vault-relative `.md` paths under `folder`, natural-sorted. "" ⇒ the vault.

    The one answer to "which notes does this folder hold?", because neither
    half of the codebase could give it for an inbox folder: the note index used
    to skip the inbox entirely, and `sources.registry.expand_folder` answers
    only for the code lane off a git-backed census — a plain Obsidian vault is
    no repo, so it returns [] for every folder in it. Between the two a folder
    argument had no listing at all, and the only caller left to produce one was
    an LLM guessing filenames.

    The index now walks the inbox, so `list_files` alone answers. The
    `list_inbox_files` fallback below stays as belt-and-braces: its failure mode
    was a model inventing filenames, which is not a failure worth re-earning to
    save four lines.
    """
    from silica.kernel.recall.paths import in_folder
    from silica.kernel.vault_manifest import active_inbox_dir

    scope = _vault_rel(folder)
    # list_files(folder) pre-filters loosely (startswith); in_folder tightens it
    # so a Foo/ argument never leaks into a sibling FooBar/.
    paths = [r.path for r in DRIVER.list_files(scope) if in_folder(r.path, scope)]
    if not paths and scope:
        inbox = active_inbox_dir()
        if inbox and in_folder(scope, inbox):
            # .md only: the unconverted files (PDFs) are silica_inbox_ls's job,
            # and a converted folder's Images/ leaves outnumber its notes 70:1.
            paths = [
                r.path for r in DRIVER.list_inbox_files()
                if r.path.endswith(".md") and in_folder(r.path, scope)
            ]
    return sorted(paths, key=_natural_key)


def _unconverted_under(folder: str) -> list[str]:
    """Non-markdown files under `folder`, natural-sorted; `Images/` excluded.

    `notes_under` lists .md only, so a folder holding nothing but PDFs came back
    as {"total": 0, "files": []} — a payload indistinguishable from a folder
    that is not there, and the agent duly reported it as non-existent. Naming
    the unconverted files is the only way the caller can tell the two apart.
    `Images/` is conversion output, not input, and outnumbers the notes 70:1.

    Answers for source folders as well as the inbox: a research library is nine
    folders of scanned books and one INDEX.md, and while the rule was
    inbox-only every one of them read as empty.
    """
    from silica.kernel.recall.paths import in_folder
    from silica.kernel.vault_manifest import active_inbox_dir

    scope = _vault_rel(folder)
    if not scope:
        return []
    inbox = active_inbox_dir()
    if inbox and in_folder(scope, inbox):
        paths: Iterable[str] = (r.path for r in DRIVER.list_inbox_files())
    else:
        paths = _vault_files_under(scope)
    return sorted(
        (
            p for p in paths
            if not p.endswith(".md")
            and in_folder(p, scope)
            and "/Images/" not in p
        ),
        key=_natural_key,
    )


def _vault_files_under(scope: str) -> list[str]:
    """Vault-relative files under `scope`, dotfiles and dot-dirs skipped."""
    import os

    from silica.config import CONFIG

    root = Path(CONFIG.vault_path or "")
    base = root / scope
    if not base.is_dir():
        return []
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            out.append((Path(dirpath) / name).relative_to(root).as_posix())
    return out


def _vault_rel(path: str) -> str:
    """`path` as a vault-relative posix path; already-relative input passes through.

    Users do paste absolute vault paths at /nucleate, and every listing below is
    keyed vault-relative. A path outside the vault is returned as given: it
    matches nothing, which is the honest answer for it.
    """
    from silica.config import CONFIG

    p = Path(path.strip())
    if p.is_absolute():
        try:
            return p.resolve().relative_to(Path(CONFIG.vault_path).resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return path.replace("\\", "/").strip("/")


class ListFilesArgs(BaseModel):
    folder: str = Field(default="", description="Optional folder path to filter results")

@tool(ListFilesArgs, cls="atomic")
def silica_files(folder: str = "") -> dict:
    """Lists vault notes and source files under a folder.

    Returns {"total", "files"} for markdown (wikilink name = filename minus
    extension), "code" for ingestible source files (feed to /nucleate — a
    folder of code is NOT empty just because it holds no .md), "unconverted"
    for inbox files needing `/convert`, and on a bare call "source_folders"
    ({folder: count}) for non-markdown material that IS in the vault — never
    report it as missing. Listings cap at 200: on "truncated", narrow with
    folder= instead of re-calling. For a bare count use "total" or the
    '## Vault map' already in context.
    """
    # bare paths, not {name, path} dicts — NoteRef.name is the
    # filename without its extension, so the dict shipped every note's name
    # twice (48% of this payload, ~2.5k tokens at the 200-entry cap).
    files = notes_under(folder)
    result: dict = {"total": len(files), "files": files[:_FILES_CAP]}
    if len(files) > _FILES_CAP:
        result["truncated"] = True
        result["hint"] = "Listing capped at 200 entries; pass folder= to narrow."
    # An inbox folder of PDFs is not an empty folder: say so, or the caller
    # concludes the path does not exist. See _unconverted_under.
    pending = _unconverted_under(folder) if folder else []
    if pending:
        result["unconverted"] = pending[:_FILES_CAP]
        result["unconverted_total"] = len(pending)
        result.setdefault(
            "hint",
            f"{len(pending)} file(s) here are not markdown yet. /nucleate handles "
            "them directly (it converts, then distills) — suggest `/nucleate "
            "<path>` for one, or `/nucleate <folder>` for all of them. Only "
            "suggest /convert when the user wants the markdown WITHOUT notes.",
        )
    # folder-scoped only — a bare call would dump a whole repo into
    # the context window, and the vault map already covers "what is here".
    if folder:
        from silica.sources.registry import expand_folder

        code = expand_folder(folder)
        if code:
            result["code"] = code[:_FILES_CAP]
            result["code_total"] = len(code)
    else:
        # Folders, never the files: a bare listing on a library of scanned books
        # showed only the notes Silica had written, so the agent told the user
        # their own PDFs were "nowhere in the vault". A count per top folder is
        # five lines and answers the question the file list was being read for.
        census = _source_folder_census()
        if census:
            result["source_folders"] = census
            result.setdefault(
                "hint",
                "These folders hold files that are not notes yet (PDFs, scans, "
                "media). They ARE in the vault — call silica_files(folder=…) to "
                "name them, or /nucleate a path to turn one into notes.",
            )
    return result


def _source_folder_census() -> dict[str, int]:
    """Top-level folder -> count of non-markdown files under it, non-empty only."""
    from collections import Counter

    from silica.kernel.recall.paths import is_inbox_path

    counts: Counter = Counter()
    for p in _vault_files_under(""):
        if p.endswith(".md") or "/" not in p or is_inbox_path(p):
            continue
        counts[p.split("/", 1)[0]] += 1
    return dict(counts.most_common(_FILES_CAP))


class ExistsArgs(BaseModel):
    path: str = Field(description="Relative path of the note in the vault")

@tool(ExistsArgs, cls="atomic")
def silica_exists(path: str) -> bool:
    """Verifies a file exists in the vault — notes, inbox, and source files
    alike (PDFs too). `read_note` only opens markdown: "I cannot read it" is
    not the same answer as "it is not there".
    """
    try:
        DRIVER.read_note(path)
        return True
    except Exception:
        pass
    from silica.config import CONFIG

    root = Path(CONFIG.vault_path or "")
    if not root.is_dir():
        return False
    try:
        target = (root / (path or "").strip()).resolve()
        target.relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return target.is_file()


# ---------------------------------------------------------------------------
# Deferred Op Store
# ---------------------------------------------------------------------------

@tool(EmptyArgs, cls="atomic")
def silica_deferred_list() -> list:
    """List all pending deferred op bundles (concepts rejected by the validator in previous runs).

    Returns summary rows — use silica_deferred_retry(content_hash) to attempt
    writing them, or silica_deferred_flush(content_hash) to discard them.
    """
    from silica.kernel.recall.deferred import get_deferred_store
    return get_deferred_store().list_all()


class DeferredFlushArgs(BaseModel):
    content_hash: str = Field(description="Content hash of the deferred bundle to permanently discard")

@tool(DeferredFlushArgs, cls="atomic", collapse="eager")
def silica_deferred_flush(content_hash: str) -> dict:
    """Discard a deferred op bundle — marks those rejected ops as permanently skipped."""
    from silica.kernel.recall.deferred import get_deferred_store
    # purge, not remove: the user's explicit discard also drops any declared
    # residue facts, which remove() would deliberately keep.
    removed = get_deferred_store().purge(content_hash)
    if removed:
        return {"removed": True, "content_hash": content_hash}
    return {"removed": False, "error": f"No deferred bundle found for {content_hash[:8]}…"}


@tool(EmptyArgs, cls="atomic")
def silica_inbox_ls() -> list:
    """Lists all files in the Inbox folder (inbox_dir), including non-markdown
    files (PDFs etc.). Non-markdown files cannot be read or nucleated directly:
    ask the user to run `/convert <path>` first, then work on the resulting .md.
    """
    refs = DRIVER.list_inbox_files()
    return [r.path for r in refs]


# ---------------------------------------------------------------------------
# Graph path / explain
# ---------------------------------------------------------------------------

class GraphPathArgs(BaseModel):
    source: str = Field(description="Source note name or vault-relative path")
    target: str = Field(description="Target note name or vault-relative path")
    max_paths: int = Field(default=1, description="Maximum number of shortest paths to return")

@tool(GraphPathArgs, cls="atomic")
def silica_graph_path(source: str, target: str, max_paths: int = 1) -> dict:
    """Shortest connection(s) between two notes over the resolved wikilink graph.

    Returns path(s) as lists of note ids, or an error dict if no path exists.
    Uses the undirected view of the resolved (EXTRACTED) wikilink graph.
    """
    import networkx as nx
    from silica.kernel.recall.graph_export import build_graph_data, edge_graph

    try:
        nodes, edges = build_graph_data(folder="")
    except Exception as exc:
        return {"error": f"Failed to build graph: {exc}"}

    # edge_graph is the shared builder; it also inserts nodes sorted, so tied
    # shortest paths come back in the same order every process.
    G = edge_graph(nodes, edges)
    real_ids: set[str] = set(G.nodes())

    # Resolve source/target: accept path or name substring match
    def _resolve(query: str) -> str | None:
        if query in real_ids:
            return query
        q_lower = query.lower().removesuffix(".md")
        for nid in real_ids:
            stem = nid.rsplit("/", 1)[-1].removesuffix(".md").lower()
            if stem == q_lower:
                return nid
        return None

    src_id = _resolve(source)
    tgt_id = _resolve(target)

    if src_id is None:
        return {"error": f"Source note not found in graph: '{source}'"}
    if tgt_id is None:
        return {"error": f"Target note not found in graph: '{target}'"}
    if src_id == tgt_id:
        return {"paths": [[src_id]], "length": 0}

    try:
        if max_paths == 1:
            path = nx.shortest_path(G, src_id, tgt_id)
            return {"paths": [path], "length": len(path) - 1}
        else:
            import itertools
            gen = nx.all_shortest_paths(G, src_id, tgt_id)
            paths = list(itertools.islice(gen, max_paths))
            return {"paths": paths, "length": len(paths[0]) - 1 if paths else 0}
    except nx.NetworkXNoPath:
        return {"error": f"No path between '{source}' and '{target}'"}
    except nx.NodeNotFound as exc:
        return {"error": f"Node not found: {exc}"}


class GraphExplainArgs(BaseModel):
    note: str = Field(description="Note name or vault-relative path to explain")
    depth: int = Field(default=1, description="Neighbourhood depth (1=direct links only)")

@tool(GraphExplainArgs, cls="atomic")
def silica_graph_explain(note: str, depth: int = 1) -> dict:
    """Explain a note's structural position: cluster, degree rank, betweenness,
    out-links, backlinks, cross-cluster bridges. Low degree + high betweenness
    = a bridge whose removal fragments the vault, worth reinforcing.

    `diagnosis` reads every vault-wide coherence signal for this one note
    (orphan/hub status, cluster cohesion, contested or drifted
    source, and its rank in the attention / integration-deficit lists. A `null`
    in a ranked field means the note did not make that list's top-k, which is
    "not among the worst", not "clean".
    """
    from silica.kernel.report.graph_report import compute_report

    try:
        # with_cooccurrence: the prerequisite direction (V2) is store-derived,
        # and "what to read first" is the question this tool answers best.
        # Memoized per vault epoch, like the analytics pass it rides on.
        report = compute_report(analytics=True, with_cooccurrence=True)
    except Exception as exc:
        return {"error": f"Failed to compute graph report: {exc}"}

    # Find the node in god_nodes or clusters
    q_lower = note.lower().removesuffix(".md")
    node_stat = None
    for n in report.god_nodes:
        if n.id.lower() == q_lower or n.id.rsplit("/", 1)[-1].removesuffix(".md").lower() == q_lower:
            node_stat = n
            break

    # Resolve via cluster members if not in god_nodes
    resolved_id: str | None = None
    if node_stat:
        resolved_id = node_stat.id
    else:
        for c in report.clusters:
            for m in c.members:
                if m.lower() == q_lower or m.rsplit("/", 1)[-1].removesuffix(".md").lower() == q_lower:
                    resolved_id = m
                    break
            if resolved_id:
                break

    if resolved_id is None:
        # last attempt: check orphans
        for o in report.orphans:
            if o.lower() == q_lower or o.rsplit("/", 1)[-1].removesuffix(".md").lower() == q_lower:
                resolved_id = o
                break

    if resolved_id is None:
        return {"error": f"Note '{note}' not found in the graph"}

    # Degree rank (rank among all nodes by degree)
    try:
        out_links = [r.path or r.name for r in DRIVER.links(resolved_id)]
        backlinks = [r.path or r.name for r in DRIVER.backlinks(resolved_id)]
    except Exception:
        out_links = []
        backlinks = []

    bridges_involving = [
        {"source": b.source, "target": b.target, "clusters": f"{b.source_cluster}↔{b.target_cluster}", "weight": b.weight}
        for b in report.bridges
        if b.source == resolved_id or b.target == resolved_id
    ]

    cluster_id = -1
    for c in report.clusters:
        if resolved_id in c.members:
            cluster_id = c.cluster_id
            break

    # Degree rank
    all_degrees = sorted(
        [(n.id, n.degree) for n in report.god_nodes],
        key=lambda x: -x[1],
    )
    degree_rank = next(
        (i + 1 for i, (nid, _) in enumerate(all_degrees) if nid == resolved_id),
        None,
    )

    degree = (node_stat.degree if node_stat else len(out_links) + len(backlinks))

    # Per-note diagnosis: every vault-wide coherence signal, read for THIS note.
    # All of it is already on the report — the signals existed only as vault
    # aggregates or as rows in GRAPH_REPORT.md, with no way to ask "how well is
    # this one note integrated". Composition only, no extra computation.
    # The co-occurrence signals (integration deficit, stale links) need
    # with_cooccurrence, which this tool does not pay for; they read `null` here.
    # Flip the compute_report call above if the diagnosis ever needs them.
    store_key = resolved_id.removesuffix(".md")  # prereq_map is store-keyed (no .md)
    cluster_stat = next((c for c in report.clusters if c.cluster_id == cluster_id), None)
    contested_note = next((c for c in report.contested if c.path == resolved_id), None)
    attention = next((a for a in report.attention_candidates if a.path == resolved_id), None)
    deficit = next((d for d in report.integration_deficits if d.path == resolved_id), None)
    stale = [
        {"source": s.source, "target": s.target}
        for s in report.stale_links
        if resolved_id in (s.source, s.target)
    ]
    diagnosis = {
        # Structural position
        "is_orphan": resolved_id in report.orphans,
        # A one-note cluster's hub is itself, which made every orphan a hub.
        "is_hub": bool(cluster_stat and cluster_stat.hub == resolved_id and cluster_stat.size > 1),
        "cluster_size": (cluster_stat.size if cluster_stat else 0),
        # How tightly its own area holds together: a well-linked note in a loose
        # cluster is integrated; the same note in a dense cluster is ordinary.
        "cluster_cohesion": (cluster_stat.cohesion if cluster_stat else 0.0),
        # Authority / freshness
        "contested": bool(contested_note),
        "contradictions": (contested_note.refs if contested_note else []),
        "drifted_source": any(
            sd.note == resolved_id.removesuffix(".md") for sd in report.source_drift
        ),
        # Ranked signals: present only when the note made the report's top-k, so
        # `null` means "not among the worst", NOT "clean".
        "attention_score": (attention.score if attention else None),
        "days_idle": (attention.days_idle if attention else None),
        "integration_deficit": (deficit.score if deficit else None),
        "stale_links": stale,
        # Seven variables (spec 2026-08-22): structural role and reading order.
        "coreness": report.core_map.get(resolved_id),
        "is_articulation": resolved_id in report.articulation,
        "surprise": next((lb.surprise for lb in report.load_bearing if lb.path == resolved_id), None),
        "dissonance": report.dissonance_map.get(resolved_id),
        "prerequisites": list(report.prereq_map.get(store_key, [])),
        "unlocks": sorted(d for d, ps in report.prereq_map.items() if store_key in ps),
    }

    return {
        "note": resolved_id,
        "cluster": cluster_id,
        "degree": degree,
        "degree_rank": degree_rank,
        "betweenness": report.betweenness_map.get(resolved_id, 0.0),
        "out_links": out_links[:depth * 10],
        "backlinks": backlinks[:depth * 10],
        "bridges": bridges_involving,
        "diagnosis": diagnosis,
    }


# ---------------------------------------------------------------------------
# Ledger steering — silica_ledger_next / silica_ledger_update
# ---------------------------------------------------------------------------

class LedgerNextArgs(BaseModel):
    run_id: str = Field(description="Run ID returned by silica_vault_report")


# Fixed byte budget per drain — payloads carry ~4KB path chunks, and a 40-task
# run would otherwise land ~40k tokens in one tool result. Tool-arg promotion
# declined 2026-08-19: no caller wants a different slice.
LEDGER_DRAIN_BYTES = 12000


@tool(LedgerNextArgs, cls="atomic")
def silica_ledger_next(run_id: str) -> dict:
    """Return EVERY task that is ready to run now, as `tasks` — not one task.

    Each entry carries its own capability (tool name), validated payload, and
    reason. Run them all: they are the frontier, so nothing in the list waits
    on anything else in it, and one silica_ledger_update per task records the
    outcomes. `remaining` counts the ready tasks that did not fit this drain —
    call again only when it is above zero. Returns {"done": true} when the plan
    is exhausted.
    """
    import orjson
    from pathlib import Path
    from silica.kernel.progress import ProgressLedger

    try:
        progress = ProgressLedger.load(run_id)
    except FileNotFoundError:
        return {"error": f"Run '{run_id}' not found"}
    except Exception as exc:
        return {"error": f"Failed to load ledger: {exc}"}

    ready = progress.ready_pending()
    if not ready:
        return {"done": True}

    tasks: list[dict] = []
    budget = LEDGER_DRAIN_BYTES
    for t in ready:
        # Load payload from disk if available
        payload: dict = {}
        if t.input_ref:
            try:
                payload = orjson.loads(Path(t.input_ref).read_bytes())
            except Exception:
                pass

        entry = {
            "task_id": t.id,
            "capability": t.capability_name,
            "payload": payload,
            "reason": payload.get("_reason", ""),
            "needs_confirmation": payload.get("needs_confirmation", False),
            "attempts": t.attempts,
        }
        budget -= len(orjson.dumps(entry))
        # The first task ships whatever it weighs: a payload fatter than the
        # whole budget must still be servable, or the run stalls forever.
        if tasks and budget < 0:
            break
        tasks.append(entry)

    return {"tasks": tasks, "remaining": len(ready) - len(tasks)}


class LedgerUpdateArgs(BaseModel):
    run_id: str = Field(description="Run ID")
    task_id: str = Field(description="Task ID returned by silica_ledger_next")
    status: str = Field(description="Outcome: done | failed | skipped | blocked")
    error: str = Field(default="", description="Error message if status is 'failed'")

@tool(LedgerUpdateArgs, cls="atomic")
def silica_ledger_update(run_id: str, task_id: str, status: str, error: str = "") -> dict:
    """Mark a task's outcome on the run's ProgressLedger and persist it.

    Returns {"ok": true, "digest": ...} so the agent has the updated state
    for the next iteration.
    """
    from silica.kernel.progress import ProgressLedger

    try:
        progress = ProgressLedger.load(run_id)
    except FileNotFoundError:
        return {"error": f"Run '{run_id}' not found"}
    except Exception as exc:
        return {"error": f"Failed to load ledger: {exc}"}

    try:
        if status == "done":
            progress.mark_done(task_id)
        elif status == "failed":
            progress.mark_failed(task_id, error=error)
        else:
            progress.set_status(task_id, status, error=error or None)  # type: ignore[arg-type]
        progress.save()
    except KeyError:
        return {"error": f"Task '{task_id}' not found in run '{run_id}'"}
    except Exception as exc:
        return {"error": f"Failed to update ledger: {exc}"}

    return {"ok": True, "digest": progress.digest()}



# ---------------------------------------------------------------------------
# Study loop
# ---------------------------------------------------------------------------

class QuizResult(BaseModel):
    path: str = Field(description="Note the question was drawn from (wikilink name or vault-relative path)")
    correct: bool = Field(description="True if the reader's answer was right")
    concepts: list[str] = Field(default_factory=list, description="The 1-3 concepts this question actually tested, as you named them when writing it")
    q: str = Field(default="", description="The question text as asked — lets later rounds avoid re-asking it")
    anchor: str = Field(default="", description="Optional citation into the note: '#Heading', or '#^id' only if that block id already exists in the note body")


class RecordQuizArgs(BaseModel):
    results: list[QuizResult] = Field(description="One entry per graded question")


@tool(RecordQuizArgs, cls="atomic")
def silica_record_quiz(results: list) -> dict:
    """Record graded quiz answers so the notes the reader failed resurface.

    Call once after grading, one entry per question. Writes no note — derived
    state feeding silica_review_queue and the report's attention list.
    Concepts are logged raw, exactly as spelled.
    """
    from silica.kernel.report import quiz

    entries = []
    for r in results:
        r = r if isinstance(r, dict) else r.model_dump()
        name = str(r.get("path") or "").strip()
        if not name:
            continue
        try:  # resolve wikilink names to the path the report keys on
            name = DRIVER.read_note(name).ref.path or name
        except Exception:
            pass  # unresolvable: log the reader's spelling rather than drop the answer
        entries.append({
            "path": name, "correct": bool(r.get("correct")),
            "concepts": r.get("concepts") or [],
            "q": r.get("q") or "", "anchor": r.get("anchor") or "",
        })

    written = quiz.record(entries)
    return {"recorded": written, "wrong": sum(1 for e in entries if not e["correct"])}


class ReviewQueueArgs(BaseModel):
    limit: int = Field(default=10, description="How many notes to return (global picker mode)")
    target: str = Field(default="", description="Optional vault path prefix: report EVERY note under it with its retention state instead of picking — the calibration read a study plan starts from")


class DoctorArgs(BaseModel):
    live: bool = Field(default=False, description="Also run one tiny PAID completion against the configured model — proves the model actually answers, not just that the config looks right. Off by default because it costs money.")


@tool(DoctorArgs, cls="atomic")
def silica_doctor(live: bool = False) -> dict:
    """Silica's own health: model, endpoints, vault, indexes, hooks — the
    `silica doctor` checks as data. Call when a capability behaves as if off
    (relatedness finds nothing, a write lands nowhere) instead of guessing.
    Credentials redacted. Never autostarts local servers: reports the state
    as it is right now. `live=true` adds the one paid check: a real completion.
    """
    from silica.config import CONFIG
    from silica.onboarding import checks

    results = checks.run_checks(CONFIG)
    if live:
        results.append(checks.live_probe(CONFIG))
    return checks.report_payload(results)


class ChangesArgs(BaseModel):
    scope: str = Field(default="session", description="'session' = this process (default); 'vault' = every session on this vault, rows with session/ts/mine")
    since: str = Field(default="", description="vault scope: ISO-8601; only rows first touched after it (pass the last `ts` you saw)")
    limit: int = Field(default=200, description="vault scope: newest rows kept")


@tool(ChangesArgs, cls="atomic")
def silica_changes(scope: str = "session", since: str = "", limit: int = 200) -> dict:
    """What changed in the vault: path, created/modified/moved/deleted, lines
    added/removed, measured against the file as it is now (an /undo empties
    its row). scope='session' is this process; scope='vault' is every
    session's ledger on this vault, so another client or a pipeline shows up
    with `session`, `ts`, `mine`; poll with `since`.
    """
    import datetime

    from silica.kernel.write import session_changes

    if scope == "vault":
        cut = None
        if since.strip():
            try:
                cut = datetime.datetime.fromisoformat(since.strip()).timestamp()
            except ValueError:
                return {"error": f"since must be ISO-8601, got {since!r}",
                        "scope": scope, "total": 0, "changes": []}
        rows = session_changes.history(since=cut, limit=limit)
        for r in rows:
            # Full precision on purpose: handed back as `since`, it must exclude
            # exactly this row, and seconds would replay it.
            r["ts"] = datetime.datetime.fromtimestamp(r["ts"]).isoformat()
        return {"scope": "vault", "session": session_changes.SESSION,
                "total": len(rows), "changes": rows}
    if scope != "session":
        return {"error": f"scope must be 'session' or 'vault', got {scope!r}",
                "scope": scope, "total": 0, "changes": []}
    rows = session_changes.rows()
    return {"scope": "session", "total": len(rows), "changes": rows}


@tool(ReviewQueueArgs, cls="atomic")
def silica_review_queue(limit: int = 10, target: str = "") -> list:
    """What the reader should review next — the learner model's picker.

    Rows carry estimated retention R (0..1, null = never measured) and a pool:
    "due" (decaying, worst first), "unexplored" (no quiz evidence, AI-written
    first), "known" (target mode only). R derives from creation dates,
    `AI: true` authorship and the graded-quiz ledger — writing counts as
    learning once; quizzes are the only proof since.
    """
    from silica.kernel.report import learner

    rows = learner.review_queue(limit=limit, target=target)
    return [
        {
            "path": r["path"],
            "R": None if r["R"] is None else round(r["R"], 3),
            "why": r["why"], "misses": r["misses"], "attempts": r["attempts"],
            "ai": r["ai"],
            # V2: what to read first, and whether those are known yet. Target
            # mode already returns rows in prerequisite order.
            "prereqs": r.get("prereqs", []), "ready": r.get("ready", True),
        }
        for r in rows
    ]
