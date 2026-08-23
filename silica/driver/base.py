# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Obsidian Driver — L0 abstraction over the vault I/O substrate.

From SILICA.md §3 L0:
  Adapter typed by DOMAIN, not by transport. Everything else talks to the
  Driver, never to disk directly. Two interchangeable backends:
  - fs: direct filesystem + index (derived from Hermes scripts)
  - ws: the Obsidian bridge plugin, installed live by `silica connect`

This module defines the Protocol (interface), domain types, and the
global DRIVER instance selected at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Domain types & Exceptions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoteRef:
    """Reference to a note in the vault."""
    name: str           # wikilink-style name (no extension)
    path: str = ""      # relative path within vault (folder/note.md)


@dataclass
class NoteContent:
    """Full content of a note."""
    ref: NoteRef
    content: str


@dataclass
class Hit:
    """Search result with context."""
    ref: NoteRef
    line: int = 0
    snippet: str = ""
    # Total occurrences of the query in this note's body — the density signal.
    # Backends cap how many Hits they materialize per note, so len(hits) alone
    # saturates at the cap and cannot rank two heavy notes. 0 = not measured
    # (backends without the counter): consumers fall back to len(hits).
    note_matches: int = 0


@dataclass
class Heading:
    """A heading in a note's outline."""
    level: int          # 1-6
    text: str
    position: int = 0   # char offset


@dataclass(frozen=True)
class Link:
    """A link between notes."""
    source: NoteRef
    target: str         # wikilink target (may be unresolved)


@dataclass
class GraphSnapshot:
    """Snapshot of the vault graph for non-regression diffing."""
    orphans: list[NoteRef] = field(default_factory=list)
    unresolved: list[Link] = field(default_factory=list)
    link_counts: dict[str, int] = field(default_factory=dict)   # note -> outgoing count
    backlink_counts: dict[str, int] = field(default_factory=dict)  # note -> incoming count


@dataclass
class Txn:
    """Transaction handle for snapshot/rollback.

    Rollback strategies (C3 / ADR-0009):
      - inverses:       authoritative list of InverseOp — consumed by silica_restore and
                        the ROLLBACK state. Single source of truth.
      - created_paths:  derived from inverses (delete_created entries); same reason.
    """
    id: str
    refs: list[NoteRef] = field(default_factory=list)
    created_paths: list[str] = field(default_factory=list)   # paths created by write ops
    inverses: list = field(default_factory=list)              # list[InverseOp] — real field, not dynamic attr

    @property
    def inverses_serialized(self) -> list[dict]:
        """Return a JSON-serializable list of dicts representing the inverse operations."""
        serialized = []
        for inv in self.inverses:
            if hasattr(inv, "model_dump"):
                serialized.append(inv.model_dump())
            elif isinstance(inv, dict):
                serialized.append(inv)
            else:
                try:
                    serialized.append(dict(inv))
                except Exception:
                    pass
        return serialized


# ---------------------------------------------------------------------------
# GraphIndexMixin — helpers shared verbatim by both backends
# ---------------------------------------------------------------------------

class GraphIndexMixin:
    """Graph-index helpers shared by the fs and ws backends.

    Subclasses build the in-memory index in ``_ensure_graph()`` and expose
    the ``_notes``/``_unresolved_links``/``_graph``
    attributes it populates, plus the three note primitives ``upsert`` needs.

    Both halves of that contract are declared below rather than left implicit:
    a mixin that reads names it does not define is only safe while every
    subclass happens to define them, and nothing was checking that.
    """

    # Built by the subclass's _ensure_graph(); read by every helper here.
    _notes: dict[str, NoteRef]
    _unresolved_links: set
    _graph: Any

    def _ensure_graph(self) -> None:
        raise NotImplementedError

    def _add_link_edge(self, source: str, target: str, *, scaffold: bool) -> None:
        """One resolved wikilink edge, carrying its class (kernel.link.ast).

        Two spellings can resolve to one note (an alias in the frontmatter and
        the title in prose); prose wins, so a scaffold occurrence never
        downgrades an edge the prose already justified.
        """
        if self._graph.has_edge(source, target):
            if not scaffold:
                self._graph[source][target]["scaffold"] = False
            return
        self._graph.add_edge(source, target, scaffold=scaffold)

    # Provided by the concrete backend; upsert() composes them.
    def read_note(self, path: str) -> Any:
        raise NotImplementedError

    def create(self, path: str, content: str) -> NoteRef:
        raise NotImplementedError

    def overwrite(self, path: str, content: str) -> NoteRef:
        raise NotImplementedError

    def _node_ref(self, path: str) -> NoteRef:
        if path in self._notes:
            return self._notes[path]
        name = path.rsplit("/", 1)[-1].removesuffix(".md")
        return NoteRef(name=name, path=path)

    def graph_data(self) -> tuple[dict, set, Any]:
        """Return (notes, unresolved_links, graph) for in-process consumers."""
        self._ensure_graph()
        return self._notes, self._unresolved_links, self._graph

    def upsert(self, path: str, content: str) -> NoteRef:
        """Write content unconditionally: create when missing, overwrite when present.

        For callers whose semantics are "refresh this staging/derived note"
        (inbox conversion, source stubs, undo restore) — NOT for vault write
        ops, which must fail loudly on an unexpected existing note (create).
        Existence-probe via read_note keeps this backend-agnostic (create's
        raise type differs across backends).
        """
        try:
            self.read_note(path)
        except Exception:
            return self.create(path, content)
        return self.overwrite(path, content)


# ---------------------------------------------------------------------------
# ObsidianDriver Protocol — the domain interface (SILICA.md §3 L0)
# ---------------------------------------------------------------------------

@runtime_checkable
class ObsidianDriver(Protocol):
    """Domain-typed interface to an Obsidian vault.

    Freshness contract (NORMATIVE from SILICA.md):
      The Driver MUST declare read-after-write semantics. After a create/
      set_prop/move, the Driver guarantees that the next read reflects the
      mutation. If the underlying cache updates asynchronously, the backend
      MUST wait/poll until settled. A method that doesn't respect the same
      freshness contract on both backends is a bug, not a difference.
    """

    # -- discovery / read --------------------------------------------------

    def search_names(self, query: str) -> list[NoteRef]:
        """Search vault note names matching query."""
        ...

    def search_context(self, query: str) -> list[Hit]:
        """Search vault content with line-level context snippets."""
        ...

    def search_context_batch(self, queries: list[str]) -> dict[str, list[Hit]]:
        """Like search_context, but for many queries in one call.

        Key = query, value = the Hits (ref + line + snippet) the corresponding
        single search_context call would return. Additive: single-query callers
        stay on search_context.
        """
        ...

    def read_note(self, ref: NoteRef | str) -> NoteContent:
        """Read a note's full content by name or ref."""
        ...

    def props_of(self, ref: NoteRef | str) -> dict:
        """Read frontmatter properties (~hundreds of tokens, no body)."""
        ...

    def outline(self, ref: NoteRef | str) -> list[Heading]:
        """Get the heading tree of a note."""
        ...

    # -- graph -------------------------------------------------------------

    def links(self, ref: NoteRef | str) -> list[NoteRef]:
        """Outgoing links from a note."""
        ...

    def backlinks(self, ref: NoteRef | str) -> list[NoteRef]:
        """Incoming links to a note."""
        ...

    def orphans(self) -> list[NoteRef]:
        """Notes with no incoming links."""
        ...

    def unresolved(self) -> list[Link]:
        """Unresolved wikilinks in the vault."""
        ...

    def graph_snapshot(self, refs: list[NoteRef] | None = None) -> GraphSnapshot:
        """Graph snapshot for non-regression gating.

        If refs is provided, performs an incremental snapshot covering only
        the touched notes and their 1-hop neighborhood.
        """
        ...

    # -- write (graph-safe) ------------------------------------------------

    def create(self, path: str, content: str) -> NoteRef:
        """Create a new note. Path is relative to vault root. Raises if file exists."""
        ...

    def upsert(self, path: str, content: str) -> NoteRef:
        """Write content unconditionally: create when missing, overwrite when present.

        Concrete implementation in GraphIndexMixin (Protocol bodies are not
        inherited by structural implementors).
        """
        ...

    def overwrite(self, path: str, content: str) -> NoteRef:
        """Overwrite an existing note in-place, preserving history.

        Unlike delete+create, this MUST NOT destroy Obsidian's version history
        or break block-references. Use for patch and overwrite op types.
        The CLI backend uses `obsidian create path=... overwrite=true`.
        The FS backend writes the file directly.
        """
        ...

    def append(self, ref: NoteRef | str, content: str) -> None:
        """Append content to an existing note."""
        ...

    def set_prop(self, ref: NoteRef | str, name: str, value: Any, type_: str = "text") -> None:
        """Set a frontmatter property on a note."""
        ...

    def move(self, ref: NoteRef | str, to: str) -> None:
        """Move/rename a note. Updates wikilinks (graph-safe)."""
        ...

    def delete(self, ref: NoteRef | str) -> None:
        """Delete a note from the vault."""
        ...

    def autolink_note(
        self,
        path: str,
        candidates: list[str] | None = None,
        title_index: list[str] | None = None,
    ) -> list[str]:
        """Wrap unlinked mentions of vault titles in `path` with links, in place.

        Returns the list of titles linked. The CLI backend delegates skip-region
        detection, link resolution, and link rendering to Obsidian's own engine
        (respecting the user's link-format preference). The FS backend uses the
        deterministic pure-Python autolink() kernel. `candidates` optionally
        restricts which titles are considered (embedding/cluster-prioritised subset).
        `title_index` optionally supplies a prebuilt disambiguated vault-title
        list so callers batching many notes avoid a per-note rebuild; when
        None the backend builds its own.
        """
        ...

    def alias_index(self) -> list[tuple[str, list[str]]]:
        """(title, aliases) pairs for notes declaring frontmatter aliases.

        Feeds autolink.build_alias_map so a mention of a note's alias links to
        the note itself. Backends that delegate autolinking to Obsidian return
        an empty list: Obsidian resolves aliases on its own.
        """
        ...

    # -- advanced ----------------------------------------------------------

    def list_files(self, folder: str = "") -> list[NoteRef]:
        """List all markdown files, optionally filtered by folder."""
        ...

    def list_inbox_files(self) -> list[NoteRef]:
        """List all files in the inbox directory."""
        ...

    # -- graph data (in-process, avoids O(N) subprocess calls) -------------

    def graph_data(self) -> tuple[dict, set, Any]:
        """Return (notes, unresolved_links, graph) for in-process consumers.

        Ensures the graph index is populated first. Used by graph_export to
        avoid O(N) CLI calls while keeping the contract explicit.
        """
        ...

    # -- transactionality --------------------------------------------------

    def snapshot_versions(self, refs: list[NoteRef]) -> Txn:
        """Snapshot current versions for later rollback."""
        ...

    def restore(self, txn: Txn) -> None:
        """Rollback a transaction.

        CAPABILITY GAP: rollback completeness is backend-dependent.
          - created_paths (undo write ops): honored by both backends.
          - versions (undo patch ops via history): CLI-backend only; the FS
            backend has no version history and no-ops these (logged).
        Prefer content-based rollback (InverseOp.restore_version with
        prior_content, applied via silica_restore) for backend-agnostic undo;
        this version-based path is a fallback only.
        """
        ...
