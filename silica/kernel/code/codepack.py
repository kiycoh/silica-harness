# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""codepack - deterministic context pack for one source file (spec-code-recall).

Not a ranker: a budgeted closure over facts the codegraph store already holds.
In the vault relevance is unknown and has to be estimated (RRF fusion, reranker);
in code it is given, and it is reachability. So no embeddings (D1), no language
server (ADR-0022), no scoring. Same repo state, same bytes, like
codegraph._serialize.

The asymmetry that makes the pack worth its budget: verbatim only what you are
about to rewrite, signatures for everything around it (D5).

# ponytail: kill by 2026-10-28 if unused. Exposure (re-verified 2026-08-19):
# NOT in the chat agent's default toolset. `silica_code_pack` is in
# constraints._CHAT_EXCLUDED, so a chat turn only sees it when the user names
# it (constraints._summoned). Reachable otherwise behind `silica mcp --all`.
# Instrumented 2026-08-19: every invocation stamps a timestamp line to
# <index>/codepack_usage.log (codedocs_tool._stamp_code_pack_use), so at the
# kill date `grep -c . ~/.silica/index/*/codepack_usage.log` answers the
# disuse question with data instead of chat silence.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from silica.kernel.code import codegraph
from silica.kernel.recall import paths as _paths

BUDGET_CHARS = 24000


def _emit(entry: dict, drop, doc: bool = False) -> str:
    """Signatures of a file, one per line, methods indented. This is the
    "everything around it" half of D5, and the fallback when the target itself
    is too big to serve whole.

    `drop(sym, name, parent)` decides what this caller hides. Whatever it hides
    also hides everything declared inside it, at any depth. A one-level rule
    re-parented an inner class's methods onto the outer class (false, and for
    the outline a duplicate of text already served verbatim above it) or left
    them as orphan indented lines with no owner at all.

    `hidden` is keyed by bare name, the only key `parent` ever carries, so a
    survivor re-opens its own name the moment it is emitted: two unrelated
    classes can share a simple name (`Outer.Builder` hidden, `Other.Builder`
    kept) and the first must not silence the second's members. Document order
    makes this safe, since `ModuleSkeleton.symbols` always emits a class before
    its own members (base.py, and every walker's class handler appends the
    class before recursing into its body)."""
    lines: list[str] = []
    hidden: set[str] = set()
    for s in entry.get("symbols", []):
        name, parent = s.get("name", ""), s.get("parent", "")
        if (parent and parent in hidden) or drop(s, name, parent):
            hidden.add(name)
            continue
        hidden.discard(name)  # a later same-named symbol that survives re-opens the name
        note = f"  # {s['doc']}" if doc and s.get("doc") else ""
        lines.append(("  " if parent else "") + s.get("signature", "") + note)
    return "\n".join(lines)


def _outline(entry: dict, skip: str = "") -> str:
    """The target's own signatures, minus `skip` (the symbol already served
    verbatim above the outline) and everything declared inside it. `skip` is a
    bare name or `Parent.name`."""
    head = skip.split(".", 1)[0]

    def addressed(s, name, parent):
        qual = f"{parent}.{name}" if parent else name
        return bool(skip) and (qual == skip or ("." not in skip
                and (parent == head or (parent == "" and name == head))))

    return _emit(entry, addressed, doc=True)


# Declaration node types per family, the same sets the codeast walkers use.
# C and C++ are absent on purpose: their names sit inside `declarator`, so a
# whole-file pack is the honest degrade (D4).
# ponytail: add the C/C++ declarator walk only if a real C target asks for it
_DECL_NODES: dict[str, tuple[str, ...]] = {
    "python": ("class_definition", "function_definition"),
    "java": ("class_declaration", "interface_declaration", "enum_declaration",
             "record_declaration", "annotation_type_declaration",
             "method_declaration", "constructor_declaration"),
    "typescript": ("class_declaration", "abstract_class_declaration",
                   "interface_declaration", "function_declaration",
                   "method_definition"),
}
_DECL_NODES["javascript"] = _DECL_NODES["typescript"]


def _find_decl(node, src: bytes, kinds: tuple[str, ...], name: str):
    """Shallowest declaration node of `kinds` whose `name` field reads `name`.
    Breadth-first on purpose: a nested class or method with the same name must
    never shadow the top-level one the selector addresses."""
    level = [node]
    while level:
        nxt = []
        for parent in level:
            for i in range(parent.named_child_count):
                child = parent.named_child(i)
                if child.type in kinds:
                    field = child.child_by_field_name("name")
                    if field is not None and src[field.start_byte:field.end_byte].decode(
                            "utf-8", errors="replace") == name:
                        return child
                nxt.append(child)
        level = nxt
    return None


def _symbol_source(source: str, language: str, selector: str) -> str | None:
    """Verbatim declaration text for "Class", "Class.member" or a top-level
    name. Reparses this one file: no offset is persisted, so Symbol and
    STORE_VERSION stay untouched (D7). None when the family has no selector
    table or the name is not there."""
    kinds = _DECL_NODES.get(language or "")
    if not kinds:
        return None
    try:
        from tree_sitter_language_pack import get_parser
        src = source.encode("utf-8")
        node = get_parser(language).parse(src).root_node
    except Exception:
        return None
    outer, _, inner = selector.partition(".")
    node = _find_decl(node, src, kinds, outer)
    if node is not None and inner:
        node = _find_decl(node, src, kinds, inner)
    if node is None:
        return None
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _target_block(source: str, entry: dict, selector: str, budget_chars: int,
                  dropped: list[str]) -> tuple[str, str]:
    """(body, mode). Verbatim when it fits, the selected symbol plus the rest
    as an outline when a selector resolves, the file outline otherwise. An
    empty outline is not an improvement, so the truncated source stays: never
    serve less than the target (spec section 6)."""
    if selector:
        picked = _symbol_source(source, entry.get("language") or "", selector)
        if picked is not None:
            rest = _outline(entry, skip=selector)
            tail = f"\n\n-- rest of file, outline --\n{rest}" if rest else ""
            return picked.rstrip("\n") + tail, "symbol"
        dropped.append(
            f"note: selector '{selector}' not found, degraded to a file-level pack")
    if len(source) <= budget_chars:
        return source.rstrip("\n"), "verbatim"
    outline = _outline(entry)
    if not outline:
        return source.rstrip("\n"), "verbatim"
    return outline, "outline"


# Each `extends`/`implements` clause is captured up to the next such keyword
# (or the end of the signature): a plain `[^{]+` is greedy and lets `extends`
# swallow a trailing `implements` clause whole, losing its bases.
_JAVA_SUPER = re.compile(
    r"\b(?:extends|implements)\s+((?:(?!\bextends\b|\bimplements\b)[^{])+)"
)
_PY_BASES = re.compile(r"\bclass\s+\w+\s*\(([^)]*)\)")
_CPP_BASES = re.compile(r"\b(?:class|struct)\s+\w+\s*:\s*([^{]+)")
# A trailing comment is cut off the signature once, before any of the three
# patterns above run, rather than excluded inside each of them. A declaration
# line is stored verbatim including its comment, so `class Config:  # noqa`
# would otherwise read `noqa` as a declared base, and `implements B // see C, D`
# would read `D`. Fabricating a base is worse than missing one: a porting agent
# acts on it. Matching the comment TOKENS and not the characters matters: a
# bare `/` is division, and excluding the character would cut the closing `>`
# off `Array<T, SIZE/2>`, leaving the comma to split one base into two.
_COMMENT = re.compile(r"//|/\*|#")
# Dots and colons included so a qualified name (`com.example.Base`,
# `Map.Entry`, `std::runtime_error`) is captured whole and reduced to its last
# segment, rather than tokenized into a package/namespace head that the
# first-identifier rule below would then mistake for the base itself.
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.:]*")
_QUALIFIER = re.compile(r"::|\.")
_GENERIC = re.compile(r"<[^<>]*>|\[[^\[\]]*\]")
_ACCESS = frozenset({"public", "private", "protected", "virtual"})


def _supertypes(signature: str) -> list[str]:
    """Declared bases of a class signature. The hierarchy is already in the
    store: `signature` is the declaration line verbatim, in all four families
    (spec section 1), so this is a regex and not an index."""
    signature = _COMMENT.split(signature, 1)[0]
    groups = [m.group(1) for m in _JAVA_SUPER.finditer(signature)]
    for rx in (_PY_BASES, _CPP_BASES):
        m = rx.search(signature)
        if m:
            groups.append(m.group(1))
    out: list[str] = []
    for group in groups:
        # Strip bracket-nested spans (generic/template parameters) before
        # splitting on comma: a multi-parameter generic base like
        # `HashMap<K, V>` or `B[C, D]` must not read as two bases. Repeated
        # until stable so a nested generic strips from the inside out.
        prev = None
        while prev != group:
            prev, group = group, _GENERIC.sub("", group)
        for part in group.split(","):
            if "=" in part:
                continue  # a Python keyword argument (metaclass=...), not a base
            # first identifier only: it drops C++ access keywords. A
            # qualified name reduces to its last segment on either separator,
            # the class itself rather than its package, namespace or
            # enclosing type.
            idents = [i for i in _IDENT.findall(part) if i not in _ACCESS]
            if idents:
                # Empty segments dropped, so a name the signature cuts short
                # (`extends B.`, a declaration wrapped after the dot) still
                # yields `B` instead of nothing.
                segments = [seg for seg in _QUALIFIER.split(idents[0]) if seg]
                base = segments[-1] if segments else ""
                if base and base not in out:
                    out.append(base)
    return out


def _hierarchy(graph, path: str, entry: dict) -> list[tuple[str, str]]:
    """(label, line) pairs: the declared bases of each top-level class in the
    target, then every class in the repo that declares one of them.

    Known limitation, by design (declared facts only, no resolution): the
    repo scan matches bases by bare name, so two same-named classes in
    different packages can produce a false `Base <- path#Sub` edge."""
    out: list[tuple[str, str]] = []
    own: list[str] = []
    for s in entry.get("symbols", []):
        if s.get("kind") != "class" or s.get("parent"):
            continue
        name = s.get("name", "")
        own.append(name)
        bases = _supertypes(s.get("signature", ""))
        if bases:
            out.append((name, f"{name} extends {', '.join(bases)}"))
    if graph is None or not own:
        return out
    for p in sorted(graph.files):
        if p == path:
            continue
        for s in graph.files[p].get("symbols", []):
            if s.get("kind") != "class":
                continue
            for base in _supertypes(s.get("signature", "")):
                if base in own:
                    label = f"{p}#{s.get('name', '')}"
                    out.append((label, f"{base} <- {label}"))
    return out


def _signatures(entry: dict) -> str:
    """Public signatures of a neighbour file. Public is spelled per family: no
    leading underscore (Python, TS; dunder names excepted, since a constructor
    is exactly the contract a port needs to read), no `private` modifier token
    anywhere in the declaration prefix (Java, which catches `private void f()`
    and the legal-but-reordered `static private void f()` alike). A private
    inner class takes its own public methods down with it, via the transitive
    rule in `_emit`.

    Known limitation: this cannot filter a private C++ member. `codeast/c.py`
    never records the `access_specifier` node (`private:` is a class-body
    section label, a sibling of the members it governs, not a per-member
    modifier token), so a private C++ method's `signature` looks identical to
    a public one. Teaching codeast about access specifiers is out of scope,
    since `codeast.Symbol` must not change, so C++ neighbour signatures
    currently show every member, public or not."""
    # Python has no `private` modifier: it spells private with the underscore
    # rule above, so there the token can only be a symbol's own name, and
    # hiding a public function over what it is called is wrong.
    modifiers = (entry.get("language") or "") != "python"

    def private(s, name, parent):
        prefix = s.get("signature", "").split("(", 1)[0].split()
        underscored = name.startswith("_") and not (
            name.startswith("__") and name.endswith("__"))
        return underscored or (modifiers and "private" in prefix)

    return _emit(entry, private)


# An import/include line always names the file it resolves to, so searching
# it unmasked would make every import a "mention" by definition and the
# filter below would exclude nothing.
# `using` names another file in C++ (`using std::string;`) EXCEPT when it
# declares a type alias (`using Alias = Neighbor;`), which is a real use and
# often the only place a neighbour is named. Masking it hid that neighbour.
_IMPORT_START = re.compile(r"^[ \t]*(?:(?:import|from|#include)\b|using\b(?![^=;]*=))")
_OPEN, _CLOSE = "([{", ")]}"


def _mask_imports(source: str) -> str:
    """`source` with every import/include statement blanked to spaces.

    Blanked, not deleted, so every offset outside the masked spans stays
    exactly where it was: `_first_mention` orders neighbours by first real use
    and that only means anything if its offsets are positions in the real file.

    A statement is not a line. Anchoring on the first physical line left the
    braced multi-line form (`import {\\n Foo,\\n} from "./mod";`, prettier's
    default past 80 columns) mostly unmasked, names and module specifier
    included, which switched the mention filter off for idiomatic TypeScript
    and for the parenthesized Python form. So the mask runs on while the
    brackets opened on the import line are still unbalanced, and stops on the
    first line that closes them. Counting brackets is a heuristic, and a
    deliberately cheap one: the alternative is a parse per neighbour.

    A blank line ends the run unconditionally, because no import statement
    spans one. Without that stop an unbalanced bracket the counter cannot
    close (`import os  # TODO(alice`) masks to end of file, and since a
    neighbour that is never mentioned is never a candidate, the whole
    neighbourhood would vanish with nothing in `dropped` to say why."""
    lines = source.split("\n")
    depth, masking = 0, False
    for i, line in enumerate(lines):
        if not masking:
            if not _IMPORT_START.match(line):
                continue
            masking, depth = True, 0
        elif not line.strip():
            masking, depth = False, 0
            continue
        depth += sum(line.count(c) for c in _OPEN) - sum(line.count(c) for c in _CLOSE)
        lines[i] = " " * len(line)
        if depth <= 0:
            masking, depth = False, 0
    return "\n".join(lines)


def _first_mention(source: str, path: str, entry: dict) -> int | None:
    """Offset of the first place `source` names this file: any top-level
    symbol name, or the file stem. `source` is expected to already have its
    import/include statements masked to blanks (`_mask_imports`), otherwise
    an import's own line always matches its own target, and the filter that
    is supposed to separate real uses from bare imports never excludes
    anything. None means it is never named outside an import, so it is not a
    neighbour. The filter took the median neighbourhood from 9 to 3 and the
    median pack from 6588 to 2351 characters (spec section 2)."""
    names = {s.get("name", "") for s in entry.get("symbols", []) if not s.get("parent")}
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if stem not in ("__init__", "index"):
        names.add(stem)
    # One alternation, one scan: the leftmost match IS the earliest mention, so
    # this replaces a search per name plus a running minimum. Alternation
    # backtracks, so a name that prefixes another (`Vec` before `Vec2`) still
    # matches at the same offset the longer one does.
    alt = "|".join(re.escape(n) for n in sorted(names) if n)
    m = re.search(rf"\b(?:{alt})\b", source) if alt else None
    return m.start() if m is not None else None


def _neighborhood(graph, path: str, entry: dict,
                  source: str) -> list[tuple[str, str]]:
    """(label, block) pairs: resolved imports first, then package siblings,
    each group by first mention with the path as the tiebreak (spec section 4).
    An import crosses a package boundary, so it is a contract you cannot see by
    opening the folder next door; a sibling is one `ls` away.

    Mentions are searched for over `source` with its own import/include
    statements masked to same-length blanks first (`_mask_imports`): an import
    always names the file it imports, so searching the raw source would make
    the mention filter vacuous for group 1 (every import "mentions itself").
    Same-length blanks keep the rest of the offsets meaningful, so
    `sorted(ranked)` still orders by real position in the file."""
    if graph is None:
        return []
    imports = [p for p in entry.get("imports", []) if p != path and p in graph.files]
    siblings: list[str] = []
    if entry.get("language") == "java":
        # D6: in Java the directory is the package, and a package sibling is
        # referenced with no import at all. One rule, instantiated per language:
        # every other family sees nothing beyond its explicit imports.
        folder = path.rpartition("/")[0]
        siblings = sorted(
            p for p, e in graph.files.items()
            if p != path and p not in imports
            and p.rpartition("/")[0] == folder and e.get("language") == "java"
        )
    body = _mask_imports(source)
    out: list[tuple[str, str]] = []
    for group in (imports, siblings):
        ranked = []
        for p in group:
            at = _first_mention(body, p, graph.files[p])
            if at is not None:
                ranked.append((at, p))
        for _, p in sorted(ranked):
            sigs = _signatures(graph.files[p])
            if sigs:
                out.append((p, f"{p}\n{sigs}"))
    return out


def code_pack(vault: Path | str, target: str,
              budget_chars: int = BUDGET_CHARS,
              sections: list[str] | None = None) -> dict:
    """Context pack for `target` ("path", "path#Class" or "path#Class.member").

    `sections` (None = all) names which of hierarchy / neighborhood / external
    / importers to emit; the target is always served. The tool is stateless,
    so the second pack in the same package would otherwise repay the same
    neighbourhood outline (paths.py came back in every pack under
    kernel/write, 2026-09-03) with no way to say "already seen".

    Raises ValueError when the target file cannot be read: a bad path is a
    caller mistake, not a state to degrade around. Every other shortfall
    (no repo, no graph, unsupported language, empty neighbourhood) degrades to
    a poorer pack and says why in `dropped` (D4).

    `dropped` holds two kinds of entry, in this order and told apart by their
    prefix. First the degrade notes, `"note: <what was given up and why>"`,
    which name no fetchable thing. Then the budget drops,
    `"<section>: <label>"`, each one a real entry the caller can ask for
    directly; that second part is always a suffix of the entry order, because
    the fill stops at the first entry that does not fit and never lets a later
    small entry jump the queue. The two used to share one shape, so an agent
    reading `dropped` as a fetch list would go looking for a file called
    "no code graph (vault is not inside a git repo)".
    """
    path, _, selector = target.partition("#")
    root = _paths.repo_root_for(vault) or Path(vault)
    # `target` arrives raw from a model (silica_code_pack), and `root / path`
    # silently discards the root when `path` is absolute and joins a `..`
    # verbatim, so the read below is the exfiltration seam: without this the
    # pack serves /etc/passwd or ~/.ssh/id_rsa into the caller's context.
    # Same boundary the code source adapter enforces (sources/code.py), through
    # the shared containment choke point rather than a second copy of the rule.
    try:
        path = _paths.contain_in_vault(path, root)
    except ValueError as exc:
        raise ValueError(f"target escapes the repository: {path!r} ({exc})") from exc
    try:
        source = (root / path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"cannot read target: {path} ({exc})") from exc

    graph = codegraph.load_codegraph(vault)
    dropped: list[str] = []
    if graph is None:
        dropped.append("note: no code graph (vault is not inside a git repo)")
        entry: dict = {}
        head_ref = ""
    else:
        head_ref = graph.head_ref
        entry = graph.files.get(path, {})
        if not entry:
            dropped.append(f"note: {path} is not in the code graph")

    # `_target_block` only sees the body, so the header line and the pack's own
    # trailing newline have to come out of its budget too, or a larger budget
    # can select verbatim over an outline that would have fit inside it.
    # "verbatim" is the longest mode word, so sizing on it bounds every mode.
    overhead = len(f"## target {path} @ {head_ref} mode: verbatim\n") + 1
    body, mode = _target_block(source, entry, selector, budget_chars - overhead, dropped)
    chunks = [f"## target {path} @ {head_ref} mode: {mode}\n{body}"]
    emitted_sections: dict[str, list[str]] = {"target": [path]}
    # one scan of the graph, not two: `fan-in` is defined as len(importers)
    importers = [(p, p) for p in (graph.importers(path) if graph is not None else [])]
    # A Python stdlib import is not a dependency anyone needs to fetch, and
    # listed in `dropped` it read as a fetchable section ("external: hashlib").
    # Other languages keep their externals: no stdlib roster to check against.
    stdlib = sys.stdlib_module_names if entry.get("language") == "python" else frozenset()
    external = [(d, d) for d in entry.get("external", []) if d.split(".")[0] not in stdlib]
    used = len(chunks[0]) + 1  # + the trailing newline the pack always ends with
    stop = False
    for name, entries in (
        ("hierarchy", _hierarchy(graph, path, entry)),
        ("neighborhood", _neighborhood(graph, path, entry, source)),
        ("external", external),
        ("importers", importers),
    ):
        if sections is not None and name not in sections:
            continue
        # len(entries), not len(emitted): the count is the repo-wide total even
        # when the budget trimmed the list printed underneath it.
        header = f"## {name}" + (f" (fan-in {len(entries)})" if name == "importers" else "")
        emitted: list[str] = []
        for label, block in entries:
            # a chunk costs "\n\n" + header + "\n" the first time, "\n" after
            cost = len(block) + (len(header) + 3 if not emitted else 1)
            if stop or used + cost > budget_chars:
                stop = True   # dropped is a suffix: a later small entry never jumps the queue
                dropped.append(f"{name}: {label}")
                continue
            used += cost
            emitted.append(block)
            emitted_sections.setdefault(name, []).append(label)
        if emitted:
            chunks.append(header + "\n" + "\n".join(emitted))
    out = {
        "text": "\n\n".join(chunks) + "\n",
        "sections": emitted_sections,
        "dropped": dropped,
        "target_mode": mode,
        "truncated": mode != "verbatim",
        "head_ref": head_ref,
        "target_chars": len(source),
    }
    if mode == "outline":
        # The one number the caller needs to escalate: the budget at which
        # `_target_block` would have served the file whole.
        out["verbatim_at"] = len(source) + overhead
    return out
