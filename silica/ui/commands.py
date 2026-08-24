# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Command:
    name: str
    group: str       # "workflow" | "direct" | "system"
    usage: str
    summary: str
    examples: tuple[str, ...] = ()
    # REPL-only: driven by the terminal session itself, so the web GUI cannot run
    # it as a chat turn (it has its own affordance instead — new chat, the header
    # panel, the thinking toggle). Kept out of the GUI's command picker.
    repl_only: bool = False


COMMANDS: tuple[Command, ...] = (
    # Workflow — agent-directed
    Command(
        name="/report",
        group="workflow",
        usage="[folder] [--top-k=N] [--embeddings]",
        summary="structural audit of the vault → steering loop",
        examples=(
            "/report Concepts/ML",
            "/report --embeddings",
            "/report --cooccurrence",
            "/report --top-k=15 --embeddings",
        ),
    ),
    Command(
        name="/nucleate",
        group="workflow",
        usage="<file...> [--target=DIR] [--hub=H]",
        summary="bring files in: notes via Injector FSM, code as skeleton stubs",
        examples=(
            "/nucleate Inbox/note.md --target=Concepts/AI",
            "/nucleate silica/cli.py",
            "/nucleate paper.pdf --target=Concepts/AI",
        ),
    ),
    Command(
        name="/promote",
        group="workflow",
        usage="[<key>]",
        summary="session memory → a note: list what keeps recurring, promote one through the gate",
        examples=("/promote", "/promote user.dog.name"),
    ),
    Command(
        name="/episodes",
        group="direct",
        usage="[--save=<path>]",
        summary="show what session memory holds: live chains, dated, grouped by key; writes nothing",
        examples=("/episodes", "/episodes --save=~/episodes.md"),
    ),
    Command(
        name="/agenda",
        group="direct",
        usage="[today|week|YYYY-MM-DD]",
        summary="per-day merge of events, dated notes, agent activity and review due",
        examples=("/agenda", "/agenda week", "/agenda 2026-09-01"),
        repl_only=True,  # the GUI has its own calendar tab
    ),
    Command(
        name="/convert",
        group="direct",
        usage="<file...> [--target=DIR]",
        summary="transcode a non-.md file (PDF) into a markdown note in the inbox",
        examples=(
            "/convert paper.pdf",
            "/convert paper.pdf --target=Concepts/AI",
        ),
    ),
    Command(
        name="/web-search",
        group="direct",
        usage='"<concept>" [--max-searches=N]',
        summary="research a concept on the web → cited findings note in the Inbox (then /nucleate)",
        examples=(
            '/web-search "retrieval-augmented generation"',
            '/web-search "graph neural networks" --max-searches=6',
        ),
    ),
    Command(
        name="/web",
        group="workflow",
        usage="[keywords]",
        summary="answer from the web instead of the vault, cited; bare /web re-asks your last question",
        examples=(
            "/web graph rewiring benchmarks",
            "/web",
        ),
    ),
    Command(
        name="/keep",
        group="direct",
        usage="",
        summary="save the last /web answer as a cited note in the Inbox (then /nucleate)",
    ),
    Command(
        name="/fetch",
        group="direct",
        usage="<url>",
        summary="read one URL (YouTube gives its transcript) → verbatim note in the Inbox",
        examples=(
            "/fetch https://arxiv.org/abs/2005.11401",
            "/fetch https://www.youtube.com/watch?v=aircAruvnKk",
        ),
    ),
    Command(
        name="/organize",
        group="workflow",
        usage='"<intent>" [--scope=FOLDER] [--file=taxonomy.yaml] [--merge] [--move-uncategorized] [--apply]',
        summary="classify and reorganize vault notes according to a taxonomy",
        examples=(
            '/organize "put AI notes in Concepts/AI, cooking notes in Life"',
            '/organize "archive Acme docs under Clients/Acme" --merge',
            "/organize --file=taxonomy.yaml --apply",
            "/organize --scope=Inbox",
        ),
    ),
    # Reader — agent-directed, strictly read-only (output in chat, never writes)
    Command(
        name="/summarize",
        group="workflow",
        usage="<note|folder...>",
        summary="read-only digest of one or more notes in chat (key points, tables)",
        examples=(
            "/summarize Concepts/AI/RAG.md",
            "/summarize Concepts/ML",
        ),
    ),
    Command(
        name="/explain",
        group="workflow",
        usage='"<concept>" [--level=intro|expert]',
        summary="explain a concept grounded in the vault, at the chosen register",
        examples=(
            '/explain "retrieval-augmented generation"',
            '/explain "backpropagation" --level=intro',
        ),
    ),
    Command(
        name="/compare",
        group="workflow",
        usage='"<A>" "<B>" [...]',
        summary="comparison table of notes/concepts; surfaces contradictions",
        examples=('/compare "RAG" "fine-tuning"',),
    ),
    Command(
        name="/quiz",
        group="workflow",
        usage="[note|folder] [--n=10]",
        summary="active-recall quiz; graded answers resurface the notes you miss. No target = review queue",
        examples=("/quiz Concepts/ML --n=5", "/quiz"),
    ),
    Command(
        name="/learn",
        group="workflow",
        usage="<area|folder|note|topic>",
        summary="guided re-learning: builds (or resumes) a syllabus note calibrated on what you still retain, then teaches step by step with quiz gates",
        examples=("/learn Concepts/ML", '/learn "differential forms"'),
    ),
    Command(
        name="/relate",
        group="workflow",
        usage="<note> [--n=8]",
        summary="typed relationship map: how/why one note relates to its vault neighbors",
        examples=("/relate Concepts/AI/RAG.md --n=6",),
    ),
    Command(
        name="/schematize",
        group="workflow",
        usage="<note|folder|topic> [--save=<path>]",
        summary="Markdown table schematizing a note, folder, or topic",
        examples=("/schematize Concepts/ML", '/schematize "the ingest pipeline"'),
    ),
    Command(
        name="/diagram",
        group="workflow",
        usage="<note|folder|topic> [--save=<path>]",
        summary="Mermaid diagram of a note, folder, or topic",
        examples=("/diagram kernel/codegraph.py", '/diagram "the ingest pipeline" --save=Concepts/diagram.md'),
    ),
    # Direct — immediate, no LLM round-trip
    Command(
        name="/status",
        group="direct",
        usage="[run_id]",
        summary="progress digest of the last run",
    ),
    Command(
        name="/embed",
        group="direct",
        usage="[folder] [--force]",
        summary="build/update embedding index",
    ),
    Command(
        name="/cooccur",
        group="direct",
        usage="[folder] [--force]",
        summary="build/update co-occurrence index (without embedder)",
    ),
    Command(
        name="/lexical",
        group="direct",
        usage="[folder] [--force]",
        summary="build/update lexical (BM25/fuzzy) index",
    ),
    Command(
        name="/wiki",
        group="direct",
        usage="[folder|path] [--overview-only] [--force]",
        summary="behavioral code wiki: ARCHITECTURE.md + one note per subsystem",
        examples=("/wiki", "/wiki kernel", "/wiki silica/kernel",
                  "/wiki /core/src/main/java/io/github/app/manager",
                  "/wiki --overview-only", "/wiki --force"),
    ),
    Command(
        name="/graph",
        group="direct",
        usage="[out.html] [folder]",
        summary="export knowledge graph",
    ),
    Command(
        name="/map",
        group="direct",
        usage="<note> [--force]",
        summary="radial mind-map rooted on a note → maps/<stem>.canvas",
    ),
    Command(
        name="/find",
        group="direct",
        usage="<query> [--k=N]",
        summary="semantic search",
    ),
    Command(
        name="/changes",
        group="direct",
        usage="",
        summary="notes this session wrote to, with added/removed line counts",
        repl_only=True,  # the GUI has its own Changes drawer, off the same ledger
    ),
    Command(
        name="/undo",
        group="direct",
        usage="[note-path]",
        summary="undo the last patch on a note",
    ),
    Command(
        name="/review",
        group="direct",
        usage="[--flush=HASH]",
        summary="inspect the async review queue (deferred ops)",
    ),
    Command(
        name="/anneal",
        group="direct",
        usage="[--steer] [--limit=N]",
        summary="retry every deferred bundle against the current vault; --steer escalates what still fails",
        examples=("/anneal", "/anneal --steer", "/anneal --limit=5"),
    ),
    Command(
        name="/revert",
        group="direct",
        usage="[run-id | --source <file>]",
        summary="revert a whole injection (per-run, LIFO), or every run derived from one source",
    ),
    Command(
        name="/dedup",
        group="direct",
        usage="[folder]",
        summary="deduplicate (sub-agent)",
    ),
    Command(
        name="/curate",
        group="direct",
        usage="[folder] [--apply]",
        summary="curate the vault: plan autolink/orphan/dedup/refine work (dry-run; --apply executes)",
    ),
    Command(
        name="/aliases",
        group="direct",
        usage="[folder] [--apply]",
        summary="propose frontmatter aliases for note titles (abbreviations, spellings); dry-run, --apply writes",
        examples=("/aliases", "/aliases Concepts/AI --apply"),
    ),
    Command(
        name="/refine",
        group="direct",
        usage="[folder]",
        summary="enrich and normalize notes (sub-agent)",
    ),
    Command(
        name="/enrich",
        group="direct",
        usage="[folder]",
        summary="enrich note semantics (sub-agent)",
    ),
    Command(
        name="/stale",
        group="direct",
        usage="[--all]",
        summary="list notes whose documents: sources changed structurally (--all includes cosmetic)",
    ),
    Command(
        name="/impact",
        group="direct",
        usage="[<git-range>]",
        summary="changed files → affected notes (documenting + 1-hop import neighbors); no range = uncommitted changes",
    ),
    Command(
        name="/plans",
        group="direct",
        usage="",
        summary="list plans/ notes grouped by status: (todo|in-progress|blocked|done)",
    ),
    Command(
        name="/path",
        group="direct",
        usage="<noteA> <noteB>",
        summary="shortest reading path between two notes (wikilinks + co-occurrence)",
        examples=('/path "RAG" "Transformers"',),
    ),
    Command(
        name="/contested",
        group="direct",
        usage="",
        summary="list notes flagged contested: true with their unresolved contradictions",
    ),
    # System
    Command(
        name="/sessions",
        group="system",
        usage="[prune <days>d]",
        summary="list saved conversations (narration + legacy); prune deletes old ones",
        examples=("/sessions", "/sessions prune 90d"),
        repl_only=True,
    ),
    Command(
        name="/resume",
        group="system",
        usage="<n|id>",
        summary="reopen a saved conversation and continue it",
        examples=("/resume 1",),
        repl_only=True,
    ),
    Command(
        name="/new",
        group="system",
        usage="",
        summary="start a fresh conversation (same as /clear); the old one stays resumable",
        repl_only=True,
    ),
    Command(
        name="/vault",
        group="system",
        usage="[path]",
        summary="show the active vault, or switch to another for this session",
    ),
    Command(
        name="/settings",
        group="system",
        usage="[<key> <value|none>]",
        summary="view or edit vault.yaml settings (language, tags) without the wizard",
        examples=(
            "/settings",
            "/settings conventions.reply_language italian",
        ),
    ),
    Command(
        name="/help",
        group="system",
        usage="",
        summary="show this help",
        repl_only=True,
    ),
    Command(
        name="/model",
        group="system",
        usage="",
        summary="show the current LLM model",
        repl_only=True,
    ),
    Command(
        name="/tools",
        group="system",
        usage="",
        summary="list registered tools",
        repl_only=True,
    ),
    Command(
        name="/clear",
        group="system",
        usage="",
        summary="reset conversation history",
        repl_only=True,
    ),
    Command(
        name="/verbose",
        group="system",
        usage="",
        summary="cycle tool progress: off → new → all → verbose",
        repl_only=True,
    ),
    Command(
        name="/thinking",
        group="system",
        usage="",
        summary="toggle display of the reasoning block",
        repl_only=True,
    ),
    Command(
        name="/incognito",
        group="system",
        usage="",
        summary="toggle capture of this session off",
        repl_only=True,
    ),
    Command(
        name="/exit",
        group="system",
        usage="",
        summary="exit silica",
        repl_only=True,
    ),
)


def command_names() -> tuple[str, ...]:
    return tuple(c.name for c in COMMANDS)


def render_help() -> None:
    from rich.padding import Padding

    from silica.ui.console import CONSOLE
    from silica.ui.style import GROUP_STYLE, command_table

    CONSOLE.print()
    CONSOLE.print("  [bold]silica commands[/]")
    CONSOLE.print()

    workflow = [c for c in COMMANDS if c.group == "workflow"]
    direct = [c for c in COMMANDS if c.group == "direct"]
    system = [c for c in COMMANDS if c.group == "system"]

    CONSOLE.print(f"  [bold {GROUP_STYLE['workflow']}]Workflow[/]  [dim]· agent-directed[/]")
    CONSOLE.print(Padding(command_table(workflow, name_style=f"bold {GROUP_STYLE['workflow']}"), (0, 0, 0, 4)))
    CONSOLE.print()
    CONSOLE.print()

    CONSOLE.print(f"  [bold {GROUP_STYLE['direct']}]Direct[/]  [dim]· immediate, no LLM[/]")
    CONSOLE.print(Padding(command_table(direct, name_style=f"bold {GROUP_STYLE['direct']}"), (0, 0, 0, 4)))
    CONSOLE.print()
    CONSOLE.print()

    sys_line = "  ·  ".join(c.name for c in system)
    CONSOLE.print(f"  [bold {GROUP_STYLE['system']}]System[/]")
    CONSOLE.print(f"    [dim]{sys_line}[/]")
    CONSOLE.print()
