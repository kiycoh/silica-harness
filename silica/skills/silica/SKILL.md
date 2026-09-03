---
name: silica
description: Use the Silica vault as persistent memory over MCP: recall before answering, capture after learning. Trigger when the user asks what they know/decided/wrote about a topic ("what do I have on X", "cosa so su", "avevo deciso"), wants something saved for later ("save this to the vault", "salva nel vault", "ricordati che"), references their vault or past notes, or when the session produced a decision or insight worth keeping across sessions.
---

# Silica: vault memory over MCP

Silica is a deterministic knowledge-graph engine over an Obsidian vault. This
skill is the usage loop; the `silica` MCP server carries the mechanics
(tools named `silica_*`; if they are deferred, load them with ToolSearch).
If the tools are missing entirely, say so and give the user the install line:

```bash
silica setup claude       # or: setup codex, setup dsh, setup opencode
```

It registers the server at user scope (every project, each serving the vault
of the folder it is opened in). With no `silica` on PATH, the same thing by
hand:

```bash
claude mcp add --scope user --transport stdio silica -- uvx --from 'silica-harness[mcp]' silica mcp
```

## Recall: search before answering

Any question about accumulated knowledge starts here, not with your own
recollection.

**Start with `silica_recall {query, k?}`.** It runs the fused retrieval and
hands back an answer-ready context: each note's query-densest window under a
rank/evidence/date header, recalled personal facts first. Answer from
`context`, and re-read only the notes named in `partial`, the rest arrived
whole. One call replaces stitching the probes below together yourself. When
the reply carries a `stale` key, those notes are flagged out of date: against
the code they document (`cosmetic`, `structural`), or, at level `source`,
against the source they were distilled from, which has since been
re-nucleated at another version. Cite them with that caveat, never as current.

Refine with those when `recall` is the wrong shape:

- Bare ranked list by meaning: `silica_semantic_search {query, k}`.
- Exact strings (error messages, names, quotes): `silica_search_context {query}`.
- Known title: `silica_search {query}`, then read it.
- Temporal ("when", "before/after", "most recent"): `silica_timeline
  {start?, end?, limit?}`, the chronological index of dated notes. Consult it
  before free-text recall, then read the linked note.

**Another vault may hold the answer.** `silica_vaults {query?}` lists the
vaults this machine knows (the active one, the personal-memory vault, every
adopted vault served here or listed by Obsidian) and, given a query, has each
vault's own index nominate a few notes and the cross-encoder score them. Read
`home` first: the vaults that hold the answer, `[]` when none does, `null`
when there is no calibrated verdict (no reranker), in which case judge by
`top` and read `coverage` before the score: `cold` means never indexed, not
"knows nothing". Then `silica_recall {query, vault: <path>}` answers from that
vault without leaving this one: read-only, the session's vault and writes stay
put. Re-read a note it lists under `partial` with
`silica_read_note {name, vault: <the same path>}`, and a note under `memory`

Never conclude "nothing in the vault" from a single miss: try at least one
semantic and one literal probe. If retrieval keeps coming back empty, or a
capability behaves as if switched off, call `silica_doctor` instead of
guessing. It reports model, endpoints, vault and index state as data, so a
missing embeddings index is a fact you read rather than an error string you
match. While the semantic leg is down, `silica_search` and
`silica_search_context` are grep-based and always work; tell the user that
`/embed`, `/cooccur` and `/lexical` in the Silica REPL rebuild whichever
index doctor reports missing.

## Ground: read before citing

- `silica_read_note {name}` before quoting or acting on a hit; for long notes
  `silica_outline {name}` first, to target the right section.
- `silica_links {name}` / `silica_props {name}` give the note's neighborhood
  and frontmatter when you need context around a hit.
- The graph leg answers what no search can: `silica_related {note, k?}` for a
  note's fused shortlist of neighbors (its `distance` is wikilink hops, so a
  high score at null distance is a missing link worth proposing),
  `silica_concepts {term?, note?}` for the embedder-free co-occurrence view of
  a term or a note, `silica_graph_explain {note}` for structural position
  (cluster, bridge, orphan).
- `silica_read_note` opens markdown only, so a PDF or a source file can fail
  to read while being perfectly present. `silica_exists {path}` tells "I
  cannot read it" apart from "it is not there", and `silica_files {folder}`
  lists notes and ingestible source files alike: a folder of code is not
  empty just because it holds no `.md`.
- A note you find wrong or stale while using it: `silica_flag_note {name,
  reason}` marks it `contested`, which demotes it at recall instead of hiding
  it, edits nothing, and reverses with `clear=True`. Flagging is not fixing;
  the human decides.

## Code and data: navigate, don't grep

- `silica_tables {folder?, column?}` is the census of the vault's tabular
  files (csv/tsv/parquet, Excel): schema per file, and `column=` answers
  "which table holds NEET?" in one call instead of head-reading every file.
  Then `silica_query_table {path, sql}` runs one read-only SELECT over that
  file (`SUMMARIZE t` first when columns are unknown; trust the schema its
  replies carry, never a guess).
- `silica_code_pack {target}` packs one source file with its real
  dependencies inside a character budget: use it before rewriting or porting
  a file, instead of ten greps.
- `silica_impact {range_spec?}` maps a code change (uncommitted by default)
  to the notes documenting the changed files and their import neighbors,
  classified cosmetic/structural: the blast-radius read before and after
  editing code the vault documents.

## Capture: write what deserves to outlive the session

What belongs: decisions and their why, non-obvious constraints, distilled
understanding, hard-won references. What does not: transcripts, code the repo
already holds, anything you could regenerate. Silica's quality gates reject
low-density notes, so write like you'd want to re-read.

1. Search first (dedup). If a note on the concept exists, extend it:
   `silica_patch_note {name, heading, snippet, source_basename}`.
   `source_basename` is provenance (the file or conversation the snippet
   came from).
2. New concept → `silica_write_note {path, body, title?, tags?, related?,
   parent?, template?}`. `body` is markdown only: frontmatter comes from the
   structured fields, scalar extras go in `props`, and a leading YAML block
   in `body` is stripped rather than honoured. It refuses to overwrite by
   design: an "already exists" error means patch instead. In a codebase
   vault, `documents` binds the note to the source files it describes,
   validated against the repo: that binding is what later makes recall
   report the note as `stale`.
3. Note shape: one atomic concept per note; YAML frontmatter with tags;
   `[[wikilinks]]` to the related notes your searches surfaced; write in the
   vault's language (read one existing note if unsure).

An appointment or a deadline is not a note: `silica_event_create {title,
start, end?, rrule?, reminder?, body?}` files it as a calendar note,
`silica_event_update {note, ...}` moves or closes the series, and
`silica_agenda {start?, days?}` reads a day back.

## Know the boundary

The MCP surface is the fast path: search, read, single-note writes. Bulk work
lives in the Silica REPL, run as `/nucleate`, `/report`, `/curate` inside
`silica`: multi-file nucleation with quality gates, dedup sweeps, taxonomy,
structural reports. When the task is bulk-shaped, say so and point there
instead of simulating the pipeline note by note.

That split belongs to the default registration, not to MCP: `silica mcp --all`
serves 58 tools instead of the curated 21, bulk operations included. Worth
naming if the user wants that surface in their own client, but it is their
call to widen it, never a lever to reach for mid-task.
