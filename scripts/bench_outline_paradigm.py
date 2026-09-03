#!/usr/bin/env python
"""A/B prototype: outline-first nucleation vs the shipped keyphrase pipeline.

Measured 2026-09-02 on the ML lecture vault: the shipped pipeline lets a
keyphrase miner decide the units (n_tok // 20, cap 40), hands the distiller
{name, excerpt} pairs with no document structure, hides same-run notes from
the related candidates, and links by cosine + BM25 only. This script is the
"after" arm of the A/B that decides whether flipping that is worth building:
the model reads the WHOLE lecture once and emits the ideas, their source
section and typed dependencies (stage A); then sees the vault outline and
proposes cross-lecture edges with a reason (stage B). Everything mechanical
stays out of the loop on purpose: the question is what the model produces
when it is allowed to name the units.

Throwaway experiment harness, not product code: no dedup guard, no gates, no
provenance ledger. Usage:

    uv run python scripts/bench_outline_paradigm.py run  OUT_VAULT L11.md L12.md ...
    uv run python scripts/bench_outline_paradigm.py measure VAULT_DIR
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

RELATIONS = ["defines", "derives", "relaxes", "applies", "bounds", "contrasts",
             "instance_of", "motivates", "generalizes", "justifies", "same_as"]

STAGE_A = """You are reading one complete lecture of a university course, converted to markdown (LaTeX kept, image descriptions inside <details>). Reconstruct the ARGUMENT of the lecture as a small graph of ideas, the way a careful student would after studying it.

Emit ONLY a JSON object:
{{
 "lesson_title": "<short title of what this lecture is about, in {lang}>",
 "spine": ["<idea title>", ...],
 "ideas": [
   {{
     "title": "<noun phrase in {lang} naming ONE idea; never a sentence fragment>",
     "section": "<heading of the source section(s) the idea comes from, verbatim>",
     "claim": "<one sentence in {lang}: what this idea asserts>",
     "depends_on": [{{"title": "<another idea title from THIS list>", "relation": "<one of {rels}>", "why": "<one sentence in {lang}>"}}]
   }}
 ]
}}
Rules:
- "spine" lists every idea title exactly once, in the order the argument develops.
- Between 6 and 16 ideas. Merge repeated slides of the same idea into one idea. Skip noise (author, email, course header, empty slides).
- Every depends_on.title MUST be another idea title of this same JSON. Relation meanings: defines (A is defined in terms of B), derives (A is derived from B), relaxes (A weakens B's assumptions), applies (A applies B), bounds (A bounds B), contrasts (A vs B), instance_of, motivates (A is why B is needed), generalizes, justifies.
- No markdown fences around the JSON.
"""

STAGE_A2 = """You are given one complete lecture (markdown, LaTeX kept) and a list of ideas already identified in it (title | source section). For EACH listed idea write its note body in {lang}: the facts themselves (definitions, formulas in LaTeX verbatim, theorem statements, algorithm steps), copied from the source, never outside knowledge, never a description of the source. Markdown. Keep every formula that belongs to the idea, drop repeated slides.

Emit ONLY a JSON object: {{"bodies": {{"<idea title verbatim>": "<markdown body>", ...}}}}
JSON strings: escape every backslash (write \\\\alpha for \\alpha), newlines as \\n. No markdown fences.
"""

STAGE_B = """You maintain a knowledge vault built from the lectures of one course. Below is the OUTLINE of the vault so far (one line per existing note: lesson | title | claim), then the ideas of a NEW lecture (title | claim).

Find the connections a careful student would draw between the NEW ideas and the EXISTING notes: where a new idea applies, relaxes, generalizes, bounds, justifies, contrasts with, is an instance of, or is the same idea as an existing one. Propose only connections you can justify in one sentence from the two claims. Prefer specific notes over the lecture-level ones.

Emit ONLY a JSON object: {{"edges": [{{"from": "<new idea title>", "to": "<existing note title>", "relation": "<one of {rels}>", "why": "<one sentence in {lang}>"}}]}}
"from" MUST be a new idea title and "to" an existing note title, both verbatim. Use "same_as" only for the same idea restated (its body will be merged into the existing note). No markdown fences.
"""

COST: list[dict] = []


def _model() -> str:
    from silica.config import CONFIG
    return CONFIG.worker_model or CONFIG.model


RAW_DIR: Path | None = None


def _ask(system: str, user: str, *, max_tokens: int) -> dict:
    from silica.agent.llm import call_llm
    t0 = time.time()
    # reasoning=False: the first probe (2026-09-02, deepseek-v4-flash) spent the
    # whole 8k completion budget thinking and returned no JSON at all.
    resp = call_llm(_model(), [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                    max_tokens=max_tokens, temperature=0.0, reasoning=False)
    u = resp.usage or {}
    COST.append({"prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
                 "seconds": round(time.time() - t0, 1), "finish": resp.finish_reason})
    text = (resp.text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if RAW_DIR:
        (RAW_DIR / f"raw_{len(COST)}.txt").write_text(text, encoding="utf-8")
    return json.loads(text[start:end + 1])


def _slug(title: str) -> str:
    return re.sub(r'[/\\:*?"<>|]', "", title).strip()


def _note(path: Path, fm: dict, body: str) -> None:
    lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            lines.extend(f'  - "{x}"' for x in v)
        else:
            lines.append(f'{k}: {v}')
    lines += ["---", "", body.rstrip(), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n" + text + "\n", encoding="utf-8")


def _strip_images(md: str) -> str:
    return re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)


def run(out: Path, lessons: list[Path], *, lang: str = "italiano") -> None:
    global RAW_DIR
    RAW_DIR = out
    notes_dir = out / "Machine learning"
    outline: list[dict] = []  # {lesson, title, claim, path}
    hub = notes_dir / "Machine learning.md"
    if not hub.exists():
        _note(hub, {"type": "Note", "tags": ["machine-learning"]}, "# Machine learning\n\n## Lessons\n")
    for les in lessons:
        name = les.stem
        src = _strip_images(les.read_text(encoding="utf-8"))
        a = _ask(STAGE_A.format(lang=lang, rels=", ".join(RELATIONS[:-1])), src, max_tokens=6000)
        ideas = {i["title"]: i for i in a["ideas"]}
        titles = list(ideas)
        for k in range(0, len(titles), 8):
            batch = titles[k:k + 8]
            bodies = _ask(STAGE_A2.format(lang=lang),
                          "LECTURE:\n" + src + "\n\nIDEAS:\n"
                          + "\n".join(f"{t} | {ideas[t].get('section', '')}" for t in batch),
                          max_tokens=12000).get("bodies", {})
            for t in batch:
                ideas[t]["body"] = bodies.get(t, "")
        merged: dict[str, str] = {}  # new title -> existing title (same_as)
        edges: list[dict] = []
        if outline:
            b = _ask(STAGE_B.format(lang=lang, rels=", ".join(RELATIONS)),
                     "EXISTING NOTES:\n" + "\n".join(f"{o['lesson']} | {o['title']} | {o['claim']}" for o in outline)
                     + "\n\nNEW LECTURE IDEAS:\n" + "\n".join(f"{i['title']} | {i['claim']}" for i in a["ideas"]),
                     max_tokens=4000)
            known = {o["title"] for o in outline}
            edges = [e for e in b.get("edges", []) if e.get("from") in ideas and e.get("to") in known]
            merged = {e["from"]: e["to"] for e in edges if e["relation"] == "same_as"}
        # Idea notes
        for title, idea in ideas.items():
            if title in merged:
                tgt = notes_dir / f"{_slug(merged[title])}.md"
                _append(tgt, f"\n## From: {name} ({idea['section']})\n\n{idea['body']}")
                continue
            deps = [d for d in idea.get("depends_on", []) if d.get("title") in ideas and d["title"] != title]
            rel_lines = [f"- {d['relation']} [[{merged.get(d['title'], d['title'])}]]: {d.get('why', '')}" for d in deps]
            rel_lines += [f"- {e['relation']} [[{e['to']}]] ({name} -> earlier lesson): {e['why']}"
                          for e in edges if e["from"] == title and e["relation"] != "same_as"]
            body = f"# {title}\n\n> {idea['claim']}\n\n{idea['body']}\n"
            if rel_lines:
                body += "\n## Relations\n" + "\n".join(rel_lines) + "\n"
            _note(notes_dir / f"{_slug(title)}.md", {
                "type": "Note", "AI": "true", "lesson": f'"{name}"',
                "sources": [f"{name}.md#{idea['section']}"],
                "parent note": f'"[[{name}]]"',
                "related": [merged.get(d["title"], d["title"]) and f"[[{merged.get(d['title'], d['title'])}]]" for d in deps]
                           + [f"[[{e['to']}]]" for e in edges if e["from"] == title and e["relation"] != "same_as"],
                "tags": ["machine-learning"],
            }, body)
        # Reverse side of cross edges on the existing notes
        for e in edges:
            if e["relation"] == "same_as":
                continue
            tgt = notes_dir / f"{_slug(e['to'])}.md"
            if tgt.exists():
                txt = tgt.read_text(encoding="utf-8")
                if "## Relations" not in txt:
                    _append(tgt, "\n## Relations")
                _append(tgt, f"- [[{e['from']}]] ({name}) {e['relation']} this: {e['why']}")
        # Lesson spine note + hub line
        spine = [t for t in a.get("spine", []) if t in ideas] or list(ideas)
        _note(notes_dir / f"{name}.md", {"type": "Note", "AI": "true", "sources": [f"{name}.md"],
                                          "parent note": '"[[Machine learning]]"', "tags": ["machine-learning"]},
              f"# {name}: {a.get('lesson_title', '')}\n\n"
              + "\n".join(f"{n}. [[{merged.get(t, t)}]]: {ideas[t]['claim']}" for n, t in enumerate(spine, 1)))
        _append(hub, f"- [[{name}]]: {a.get('lesson_title', '')}")
        outline += [{"lesson": name, "title": t, "claim": i["claim"]} for t, i in ideas.items() if t not in merged]
        (out / f"stage_{name}.json").write_text(json.dumps({"A": a, "edges": edges}, ensure_ascii=False, indent=1))
    (out / "cost.json").write_text(json.dumps({"model": _model(), "calls": COST,
                                               "prompt": sum(c["prompt"] or 0 for c in COST),
                                               "completion": sum(c["completion"] or 0 for c in COST),
                                               "seconds": round(sum(c["seconds"] for c in COST), 1)}, indent=1))
    print(json.dumps(json.loads((out / "cost.json").read_text()), indent=1))


# ----------------------------------------------------------------- measure --
_WL = re.compile(r"\[\[([^\]|#]+)")


def measure(vault: Path) -> dict:
    from silica.kernel.text.candidates import is_fragment
    notes: dict[str, dict] = {}
    for p in vault.rglob("*.md"):
        rel = p.relative_to(vault).as_posix()
        if rel.split("/")[0] in ("Inbox", "Done", "sources") or p.name.startswith("."):
            continue
        t = p.read_text(encoding="utf-8")
        fm = t.split("---", 2)[1] if t.startswith("---") else ""
        body = t.split("---", 2)[2] if t.startswith("---") else t
        # Two YAML dialects reach the vault: the template writes `- "x"`, the
        # overwrite path (expand / dedup settle) dumps `- x` or `- 'x'`.
        srcs = [x.strip().strip("\"'") for x in re.findall(r"^\s*- (.+)$", (re.search(r"sources:\n((?:\s*- .*\n)+)", fm) or [None, ""])[1], re.M)]
        parent = (re.search(r"parent note:\s*[\"']?\[\[([^\]]+)\]\]", fm) or [None, None])[1]
        section = (re.search(r'^section:\s*"?(.+?)"?\s*$', fm, re.M) or [None, ""])[1]
        # The FIRST source authored the note; a later lecture that patched a
        # relation line into it is stamped too, and must not make the pair intra.
        notes[p.stem] = {"sources": srcs, "section": section,
                         "lessons": {re.sub(r"\.md.*$", "", srcs[0])} if srcs else set(),
                         "parent": parent, "links": set(_WL.findall(body)) - {p.stem},
                         "related": set(re.findall(r"\[\[([^\]|]+)\]\]", (re.search(r"related:\n((?:\s*- .*\n)+)", fm) or [None, ""])[1])),
                         "typed": len(re.findall(r"^- \w+ \[\[|^- \[\[[^\]]+\]\] \([^)]+\) \w+ this", body, re.M))}
    ideas = {k: v for k, v in notes.items() if v["sources"] and not re.fullmatch(r"Lezione \d+", k) and v["section"] != "(outline)"}
    hub_names = {"Machine learning"} | {k for k in notes if re.fullmatch(r"Lezione \d+", k)}
    intra = cross = 0
    edge_rows = []
    for a, va in ideas.items():
        for b in (va["links"] | va["related"]):
            if b not in ideas or b == a:
                continue
            shared = va["lessons"] & ideas[b]["lessons"]
            kind = "intra" if shared else "cross"
            if shared:
                intra += 1
            elif va["lessons"] and ideas[b]["lessons"]:
                cross += 1
            edge_rows.append(f"{kind}: {a} -> {b}")
    junk = [k for k in ideas if is_fragment(k)]
    return {
        "idea_notes": len(ideas),
        "M1_section_provenance": round(sum(bool(v["section"]) or any("#" in s for s in v["sources"]) for v in ideas.values()) / max(1, len(ideas)), 2),
        "M2_is_fragment_titles": junk,
        "M3_intra_links": intra,
        "M3_parent_not_hub": round(sum(v["parent"] not in hub_names and v["parent"] != "Machine learning" for v in ideas.values()) / max(1, len(ideas)), 2),
        "M3_typed_edge_lines": sum(v["typed"] for v in ideas.values()),
        "M4_cross_links": cross,
        "titles": sorted(ideas),
        "edges": sorted(edge_rows),
    }


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]])
    else:
        print(json.dumps(measure(Path(sys.argv[2])), ensure_ascii=False, indent=1))
