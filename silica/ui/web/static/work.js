// Work: the right sidebar's third mode, and the only surface in this app that
// reads the narration. It was a column of its own at that same right edge until
// the note drawer, also at that edge, was found to be switching it off on every
// open; app.js owns the sidebar and this file fills one of its three panes.
//
// `silica/agent/narration.py` has written an append-only account of every run
// since it shipped, and `/narration/sse` has served it live for just as long.
// Nothing rendered it. So this file is a PROJECTION of a stream that already
// exists, not new plumbing: every duration, token delta, phase and write below
// is a record the agent already wrote.
//
// It loads after app.js and uses two of its globals on purpose — openNote() to
// read a source and send() to revert a write — rather than restating either.

// --- projection:begin --------------------------------------------------------
// Everything between this marker and projection:end is pure: no DOM, no fetch,
// no globals. tests/test_work_panel.py lifts the block out and runs it under
// node, which is how the fold below is checked at all.
// The narration payload carries the RAW tool name and the raw args, and the
// browser has no renderer, so the verb table has to live here too. A copy that
// nothing checks is a copy that goes stale on the next tool, so
// tests/test_work_panel.py pins it entry-for-entry to _TOOL_DESC in
// silica/ui/renderer.py and fails when the two disagree.
const TOOL_VERBS = {
  "silica_search": ["search", "query"],
  "silica_search_context": ["search", "query"],
  "silica_read_note": ["read", "name"],
  "silica_write_note": ["write note", "path"],
  "silica_patch_note": ["patch note", "name"],
  "silica_flag_note": ["flag note", "name"],
  "silica_record_quiz": ["record quiz", null],
  "silica_review_queue": ["review queue", null],
  "silica_props": ["props", "name"],
  "silica_outline": ["outline", "name"],
  "silica_links": ["links", "name"],
  "silica_backlinks": ["backlinks", "name"],
  "silica_orphans": ["orphans", null],
  "silica_unresolved": ["unresolved links", null],
  "silica_files": ["list files", "folder"],
  "silica_exists": ["exists", "path"],
  "silica_deferred_list": ["deferred list", null],
  "silica_deferred_flush": ["deferred flush", null],
  "silica_deferred_retry": ["deferred retry", null],
  "silica_anneal": ["anneal", null],
  "silica_inbox_ls": ["inbox", null],
  "silica_recon": ["recon", "inbox_file"],
  "silica_payload": ["payload", "recon_report_path"],
  "silica_sanitize": ["sanitize", "distiller_output_path"],
  "silica_validate_ops": ["validate", "ops_json_path"],
  "silica_bulk_write": ["bulk write", "ops_json_path"],
  "silica_lint": ["lint", "note_name"],
  "silica_run_injector": ["injector", "inbox_file"],
  "silica_delete": ["delete", "ref"],
  "silica_snapshot": ["snapshot", "ops_json_path"],
  "silica_restore": ["restore", "txn_id"],
  "silica_cleanup": ["cleanup", "inbox_file"],
  "web_search": ["web search", "query"],
  "web_fetch": ["web fetch", "url"],
  "silica_web_answer": ["web answer", "question"],
};

// How a tool changes the vault. Same keys as _TOOL_EFFECT in web/callback.py,
// and the reason a patch target is a WRITE here and never also a source: the
// panel's two lists are derived from one table, so a note cannot appear in both.
const WRITE_OPS = {
  "silica_write_note": "write",
  "silica_patch_note": "patch",
  "silica_flag_note": "flag",
  "silica_bulk_write": "bulk write",
  "silica_restore": "restore",
  "silica_delete": "delete",
  "silica_move": "move",
};

// Arg keys that name a note, the allowlist from web/callback.py. Missing one
// only drops a row from Sources; it never names the wrong note.
const NOTE_KEYS = ["name", "path", "note", "note_path", "ref"];

const fmtSecs = (s) => (s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m${String(Math.floor(s % 60)).padStart(2, "0")}s`);
const fmtTok = (n) => Number(n).toLocaleString("en-US");

function toolVerb(name) {
  const e = TOOL_VERBS[name];
  return e ? e[0] : String(name || "").replace(/^silica_/, "").replace(/_/g, " ");
}

function toolTarget(name, args) {
  const a = args || {};
  if (name === "silica_move") return `${a.ref || ""} → ${a.to || ""}`.trim();
  const e = TOOL_VERBS[name];
  const key = e && e[1];
  const raw = key ? String(a[key] == null ? "" : a[key]).trim() : "";
  // web_fetch's target is a URL, and a URL in a 372px column is a wall. The
  // host is what you judge a source by at a glance; the full link arrives in
  // the turn's own footer anyway.
  if (name === "web_fetch" && raw) { try { return new URL(raw).host; } catch { return raw; } }
  return raw;
}

// --- the projection ----------------------------------------------------------
// One pass over the beats of a session, returning the CURRENT run: a user turn
// starts a new one, so the panel shows what is happening rather than the whole
// file. Pinned by tests/test_work_panel.py, which runs this function under node.
function projectRun(beats) {
  let title = "", model = "", t0 = null, tLast = null;
  let rows = [], writes = [], sources = [];
  const seen = new Set();
  let spans = {};          // span id -> row (or {ts} for a call, which has no row)
  let openThoughts = [];   // innermost last: a call folds into the one it ran inside
  for (const b of beats || []) {
    const kind = b.kind, st = b.status, id = b.id, p = b.payload || {};
    const terminal = st !== "running";
    if (b.ts != null) tLast = b.ts;
    if (kind === "session") continue;
    if (kind === "turn") {
      const m = p.message || {};
      // Only the human's turn starts a run. The assistant and tool turns are
      // the same exchange, and resetting on them would blank the panel between
      // the question and the answer, which is exactly when it is being read.
      if (m.role === "user" && typeof m.content === "string") {
        title = m.content.split("\n")[0];
        t0 = b.ts; rows = []; writes = []; sources = []; seen.clear();
        spans = {}; openThoughts = [];
      }
      continue;
    }
    if (kind === "thought") {
      if (!terminal) {
        const row = { kind: "think", text: "", sub: "", dur: "", running: true, seq: b.seq };
        rows.push(row); spans[id] = { row, ts: b.ts }; openThoughts.push(row);
      } else if (spans[id]) {
        const row = spans[id].row;
        row.text = String(p.text || b.summary || "").trim();
        row.dur = fmtSecs(p.duration_s != null ? p.duration_s : b.ts - spans[id].ts);
        row.running = false;
        openThoughts = openThoughts.filter((r) => r !== row);
      }
      continue;
    }
    if (kind === "call") {
      // A call opens INSIDE the thought it produces and closes on the same tick,
      // and neither carries the other as a parent, so the fold keys on nesting
      // and not on `parent`. Printing both rows prints the same seconds twice.
      if (!terminal) { spans[id] = { ts: b.ts }; if (p.model) model = p.model; continue; }
      const bits = [];
      if (p.prompt_tokens != null) bits.push(`${fmtTok(p.prompt_tokens)} → ${fmtTok(p.completion_tokens || 0)} tok`);
      if (p.cached_tokens) bits.push(`${fmtTok(p.cached_tokens)} cached`);
      const sub = bits.join(" · ");
      const host = openThoughts[openThoughts.length - 1];
      if (host) { host.sub = sub; }
      else {
        rows.push({ kind: "call", verb: "call", target: p.model || model, sub: sub,
                    dur: spans[id] ? fmtSecs(b.ts - spans[id].ts) : "", running: false, seq: b.seq });
      }
      continue;
    }
    if (kind === "tool") {
      const name = p.name || "";
      if (!terminal) {
        const row = { kind: "tool", name: name, verb: toolVerb(name), target: toolTarget(name, p.args),
                      sub: "", dur: "", running: true, failed: false, seq: b.seq };
        rows.push(row); spans[id] = { row, ts: b.ts, args: p.args || {} };
        continue;
      }
      const sp = spans[id];
      if (!sp) continue;
      const row = sp.row;
      row.running = false;
      row.failed = st === "failed";
      row.dur = fmtSecs(b.ts - sp.ts);
      const op = WRITE_OPS[row.name];
      if (op) {
        writes.push({ op: op, target: row.target, failed: row.failed, seq: b.seq });
      } else {
        const key = NOTE_KEYS.find((k) => sp.args[k]);
        const nm = key ? String(sp.args[key]).trim() : "";
        if (nm && !seen.has(nm)) { seen.add(nm); sources.push({ name: nm, verb: row.verb }); }
      }
      continue;
    }
    if (kind === "write") {
      writes.push({ op: p.op || "write", target: p.path || b.summary || "",
                    failed: st === "rolled_back" || st === "failed", seq: b.seq });
      continue;
    }
    if (kind === "phase") {
      const label = String(p.phase || b.summary || "").trim();
      const pos = [];
      if (p.file_total) pos.push(`file ${p.file_idx || 0} of ${p.file_total}`);
      if (p.chunk_total) pos.push(`chunk ${p.chunk_idx || 0} of ${p.chunk_total}`);
      const last = rows[rows.length - 1];
      if (last && last.kind === "phase" && last.target === label && last.running) {
        last.running = !terminal; last.sub = pos.join(" · ");
        if (terminal && spans[id]) last.dur = fmtSecs(b.ts - spans[id].ts);
        continue;
      }
      rows.push({ kind: "phase", verb: "phase", target: label, sub: pos.join(" · "),
                  dur: "", running: !terminal, seq: b.seq });
      spans[id || label] = { ts: b.ts };
      continue;
    }
    if (kind === "cancel") {
      for (const r of rows) r.running = false;
      continue;
    }
  }
  const running = rows.some((r) => r.running);
  const tools = rows.filter((r) => r.kind === "tool").length;
  const thinks = rows.filter((r) => r.kind === "think").length;
  const shape = [thinks ? `${thinks} calls` : "", tools ? `${tools} tools` : ""].filter(Boolean).join(" · ");
  return { title, model, running, rows, writes, sources, shape,
           elapsed: t0 != null && tLast != null ? fmtSecs(tLast - t0) : "" };
}

// --- the report projection ---------------------------------------------------
// On the metrics view this column stops narrating the run and states the
// report. Everything the metrics view shows is derived from the report in front
// of it; the one thing it cannot derive is what MOVED, which is why the server
// keeps the previous reading (silica/kernel/report/history.py) and hands it back
// with the payload.
//
// Direction is half the reading: "5 closed" and "5 new" are the same integer and
// opposite news, so each signal carries the word for each direction rather than
// a signed number the reader has to interpret.
//
// A direction that names a noun carries both its forms, because "1 communities
// merged" is the kind of sentence that makes a reader stop trusting the number
// beside it.
const REPORT_ROWS = [
  ["unresolved", "unresolved", "closed", "new"],
  ["orphans", "orphans", "linked up", ["new note nothing links to", "new notes nothing links to"]],
  ["lean_notes", "lean notes", "filled out", ["new thin note", "new thin notes"]],
  // "areas > 1 note" and not "areas": the top strip states 91 areas for this
  // vault and the tile beside this panel states 24, because the strip counts
  // every community and the metrics view drops the singletons. The signal here
  // is the metrics one, so it carries the metrics view's own words rather than
  // a third number under a word already spoken for.
  ["areas", "areas > 1 note", ["community merged", "communities merged"],
   ["new community", "new communities"]],
  ["notes", "notes", "removed", "written"],
];

// Which way is good. `areas` and `notes` are in neither set on purpose: a vault
// that grew has more of both and nothing is wrong, so colouring the delta would
// congratulate or scold the user for writing.
const REPORT_GOOD_DOWN = new Set(["unresolved", "orphans", "lean_notes"]);

const plural = (word, n) => (Array.isArray(word) ? word[n === 1 ? 0 : 1] : word);

function projectReport(payload) {
  const rep = (payload && payload.report) || {};
  const now = rep.signals || {};
  const was = rep.previous || null;
  const rows = [];
  for (const [key, label, down, up] of REPORT_ROWS) {
    const v = now[key];
    if (v == null) continue;
    const w = was && was[key] != null ? was[key] : null;
    const d = w == null ? null : v - w;
    let sub;
    if (w == null) sub = "first reading";
    else if (d === 0) sub = `was ${fmtTok(w)} · unchanged`;
    else sub = `was ${fmtTok(w)} · ${fmtTok(Math.abs(d))} ${plural(d < 0 ? down : up, Math.abs(d))}`;
    rows.push({
      key: key, label: label, value: fmtTok(v), sub: sub, delta: d, was: w,
      good: d == null || d === 0 || !REPORT_GOOD_DOWN.has(key) ? null : d < 0,
    });
  }
  return {
    // The depth rides in the payload because the two are not the same report:
    // the co-occurrence leg is what separates them, and it is the expensive one.
    title: rep.depth === "full" ? "Structural audit · full report" : "Structural audit",
    meta: [rep.elapsed_s != null ? fmtSecs(rep.elapsed_s) : "",
           rep.notes != null ? `${fmtTok(rep.notes)} notes` : "",
           // Said out loud because it is the thing people assume otherwise: the
           // whole report is graph arithmetic over the vault, not a model call.
           "local, no model"].filter(Boolean),
    at: rep.at || null,
    since: rep.since || null,
    rows: rows,
  };
}

// --- the node projection -----------------------------------------------------
// This column states the node you pointed at, on every view and not only on
// explore. It is the ONLY reader of /context now. The note drawer used to draw
// the same payload as its `context` mode, and the two had already drifted: the
// drawer showed snippets this projection threw away, this projection showed
// degree the drawer threw away, and neither header said which of the two you
// were looking at.
//
// Two sources meet here and neither is asked twice: the graph's own message
// carries the facts it had to compute in order to DRAW the node (degree, state,
// area), and /context carries what the vault knows about it.
//
// `loading` is a state and not an absence: the head facts arrive with the click
// and the sections one round trip later, so the panel says the node's name
// immediately rather than staying on the previous node until the fetch lands.
function projectNode(node, ctx) {
  const n = node || {};
  const c = ctx || {};
  const rel = c.related || {};
  const sug = c.suggested || [];
  // The count is kept apart from the words because it is painted apart: it is
  // the one figure on this line, and three grey strings in a row read as one
  // sentence — the more so when the area's own label already contains a dot.
  const meta = [];
  // "note" is the default state, so printing it says nothing; hub and orphan
  // are the readings worth a word, and they come from nodeState() in the frame
  // rather than from a second threshold invented here.
  if (n.state && n.state !== "note") meta.push(n.state);
  if (n.area) meta.push(n.area);
  const name = (p) => String(p || "").split("/").pop().replace(/\.md$/, "");
  const linked = new Set(
    [...(rel.outgoing || []), ...(rel.backlinks || [])].map((r) => r.path).filter(Boolean));
  return {
    title: c.title || n.name || name(n.path),
    path: n.path || c.path || "",
    ghost: !!(n.ghost || c.ghost),
    count: n.links == null ? "" : n.links + (n.links === 1 ? " link" : " links"),
    meta: meta,
    loading: !ctx,
    error: c.error || "",
    concepts: (c.concepts || []).slice(0, 12),
    // What the note SAYS. It used to be the drawer's half and it is here for
    // one reason: this column is what you look at to decide whether to spend
    // the SECOND click, and "12 links, cluster dati" does not answer that. The
    // server already drops snippets under 700 characters, so a note short
    // enough to read whole never pays for an extract of itself.
    says: c.snippets || [],
    // A frontmatter `related:` is not a body wikilink. One is a claim the
    // author made about the note, the other is a sentence that happened to name
    // another note. But a note that is BOTH is one relation, not two, and
    // listing every entry stated those rows twice on one panel - which is the
    // duplicate this whole surface exists to remove, rebuilt one heading lower.
    // Measured on the dev vault the day this shipped, 7 of 7 entries were
    // duplicates, so the full list would have been pure noise and the filtered
    // one is empty until it has something to say.
    //
    // What is left is the half nothing else states: declared and never linked.
    // An entry that resolves to nothing has no path to match on, so a dead
    // `related:` still lists, which is the one case worth acting on.
    related: (rel.frontmatter || []).filter((r) => !r.path || !linked.has(r.path)),
    out: rel.outgoing || [],
    from: rel.backlinks || [],
    // Two flavours of one list, split because they are two different offers:
    // a ghost is a link you already wrote and never filled, a note is a
    // relative the vault computed and you never linked.
    missing: sug.filter((x) => x.kind === "ghost"),
    similar: sug.filter((x) => x.kind === "note"),
    // Where the note SITS, as opposed to what it says or who it links.
    // Passed through whole and read by structureRows(), which is where each
    // number is turned into the sentence it means: the projection stays a
    // projection, and the thresholds live in one place a test can reach.
    structure: c.structure || null,
    // Why a section came back thin, from the machine that came back thin.
    // Without it "Similar (0)" reads as "this note has no relatives" when what
    // it means is "nobody has run /embed".
    hint: c.hint || "",
  };
}

// Six numbers, and what each one is allowed to CLAIM. The gates decided this
// wording, not taste (docs/adr/0027):
//
//   coreness / cut vertex  PASSED, so they are allowed an imperative reading:
//     a held refine and a warning swatch.
//   surprise               PASSED as part of the same variable; stated as a
//     comparison ("more crossings than links"), never as a verdict.
//   dissonance             FAILED its judge gate (notes it called misfiled were
//     filed right 13/14 and 16/19), so it says what it MEASURED - the share of
//     links leaving the note's semantic zone - and never that the note is in
//     the wrong place. No warning swatch, no action.
//
// Returns [{text, warn, why}]; an empty array means nothing worth a row, which
// is why an ordinary note in the middle of its area prints no section at all.
function structureRows(st) {
  if (!st) return [];
  const out = [];
  if (st.articulation) {
    out.push({
      text: "load-bearing" + (st.strands
        ? ": removing it strands " + st.strands + (st.strands === 1 ? " note" : " notes")
        : ""),
      warn: true,
      why: "a cut vertex - every path between those notes and the rest goes through this one",
    });
  }
  if (st.coreness >= 2) {
    out.push({
      text: "core " + st.coreness,
      why: "sits inside a group where every note has at least " + st.coreness
         + " links to the same group",
    });
  }
  // Both tails are worth a row and they read as opposites, so the sign is
  // spelled out in words rather than left as a signed percentage.
  if (st.surprise >= 20) {
    out.push({
      text: "more crossings than links (+" + st.surprise + ")",
      why: "paths through the vault cross it far more than its own link count "
         + "would predict: a bridge worth reinforcing",
    });
  } else if (st.surprise <= -20) {
    out.push({
      text: "more links than crossings (" + st.surprise + ")",
      why: "well linked, but few paths route through it: a member, not a junction",
    });
  }
  if (st.dissonance != null && st.dissonance >= 50) {
    out.push({
      text: st.dissonance + "% of its links leave its zone",
      why: "the two partitions disagree here: what it links and what it reads "
         + "like are not the same neighbourhood. A reading, not a fault.",
    });
  }
  return out;
}

// --- projection:end ----------------------------------------------------------

// --- the surface -------------------------------------------------------------
(function () {
  // Three elements of the right sidebar's `work` mode: its pane, and the two
  // marks that ride in the sidebar's one header. Looked up from the document and
  // not from a wrapper of their own — this file used to own a whole #work panel
  // with a header, a name and a close button, and all three belonged to being a
  // second surface at an edge that already had one. app.js owns whether the
  // sidebar is open and which mode it shows; this file only fills the pane.
  const body = document.getElementById("work-body");
  if (!body) return;
  const head = document.getElementById("work-state");
  // What this mode is ABOUT, beside the segment that says what it is. Empty on
  // the run itself: the segment is already named after it.
  const scope = document.getElementById("work-scope");

  let beats = [], sid = null, curSid = null, run = null, tick = null;
  // Which tab is up: on metrics this column is the Report, on explore the Node.
  // Read from the DOM and not left at "chat" until the first silica:view: app.js
  // selects the boot tab while it is parsing, which is BEFORE this file loads,
  // so a deep link to #metrics or #explore used to land on a column narrating
  // the run on a view that has something else to say - and stayed that way
  // until the reader clicked a tab.
  const bootTab = document.querySelector(".tab.active");
  let view = bootTab ? bootTab.dataset.tab : "chat";
  let report = null;   // the last /metrics payload, which rides app.js's fetch
  let node = null, nodeCtx = null;  // on explore: the node pointed at, and its context

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  function sectionHead(label, count) {
    const h = el("div", "wk-sec");
    h.appendChild(el("span", "wk-sec-t", label));
    if (count != null) h.appendChild(el("span", "wk-sec-n", String(count)));
    return h;
  }

  function beatRow(r) {
    const row = el("div", "wk-beat" + (r.running ? " running" : "") + (r.failed ? " failed" : ""));
    row.appendChild(el("span", "wk-mark wk-" + r.kind));
    const lab = el("div", "wk-lab");
    if (r.kind === "think") {
      lab.appendChild(el("div", "wk-thought", r.text || "thinking"));
    } else {
      const v = el("div", "wk-v");
      v.appendChild(el("span", "wk-verb", r.verb));
      if (r.target) v.appendChild(el("span", "wk-target", r.target));
      lab.appendChild(v);
    }
    if (r.sub) lab.appendChild(el("div", "wk-sub", r.sub));
    row.appendChild(lab);
    row.appendChild(el("div", "wk-dur", r.dur));
    return row;
  }

  function writeCardEl(w) {
    const card = el("div", "wk-write" + (w.failed ? " failed" : ""));
    const h = el("div", "wk-write-h");
    h.appendChild(el("span", "wk-op", w.failed ? "rolled back" : w.op));
    if (w.failed) h.appendChild(el("span", "wk-note", "vault unchanged"));
    card.appendChild(h);
    card.appendChild(el("div", "wk-path", w.target));
    if (!w.failed && w.target) {
      const acts = el("div", "wk-acts");
      const read = el("button", "wk-mini go", "read");
      read.type = "button";
      read.addEventListener("click", () => { if (window.openNote) window.openNote(w.target); });
      const undo = el("button", "wk-mini", "revert");
      undo.type = "button";
      // `/undo <path>` is the same command the transcript's write card sends,
      // and it takes the rest of the line, so a note name with spaces survives.
      undo.addEventListener("click", () => { if (window.send) window.send("/undo " + w.target); });
      acts.appendChild(read); acts.appendChild(undo);
      card.appendChild(acts);
    }
    return card;
  }

  function sourceRow(s) {
    const row = el("button", "wk-src");
    row.type = "button";
    row.appendChild(el("span", "wk-src-sw"));
    const t = el("div", "wk-src-t");
    t.appendChild(el("div", "wk-src-n", s.name));
    t.appendChild(el("div", "wk-src-v", s.verb));
    row.appendChild(t);
    row.addEventListener("click", () => { if (window.openNote) window.openNote(s.name); });
    return row;
  }

  // The top strip says the same word as this header and takes it from the same
  // projection: two paints, one computation, so the strip cannot claim idle
  // while the panel beside it is filling. It also keeps meaning something when
  // the panel is toggled off, which is when it is the only thing saying it.
  const topRun = document.getElementById("top-run");
  const topElapsed = document.getElementById("top-elapsed");

  function paintTopRun(run, state) {
    if (!topElapsed) return;
    const live = !!run.rows.length;
    topElapsed.textContent = live ? run.elapsed : "";
    topElapsed.title = live ? state : "";
    // Hidden and not merely empty: .tz reserves 2.2rem of padding whether or
    // not it has anything in it, and an empty compartment that never fills is
    // a gap the eye keeps returning to.
    if (topRun) topRun.hidden = !live;
  }

  // The spine is the run's own progress bar, so it has to end at the beat that
  // is RUNNING rather than at the fraction the mock happened to draw. Measured
  // and not counted: a thought row is two lines and a tool row is one, so "the
  // fourth of six beats" is nowhere near two thirds of the way down the column.
  // The insets come from the rule's own computed style because the gradient is
  // laid out in the pseudo-element's box, not the list's — reading them keeps
  // the cyan on the beat when those paddings change in CSS alone.
  function paintSpine(tl) {
    const live = tl.querySelectorAll(".wk-beat.running");
    const last = live[live.length - 1];
    if (!last) { tl.style.removeProperty("--spine"); return; }
    const box = tl.getBoundingClientRect();
    const cs = getComputedStyle(tl, "::before");
    const top = box.top + (parseFloat(cs.top) || 0);
    const span = box.height - (parseFloat(cs.top) || 0) - (parseFloat(cs.bottom) || 0);
    // A hidden panel measures zero, and a percentage of zero is NaN in a
    // gradient, which paints the whole rule accent.
    if (!(span > 0)) { tl.style.removeProperty("--spine"); return; }
    const b = last.getBoundingClientRect();
    const pct = ((b.top + b.height / 2 - top) / span) * 100;
    tl.style.setProperty("--spine", Math.max(0, Math.min(100, pct)).toFixed(1) + "%");
  }

  // --- the Report panel ------------------------------------------------------
  // Same column, different question. On metrics the run is not what you are
  // reading, the report is, and its two things worth a column are what moved
  // since the last one and what to do about it.
  const shortDate = (iso) => {
    const d = new Date(iso);
    return isNaN(d) ? "" : d.toLocaleDateString(undefined, { day: "numeric", month: "short" }).toLowerCase();
  };
  // h23 rather than the locale's own cycle: the metrics header a few pixels to
  // the left prints "2026-08-22 15:05" from the server, and one clock in this
  // app cannot say 15:05 while the one beside it says 05:05 PM.
  const clockOf = (iso) => {
    const d = new Date(iso);
    return isNaN(d) ? "" : d.toLocaleTimeString(undefined,
      { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
  };

  // The two bulk turns app.js already offers under the evidence panes, taken
  // from there rather than restated: the Report is a second door onto the same
  // work, and two copies of a write prompt are two turns that can drift apart.
  function reportActions(d) {
    const T = d.totals || {};
    const acts = [];
    const topN = Math.min(20, T.dangling_links || 0);
    if (topN > 0 && window.bulkWritePrompt) {
      acts.push({ label: `write the ${topN} most-referenced targets`,
                  note: `${fmtTok(d.dangling_top_refs || 0)} links`,
                  sig: "dangling", prompt: window.bulkWritePrompt(d, topN) });
    }
    const orphans = T.orphans || 0;
    if (orphans && window.bulkAutolinkPrompt) {
      acts.push({ label: "autolink the orphans into their areas",
                  note: `${fmtTok(orphans)} notes`,
                  sig: "orphans", prompt: window.bulkAutolinkPrompt(orphans) });
    }
    return acts;
  }

  function suggestedCard(a) {
    const card = el("div", "wk-write suggested");
    const h = el("div", "wk-write-h");
    h.appendChild(el("span", "wk-op", "suggested"));
    h.appendChild(el("span", "wk-num", a.note));
    card.appendChild(h);
    card.appendChild(el("div", "wk-path", a.label));
    const acts = el("div", "wk-acts");
    // Same contract as every action in the metrics view: it drafts the turn and
    // leaves it in the composer. A panel that reports on the vault does not get
    // to change it.
    const go = el("button", "wk-mini go", "run it");
    go.type = "button";
    go.addEventListener("click", () => { if (window.prefillChat) window.prefillChat(a.prompt); });
    // "Preview" means the rows it would touch, and those rows are already a
    // pane in the view beside this one, so it points there instead of growing
    // a second list of the same targets.
    const prev = el("button", "wk-mini", "preview ops");
    prev.type = "button";
    prev.addEventListener("click", () => { if (window.selectMetricSignal) window.selectMetricSignal(a.sig); });
    acts.appendChild(go); acts.appendChild(prev);
    card.appendChild(acts);
    return card;
  }

  // What moved, read positionally instead of parsed row by row.
  //
  // NOT a slope chart, though that is the form this answers to. A slope chart
  // puts every series on one axis, and these counts do not share one: notes runs
  // to the hundreds and gaps to single digits, so a shared axis draws eight of
  // the nine as a flat line at the bottom and the ninth as the whole chart. Given
  // each its own axis instead, the slope ANGLE stops meaning anything across
  // lines, which is the only thing a slope chart is read for.
  //
  // Proportional change is the currency that does compare them: gaps 12 -> 9 is a
  // quarter gone, notes 686 -> 690 is a rounding error, and a reader wants to see
  // that difference without dividing. One axis, zero in the middle, and the row's
  // own absolute figures stay in the beat below.
  function movementStrip(rows) {
    const moved = rows.filter((r) => r.delta != null && r.delta !== 0);
    if (moved.length < 2) return null;   // one bar is a number with decoration
    const share = (r) => {
      // From zero there is no ratio to take. Full width in the direction of the
      // move, and the beat below is what says the count was zero before.
      if (!r.was) return r.delta > 0 ? 1 : -1;
      return Math.max(-1, Math.min(1, r.delta / r.was));
    };
    // Scaled to the largest move here, not to +-100%. Between two consecutive
    // reports these are single-digit percentages, so against a full-scale axis
    // every bar renders under two pixels and the strip draws nothing. The bar
    // ranks the movers against each other and the printed figure carries the
    // magnitude -- the same division of labour barChart already uses, where the
    // bar is scaled to the row maximum and the value is written at the tip.
    const top = Math.max(...moved.map((r) => Math.abs(share(r))), 0.001);
    const strip = el("div", "wk-mv");
    for (const r of moved) {
      const p = share(r);
      const row = el("div", "wk-mv-r");
      row.title = `${r.label}: ${r.sub}`;
      row.appendChild(el("div", "wk-mv-l", r.label));
      const track = el("div", "wk-mv-t");
      const bar = el("i", "wk-mv-b" + (r.good === true ? " good" : r.good === false ? " bad" : ""));
      // Anchored at the centre and grown outward, so the two directions are the
      // two sides of one line rather than two bar charts side by side. Floored
      // at 3% of the half-track: a move small enough to render as a hairline is
      // still a move, and the row is here precisely because something changed.
      bar.style.width = Math.max(3, (Math.abs(p) / top) * 50) + "%";
      bar.style[p < 0 ? "right" : "left"] = "50%";
      track.appendChild(bar);
      row.appendChild(track);
      row.appendChild(el("div", "wk-mv-n",
        r.was ? (p > 0 ? "+" : "−") + Math.round(Math.abs(r.delta / r.was) * 100) + "%" : "new"));
      strip.appendChild(row);
    }
    return strip;
  }

  function renderReport() {
    const r = projectReport(report);
    scope.textContent = "report";
    head.textContent = clockOf(r.at);
    head.className = "wk-num";
    body.textContent = "";

    const rh = el("div", "wk-run");
    rh.appendChild(el("div", "wk-run-t", r.title));
    const meta = el("div", "wk-run-m");
    for (const m of r.meta) meta.appendChild(el("span", "wk-num", m));
    rh.appendChild(meta);
    body.appendChild(rh);

    if (r.rows.length) {
      body.appendChild(sectionHead("Since the last report",
        r.since ? shortDate(r.since) : "first"));
      const strip = movementStrip(r.rows);
      if (strip) body.appendChild(strip);
      const tl = el("div", "wk-tl");
      for (const row of r.rows) {
        const beat = el("div", "wk-beat");
        beat.appendChild(el("span", "wk-mark wk-tool"));
        const lab = el("div", "wk-lab");
        const v = el("div", "wk-v");
        v.appendChild(el("span", "wk-verb", row.label));
        v.appendChild(el("span", "wk-target", row.value));
        lab.appendChild(v);
        lab.appendChild(el("div", "wk-sub", row.sub));
        beat.appendChild(lab);
        const d = el("div", "wk-dur" + (row.good === true ? " good" : row.good === false ? " bad" : ""),
                     row.delta == null ? "" : (row.delta > 0 ? "+" : row.delta < 0 ? "−" : "") + fmtTok(Math.abs(row.delta)));
        beat.appendChild(d);
        tl.appendChild(beat);
      }
      body.appendChild(tl);
    }

    const acts = reportActions(report);
    if (acts.length) {
      body.appendChild(sectionHead("Act on this report"));
      for (const a of acts) body.appendChild(suggestedCard(a));
    }
  }

  // --- the Node panel --------------------------------------------------------
  // Same column, the third question. The rows are the source rows the Sources
  // section already uses: a name, what it is, and one thing you can do about it
  // is exactly their shape, and a second row layout for the same shape is a
  // second thing to keep in step.
  // Where a note sits, which is the one thing the name does not say and the
  // reason two notes with the same title are not the same row.
  const folderOf = (p) => {
    const i = String(p || "").lastIndexOf("/");
    return i > 0 ? p.slice(0, i) : "";
  };

  function nodeRow(r, opt) {
    const o = opt || {};
    const row = el("div", "wk-src" + (o.action ? " has-act" : ""));
    row.appendChild(el("span", "wk-src-sw" + (o.warn ? " warn" : "")));
    const t = el("button", "wk-srct");
    t.type = "button";
    t.appendChild(el("div", "wk-src-n", r.name || r.path || ""));
    const v = el("div", "wk-src-v");
    if (o.why) v.appendChild(el("span", null, o.why));
    // A meter only where a real number exists. A wikilink has no strength, so
    // those rows get no bar at all rather than a bar at some default width,
    // which would rank rows nobody ranked.
    if (o.score > 0) {
      const m = el("span", "wk-mtr");
      const fill = el("i");
      fill.style.width = Math.max(6, Math.min(100, Math.round(o.score * 100))) + "%";
      m.appendChild(fill);
      v.appendChild(m);
    }
    if (v.childElementCount) t.appendChild(v);
    // Pointing at a row walks the column to that node, which is the same verb
    // the graph click has. A row with no note behind it (a ghost) has nowhere
    // to walk to, and its offer is the button on the right instead.
    if (r.path && window.showNode) {
      t.addEventListener("click", () => window.showNode({ path: r.path }));
    } else {
      t.disabled = true;
      t.style.cursor = "default";
    }
    row.appendChild(t);
    if (o.action) {
      const b = el("button", "wk-mini" + (o.go ? " go" : ""), o.action);
      b.type = "button";
      b.addEventListener("click", () => { if (window.prefillChat) window.prefillChat(o.prompt); });
      row.appendChild(b);
    }
    return row;
  }

  // Five rows say what kind of neighbourhood a note sits in. A hub with sixty
  // backlinks says it in sixty and buries every section under it, which is what
  // this column would have inherited the moment it took the drawer's lists
  // over: the floor lived in cxList, and cxList is what left. <details> owns
  // the tail because the browser already owns that toggle - no open/closed
  // state to keep in JS, and it survives a re-render by not existing across one.
  const NODE_VISIBLE = 5;

  function nodeSection(label, rows, opt) {
    if (!rows.length) return;
    body.appendChild(sectionHead(label, rows.length));
    const mk = (r) => nodeRow(r, typeof opt === "function" ? opt(r) : opt);
    for (const r of rows.slice(0, NODE_VISIBLE)) body.appendChild(mk(r));
    const rest = rows.slice(NODE_VISIBLE);
    if (!rest.length) return;
    const more = el("details", "wk-more");
    more.appendChild(el("summary", null, rest.length + " more"));
    for (const r of rest) more.appendChild(mk(r));
    body.appendChild(more);
  }

  function renderNode() {
    const d = projectNode(node, nodeCtx);
    scope.textContent = "node";
    head.textContent = d.ghost ? "unresolved" : "";
    head.className = "wk-num";
    body.textContent = "";

    const rh = el("div", "wk-run");
    rh.appendChild(el("div", "wk-run-t", d.title || "node"));
    if (d.path) {
      const m = el("div", "wk-run-m");
      m.appendChild(el("span", "wk-num", d.path));
      rh.appendChild(m);
    }
    if (d.count || d.meta.length) {
      const m = el("div", "wk-run-m");
      if (d.count) m.appendChild(el("span", "wk-num strong", d.count));
      for (const bit of d.meta) m.appendChild(el("span", "wk-num", bit));
      rh.appendChild(m);
    }
    const acts = el("div", "wk-acts");
    if (d.path) {
      // The reader is one click away and not gone: pointing at a node fills
      // this column, naming one still opens the drawer, and this is the button
      // that turns the first into the second.
      const read = el("button", "wk-mini go", "read");
      read.type = "button";
      read.addEventListener("click", () => { if (window.openNote) window.openNote(d.path); });
      acts.appendChild(read);
    }
    const ask = el("button", "wk-mini", "open in chat");
    ask.type = "button";
    ask.addEventListener("click", () => {
      if (!window.prefillChat) return;
      window.prefillChat(d.ghost && window.ghostWritePrompt
        ? window.ghostWritePrompt(d.title, d.from)
        : 'Tell me about "' + d.title + '", grounded in what the vault already says.');
    });
    acts.appendChild(ask);
    rh.appendChild(acts);
    body.appendChild(rh);

    if (d.error) {
      const e = el("div", "wk-empty");
      e.appendChild(el("div", "wk-empty-d", d.error));
      body.appendChild(e);
      return;
    }
    if (d.loading) {
      const e = el("div", "wk-empty");
      e.appendChild(el("div", "wk-empty-d", "reading the vault…"));
      body.appendChild(e);
      return;
    }

    // Above the graph half on purpose: what a note says is the cheaper read,
    // and it is the one that settles whether the neighbourhood below is worth
    // walking.
    if (d.says.length) {
      body.appendChild(sectionHead("Says", d.says.length));
      for (const sn of d.says) {
        const row = el("div", "wk-snip");
        if (sn.heading) row.appendChild(el("div", "wk-snip-h", sn.heading));
        row.appendChild(el("div", "wk-snip-t", sn.text || ""));
        body.appendChild(row);
      }
    }

    if (d.concepts.length) {
      body.appendChild(sectionHead("Concepts", d.concepts.length));
      const pills = el("div", "wk-pills");
      for (const c of d.concepts) {
        const b = el("button", "wk-pill", c.concept);
        b.type = "button";
        b.title = "weight " + (c.weight || 1) + ": light its notes in the graph";
        b.addEventListener("click", () => {
          if (window.lightConcept) window.lightConcept(c.concept, b);
        });
        pills.appendChild(b);
      }
      body.appendChild(pills);
    }

    // Where it sits, between what it says and who it links: the three are the
    // same question at three ranges, and this is the middle one.
    const srows = structureRows(d.structure);
    if (srows.length) {
      body.appendChild(sectionHead("Structure", null));
      for (const r of srows) {
        const row = el("div", "wk-str" + (r.warn ? " warn" : ""));
        row.appendChild(el("span", "wk-str-sw"));
        const t = el("div", "wk-str-t");
        t.appendChild(el("div", "wk-str-n", r.text));
        t.appendChild(el("div", "wk-str-w", r.why));
        row.appendChild(t);
        body.appendChild(row);
      }
    }
    // The one direction the vault knows. Above the undirected lists because a
    // reading order answers "what now" and they answer "what else".
    const st = d.structure || {};
    nodeSection("Read first", st.prerequisites || [], (r) => ({ why: folderOf(r.path) }));
    nodeSection("Unlocks", st.unlocks || [], (r) => ({ why: folderOf(r.path) }));

    // Same grammar as "Similar, not linked" on purpose: both name a relation
    // the vault knows about and the prose does not carry.
    nodeSection("Declared, not linked", d.related, (r) => ({ why: folderOf(r.path) }));
    nodeSection("Links out", d.out, (r) => ({ why: folderOf(r.path) }));
    nodeSection("Linked from", d.from, (r) => ({ why: folderOf(r.path) }));
    nodeSection("Missing", d.missing, (r) => ({
      warn: true, why: r.why, action: "write it", go: true,
      prompt: window.writeGhostPrompt ? window.writeGhostPrompt(r.name, d.title) : "",
    }));
    // The only section whose `why` is not the folder: it names the machine that
    // found the row ("2 hops away"). So the folder is appended rather than
    // swapped in - drop it and two notes both called Cell are a coin flip here
    // and nowhere else on the panel.
    nodeSection("Similar, not linked", d.similar, (r) => ({
      why: [r.why, folderOf(r.path)].filter(Boolean).join(" · "),
      score: r.score || 0, action: "link",
      prompt: window.linkNotesPrompt ? window.linkNotesPrompt(d.title, r.name) : "",
    }));

    // Under the sections it explains, never over them: it is the reason a count
    // is low, and a reason printed before the count warns about something the
    // reader has not seen yet.
    if (d.hint) body.appendChild(el("div", "wk-hint", d.hint));

    if (!d.says.length && !d.concepts.length && !d.related.length && !d.out.length
        && !d.from.length && !d.missing.length && !d.similar.length && !srows.length
        && !(st.prerequisites || []).length && !(st.unlocks || []).length) {
      const e = el("div", "wk-empty");
      e.appendChild(el("div", "wk-empty-d",
        "nothing indexed for this note yet. Run /report or /embed to build the graph"));
      body.appendChild(e);
    }
  }

  function render() {
    run = projectRun(beats);
    const state = run.running ? "live" : (run.rows.length ? "done" : "idle");
    // The top strip says the run's state on every view, this column only on
    // some, so the strip is painted before the branch and not inside it.
    paintTopRun(run, state);
    // The node you pointed at outranks both the run and the report, on either
    // view that is made of rows pointing at notes. It used to be explore alone,
    // because everywhere else the note drawer answered instead - with a second
    // rendering of this same /context payload. There is one rendering now, so
    // the gate had to widen with it or a metrics row would route here and land
    // on a column still drawing the report.
    //
    // Chat and calendar stay out: nothing on either selects a node, so a node
    // showing beside a transcript is a selection made on another view that you
    // can neither see the source of nor drop from here. With nothing selected
    // this column is the Work panel it has always been, which is what a graph
    // you are still looking around in has to say.
    if ((view === "graph" || view === "metrics") && node) { renderNode(); return; }
    if (view === "metrics" && report) { renderReport(); return; }
    scope.textContent = "";
    head.textContent = state;
    head.className = "wk-state" + (run.running ? " live" : "");
    body.textContent = "";

    if (!run.rows.length) {
      const e = el("div", "wk-empty");
      e.appendChild(el("div", "wk-empty-t", "Nothing running"));
      e.appendChild(el("div", "wk-empty-d",
        "Every step of the next turn lands here: what the model thought, which "
        + "tools it ran, what it read and what it changed in the vault."));
      body.appendChild(e);
      return;
    }

    const rh = el("div", "wk-run");
    rh.appendChild(el("div", "wk-run-t", run.title || "run"));
    const meta = el("div", "wk-run-m");
    if (run.elapsed) meta.appendChild(el("span", "wk-num", run.elapsed));
    if (run.shape) meta.appendChild(el("span", "wk-num", run.shape));
    if (run.model) meta.appendChild(el("span", "wk-num", run.model.split("/").pop()));
    rh.appendChild(meta);
    body.appendChild(rh);

    const tl = el("div", "wk-tl" + (run.running ? " live" : ""));
    for (const r of run.rows) tl.appendChild(beatRow(r));
    body.appendChild(tl);
    paintSpine(tl);

    if (run.writes.length) {
      body.appendChild(sectionHead("Writes", run.writes.length));
      for (const w of run.writes) body.appendChild(writeCardEl(w));
    }
    if (run.sources.length) {
      body.appendChild(sectionHead("Sources", run.sources.length));
      for (const s of run.sources) body.appendChild(sourceRow(s));
    }
  }

  // A running turn has to tick, or the elapsed reads as a clock that stopped
  // between beats — and a distill chunk is nine seconds of no beats at all.
  function retick() {
    if (tick) { clearInterval(tick); tick = null; }
    if (run && run.running) tick = setInterval(render, 500);
  }

  function ingest(payload) {
    const incoming = payload.beats || [];
    if (!incoming.length) return;
    const bsid = payload.sid || incoming[0].sid;
    // /reset opens a new narration session on the same stream. Adopt it and drop
    // the old run rather than interleaving two conversations in one panel.
    if (bsid && curSid && bsid !== curSid) beats = [];
    if (bsid) curSid = bsid;
    for (const b of incoming) {
      if (!beats.length || b.seq > beats[beats.length - 1].seq) beats.push(b);
    }
    render();
    retick();
  }

  // A session switch (History, or the wordmark's new chat) replays nothing onto
  // the bus, so the live stream cannot see it. /narration is the snapshot of
  // whichever session is current now; the stream keeps appending to it.
  async function refresh() {
    beats = []; curSid = null; render(); retick();
    try {
      const d = await (await fetch("/narration")).json();
      ingest(d);
    } catch { /* the next beat fills it; a snapshot that fails is not an error to report */ }
  }
  document.addEventListener("silica:session", refresh);
  document.addEventListener("silica:view", (e) => { view = e.detail; render(); });
  // The graph hands over what it drew (the node) and app.js the round trip that
  // followed (its context). Both arrive here as one event, twice: once bare so
  // the head paints on the click, once filled.
  document.addEventListener("silica:node", (e) => {
    const d = e.detail;
    node = d ? d.node : null;
    nodeCtx = d ? d.context : null;
    // A pane the reader is not on is not a pane to write into unseen: pointing
    // at a node is a request to see what it is, so it raises the sidebar on this
    // mode. The BARE announce only — that is the one carrying the click. The
    // filled one arrives a round trip later, by which time the reader may have
    // opened a note, and raising the pane again would take back a surface they
    // asked for after they asked for it. (It could not before: this pane was a
    // separate panel that `body.note-open #work` simply hid, so a late announce
    // wrote into something already off-screen.)
    if (node && !nodeCtx && window.openWorkDrawer) window.openWorkDrawer();
    render();
  });
  // One fetch, two surfaces: app.js loads /metrics for the view and hands the
  // same payload here rather than this file measuring the vault a second time.
  document.addEventListener("silica:report", (e) => { report = e.detail; render(); });

  let es = null;
  function connect() {
    if (es) es.close();
    es = new EventSource("/narration/sse");
    // The SSE id: field is the seq, so a dropped connection resumes from
    // Last-Event-ID with no cursor of our own.
    es.addEventListener("beats", (e) => {
      try { ingest(JSON.parse(e.data)); } catch { /* one malformed frame is not the stream */ }
    });
    es.onerror = () => { /* EventSource retries by itself; a closed panel is not an error to report */ };
  }

  render();
  connect();
})();
