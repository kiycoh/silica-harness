// Vanilla client: POST /chat returns text/event-stream, read incrementally via
// the body's ReadableStream (not EventSource — that only does GET).
const $ = (s) => document.querySelector(s);
const log = $("#log");
const input = $("#input");
const stopBtn = $("#stop");

let streaming = false;
let activeTab = "chat";

// --- notifications + screen-reader status ------------------------------------
// A hairline toast strip fills the silent catch(){} gaps: a failed background
// fetch now says so instead of leaving a stale "—". Two levels only (info =
// accent, error = gold/caution) — the palette reserves no third UI signal.
// Every notify() also lands in the polite SR region, so the streaming
// transcript itself needn't be a chatty live region.
const srStatus = $("#sr-status");
const toasts = $("#toasts");
function announce(msg) { if (srStatus) srStatus.textContent = msg; }
// Cap the visible stack. Two diagnostics fire on every load and an error turn
// stacked five, ~300px of grey boxes sitting on top of #send — so whatever you
// had just done, the screen closed on a debug message covering the primary
// control. Older ones roll up behind a "+N" the user can expand.
const TOAST_MAX = 2;

function rollUpToasts() {
  const all = [...toasts.querySelectorAll(".toast")];
  const overflow = Math.max(0, all.length - TOAST_MAX);
  all.forEach((t, i) => { t.hidden = i < overflow; });
  let more = toasts.querySelector(".toast-more");
  if (!overflow) { if (more) more.remove(); return; }
  if (!more) {
    more = document.createElement("button");
    more.type = "button";
    more.className = "toast-more";
    // recompute on click: the stack keeps changing under this handler
    more.addEventListener("click", () => {
      toasts.querySelectorAll(".toast").forEach((t) => { t.hidden = false; });
      more.remove();
    });
  }
  toasts.prepend(more);
  more.textContent = `+${overflow} more`;
}

function notify(msg, level = "error") {
  announce(msg);
  if ([...toasts.querySelectorAll(".toast")].some((t) => t.textContent === msg)) return; // dedupe visible
  const t = document.createElement("div");
  t.className = "toast " + level;
  t.textContent = msg;
  t.title = msg; // CSS clamps the box to 3 lines; the full text stays reachable
  const kill = () => { t.remove(); rollUpToasts(); };
  t.addEventListener("click", kill);
  toasts.appendChild(t);
  setTimeout(kill, level === "error" ? 6000 : 3000);
  rollUpToasts();
}

// Name what a tool acted on. The verb alone ("write note") never told the user
// which file the agent touched in their own vault. One formatter, so a replayed
// transcript reads exactly like the stream that produced it.
const toolLabel = (t) => (t.target ? `${t.name} "${t.target}"` : t.name);

// --- injector pipeline block -------------------------------------------------
// Every other tool is one flat line; a nucleate run is minutes of work with a
// 15-phase cycle inside it, and a lone spinner reading "injector" was the whole
// of what the GUI said about it. The TUI has always shown the phases (it holds
// the only subscriber to the FSM's phase stream) — this is the same information,
// laid out for a surface that has vertical space and no 12fps redraw budget.

const PHASE_MARK = { done: "✓", running: "◉", failed: "✗", pending: "·" };
const SUMMARY_MARK = { ok: "✓", partial: "◐", empty: "⊘", failed: "✗" };

const fmtDur = (s) =>
  s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m${String(Math.floor(s % 60)).padStart(2, "0")}s`;

// One line of counts for a finished run. Shared by the live tool_done event and
// by transcript replay, so reopening a chat restates what it said while running.
function injectorSummaryLine(label, s) {
  const bits = [];
  if (s.files) bits.push(s.files === 1 ? "1 file" : `${s.files} files`);
  if (s.notes) bits.push(`${s.notes} notes`);
  if (s.links) bits.push(`${s.links} links`);
  if (s.reason) bits.push(s.reason);
  if (s.kind === "partial" && s.failed_chunks.length) {
    bits.push(`${s.failed_chunks.length} of ${s.committed + s.failed_chunks.length} chunks failed`);
  }
  const mark = SUMMARY_MARK[s.kind] || "✓";
  return `${mark} injector · ${label}${bits.length ? "   " + bits.join(" · ") : ""}`;
}

function makePipelineBlock(label, tracks) {
  const el = document.createElement("div");
  el.className = "tool tool-pipeline running";
  el.innerHTML =
    `<div class="pipe-head"><span class="pipe-title"></span><span class="pipe-pos"></span></div>` +
    `<div class="pipe-track pipe-file"></div><div class="pipe-track pipe-chunk"></div>`;
  const head = el.querySelector(".pipe-title");
  const pos = el.querySelector(".pipe-pos");
  head.textContent = `» injector · ${label}`;

  // Both tracks are drawn once, greyed out, so the pipeline reads as a known
  // sequence with a position in it rather than a list that grows as it goes.
  const rows = {};
  for (const [scope, names] of Object.entries(tracks)) {
    const box = el.querySelector(scope === "file" ? ".pipe-file" : ".pipe-chunk");
    for (const name of names) {
      const r = document.createElement("div");
      r.className = "pipe-phase pending";
      r.innerHTML = `<span class="pipe-mark">·</span><span class="pipe-name"></span><span class="pipe-time"></span>`;
      r.querySelector(".pipe-name").textContent = name;
      box.appendChild(r);
      rows[`${scope}:${name}`] = r;
    }
  }

  let chunkKey = null;      // resets the chunk track when the run moves on
  let running = null;       // { row, at } — the phase whose timer is ticking
  // The TUI gets a live timer free from Rich re-rendering at 12fps; here the
  // running row is ticked locally from when its event arrived, so the server
  // sends elapsed only once, on done.
  const timer = setInterval(() => {
    if (running) running.row.querySelector(".pipe-time").textContent = fmtDur((Date.now() - running.at) / 1000);
  }, 100);

  function setRow(row, state, secs) {
    const undo = row.dataset.rollback === "1";
    // A completed rollback is not a step that went well: it is the undo of one
    // that did not. Ticking it in the same grey as `write` read as success.
    row.className = `pipe-phase ${undo && state !== "pending" ? "failed" : state}`;
    row.querySelector(".pipe-mark").textContent =
      undo && state !== "pending" ? "↳" : (PHASE_MARK[state] || "·");
    if (secs != null) row.querySelector(".pipe-time").textContent = fmtDur(secs);
  }

  return {
    el,
    applyPhase(ev) {
      // Position rides on every event, so a dropped one cannot leave the header
      // naming the wrong file or chunk — the next event restates all of it.
      const bits = [];
      if (ev.file_total > 1) bits.push(`file ${ev.file_idx + 1}/${ev.file_total}`);
      if (ev.chunk_total > 0) bits.push(`chunk ${ev.chunk_idx + 1}/${ev.chunk_total}`);
      pos.textContent = bits.join(" · ");
      if (ev.source_file) head.textContent = `» injector · ${ev.source_file}`;

      const key = `${ev.file_idx}:${ev.chunk_idx}`;
      if (ev.scope === "chunk" && key !== chunkKey) {
        chunkKey = key;
        for (const [k, r] of Object.entries(rows)) {
          if (k.startsWith("chunk:")) { setRow(r, "pending"); r.querySelector(".pipe-time").textContent = ""; }
        }
      }

      // rollback is not in either track: it is an exception branch, and drawing
      // it as a pending step made every healthy run advertise a rollback that
      // was never coming. It gets appended only when it actually fires.
      // ev.phase is the display label (the server maps it), so this is an exact
      // match — an id-to-label rule here would have to cover hub_update/hub-update,
      // and guessing left that phase permanently grey.
      let row = rows[`${ev.scope}:${ev.phase}`];
      if (!row && ev.phase === "rollback") {
        row = rows["exception:rollback"];
        if (!row) {
          row = document.createElement("div");
          row.className = "pipe-phase";
          row.dataset.rollback = "1";
          row.innerHTML = `<span class="pipe-mark">↳</span><span class="pipe-name">rollback</span><span class="pipe-time"></span>`;
          el.querySelector(".pipe-chunk").appendChild(row);
          rows["exception:rollback"] = row;
        }
      }
      if (!row) return;
      if (ev.status === "running") {
        setRow(row, "running");
        running = { row, at: Date.now() };
      } else {
        const secs = ev.elapsed != null ? ev.elapsed
          : (running && running.row === row ? (Date.now() - running.at) / 1000 : null);
        setRow(row, ev.status === "failed" ? "failed" : "done", secs);
        if (running && running.row === row) running = null;
      }
    },
    finish(summary) {
      clearInterval(timer);
      running = null;
      const s = summary || { kind: "failed", reason: "", notes: 0, links: 0, files: 0, committed: 0, failed_chunks: [] };
      const name = (head.textContent || "").replace(/^» injector · /, "");
      el.classList.remove("running");
      el.classList.add(s.kind);
      head.textContent = injectorSummaryLine(name, s);
      // A good run collapses to its one line; a bad one keeps the track open on
      // the phase that broke, which is the only time the detail is worth rows.
      if (s.kind === "ok" || s.kind === "empty") {
        el.classList.add("collapsed");
        pos.textContent = "";
      } else if (s.failed_chunks && s.failed_chunks.length) {
        const d = document.createElement("div");
        d.className = "pipe-failed";
        d.textContent = s.failed_chunks.map((f) => `✗ ${f.chunk}${f.phase ? " " + f.phase : ""}`).join(" · ");
        el.appendChild(d);
      }
    },
  };
}

function bubble(role) {
  const el = document.createElement("div");
  el.className = "msg " + (role === "user" ? "user" : "silica");
  el.innerHTML = `<div class="role">${role === "user" ? "you" : "silica"}</div><div class="body"></div>`;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el.querySelector(".body");
}

// One vault change as an object that stays in the transcript: what happened, to
// which note, and the way back out of it. `effect` is written | moved | deleted |
// failed. The card owns its own revert, so you undo the write you are looking at
// rather than the whole turn.
const WRITE_COPY = {
  written: { label: "written", act: "revert", hint: "restore this note to its state before the turn" },
  moved: { label: "moved", act: "revert", hint: "move this note back" },
  deleted: { label: "deleted", act: "restore", hint: "bring this note back" },
  failed: { label: "not written", act: null, hint: "" },
};

function writeCard(ref, effect, verb) {
  const copy = WRITE_COPY[effect] || WRITE_COPY.written;
  const card = document.createElement("div");
  card.className = "wcard " + effect;
  const op = document.createElement("span");
  op.className = "wc-op";
  op.textContent = copy.label;
  const path = document.createElement("span");
  // A deleted note has no page to open, and a failed write may have created
  // nothing at all: keep the path as a record, drop the click, or it routes to
  // /note and answers "not found in vault".
  // `wc-open`, not `note-link`: a card's path is the thing that changed, not a
  // citation, and borrowing the chip's class also borrowed its cyan underline —
  // which `:is(.msg, #note-body, …) .note-link` carries at ID specificity, so no
  // amount of class stacking could take it back off. The delegated open-note
  // handler matches both classes.
  const openable = effect !== "deleted" && effect !== "failed";
  path.className = "wc-path" + (openable ? " wc-open" : "");
  if (openable) path.dataset.path = ref;
  path.textContent = ref;
  path.title = ref;
  card.append(op, path);
  if (!copy.act) {
    const n = document.createElement("span");
    n.className = "wc-note";
    n.textContent = verb ? `${verb} failed · vault unchanged` : "vault unchanged";
    card.appendChild(n);
    return card;
  }
  // How big the change was. Filled in by loadChanges() from the same /changes
  // payload the sidebar reads, so the number on the card, the number on the
  // sidebar row and the number in the header of the diff this card opens are
  // one number off one baseline. A card that renders before that payload lands
  // simply has no tally yet, rather than a wrong one.
  const tally = document.createElement("span");
  tally.className = "wc-tally";
  tally.dataset.for = ref;
  card.appendChild(tally);
  const b = document.createElement("button");
  b.type = "button";
  b.className = "wc-act";
  b.textContent = copy.act;
  b.title = copy.hint;
  b.addEventListener("click", () => {
    b.disabled = true;
    card.classList.add("reverting");
    send("/undo " + ref);
  });
  card.appendChild(b);
  return card;
}

// A button that asks before it acts, without a modal. Reverting is itself a
// mutation of a corpus the user means to keep, and in bulk it was the least
// guarded action in the app: one click took back every note a turn had written,
// with no preview and nothing to press to get back. The button arms instead, so
// the second click is the decision and the first one costs nothing to abandon —
// doing nothing disarms it. No new markup, no focus trap, and the keyboard path
// is the one the button already had.
const ARM_MS = 4000;
function armThenRun(btn, label, armed, run) {
  let timer = null;
  const disarm = () => { timer = null; btn.classList.remove("armed"); btn.textContent = label; };
  btn.addEventListener("click", () => {
    if (timer) {
      clearTimeout(timer);
      disarm();
      btn.disabled = true;
      run();
      return;
    }
    btn.classList.add("armed");
    btn.textContent = armed;
    timer = setTimeout(disarm, ARM_MS);
  });
  btn.addEventListener("blur", () => { if (timer) { clearTimeout(timer); disarm(); } });
}

// Raw exception text used to land in the transcript as the agent's own speech:
// "HTTPError 502 … (request id 4f2a-9c11-bd03)". Say what happened and what to do
// about it, in the product's own language. The original text is never discarded —
// it stays on the element's title, because the person debugging this needs it.
const ERROR_PLAIN = [
  [/\b(50[0-9]|timeout|timed out|connection|ECONNREFUSED|unreachable)\b/i,
    "the model endpoint didn't answer. Try again, or check the endpoint under the model button."],
  [/\b(401|403|unauthorized|forbidden|api[_ -]?key)\b/i,
    "the provider rejected the credentials. Check the API key for this endpoint."],
  [/\b(429|rate limit)\b/i, "the provider is rate-limiting. Wait a moment and try again."],
  [/lint failed/i, "the write was rolled back because the note would have broken a vault rule. The vault is unchanged."],
  [/not found in vault|no such note/i, "that note isn't in the vault under that name."],
  [/context length|too many tokens|max_tokens/i,
    "the conversation outgrew the model's context. Start a new chat, or narrow the question."],
];

function plainError(raw) {
  const s = String(raw || "");
  for (const [re, msg] of ERROR_PLAIN) if (re.test(s)) return msg;
  return s;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// Hover-revealed "copy" button in a message body's corner. getText() is called
// at click time so live turns can hand back their accumulated raw markdown.
function addCopyBtn(bodyEl, getText) {
  const b = document.createElement("button");
  b.className = "copy-btn";
  b.type = "button";
  b.textContent = "copy";
  b.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(getText()); b.textContent = "copied"; }
    catch { b.textContent = "failed"; }
    setTimeout(() => (b.textContent = "copy"), 1200);
  });
  bodyEl.appendChild(b);
}

// ponytail: lazy live markdown for the streaming turn — headings, bold, italic,
// inline + fenced code, bullet/ordered lists, links, rules, GFM tables. Re-parses the whole segment
// on every delta (O(n²) over the turn, fine at KB scale; parse from the last
// block boundary if very long turns ever stutter). The server re-renders
// the canonical answer (wikilinks, callouts, mermaid) on `done` for uninterrupted
// turns; swap in a vendored parser if full CommonMark is ever needed here.
function mdLite(src) {
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  // `[[note|alias]]` -> the same .note-link the server render emits. The target
  // is passed through raw: /note resolves titles and paths itself, so the live
  // segment needs no vault index (and an unresolvable one just says so on click).
  const wiki = (t) =>
    t.replace(/\[\[([^\]|\n]+?)(?:\|([^\]\n]+?))?\]\]/g, (_m, target, alias) => {
      const path = target.split("#")[0].trim();
      const shown = (alias || path.split("/").pop().replace(/\.md$/, "")).trim();
      return `<a class="note-link" data-path="${path.replace(/"/g, "&quot;")}">${shown}</a>`;
    });
  // Only these schemes become a live href. mdLite builds its anchors by hand and
  // gets none of the validateLink pass markdown-it runs server-side, so a
  // model-authored `[x](javascript:…)` landed as a live anchor in the app's own
  // origin. A whitelist, not a blocklist: `java\tscript:` walks straight through
  // a blocklist and the browser still runs it. Unsafe → the text, no anchor.
  const safeHref = (u) => {
    const s = u.trim().replace(/[\x00-\x1f]/g, "");
    const scheme = /^([a-z][a-z0-9+.-]*):/i.exec(s);
    return !scheme || ["http", "https", "mailto"].includes(scheme[1].toLowerCase()) ? s : null;
  };
  // Sentence punctuation and an unbalanced closing paren belong to the prose, not
  // to the URL — the same call linkify-it makes server-side, so a citation ending
  // in a full stop links the same in both renders (a Wikipedia URL's own balanced
  // parens survive).
  const trimUrl = (u) => {
    let end = u.length;
    const count = (s, c) => (s.split(c).length - 1);
    for (;;) {
      if (end > 0 && ".,;:!?".includes(u[end - 1])) { end--; continue; }
      const head = u.slice(0, end);
      if (end > 0 && u[end - 1] === ")" && count(head, "(") < count(head, ")")) { end--; continue; }
      return u.slice(0, end);
    }
  };
  // Both link forms in ONE pass. A bare URL matched before the markdown form eats
  // the target inside `](…)`; matched after, it re-matches the URL already sitting
  // in `href="…"` and nests the anchors. The lookbehind keeps it off the target of
  // a `[[…]]` that wiki() already turned into an attribute.
  const LINK = /\[([^\]]+)\]\(([^)\s]+)\)|(?<![\w"'=@./-])(https?:\/\/[^\s<>"'`]+)/g;
  const inline = (t) => {
    // Code spans are parked as placeholders for the rest of the pass: emphasis and
    // links used to be applied INSIDE the <code> they had already produced, so
    // `` `https://x` `` would have come out of this change as a live anchor.
    const code = [];
    return wiki(esc(t))
      .replace(/`([^`]+)`/g, (_m, c) => `\u0000${code.push(c) - 1}\u0000`)
      .replace(/\*\*([^*]+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*\n]+?)\*/g, "<em>$1</em>")
      .replace(LINK, (_m, txt, target, bare) => {
        if (bare === undefined) {
          const h = safeHref(target);
          return h ? `<a href="${h.replace(/"/g, "&quot;")}">${txt}</a>` : txt;
        }
        const u = trimUrl(bare);
        return `<a href="${u.replace(/"/g, "&quot;")}">${u}</a>` + bare.slice(u.length);
      })
      .replace(/\u0000(\d+)\u0000/g, (_m, i) => `<code>${code[i]}</code>`);
  };
  const lines = src.split("\n");
  const out = [];
  let i = 0, list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const HR = /^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/;
  // A GFM delimiter row: pipes, colons, spaces, and at least one dash.
  const DELIM = /^\s*\|?[\s:|-]*-[\s:|-]*$/;
  const isBlock = (l) => /^```|^#{1,6}\s|^\s*[-*]\s|^\s*\d+\.\s|^\s*\|/.test(l) || HR.test(l);
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { closeList(); i++; continue; }
    if (HR.test(line)) { closeList(); out.push("<hr>"); i++; continue; }
    // GFM table: a piped header row whose next line is the delimiter. Checked
    // before the paragraph branch, or the whole grid collapses into one <p> of
    // pipes — which is what every tool-interrupted turn was showing, since only
    // uninterrupted turns get upgraded to the server render.
    // Escaped `\|` cells and per-column alignment match the server render:
    // the escape parks as U+0001 so the split never sees it, and the delimiter
    // row's colons become the same inline text-align markdown-it emits.
    if (/^\s*\|/.test(line) && DELIM.test(lines[i + 1] || "")) {
      closeList();
      const cells = (l) =>
        l.trim().replace(/\\\|/g, "\u0001").replace(/^\||\|$/g, "").split("|")
          .map((c) => c.trim().replace(/\u0001/g, "|"));
      const head = cells(line);
      const align = cells(lines[i + 1]).map((c) =>
        /^:-+:$/.test(c) ? "center" : /^-+:$/.test(c) ? "right" : /^:-+$/.test(c) ? "left" : "");
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(cells(lines[i++]));
      const td = (c, j, tag) =>
        `<${tag}${align[j] ? ` style="text-align:${align[j]}"` : ""}>${inline(c)}</${tag}>`;
      const tr = (cs, tag) => `<tr>${cs.map((c, j) => td(c, j, tag)).join("")}</tr>`;
      out.push(
        `<table><thead>${tr(head, "th")}</thead><tbody>${rows.map((r) => tr(r, "td")).join("")}</tbody></table>`
      );
      continue;
    }
    if (/^```/.test(line)) {
      closeList();
      const buf = []; i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++; // closing fence (or EOF while still streaming)
      out.push(`<pre><code>${esc(buf.join("\n"))}</code></pre>`);
      continue;
    }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { closeList(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); i++; continue; }
    const item = line.match(/^\s*(?:[-*]|\d+\.)\s+(.*)$/);
    if (item) {
      const want = /^\s*\d/.test(line) ? "ol" : "ul";
      if (list !== want) { closeList(); out.push(`<${want}>`); list = want; }
      out.push(`<li>${inline(item[1])}</li>`); i++; continue;
    }
    closeList();
    // The current line always goes in, even when isBlock() calls it a block. A
    // table header whose delimiter row has not streamed in yet lands here: it is
    // a block by isBlock(), so an empty paragraph would leave `i` untouched and
    // the outer loop would spin forever, growing `out` until the tab threw
    // "RangeError: Invalid array length" — which killed the SSE reader and ate
    // the rest of the answer. Consuming one line per pass is what guarantees the
    // parser terminates, whatever the half-arrived block looks like.
    const para = [lines[i++]];
    while (i < lines.length && lines[i].trim() && !isBlock(lines[i])) para.push(lines[i++]);
    out.push(`<p>${para.map(inline).join("<br>")}</p>`);
  }
  closeList();
  return out.join("");
}

// A JSON payload that rides a header. Fail-open by design: a meter is a reading,
// and a malformed one must leave the app answering questions, not stop it.
function jsonHeader(r, name) {
  try { return JSON.parse(r.headers.get(name) || "null"); } catch { return null; }
}

function fmtTokens(n) {
  n = Number(n) || 0;
  return n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);
}
// --- the context ring -------------------------------------------------------
// One number, three readings: the ring is how full the window is, the panel is
// what filled it, and the caution level is the one fill that means something.
// The parts are counted server-side by _context_breakdown, which charges
// litellm's per-call chat envelope exactly once so they sum to the total printed
// beside them — a meter that invites you to add up its own segments has to
// survive that.
const CTX_R = 8;                    // tracks the <circle r> in index.html
const CTX_C = 2 * Math.PI * CTX_R;
// In the order the model meets them, which is also the reverse of the order
// they stop being yours: the instructions and the tool block never move, the
// read results are what compaction collapses first. The tool definitions are
// their own part rather than part of `instructions` because they are the
// largest resident of an idle window (7.2k against 2.1k), and a window that
// counted only the messages reported a sixth of what the provider billed.
// Ordinal ramp, not the interface palette: this is a bar series and --accent is
// reserved for what you can click.
const CTX_PARTS = [
  ["system", "instructions", "ord-1"],
  ["tool_specs", "tool definitions", "ord-2"],
  ["tool_io", "recall + tool results", "ord-3"],
  ["messages", "conversation", "ord-4"],
];
let ctxMeter = { used: 0, max: 0, parts: null, compactAt: 0.6 };

function setCtxTokens(used, max, parts, compactAt) {
  ctxMeter = {
    used: Number(used) || 0,
    max: Number(max) || 0,
    parts: parts || null,
    compactAt: Number(compactAt) || ctxMeter.compactAt,
  };
  const ring = $("#ctx-ring");
  if (!ring) return;
  // An empty window has nothing to read: on a fresh chat the ring would be a
  // blank circle beside send, which is a control that answers nothing. It
  // appears with the first turn, the way #side-changes appears with the first
  // write.
  ring.hidden = !(ctxMeter.max && ctxMeter.used);
  if (ring.hidden) { closeCtxPanel(); return; }
  const frac = Math.min(1, ctxMeter.used / ctxMeter.max);
  const pct = Math.round(frac * 100);
  ring.querySelector(".ctx-fill").style.strokeDasharray =
    `${(CTX_C * frac).toFixed(2)} ${CTX_C.toFixed(2)}`;
  // Amber is caution in this palette and nothing else, so it arms where the
  // condition it names actually begins — the fill at which the loop starts
  // collapsing old read results — not at an eyeballed "looks full".
  ring.dataset.level = frac >= ctxMeter.compactAt ? "high" : "";
  ring.setAttribute("aria-label",
    `context window ${pct}% full, ${fmtTokens(ctxMeter.used)} of ${fmtTokens(ctxMeter.max)}`);
  ring.title = `context ${pct}% · ${fmtTokens(ctxMeter.used)}/${fmtTokens(ctxMeter.max)}`;
  if (!$("#ctx-panel").hidden) renderCtxPanel();
}

function renderCtxPanel() {
  const panel = $("#ctx-panel");
  const { used, max, parts, compactAt } = ctxMeter;
  const pct = max ? Math.round((used / max) * 100) : 0;
  const free = Math.max(0, max - used);
  const rows = [];
  // The bar is the composition of what is IN the window, at full width, and the
  // ring beside it is how full. Splitting the two questions is what keeps the
  // bar readable: a bar that also carried the free space spent 98% of itself on
  // the empty end at every fill anyone actually works at, and compressed the
  // three parts it exists to compare into a six-pixel sliver.
  const segs = [];
  for (const [key, label, tier] of CTX_PARTS) {
    const n = (parts && Number(parts[key])) || 0;
    if (n > 0) segs.push(`<div class="stack-seg ${tier}" style="flex:${n}"></div>`);
    rows.push(`<div class="ctxp-row"><span class="swatch ${tier}"></span>`
      + `<span class="ctxp-k">${label}</span>`
      + `<span class="ctxp-v">${fmtTokens(n)}</span></div>`);
  }

  // Stated rather than implied by the empty part of the bar: what a reader wants
  // at 70% is how much room is left, and reading it off a track is a subtraction
  // they should not have to do.
  rows.push(`<div class="ctxp-row free">`
    + `<span class="ctxp-k">free</span>`
    + `<span class="ctxp-v">${fmtTokens(free)}</span></div>`);
  // The one thing this panel exists to warn about, and only while it is true.
  const note = used / max >= compactAt
    ? `<p class="ctxp-note">Past ${Math.round(compactAt * 100)}%: old read results`
      + ` are being collapsed to stubs that name the call to re-issue.</p>`
    : "";
  panel.innerHTML = `<div class="ctxp-head"><span class="ctxp-title">Context</span>`
    + `<span class="ctxp-fig">${fmtTokens(used)} / ${fmtTokens(max)}</span>`
    + `<span class="ctxp-pct">${pct}%</span></div>`
    + `<div class="stack">${segs.join("")}</div>`
    + rows.join("") + note;
}

function closeCtxPanel() {
  const panel = $("#ctx-panel");
  if (!panel || panel.hidden) return;
  panel.hidden = true;
  $("#ctx-ring").setAttribute("aria-expanded", "false");
}

// Both send buttons go dead for the length of a turn: the server answers one at
// a time (409 otherwise), and an enabled-looking button that discards the click
// is worse than a disabled one.
function setSendDisabled(v) {
  $("#send").disabled = v;
  $("#dock-send").disabled = v;
  // Both edges of a turn in one place: an open settings panel locks itself while
  // a response runs and unlocks when it ends, rather than waiting for a 409.
  stSetBusy(v);
}

// `retry` re-runs the exact same turn. It is a callback rather than the text so
// that /find, the dock and a nucleate-and-ask all retry the thing THEY sent.
async function runTurn(fetchPromise, pendingLabel = "working", retry = null) {
  if (streaming) return;
  streaming = true;
  stopBtn.hidden = false;
  setSendDisabled(true);
  announce("silica is responding");
  const body = bubble("silica");
  // flow = thinking blocks, tool groups and text segments interleaved in arrival
  // order, so the transcript reads chronologically: think, tools, think, tools,
  // text… (Claude-style). In this agent the connective tissue between tool calls
  // is *thinking*, so it must interleave too or tools pile into one group.
  const flow = document.createElement("div");
  body.appendChild(flow);

  // The live iridescent caret is ONE physical element, re-parented onto
  // whatever is streaming right now (thinking body / tool group / text tail).
  const caret = document.createElement("span");
  caret.className = "caret";
  caret.textContent = "▍";

  const toolEls = {};
  const texts = [];    // every text segment { el, raw }, for the copy button
  // ref → effect ("read" | "written" | "moved" | "deleted"), for the footer.
  // A ref is only recorded once its tool SUCCEEDS: a failed write must not be
  // reported as written, in the one place the user looks to trust the agent.
  const touched = new Map();
  // ref → the verb that failed. A write that fails lints and self-reverts used to
  // leave NOTHING behind: its refs were dropped, so the footer showed no chip and
  // the only trace was a tool line with a raw exception in it. "the vault did not
  // change" is a result the user needs stated, not inferred from an absence.
  const failed = new Map();
  const claimed = {};  // call id → { refs, effect, verb }, held until tool_done
  const pipes = {};    // call id → injector pipeline block, held until tool_done
  let curPipe = null;  // the block phase events currently belong to
  let curText = null;   // open markdown segment { el, raw }
  let curTools = null;  // open group of consecutive tools
  let curThink = null;  // open thinking block { details, body, raw }
  let segments = 0;     // text runs so far; an uninterrupted one upgrades to server html
  // Segments painted since the last tool block — all a `reset` can still take
  // back. A tool result is committed, so anything above one stands, and the
  // *open* segment is not the unit: a retry that streamed think→text has already
  // let go of its thinking block by the time the retraction arrives.
  let live = [];

  // Opening one segment kind closes the other two; a thinking block collapses
  // as it closes (it stays open only while it is the live tail).
  function close(keep) {
    if (keep !== "text") curText = null;
    if (keep !== "tools") curTools = null;
    if (keep !== "think" && curThink) { curThink.details.open = false; curThink = null; }
  }
  function thinkSeg() {
    if (curThink) return curThink;
    close("think");
    const details = document.createElement("details");
    details.className = "thinking";
    details.open = true;
    details.innerHTML = `<summary>thinking</summary><div class="thinking-body"></div>`;
    flow.appendChild(details);
    curThink = { details, body: details.querySelector(".thinking-body"), raw: "" };
    live.push(curThink);
    return curThink;
  }
  function textSeg() {
    if (curText) return curText;
    close("text");
    const el = document.createElement("div");
    el.className = "stream-text";
    flow.appendChild(el);
    curText = { el, raw: "" };
    texts.push(curText);
    live.push(curText);
    segments++;
    return curText;
  }
  // Retract the model's output for a server-sent `reset` delta. `textOnly` keeps
  // the thinking: a turn that resolved into a tool call retracts the preamble it
  // streamed, not the reasoning that produced the call. A full reset (a retry
  // replays the attempt from the top) takes the reasoning too, or the thinking
  // block ends up holding both passes.
  function dropLiveSegments(textOnly) {
    for (const seg of live) {
      if (seg.details) {
        if (!textOnly) seg.details.remove();
        continue;
      }
      seg.el.remove();
      const i = texts.indexOf(seg);
      if (i >= 0) texts.splice(i, 1);
      segments--; // keeps `done`'s "uninterrupted turn" upgrade test honest
    }
    live = textOnly ? live.filter((s) => s.details) : [];
    curText = null;
    if (!textOnly) curThink = null;
    peekRollback();
    // The drop detached the caret with the segment it lived in; a retry can
    // back off for seconds, and a bubble with no activity marker reads as done.
    flow.appendChild(caret);
  }
  function toolsGroup() {
    if (curTools) return curTools;
    close("tools");
    const g = document.createElement("div");
    g.className = "tools";
    flow.appendChild(g);
    live = [];  // a tool result commits everything above it
    peekMark(); // …including the dock's copy of it
    return (curTools = g);
  }
  const flowMsg = (s) => { const d = document.createElement("div"); d.className = "stream-text"; d.textContent = s; flow.appendChild(d); };

  // The first SSE event can be minutes away — /nucleate converts a PDF before
  // the turn even starts — and until it lands `flow` is empty, so the answer
  // block read as a hang next to a live Stop button. Park a pulsing line and the
  // caret there and drop them on the first event, as openPeek() does for the dock.
  const pending = document.createElement("div");
  pending.className = "tools";
  pending.innerHTML = `<div class="tool">» ${escapeHtml(pendingLabel)} …</div>`;
  flow.appendChild(pending);
  pending.appendChild(caret);

  try {
    const resp = await fetchPromise;
    if (resp.status === 409) { flowMsg("(a turn is already in progress)"); return; }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        handle(JSON.parse(line.slice(6)));
      }
    }
  } catch (e) {
    flowMsg("error: " + e);
    peekError(String(e));
    notify("the turn failed: " + e);
  } finally {
    streaming = false;
    stopBtn.hidden = true;
    setSendDisabled(false);
    pending.remove(); // no-op if the first event already dropped it
    caret.remove(); // no-op if a rerender already detached it
    freezePeek(); // done or aborted — stop mirroring, keep the preview up
    if (curThink) curThink.details.open = false; // aborted mid-thought — still collapse
    // A mutation is an object, a read is a citation, and they no longer share a
    // row of identical chips. The product's whole claim is that a write to your
    // vault is safe to delegate, and a write used to announce itself as a 12px
    // chip with a small outlined button beside it — the least prominent thing in
    // its own footer. Each one now gets a card it can carry a state on, and its
    // own revert, so you undo the write you are looking at rather than the turn.
    const mutations = [...touched].filter(([, e]) => e !== "read");
    if (mutations.length || failed.size) {
      const w = document.createElement("div");
      w.className = "writes";
      for (const [ref, effect] of mutations) w.appendChild(writeCard(ref, effect, null));
      for (const [ref, verb] of failed) w.appendChild(writeCard(ref, "failed", verb));
      // One button for the whole turn stays, but only when there is more than one
      // card: with a single write, per-card revert already says it better.
      if (mutations.length > 1) {
        const u = document.createElement("button");
        u.type = "button";
        u.className = "undo-turn";
        const label = `revert all ${mutations.length} changes`;
        u.textContent = label;
        u.title = "run /undo for every note this turn touched";
        armThenRun(u, label, `click again to revert ${mutations.length}`, () => send("/undo"));
        w.appendChild(u);
      }
      flow.appendChild(w);
    }
    const reads = [...touched].filter(([, e]) => e === "read").map(([r]) => r);
    if (reads.length) {
      const s = document.createElement("div");
      s.className = "sources";
      const g = document.createElement("div");
      g.className = "sgroup read";
      g.innerHTML = '<span class="sources-label">read</span>';
      for (const ref of reads) {
        const c = document.createElement("span");
        c.className = "note-link";
        c.dataset.path = ref; // delegated click → note drawer
        c.textContent = ref.split("/").pop().replace(/\.md$/, "");
        g.appendChild(c);
      }
      s.appendChild(g);
      flow.appendChild(s);
    }
    const answer = texts.map((t) => t.raw).join("\n\n").trim();
    if (answer) addCopyBtn(body, () => answer);
    loadSessions(); // turn saved server-side — refresh titles/order
    loadVaultInfo(); // a turn may have written notes — refresh stats + tree
    loadChanges();   // …and the sidebar's record of what it changed
    markVaultChanged(); // a turn may have written notes
  }

  function handle(ev) {
    pending.remove(); // something arrived — the placeholder has done its job
    if (ev.type === "delta" && ev.kind === "reset") {
      // The server retracts what it just streamed: a transient retry replays the
      // whole attempt (agent/llm.py), and a turn that resolved into tool calls
      // streamed a preamble, never an answer (agent/loop.py). Without this branch
      // the event fell through and the replay was spliced under the truncated
      // first take, so the GUI showed a duplicated answer the TUI did not.
      // A reset's `text` is the retraction scope, not delta text: "" takes the
      // whole attempt (reasoning included), "text" the answer alone.
      dropLiveSegments(ev.text === "text");
    } else if (ev.type === "delta" && ev.kind === "reasoning") {
      const th = thinkSeg();
      th.raw += ev.text;
      th.body.textContent = th.raw;
      th.body.appendChild(caret);
      th.body.scrollTop = th.body.scrollHeight; // follow the caret in the capped box
    } else if (ev.type === "delta" && ev.kind === "text") {
      const seg = textSeg();
      seg.raw += ev.text;
      seg.el.innerHTML = mdLite(seg.raw);
      (seg.el.lastElementChild || seg.el).appendChild(caret); // inline at the text tail
      peekDelta(ev.text);
    } else if (ev.type === "tool_start") {
      if (ev.pipeline) {
        // A nucleate run gets the block instead of a line; tool calls are
        // dispatched one at a time (agent/loop.py), so the phase events that
        // follow belong to this one until its tool_done arrives.
        const p = makePipelineBlock(ev.target || "?", ev.pipeline);
        toolsGroup().appendChild(p.el);
        curTools.appendChild(caret);
        pipes[ev.id] = p;
        curPipe = p;
        claimed[ev.id] = { refs: ev.notes || [], effect: ev.effect || "read", verb: ev.name };
        return;
      }
      const t = makeToolRow(toolLabel(ev));
      t.section("in", ev.input);
      toolsGroup().appendChild(t.el);
      curTools.appendChild(caret);
      toolEls[ev.id] = t;
      claimed[ev.id] = { refs: ev.notes || [], effect: ev.effect || "read", verb: ev.name };
    } else if (ev.type === "phase") {
      if (curPipe) curPipe.applyPhase(ev);
      bumpChanges(); // notes land mid-run, not at tool_done
    } else if (ev.type === "tool_done") {
      bumpChanges(); // a write tool that emits no phases still changed the vault
      if (pipes[ev.id]) {
        pipes[ev.id].finish(ev.summary);
        if (curPipe === pipes[ev.id]) curPipe = null;
        delete pipes[ev.id];
      }
      const t = toolEls[ev.id];
      if (t) { t.section("out", ev.output); t.finish("done", ev.ms); }
      const c = claimed[ev.id];
      if (c) {
        // A mutation always wins over a read of the same note; a read never
        // downgrades a write recorded earlier in the turn.
        for (const r of c.refs) if (c.effect !== "read" || !touched.has(r)) touched.set(r, c.effect);
        delete claimed[ev.id];
      }
    } else if (ev.type === "tool_error") {
      if (pipes[ev.id]) {
        // No summary to read: the tool raised instead of returning a verdict, so
        // the block keeps the track open on whatever phase was in flight.
        pipes[ev.id].finish(null);
        if (curPipe === pipes[ev.id]) curPipe = null;
        delete pipes[ev.id];
      }
      const t = toolEls[ev.id];
      if (t) {
        t.setLabel(toolLabel(ev) + " · " + plainError(ev.error));
        // The raw text moves out of a `title` and into the card: a tooltip on a
        // row that is already a disclosure is a second, worse copy of the same
        // affordance, and it cannot be selected or copied.
        t.section("error", { text: ev.error, cut: 0 });
        t.finish("error", ev.ms);
      }
      const f = claimed[ev.id];
      // Still not claimed as written — but now recorded as a mutation that did
      // NOT land, so the turn can say so in the footer instead of going quiet.
      if (f && f.effect !== "read") for (const r of f.refs) if (!touched.has(r)) failed.set(r, f.verb || "write");
      delete claimed[ev.id]; // it failed: do not claim its notes
    } else if (ev.type === "batch") {
      toolsGroup().appendChild(makeToolRow(ev.kind + " · " + ev.label).el);
      curTools.appendChild(caret);
    } else if (ev.type === "done") {
      // Uninterrupted answer (no tool split the text) → upgrade the live md to the
      // canonical server render (wikilinks, callouts, mermaid). Interleaved turns
      // keep their live segments; they render canonically on the next reload.
      if (segments === 0 && (ev.html || ev.answer)) {
        const seg = textSeg();
        seg.raw = ev.answer || ""; // keep the copy button fed on no-delta turns
        seg.el.innerHTML = ev.html || escapeHtml(ev.answer || "");
      } else if (segments === 1 && curText && (ev.html || ev.answer)) {
        curText.el.innerHTML = ev.html || escapeHtml(ev.answer || "");
      }
      close(""); // collapse any open thinking, end all segments
      if (ev.hint) {
        // Informational only: every recall call this turn came back empty. It
        // arms nothing — /web works the same with or without it.
        const h = document.createElement("div");
        h.className = "turn-hint";
        h.textContent = ev.hint;
        flow.appendChild(h);
      }
      setCtxTokens(ev.context_tokens, ev.max_context_tokens, ev.context_parts, ev.compact_at);
      peekDone(ev); // card gets the canonical OFM render
      announce("response ready");
    } else if (ev.type === "error") {
      close("");
      peekError(plainError(ev.error));
      // The error belongs where it happened, not in a corner: it used to render
      // as a raw exception line attributed to the agent as speech AND as a toast
      // over the send button, and neither offered a way to try again — so a
      // failed turn ended the conversation and took the typed question with it.
      const box = document.createElement("div");
      box.className = "turn-error";
      const msg = document.createElement("div");
      msg.className = "te-msg";
      msg.textContent = plainError(ev.error);
      msg.title = ev.error; // raw text stays reachable
      box.appendChild(msg);
      if (retry) {
        const r = document.createElement("button");
        r.type = "button";
        r.className = "te-retry";
        r.textContent = "try again";
        r.addEventListener("click", () => {
          if (streaming) return;
          box.remove();
          retry();
        });
        box.appendChild(r);
      }
      flow.appendChild(box);
      announce("the turn failed: " + plainError(ev.error));
    }
    log.scrollTop = log.scrollHeight;
  }
}

function send(text, replay = false) {
  if (!text.trim() || streaming) return;
  // A retry re-runs the turn but must not stack a second copy of your question
  // in the transcript: the bubble from the first attempt is still there.
  if (!replay) bubble("user").textContent = text;
  const find = text.trim().match(/^\/find\s*(.*)$/);
  if (find) { runFind(find[1]); return; }
  runTurn(fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }), "working", () => send(text, true));
}

// /find bypasses the agent entirely — same "direct tool, no LLM" pattern as
// the /graph and /map tabs, just rendered inline as a result bubble.
async function runFind(rest) {
  const body = bubble("silica");
  // dock-launched /find: mirror the result bubble into the card (no SSE stream
  // here, so the peek would otherwise sit at "thinking" forever)
  const mirror = () => { if (peek) { peek.body.innerHTML = body.innerHTML; freezePeek(); } };
  let k = 5;
  const tokens = [];
  for (const part of rest.trim().split(/\s+/)) {
    const m = part.match(/^--k=(\d+)$/);
    if (m) k = parseInt(m[1], 10);
    else if (part) tokens.push(part);
  }
  const query = tokens.join(" ");
  if (!query) { body.textContent = "usage: /find <query> [--k=N]"; mirror(); return; }
  body.textContent = "searching…";
  try {
    const r = await fetch("/find?q=" + encodeURIComponent(query) + "&k=" + k);
    body.innerHTML = await r.text();
  } catch (e) {
    body.textContent = "error: " + e;
  }
  mirror();
}

// --- tool rows --------------------------------------------------------------
// A tool used to be one line that said `read Etica.md` and stopped. That is the
// verb and the object; what it cost and what came back were nowhere, so every
// step of a turn had to be taken on trust. The row now carries its own duration
// and opens onto the call it made and the answer it got — the two facts that
// separate "the agent read the note" from "the agent says it read the note".
//
// Server-measured duration, not a client timer started when the event painted:
// an SSE frame lands a frame late and a backgrounded tab stops scheduling
// altogether, so a clock on this side reports the transport as the tool's cost.
function fmtMs(ms) {
  if (!(ms >= 0)) return "";
  // A sub-millisecond call printed as "0ms" reads as "not measured", which is
  // the opposite of what it means: the tool was a cache hit.
  if (ms < 1) return "<1ms";
  if (ms < 1000) return Math.round(ms) + "ms";
  return (ms / 1000).toFixed(ms < 10000 ? 1 : 0) + "s";
}

function makeToolRow(label) {
  const el = document.createElement("div");
  el.className = "tool";
  el.dataset.state = "running";
  const row = document.createElement("button");
  row.type = "button";
  row.className = "tool-row";
  // Inert until it has a card. A transcript replay carries no results, so a
  // reloaded chat would otherwise put six focusable buttons in the tab order
  // that do nothing when activated — and a keyboard user has no hover to tell
  // them apart from the rows that open.
  row.disabled = true;
  row.innerHTML = '<span class="tool-mark" aria-hidden="true">\u00bb</span>'
    + '<span class="tool-label"></span><span class="tool-ms"></span>'
    + '<span class="tool-chev" aria-hidden="true">\u25b8</span>';
  const card = document.createElement("div");
  card.className = "tool-card";
  card.hidden = true;
  el.append(row, card);
  const labelEl = row.querySelector(".tool-label");
  labelEl.textContent = label;

  // Only a row with something to open is a control. The rest keeps the same
  // geometry and stays inert, so a group of tools does not read as a row of
  // buttons half of which do nothing.
  let sections = 0;
  function section(kind, payload) {
    if (!payload || !payload.text) return;
    if (sections) card.appendChild(mkEl("div", "tc-div"));
    const sec = mkEl("div", "tc-sec");
    sec.appendChild(mkEl("span", "tc-k", kind));
    // A div, not a <pre>: `:is(.msg, …) pre` gives every pre in a turn the code
    // well's own surface, and this text already sits inside one. `white-space`
    // below is what <pre> was ever here for, and .tc-text sets it.
    const pre = mkEl("div", "tc-text", payload.text);
    if (payload.cut) {
      // Named, not elided: a result that genuinely ends in an ellipsis and one
      // that was cut look identical once you print "…" and say nothing.
      pre.appendChild(mkEl("span", "tc-cut", `\u2026 ${payload.cut} more characters`));
    }
    sec.appendChild(pre);
    card.appendChild(sec);
    sections++;
    el.dataset.open = "shut";
    row.disabled = false;
    row.setAttribute("aria-expanded", "false");
  }
  row.addEventListener("click", () => {
    if (!sections) return;
    const opening = card.hidden;
    card.hidden = !opening;
    el.dataset.open = opening ? "open" : "shut";
    row.setAttribute("aria-expanded", opening ? "true" : "false");
  });

  return {
    el,
    section,
    setLabel(text) { labelEl.textContent = text; },
    finish(state, ms) {
      el.dataset.state = state;                          // done | error
      row.querySelector(".tool-mark").textContent = state === "error" ? "\u2717" : "\u2713";
      const t = fmtMs(ms);
      if (t) row.querySelector(".tool-ms").textContent = t;
    },
  };
}

// --- composer ---------------------------------------------------------------
function autoGrow(el) {
  el.style.height = "auto";
  const border = el.offsetHeight - el.clientHeight; // box-sizing: border-box
  el.style.height = (el.scrollHeight + border) + "px"; // clamped visually by CSS max-height
}
$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  // Guard BEFORE clearing. send() and nucleateStaged() both bail out on
  // `streaming`, so clearing first silently destroyed a follow-up typed while
  // the answer was still landing — the most natural thing to do on this surface.
  // #dock-composer already had the check in this order.
  if (streaming) return;
  const t = input.value;
  input.value = "";
  autoGrow(input);
  renderCommands(input.value); // clearing by hand fires no `input` event — dismiss the picker
  renderHighlight();           // …and clear the bands with it
  if (staged.length) nucleateStaged(t); // files attached: upload + act on them together
  else send(t);
});
let allCommands = [];
let filteredCommands = [];
let cmdSelIdx = -1;

fetch("/commands").then(r => r.json()).then(data => allCommands = data || []).catch(() => {});

function renderCommands(q) {
  const box = $("#commands");
  syncQuick(); // every path that changes the box comes through here
  if (!q.startsWith("/")) {
    box.hidden = true;
    return;
  }
  const search = q.substring(1).toLowerCase();
  
  filteredCommands = allCommands.map(cmd => {
    let score = 0;
    const name = cmd.name.substring(1).toLowerCase();
    if (name === search) score = 10;
    else if (name.startsWith(search)) score = 5;
    else if (name.includes(search)) score = 3;
    else {
      let i = 0;
      let matched = true;
      for (const c of search) {
        i = name.indexOf(c, i);
        if (i === -1) { matched = false; break; }
        i++;
      }
      if (matched && search.length > 0) score = 1;
    }
    return { cmd, score };
  }).filter(x => x.score > 0).sort((a, b) => b.score - a.score || a.cmd.name.localeCompare(b.cmd.name)).map(x => x.cmd);

  if (!filteredCommands.length) {
    box.hidden = true;
    return;
  }
  
  cmdSelIdx = 0;
  box.innerHTML = "";
  filteredCommands.forEach((c, i) => {
    const el = document.createElement("button");
    el.className = "cmd-item" + (i === cmdSelIdx ? " sel" : "");
    el.type = "button";
    el.innerHTML = `<span class="cmd-name">${c.name}</span><span class="cmd-summary">${escapeHtml(c.usage ? c.usage + " · " + c.summary : c.summary)}</span>`;
    el.title = c.usage ? c.usage + " · " + c.summary : c.summary;
    el.addEventListener("click", () => pickCommand(c));
    box.appendChild(el);
  });
  box.hidden = false;
}

function updateCmdSel() {
  const box = $("#commands");
  Array.from(box.children).forEach((el, i) => {
    el.classList.toggle("sel", i === cmdSelIdx);
    if (i === cmdSelIdx) el.scrollIntoView({ block: "nearest" });
  });
}

function pickCommand(c) {
  input.value = c.name + (c.usage ? " " : "");
  input.focus();
  renderCommands(input.value);
  renderHighlight();
}

// --- the composer mirror ----------------------------------------------------
// A layer exactly the size of the field, behind it, drawing what the box has
// understood: a band under the leading command, under every [[wikilink]] and
// under every path, plus the greyed rest of a command you have started typing.
//
// It draws BACKGROUNDS and never glyphs. The textarea keeps its own opaque
// text, caret and selection on top, so the two layers can only ever disagree by
// a band sitting a pixel off — where the usual mirror trick (transparent
// textarea, glyphs in the backdrop) disagrees by showing the wrong characters,
// which is what makes it flicker on IME input and lag a frame behind a wheel
// scroll. The cost of the safe version is that the tokens are tinted rather
// than coloured; the benefit is that nothing here can ever ghost.
const inputHl = $("#input-hl");
// [[wikilink]] first, so a link is never re-matched as a path by its own text.
const HL_TOKENS = [
  [/\[\[[^\][]+\]\]/g, "link"],
  [/(?:^|\s)((?:[\w.\-]+\/)+[\w.\-]+\.md)/g, "path"],
];

function renderHighlight() {
  const raw = input.value;
  if (!raw) { inputHl.textContent = ""; inputHl.scrollTop = 0; return; }
  // Marks by offset, then one pass: overlapping patterns would otherwise nest
  // spans inside each other and the second band would paint over the first.
  const marks = [];
  const cmd = /^\/[a-z-]+/.exec(raw);
  if (cmd) marks.push([0, cmd[0].length, "cmd"]);
  for (const [re, cls] of HL_TOKENS) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(raw))) {
      const hit = m[1] !== undefined ? m[1] : m[0];
      const at = m.index + m[0].indexOf(hit);
      if (!marks.some(([a, b]) => at < b && at + hit.length > a))
        marks.push([at, at + hit.length, cls]);
    }
  }
  marks.sort((a, b) => a[0] - b[0]);
  let out = "";
  let at = 0;
  for (const [start, end, cls] of marks) {
    out += escapeHtml(raw.slice(at, start));
    out += `<span class="hl-${cls}">${escapeHtml(raw.slice(start, end))}</span>`;
    at = end;
  }
  out += escapeHtml(raw.slice(at));
  // The ghost completes the command you are typing, and only while exactly one
  // command can still be meant: offering the first of five is a guess wearing
  // the clothes of an answer, and the picker below already lists all five.
  if (cmd && cmd[0].length === raw.length) {
    const hits = allCommands.filter((c) => c.name.startsWith(raw) && c.name !== raw);
    if (hits.length === 1) out += `<span class="hl-ghost">${escapeHtml(hits[0].name.slice(raw.length))}</span>`;
  }
  inputHl.innerHTML = out;
  inputHl.scrollTop = input.scrollTop;
}

input.addEventListener("input", () => {
  autoGrow(input);
  renderCommands(input.value);
  renderHighlight();
});
// Past the 40vh cap the field scrolls, and a band that stays put while its word
// leaves the viewport is worse than no band at all.
input.addEventListener("scroll", () => { inputHl.scrollTop = input.scrollTop; });

input.addEventListener("keydown", (e) => {
  const box = $("#commands");
  if (!box.hidden && filteredCommands.length > 0) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      cmdSelIdx = (cmdSelIdx + 1) % filteredCommands.length;
      updateCmdSel();
      return;
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      cmdSelIdx = (cmdSelIdx - 1 + filteredCommands.length) % filteredCommands.length;
      updateCmdSel();
      return;
    } else if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (cmdSelIdx >= 0 && cmdSelIdx < filteredCommands.length) {
        pickCommand(filteredCommands[cmdSelIdx]);
      }
      return;
    } else if (e.key === "Escape") {
      e.preventDefault();
      box.hidden = true;
      return;
    }
  }

  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#composer").requestSubmit();
    box.hidden = true;
  }
});

// --- dock composer (graph/map) — same conversation, mirrored into the card ---
// The turn is a real chat turn (user bubble + transcript land in the chat tab);
// the dock card is a lens showing only the latest exchange.
const dockInput = $("#dock-input");
$("#dock-composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const t = dockInput.value;
  if (!t.trim() || streaming) return;
  dockInput.value = "";
  autoGrow(dockInput);
  openPeek(t.trim());
  send(t);
});
dockInput.addEventListener("input", () => autoGrow(dockInput));
dockInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#dock-composer").requestSubmit();
  }
});
stopBtn.addEventListener("click", () => fetch("/stop", { method: "POST" }));
// Optimistic: clear the transcript at once (the reset itself is a cached-seed
// copy server-side, but never make the click wait on the network).
$("#brand-logo").addEventListener("click", async () => {
  if (streaming) return;
  log.innerHTML = "";
  await fetch("/reset", { method: "POST" });
  announceSession();
  document.querySelector('.tab[data-tab="chat"]').click(); // surface the loaded chat
  loadVault();
  loadSessions();
});

