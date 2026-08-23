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
// they stop being yours: the instructions never move, the read results are what
// compaction collapses first. Ordinal ramp, not the interface palette: this is a
// bar series and --accent is reserved for what you can click.
const CTX_PARTS = [
  ["system", "instructions", "ord-1"],
  ["tools", "recall + tool results", "ord-2"],
  ["messages", "conversation", "ord-3"],
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

// --- unified sidebar (stats · search · files · history) ----------------------
if (localStorage.getItem("sidebar-collapsed") === "1")
  document.body.classList.add("sidebar-collapsed");
$("#sidebar-toggle").addEventListener("click", () => {
  // Below the floor the rail is already a strip of icons and there is nothing
  // left to collapse; the same button summons it over the transcript instead.
  if (isNarrow()) { toggleRail(railSection || "side-files"); return; }
  const collapsed = document.body.classList.toggle("sidebar-collapsed");
  localStorage.setItem("sidebar-collapsed", collapsed ? "1" : "0");
  sidebarYielded = false; // an explicit choice outranks the drawer's auto-yield
});

// --- the 1120 floor: the deck folds, it does not reflow ----------------------
// Below the width that holds rail + reading measure + work panel side by side,
// the rail becomes five icons and the work panel becomes an overlay over the
// transcript. Neither disappears: what a fold must never do is take a surface
// away and leave nothing where it was.
//
// The threshold itself lives in ONE place, the media query in app.css that sets
// --narrow. Both this file and work.js read it back through here rather than
// each carrying a pixel count of its own, because two constants that must match
// are two constants that eventually do not.
function isNarrow() {
  return getComputedStyle(document.body).getPropertyValue("--narrow").trim() === "1";
}
window.isNarrow = isNarrow;

let railSection = null; // which compartment the summoned rail is showing

function toggleRail(sec) {
  const open = railSection !== sec;
  railSection = open ? sec : null;
  document.body.classList.toggle("rail-open", open);
  for (const b of document.querySelectorAll("#railmini .rm"))
    b.classList.toggle("on", open && b.dataset.sec === sec);
  if (!open) return;
  // The compartment you asked for is the one that opens; the others keep the
  // state you left them in, so summoning Files twice does not re-collapse the
  // tree you just expanded.
  const el = document.getElementById(sec);
  if (el) { el.open = true; el.scrollIntoView({ block: "nearest" }); }
}

function closeRail() {
  if (!railSection) return;
  railSection = null;
  document.body.classList.remove("rail-open");
  for (const b of document.querySelectorAll("#railmini .rm")) b.classList.remove("on");
}

$("#railmini").addEventListener("click", (e) => {
  const b = e.target.closest(".rm");
  if (b) toggleRail(b.dataset.sec);
});
// The summoned rail sits over the transcript, so a click in the transcript is
// a dismissal. The rail itself and the strip that summoned it are not.
document.addEventListener("click", (e) => {
  if (!railSection) return;
  if (e.target.closest("#sidebar, #railmini, #sidebar-toggle")) return;
  closeRail();
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeRail(); });

// A compartment the vault cannot fill yet (Pinned before the first pin, This
// session before the first write) hides itself, and an icon standing for a
// hidden compartment is a button that opens nothing. app.js unhides those
// sections from four different places, so watch the result rather than adding a
// fifth call site to each.
function syncRailIcons() {
  for (const b of document.querySelectorAll("#railmini .rm")) {
    const sec = document.getElementById(b.dataset.sec);
    b.hidden = !sec || sec.hidden;
  }
}
new MutationObserver(syncRailIcons).observe($("#sidebar"),
  { attributes: true, attributeFilter: ["hidden"], subtree: true });

function syncNarrow() {
  const narrow = isNarrow();
  $("#railmini").hidden = !narrow;
  if (!narrow) closeRail();
  syncRailIcons();
}
syncNarrow();

// Vault stats + file tree, from /vault_info. Best-effort: on error the placeholders stay.
async function loadVaultInfo() {
  try {
    const r = await fetch("/vault_info");
    const data = await r.json();
    if (data.error) return;
    if (data.path) setVaultPath(data.path); // follows a /vault switch
    renderTopCounts(data);            // the strip's copy, the only one now
    $("#tree").innerHTML = data.tree || "";
    syncTreePins();          // a fresh tree knows nothing about the pins yet
    renderRailAreas(data);            // the rail's spectrum, from the same call
    renderVaultFacts(data);           // the chat landing's counted line
    renderMapPicker(data.hubs || []); // map landing: best-connected notes
    buildNoteIndex();                 // explore note search reads the fresh tree
    applySidebarFilter();
  } catch { notify("couldn't refresh vault stats"); }
}

// --- the top strip: identity and counts, stated once -------------------------
// These numbers used to live in the rail as a 2x2 board AND in metrics as four
// of its rates, which is the same fact in two places that can disagree. Here
// they are true on every view and cost the rail nothing.

// The folder's NAME in the strip, its path on hover. The rail printed the whole
// path over two wrapped lines; what identifies a vault at a glance is its last
// segment, and the rest is one hover and the settings panel away.
function setVaultPath(path) {
  const el = $("#top-vname");
  const was = el.title;
  const name = String(path || "").replace(/\/+$/, "").split("/").pop();
  el.textContent = name || "";
  el.title = path || "";
  // …and in full at the bottom of the rail. The strip's hover is the wrong home
  // for the answer to "which of my two vaults is this window", which is a thing
  // you check while reading something else.
  const foot = $("#railfoot");
  foot.textContent = path || "";
  foot.title = path || "";
  foot.hidden = !path;
  if (path) $("#top-vault").hidden = false;
  if (path && path !== was) loadPins(); // pins are per vault, and this is the switch
}

// A cluster count made entirely of singletons names nothing, so it is stated
// only when there are areas worth naming. One function, because the strip and
// the chat landing both state this number and two copies of the rule are two
// numbers that can disagree about the same vault.
function areaCount(data) {
  return (data.topics || []).length ? data.clusters : 0;
}

function renderTopCounts(data) {
  const bits = [];
  if (data.notes) bits.push(nfmt(data.notes) + (data.notes === 1 ? " note" : " notes"));
  if (data.links) bits.push(nfmt(data.links) + " links");
  const areas = areaCount(data);
  if (areas) bits.push(nfmt(areas) + " areas");
  $("#top-counts").textContent = bits.join("  ·  ");
  const broken = $("#top-broken");
  broken.hidden = !data.unresolved;
  broken.textContent = nfmt(data.unresolved || 0) + " broken";
  broken.title = data.unresolved + " unresolved wikilinks: targets that do not exist yet. Opens metrics.";
  $("#top-vault").hidden = !(bits.length || data.unresolved || $("#top-vname").textContent);
}

// A number you cannot act on from where it is stated is a number you read once.
$("#top-broken").addEventListener("click", () => {
  document.querySelector('.tab[data-tab="metrics"]').click();
});

// Two conditions, one hidden flag, and they arrive at different times: the
// areas come from a fetch, the view from a click. Whichever moves calls this,
// so neither can leave the compartment showing on chat or missing on explore.
// The areas ARE the colouring of the graph and the rows of the areas surface;
// on a transcript they are a list nothing on screen refers to.
let railHasAreas = false;

function syncAreasRail() {
  $("#side-areas").hidden = !railHasAreas || activeTab !== "graph";
}

// The rail's area spectrum, from the same /vault_info the landing reads: one
// fetch, two readings. The bar is proportional to the largest area, because
// what the rail is for here is the SHAPE of the distribution; the exact size is
// the figure beside it. The summary counts every area, not the five that fit.
function renderRailAreas(data) {
  const rows = (data.topics || []).filter((t) => t.label);
  const box = $("#areas");
  box.innerHTML = "";
  railHasAreas = rows.length > 0;
  syncAreasRail();
  $("#areas-count").textContent = rows.length ? nfmt(data.clusters) : "";
  const top = rows[0] ? rows[0].size : 1;
  for (const t of rows) {
    const row = mkEl("div", "area");
    row.title = t.label + " · " + t.size + " notes";
    row.appendChild(mkEl("span", "an", t.label));
    row.appendChild(mkEl("span", "ac", nfmt(t.size)));
    const bar = mkEl("span", "abar");
    const fill = mkEl("i");
    fill.style.width = Math.max(4, Math.round((t.size / (top || 1)) * 100)) + "%";
    bar.appendChild(fill);
    row.appendChild(bar);
    box.appendChild(row);
  }
}

// --- the rail's Layout compartment (explore only) ----------------------------
// The ONE place the five surfaces are named. They used to be a row of tabs in
// the graph toolbar as well, a hand's width from this list: two controls for one
// choice, and the rail is the one that survives a narrow window. The list is
// data here rather than markup because the toolbar copy is gone — there is no
// second DOM to read it back off.
const LAYOUT_MODES = [
  ["graph", "Graph", "wikilink structure + semantic k-NN overlay; toggle the layers in the HUD"],
  ["map", "Map", "radial map rooted on one note"],
  // Three surfaces over the same graph that are NOT link-space: graph and map
  // both lay notes out by how they connect, so neither can show where a note
  // SITS, how two areas couple as a whole, or what order the vault could be
  // read in.
  ["folders", "Folders", "the vault as folders, shaded by how much each folder mixes areas"],
  ["areas", "Areas", "area x area coupling: every pair at once, not a top-N list"],
  ["read", "Read", "a reading order derived from hubs and their links"],
  // The sixth, and the only one that is not undirected: the four above lay
  // notes out by how they connect, where they are filed, or how two groups
  // couple, and none of them can answer "what do I read BEFORE this".
  ["path", "Path", "reading order around one note: what RefD says comes before it, and what it unlocks"],
];

function buildLayoutRail() {
  const box = $("#layout-modes");
  box.innerHTML = "";
  for (const [mode, label, why] of LAYOUT_MODES) {
    const row = mkEl("button", "lay");
    row.type = "button";
    row.dataset.gmode = mode;
    row.title = why;
    row.appendChild(mkEl("span", "sw"));
    row.appendChild(mkEl("span", "t", label));
    row.addEventListener("click", () => setGraphMode(mode));
    box.appendChild(row);
  }
  syncLayoutRail();
}

function syncLayoutRail() {
  document.querySelectorAll("#layout-modes .lay")
    .forEach((b) => setActive(b, b.dataset.gmode === graphMode));
}

// --- pinned notes ------------------------------------------------------------
// The one thing in the rail the vault cannot derive. Files is alphabetical,
// History is chronological, Areas is computed: none of them can say "this one
// matters". Stored per vault, because a pin is a statement about the folder you
// are reading and switching vaults must not carry one vault's pins into another.
let pinnedPaths = [];

function pinKey() { return "pinned:" + ($("#top-vname").title || ""); }

function loadPins() {
  try {
    const raw = JSON.parse(localStorage.getItem(pinKey()) || "[]");
    pinnedPaths = Array.isArray(raw) ? raw.filter((x) => typeof x === "string") : [];
  } catch { pinnedPaths = []; } // a hand-edited key is not an error to report
  renderPins();
}

function savePins() {
  // Quota is the one failure worth saying out loud: silently, the rail forgets
  // a pin the moment the tab closes and the user never learns why.
  try { localStorage.setItem(pinKey(), JSON.stringify(pinnedPaths)); }
  catch { notify("couldn't save that pin"); }
  renderPins();
}

function togglePin(path) {
  if (!path) return;
  const i = pinnedPaths.indexOf(path);
  if (i >= 0) pinnedPaths.splice(i, 1);
  else pinnedPaths.unshift(path); // newest first: the pin you just made is visible
  savePins();
}

function syncPinButton() {
  const btn = $("#note-pin");
  const on = !!lastViewedPath && pinnedPaths.includes(lastViewedPath);
  btn.setAttribute("aria-pressed", String(on));
  btn.classList.toggle("on", on);
  btn.querySelector("span").textContent = on ? "pinned" : "pin";
}

// The same pin, stated on the row it came from. Pinned is a state of the NOTE,
// so the tree has to show it: a toggle that only ever looks unpressed is a
// button the reader has to click twice to learn what it did.
function syncTreePins() {
  const on = new Set(pinnedPaths);
  for (const b of $("#tree").querySelectorAll(".tree-pin")) {
    const lit = on.has(b.dataset.pin);
    b.setAttribute("aria-pressed", String(lit));
    b.classList.toggle("on", lit);
    b.title = lit ? "unpin this note" : "keep this note in the rail";
  }
}

function renderPins() {
  const box = $("#pinned");
  box.innerHTML = "";
  for (const path of pinnedPaths) {
    const row = mkEl("div", "pin-row");
    row.dataset.path = path;
    row.title = path;
    row.appendChild(mkEl("span", "sw"));
    row.appendChild(mkEl("span", "t", path.replace(/\.md$/, "").split("/").pop()));
    const x = mkEl("button", "pin-x", "✕");
    x.type = "button";
    x.setAttribute("aria-label", "unpin " + path);
    row.appendChild(x);
    box.appendChild(row);
  }
  $("#side-pinned").hidden = !pinnedPaths.length;
  $("#pinned-count").textContent = pinnedPaths.length || "";
  syncPinButton();
  syncTreePins();
  applySidebarFilter();
}

$("#pinned").addEventListener("click", (e) => {
  const row = e.target.closest(".pin-row");
  if (!row) return;
  if (e.target.closest(".pin-x")) { togglePin(row.dataset.path); return; }
  openNote(row.dataset.path);
});
$("#note-pin").addEventListener("click", () => togglePin(lastViewedPath));

// --- the chat landing --------------------------------------------------------
// Two lines about the folder you opened, over the wordmark. The counted one is
// this function and costs nothing; the written one is loadVaultBrief below and
// costs a call, so it is the one that can be switched off. When it is off the
// prose slot falls back to naming the topics — the landing must never describe
// a vault by its size alone, which is the one thing size cannot say.
let vaultTopics = [];

function renderVaultFacts(data) {
  vaultTopics = (data.topics || []).map((t) => t.label).filter(Boolean);
  const bits = [];
  if (data.notes) bits.push(nfmt(data.notes) + (data.notes === 1 ? " note" : " notes"));
  if (data.links) bits.push(nfmt(data.links) + " links");
  const areas = areaCount(data);
  if (areas) bits.push(nfmt(areas) + " areas");
  $("#vh-facts").textContent = bits.join("  ·  ");
  if (!$("#vh-brief").dataset.written) renderTopicLine();
}

// The fallback sentence. The labels are the vault's own words and carry the
// content colour; everything around them is chrome, which is what keeps a
// label like "mystery · sophia" readable inside a sentence that also joins
// with punctuation.
function renderTopicLine() {
  const el = $("#vh-brief");
  el.replaceChildren();
  const t = vaultTopics.slice(0, 3);
  if (!t.length) return;
  el.appendChild(document.createTextNode("Densest around "));
  t.forEach((label, i) => {
    if (i) el.appendChild(document.createTextNode(i === t.length - 1 ? " and " : ", "));
    el.appendChild(mkEl("b", "", label));
  });
  el.appendChild(document.createTextNode("."));
}

async function loadVaultBrief() {
  try {
    const d = await (await fetch("/vault_brief")).json();
    if (!d.enabled || !d.text) return; // off, or the worker had nothing to say
    const el = $("#vh-brief");
    el.dataset.written = "1";
    el.textContent = d.text;
  } catch { /* the counted line already answered the question */ }
}

// Tree click routing follows the active view: in explore's map mode a click
// roots the radial map on the note; otherwise it opens the note drawer (which
// also mirrors focus into the graph iframe via focusGraphNode).
$("#tree").addEventListener("click", (e) => {
  // The pin sits inside the row, so it is tested first: otherwise the row's own
  // click wins and pinning a note would also open it, which is the opposite of
  // what a pin is for (naming it without going there).
  const pin = e.target.closest(".tree-pin");
  if (pin) { togglePin(pin.dataset.pin); return; }
  const leaf = e.target.closest(".tree-note");
  if (!leaf) return;
  const path = leaf.dataset.id;
  if (activeTab === "graph" && graphMode === "map") rootMap(path);
  else openNote(path);
});

// --- changes (what this session did to the vault) ----------------------------
// The list is the server's, not the transcript's: it survives a reload, folds
// five writes to one note into one row, and empties a row when /undo puts the
// bytes back. A row opens the drawer on the diff, which is the only place in the
// app where you can read what actually changed rather than what was claimed.
const changedPaths = new Set();
const KIND_MARK = { created: "+", deleted: "−", moved: "→", modified: "±" };

// A turn-end refresh is enough for a one-note write, and wrong for a run: an
// injector writes for minutes before its tool returns, so the notes were on disk
// — Obsidian showing them — while this list stayed empty and read as broken.
// Every phase bumps it, throttled: a run emits one every few hundred ms, and
// /changes re-reads each tracked note off disk. The trailing edge is covered by
// the turn-end call, so a leading-edge throttle drops nothing.
let lastBump = 0;
function bumpChanges() {
  const now = performance.now();
  if (now - lastBump < 2000) return;
  lastBump = now;
  loadChanges();
}

async function loadChanges() {
  let rows = [];
  try {
    rows = await (await fetch("/changes")).json();
  } catch { return; } // ambient, not an errand: a failed poll says nothing
  changedPaths.clear();
  const box = $("#changes");
  box.innerHTML = "";
  for (const r of rows) {
    changedPaths.add(r.path);
    const row = mkEl("div", "chg-row " + r.kind);
    row.dataset.path = r.path;
    row.title = r.from ? `${r.from} → ${r.path}` : r.path;
    row.appendChild(mkEl("span", "chg-mark", KIND_MARK[r.kind] || "±"));
    row.appendChild(mkEl("span", "chg-name", r.name));
    const tally = mkEl("span", "chg-tally");
    if (r.added) tally.appendChild(mkEl("span", "chg-add", "+" + r.added));
    if (r.removed) tally.appendChild(mkEl("span", "chg-del", "−" + r.removed));
    if (!r.added && !r.removed) tally.appendChild(mkEl("span", "chg-quiet", r.kind));
    row.appendChild(tally);
    box.appendChild(row);
  }
  $("#side-changes").hidden = !rows.length;
  $("#changes-count").textContent = rows.length || "";
  stampWriteTallies(rows);
  applySidebarFilter();
  syncDrawerMode(); // the diff tab may have just become available for the open note
}

// Every write card in the transcript, including ones a page reload replayed,
// takes its tally from the payload above rather than counting anything itself.
function stampWriteTallies(rows) {
  // The two halves are keyed differently and have to be made to meet. A card is
  // stamped with the tool's own argument — silica_patch_note(name:"Photosynthesis")
  // — while /changes reports the path the tool resolved that to,
  // "Biology/Photosynthesis.md". Exact path first, then the same path without
  // .md, then the bare basename but only when it is unambiguous across the
  // payload. Without this the tally never filled and every card click fell
  // through openDiff's baseline lookup to openNote, so the diff the card exists
  // to open was never the thing that opened.
  const by = new Map(), byBase = new Map();
  for (const r of rows) {
    const stem = r.path.replace(/\.md$/, "");
    by.set(r.path, r);
    by.set(stem, r);
    const base = stem.split("/").pop();
    byBase.set(base, byBase.has(base) ? null : r); // null = ambiguous, never used
  }
  for (const el of document.querySelectorAll(".wc-tally")) {
    const key = el.dataset.for;
    const r = by.get(key) || byBase.get(key) || null;
    el.textContent = "";
    if (!r) continue;
    // Re-point the card at the resolved path so its click opens the diff.
    const open = el.parentElement && el.parentElement.querySelector(".wc-open");
    if (open) open.dataset.path = r.path;
    if (r.added) el.appendChild(mkEl("span", "chg-add", "+" + r.added));
    if (r.removed) el.appendChild(mkEl("span", "chg-del", "−" + r.removed));
  }
}

$("#changes").addEventListener("click", (e) => {
  const row = e.target.closest(".chg-row");
  if (row) openDiff(row.dataset.path);
});

// One search box filters both the file tree and the chat history.
function applySidebarFilter() {
  const q = $("#side-search").value.trim().toLowerCase();
  // notes: substring on name or full path
  $("#tree").querySelectorAll(".tree-note").forEach((el) => {
    const off = !!q && !el.textContent.toLowerCase().includes(q) &&
                !(el.dataset.id || "").toLowerCase().includes(q);
    el.hidden = off;
    // The pin rides in a wrapper, so hiding the label alone would leave a bare
    // pin on an empty line. Both, because every count below still asks the
    // label whether it is hidden.
    const row = el.closest(".tree-row");
    if (row) row.hidden = off;
  });
  // folders: hide if nothing visible remains inside; reveal matches while searching
  $("#tree").querySelectorAll("details").forEach((d) => {
    const any = Array.from(d.querySelectorAll(".tree-note")).some((n) => !n.hidden);
    d.hidden = !!q && !any;
    if (q && any) d.open = true;
  });
  // pinned notes: same substring rule again, so a filtered rail is filtered
  // everywhere and not just below the fold
  $("#pinned").querySelectorAll(".pin-row").forEach((el) => {
    el.hidden = !!q && !el.textContent.toLowerCase().includes(q) &&
                !(el.dataset.path || "").toLowerCase().includes(q);
  });
  // changed notes: same substring rule as the tree, on the same names
  $("#changes").querySelectorAll(".chg-row").forEach((el) => {
    el.hidden = !!q && !el.textContent.toLowerCase().includes(q) &&
                !(el.dataset.path || "").toLowerCase().includes(q);
  });
  // sessions: substring on title; while searching, the expand cap is lifted
  $("#sessions").querySelectorAll(".session").forEach((el) => {
    el.hidden = (!!q && !el.textContent.toLowerCase().includes(q)) ||
                (!q && !sessionsExpanded && +el.dataset.idx >= SESSION_CAP);
  });
  $("#sessions-more").hidden = !!q || sessionsExpanded || sessionCount <= SESSION_CAP;

  // Say what the filter did. It used to empty both lists in silence, so a real
  // zero and a typo rendered pixel-identical and the only way to learn which was
  // to clear the field. It also only ever matched names, never note bodies, and
  // the placeholder never said so — hence the offer to ask the vault instead.
  const notes = $("#tree").querySelectorAll(".tree-note:not([hidden])").length;
  const chats = $("#sessions").querySelectorAll(".session:not([hidden])").length;
  const box = $("#side-search-status");
  box.hidden = !q;
  if (!q) return;
  if (notes || chats) {
    box.textContent = `${notes} note${notes === 1 ? "" : "s"}, ${chats} chat${chats === 1 ? "" : "s"} by name`;
    box.classList.remove("empty");
  } else {
    box.textContent = `no name matches "${$("#side-search").value.trim()}".`;
    box.classList.add("empty");
    const ask = document.createElement("button");
    ask.type = "button";
    ask.className = "linklike";
    ask.textContent = "search inside the notes";
    ask.addEventListener("click", () => {
      input.value = "/find " + $("#side-search").value.trim();
      input.focus();
      autoGrow(input);
    });
    box.appendChild(ask);
  }
}
$("#side-search").addEventListener("input", applySidebarFilter);

// collapse every open folder in the tree. The button lives inside the Files
// <summary>, so a click on it would also toggle the whole section. Cancelling
// on the summary itself is what reliably suppresses that: preventDefault from
// the button's own listener is order-dependent and stopPropagation does not
// consistently reach summary's activation behaviour.
$("#side-files").querySelector("summary").addEventListener("click", (e) => {
  if (e.target.closest("#tree-collapse")) e.preventDefault();
});

$("#tree-collapse").addEventListener("click", () => {
  for (const d of $("#tree").querySelectorAll("details[open]")) d.open = false;
});

// --- history (last sidebar section; capped, "expand" reveals the rest) -------
const SESSION_CAP = 8;
let sessionsExpanded = false;
let sessionCount = 0;

$("#sessions-more").addEventListener("click", () => {
  sessionsExpanded = true;
  applySidebarFilter();
});

async function loadSessions() {
  try {
    const r = await fetch("/sessions");
    const current = r.headers.get("X-Silica-Session") || "";
    const box = $("#sessions");
    box.innerHTML = "";
    const sessions = await r.json();
    sessionCount = sessions.length;
    sessions.forEach((s, i) => {
      const el = document.createElement("div");
      el.className = "session" + (s.id === current ? " active" : "");
      el.dataset.idx = i;
      el.textContent = s.title || "untitled";
      el.title = s.title || "";
      el.addEventListener("click", () => openSession(s.id));
      box.appendChild(el);
    });
    $("#sessions-more").textContent = "+ " + Math.max(0, sessionCount - SESSION_CAP) + " more";
    applySidebarFilter();
  } catch { notify("couldn't load chat history"); }
}

// The work panel projects whichever narration session is current, and the two
// places that switch it are the only ones that know. A load replays no beats
// onto the BUS — read_beats() walks the file — so a live subscriber cannot see
// the switch happen and has to be told.
const announceSession = () => document.dispatchEvent(new CustomEvent("silica:session"));

async function openSession(id) {
  if (streaming) return;
  try {
    const r = await fetch("/session/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    if (!r.ok) { notify("couldn't load that chat"); return; }
  } catch { notify("couldn't load that chat"); return; }
  document.querySelector('.tab[data-tab="chat"]').click(); // surface the loaded chat
  announceSession();
  await loadVault();
  loadSessions();
}

// --- tabs -------------------------------------------------------------------
// Rebuilding the graph (Louvain + cooccurrence labels) is not free — only do it
// when the vault might actually have changed (graphStale), not on every switch
// back into the tab. A turn that writes notes sets graphStale = true.
let graphStale = true;
// One vocabulary for "which of these is showing". `.active` paints it and this
// says it out loud, so a segmented control is not a state only a sighted user
// can read. `aria-pressed` and not `aria-selected`, because these are groups of
// buttons rather than an ARIA tablist with its roving tabindex — and because it
// is what the quick actions already carry, so the app keeps one convention
// instead of gaining a second.
function setActive(btn, on) {
  btn.classList.toggle("active", on);
  btn.setAttribute("aria-pressed", on ? "true" : "false");
}

// Switching tabs is a function, not only a click: a synthetic .click() bubbles
// to the document's outside-click handler, which closes the note drawer. Every
// caller that needs the drawer to survive the switch (the context drawer's
// concept cloud, its suggested rows) calls this instead.
function showTab(tab) {
  activeTab = tab;
  if (tab === "chat") closePeek(); // stream visible → card redundant
  $("#dock").hidden = tab !== "graph"; // ask-from-here lives on the graph + map only
  document.querySelectorAll(".tab").forEach((b) => setActive(b, b.dataset.tab === tab));
  $("#view-chat").classList.toggle("active", tab === "chat");
  $("#view-graph").classList.toggle("active", tab === "graph");
  $("#view-calendar").classList.toggle("active", tab === "calendar");
  $("#view-metrics").classList.toggle("active", tab === "metrics");
  // The rail's Layout rows name the surfaces of ONE view, so they leave with it.
  // Areas is the same fact about the same view: what the graph is coloured by.
  $("#side-layout").hidden = tab !== "graph";
  syncAreasRail();
  if (tab === "graph") setGraphMode(graphMode); // load the active mode's content
  if (tab === "calendar") loadCalendar();
  if (tab === "metrics") loadMetrics();
  // Deep-linkable views: the hash names the tab ("explore" is the label users
  // see for the graph view), so a pasted URL opens on the right screen.
  const slug = tab === "graph" ? "explore" : tab;
  if (location.hash !== "#" + slug) history.replaceState(null, "", "#" + slug);
  // The third column reads this: on metrics it stops narrating the run and
  // becomes the Report panel. It is a separate file and has no other way to
  // know which view is up.
  document.dispatchEvent(new CustomEvent("silica:view", { detail: tab }));
}
$(".tabs").addEventListener("click", (e) => {
  const tab = e.target.dataset.tab;
  if (tab) showTab(tab);
});
// #explore / #metrics / #chat on the URL select the tab, at load and on manual
// hash edits alike. Unknown hashes are left alone (note anchors, etc.).
function tabFromHash() {
  const slug = (location.hash || "").replace(/^#/, "");
  const tab = slug === "explore" ? "graph" : slug;
  if (["chat", "graph", "calendar", "metrics"].includes(tab) && tab !== activeTab) showTab(tab);
}
window.addEventListener("hashchange", tabFromHash);

// --- theme ------------------------------------------------------------------
// The palette itself is CSS, and the <head> script owns resolving "auto" — this
// is only the two things CSS cannot reach. The iframes get the resolved value
// on their URL rather than inheriting it, because they are separate documents
// with their own <head> script and no way to read this one's :root. Mermaid
// gets a repaint because a rendered diagram is baked SVG.
const liveTheme = () => document.documentElement.dataset.theme || "dark";

function applyThemePref(pref) {
  document.documentElement.dataset.themePref = pref;
  document.documentElement.__silicaPaintTheme?.();
}

// One watcher for both ways the theme moves: the settings row writes the
// preference, and the OS moving under an "auto" preference fires the <head>
// script's own listener. Both land on data-theme, so that is the only thing
// worth watching — and watching the result rather than the causes is what keeps
// this from needing a second call site every time a new one appears.
new MutationObserver(() => {
  repaintMermaid();
  graphStale = true;
  if (activeTab === "graph") {
    setGraphMode(graphMode);
    if (graphMode === "map" && mapRootedPath) rootMap(mapRootedPath);
  }
}).observe(document.documentElement, { attributeFilter: ["data-theme"] });

// --- explore tab: network graph | radial map ---------------------------------
// Two modes in one view, one toolbar. "graph" is one build (wikilink structure
// + semantic k-NN overlay, layers toggled in the frame's HUD); "map" is a radial
// map rooted on one note (/map), which needs a root, so it opens on a hub-picker
// landing. Each mode owns its iframe so switching back doesn't rebuild the graph.
let graphMode = "graph";
let mapRootedPath = null; // note the radial map is rooted on, or null → picker

// Show one mode: toggle which frame/picker is visible, and rebuild /graph only
// when the vault changed under us. Also the entry point when switching INTO the
// explore tab, so it must be idempotent.
function setGraphMode(m) {
  graphMode = m;
  syncRefreshCue(); // the offer belongs to the graph surface alone
  syncLayoutRail();
  // Only the graph has two renderers. On the other four the segment would be a
  // control with nothing to switch.
  $("#renderer-tabs").hidden = m !== "graph";
  const isMap = m === "map";
  // folders / areas / read render in-page. They take the whole pane, so both
  // iframes hide and the note search goes with them: it flies the graph camera
  // and roots the map, and neither means anything on a treemap or a matrix.
  const isShape = m in SHAPE_VIEWS;
  // path renders in the same pane as the three shape views and is NOT one of
  // them: it is rooted on a note, so it keeps the search the shape views hide.
  const isPath = m === "path";
  $("#shape-pane").hidden = !(isShape || isPath);
  $("#node-search-wrap").hidden = isShape;
  // The pane's top padding clears a toolbar carrying only the renderer segment.
  // path is the one pane surface that KEEPS the note search, which makes that
  // toolbar 67px tall, and at 1100px the ladder's own title landed underneath
  // it (measured 2026-08-22: title top 97, search bottom 104). The three shape
  // views hide the search and need no such reservation.
  $("#shape-pane").classList.toggle("with-bar", isPath);
  $("#graph-frame").hidden = isMap || isShape || isPath;
  $("#map-frame").hidden = !isMap || !mapRootedPath;
  $("#map-picker").hidden = !isMap || !!mapRootedPath;
  closeNodeResults();
  if (isShape || isPath) {
    $("#graph-loading").hidden = true;
    $("#map-loading").hidden = true;
    if (isPath) { drawPath(); $("#node-search").focus(); } else { drawShape(); }
    return;
  }
  $("#shape-loading").hidden = true;
  if (isMap) {
    $("#graph-loading").hidden = true;
    if (mapRootedPath) $("#map-loading").hidden = true;
    $("#node-search").focus();
  } else {
    $("#map-loading").hidden = true;
    if (graphStale) {
      $("#graph-loading").hidden = false;
      $("#graph-frame").src = "/graph?theme=" + liveTheme() + "&t=" + Date.now();
      graphStale = false;
      syncRefreshCue(); // …and this rebuild is what it was offering
    }
  }
}

$("#graph-refresh").addEventListener("click", () => {
  setGraphMode("graph"); // graphStale is already true, so this is the rebuild
});

$("#graph-bar").addEventListener("click", (e) => {
  // The renderer lives inside the frame (it owns the WebGL/canvas instance), so
  // this asks rather than sets. Nothing is painted here on the way out: the
  // frame answers with the mode it actually built, which is the only value that
  // cannot be a lie, and syncRenderer() paints that.
  const r = e.target.dataset.renderer;
  if (!r) return;
  const f = $("#graph-frame");
  if (f.contentWindow) f.contentWindow.postMessage({ type: "silica-set-renderer", mode: r }, "*");
});

// The frame states its renderer on every build and on every switch, embedded or
// not. Before the first answer the segment shows neither: an unanswered toolbar
// that guesses 3D is a toolbar that is wrong for as long as the graph takes to
// build.
function syncRenderer(mode) {
  document.querySelectorAll("#renderer-tabs button")
    .forEach((b) => setActive(b, b.dataset.renderer === mode));
}

// #graph-frame finishes loading only once the server is done building — drop the
// loader then and re-sync the focus dim state after a (re)load.
$("#graph-frame").addEventListener("load", () => {
  $("#graph-loading").hidden = true;
  replayGraphFocus(); // re-sync whatever is focused after a (re)load
  syncDrawerToViews(); // ditto for the drawer, which hides this frame's HUD
});
$("#map-frame").addEventListener("load", () => { $("#map-loading").hidden = true; });

// --- explore: three surfaces that are not link-space -------------------------
// graph and map both lay notes out by how they CONNECT. Three questions that
// shape cannot answer: where does a note sit, how do two areas couple as a
// whole, and in what order could this be read. One /shape load feeds all three.
let shapeData = null;
let shapeLoading = false;
let folderPrefix = []; // drill state for the containment view

async function loadShape() {
  if (shapeData || shapeLoading) return shapeData;
  shapeLoading = true;
  $("#shape-loading").hidden = false;
  try {
    const d = await (await fetch("/shape")).json();
    if (d.error) { notify("couldn't read the vault shape: " + d.error); return null; }
    shapeData = d;
    return d;
  } catch { notify("couldn't read the vault shape"); return null; }
  finally { shapeLoading = false; $("#shape-loading").hidden = true; }
}

// Squarified treemap. ~30 lines instead of d3-hierarchy: the vendored bundles
// are the graph renderers, and pulling a layout library in for one rect split
// would be the largest dependency on the page by a wide margin.
// Returns [{...item, x, y, w, h}] in the given rect.
function squarify(items, x, y, w, h) {
  const out = [];
  let rest = items.filter((i) => i.value > 0).sort((a, b) => b.value - a.value);
  while (rest.length) {
    const total = rest.reduce((s, i) => s + i.value, 0);
    const vertical = w < h;          // lay the next row along the shorter side
    const side = vertical ? w : h;
    // Grow the row while the worst aspect ratio in it keeps improving.
    let row = [], best = Infinity, sum = 0;
    for (const it of rest) {
      const trial = sum + it.value;
      const thickness = (trial / total) * (vertical ? h : w);
      const worst = Math.max(
        ...[...row, it].map((r) => {
          const len = (r.value / trial) * side;
          return Math.max(thickness / len, len / thickness);
        }));
      if (row.length && worst > best) break;
      row.push(it); sum = trial; best = worst;
    }
    const thickness = (sum / total) * (vertical ? h : w);
    let off = 0;
    for (const it of row) {
      const len = (it.value / sum) * side;
      out.push(vertical
        ? { ...it, x: x + off, y, w: len, h: thickness }
        : { ...it, x, y: y + off, w: thickness, h: len });
      off += len;
    }
    if (vertical) { y += thickness; h -= thickness; } else { x += thickness; w -= thickness; }
    rest = rest.slice(row.length);
  }
  return out;
}

// One level of the containment tree at `prefix`: immediate children, each with
// its note count and how much of it belongs to a single area.
function folderLevel(notes, prefix, real) {
  const pre = prefix.length ? prefix.join("/") + "/" : "";
  const kids = new Map();
  for (const n of notes) {
    if (!n.path.startsWith(pre)) continue;
    const tail = n.path.slice(pre.length);
    const cut = tail.indexOf("/");
    const name = cut === -1 ? tail : tail.slice(0, cut);
    let k = kids.get(name);
    if (!k) kids.set(name, k = { name, folder: cut !== -1, count: 0, areas: new Map() });
    k.count++;
    // Only multi-note areas count toward purity. A singleton carries a group id
    // like any other community, so counting it would let a folder of six
    // unrelated notes report six areas and a purity of 1/6 — a number about the
    // clustering's tail, not about the filing. `real` is the same set the
    // matrix draws, so the two surfaces agree on what an area is.
    if (real.has(n.area)) k.areas.set(n.area, (k.areas.get(n.area) || 0) + 1);
  }
  return [...kids.values()].map((k) => {
    const placed = [...k.areas.values()].reduce((s, v) => s + v, 0);
    const top = Math.max(0, ...k.areas.values());
    return { ...k, value: k.count, placed,
             purity: placed ? top / placed : null, spread: k.areas.size };
  });
}

// The containment view. Area is note count; the fill is IMPURITY, so a folder
// whose notes all belong to one area is nearly blank and one that mixes nine
// areas is solid. That direction is deliberate: the question this surface
// answers is where filing and meaning disagree, so the disagreements are the
// ones that should be loud. Colouring by area instead would need a 26-hue
// categorical palette, which is exactly what the viz tokens exist to avoid.
function renderFolders(s) {
  const pane = mkEl("div", "shape-body");
  const head = mkEl("div", "shape-head");
  const crumbs = mkEl("div", "fcrumb");
  const mk = (label, depth) => {
    const b = mkEl("button", "fcrumb-b", label);
    b.type = "button";
    b.addEventListener("click", () => { folderPrefix = folderPrefix.slice(0, depth); drawShape(); });
    return b;
  };
  crumbs.appendChild(mk("vault", 0));
  folderPrefix.forEach((p, i) => {
    crumbs.appendChild(mkEl("span", "fcrumb-sep", "/"));
    crumbs.appendChild(mk(p, i + 1));
  });
  head.appendChild(crumbs);
  head.appendChild(mkEl("span", "shape-sub",
    "area = notes · fill = how much the folder mixes areas · click a folder to descend"));
  pane.appendChild(head);

  const rows = folderLevel(s.notes, folderPrefix, new Set(s.areas.map((a) => a.id)));
  if (!rows.length) { pane.appendChild(mkEl("p", "mempty", "Nothing here.")); return pane; }

  const box = mkEl("div", "tmap");
  // A fixed viewBox with percentage-positioned tiles: the layout is computed in
  // an abstract 100x100 box and the container decides the pixels, so a resize
  // needs no relayout and no observer.
  for (const t of squarify(rows, 0, 0, 100, 100)) {
    const tile = mkEl("div", "tmap-tile" + (t.folder ? " folder" : ""));
    tile.style.cssText = `left:${t.x}%;top:${t.y}%;width:${t.w}%;height:${t.h}%`;
    const impurity = t.purity === null ? 0 : 1 - t.purity;
    tile.style.setProperty("--i", impurity.toFixed(3));
    const pur = t.purity === null ? "no area" : `${Math.round(t.purity * 100)}% one area`;
    tile.title = `${t.name} · ${t.count} notes · ${pur}`
      + (t.spread > 1 ? ` · spans ${t.spread} areas` : "");
    const lbl = mkEl("div", "tmap-lbl");
    lbl.appendChild(mkEl("span", "tmap-name", t.name));
    lbl.appendChild(mkEl("span", "tmap-n", nfmt(t.count)));
    tile.appendChild(lbl);
    if (t.folder) {
      tile.addEventListener("click", () => { folderPrefix = [...folderPrefix, t.name]; drawShape(); });
    } else {
      tile.dataset.path = (folderPrefix.length ? folderPrefix.join("/") + "/" : "") + t.name;
      tile.classList.add("clickable");
    }
    box.appendChild(tile);
  }
  pane.appendChild(box);

  const worst = rows.filter((r) => r.purity !== null && r.placed >= 3)
    .sort((a, b) => a.purity - b.purity)[0];
  pane.appendChild(mkEl("p", "mnote", worst
    ? `Most mixed here: ${worst.name}, ${Math.round(worst.purity * 100)}% in its biggest area across ${worst.spread}.`
    : "Too few placed notes here to read purity."));
  return pane;
}

// Area x area coupling. Every pair at once, where the metrics tab's gap list is
// a top-N: an absence is only readable against the pairs that are present, and
// a ranked list of the emptiest pairs cannot show that.
// The grid itself is `couplingMatrix`, which the metrics tab draws too: this
// surface owns the framing and the caveat, not the cells.
function renderAreas(s) {
  const pane = mkEl("div", "shape-body");
  const head = mkEl("div", "shape-head");
  head.appendChild(mkEl("strong", null, "Area coupling"));
  const pairs = s.areas.length * (s.areas.length - 1) / 2;
  let linked = 0;
  for (let i = 0; i < s.areas.length; i++) {
    for (let j = i + 1; j < s.areas.length; j++) if (s.matrix[i][j]) linked++;
  }
  head.appendChild(mkEl("span", "shape-sub",
    `${s.areas.length} areas · ${linked} of ${pairs} pairs share a link · diagonal is cohesion`));
  pane.appendChild(head);

  pane.appendChild(couplingMatrix(s.areas, s.matrix));
  pane.appendChild(mkEl("p", "mnote",
    `${nfmt(s.totals.singletons)} single-note areas are left out: each would be a row and a column `
    + "of zeroes, and 65 of them would bury the 26 that carry the vault."));
  return pane;
}

// A reading order, derived and not authored. The one surface here that is not a
// layout: it answers "where do I start and what next", which no arrangement of
// nodes in space can, because space has no order.
function renderReading(s) {
  const pane = mkEl("div", "shape-body read");
  const head = mkEl("div", "shape-head");
  const r = s.reading;
  head.appendChild(mkEl("strong", null, "A way through"));
  head.appendChild(mkEl("span", "shape-sub",
    `${r.stops.length} stops · areas biggest first, each hub then what it opens onto`));
  pane.appendChild(head);
  // The path is a roll on a reading measure, so on a wide screen it cannot fill
  // the pane and should not try. What the width is worth here is a way back:
  // 24 stops across 8 areas scroll past the top, and the rail is the only place
  // that can say where in the path you are. Rail and roll centre as one cluster.
  const rail = mkEl("nav", "rpath-rail");
  rail.setAttribute("aria-label", "areas in this path");
  pane.appendChild(rail);
  // The vault holds notes that share a name across folders, so a path can list
  // the same label twice for two different files. Where that happens the parent
  // folder rides along: two identical rows pointing at different notes is worse
  // than a longer label, and silently dropping one would be worse still.
  const seenLabel = new Map();
  for (const st of r.stops) seenLabel.set(st.label, (seenLabel.get(st.label) || 0) + 1);
  const ol = mkEl("ol", "rpath");
  let area = null;
  let ai = 0;
  let ordinal = 0;
  const marks = [];
  for (const stop of r.stops) {
    if (stop.area !== area) {
      area = stop.area;
      const id = "rp-a" + (ai++);
      const h = mkEl("li", "rpath-area", area);
      h.id = id;
      ol.appendChild(h);
      const link = mkEl("button", "rpath-rail-i", area);
      link.type = "button";
      link.title = area;
      link.addEventListener("click", () => h.scrollIntoView({ block: "start", behavior: "smooth" }));
      rail.appendChild(link);
      marks.push([h, link]);
    }
    const li = mkEl("li", "rpath-stop clickable");
    li.dataset.path = stop.path;
    // The number is not decoration on this one surface: the whole claim of a
    // reading path is its order, and without the ordinal the roll reads as a
    // grouped list of hubs, which is what the folders view already is.
    li.appendChild(mkEl("span", "rpath-n", String(++ordinal)));
    // The full path, not the parent folder: the notes that collide here are
    // forks of each other under `silica/`, so they share every segment except
    // the first, and one parent segment disambiguated nothing.
    const name = seenLabel.get(stop.label) > 1
      ? stop.path.replace(/\.md$/, "") : stop.label;
    const n = mkEl("span", "rpath-name", name);
    n.title = stop.path;
    li.appendChild(n);
    li.appendChild(mkEl("span", "rpath-why", stop.why));
    ol.appendChild(li);
  }
  pane.appendChild(ol);
  // The cut is stated rather than left to be inferred from a heading count: the
  // path stops at a readable length, and 18 unmentioned areas would make this
  // read as a tour of the whole vault.
  const cut = r.areas_total - r.areas_covered;
  pane.appendChild(mkEl("p", "mnote",
    "Derived from link structure alone, so it promises adjacency and not importance: "
    + "each stop is linked to something already read."
    + (cut > 0 ? ` ${nfmt(cut)} smaller areas are past the end of the path.` : "")));
  // Which area you are in, marked on the rail. Observed rather than measured on
  // scroll: this pane holds a few hundred rows and a scroll handler reading
  // offsets off all of them is the one thing that could make a list janky.
  if (marks.length) {
    const io = new IntersectionObserver((entries) => {
      for (const en of entries) {
        const pair = marks.find(([h]) => h === en.target);
        if (pair) pair[1].classList.toggle("here", en.isIntersecting);
      }
      // Nothing intersecting means every heading is above the fold: keep the
      // last one that was, so the rail never goes blank mid-area.
      if (!rail.querySelector(".here")) {
        const above = marks.filter(([h]) => h.getBoundingClientRect().top < 200).pop();
        if (above) above[1].classList.add("here");
      }
    }, { root: $("#shape-pane"), rootMargin: "-56px 0px -70% 0px" });
    for (const [h] of marks) io.observe(h);
  }
  return pane;
}

const SHAPE_VIEWS = { folders: renderFolders, areas: renderAreas, read: renderReading };

async function drawShape() {
  const pane = $("#shape-pane");
  const render = SHAPE_VIEWS[graphMode];
  if (!render) return;
  const s = shapeData || await loadShape();
  if (!s || !SHAPE_VIEWS[graphMode]) return; // mode may have changed while loading
  pane.innerHTML = "";
  pane.appendChild(SHAPE_VIEWS[graphMode](s));
}

// Same convention the metrics rows use: a shape row is a measurement ABOUT a
// note, not the note, so it points rather than names and fills the work panel.
$("#shape-pane").addEventListener("click", (e) => {
  // The path surface puts a re-root button INSIDE its chips, so the chip's own
  // "select this note" verb has to stand aside for it. Registration order on
  // one node cannot be relied on, so the generic handler declines rather than
  // the specific one shouting.
  if (e.target.closest("[data-root]")) return;
  const el = e.target.closest("[data-path]");
  // Off a row is a deselection, the same way the graph's background is one:
  // without it a row click is a one-way door and the panel keeps stating a note
  // you have stopped looking at.
  if (el && el.dataset.path) showNode({ path: el.dataset.path });
  else announceNode(null, null);
});

// --- metrics tab -------------------------------------------------------------
// Everything the L1 graph report measures, as cards. Charts are HTML tables:
// the bar IS the row, so the chart and its table view are one DOM — every value
// stays readable without a hover, and there is no chart/table toggle to keep in
// sync. Deliberately library-free; a bar is a div with a width.
//
// Palette (validated with the dataviz skill's checker against the --page
// surface, dark mode): magnitude uses the accent hue snapped into the dark
// lightness band, the energy chart is diverging accent↔amber over a neutral
// zero rule, and reliability tiers take a 3-step ordinal ramp of the accent.
// The chrome tokens themselves (--accent, --warn) sit above the band and stay
// where they are — they light chrome, not fills.
// Two depths, because the report's co-occurrence leg costs ~100x the rest
// (one expanded ranking per note). The tab opens at structural depth in a
// couple of seconds; the four PROPOSED signals are a second, explicit pass the
// reader asks for. E(vault) is labelled with the depth it was measured at —
// its `deficits` term is absent from the cheap pass, and on a real vault that
// term dominates, so an unlabelled number would compare two different things.
let metricsStale = true;
let metricsLoading = false;
let metricsDepth = "structural";

let metricsAbort = null;

async function loadMetrics(force = false, proposals = false) {
  if (metricsLoading) return;
  if (!metricsStale && !force && !(proposals && metricsDepth !== "full")) return;
  metricsLoading = true;
  const body = $("#metrics-body");
  const loading = $("#metrics-loading");
  // `.fl-msg`, not `div:last-child`: the overlay now ends with a cancel button,
  // so the positional selector matched nothing and this line threw before the
  // overlay was ever shown — the whole metrics load failed silently.
  loading.querySelector(".fl-msg").textContent = proposals
    ? "Running the co-occurrence delta over every note."
    : "Measuring the vault.";
  loading.hidden = false;
  body.style.opacity = body.childElementCount ? "0.45" : ""; // hold the last render, no skeleton flash
  // A full report takes ~20s behind an indeterminate spinner, with no way out
  // short of switching tab and hoping. The previous render stays underneath at
  // reduced opacity, so cancelling leaves you exactly where you were.
  metricsAbort = new AbortController();
  let data = null;
  try {
    data = await (await fetch("/metrics" + (proposals ? "?proposals=1" : ""),
                              { signal: metricsAbort.signal })).json();
  } catch (e) {
    // Reaching the endpoint is the ONLY failure this message can honestly
    // describe. renderMetrics used to sit inside this try, so a bug in the view
    // -- a chart reading a field the payload stopped carrying, say -- surfaced
    // as "couldn't measure the vault" over a body left half-drawn, which sends
    // every render bug to the wrong place to look and blames the vault for it.
    if (e.name !== "AbortError") notify("couldn't measure the vault");
  } finally {
    metricsAbort = null;
    metricsLoading = false;
    loading.hidden = true;
    body.style.opacity = "";
  }
  if (!data) return;                         // aborted, or the fetch failed above
  if (data.error) { notify("metrics unavailable: " + data.error); return; }
  metricsDepth = data.depth || "structural";
  // The Report panel is the same payload read a different way, so it rides this
  // fetch rather than making a second one that would recompute the vault. Sent
  // BEFORE the render: the two are separate surfaces over one reading, and a
  // chart that throws should not also blank the panel in the third column.
  document.dispatchEvent(new CustomEvent("silica:report", { detail: data }));
  try {
    renderMetrics(data);
  } catch (e) {
    // The vault was measured; this file could not draw it. Said out loud and
    // logged with the stack, because the half-built body left on screen is the
    // most misleading state this view has -- it reads as a complete report that
    // happens to be missing its last few cards.
    console.error("metrics: render failed", e);
    body.innerHTML = "";
    body.appendChild(mkEl("p", "mempty",
      "The vault was measured, but this view could not draw the result. "
      + "The console carries the error."));
    return;
  }
  metricsStale = false;
}

$("#metrics-refresh").addEventListener("click", () => loadMetrics(true, metricsDepth === "full"));

// A metrics row is a measurement about a note's PLACE in the vault, so it fills
// the work panel: same question, same answer, same surface as a graph node. It
// used to open the drawer's context mode instead, which is how one payload came
// to have two renderers that disagreed.
$("#metrics-body").addEventListener("click", (e) => {
  if (e.target.id === "metrics-proposals") { loadMetrics(true, true); return; }
  const row = e.target.closest("[data-path]");
  // Off a row is a deselection, the same way the graph's background is one, and
  // here it is what makes the report reachable again after a row has taken the
  // column over.
  if (row && row.dataset.path) showNode({ path: row.dataset.path });
  else announceNode(null, null);
});

const mkEl = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text; // labels are vault data — never innerHTML
  return n;
};

// A card: hairline compartment, micro-label title, optional one-line note.
// A div, not a <header>: the global `header>* {display:flex}` rule would flatten
// the title and its subtitle onto one line.
function mCard(title, sub) {
  const c = mkEl("section", "mcard");
  // The worklist at the top of the view points at cards by their title, so the
  // title has to be addressable and not just printed.
  c.dataset.card = title;
  const h = mkEl("div", "mcard-head");
  h.appendChild(mkEl("h3", null, title));
  if (sub) h.appendChild(mkEl("span", "mcard-sub", sub));
  c.appendChild(h);
  return c;
}

function mEmpty(card, msg) { card.appendChild(mkEl("p", "mempty", msg)); return card; }

// Magnitude chart: one hue, bars grow from a single baseline, value at the tip.
// rows: [{label, value, path?, title?, note?}]
function barChart(rows, { fmt = (v) => nfmt(v), tone = "accent" } = {}) {
  const max = Math.max(...rows.map((r) => Math.abs(r.value)), 1);
  const t = mkEl("table", "chart bars");
  const tb = mkEl("tbody");
  for (const r of rows) {
    const tr = mkEl("tr");
    if (r.path) { tr.dataset.path = r.path; tr.classList.add("clickable"); }
    if (r.title) tr.title = r.title;
    const th = mkEl("th", null, r.label);
    th.scope = "row";
    const td = mkEl("td", "cell");
    const bar = mkEl("div", "bar " + tone);
    bar.style.width = (Math.abs(r.value) / max) * 100 + "%";
    td.appendChild(bar);
    const val = mkEl("td", "num", fmt(r.value));
    tr.append(th, td, val);
    if (r.note !== undefined) tr.appendChild(mkEl("td", "num sub", r.note));
    tb.appendChild(tr);
  }
  t.appendChild(tb);
  return t;
}

// Waterfall: the right form for an additive decomposition. Each bar starts
// where the previous one ended, and the last bar IS the total — so the chart
// says what the hero number is made of, which a common-baseline chart cannot.
// It also survives the scale: E's terms span three orders of magnitude on a
// real vault, and off a shared baseline the small ones paint as 2px slivers
// that read "measured, came out flat". Stacked end to end they are steps.
// Cool arm lowers the total, warm arm raises it, neutral rule marks zero.
function waterfall(rows, total, { negLabel, posLabel }) {
  let cum = 0, lo = 0, hi = 0;
  const steps = rows.map((r) => {
    const start = cum;
    cum += r.value;
    lo = Math.min(lo, cum);
    hi = Math.max(hi, cum);
    return { label: r.label, value: r.value, start, end: cum };
  });
  hi = Math.max(hi, total);
  lo = Math.min(lo, total);
  const span = hi - lo || 1;
  const at = (v) => ((v - lo) / span) * 100;

  const wrap = mkEl("div", "diverge");
  wrap.appendChild(mLegend([
    { tone: "accent", label: negLabel },
    { tone: "amber", label: posLabel },
  ]));
  const t = mkEl("table", "chart bars waterfall");
  t.style.setProperty("--zero", at(0) + "%");
  const tb = mkEl("tbody");
  const addRow = (label, from, to, value, tone, cls) => {
    const tr = mkEl("tr", cls);
    const num = (value > 0 ? "+" : "") + value.toFixed(2);
    tr.title = `${label}: ${num}`;
    const th = mkEl("th", null, label);
    th.scope = "row";
    const td = mkEl("td", "cell");
    const bar = mkEl("div", "bar " + tone);
    bar.style.left = at(Math.min(from, to)) + "%";
    bar.style.width = (Math.abs(to - from) / span) * 100 + "%";
    td.appendChild(bar);
    tr.append(th, td, mkEl("td", "num", num));
    tb.appendChild(tr);
  };
  for (const s of steps) {
    addRow(s.label, s.start, s.end, s.value, s.value < 0 ? "accent" : "amber");
  }
  addRow("E(vault)", 0, total, total, "total", "total");
  t.appendChild(tb);
  wrap.appendChild(t);
  return wrap;
}

// Histogram: columns, because the x-axis is an ordered numeric scale and
// position has to read left-to-right. One hue — the bins are a single series
// ("notes"), and their order is already carried by position, so spending the
// identity channel on a ramp would re-encode what the axis says.
// Every column is capped at 24px and labeled on the cap, so the values are
// readable without hovering; the row beneath is the axis.
function histogram(bins) {
  const max = Math.max(...bins.map((b) => b.count), 1);
  // No wrapper: it used to be a div.hist, which is the tail chart's class and
  // carries `height: 44px` -- a 120px track, its caps and its axis inside a
  // 44px box, overflowing the card and painting over the note under it.
  const plot = mkEl("div", "hist-plot");
  for (const b of bins) {
    const col = mkEl("div", "hist-col");
    col.title = `degree ${b.label}: ${nfmt(b.count)} notes`;
    // A zero bin gets a labeled slot but no mark: painting a stub would say
    // "small", and the reading here is "none".
    col.appendChild(mkEl("div", "hist-cap", b.count ? nfmt(b.count) : ""));
    // The track is the only fixed-height box, so the bar's percentage resolves
    // against the plot area alone and the cap/tick bands sit outside it — a
    // column chart whose fixed height swallowed its own axis labels would make
    // the card grow a nested scrollbar.
    const track = mkEl("div", "hist-track");
    const bar = mkEl("div", "hist-bar" + (b.count ? "" : " empty"));
    bar.style.height = (b.count / max) * 100 + "%";
    track.appendChild(bar);
    col.appendChild(track);
    col.appendChild(mkEl("div", "hist-tick", b.label));
    plot.appendChild(col);
  }
  return plot;
}

// A legend is always present for two or more series — identity never rests on
// color alone. Single-series charts get none; their title already names them.
//
// `note` puts the segment's own figure on the key. A stacked bar shows shares
// and a reader who wants the count had to find it again in a table underneath,
// which is the same three numbers rendered twice — and two renderings of one
// measurement are two things that can disagree after an edit.
function mLegend(items) {
  const l = mkEl("div", "mlegend");
  for (const it of items) {
    const row = mkEl("span", "mlegend-item");
    row.appendChild(mkEl("i", "swatch " + it.tone));
    row.appendChild(mkEl("span", null, it.label));
    if (it.note) row.appendChild(mkEl("span", "mlegend-n", it.note));
    l.appendChild(row);
  }
  return l;
}

// Meter: one ratio against its limit. Fill and track are steps of one ramp.
function meter(done, total, label) {
  const w = mkEl("div", "meter-wrap");
  const track = mkEl("div", "meter");
  const fill = mkEl("div", "meter-fill");
  fill.style.width = (total ? (done / total) * 100 : 0) + "%";
  track.appendChild(fill);
  w.append(track, mkEl("div", "meter-lbl", label));
  return w;
}

// Ordinal part-to-whole: one stacked bar, steps of a single hue in rank order,
// 2px surface gaps doing the separating (never a stroke around a segment).
function stackedBar(segs) {
  const total = segs.reduce((s, x) => s + x.value, 0) || 1;
  const bar = mkEl("div", "stack");
  for (const s of segs) {
    if (!s.value) continue;
    const seg = mkEl("div", "stack-seg " + s.tone);
    seg.style.width = (s.value / total) * 100 + "%";
    seg.title = s.label + ": " + nfmt(s.value);
    bar.appendChild(seg);
  }
  return bar;
}

// Session × area matrix. A time axis is the wrong form for this data and the
// vault says so: the claim clocks land on a handful of days inside a couple of
// months, with the odd straggler years back, so a linear date axis spends its
// width on the gap and smears everything that matters into one column. Dropping
// the duration and keeping the ordering leaves what actually varies — which
// areas recur session after session, and which never come up at all.
// Cells carry the count as text, not just as intensity: the tab's rule is that
// every value reads without a hover, and colour here is the second encoding.
function sessionMatrix(s) {
  const wrap = mkEl("div", "smx-scroll");
  const g = mkEl("div", "smx");
  g.style.setProperty("--cols", s.areas.length);
  const max = Math.max(...s.days.map((d) => Math.max(...Object.values(d.cells), 0)), 1);

  g.appendChild(mkEl("div", "smx-corner"));
  for (const a of s.areas) {
    const h = mkEl("div", "smx-col", a.label);
    h.title = `${a.label} · ${a.total} claims across ${s.days.length} sessions`;
    g.appendChild(h);
  }
  g.appendChild(mkEl("div", "smx-col smx-tot", "total"));

  for (const d of s.days) {
    const lbl = mkEl("div", "smx-row", d.date);
    lbl.title = `${d.date}: ${d.notes} claims`;
    g.appendChild(lbl);
    for (const a of s.areas) {
      const n = d.cells[a.id] || 0;
      // An empty cell gets its slot and no mark: painting a stub would say
      // "a little", and the reading is "that area saw nothing that day".
      const c = mkEl("div", "smx-cell" + (n ? "" : " empty"), n ? String(n) : "");
      if (n) {
        // sqrt, so a session of 1 stays visible next to one of 12 instead of
        // resolving to a tint indistinguishable from empty.
        c.style.setProperty("--i", Math.sqrt(n / max).toFixed(3));
        c.title = `${d.date} · ${a.label}: ${n}`;
        if (a.path) { c.dataset.path = a.path; c.classList.add("clickable"); }
      }
      g.appendChild(c);
    }
    g.appendChild(mkEl("div", "smx-cell smx-tot", String(d.notes)));
  }
  wrap.appendChild(g);
  return wrap;
}

// --- the four forms this view had no way to draw -----------------------------
// Every chart above answers a question about ONE reading of the vault. These
// four answer questions a single reading cannot hold: which way a count is
// going, how two measurements sit against each other, which areas touch, and
// how the vault's mass is distributed. Each is inline SVG or CSS boxes for the
// same reason `squarify` is thirty lines rather than d3-hierarchy: the vendored
// graph bundles are the only libraries on this page and a layout dependency for
// a polyline would be the largest of them.

const SVGNS = "http://www.w3.org/2000/svg";
const svgEl = (name, attrs) => {
  const el = document.createElementNS(SVGNS, name);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
};

// One series, its own scale, no axis. A sparkline is a direction, not a
// measurement: the number beside it is the measurement, which is why every
// caller here prints one.
//
// Two points is the floor. A single reading drawn as a line would be a flat
// mark, and a flat mark is what an unchanged series looks like — the reader
// cannot tell "never measured twice" from "measured, holding steady", and only
// one of those is true of a vault nobody has run a report on yet. Callers get
// null and leave the slot empty, the same way an empty histogram bin gets its
// label and no bar.
function sparkline(values, { w = 84, h = 22, title = "" } = {}) {
  const pts = values.filter((v) => typeof v === "number" && isFinite(v));
  if (pts.length < 2) return null;
  const lo = Math.min(...pts), hi = Math.max(...pts);
  // A flat series has no range to scale against. Centred rather than pinned to
  // an edge: at y=0 it would read as a maximum and at y=h as a floor, and it is
  // neither. `span` is the divisor everywhere below, so it can never be 0.
  const span = hi - lo || 1;
  const flat = hi === lo;
  const x = (i) => (i / (pts.length - 1)) * w;
  const y = (v) => (flat ? h / 2 : h - ((v - lo) / span) * h);
  // 1px of inset top and bottom: a stroke centred on y=0 renders half outside
  // the box and the peak of every rising series is clipped to a hairline.
  const pad = 1.5;
  const yi = (v) => pad + (y(v) / h) * (h - pad * 2);

  // Rendered at its viewBox size, so nothing scales: a stretched sparkline
  // draws its head dot as an ellipse and its stroke as a wedge, and both read
  // as data. Callers pick the box; the CSS only stops it overflowing a narrow
  // one, it never stretches it.
  const svg = svgEl("svg", {
    class: "spark", viewBox: `0 0 ${w} ${h}`, width: w, height: h,
    "aria-hidden": "true", focusable: "false",
  });
  const d = pts.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(2)},${yi(v).toFixed(2)}`).join("");
  svg.appendChild(svgEl("path", { class: "spark-l", d }));
  // The head, so "where it ended" reads without counting points from the left.
  svg.appendChild(svgEl("circle", {
    class: "spark-h", cx: x(pts.length - 1).toFixed(2), cy: yi(pts[pts.length - 1]).toFixed(2), r: 1.9,
  }));
  if (title) {
    const t = svgEl("title");
    t.textContent = title;
    svg.appendChild(t);
  }
  return svg;
}

// A delta chip: the move, and whether the move was the good direction.
// `good` is the sign that means progress, because it differs per signal — one
// fewer orphan is progress, one fewer note is not — and a single hardcoded
// direction would paint half the band the wrong colour.
function deltaChip(now, then, good, fmt = nfmt) {
  if (typeof then !== "number" || typeof now !== "number") return null;
  const d = now - then;
  // Formatted through the caller's own formatter, or a 0.4 move on a ratio
  // rounds to "0" and a tile that changed reads as one that held still.
  const mag = fmt(Math.abs(d));
  // And when the formatter rounds the move away entirely, it is not a move the
  // reader can see: "+0.0" claims a direction and then prints the figure that
  // denies it. No non-zero digit survived the rounding, so it reads as no
  // change -- which is what the value beside it already shows.
  const moved = /[1-9]/.test(String(mag));
  // good === 0 means the count is not progress in either direction -- more
  // areas is fragmentation or coverage depending on the vault, and painting a
  // guess green would be the view asserting something it cannot know.
  const tone = !moved || !good ? "" : (d > 0) === (good > 0) ? " up" : " down";
  const sign = !moved ? "±" : d > 0 ? "+" : "−";
  return mkEl("span", "dchip" + tone, sign + mag);
}

// Every chart that measures its own container registers its observer here, so
// a re-render can release them. ResizeObserver holds its targets strongly, and
// `renderMetrics` rebuilds the whole body on every recompute -- without this,
// each run would leave one live observation per chart pointing at a detached
// subtree, and the leak would grow with how often the tab is opened.
let chartObservers = [];

function releaseChartObservers() {
  for (const o of chartObservers) o.disconnect();
  chartObservers = [];
}

// Two continuous measurements at once, split at their own medians.
//
// The median and not a fixed threshold: these axes have no absolute scale -- a
// betweenness of 0.04 is high on one vault and unremarkable on another -- so the
// only honest split is the vault's own middle, and the quadrant caption then
// reads "busier than typical HERE".
//
// The svg is drawn at its container's measured width rather than scaled from a
// fixed viewBox. The cards here run from ~330px in a three-column layout to
// ~650px in one, so a fixed box would be scaled by 2x between them: one unit is
// one CSS pixel at every width instead, which keeps a 9px axis label 9px and
// stops the quadrant captions outgrowing the field they annotate. It also makes
// the hover ring's position a plain read of the point's own coordinates.
//
// Points are placed by value and drawn once. Hover is delegated from the svg and
// reads `data-i` off the target rather than binding two listeners per dot, and
// the ring is drawn over the field instead of restyling the hovered circle, so
// pointer movement never touches the field's DOM.
function scatter(points, {
  x, y, r, label, xLabel, yLabel, quadrants = [], height = 240, onPick,
} = {}) {
  const wrap = mkEl("div", "sc-wrap");
  const data = points.filter((p) => isFinite(x(p)) && isFinite(y(p)));
  if (data.length < 2) {
    wrap.appendChild(mkEl("p", "mempty", "Too few measured notes to plot."));
    return wrap;
  }
  const H = height;
  const padL = 30, padR = 12, padT = 14, padB = 24;
  const plotH = H - padT - padB;
  const xs = data.map(x), ys = data.map(y);
  const xMax = Math.max(...xs) || 1, yMax = Math.max(...ys) || 1;
  const med = (vals) => {
    const s = [...vals].sort((a, b) => a - b), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };
  const rMax = r ? Math.max(...data.map(r), 1) : 1;
  const sr = (p) => (r ? 1.6 + Math.sqrt(r(p) / rMax) * 4.2 : 2.6);
  // 4% of headroom, so a point at the maximum sits inside the frame rather than
  // half-outside it on the axis line.
  const sy = (v) => padT + (1 - v / (yMax * 1.04)) * plotH;
  const midY = sy(med(ys));

  const tip = mkEl("div", "sc-tip");
  tip.hidden = true;
  // Rebuilt on a width change rather than patched: every x coordinate is a
  // function of the width, so a partial update would have to touch every node
  // anyway, and a fresh subtree cannot leave a stale one behind.
  let sx = null, midX = null, W = 0;

  function draw(width) {
    W = Math.max(240, Math.round(width));
    const plotW = W - padL - padR;
    sx = (v) => padL + (v / (xMax * 1.04)) * plotW;
    midX = sx(med(xs));

    const svg = svgEl("svg", {
      class: "sc", viewBox: `0 0 ${W} ${H}`, width: W, height: H,
      role: "img", "aria-label": `${yLabel} against ${xLabel}, one dot per note`,
    });
    // One quadrant washed, not four coloured panels: the dots are the content,
    // and a tint behind all four would spend the visual budget re-saying what
    // the two dashed guides already say.
    for (const q of quadrants) {
      if (!q.tint) continue;
      svg.appendChild(svgEl("rect", {
        class: "sc-tint",
        x: q.at[0] === "r" ? midX : padL, y: q.at[1] === "t" ? padT : midY,
        width: (q.at[0] === "r" ? W - padR - midX : midX - padL).toFixed(2),
        height: (q.at[1] === "t" ? midY - padT : H - padB - midY).toFixed(2),
      }));
    }
    for (const [x1, y1, x2, y2, cls] of [
      [padL, padT, padL, H - padB, "sc-ax"], [padL, H - padB, W - padR, H - padB, "sc-ax"],
      [midX, padT, midX, H - padB, "sc-med"], [padL, midY, W - padR, midY, "sc-med"],
    ]) svg.appendChild(svgEl("line", { class: cls, x1, y1, x2, y2 }));

    const field = svgEl("g", { class: "sc-field" });
    data.forEach((p, i) => {
      field.appendChild(svgEl("circle", {
        class: "sc-dot", "data-i": i,
        cx: sx(x(p)).toFixed(2), cy: sy(y(p)).toFixed(2), r: sr(p).toFixed(2),
      }));
    });
    svg.appendChild(field);
    const ring = svgEl("circle", { class: "sc-ring", r: 0, cx: 0, cy: 0 });
    svg.appendChild(ring);

    for (const q of quadrants) {
      const t = svgEl("text", {
        class: "sc-q",
        x: q.at[0] === "r" ? W - padR - 5 : padL + 5,
        y: q.at[1] === "t" ? padT + 10 : H - padB - 5,
        "text-anchor": q.at[0] === "r" ? "end" : "start",
      });
      t.textContent = q.label;
      svg.appendChild(t);
    }
    const ax = (attrs, text) => {
      const t = svgEl("text", { class: "sc-al", ...attrs });
      t.textContent = text;
      svg.appendChild(t);
    };
    ax({ x: padL + plotW / 2, y: H - 4, "text-anchor": "middle" }, xLabel);
    ax({ x: 9, y: padT + plotH / 2, "text-anchor": "middle",
         transform: `rotate(-90 9 ${padT + plotH / 2})` }, yLabel);

    const at = (t) => {
      const raw = t && t.getAttribute && t.getAttribute("data-i");
      return raw == null ? null : data[Number(raw)];
    };
    svg.addEventListener("mouseover", (e) => {
      const p = at(e.target);
      if (!p) return;
      ring.setAttribute("cx", sx(x(p)).toFixed(2));
      ring.setAttribute("cy", sy(y(p)).toFixed(2));
      ring.setAttribute("r", (sr(p) * 1.9).toFixed(2));
      tip.textContent = label(p);
      tip.hidden = false;
    });
    svg.addEventListener("mouseout", (e) => {
      if (at(e.target)) { ring.setAttribute("r", 0); tip.hidden = true; }
    });
    if (onPick) {
      svg.classList.add("pick");
      svg.addEventListener("click", (e) => { const p = at(e.target); if (p) onPick(p); });
    }
    const prev = wrap.querySelector("svg");
    if (prev) wrap.replaceChild(svg, prev);
    else wrap.insertBefore(svg, tip);
  }

  // The tooltip goes in first: `draw` inserts the svg before it, and a node
  // that is not yet a child is not a reference point insertBefore will accept.
  wrap.appendChild(tip);
  // A first paint at a plausible width, so the card has its height before the
  // observer fires and the section below it does not jump a frame later.
  draw(360);
  if (window.ResizeObserver) {
    let lastW = 0;
    const ro = new ResizeObserver((entries) => {
      const w = Math.round(entries[0].contentRect.width);
      // Width only. The tooltip appearing changes the wrapper's HEIGHT, and
      // redrawing on that would be a redraw per pointer entry -- and, since the
      // redraw itself can change the height, a loop.
      if (w > 0 && w !== lastW) { lastW = w; draw(w); }
    });
    ro.observe(wrap);
    chartObservers.push(ro);
  }
  return wrap;
}

// Area x area coupling, drawn once for both surfaces that show it.
//
// `/shape` owns the containment and coupling views; the metrics tab reaches the
// same grid from its structural-gaps row. Two renderers over one payload is two
// places to fix a contrast bug in, and the payloads are already identical by
// construction (`_area_matrix` mirrors `/shape`'s shape for exactly this).
//
// Two scales in one grid, so they get two treatments: off-diagonal cells ramp on
// the accent by inter-area link count, the diagonal is neutral and carries the
// area's own cohesion. A shared ramp would put a 0.11 cohesion and 11 links in
// the same ink and invite reading one as the other.
function couplingMatrix(areas, matrix, { dense = true } = {}) {
  let max = 1;
  for (let i = 0; i < areas.length; i++) {
    for (let j = 0; j < areas.length; j++) if (i !== j) max = Math.max(max, matrix[i][j]);
  }
  const wrap = mkEl("div", "smx-scroll");
  const g = mkEl("div", "smx" + (dense ? " dense" : ""));
  g.style.setProperty("--cols", areas.length);
  g.appendChild(mkEl("div", "smx-corner"));
  for (const a of areas) {
    const h = mkEl("div", "smx-col", a.label);
    h.title = `${a.label} · ${a.size} notes · cohesion ${a.cohesion}`;
    g.appendChild(h);
  }
  areas.forEach((a, i) => {
    const lbl = mkEl("div", "smx-row", a.label);
    lbl.title = `${a.label} · ${a.size} notes`;
    if (a.path) { lbl.dataset.path = a.path; lbl.classList.add("clickable"); }
    g.appendChild(lbl);
    areas.forEach((b, j) => {
      const v = matrix[i][j];
      if (i === j) {
        // ".55" saves two characters in a 22px cell, but only where there IS a
        // leading zero: slicing it off 1.00 printed ".00", so a perfectly
        // cohesive area read as the least cohesive one on the grid.
        const coh = a.cohesion >= 1 ? "1" : a.cohesion ? a.cohesion.toFixed(2).slice(1) : "";
        const c = mkEl("div", "smx-cell diag", coh);
        c.title = `${a.label}: cohesion ${a.cohesion}, ${a.intra} linked pairs inside ${a.size} notes`;
        g.appendChild(c);
        return;
      }
      const c = mkEl("div", "smx-cell" + (v ? "" : " empty"), v ? String(v) : "");
      if (v) {
        c.style.setProperty("--i", Math.sqrt(v / max).toFixed(3));
        c.title = `${a.label} ↔ ${b.label}: ${v} linked note pairs`;
      } else {
        c.title = `${a.label} ↮ ${b.label}: nothing links them`;
      }
      g.appendChild(c);
    });
  });
  wrap.appendChild(g);
  return wrap;
}

// Areas as area. A ranked bar list can only show the top fourteen before the
// rows become stubs, and the tail then has to be stated in a sentence — "94
// smaller areas hold the other 210 notes" — which is the one part of the
// distribution a reader cannot see. Every area gets a tile here, and the small
// ones being small IS the reading.
//
// The fill is a lens: with no signal selected it ramps on cohesion, so a
// loosely-held area reads pale; with one selected it ramps on that signal's
// share of the area, so the areas carrying the work light up. One field
// recoloured rather than one card per signal, which is what a metrics view
// turns into if every measurement gets its own chart.
function areaTreemap(areas, { lens = null, lensLabel = "" } = {}) {
  const rows = areas.map((a) => ({ ...a, value: a.size }));
  const box = mkEl("div", "tmap tmap-a");
  // Under a lens: the per-note SHARE, not the raw count. A 60-note area with 6
  // hits and a 6-note area with 6 hits paint the same on a count ramp, and only
  // one of them is a problem.
  const value = lens
    ? (a) => (a.size ? (lens[String(a.id)] || 0) / a.size : 0)
    : (a) => a.cohesion || 0;
  // Normalised to the loudest tile, then square-rooted -- the same treatment
  // and the same reason as the session matrix's cells.
  //
  // Both readings are bounded in [0, 1], and normalising alone does nothing to
  // cohesion because the top of that range is already occupied: a two-note area
  // whose notes link scores exactly 1.0 by construction, and a vault has
  // several. The areas that actually carry it sit between 0.05 and 0.2, so on a
  // linear ramp every tile a reader can see is painted at 2-9% alpha and the
  // field reads as blank while three 20px tiles burn solid. sqrt lifts the
  // populated end without reordering anything.
  //
  // The exact figure rides the tooltip, and cohesion is printed outright on the
  // coupling matrix's own diagonal; the fill is here to rank the tiles against
  // each other. 1e-4 is a divide-by-zero floor, not a threshold.
  const top = Math.max(...rows.map(value), 1e-4);
  for (const t of squarify(rows, 0, 0, 100, 100)) {
    const tile = mkEl("div", "tmap-tile clickable");
    tile.style.cssText = `left:${t.x}%;top:${t.y}%;width:${t.w}%;height:${t.h}%`;
    const hits = lens ? (lens[String(t.id)] || 0) : 0;
    tile.style.setProperty("--i", Math.sqrt(value(t) / top).toFixed(3));
    tile.title = lens
      ? `${t.label} · ${t.size} notes · ${hits} ${lensLabel}`
      : `${t.label} · ${t.size} notes · cohesion ${t.cohesion}`;
    if (t.path) tile.dataset.path = t.path;
    const lbl = mkEl("div", "tmap-lbl");
    lbl.appendChild(mkEl("span", "tmap-name", t.label));
    lbl.appendChild(mkEl("span", "tmap-n", lens && hits ? `${hits} / ${t.size}` : nfmt(t.size)));
    tile.appendChild(lbl);
    box.appendChild(tile);
  }
  return box;
}

// cols: [{key, label, num?}] — `num` right-aligns and tabularises the column.
function mTable(cols, rows) {
  const t = mkEl("table", "chart data");
  const thead = mkEl("thead");
  const hr = mkEl("tr");
  for (const c of cols) {
    const th = mkEl("th", c.num ? "num" : null, c.label);
    th.scope = "col";
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  const tb = mkEl("tbody");
  for (const r of rows) {
    const tr = mkEl("tr");
    if (r._path) { tr.dataset.path = r._path; tr.classList.add("clickable"); }
    if (r._title) tr.title = r._title;
    for (const c of cols) {
      // An action column builds its own cell: a measurement can carry the move
      // it suggests, and the row click keeps meaning "open that note".
      if (c.el) {
        const td = mkEl("td", "act");
        td.appendChild(c.el(r));
        tr.appendChild(td);
        continue;
      }
      const td = mkEl("td", c.num ? "num" : null, String(r[c.key] ?? ""));
      // Text columns are clamped to keep the card from growing a scrollbar; the
      // full value has to stay reachable, so it rides the cell's own tooltip.
      if (!c.num) td.title = td.textContent;
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  t.append(thead, tb);
  const wrap = mkEl("div", "tscroll");
  wrap.appendChild(t);
  return wrap;
}

const nfmt = (n) => (typeof n === "number" ? n.toLocaleString() : String(n));

// The one action a metrics row carries: a gap names two areas, and the move it
// suggests is a write, so it drafts the turn and the agent's gate still owns the
// write. The turn has to leave room for "there is no bridge here" — the gap is a
// shape in the link graph, not evidence that the two areas belong connected.
// stopPropagation: the row itself is clickable and means "open that note".
function mBridgeBtn(a, b) {
  const btn = mkEl("button", "m-do", "bridge");
  btn.type = "button";
  btn.title = "draft a note that connects these two areas";
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    prefillChat(
      'Nothing links "' + a + '" and "' + b + '", the hubs of two areas ' +
      "of the vault that stand apart. Read both, and if a real connection exists, " +
      "write the note that states it and link it to each side. If there isn't one, " +
      "say so instead of inventing it.");
  });
  return btn;
}

// A cut list must never read as the whole list.
function mMore(shown, total, noun) {
  return shown < total ? mkEl("p", "mnote", `showing ${shown} of ${nfmt(total)} ${noun}`) : null;
}

// --- the evidence pane -------------------------------------------------------
// One signal at a time: what it counts, the reading that says what the SHAPE of
// the list means, the rows themselves, and the action on the row that carries
// it. A report whose rows cannot be acted on is a report nobody runs twice.
//
// Every pane is built from the payload the report already returned. The only
// two fields added for this were the ones the unresolved tail could not be
// stated without: who references a missing target, and the distribution of
// references over ALL targets rather than the twelve that fit on screen.
let metricsSignal = "dangling"; // survives a recompute: you came back to check one number

function evHead(title, sub) {
  const h = mkEl("div", "ev-h");
  h.appendChild(mkEl("span", "ev-t", title));
  if (sub) h.appendChild(mkEl("span", "ev-n", sub));
  return h;
}

// The reading, not a caption. It says what the numbers mean together, which is
// the one thing a table of them cannot.
const evProse = (text) => mkEl("p", "ev-s", text);

// An action drafts the turn and leaves it in the composer: the vault is changed
// by a turn you sent, never by a metrics view deciding on your behalf.
function evAct(label, prompt, primary) {
  const b = mkEl("button", "mini" + (primary ? " go" : ""), label);
  b.type = "button";
  b.addEventListener("click", (e) => { e.stopPropagation(); prefillChat(prompt); });
  return b;
}

function evEmpty(box, text) {
  box.appendChild(mkEl("p", "mempty", text));
  return box;
}

function evDangling(d, T) {
  const box = mkEl("div", "ev");
  const targets = T.dangling_links || 0, refs = T.unresolved || 0;
  box.appendChild(evHead("Unresolved links",
    `${nfmt(targets)} targets · ${nfmt(refs)} references`));
  if (!d.dangling?.length) return evEmpty(box, "None. Every wikilink resolves.");

  const hist = d.dangling_hist || [];
  const once = (hist.find((h) => h.refs === 1) || {}).targets || 0;
  const carried = d.dangling_top_refs || 0;
  const topN = Math.min(20, targets);
  // Stated only when the concentration is real: on a vault whose targets are
  // all referenced once, "the top twenty carry most of it" would be a sentence
  // about nothing. The number of targets left in the tail is what decides
  // whether writing twenty notes is a move or a rounding error.
  if (targets > topN && refs && carried) {
    box.appendChild(evProse(
      `${Math.round((carried / refs) * 100)}% of the references point at ${nfmt(topN)} targets. `
      + `Writing those ${nfmt(topN)} notes closes ${nfmt(carried)} broken links; `
      + `${nfmt(once)} of the rest are referenced once each and are better left alone `
      + `than stubbed.`));
  }

  box.appendChild(mTable(
    [{ key: "target", label: "Target" }, { key: "refs", label: "Refs", num: true },
     { key: "src", label: "Referenced from" },
     { label: "", el: (r) => evAct("write it", writePrompt(r._row), true) }],
    d.dangling.map((r) => ({ target: r.target, refs: r.refs, _row: r, src: srcLabel(r) })),
  ));
  const more = mMore(d.dangling.length, targets, "targets");
  if (more) box.appendChild(more);

  if (hist.length > 1) box.appendChild(evTail(hist));
  if (targets > topN) {
    const foot = mkEl("div", "ev-act");
    foot.appendChild(evAct(`write the top ${topN}`, bulkWritePrompt(d, topN), true));
    box.appendChild(foot);
  }
  return box;
}

// The two bulk turns this view offers, named once because the Report panel in
// the third column offers the same two: a prompt stated twice is two turns that
// can drift apart, and the one people would notice is the one that writes.
function bulkWritePrompt(d, topN) {
  return `These ${topN} wikilink targets are the most referenced ones that do not exist yet: `
    + (d.dangling || []).slice(0, topN).map((r) => `"${r.target}"`).join(", ")
    + `. Write them, each grounded in the notes that already link to it, and stop `
    + `if a target turns out to be a typo rather than a missing note.`;
}

function bulkAutolinkPrompt(n) {
  return `${nfmt(n)} notes in the vault have nothing linking to them. Go through them, `
    + `and for each one add the wikilinks that genuinely belong, from the notes that `
    + `already cover the same ground. Skip the ones that have no real neighbour.`;
}
window.bulkWritePrompt = bulkWritePrompt;
window.bulkAutolinkPrompt = bulkAutolinkPrompt;

// Two names and a count, without the extension. Three full filenames run past
// sixty characters and take the width from the target column, which is the one
// the row is named by; the full set is what the write prompt carries.
function srcLabel(r) {
  const names = (r.from || []).map(noExt);
  const shown = names.slice(0, 2);
  const more = names.length - shown.length + (r.from_more || 0);
  return shown.join(", ") + (more ? ` +${more}` : "");
}

const noExt = (n) => String(n).replace(/\.md$/, "");

// The same sentence the ghost drawer and the explore panel send, so a target
// written from here and one written from either of those arrive as the same
// turn. It used to be a hand-kept copy of that sentence, which had already
// drifted: this one left the source names unquoted.
const writePrompt = (r) => ghostWritePrompt(r.target, r.from);

// The tail: how many targets are asked for once, twice, fourteen times. Linear
// height on purpose, and the caption names both ends -- the skew IS the reading
// ("431 asked for once" is what makes stubbing them the wrong move), and a log
// axis that needed a disclaimer would be a chart arguing with its own caption.
function evTail(hist) {
  const wrap = mkEl("div", "ev-tail");
  wrap.appendChild(mkEl("div", "ev-tl", "The tail: references per target"));
  const chart = mkEl("div", "hist");
  const top = Math.max(...hist.map((h) => h.targets));
  const by = new Map(hist.map((h) => [h.refs, h.targets]));
  // Every bin from 1 to the maximum, including the empty ones. The payload
  // carries only the bins that exist, and drawing those side by side would put
  // "9 refs" next to "12 refs" at even spacing, which hides the gap that is
  // half of what a tail looks like.
  for (let refs = hist[0].refs; refs <= hist[hist.length - 1].refs; refs++) {
    const n = by.get(refs) || 0;
    const bar = mkEl("i", n === top ? "hot" : null);
    bar.style.height = (n ? Math.max(1, Math.round((n / top) * 100)) : 0) + "%";
    bar.title = `${nfmt(n)} target${n === 1 ? "" : "s"} referenced `
      + `${refs} time${refs === 1 ? "" : "s"}`;
    chart.appendChild(bar);
  }
  wrap.appendChild(chart);
  const lo = hist[0], hi = hist[hist.length - 1];
  wrap.appendChild(mkEl("div", "ev-tc",
    `${lo.refs} ref · ${nfmt(lo.targets)} targets  →  ${hi.refs} refs · ${nfmt(hi.targets)} target`
    + (hi.targets === 1 ? "" : "s")));
  return wrap;
}

function evOrphans(d, T) {
  const box = mkEl("div", "ev");
  box.appendChild(evHead("Orphans", `${nfmt(T.orphans || 0)} notes`));
  if (!d.orphans?.length) return evEmpty(box, "None. Every note is reachable.");
  box.appendChild(evProse(
    "Nothing links to these. They may still link out: an orphan is a fact about "
    + "what points AT a note, which is what recall follows and what the graph "
    + "cannot reach."));
  box.appendChild(mTable(
    [{ key: "label", label: "Note" },
     { label: "", el: (r) => evAct("link it",
       `Read "${r.label}" and link it into the vault: find the notes it belongs beside `
       + `and add the wikilinks in both directions where they are genuine. Do not invent `
       + `a connection to place it.`) }],
    d.orphans.map((o) => ({ ...o, _path: o.path })),
  ));
  const more = mMore(d.orphans.length, T.orphans || 0, "orphans");
  if (more) box.appendChild(more);
  const foot = mkEl("div", "ev-act");
  foot.appendChild(evAct("autolink them all", bulkAutolinkPrompt(T.orphans || 0), true));
  box.appendChild(foot);
  return box;
}

function evLean(d, T) {
  const box = mkEl("div", "ev");
  box.appendChild(evHead("Lean notes", `${nfmt(T.lean_notes || 0)} notes`));
  if (!d.lean_notes?.length) return evEmpty(box, "None. Every note carries its topic.");
  const limit = d.lean_limit || 0;
  box.appendChild(evProse(
    "Too thin to carry their topic: a title and a line or two. They are not "
    + "wrong, they are unfinished, and they are what a recall answer has least "
    + "to quote from."
    + (limit ? ` A note is counted here when its body is under ${nfmt(limit)} `
        + "characters, frontmatter excluded." : "")));
  box.appendChild(mTable(
    // The measurement that put each row in this list, on the row. Without it
    // "too thin" is a verdict the pane asks you to take on trust, and a note
    // three characters under the limit is indistinguishable from an empty one
    // -- which is the difference between finishing a note and writing it.
    [{ key: "label", label: "Note" },
     { key: "chars", label: "Chars", num: true },
     { label: "", el: (r) => evAct("expand it",
       `The note "${r.label}" is too thin to carry its topic. Expand it from what the vault `
       + `already holds, and say what is still missing rather than padding it.`) }],
    d.lean_notes.map((x) => ({
      ...x, _path: x.path,
      // "empty" and not "0": a note with no body at all is a different job from
      // a short one, and the report's own triage already separates them.
      chars: x.chars ? nfmt(x.chars) : "empty",
      _title: x.chars
        ? `${nfmt(x.chars)} characters` + (limit ? ` · ${nfmt(limit - x.chars)} under the limit` : "")
        : "no body at all",
    })),
  ));
  const more = mMore(d.lean_notes.length, T.lean_notes || 0, "notes");
  if (more) box.appendChild(more);
  return box;
}

function evGaps(d, T) {
  const box = mkEl("div", "ev");
  box.appendChild(evHead("Structural gaps", `${nfmt(T.structural_gaps || 0)} area pairs`));
  if (!d.gaps?.length) return evEmpty(box, "No gaps measured.");
  // Sizes, not the absent-link fraction: that fraction reads 99.7-100% on every
  // row of a real vault, so it cannot explain why row 1 outranks row 20. Size ×
  // size ÷ (1 + links) is the actual ranking, and with both sizes on the row
  // the order is readable instead of asserted.
  box.appendChild(evProse(
    "Two well-formed areas with almost nothing between them. Ranked by "
    + "size × size ÷ (1 + links), so the pair at the top is the one where the "
    + "most notes stand apart."));
  // The ranking is a top-N of the emptiest pairs, and a list of absences cannot
  // show what they are absences against: twelve rows reading "0 links" look the
  // same on a vault where everything couples and on one where nothing does. The
  // grid is every pair at once, so a hole is legible as a hole. Same renderer
  // and same payload shape as `/shape`'s coupling view, which is the surface
  // that owns the framing; here it is evidence for one worklist row.
  if (d.area_matrix) {
    box.appendChild(couplingMatrix(d.area_matrix.areas, d.area_matrix.matrix));
    box.appendChild(mkEl("p", "mnote",
      "every pair of multi-note areas · a filled cell is linked note pairs, "
      + "the diagonal is the area's own cohesion, an empty cell is a hole"));
  }
  box.appendChild(mTable(
    [{ key: "pair", label: "Area hubs" }, { key: "sizes", label: "Notes", num: true },
     { key: "inter_edges", label: "Links", num: true },
     { label: "", el: (r) => mBridgeBtn(r._a, r._b) }],
    d.gaps.map((g) => ({
      pair: g.a + " ↮ " + g.b, sizes: `${g.size_a} × ${g.size_b}`,
      inter_edges: g.inter_edges, _path: g.a_path, _a: g.a, _b: g.b,
    })),
  ));
  return box;
}

// The deck's second fix, and the reason this pane is not a table: the view used
// to print fifteen rows of `113 · 0 · 114`. Three columns, one value each,
// fifteen times. A constant column is not data, so when every candidate carries
// the same idle count and none was ever missed, the pane states the one figure
// and the reading it supports instead of ranking notes by a score that is
// measuring the vault sitting still.
function evAttention(d, T, rows, measured) {
  const box = mkEl("div", "ev");
  box.appendChild(evHead("Attention", `${nfmt(rows.length)} candidates`));
  if (!rows.length) return evEmpty(box, "Nothing overdue.");

  const idles = rows.map((a) => a.days_idle || 0);
  const flat = idles.every((v) => v === idles[0]);
  const noMiss = rows.every((a) => !(a.misses || 0));
  if (flat && noMiss) {
    const fig = mkEl("div", "ev-fig");
    fig.appendChild(mkEl("div", "ev-figv", nfmt(idles[0])));
    const when = new Date(Date.now() - idles[0] * 86400000);
    fig.appendChild(mkEl("p", "ev-figc",
      `Days since any of the ${rows.length} was read. All ${rows.length} carry the same figure `
      + `and none was ever missed in recall, so the score is measuring the vault sitting still, `
      + `not any note in it.`
      + (idles[0] > 0 ? ` Last read ${when.toLocaleDateString(undefined,
          { day: "numeric", month: "long", year: "numeric" })}.` : "")));
    box.appendChild(fig);
    // The candidates themselves, which this pane used to withhold. The reading
    // above is about the SCORE, and it is correct: with idle flat and misses at
    // zero the ranking carries no signal. It is not a reason to answer "20" and
    // show nothing -- the worklist row says twenty notes are waiting and the
    // only question it raises is which twenty.
    //
    // No score column, because that is the column the reading just disowned.
    // Links instead: with the other two inputs constant the score reduces to a
    // function of degree, so links is both the one input that varies and the
    // reason the rows are in this order.
    // Every column that VARIES, and none that does not. Measured on this vault
    // all twenty candidates carry 0 links and 0 attempts, so a Links column
    // would be twenty zeroes: the defect this pane was built to remove,
    // reintroduced one column over. The check is on the data rather than on a
    // hardcoded column list, because which input is flat depends on the vault.
    const varies = (k) => new Set(rows.map((a) => a[k] || 0)).size > 1;
    const cols = [{ key: "label", label: "Note" }];
    for (const [k, lbl] of [["degree", "Links"], ["attempts", "Quizzed"]]) {
      if (varies(k)) cols.push({ key: k, label: lbl, num: true });
    }
    // And the order is only worth asserting where something can order them.
    // With every input identical the sort is a tie broken arbitrarily, and
    // "least linked first" would be a caption about nothing.
    box.appendChild(evProse(cols.length > 1
      ? "The candidates. Nothing was missed and every note is idle the same "
        + "number of days, so the score ranks them by what is left: the columns "
        + "below."
      : `The candidates. All ${nfmt(rows.length)} carry the same reading, `
        + `${nfmt(rows[0].degree || 0)} resolved `
        + (rows[0].degree === 1 ? "link" : "links") + " and "
        + (rows[0].attempts ? `${nfmt(rows[0].attempts)} recall attempts` : "no recall attempt")
        + ", so nothing separates them and the order below is arbitrary."));
    cols.push({ label: "", el: (r) => evAct("quiz me", `/quiz ${shellQuote(r._path)}`) });
    box.appendChild(mTable(cols, rows.map((a) => ({
      ...a, _path: a.path,
      _title: `${a.days_idle}d idle · never missed · ${a.degree} links`,
    }))));
    const more = mMore(rows.length, T.attention_candidates || 0, "candidates");
    if (more) box.appendChild(more);
    return box;
  }
  if (!measured) {
    return evEmpty(box, "No signal yet. Every candidate was touched today and none "
      + "has been quizzed, so idle days and recall misses are both zero and there is "
      + "nothing to rank.");
  }
  box.appendChild(evProse("Ranked by (idle + 1)(1 + misses) ÷ ((1 + links)(1 + correct)): "
    + "long unread, missed when it should have been recalled, and weakly linked."));
  // The score is a product of two measurements, and a product hides which of
  // them is doing the work: a note idle for 200 days and never missed scores
  // like one idle for 20 and missed ten times. Plotted apart they separate, and
  // the degenerate case this pane guards against above -- every candidate on
  // both floors -- is visible as a single stack on the origin rather than
  // asserted in prose.
  box.appendChild(scatter(rows, {
    x: (a) => a.days_idle || 0,
    y: (a) => a.misses || 0,
    r: (a) => (a.degree || 0) + 1,
    xLabel: "days idle", yLabel: "recall misses",
    label: (a) => `${a.label} · ${a.days_idle}d idle · ${a.misses} missed `
      + `· ${a.degree} links · score ${a.score}`,
    quadrants: [
      { at: "rt", label: "idle and missed", tint: true },
      { at: "rb", label: "idle, never missed" },
      { at: "lt", label: "missed, read recently" },
    ],
    height: 210,
    onPick: (a) => showNode({ path: a.path }),
  }));
  box.appendChild(mkEl("p", "mnote",
    "dot size = links · split at this vault's own medians, since neither axis "
    + "has a scale that means anything across vaults"));
  // Idle and misses are the scatter's two axes now, so the columns that
  // reprinted them are gone: the table is what you act from, not a second
  // rendering of the chart above it.
  box.appendChild(mTable(
    [{ key: "label", label: "Note" }, { key: "score", label: "Score", num: true },
     { label: "", el: (r) => evAct("quiz me", `/quiz ${shellQuote(r._path)}`) }],
    rows.map((a) => ({ ...a, _path: a.path,
                       _title: `${a.days_idle}d idle · ${a.misses} missed · ${a.degree} links` })),
  ));
  return box;
}

function evList(title, sub, rows, cols, empty) {
  const box = mkEl("div", "ev");
  box.appendChild(evHead(title, sub));
  if (!rows?.length) return evEmpty(box, empty);
  box.appendChild(mTable(cols, rows));
  return box;
}

function evidence(key, d, T, attnRows, attnMeasured) {
  switch (key) {
    case "dangling": return evDangling(d, T);
    case "orphans": return evOrphans(d, T);
    case "lean": return evLean(d, T);
    case "gaps": return evGaps(d, T);
    case "attention": return evAttention(d, T, attnRows, attnMeasured);
    case "contested":
      return evList("Contested", `${nfmt(T.contested || 0)} notes`,
        (d.contested || []).map((c) => ({
          label: c.label, refs: (c.refs || []).join(", "), _path: c.path })),
        [{ key: "label", label: "Note" }, { key: "refs", label: "Conflicts with" }],
        "None. No note flags a conflict.");
    case "drift":
      return evList("Source drift", `${nfmt(T.source_drift || 0)} notes`,
        (d.source_drift || []).map((x) => ({ ...x, _path: x.path })),
        [{ key: "label", label: "Note" }, { key: "source", label: "Source" }],
        "None. Every note is level with its source.");
    case "sprawling":
      return evList("Sprawling notes", `${nfmt(T.sprawling || 0)} notes`,
        (d.sprawling || []).map((x) => ({ ...x, _path: x.path })),
        [{ key: "label", label: "Note" }, { key: "concepts", label: "Concepts", num: true },
         { key: "entropy", label: "Bits", num: true },
         { key: "flatness", label: "Flatness", num: true }],
        "None. Every note has a concept that dominates it.");
    case "deficits":
      return evList("Integration deficits", `${nfmt(T.integration_deficits || 0)} notes`,
        (d.deficits || []).map((x) => ({ ...x, _path: x.path })),
        [{ key: "label", label: "Note" }, { key: "concepts", label: "Concepts", num: true },
         { key: "degree", label: "Links", num: true }, { key: "score", label: "Score", num: true }],
        "None measured.");
    default: {
      const box = mkEl("div", "ev");
      box.appendChild(evHead("Nothing to act on", ""));
      return evEmpty(box, "Every signal the report measures is at zero. "
        + "The readings below still describe the vault's shape.");
    }
  }
}

// Which stored reading each worklist signal moves with, and which way is
// progress. `report_history.jsonl` keeps only the counts that mean the same
// thing at either depth (see history.py's SIGNALS), so four of the nine rows
// have no series at all. Those get no sparkline rather than a flat one: a flat
// line is what an unchanged signal looks like, and a reader cannot tell that
// from one the store has never seen twice.
//
// `good` is the sign that means progress. One fewer orphan is progress and one
// fewer note is not, so the direction is per row and never global.
const SIGNAL_SERIES = {
  dangling: { key: "dangling_links", good: -1 },
  lean: { key: "lean_notes", good: -1 },
  orphans: { key: "orphans", good: -1 },
  gaps: { key: "structural_gaps", good: -1 },
  contested: { key: "contested", good: -1 },
};

// The stored series for one signal, oldest first. Readings that predate the
// signal simply have no value for it and are skipped rather than read as zero:
// history.py appends a key the first time it is emitted, so an older line
// missing `contested` means "not yet measured", not "none".
function seriesOf(history, key) {
  return (history || []).map((r) => r.signals && r.signals[key])
    .filter((v) => typeof v === "number");
}

// The worklist's right-hand slot: where the count has been going, and by how
// much since the last report that differed. Null when the store cannot say.
function signalTrend(history, sigKey, { w = 84 } = {}) {
  const meta = SIGNAL_SERIES[sigKey];
  if (!meta) return null;
  const vals = seriesOf(history, meta.key);
  const line = sparkline(vals, { w, title: `${meta.key}: ${vals.join(" → ")}` });
  if (!line) return null;
  const slot = mkEl("span", "wl-b");
  slot.appendChild(line);
  const chip = deltaChip(vals[vals.length - 1], vals[vals.length - 2], meta.good);
  if (chip) slot.appendChild(chip);
  return slot;
}

function renderMetrics(d) {
  const body = $("#metrics-body");
  releaseChartObservers();   // the subtree about to be dropped owns some
  body.innerHTML = "";
  const T = d.totals || {};
  $("#metrics-stamp").textContent = d.generated_at ? d.generated_at.slice(0, 16).replace("T", " ") : "";

  const e = d.energy || { total: 0, terms: [] };
  const full = d.depth === "full";

  // --- the standing: what to do, and how the vault is doing -------------------
  // This view used to open on E(vault) at 52px — a number its own caption calls
  // a thermometer and not a target — over six rates, over sixteen cards of
  // equal weight. Three surfaces, none of them answering the question you open
  // a metrics tab with. E is now a reading inside Structure, where the terms
  // that make it up are, and the top of the page is the worklist plus the four
  // rates that put it in proportion.
  const links = T.links || 0, notes = T.notes || 0;
  const orphans = T.orphans || 0;
  const zeroBin = d.degree_histogram?.[0];
  const isolated = zeroBin && zeroBin.lo === 0 ? zeroBin.count : 0;
  const pct = (n, of) => (of ? Math.round((n / of) * 100) + "%" : "—");

  // Attention abstains when it has nothing to measure. Its score is
  // (idle+1)(1+misses) / ((1+degree)(1+correct)), so a vault whose notes were
  // all touched today and never quizzed scores every candidate at exactly 1
  // with both inputs on the floor — and the view then claimed N notes were
  // "idle and missed in recall" above a table of identical rows reading
  // 0 · 0 · 1. Both inputs at their floor is not a measurement. The rows are
  // ranked best-first, so if the top ones carry no signal, none of them do.
  const attnRows = d.attention || [];
  const attnMeasured = attnRows.some((a) => (a.days_idle || 0) > 0 || (a.misses || 0) > 0);
  const attnCount = attnRows.length && !attnMeasured ? 0 : (T.attention_candidates || 0);

  // --- the worklist is the navigation, the pane beside it is the evidence ----
  // This view used to be three walls of tables: 63 orphans and 532 unresolved
  // targets printed as rows, in cards you had to go find, with nothing to do
  // about any of them from the view that found them. The worklist on the left
  // is now what you steer with, the pane on the right is THAT row's evidence,
  // and the action sits on the row that carries it.
  const mv = mkEl("section", "mv");
  const left = mkEl("div", "mv-l");
  const right = mkEl("div", "mv-r");
  left.appendChild(mkEl("div", "mv-lh", "What needs attention"));

  // Every signal the report measures, zeroes included. The zeroes are the part
  // a top-four worklist could not say, and they are the reason the Maintenance
  // table existed: a row reading 0 is an answer, it just is not a destination.
  const signals = [
    ["dangling", T.dangling_links, "unresolved links", "wikilinks with no target"],
    ["lean", T.lean_notes, "lean notes", "too thin to carry their topic"],
    ["orphans", T.orphans, "orphans", "nothing links to them"],
    ["gaps", T.structural_gaps, "structural gaps", "areas that should connect and do not"],
    ["attention", attnCount, "waiting on attention", "idle and missed in recall"],
    // "—" not "0" when the leg that measures it never ran: a printed zero reads
    // as "measured, came out flat".
    ["deficits", full ? T.integration_deficits : null, "integration deficits",
     "concept-rich text, few wikilinks"],
    ["contested", T.contested, "contested", "frontmatter flags a conflict"],
    ["drift", T.source_drift, "source drift", "the source moved on without the note"],
    // V7. A row, not a card: it is a list of notes to act on, which is what the
    // worklist is for. Its own judge gate came back uninformative (the judge
    // would split 60-90% of RANDOM notes too), so the row says what it measured
    // - broad and flat - and never that the note is wrong.
    ["sprawling", full ? T.sprawling : null, "sprawling notes",
     "many concepts, none of them dominant"],
  ].sort((a, b) => (b[1] || 0) - (a[1] || 0));

  let firstRow = null;
  for (const [key, n, label, means] of signals) {
    const live = n > 0;
    // A row that goes nowhere stays a div: a button that does nothing is worse
    // than a line of text, and a zero has no evidence to show.
    const row = mkEl(live ? "button" : "div", "wl" + (live ? "" : " zero"));
    if (live) { row.type = "button"; row.dataset.sig = key; }
    row.appendChild(mkEl("span", "wl-q", n === null ? "—" : nfmt(n)));
    row.appendChild(mkEl("span", "wl-l", label));
    row.appendChild(mkEl("span", "wl-d", means));
    if (live) {
      // The slot that held a magnitude bar. n/max across these rows compared
      // counts with no shared unit -- unresolved links against lean notes
      // against area pairs -- so the bar ranked them on a scale nobody can
      // name, and the sort order plus the printed count already said which was
      // biggest. What a worklist actually raises is which way a count is going.
      const trend = signalTrend(d.history, key);
      if (trend) row.appendChild(trend);
      row.addEventListener("click", () => selectSignal(key));
      if (!firstRow) firstRow = key;
    }
    left.appendChild(row);
  }
  if (!signals.some(([, n]) => n > 0)) {
    left.appendChild(mkEl("p", "mempty", "Nothing is out of place. Every note is "
      + "reachable, every wikilink resolves."));
  }

  // The four rates that put those counts in proportion. Not counts: notes,
  // links, areas and broken links are stated in the top strip on every view,
  // and printing them again spends a row on nothing.
  // Three of these tiles and the worklist above them all count something about
  // links, and the panel used to print "36 orphans", "3% orphaned" and "46 no
  // link at all" side by side with nothing saying how the three relate. They
  // are not the same measure: an orphan has nothing pointing AT it and may link
  // out freely, "no link either way" has neither direction and includes the
  // staging notes orphans deliberately skips. Each tile carries its own
  // definition rather than leaving the reader to guess which number to believe.
  const kpi = mkEl("div", "mkpi");
  // A rate over time, derived per reading rather than from today's counts: a
  // links-per-note that fell because the vault gained notes is a different
  // event from one that fell because links were removed, and only the series
  // built reading-by-reading can tell them apart.
  const rate = (num, den) => (d.history || [])
    .map((r) => (r.signals && r.signals[den] ? r.signals[num] / r.signals[den] : null))
    .filter((v) => typeof v === "number");
  const d1 = (v) => v.toFixed(1);
  const pp = (v) => Math.round(v * 100) + "%";
  const tiles = [
    ["Links / note", notes ? d1(links / notes) : "0", false,
      "resolved wikilinks divided by notes",
      rate("links", "notes"), +1, d1],
    ["Orphaned", pct(orphans, notes), orphans > 0,
      `share of notes nothing links to (${nfmt(orphans)} of ${nfmt(notes)}). They may still link out.`,
      rate("orphans", "notes"), -1, pp],
    // No series: the reading comes from the degree histogram, which the report
    // store does not keep. An empty slot, never a flat line.
    ["No link either way", nfmt(isolated), isolated > 0,
      "notes with no link in and none out. Counted over every note, including the staging folders orphans skips.",
      [], +1, nfmt],
    ["Areas > 1 note", nfmt((d.clusters || []).filter((c) => c.size > 1).length), false,
      "structural areas holding more than a single note",
      seriesOf(d.history, "areas"), +1, nfmt],
  ];
  for (const [lbl, val, warn, why, vals, good, fmt] of tiles) {
    const st = mkEl("div", "stat");
    if (why) st.title = why;
    const head = mkEl("div", "stat-h");
    head.appendChild(mkEl("div", "val" + (warn ? " warn" : ""), nfmt(val)));
    const chip = deltaChip(vals[vals.length - 1], vals[vals.length - 2], good, fmt);
    if (chip) head.appendChild(chip);
    st.appendChild(head);
    st.appendChild(mkEl("div", "lbl", lbl));
    const line = sparkline(vals, { w: 96, h: 18, title: vals.map(fmt).join(" → ") });
    if (line) st.appendChild(line);
    kpi.appendChild(st);
  }
  left.appendChild(kpi);

  // Set by the Areas card once it exists. Selecting a worklist row recolours
  // that field as well as filling the pane, which is the difference between a
  // view with sixteen small charts and a view with one field read sixteen ways.
  // A no-op until then, because the pane is built and selected before the
  // sections below it are.
  let relens = () => {};

  // The pane holds one signal at a time, and the choice survives a recompute:
  // the number you were reading is exactly the one you came back to check.
  function selectSignal(key) {
    metricsSignal = key;
    for (const r of left.querySelectorAll(".wl[data-sig]"))
      r.classList.toggle("on", r.dataset.sig === key);
    right.textContent = "";
    right.appendChild(evidence(key, d, T, attnRows, attnMeasured));
    right.scrollTop = 0;
    relens(key);
  }
  // `preview ops` in the Report panel means "show me the rows this would touch",
  // and those rows are already a pane here. Re-exported on every render because
  // selectSignal closes over THIS render's panes; the previous one is detached.
  window.selectMetricSignal = selectSignal;
  mv.appendChild(left);
  mv.appendChild(right);
  body.appendChild(mv);
  selectSignal(signals.some(([k, n]) => k === metricsSignal && n > 0) ? metricsSignal : firstRow);

  // --- how the vault moved (graft: the series the store already kept) --------
  // Small multiples, one frame per signal, and deliberately NOT one chart with
  // nine lines. These counts do not share a domain: notes and links run to the
  // hundreds while gaps and contested run to single digits, so on one linear
  // axis the four that matter resolve to a flat line along the bottom -- the
  // same failure the waterfall exists to avoid for E's terms. Scaling each
  // series to its own range inside one frame would fix the flatness by lying
  // about magnitude: a signal that moved by 1 would swing as hard as one that
  // moved by 100.
  //
  // What the panels DO share is the x axis, because every panel is drawn from
  // the same readings in the same order. That is the comparison this band is
  // for: which signals moved together.
  //
  // The series is depth-independent by construction (history.py stores only
  // counts that mean the same thing at either depth), so a structural report
  // and a full one land on the same line without one of them reading as a
  // hundred deficits closed by nobody.
  const hist = d.history || [];
  const band = mkEl("section", "mband");
  const bh = mkEl("div", "mband-h");
  bh.appendChild(mkEl("span", "mband-t", "How the vault moved"));
  const series = [
    ["notes", "notes", +1, nfmt],
    ["links", "links", +1, nfmt],
    ["unresolved", "unresolved refs", -1, nfmt],
    ["dangling_links", "missing targets", -1, nfmt],
    ["orphans", "orphans", -1, nfmt],
    ["lean_notes", "lean notes", -1, nfmt],
    ["structural_gaps", "gaps", -1, nfmt],
    ["contested", "contested", 0, nfmt],
    ["areas", "areas", 0, nfmt],
  ].map(([key, label, good, fmt]) => [key, label, good, fmt, seriesOf(hist, key)]);
  const panels = series.filter(([, , , , vals]) => vals.length >= 2);
  // One reading is a state, not a movement -- but it is a real state, and the
  // figures in it are the points every future line will start from. So the band
  // still draws its frames, with the value and no line: what is missing here is
  // the second reading, not the measurement.
  const waiting = panels.length ? [] : series.filter(([, , , , v]) => v.length === 1);

  if (waiting.length) {
    bh.appendChild(mkEl("span", "mband-s",
      `${nfmt(hist.length)} reading stored · a direction needs two`));
    band.appendChild(bh);
    // Said in full rather than in a half-line, because the thing a reader has
    // to learn here is not "it is empty" but WHY it is empty and what fills it:
    // the store appends only when a count actually moves, so reopening this tab
    // on an unchanged vault will never produce the second point, and waiting is
    // the wrong response.
    band.appendChild(mkEl("p", "mband-p",
      "This band reads which counts are rising and which are falling, one frame "
      + "per signal. It needs two readings to draw a line, and a reading is only "
      + "stored when a count actually moves: recomputing an unchanged vault "
      + "deliberately does not manufacture one. Write a note, resolve a broken "
      + "link, run a nucleation: the next report becomes the second point and "
      + "these figures start carrying a direction."));
    const grid = mkEl("div", "mband-g");
    for (const [, label, , fmt, vals] of waiting) {
      const cell = mkEl("div", "mband-c waiting");
      const top = mkEl("div", "mband-ct");
      top.appendChild(mkEl("span", "mband-v", fmt(vals[0])));
      cell.appendChild(top);
      // A dotted rule where the line will go. Dotted because nothing in this
      // view draws a VALUE that way, so it cannot be misread as a flat series
      // -- which is exactly what a solid stroke here would say, and the one
      // thing the store cannot claim yet.
      cell.appendChild(mkEl("div", "mband-wait"));
      cell.appendChild(mkEl("div", "mband-l", label));
      cell.title = `${label}: ${fmt(vals[0])} at the only stored reading`;
      grid.appendChild(cell);
    }
    band.appendChild(grid);
  } else if (!panels.length) {
    // Nothing stored at all: no figures to show either, so there is no frame to
    // draw. A vault reaches this only before its first report is ever filed.
    bh.appendChild(mkEl("span", "mband-s", "no readings stored yet"));
    band.appendChild(bh);
    band.appendChild(mkEl("p", "mband-p",
      "The first report files one. From the second onward, this band shows which "
      + "counts are rising and which are falling."));
  } else {
    const first = hist[0].at || "", last = hist[hist.length - 1].at || "";
    bh.appendChild(mkEl("span", "mband-s",
      `${nfmt(hist.length)} readings · ${first.slice(0, 10)} to ${last.slice(0, 10)} `
      + "· one frame per signal, oldest left, each on its own scale"));
    band.appendChild(bh);
    const grid = mkEl("div", "mband-g");
    for (const [key, label, good, fmt, vals] of panels) {
      const cell = mkEl("div", "mband-c");
      const top = mkEl("div", "mband-ct");
      top.appendChild(mkEl("span", "mband-v", fmt(vals[vals.length - 1])));
      const chip = deltaChip(vals[vals.length - 1], vals[vals.length - 2], good, fmt);
      if (chip) top.appendChild(chip);
      cell.appendChild(top);
      const line = sparkline(vals, { w: 130, h: 26,
        title: `${label}: ${vals.map(fmt).join(" → ")}` });
      if (line) cell.appendChild(line);
      cell.appendChild(mkEl("div", "mband-l", label));
      // The whole series on the frame, not only the sparkline's own tooltip:
      // the number under the pointer has to be readable without hitting a 130px
      // path with a 1px stroke.
      cell.title = `${label}: ${vals.map(fmt).join(" → ")}`;
      grid.appendChild(cell);
    }
    band.appendChild(grid);
  }
  body.appendChild(band);

  // --- two sections, one question each ---------------------------------------
  // Health used to hold the seven lists and to open on load, because the
  // worklist above pointed into it. Those lists are the evidence pane now, so
  // what is left here are readings you come looking for, and neither unrolls
  // by itself: sixteen cards open at once is what made this view unreadable.
  const sec = (title, sub, open) => {
    const box = mkEl("details", "msec");
    box.open = open;
    const sum = mkEl("summary", "");
    sum.appendChild(mkEl("span", "msec-t", title));
    sum.appendChild(mkEl("span", "msec-s", sub));
    box.appendChild(sum);
    const g = mkEl("div", "mgrid");
    box.appendChild(g);
    body.appendChild(box);
    return g;
  };
  const structure = sec("Structure", "how the vault is shaped, and how tightly", false);
  const activity = sec("Activity", "what wrote it, and what could be added", false);
  // Cards route by section from here; `grid` stays the name the card builders
  // below already use for the one they belong to.
  let grid = structure;

  // --- E(vault) and its decomposition ----------------------------------------
  // The reading and the terms that make it up, in one card. Apart they were a
  // display number with no explanation next to it and an explanation with no
  // number; the whole point of E is that it is a sum you can open.
  const ec = mCard("E(vault) · lattice energy",
    full ? "measured at full depth" : "structural depth · integration deficits not measured");
  ec.appendChild(mkEl("div", "hero-val", (e.total > 0 ? "+" : "") + e.total.toFixed(2)));
  ec.appendChild(mkEl("p", "mnote",
    "Lower is more coherent. A thermometer, not a target: read it to compare runs, "
    + "never descend it. "
    + (full
      ? "Comparable only to other full-depth readings."
      : "Not comparable to a full-depth E.")));
  if (d.discourse_state) ec.appendChild(mkEl("div", "chip", "discourse: " + d.discourse_state));
  ec.appendChild(mkEl("p", "mcard-sub", `the ${e.terms.length} terms that sum to E`));
  ec.appendChild(waterfall(
    e.terms.map((t) => ({ label: t.name, value: t.value })), e.total,
    { negLabel: "bonds formed (lowers E)", posLabel: "entropic cost (raises E)" },
  ));
  grid.appendChild(ec);

  // --- areas, and the field every signal is read against ---------------------
  // This was a ranked bar list of the top fourteen with the rest stated in a
  // sentence: "94 smaller areas hold the other 210 notes". That sentence is the
  // one part of the distribution a reader cannot see, and the tail could not
  // become a fifteenth bar because an aggregate never shares a magnitude scale
  // with the things it aggregates. As area, every area is drawn and the small
  // ones being small IS the reading.
  //
  // It is also the view's one field. Selecting a worklist row recolours it by
  // that signal's share of each area, which is what turns nine measurements
  // into nine readings of one vault rather than nine charts.
  // `label`, because that is what areaTreemap and the coupling matrix both read
  // and what `/shape` already ships. `/metrics` names the same field `hub`, and
  // an unmapped tile does not fail loudly -- it renders with an empty name.
  const areaField = (d.clusters || []).filter((c) => c.size > 1)
    .map((c) => ({ ...c, label: c.hub || "#" + c.id }));
  const cl = mCard("Areas", `${nfmt(areaField.length)} multi-note areas · sized by notes`);
  if (areaField.length) {
    const holder = mkEl("div", "tmap-holder");
    const legend = mkEl("p", "mnote");
    cl.append(holder, legend);
    // Why a signal has no per-area tally, stated where the fill falls back
    // rather than left as a colour that silently stops meaning anything.
    const unplaceable = {
      dangling: "a missing target has no note, so it belongs to no area.",
      gaps: "a gap is already a fact about a PAIR of areas.",
    };
    relens = (key) => {
      const lens = (d.signal_areas || {})[key] || null;
      const meta = signals.find(([k]) => k === key);
      const noun = meta ? meta[2] : "selected";
      holder.textContent = "";
      holder.appendChild(areaTreemap(areaField, { lens, lensLabel: noun }));
      legend.textContent = (lens
        ? `fill = share of each area's notes that are ${noun}, so a small area `
          + "carrying three reads louder than a large one carrying five"
        : (unplaceable[key] ? unplaceable[key] + " Fill is cohesion: "
            : "fill = cohesion, ")
          + "the share of possible links inside an area that exist")
        // Said because the ramp is relative: the darkest tile is the loudest
        // here, not a full score, and a reader comparing two runs by eye would
        // otherwise read a rescale as a change in the vault.
        + " · shaded against the loudest area, not an absolute scale";
    };
    relens(metricsSignal);
  } else mEmpty(cl, "No communities yet. Link some notes.");
  grid.appendChild(cl);

  // --- degree distribution ---------------------------------------------------
  // An empty vault gets no card at all: the endpoint still returns one zeroed
  // bin, and "every note carries at least one link" is a silly thing to say
  // about no notes.
  if (d.degree_histogram?.some((b) => b.count)) {
    const dh = mCard("Link distribution", "notes by how many resolved links they carry");
    dh.appendChild(histogram(d.degree_histogram));
    const isolated = d.degree_histogram[0];
    dh.appendChild(mkEl("p", "mnote",
      isolated && isolated.lo === 0 && isolated.count
        ? `${nfmt(isolated.count)} notes carry no resolved link at all`
        : "every note carries at least one resolved link"));
    grid.appendChild(dh);
  }

  // --- hubs ------------------------------------------------------------------
  // Two measurements that a table can only list one under the other, sorted by
  // one of them. They do not agree, and where they disagree is the reading: a
  // note with forty links and no betweenness is a hub of its own area that
  // nothing routes THROUGH, which the table's degree order puts at the top and
  // gives no way to notice. In/out degree stay on the tooltip, as before:
  // degree is their sum, and the split is a detail of one note, not a shape.
  const hb = mCard("Hubs", "links against traffic · the top 20 by degree");
  if (d.hubs?.length > 1) {
    hb.appendChild(scatter(d.hubs, {
      x: (h) => h.degree || 0,
      y: (h) => h.betweenness || 0,
      xLabel: "resolved links", yLabel: "betweenness",
      label: (h) => `${h.label} · ${h.area} · ${h.degree} links `
        + `(${h.in} in, ${h.out} out) · betweenness ${h.betweenness}`,
      quadrants: [
        { at: "rt", label: "central both ways" },
        { at: "rb", label: "linked, off the paths", tint: true },
        { at: "lt", label: "on the paths, lightly linked" },
      ],
      height: 230,
      onPick: (h) => showNode({ path: h.path }),
    }));
    hb.appendChild(mkEl("p", "mnote",
      "split at the medians of these twenty, not at a fixed threshold: neither "
      + "axis has a value that means the same thing on another vault"));
  } else if (d.hubs?.length) {
    // One point has no median to split on and no shape to read. The row it
    // would have plotted is the whole answer.
    hb.appendChild(mkEl("p", "mnote",
      `${d.hubs[0].label} is the only connected note: ${d.hubs[0].degree} links.`));
  } else mEmpty(hb, "No connected notes yet.");
  grid.appendChild(hb);

  grid = activity;

  // --- reliability tiers -----------------------------------------------------
  if (d.temporal) {
    const tp = d.temporal, bt = tp.by_tier || {};
    const tiers = [
      { tone: "ord-3", label: "human", value: bt["3"] || 0 },
      { tone: "ord-2", label: "grounded", value: bt["2"] || 0 },
      { tone: "ord-1", label: "distilled", value: bt["1"] || 0 },
    ];
    const tc = mCard("Reliability", `${nfmt(tp.notes_scanned)} notes scanned`);
    const scanned = tp.notes_scanned || 0;
    tc.appendChild(mLegend(tiers.map((t) => ({
      tone: t.tone, label: t.label,
      note: `${nfmt(t.value)}${scanned ? " · " + Math.round((t.value / scanned) * 100) + "%" : ""}`,
    }))));
    tc.appendChild(stackedBar(tiers));
    // A ratio against its limit is a meter, and it was a table row printing
    // "412 / 686" for a reader to divide.
    tc.appendChild(meter(tp.stamped, scanned,
      `${nfmt(tp.stamped)} of ${nfmt(scanned)} carry a claim stamp`));
    // What is left is three facts, and three facts are a sentence. A two-column
    // table around them is a grid drawn to hold nothing.
    const facts = [
      `${nfmt(tp.superseded_sections)} notes carry a Superseded section`,
      `${nfmt(tp.superseded_notes)} were merged away`,
    ];
    if (tp.oldest_valid_from) facts.push(`earliest claim ${tp.oldest_valid_from}`);
    tc.appendChild(mkEl("p", "mnote", facts.join(" · ")));
    grid.appendChild(tc);
  }

  grid = activity;

  // --- write sessions --------------------------------------------------------
  // What wrote the vault, not when its subjects happened: only a nucleated note
  // carries a claim clock. Reads as coverage — the areas the writing keeps
  // landing in, and the ones it has never reached.
  if (d.sessions?.days?.length) {
    const s = d.sessions;
    const sc = mCard("Write sessions",
      `${s.days.length} days · ${s.areas.length} of ${s.areas_total} areas written into`);
    sc.appendChild(sessionMatrix(s));
    // The unmeasured majority, named. A matrix that omitted it would read as
    // "the whole vault, over 9 days", which is the opposite of true.
    const caveats = [`${nfmt(s.undated)} notes carry no claim clock and have no place here`];
    if (s.untouched) {
      caveats.push(`${nfmt(s.untouched)} areas have never been written into`);
    }
    // Not a rounding loss: these are notes whose name resolves to two areas at
    // once, so any column would be a guess. Printed because the row totals
    // otherwise silently fall short of the dated count.
    if (s.ambiguous) {
      caveats.push(`${nfmt(s.ambiguous)} claims sit on a name that two areas both hold`);
    }
    sc.appendChild(mkEl("p", "mnote", caveats.join(" · ")));
    grid.appendChild(sc);
  }

  // --- code coverage ---------------------------------------------------------
  if (d.code_coverage) {
    const cc = d.code_coverage;
    const card = mCard("Code coverage", "source files with at least one note");
    card.appendChild(meter(cc.documented, cc.total,
      `${nfmt(cc.documented)} / ${nfmt(cc.total)} files documented`));
    if (cc.undocumented?.length) {
      card.appendChild(mTable(
        [{ key: "path", label: "Undocumented" }, { key: "fan_in", label: "Fan-in", num: true }],
        cc.undocumented.slice(0, 10),
      ));
    }
    grid.appendChild(card);
  }

  // --- what the recent writing is about (V6) ---------------------------------
  // A card in Activity and NOT a worklist row: a burst is not something to fix,
  // it is what the last fortnight of writing turned out to be about. The window
  // is the last 14 days of WRITING, not of wall-clock time, so a vault written
  // two months ago still has a most-recent fortnight and this card still says
  // something. z is a one-proportion score against the concept's own baseline
  // share, which is why a concept that is everywhere never bursts.
  if (full && d.bursting?.length) {
    const bc = mCard("Bursting concepts", "over-represented in the last 14 days of writing");
    bc.appendChild(mTable(
      [{ key: "concept", label: "Concept" }, { key: "recent", label: "Recent", num: true },
       { key: "total", label: "All", num: true }, { key: "z", label: "z", num: true }],
      d.bursting.slice(0, 12),
    ));
    grid.appendChild(bc);
  }

  grid = structure;

  // --- bridges ----------------------------------------------------------------
  // Gaps moved into the worklist's evidence pane, beside the row that counts
  // them. Bridges stay a card: they are not a signal to act on, they are a
  // reading about how the areas actually touch.
  const br = mCard("Surprising bridges", "cross-area links between otherwise distant notes");
  if (d.bridges?.length) {
    // One ranked measure over named things, which is what a magnitude chart is
    // for. As a two-column table the order was asserted by row position and the
    // distances between the weights -- whether the top bridge is twice the
    // second or a hair above it -- were left for the reader to subtract.
    br.appendChild(barChart(d.bridges.map((b) => ({
      label: b.source + " ↔ " + b.target, value: b.weight, path: b.source_path,
      title: `${b.source} ↔ ${b.target}: surprise ${b.weight}`,
    }))));
  } else mEmpty(br, "No cross-area links yet.");
  grid.appendChild(br);

  grid = activity;

  // --- proposals (not authoritative) -----------------------------------------
  // The co-occurrence leg runs one expanded ranking per note, so it is minutes
  // on a real vault. Asked for, never assumed.
  if (!full) {
    const ask = mCard("Proposals", "co-occurrence delta, not yet measured");
    ask.classList.add("proposed");
    ask.appendChild(mkEl("p", "mempty",
      "Autolink candidates, stale links, missing hubs and integration deficits "
      + "come from comparing the co-occurrence graph against the wikilinks. "
      + "That pass ranks every note against every other, so it grows with the "
      + "square of the vault: seconds here, longer on a big one."));
    const btn = mkEl("button", "mbtn", "measure proposals");
    btn.type = "button";
    btn.id = "metrics-proposals";
    ask.appendChild(btn);
    grid.appendChild(ask);
  }
  const props = [
    ["Duplicate pairs", "embeddings propose · graph disposes", d.duplicates,
     [{ key: "pair", label: "Pair" }, { key: "score", label: "Cosine", num: true },
      { key: "band", label: "Band" }],
     (x) => ({ pair: x.a + " ↔ " + x.b, score: x.score, band: x.confirmed ? "merge?" : "link", _path: x.a_path }),
     (T.confirmed_duplicates || 0) + (T.duplicate_pairs || 0), "pairs"],
    ["Autolink candidates", "co-mentioned in text, never linked", d.autolinks,
     [{ key: "pair", label: "Pair" }, { key: "shared", label: "Shared concepts" },
      { key: "weight", label: "Weight", num: true }],
     (x) => ({ pair: x.a + " ↔ " + x.b, shared: (x.shared || []).join(", "), weight: x.weight, _path: x.a_path }),
     T.autolink_candidates || 0, "pairs"],
    ["Stale links", "linked, but share no concepts in text", d.stale_links,
     [{ key: "pair", label: "Pair" }],
     (x) => ({ pair: x.a + " ↔ " + x.b, _path: x.a_path }),
     T.stale_links || 0, "links"],
    ["Missing hubs", "central concepts with no note of their own", d.missing_hubs,
     [{ key: "concept", label: "Concept" }, { key: "centrality", label: "Centrality", num: true }],
     (x) => x, T.missing_hubs || 0, "concepts"],
  ];
  for (const [title, sub, rows, cols, map, total, noun] of props) {
    if (!rows?.length) continue;
    const c = mCard(title, sub);
    c.classList.add("proposed");
    c.appendChild(mTable(cols, rows.map(map)));
    const more = total ? mMore(rows.length, total, noun) : null;
    if (more) c.appendChild(more);
    grid.appendChild(c);
  }
}

// --- map landing: root the radial map on a note; hub-picker until one is set --
function rootMap(path) {
  mapRootedPath = path;
  if (graphMode !== "map") setGraphMode("map");
  $("#map-picker").hidden = true;
  $("#map-frame").hidden = false;
  $("#map-loading").hidden = false;
  $("#map-frame").src = "/map?note=" + encodeURIComponent(path)
    + "&theme=" + liveTheme() + "&t=" + Date.now();
  closeNodeResults();
}

function renderMapPicker(hubs) {
  const box = $("#map-picker-list");
  box.innerHTML = "";
  for (const h of hubs) {
    const row = document.createElement("div");
    row.className = "hub-row";
    row.dataset.path = h.path;
    row.innerHTML = '<span class="hub-name"></span><span class="hub-deg"></span>';
    row.querySelector(".hub-name").textContent = h.name;
    row.querySelector(".hub-deg").textContent = h.degree;
    box.appendChild(row);
  }
}
$("#map-picker-list").addEventListener("click", (e) => {
  const row = e.target.closest(".hub-row");
  if (row) rootMap(row.dataset.path);
});

// --- path: the reading order around one note (V2, RefD) ----------------------
// The sixth surface, and the only directed one. Rungs are LONGEST-path depth,
// not hop count: on a ladder a shortcut edge would otherwise put a note on the
// same rung as its own prerequisite, and the rung is the whole reading.
//
// Wires are drawn after layout from the chips' own boxes rather than laid out
// in SVG: the chips are text of unpredictable width, and measuring what the
// browser already placed is the only way the two can agree. They are redrawn on
// resize for the same reason.
let pathRootedPath = null;
let pathWireObserver = null;
let pathEdges = [];

function rootPath(path) {
  pathRootedPath = path;
  if (graphMode !== "path") setGraphMode("path");
  else drawPath();
  closeNodeResults();
}

async function drawPath() {
  const pane = $("#shape-pane");
  pane.innerHTML = "";
  $("#shape-loading").hidden = false;
  let d;
  try {
    d = await (await fetch("/path?note=" + encodeURIComponent(pathRootedPath || ""))).json();
  } catch {
    d = { error: "couldn't read the reading order" };
  }
  $("#shape-loading").hidden = true;
  if (graphMode !== "path") return; // the reader switched surface mid-flight
  pane.innerHTML = "";
  pane.appendChild(renderPath(d));
  drawPathWires();
}

function pathRungLabel(depth) {
  if (depth === 0) return "this note";
  if (depth === -1) return "read first";
  if (depth === 1) return "unlocks";
  return depth < 0 ? (-depth) + " steps before" : depth + " steps after";
}

function renderPath(d) {
  const wrap = mkEl("div", "pl-wrap");
  if (d.error) {
    wrap.appendChild(mkEl("p", "pl-empty", d.error));
    return wrap;
  }
  // The landing. Same job as the map picker and a different ranking: a note
  // roots a ladder by what reads around it, which the file tree cannot say.
  if (d.picks) {
    wrap.appendChild(mkEl("div", "pl-h", "Pick a note to place in its reading order"));
    if (d.hint) wrap.appendChild(mkEl("p", "pl-empty", d.hint));
    if (!d.picks.length && !d.hint) {
      wrap.appendChild(mkEl("p", "pl-empty",
        "no note has a prerequisite yet. RefD only speaks where the co-occurrence "
        + "index has enough related notes to compare."));
    }
    const list = mkEl("div", "pl-picks");
    for (const pk of d.picks) {
      const row = mkEl("button", "pl-pick");
      row.type = "button";
      row.dataset.root = pk.path;
      row.appendChild(mkEl("span", "pl-pick-n", pk.name));
      row.appendChild(mkEl("span", "pl-pick-d",
        pk.before + " before · " + pk.after + " after"));
      list.appendChild(row);
    }
    wrap.appendChild(list);
    return wrap;
  }

  const head = mkEl("div", "pl-h");
  head.appendChild(mkEl("span", "pl-h-t", (d.root && d.root.name) || "reading order"));
  wrap.appendChild(head);
  if (d.hint) {
    wrap.appendChild(mkEl("p", "pl-empty", d.hint));
    return wrap;
  }

  const lad = mkEl("div", "pl");
  const wires = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  wires.setAttribute("class", "pl-wires");
  wires.setAttribute("aria-hidden", "true");
  lad.appendChild(wires);
  for (const lv of d.levels) {
    const rung = mkEl("div", "pl-rung" + (lv.depth === 0 ? " here" : ""));
    rung.appendChild(mkEl("div", "pl-lbl", pathRungLabel(lv.depth)));
    const notes = mkEl("div", "pl-notes");
    for (const n of lv.notes) {
      const chip = mkEl("div", "pl-n" + (n.root ? " root" : "") + (n.cyclic ? " cyc" : ""));
      chip.dataset.path = n.path;
      chip.dataset.plId = n.path;
      chip.appendChild(mkEl("span", "pl-n-t", n.name));
      if (!n.root) {
        const re = mkEl("button", "pl-re", "⤴");
        re.type = "button";
        re.dataset.root = n.path;
        re.title = "place this note in its own reading order";
        notes.appendChild(chip);
        chip.appendChild(re);
      } else {
        notes.appendChild(chip);
      }
    }
    rung.appendChild(notes);
    lad.appendChild(rung);
  }
  wrap.appendChild(lad);

  const foot = [];
  if (d.truncated) foot.push("cut at 60 notes: the ladder continues past this");
  // A cycle is a disagreement the vault really contains, not corrupt data, so
  // it is reported rather than hidden. The edges that close it are dropped from
  // the layout, which is why the count is worth stating at all.
  if (d.cycles) {
    foot.push(d.cycles + (d.cycles === 1 ? " note is" : " notes are")
      + " in a prerequisite cycle: their edges are not drawn");
  }
  if (foot.length) wrap.appendChild(mkEl("p", "pl-foot", foot.join(" · ")));
  pathEdges = d.edges || [];
  return wrap;
}

function drawPathWires() {
  const lad = $("#shape-pane").querySelector(".pl");
  const svg = lad && lad.querySelector(".pl-wires");
  if (!svg) return;
  const box = lad.getBoundingClientRect();
  svg.setAttribute("viewBox", "0 0 " + Math.round(box.width) + " " + Math.round(box.height));
  svg.setAttribute("width", Math.round(box.width));
  svg.setAttribute("height", Math.round(box.height));
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const at = (path) => {
    const el = lad.querySelector('[data-pl-id="' + CSS.escape(path) + '"]');
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: r.left - box.left + r.width / 2, top: r.top - box.top, bot: r.bottom - box.top };
  };
  for (const e of pathEdges) {
    const a = at(e.from), b = at(e.to);
    if (!a || !b) continue;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const y1 = a.bot, y2 = b.top;
    const mid = (y1 + y2) / 2;
    path.setAttribute("d", "M" + a.x + " " + y1 + " C" + a.x + " " + mid
      + " " + b.x + " " + mid + " " + b.x + " " + y2);
    path.setAttribute("class", "pl-wire");
    svg.appendChild(path);
  }
  if (!pathWireObserver && window.ResizeObserver) {
    pathWireObserver = new ResizeObserver(() => {
      if (graphMode === "path") drawPathWires();
    });
    pathWireObserver.observe($("#shape-pane"));
  }
}

$("#shape-pane").addEventListener("click", (e) => {
  const b = e.target.closest("[data-root]");
  if (b) rootPath(b.dataset.root);
});

// --- explore note search (network: fly the camera · map: root the map) --------
// A fuzzy ranked picker over the vault's notes, indexed from the sidebar tree —
// same title→prefix→substring→path ranking the graph viewer's own search uses.
let noteIdx = [];      // [{name, path, ln, lp}]
let nodeResults = [];  // current ranked matches
let nodeSel = -1;

function buildNoteIndex() {
  noteIdx = Array.from($("#tree").querySelectorAll(".tree-note")).map((el) => {
    const name = el.textContent, path = el.dataset.id || "";
    return { name, path, ln: name.toLowerCase(), lp: path.toLowerCase() };
  });
}

function scoreNote(n, q) {
  if (n.ln === q) return 5;
  if (n.ln.startsWith(q)) return 4;
  if (n.ln.includes(q)) return 3;
  if (n.lp.includes(q)) return 2;
  return 0;
}

function renderNodeResults(raw) {
  const q = raw.trim().toLowerCase();
  const box = $("#node-results");
  if (!q) { closeNodeResults(); return; }
  nodeResults = noteIdx
    .map((n) => [scoreNote(n, q), n])
    .filter((p) => p[0] > 0)
    .sort((a, b) => b[0] - a[0] || a[1].name.localeCompare(b[1].name))
    .slice(0, 12)
    .map((p) => p[1]);
  nodeSel = nodeResults.length ? 0 : -1;
  box.innerHTML = "";
  nodeResults.forEach((n, i) => {
    const el = document.createElement("div");
    el.className = "node-result" + (i === nodeSel ? " sel" : "");
    el.innerHTML = '<span class="nr-name"></span><span class="nr-path"></span>';
    el.querySelector(".nr-name").textContent = n.name;
    el.querySelector(".nr-path").textContent = n.path;
    el.addEventListener("click", () => pickNote(n.path));
    box.appendChild(el);
  });
  box.hidden = nodeResults.length === 0;
}

function closeNodeResults() {
  $("#node-results").hidden = true;
  nodeResults = [];
  nodeSel = -1;
}

function moveNodeSel(d) {
  nodeSel = (nodeSel + d + nodeResults.length) % nodeResults.length;
  document.querySelectorAll("#node-results .node-result").forEach((el, i) => el.classList.toggle("sel", i === nodeSel));
}

function pickNote(path) {
  if (graphMode === "map") {
    rootMap(path);
  } else if (graphMode === "path") {
    rootPath(path);
  } else { // network: locate the note and fly the graph camera to it
    const f = $("#graph-frame");
    if (f.contentWindow) f.contentWindow.postMessage({ type: "silica-goto-path", path }, "*");
  }
  $("#node-search").value = "";
  closeNodeResults();
}

$("#node-search").addEventListener("input", (e) => renderNodeResults(e.target.value));
$("#node-search").addEventListener("keydown", (e) => {
  if (!nodeResults.length) return;
  if (e.key === "ArrowDown") { e.preventDefault(); moveNodeSel(1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); moveNodeSel(-1); }
  else if (e.key === "Enter") { e.preventDefault(); if (nodeSel >= 0) pickNote(nodeResults[nodeSel].path); }
  else if (e.key === "Escape") { $("#node-search").value = ""; closeNodeResults(); }
});


// --- attachments: drop / "+" accumulate files as chips above the input; they
// are NOT nucleated on drop. The next composer submit uploads them together with
// the typed message, so the agent acts on the files per the user's instruction.
let staged = []; // File objects awaiting the next submit
const attachEls = $("#attachments");

function renderAttachments() {
  attachEls.innerHTML = "";
  attachEls.hidden = staged.length === 0;
  syncQuick(); // staged files are what the next message does
  staged.forEach((f, i) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<span class="chip-name"></span><button type="button" class="chip-x" title="remove">✕</button>`;
    chip.querySelector(".chip-name").textContent = f.name;
    chip.querySelector(".chip-x").addEventListener("click", () => { staged.splice(i, 1); renderAttachments(); });
    attachEls.appendChild(chip);
  });
}
function addFiles(fileList) {
  for (const f of fileList) staged.push(f);
  renderAttachments();
}

// Upload every staged file + the typed text as one turn (server stages them —
// converts PDFs, stubs code — then the agent works on them per `text`).
function nucleateStaged(text) {
  if (streaming || !staged.length) return;
  const names = staged.map((f) => f.name);
  bubble("user").textContent = (text.trim() ? text.trim() + "\n" : "") + "⇪ " + names.join(", ");
  const fd = new FormData();
  for (const f of staged) fd.append("files", f);
  fd.append("text", text);
  staged = [];
  renderAttachments();
  runTurn(fetch("/nucleate", { method: "POST", body: fd }), "staging " + names.length + (names.length === 1 ? " file" : " files"));
}

let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; document.body.classList.add("dragging"); });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("dragleave", (e) => { e.preventDefault(); if (--dragDepth <= 0) document.body.classList.remove("dragging"); });
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dragging");
  if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
});

// "+" opens the native picker, constrained to what the nucleate lanes accept.
const nucleateInput = $("#nucleate-file");
fetch("/supported_types")
  .then((r) => r.json())
  .then((d) => { nucleateInput.accept = (d.extensions || []).join(","); })
  .catch(() => {}); // accept="" just means the picker shows all files
$("#attach").addEventListener("click", () => nucleateInput.click());
nucleateInput.addEventListener("change", () => {
  addFiles(nucleateInput.files);
  nucleateInput.value = ""; // reset so re-picking the same file fires change again
});

// --- note panel (right overlay drawer; opens from .note-link, the graph, and the map) -
const notePanel = $("#note-panel");
let lastNotePath = null;   // note currently open in the drawer
let lastViewedPath = null; // survives close — feeds the header reopen button

// The dock inset and the drawer width must agree; CSS reads it as --note-w.
function setNoteW(w) {
  document.documentElement.style.setProperty("--note-w", w + "px");
  syncDrawerToViews(); // the frame parks its focus bar against the drawer's edge
}

// Mirror the open note onto the graph + map iframes: the matching node + its
// 1-hop neighbours go full-opacity, everything else dims. No-op harmlessly if
// a tab was never opened (contentWindow still exists, message just has no
// listener yet).
function postToViews(msg) {
  for (const id of ["#graph-frame", "#map-frame"]) {
    const frame = $(id);
    if (frame.contentWindow) frame.contentWindow.postMessage(msg, "*");
  }
}
// The last focus INTENT, replayed whenever a view (re)loads. A frame that is
// still loading drops the message, and /graph is rebuilt on every entry into
// the explore tab — so "light these notes" issued from the chat tab used to be
// posted into a loading iframe and then overwritten by the load handler.
let graphFocus = [];
function focusGraphNode(path) {
  graphFocus = path ? [path] : [];
  replayGraphFocus();
}
// Same, for a SET: a concept lights every note that carries it.
function focusGraphNodes(paths) {
  graphFocus = paths || [];
  replayGraphFocus();
}
// Both shapes, in order: the radial map only speaks the single-path message,
// the graph speaks both and the set arrives second, so it wins there.
function replayGraphFocus() {
  postToViews({ type: "silica-focus-path", path: graphFocus[0] || null });
  if (graphFocus.length > 1) postToViews({ type: "silica-focus-paths", paths: graphFocus });
}

// Explore does not inset for the drawer the way chat does — the graph keeps its
// full width and the drawer overlays it, so the frame's own HUD sits under a
// translucent panel and reads through the note. The frame cannot see the
// drawer, so tell it. Replayed on frame load like the focus state, because
// /graph is rebuilt on every entry into the tab.
function syncDrawerToViews() {
  const open = document.body.classList.contains("note-open");
  postToViews({ type: "silica-host-drawer", open, width: open ? notePanel.offsetWidth : 0 });
}

// Mermaid is a 3.5MB vendored bundle, so it loads on demand — only the first
// time an opened note actually contains a ```mermaid fence. Render failures
// leave the fence as plain text (suppressErrorRendering).
let mermaidLoad = null;

// Read the palette instead of restating it. The old block carried five hex
// literals copied out of app.css, and by the time anyone looked they were two
// revisions stale — which is the failure mode a second theme turns from a
// blemish into a wrong-coloured diagram on a white page. getComputedStyle on
// :root returns whichever ramp is live, so this cannot drift from either.
// `base` in both themes, never "dark"/"default". Mermaid derives a theme object
// on the first initialize() and a later one does not rebuild it: switching
// theme:"dark" -> theme:"default" re-rendered every diagram in mermaid's own
// lavender defaults with our themeVariables sitting in the live config, ignored.
// `base` is the theme that exists to be driven entirely by those variables, so
// nothing has to be re-derived and there is one code path instead of two.
function mermaidConfig() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  const light = document.documentElement.dataset.theme === "light";
  return {
    startOnLoad: false, suppressErrorRendering: true, theme: "base",
    fontFamily: "Martian Mono, ui-monospace, monospace",
    themeVariables: {
      darkMode: !light,
      background: v("--void"),
      primaryColor: v("--slate-2"),
      primaryTextColor: v("--frost"),
      primaryBorderColor: v("--line-2"),
      lineColor: v("--ash"),
      // base derives the rest from these four, and derives them wrong when the
      // ramp is not a neutral gray: naming them is cheaper than correcting it.
      secondaryColor: v("--slate"),
      tertiaryColor: v("--sheet"),
      mainBkg: v("--slate-2"),
      textColor: v("--text"),
    },
  };
}

function renderMermaid(root) {
  const blocks = root.querySelectorAll("pre.mermaid");
  if (!blocks.length) return;
  // Keep the source: mermaid.run() replaces the fence's text with an <svg>, so
  // without this a repaint has nothing left to re-render from.
  blocks.forEach((b) => { if (!b.dataset.mmd) b.dataset.mmd = b.textContent; });
  mermaidLoad ||= new Promise((resolve) => {
    const s = document.createElement("script");
    s.src = "/static/mermaid.min.js";
    s.onload = () => { mermaid.initialize(mermaidConfig()); resolve(); };
    document.head.appendChild(s);
  });
  mermaidLoad.then(() => mermaid.run({ nodes: blocks }).catch(() => {}));
}

// A diagram is baked SVG, so it does not follow a token swap the way the rest
// of the page does — every rendered fence has to be rewound to its source and
// drawn again. Nothing to do if the bundle was never loaded.
function repaintMermaid() {
  if (!mermaidLoad) return;
  mermaidLoad.then(() => {
    mermaid.initialize(mermaidConfig());
    const done = document.querySelectorAll("pre.mermaid[data-processed]");
    if (!done.length) return;
    done.forEach((b) => {
      b.removeAttribute("data-processed");
      b.textContent = b.dataset.mmd || "";
    });
    mermaid.run({ nodes: done }).catch(() => {});
  });
}

// The transcript is what the drawer exists to serve, so it gets a floor and the
// two panes beside it negotiate around it.
//
// The old rule was a fixed 1100px breakpoint, and it made the layout NON-MONOTONIC:
// below 1100 the sidebar yielded, but at 1200 it stayed (264px) while the drawer
// kept its full 630, leaving 306px of log — measured at 21 characters of prose per
// line, worse than at the 900px floor. The window got bigger and the transcript got
// smaller. 1280 and 1366 are the two commonest laptop widths after 1440, so the
// broken band was the common case.
const MIN_PROSE = 560;   // px of transcript that must survive, ~47ch of prose once
                         // the sheet's and the row's padding come out of it
const MIN_DRAWER = 320;  // below this the drawer stops being worth its own pane
const SIDE_W = 264;      // must match --side-w
let sidebarYielded = false;

// What the drawer may take before the transcript drops below its floor.
function drawerBudget(sidebarOn) {
  return window.innerWidth - (sidebarOn ? SIDE_W : 0) - MIN_PROSE;
}

// Single owner of the open-drawer layout: decides whether the sidebar can stay,
// then sizes the drawer to whatever is left over the floor. Runs on open, on
// resize and after a drag, so the same window can no longer sit in two different
// layouts depending on the order those happened in.
function fitPanes() {
  if (!document.body.classList.contains("note-open")) return;
  const userCollapsed = document.body.classList.contains("sidebar-collapsed") && !sidebarYielded;
  if (!userCollapsed) {
    if (drawerBudget(true) >= MIN_DRAWER) restoreYieldedSidebar();
    else yieldSidebarToDrawer();
  }
  const sidebarOn = !document.body.classList.contains("sidebar-collapsed");
  const want = parseInt(localStorage.getItem("note-width"), 10) || 630;
  const w = Math.max(MIN_DRAWER, Math.min(want, drawerBudget(sidebarOn)));
  notePanel.style.width = w + "px";
  setNoteW(w);
}

function yieldSidebarToDrawer() {
  if (document.body.classList.contains("sidebar-collapsed")) return;
  document.body.classList.add("sidebar-collapsed");
  sidebarYielded = true;
}

function restoreYieldedSidebar() {
  if (!sidebarYielded) return;
  document.body.classList.remove("sidebar-collapsed");
  sidebarYielded = false;
}

// The drawer is the NOTE: `note` is the reader and `diff` is the same file
// against how it stood before this session touched it. It carried a third mode,
// `context`, which drew the /context payload the work panel's node scope also
// draws - two panels at the same edge, the same sections, neither header saying
// which of them you were reading. They had already drifted apart by the time it
// was noticed, each dropping a section the other kept. The panel took the
// payload; this drawer kept the prose, and each fact now has one home.
//
// The click contract, unchanged in principle and now stated once: NAMING a note
// means "I want to read it", so a wikilink, the file tree and a search hit land
// here. POINTING at one means "what is this", so a graph node, a map card and a
// metrics row fill the work panel instead. A second click on a node is "read
// it" said with the gesture, and lands back here.
let drawerMode = "note";

function syncDrawerMode() {
  const path = lastNotePath || lastViewedPath;
  document.querySelectorAll("#note-mode button").forEach((b) => {
    setActive(b, b.dataset.mode === drawerMode);
    // A note this session never touched has no diff, and an enabled tab onto an
    // empty pane is a promise the drawer cannot keep.
    if (b.dataset.mode === "diff") {
      b.disabled = !changedPaths.has(path);
      b.title = b.disabled ? "this session has not changed this note"
                           : "what this session changed in this note";
    }
  });
  $("#note-body").hidden = drawerMode !== "note";
  $("#note-diff").hidden = drawerMode !== "diff";
}

// Shared tail of both openers: raise the panel and let fitPanes negotiate the
// widths. Kept in one place so the two modes cannot drift apart on layout.
function showDrawer(title) {
  $("#note-title").textContent = title || "";
  notePanel.classList.add("open");
  notePanel.setAttribute("aria-hidden", "false");
  document.body.classList.add("note-open"); // dock + chat inset to the drawer's edge
  syncDrawerToViews();
  fitPanes(); // owns the sidebar decision AND the drawer width, in that order
  $("#note-last").querySelector("span").textContent = title || "";
  syncPinButton(); // the toggle states THIS note, so it is re-read per open
}

async function openNote(path) {
  if (!path) return;
  lastNotePath = path;
  lastViewedPath = path;
  drawerMode = "note";
  syncDrawerMode();
  focusGraphNode(path);
  try {
    const r = await fetch("/note?path=" + encodeURIComponent(path));
    const data = await r.json();
    $("#note-body").innerHTML = data.html || "";
    renderMermaid($("#note-body"));
    $("#note-body").scrollTop = 0;
    showDrawer(data.title || path);
  } catch { notify("couldn't open that note"); }
}

// --- the Node panel: what you pointed at, in the third column ----------------
// Pointing at a node fills the third column instead of opening a drawer over
// the surface you are reading. The click contract itself does not move: naming
// a note still means "read it" and still lands in the drawer; pointing at one
// still means "what is this". What moved is the ANSWER, which used to be drawn
// twice - here for explore and in the drawer's context mode for every other
// view - off the one /context call. One reader now, so a node answers the same
// on the graph, the map, a metrics row and a shape row.
//
// The head facts (degree, state, area) ride in on the frame's message rather
// than being asked for again: the frame computed all three to draw the node you
// clicked, and a second source for them is a second answer that can disagree
// with the picture in front of you.
const announceNode = (node, context) =>
  document.dispatchEvent(new CustomEvent("silica:node",
    { detail: node ? { node: node, context: context } : null }));

let nodeSeq = 0; // two fast clicks: only the last one's context may land

async function showNode(target) {
  const path = target.path || "";
  const ghost = !!target.ghost || !path;
  if (!path && !target.name) return;
  if (!ghost) { lastNotePath = path; lastViewedPath = path; }
  focusGraphNode(ghost ? null : path);
  const seq = ++nodeSeq;
  announceNode(target, null);   // the head paints now; the sections follow
  try {
    const q = ghost
      ? "ghost=1&name=" + encodeURIComponent(target.name || "")
      : "path=" + encodeURIComponent(path);
    const data = await (await fetch("/context?" + q)).json();
    if (seq === nodeSeq) announceNode(target, data);
  } catch {
    if (seq === nodeSeq) announceNode(target, { error: "couldn't read that note's context" });
  }
}
window.showNode = showNode;

$("#note-mode").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-mode]");
  if (!b || b.disabled || b.dataset.mode === drawerMode) return;
  const path = lastNotePath || lastViewedPath;
  if (b.dataset.mode === "diff") openDiff(path); else openNote(path);
});

// --- diff mode ---------------------------------------------------------------
// The note against how it stood before this session touched it. Red is what left
// the file, green is what arrived; the gaps are the unchanged stretches the
// server left out. Rows are DOM nodes, never innerHTML: every line here is a line
// of the user's own vault.
async function openDiff(path) {
  if (!path) return;
  lastNotePath = path;
  lastViewedPath = path;
  drawerMode = "diff";
  syncDrawerMode();
  focusGraphNode(path);
  const box = $("#note-diff");
  box.className = "dl-wait";
  box.textContent = "reading the diff…";
  showDrawer(path.split("/").pop().replace(/\.md$/, ""));
  let d;
  try {
    d = await (await fetch("/changes/diff?path=" + encodeURIComponent(path))).json();
  } catch {
    box.className = "dl-wait";
    box.textContent = "couldn't read that diff";
    return;
  }
  // The session never touched this note, so there is nothing to compare it
  // against. Show the note rather than an empty diff pretending to be one.
  if (d.baseline === false) return openNote(path);
  box.className = "";
  box.innerHTML = "";
  const head = mkEl("div", "dl-head");
  head.appendChild(mkEl("span", "dl-kind " + d.kind, d.kind));
  if (d.from) head.appendChild(mkEl("span", "dl-from", d.from + " →"));
  if (d.added) head.appendChild(mkEl("span", "chg-add", "+" + d.added));
  if (d.removed) head.appendChild(mkEl("span", "chg-del", "−" + d.removed));
  box.appendChild(head);
  if (!d.lines.length) {
    box.appendChild(mkEl("div", "dl-empty", d.kind === "moved"
      ? "moved, and the bytes are unchanged"
      : "no difference left: this note is back to how it started"));
  }
  const track = mkEl("div", "dl-track");
  const CLS = { "+": "dl-add", "-": "dl-del", "@": "dl-gap" };
  for (const l of d.lines) {
    const cls = CLS[l.op] || "dl-ctx";
    // A gap is the elision itself, not a line of the note.
    track.appendChild(mkEl("div", "dl-line " + cls, l.op === "@" ? "⋯" : l.op + l.text));
  }
  box.appendChild(track);
  if (d.clipped)
    box.appendChild(mkEl("div", "dl-empty",
      `${d.clipped} more lines. Open the note for the rest`));
  box.scrollTop = 0;
  showDrawer(d.name || path);
}

function closeNote() {
  notePanel.classList.remove("open");
  notePanel.setAttribute("aria-hidden", "true");
  document.body.classList.remove("note-open");
  syncDrawerToViews();
  restoreYieldedSidebar();
  lastNotePath = null; // lastViewedPath survives — the header button can reopen
  focusGraphNode(null);
}
$("#note-last").addEventListener("click", () => {
  if (lastViewedPath) openNote(lastViewedPath);
});

// The two turns a suggestion drafts. They live here and not in work.js, which
// is the only surface that offers them, because a prompt written twice is two
// turns that drift the first time one of them is reworded - the same rule the
// metrics view's bulk prompts already follow.
function writeGhostPrompt(name, from) {
  return 'Write the note "' + name + '", which "' + from + '" already links to.';
}

function linkNotesPrompt(a, b) {
  return 'Check whether "' + a + '" and "' + b + '" belong linked, and if they do, '
       + "add the wikilink in whichever direction reads right.";
}

// The unresolved link as a subject of its own: it has no body, and the notes
// that already point at it are the only material there is to write it from.
// Three callers hand it three shapes of the same list (backlink rows, ghost
// source names, a metrics row's `from`), so it takes either and quotes both:
// an unquoted list of names ending in a comma is a sentence the model has to
// guess the boundaries of.
function ghostWritePrompt(title, sources) {
  const from = (sources || []).map((s) => '"' + noExt(s.name || s) + '"').join(", ");
  return 'Write the note "' + title + '".'
       + (from ? " It is already linked from " + from
                 + ", so ground it in what those notes already say." : "");
}
window.writeGhostPrompt = writeGhostPrompt;
window.linkNotesPrompt = linkNotesPrompt;
window.ghostWritePrompt = ghostWritePrompt;

// A concept is a set of notes, so it focuses a set. Clicked from chat it also
// switches to explore first — there is nothing to see otherwise.
// Exported: the explore panel's concept pills light the same notes through this
// one function, rather than growing a second copy of "resolve the term, then
// focus whatever carries it".
async function lightConcept(term, btn) {
  document.querySelectorAll(".wk-pill.lit").forEach((e) => e.classList.remove("lit"));
  btn.classList.add("lit");
  if (activeTab !== "graph") showTab("graph");
  try {
    const d = await (await fetch("/concept?term=" + encodeURIComponent(term))).json();
    if (!d.notes || !d.notes.length) { notify("no notes carry “" + term + "”"); return; }
    focusGraphNodes(d.notes);
  } catch { notify("couldn't resolve that concept"); }
}
window.lightConcept = lightConcept;

// Suggested rows never write. They prefill a chat turn and hand it back to you,
// so the write still goes through the agent's gate — validate, checkpoint, undo
// journal — instead of a drawer button reaching the disk on its own.
function prefillChat(text) {
  showTab("chat");
  const box = $("#input");
  box.value = text;
  box.focus();
  box.dispatchEvent(new Event("input")); // let the autosize/palette hooks see it
}

// "map" button in the drawer header — jump to explore's map mode, rooted here.
// Capture the path FIRST: the programmatic tab .click() bubbles to the document
// outside-click handler, which closes the drawer and nulls lastNotePath
// synchronously before rootMap runs (else note=null). Pre-set graphMode so the
// tab-enter goes straight to map instead of loading the graph first.
$("#note-map").addEventListener("click", () => {
  const note = lastNotePath;
  if (!note) return;
  graphMode = "map";
  document.querySelector('.tab[data-tab="graph"]').click();
  rootMap(note);
});

// summarize / explain / quiz — dispatch the reader slash-command for the open
// note as a chat turn. The drawer stays open (the peek dock tucks under it and
// mirrors the turn), so the note you launched from is never lost.
const shellQuote = (s) => '"' + String(s).replace(/"/g, '\\"') + '"';
function drawerReader(makeCmd) {
  if (!lastNotePath || streaming) return; // streaming: send() would no-op — no peek either
  const cmd = makeCmd(lastNotePath, $("#note-title").textContent.trim());
  if (activeTab !== "chat") openPeek(cmd); // on chat the stream is already visible
  send(cmd);
}
$("#note-summarize").addEventListener("click", () => drawerReader((p) => "/summarize " + shellQuote(p)));
$("#note-explain").addEventListener("click", () => drawerReader((p, t) => "/explain " + shellQuote(t || p)));
$("#note-quiz").addEventListener("click", () => drawerReader((p) => "/quiz " + shellQuote(p)));
$("#note-relate").addEventListener("click", () => drawerReader((p) => "/relate " + shellQuote(p)));

// --- dock card (rendered answer for a dock- or drawer-launched turn) ---------
// Not a re-implementation of the chat flow: no tools, no thinking text. Title =
// the dispatched prompt; body = pulsing "thinking", then the answer as live
// markdown (mdLite), upgraded to the canonical OFM render on `done` — so
// wikilinks in the card open the note drawer and focus the graph. One exchange
// only; the next one replaces it. "open in chat" → the full transcript.
const peekEl = $("#peek");
let peek = null; // { body, caret, raw } while a turn is being mirrored
function openPeek(title) {
  const body = $("#peek-body");
  body.className = "";
  body.textContent = "thinking";
  const caret = document.createElement("span"); // own instance: the chat caret is a
  caret.className = "caret";                    // single element, re-parented live
  caret.textContent = "▍";
  body.appendChild(caret);
  $("#peek-title").textContent = title;
  peekEl.hidden = false;
  peek = { body, caret, raw: "", mark: 0 };
}
function closePeek() {
  peekEl.hidden = true;
  peek = null;
}
// Freeze: stop mirroring, drop the caret, leave the card up until dismissed.
function freezePeek() {
  if (!peek) return;
  peek.caret.remove();
  peek = null;
}
function peekDelta(text) {
  if (!peek) return;
  peek.raw += text;
  peek.body.innerHTML = mdLite(peek.raw);
  (peek.body.lastElementChild || peek.body).appendChild(peek.caret);
  peek.body.scrollTop = peek.body.scrollHeight;
}
// The dock mirrors the same text deltas as one flat string, so a retraction has
// to cut it back too — to the last tool block, not to zero: text the chat pane
// already committed above one still stands.
function peekMark() { if (peek) peek.mark = peek.raw.length; }
function peekRollback() {
  if (!peek) return;
  peek.raw = peek.raw.slice(0, peek.mark);
  peek.body.innerHTML = mdLite(peek.raw);
  (peek.body.lastElementChild || peek.body).appendChild(peek.caret);
}
// `done` upgrade: the server's canonical OFM render (wikilinks, callouts, math),
// same swap the chat pane does. Also covers no-delta turns (raw still empty).
function peekDone(ev) {
  if (!peek) return;
  if (ev.html || ev.answer) peek.body.innerHTML = ev.html || escapeHtml(ev.answer);
  freezePeek();
}
function peekError(msg) {
  if (!peek) return;
  peek.body.classList.add("error");
  peek.body.textContent = "error: " + msg;
  peek = null; // frozen; card stays until dismissed
}
$("#peek-open-chat").addEventListener("click", () => {
  document.querySelector('.tab[data-tab="chat"]').click(); // tab handler closes the peek
});
$("#peek-close").addEventListener("click", closePeek);

// --- note panel resize (drag left edge, clamped) ----------------------------
const NOTE_MIN_W = 280, NOTE_MAX_W = 800;
const savedNoteWidth = parseInt(localStorage.getItem("note-width"), 10);
if (savedNoteWidth) notePanel.style.width = Math.min(NOTE_MAX_W, Math.max(NOTE_MIN_W, savedNoteWidth)) + "px";
// Read the rendered width, not the inline style: with no saved width the inline
// style is "" and the old `|| 420` fallback set --note-w to 420 while the panel
// rendered at its stylesheet width, so the header and dock reserved 210px too
// little and the drawer covered #stop and #dock-send on every fresh profile.
const syncNoteW = () => setNoteW(Math.round(notePanel.getBoundingClientRect().width));
syncNoteW();
// Toasts now hang under the header instead of over the composer, so they need its
// REAL height: the strip wraps to two rows when the drawer is open on a narrow
// window, and a hardcoded offset would put them on top of it there.
const headerEl = document.querySelector("header");
// Unrounded: this also sets the note drawer's title strip, so that the two bands
// across the top of the window meet flush. Math.round turned a 36.5px header
// into a 37px strip beside it, and half a pixel of step is still a step. The
// toast offset that first needed this value does not care about the fraction.
const syncHeaderH = () => document.documentElement.style.setProperty(
  "--header-h", headerEl.getBoundingClientRect().height + "px");
syncHeaderH();
new ResizeObserver(syncHeaderH).observe(headerEl);
// The drawer's width is viewport-relative, so it changes with the window.
// --note-w drives the header and dock insets, so it has to follow or they reserve
// the wrong gap and the drawer covers #stop / #dock-send again. fitPanes() is the
// one that re-negotiates against the prose floor; syncNoteW is the fallback for a
// resize with the drawer closed.
window.addEventListener("resize", () => { syncNarrow(); fitPanes(); syncNoteW(); });
let resizingNote = false; // guards the outside-click-closes handler below: a drag
                           // that ends outside #note-panel fires a "click" there too
$("#note-resize").addEventListener("mousedown", (e) => {
  e.preventDefault();
  resizingNote = true;
  const startX = e.clientX, startWidth = notePanel.getBoundingClientRect().width;
  const onMove = (e2) => {
    // The drag is clamped by the same prose floor the automatic fit obeys, so a
    // user cannot hand-drag the transcript down to four words a line either.
    const cap = Math.max(MIN_DRAWER, drawerBudget(!document.body.classList.contains("sidebar-collapsed")));
    const w = Math.min(NOTE_MAX_W, cap, Math.max(NOTE_MIN_W, startWidth + (startX - e2.clientX)));
    notePanel.style.width = w + "px";
    // Read the rendered width, not the requested one: max-width can clamp the
    // drawer on a narrow window, and --note-w must never disagree with it.
    setNoteW(Math.round(notePanel.getBoundingClientRect().width));
  };
  const onUp = () => {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    localStorage.setItem("note-width", Math.round(notePanel.getBoundingClientRect().width));
    setTimeout(() => { resizingNote = false; }, 0); // clear after this click event finishes
  };
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
});
// One delegated handler: .note-link (chat OR in-panel → in-place nav) opens the
// drawer; a click outside an open drawer closes it. The sidebar and the dock
// are persistent instruments — picking a note, toggling a folder, or typing a
// question about the open note must not close the drawer or reset the graph
// focus, so they never count as "outside". Neither does the reopen button
// (its own listener would immediately fight the close).
document.addEventListener("click", (e) => {
  if (resizingNote) return;
  // dismiss the explore note-search dropdown on any click outside it (a result
  // click runs its own handler first, so pickNote still fires)
  if (!e.target.closest("#node-search-wrap")) closeNodeResults();
  // A citation and a change are two different questions, so they get two
  // different answers. `.note-link` asks what a note says, and opens the note.
  // `.wc-open` is the path on a write card, where the only thing worth reading
  // is what actually changed rather than what was claimed — so it opens the
  // diff, which falls back to the note when the session holds no baseline.
  const changed = e.target.closest(".wc-open");
  if (changed) { e.preventDefault(); openDiff(changed.dataset.path); return; }
  const link = e.target.closest(".note-link");
  if (link) { e.preventDefault(); openNote(link.dataset.path); return; }
  // An external link opens in its own tab. The app has no internal <a href> of
  // its own — every in-app move is JS — so any href in the flow came out of a
  // rendered answer or note, and following it in place tore down the SPA: the
  // turn, the open drawer and the graph focus all went with it. Scoped to
  // http(s) on purpose: a `[text](nota.md)` still resolves against the origin
  // the way it does today, rather than opening a new tab on a 404.
  const ext = e.target.closest('a[href^="http:"], a[href^="https:"]');
  if (ext) { e.preventDefault(); window.open(ext.href, "_blank", "noopener"); return; }
  if (notePanel.classList.contains("open") &&
      !e.target.closest("#note-panel") && !e.target.closest("#sidebar") &&
      !e.target.closest("#dock") && !e.target.closest("#note-last")) closeNote();
});
$("#note-close").addEventListener("click", closeNote);
// (Escape is handled once, at the bottom of this file, in priority order.)
// Graph node clicks (in the iframe) post a message up when embedded.
window.addEventListener("message", (e) => {
  if (!e.data) return;
  if (e.data.type === "silica-open-note") openNote(e.data.path);
  // Graph nodes and map cards POINT rather than name, so they fill the work
  // panel - on every view, not only explore. A ghost node arrives with no path
  // at all, and the panel is the only surface that can say anything about an
  // unresolved link. The second click on the same node arrives separately, as
  // silica-open-note above.
  if (e.data.type === "silica-open-context") showNode(e.data);
  // Clicking the background is a deselection, and a panel still showing the
  // node you just dropped is the app disagreeing with the view.
  if (e.data.type === "silica-clear-node") announceNode(null, null);
  // The frame answers with the renderer it actually built.
  if (e.data.type === "silica-renderer") syncRenderer(e.data.mode);
});

// --- session bootstrap (re-render server-side history; never resets on load) -
async function loadVault() {
  try {
    const r = await fetch("/messages");
    setVaultPath(r.headers.get("X-Silica-Vault") || "");
    setCtxTokens(r.headers.get("X-Silica-Context-Tokens"), r.headers.get("X-Silica-Max-Context-Tokens"),
                 jsonHeader(r, "X-Silica-Context-Parts"), r.headers.get("X-Silica-Compact-At"));
    const msgs = await r.json();
    log.innerHTML = "";
    // One reply is ONE bubble. The model's text arrives split around its own
    // tool calls, so the history holds several assistant messages per turn —
    // rendering each as its own bubble made a single answer read as three
    // separate replies, with the steps between them missing entirely.
    let turn = null;
    for (const m of msgs) {
      if (m.role === "user") { turn = null; bubble("user").textContent = m.content; continue; }
      if (!turn) {
        turn = { body: bubble("silica"), raw: [] };
        const t = turn;
        addCopyBtn(t.body, () => t.raw.join("\n\n"));
      }
      // Thinking first, then what it produced — the order the stream had. It
      // replays collapsed: the live block is open only while it is the tail.
      if (m.thinking) {
        const d = document.createElement("details");
        d.className = "thinking";
        d.innerHTML = '<summary>thinking</summary><div class="thinking-body"></div>';
        d.querySelector(".thinking-body").textContent = m.thinking;
        turn.body.appendChild(d);
      }
      if (m.content) {
        const seg = document.createElement("div");
        seg.className = "stream-text";
        seg.innerHTML = m.html || escapeHtml(m.content);
        turn.body.appendChild(seg);
        turn.raw.push(m.content);
      }
      if (m.tools && m.tools.length) {
        const g = document.createElement("div");
        g.className = "tools";
        for (const t of m.tools) {
          if (!t.summary) {
            // Same row object as a live turn, so a reloaded chat keeps the shape
            // it had while streaming. It opens onto nothing: a stored assistant
            // message holds the call it made, not what came back, and a card
            // that says "no output" for a read that returned a whole note would
            // be inventing an absence.
            const row = makeToolRow(toolLabel(t));
            row.finish(t.error ? "error" : "done");
            g.appendChild(row.el);
            continue;
          }
          // The run's outcome, restated from the stored tool result. Without it
          // a reloaded chat could only say the injector had run — not what it
          // wrote, or which chunks died.
          const d = document.createElement("div");
          d.className = "tool tool-pipeline collapsed " + t.summary.kind;
          d.innerHTML = `<div class="pipe-head"><span class="pipe-title"></span></div>`;
          d.querySelector(".pipe-title").textContent = injectorSummaryLine(t.target || "?", t.summary);
          if (t.summary.failed_chunks.length) {
            const f = document.createElement("div");
            f.className = "pipe-failed";
            f.textContent = t.summary.failed_chunks.map((x) => `✗ ${x.chunk}${x.phase ? " " + x.phase : ""}`).join(" · ");
            d.appendChild(f);
            d.classList.remove("collapsed");
          }
          g.appendChild(d);
        }
        turn.body.appendChild(g);
      }
    }
    log.scrollTop = log.scrollHeight;
  } catch { notify("couldn't load the conversation"); }
}
// --- quick-action launch pad (empty chat only; CSS collapses it on first turn).
// A segmented control, not four buttons: the pill says what the next message
// will do. It is DERIVED, never stored — from the command already in the box,
// or from files waiting to be nucleated — so a segment cannot claim a mode the
// composer isn't in. Picking a segment only prefills; the user still hits enter.
const qaTrack = $(".qa-track");
const qaPill = $("#qa-pill");

function syncQuick() {
  const cmd = (input.value.match(/^\/[a-z-]+/) || [""])[0];
  const want = staged.length ? "/nucleate" : cmd; // staged files outrank a typed command
  const segs = Array.from(qaTrack.querySelectorAll(".qa"));
  const on = segs.find((b) => b.dataset.action === want) || segs[0]; // unknown command → ask
  segs.forEach((b) => b.setAttribute("aria-pressed", String(b === on)));
  // The pill sits at the track's padding-box origin, so the offset is just the
  // segment's rect minus that origin (clientLeft/Top are the track's border).
  // Both axes: the track wraps on a narrow pane and centres each row, so the
  // segments move vertically AND horizontally under it.
  const a = on.getBoundingClientRect(), t = qaTrack.getBoundingClientRect();
  const x = a.left - t.left - qaTrack.clientLeft, y = a.top - t.top - qaTrack.clientTop;
  // Vertically the pill is the full height of its ROW of the track, not of the
  // segment: it has to read as a lens sliding along the container. The inset is
  // measured off the first segment (always on row one), never hardcoded.
  const pad = segs[0].getBoundingClientRect().top - t.top - qaTrack.clientTop;
  qaPill.style.width = on.offsetWidth + "px";
  qaPill.style.height = on.offsetHeight + 2 * pad + "px";
  qaPill.style.transform = `translate(${x}px, ${y - pad}px)`;
}

$("#quick-actions").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-action]");
  if (!btn) return;
  const a = btn.dataset.action;
  input.value = input.value.replace(/^\/[a-z-]*\s*/, ""); // swap the command, keep the text
  if (a) input.value = a + " " + input.value;
  input.focus();
  autoGrow(input);
  renderCommands(input.value); // syncQuick runs from here
});

syncQuick();
// Lexend loads async and the segments get wider when it lands; the track also
// re-wraps whenever the sidebar or the drawer renegotiates the pane. Both are
// resizes of the track, so one observer covers them.
new ResizeObserver(syncQuick).observe(qaTrack);

// --- the chat's model line ---------------------------------------------------
// Its own cheap read, kept separate from /settings: that one probes four
// endpoints for their model lists, and the line must not wait seconds to say
// which model answers you.
//
// The worker half only appears when there IS a second model: it defaults to
// empty and every call site falls back to the chat model, so printing the same
// name twice would claim two models where one is configured.
async function loadConfig() {
  try {
    const c = await (await fetch("/config")).json();
    const short = (m) => (m || "").split("/").pop();
    const box = $("#chat-models");
    box.textContent = "";
    const pair = (lbl, val) => {
      box.appendChild(mkEl("span", "cm-lbl", lbl));
      box.appendChild(mkEl("span", "cm-val", val));
    };
    pair("model", c.model ? short(c.model) : "no model");
    if (c.worker_model && c.worker_model !== c.model) pair("worker", short(c.worker_model));
  } catch { notify("couldn't load session config"); }
}
$("#chat-models").addEventListener("click", () => openSettings());
$("#metrics-cancel").addEventListener("click", () => {
  if (metricsAbort) metricsAbort.abort();
});

// --- help panel -------------------------------------------------------------
// There was no help surface anywhere in the app: no shortcut list, no tour, no
// docs link, and twelve buttons whose only label was a `title`. This is the
// smallest thing that answers "what can I do here" without leaving the window.
const helpPanel = $("#help-panel");
const helpBtn = $("#help-btn");
function closeHelpPanel() {
  helpPanel.hidden = true;
  helpBtn.setAttribute("aria-expanded", "false");
}
helpBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  const opening = helpPanel.hidden;
  helpPanel.hidden = !opening;
  helpBtn.setAttribute("aria-expanded", opening ? "true" : "false");
});

document.addEventListener("click", (e) => {
  if (!helpPanel.hidden && !e.target.closest("#help-panel") && !e.target.closest("#help-btn"))
    closeHelpPanel();
  if (!e.target.closest("#ctx-panel") && !e.target.closest("#ctx-ring")) closeCtxPanel();
  if (!noticesPanel.hidden && !e.target.closest("#notices-panel") && !e.target.closest("#notices-btn"))
    closeNotices();
});

$("#ctx-ring").addEventListener("click", (e) => {
  e.stopPropagation();
  const panel = $("#ctx-panel");
  const opening = panel.hidden;
  if (opening) renderCtxPanel();
  panel.hidden = !opening;
  $("#ctx-ring").setAttribute("aria-expanded", opening ? "true" : "false");
});
// --- dictation: microphone → 16 kHz mono WAV → /stt --------------------------
// Whisper transcribes a clip in one pass, so there is no live partial text to
// stream in and the button has to carry the state on its own: idle, recording,
// transcribing. The WAV conversion happens here rather than on the server
// because MediaRecorder can only produce webm/opus, whisper.cpp's server reads
// WAV unless it was built with ffmpeg, and converting in the browser costs a
// dependency on neither side.
const sttPanel = $("#stt-panel");
// A recording nobody stopped is otherwise a twenty-minute upload and a wait to
// match, so the take ends itself.
const MIC_MAX_MS = 60000;

function showSttPanel(why) {
  $("#stt-why").textContent = why;
  sttPanel.hidden = false;
}
$("#stt-close").addEventListener("click", () => { sttPanel.hidden = true; });

// A success is cached for the page's life; a failure is not. Someone who reads
// the panel, starts whisper-server and clicks again should get a microphone,
// not the same panel until they reload.
let sttProbe = null;
async function sttAvailable() {
  if (sttProbe) return sttProbe;
  const answer = await fetch("/stt")
    .then((r) => r.json())
    .catch(() => ({ ok: false, detail: "the silica server did not answer" }));
  if (answer.ok) sttProbe = answer;
  return answer;
}

// Canonical 44-byte header + PCM16, written out by hand: the alternative is a
// dependency for twenty lines of byte-poking.
function wavFromPcm(samples, rate) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const ascii = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  ascii(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); ascii(8, "WAVE");
  ascii(12, "fmt "); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);   // PCM
  view.setUint16(22, 1, true);   // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);   // block align
  view.setUint16(34, 16, true);  // bits
  ascii(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view.buffer], { type: "audio/wav" });
}

// Resample and downmix in one render pass. Hand-rolling either would be a worse
// filter than the one the browser already ships.
async function toWav16k(blob) {
  const ctx = new AudioContext();
  let decoded;
  try { decoded = await ctx.decodeAudioData(await blob.arrayBuffer()); }
  finally { ctx.close(); }
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  return wavFromPcm((await off.startRendering()).getChannelData(0), 16000);
}

// Whisper mishears proper nouns and invents punctuation, and what lands here can
// become a write to the vault, so the text goes in for review and never sends
// itself. At the cursor, so dictating into a half-typed message works.
function insertAtCursor(box, text) {
  if (!text) { notify("nothing was transcribed", "info"); return; }
  const at = box.selectionStart ?? box.value.length;
  const before = box.value.slice(0, at);
  const after = box.value.slice(box.selectionEnd ?? at);
  const sep = before && !/\s$/.test(before) ? " " : "";
  box.value = before + sep + text + after;
  const caret = (before + sep + text).length;
  box.setSelectionRange(caret, caret);
  box.focus();
  box.dispatchEvent(new Event("input", { bubbles: true })); // autogrow + send state
}

function attachMic(box, btn) {
  let rec = null;
  let chunks = [];
  let cap = null;
  const stop = () => { if (rec && rec.state !== "inactive") rec.stop(); };

  btn.addEventListener("click", async () => {
    if (rec) { stop(); return; } // the second click ends the take
    const avail = await sttAvailable();
    if (!avail.ok) {
      showSttPanel(avail.detail || "no transcription endpoint is answering");
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      // Denied, or no device at all: the browser's to fix, not silica's.
      notify("no microphone: " + plainError((err && err.message) || err));
      return;
    }
    chunks = [];
    rec = new MediaRecorder(stream);
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = async () => {
      clearTimeout(cap);
      // Release the device, which is also what drops the browser's own
      // recording indicator — leaving it lit would say silica is still listening.
      stream.getTracks().forEach((t) => t.stop());
      rec = null;
      btn.classList.remove("recording");
      if (!chunks.length) return;
      btn.classList.add("busy");
      announce("transcribing");
      try {
        const wav = await toWav16k(new Blob(chunks, { type: chunks[0].type }));
        const form = new FormData();
        form.append("audio", wav, "clip.wav");
        const resp = await fetch("/stt", { method: "POST", body: form });
        const body = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(body.detail || resp.statusText);
        insertAtCursor(box, (body.text || "").trim());
      } catch (err) {
        notify("dictation failed: " + plainError((err && err.message) || err));
      } finally {
        btn.classList.remove("busy");
      }
    };
    rec.start();
    btn.classList.add("recording");
    announce("recording, click again to stop");
    cap = setTimeout(stop, MIC_MAX_MS);
  });
}

document.querySelectorAll(".mic").forEach((b) => {
  const box = document.getElementById(b.dataset.for);
  if (box) attachMic(box, b);
});

// --- settings ----------------------------------------------------------------
// Rows are built from /settings, never hardcoded here: the table lives in
// silica/ui/web/settings.py, so what the panel offers and what the server will
// accept are the same list. Saving is per row — no save button, no dirty state,
// no exit dialog. Toggles and pick-lists apply at once, text fields on blur or
// Enter (the browser's own `change`), never on every keystroke.
const stModal = $("#st-modal");
const stBackdrop = $("#st-backdrop");
const stSheet = $("#st-sheet");
const stPanel = $("#st-panel");
const stTabs = $("#st-tabs");
const stSearch = $("#st-search");
const settingsBtn = $("#settings-btn");
const ST_EXTRA = ["Endpoints", "Diagnostics", "About"];
const stState = { data: null, section: "Session", controls: [], uid: 0 };

// A key is shown as its own head and tail: enough to tell an OpenRouter key from
// a stale one without putting the secret on screen.
function maskKey(v) {
  if (!v) return "";
  return v.length <= 12 ? "•".repeat(8) : v.slice(0, 5) + "••••" + v.slice(-4);
}

function stEl(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
}

function stSectionEl(name) {
  const s = stEl("section", "st-section");
  s.dataset.section = name;
  s.appendChild(stEl("div", "st-section-title", name));
  return s;
}

function stNote(rowEl, cls, text) {
  const note = rowEl.querySelector(".st-note");
  note.className = "st-note " + cls;
  note.textContent = text;
}

function stValueOf(row, input) {
  return row.kind === "toggle" ? String(input.checked) : input.value;
}

function stRevert(row, input) {
  const prev = input.dataset.prev || "";
  if (row.kind === "toggle") input.checked = prev === "true";
  else input.value = prev;
}

// One control per kind. Every list row is an <input list> with a <datalist>
// rather than a dropdown: _endpoint_model_ids returns [] on any error, so the
// list is empty exactly when the endpoint is down — which is the moment you
// opened this panel to fix it. A datalist degrades to a text field on its own.
function stBuildControl(row, rowEl, labelEl) {
  if (row.kind === "readonly") {
    // Registered like any control so a write that derives it (safe mode sets
    // write_dir) refreshes the readout instead of leaving it stale until reopen.
    const ro = stEl("span", "st-ro", row.value || "—");
    stState.controls.push({ row, input: ro });
    return ro;
  }
  const id = `st-c${++stState.uid}`;
  labelEl.setAttribute("for", id);
  let input;
  if (row.kind === "toggle") {
    input = stEl("input");
    input.type = "checkbox";
    input.checked = row.value === "true";
  } else if (row.kind === "enum") {
    input = stEl("select");
    const opts = row.options.includes(row.value) || !row.value
      ? row.options : [row.value, ...row.options];
    for (const o of opts) input.appendChild(new Option(o, o));
    input.value = row.value;
  } else {
    input = stEl("input");
    input.type = "text";
    input.spellcheck = false;
    input.autocomplete = "off";
    if (row.kind === "int") input.inputMode = "numeric";
    input.value = row.kind === "secret" ? maskKey(row.value) : row.value;
    if (row.options.length) {
      const dl = stEl("datalist");
      dl.id = id + "-list";
      for (const o of row.options) dl.appendChild(new Option(o, o));
      input.setAttribute("list", dl.id);
      rowEl.appendChild(dl);
    }
  }
  input.id = id;
  input.dataset.prev = stValueOf(row, input);
  if (row.locked) {
    input.disabled = true;
    input.dataset.locked = "1";
  }
  input.addEventListener("change", () => stCommit(row, input, rowEl));
  stState.controls.push({ row, input });
  return input;
}

function stRowEl(row) {
  const el = stEl("div", "st-row");
  el.dataset.key = row.key;
  el.dataset.search = `${row.label} ${row.help} ${row.key}`.toLowerCase();
  const label = stEl("div", "st-label");
  const name = stEl("label", "st-name", row.label);
  label.appendChild(name);
  if (row.help) label.appendChild(stEl("div", "st-help", row.help));
  if (row.warn) label.appendChild(stEl("div", "st-warn", "⚠ " + row.warn));
  el.appendChild(label);

  const ctl = stEl("div", "st-ctl");
  const input = stBuildControl(row, el, name);
  ctl.appendChild(input);
  // A key already set is shown masked and read-only: the eye is how you say
  // "I mean to replace this", so a mask can never be saved as a value.
  if (row.kind === "secret" && row.value && !row.locked) {
    const eye = stEl("button", "st-eye", "👁");
    eye.type = "button";
    eye.title = "reveal and replace";
    eye.setAttribute("aria-label", `reveal ${row.label}`);
    input.readOnly = true;
    eye.addEventListener("click", () => {
      input.readOnly = false;
      input.value = row.value;
      input.dataset.prev = row.value;
      eye.remove();
      input.focus();
    });
    ctl.appendChild(eye);
  }
  el.appendChild(ctl);

  const note = stEl("div", "st-note");
  if (row.locked) note.textContent = `🔒 defined in the environment (${row.key})`;
  else if (row.kind === "secret" && row.value) note.textContent = "set · reveal to replace";
  el.appendChild(note);
  return el;
}

// Every control bound to a key this write touched, resynced: `thinking` is both
// the session's live toggle and a display preference, and a provider change
// drags the model and the base url with it.
function stSyncKey(key, value) {
  for (const { row, input } of stState.controls) {
    if (row.key !== key) continue;
    row.value = value;
    if (row.kind === "readonly") { input.textContent = value || "—"; continue; }
    if (row.kind === "toggle") input.checked = value === "true";
    else if (row.kind === "secret") input.value = input.readOnly ? maskKey(value) : value;
    else input.value = value;
    input.dataset.prev = stValueOf(row, input);
  }
}

async function stCommit(row, input, rowEl) {
  const value = stValueOf(row, input);
  if (value === input.dataset.prev) return;
  if (row.warn && !(await stConfirmRow(row, value))) {
    stRevert(row, input);
    stNote(rowEl, "", "");
    return;
  }
  stNote(rowEl, "pending", "saving…");
  let resp = null, data = null;
  try {
    resp = await fetch(row.confirm ? "/settings/confirm" : "/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: row.key, value }),
    });
    data = await resp.json();
  } catch { /* falls through to the failure branch */ }
  if (!resp || !resp.ok || (data && data.ok === false)) {
    const why = (data && (data.detail || data.error)) || "could not save";
    stNote(rowEl, "bad", "✕ " + why);
    stRevert(row, input);
    if (resp && resp.status === 409) stSetBusy(true);
    return;
  }
  const keys = Object.keys(data.values || {});
  stNote(rowEl, "good", keys.length > 1
    ? `✓ ${keys.length} keys saved to ${data.path}`
    : `✓ saved to ${data.path}`);
  for (const [k, v] of Object.entries(data.values || {})) stSyncKey(k, v);
  for (const n of data.notes || []) notify(n, "info");
  if (data.reindex) {
    const n = data.reindex.indexed ?? data.reindex.embedded ?? "";
    stNote(rowEl, "good", `✓ saved to ${data.path} · re-indexed${n === "" ? "" : " " + n} notes`);
  }
  if (row.key === "SILICA_MODEL" || row.key === "SILICA_PROVIDER") loadConfig();
  // The graph document bakes these in at render time, so the change only lands
  // on a rebuild. Stale it, and rebuild now if that view is the one on screen.
  if (row.key === "SILICA_THEME") applyThemePref((data.values || {})[row.key] || value);
  if (row.key === "SILICA_GRAPH_PARTICLES" || row.key === "SILICA_GRAPH_SHADING") {
    graphStale = true;
    if (activeTab === "graph" && graphMode === "graph") setGraphMode("graph");
  }
  if (row.key === "SILICA_VAULT") { loadVault(); loadVaultInfo(); loadSessions(); loadChanges(); }
}

// --- sheets: confirmations and the bug report, inside the modal so the focus
// trap keeps holding.
let stSheetResolve = null;
function openSheet(title, body, actions) {
  $("#st-sheet-title").textContent = title;
  const bodyEl = stSheet.querySelector(".st-sheet-body");
  const actEl = stSheet.querySelector(".st-sheet-actions");
  bodyEl.innerHTML = "";
  actEl.innerHTML = "";
  bodyEl.appendChild(body);
  for (const [label, fn, kind] of actions) {
    const b = stEl("button", "st-btn " + (kind || ""), label);
    b.type = "button";
    b.addEventListener("click", () => fn());
    actEl.appendChild(b);
  }
  stSheet.hidden = false;
  actEl.querySelector("button").focus();
}

function closeSheet(answer) {
  stSheet.hidden = true;
  const resolve = stSheetResolve;
  stSheetResolve = null;
  if (resolve) resolve(!!answer);
}

// The consequence, named, before the change happens — and the button says what
// it will do, not "ok".
const ST_CONFIRM = {
  SILICA_VAULT: (row, value) => [
    "switch vault?",
    `silica will read and write ${value} instead.\nevery index is rebuilt for the new folder.`,
    "switch",
  ],
  SILICA_EMBEDDING_MODEL: (row) => [
    "change the embedding model?",
    `the vectors already stored were produced by ${row.value || "another model"}.\n` +
    "new queries cannot be compared against them.\n" +
    "repairing this means a full re-index, which takes a while on a large vault.",
    "change and re-index",
  ],
  SILICA_EMBEDDING_BASE_URL: (row) => [
    "change where embeddings come from?",
    "a different server means different vectors, even under the same model name.\n" +
    "repairing this means a full re-index.",
    "change and re-index",
  ],
  SILICA_COOCCURRENCE_LANG: () => [
    "change the vault language?",
    "the language is frozen per vault. changing it after notes exist\n" +
    "makes old keywords disagree with new ones.",
    "change",
  ],
};

function stConfirmRow(row, value) {
  const build = ST_CONFIRM[row.key] || (() => [`change ${row.label}?`, row.warn, "change"]);
  const [title, body, ok] = build(row, value);
  const el = stEl("p", "st-sheet-text", body);
  return new Promise((resolve) => {
    stSheetResolve = resolve;
    openSheet(title, el, [
      ["cancel", () => closeSheet(false)],
      [ok, () => closeSheet(true), "primary"],
    ]);
  });
}

// --- the three sections that are not config rows ------------------------------
const ST_DOT = { ok: "●", warn: "◐", fail: "○", unknown: "○" };

function stInfoRow(host, label, value, cls) {
  const el = stEl("div", "st-row" + (cls ? " " + cls : ""));
  el.dataset.search = `${label} ${value}`.toLowerCase();
  el.appendChild(stEl("div", "st-label", label));
  el.appendChild(stEl("div", "st-ctl-text", value));
  host.appendChild(el);
  return el;
}

async function renderEndpoints(host) {
  host.querySelectorAll(".st-row, .st-note-line").forEach((e) => e.remove());
  host.appendChild(stEl("div", "st-note-line",
    "reachability is checked with a real request, not an open port"));
  let rows = [];
  try { rows = await (await fetch("/endpoints")).json(); }
  catch { host.appendChild(stEl("div", "st-note-line", "could not probe the endpoints")); return; }
  for (const e of rows) {
    const row = stEl("div", "st-row");
    row.dataset.search = `${e.label} ${e.url} endpoint`.toLowerCase();
    const label = stEl("div", "st-label");
    label.appendChild(stEl("span", "st-name", e.label));
    label.appendChild(stEl("div", "st-help", e.url || "not configured"));
    row.appendChild(label);
    const state = stEl("div", "st-ctl-text " + (e.up ? "up" : "down"));
    state.textContent = e.up
      ? `● up${e.models ? ` · ${e.models} model${e.models === 1 ? "" : "s"}` : ""}`
      : `○ down${e.command || !e.local ? "" : " · no start command set"}`;
    row.appendChild(state);
    const note = stEl("div", "st-note");
    if (e.command) note.textContent = `start command read-only · edit ${e.command_key} in the .env`;
    row.appendChild(note);
    if (!e.up && e.local && e.command) {
      const start = stEl("button", "st-btn", "start");
      start.type = "button";
      start.addEventListener("click", async () => {
        start.disabled = true;
        note.className = "st-note pending";
        note.textContent = `starting ${e.label}… loading a model takes a while`;
        let out = null;
        try {
          out = await (await fetch("/endpoints/start", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ label: e.label }),
          })).json();
        } catch { /* reported below */ }
        if (out && out.ok) renderEndpoints(host);
        else {
          note.className = "st-note bad";
          note.textContent = (out && out.error)
            || `${e.label} did not come up · see ${out ? out.log : "~/.silica/logs"}`;
          start.disabled = false;
        }
      });
      state.appendChild(start);
    }
    host.appendChild(row);
  }
}

async function renderDiagnostics(host) {
  host.querySelectorAll(".st-row, .st-note-line").forEach((e) => e.remove());
  let rows = [];
  try { rows = await (await fetch("/health?all=1")).json(); }
  catch { host.appendChild(stEl("div", "st-note-line", "could not run the checks")); return; }
  if (!rows.length) { host.appendChild(stEl("div", "st-note-line", "everything checks out")); return; }
  for (const r of rows) {
    const el = stInfoRow(host, r.name, `${ST_DOT[r.status] || "○"} ${r.detail}`, "st-check-" + r.status);
    if (r.hint) el.appendChild(stEl("div", "st-note", r.hint));
  }
}

function renderAbout(host, data) {
  host.querySelectorAll(".st-row, .st-note-line").forEach((e) => e.remove());
  stInfoRow(host, "version", `silica ${data.version}`);
  stInfoRow(host, "updates", data.behind
    ? `${data.behind} commit${data.behind === 1 ? "" : "s"} behind · update with \`silica update\``
    : "up to date");
  const row = stEl("div", "st-row");
  row.dataset.search = "report a bug issue github";
  row.appendChild(stEl("div", "st-label", "report a bug"));
  const btn = stEl("button", "st-btn st-safe", "report a bug");
  btn.type = "button";
  btn.addEventListener("click", () => openBugReport(data.issues_url));
  const ctl = stEl("div", "st-ctl");
  ctl.appendChild(btn);
  row.appendChild(ctl);
  host.appendChild(row);
}

// The attached payload is built by the server, not read off this panel: the API
// key fields are one querySelector away from here, and an issue is public.
async function openBugReport(fallbackUrl) {
  let data = { payload: "", issues_url: fallbackUrl };
  try { data = await (await fetch("/bug_report")).json(); } catch { /* file it bare */ }
  const body = stEl("div", "st-bug");
  body.appendChild(stEl("label", "st-bug-label", "what happened?"));
  const what = stEl("textarea", "st-bug-what");
  what.rows = 4;
  what.placeholder = "what you did, what you expected, what happened instead";
  body.appendChild(what);
  body.appendChild(stEl("label", "st-bug-label", "this will be attached · edit it if you like"));
  const payload = stEl("textarea", "st-bug-payload");
  payload.rows = 8;
  payload.value = data.payload;
  body.appendChild(payload);
  body.appendChild(stEl("div", "st-note-line",
    "your vault path is shortened to ~ · api keys are never included"));
  openSheet("report a bug", body, [
    ["cancel", () => closeSheet(false)],
    ["open on github ↗", () => {
      const title = (what.value.trim().split("\n")[0] || "bug report").slice(0, 80);
      const text = `${what.value.trim()}\n\n\`\`\`\n${payload.value}\n\`\`\`\n`;
      window.open(
        `${data.issues_url}?title=${encodeURIComponent(title)}&body=${encodeURIComponent(text)}`,
        "_blank", "noopener");
      closeSheet(true);
    }, "primary"],
  ]);
  what.focus();
}

// --- render, filter, open, close ---------------------------------------------
function stRender(data) {
  stState.data = data;
  stState.controls = [];
  stPanel.innerHTML = "";
  stTabs.innerHTML = "";
  $("#st-env").textContent = data.env_path;
  for (const section of data.sections) {
    const el = stSectionEl(section.name);
    for (const row of section.rows) el.appendChild(stRowEl(row));
    stPanel.appendChild(el);
  }
  for (const name of ST_EXTRA) stPanel.appendChild(stSectionEl(name));
  stPanel.appendChild(stEl("div", "st-empty"));
  for (const name of [...data.sections.map((s) => s.name), ...ST_EXTRA]) {
    const b = stEl("button", "st-tab", name);
    b.type = "button";
    b.dataset.section = name;
    b.addEventListener("click", () => stShow(name));
    stTabs.appendChild(b);
  }
  renderAbout(stPanel.querySelector('[data-section="About"]'), data);
  renderEndpoints(stPanel.querySelector('[data-section="Endpoints"]'));
  const diagnostics = stPanel.querySelector('[data-section="Diagnostics"]');
  // The checks are a snapshot of a machine that keeps changing — starting the
  // server this panel just told you was down is the whole point.
  const recheck = stEl("button", "st-btn st-safe", "recheck");
  recheck.type = "button";
  recheck.addEventListener("click", () => renderDiagnostics(diagnostics));
  diagnostics.querySelector(".st-section-title").appendChild(recheck);
  renderDiagnostics(diagnostics);
  stShow(stState.section);
  stSetBusy(data.busy || streaming);
}

function stShow(name) {
  stState.section = name;
  stSearch.value = "";
  stFilter();
  stPanel.scrollTop = 0;
}

// The search reaches every section at once — the rows are already in the DOM,
// so it needs no index and no second surface.
function stFilter() {
  const q = stSearch.value.trim().toLowerCase();
  let hits = 0;
  for (const section of stPanel.querySelectorAll(".st-section")) {
    let any = 0;
    for (const row of section.querySelectorAll(".st-row")) {
      const hit = !q || (row.dataset.search || "").includes(q);
      row.hidden = !hit;
      if (hit) any++;
    }
    section.hidden = q ? !any : section.dataset.section !== stState.section;
    if (!section.hidden) hits += any;
  }
  const empty = stPanel.querySelector(".st-empty");
  empty.hidden = !(q && !hits);
  empty.textContent = `no setting matches "${q}"`;
  for (const b of stTabs.querySelectorAll(".st-tab"))
    setActive(b, !q && b.dataset.section === stState.section);
}

// One rule, not a list of which rows a turn happens to read: that list would rot
// at the first new tool and no test would catch it.
function stSetBusy(busy) {
  $("#st-busy").hidden = !busy;
  // `.st-safe` reads and writes nothing — rechecking the diagnostics or filing a
  // bug is exactly what you want to do while a turn is misbehaving.
  for (const el of stPanel.querySelectorAll("input, select, .st-btn:not(.st-safe), .st-eye")) {
    if (el.dataset.locked === "1") continue;
    el.disabled = !!busy;
  }
}

function stTrapFocus(e) {
  if (e.key !== "Tab" || stModal.hidden) return;
  const scope = stSheet.hidden ? stModal : stSheet;
  const focusable = [...scope.querySelectorAll(
    'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled])'
  )].filter((el) => el.offsetParent !== null && !el.closest("[hidden]"));
  if (!focusable.length) return;
  const first = focusable[0], last = focusable[focusable.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

async function openSettings() {
  stBackdrop.hidden = false;
  stModal.hidden = false;
  settingsBtn.setAttribute("aria-expanded", "true");
  stPanel.innerHTML = "";
  stPanel.appendChild(stEl("div", "st-note-line", "reading your configuration…"));
  stSearch.focus();
  try {
    stRender(await (await fetch("/settings")).json());
  } catch {
    stPanel.innerHTML = "";
    stPanel.appendChild(stEl("div", "st-note-line", "could not read the settings"));
  }
}

function closeSettings() {
  if (!stSheet.hidden) closeSheet(false);
  stModal.hidden = true;
  stBackdrop.hidden = true;
  settingsBtn.setAttribute("aria-expanded", "false");
  // Always the gear, not wherever focus happened to be: it is the control the
  // modal came out of, and it is where a keyboard user expects to land back.
  settingsBtn.focus();
}

settingsBtn.addEventListener("click", () => {
  if (stModal.hidden) openSettings(); else closeSettings();
});
$("#st-close").addEventListener("click", closeSettings);
stBackdrop.addEventListener("click", closeSettings);
stSearch.addEventListener("input", stFilter);
document.addEventListener("keydown", stTrapFocus);

// One Escape handler for the whole app: there used to be two independent ones,
// so a single press with a panel open over a note closed both at once.
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  // The settings modal is the only surface with a backdrop: whatever is under
  // it is unreachable, so it answers first, and its own sheet before it.
  if (!stSheet.hidden) { closeSheet(); return; }
  if (!stModal.hidden) { closeSettings(); return; }
  if (!sttPanel.hidden) { sttPanel.hidden = true; return; }
  if (!helpPanel.hidden) { closeHelpPanel(); return; }
  if (!noticesPanel.hidden) { closeNotices(); return; }
  if (!$("#ctx-panel").hidden) { closeCtxPanel(); return; }
  closeNote();
});

// --- boot health: the doctor's non-ok rows as toasts -------------------------
// The TUI logs a degraded embedder/reranker to stderr; the browser sees none of
// that, so without this a server left down just makes recall quietly worse.
// Boot health is CONFIGURATION, not an event: it is equally true a second after
// load and ten minutes in. As toasts these fired on every single load, stacked
// over whatever the user was reading, and one of them is a five-line JSON hooks
// blob — so the app's resting state was a debug message. They live in the
// sidebar now, where they stay legible and dismissible for as long as they are
// true, and the toast strip goes back to meaning "something just happened".
const noticesBtn = $("#notices-btn");
const noticesPanel = $("#notices-panel");

function closeNotices() {
  noticesPanel.hidden = true;
  noticesBtn.setAttribute("aria-expanded", "false");
}

// The chip IS the count: with nothing to report it is not a chip that says
// zero, it is absent, and the strip goes back to being about the vault.
function setNoticeCount(n) {
  noticesBtn.hidden = !n;
  noticesBtn.textContent = n + (n === 1 ? " notice" : " notices");
  if (!n) closeNotices();
}

noticesBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  const opening = noticesPanel.hidden;
  noticesPanel.hidden = !opening;
  noticesBtn.setAttribute("aria-expanded", opening ? "true" : "false");
});

async function loadHealth() {
  const box = $("#boot-notices");
  try {
    const rows = await (await fetch("/health")).json();
    box.innerHTML = "";
    for (const r of rows) {
      const n = document.createElement("div");
      n.className = "notice";
      const t = document.createElement("div");
      t.className = "notice-text";
      t.textContent = r.name + ": " + r.detail;
      n.appendChild(t);
      if (r.hint) {
        // Hints collapsed behind a disclosure: three stacked notices with a
        // five-line JSON hooks blob were eating half the sidebar on first
        // load. One line each until the user asks for the remedy.
        const d = document.createElement("details");
        const s = document.createElement("summary");
        s.textContent = "how to fix";
        d.appendChild(s);
        const h = document.createElement("div");
        h.className = "notice-hint";
        h.textContent = r.hint;
        d.appendChild(h);
        n.appendChild(d);
      }
      const x = document.createElement("button");
      x.type = "button";
      x.className = "notice-x";
      x.setAttribute("aria-label", "dismiss this notice");
      x.textContent = "✕";
      x.addEventListener("click", () => { n.remove(); setNoticeCount(box.childElementCount); });
      n.appendChild(x);
      box.appendChild(n);
      announce(r.name + ": " + r.detail);
    }
    setNoticeCount(rows.length);
  } catch { /* the page works without the report; don't toast about the toast */ }
}

loadVault();
loadSessions();
loadVaultInfo();
loadVaultBrief(); // a call, so it is its own request: the landing renders without it
loadChanges(); // the server's ledger outlives the tab — a reload keeps the list
loadConfig(); // header shows the active model without opening the panel
loadHealth(); // a chat/embedder/reranker server that isn't up says so, once, here
// --- calendar ----------------------------------------------------------------
// Month grid + week view over GET /calendar (the 4-axis agenda payload), with
// nodus-style lane-packing for multi-day bars: per week row, spans pack into
// the first free lane, Mon–Sun clipped; in month mode lanes cap at 3 and the
// overflow folds into the day's "+N". The agenda panel shows the upcoming 7
// days, or the one day clicked in the grid. POST /reminders every 30 s IS the
// reminder tick (setInterval stays alive in hidden tabs); delivered ones land
// on the toast strip.

let calMode = "month";        // "month" | "week"
let calAnchor = new Date();   // any date inside the visible month/week
let calSelected = null;       // "YYYY-MM-DD" the agenda panel focuses on, or null
let calDays = {};             // date -> DayRow of the visible window
let calUpcoming = null;       // DayRows of today+7, for the default agenda panel
let calBursting = [];         // V6: what the last fortnight of WRITING was about

const CAL_LANE_CAP = 3;       // month mode: visible multi-day lanes per week
const CAL_CHIP_CAP = 2;       // month mode: visible timed chips per day
const CAL_DERIVED_CAP = 3;    // month mode: chips + derived lines together
const CAL_AGENDA_CAP = 6;     // agenda: rows per axis before "show all"

function calFmt(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
function calMonday(d) {
  const x = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  x.setDate(x.getDate() - ((x.getDay() + 6) % 7));
  return x;
}
function calWindow() {
  if (calMode === "week") return { start: calMonday(calAnchor), days: 7 };
  const first = new Date(calAnchor.getFullYear(), calAnchor.getMonth(), 1);
  return { start: calMonday(first), days: 42 };
}

async function loadCalendar() {
  const w = calWindow();
  // Only on a cold open. Every month step refetches, and flashing a spinner
  // over a grid that is already on screen would be the louder of the two lies.
  const frame = $("#cal-loading");
  const cold = !Object.keys(calDays).length;
  if (cold) frame.hidden = false;
  try {
    const [grid, up] = await Promise.all([
      fetch(`/calendar?start=${calFmt(w.start)}&days=${w.days}`).then((r) => r.json()),
      fetch("/calendar?start=today&days=7").then((r) => r.json()),
    ]);
    if (grid.error) { notify(grid.error); return; }
    calDays = {};
    for (const row of grid.days) calDays[row.date] = row;
    calUpcoming = up.error ? null : up.days;
    calBursting = up.bursting || [];
    renderCalendar();
    renderCalAgenda();
  } catch { notify("couldn't load the calendar"); }
  finally { frame.hidden = true; }
}

// A bar is anything that must draw as a span: all-day, or crossing midnight.
function calIsBar(e) {
  return e.all_day || (e.end && e.end.slice(0, 10) !== e.start.slice(0, 10));
}

// nodus layoutCalendarWeek: first-fit lanes over column intervals.
function calPackLanes(spans) {
  spans.sort((a, b) => a.c0 - b.c0 || (b.c1 - b.c0) - (a.c1 - a.c0));
  const lanes = [];
  for (const s of spans) {
    let lane = lanes.findIndex((l) => l.every((o) => o.c1 < s.c0 || o.c0 > s.c1));
    if (lane === -1) { lane = lanes.length; lanes.push([]); }
    lanes[lane].push(s);
    s.lane = lane;
  }
  return lanes.length;
}

// "nucleate `07-part.md` → 2 new, 26 patch, 2 deferred · run 5e88feb0"
// The run id is provenance, not reading matter: it rides the row's tooltip and
// leaves the line. Anything that doesn't match comes through whole rather than
// being dropped — the log's shape is the agent's to change.
function calActivity(s) {
  const at = s.lastIndexOf(" · run ");
  const body = at > 0 ? s.slice(0, at) : s;
  const m = body.match(/^(\S+)\s+`(.+?)`\s*(?:→\s*(.*))?$/);
  if (!m) return { verb: "", what: body, detail: "" };
  return { verb: m[1], what: m[2], detail: (m[3] || "").trim() };
}

// What a day's four axes look like in ONE cell. Events are the only axis that
// is an appointment, so they stay solid chips; the other three are things that
// already happened or are waiting, and they render as quiet counted lines. A
// month of a working vault used to be 42 empty boxes beside an agenda holding
// 60 rows: every axis but `events` was agenda-only.
function calDerived(row) {
  const out = [];
  if (!row) return out;
  if (row.review?.length) {
    out.push({ cls: "due", text: row.review.length + " to review",
               title: row.review.length + " notes the review queue has due" });
  }
  const verbs = new Map();
  for (const a of row.activity || []) {
    const v = calActivity(a).verb || "wrote";
    verbs.set(v, (verbs.get(v) || 0) + 1);
  }
  for (const [v, n] of [...verbs].sort((a, b) => b[1] - a[1])) {
    out.push({ cls: "", text: `${v} ×${n}`, title: `${n} ${v} runs on this day` });
  }
  if (row.notes?.length) {
    out.push({ cls: "", text: row.notes.length + (row.notes.length === 1 ? " note" : " notes"),
               title: row.notes.length + " notes whose claim clock lands here" });
  }
  return out;
}

function calEventEl(e, cls, withTime) {
  const el = document.createElement("div");
  el.className = cls + (e.status ? " " + e.status : "");
  if (withTime && !e.all_day) {
    const t = document.createElement("span");
    t.className = "t";
    t.textContent = e.start.slice(11, 16);
    el.appendChild(t);
  }
  el.appendChild(document.createTextNode((e.status === "done" ? "✓ " : "") + e.title));
  el.title = e.title;
  el.addEventListener("click", (ev) => { ev.stopPropagation(); openNote(e.path); });
  return el;
}

function calBuildWeek(dates, tall) {
  const week = document.createElement("div");
  week.className = "cal-week" + (tall ? " wk" : "");

  // Reconstruct spans from the per-day buckets: same (stem, start) on
  // consecutive days is one occurrence.
  const spans = new Map();
  dates.forEach((date, i) => {
    for (const e of (calDays[date] || {}).events || []) {
      if (!calIsBar(e)) continue;
      const k = e.stem + "|" + e.start;
      const s = spans.get(k) || { c0: i, c1: i, ev: e };
      s.c1 = i;
      spans.set(k, s);
    }
  });
  const bars = [...spans.values()];
  const laneTotal = calPackLanes(bars);
  const laneCap = tall ? laneTotal : Math.min(laneTotal, CAL_LANE_CAP);
  // repeat(0, …) is invalid CSS and would drop the whole declaration
  week.style.gridTemplateRows =
    laneCap > 0 ? `20px repeat(${laneCap}, 20px) 1fr` : "20px 1fr";

  const todayStr = calFmt(new Date());
  const overflow = new Array(7).fill(0);
  const visMonth = calAnchor.getMonth();

  dates.forEach((date, i) => {
    const cell = document.createElement("div");
    cell.className = "cal-day";
    // Explicit column: a cell with a definite row span but an auto column
    // would be pushed past any bar-occupied column by the sparse placement
    // cursor, spilling days 5..7 into implicit tracks (measured: 10 columns).
    cell.style.gridColumn = String(i + 1);
    if (i === 6) cell.classList.add("c7");
    if (date === todayStr) cell.classList.add("today");
    if (date === calSelected) cell.classList.add("selected");
    const d = new Date(date + "T00:00");
    if (calMode === "month" && d.getMonth() !== visMonth) cell.classList.add("other");
    const num = document.createElement("div");
    num.className = "cal-num";
    const nn = document.createElement("span");
    nn.textContent = d.getDate();
    num.appendChild(nn);
    cell.appendChild(num);

    if (laneCap > 0) {
      // Hold the lane band open: chips flow in normal cell layout while bars
      // paint on the overlapping grid rows — without this the first chip
      // renders underneath a bar.
      const spacer = document.createElement("div");
      spacer.style.flex = `0 0 ${laneCap * 20}px`;
      cell.appendChild(spacer);
    }
    const chips = document.createElement("div");
    chips.className = "cal-chips";
    const timed = ((calDays[date] || {}).events || []).filter((e) => !calIsBar(e));
    const cap = tall ? timed.length : CAL_CHIP_CAP;
    timed.slice(0, cap).forEach((e) => chips.appendChild(calEventEl(e, "cal-chip", true)));
    overflow[i] += Math.max(0, timed.length - cap);
    // The three derived axes, under whatever appointments the day holds. They
    // are counts of what is already true, so they never take the space an
    // event wanted: the cap is what is left of the cell after the chips.
    const derived = calDerived(calDays[date]);
    const dcap = tall ? derived.length : Math.max(0, CAL_DERIVED_CAP - Math.min(timed.length, cap));
    derived.slice(0, dcap).forEach((d) => {
      const el = mkEl("div", "cal-sig" + (d.cls ? " " + d.cls : ""), d.text);
      el.title = d.title;
      chips.appendChild(el);
    });
    overflow[i] += Math.max(0, derived.length - dcap);
    cell.appendChild(chips);

    const more = document.createElement("div");
    more.className = "cal-more";
    cell.appendChild(more);

    cell.addEventListener("click", () => {
      calSelected = calSelected === date ? null : date;
      renderCalendar();
      renderCalAgenda();
    });
    week.appendChild(cell);
  });

  for (const s of bars) {
    if (s.lane >= laneCap) {
      for (let c = s.c0; c <= s.c1; c++) overflow[c] += 1;
      continue;
    }
    const bar = calEventEl(s.ev, "cal-bar", false);
    bar.style.gridColumn = `${s.c0 + 1} / ${s.c1 + 2}`;
    bar.style.gridRow = String(s.lane + 2);
    week.appendChild(bar);
  }
  overflow.forEach((n, i) => {
    if (n > 0) week.children[i].querySelector(".cal-more").textContent = `+${n} more`;
  });
  return week;
}

function renderCalendar() {
  const grid = $("#cal-grid");
  grid.replaceChildren();
  const dow = document.createElement("div");
  dow.className = "cal-dow";
  for (const n of ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]) {
    const c = document.createElement("div");
    c.textContent = n;
    dow.appendChild(c);
  }
  grid.appendChild(dow);

  const w = calWindow();
  const dates = [];
  for (let i = 0; i < w.days; i++) {
    const d = new Date(w.start);
    d.setDate(d.getDate() + i);
    dates.push(calFmt(d));
  }
  for (let r = 0; r < dates.length / 7; r++) {
    grid.appendChild(calBuildWeek(dates.slice(r * 7, r * 7 + 7), calMode === "week"));
  }

  const t = $("#cal-title");
  if (calMode === "month") {
    t.textContent = calAnchor.toLocaleString("en-US", { month: "long", year: "numeric" });
  } else {
    const a = calMonday(calAnchor);
    const b = new Date(a); b.setDate(b.getDate() + 6);
    const f = (d) => d.toLocaleString("en-US", { month: "short", day: "numeric" });
    t.textContent = `${f(a)} – ${f(b)}, ${b.getFullYear()}`;
  }
}

function calAgendaDay(row) {
  const sec = document.createElement("div");
  sec.className = "cal-ag-day";
  const d = new Date(row.date + "T00:00");
  const h = document.createElement("div");
  h.className = "cal-ag-date";
  h.textContent = d.toLocaleString("en-US", { month: "short", day: "numeric" });
  const wd = document.createElement("span");
  wd.className = "wd";
  wd.textContent = d.toLocaleString("en-US", { weekday: "long" });
  h.appendChild(wd);
  sec.appendChild(h);

  let any = false;
  for (const e of row.events) {
    any = true;
    const it = document.createElement("div");
    it.className = "cal-ag-item ev" + (e.status ? " " + e.status : "");
    const when = document.createElement("span");
    when.className = "when";
    when.textContent = e.all_day ? "all-day" : e.start.slice(11, 16);
    it.appendChild(when);
    const ti = document.createElement("span");
    ti.textContent = e.title;
    it.appendChild(ti);
    it.addEventListener("click", () => openNote(e.path));
    sec.appendChild(it);
  }
  // An axis renders six rows and then says how many it is holding back. A
  // nucleation run puts sixty lines on one day, and sixty rows of a path with
  // a run id after it is not an agenda, it is a log file in a 300px column.
  const axis = (kind, items, build) => {
    if (!items?.length) return;
    any = true;
    const rows = items.map((x) => {
      const it = mkEl("div", "cal-ag-item");
      const k = mkEl("span", "when cal-ag-kind", kind);
      it.appendChild(k);
      build(it, x);
      return it;
    });
    rows.slice(0, CAL_AGENDA_CAP).forEach((r) => sec.appendChild(r));
    if (rows.length <= CAL_AGENDA_CAP) return;
    const more = mkEl("button", "cal-ag-more",
      `show ${rows.length - CAL_AGENDA_CAP} more`);
    more.type = "button";
    more.addEventListener("click", () => {
      rows.slice(CAL_AGENDA_CAP).forEach((r) => sec.insertBefore(r, more));
      more.remove();
    });
    sec.appendChild(more);
  };

  axis("note", row.notes, (it, n) => it.appendChild(document.createTextNode(n.label)));
  // Verb, then the note it touched, then what it did to it — three fields
  // instead of one sentence, because the file name is the part you scan for.
  axis("agent", row.activity, (it, a) => {
    const p = calActivity(a);
    it.title = a;
    const t = mkEl("span", "cal-ag-txt");
    if (p.verb) t.appendChild(mkEl("b", "", p.verb + " "));
    t.appendChild(document.createTextNode(p.what));
    if (p.detail) t.appendChild(mkEl("i", "", " " + p.detail));
    it.appendChild(t);
  });
  // The queue names a path; a path in a narrow column ellipsises to its
  // folder, which is the half you already know. The note leads, the folder
  // trails at meta weight, and the whole path stays on the tooltip.
  axis("review", row.review, (it, r) => {
    const path = r.path || "";
    const cut = path.lastIndexOf("/");
    const t = mkEl("span", "cal-ag-txt");
    t.appendChild(document.createTextNode(path.slice(cut + 1).replace(/\.md$/, "")));
    if (cut > 0) t.appendChild(mkEl("i", "", " " + path.slice(0, cut)));
    it.title = path;
    it.classList.add("clickable");
    it.addEventListener("click", () => openNote(path));
    it.appendChild(t);
  });
  if (!any) {
    const e = document.createElement("div");
    e.className = "cal-ag-empty";
    e.textContent = "nothing scheduled";
    sec.appendChild(e);
  }
  return sec;
}

// The one reading in this app whose axis is time and not space, so it rides the
// only tab whose axis is time. Above the days rather than beside them: it frames
// what follows ("this fortnight has been about X"), and a frame printed after
// the thing it frames is a footnote.
//
// Concepts, not notes, so the pills do not open anything: clicking one lights
// its notes in the graph, the same verb the note drawer's concept cloud has.
function calBurstStrip() {
  if (!calBursting.length) return null;
  const box = mkEl("div", "cal-burst");
  box.appendChild(mkEl("div", "cal-ag-head", "this fortnight of writing"));
  const pills = mkEl("div", "cal-burst-p");
  for (const b of calBursting) {
    const pill = mkEl("button", "wk-pill", b.concept);
    pill.type = "button";
    pill.title = `${b.recent} of the ${b.total} notes carrying it were written in the `
      + `window (z ${b.z}): light its notes in the graph`;
    pill.addEventListener("click", () => lightConcept(b.concept, pill));
    pills.appendChild(pill);
  }
  box.appendChild(pills);
  return box;
}

function renderCalAgenda() {
  const panel = $("#cal-agenda");
  panel.replaceChildren();
  const burst = calBurstStrip();
  if (burst) panel.appendChild(burst);
  const head = document.createElement("div");
  head.className = "cal-ag-head";
  if (calSelected && calDays[calSelected]) {
    head.textContent = "selected day";
    panel.appendChild(head);
    panel.appendChild(calAgendaDay(calDays[calSelected]));
  } else if (calUpcoming) {
    head.textContent = "next 7 days";
    panel.appendChild(head);
    for (const row of calUpcoming) panel.appendChild(calAgendaDay(row));
  }
}

$("#cal-prev").addEventListener("click", () => {
  if (calMode === "month") calAnchor.setMonth(calAnchor.getMonth() - 1);
  else calAnchor.setDate(calAnchor.getDate() - 7);
  loadCalendar();
});
$("#cal-next").addEventListener("click", () => {
  if (calMode === "month") calAnchor.setMonth(calAnchor.getMonth() + 1);
  else calAnchor.setDate(calAnchor.getDate() + 7);
  loadCalendar();
});
$("#cal-today").addEventListener("click", () => {
  calAnchor = new Date();
  calSelected = null;
  loadCalendar();
});
$("#cal-mode").addEventListener("click", (e) => {
  const m = e.target.dataset.calmode;
  if (!m || m === calMode) return;
  calMode = m;
  document.querySelectorAll("#cal-mode button").forEach((b) => setActive(b, b.dataset.calmode === m));
  loadCalendar();
});

// --- vault freshness ---------------------------------------------------------
// The explore surfaces are built once and cached, and until now the only thing
// that invalidated them was a chat turn in THIS tab — so an Obsidian edit, or a
// `silica nucleate` in a terminal, left them drawing a vault that no longer
// existed. /vault_version is a digest of the note roster plus the derived
// indexes; when it moves, so did the vault.
//
// A poll and not the BUS the turn stream already runs on: that bus is
// in-process, so it carries this browser's own agent and nothing else, and the
// writes this exists for are precisely the ones from outside it.
let vaultVersion = null;

// Cheap surfaces redraw; the expensive one offers. Rebuilding the graph
// document costs the camera, the zoom and the focused node — the reader's place
// in it — so out-of-band changes never take that away, they put a button up.
function markVaultChanged() {
  graphStale = true;
  metricsStale = true;
  // The one place shape is dropped: the other two `graphStale` sites are a
  // theme flip and a render setting, and neither moves a note. folders/areas/
  // read take their colours from tokens, so they survive both untouched.
  shapeData = null;
  if (activeTab !== "graph") return; // rebuilt on the way back into the tab
  if (graphMode in SHAPE_VIEWS) drawShape();
  else if (graphMode === "map" && mapRootedPath) rootMap(mapRootedPath);
  syncRefreshCue();
}

// Derived state, never a flag someone has to remember to clear: the offer shows
// exactly when the graph is the surface on screen AND it is out of date. On map
// or folders/areas/read it must not, because those redrew themselves — an offer
// there would point at a staleness the reader cannot see, on a view that has
// none.
function syncRefreshCue() {
  $("#graph-refresh").hidden = !(graphMode === "graph" && graphStale);
}

async function pollVaultVersion() {
  if (document.visibilityState !== "visible") return; // no view to keep fresh
  try {
    const { version } = await (await fetch("/vault_version")).json();
    if (!version) return;            // unreadable vault answers "": nothing to say
    if (vaultVersion === null) vaultVersion = version; // first read is the baseline
    else if (version !== vaultVersion) { vaultVersion = version; markVaultChanged(); }
  } catch { /* a stale view is not worth a toast; the next tick retries */ }
}
setInterval(pollVaultVersion, 15000);
// Coming back from Obsidian is the whole scenario, and it is a tab switch —
// waiting out the interval would show the stale graph for the worst 15 seconds.
document.addEventListener("visibilitychange", pollVaultVersion);
pollVaultVersion();

// The poll is the tick: the endpoint computes due, advances the sidecar marks,
// and returns what to show. At-most-once shared with the REPL daemon.
setInterval(async () => {
  try {
    const d = await (await fetch("/reminders", { method: "POST" })).json();
    for (const r of d.due || []) {
      notify((r.late ? "late reminder: " : "reminder: ") + r.title + " · " + r.start.slice(0, 16), "info");
    }
  } catch { /* a reminder is a courtesy; the next tick retries */ }
}, 30000);

buildLayoutRail();

// Land on chat — it's the primary surface — unless the URL names another view
// (#explore, #calendar, #metrics): a pasted deep link must win over the default.
// Read the hash BEFORE the default click: showTab rewrites it via replaceState.
const bootSlug = (location.hash || "").replace(/^#/, "");
const bootTab = bootSlug === "explore" ? "graph" : bootSlug;
document.querySelector(
  `.tab[data-tab="${["chat", "graph", "calendar", "metrics"].includes(bootTab) ? bootTab : "chat"}"]`
).click();
