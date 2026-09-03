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
// literals copied out of app-base.css, and by the time anyone looked they were two
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
let sidebarYielded = false;

// The rail is draggable, so its width is read and never assumed: a constant
// here would go stale the first time anyone touched the handle, and the drawer
// would size itself against a rail that is no longer that wide.
function sideW() {
  return parseInt(getComputedStyle(document.documentElement)
    .getPropertyValue("--side-w"), 10) || 264;
}

// What the drawer may take before the transcript drops below its floor.
function drawerBudget(sidebarOn) {
  return window.innerWidth - (sidebarOn ? sideW() : 0) - MIN_PROSE;
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

// This drawer is the app's ONE right-edge surface, in three modes. Two are the
// same file read two ways: `note` is the reader, `diff` is that file against how
// it stood before this session touched it. The third, `node`, is what you
// POINTED at, which is a different subject from either.
//
// That third mode was called `work` and drew two unrelated things behind one
// name: the run on chat, the node on explore. A segment cannot be named after
// one of its two subjects. The run moved to the left rail's Work compartment,
// where nothing competes with it and it stays readable WHILE a note is open;
// this mode kept the node, and now carries the name of what it draws.
//
// (It also once carried a `context` mode, which drew the /context payload the
// work panel's node scope also drew: two panels at the same edge, the same
// sections, neither header saying which you were reading. That one was deleted
// rather than merged - the panel took the payload, this drawer kept the prose.)
//
// The click contract, unchanged in principle and now stated once: NAMING a note
// means "I want to read it", so a wikilink, the file tree and a search hit land
// on `note`. POINTING at one means "what is this", so a graph node, a map card
// and a metrics row land on `node`. A second click on a node is "read it" said
// with the gesture, and lands back on `note`.
let drawerMode = "note";

function syncDrawerMode() {
  const path = lastNotePath || lastViewedPath;
  const open = notePanel.classList.contains("open");
  document.querySelectorAll("#note-mode button").forEach((b) => {
    const on = b.dataset.mode === drawerMode;
    setActive(b, on);
    // A note this session never touched has no diff, a session that has opened
    // none has no note, and a session that has pointed at nothing has no node:
    // an enabled segment onto an empty pane is a promise the sidebar cannot
    // keep. Never the mode that is SHOWING, though - the header toggle can open
    // the node pane on its own empty state, and a lit segment you cannot press
    // reads as a broken control rather than as a pane with nothing in it.
    if (b.dataset.mode === "diff") {
      b.disabled = !on && !changedPaths.has(path);
      b.title = b.disabled ? "this session has not changed this note"
                           : "what this session changed in this note";
    } else if (b.dataset.mode === "node") {
      b.disabled = !on && !nodePicked;
      b.title = b.disabled ? "point at a node on explore or metrics"
                           : "what you pointed at";
    } else if (b.dataset.mode === "note") {
      b.disabled = !on && !path;
      b.title = b.disabled ? "no note opened yet" : "read the note";
    }
  });
  $("#note-body").hidden = drawerMode !== "note";
  $("#note-diff").hidden = drawerMode !== "diff";
  $("#node-pane").hidden = drawerMode !== "node";
  // The five actions act ON the open note; the two chips belong to the node.
  // Each row states the mode it came from and nothing else.
  $("#note-actions").hidden = drawerMode === "node";
  $("#node-scope").hidden = drawerMode !== "node";
  $("#node-state").hidden = drawerMode !== "node";
  // Pressed means the drawer is OPEN, and nothing finer: this is the only
  // control it has once it is shut, and which of the three modes is showing is
  // what the segment above already says.
  setActive($("#drawer-toggle"), open);
  $("#drawer-toggle").title = open ? "hide the note sidebar" : "show the note sidebar";
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
  syncPinButton(); // the toggle states THIS note, so it is re-read per open
  syncDrawerMode(); // the segments and the header toggle key on `open`, set above
}

// Node opens without a path, which is the whole difference between it and the
// other two: it is about what you pointed at, which may be a ghost with no file
// behind it at all. The empty title leaves #note-title as the header's flex
// spacer, with the two chips in it instead.
let nodePicked = false; // has anything been pointed at this session

function openNodePane() {
  drawerMode = "node";
  showDrawer("");
}
// The flag is set HERE and not in openNodePane, because this is the only entry
// that is an actual pick. The segment above calls the same opener and is
// disabled until this has fired, and the header toggle falls back to this pane
// on a session that has read nothing — neither of those is a node.
window.openNodeDrawer = () => { nodePicked = true; openNodePane(); };

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
  if (b.dataset.mode === "node") openNodePane();
  else if (b.dataset.mode === "diff") openDiff(path);
  else openNote(path);
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
  lastNotePath = null; // lastViewedPath survives — the NOTE segment can reopen
  focusGraphNode(null);
  syncDrawerMode(); // the segments key on `open`, cleared above
}

// The header's one control for the drawer: open it, or shut it. It does not
// pick a mode, it RESTORES one — whichever you left it on, degraded to the next
// one that has something behind it, because a segment can go dead between two
// opens (a diff needs this session to have touched the file).
function reopenDrawer() {
  const path = lastNotePath || lastViewedPath;
  if (drawerMode === "diff" && changedPaths.has(path)) { openDiff(path); return; }
  if (drawerMode !== "node" && path) { openNote(path); return; }
  if (drawerMode === "node" && nodePicked) { openNodePane(); return; }
  if (path) { openNote(path); return; }
  // Nothing read and nothing pointed at: the node pane is the one mode with a
  // real thing to say about that, so it opens on its own empty state rather
  // than on a reader with no file in it.
  openNodePane();
}

$("#drawer-toggle").addEventListener("click", () => {
  if (notePanel.classList.contains("open")) closeNote();
  else reopenDrawer();
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
  // Clicking the lit one drops it. A concept lights a SET, and a set you can
  // only ever swap for another set is a mode you are stuck in: there was no
  // gesture at all for "show me the graph again", short of opening a note.
  const drop = btn.classList.contains("lit");
  document.querySelectorAll(".wk-pill.lit").forEach((e) => e.classList.remove("lit"));
  if (drop) { focusGraphNodes([]); return; }
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

// The same jump for the reading order, which the node panel's ladder footer
// hands off to. Exported rather than duplicated: the drawer knows which note it
// is on and nothing else about explore's modes, and rootPath is the one place
// that knows how to enter Path rooted somewhere.
window.openLadder = (note) => {
  if (!note) return;
  // rootPath BEFORE the tab, and no pre-set graphMode: the map button can
  // pre-set its mode because rootMap is replayed on tab-enter, and path has no
  // such replay - showTab's setGraphMode(graphMode) draws off pathRootedPath
  // and nothing else. Pre-setting the mode therefore started an UNROOTED draw
  // first, and the landing measures forty ladders, so it also finished last
  // and overwrote the rooted one: Path opened on the picker every time.
  rootPath(note);
  document.querySelector('.tab[data-tab="graph"]').click();
};

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

// --- rail resize (drag right edge, clamped) ---------------------------------
// The mirror of #note-resize below, and it exists for the same reason: the one
// line that says WHICH vault you are reading sat at the foot of a 264px column
// and was clipped there, so the answer was reachable only by hovering it.
const SIDE_MIN_W = 200, SIDE_MAX_W = 520;

function setSideW(w) {
  document.documentElement.style.setProperty("--side-w", w + "px");
}

const savedSideWidth = parseInt(localStorage.getItem("side-width"), 10);
if (savedSideWidth) setSideW(Math.min(SIDE_MAX_W, Math.max(SIDE_MIN_W, savedSideWidth)));

$("#side-resize").addEventListener("mousedown", (e) => {
  e.preventDefault();
  const startX = e.clientX, startWidth = sideW();
  const onMove = (e2) => {
    // The same prose floor the drawer obeys: widening the rail must not be a
    // way around it, or the two panes together squeeze the transcript to
    // nothing from opposite edges.
    const open = document.body.classList.contains("note-open");
    const cap = open ? window.innerWidth - MIN_PROSE - MIN_DRAWER : SIDE_MAX_W;
    setSideW(Math.min(SIDE_MAX_W, Math.max(SIDE_MIN_W,
      Math.min(cap, startWidth + (e2.clientX - startX)))));
  };
  const onUp = () => {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    localStorage.setItem("side-width", sideW());
    fitPanes(); // the drawer's budget just changed under it
  };
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
});

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
syncDrawerMode(); // segments start correct: `note` is disabled with no note
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
// What the sidebar was showing BEFORE this click reached anything: the mode, or
// null for closed. Capture phase, so it is sampled before any element handler
// can change it. The outside-click rule below closes the sidebar, and the three
// modes made that rule wrong in two directions at once: a metrics or shape row
// POINTS at a note, which raises the sidebar on `work` from its own listener,
// and the delegated handler running after it on the way up would read the panel
// as open and close what the click just asked for — whether the click OPENED the
// sidebar or SWUNG it from `note`. So the rule is not "was it open" but "did this
// click leave it alone": a click that opened or moved the sidebar never closes it.
let drawerBefore = null;
const drawerNow = () => (notePanel.classList.contains("open") ? drawerMode : null);
document.addEventListener("click", () => { drawerBefore = drawerNow(); }, true);
// One delegated handler: .note-link (chat OR in-panel → in-place nav) opens the
// drawer; a click outside an open drawer closes it. The sidebar and the dock
// are persistent instruments — picking a note, toggling a folder, or typing a
// question about the open note must not close the drawer or reset the graph
// focus, so they never count as "outside". Neither does the sidebar toggle
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
  // isConnected FIRST, and not as one more `closest` — a row inside the drawer
  // whose own handler re-renders the list has already been detached by the time
  // this runs, and closest() on a detached node answers "outside" for every
  // selector below. That is what closed the sidebar on every Read-first click:
  // the panel the click asked to walk was the panel it took away.
  if (drawerBefore && drawerBefore === drawerNow() &&  // untouched by this click
      e.target.isConnected &&
      !e.target.closest("#note-panel") && !e.target.closest("#sidebar") &&
      !e.target.closest("#dock")) closeNote();
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

