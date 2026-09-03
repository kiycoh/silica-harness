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
// Below the width that holds rail + reading measure + right sidebar side by
// side, the rail becomes five icons summoned over the transcript. It does not
// disappear: what a fold must never do is take a surface away and leave nothing
// where it was. (The work panel folded here too, until it stopped being a panel
// — as a mode of the sidebar it folds the way the sidebar does, which is why
// work.js no longer asks this question and the export it asked through is gone.)
//
// The threshold itself lives in ONE place, the media query in app-node.css that sets
// --narrow, read back through here rather than carried as a pixel count as well,
// because two constants that must match are two constants that eventually do not.
function isNarrow() {
  return getComputedStyle(document.body).getPropertyValue("--narrow").trim() === "1";
}

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

// A vault with no named areas has nothing to legend, and an empty fold is a
// row that opens onto nothing. The view gate that used to ride here is gone
// with the compartment: #legend lives inside #view-graph, so it is already
// only on the view its contents describe.
let railHasAreas = false;

function syncAreasRail() {
  $("#lg-areas").hidden = !railHasAreas;
}

// The legend's area spectrum, from the same /vault_info the landing reads: one
// fetch, two readings. The bar is proportional to the largest area, because
// what this is for is the SHAPE of the distribution; the exact size is the
// figure beside it. The summary counts every area, not the five that fit.
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

// --- the legend's Layout compartment -----------------------------------------
// The ONE place the six surfaces are named. They used to be a row of tabs in
// the graph toolbar as well, a hand's width from this list; then this list was
// in the LEFT rail, a screen's width from the picture it switched. It is
// against the picture now. The list is data rather than markup because there is
// no second DOM to read it back off.
//
// The third field is not a tooltip only: it is the surface's KEY, and the same
// string is what #lg-surface states about the marks in front of you.
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
  // Closed, the fold has to say which of the six you are on, or it is a row
  // that names the control and not the state.
  const now = LAYOUT_MODES.find(([m]) => m === graphMode);
  $("#lg-layout-now").textContent = now ? now[1] : "";
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

