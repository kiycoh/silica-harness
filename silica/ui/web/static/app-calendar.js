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

// AFTER the boot tab, not with the rest of the drawer's setup: switching tabs
// closes the sidebar, and a restore that ran before this line was undone by it
// on every load - the panel opened and shut inside one frame, which reads as the
// preference never having been saved. Closed until asked for, then remembered:
// the sidebar on `work` narrates a run, and the state it would open on by
// default is the state it spends most of its time in - empty, beside a
// transcript that is 630px narrower for it.
