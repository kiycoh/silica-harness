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
