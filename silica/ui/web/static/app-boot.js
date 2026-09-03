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
// Archivo loads async and the segments get wider when it lands; the track also
// re-wraps whenever the sidebar or the drawer renegotiates the pane. Both are
// resizes of the track, so one observer covers them.
new ResizeObserver(syncQuick).observe(qaTrack);

// --- the composer hint, at the width it actually gets ------------------------
// The placeholder is a name plus a parenthetical key hint, and CSS cannot
// shorten it: Chromium drops text-overflow on a <textarea> placeholder, so the
// ellipsis declared for exactly this in app-chat.css never renders and a squeezed
// field hard-cuts mid-word. At the 900px floor with the work drawer folded open
// the field is 223px wide and the string wants 263px, which showed "⇧⏎ nev".
// So drop the clause, not the characters. Canvas measures the same string the
// field will draw; the 6px margin covers what the 2d context cannot carry from
// the element, which is the variable-width axis Archivo is here for.
const phMeter = document.createElement("canvas").getContext("2d");
function fitPlaceholder(el) {
  const full = el.dataset.phFull || (el.dataset.phFull = el.placeholder);
  const cs = getComputedStyle(el);
  phMeter.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
  const room = el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const fits = phMeter.measureText(full).width + 6 <= room;
  // strip the trailing parenthetical, keeping the name the field is for
  el.placeholder = fits ? full : full.replace(/\s*\([^()]*\)\s*$/, "");
}
// Changing a placeholder resizes nothing, so observing the field it sits in
// cannot feed itself.
for (const el of [$("#input"), $("#dock-input")]) {
  fitPlaceholder(el);
  new ResizeObserver(() => fitPlaceholder(el)).observe(el);
}

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

