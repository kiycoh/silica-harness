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

