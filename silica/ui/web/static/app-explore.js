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
  buildSurfaceLegend();
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

// --- the legend's per-surface half -------------------------------------------
// What the marks in front of you MEAN, for the one surface you are on. Every
// surface already stated this somewhere - the shape views printed it as a grey
// line inside their own pane, the graph hid its renderer in a toolbar - so this
// is one home for six scattered captions, not six new ones.
//
// A key and, where the surface has one, its controls. Only the graph has a
// control here today: the two renderers. The blank left by the others is not a
// gap to fill with knobs - a control that changes nothing worth changing is
// worse than none - but the obvious candidates when they earn their keep are a
// depth cap on map, a fill rule on folders (mixing vs size), a sort on the
// areas matrix, and a "areas covered" cut on read.
const SURFACE_KEYS = {
  graph: ["node = note · size = how much the vault routes through it · colour = area",
          "lines are filtered in the panel below"],
  map: ["ring = hops from the root · angle = how alike two notes read",
        "pick a different root from the search above"],
  folders: ["tile area = notes in the folder · fill = how much the folder mixes areas",
            "click a folder to descend, the crumbs walk back"],
  areas: ["cell = links between two areas · diagonal = the area's own cohesion"],
  read: ["stops in area order, biggest first · each hub then what it opens onto"],
  path: ["− comes before this note · + is what it unlocks · ● is where you are",
         "RefD over the resolved links; re-root from any chip"],
};

function buildSurfaceLegend() {
  const box = $("#lg-surface");
  box.innerHTML = "";
  const key = SURFACE_KEYS[graphMode] || [];
  // Only the graph has two renderers. On the other five a segment here would be
  // a control with nothing to switch.
  if (graphMode === "graph") {
    const seg = mkEl("div", "gmode-tabs");
    seg.id = "renderer-tabs";
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "renderer");
    for (const [r, why] of [["3d", "WebGL, three dimensions"],
                            ["2d", "canvas, two dimensions, with note labels"]]) {
      const b = mkEl("button", null, r.toUpperCase());
      b.type = "button";
      b.dataset.renderer = r;
      b.title = why;
      b.setAttribute("aria-pressed", "false");
      seg.appendChild(b);
    }
    box.appendChild(seg);
    syncRenderer("");   // paint the fresh pair from what the frame last said
  }
  for (const line of key) box.appendChild(mkEl("div", "lg-key", line));
  syncLegendOffset();
}

$("#lg-surface").addEventListener("click", (e) => {
  // The renderer lives inside the frame (it owns the WebGL/canvas instance), so
  // this asks rather than sets. Nothing is painted here on the way out: the
  // frame answers with the mode it actually built, which is the only value that
  // cannot be a lie, and syncRenderer() paints that.
  const r = e.target.dataset.renderer;
  if (!r) return;
  const f = $("#graph-frame");
  if (f.contentWindow) f.contentWindow.postMessage({ type: "silica-set-renderer", mode: r }, "*");
});

// The frame's HUD is anchored to the frame's own top-right, which is where this
// legend sits. It cannot see this element, so it is told how tall it is and
// parks underneath — the two then read as one stack against that edge rather
// than as one panel covering another.
function syncLegendOffset() {
  const f = $("#graph-frame");
  if (!f || !f.contentWindow) return;
  const h = $("#legend").getBoundingClientRect().height;
  f.contentWindow.postMessage({ type: "silica-legend-h", h: Math.round(h) }, "*");
}
new ResizeObserver(syncLegendOffset).observe($("#legend"));

// The frame states its renderer on every build and on every switch, embedded or
// not. Before the first answer the segment shows neither: an unanswered toolbar
// that guesses 3D is a toolbar that is wrong for as long as the graph takes to
// build.
// Remembered, because the segment is now BUILT per surface rather than hidden:
// leaving graph for folders and coming back makes a fresh pair of buttons, and
// the frame announces only on a build or a switch. Without this the segment
// came back blank on a graph that was plainly still in 2D.
let lastRenderer = "";

function syncRenderer(mode) {
  if (mode) lastRenderer = mode;
  document.querySelectorAll("#renderer-tabs button")
    .forEach((b) => setActive(b, b.dataset.renderer === lastRenderer));
}

// #graph-frame finishes loading only once the server is done building — drop the
// loader then and re-sync the focus dim state after a (re)load.
$("#graph-frame").addEventListener("load", () => {
  $("#graph-loading").hidden = true;
  replayGraphFocus(); // re-sync whatever is focused after a (re)load
  syncDrawerToViews(); // ditto for the drawer, which hides this frame's HUD
  // …and for the legend. A contentWindow exists long before the document in it
  // has a listener, so every earlier post was dropped: measured, the HUD sat at
  // top:10px under a legend 203px tall and lost its first six rows.
  syncLegendOffset();
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
    `${s.areas.length} areas · ${linked} of ${pairs} pairs share a link`));
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
  head.appendChild(mkEl("span", "shape-sub", `${r.stops.length} stops`));
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

