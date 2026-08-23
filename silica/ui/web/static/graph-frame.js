// Embedded in the web app's iframe: the app's own sidebar (stats/search/tree)
// replaces the internal one; only the graph + HUD legend remain.
if (window.parent !== window) document.body.classList.add("embedded");

// Everything the server decided for this document rides one JSON island, read
// here and nowhere else. Not JS literals spliced into this file: the code is
// then a static file a linter can parse and a test can run, and the data is the
// one thing per document that is not (ADR-0026). The island is parsed, not
// evaluated, so a value can never be code.
const DATA = JSON.parse(document.getElementById("graph-data").textContent);

const RAW_NODES = DATA.nodes;
const RAW_EDGES = DATA.edges;
const COMM_LABELS = DATA.comm_labels;
// The semantic partition: node.sgroup -> zone. Disjoint from node.group in every
// way that matters — different edges, different ids, different colours.
const ZONES = DATA.zones;
// Only the label is indexed. There was a ZONE_COLOR map beside it, read by
// nodeColor alone; now that the notes keep their community colour it has no
// reader left — the hulls and the zone names take z.color straight off ZONES.
const ZONE_LABEL = {};
ZONES.forEach(z => { ZONE_LABEL[z.id] = z.label; });
// The zone's hue for the live floor. Both rides on the zone (see Zone in
// graph_export) because the phase shift that keeps zone i off community i is
// declared there, and recomputing it here would be a second place to break it.
const zoneColor = z => (LIGHT && z.color_paper) || z.color;

const outDeg = {}, inDeg = {};
RAW_EDGES.forEach(e => {
  outDeg[e.from] = (outDeg[e.from] || 0) + 1;
  inDeg[e.to]   = (inDeg[e.to]   || 0) + 1;
});

const NODE_BY_ID = {};
RAW_NODES.forEach(n => { NODE_BY_ID[n.id] = n; });

const neighbors = {};
RAW_EDGES.forEach(e => {
  (neighbors[e.from] = neighbors[e.from] || new Set()).add(e.to);
  (neighbors[e.to]   = neighbors[e.to]   || new Set()).add(e.from);
});

let focusIds = [];  // the focused set; [] = nothing focused
const NO_NEIGHBOURS = new Set();

// Highlight a SET of nodes and their 1-hop neighbours; dim everything else.
// A set, not one id: the context drawer lights every note carrying a concept,
// and a single-node focus is just the one-element case.
//
// Split in two on purpose: computeFocus only writes the _dim flags onto the
// shared node/link objects, applyFocus also repaints. Those objects outlive the
// renderer instance, so a fresh one picks the flags up in its first graphData()
// digest — which is what lets a rebuild compute and skip the repaint. The
// repaint is not free (see refreshPaint).
function computeFocus(ids) {
  focusIds = (ids == null ? [] : [].concat(ids)).filter(id => NODE_BY_ID[id]);
  const on = new Set(focusIds);
  // Lit = the focused nodes plus their 1-hop neighbours; an edge stays lit when
  // EITHER endpoint is focused (neighbour-to-neighbour edges dim, as before).
  const lit = new Set(focusIds);
  focusIds.forEach(id => (neighbors[id] || NO_NEIGHBOURS).forEach(nb => lit.add(nb)));
  RAW_NODES.forEach(n => { n._dim = on.size > 0 && !lit.has(n.id); });
  RAW_EDGES.forEach(e => { e._dim = on.size > 0 && !on.has(e.from) && !on.has(e.to); });
  updateFocusBar();
}

// --- The banner: say out loud that you are not looking at the whole vault ---
// Two mechanisms hide notes — the community filter and the focus dim — and
// neither used to announce itself. Both land here, so one line covers both, and
// it names the way out (Esc) next to the reason it is needed.
function updateFocusBar() {
  const bar = document.getElementById("focus-bar");
  const parts = [];
  // Notes off is the loudest filter of the three: with the zones off too it
  // empties the frame outright, and an empty frame that says nothing reads as a
  // broken view rather than a chosen one.
  if (!showNotes) parts.push(showZones ? "<b>zones only</b>" : "<b>notes hidden</b>");
  if (activeCommunity !== -2) {
    const label = COMM_LABELS[activeCommunity] || ("cluster " + activeCommunity);
    const n = RAW_NODES.filter(x => x.group === activeCommunity).length;
    parts.push("cluster <b>" + escHtml(label) + "</b> · " + n + " notes");
  }
  if (focusIds.length) {
    const lit = new Set(focusIds);
    focusIds.forEach(id => (neighbors[id] || NO_NEIGHBOURS).forEach(nb => lit.add(nb)));
    const nearby = Math.max(0, lit.size - focusIds.length);
    // One note focused: its name is already on screen above this bar, in the
    // drawer header. Repeating it here is noise, so only a set says what it is.
    const head = focusIds.length === 1 ? "" : "<b>" + focusIds.length + " notes</b> ";
    parts.push(head + "+ " + nearby + " neighbour" + (nearby === 1 ? "" : "s"));
  }
  bar.innerHTML = parts.length
    ? parts.join(" · ") + ' <span class="esc">· Esc to clear</span>' : "";
  bar.style.display = parts.length ? "block" : "none";
}

const escHtml = s => String(s).replace(/[&<>]/g, c => (
  {'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function applyFocus(ids) { computeFocus(ids); refreshPaint(); }

function fitGraph() { Graph.zoomToFit(400, 40); wake(600); }

// --- what the layout is doing, bottom-left ----------------------------------
// Ticks and not seconds: a tick is the layout's own unit, so the same graph
// settles in the same number of them on any machine, while the seconds say more
// about the laptop than about the vault.
//
// The animated tail is counted here; the synchronous warmup is ADDED rather
// than observed, because the bundle runs it as a plain forceLayout.tick() loop
// with no onEngineTick call. Adding the count we passed is exact in both paths
// and not an estimate: a seeded build warms up 0 ticks, and a cold one runs at
// most 240 against a schedule that only reaches ALPHA_MIN at ~300, so the loop's
// own alpha break can never cut it short.
let simTicks = 0, warmupRun = 0;
const gnum = n => n.toLocaleString("en-US");

function paintStat(tail) {
  const el = document.getElementById("gstat-n");
  if (!el) return;
  el.textContent = gnum(RAW_NODES.length) + " nodes · " + gnum(RAW_EDGES.length)
                 + " edges · " + tail;
}

// The chip beside fit. A reheat is a perturbation of a settled layout, which is
// what the force sliders already do — same call, so there is one definition of
// "stir it" and not two that decay differently.
function reheat() {
  simTicks = 0; warmupRun = 0;   // this settle belongs to the reheat, not the build
  paintStat("settling");
  applyForces(true);
}

// Undim AND reframe — the background click means "show me everything again".
function clearFocus() {
  applyFocus(null);
  Graph.zoomToFit(600, 40);
  wake(700);   // the refit is a camera tween; it needs frames to run
}

// Re-pass the colour accessors so the renderer repaints without touching the
// simulation. 2D also needs nodeCanvasObject re-passed — the canvas draw reads
// _dim itself, and force-graph caches nothing per node between frames, so a
// plain redraw suffices there; the re-pass is what schedules it.
//
// Cheap in 2D only. force-graph declares nodeColor/linkColor with
// `triggerUpdate:false`, so the re-pass just raises needsRedraw. 3d-force-graph
// does NOT: it lists nodeColor/nodeVisibility/linkColor/linkVisibility among the
// props that re-run the whole node and link digest, so every re-pass rebuilds
// the material of every node and every link. Never call this when a rebuild is
// about to happen anyway.
function refreshPaint() {
  Graph.nodeColor(Graph.nodeColor());
  // 3D links live in the merged LineSegments: one buffer rewrite against the
  // full link digest (9k material rebuilds) the accessor re-pass would cost.
  // With PARTICLES on the lib still owns the photon carriers, so those keep
  // the re-pass beside the merge.
  if (is2D() || PARTICLES) Graph.linkColor(Graph.linkColor());
  if (!is2D()) repaintLinkSeg();
  wake(120);
}

// --- the canvas's own palette ----------------------------------------------
// CSS tokens stop at the edge of the canvas: a WebGL material and a 2D
// fillStyle both need a literal, and both are set on a hot path where reading
// getComputedStyle per node is not an option. So the two sets live here, picked
// once at load. Every value is the light twin of the one beside it, chosen
// against the floor it lands on rather than by inverting a channel.
const LIGHT = document.documentElement.dataset.theme === "light";
const GP = LIGHT ? {
  dim: '#DCD5C7',          // a node filtered out of focus: toward the paper
  ghost: '#C9C4D6',        // unresolved link — the faintest thing the floor holds
  fallback: '#6B6F8C',     // no community
  label: '#1A1815', ghostLabel: '#615B4F',
  linkDim: '#E5DFD2',
  bg: '#EFEAE0', bgHex: 0xEFEAE0,
  ringHub: '#096275', ringOrphan: '#615B4F', ringCut: '#7A5305',
} : {
  dim: '#1d192f',
  ghost: '#484867',        // unlit, never black
  fallback: '#565a77',
  label: '#EBEFF8', ghostLabel: '#838DA7',
  linkDim: '#141221',
  bg: '#0D0917', bgHex: 0x0D0917,
  ringHub: '#35C6E8', ringOrphan: '#838DA7', ringCut: '#E0A93B',
};

let activeCommunity = -2;
let showExtracted = true;
let showAmbiguous = false;
let showGaps = true;
let showSimilar = true;
// The three inferred layers, all off: the frame opens on what you WROTE, and
// every layer that is a claim about what you did not write has to be asked for.
let showProposed = false;
let showCoupled = false;
let showDiscord = false;
let showZones = false;   // the semantic layer is asked for, never assumed
let showNotes = true;    // the macro read: zones alone in the frame

// --- Node color = its community color, flat -------------------------------
// One hue per community: every node in a community shares the exact color,
// hub or leaf. Degree is shown by size, never by washing the hue out.
function nodeColor(n) {
  // Solid darken-to-background dim, decided: rgba() only if visual
  // verification ever shows 3d-force-graph honours per-node alpha.
  // Neutrals are blue-violet, never gray: the mascot has no gray facet, only
  // unlit ones. Each value below holds the luminance of the gray it replaces.
  if (n._dim) return GP.dim;
  if (n.type === 'ghost') return GP.ghost;
  // The node's colour is the STRUCTURAL community, always, whatever else is on
  // screen. It used to hand the channel over to the semantic partition while
  // the zone layer was up, which read as a bug and was one: ADR-0023 says the
  // two partitions "coexist and never substitute for each other" and "share
  // neither colour key nor id space", and a channel that swaps between them
  // is a substitution by definition. The semantic layer owns the hulls and the
  // names; it does not get to repaint the notes.
  const c = n.color || {};
  return (LIGHT ? c.background_paper || c.background : c.background) || GP.fallback;
}

// --- Node state = a ring, on a channel colour is not already using ----------
// Nothing new is computed here: betweenness rides on every node already (it is
// what sizes them) and the in-degree comes from the edge list. The states are
// the ones the vault report already names, so the view and the report agree.
//
// Orphan matches graph_report's definition exactly — in-degree zero over
// RESOLVED wikilinks only. Counting SIMILAR would erase the state entirely
// (k-NN gives almost every note an in-edge), which is the opposite of the
// question "is anything in the vault pointing here".
const linkInDeg = {};
RAW_EDGES.forEach(e => {
  if (e.type === "EXTRACTED") linkInDeg[e.to] = (linkInDeg[e.to] || 0) + 1;
});

// Hub = top decile of the nodes that have any betweenness at all. A fixed
// threshold would call half a dense vault a hub and none of a sparse one;
// the decile asks the same question of both.
const _bets = RAW_NODES.map(n => n.betweenness || 0).filter(b => b > 0).sort((a, b) => a - b);
const HUB_MIN = _bets.length ? _bets[Math.floor(_bets.length * 0.9)] : Infinity;

// Priority is deliberate: a crossing with no backlinks is worth reading as a
// hub, not as an orphan.
function nodeState(n) {
  if (n.type === "ghost") return "ghost";
  if ((n.betweenness || 0) >= HUB_MIN) return "hub";
  if (!linkInDeg[n.id]) return "orphan";
  return "note";
}

// Only the states with NO channel of their own get a ring. "note" is the
// default, so ringing it would say nothing; ghost already has three markers
// (its own unlit colour, a smaller radius, a dimmer label) and there are 468 of
// them in a 682-note vault — a fourth marker on the most numerous state turns
// the whole view into an alarm about links you have not written yet.
const STATE_RING = { hub: GP.ringHub, orphan: GP.ringOrphan };

// Counts never change (the states come from the exported data), so this runs
// once at load, not per repaint.
function syncStateLegend() {
  const c = { hub: 0, orphan: 0, ghost: 0, note: 0 };
  RAW_NODES.forEach(n => c[nodeState(n)]++);
  Object.keys(c).forEach(k => {
    const el = document.getElementById("st-" + k);
    if (el) el.textContent = c[k];
  });
}
syncStateLegend();

// --- Density-aware forces ---------------------------------------------------
// The lib's d3 defaults (charge -60 in 3D, link distance 30) collapse dense
// graphs into a hairball: equilibrium spacing must grow with avg degree or
// neighborhoods overlap. sqrt keeps sparse graphs (k<=2) exactly as before
// (scale=1) and opens dense ones up to 4x. Sliders multiply on top of this
// baseline, so the auto-scaling stays authoritative as the vault grows.
const AVG_DEG = RAW_NODES.length ? 2 * RAW_EDGES.length / RAW_NODES.length : 0;
const FORCE_SCALE = Math.min(4, Math.max(1, Math.sqrt(AVG_DEG / 2)));
// Same auto-scaled baseline in both modes; only the per-mode constants differ.
// 2D has one dimension fewer to disperse into, so at the same charge the plane
// packs tighter than the sphere: repulsion and rest length are opened up until
// x1 reads the same in both. (Tuned by eye — that IS what "looks right" means.)
const CHARGE_2D_K = 1.8, DIST_2D_K = 1.5;
const baseCharge = () => -60 * FORCE_SCALE * FORCE_SCALE * (is2D() ? CHARGE_2D_K : 1);
const baseDist   = () => 30 * FORCE_SCALE * (is2D() ? DIST_2D_K : 1);
// --- When the layout is allowed to stop ------------------------------------
// A tick COUNT is the wrong gate and was measurably too small. d3 decays alpha
// by a fixed 2.28% per tick regardless of graph size, so it reaches the 0.001
// that d3 itself calls converged at tick 300, always. The old budget
// (100 + min(200, N/10)) gave 155 ticks on a 550-note vault: alpha 0.028, still
// 28x above convergence. The view froze on a layout that was genuinely half
// unfolded — not a perception problem, an arithmetic one.
//
// So hand the gate to the physics. Both bundles already check it and neither
// had it switched on (d3AlphaMin defaults to 0 = disabled):
//
//   ++cntTicks > cooldownTicks || now - startTickTime > cooldownTime
//     || d3AlphaMin > 0 && forceLayout.alpha() < d3AlphaMin   ->  onEngineStop
//
// cooldownTicks and cooldownTime both go to Infinity on purpose. Alpha decays
// deterministically, so it ALWAYS converges; a tick or wall-clock ceiling can
// no longer protect against anything, it can only re-introduce the early cut
// through a second door. (cooldownTime's 15s default would have done exactly
// that on the slower renderer.)
const ALPHA_MIN = 0.001;      // d3's own convergence point, reached at tick ~300

// warmupTicks runs the same tick() in a plain loop with nothing painted, and
// consumes the SAME alpha schedule — the bundle's warmup loop carries the
// d3AlphaMin check too. So warmup and the animated tail split one 300-tick
// budget, and the split is what you actually watch.
//
// Split per renderer, because the two pay completely different prices for a
// painted tick: both libs advance the layout exactly one tick per RENDERED
// frame, and the 2D canvas (553 arcs + labels + 4566 strokes per frame, all
// CPU) runs that at roughly 19 ticks/s against WebGL's 60. An even split makes
// 2D take three times as long to settle for the identical layout. Weighting the
// warmup by the renderer buys back a comparable settle time in both.
//
// Two things the plain `() => is2D() ? 240 : 150` got wrong on a big vault.
//
// It ran a warmup even when the layout was SEEDED. Seeded means FAST_DECAY, and
// FAST_DECAY reaches ALPHA_MIN at tick 66 — inside either count. So the engine
// stopped in the middle of the synchronous warmup, before the renderer had ever
// drawn a frame, and the fit deferred to onEngineStop then measured a scene
// whose meshes were all still sitting at the origin. Camera 41 units from the
// centre of a 3,900-unit graph: the 2D -> 3D switch opening fully zoomed in.
// A cached layout does not need a warmup. It IS the warmup.
//
// And the counts were absolute, while the cost of a tick is not. Every warmup
// tick is a frame the browser cannot paint, and a tick is roughly linear in
// links: measured 8.94ms at 12,232 links (link 2.5 + charge 6.4), so the 240
// tuned at ~4.5k links stopped being ~0.8s of freeze and became ~2.1s. Hold the
// freeze near what it was tuned to be, rather than the number of ticks.
const WARMUP_REF_EDGES = 4566;   // the graph the counts above were tuned on
function WARMUP_TICKS(seeded) {
  if (seeded) return 0;
  const base = is2D() ? 240 : 150;
  const scale = Math.min(1, WARMUP_REF_EDGES / Math.max(1, RAW_EDGES.length));
  return Math.max(30, Math.round(base * scale));
}

// --- Layout cache: pay the 300 ticks once, not once per load ----------------
// The honest gate above costs about twice the ticks of the wrong one. Cached
// positions hand that back from the second load on: seed x/y/z from the last
// settled layout and raise the decay so the sim only has to rerelax, ~66 ticks,
// which the warmup then swallows whole. The graph opens already settled.
//
// One slot per renderer, holding its own fingerprint — a slot per fingerprint
// would accumulate a dead entry for every force-slider position ever dragged.
// The fingerprint covers the node set AND the force multipliers: a layout
// settled under different forces is not this layout.
const LAYOUT_KEY = "silica-graph-layout";
const FAST_DECAY = 0.1;   // alpha 1 -> ALPHA_MIN in ~66 ticks
const NODE_FP = (() => {
  let h = RAW_NODES.length;
  for (const n of RAW_NODES)
    for (let i = 0; i < n.id.length; i++) h = (h * 31 + n.id.charCodeAt(i)) | 0;
  return h;
})();
const layoutFp = () => NODE_FP + ":" + forceMul.repel.toFixed(3) + ":" +
  forceMul.dist.toFixed(3) + ":" + forceMul.center.toFixed(2);

// True when the positions were seeded, i.e. the sim only has to rerelax.
function loadLayout() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(LAYOUT_KEY + "-" + mode)); } catch (e) {}
  if (!saved || saved.fp !== layoutFp() || !saved.pos) return false;
  // Positions are keyed by id, not by index: a vault that gained a note between
  // two loads has a different fingerprint anyway, but by-index would silently
  // scatter every node past the insertion point if that ever stopped holding.
  let hit = 0;
  RAW_NODES.forEach(n => {
    const p = saved.pos[n.id];
    if (!p) return;
    n.x = p[0]; n.y = p[1]; n.z = p[2];
    hit++;
  });
  return hit === RAW_NODES.length;
}

function saveLayout() {
  const pos = {};
  // Rounded to whole graph units: the layout is thousands of units across, the
  // decimals are noise, and they triple the size of what goes into storage.
  RAW_NODES.forEach(n => {
    pos[n.id] = [Math.round(n.x || 0), Math.round(n.y || 0), Math.round(n.z || 0)];
  });
  try {
    localStorage.setItem(LAYOUT_KEY + "-" + mode,
                         JSON.stringify({ fp: layoutFp(), pos: pos }));
  } catch (e) { /* quota or blocked storage -> next load just re-settles */ }
}

// --- 2D labels: node radius + zoom LOD --------------------------------------
// Base size is whatever the smallest real note got (16 today, or 16+40*b once
// betweenness sizing runs); anything above it is a node that stands out, and
// those are the ones worth a label before you have zoomed in.
const BASE_SIZE = RAW_NODES.reduce(
  (m, n) => n.type === "ghost" ? m : Math.min(m, n.size || 16), Infinity);
const NODE_REL_SIZE = 1.5;  // r = sqrt(val) * this — 16 => 6px, a 56 hub => 11px
const nodeRadius = n => Math.sqrt(Math.max(0, n.size || 16)) * NODE_REL_SIZE;

// Circle + text. Below ~0.6 zoom only dots (at that scale the text is a smear
// and the labels outnumber the pixels); 0.6-1.5 the standouts; above 1.5 all of
// them. Font size divides by the zoom so text keeps a constant SCREEN size.
function drawNode(n, ctx, scale) {
  const r = nodeRadius(n);
  ctx.beginPath();
  ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
  ctx.fillStyle = nodeColor(n);
  ctx.fill();
  if (n._dim || scale < 0.6) return;                       // dimmed: dot only
  // The ring is detail, so it lives behind the same zoom gate as the labels.
  // Drawing it at every scale flooded the zoomed-out view: rings outnumbered
  // the pixels between them and the whole graph read as one alarm.
  // It sits OUTSIDE the disc so the community hue keeps its full area, and both
  // offset and width divide by the zoom to stay constant on screen.
  const ring = STATE_RING[nodeState(n)];
  if (ring) {
    ctx.beginPath();
    ctx.arc(n.x, n.y, r + 2 / scale, 0, 2 * Math.PI);
    ctx.lineWidth = 1.5 / scale;
    ctx.strokeStyle = ring;
    ctx.stroke();
  }
  // A second node channel rather than a fourth state: a cut vertex is usually a
  // hub as well, and nodeState() can only answer with one word. The ring sits
  // OUTSIDE the state ring, so a note that is both shows both readings instead
  // of one overwriting the other.
  //
  // Always on, and it has no toggle. It was off by default on the argument that
  // 128 amber rings on a 709-note vault is an alarm about a normal graph, and
  // that traded the wrong thing away: a reading nobody switches on is a reading
  // nobody has. The ring is 1.2px at the node's own scale and carries no fill,
  // which is what keeps 128 of them a texture rather than an alarm.
  if (n.cut) {
    ctx.beginPath();
    ctx.arc(n.x, n.y, r + 4.5 / scale, 0, 2 * Math.PI);
    ctx.lineWidth = 1.2 / scale;
    ctx.strokeStyle = GP.ringCut;
    ctx.stroke();
  }
  if (scale < 1.5 && (n.size || 16) <= BASE_SIZE) return;  // mid zoom: standouts
  ctx.font = (11 / scale) + 'px Lexend, system-ui, sans-serif';
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillStyle = n.type === "ghost" ? GP.ghostLabel : GP.label;
  ctx.fillText(n.label, n.x, n.y + r + 2 / scale);
}

// Hit area follows the circle, not the label: clicking the text of a dense
// cluster would otherwise pick whichever node's label happened to be on top.
function paintNodeArea(n, color, ctx) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(n.x, n.y, nodeRadius(n) + 2, 0, 2 * Math.PI);
  ctx.fill();
}

// --- Edge weight: what you wrote outranks what was inferred -----------------
// Every edge already declares its own opacity and width (graph_export), and
// both were dead: the 3D branch drew every link at the bundle's flat 0.2 and
// the 2D branch at a flat width of 1. So the two layers that matter most —
// 1341 wikilinks you wrote and 2718 similarities a model guessed — arrived on
// screen at identical weight, separated by hue alone.
//
// The alpha lives in the colour string because that is the only per-link seam
// the 3D bundle has: linkOpacity is global, but it MULTIPLIES the alpha it
// parses out of an rgba(), so setting it to 1 makes the per-edge alpha the
// final word. 2D reads the same string straight into strokeStyle.
const LINK_ALPHA_2D = 0.55;   // see the comment on its use below
const EDGE_FALLBACK = LIGHT ? DATA.colors.extracted_paper : DATA.colors.extracted;

// --- Edge family: a broken line is a link that is not there ------------------
// Weight ranks the layers; it does not SEPARATE them, and seven kinds on the
// hue channel alone is more than a 1px antialiased line can carry. So a second
// channel, carrying the one division that survives being drawn: GAP is a bridge
// between two areas that does not exist, PROPOSED is a link the shape of the
// vault predicts and nobody wrote, COUPLED is two notes from one source "that
// never came to link each other". All three draw an absence. Everything else on
// the canvas is a relation that HOLDS - a wikilink you wrote, the discord
// recolour of those same wikilinks, and the k-NN similarity, which is measured
// and not proposed - so all of it stays solid. It is the dashed line's oldest
// meaning on a map: the road that is planned, the border nobody agreed.
//
// SIMILAR was in this set for one build, on the wider rule "solid is what you
// wrote". Wrong twice. Wrong by meaning, because a similarity is not a missing
// link, it is a different KIND of link. And wrong by weight: it is 2901 of the
// 4246 visible edges, so dashing it turned the layer that has to read as ground
// into 8703 dots and made it the subject - which is the same mistake the
// particle budget below already refused for the same layer, in the same words.
// The dash is worth having on the three rows you can COUNT; on a field it is
// texture. SIMILAR was never the layer anyone confused: it owns the only
// saturated cool hue, the lowest opacity and the thinnest width already.
const ABSENT = new Set(["GAP", "PROPOSED", "COUPLED"]);
const isDashed = l => ABSENT.has(l.type);

// Three dashes and two gaps, as fractions of the link. Ink 0.78 of the length,
// split so a dash lands on each end: a broken line that stopped short of its
// target would read as a link that does not arrive, which is a different claim
// from the one this channel makes.
//
// ponytail: the count is fixed, so a long link gets long dashes. Fine at this
// vault's spread; the fix if a picture ever reads as noise is a period in world
// units with a per-edge count, which costs a variable-length span the segOff
// table below already knows how to carry. Revisit if a vault's longest link is
// past 6x its median.
const DASH_SEGS = 3;
const DASH_T = [0, 0.26, 0.37, 0.63, 0.74, 1];
const DASH_2D = [6, 2];

// Memoised because 2D calls this for every visible link on every frame, and
// building the string costs ~1ms per frame across 4.6k edges — measured, not
// assumed. An edge's colour depends on nothing but its own (fixed) data, its
// dim state and the renderer, so those two are the whole cache key.
function linkPaint(l) {
  // The discord flag is part of the key, not a branch after the lookup: it
  // changes the colour of an edge whose data is otherwise fixed, so a cache
  // that ignored it would keep painting the last answer after the toggle.
  const lit = showDiscord && l.discord;
  const key = (l._dim ? "d" : "") + (is2D() ? "2" : "3") + (lit ? "x" : "");
  if (l.__paintKey === key) return l.__paint;
  let out;
  if (l._dim) out = GP.linkDim;
  else {
    const c = lit
      ? { color: DATA.colors.discord, paper: DATA.colors.discord_paper, opacity: 0.85 }
      : (l.color || {});
    const hex = (LIGHT ? c.paper || c.color : c.color) || EDGE_FALLBACK;
    const n = parseInt(hex.slice(1), 16);
    // 2D stacks every edge on one flat plane with no depth to thin the far ones
    // out, so the same alphas that read as structure in 3D read as a mat there.
    // One scale factor rather than a second table: the RANK is what is worth
    // preserving between the two views, not the absolute values.
    const a = (c.opacity == null ? 0.6 : c.opacity) * (is2D() ? LINK_ALPHA_2D : 1);
    out = "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," +
          (n & 255) + "," + a.toFixed(3) + ")";
  }
  l.__paintKey = key; l.__paint = out;
  return out;
}

// --- 3D: every link in ONE LineSegments -------------------------------------
// The bundle builds a separate THREE.Line per link: one draw call and one
// per-tick geometry write each, which on this vault is ~9k of either and
// ~36ms a frame before a single sphere is drawn — measured, and the reason
// the 3D view lagged. So the lib's lines are never shown (linkVisibility
// false) and never positioned (linkPositionUpdate returns true), and this
// layer draws every visible link as one LineSegments: one object, one draw
// call, two vertices per link, RGBA per vertex so the per-edge alpha rank
// survives the merge. Fog applies to the shared material like it did to the
// per-link ones.
//
// No THREE global exists (same constraint as faceteNodes below), and no
// LineSegments instance exists to steal a constructor from. A Line instance
// does — and the renderer branches on the isLineSegments FLAG, not the class,
// so a Line wearing the flag renders as GL_LINES. The donor is the ONE line
// the lib is allowed to build: its visibility accessor admits RAW_EDGES[0]
// only, and an accessor that returns false makes the lib skip creating the
// object outright (verified, not assumed) — so the whole per-link fleet, its
// 9k materials included, is simply never born. The donor itself stays
// degenerate at the origin (linkPositionUpdate never lets it be positioned)
// and draws nothing.
//
// Particles are the one carve-out. A photon group is only created for a link
// whose line object exists (verified: admit the link, photons appear; skip
// it, they never do), so the links that may carry photons — GAP always,
// SIMILAR for the drift — stay lib-owned whenever PARTICLES is on: real
// lines, lib-positioned, excluded from the merge. PARTICLES off (the
// default) merges everything behind the one donor.
const libOwnsLink = l => PARTICLES && (l.type === "GAP" || l.type === "SIMILAR");
let LinkSeg = null;   // { obj, pos, col, colorCls, edges }

// The same conversion Color.setStyle applies (sRGB into the working space),
// without setStyle's per-call "alpha will be ignored" console warning.
function segRGBA(Color, s) {
  let r, g, b, a = 1;
  if (s[0] === "#") {
    const n = parseInt(s.slice(1), 16);
    r = (n >> 16) & 255; g = (n >> 8) & 255; b = n & 255;
  } else {
    const p = s.match(/rgba?\(([^)]+)\)/)[1].split(",").map(Number);
    r = p[0]; g = p[1]; b = p[2];
    if (p.length > 3) a = p[3];
  }
  const c = new Color().setRGB(r / 255, g / 255, b / 255, "srgb");
  return [c.r, c.g, c.b, a];
}

function buildLinkSeg() {
  const line = RAW_EDGES.length && RAW_EDGES[0].__lineObj;
  if (!line) return;               // digest not run yet; next frame retries
  const E = RAW_EDGES.length;
  // Segments, not links: a dashed edge is DASH_SEGS disjoint pairs in the same
  // buffer. Counted rather than allocated at the worst case, because the dashed
  // rows are the three SMALL ones (18 edges of 4938 on the reference vault), so
  // the true size is E+36 where DASH_SEGS * E would have been three times the
  // buffer to draw the same picture.
  const S = RAW_EDGES.reduce((n, e) => n + (isDashed(e) ? DASH_SEGS : 1), 0);
  const pos = new Float32Array(S * 6);
  const col = new Float32Array(S * 8);
  const Attr = line.geometry.getAttribute("position").constructor;
  const geom = new (line.geometry.constructor)();
  geom.setAttribute("position", new Attr(pos, 3));
  geom.setAttribute("color", new Attr(col, 4));
  const mat = new (line.material.constructor)({ vertexColors: true, transparent: true });
  const obj = new (line.constructor)(geom, mat);
  obj.isLineSegments = true;       // the renderer reads the flag, not the class
  obj.type = "LineSegments";
  obj.frustumCulled = false;       // positions churn every tick; skip bounds
  obj.raycast = () => {};        // the pointer belongs to the nodes
  (line.parent || Graph.scene()).add(obj);
  // segOff[i] is where edge i's segments start and segOff[i+1] where they end,
  // so the span carries the count and no second array has to stay in step with
  // it. E+1 long for the sentinel.
  LinkSeg = { obj, pos, col, colorCls: line.material.color.constructor,
              edges: [], off: new Uint32Array(E + 1) };
  repaintLinkSeg();
}

// Colours + the visible set — only when they change (filters, focus, dim),
// which is what refreshPaint/applyFilters call in place of the accessor
// re-pass that used to rebuild 9k materials.
function repaintLinkSeg() {
  if (!LinkSeg) return;
  const edges = LinkSeg.edges = RAW_EDGES.filter(e => !e._hidden && !libOwnsLink(e));
  const off = LinkSeg.off;
  let s = 0;
  for (let i = 0; i < edges.length; i++) {
    const c = segRGBA(LinkSeg.colorCls, linkPaint(edges[i]));
    off[i] = s;
    const k = isDashed(edges[i]) ? DASH_SEGS : 1;
    for (let j = 0; j < k; j++) {
      LinkSeg.col.set(c, (s + j) * 8);
      LinkSeg.col.set(c, (s + j) * 8 + 4);
    }
    s += k;
  }
  off[edges.length] = s;
  LinkSeg.obj.geometry.setDrawRange(0, s * 2);
  LinkSeg.obj.geometry.getAttribute("color").needsUpdate = true;
  writeLinkSegPositions();
}

function writeLinkSegPositions() {
  const { pos, edges, off, obj } = LinkSeg;
  for (let i = 0; i < edges.length; i++) {
    const a = NODE_BY_ID[edges[i].from], b = NODE_BY_ID[edges[i].to];
    const ax = a.x || 0, ay = a.y || 0, az = a.z || 0;
    const s = off[i], k = off[i + 1] - s, o = s * 6;
    // The solid case keeps its straight write. It is most of the buffer on any
    // vault, and it runs every tick a node can still move.
    if (k === 1) {
      pos[o]     = ax;       pos[o + 1] = ay;       pos[o + 2] = az;
      pos[o + 3] = b.x || 0; pos[o + 4] = b.y || 0; pos[o + 5] = b.z || 0;
      continue;
    }
    const dx = (b.x || 0) - ax, dy = (b.y || 0) - ay, dz = (b.z || 0) - az;
    for (let j = 0; j < k; j++) {
      const q = o + j * 6, t0 = DASH_T[j * 2], t1 = DASH_T[j * 2 + 1];
      pos[q]     = ax + dx * t0; pos[q + 1] = ay + dy * t0; pos[q + 2] = az + dz * t0;
      pos[q + 3] = ax + dx * t1; pos[q + 4] = ay + dy * t1; pos[q + 5] = az + dz * t1;
    }
  }
  obj.geometry.getAttribute("position").needsUpdate = true;
}

// Per frame beside the label layers (the loop always runs in 3D): build once
// the lib's digest has produced a carrier, then follow the nodes — but only
// while they can move. A settled, unwoken graph skips the write entirely.
function linkSegStep() {
  if (is2D()) return;
  if (!LinkSeg) { buildLinkSeg(); return; }
  if (!simRunning && performance.now() >= awakeUntil) return;
  writeLinkSegPositions();
}

// --- 3D: the same crystal the mark is cut from ------------------------------
// PARTICLES/SHADING come from the settings panel (Display). Both off is the
// bundle's own look: smooth lit spheres, no fog, still edges.
const PARTICLES = DATA.particles;
const SHADING = DATA.shading;
// Everything here is a bundle default undone. Out of the box the scene is lit
// by a GRAY ambient at full strength plus a white key, the spheres are smooth-
// shaded, and there is no fog — which is a fine neutral rig for a demo and the
// wrong one for this palette. A substrate built as blue-black crystal, lit flat
// by an office light, renders as plastic beads: that, not the library, is what
// made this view read as a stock 3D graph while the rest of the app did not.
//
// No THREE global exists to build with (see the DOM label layers below for the
// same constraint). It is not needed: the bundle hands out the scene, and every
// constructor this wants is reachable from an object already inside it.
// Vendored+pinned bundles are what makes that safe to lean on.
//
// Two rigs, because light is not the same scene with the background swapped.
// The community colours already dropped ~28 lightness points to survive a paper
// floor (_COMMUNITY_LIGHTNESS_ON_PAPER), and a multiplying ambient at crystal
// strength takes them the rest of the way to mud. So on paper the ambient rises
// toward white and the key comes down: the facets still split, but the light
// falling on them is a room's, not a lamp inside the stone. Fog travels with
// the floor either way — it is the same depth cue, sold into white instead of
// out of black.
const CRYSTAL = LIGHT ? {
  ambient: 0xEFEAE0, ambientI: 1.55,
  key: 0xFFF6E4, keyI: 1.35,      // warm key, matching the paper it lands on
  fogNear: 0.55, fogFar: 2.30,    // paper fog swallows less: white on white
  fov: 42,
} : {
  // Ambient MULTIPLIES each node's own colour, so this is the one value that
  // can silently destroy the community channel: a saturated violet here turned
  // every community violet, and a bright one flattened the facets back into
  // spheres. Low enough that lit and unlit facets separate, unsaturated enough
  // that the hue survives into the unlit half.
  ambient: 0x9B93C6, ambientI: 1.05,
  key: 0xDCE8FF, keyI: 2.40,      // cool key, off-axis so facets split
  fogNear: 0.45, fogFar: 2.10,    // as fractions of the camera's distance
  fov: 42,                        // 50 is the wide-angle look; this is a lens
};

// Re-facet after every wholesale material rebuild. refreshPaint triggers one on
// every focus change (see its comment), so this cannot be a one-shot. The flag
// rides on the material itself: a rebuilt material simply arrives without one,
// which makes the repeat cost a 1150-entry loop that writes nothing.
// __threeObj is the bundle's own per-node handle — 18x cheaper than walking the
// scene, which also carries ~2700 particle meshes that want none of this.
function faceteNodes() {
  let Color = null;
  for (const n of RAW_NODES) {
    const m = n.__threeObj && n.__threeObj.material;
    if (!m) continue;
    if (!Color) Color = m.color.constructor;
    if (m.__silica) continue;
    // A 6-segment sphere stops pretending to be round and becomes a facet
    // cluster. The geometry never changed; it just stopped being smoothed.
    m.flatShading = true;
    m.needsUpdate = true;
    m.__silica = 1;
  }
  return Color;
}

function styleScene() {
  if (!SHADING || is2D() || !Graph || !Graph.scene) return;
  const sc = Graph.scene();
  const Color = faceteNodes();
  if (!Color) return;
  // Per-light flags, not one scene-level one. The bundle populates its scene
  // lazily: at construction and immediately after graphData it holds only the
  // background mesh, and the lights and the node Group appear some frames
  // later. A single "already styled" mark set the moment the nodes showed up
  // could therefore land in a window where the lights had not, and then said
  // done forever — which is exactly what it did. Flagging each light drops the
  // ordering assumption: one that arrives late, or gets replaced, is styled on
  // the next frame instead of never. Direct children only, so this is four
  // entries rather than a walk over ~3900 meshes.
  for (const o of sc.children) {
    if (!o.isLight || o.__silica) continue;
    if (o.type === "AmbientLight") {
      o.color = new Color(CRYSTAL.ambient); o.intensity = CRYSTAL.ambientI;
    } else if (o.type === "DirectionalLight") {
      o.color = new Color(CRYSTAL.key); o.intensity = CRYSTAL.keyI;
      o.position.set(0.55, 1, 0.75);
    } else continue;
    o.__silica = 1;
  }
  if (sc.__silicaLit) return;
  // Linear fog, not exponential. Exponential density is anchored to world units,
  // and the camera here travels three orders of magnitude between a fitted vault
  // and a single note: one density either does nothing up close or swallows the
  // whole graph the moment you pull back — which is exactly what it did. Linear
  // near/far ride the camera instead, so the depth cue reads the same at every
  // zoom. No Fog constructor is reachable (no instance exists to borrow one
  // from), but the renderer reads the shape, not the class.
  sc.fog = { isFog: true, isFogExp2: false, name: "",
             color: new Color(GP.bgHex), near: 1, far: 2 };
  const cam = Graph.camera();
  cam.fov = CRYSTAL.fov;
  cam.updateProjectionMatrix();
  sc.__silicaLit = 1;
}

// Per frame, beside the label layers: the fog slab has to follow the camera or
// it is just a fixed band the graph flies through.
function fogStep() {
  if (!SHADING || is2D() || !Graph || !Graph.scene) return;
  const sc = Graph.scene();
  if (!sc.fog) return;
  const p = Graph.camera().position;
  const d = Math.hypot(p.x, p.y, p.z);
  sc.fog.near = d * CRYSTAL.fogNear;
  sc.fog.far = d * CRYSTAL.fogFar;
}

// --- the renderer, either dimension ----------------------------------------
// /graph regenerates the whole document per request (cooccurrence refresh + kNN
// + Louvain), so the switch must never reload: it destroys the instance and
// rebuilds from the RAW_NODES/RAW_EDGES already in the page. The two libs are
// kapsule siblings — one builder, four branches (fly-to, sphere detail, link
// width, labels), everything else shared.
const MODE_KEY = "silica-graph-mode";
let mode = "3d";
try { if (localStorage.getItem(MODE_KEY) === "2d") mode = "2d"; } catch (e) {}
const is2D = () => mode === "2d";

let Graph = null;
let fitPending = false;  // one-shot zoomToFit after a rebuild

// --- render budget: stop paying 60fps for a picture that stopped moving -----
// Neither bundle idles on its own here. 3d-force-graph's _animationCycle is
// unconditional: tickFrame + render + requestAnimationFrame, every frame,
// forever. force-graph DOES have autoPauseRedraw (default on), but its wake
// condition includes `links.some(l => l.__photons.length)` — so the five GAP
// particle links hold the whole canvas awake, repainting all 553 nodes at 60Hz
// to move ten dots. On a settled graph that is the single largest cost in this
// view, and it is pure waste: nothing on screen changes.
//
// So: full rate while the layout settles, the camera tweens or the pointer is
// over the graph; IDLE_FPS when only particles still move; nothing when even
// those are off. The failure mode is chosen — a missed wake signal leaves the
// loop running, i.e. exactly today's behaviour, never a frozen view.
const IDLE_FPS = 20;
const WAKE_MS = 1200;   // trackball inertia keeps the camera moving after pointerup
let awakeUntil = 0, simRunning = true, idleTick = null, sleepTimer = null;

// Both particle layers keep the canvas awake, so both have to be asked. The
// similarity layer is on by default, which means the idle tick now effectively
// always runs — that is the price of the drift, and it is the reason the drift
// rides the 20fps budget instead of the full frame rate.
const particlesMoving = () => PARTICLES && RAW_EDGES.some(e =>
  !e._dim && !e._hidden &&
  ((e.type === "GAP" && showGaps) || (e.type === "SIMILAR" && showSimilar)));

function renderBudget() {
  if (!Graph) return;
  clearInterval(idleTick); idleTick = null;
  if (simRunning || performance.now() < awakeUntil) { Graph.resumeAnimation(); return; }
  Graph.pauseAnimation();
  // resumeAnimation() runs one cycle synchronously and re-arms rAF;
  // pauseAnimation() cancels the re-arm. Net effect: exactly one frame.
  if (particlesMoving()) idleTick = setInterval(
    () => { Graph.resumeAnimation(); Graph.pauseAnimation(); }, 1000 / IDLE_FPS);
}

// Keep rendering at full rate for `ms`. Every mutation and every camera tween
// calls this; the tail covers control inertia and tween duration.
function wake(ms = WAKE_MS) {
  awakeUntil = Math.max(awakeUntil, performance.now() + ms);
  clearTimeout(sleepTimer);
  sleepTimer = setTimeout(renderBudget, awakeUntil - performance.now() + 30);
  renderBudget();
}

// --- dynamic resolution: motion may be soft, stillness never ----------------
// The 2D canvas is fill-bound: a full repaint costs ~15ms per megapixel of
// backing store on software raster, so a hot simulation on a 2263x1339 canvas
// delivers 15-22fps with the main thread mostly idle — measured ~8ms of script
// against 45-90ms between frames, rAF starved by raster back-pressure, which
// is why no JS profile ever showed it. Pixels are the cost: the same scene at
// half ratio (a quarter of the pixels) measured ~240fps. So while repaints are
// STREAMING in full-rate mode the backing store drops to half ratio — softness
// is invisible in motion — and when the stream ends the full ratio returns
// with one crisp repaint. A still frame is never soft, and the idle particle
// tick (parked, 20fps) never triggers this: what it paints IS stillness.
//
// One overridden getter keeps every consumer consistent: the lib reads
// window.devicePixelRatio live for the backing store (resize path), the
// per-frame canvas transform, and the shadow-canvas picking coords. 2D only —
// the WebGL renderer is not fill-bound and keeps its native ratio.
// Captured BEFORE the override, and re-read on resize: a browser zoom or a drag
// to a HiDPI monitor changes the real ratio, and once the getter below is in
// place no later read can see it — a one-shot snapshot left the canvas at the
// old device ratio for the rest of the session.
const dprNative = (() => {
  const d = Object.getOwnPropertyDescriptor(window, "devicePixelRatio");
  const get = d && d.get;
  const seed = window.devicePixelRatio || 1;
  return () => (get ? get.call(window) : seed) || 1;
})();
let DPR_REAL = dprNative();
const DRS_SCALE = 0.5;
let dprScale = 1;
Object.defineProperty(window, "devicePixelRatio",
  { get: () => DPR_REAL * dprScale, configurable: true });
window.addEventListener("resize", () => { DPR_REAL = dprNative(); });

function setRes(scale) {
  if (dprScale === scale || !Graph || !is2D()) return;
  dprScale = scale;
  // A same-value set still runs the lib's resize path: it rebuilds both
  // canvases at the new ratio (verified live) — and leaves them BLANK, so a
  // repaint must follow. Going low, the stream's next frame is that repaint;
  // going crisp, paint the one frame here and hand the loop to the budget.
  //
  // The resize path also re-centers: it estimates the old CSS size as
  // canvas.width / devicePixelRatio and shifts the zoom transform by half the
  // difference. That estimate divides OLD pixels by the NEW ratio, so on a
  // ratio change (the only reason we are here) it is wrong by exactly the
  // ratio step and the camera lurches half a screen per transition. The CSS
  // size never changes in this resize, so the correct shift is zero: save the
  // camera, let the lib do its arithmetic, put the camera back.
  const z = Graph.zoom(), c = Graph.centerAt();
  Graph.width(Graph.width()).height(Graph.height());
  Graph.zoom(z);
  Graph.centerAt(c.x, c.y);
  // Both directions paint one frame right here: the resize left the canvases
  // blank, and a repaint may not otherwise be due (a grab that has not moved
  // yet, a restore on a parked loop). That paint re-enters drsOnPaint, which
  // re-arms the restore timer — an engage with no follow-up stream still finds
  // its way back to crisp. renderBudget re-derives the loop state the pause
  // just clobbered.
  Graph.resumeAnimation(); Graph.pauseAnimation(); renderBudget();
}

// The burst detector below covers streams that no pointer announces (layout
// settle, slider reheat, camera tweens): three rapid repaints in full-rate
// mode prove a stream. Pointer-driven camera work does NOT wait for that
// warm-up — its three full-res paints cost 150-270ms at the START of every
// pan stroke, and with pan-pause-pan exploration each resume paid it again,
// which read as "panning the settled graph is slow". A drag or a wheel notch
// is motion by declaration: the graph-wrap listener at the bottom of this
// file calls setRes(DRS_SCALE) directly, before the first heavy paint.
let drsTimer = null, drsLast = 0, drsBurst = 0;
function drsOnPaint() {
  const now = performance.now();
  const streaming = simRunning || now < awakeUntil;
  if (streaming) {
    drsBurst = now - drsLast < 250 ? drsBurst + 1 : 0;
    // Deferred: resizing from inside the paint would clear the canvas mid-frame.
    if (drsBurst >= 3 && dprScale === 1) setTimeout(() => setRes(DRS_SCALE));
  } else drsBurst = 0;
  drsLast = now;
  clearTimeout(drsTimer);
  // Only a STREAMING paint arms the restore timer. The idle particle tick paints
  // every 1000/IDLE_FPS = 50ms, which is shorter than this window — so letting it
  // re-arm meant the timer could never elapse and one pan left the settled graph
  // permanently soft, inverting this block's whole premise. What the idle tick
  // paints IS stillness, so it restores immediately instead. The restore paint
  // re-enters here with dprScale already 1, which arms nothing: no oscillation.
  if (streaming) drsTimer = setTimeout(() => setRes(1), 250);
  else if (dprScale !== 1) drsTimer = setTimeout(() => setRes(1));
}

// zoomToFit measures the SCENE, not the data. 3d-force-graph's getGraphBbox
// unions the node meshes' world boxes, and a mesh only takes its node's
// position on a rendered frame — so fitting before the first frame measures
// 1,445 meshes stacked at the origin and returns the largest node RADIUS as the
// graph's extent. Measured: bbox +/-14 against positions spanning +/-2,300, and
// a camera parked 41 units from the centre of a 3,900-unit graph, which is what
// "the 3D view opens fully zoomed in" actually is.
//
// So the fit waits for a frame it can trust. wake() is first because the budget
// loop may be parked, and a paused loop paints nothing to wait for; the two
// frames then cover our own callback landing ahead of the library's render on
// the same tick.
function fitWhenPainted(G) {
  wake(1200);
  requestAnimationFrame(() => requestAnimationFrame(() => {
    G.zoomToFit(400, 40);
    wake(600);
  }));
}

function buildGraph() {
  // Direct write, not setRes: the old instance is about to die, and the new
  // one (either renderer) must initialize its canvas at the native ratio.
  dprScale = 1;
  const el = document.getElementById("graph");
  if (Graph) {
    // Loud on failure: a swallowed teardown leaks the WebGL context, and the
    // browser caps them at ~16 — the graph would go black after enough mode
    // switches with nothing in the console to say why.
    try { Graph._destructor(); } catch (e) { console.warn("graph teardown failed", e); }
    el.innerHTML = "";
    LinkSeg = null;   // died with the scene; the next 3D build remakes it
  }
  const G = is2D() ? new ForceGraph(el) : new ForceGraph3D(el);
  // Seeded positions and the decay that goes with them, decided before the data
  // lands: on a cache hit the whole rerelaxation fits inside the warmup, so the
  // graph appears already settled instead of unfolding a layout you have
  // watched unfold before.
  const seeded = loadLayout();
  const warm = WARMUP_TICKS(seeded);
  warmupRun = warm; simTicks = 0;
  paintStat("settling");
  G.warmupTicks(warm)
    .d3AlphaMin(ALPHA_MIN)
    .d3AlphaDecay(seeded ? FAST_DECAY : 0.0228)
    .cooldownTicks(Infinity)   // alpha is the gate; see ALPHA_MIN
    .cooldownTime(Infinity)
    .backgroundColor(GP.bg)
    .linkSource("from").linkTarget("to")
    .nodeVal("size")
    .nodeColor(nodeColor)
    .linkColor(linkPaint)
    // Motion. The line above this one used to read "structural gaps have no dash
    // in WebGL, mark them by motion instead", which was true of the per-link
    // THREE.Lines this frame drew before the merge and is not true of what the
    // merge left behind: LineSegments draws disjoint vertex pairs, so a dash is
    // what that buffer natively IS (see writeLinkSegPositions). GAP now carries
    // both, and needs to - with PARTICLES on it is lib-owned (libOwnsLink),
    // which puts it outside the merge and out of reach of the dash, so motion is
    // the only channel it keeps in that config. SIMILAR is solid by design (see
    // ABSENT) and motion is the whole of what marks it. Amber particles stream
    // along the absent bridge. The similarity layer gets the
    // same treatment at a fraction of the weight: one particle instead of two,
    // pre-dimmed almost into the background, and six times the speed, so two
    // thousand of them read as a faint drift through the semantic neighbourhood
    // rather than as two thousand moving dots. Both stop when the link is dimmed
    // or filtered out, so focus mode stays quiet and unticking Similar takes its
    // meshes with it. Same binding in both libs, so the layers read the same
    // either way.
    .linkDirectionalParticles(l => (!PARTICLES || l._dim || l._hidden) ? 0
      : l.type === "GAP" ? 2 : l.type === "SIMILAR" ? 1 : 0)
    .linkDirectionalParticleSpeed(l => l.type === "SIMILAR" ? 0.06 : 0.01)
    .linkDirectionalParticleColor(l => l.type === "SIMILAR"
      ? (LIGHT ? DATA.colors.similar_particle_paper : DATA.colors.similar_particle)
      : (LIGHT ? DATA.colors.gap_paper : DATA.colors.gap))
    .linkDirectionalParticleWidth(l => l.type === "SIMILAR" ? 1.5 : 2)
    .nodeVisibility(n => !n._hidden)
    .linkVisibility(l => !l._hidden)
    .onNodeClick((node, event) => { selectNode(node, event); applyFocus(node.id); })
    .onBackgroundClick(() => { closeDrawer(); clearFocus(); })
    // The alpha gate and a drag deadlock each other. Both bundles reheat a drag
    // the d3 way — d3AlphaTarget(0.3).resetCountdown() on every drag event — but
    // alpha only climbs toward that target INSIDE tick(), and tick() is exactly
    // what the gate refuses to run while alpha still sits at the settled value
    // it stopped on. So the engine stops again on the same frame, forever. The
    // grabbed node still follows the pointer (both libs write its position
    // directly), which is why the symptom is one node moving through a graph
    // that has stopped answering. Lift the gate for the drag and let alpha do
    // what it was going to do; dragend puts the gate back, alphaTarget returns
    // to 0, and the layout settles into onEngineStop as usual.
    .onNodeDrag(() => {
      if (simRunning) return;
      simRunning = true;
      G.d3AlphaMin(0);
      renderBudget();   // a still pointer stops waking the loop mid-drag
    })
    .onNodeDragEnd(() => G.d3AlphaMin(ALPHA_MIN))
    .onEngineTick(() => { simTicks++; })
    .onEngineStop(() => {
      simRunning = false;
      paintStat("settled in " + gnum(warmupRun + simTicks) + " ticks");
      for (const k in mstMemo) delete mstMemo[k];  // positions just moved
      saveLayout();
      measureGraphRadius();   // the label thresholds are a fraction of it
      if (fitPending) { fitPending = false; fitWhenPainted(G); }
      else renderBudget();
    });

  if (is2D()) {
    // Width 0 is invisible on canvas (GL draws a 1px line for it, 2D draws
    // nothing), and the labels ARE the point here, so pay for them. Width is
    // per-edge because on canvas it is free, and it is the second half of the
    // rank the alpha starts: a wikilink is both stronger and thicker than the
    // similarity that a model proposed beside it.
    G.linkWidth(l => l.width || 1)
      // The dash the 3D side has to build by hand (see LinkSeg), free here:
      // the canvas painter ends on setLineDash(accessor || []). Paint-time and
      // untriggered, so a family that costs one buffer rewrite over there costs
      // nothing at all on this side.
      //
      // ponytail: the pattern is in graph units, and the painter runs inside
      // the zoom transform, so dashes lengthen as you zoom in. It reads as the
      // whole picture growing rather than as a pattern changing, which is why
      // it stays. Screen-space would mean owning the link painter outright
      // (linkCanvasObject) to divide by scale the way drawNode does its font.
      .linkLineDash(l => isDashed(l) ? DASH_2D : null)
      .nodeRelSize(NODE_REL_SIZE)
      .nodeCanvasObject(drawNode)
      .nodePointerAreaPaint(paintNodeArea)
      // The picking canvas repaints on a ~800ms debounce, and its link pass
      // strokes every edge at width+4px — 40-80ms on this vault, landing as a
      // visible hitch once or twice a second through any pan. Nothing hovers
      // or clicks a link here (no linkLabel, no onLinkHover), so the pass
      // buys nothing: paint no link areas at all. Node picking keeps its own
      // painter above. Measured: worst 2D frame 78ms -> 9ms.
      .linkPointerAreaPaint(() => {})
      // Pre, not Post: a zone is the ground the notes stand on. 2D only — the
      // 3D bundle hands out no THREE, so there the zones are colour and name.
      // Every real 2D repaint passes here, which is what makes it the seam
      // where the resolution governor watches for streams.
      .onRenderFramePre((ctx, scale) => { drsOnPaint(); drawZones(ctx, scale); });
  } else {
    // Perf on big vaults (1200+ notes): the bundle gives every link its own
    // THREE.Line — a draw call and a per-tick buffer write each. The links are
    // drawn merged instead (see LinkSeg above), and the visibility accessor
    // admits exactly one lib line into existence: the constructor donor. A
    // falsy accessor result skips object creation entirely, so the other 9k
    // Lines and their materials are never built at all. linkWidth 0 keeps the
    // donor a cheap Line rather than a cylinder mesh; fewer sphere segments.
    G.linkWidth(0).nodeResolution(6)
      .linkVisibility(l => l === RAW_EDGES[0] || (libOwnsLink(l) && !l._hidden))
      // "handled" for merged links and for the donor, which stays degenerate;
      // lib-owned particle carriers keep the default update and really move.
      .linkPositionUpdate((o, coords, l) => !libOwnsLink(l))
      // 1 so the per-edge alpha in linkPaint is the final opacity rather than
      // being scaled by a second global. The bundle's default 0.2 is what put
      // every link at the same weight in the first place.
      .linkOpacity(1)
      // The bundle's own onboarding line, bottom centre of every scene it has
      // ever rendered. It is the single most recognisable thing about the
      // library, it teaches three mouse bindings nobody needed taught, and it
      // is not written in this app's voice.
      .showNavInfo(false)
      // 0.75 is the default, and at 0.75 every node shows through every other
      // one: the cluster reads as gas. Solid nodes plus the fog below carry the
      // depth instead, which is the reading that was wanted all along.
      .nodeOpacity(0.96)
      // The default tooltip is the bare label in the library's own black
      // rounded box. This one is a Silica compartment (see .float-tooltip-kap)
      // and it answers the question the hover is actually asking at a distance
      // the note labels do not reach: which cluster is this, and is it special.
      //
      // The community comes first because that is what the node's colour means,
      // always. The zone is APPENDED when its layer is up, never substituted:
      // in 3D there are no hulls to carry it (onRenderFramePre is 2D-only), so
      // without this the semantic layer would be floating names and nothing an
      // individual note could be checked against.
      // The zone gets its OWN line rather than joining the first: community
      // labels already contain " · " inside themselves, so appending to them
      // produced "etica · sistemi · zone: etica · morale", one run with no seam
      // where the second partition starts.
      .nodeLabel(n => {
        const bits = [];
        if (COMM_LABELS[n.group]) bits.push(COMM_LABELS[n.group]);
        const st = nodeState(n);
        if (st !== "note") bits.push(st);
        const zone = (showZones && n.sgroup >= 0) ? ZONE_LABEL[n.sgroup] : null;
        // Appended on the same terms as the zone, and for the same reason: 3D
        // has no ring, so with the layer up this line is the only place the
        // reading can land at all.
        const cut = n.cut
          ? "load-bearing · strands " + (n.strands || 0) : null;
        return '<div class="g3d-tip"><b>' + escHtml(n.label) + '</b>' +
          (bits.length ? '<i>' + escHtml(bits.join(" · ")) + '</i>' : '') +
          (zone ? '<i>zone ' + escHtml(zone) + '</i>' : '') +
          (cut ? '<i>' + escHtml(cut) + '</i>' : '') + '</div>';
      });
  }
  // Forces before the data, and the data last of all. The warmup loop runs the
  // moment graphData lands, so anything set afterwards shapes only the animated
  // tail: with the tuned forces arriving late, the bulk of the layout was being
  // built by d3's untouched defaults and then nudged at low alpha. Widening the
  // warmup made that worse, which is how it surfaced.
  applyForces(false, G);
  G.graphData({ nodes: RAW_NODES, links: RAW_EDGES });
  return G;
}

// Switching preserves everything that is not the camera: edge filters, the
// community filter, the focused set and the search box all live outside the
// instance. The camera cannot be preserved — there is no sane mapping from a 3D
// camera to a 2D pan/zoom — so it refits once the new layout settles.
function setMode(m) {
  // Announced before the early return as well: the host toolbar has to hear the
  // mode even when the answer is "already in it", or a click that changes
  // nothing leaves the segment showing the mode it just moved away from.
  announceMode(m);
  if (m === mode && Graph) return;
  const rebuild = Graph !== null;   // a switch always refits; a first build may not
  mode = m;
  try { localStorage.setItem(MODE_KEY, m); } catch (e) {}
  document.querySelectorAll("#mode-toggle button")
    .forEach(b => b.classList.toggle("active", b.dataset.mode === m));
  // The rings are a canvas draw. Leaving their legend up in 3D would promise a
  // channel that mode does not paint.
  document.getElementById("state-legend").style.display = is2D() ? "" : "none";
  fitPending = fitPending || rebuild;
  // Flags BEFORE the build, repaint never: the new instance reads _hidden/_dim
  // in its first graphData() digest. Recomputing them afterwards used to cost
  // four more full digests to arrive at the state the build already had.
  computeFilters();
  computeFocus(focusIds);
  simRunning = true;
  Graph = buildGraph();    // owns the forces now: they must precede graphData
  window.__G = Graph;      // console/harness handle: frame-cost probes in hidden tabs
  styleScene();            // first frame already lit; a no-op if 2D
  syncZoneLoop();          // the note-name layer is 3D-only, so the mode owns it
  renderBudget();
}

// The app's toolbar carries this control when embedded (the HUD's own segment
// is hidden there), and it paints from THIS message rather than from the click:
// the frame is the only place that knows which renderer actually got built.
function announceMode(m) {
  if (window.parent === window) return;
  window.parent.postMessage({ type: "silica-renderer", mode: m }, "*");
}

// Slider multipliers persist across sessions; the baseline is never persisted
// (recomputed from the current graph each load).
const FORCES_KEY = "silica-graph-forces";
let forceMul = { repel: 1, dist: 1, center: 1 };
try {
  Object.assign(forceMul, JSON.parse(localStorage.getItem(FORCES_KEY)) || {});
} catch (e) { /* corrupt or blocked storage -> auto defaults */ }

// G is passed explicitly during a build, where the instance is not yet the
// global one: the forces have to be in place before graphData runs the warmup.
function applyForces(reheat, G) {
  G = G || Graph;
  // distanceMax bounds both over-dispersion and per-tick cost on big graphs.
  G.d3Force("charge").strength(baseCharge() * forceMul.repel)
    .distanceMax(600 * FORCE_SCALE);
  G.d3Force("link").distance(baseDist() * forceMul.dist);
  // Center capped at 1: d3 forceCenter shifts positions directly, >1 oscillates.
  G.d3Force("center").strength(Math.min(1, forceMul.center));
  // A slider moves an already-settled layout, so the reheat is a perturbation,
  // not a cold start: the fast decay is the right budget for it, and without it
  // dragging a slider would cost the full 300-tick schedule every time.
  if (reheat) {
    G.d3AlphaDecay(FAST_DECAY).d3ReheatSimulation();
    simRunning = true;
    renderBudget();
  }
}

// Log-scale track for the multiplier sliders: x1 sits mid-track and the
// useful 0.2-1 range gets half the travel instead of a sliver.
const fromSlider = v => Math.pow(10, +v);
const toSlider = m => Math.log10(m);

function syncForceUI() {
  document.getElementById("sl-repel").value = toSlider(forceMul.repel);
  document.getElementById("sl-dist").value = toSlider(forceMul.dist);
  document.getElementById("sl-center").value = forceMul.center;
  document.getElementById("fv-repel").textContent = forceMul.repel.toFixed(1) + "\u00d7";
  document.getElementById("fv-dist").textContent = forceMul.dist.toFixed(1) + "\u00d7";
  document.getElementById("fv-center").textContent = (+forceMul.center).toFixed(2);
}

function onForceSlider() {
  forceMul.repel = fromSlider(document.getElementById("sl-repel").value);
  forceMul.dist = fromSlider(document.getElementById("sl-dist").value);
  forceMul.center = +document.getElementById("sl-center").value;
  try { localStorage.setItem(FORCES_KEY, JSON.stringify(forceMul)); } catch (e) {}
  syncForceUI();
  applyForces(true);
}

function resetForces() {
  forceMul = { repel: 1, dist: 1, center: 1 };
  try { localStorage.removeItem(FORCES_KEY); } catch (e) {}
  syncForceUI();
  applyForces(true);
}

syncForceUI();

// Same split as computeFocus/applyFocus: flags on the shared objects here,
// repaint only when there is no rebuild coming to read them for free.
function computeFilters() {
  RAW_NODES.forEach(n => {
    n._hidden = !showNotes || (activeCommunity !== -2 && n.group !== activeCommunity);
  });
  RAW_EDGES.forEach(e => {
    // Notes off takes the edges with it: an edge between two invisible nodes is
    // a line to nowhere, and the macro read is exactly the one that cannot
    // afford 2718 of them.
    e._hidden = !showNotes ||
                (e.type === "EXTRACTED" && !showExtracted) ||
                (e.type === "AMBIGUOUS" && !showAmbiguous) ||
                (e.type === "GAP" && !showGaps) ||
                (e.type === "SIMILAR" && !showSimilar) ||
                (e.type === "PROPOSED" && !showProposed) ||
                (e.type === "COUPLED" && !showCoupled);
  });
}

function applyFilters() {
  computeFilters();
  // Re-pass the current accessor to force a visibility refresh without resetting the physics layout
  Graph.nodeVisibility(Graph.nodeVisibility());
  // 3D link visibility lives in the merged buffer; the accessor re-pass stays
  // for 2D and, with PARTICLES on, for the lib-owned photon carriers.
  if (is2D() || PARTICLES) Graph.linkVisibility(Graph.linkVisibility());
  if (!is2D()) repaintLinkSeg();
  wake(120);   // and re-evaluate the idle tick: gaps may have just been toggled
}

function updateEdgeFilter() {
  showExtracted = document.getElementById("cb-extracted").checked;
  showAmbiguous = document.getElementById("cb-ambiguous").checked;
  showGaps = document.getElementById("cb-gaps").checked;
  const cbSim = document.getElementById("cb-similar");
  if (cbSim) showSimilar = cbSim.checked;
  // Each of the four is rendered only where its layer has something in it, so
  // each is read defensively rather than assumed into existence.
  const cbProp = document.getElementById("cb-proposed");
  if (cbProp) showProposed = cbProp.checked;
  const cbCoup = document.getElementById("cb-coupled");
  if (cbCoup) showCoupled = cbCoup.checked;
  const cbDisc = document.getElementById("cb-discord");
  if (cbDisc) {
    showDiscord = cbDisc.checked;
    // Discord marks links that are already drawn, so it is a repaint and not a
    // filter: nothing appears or disappears, ~450 lines change colour.
    refreshPaint();
  }
  applyFilters();
}

// --- The semantic zone layer ------------------------------------------------
// Hull + name around the members of each k-NN Louvain cluster. Not a second
// view and not a second graph: the same frame, one more layer, so the two
// partitions are read against each other instead of one after the other.
//
// The k-NN FORCES are not toggled here and never were — d3's link force reads
// every link in graphData, and linkVisibility is a render-time accessor, so the
// SIMILAR edges pull whether or not they are drawn. That decoupling is what
// makes a hull honest: on a pure-wikilink layout the members of a semantic
// cluster sit all over the frame and its hull would be a lie about space.
// The region is a CORRIDOR OF CONSTANT WIDTH along the zone's minimum spanning
// tree, not a hull around it.
// A zone has to LOOK like one region — Louvain hands back a partition, and a
// grouping drawn as scattered islands reads as several. But a CONVEX hull buys
// that continuity by claiming the members are contiguous in SPACE, and they are
// not: the layout is driven by the wikilinks too, so a semantic zone is
// routinely scattered and its hull swallows whatever lies between the pieces.
//
// Measured on the 682-note vault, one settled layout, counting foreign notes
// that fall inside a zone's drawn region:
//
//   convex hull            675   1.88 regions over the average note   max 6
//   plain discs r=0.7·d    110   1.16                                 max 3   NOT continuous
//   MST corridor r=0.35·d  119   1.17                                 max 3   continuous
//
// So continuity is nearly free as long as the corridors are the SHORTEST set
// that joins the members — which is what a minimum spanning tree is. Striping
// every intra-zone k-NN edge instead was measured too: 2406 edges, many of them
// long sweeps across the frame, 300 foreign notes at a third of this width.
//
// What overlap survives is the real thing (spec §3): wikilink pull dragging a
// note into another zone's neighbourhood. The gaps stay gaps, which is the
// information the tessellation was rejected to keep.
//
// Width follows the layout, never a constant: baseDist() IS the link force's
// rest length, and it landed within 15% of this vault's measured median
// nearest-neighbour distance (84 vs 73). So the zones breathe with the vault's
// density and with the Link-distance slider instead of going blobby or grainy.
// Two widths, because one does not read: at a single width wide enough to fuse
// neighbouring branches the corridors claim the whole neighbourhood, and at one
// narrow enough to be honest the zone reads as a tangle of tubes rather than a
// territory. So the members get a BULB and the tree gets an ISTHMUS, and the
// isthmus only shows where the bulbs do not already touch.
const ZONE_BULB = () => 0.70 * baseDist() * forceMul.dist;   // fuses at typical spacing
const ZONE_LINK = () => 0.22 * baseDist() * forceMul.dist;   // half-width of a bridge
const ZONE_ALPHA = 0.13;   // a backdrop; the notes stay the figure

// Prim, O(m²) — the textbook array form, not the naive rescan: with the tree
// scanned per candidate it is O(m³) and cost 7.9ms a frame on the largest zone
// here, against 1.0ms for all 18 zones this way.
//
// While the engine is stopped the positions are frozen, so drawZones serves
// each zone's tree from mstMemo (cleared on every onEngineStop, bypassed
// while simRunning): a pan or zoom over a settled layout stops paying Prim
// per frame. Member count guards a same-id zone whose membership changed.
const mstMemo = {};
function zoneMST(ms) {
  if (ms.length < 2) return [];
  const n = ms.length, used = new Array(n).fill(false),
        best = new Array(n).fill(Infinity), from = new Array(n).fill(0), out = [];
  used[0] = true;
  for (let j = 1; j < n; j++) best[j] = (ms[0].x - ms[j].x) ** 2 + (ms[0].y - ms[j].y) ** 2;
  for (let k = 1; k < n; k++) {
    let b = -1;
    for (let j = 0; j < n; j++) if (!used[j] && (b < 0 || best[j] < best[b])) b = j;
    used[b] = true;
    out.push([ms[from[b]], ms[b]]);
    for (let j = 0; j < n; j++) if (!used[j]) {
      const d = (ms[b].x - ms[j].x) ** 2 + (ms[b].y - ms[j].y) ** 2;
      if (d < best[j]) { best[j] = d; from[j] = b; }
    }
  }
  return out;
}

// Members per zone, positions only — recomputed per frame because the layout
// is still moving for most of the time a zone is on screen.
function zoneMembers() {
  const by = {};
  RAW_NODES.forEach(n => {
    if (!(n.sgroup >= 0) || typeof n.x !== "number") return;
    (by[n.sgroup] = by[n.sgroup] || []).push(n);
  });
  return by;
}

// 2D only: onRenderFramePre draws under the nodes.
//
// ONE fill per zone, never a fill plus a stroke: the two widths would have to be
// two passes, and two passes double-composite where they meet — a darker lozenge
// ringing every member, an accidental density map. So the isthmus goes into the
// SAME path as the bulbs, as a quad rather than a stroked segment, and nonzero
// winding unions the lot. Everything inside one fill() composites exactly once,
// however much of it overlaps, which is what lets dense areas fuse flat.
//
// The quad is wound consistently (left normal, so its handedness does not follow
// the edge's direction) and the arcs are drawn anticlockwise to match: mixed
// winding would subtract the overlaps and punch holes where a bridge meets a bulb.
function drawZones(ctx) {
  if (!showZones || !ZONES.length) return;
  const by = zoneMembers();
  const R = ZONE_BULB(), w = ZONE_LINK();
  ZONES.forEach(z => {
    const members = by[z.id];
    if (!members) return;
    ctx.save();
    ctx.globalAlpha = ZONE_ALPHA;
    ctx.fillStyle = zoneColor(z);
    ctx.beginPath();
    members.forEach(n => {
      ctx.moveTo(n.x + R, n.y);
      ctx.arc(n.x, n.y, R, 0, 2 * Math.PI, true);
    });
    let mst = !simRunning && mstMemo[z.id];
    if (!mst || mst.n !== members.length) {
      mst = { n: members.length, seg: zoneMST(members) };
      if (!simRunning) mstMemo[z.id] = mst;
    }
    mst.seg.forEach(([a, b]) => {
      const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len * w, ny = dx / len * w;
      ctx.moveTo(a.x + nx, a.y + ny);
      ctx.lineTo(b.x + nx, b.y + ny);
      ctx.lineTo(b.x - nx, b.y - ny);
      ctx.lineTo(a.x - nx, a.y - ny);
      ctx.closePath();
    });
    ctx.fill();
    ctx.restore();
  });
}

const zoneEls = {};
function buildZoneLabels() {
  const layer = document.getElementById("zone-labels");
  if (!layer || !ZONES.length) return;
  layer.innerHTML = ZONES.map(z =>
    '<div class="zone-label" data-id="' + z.id + '" style="color:' + zoneColor(z) + '">' +
    escHtml(z.label) + '</div>').join("");
  layer.querySelectorAll(".zone-label").forEach(el => { zoneEls[el.dataset.id] = el; });
}

// In 3D a centroid behind the camera still projects to a point on screen — a
// mirrored one. A zone name planted over the wrong cluster is worse than no
// name, so ask which side of the camera it is on first. Plain arithmetic on
// camera.position and the controls' target: the bundle exposes no THREE.
function inFrontOfCamera(c) {
  if (is2D()) return true;
  const cam = Graph.camera && Graph.camera();
  const ctr = Graph.controls && Graph.controls();
  if (!cam || !cam.position || !ctr || !ctr.target) return true;
  const p = cam.position, t = ctr.target;
  return (c.x - p.x) * (t.x - p.x) + (c.y - p.y) * (t.y - p.y) + (c.z - p.z) * (t.z - p.z) > 0;
}

function positionZoneLabels() {
  const by = zoneMembers();
  ZONES.forEach(z => {
    const el = zoneEls[z.id];
    if (!el) return;
    const members = by[z.id];
    if (!showZones || !members) { el.style.display = "none"; return; }
    let x = 0, y = 0, zc = 0;
    members.forEach(n => { x += n.x; y += n.y; zc += n.z || 0; });
    x /= members.length; y /= members.length; zc /= members.length;
    // Snap to the member nearest the centroid. A scattered zone's centroid sits
    // in the hole between its pieces — now that the region no longer fills that
    // hole, a name parked there labels empty space, or worse, someone else's.
    // O(n) per zone per frame, n <= 92 here.
    let c = null, best = Infinity;
    members.forEach(n => {
      const d = (n.x - x) ** 2 + (n.y - y) ** 2 + ((n.z || 0) - zc) ** 2;
      if (d < best) { best = d; c = { x: n.x, y: n.y, z: n.z || 0 }; }
    });
    if (!c || !inFrontOfCamera(c)) { el.style.display = "none"; return; }
    const s = Graph.graph2ScreenCoords(c.x, c.y, c.z);
    el.style.display = "block";
    el.style.transform =
      "translate(" + s.x + "px," + s.y + "px) translate(-50%,-50%)";
  });
}

// --- Note names in 3D -------------------------------------------------------
// 2D paints its labels on the canvas and gates them on the ZOOM: dots below
// 0.6, the standouts up to 1.5, everything above it. 3D has no zoom scalar, so
// the same idea rides the only distance it does have — camera to node. Move in,
// names appear; the reading is identical, the quantity is not a guess.
//
// DOM, like the zone names above, and for the same two reasons: the bundle
// hands out no THREE to build a sprite with, and one positioning routine is
// enough. The cost is bounded by a fixed pool of divs rather than by the vault:
// 553 absolutely-positioned elements written every frame is a real bill, and
// past the first few dozen the names overlap into a smear anyway.
const LABEL_POOL = 60;
const labelEls = [];
function buildNodeLabels() {
  const layer = document.getElementById("node-labels");
  if (!layer || labelEls.length) return;
  for (let i = 0; i < LABEL_POOL; i++) {
    const el = document.createElement("div");
    el.className = "node-label";
    layer.appendChild(el);
    labelEls.push(el);
  }
}

// The thresholds are in graph units and have to scale with the layout, which is
// thousands of units across on a big vault and hundreds on a small one. Mean
// distance from the centroid is that scale, recomputed whenever the layout
// settles rather than per frame.
let GRAPH_R = 0;
function measureGraphRadius() {
  let x = 0, y = 0, z = 0, n = 0;
  RAW_NODES.forEach(p => { if (p.x != null) { x += p.x; y += p.y; z += p.z || 0; n++; } });
  if (!n) return;
  x /= n; y /= n; z /= n;
  let sum = 0;
  RAW_NODES.forEach(p => {
    if (p.x == null) return;
    sum += Math.hypot(p.x - x, p.y - y, (p.z || 0) - z);
  });
  GRAPH_R = sum / n;
}

function positionNodeLabels() {
  if (!labelEls.length) return;
  const cam = !is2D() && Graph.camera && Graph.camera();
  // 2D draws its own labels on the canvas; showing these there would double
  // every name. Same for notes-off, where there is nothing to name.
  if (!cam || !cam.position || !showNotes || !GRAPH_R) {
    labelEls.forEach(el => { el.style.display = "none"; });
    return;
  }
  const far = 3.0 * GRAPH_R;    // beyond this, no name at all
  const near = 1.2 * GRAPH_R;   // inside this, every note gets one
  const p = cam.position;
  const cand = [];
  for (const n of RAW_NODES) {
    if (n._hidden || n._dim || n.x == null) continue;
    const d = Math.hypot(n.x - p.x, n.y - p.y, (n.z || 0) - p.z);
    if (d > far) continue;
    // Between near and far only the standouts, exactly the rule 2D applies
    // between zoom 0.6 and 1.5. Ghosts are names nothing carries; they are
    // already the most numerous thing in the frame and stay unnamed.
    if (d > near && ((n.size || 16) <= BASE_SIZE || n.type === "ghost")) continue;
    if (!inFrontOfCamera(n)) continue;
    cand.push([d, n]);
  }
  // Nearest first, then the pool cuts the tail: when more notes qualify than
  // there are slots, the ones you flew towards are the ones that get named.
  cand.sort((a, b) => a[0] - b[0]);
  const shown = Math.min(cand.length, LABEL_POOL);
  for (let i = 0; i < shown; i++) {
    const d = cand[i][0], n = cand[i][1];
    const el = labelEls[i];
    const s = Graph.graph2ScreenCoords(n.x, n.y, n.z || 0);
    if (el._id !== n.id) { el.textContent = n.label; el._id = n.id; }
    el.style.display = "block";
    // Fade with distance so names arrive instead of popping. Floored, because a
    // label at 5% is a smudge, not a word.
    el.style.opacity = Math.max(0.35, Math.min(1, 1.15 - d / far));
    el.style.color = n.type === "ghost" ? GP.ghostLabel : GP.label;
    el.style.transform =
      "translate(" + s.x + "px," + s.y + "px) translate(-50%,6px)";
  }
  for (let i = shown; i < labelEls.length; i++) labelEls[i].style.display = "none";
}

// Its own rAF, not the render budget's: a camera orbit moves the labels without
// the simulation running, and this loop only writes transforms on a bounded set
// of divs — nothing is rendered. It exists only while a layer that needs it is
// on. One loop drives both layers, so a 3D graph with zones up pays for one.
let zoneRaf = null;
const labelsWanted = () => showZones || !is2D();
function syncZoneLoop() {
  if (labelsWanted() && zoneRaf === null) {
    const step = () => {
      positionZoneLabels();
      positionNodeLabels();
      // All 3D-only and all cheap. styleScene is here rather than wired to
      // each rebuild site because the rebuilds are the bundle's, not ours, and
      // land whenever it decides: a frame is the one moment we know they have.
      // linkSegStep rides the same fact — the link carriers it builds from
      // appear whenever the digest does.
      linkSegStep();
      styleScene();
      fogStep();
      zoneRaf = requestAnimationFrame(step);
    };
    zoneRaf = requestAnimationFrame(step);
  } else if (!labelsWanted() && zoneRaf !== null) {
    cancelAnimationFrame(zoneRaf);
    zoneRaf = null;
    positionZoneLabels();   // one last pass to hide them
    positionNodeLabels();
  }
}

function updateZoneFilter() {
  const cbZones = document.getElementById("cb-zones");
  const cbNodes = document.getElementById("cb-zone-nodes");
  if (cbZones) showZones = cbZones.checked;
  if (cbNodes) showNotes = cbNodes.checked;
  // No refreshPaint: toggling zones no longer touches a single node's colour,
  // and in 3D that call rebuilds the material of every node and every link (see
  // its comment). applyFilters already re-passes the visibility accessors and
  // wakes the renderer, which is all the hull layer needs to appear.
  applyFilters();
  syncZoneLoop();
  updateFocusBar();
}

buildZoneLabels();
buildNodeLabels();

// First build: the mode comes from localStorage (3D on a fresh profile), and
// setMode owns the whole bring-up — instance, forces, filters, focus. A 2D
// first paint has no default camera worth keeping, so it refits; 3D keeps the
// lib's own initial framing.
//
// It runs down here, after both label layers exist, because setMode starts the
// label loop: the loop's own state is declared above in this section, and a
// bring-up from further up the file would read it before it is initialised.
fitPending = is2D();
setMode(mode);

function filterCommunity(cid) {
  activeCommunity = cid;
  document.querySelectorAll(".legend-item").forEach(el => el.classList.remove("active"));
  const el = cid === -2
    ? document.getElementById("legend-all")
    : document.querySelector(`[data-community="${cid}"]`);
  if (el) el.classList.add("active");
  applyFilters();
  updateFocusBar();
  if (cid !== -2) {
    Graph.zoomToFit(400, 50, n => n.group === cid); // isolate: fit camera to the filtered set
    wake(600);
  }
}

// --- Communities legend: sort by size, toggling ascending <-> descending ----
let communitySortAsc = true;
function toggleCommunitySort() {
  const box = document.getElementById("legend-box");
  const allItem = document.getElementById("legend-all");
  const items = Array.from(box.querySelectorAll(".legend-item[data-community]"));
  items.sort((a, b) => (+a.dataset.size - +b.dataset.size) * (communitySortAsc ? 1 : -1));
  items.forEach(el => box.insertBefore(el, allItem));
  document.getElementById("sort-communities").textContent = communitySortAsc ? "size ↑" : "size ↓";
  communitySortAsc = !communitySortAsc;
}

// --- Search → ranked results → fly-to-focus -------------------------------
// Search by what people actually remember: title first, then path, then the
// cluster they were browsing. Choosing a result flies the
// camera to the node and selects it — the graph answers "where is it", not
// just "is it somewhere in this cloud".
let results = [], selIdx = -1;

function scoreNode(n, q) {
  if (n.type === 'ghost') return 0;
  const label = (n.label || '').toLowerCase();
  if (label === q)            return 5;
  if (label.startsWith(q))    return 4;
  if (label.includes(q))      return 3;
  if ((n.path || '').toLowerCase().includes(q)) return 2;
  const cl = COMM_LABELS[n.group];
  if (cl && cl.toLowerCase().includes(q)) return 1;
  return 0;
}

function renderResults(q) {
  const box = document.getElementById("search-results");
  if (!q) { box.className = ""; box.innerHTML = ""; results = []; selIdx = -1; return; }
  results = RAW_NODES
    .map(n => [scoreNode(n, q), n])
    .filter(p => p[0] > 0)
    .sort((a, b) => b[0] - a[0] || a[1].label.localeCompare(b[1].label))
    .slice(0, 12)
    .map(p => p[1]);
  selIdx = results.length ? 0 : -1;

  const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  const sub = n => {
    const cl = COMM_LABELS[n.group];
    return cl ? '<em>' + esc(cl) + '</em>' : esc(n.path || n.type);
  };
  box.innerHTML =
    '<div id="search-count">' + (results.length || 'no') +
      ' result' + (results.length === 1 ? '' : 's') + '</div>' +
    results.map((n, i) =>
      '<div class="result-item' + (i === selIdx ? ' sel' : '') +
        '" onclick="chooseResult(' + i + ')">' +
        '<span class="result-name">' + esc(n.label) + '</span>' +
        '<span class="result-sub">' + sub(n) + '</span>' +
      '</div>').join("");
  box.className = "open";
}

// Shared selection path for tree clicks and search results: open the note view
// and fly the camera. Task 3 adds neighbour dimming here.
function chooseNode(node) {
  if (!node) return;
  selectNode(node);
  focusNode(node);
  applyFocus(node.id);
}

function chooseResult(i) {
  const n = results[i];
  if (!n) return;
  selIdx = i;
  chooseNode(n);
}

function moveSel(d) {
  if (!results.length) return;
  selIdx = (selIdx + d + results.length) % results.length;
  document.querySelectorAll("#search-results .result-item")
    .forEach((el, i) => el.classList.toggle("sel", i === selIdx));
}

function onSearch(q) { renderResults(q.trim().toLowerCase()); }

function onSearchKey(e) {
  if (e.key === "Enter")          { e.preventDefault(); chooseResult(selIdx); }
  else if (e.key === "ArrowDown") { e.preventDefault(); moveSel(1); }
  else if (e.key === "ArrowUp")   { e.preventDefault(); moveSel(-1); }
  else if (e.key === "Escape")    { document.getElementById("search").value = ""; renderResults(""); }
}

// Fly to a node: a camera move in 3D, a pan + zoom in 2D. Coords (node.x/y/z)
// exist once the layout has run (cooldownTicks); before that they default to 0
// and the view simply recentres — harmless.
function focusNode(node) {
  wake(1100);   // 900ms tween, either mode — it needs frames to actually fly
  if (is2D()) {
    Graph.centerAt(node.x || 0, node.y || 0, 900);
    Graph.zoom(2.5, 900);
    return;
  }
  const r = Math.hypot(node.x || 0, node.y || 0, node.z || 0) || 1;
  const k = 1 + 90 * 3 / r;
  Graph.cameraPosition(
    { x: (node.x || 0) * k, y: (node.y || 0) * k, z: (node.z || 0) * k },
    node, 900
  );
}

// `event` is the MouseEvent, and it is a parameter rather than a lookup because
// its `detail` is the browser's own click counter: 1 for the first click of a
// gesture, 2 for the second. That is what separates "what is this" from "read
// it" without a timer -- a 250ms window would have to hold the FIRST click back
// to find out whether a second was coming, which makes every single click on
// the graph feel broken to save a gesture nobody has made yet. Both clicks fire
// here instead: the first paints the panel, the second raises the drawer over
// it. Callers with no gesture behind them (tree, search) pass nothing and get
// the one-click reading, which is the right default for a name.
function selectNode(node, event) {
  // Embedded in the web-UI iframe: hand off to the parent instead of opening
  // this internal metadata drawer (avoids two stacked drawers). A graph click
  // POINTS -- "what is this, and what is around it" -- so it fills the parent's
  // work panel. Ghost nodes ride the same message: they have no path, and that
  // panel is the only surface that can say anything about an unresolved link.
  if (window.parent !== window) {
    // The head facts ride along because this frame HAD to compute all three to
    // draw the node you just clicked: degree from the edge list, state from
    // nodeState (the same rule the HUD's Node state legend counts), area from
    // the community it is coloured by. Asking the server for them again would
    // be a second answer that can disagree with the picture on screen.
    window.parent.postMessage({
      type:  "silica-open-context",
      path:  node.path || "",
      name:  node.label || "",
      ghost: node.type === "ghost",
      links: (outDeg[node.id] || 0) + (inDeg[node.id] || 0),
      state: nodeState(node),
      area:  (Number.isInteger(node.group) && node.group >= 0 && COMM_LABELS[node.group]) || "",
    }, "*");
    // ...and the second click of a double click ALSO means "read it". Sent
    // after the context message, never instead of it, so one click is never
    // spent finding out whether a second is coming. A ghost has no file behind
    // it, so double-clicking one stays a single answer.
    if (event && event.detail >= 2 && node.path)
      window.parent.postMessage({ type: "silica-open-note", path: node.path }, "*");
    return;
  }
  document.getElementById("drawer-title").textContent = node.label;
  document.getElementById("drawer-path").textContent  = node.path || "(ghost node)";
  const commText = (Number.isInteger(node.group) && node.group >= 0 && COMM_LABELS[node.group])
    ? ` · ${COMM_LABELS[node.group]}` : "";
  const betwText = node.betweenness ? ` · betweenness ${node.betweenness}` : "";
  document.getElementById("drawer-meta").textContent = `${node.type}${commText}${betwText}`;
  document.getElementById("drawer-out").textContent = outDeg[node.id] || 0;
  document.getElementById("drawer-in").textContent  = inDeg[node.id]  || 0;

  document.getElementById("drawer").classList.add("open");
}

// (Direct clicks in the view get the same dim-non-neighbours treatment as
// tree/search picks, but skip focusNode's fly — the user is already looking at
// this spot, recentring would just be jarring. Bound in buildGraph, so a mode
// switch rebinds them.)

// The embedding page (chat + note-panel) tells us which note is open
// elsewhere — e.g. a link followed inside the note panel itself — so the
// graph mirrors it. Dim only, no camera move (same reasoning as above).
window.addEventListener("message", e => {
  if (e.data && e.data.type === "silica-focus-path") {
    applyFocus(NODE_BY_ID[e.data.path] ? e.data.path : null);
  }
  // Same, for a SET of notes: the context drawer's concept cloud lights every
  // note carrying the clicked concept at once.
  if (e.data && e.data.type === "silica-focus-paths") {
    applyFocus(e.data.paths || []);
  }
  // The explore toolbar's note search asks us to *locate* a note: fly the
  // camera to it and dim to its neighbourhood, without opening the drawer
  // (selectNode would) — the user is searching the cloud, not inspecting yet.
  if (e.data && e.data.type === "silica-goto-path") {
    const n = NODE_BY_ID[e.data.path];
    if (n) { focusNode(n); applyFocus(n.id); }
  }
  // The note drawer covers this frame's right edge, which is where the HUD is.
  // Its width comes with the message because the drawer is resizable — the
  // focus bar parks against its edge, not against a constant.
  // The app's toolbar owns the renderer segment when embedded.
  if (e.data && e.data.type === "silica-set-renderer") {
    setMode(e.data.mode === "2d" ? "2d" : "3d");
  }
  if (e.data && e.data.type === "silica-host-drawer") {
    document.body.classList.toggle("host-drawer-open", !!e.data.open);
    document.body.style.setProperty("--drawer-w", (e.data.width || 0) + "px");
  }
});

function closeDrawer() {
  document.getElementById("drawer").classList.remove("open");
  // Embedded there is no drawer here to close: this is the deselect, and the
  // app's Node panel is the surface that was showing what got deselected. A
  // panel still describing the node you just dropped is the app disagreeing
  // with the view it sits beside.
  if (window.parent !== window) {
    window.parent.postMessage({ type: "silica-clear-node" }, "*");
  }
}

document.getElementById("file-tree").addEventListener("click", e => {
  const leaf = e.target.closest(".tree-note");
  if (leaf) chooseNode(NODE_BY_ID[leaf.dataset.id]);
});

// --- Esc: back to the whole vault -------------------------------------------
// Undone in the order it was applied — focus first, then the community filter —
// so one press never throws away two decisions at once. The search box owns its
// own Escape (it clears the query), so it is skipped here.
document.addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  if (e.target && e.target.id === "search") return;
  if (focusIds.length) { closeDrawer(); clearFocus(); }
  else if (activeCommunity !== -2) { filterCommunity(-2); fitGraph(); }
  // Last rung, because it is the one the banner promises Esc will undo: the
  // zone layer stays on, only the notes come back.
  else if (!showNotes) {
    const cb = document.getElementById("cb-zone-nodes");
    if (cb) { cb.checked = true; updateZoneFilter(); }
  }
});

// The loop sleeps between interactions, so anything the renderer itself
// services on a frame — the hover raycast, the cursor, drag, wheel zoom,
// control inertia — has to wake it first. Capture phase because force-graph
// reads its hover target at click time and only a rendered frame refreshes it:
// on touch there is no pointermove before the tap.
//
// The same events also cancel the pending auto-fit. The layout settles about a
// second and a half after the view opens, and until now the fit fired then
// regardless: if you had already grabbed the graph and moved somewhere, it
// yanked the camera back. Taking hold of the view means you have chosen your
// framing, so the fit stands down. Hovering is not taking hold, so pointermove
// only wakes the loop.
["pointerdown", "pointermove", "wheel", "touchstart"].forEach(t =>
  document.getElementById("graph-wrap").addEventListener(t, e => {
    wake();
    if (e.type !== "pointermove") fitPending = false;
    // Camera manipulation drops the resolution BEFORE the first heavy paint:
    // capture phase runs ahead of d3-zoom's own handlers, so the very frame
    // this gesture dirties already paints at the motion ratio. A move with a
    // button held is a drag (camera or node), a wheel notch is a zoom; hover
    // and plain clicks never blur. Restore rides the usual no-paints timer.
    if (e.type === "wheel" || (e.type === "pointermove" && e.buttons))
      setRes(DRS_SCALE);
  }, { capture: true, passive: true }));
