#!/usr/bin/env python3
"""Emit the three diagrams the README uses: the write path, the read path, and
the schematic of what a change passes through.

    python3 assets/build_diagrams.py

The first two are SMIL, because GitHub serves a README image through camo and runs no
script: <animate> is the only thing that moves there. Every element in a file
shares one dur and one repeatCount, and staging is done with keyTimes, which is
how SMIL keeps a storyboard in sync without a timeline object to hang it on.
Writing those fractions by hand is how the timing rots, so here they are written
in seconds and divided by the loop.

All three are authored at the width the README shows them at, so 11px is 11px
and not 8px after a scale-down, and all three are dark only: a light variant is
a second copy of the geometry to keep in step, and no diagram in this README has
ever had one.

The schematic is checked against the tree by check(), because the last one drifted
for a month: it still drew a CDP backend that had been replaced by the plugin
bridge, an FSM state that had been deleted, and five kernel groups that are now
eight lanes.
"""

import re
from pathlib import Path

HERE = Path(__file__).parent

# The graphite ramp of docs/specs/visual-identity.md, token for token, so a
# README diagram and the running app are the same product. They used to be
# GitHub's own dark surface, which made the card sit on the page rather than on
# top of it, at the cost of Silica looking like whatever host it was embedded in.
BG, PANEL, RAISED, LINE = "#0D0E0F", "#1B1E21", "#24272B", "#2F343A"
DIM, TEXT, BRIGHT = "#969CA6", "#C5CAD2", "#F4F5F7"
# One signal hue at 189deg, and the three status colours that are older than any
# brand. There is no --violet in the system any more, so the one thing that still
# needs a second hue -- the model, marked as data and not as a control -- takes a
# stop off the community arc the graph itself colours nodes with (212-306deg,
# 0.66 saturation): here graph_export._community_color(4).
ACCENT, MODEL = "#22B4CC", "#9C80E5"
GREEN, RED, AMBER = "#4FD08A", "#EB778E", "#E0A93B"
# Archivo and Martian Mono are named first and never load: GitHub serves a README
# image through camo, which runs no script and fetches no font. The stack behind
# them is what actually renders, so the diagrams are drawn to survive it.
SANS = "Archivo, Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'Martian Mono', ui-monospace, SFMono-Regular, Menlo, monospace"
# 90-degree corners on every compartment, per the identity: radius is spent only
# where you read or aim, and a schematic is all chrome. Badges keep the 4px step.
BADGE_R = 4

FADE = 0.3  # seconds an element takes to arrive or to leave


def clock(keys: list[tuple[float, object]], loop: float) -> str:
    """keyTimes and values from (second, value) pairs.

    keyTimes has to start at 0 and end at 1 or the whole animation is in error
    and never runs, which is silent: the frame just renders as if the element
    had no animation at all. So the last state is held out to the end of the
    loop here rather than at every call site."""
    assert keys[0][0] == 0, "an animation has to say what it looks like at 0"
    if keys[-1][0] < loop:
        keys = keys + [(loop, keys[-1][1])]
    times = ";".join(f"{min(t / loop, 1):.4f}" for t, _ in keys)
    values = ";".join(str(v) for _, v in keys)
    return (f'values="{values}" keyTimes="{times}" dur="{loop}s" '
            f'repeatCount="indefinite"')


def linger(t_in: float, loop: float, off_at: float = 0.6) -> str:
    """Visible at t=0, cleared at off_at, back at t_in, held to the end.

    The frame a still renderer shows is t=0, and a storyboard that starts empty
    shows an empty diagram there. Anything that is part of the finished picture
    is drawn at 0 too: its trailing stretch and its leading one meet across the
    loop boundary, so on a renderer that does animate nothing flickers."""
    keys = [(0, 1), (off_at, 1), (off_at + FADE, 0),
            (t_in, 0), (t_in + FADE, 1), (loop, 1)]
    return f'<animate attributeName="opacity" {clock(keys, loop)}/>'


def fade(t_in: float, t_out: float, loop: float) -> str:
    """Arrive at t_in, leave at t_out, invisible either side."""
    keys = [(0, 0), (t_in, 0), (t_in + FADE, 1),
            (t_out, 1), (min(t_out + FADE, loop), 0)]
    return f'<animate attributeName="opacity" {clock(keys, loop)}/>'


def move(path: str, stops: list[tuple[float, float]], loop: float) -> str:
    """Walk `path` through (second, progress) stops. Repeat a progress value to
    stand still, which is what the proposal does while the gate reads it."""
    return (f'<animateMotion path="{path}" calcMode="linear" rotate="0" '
            f'{clock(stops, loop).replace("values=", "keyPoints=", 1)}/>')


def panel(x, y, w, h, r=0, fill=PANEL, stroke=LINE, extra="") -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1"{extra}/>')


def label(x, y, s, size=13, fill=TEXT, weight=400, anchor="start",
          family=SANS) -> str:
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-family="{family}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{s}</text>')


def arrow(x0, x1, y, colour=LINE) -> str:
    return (f'<path d="M{x0} {y} L{x1 - 9} {y}" stroke="{colour}" '
            f'stroke-width="1.5" fill="none"/>'
            f'<path d="M{x1 - 10} {y - 5} L{x1 - 1} {y} L{x1 - 10} {y + 5} Z" '
            f'fill="{colour}"/>')


def chip(w, h, stroke, text, size=12, fill=RAISED, colour=BRIGHT,
         family=MONO) -> str:
    """A card centred on its own origin, so animateMotion can carry it.

    One raised fill for every chip and the stroke carrying the meaning, which is
    how a chip is built in the app. The three hand-tinted fills this replaced
    were each a colour nothing else in the system could name."""
    return (f'<rect x="{-w / 2}" y="{-h / 2}" width="{w}" height="{h}" '
            f'rx="{BADGE_R}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
            + label(0, 4.5, text, size, colour, 500, "middle", family))


def head(w, h, title, aria, desc) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{aria}" '
            f'font-family="{SANS}">\n<title>{title}</title>\n<desc>{desc}</desc>\n'
            f'<rect width="{w}" height="{h}" fill="{BG}"/>\n'
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" '
            f'fill="none" stroke="{LINE}"/>')


# --------------------------------------------------------------------------- #
# 1. the write path: what the gate does to an edit before the vault sees it
# --------------------------------------------------------------------------- #

def gate_svg() -> str:
    L = 13.0   # one accepted write, then one refused, with room to read both
    W, H = 880, 330
    ROW = 170  # everything travels along this line
    CHECKS = (("parse", 164), ("structure", 190), ("links resolve", 216))
    # The proposal stops at the gate's door and vanishes into it rather than
    # parking on top of the checks it is being read by. It comes out the far
    # side, so the three panels need a run of clear canvas either side.
    RUN = f"M212 {ROW} L290 {ROW} L560 {ROW} L640 {ROW}"
    DOOR, EXIT = 78 / 428, 348 / 428
    BACK = f"M290 {ROW} L212 {ROW}"
    ARC = "M754 252 C 720 300, 520 302, 424 250"

    out = [head(W, H, "How a write reaches the vault",
                "An edit proposed by the model is checked by the gate, lands in "
                "the note and is read back. A second edit fails the structural "
                "check and is sent back, leaving the note untouched.",
                "Two passes. The first edit passes parse, structure and links, "
                "lands as a new line, and a pulse travels back from the note to "
                "the gate to read it after it landed. The second edit fails the "
                "structure check, returns to the model, and the note is "
                "unchanged.")]

    out.append(label(28, 36, "Nothing lands unparsed, and what lands is read back",
                     17, BRIGHT, 600))
    out.append(f'<rect x="28" y="48" width="42" height="3" rx="1.5" fill="{ACCENT}"/>')

    # the three stations
    out.append(panel(28, 116, 140, 108))
    out.append(f'<circle cx="48" cy="146" r="3.5" fill="{MODEL}"/>')
    out.append(label(60, 150, "the model", 13, DIM))
    out.append(label(48, 178, "proposes", 13, TEXT))
    out.append(label(48, 202, "write_note()", 11.5, DIM, family=MONO))

    out.append(panel(336, 104, 176, 132))
    out.append(label(358, 132, "the gate, one way in", 12, DIM))
    for name, y in CHECKS:
        out.append(f'<circle cx="364" cy="{y - 4}" r="6" fill="none" stroke="{LINE}"/>')
        out.append(label(382, y, name, 13, DIM))

    out.append(panel(680, 96, 172, 148))
    out.append(label(700, 124, "the note", 13, DIM))
    for i, y in enumerate((146, 162, 178)):
        out.append(f'<rect x="700" y="{y}" width="{132 - i * 22}" height="6" '
                   f'rx="3" fill="{LINE}"/>')

    out.append(arrow(176, 336, ROW))
    out.append(arrow(520, 680, ROW))
    out.append(f'<path d="{ARC}" stroke="{LINE}" stroke-width="1.5" fill="none" '
               f'stroke-dasharray="4 4"/>')
    out.append(label(590, 300, "read back after it lands", 12, DIM, anchor="middle"))

    # pass 1, accepted. One card: it goes in the door, the checks run on it, and
    # it comes out the far side. The opacity gap is the time it spends inside.
    inside = [(0, 0), (0.6, 0), (0.9, 1), (1.5, 1), (1.7, 0),
              (2.9, 0), (3.1, 1), (3.4, 1), (3.6, 0)]
    out.append(f'<g opacity="0"><animate attributeName="opacity" '
               f'{clock(inside, L)}/><g>'
               f'{move(RUN, [(0, 0), (0.6, 0), (1.5, DOOR), (2.9, DOOR), (3.05, EXIT), (3.5, 1)], L)}'
               f'{chip(76, 26, MODEL, "+ 2 lines", 11.5, RAISED, BRIGHT)}</g></g>')
    # the gate lights up as it swallows the card, or the card just disappears
    for t in (1.4, 7.8):
        out.append(f'<rect x="336" y="104" width="176" height="132" '
                   f'fill="none" stroke="{ACCENT}" opacity="0">{fade(t, t + 0.5, L)}</rect>')
    for i, (name, y) in enumerate(CHECKS):
        t = 1.8 + i * 0.4
        out.append(f'<g opacity="0">{fade(t, 6.2, L)}'
                   f'<circle cx="364" cy="{y - 4}" r="6" fill="{GREEN}"/>'
                   f'{label(382, y, name, 13, TEXT)}</g>')

    # the line it adds stays put for the rest of the loop: that is the point
    out.append(f'<g>{linger(3.5, L)}'
               f'<rect x="700" y="194" width="112" height="6" rx="3" fill="{GREEN}"/></g>')
    out.append(f'<circle r="4.5" fill="{ACCENT}" opacity="0">{fade(4.2, 5.5, L)}'
               f'{move(ARC, [(0, 0), (4.3, 0), (5.4, 1)], L)}</circle>')
    out.append(f'<g opacity="0">{fade(5.1, 6.4, L)}'
               f'<rect x="700" y="212" width="86" height="22" rx="{BADGE_R}" fill="{RAISED}" '
               f'stroke="{GREEN}"/>'
               f'{label(743, 227, "verified", 11.5, GREEN, 500, "middle")}</g>')

    # pass 2, refused: parse clears, structure does not, and the run ends there
    out.append(f'<g opacity="0"><animate attributeName="opacity" '
               f'{clock([(0, 0), (7.0, 0), (7.3, 1), (7.9, 1), (8.1, 0)], L)}/><g>'
               f'{move(RUN, [(0, 0), (7.0, 0), (7.9, DOOR), (L, DOOR)], L)}'
               f'{chip(76, 26, MODEL, "+ 5 lines", 11.5, RAISED, BRIGHT)}</g></g>')
    out.append(f'<g>{linger(8.2, L)}'
               f'<circle cx="364" cy="160" r="6" fill="{GREEN}"/>'
               f'{label(382, 164, "parse", 13, TEXT)}</g>')
    out.append(f'<g>{linger(8.6, L)}'
               f'<circle cx="364" cy="186" r="6" fill="{RED}"/>'
               f'{label(382, 190, "structure", 13, TEXT)}</g>')
    out.append(f'<g opacity="0">{fade(9.0, 11.2, L)}<g>'
               f'{move(BACK, [(0, 0), (9.1, 0), (9.9, 1)], L)}'
               f'{chip(96, 26, RED, "sent back", 11.5, RAISED, RED, SANS)}</g></g>')
    out.append(f'<g>{linger(8.6, L)}'
               f'<rect x="700" y="212" width="98" height="22" rx="{BADGE_R}" fill="{RAISED}" '
               f'stroke="{LINE}"/>'
               f'{label(749, 227, "unchanged", 11.5, DIM, 500, "middle")}</g>')

    out.append("</svg>\n")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 2. the read path: three legs, fused by rank
# --------------------------------------------------------------------------- #

def grounding_svg() -> str:
    L = 12.0
    W, H = 880, 330
    # (leg, what it is, colour, y, the order it returns, optional)
    LEGS = (("embeddings", "semantic similarity", MODEL, 92, ("A", "B", "C"), False),
            # BM25 term weighting inside this leg is what made it worth fusing at
            # all: raw tf scored 0.51 recall@10 alone against 0.86 with it
            # (ADR-0029), which is why it is the default and not a knob.
            ("co-occurrence", "concept graph, BM25, no model", ACCENT, 152, ("B", "A", "D"), False),
            ("lexical", "BM25 and fuzzy, no model", ACCENT, 212, (), True))
    # RRF over those two orders, k=60: B .0325, A .0323, D .0161, C .0159
    FUSED = (("B", (MODEL, ACCENT)), ("A", (MODEL, ACCENT)),
             ("D", (ACCENT,)), ("C", (MODEL,)))

    out = [head(W, H, "How an answer is grounded",
                "A question runs down three independent legs. Two return "
                "ranked notes, the opt-in lexical leg abstains, and what comes "
                "back is fused by rank.",
                "The question reaches the embeddings leg, the co-occurrence leg "
                "and the opt-in lexical leg. The lexical leg abstains. The "
                "other two return their own rankings, which fuse by rank, so a "
                "note both legs found outranks a note only one of them found.")]
    out.append(label(28, 36, "One question, three legs, fused by rank", 17, BRIGHT, 600))
    out.append(f'<rect x="28" y="48" width="42" height="3" rx="1.5" fill="{ACCENT}"/>')

    out.append(panel(28, 152, 132, 44))
    out.append(label(94, 179, "your question", 12.5, TEXT, 500, "middle"))

    for i, (name, sub, colour, y, ranks, opt) in enumerate(LEGS):
        dash = ' stroke-dasharray="5 4"' if opt else ""
        wire = f"M160 174 C 188 174, 188 {y + 26}, 206 {y + 26}"
        out.append(f'<path d="{wire}" stroke="{LINE}" stroke-width="1.5" '
                   f'fill="none"{dash}/>')
        out.append(panel(214, y, 214, 52, 9, PANEL, colour, dash))
        out.append(label(232, y + 22, name, 13, BRIGHT, 600))
        if opt:
            out.append(f'<g opacity="0">{fade(0.7, 2.4, L)}'
                       f'{label(232, y + 39, sub, 11.5, DIM)}</g>')
        else:
            out.append(label(232, y + 39, sub, 11.5, DIM))
        out.append(f'<circle r="3.5" fill="{colour}" opacity="0">'
                   f'{fade(0.6 + i * 0.2, 2.0 + i * 0.2, L)}'
                   f'{move(wire, [(0, 0), (0.7 + i * 0.2, 0), (1.6 + i * 0.2, 1), (L, 1)], L)}'
                   f'</circle>')

    # the leg with nothing to say says so, and the pool carries on without it
    out.append(f'<g>{linger(2.4, L)}'
               f'{label(232, 251, "abstains, nothing to say", 11.5, AMBER, 500)}</g>')

    out.append(panel(508, 140, 152, 68, 9, PANEL, ACCENT))
    out.append(label(584, 168, "RRF fusion", 13, BRIGHT, 600, "middle"))
    out.append(label(584, 187, "by rank, not by score", 11, DIM, anchor="middle"))
    for y, opt in ((118, False), (178, False), (238, True)):
        dash = ' stroke-dasharray="5 4"' if opt else ""
        out.append(f'<path d="M428 {y} C 468 {y}, 468 174, 500 174" stroke="{LINE}" '
                   f'stroke-width="1.5" fill="none"{dash}/>')

    # each leg hands over its own order, and they disagree
    for i, (name, sub, colour, y, ranks, opt) in enumerate(LEGS):
        for j, key in enumerate(ranks):
            t0 = 2.9 + i * 0.25 + j * 0.45
            wire = f"M428 {y + 26} C 468 {y + 26}, 468 174, 498 174"
            out.append(f'<g opacity="0">{fade(t0, t0 + 0.9, L)}<g>'
                       f'{move(wire, [(0, 0), (t0 + 0.1, 0), (t0 + 1.0, 1), (L, 1)], L)}'
                       f'<circle r="11" fill="{RAISED}" stroke="{colour}" stroke-width="1.2"/>'
                       f'{label(0, 4, key, 11.5, BRIGHT, 600, "middle", MONO)}</g></g>')

    out.append(f'<g>{linger(5.4, L)}'
               f'{label(700, 108, "ranked notes", 12, DIM)}</g>')
    for i, (key, dots) in enumerate(FUSED):
        y, t0 = 126 + i * 38, 5.6 + i * 0.35
        row = [f'<g>{linger(t0, L)}',
               f'<rect x="700" y="{y}" width="152" height="30" fill="{PANEL}" '
               f'stroke="{LINE}"/>',
               label(716, y + 20, str(i + 1), 11, DIM, family=MONO),
               label(738, y + 20, key, 12.5, BRIGHT, 600, family=MONO)]
        for k, colour in enumerate(dots):
            cx = 836 - (len(dots) - 1 - k) * 13
            row.append(f'<circle cx="{cx}" cy="{y + 15}" r="4" fill="{colour}"/>')
        out.append("".join(row) + "</g>")
    out.append(f'<g>{linger(7.4, L)}'
               # Both lines have to clear x=880 in Archivo, which is wider than
               # the Inter that renders on GitHub: the old wording fit only in
               # the fallback and would have clipped wherever Archivo is present.
               f'{label(700, 292, "Both legs found B and A,", 11.5, DIM)}'
               f'{label(700, 307, "so they outrank D and C", 11.5, DIM)}</g>')

    out.append("</svg>\n")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 3. the schematic: everything a change passes through, in the order it does
# --------------------------------------------------------------------------- #

# (command, what it is). Four ways in, which is what the README calls them.
WAYS_IN = (("silica", "terminal, Rich TUI"),
           ("silica --gui", "web GUI, FastAPI and SSE"),
           ("silica connect", "the Obsidian plugin"),
           ("silica mcp", "memory for another agent"))
# (module under silica/sources, what to call it in public)
SOURCES = (("prose", "prose"), ("code", "code"), ("convert", "documents"),
           ("notebook", "notebooks"), ("transcript", "transcripts"),
           ("web_fetch", "web pages"), ("web_research", "web research"))
# the states of the Injector FSM, in the order they run, grouped by what they do
FSM_PHASES = (
    ("reads the vault and the payload", ("RECON", "PAYLOAD", "SALIENCE", "COLLISION")),
    ("decides what to write", ("DELEGATE", "SANITIZE", "VALIDATE")),
    ("writes, then repairs the graph",
     ("SNAPSHOT", "WRITE", "HUB_UPDATE", "AUTOLINK", "BACKLINK", "LINT", "CLEANUP")),
)
# (directory under silica/kernel, what lives there)
LANES = (("text", "recon, keyphrase,|sanitize, title"),
         ("link", "autolink, rename,|correlate, lint"),
         ("recall", "embeddings,|co-occurrence,|fusion, rerank"),
         ("organize", "classify,|taxonomy"),
         ("write", "ops, provenance,|undo journal"),
         ("code", "AST graph,|code wiki"),
         ("calendar", "events,|occurrences,|reminders"),
         ("report", "dedup, quiz,|vault energy"))
# (module under silica/driver, what it drives)
BACKENDS = (("fs_backend", "fs backend", "the default: plain files, git friendly"),
            ("ws_backend", "ws backend", "the Obsidian plugin, over its rpc bridge"))
# leashed sub-agents, fed by the FSM's work queue. One line, not six more boxes
CAPABILITIES = ("dedup", "refine", "enrich", "expand", "orphan", "codewiki")


def band(x, y, text) -> str:
    return (f'<text x="{x}" y="{y}" font-size="10.5" fill="{ACCENT}" font-weight="600" '
            f'letter-spacing="0.09em" font-family="{SANS}">{text.upper()}</text>')


def down(x, y0, y1, note="") -> str:
    out = (f'<path d="M{x} {y0} L{x} {y1 - 8}" stroke="{LINE}" stroke-width="1.5"/>'
           f'<path d="M{x - 5} {y1 - 9} L{x} {y1 - 1} L{x + 5} {y1 - 9} Z" fill="{LINE}"/>')
    if note:
        out += label(x + 12, (y0 + y1) / 2 + 4, note, 10.5, DIM)
    return out


def architecture_svg() -> str:
    W, H = 880, 728
    X, INNER = 24, 832
    MID = X + INNER / 2
    out = [head(W, H, "What a change passes through",
                "Four ways in, one gate, eight kernel lanes, two backends, and "
                "a vault of plain markdown beside a private index.",
                "A change enters through the terminal, the web GUI, the Obsidian "
                "plugin or the MCP server. External content stages in Inbox. "
                "Everything then passes the Injector FSM, whose states call the "
                "eight kernel lanes, and the ops it emits reach the vault "
                "through the filesystem backend or the Obsidian plugin.")]
    out.append(label(X, 38, "Every mutation takes the same path", 17, BRIGHT, 600))
    out.append(f'<rect x="{X}" y="50" width="42" height="3" rx="1.5" fill="{ACCENT}"/>')

    out.append(band(X, 84, "four ways in"))
    w = (INNER - 3 * 12) / 4
    for i, (cmd, what) in enumerate(WAYS_IN):
        bx = X + i * (w + 12)
        out.append(panel(bx, 94, w, 50))
        out.append(label(bx + 16, 118, cmd, 12.5, BRIGHT, 500, family=MONO))
        out.append(label(bx + 16, 134, what, 10.5, DIM))
    out.append(down(MID, 144, 176))

    out.append(band(X, 172, "what can come in"))
    out.append(panel(X, 182, INNER, 56))
    out.append(label(X + 16, 206, " · ".join(s for _, s in SOURCES), 12.5, TEXT))
    out.append(label(X + 16, 226, "external content normalizes into Inbox/ and reaches "
                                  "the vault only through the gate below", 10.5, DIM))
    out.append(down(MID, 238, 276))

    out.append(band(X, 272, "the gate"))
    out.append(panel(X, 282, INNER, 130, 10, PANEL, AMBER))
    out.append(label(X + 16, 306, "Injector FSM", 13, BRIGHT, 600))
    out.append(label(X + 118, 306, "deterministic, per file, the only way in", 11, DIM))
    for i, (what, states) in enumerate(FSM_PHASES):
        y = 332 + i * 22
        out.append(label(X + 16, y, what, 11, DIM))
        out.append(label(X + 216, y, " · ".join(states), 11, TEXT, family=MONO))
    out.append(f'<rect x="{X + 16}" y="{384}" width="{INNER - 32}" height="1" '
               f'fill="{LINE}"/>')
    out.append(label(X + 16, 403, "any gate fails, ROLLBACK applies the inverse ops "
                                  "and restores the pre-write snapshot", 11, AMBER))
    out.append(down(MID, 412, 450, "calls"))

    # The lane count is read off LANES, which check() pins to the tree: the
    # count was written into the band label and the width divisor at three
    # sites, so the calendar lane landed as an assertion failure rather than
    # as a re-laid-out row.
    out.append(band(X, 446, f"kernel, {len(LANES)} lanes"))
    w = (INNER - (len(LANES) - 1) * 8) / len(LANES)
    for i, (name, gloss) in enumerate(LANES):
        bx = X + i * (w + 8)
        out.append(panel(bx, 456, w, 70))
        out.append(label(bx + 12, 478, name, 12, BRIGHT, 600, family=MONO))
        for j, line in enumerate(gloss.split("|")):
            out.append(label(bx + 12, 496 + j * 13, line, 9.5, DIM))
    out.append(label(X, 545, "the same lanes back six leashed capabilities the FSM "
                     "work queue feeds: " + ", ".join(CAPABILITIES), 10.5, DIM))
    out.append(down(MID, 556, 590, "ops"))

    out.append(band(X, 586, "how they land"))
    w = (INNER - 12) / 2
    for i, (_, name, gloss) in enumerate(BACKENDS):
        bx = X + i * (w + 12)
        out.append(panel(bx, 596, w, 48))
        out.append(label(bx + 16, 618, name, 12.5, BRIGHT, 600, family=MONO))
        out.append(label(bx + 16, 634, gloss, 10.5, DIM))
    out.append(down(MID, 644, 672))

    for i, (name, gloss, mono) in enumerate(
            (("your vault", "plain .md, readable with or without Silica", False),
             ("~/.silica", "index, embeddings, ledgers, undo journal", True))):
        bx = X + i * (w + 12)
        out.append(panel(bx, 672, w, 48, 10, PANEL, GREEN))
        out.append(label(bx + 16, 694, name, 12.5, BRIGHT, 600,
                         family=MONO if mono else SANS))
        out.append(label(bx + 16, 710, gloss, 10.5, DIM))

    out.append("</svg>\n")
    return "\n".join(out)


def check() -> None:
    """The schematic drifted for a month because nothing compared it to the
    tree. These are the four claims that went stale, so these are the four the
    build refuses to ship wrong."""
    root = HERE.parent / "silica"
    src = (root / "router" / "orchestrator.py").read_text()
    enum = src.split("class InjectorState(Enum):", 1)[1].split("class ", 1)[0]
    states = set(re.findall(r"^\s+([A-Z_]+) = auto\(\)", enum, re.M))
    drawn = {s for _, group in FSM_PHASES for s in group}
    assert drawn == states - {"INIT", "ROLLBACK", "DONE", "ERROR"}, (
        f"FSM drifted: {drawn ^ (states - {'INIT', 'ROLLBACK', 'DONE', 'ERROR'})}")

    lanes = {p.name for p in (root / "kernel").iterdir()
             if p.is_dir() and not p.name.startswith("__")}
    assert {n for n, _ in LANES} == lanes, f"kernel lanes drifted: {lanes}"

    for module, _ in SOURCES:
        assert (root / "sources" / f"{module}.py").exists(), f"no source {module}"
    for module, _, _ in BACKENDS:
        assert (root / "driver" / f"{module}.py").exists(), f"no backend {module}"
    for cap in CAPABILITIES:
        assert (root / "capabilities" / f"{cap}.py").exists(), f"no capability {cap}"


def main() -> None:
    check()
    for name, svg in (("gate.svg", gate_svg()), ("grounding.svg", grounding_svg()),
                      ("architecture.svg", architecture_svg())):
        path = HERE / name
        before = path.read_text() if path.exists() else ""
        path.write_text(svg)
        verb = "unchanged" if svg == before else "updated"
        print(f"{name}: {verb} ({len(svg) // 1024} KB, "
              f"{svg.count('<animate')} animations)")


if __name__ == "__main__":
    main()
