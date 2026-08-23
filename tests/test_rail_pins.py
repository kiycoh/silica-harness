# tests/test_rail_pins.py
"""A pin is made where the note is, which is the tree.

Pinning used to require opening the note: the toggle lived in the drawer's
actions row, so naming a note as one that matters cost a read of it first. The
tree row now carries the same toggle. Same list, same storage, same key -- the
two surfaces are two faces of one state, and the tests below pin exactly that,
plus the two things a per-row control breaks if nobody watches: the search
filter (which hides labels, and would have left a bare pin on an empty line) and
the frame's copy of the same tree (which has no rail to pin into).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "silica" / "ui" / "web" / "static"
APP_JS = WEB / "app.js"
APP_CSS = WEB / "app.css"

NODES = [
    {"id": "Concepts/Neural.md", "label": "Neural", "type": "note", "path": "Concepts/Neural.md"},
    {"id": "Boats.md", "label": "Boats", "type": "note", "path": "Boats.md"},
    {"id": "Ghost", "label": "Ghost", "type": "ghost", "path": ""},
]


def test_the_tree_carries_a_pin_only_where_there_is_a_rail_to_pin_into():
    """The same render_tree draws the rail's tree and the graph frame's own
    sidebar. A pin in the frame would point at a rail that document does not
    have, so the action is asked for rather than assumed."""
    from silica.ui.web.graph_view import render_tree

    plain = render_tree(NODES)
    assert "tree-pin" not in plain and "tree-row" not in plain

    with_pins = render_tree(NODES, actions=True)
    assert with_pins.count('class="tree-pin"') == 2      # two notes, no ghost
    assert with_pins.count('class="tree-row"') == 2
    # the pin names the note it pins, for the click and for the screen reader
    assert 'data-pin="Concepts/Neural.md"' in with_pins
    # the leaf as the tree prints it, extension included: the label a screen
    # reader reads has to be the label a sighted reader sees
    assert 'aria-label="pin Neural.md"' in with_pins
    assert 'aria-pressed="false"' in with_pins
    # a button beside the row button, never inside it: nested buttons do not
    # survive the parser, and the row would swallow the pin's click
    assert "<button" not in with_pins.split('class="tree-note"')[1].split("</button>")[0]


def test_the_rail_asks_for_the_pins_and_the_frame_does_not():
    server = (ROOT / "silica" / "ui" / "web" / "server.py").read_text()
    assert "render_tree(nodes, actions=True)" in server
    frame = (ROOT / "silica" / "ui" / "web" / "graph_view.py").read_text()
    assert "tree_html = render_tree(nodes)" in frame


def test_the_pin_click_does_not_also_open_the_note():
    """The pin sits inside the row, so the row's own handler would win: pinning
    a note would open it, which is the opposite of what a pin is for."""
    app = APP_JS.read_text()
    handler = app[app.index('$("#tree").addEventListener("click"'):]
    handler = handler[:handler.index("});")]
    assert handler.index('.closest(".tree-pin")') < handler.index('.closest(".tree-note")')
    assert "togglePin(pin.dataset.pin); return;" in handler
    # one togglePin, so the drawer's pin and the tree's pin cannot disagree
    assert app.count("function togglePin(") == 1


def test_a_pinned_row_says_so_in_the_tree():
    """Pinned is a state of the NOTE. A toggle that only ever looks unpressed is
    a button you have to click twice to learn what it did -- and the tree is
    re-rendered from /vault_info, which knows nothing about the pins."""
    app = APP_JS.read_text()
    assert app.count("function syncTreePins(") == 1
    pins = app[app.index("function renderPins()"):]
    assert "syncTreePins();" in pins[:pins.index("\n}")], "renderPins does not re-sync the tree"
    load = app[app.index("async function loadVaultInfo()"):]
    load = load[:load.index("\n}")]
    assert "syncTreePins();" in load, "a fresh tree keeps the old pressed states"


def test_the_filter_hides_the_row_and_not_only_its_label():
    """Hiding the label alone leaves a bare pin floating on an empty line."""
    app = APP_JS.read_text()
    f = app[app.index("function applySidebarFilter()"):]
    f = f[:f.index('$("#pinned")')]
    assert '.closest(".tree-row")' in f and "row.hidden = off;" in f
    # both, because every count below still asks the label whether it is hidden
    assert "el.hidden = off;" in f


def test_the_pin_is_reachable_without_a_pointer():
    """Hover-only affordances are the half a CSS rule always forgets."""
    css = APP_CSS.read_text()
    start = css.index(".tree-pin {")
    block = css[start:css.index("}", start)]
    assert "opacity: 0" in block
    assert ".tree-pin:focus-visible" in css and ".tree-row:hover .tree-pin" in css
    assert ".tree-pin.on" in css   # a made pin stays visible when the pointer leaves
