# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""silica_recall must say which notes are worth re-reading.

Measured on a real chat: 3 of 9 tool calls were silica_read_note on notes whose
FULL body recall had already delivered (render(windowed=True) emits the excerpt,
and for a short note the excerpt is the whole body). `partial` names the notes
that really are a slice, so the rest need no second call.
"""
from silica.kernel.recall.perception import NoteBlock, Perception
from silica.tools import graph


def _perception(*blocks):
    return Perception(query="q", blocks=list(blocks))


def _patch(monkeypatch, perception):
    import silica.kernel.recall.perception as perception_mod

    monkeypatch.setattr(perception_mod, "perceive",
                        lambda query, now, k, **kw: perception, raising=False)


def test_windowed_note_is_partial_whole_note_is_not(monkeypatch):
    _patch(monkeypatch, _perception(
        NoteBlock(path="A", date="", evidence="", body="line1\nline2\nline3", excerpt="line2"),
        NoteBlock(path="B", date="", evidence="", body="short body", excerpt="short body"),
    ))
    out = graph.silica_recall("q")
    assert out["notes"] == ["A", "B"]
    assert out["partial"] == ["A"]


def test_surrounding_whitespace_is_not_a_difference(monkeypatch):
    _patch(monkeypatch, _perception(
        NoteBlock(path="A", date="", evidence="", body="  body\n", excerpt="body"),
    ))
    assert graph.silica_recall("q")["partial"] == []


def test_no_blocks_is_an_empty_answer(monkeypatch):
    _patch(monkeypatch, _perception())
    out = graph.silica_recall("q")
    assert out["notes"] == [] and out["partial"] == []


def test_render_header_carries_the_stale_token():
    """The model answers from the context string, so a side map alone never
    reaches it — the flag has to be in the header."""
    p = _perception(NoteBlock(path="wiki/m", date="", evidence="embed:0.9",
                              body="b", excerpt="b"))
    ctx = p.render(stale={"wiki/m.md": "structural"})
    assert ctx.splitlines()[0] == "[#1 | wiki/m | embed:0.9 | stale:structural]"


def test_unwindowed_render_carries_the_stale_token_too():
    """The legacy A/B arm must not silently swallow the flag — a kwarg that
    works in one layout and no-ops in the other is a trap for the next caller."""
    p = _perception(NoteBlock(path="wiki/m", date="", evidence="", body="b",
                              excerpt="w"))
    assert p.render(windowed=False,
                    stale={"wiki/m.md": "cosmetic"}) == "[stale:cosmetic]\nb"


def test_render_without_stale_is_byte_identical():
    """No stale map, no change: the validated perception string is frozen."""
    p = _perception(NoteBlock(path="wiki/m", date="", evidence="", body="b",
                              excerpt="b"))
    assert p.render() == p.render(stale=None) == p.render(stale={})


def test_recall_payload_carries_map_and_token(monkeypatch):
    from silica.kernel.code import codedocs

    _patch(monkeypatch, _perception(
        NoteBlock(path="wiki/m", date="", evidence="", body="b", excerpt="b"),
        NoteBlock(path="fresh/n", date="", evidence="", body="b", excerpt="b"),
    ))
    monkeypatch.setattr(codedocs, "peek", lambda v: {"wiki/m.md": "structural"})
    out = graph.silica_recall("q")
    assert out["stale"] == {"wiki/m": "structural"}
    assert "stale:structural" in out["context"]   # the model answers from context


def test_recall_map_absent_when_nothing_stale(monkeypatch):
    """Zero bytes on the common case: no key, no token."""
    from silica.kernel.code import codedocs

    _patch(monkeypatch, _perception(
        NoteBlock(path="A", date="", evidence="", body="b", excerpt="b")))
    monkeypatch.setattr(codedocs, "peek", lambda v: {})
    out = graph.silica_recall("q")
    assert "stale" not in out
    assert "stale:" not in out["context"]


def test_peek_failure_never_fails_recall(monkeypatch):
    """Flags are an aid; answering is the operation and must survive."""
    from silica.kernel.code import codedocs

    _patch(monkeypatch, _perception(
        NoteBlock(path="A", date="", evidence="", body="b", excerpt="b")))

    def boom(v):
        raise RuntimeError("boom")

    monkeypatch.setattr(codedocs, "peek", boom)
    out = graph.silica_recall("q")
    assert out["notes"] == ["A"] and "stale" not in out
