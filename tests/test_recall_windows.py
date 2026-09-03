# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Recall windows: cut on line starts, and fewer of them past the top ranks.

Measured 2026-09-03 on this repo: a k=15 recall was 42k chars (~10.6k tokens)
and its excerpts opened mid-word ("matic gate does not", "mportant: int"),
because `best_window_spans` picks offsets on a 250-char stride with no regard
for line boundaries. Cutting k was refuted earlier (ranks 9-15 carried 9 gold
answers), so the saving comes from the window count per rank instead.
"""
from __future__ import annotations


from silica.kernel.recall.rerank import best_window_spans

LINES = [f"line {i:03d} filler words here" for i in range(60)]
LINES[30] = "line 030 the yoga class is on tuesday evening"
TEXT = "\n".join(LINES)


def test_raw_pick_lands_mid_line():
    # The premise: without snapping the offset is a stride multiple, not a
    # line start. If this stops holding, the snap test below proves nothing.
    (p, _), = best_window_spans(TEXT, "yoga class tuesday", 200)
    assert p > 0 and TEXT[p - 1] != "\n"


def test_snap_moves_the_offset_to_the_previous_line_start():
    (p, s), = best_window_spans(TEXT, "yoga class tuesday", 200, snap=True)
    assert p == 0 or TEXT[p - 1] == "\n"
    assert s.startswith("line ")
    assert "yoga class is on tuesday" in s


def _perceive_with(monkeypatch, bodies: list[str], query: str, k: int, **kw):
    from silica.kernel.recall import perception
    from silica.kernel.recall.relatedness import RelatedNote

    results = [RelatedNote(path=f"n{i}", name=f"n{i}", score=1.0 - i / 100, evidence=["cooccur:w1"])
               for i in range(len(bodies))]
    monkeypatch.setattr(perception, "facade_retrieve", lambda *a, **k_: (results, None))
    monkeypatch.setattr(perception, "_read_dated_body",
                        lambda path, origin="vault": ("", None, bodies[int(path[1:])], []))
    monkeypatch.setattr(perception, "_recall_facts", lambda *a, **k_: ([], [], ""))
    return perception.perceive(query, now="2026-09-03", k=k, with_facts=False, **kw)


THREE_HITS = ("pad line\n" * 40 + "the yoga class\n" + "pad line\n" * 40
              + "yoga again\n" + "pad line\n" * 40 + "yoga once more\n" + "pad line\n" * 40)


def test_ranks_past_five_get_one_window(monkeypatch):
    p = _perceive_with(monkeypatch, [THREE_HITS] * 7, "yoga", k=7, window_chars=120)
    assert p.blocks[0].excerpt.count("[…]") == 2   # three windows at the top
    assert p.blocks[4].excerpt.count("[…]") == 2
    assert p.blocks[5].excerpt.count("[…]") == 0   # one window from rank 6 on
    assert "yoga" in p.blocks[5].excerpt


def test_perceive_excerpts_start_on_line_boundaries(monkeypatch):
    p = _perceive_with(monkeypatch, [THREE_HITS], "yoga", k=1, window_chars=120)
    for seg in p.blocks[0].excerpt.split("\n[…]\n"):
        at = THREE_HITS.find(seg)
        assert at >= 0
        assert at == 0 or THREE_HITS[at - 1] == "\n", repr(seg[:30])
