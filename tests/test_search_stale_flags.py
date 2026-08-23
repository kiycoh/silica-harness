# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Recall payload stale flags (spec-stale-triggers §3): peek-only, zero bytes
when nothing is stale, and a peek failure never fails the search."""
from types import SimpleNamespace

from silica.driver.base import Hit, NoteRef
from silica.kernel.code import codedocs
from silica.tools import atomic


def _fake_driver(monkeypatch, **methods):
    """Replace the module's DRIVER name, never patch the proxy's attributes.

    `_DriverProxy.__getattr__` forwards to the live driver, so a setattr on the
    proxy is recorded by monkeypatch with that driver's bound method as the
    "old value" and restored as an instance attribute at teardown: every later
    test then reads a vault that no longer exists. Measured 2026-08-23 as three
    undo-journal tests reverting nothing when run after this file.
    """
    fake = atomic.DRIVER if isinstance(atomic.DRIVER, SimpleNamespace) else SimpleNamespace()
    for name, fn in methods.items():
        setattr(fake, name, fn)
    monkeypatch.setattr(atomic, "DRIVER", fake)


def _refs(monkeypatch, names):
    refs = [NoteRef(name=n, path=f"F/{n}.md") for n in names]
    _fake_driver(monkeypatch, search_names=lambda q: refs)


def _hits(monkeypatch, pairs):
    hits = [Hit(ref=NoteRef(name=n, path=p), line=1, snippet="s") for n, p in pairs]
    _fake_driver(monkeypatch, search_context=lambda q: hits)


def test_search_carries_a_stale_map(monkeypatch):
    _refs(monkeypatch, ["Alpha", "Beta"])
    monkeypatch.setattr(codedocs, "peek", lambda v: {"F/Alpha.md": "structural"})
    out = atomic.silica_search("a")
    assert out["stale"] == {"F/Alpha.md": "structural"}


def test_search_fresh_vault_has_no_stale_key(monkeypatch):
    _refs(monkeypatch, ["Alpha"])
    monkeypatch.setattr(codedocs, "peek", lambda v: {})
    assert "stale" not in atomic.silica_search("a")


def test_search_map_lists_only_returned_paths(monkeypatch):
    _refs(monkeypatch, ["Alpha"])
    monkeypatch.setattr(codedocs, "peek",
                        lambda v: {"F/Alpha.md": "cosmetic", "F/Other.md": "structural"})
    assert atomic.silica_search("a")["stale"] == {"F/Alpha.md": "cosmetic"}


def test_search_context_flags_stale_hits(monkeypatch):
    _hits(monkeypatch, [("A", "F/A.md"), ("B", "F/B.md")])
    monkeypatch.setattr(codedocs, "peek", lambda v: {"F/A.md": "cosmetic"})
    by_path = {h["path"]: h for h in atomic.silica_search_context("s")["hits"]}
    assert by_path["F/A.md"]["stale"] == "cosmetic"
    assert "stale" not in by_path["F/B.md"]


def test_peek_failure_never_fails_the_search(monkeypatch):
    _refs(monkeypatch, ["Alpha"])
    _hits(monkeypatch, [("A", "F/A.md")])

    def boom(v):
        raise RuntimeError("boom")

    monkeypatch.setattr(codedocs, "peek", boom)
    assert atomic.silica_search("a")["paths"] == ["F/Alpha.md"]
    assert atomic.silica_search_context("s")["hits"]


# ---------------------------------------------------------------------------
# Source drift rides the same flag: a note derived from a superseded source
# version is stale in the same sense a code note is after a refactor.
# ---------------------------------------------------------------------------

from silica.kernel.write import provenance  # noqa: E402


def test_search_flags_a_note_whose_source_drifted(monkeypatch):
    _refs(monkeypatch, ["Alpha", "Beta"])
    monkeypatch.setattr(codedocs, "peek", lambda v: {})
    monkeypatch.setattr(provenance, "drift_map", lambda **k: {"F/Beta.md": "lec.md"})
    assert atomic.silica_search("a")["stale"] == {"F/Beta.md": "source"}


def test_code_staleness_wins_when_a_note_has_both(monkeypatch):
    """One flag per note: the code level is the more specific claim."""
    _refs(monkeypatch, ["Alpha"])
    monkeypatch.setattr(codedocs, "peek", lambda v: {"F/Alpha.md": "structural"})
    monkeypatch.setattr(provenance, "drift_map", lambda **k: {"F/Alpha.md": "lec.md"})
    assert atomic.silica_search("a")["stale"] == {"F/Alpha.md": "structural"}


def test_search_context_flags_drifted_hits(monkeypatch):
    _hits(monkeypatch, [("A", "F/A.md"), ("B", "F/B.md")])
    monkeypatch.setattr(codedocs, "peek", lambda v: {})
    monkeypatch.setattr(provenance, "drift_map", lambda **k: {"F/B.md": "lec.md"})
    by_path = {h["path"]: h for h in atomic.silica_search_context("s")["hits"]}
    assert by_path["F/B.md"]["stale"] == "source"
    assert "stale" not in by_path["F/A.md"]


def test_a_drift_map_failure_never_fails_the_search(monkeypatch):
    _refs(monkeypatch, ["Alpha"])
    monkeypatch.setattr(codedocs, "peek", lambda v: {"F/Alpha.md": "cosmetic"})

    def boom(**k):
        raise RuntimeError("ledger unreadable")

    monkeypatch.setattr(provenance, "drift_map", boom)
    assert atomic.silica_search("a")["stale"] == {"F/Alpha.md": "cosmetic"}


def test_read_note_carries_the_drift_banner(monkeypatch):
    from silica.driver.base import NoteContent, NoteRef

    nc = NoteContent(ref=NoteRef(name="B", path="F/B.md"), content="---\ntitle: B\n---\n\nbody\n")
    _fake_driver(monkeypatch, read_note=lambda name: nc)
    monkeypatch.setattr(provenance, "drift_map", lambda **k: {"F/B.md": "lec.md"})

    out = atomic.silica_read_note("B")

    assert out.startswith("> [stale] ")
    assert "lec.md" in out.splitlines()[0]
    assert out.endswith("body\n")


def test_read_note_without_drift_is_untouched(monkeypatch):
    from silica.driver.base import NoteContent, NoteRef

    nc = NoteContent(ref=NoteRef(name="B", path="F/B.md"), content="---\ntitle: B\n---\n\nbody\n")
    _fake_driver(monkeypatch, read_note=lambda name: nc)
    monkeypatch.setattr(provenance, "drift_map", lambda **k: {})

    assert atomic.silica_read_note("B") == nc.content
