# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""An `ADR-0003` prose reference resolves to the note filed as `0003-*`.

The generator (kernel/link/ast.extract_links) emits the token; without this
rule every such edge would land in the unresolved set and read to the graph
gate as a dangling link, which is worse than no edge at all.
"""
from __future__ import annotations

from pathlib import Path


def _bind(vault: Path, monkeypatch) -> None:
    import silica.driver
    from silica.config import CONFIG

    vault.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(silica.driver, "_driver", None)


def test_adr_token_resolves_to_the_numbered_note(tmp_path, monkeypatch):
    _bind(tmp_path / "v", monkeypatch)
    from silica.driver import DRIVER

    DRIVER.create("docs/adr/0003-llm-adjudicates-patches.md", "# ADR 3\n")
    DRIVER.create("docs/adr/0029-two-leg-fusion.md", "Keeps ADR-0003 intact.\n")
    assert [r.path for r in DRIVER.links("0029-two-leg-fusion")] == \
        ["docs/adr/0003-llm-adjudicates-patches.md"]
    assert [r.path for r in DRIVER.backlinks("0003-llm-adjudicates-patches")] == \
        ["docs/adr/0029-two-leg-fusion.md"]


def test_unknown_adr_number_stays_unresolved(tmp_path, monkeypatch):
    _bind(tmp_path / "v", monkeypatch)
    from silica.driver import DRIVER

    DRIVER.create("docs/adr/0029-two-leg-fusion.md", "See ADR-0099.\n")
    # Same contract as a dangling [[wikilink]]: a ghost ref in links(), an
    # entry in unresolved(), never a wrong note.
    assert [r.path for r in DRIVER.links("0029-two-leg-fusion")] == ["ADR-0099.md"]
    assert [(l.source.path, l.target) for l in DRIVER.unresolved()] == \
        [("docs/adr/0029-two-leg-fusion.md", "ADR-0099")]
