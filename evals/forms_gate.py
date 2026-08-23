# SPDX-License-Identifier: AGPL-3.0-or-later
"""Nucleation-forms gate (docs/specs/nucleation-forms.md, "Eval harness").

Frozen fixture: evals/forms_vault (the creator-audit vault, 19 notes + the
three failing inputs + one study input). All gates are mechanical; there is
no LLM judge. Needs the production model + embedding endpoint, so this is an
eval, not a pytest.

Run:  uv run python -m evals.forms_gate [--keep]
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

FIXTURE = Path(__file__).parent / "forms_vault"

# Gate 1 checklist: the sponsor call's eight operational facts, as
# string-matchable anchors (the lenses mandate verbatim numbers/names/dates).
SPONSOR_FACTS = {
    "package": re.compile(r"2 integrations? \+ 1 dedicated|two integrations.{0,30}one dedicated", re.I),
    "budget": re.compile(r"mid five figures", re.I),
    "six-month ask": re.compile(r"6[ -]months?|six[ -]months?", re.I),
    "90-day counter": re.compile(r"90 ?days?", re.I),
    "deliverables": re.compile(r"10-?12 min|60-?90 ?s", re.I),
    "thumbnail": re.compile(r"thumbnail", re.I),
    "timeline": re.compile(r"eptember", re.I),
    "decision-maker": re.compile(r"Marta", re.I),
}


def _fail(name: str, detail: str) -> dict:
    return {"gate": name, "ok": False, "detail": detail}


def _ok(name: str, detail: str = "") -> dict:
    return {"gate": name, "ok": True, "detail": detail}


def _bodies(vault: Path, exclude_inbox: bool = True) -> dict[str, str]:
    out = {}
    for p in vault.rglob("*.md"):
        rel = p.relative_to(vault).as_posix()
        if exclude_inbox and rel.casefold().startswith(("inbox/", "done/")):
            continue
        out[rel] = p.read_text(encoding="utf-8", errors="replace")
    return out


def main() -> int:
    keep = "--keep" in sys.argv
    work = Path(tempfile.mkdtemp(prefix="forms-gate-"))
    vault = work / "vault"
    shutil.copytree(FIXTURE, vault)
    (vault / "vault.yaml").write_text('write_dir: ""\n', encoding="utf-8")
    os.environ["SILICA_VAULT"] = str(vault)

    # Imports AFTER the env pin: CONFIG reads the environment at import time.
    from silica.config import CONFIG

    CONFIG.vault_path = str(vault)
    import silica.driver as driver

    driver._driver = None

    baseline = _bodies(vault)

    results: list[dict] = []

    # ---- Gate 0: form verdict, pinned per input --------------------------
    import silica.kernel.forms as forms

    pins = {
        "Inbox/raw-sponsor-call-2026-06-12.md": "transcript",
        "Inbox/clipped-article-inference-pricing.md": "clip",
        # ep180 is a .txt: it is converted first, so its verdict is checked
        # post-run via the filing outcome (gate 4) rather than here.
    }
    for rel, expected in pins.items():
        got = forms.resolve((vault / rel).read_text(encoding="utf-8"))
        if got.form != expected:
            results.append(_fail("0 form-verdict", f"{rel}: {got.form!r} != {expected!r} ({got.origin})"))
        else:
            results.append(_ok("0 form-verdict", f"{rel} -> {expected} ({got.origin})"))
    ep180_txt = (vault / "Inbox/transcript-dump-ep180-rough.txt").read_text(encoding="utf-8")
    got = forms.resolve(ep180_txt)
    results.append(
        _ok("0 form-verdict", f"ep180 -> draft ({got.origin})")
        if got.form == "draft"
        else _fail("0 form-verdict", f"ep180: {got.form!r} != 'draft' ({got.origin})")
    )

    # ---- Run the product path -------------------------------------------
    from silica.cli import _expand_workflow_shortcut

    for cmd in ("/embed", "/cooccur"):
        _expand_workflow_shortcut(cmd)
    out = _expand_workflow_shortcut("/nucleate Inbox/*")
    if out:
        # the deterministic path returns ""; anything else fell to the agent
        results.append(_fail("run", f"dispatch fell through to the agent: {out[:120]}"))

    after = _bodies(vault)
    new_or_changed = {
        rel: body for rel, body in after.items()
        if baseline.get(rel) != body
    }
    produced = "\n\n".join(new_or_changed.values())

    # ---- Gate 1: sponsor-call fact coverage (>= 7/8) ---------------------
    missing = [name for name, rx in SPONSOR_FACTS.items() if not rx.search(produced)]
    if len(missing) <= 1:
        results.append(_ok("1 fact-coverage", f"{8 - len(missing)}/8 (missing: {missing or 'none'})"))
    else:
        results.append(_fail("1 fact-coverage", f"{8 - len(missing)}/8, missing {missing}"))

    # ---- Gate 2: register -----------------------------------------------
    bad_register = [
        rel for rel, body in new_or_changed.items()
        if re.search(r"\bthe creator\b|\bthe user\b", body, re.I)
    ]
    results.append(
        _ok("2 register") if not bad_register
        else _fail("2 register", f"third-person owner in {bad_register}")
    )
    definitional = [
        rel for rel, body in new_or_changed.items()
        if re.match(r"\s*[A-Z][\w \-]{0,40} (is|are) an? ", body.split("---")[-1].strip())
    ]
    results.append(
        _ok("2 no-definitional-opener") if not definitional
        else _fail("2 no-definitional-opener", f"{definitional}")
    )

    # ---- Gate 3: clip commentary survives as prose -----------------------
    commentary = re.compile(r"contradicts?.{0,80}(ep ?142|episode 142)", re.I | re.S)
    results.append(
        _ok("3 clip-commentary") if commentary.search(produced)
        else _fail("3 clip-commentary", "the 'contradicts ep 142' claim is not stated in any body")
    )

    # ---- Gate 4: draft filing -------------------------------------------
    filed = [
        rel for rel, body in new_or_changed.items()
        if re.search(r"^form: .?draft", body, re.M)
    ]
    if len(filed) == 1:
        body = new_or_changed[filed[0]]
        core = "the number that convinced me was five months"
        results.append(
            _ok("4 draft-filing", filed[0]) if core in body
            else _fail("4 draft-filing", f"{filed[0]} lost the body")
        )
    else:
        results.append(_fail("4 draft-filing", f"expected exactly 1 filed draft, got {filed}"))

    # ---- Gate 5: study regression (residue ~empty on study input) --------
    log = (vault / "log.md").read_text(encoding="utf-8", errors="replace") if (vault / "log.md").exists() else ""
    results.append(
        _fail("5 study-residue", "residue declared on the study input")
        if re.search(r"residue `study", log)
        else _ok("5 study-residue")
    )

    # ---- Report ----------------------------------------------------------
    failed = [r for r in results if not r["ok"]]
    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"[{mark}] {r['gate']}  {r['detail']}")
    print(f"\n{len(results) - len(failed)}/{len(results)} gates passed")
    if keep or failed:
        print(f"workdir kept: {vault}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
