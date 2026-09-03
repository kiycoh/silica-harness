"""Outline-first nucleation lane (docs/specs/nucleation-paradigm-ab.md).

The model reads the whole source and names the ideas, their section and their
typed dependencies; the mechanical stages only guard. These tests pin the pure
seams: parsing, coverage, op assembly, edge selection, and the call plan.
"""
from __future__ import annotations

import os
import re

import pytest

from silica.kernel import outline as ol


_SRC = """## Machine Learning (9 CFU)
[[Giosuè Lo Bosco]] giosue.lobosco@unipa.it
## Support vector machines
## Margine
Il margine geometrico e' la distanza euclidea dall'iperpiano.
$$\\gamma_r = \\min_{x_i \\in T} d_e(x_i, r)$$
## dove:
M = insieme dei punti misclassificati
## Errore del percettrone online
(T. di Novikoff) il numero di errori e' limitato da $(2R/\\gamma)^2$.
## Errore del percettrone online
dimostrazione
## ***Formulazione lagrangiana:***
Si incorporano i vincoli nella funzione obiettivo.
"""

_OUTLINE_JSON = {
    "lesson_title": "SVM: iperpiano di margine massimale",
    "spine": ["Margine geometrico", "Errore del percettrone online", "Bogus"],
    "ideas": [
        {"title": "Margine geometrico", "section": "Margine",
         "claim": "Il margine e' la distanza minima dall'iperpiano.",
         "depends_on": []},
        {"title": "Errore del percettrone online", "section": "Errore del percettrone online",
         "claim": "Novikoff limita gli errori con (2R/gamma)^2.",
         "depends_on": [{"title": "Margine geometrico", "relation": "bounds", "why": "il bound e' in gamma"},
                        {"title": "Nope", "relation": "applies", "why": "unknown target"},
                        {"title": "Margine geometrico", "relation": "invents", "why": "bad relation"}]},
        {"title": "Margine geometrico", "section": "Margine", "claim": "duplicate title", "depends_on": []},
    ],
}


def test_parse_outline_keeps_order_drops_unknown_targets_and_duplicates():
    out = ol.parse_outline(_OUTLINE_JSON)
    assert [i.title for i in out.ideas] == ["Margine geometrico", "Errore del percettrone online"]
    assert out.spine == ["Margine geometrico", "Errore del percettrone online"]
    deps = out.ideas[1].depends_on
    assert deps == [{"title": "Margine geometrico", "relation": "bounds", "why": "il bound e' in gamma"}]


def test_source_headings_drop_noise_and_repeats():
    heads = ol.source_headings(_SRC)
    assert heads == ["Support vector machines", "Margine", "Errore del percettrone online",
                     "Formulazione lagrangiana:"]


def test_coverage_gaps_names_the_sections_no_idea_claims():
    out = ol.parse_outline(_OUTLINE_JSON)
    gaps = ol.coverage_gaps(out, ol.source_headings(_SRC))
    assert gaps == ["Support vector machines", "Formulazione lagrangiana:"]


def test_select_edges_filters_and_caps_per_target():
    ideas = {"A", "B", "C", "D", "E"}
    existing = {"X": "T/X.md", "Y": "T/Y.md", "Lezione 10": "T/Lezione 10.md"}
    raw = [
        {"from": "A", "to": "X", "relation": "applies", "why": "una frase abbastanza lunga da contare"},
        {"from": "B", "to": "X", "relation": "relaxes", "why": "una frase abbastanza lunga da contare"},
        {"from": "C", "to": "X", "relation": "bounds", "why": "una frase abbastanza lunga da contare"},
        {"from": "D", "to": "X", "relation": "justifies", "why": "una frase abbastanza lunga da contare"},  # 4th on X: dropped
        {"from": "E", "to": "Y", "relation": "same_as", "why": "una frase abbastanza lunga da contare"},
        {"from": "A", "to": "Y", "relation": "applies", "why": "short"},                 # why too short
        {"from": "A", "to": "Lezione 10", "relation": "applies", "why": "una frase abbastanza lunga da contare"},  # spine target
        {"from": "Z", "to": "X", "relation": "applies", "why": "una frase abbastanza lunga da contare"},  # unknown from
        {"from": "A", "to": "X", "relation": "wat", "why": "una frase abbastanza lunga da contare"},      # bad relation
    ]
    kept = ol.select_edges(raw, ideas=ideas, existing=existing, spine_titles={"Lezione 10"})
    assert [(e["from"], e["to"], e["relation"]) for e in kept] == [
        ("A", "X", "applies"), ("B", "X", "relaxes"), ("C", "X", "bounds"), ("E", "Y", "same_as"),
    ]


def test_select_edges_accepts_an_echoed_outline_row_as_the_target():
    """Stage B is shown "lesson | title | claim" rows and answers with the row
    it read; exact matching threw away all 22 edges of a live run (2026-09-02)."""
    raw = [{"from": "SVDD", "to": "Lezione 11 | Margine geometrico", "relation": "applies",
            "why": "una frase abbastanza lunga da contare"}]
    kept = ol.select_edges(raw, ideas={"SVDD"}, existing={"Margine geometrico": "T/M.md"})
    assert [(e["from"], e["to"]) for e in kept] == [("SVDD", "Margine geometrico")]


def test_select_edges_matches_titles_case_insensitively():
    raw = [{"from": "support vector machines", "to": "MARGINE GEOMETRICO", "relation": "applies",
            "why": "una frase abbastanza lunga da contare"}]
    kept = ol.select_edges(raw, ideas={"Support Vector Machines"}, existing={"Margine geometrico": "T/M.md"})
    assert kept == [{"from": "Support Vector Machines", "to": "Margine geometrico", "relation": "applies",
                     "why": "una frase abbastanza lunga da contare"}]


def _ops():
    out = ol.parse_outline(_OUTLINE_JSON)
    out.ideas[0].body = "Il margine geometrico e' la distanza euclidea.\n$$\\gamma_r$$"
    out.ideas[1].body = "Novikoff: $(2R/\\gamma)^2$."
    edges = [
        {"from": "Errore del percettrone online", "to": "Bound sul rischio", "relation": "contrasts",
         "why": "due bound espressi nello stesso rapporto R/gamma"},
        {"from": "Margine geometrico", "to": "Margine", "relation": "same_as",
         "why": "la stessa definizione ripetuta nella lezione precedente"},
    ]
    existing = {"Bound sul rischio": "ML/Bound sul rischio.md", "Margine": "ML/Margine.md"}
    return out, ol.outline_ops(out, target="ML", hub="Machine learning",
                               source_basename="Lezione 11.md", edges=edges, existing=existing)


def test_outline_ops_write_per_idea_with_section_parent_and_relations():
    out, ops = _ops()
    writes = {o["heading"]: o for o in ops if o["op"] == "write"}
    assert set(writes) == {"Margine geometrico", "Errore del percettrone online", "Lezione 11"}
    err = writes["Errore del percettrone online"]
    assert err["path"] == "ML/Errore del percettrone online.md"
    assert err["source_basename"] == "Lezione 11.md"
    assert err["hub"] == "Machine learning"
    assert err["parent"] == "Lezione 11"
    assert err["section"] == "Errore del percettrone online"
    assert err["snippet"].startswith("> Novikoff limita gli errori")
    assert "## Relations" in err["snippet"]
    assert "- bounds [[Margine geometrico]]: il bound e' in gamma" in err["snippet"]
    assert "- contrasts with [[Bound sul rischio]] (Lezione 11 -> earlier): due bound" in err["snippet"]
    assert set(err["related"]) == {"Margine geometrico", "Bound sul rischio"}


def test_outline_ops_spine_note_lists_ideas_in_argument_order():
    out, ops = _ops()
    spine = next(o for o in ops if o["heading"] == "Lezione 11")
    assert spine["path"] == "ML/Lezione 11.md"
    assert spine["section"] == ol.SPINE_SECTION
    body = spine["snippet"]
    assert body.index("[[Margine geometrico]]") < body.index("[[Errore del percettrone online]]")
    assert "SVM: iperpiano di margine massimale" in body


def test_outline_ops_cross_edge_patches_the_existing_note():
    out, ops = _ops()
    patches = [o for o in ops if o["op"] == "patch"]
    assert len(patches) == 1
    p = patches[0]
    assert p["path"] == "ML/Bound sul rischio.md"
    assert p["heading"] == "Bound sul rischio"
    assert "[[Errore del percettrone online]]" in p["snippet"] and "contrasts with this" in p["snippet"]


def test_outline_ops_same_as_becomes_a_soft_near_title_flag_not_a_patch():
    from silica.router.states.distill import _NEAR_TITLE_RE
    out, ops = _ops()
    w = next(o for o in ops if o["heading"] == "Margine geometrico")
    m = _NEAR_TITLE_RE.search(w["review"])
    assert m and m.group(1) == "Margine" and m.group(2) == "ML/Margine.md"
    assert not any(o["op"] == "patch" and o["path"] == "ML/Margine.md" for o in ops)


def test_concept_entries_whitelist_ideas_spine_and_patch_targets():
    out, ops = _ops()
    entries = ol.concept_entries(ops)
    names = {e["name"]: e for e in entries}
    assert set(names) == {"Margine geometrico", "Errore del percettrone online", "Lezione 11", "Bound sul rischio"}
    assert names["Bound sul rischio"]["vault_collision"]["path"] == "ML/Bound sul rischio.md"
    assert names["Margine geometrico"]["vault_collision"] is None


def test_section_text_joins_repeated_headings_and_stops_at_the_next_one():
    txt = ol.section_text(_SRC, "Errore del percettrone online")
    assert "Novikoff" in txt and "dimostrazione" in txt
    assert "Formulazione lagrangiana" not in txt and "Margine" not in txt.split("\n", 1)[1]
    assert ol.section_text(_SRC, "Formulazione lagrangiana:") .startswith("## ***Formulazione lagrangiana:***")
    assert ol.section_text(_SRC, "Nope") == ""


def test_concept_entries_carry_the_section_as_excerpt():
    out, ops = _ops()
    names = {e["name"]: e for e in ol.concept_entries(ops, _SRC)}
    assert names["Margine geometrico"]["inbox_excerpt"].startswith("## Margine")
    assert "Novikoff" in names["Errore del percettrone online"]["inbox_excerpt"]
    assert names["Lezione 11"]["inbox_excerpt"] == _SRC  # spine: whole source


def test_validate_accepts_the_lane_ops(tmp_vault, monkeypatch):
    from silica.kernel.write.ops import OpType
    from silica.kernel.write.validate import validate_operations
    monkeypatch.setenv("SILICA_MIN_WRITE_SNIPPET_CHARS", "50")
    tmp_vault.note("ML/Bound sul rischio.md", "---\ntype: Note\n---\n# Bound sul rischio\n\nbody\n")
    tmp_vault.note("ML/Margine.md", "---\ntype: Note\n---\n# Margine\n\nbody\n")
    out, ops = _ops()
    payload = [{"batches": [{"inbox_file": "Inbox/Lezione 11.md", "concepts": ol.concept_entries(ops, _SRC)}]}]
    validated, rejected = validate_operations(ops, payload, "ML", hub="Machine learning")
    assert [r.reason for r in rejected] == []
    by = {v.heading: v for v in validated}
    err = by["Errore del percettrone online"]
    assert err.parent == "Lezione 11"            # forward reference to the spine written in the same batch
    assert err.section == "Errore del percettrone online"
    assert by["Margine geometrico"].review.startswith("near_title candidate='Margine'")
    assert by["Bound sul rischio"].op == OpType.patch and by["Bound sul rischio"].path == "ML/Bound sul rischio.md"
    assert by["Lezione 11"].section == ol.SPINE_SECTION


def test_vault_outline_reads_claims_and_skips_spines(tmp_vault):
    tmp_vault.note("ML/Margine.md", '---\nsources:\n  - "Lezione 10.md"\n---\n# Margine\n\n> Distanza dall\'iperpiano.\n\ncorpo\n')
    tmp_vault.note("ML/Kernel.md", '---\ntype: Note\n---\n# Kernel\n\nUn kernel e\' un prodotto interno in uno spazio trasformato. Altro testo.\n')
    tmp_vault.note("ML/Lezione 10.md", f'---\nsection: "{ol.SPINE_SECTION}"\n---\n# Lezione 10\n\n1. [[Margine]]\n')
    tmp_vault.note("ML/Machine learning.md", "# hub\n")
    rows = ol.vault_outline("ML", exclude_titles={"Machine learning"})
    by = {r["title"]: r for r in rows}
    assert set(by) == {"Margine", "Kernel"}
    assert by["Margine"]["claim"] == "Distanza dall'iperpiano."
    assert by["Margine"]["lesson"] == "Lezione 10"
    assert by["Kernel"]["claim"].startswith("Un kernel e' un prodotto interno")
    assert by["Margine"]["path"] == "ML/Margine.md"


def test_run_outliner_call_plan_and_result_shape():
    calls: list[str] = []

    def ask(system: str, user: str, *, max_tokens: int) -> dict:
        calls.append(system.split("\n", 1)[0])
        if system.startswith(ol.STAGE_A_TAG):
            return _OUTLINE_JSON
        if system.startswith(ol.STAGE_GAP_TAG):
            return {"ideas": [{"title": "Formulazione lagrangiana", "section": "Formulazione lagrangiana:",
                               "claim": "I vincoli entrano nella funzione obiettivo.", "depends_on": []}],
                    "skips": [{"section": "Support vector machines", "reason": "titolo vuoto"}]}
        if system.startswith(ol.STAGE_BODIES_TAG):
            titles = re.findall(r"^(.+?) \| ", user.split("IDEAS:\n", 1)[1], re.M)
            return {"bodies": {t: f"corpo di {t}" for t in titles}}
        if system.startswith(ol.STAGE_B_TAG):
            return {"edges": [{"from": "Margine geometrico", "to": "Margine", "relation": "same_as",
                               "why": "la stessa definizione ripetuta nella lezione precedente"}]}
        raise AssertionError(system[:40])

    res = ol.run_outliner(
        source_text=_SRC, source_basename="Lezione 11.md", target="ML", hub="Machine learning",
        language="Italian",
        vault_outline=[{"title": "Margine", "claim": "Distanza.", "lesson": "Lezione 10", "path": "ML/Margine.md"}],
        ask=ask,
    )
    assert calls == [ol.STAGE_A_TAG, ol.STAGE_GAP_TAG, ol.STAGE_BODIES_TAG, ol.STAGE_B_TAG]
    heads = {o["heading"] for o in res["updates"] if o["op"] == "write"}
    assert heads == {"Margine geometrico", "Errore del percettrone online", "Formulazione lagrangiana", "Lezione 11"}
    assert next(o for o in res["updates"] if o["heading"] == "Formulazione lagrangiana")["snippet"].endswith("corpo di Formulazione lagrangiana") or "corpo di Formulazione lagrangiana" in next(o for o in res["updates"] if o["heading"] == "Formulazione lagrangiana")["snippet"]
    assert res["ephemerals"] == []
    assert res["gaps"] == []  # the gap pass answered every uncovered section
    assert {e["name"] for e in res["concepts"]} >= heads
    assert "Support vector machines" in res["outline"]["skips"][0]["section"]


def test_run_outliner_bodies_in_batches_of_eight_and_no_stage_b_on_empty_vault():
    big = {"lesson_title": "t", "spine": [f"I{i}" for i in range(11)],
           "ideas": [{"title": f"I{i}", "section": "S", "claim": "c", "depends_on": []} for i in range(11)]}
    calls: list[str] = []

    def ask(system: str, user: str, *, max_tokens: int) -> dict:
        calls.append(system.split("\n", 1)[0])
        if system.startswith(ol.STAGE_A_TAG):
            return big
        if system.startswith(ol.STAGE_BODIES_TAG):
            titles = re.findall(r"^(.+?) \| ", user.split("IDEAS:\n", 1)[1], re.M)
            assert len(titles) <= 8
            return {"bodies": {t: "b" for t in titles}}
        raise AssertionError(system[:40])

    res = ol.run_outliner(source_text="## S\ntext", source_basename="L.md", target="T", hub="H",
                          language="English", vault_outline=[], ask=ask)
    assert calls == [ol.STAGE_A_TAG, ol.STAGE_BODIES_TAG, ol.STAGE_BODIES_TAG]
    assert len([o for o in res["updates"] if o["op"] == "write"]) == 12


def test_run_outliner_retry_regenerates_only_the_named_bodies():
    calls: list[str] = []

    def ask(system: str, user: str, *, max_tokens: int) -> dict:
        calls.append(system.split("\n", 1)[0])
        titles = re.findall(r"^(.+?) \| ", user.split("IDEAS:\n", 1)[1], re.M)
        assert titles == ["Margine geometrico"]
        assert "steer me" in user
        return {"bodies": {"Margine geometrico": "corpo nuovo"}}

    prior = ol.parse_outline(_OUTLINE_JSON)
    res = ol.run_outliner(source_text=_SRC, source_basename="Lezione 11.md", target="ML", hub="H",
                          language="Italian", vault_outline=[], ask=ask,
                          outline=prior, only_titles={"Margine geometrico"}, steer_context="steer me")
    assert calls == [ol.STAGE_BODIES_TAG]
    assert [o["heading"] for o in res["updates"]] == ["Margine geometrico"]
    assert "corpo nuovo" in res["updates"][0]["snippet"]


def test_lane_for_routes_document_profiles_to_outline(monkeypatch):
    monkeypatch.setattr("silica.config.CONFIG.nucleate_lane", "outline")
    assert ol.lane_for("default") == "outline"
    assert ol.lane_for(None) == "outline"
    assert ol.lane_for("transcript") == "outline"
    assert ol.lane_for("extractive") == "outline"
    assert ol.lane_for("clip") == "keyphrase"
    assert ol.lane_for("promotion") == "keyphrase"
    monkeypatch.setattr("silica.config.CONFIG.nucleate_lane", "keyphrase")
    assert ol.lane_for("default") == "keyphrase"


def test_render_write_stamps_the_section(tmp_vault):
    from silica.kernel.write.bulk import render_write
    from silica.kernel.write.ops import Op, OpType
    op = Op(op=OpType.write, heading="Margine geometrico", source_basename="Lezione 11.md",
            path="ML/Margine geometrico.md", snippet="x" * 300, hub="Machine learning",
            section="Margine")
    content = render_write(op)
    assert re.search(r'^section: "?Margine"?$', content.split("\n---\n")[0], re.M)


# ------------------------------------------------------------ FSM seams --
from unittest.mock import MagicMock, patch

from silica.router import states
from silica.router.orchestrator import InjectorFSM, InjectorState


def _outline_chunk(text: str = "## Margine\ncorpo") -> dict:
    return {"schema_version": 1, "lane": "outline",
            "batches": [{"inbox_file": "Inbox/test.md", "concepts": []}], "source_text": text}


def _outliner_result(**over) -> dict:
    base = {
        "updates": [{"op": "write", "path": "TargetDir/Margine geometrico.md", "heading": "Margine geometrico",
                     "source_basename": "test.md", "snippet": "x", "section": "Margine"}],
        "ephemerals": [],
        "concepts": [{"name": "Margine geometrico", "action_hint": "create", "inbox_excerpt": "", "vault_collision": None}],
        "outline": {"lesson_title": "t", "spine": ["Margine geometrico"],
                    "ideas": [{"title": "Margine geometrico", "section": "Margine", "claim": "c", "body": "x", "depends_on": []}],
                    "skips": []},
        "gaps": [],
    }
    base.update(over)
    return base


@patch("silica.router.states.distill.vault_outline",
       return_value=[{"title": "Margine", "claim": "c", "lesson": "Lezione 10", "path": "TargetDir/Margine.md"}])
@patch("silica.router.states.distill.run_outliner")
def test_delegate_routes_outline_chunks_to_the_outliner_and_fills_the_whitelist(mock_run, mock_vo):
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._chunks = [_outline_chunk()]
    fsm._current_chunk_idx = 0
    fsm.context["file_0_language"] = "Italian"
    fsm.state = InjectorState.DELEGATE
    mock_run.return_value = _outliner_result(gaps=["Sezione persa"])
    with patch.object(fsm, "_make_tmp", return_value="tmp.json"):
        fsm.step()
    kw = mock_run.call_args.kwargs
    assert kw["source_text"] == "## Margine\ncorpo"
    assert kw["source_basename"] == "test.md"
    assert kw["target"] == "TargetDir" and kw["hub"] == "TargetDir"
    assert kw["language"] == "Italian"
    assert kw["vault_outline"][0]["title"] == "Margine"
    assert kw["outline"] is None and kw["only_titles"] is None
    # VALIDATE's heading whitelist reads the chunk: the model's titles are now in it.
    assert fsm._chunks[0]["batches"][0]["concepts"][0]["name"] == "Margine geometrico"
    assert fsm.context["chunk_0_outline"]["spine"] == ["Margine geometrico"]
    assert fsm.state == InjectorState.SANITIZE


@patch("silica.router.states.distill.vault_outline", return_value=[])
@patch("silica.router.states.distill.run_outliner")
def test_delegate_steer_retry_regenerates_only_rejected_bodies(mock_run, mock_vo):
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    chunk = _outline_chunk()
    chunk["batches"][0]["concepts"] = _outliner_result()["concepts"]
    fsm._chunks = [chunk]
    fsm._current_chunk_idx = 0
    fsm.state = InjectorState.DELEGATE
    fsm.context["chunk_0_outline"] = _outliner_result()["outline"]
    fsm.context["chunk_0_retry_payload"] = {**chunk, "batches": [{"inbox_file": "Inbox/test.md",
                                                                  "concepts": [{"name": "Margine geometrico"}]}]}
    fsm.context["chunk_0_steer_context"] = "snippet too short"
    fsm.context["chunk_0_hash"] = "h"
    mock_run.return_value = _outliner_result()
    with patch.object(fsm, "_make_tmp", return_value="tmp.json"):
        fsm.step()
    kw = mock_run.call_args.kwargs
    assert kw["only_titles"] == {"Margine geometrico"}
    assert kw["outline"].spine == ["Margine geometrico"]
    assert kw["steer_context"] == "snippet too short"
    assert fsm.state == InjectorState.SANITIZE


def test_assemble_file_chunks_outline_lane_makes_one_full_text_chunk():
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    recon = {"success": True, "outline_lane": True, "file": "Inbox/test.md", "source_text": "## A\nbody"}
    with patch("silica.router.orchestrator.silica_payload") as mock_payload:
        res, chunks = states.setup._assemble_file_chunks(fsm, recon)
    mock_payload.assert_not_called()
    assert chunks == [_outline_chunk("## A\nbody")]
    assert res == {"chunks": chunks}


def test_recon_skips_the_miner_on_the_outline_lane(monkeypatch):
    monkeypatch.setattr("silica.config.CONFIG.nucleate_lane", "outline")
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.state = InjectorState.RECON
    monkeypatch.setattr(states.setup, "_pin_file_profile",
                        lambda fsm, fi, f: fsm.context.__setitem__(f"file_{fi}_profile", "default"))
    monkeypatch.setattr(states.setup, "read_source_text", lambda rel: "## A\nbody")
    with patch("silica.router.orchestrator.silica_recon") as mock_recon:
        fsm.step()
    mock_recon.assert_not_called()
    rec = fsm.context["recon"][-1]
    assert rec["outline_lane"] is True and rec["source_text"] == "## A\nbody"
    assert fsm.context["file_0_lane"] == "outline"
    assert fsm.state == InjectorState.PAYLOAD


def test_recon_keeps_the_miner_for_memory_forms(monkeypatch):
    monkeypatch.setattr("silica.config.CONFIG.nucleate_lane", "outline")
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.state = InjectorState.RECON
    monkeypatch.setattr(states.setup, "_pin_file_profile",
                        lambda fsm, fi, f: fsm.context.update({f"file_{fi}_profile": "clip",
                                                                f"file_{fi}_form_origin": "stamp"}))
    with patch("silica.router.orchestrator.silica_recon", return_value={"success": True, "concepts": []}) as mock_recon:
        fsm.step()
    mock_recon.assert_called_once_with("Inbox/test.md")
    assert fsm.context["file_0_lane"] == "keyphrase"


def test_collision_leaves_outline_chunks_untouched(monkeypatch):
    from silica.router.states import collision
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    chunk = _outline_chunk()
    fsm._chunks = [chunk]
    fsm._current_chunk_idx = 0
    store = MagicMock()
    store.__len__ = lambda _self: 5
    monkeypatch.setattr("silica.kernel.recall.embed.get_store", lambda: store)
    collision.collision_pass(fsm, 0)
    assert fsm._chunks[0] == chunk
    assert fsm._chunks[0]["lane"] == "outline" and fsm._chunks[0]["source_text"] == "## Margine\ncorpo"


def test_near_title_gate_ignores_titles_that_differ_only_by_number(tmp_vault, monkeypatch):
    """'Lezione 12' is not a near-duplicate of 'Lezione 11': a number in a
    title is identity (lectures, chapters, versions). Measured 2026-09-02: the
    gate flagged the lesson-12 spine at ratio 0.9 and the judge merged it into
    lesson 11."""
    from silica.kernel.write.validate import validate_operations
    monkeypatch.setenv("SILICA_MIN_WRITE_SNIPPET_CHARS", "50")
    tmp_vault.note("ML/Lezione 11.md", "---\ntype: Note\n---\n# Lezione 11\n\n1. [[a]]\n")
    ops = [{"op": "write", "path": "ML/Lezione 12.md", "heading": "Lezione 12", "title": "Lezione 12",
            "source_basename": "Lezione 12.md", "hub": "Machine learning", "snippet": "> t\n\n" + "1. [[b]]: c\n" * 20}]
    payload = [{"batches": [{"inbox_file": "Inbox/Lezione 12.md",
                             "concepts": [{"name": "Lezione 12", "inbox_excerpt": "x" * 400}]}]}]
    validated, rejected = validate_operations(ops, payload, "ML", hub="Machine learning")
    assert rejected == []
    op = next(v for v in validated if v.heading == "Lezione 12")
    assert op.op.value == "write" and op.path == "ML/Lezione 12.md"   # not coerced onto Lezione 11
    assert not (op.review or "").startswith("near_title")


def test_numbers_differ_separates_lectures_but_folds_enumerators():
    from silica.kernel.text.title import near_titles, numbers_differ
    assert numbers_differ("Lezione 11", "Lezione 12")
    assert numbers_differ("Capitolo 3", "capitolo 4")
    assert not numbers_differ("Foo", "Foo 1")          # a slide enumerator on one side only
    assert not numbers_differ("Lezione 11", "lezione 11")
    assert near_titles("Capitolo 4", ["Capitolo 3"]) == []
    assert near_titles("Descriptor", ["Description"])   # the band still fires without numbers


def test_cold_intra_chunk_predicate_keeps_two_lectures_apart():
    from silica.router.states.collision import _cold_intra_chunk_near_dup
    assert not _cold_intra_chunk_near_dup(("Lezione 11", "a" * 50), ("Lezione 12", "b" * 50))
    assert _cold_intra_chunk_near_dup(("Neurone artificiale", "a"), ("Neurone Artificiale 1", "b"))



def test_outline_language_follows_the_manifest_then_detection(monkeypatch):
    from silica.router.states import distill
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm._chunks = [_outline_chunk("## Margine\nIl margine geometrico è la distanza euclidea dall'iperpiano di separazione.")]
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return _outliner_result()
    monkeypatch.setattr(distill, "run_outliner", fake_run)
    monkeypatch.setattr(distill, "vault_outline", lambda *a, **k: [])
    distill._run_outline_chunk(fsm, 0, fsm._chunks[0], None, None)
    assert seen["language"] == "Italian"          # no pin, no manifest: detected from the source
    fsm.context["file_0_language"] = "French"
    distill._run_outline_chunk(fsm, 0, fsm._chunks[0], None, None)
    assert seen["language"] == "French"           # the per-file pin wins


def test_default_ask_gives_up_at_the_distiller_deadline(monkeypatch):
    """OpenRouter trickles keep-alive bytes on a dead upstream, so only wall
    clock bounds a stage call (same helper as run_distiller). Measured
    2026-09-02: the live run sat 10 minutes on one open socket."""
    import threading, time
    from silica.kernel import outline as ol

    class _Provider:
        model = "fake"

        def call_llm(self, **kw):
            cancel = kw.get("cancel")
            for _ in range(50):
                if cancel is not None and cancel.is_set():
                    return None
                time.sleep(0.05)
            return None

    monkeypatch.setattr("silica.agent.providers.get_provider", lambda cfg, role="worker": _Provider())
    monkeypatch.setenv("DISTILLER_TIMEOUT", "0.2")
    t0 = time.time()
    with pytest.raises(TimeoutError):
        ol._default_ask("s", "u", max_tokens=10)
    assert time.time() - t0 < 1.5


def test_default_ask_uses_a_response_schema_and_retries_once_on_no_json(monkeypatch):
    """Structured decode is the distiller's proven guard against a reply that
    is thinking text or a cut JSON (live4, 2026-09-02: finish=stop with no
    brace, then finish=length). One retry with a doubled budget on a cut."""
    from types import SimpleNamespace
    from silica.kernel import outline as ol
    calls = []

    class _Provider:
        model = "fake"

        def call_llm(self, **kw):
            calls.append(kw)
            if len(calls) == 1:
                return SimpleNamespace(text="I will now think about the lecture", finish_reason="length")
            return SimpleNamespace(text='{"ideas": []}', finish_reason="stop")

    monkeypatch.setattr("silica.agent.providers.get_provider", lambda cfg, role="worker": _Provider())
    out = ol._default_ask(ol.STAGE_A, "src", max_tokens=100)
    assert out == {"ideas": []}
    assert len(calls) == 2
    assert calls[0]["response_schema"] is None and calls[0]["reasoning"] is False
    assert calls[1]["max_tokens"] == 200          # the cut reply gets double the budget
    assert calls[1]["response_schema"] is ol.OutlineReply   # and a different decode mode


def test_recon_ignores_a_sniffed_clip_for_the_lane(monkeypatch):
    """The sniffer is an LLM call whose verdict changes run to run; only a
    stamp or the run-level profile may route a file off the outline lane."""
    monkeypatch.setattr("silica.config.CONFIG.nucleate_lane", "outline")
    fsm = InjectorFSM("Inbox/test.md", "TargetDir")
    fsm.state = InjectorState.RECON
    monkeypatch.setattr(states.setup, "_pin_file_profile",
                        lambda fsm, fi, f: fsm.context.update({f"file_{fi}_profile": "clip",
                                                                f"file_{fi}_form_origin": "sniff"}))
    monkeypatch.setattr(states.setup, "read_source_text", lambda rel: "## A\nbody")
    with patch("silica.router.orchestrator.silica_recon") as mock_recon:
        fsm.step()
    mock_recon.assert_not_called()
    assert fsm.context["file_0_lane"] == "outline"
