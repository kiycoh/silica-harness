# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Episodic memory lane — short-term fact store with supersedes chains and TTL
(docs spec 2026-07-14). Store unit tests use an explicit path; no global state."""
from __future__ import annotations

import json

from silica.kernel.recall.episodic import EpisodicStore, Fact


def _store(tmp_path):
    return EpisodicStore(path=tmp_path / "episodic.json")


def test_capture_new_key_persists_round_trip(tmp_path):
    store = _store(tmp_path)
    store.capture(
        [{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
        run_id="run_a3f2",
        seen="2026-07-14",
    )

    reloaded = EpisodicStore(path=tmp_path / "episodic.json")
    facts = reloaded.live_facts()
    assert len(facts) == 1
    f = facts[0]
    assert f.key == "user.dog.name"
    assert f.text == "Il mio cane si chiama Tom"
    assert f.first_seen == "2026-07-14"
    assert f.last_seen == "2026-07-14"
    assert f.runs == ["run_a3f2"]
    assert f.supersedes is None
    assert f.status == "live"


def test_reinforce_same_key_same_normalized_text(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_1", seen="2026-06-10")
    # Same fact, different casing/punctuation — reinforces, no new fact.
    store.capture([{"key": "user.dog.name", "text": "il mio cane si chiama tom!"}],
                  run_id="run_2", seen="2026-07-01")
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_2", seen="2026-07-02")

    facts = store.live_facts()
    assert len(facts) == 1
    f = facts[0]
    assert f.first_seen == "2026-06-10"
    assert f.last_seen == "2026-07-02"
    assert f.runs == ["run_1", "run_2"]  # run_2 appended once


def test_supersede_same_key_different_text_keeps_chain(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Rex"}],
                  run_id="run_1", seen="2026-03-01")
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_2", seen="2026-06-10")

    live = store.live_facts()
    assert len(live) == 1
    head = live[0]
    assert head.text == "Il mio cane si chiama Tom"
    assert head.first_seen == "2026-06-10"

    old = next(f for f in store.facts if f.id == head.supersedes)
    assert old.text == "Il mio cane si chiama Rex"
    assert old.status == "superseded"
    assert old.supersedes is None

    # Chain grows: a third value points at the second.
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Ugo"}],
                  run_id="run_3", seen="2026-07-01")
    (head2,) = store.live_facts()
    assert head2.supersedes == head.id
    assert next(f for f in store.facts if f.id == head.id).status == "superseded"


class _FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _BrokenEmbedder:
    def embed(self, texts):
        raise RuntimeError("embedder down")


def test_capture_embeds_new_facts_when_embedder_served(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Tom"}],
                  run_id="r1", seen="2026-07-14", embedder=_FakeEmbedder())
    (f,) = store.live_facts()
    assert f.vec == [1.0, 0.0]


def test_capture_without_embedder_or_broken_embedder_skips_silently(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "a.b", "text": "x"}], run_id="r1", seen="2026-07-14")
    store.capture([{"key": "c.d", "text": "y"}],
                  run_id="r1", seen="2026-07-14", embedder=_BrokenEmbedder())
    assert all(f.vec is None for f in store.live_facts())


class _KeyedEmbedder:
    """Fixed vec per exact input string; unknown inputs (fact texts) get a
    distinct axis, so key vectors are deterministic in tests."""

    _TABLE = {
        "user photo pic": [1.0, 0.0, 0.0],
        "user photo shot": [0.96, 0.28, 0.0],   # cos vs pic = 0.96
        "user hobby piano": [0.0, 1.0, 0.0],
    }

    def embed(self, texts):
        return [self._TABLE.get(t, [0.0, 0.0, 1.0]) for t in texts]


def test_synonym_keys_start_separate_chains(tmp_path):
    """Keys match canonically, never by embedding proximity: `photo.pic` and
    `photo.shot` are two chains, not one (the embed-snap fallback that would
    have joined them was refuted by the 2026-08-02 audit and removed)."""
    store = _store(tmp_path)
    store.capture([{"key": "user.photo.pic", "text": "a"}],
                  run_id="r1", seen="2026-07-18", embedder=_KeyedEmbedder())
    store.capture([{"key": "user.photo.shot", "text": "b"}],
                  run_id="r2", seen="2026-07-18", embedder=_KeyedEmbedder())
    assert len(store.live_facts()) == 2


class _TextEmbedder:
    """Fixed vec per fact text — drives the supersede gate deterministically."""

    _TABLE = {
        "the dog is named Tom": [1.0, 0.0, 0.0],
        "the dog is named Rex": [0.9, 0.436, 0.0],   # cos vs Tom ≈ 0.9
        "pottery class started yesterday": [0.0, 1.0, 0.0],  # cos vs Tom = 0
    }

    def embed(self, texts):
        return [self._TABLE.get(t, [0.0, 0.0, 1.0]) for t in texts]


def test_supersede_gate_forks_distinct_facts_under_a_reused_key(tmp_path):
    """The key-collision fix: same slot key, unrelated text — both facts stay
    live as sibling chains, no fabricated 'previously'."""
    store = _store(tmp_path)
    store.capture([{"key": "user.event.date", "text": "the dog is named Tom"}],
                  run_id="r1", seen="2026-08-01",
                  embedder=_TextEmbedder(), supersede_tau=0.7)
    store.capture([{"key": "user.event.date", "text": "pottery class started yesterday"}],
                  run_id="r2", seen="2026-08-02",
                  embedder=_TextEmbedder(), supersede_tau=0.7)
    live = store.live_facts()
    assert len(live) == 2
    assert all(f.supersedes is None for f in live)
    assert {f.key for f in live} == {"user.event.date"}


def test_supersede_gate_lets_a_genuine_update_supersede(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "the dog is named Tom"}],
                  run_id="r1", seen="2026-08-01",
                  embedder=_TextEmbedder(), supersede_tau=0.7)
    store.capture([{"key": "user.dog.name", "text": "the dog is named Rex"}],
                  run_id="r2", seen="2026-08-02",
                  embedder=_TextEmbedder(), supersede_tau=0.7)
    live = store.live_facts()
    assert len(live) == 1
    assert live[0].text == "the dog is named Rex"
    assert len(store.chain(live[0])) == 2


def test_supersede_gate_abstains_without_an_embedder(tmp_path):
    """No vectors -> legacy supersede, never a silent behavior flip."""
    store = _store(tmp_path)
    store.capture([{"key": "k", "text": "the dog is named Tom"}],
                  run_id="r1", seen="2026-08-01", supersede_tau=0.7)
    store.capture([{"key": "k", "text": "pottery class started yesterday"}],
                  run_id="r2", seen="2026-08-02", supersede_tau=0.7)
    assert len(store.live_facts()) == 1


def test_supersede_gate_abstains_when_the_head_has_no_vec(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "k", "text": "the dog is named Tom"}],
                  run_id="r1", seen="2026-08-01")  # no embedder: head unembedded
    store.capture([{"key": "k", "text": "pottery class started yesterday"}],
                  run_id="r2", seen="2026-08-02",
                  embedder=_TextEmbedder(), supersede_tau=0.7)
    assert len(store.live_facts()) == 1


def test_supersede_gate_ships_off(monkeypatch):
    """Off until something moves the product metric: the conv-26 answer A/B was
    null (52.8% -> 53.8%, p=0.845). The gate works — it just has not earned the
    default. config.py loads ~/.silica/.env at import, so the default_factory
    reads the developer's own pin unless it is cleared here."""
    from silica.config import SilicaConfig

    monkeypatch.delenv("SILICA_EPISODIC_SUPERSEDE_TAU", raising=False)
    assert SilicaConfig().episodic_supersede_tau == 0


def test_capture_from_distill_wires_supersede_tau_from_config(tmp_path, monkeypatch):
    import silica.agent.providers as providers
    from silica.config import CONFIG
    from silica.kernel.recall.episodic import EpisodicStore, capture_from_distill

    monkeypatch.setattr(CONFIG, "episodic_supersede_tau", 0.7, raising=False)
    monkeypatch.setattr(providers, "get_embedder", lambda cfg: _TextEmbedder())
    capture_from_distill(
        {"ephemerals": [{"key": "user.slot", "text": "the dog is named Tom"}]},
        run_id="r1", seen="2026-08-01")
    capture_from_distill(
        {"ephemerals": [{"key": "user.slot", "text": "pottery class started yesterday"}]},
        run_id="r2", seen="2026-08-02")
    assert len(EpisodicStore().live_facts()) == 2


def test_episodic_home_resolves_even_when_active_vault_is_memory_vault(tmp_path, monkeypatch):
    """Unlike memory_lane.memory_vault(), episodic_home never abstains."""
    from silica.config import CONFIG
    from silica.kernel.recall.episodic import episodic_home

    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(CONFIG, "memory_vault", str(vault), raising=False)
    monkeypatch.setattr(CONFIG, "vault_path", str(vault), raising=False)
    assert episodic_home() == vault.resolve()


def test_corrupt_store_file_is_quarantined_not_fatal(tmp_path):
    p = tmp_path / "episodic.json"
    p.write_text("{not json", encoding="utf-8")
    store = EpisodicStore(path=p)
    assert store.live_facts() == []
    # Original bytes preserved aside, store restarts empty.
    assert any(".corrupt." in q.name for q in tmp_path.iterdir())


def test_sweep_evaporates_whole_chain_by_head_last_seen(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Rex"}], run_id="r1", seen="2026-01-01")
    store.capture([{"key": "user.dog.name", "text": "Tom"}], run_id="r2", seen="2026-02-01")
    store.capture([{"key": "user.city", "text": "Torino"}], run_id="r2", seen="2026-07-01")

    removed = store.sweep(now="2026-07-14", ttl_days=90)
    # dog chain head last_seen 2026-02-01 is >90d old: head AND superseded
    # ancestor evaporate together; city (13d old) survives.
    assert removed == 1
    assert {f.key for f in store.facts} == {"user.city"}

    # Reloaded store reflects the sweep (sweep persists).
    assert {f.key for f in EpisodicStore(path=store.path).facts} == {"user.city"}


def test_sweep_reinforcement_resets_timer_and_zero_ttl_never_expires(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Tom"}], run_id="r1", seen="2026-01-01")
    store.capture([{"key": "user.dog.name", "text": "Tom"}], run_id="r2", seen="2026-07-10")
    assert store.sweep(now="2026-07-14", ttl_days=90) == 0
    assert len(store.live_facts()) == 1

    store.capture([{"key": "old.fact", "text": "x"}], run_id="r1", seen="2020-01-01")
    assert store.sweep(now="2026-07-14", ttl_days=0) == 0  # 0 = never expire
    assert len(store.live_facts()) == 2


def test_nucleation_candidates_count_distinct_runs_across_chain(tmp_path):
    store = _store(tmp_path)
    # user.dog.name: 3 distinct runs spread over a supersede (Rex r1+r2, Tom r3)
    store.capture([{"key": "user.dog.name", "text": "Rex"}], run_id="r1", seen="2026-06-10")
    store.capture([{"key": "user.dog.name", "text": "Rex"}], run_id="r2", seen="2026-06-20")
    store.capture([{"key": "user.dog.name", "text": "Tom"}], run_id="r3", seen="2026-07-01")
    # user.city: 2 runs only — below threshold
    store.capture([{"key": "user.city", "text": "Torino"}], run_id="r1", seen="2026-06-10")
    store.capture([{"key": "user.city", "text": "Torino"}], run_id="r2", seen="2026-06-20")

    cands = store.nucleation_candidates(min_runs=3)
    assert len(cands) == 1
    c = cands[0]
    assert c.key == "user.dog.name"
    assert c.run_count == 3
    assert c.since == "2026-06-10"


def test_a_promoted_chain_leaves_the_candidate_queue(tmp_path):
    """Promotion is the exit from the queue: the note exists, stop suggesting it."""
    store = _store(tmp_path)
    for rid in ("r1", "r2", "r3"):
        store.capture([{"key": "user.dog.name", "text": "Tom"}], run_id=rid,
                      seen="2026-06-10")

    (head,) = store.live_facts()
    head.promoted = "Concepts/Dog.md"
    store.save()

    reloaded = EpisodicStore(path=tmp_path / "episodic.json")
    assert reloaded.live_facts()[0].promoted == "Concepts/Dog.md"
    assert reloaded.nucleation_candidates(min_runs=3) == []


def test_promotion_stub_survives_the_payload_window_whole(tmp_path):
    """Regression, measured live: with `## <key>` headings per attribute, the
    payload builder's heading-section match reduced the excerpt to ONE
    attribute's line — the distiller never saw the entity, emitted empty
    bodies, and every /promote ended no_ops. The stub must reach the distiller
    whole, whatever concept the keyphrase extractor picks."""
    from silica.kernel.recall.episodic import promotion_stub
    from silica.kernel.text.payload import extract_excerpt_from_content

    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Rex"}], run_id="r1", seen="2026-06-10")
    store.capture([{"key": "user.dog.breed", "text": "pastore tedesco"}], run_id="r1", seen="2026-06-10")

    heads = sorted(store.live_facts(), key=lambda f: f.key)
    stub = promotion_stub(heads, store=store)

    for concept in ("user.dog.name", "user.dog.breed", "Rex"):
        excerpt = extract_excerpt_from_content(stub, concept, 450)
        assert "Rex" in excerpt and "pastore tedesco" in excerpt, (
            f"concept {concept!r} windowed the stub down to: {excerpt!r}"
        )


def test_promotion_stub_carries_the_chain_and_its_provenance(tmp_path):
    """What /promote feeds the gate: current value, dated history, provenance."""
    from silica.kernel.recall.episodic import promotion_stub

    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Rex"}], run_id="r1", seen="2026-06-10")
    store.capture([{"key": "user.dog.name", "text": "Rex"}], run_id="r2", seen="2026-06-20")
    store.capture([{"key": "user.dog.name", "text": "Tom"}], run_id="r3", seen="2026-07-01")

    (head,) = store.live_facts()
    text = promotion_stub([head], store=store)

    front, body = text.split("---\n")[1], text.split("---\n")[2]
    assert "episodic_key: user.dog" in front       # the entity is the note
    assert "episodic_attributes: user.dog.name" in front
    assert "first_seen: 2026-06-10" in front  # the chain's oldest, not the head's
    assert "last_seen: 2026-07-01" in front
    assert "sessions: 3" in front
    assert "Tom" in body
    assert "Rex" in body and "2026-06-10" in body  # superseded history survives


def test_recall_ranks_by_embedding_when_vectors_exist(tmp_path):
    store = _store(tmp_path)

    class _E:
        def embed(self, texts):
            return [[1.0, 0.1] if "cane" in t else [0.1, 1.0] for t in texts]

    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"},
                   {"key": "user.city", "text": "Vivo a Torino"}],
                  run_id="r1", seen="2026-07-14", embedder=_E())

    # floor=0 isolates the ranking assertion; the floor itself is covered below.
    hits = store.recall("come si chiama il mio cane", query_vec=[1.0, 0.0],
                        k=2, now="2026-07-14", floor=0.0)
    assert [h.fact.key for h in hits] == ["user.dog.name", "user.city"]
    assert hits[0].score > hits[1].score


def test_recall_floor_drops_off_topic_facts_on_the_embed_leg(tmp_path):
    """Without a floor, `score > 0` never rejects a cosine, so top-k ships the
    whole store on every query (measured: an 11-fact store returned the same 10
    facts for "pasta recipe with tomatoes" as for an on-topic question)."""
    store = _store(tmp_path)

    class _E:
        def embed(self, texts):
            return [[1.0, 0.1] if "cane" in t else [0.1, 1.0] for t in texts]

    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"},
                   {"key": "user.city", "text": "Vivo a Torino"}],
                  run_id="r1", seen="2026-07-14", embedder=_E())

    hits = store.recall("come si chiama il mio cane", query_vec=[1.0, 0.0],
                        k=5, now="2026-07-14", floor=0.5)
    assert [h.fact.key for h in hits] == ["user.dog.name"]  # 0.0995 cosine cut
    # An off-topic query clears nothing, so perceive() emits no facts block.
    assert store.recall("qualcosa di totalmente altro", query_vec=[0.0, 0.0],
                        k=5, now="2026-07-14", floor=0.5) == []
    # The lexical leg keeps its own scale: a 2-term query matching 1 term is 0.5.
    assert store.recall("cane Torino", query_vec=None, k=5,
                        now="2026-07-14", floor=0.5)


def test_recall_lexical_fallback_without_vectors_and_live_only(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Rex"}],
                  run_id="r1", seen="2026-07-01")
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="r2", seen="2026-07-10")
    store.capture([{"key": "user.meeting", "text": "Riunione lunedì"}],
                  run_id="r2", seen="2026-07-10")

    hits = store.recall("cane", query_vec=None, k=5, now="2026-07-14")
    # Only the live head of the dog chain matches; superseded Rex never
    # surfaces as its own hit even though its text also matches.
    assert [h.fact.text for h in hits][0] == "Il mio cane si chiama Tom"
    assert all(h.fact.status == "live" for h in hits)
    # Key segments count as lexical signal too.
    hits_by_key = store.recall("dog name", query_vec=None, k=5, now="2026-07-14")
    assert hits_by_key and hits_by_key[0].fact.key == "user.dog.name"


def test_recall_filters_expired_chains_without_mutating_store(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.old", "text": "vecchio fatto sul cane"}],
                  run_id="r1", seen="2026-01-01")
    hits = store.recall("cane", query_vec=None, k=5, now="2026-07-14", ttl_days=90)
    assert hits == []
    # Recall never deletes: sweep at digest time is the only deleter.
    assert len(store.facts) == 1


def test_render_includes_chain_history_with_dates(tmp_path):
    from silica.kernel.recall import episodic

    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Rex"}],
                  run_id="r1", seen="2026-03-01")
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="r2", seen="2026-06-10")
    store.capture([{"key": "user.city", "text": "Vivo a Torino"}],
                  run_id="r2", seen="2026-06-10")

    hits = store.recall("cane Torino", query_vec=None, k=5, now="2026-07-14")
    text = episodic.render(hits, store=store)
    assert "- [since 2026-06-10] Il mio cane si chiama Tom" in text
    assert "(previously: Il mio cane si chiama Rex, 2026-03-01 to 2026-06-10)" in text
    assert "- [since 2026-06-10] Vivo a Torino" in text
    # No chain for the city fact — no "previously" line for it.
    assert text.count("previously") == 1


def test_render_empty_hits_is_empty_string(tmp_path):
    from silica.kernel.recall import episodic

    assert episodic.render([], store=_store(tmp_path)) == ""


def test_distiller_output_parses_with_and_without_ephemerals():
    from silica.kernel.write.ops import DistillerOutput

    legacy = DistillerOutput.model_validate({"updates": []})
    assert legacy.ephemerals == []

    doc = DistillerOutput.model_validate({
        "updates": [],
        "ephemerals": [{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
    })
    assert doc.ephemerals[0].key == "user.dog.name"
    assert doc.ephemerals[0].text == "Il mio cane si chiama Tom"


def test_config_episodic_fields_env_overrides(monkeypatch):
    from silica.config import SilicaConfig

    assert SilicaConfig().episodic_ttl_days == 90
    assert SilicaConfig().episodic_nucleation_runs == 3
    monkeypatch.setenv("SILICA_EPISODIC_TTL_DAYS", "0")
    monkeypatch.setenv("SILICA_EPISODIC_NUCLEATION_RUNS", "5")
    cfg = SilicaConfig()
    assert cfg.episodic_ttl_days == 0
    assert cfg.episodic_nucleation_runs == 5


def test_a_store_written_before_the_provenance_stamps_loads_unchanged(tmp_path):
    """`vault`/`notes` are additive: pre-phase-E stores must still open."""
    path = tmp_path / "episodic.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "next_id": 2,
        "facts": [{"id": "f_0001", "key": "user.city", "text": "Torino",
                   "first_seen": "2026-07-01", "last_seen": "2026-07-01",
                   "runs": ["r1"]}],
    }), encoding="utf-8")

    (fact,) = EpisodicStore(path=path).live_facts()

    assert fact.vault is None
    assert fact.notes == []


def test_capture_from_distill_routes_ephemerals_and_never_raises(tmp_path, monkeypatch):
    from silica.kernel.recall import episodic

    monkeypatch.setattr(episodic, "store_path", lambda: tmp_path / "episodic.json")
    result = {
        "updates": [],
        "ephemerals": [{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"},
                       {"key": "", "text": "junk ignored"}],
    }
    episodic.capture_from_distill(result, run_id="run_x", seen="2026-07-14")
    (f,) = EpisodicStore(path=tmp_path / "episodic.json").live_facts()
    assert f.key == "user.dog.name"
    assert f.runs == ["run_x"]

    # No ephemerals / broken store: silent no-op, ingest must never fail.
    episodic.capture_from_distill({"updates": []}, run_id="r", seen="2026-07-14")
    monkeypatch.setattr(episodic, "store_path",
                        lambda: (_ for _ in ()).throw(RuntimeError("disk gone")))
    episodic.capture_from_distill(result, run_id="r", seen="2026-07-14")


def test_digest_sweeps_and_lists_nucleation_candidates(tmp_path, monkeypatch):
    import datetime as _dt

    from silica.kernel.recall import episodic
    from silica.kernel.progress import ProgressLedger

    monkeypatch.setattr(episodic, "store_path", lambda: tmp_path / "episodic.json")
    store = EpisodicStore(path=tmp_path / "episodic.json")
    today = _dt.date.today().isoformat()
    for rid in ("r1", "r2", "r3"):
        store.capture([{"key": "user.dog.name", "text": "Tom"}], run_id=rid, seen=today)
    store.capture([{"key": "user.stale", "text": "old"}], run_id="r0", seen="2020-01-01")

    text = ProgressLedger.new(mode="test").digest()
    assert ("episodic candidate: user.dog.name (3 runs since "
            f"{today}) -> /promote user.dog.name") in text
    # Sweep ran: the 2020 chain evaporated from the persisted store.
    assert {f.key for f in EpisodicStore(path=tmp_path / "episodic.json").facts} == {"user.dog.name"}


def test_distiller_prompt_routes_ephemerals():
    from silica.kernel.prep_delegation import render_prompt

    prompt = render_prompt(target="Notes", source_text="some english text")
    assert '"ephemerals"' in prompt
    assert "user.dog.name" in prompt  # the canonical key example
    assert "entity.attribute" in prompt


def test_normalize_key_merges_morphological_variants():
    from silica.kernel.recall.episodic import normalize_key

    assert normalize_key("model_kits.gifts") == normalize_key("model_kit.gift")
    assert normalize_key("User.Car.Model") == "user.car.model"
    assert normalize_key("user.cities") == normalize_key("user.city")
    # snowball, not naive strip-s: these survive intact
    assert normalize_key("user.status") == "user.status"
    assert normalize_key("user.address") == "user.address"
    # dots stay segment separators, underscores stay token separators
    assert normalize_key("model_kits.last_project") == "model_kit.last_project"


def test_normalize_key_folds_change_marker_tokens():
    # LoCoMo smoke 2026-07-18: models bake the CHANGE into the key
    # ("aspiration_reinforced") despite the prompt's key-discipline block.
    # Mechanical lever: change-marker tokens fold away at lookup time so the
    # variant MATCHES the clean head. Stored spelling is never rewritten.
    from silica.kernel.recall.episodic import normalize_key

    assert (normalize_key("caroline.counseling.aspiration_reinforced")
            == normalize_key("caroline.counseling.aspiration"))
    assert normalize_key("user.job_update") == normalize_key("user.job")
    assert normalize_key("user.job_updated") == normalize_key("user.job")
    assert normalize_key("sam.trip.new") == normalize_key("sam.trip")
    assert normalize_key("user.diet.changed") == normalize_key("user.diet")
    assert normalize_key("user.plan.v2") == normalize_key("user.plan")
    # a key that is nothing but markers must not normalize to empty
    assert normalize_key("new") != ""


def test_normalize_key_folds_change_markers_per_language():
    # A store frozen non-english decorates keys in its own language
    # (`utente.lavoro_aggiornato`); the marker set must fold per language or
    # every decorated key starts a parallel chain (the key-collision defect,
    # mirrored: here the chain SPLITS instead of burying).
    from silica.kernel.recall.episodic import normalize_key

    cases = [
        ("danish",     "bruger.job_opdateret",           "bruger.job"),
        ("dutch",      "gebruiker.baan_bijgewerkt",      "gebruiker.baan"),
        ("dutch",      "gebruiker.nieuwe_baan",          "gebruiker.baan"),
        ("finnish",    "käyttäjä.työ_päivitetty",        "käyttäjä.työ"),
        ("french",     "utilisateur.travail_modifié",    "utilisateur.travail"),
        ("german",     "benutzer.job_aktualisiert",      "benutzer.job"),
        ("hungarian",  "felhasználó.munka_frissített",   "felhasználó.munka"),
        ("italian",    "utente.lavoro_aggiornato",       "utente.lavoro"),
        ("norwegian",  "bruker.jobb_oppdatert",          "bruker.jobb"),
        ("portuguese", "usuário.trabalho_atualizado",    "usuário.trabalho"),
        ("romanian",   "utilizator.loc_actualizat",      "utilizator.loc"),
        ("romanian",   "utilizator.adresa_nouă",         "utilizator.adresa"),
        ("russian",    "пользователь.работа_обновлена",  "пользователь.работа"),
        ("spanish",    "usuario.trabajo_actualizado",    "usuario.trabajo"),
        ("swedish",    "användare.jobb_uppdaterad",      "användare.jobb"),
    ]
    for lang, decorated, clean in cases:
        assert normalize_key(decorated, lang=lang) == normalize_key(clean, lang=lang), \
            f"{lang}: {decorated} must fold to {clean}"
    # all-marker keys must not normalize to empty in any language
    assert normalize_key("nuovo", lang="italian") != ""
    assert normalize_key("новый", lang="russian") != ""


def test_normalize_key_language_markers_never_eat_real_nouns():
    # The dangerous direction: a marker stem that collides with a common
    # attribute noun folds that noun out of every key (nieuws = news, not
    # "new"; nytta/nytte = benefit; nouvelles = news). These must survive.
    from silica.kernel.recall.episodic import normalize_key

    cases = [
        ("english",   "user.news_source",          "user.source"),
        ("dutch",     "gebruiker.nieuws_bron",     "gebruiker.bron"),
        ("french",    "utilisateur.nouvelles_pref", "utilisateur.pref"),
        ("swedish",   "användare.nytta_poäng",     "användare.poäng"),
        ("norwegian", "bruker.nyttig_info",        "bruker.info"),
        ("danish",    "bruger.nytte_score",        "bruger.score"),
    ]
    for lang, key, over_folded in cases:
        assert normalize_key(key, lang=lang) != normalize_key(over_folded, lang=lang), \
            f"{lang}: the first token of {key} must survive folding"


def test_marker_stems_english_is_byte_identical_to_the_old_set():
    # The word list replaced a hand-kept stem set; english identity is pinned
    # so existing english stores keep their exact supersede-chain matching.
    from silica.kernel.recall.episodic import _marker_stems

    assert _marker_stems("english") == frozenset(
        {"reinforc", "reaffirm", "updat", "new", "chang"})


def test_marker_stems_uncovered_language_falls_back_to_english_words():
    # An uncovered snowball language stems the ENGLISH word list with its own
    # stemmer — marker and key token go through the same stemmer, so a store
    # whose model decorates in english still folds.
    from silica.kernel.recall.episodic import normalize_key

    assert (normalize_key("user.job_updated", lang="estonian")
            == normalize_key("user.job", lang="estonian"))


def test_capture_change_marker_variant_supersedes_clean_head(tmp_path):
    # aspiration_reinforced arriving after aspiration must extend the SAME
    # chain, not open a parallel one.
    store = _store(tmp_path)
    store.capture([{"key": "elena.counseling.aspiration",
                    "text": "Elena wants to become a counselor"}],
                  run_id="s1", seen="2026-05-01")
    store.capture([{"key": "elena.counseling.aspiration_reinforced",
                    "text": "Elena reaffirmed her counseling aspiration"}],
                  run_id="s4", seen="2026-05-20")

    live = store.live_facts()
    assert len(live) == 1
    head = live[0]
    assert head.text == "Elena reaffirmed her counseling aspiration"
    assert head.supersedes is not None  # chained, not parallel


def test_normalize_key_idempotent():
    from silica.kernel.recall.episodic import normalize_key

    for k in ("model_kits.gifts", "user.preferences.color", "user.cities",
              "assistant.recipe.oven_temp"):
        once = normalize_key(k)
        assert normalize_key(once) == once


def test_capture_links_chain_across_plural_key_variants(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "model_kit.project", "text": "Working on a Spitfire kit"}],
                  run_id="run_1", seen="2026-06-01")
    store.capture([{"key": "model_kits.projects", "text": "Now building a B-29 kit"}],
                  run_id="run_2", seen="2026-06-20")

    live = store.live_facts()
    assert len(live) == 1
    head = live[0]
    assert head.key == "model_kits.projects"   # raw key stored as emitted
    assert head.text == "Now building a B-29 kit"
    old = next(f for f in store.facts if f.id == head.supersedes)
    assert old.key == "model_kit.project" and old.status == "superseded"


def test_capture_reinforces_across_key_variants_when_text_matches(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "model_kit.project", "text": "Building a Spitfire"}],
                  run_id="run_1", seen="2026-06-01")
    store.capture([{"key": "model_kits.project", "text": "building a spitfire!"}],
                  run_id="run_2", seen="2026-06-10")

    live = store.live_facts()
    assert len(live) == 1
    assert live[0].key == "model_kit.project"   # first spelling kept
    assert live[0].runs == ["run_1", "run_2"]
    assert live[0].last_seen == "2026-06-10"


def test_capture_matches_legacy_head_written_before_layer_a(tmp_path):
    # A store written before normalization existed: raw plural head on disk.
    store = _store(tmp_path)
    store.facts.append(Fact(id="f_0001", key="model_kits.gifts",
                            text="Got a B-29 kit", first_seen="2026-05-01",
                            last_seen="2026-05-01", runs=["run_0"]))
    store.next_id = 2
    store.save()

    store.capture([{"key": "model_kit.gifts", "text": "Got a Camaro kit"}],
                  run_id="run_1", seen="2026-06-01")
    live = store.live_facts()
    assert len(live) == 1
    assert live[0].supersedes == "f_0001"


def test_key_vocabulary_lists_live_heads_by_recency_with_cap(tmp_path):
    from silica.kernel.recall.episodic import key_vocabulary

    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Tom"}],
                  run_id="r1", seen="2026-01-01")
    store.capture([{"key": "user.car.model", "text": "Panda"}],
                  run_id="r2", seen="2026-03-01")
    # Supersede user.dog.name: only the head key surfaces, once.
    store.capture([{"key": "user.dog.name", "text": "Rex"}],
                  run_id="r3", seen="2026-04-01")

    assert key_vocabulary(store) == ["user.dog.name", "user.car.model"]
    assert key_vocabulary(store, cap=1) == ["user.dog.name"]


def test_key_vocabulary_section_renders_or_abstains(tmp_path):
    from silica.kernel.recall.episodic import key_vocabulary_section

    store = _store(tmp_path)
    assert key_vocabulary_section(store) is None   # empty store: no section

    store.capture([{"key": "user.car.model", "text": "Panda"}],
                  run_id="r1", seen="2026-01-01")
    section = key_vocabulary_section(store)
    assert section is not None
    assert section.startswith("## Episodic keys")
    assert "user.car.model" in section


def test_key_tokens_stemmed_entity_prefix_dropped():
    from silica.kernel.recall.episodic import key_tokens

    # The probe-proven KU pair shares exactly two stemmed tokens.
    shared = (key_tokens("user.fitness.tournament.date")
              & key_tokens("user.tennis_tournament_date"))
    assert len(shared) == 2
    # Entity prefixes are dropped, never tokens.
    assert not key_tokens("user.laundry.schedule") & {"user", "assist"}
    # Morphological variants and user/assistant prefixes merge.
    assert key_tokens("assistant.laundry.tips") == key_tokens("user.laundry.tip")
    # Single-char tokens are noise, not alphabet.
    assert key_tokens("user.a_b.c") == set()


# ---------------------------------------------------------------------------
# Key schema enforcement (ADR-0021): declared grammar, fold never reject
# ---------------------------------------------------------------------------


def _schema(**kw):
    from silica.kernel.vault_manifest import EpisodicKeySchema

    return EpisodicKeySchema(**kw)


def test_enforce_compliant_key_passes_through():
    from silica.kernel.recall.episodic import enforce_key_schema

    assert enforce_key_schema("user.dog.name", _schema()) == "user.dog.name"


def test_enforce_unknown_prefix_folds_under_default():
    from silica.kernel.recall.episodic import enforce_key_schema

    assert enforce_key_schema("dog.name", _schema()) == "user.dog.name"


def test_enforce_depth_folds_tail_segments():
    from silica.kernel.recall.episodic import enforce_key_schema

    assert (enforce_key_schema("assistant.recipe.oven.temp", _schema())
            == "assistant.recipe.oven_temp")


def test_enforce_prefix_match_is_canonical_stored_form_untouched():
    from silica.kernel.recall.episodic import enforce_key_schema

    # "assistants" stems to the same canonical form as "assistant": it IS a
    # known prefix, and the stored spelling is never rewritten.
    assert enforce_key_schema("assistants.diet", _schema()) == "assistants.diet"


def test_enforce_prepend_then_depth_fold_compose():
    from silica.kernel.recall.episodic import enforce_key_schema

    # Unknown prefix adds a segment, so the tail folds to honor max_depth.
    assert (enforce_key_schema("weather.city.today", _schema())
            == "user.weather.city_today")


def test_enforce_is_idempotent():
    from silica.kernel.recall.episodic import enforce_key_schema

    schema = _schema()
    for raw in ("dog.name", "assistant.recipe.oven.temp", "weather.city.today",
                "user.dog.name"):
        once = enforce_key_schema(raw, schema)
        assert enforce_key_schema(once, schema) == once


def test_enforce_drops_empty_segments():
    from silica.kernel.recall.episodic import enforce_key_schema

    assert enforce_key_schema("user..dog.name", _schema()) == "user.dog.name"


def test_enforce_honors_custom_schema_values():
    from silica.kernel.recall.episodic import enforce_key_schema

    schema = _schema(prefixes=("team",), default_prefix="team", max_depth=2)
    assert enforce_key_schema("velocity.sprint", schema) == "team.velocity_sprint"
    assert enforce_key_schema("team.velocity", schema) == "team.velocity"


def test_capture_with_schema_stores_enforced_key(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_1", seen="2026-07-16", schema=_schema())
    facts = store.live_facts()
    assert len(facts) == 1
    assert facts[0].key == "user.dog.name"


def test_capture_without_schema_stores_raw_key(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "assistant.recipe.oven.temp", "text": "180 gradi"}],
                  run_id="run_1", seen="2026-07-16")
    assert store.live_facts()[0].key == "assistant.recipe.oven.temp"


def test_capture_from_distill_loads_schema_from_memory_vault(tmp_path, monkeypatch):
    from silica.kernel.recall import episodic

    vault = tmp_path / "memvault"
    vault.mkdir()
    (vault / "vault.yaml").write_text(
        "conventions:\n  episodic_keys: {}\n", encoding="utf-8")
    monkeypatch.setattr(episodic, "episodic_home", lambda: vault)
    monkeypatch.setattr(episodic, "store_path", lambda: tmp_path / "episodic.json")

    result = {"ephemerals": [{"key": "dog.name", "text": "Si chiama Tom"}]}
    episodic.capture_from_distill(result, run_id="run_1", seen="2026-07-16")
    (f,) = EpisodicStore(path=tmp_path / "episodic.json").live_facts()
    assert f.key == "user.dog.name"


def test_capture_from_distill_without_schema_block_keeps_raw_key(tmp_path, monkeypatch):
    from silica.kernel.recall import episodic

    vault = tmp_path / "memvault"
    vault.mkdir()  # no vault.yaml at all
    monkeypatch.setattr(episodic, "episodic_home", lambda: vault)
    monkeypatch.setattr(episodic, "store_path", lambda: tmp_path / "episodic.json")

    result = {"ephemerals": [{"key": "dog.name", "text": "Si chiama Tom"}]}
    episodic.capture_from_distill(result, run_id="run_1", seen="2026-07-16")
    (f,) = EpisodicStore(path=tmp_path / "episodic.json").live_facts()
    assert f.key == "dog.name"


def test_capture_schema_converges_raw_variant_onto_existing_chain(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_1", seen="2026-07-01", schema=_schema())
    # Same fact arrives with a prefixless key: enforcement folds it onto the
    # same chain instead of starting a parallel one.
    store.capture([{"key": "dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_2", seen="2026-07-16", schema=_schema())
    facts = store.live_facts()
    assert len(facts) == 1
    assert facts[0].runs == ["run_1", "run_2"]



def test_store_freezes_key_language_and_merges_non_english_variants(tmp_path):
    """The defect this fixes: an Italian store stemmed as English left
    `modello`/`modelli` distinct, so the supersede chain split and recall
    returned both values of one attribute."""
    store = _store(tmp_path)
    store.capture([{"key": "utente.auto.modello", "text": "La mia auto è una Panda"}],
                  run_id="run_1", seen="2026-07-14")
    assert store.lang == "italian"

    store.capture([{"key": "utente.auto.modelli", "text": "Ho cambiato, ora è una Punto"}],
                  run_id="run_2", seen="2026-07-15")

    live = store.live_facts()
    assert len(live) == 1, [f.key for f in live]
    assert live[0].text == "Ho cambiato, ora è una Punto"
    assert live[0].supersedes is not None

    # frozen on disk, so lookup identity cannot drift as the store grows
    from silica.kernel.recall.episodic import _unpack_store

    doc = _unpack_store((tmp_path / "episodic.json").read_bytes())
    assert doc["lang"] == "italian"
    assert EpisodicStore(path=tmp_path / "episodic.json").lang == "italian"


def test_english_store_keeps_its_identity(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.car.model", "text": "My car is a Panda, a small city car"}],
                  run_id="run_1", seen="2026-07-14")
    assert store.lang == "english"
    store.capture([{"key": "user.car.models", "text": "I switched, it is a Punto now"}],
                  run_id="run_2", seen="2026-07-15")
    assert len(store.live_facts()) == 1


def test_frozen_language_survives_a_later_switch(tmp_path):
    """Re-detecting per capture would re-partition live chains mid-life."""
    store = _store(tmp_path)
    store.capture([{"key": "user.car.model", "text": "My car is a small red city car"}],
                  run_id="run_1", seen="2026-07-14")
    store.capture([{"key": "utente.cane.nome", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_2", seen="2026-07-15")
    assert store.lang == "english"
