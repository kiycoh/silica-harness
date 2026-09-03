"""Shared pytest fixtures for the silica-harness test suite."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _fresh_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the global BUS singleton for every test to prevent cross-test contamination."""
    import silica.agent.bus as bus_mod
    monkeypatch.setattr(bus_mod, "BUS", bus_mod.EventBus())


@pytest.fixture(autouse=True)
def _fresh_narrator(monkeypatch: pytest.MonkeyPatch):
    """Fresh NARRATOR per test, same contract as _fresh_bus: the singleton
    holds an open file handle + flock, so a leaked one would pin a tmp dir
    (and a beat from one test could land in another's session)."""
    import silica.agent.narration as narration_mod
    fresh = narration_mod.Narrator()
    monkeypatch.setattr(narration_mod, "NARRATOR", fresh)
    yield fresh
    fresh.close()


@pytest.fixture(autouse=True)
def _no_form_sniff_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The form sniff must never reach a real endpoint from a test.

    Raising makes sniff_form degrade to "" (its designed offline posture);
    tests that want a specific verdict monkeypatch forms.sniff_form or
    forms.call_llm themselves, which overrides this stub. The memo is cleared
    so one test's verdict can never leak into another's."""
    import silica.kernel.forms as forms

    forms._sniff_memo.clear()

    def _offline(*a, **k):
        raise RuntimeError("form sniff disabled in tests")

    monkeypatch.setattr(forms, "call_llm", _offline)

    # Same posture for the residue verification: never a real endpoint from a
    # test. Its designed offline degrade is [] (CLEANUP proceeds); residue
    # tests monkeypatch finalize.residue_facts themselves, overriding this.
    # Both dispatch seams are stubbed for the same reason (decompose at chunk
    # attach, evidence+judge at the last chunk's WRITE — each submits LLM
    # calls to a pool); test_residue_dispatch binds the real implementations
    # at import time, before this fixture patches the attrs.
    import silica.router.states.finalize as _finalize

    monkeypatch.setattr(_finalize, "residue_facts", lambda *a, **k: [])
    monkeypatch.setattr(_finalize, "maybe_dispatch_residue_check", lambda *a, **k: None)
    monkeypatch.setattr(_finalize, "maybe_dispatch_residue_decompose", lambda *a, **k: None)
    # The gate's late-dispatch fallback submits the whole verification to a
    # pool; without this stub every FSM cleanup test would hit the network.
    monkeypatch.setattr(_finalize, "_verify_now",
                        lambda *a, **k: {"missing": [], "total": 0, "judged": 0,
                                         "failures": 0, "off_theme": 0})


@pytest.fixture(autouse=True)
def _restore_tools_registry() -> None:
    """Undo any registration a test leaves in the global TOOLS dict.

    TOOLS is module-level and `@tool` writes into it at import time; ten test
    files register into it by hand. That leak is the first mechanism behind the
    2026-08-05 suite flakiness (a real eager tool registered by one test broke
    another's exact-set assertion under pytest-randomly), and it was closed per
    test with try/finally. Per test is convention, so the eleventh author
    reintroduces it. Snapshot and restore, never clear: registration happens at
    import, so clearing would empty the registry for the rest of the session.

    Restoring the snapshot wholesale is wrong for the same reason. The first
    test to import a tool module registers it DURING the test, and the import
    is cached, so deleting what it added unregisters those tools for the whole
    session — that is how `silica_code_pack` went missing from
    `exposed_tools(all_tools=True)` two tests after `test_mcp_surface` imported
    it. So an addition is kept when it came from a `silica.*` module (the
    product registry filling in lazily) and dropped otherwise (a fake defined
    in a test).

    ponytail: this hides a leaking test instead of failing it. Stability was
    chosen over catching the author; assert the registry is unchanged here if
    that turns out to be the wrong trade.
    """
    from silica.tools import TOOLS
    snapshot = dict(TOOLS)
    yield
    TOOLS.update(snapshot)  # undo overwrites and deletions
    for name in set(TOOLS) - set(snapshot):
        if not getattr(TOOLS[name].fn, "__module__", "").startswith("silica"):
            del TOOLS[name]


@pytest.fixture(autouse=True)
def _reset_run_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the process-wide 429 pacing floor so it can't leak between tests
    (a leaked cooldown would make a later real retry sleep for seconds)."""
    import silica.agent.llm as llm_mod
    monkeypatch.setattr(llm_mod, "_run_cooldown", 0.0)


@pytest.fixture(autouse=True)
def _keyphrase_lane_under_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FSM suites mock silica_recon/silica_payload and count their calls:
    they test the keyphrase pipeline. The outline lane (the default since
    2026-09-02) bypasses both, so it is opted into per test
    (tests/test_outline_lane.py) instead of silently emptying these mocks."""
    monkeypatch.setattr("silica.config.CONFIG.nucleate_lane", "keyphrase")


@pytest.fixture(autouse=True)
def _no_recon_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable silica_recon's network embedder by default: recon falls back to the
    deterministic mined rank. Keeps the suite fast and offline; the rerank path is
    covered by test_keyphrase (FakeEmbedder) and the SILICA_EVAL golden eval."""
    import silica.tools.pipeline as pipe_mod
    monkeypatch.setattr(pipe_mod, "_recon_embedder", lambda: None)


@pytest.fixture(autouse=True)
def _disable_index_sweep(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the invocation-time index sweep and sandbox its stamp sidecar.

    facade_retrieve and the injector sweep the vault's derived indexes
    (kernel/recall/sync.py) before reading them; in tests that would stat
    stub drivers and write the developer's real ~/.silica/index stamp file.
    Tests that exercise the sweep re-enable CONFIG.index_sweep explicitly
    (tests/test_index_sync.py)."""
    monkeypatch.setattr("silica.config.CONFIG.index_sweep", False)
    import silica.kernel.recall.sync as sync_mod
    monkeypatch.setattr(sync_mod, "_stamps_path", lambda: tmp_path / "sync_stamps.json")
    monkeypatch.setattr(sync_mod, "_last_sweep", 0.0)


@pytest.fixture(autouse=True)
def _isolate_embed_legacy_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard against the real ~/.silica/index/embeddings.json leaking into tests
    via the legacy-migration fallback. Any test that redirects _index_path to a
    non-existent tmp file would otherwise fall back to the developer's real index."""
    import silica.kernel.recall.embed as embed_mod
    monkeypatch.setattr(embed_mod, "_LEGACY_INDEX_PATH", tmp_path / "legacy_embed.json")


@pytest.fixture(autouse=True)
def _isolate_cooccurrence_index(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the default co-occurrence index to a per-test tmp path.

    The post-write freshness hook refreshes the co-occurrence index with no
    embedder gate (it is the embedder-free stable leg), so any test that drives
    the write handler would otherwise write the user's real
    ~/.silica/index/cooccurrence.json. Tests that need a store pass an explicit
    path; this only redirects the default.
    """
    import silica.kernel.recall.cooccurrence as cooc_mod
    monkeypatch.setattr(cooc_mod, "_index_path", lambda: tmp_path / "cooccurrence_index.json")


@pytest.fixture(autouse=True)
def _isolate_silica_home(tmp_path_factory, monkeypatch: pytest.MonkeyPatch):
    """Sandbox every ~/.silica default under the test's tmp dir.

    The ledger/undo-journal/checkpoint singletons are first-caller-wins (the
    path argument is ignored once initialised) and default to the developer's
    real ~/.silica/*.db; paths._SILICA_HOME feeds tmp/, index/ and session
    capture. An unpatched caller (orchestrator record, CLEANUP) would write
    real state, pin its store for the rest of the process, and let concurrent
    suite runs interfere with each other through the shared files.

    Outside tmp_path on purpose: path_lease drops its flock files under
    _SILICA_HOME, and tests assert over tmp_path's exact contents.
    """
    home = tmp_path_factory.mktemp("silica-home")
    import silica.kernel.recall.paths as paths_mod
    import silica.kernel.write.checkpoints as cp_mod
    import silica.kernel.write.ledger as ledger_mod
    import silica.kernel.write.undo_journal as journal_mod

    monkeypatch.setattr(paths_mod, "_SILICA_HOME", home)
    monkeypatch.setattr(ledger_mod, "_DEFAULT_LEDGER_PATH", home / "ledger.db")
    monkeypatch.setattr(journal_mod, "_DEFAULT_JOURNAL_PATH", home / "undo_journal.db")
    monkeypatch.setattr(cp_mod, "_DEFAULT_CHECKPOINT_PATH", home / "checkpoints.db")
    singletons = ((ledger_mod, "_ledger"), (journal_mod, "_store"), (cp_mod, "_store"))
    for mod, attr in singletons:
        monkeypatch.setattr(mod, attr, None)
    yield
    # Close whatever the test created before monkeypatch restores the attrs,
    # so per-test sqlite files don't accumulate open fds across the run.
    for mod, attr in singletons:
        store = getattr(mod, attr)
        if store is not None:
            try:
                store.close()
            except Exception:
                pass


@pytest.fixture(autouse=True)
def _isolate_distill_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the distiller reply cache to a per-test tmp path.

    Any test driving run_distiller with the cache armed would otherwise write
    the developer's real ~/.silica/cache/distill, and a stored reply would
    then replay into an unrelated test on the next run. The flag is cleared
    suite-wide too, so a developer's .env cannot decide whether a test that
    distills twice sees one call or two; tests that want the cache set it.
    """
    import silica.kernel.distill_cache as cache_mod
    monkeypatch.delenv("SILICA_DISTILL_CACHE", raising=False)
    monkeypatch.setattr(cache_mod, "cache_root", lambda: tmp_path / "distill_cache")


@pytest.fixture(autouse=True)
def _isolate_episodic_store(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the default episodic store to a per-test tmp path.

    progress.digest() sweeps the episodic store and the distill state captures
    into it; without this, any test driving those paths would read/write the
    developer's real ~/.silica/index/<digest>/episodic.json. Tests that need a
    store pass an explicit path; this only redirects the default.
    """
    import silica.kernel.recall.episodic as ep_mod
    monkeypatch.setattr(ep_mod, "store_path", lambda: tmp_path / "episodic_default.json")


@pytest.fixture(autouse=True)
def _isolate_contested_register(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the contested-note register to a per-test tmp path.

    silica_flag_note and progress.digest() read/write index_dir()/
    contested_register.json; without this a test driving the flag tool would
    leave the developer's real ~/.silica/index littered.
    """
    import silica.kernel.contested_register as reg_mod
    monkeypatch.setattr(reg_mod, "_register_path",
                        lambda: tmp_path / "contested_register.json")


@pytest.fixture(autouse=True)
def _isolate_cluster_ctx_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the vault-cluster ctx cache (Scaling E) to a per-test tmp path.

    build_vault_graph_ctx persists the cluster ctx under index_dir(); a test that
    runs it without isolating the vault would otherwise write into the developer's
    real ~/.silica index AND a cache from one test could leak into the next. Per
    tmp_path keeps each test's cache private and out of the real index.
    """
    import silica.kernel.recall.graph_export as ge_mod
    monkeypatch.setattr(
        ge_mod, "cluster_ctx_path", lambda: tmp_path / "clusters_ctx.json"
    )


@pytest.fixture(autouse=True)
def _isolate_deferred_store(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the deferred review queue to a per-test tmp path.

    The pipeline defers ops through get_deferred_store() with no explicit path;
    before this fixture existed, every FSM test that hit a defer path wrote its
    fixtures into the developer's real global store (the 221 «lint failed:
    ['e']» bundles). Also points the legacy migration source at an empty tmp
    dir so the one-shot adoption never reads the real ~/.silica/deferred.
    """
    import silica.kernel.recall.deferred as deferred_mod
    monkeypatch.setattr(deferred_mod, "_store_dir", lambda: tmp_path / "deferred_store")
    monkeypatch.setattr(deferred_mod, "_LEGACY_DEFERRED_DIR", tmp_path / "deferred_legacy")
    deferred_mod._stores.clear()
    yield
    deferred_mod._stores.clear()


@pytest.fixture(autouse=True)
def _isolate_undo_journal(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the undo journal to a per-test tmp DB.

    run_subagent_batch and the organizer FSM now open a journal run with no
    explicit path; without this every batch/organize test would write into the
    developer's real ~/.silica/undo_journal.db.
    """
    # Lazy, like the other store fixtures: only redirect the DEFAULT path and
    # reset the singleton — the DB (and its dir + WAL sidecars) is created only
    # when a test actually opens the journal, so tests that assert on tmp_path's
    # contents never see a stray undo_journal/ dir.
    import silica.kernel.write.undo_journal as uj
    monkeypatch.setattr(uj, "_DEFAULT_JOURNAL_PATH", tmp_path / "undo_journal" / "j.db")
    uj._store = None
    yield
    uj._store = None


@pytest.fixture(autouse=True)
def _clear_store_singletons() -> None:
    """Reset the cached store singletons (Fix 3 seam) around every test.

    `get_store`/`get_cooccur_store` keep a process-lifetime instance keyed by
    index path; without this, an instance built under one test's monkeypatched
    `_index_path` would leak into the next. Clear before AND after to also drop
    state seeded by import-time or session-scoped fixtures.
    """
    import silica.kernel.recall.embed as embed_mod
    import silica.kernel.recall.cooccurrence as cooc_mod
    embed_mod.clear()
    cooc_mod.clear()
    yield
    embed_mod.clear()
    cooc_mod.clear()


@pytest.fixture(autouse=True)
def _reset_overlay_cache() -> None:
    """Reset the module-level overlay cache before every test.

    Prevents a test that calls get_active_overlay() (or monkeypatches the vault
    path) from polluting the cached result seen by subsequent tests.
    """
    import silica.kernel.text.overlay as overlay_mod
    overlay_mod.reset_overlay_cache()


@pytest.fixture(autouse=True)
def _reset_manifest_cache() -> None:
    """Reset the module-level vault-manifest cache before every test.

    Mirrors `_reset_overlay_cache`: `ofm.ofm_lint` and `prep_delegation.render_prompt`
    now resolve `conventions:` from `get_active_manifest()`, so a test that sets
    CONFIG.vault_path (e.g. via the `tmp_vault` fixture) would otherwise leak a
    cached manifest — with its vault.yaml-derived conventions — into whichever
    test runs next in the same process.
    """
    import silica.kernel.vault_manifest as manifest_mod
    manifest_mod.reset_manifest_cache()


@pytest.fixture(autouse=True)
def _isolate_vault_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point CONFIG.vault_path at a per-test tmp dir by default.

    At import CONFIG.vault_path is the developer's REAL configured vault (from
    .env). run_log.append_log_line falls back to it whenever no explicit
    vault_path is passed, so any unit test that drove a CLEANUP (or the curator)
    leaked `nucleate test.md → 0 new…` lines into the real log.md. This runs
    before requested fixtures, so tmp_vault and in-test monkeypatches still win
    for tests that need a specific vault.
    """
    import silica.config as config_mod
    # Lazy, like the other store fixtures: don't create the dir (a bare
    # tmp_path/… would pollute tests that assert on tmp_path's contents). Any
    # code that actually writes here (append_log_line) creates it on demand.
    safe_vault = tmp_path / "isolated_vault"
    monkeypatch.setattr(config_mod.CONFIG, "vault_path", str(safe_vault))


@pytest.fixture(scope="session")
def synthetic_vault() -> Path:
    """Return the path to the synthetic test vault, building it if needed.

    Session-scoped: built exactly once per pytest run.
    Location: tests/fixtures/synthetic_vault/ (or SILICA_TEST_VAULT env var).
    """
    from tests.fixtures.vault_factory import build_synthetic_vault, _resolve_root
    return build_synthetic_vault(_resolve_root())


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    """Provide a temporary filesystem-backed vault for unit tests.

    Returns a helper with:
      .note(rel, content="") -> str   — create a note, return absolute path
      .read(path) -> str              — read note at absolute path
      .write(path, content)           — overwrite note at absolute path
    """
    import silica.config
    import silica.driver

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    monkeypatch.setattr(silica.config.CONFIG, "vault_path", str(vault_dir))
    silica.driver._driver = None  # reset lazy singleton

    class _VaultHelper:
        def note(self, rel: str, content: str = "") -> str:
            p = vault_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return str(p)

        def read(self, path: str) -> str:
            from pathlib import Path as _Path
            return _Path(path).read_text(encoding="utf-8")

        def write(self, path: str, content: str) -> None:
            from pathlib import Path as _Path
            _Path(path).write_text(content, encoding="utf-8")

    yield _VaultHelper()
    silica.driver._driver = None  # reset after test


@pytest.fixture(autouse=True)
def _isolate_stale_snapshot(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the stale-snapshot cache to a per-test tmp path, keyed on vault.

    Intent is for read_warning/stale_count (future tasks) to warm this
    snapshot on any documents: note; without this fixture, any such test
    would write the developer's real ~/.silica index. Keying the stub on
    `vault` (not one fixed filename) matters because two vaults can share a
    single repo root — e.g. a codebase-mode `<repo>/.silica` vault and a
    docs vault both resolve to the same HEAD — so a fixed filename would let
    snapshot(vault_a) leak into peek(vault_b) as a false green. The cache
    lives in a subdirectory so it can never collide with a vault rooted at
    tmp_path itself.
    """
    import hashlib
    from silica.kernel.code import codedocs
    monkeypatch.setattr(
        codedocs, "_snapshot_path",
        lambda vault: (tmp_path / "stale-cache"
                       / (hashlib.sha1(str(vault).encode()).hexdigest()[:8] + ".json")),
    )
