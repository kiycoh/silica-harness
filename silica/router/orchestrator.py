# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""L3 Router / Orchestrator for Silica — Injector FSM (S2.3 complete).

From SILICA.md §3 L3 & §7.3:
  Deterministic state machine for the Injector pipeline.
  Gates: >= 10% rejection rate -> abort + rollback.

Contracts applied (see silica_architecture_addendum.md):
  C1 — ops_path carries list[Op]-compatible dicts after VALIDATE.
  C2 — freshness via per-op postconditions in CLI backend.
  C3 — build_txn() builds InverseOp entries; ROLLBACK applies them.
  C4 — VALIDATE overwrites ops_path; SNAPSHOT/WRITE read that same file.
  C5 — ledger records ops; CLEANUP only reachable from DONE state.

S2.3 change: DELEGATE calls the real Distiller LLM via prep_delegation.
S2.3 change: SNAPSHOT uses build_txn() directly (no _txn_obj leak).
S2.3 change: ledger.py integrated (CLEANUP writes 'committed', ROLLBACK marks 'rolled_back').
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import contextmanager
from enum import Enum, auto
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor

    from silica.driver.base import Txn, GraphSnapshot
    from silica.driver.base import NoteRef
    from silica.kernel.write.undo_journal import InverseOp
    from silica.router.prefetch import DistillPrefetcher

from silica.driver import DRIVER
from silica.config import CONFIG
from silica.tools.composed import (
    silica_lint,
    silica_payload,
    silica_recon,
    silica_sanitize,
    silica_validate_ops,
)
from silica.kernel.write.ops import OpType
from silica.kernel.write.ops_io import load_ops
from silica.kernel.recall.paths import to_vault_relative
from silica.router.base_fsm import BaseFSM
# Imported for the states modules (and tests), which resolve patchable
# collaborators through this module's namespace — see silica.router.states.
from silica.router import states

logger = logging.getLogger(__name__)


# Recipe phases that run once per inbox file, before that file has any chunks.
# Everything else in injector.yaml is per-chunk, except rollback (on_gate_fail).
# Mirrored for display by _FILE_PHASES/_CHUNK_PHASES in silica/ui/renderer.py;
# tests/test_phase_track.py pins both against the recipe.
_FILE_SCOPE_PHASES = frozenset({"recon", "payload", "salience"})


def _refresh_cooccurrence_for_ops(
    ops: list,
    committed_paths: set,
    *,
    read_body: Callable[[str], str],
    lang: str = "english",
    store: Any | None = None,
    save: bool = True,
) -> int:
    """Refresh the embedder-free co-occurrence index for committed write/patch ops.

    The freshness twin of the embedding refresh, but the STABLE leg: it imports
    only the cooccurrence module (never the embedder/provider stack), so the
    index stays fresh even when LM Studio is down. Uses build_index(force=True)
    so a note's prior contribution is replaced, never inflated, with a single
    save — replacement semantics only: it deliberately does NOT pass
    refreeze=True, so the store's frozen stemming language is never re-detected
    from a write batch (re-detection is reserved for /cooccur --force).
    Note edges need no step here: they are derived from the contributions on
    read (CooccurStore.note_adjacency, ADR-0029). Best-effort: a per-note read
    failure is skipped and the whole call never raises. Returns the number of
    notes refreshed.
    """
    from silica.kernel.recall.cooccurrence import build_index as _cooccur_build

    notes: list[tuple[str, str, str]] = []
    concepts_by_path: dict[str, list[str]] = {}
    seen: set[str] = set()
    for op in ops:
        path = op.touched_ref()
        if op.op not in (OpType.write, OpType.patch) or not path:
            continue
        if path not in committed_paths or path in seen:
            continue
        seen.add(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        idx_path = path.removesuffix(".md")
        try:
            body = read_body(path) or ""
        except Exception:
            continue
        notes.append((idx_path, stem, body))
        # #9: forward LLM-extracted concept phrases to reinforce this note.
        op_concepts = getattr(op, "concepts", None)
        if op_concepts:
            concepts_by_path[idx_path] = op_concepts

    if not notes:
        return 0
    try:
        # save=False defers the single save to the end-of-run flush, which
        # rewrites the same singleton.
        built = _cooccur_build(notes, store=store, lang=lang, force=True,
                               concepts_by_path=concepts_by_path or None, save=False)
        if save:
            built.save()
    except Exception as exc:
        logger.debug("WRITE: cooccur refresh skipped (%s)", exc)
        return 0
    return len(notes)


def _commit_docs_for_ops(
    ops: list,
    committed_paths: set,
    *,
    vault: str,
    git_commit: str,
) -> str | None:
    """Commit touched vault paths for write/patch ops to git.

    Git safety net behind SILICA_GIT_COMMIT=auto: an additive snapshot on top
    of the undo journal (ADR-0002), never a replacement. Best-effort: no git
    binary, vault outside a repo, nothing staged, or any subprocess failure all
    yield None and never raise. Commits ONLY vault paths — the out-of-vault
    guard lives inside gitstate.commit_docs, so a bug cannot commit source files.
    """
    from silica.kernel.code import gitstate
    from pathlib import Path as _Path

    if git_commit != "auto" or not vault:
        return None

    seen: set[str] = set()
    abs_paths: list[_Path] = []
    for op in ops:
        path = op.touched_ref()
        if op.op not in (OpType.write, OpType.patch) or not path:
            continue
        if path not in committed_paths or path in seen:
            continue
        seen.add(path)
        abs_paths.append(_Path(vault) / path)

    if not abs_paths:
        return None

    try:
        from silica.kernel.recall.paths import repo_root_for

        root = repo_root_for(vault)
        if root is None:
            return None
        n = len(abs_paths)
        return gitstate.commit_docs(root, vault, abs_paths, f"silica: write {n} note(s)")
    except Exception as _ge:
        logger.debug("WRITE: git auto-commit skipped (%s)", _ge)
        return None


class InjectorState(Enum):
    INIT = auto()
    RECON = auto()         # Phase 1
    PAYLOAD = auto()       # Phase 2.0
    SALIENCE = auto()      # Phase 2.05 — thematic salience gate (drop off-theme concepts)
    COLLISION = auto()     # Phase 5 — dedup routing: high-sim→patch, borderline→defer, low→write
    DELEGATE = auto()      # Phase 2.1 — real Distiller LLM
    SANITIZE = auto()      # Phase 2.2
    VALIDATE = auto()      # Phase 2.3 (Gate) — C4: overwrites ops_path
    SNAPSHOT = auto()      # Phase 2.5 — C3: builds InverseOp Txn
    WRITE = auto()         # Phase 3
    HUB_UPDATE = auto()    # Phase 3.5 — patch Hub note with MOC links
    AUTOLINK = auto()      # Phase 4 — inject wikilinks into touched notes
    BACKLINK = auto()      # Phase 4.5 — reverse: inject links to new notes into pre-existing ones
    LINT = auto()          # Phase 5 (Gate)
    CLEANUP = auto()       # Phase 5 — C5: only from DONE
    ROLLBACK = auto()      # On gate fail — C3: apply inverses
    DONE = auto()
    ERROR = auto()


@contextmanager
def phase(fsm, task_id: str, capability_name: str):
    """Bracket a handler's happy path: 'running' on entry, then 'done' +
    _transition_success() on clean exit. An exception propagates WITHOUT
    emitting 'done'/transition (the caller's raise routes to ROLLBACK/error
    exactly as before). Fits only linear handlers with a single trailing
    success; handlers with early-exit transitions, a split done/transition,
    or ROLLBACK routing keep their explicit progress notes.

    A free function (not an FSM method) so it depends only on `_progress_note`
    and `_transition_success`: handler unit tests keep stubbing those two on a
    plain fake without needing to know about (or bind) the concrete FSM."""
    fsm._progress_note(task_id, capability_name, "running")
    yield
    fsm._progress_note(task_id, capability_name, "done")
    fsm._transition_success()


class InjectorFSM(BaseFSM[InjectorState]):
    """Deterministic state machine for the Injector pipeline (S2.3 complete)."""

    # The residue lane's state. It lives on the FSM because it spans files
    # (a decompose dispatched for file N+1 while N is still distilling) but it
    # is OWNED by silica.router.states.finalize, which is the only module that
    # writes these. Declared here rather than left to spring into existence on
    # first assignment: they are part of the FSM's surface, and a reader of this
    # class could not otherwise know the lane exists.
    _residue_executor: ThreadPoolExecutor | None = None
    _residue_decompose: dict[int, Future] | None = None
    _residue_ready: tuple[int, dict[str, Any]] | None = None
    _residue_future: tuple[int, list[Any], list[Future]] | None = None
    _residue_pending: list[tuple] | None = None
    # File indices already accounted in the run log — the success and failure
    # paths both conclude a file, and this keeps them from double-recording.
    _files_logged: set[int] | None = None
    # Ops the anneal recovered from the deferred store; read by Coordinator.
    _annealed_ops: int = 0
    # The distill lane's read-ahead worker, owned by states.distill.
    _prefetcher: DistillPrefetcher | None = None
    # One vault listing per run, memoized for the title checks in states.linking.
    _run_title_refs: list[NoteRef] | None = None

    def __init__(
        self,
        inbox_file: str = "",
        target_dir: str = "",
        hub: str | None = None,
        *,
        inbox_files: list[str] | None = None,
        resume_run_id: str | None = None,
        seen_override: str | None = None,
        keep_sources: bool = False,
        episodic_capture: bool = True,
        distill_profile: str | None = None,
    ):
        # Normalize to a list. inbox_files takes precedence; inbox_file is a
        # compat shim inserted at position 0 if not already present.
        files: list[str] = list(inbox_files or [])
        if inbox_file and inbox_file not in files:
            files.insert(0, inbox_file)
        if not files:
            raise ValueError("At least one inbox file must be provided")
        self.inbox_files: list[str] = [to_vault_relative(f) for f in files]
        self.inbox_file: str = self.inbox_files[0]  # first file; compat with single-file callers
        from silica.kernel.recall.paths import resolve_target_dir
        from silica.kernel.vault_manifest import in_write_dir
        # Case-fold against the REAL tree first (the mirror often has no folder
        # for this note yet), then rebase into the write boundary. Rebasing here
        # rather than in validate's local copy is what makes HUB_UPDATE and
        # anneal agree with the ops: both build `<target_dir>/<hub>.md`, and a
        # target_dir left outside the boundary sent them looking for a hub at a
        # path this run can never have written.
        target_dir = in_write_dir(resolve_target_dir(target_dir))
        self.target_dir = target_dir

        # Hub sanity check: if not specified, inherit the folder name of target_dir
        if not hub and target_dir:
            import os
            hub = os.path.basename(target_dir.rstrip("/\\"))
        self.hub = hub

        # Bench-only episodic clock: when set, capture_from_distill dates
        # facts with this ISO day instead of the ingest day (LoCoMo e2e leg).
        self.seen_override = seen_override

        # Verbatim source leaves (spec-harness-promotion §2): /ingest
        # --keep-sources. Conversation captures (seen_override set) always
        # leave a leaf — their source is ephemeral, otherwise lost.
        self.keep_sources = keep_sources

        # Off for /promote: that run distills a render of the episodic store,
        # so capturing its output back would nest the chain inside itself.
        self.episodic_capture = episodic_capture

        # Per-RUN distill profile, prompt and gate together (None = the
        # process-global resolution: SILICA_DISTILL_PROFILE > manifest).
        # /promote runs "extractive": a promotion stub is finished verbatim
        # content, and the default authoring lens + 275-char floor rejects
        # every honest distillation of it.
        self.distill_profile = distill_profile

        self.state = InjectorState.INIT
        self.context: dict[str, Any] = {}
        self._tmp_files: list[str] = []
        self._txn: Txn | None = None  # holds the live Txn object for ROLLBACK
        self._undo_run_id: str | None = None          # journal run for this inject
        self._run_inverses: list[tuple[str, InverseOp, str | None]] = []  # (path, inverse, post_hash)
        self._pre_graph: GraphSnapshot | None = None  # S3.2 pre-write graph snapshot

        # Optional producer channel to the leashed sub-agent pool.  Set by the
        # Coordinator; when None the FSM runs standalone (legacy behaviour) and
        # never produces work items.
        self.work_queue: Any | None = None

        # Optional run-scoped memory of non-blocking warnings (orphans).  Set by
        # the Coordinator; drained for repair at end of run.  None ⇒ no recording.
        self.warning_ledger: Any | None = None

        # Per-file content info — populated by run() before _run_loop starts
        self._file_canonicals: list[str] = []
        self._file_content_hashes: list[str] = []
        self._file_valid_from: list[str | None] = []  # source event clock; None → no stamp
        self._committed_file_indices: set[int] = set()  # indices of already-committed files

        # Iterative chunk processing state fields
        self._chunks: list[dict] = []
        # Monotonic union of every chunk's concept stems, for the LINT graph-diff
        # gate. Folded incrementally (only chunks appended since the last LINT) to
        # avoid an O(chunks × concepts) rescan on every chunk's LINT.
        self._run_concept_stems: set[str] = set()
        self._run_concept_stems_n: int = 0
        self._current_chunk_idx: int = 0
        # Per-file pipeline: setup states (RECON→SALIENCE) run one file at a
        # time; the FSM loops back to RECON for the next file after the current
        # file's chunks are written. Keyed by global inbox-file index so
        # committed-file skips never desync fi from inbox_files.
        self._current_file_idx: int = 0
        self._file_chunks: dict[int, dict] = {}  # fi → {"source_file": str, "chunks": [...]}
        self._chunk_flat_to_fi_ci: dict[int, tuple[int, int]] = {}  # flat_idx → (file_idx, chunk_idx)
        self._last_running_phase: str = ""  # phase in flight, for the failure ledger

        # S3.3: Load the recipe for dynamic configuration. The recipe is bundled
        # package data — if it's missing the install is broken; fail fast.
        from silica.router.recipe_parser import load_recipe
        self._recipe = load_recipe("injector")
        self._has_collision_phase = any(
            p.get("id") == "collision" for p in self._recipe.get("phases", [])
        )

        # Run facade — TaskLedger (immutable plan, built from the recipe) +
        # ProgressLedger (mutable state) + RunManifest (short-term memory)
        # under one run_id; the resume fallback dance lives in Run.resume.
        from silica.kernel.progress import PlanStep, Run
        _checkpoints = [
            PlanStep(
                id=p["id"],
                kind=p.get("kind", "mechanical"),
                objective=p.get("tool", p.get("worker", p["id"])),
            )
            for p in self._recipe.get("phases", [])
        ]
        _run_kwargs: dict[str, Any] = dict(
            mode="inject",
            user_request=f"inject {', '.join(self.inbox_files)} → {target_dir}",
            checkpoints=_checkpoints,
            inputs={
                "inbox_files": self.inbox_files,
                "inbox_file": self.inbox_file,
                "target_dir": target_dir,
                "hub": hub or "",
            },
        )
        # NB: named _run because `run` would shadow the FSM's run() entry point
        run = Run.resume(resume_run_id, **_run_kwargs) if resume_run_id else Run.new(**_run_kwargs)
        self._run = run
        self.progress = run.progress
        self.manifest = run.manifest
        self.task_ledger = run.task_ledger
        if not run.resumed:
            self.progress.add_task("recon",   task_id="recon")
            self.progress.add_task("payload", task_id="payload", depends_on=["recon"])
            self.progress.save()

        # BaseFSM contract
        self._phase_label = "Injector"
        self._done_state = InjectorState.DONE
        self._error_state = InjectorState.ERROR
        self._rollback_state = InjectorState.ROLLBACK
        self._phase_to_state: dict[str, InjectorState] = {
            "recon":      InjectorState.RECON,
            "payload":    InjectorState.PAYLOAD,
            "salience":   InjectorState.SALIENCE,
            "collision":  InjectorState.COLLISION,
            "distill":    InjectorState.DELEGATE,
            "sanitize":   InjectorState.SANITIZE,
            "validate":   InjectorState.VALIDATE,
            "snapshot":   InjectorState.SNAPSHOT,
            "write":      InjectorState.WRITE,
            "hub_update": InjectorState.HUB_UPDATE,
            "autolink":   InjectorState.AUTOLINK,
            "backlink":   InjectorState.BACKLINK,
            "lint":       InjectorState.LINT,
            "cleanup":    InjectorState.CLEANUP,
            "rollback":   InjectorState.ROLLBACK,
        }

        # S2.2.1: Handlers mapping and error policy.
        self._HANDLERS = {
            InjectorState.RECON:      lambda: states.setup.handle_recon(self),
            InjectorState.PAYLOAD:    lambda: states.setup.handle_payload(self),
            InjectorState.SALIENCE:   lambda: states.setup.handle_salience(self),
            InjectorState.COLLISION:  lambda: states.collision.handle_collision(self),
            InjectorState.DELEGATE:   lambda: states.distill.handle_delegate(self),
            InjectorState.SANITIZE:   lambda: states.distill.handle_sanitize(self),
            InjectorState.VALIDATE:   lambda: states.distill.handle_validate(self),
            InjectorState.SNAPSHOT:   lambda: states.write.handle_snapshot(self),
            InjectorState.WRITE:      lambda: states.write.handle_write(self),
            InjectorState.HUB_UPDATE: lambda: states.write.handle_hub_update(self),
            InjectorState.AUTOLINK:   lambda: states.linking.handle_autolink(self),
            InjectorState.BACKLINK:   lambda: states.linking.handle_backlink(self),
            InjectorState.LINT:       lambda: states.finalize.handle_lint(self),
            InjectorState.CLEANUP:    lambda: states.finalize.handle_cleanup(self),
            InjectorState.ROLLBACK:   lambda: states.finalize.handle_rollback(self),
        }

        self._ON_ERROR = {
            # Setup phases: abort the whole run on failure
            InjectorState.RECON: InjectorState.ERROR,
            InjectorState.PAYLOAD: InjectorState.ERROR,
            # Per-chunk phases: contain failure at chunk level via rollback
            InjectorState.DELEGATE: InjectorState.ROLLBACK,
            InjectorState.SANITIZE: InjectorState.ROLLBACK,
            InjectorState.VALIDATE: InjectorState.ROLLBACK,
            InjectorState.SNAPSHOT: InjectorState.ROLLBACK,
            InjectorState.WRITE: InjectorState.ROLLBACK,
            InjectorState.HUB_UPDATE: InjectorState.ROLLBACK,
            InjectorState.LINT: InjectorState.ROLLBACK,
        }

        # Phases the recipe declares best_effort: an unhandled failure skips to
        # the next phase instead of aborting the run (A26). Without this a
        # best-effort phase whose handler doesn't self-guard (e.g. collision_pass
        # tail) would route to ERROR — for post-write AUTOLINK/BACKLINK that
        # bypasses ROLLBACK and strands a half-committed chunk.
        self._best_effort_states = {
            self._phase_to_state[p["id"]]
            for p in self._recipe.get("phases", [])
            if p.get("best_effort") and p.get("id") in self._phase_to_state
        }

    def _get_chunks_from_context_if_empty(self) -> None:
        """Helper to extract chunks from self.context['payload'] if self._chunks is empty."""
        if not self._chunks and "payload" in self.context:
            res = self.context["payload"]
            if "chunks" in res:
                self._chunks = res["chunks"]
            elif "payload" in res:
                self._chunks = [res["payload"]]
            else:
                self._chunks = [res]

    def _progress_note(
        self,
        task_id: str,
        capability_name: str,
        status: str,
        *,
        output_ref: str | None = None,
        content_hash: str | None = None,
        error: str | None = None,
    ) -> None:
        """Shadow: record FSM progress in ProgressLedger; never affects FSM control flow."""
        try:
            if not any(t.id == task_id for t in self.progress.tasks):
                self.progress.add_task(capability_name, task_id=task_id)
            if status == "done":
                self.progress.mark_done(task_id, output_ref=output_ref, content_hash=content_hash)
            elif status == "failed":
                self.progress.mark_failed(task_id, error or "")
            else:
                self.progress.set_status(task_id, status, error=error)  # type: ignore[arg-type]
            self.progress.save()
        except Exception as _e:
            logger.debug("progress shadow error (suppressed): %s", _e)

        # The phase in flight, for the failure ledger. rollback is excluded so it
        # cannot overwrite the gate that actually failed — it runs as a
        # consequence of that failure, and the order of the two is not fixed.
        if status == "running" and capability_name != "rollback":
            self._last_running_phase = capability_name

        # Surface the transition to whatever frontend is watching (no-op if none).
        try:
            from silica.agent.events import PhaseEvent
            from silica.ui.renderer import emit_phase
            from silica.agent import narration as _narr_mod
            pe = PhaseEvent(phase=capability_name, status=status,
                            **self._phase_position(capability_name))
            emit_phase(pe)
            _narr_mod.NARRATOR.on_render_event(pe)   # durable twin of the bus event
        except Exception:
            pass

    def _failed_phase_id(self) -> str:
        """The phase the current chunk died in, or "" if none was recorded."""
        return getattr(self, "_last_running_phase", "") or ""

    def _phase_position(self, capability_name: str) -> dict:
        """Where the run is, for a PhaseEvent. Every event restates all of it.

        file_idx comes from _current_file_idx, NOT from the chunk map: the map
        still points into the previous file during the next file's
        RECON/PAYLOAD (it only gains that file's entries at its own
        PAYLOAD), so deriving it there names the wrong document.

        chunk_total is this FILE's chunk count. The run-wide total does not exist
        until the last file is partitioned — `_chunks` grows one file-group at a
        time — so a run-wide denominator would keep moving under the reader.
        """
        fi = self._current_file_idx
        _, ci = self._chunk_flat_to_fi_ci.get(self._current_chunk_idx, (fi, 0))
        scope = ("exception" if capability_name == "rollback"
                 else "file" if capability_name in _FILE_SCOPE_PHASES
                 else "chunk")
        return {
            "scope": scope,
            "file_idx": fi,
            "file_total": len(self.inbox_files),
            # A file-scope phase runs before this file has chunks; reporting the
            # previous file's chunk index there would rewind the counter.
            "chunk_idx": 0 if scope == "file" else ci,
            "chunk_total": len(self._file_chunks.get(fi, {}).get("chunks", [])),
            "source_file": self.inbox_files[fi] if fi < len(self.inbox_files) else "",
        }

    @property
    def _chunk_ctx(self) -> dict:
        """Per-chunk volatile state namespace — cleared atomically on each chunk boundary."""
        return self.context.setdefault("chunk", {})

    def _save_knowledge_block(self, chunk_idx: int, ops_path: str) -> str:
        """Persist validated ops to a stable (non-tmp) path in the run directory.

        Returns the persistent path so it can be stored as a task output_ref
        and reused on re-runs (content-addressed idempotency).
        """
        import shutil
        kb_dir = self.progress.run_dir / "checkpoints" / f"chunk_{chunk_idx}"
        kb_dir.mkdir(parents=True, exist_ok=True)
        kb_path = str(kb_dir / "validated_ops.json")
        shutil.copy2(ops_path, kb_path)
        return kb_path

    @staticmethod
    def _retryable(op: dict) -> bool:
        """Deferred retry replays ops verbatim — an op with no payload re-fails
        identically forever, so it never earns a slot in the store. skip ops do
        nothing on retry; write/patch with an empty snippet and overwrite with
        empty content have nothing to write."""
        t = op.get("op")
        if t == "skip":
            return False
        if t in ("write", "patch") and not (op.get("snippet") or "").strip():
            return False
        if t == "overwrite" and not (op.get("content") or "").strip():
            return False
        return True

    def _defer_ops(
        self,
        rejected_ops: list[dict],
        rejection_reasons: dict[str, str],
        *,
        phase: str,
    ) -> bool:
        """Persist rejected/failed ops to the deferred store, merging with any
        bundle already saved under this source file's content hash.

        Every defer site (COLLISION, VALIDATE, WRITE) funnels through here so
        the bundle's merge semantics live in exactly one place: because all
        phases of all chunks of one file share the same content_hash, a later
        phase (or a later chunk) must NOT clobber ops an earlier one deferred —
        they accumulate. Returns True iff a bundle was written.
        """
        kept = [op for op in rejected_ops if not isinstance(op, dict) or self._retryable(op)]
        if len(kept) < len(rejected_ops):
            logger.info(
                "%s: %d empty-payload op(s) not deferred (verbatim retry would re-fail)",
                phase, len(rejected_ops) - len(kept),
            )
        rejected_ops = kept
        if not rejected_ops:
            return False
        content_hash = self._current_content_hash
        if not content_hash:
            logger.warning(
                "%s: %d op(s) to defer but no content_hash — deferred store skipped.",
                phase, len(rejected_ops),
            )
            return False
        # Persist the payload the ops were validated against, so the deferred
        # retry re-validates with the same grounding/heading/collision checks
        # instead of the strictly weaker empty-payload pass (finding 2).
        # Best-effort: an early defer site (SETUP) has no payload yet.
        payloads: list[dict] = []
        try:
            if self._chunks and self._current_chunk_idx < len(self._chunks):
                payloads = [self._chunks[self._current_chunk_idx]]
            else:
                pd = self.context.get("payload", {})
                if "chunks" in pd:
                    payloads = list(pd["chunks"])
                elif "payload" in pd:
                    payloads = [pd["payload"]]
        except Exception:
            payloads = []
        try:
            from silica.kernel.recall.deferred import get_deferred_store
            store = get_deferred_store()
            existing = store.get(content_hash) or {}
            # Accumulation is per-run. A bundle saved against a DIFFERENT target
            # belongs to an earlier run of the same source that aimed somewhere
            # else: its op paths point into that old folder, so retrying them
            # re-fails forever and counting them tells the user a successful run
            # dropped work it never touched ("34 new, 42 deferred", where 41
            # were the previous run's).
            if existing and existing.get("target_dir") != self.target_dir:
                logger.info(
                    "%s: dropping %d deferred op(s) from an earlier run targeting %r",
                    phase, len(existing.get("rejected_ops", [])),
                    existing.get("target_dir"),
                )
                existing = {}
            store.put(
                content_hash=content_hash,
                source_path=self._current_source_file,
                target_dir=self.target_dir,
                hub=self.hub,
                rejected_ops=list(existing.get("rejected_ops", [])) + rejected_ops,
                rejection_reasons={**existing.get("rejection_reasons", {}), **rejection_reasons},
                phase=phase,
                payloads=list(existing.get("payloads", [])) + payloads,
            )
            return True
        except Exception as _de:
            logger.warning("%s: failed to save deferred ops: %s", phase, _de)
            return False

    def _note_renucleation(self, inbox_file: str, content_hash: str) -> None:
        """Record that `inbox_file` was nucleated before at another version.

        The ledger knew (check_renucleate existed, tested, uncalled) and the
        user was never told that the notes of the previous version stay in
        place beside the new ones. No prompt belongs in this path, which runs
        headless from the GUI and MCP, so the fact is said where it can be
        acted on: the log at start, the run report (`files_summary`) and the
        completion line at the end, each naming the count and the command
        that is the operator's actual choice, `/revert --source`.
        """
        if not content_hash:
            return  # an unreadable file makes no claim about its history
        from silica.kernel.write.provenance import check_renucleate

        basename = os.path.basename(inbox_file)
        modified, prior = check_renucleate(basename, content_hash)
        if not modified:
            return
        self.context.setdefault("renucleated", {})[basename] = prior
        logger.warning(
            "NUCLEATE: %s changed since its last nucleation; %d note(s) derive from "
            "the previous version and stay in place (/revert --source %s removes them)",
            basename, prior, basename)

    def run(self) -> dict[str, Any]:
        """Execute the pipeline end-to-end (single or multi-file)."""
        from silica.kernel.write.ledger import get_ledger
        ledger = get_ledger()

        # Compute per-file canonicals and content hashes; track committed status
        from silica.kernel.write.provenance import source_event_date
        self._file_canonicals = []
        self._file_content_hashes = []
        self._file_valid_from = []
        # One is_committed() lookup per file: accumulate the committed indices here
        # and derive all_committed from the set (was a second pass of lookups).
        for i, inbox_file in enumerate(self.inbox_files):
            canonical = self._source_canonical_for(inbox_file)
            self._file_canonicals.append(canonical)
            try:
                content_bytes = DRIVER.read_note(inbox_file).content.encode("utf-8")
                content_hash = hashlib.sha256(content_bytes).hexdigest()
            except Exception:
                try:
                    content_bytes = open(inbox_file, "rb").read()
                    content_hash = hashlib.sha256(content_bytes).hexdigest()
                except OSError:
                    content_bytes = b""  # never carry the previous file's bytes forward
                    content_hash = ""
            self._file_content_hashes.append(content_hash)
            self._note_renucleation(inbox_file, content_hash)
            # Resolve the event clock once per file, off the bytes already read.
            # None (undated source) stamps nothing: the run date is the ingest
            # clock, and stamping it as valid_from would feed note_clock a fake
            # freshness that defeats suppress_contest's recency veto.
            self._file_valid_from.append(
                source_event_date(
                    content_bytes.decode("utf-8", "replace"),
                    getattr(self, "seen_override", None),
                )
            )
            if ledger.is_committed(canonical, content_hash=content_hash):
                self._committed_file_indices.add(i)

        all_committed = len(self._committed_file_indices) == len(self.inbox_files)

        # Compat keys for first file (used by single-file code paths and RECON)
        self.context["source_canonical"] = self._file_canonicals[0] if self._file_canonicals else ""
        self.context["source_content_hash"] = self._file_content_hashes[0] if self._file_content_hashes else ""

        if all_committed:
            self.context["final_status"] = "already_nucleated"
            return self.context

        # Only open a journal run when the pipeline will actually execute writes.
        from silica.kernel.write.undo_journal import get_undo_journal
        # The ledger's run id travels with the journal row: it is the join
        # `/revert --source` needs, and the journal's own uuid is a different
        # keyspace that the ledger never sees.
        self._undo_run_id = get_undo_journal().start_run(
            source=self.inbox_file, vault=getattr(CONFIG, "vault_path", None) or None,
            ledger_run_id=getattr(getattr(self, "progress", None), "run_id", None),
        )

        # Fix A: repair index drift before this run reads the indexes — crash
        # drift and out-of-band creates/edits/deletes (kernel/recall/sync.py).
        # No-op/sub-ms when in sync.
        from silica.kernel.recall.sync import sweep
        sweep(force=True)

        # Per-file pipeline: start at the first uncommitted file (committed
        # files are skipped entirely — no recon/embedding spent on them).
        self._current_file_idx = self._next_uncommitted_file_idx(0)

        self.state = InjectorState.RECON
        return self._run_loop()

    def _run_loop(self) -> dict[str, Any]:
        cancelled = False
        try:
            return super()._run_loop()
        except KeyboardInterrupt:
            # Ctrl+C means "stop writing to my vault", so the finally below must
            # stay pure cleanup. _boundary_anneal is not cleanup: it sweeps the
            # deferred store and creates notes, so it used to land a whole batch
            # of writes ~30s AFTER the user interrupted — including bundles
            # deferred by earlier runs — and none of them journalled for /revert
            # (CLEANUP never ran). The bundles keep, so the next run recovers them.
            cancelled = True
            raise
        finally:
            if self._prefetcher is not None:
                self._prefetcher.shutdown()
            if self._residue_executor is not None:
                self._residue_executor.shutdown(wait=False, cancel_futures=True)
            if not cancelled:
                self._boundary_anneal()
            self._flush_indexes()

    def _on_step_error(self, exc: Exception) -> bool:
        """Best-effort phases skip to the next one; per-chunk phases route to
        ROLLBACK without the base's live-txn guard (WRITE can fail before the
        txn is assigned) and record the reason in the per-chunk namespace."""
        if self.state in self._best_effort_states:
            logger.warning(
                "FSM: best-effort phase %s failed (%s) — skipping to next phase",
                self.state.name, exc,
            )
            self._transition_success()
            return True
        if self._ON_ERROR.get(self.state) == self._rollback_state:
            self._chunk_ctx["abort_reason"] = str(exc)
            self.state = self._rollback_state
            return True
        return False

    def _boundary_anneal(self) -> None:
        """Mechanical recovery sweep, once per run: re-validate every deferred
        bundle against the now-larger vault and write what passes. No LLM
        (steer=False) — the escalation pass stays the opt-in silica_anneal tool.
        This is what lets the in-run gate stay strict: anything it defers gets a
        batched second chance here, off the critical path.

        ponytail: sweeps the whole deferred store each run; if a vault
        accumulates many unfixable bundles, gate on this-run deferrals instead.
        Kill-switch: SILICA_BOUNDARY_ANNEAL=0.
        """
        import os
        if os.getenv("SILICA_BOUNDARY_ANNEAL", "1") == "0":
            return
        try:
            from silica.kernel.recall.deferred import get_deferred_store
            if not get_deferred_store().list_all():
                return
            from silica.agent.commit import _current_ledger_run, _current_undo_run
            from silica.tools.pipeline import silica_anneal

            # This sweep writes inside the run's own `finally`, so its notes are
            # part of the run the user just started: they ride the run's journal
            # entry (else /revert undoes the anneal and leaves the nucleation)
            # and carry the run's ledger id (else the dangling-link sweep, which
            # matches on progress.run_id, cannot see them).
            toks = (_current_undo_run.set(self._undo_run_id),
                    _current_ledger_run.set(getattr(self.progress, "run_id", None)))
            try:
                res = silica_anneal(steer=False)
            finally:
                _current_undo_run.reset(toks[0])
                _current_ledger_run.reset(toks[1])
            # Read back by the coordinator's coverage summary: the recovery was
            # info-level only, so a run that rescued a batch of ops said nothing
            # about it to the user who asked for the nucleation.
            self._annealed_ops = int(res.get("written") or 0)
            if res.get("written"):
                logger.info("boundary anneal: recovered %d deferred op(s)", res.get("written"))
                self._lift_recovered_partial()
            if res.get("flagged_ops"):
                # Soft-gate landings from the sweep: the retry path has no
                # judge and no ledger, but this runs inside the run (queue
                # still open, workers alive), so they get the same settle as
                # a chunk write: ledger row, judge with loser_path, expand.
                from silica.kernel.write.ops import Op
                from silica.router.states.write import _settle_flagged
                ops = [Op.model_validate(o) for o in res["flagged_ops"]]
                _settle_flagged(self, ops, committed={o.touched_ref() for o in ops})
        except Exception as e:
            logger.debug("boundary anneal skipped (%s)", e)

    def _lift_recovered_partial(self) -> None:
        """Drop the "partial" verdict when the anneal recovered every failure.

        has_partial_failure is a one-way latch and CLEANUP computes
        final_status before this sweep runs, so a run that ended complete still
        announced `partial` (measured 2026-08-16: three ops failed lint/write,
        all three notes written by the anneal seconds later, verdict unchanged).
        Only WRITE-deferred ops are recoverable here: a chunk that rolled back
        recorded failed_chunks and keeps its verdict, and a file whose bundle
        survives the anneal really is partial.
        """
        if not self.context.get("has_partial_failure") or self.context.get("failed_chunks"):
            return
        from silica.kernel.recall.deferred import get_deferred_store

        store = get_deferred_store()
        if any(store.get(h) for h in (self._file_content_hashes or []) if h):
            return
        self.context["has_partial_failure"] = False
        self.context["final_status"] = (
            "Success" if self.context.get("run_had_ops") else "no_ops"
        )

    def _flush_indexes(self) -> None:
        """Persist the deferred embed + co-occurrence upserts once per run (Fix A).

        The write path upserts into the shared in-memory singletons with
        save=False and marks the index dirty; this single flush rewrites each
        dirty index file once instead of per note (1.17s/note at 10k). Gated on
        the dirty flags so a run that wrote nothing (or had the embedder down)
        never rewrites the index. Runs in the _run_loop finally so it fires on
        success, error, and Ctrl+C; a hard kill is repaired by the index sweep.
        """
        ctx = getattr(self, "context", {})
        if ctx.get("_embed_dirty"):
            try:
                from silica.kernel.recall.embed import get_store
                get_store().save()
            except Exception as e:
                logger.debug("flush: embed index save skipped (%s)", e)
        if ctx.get("_cooccur_dirty"):
            try:
                from silica.kernel.recall.cooccurrence import get_cooccur_store
                get_cooccur_store().save()
            except Exception as e:
                logger.debug("flush: cooccur index save skipped (%s)", e)
        if ctx.get("_lexical_dirty"):
            try:
                from silica.kernel.recall.lexical import get_lexical_store
                get_lexical_store().save()
            except Exception as e:
                logger.debug("flush: lexical index save skipped (%s)", e)

    def _next_uncommitted_file_idx(self, start: int) -> int:
        """Return the first file index >= start not already committed in the ledger."""
        idx = start
        committed = self._committed_file_indices  # always set in __init__
        while idx < len(self.inbox_files) and idx in committed:
            logger.info("Skipping already-committed file %d: %s", idx, self.inbox_files[idx])
            idx += 1
        return idx

    def _advance_file_or_done(self) -> bool:
        """Per-file pipeline: move to the next uncommitted inbox file (→ RECON).

        Returns True when a next file exists (state set to RECON), False when
        none remain (caller concludes the run).
        """
        next_fi = self._next_uncommitted_file_idx(self._current_file_idx + 1)
        if next_fi >= len(self.inbox_files):
            return False
        self._current_file_idx = next_fi
        logger.info(
            "Advancing to file %d/%d: %s",
            next_fi + 1, len(self.inbox_files), self.inbox_files[next_fi],
        )
        self.state = InjectorState.RECON
        return True

    def _on_pipeline_end(self) -> None:
        """Check if there are more chunks to process or if the queue is empty."""
        # Clear the per-chunk volatile namespace atomically before advancing
        self.context.pop("chunk", None)
        self._txn = None
        self._pre_graph = None
        self._get_chunks_from_context_if_empty()
        next_idx = self._next_uncommitted_chunk_idx(self._current_chunk_idx + 1)
        # A warm-attached next file's chunks sit in the flat list before its
        # RECON ever ran: a chunk belonging to a later file means THIS file is
        # done, and the per-file machinery must advance through RECON (the
        # warmed states fast-path via their guards; the cursor is positioned
        # by the warmed PAYLOAD).
        if next_idx < len(self._chunks) and (
            self._chunk_flat_to_fi_ci.get(next_idx, (self._current_file_idx, 0))[0]
            == self._current_file_idx
        ):
            self._current_chunk_idx = next_idx
            logger.info(f"✔ Batch completed successfully. Advancing to batch {self._current_chunk_idx + 1}")
            # Restart per-chunk loop from COLLISION (Phase 5) if present, else DELEGATE
            self.state = InjectorState.COLLISION if self._has_collision_phase else InjectorState.DELEGATE
        elif self._advance_file_or_done():
            pass  # routed to RECON; the next phase event carries the new position
        else:
            logger.info("🎉 All batched chunks have been successfully injected and verified!")
            # Parked residue verifications resolve here, before the executor
            # shutdown in _run_loop's finally would cancel their futures.
            try:
                from silica.router.states import finalize as _fz
                _fz.flush_residue_pending(self, wait=True)
            except Exception as _fe:
                logger.debug("residue flush failed (non-fatal): %s", _fe)
            self.state = InjectorState.DONE

    def _source_canonical_for(self, inbox_file: str) -> str:
        """Vault-relative canonical path for an arbitrary inbox file (no .md, lowercase)."""
        vault_path = getattr(CONFIG, "vault_path", None) or ""
        if vault_path:
            try:
                from pathlib import Path as _P
                rel = _P(inbox_file).relative_to(_P(vault_path)).as_posix()
                return rel.removesuffix(".md").lower()
            except ValueError:
                pass
        return os.path.splitext(os.path.basename(inbox_file))[0].lower()

    def _chunk_task_id(self, cap: str, idx: int | None = None) -> str:
        """Task ID for a chunk (default: current) using the f{fi}_c{ci}_{cap} scheme."""
        flat = self._current_chunk_idx if idx is None else idx
        fi, ci = self._chunk_flat_to_fi_ci.get(flat, (0, flat))
        return f"f{fi}_c{ci}_{cap}"

    def _contain_chunk_failure(self) -> None:
        """Contain a per-chunk failure: mark failed tasks, reset context, advance.

        Called after rollback completes (or as a no-op when no txn existed).
        Preserves all previously committed chunks — only the failing chunk is
        affected.  If more chunks remain, the FSM restarts from COLLISION;
        otherwise it concludes with final_status="partial".
        """
        idx = self._current_chunk_idx
        fi, ci = self._chunk_flat_to_fi_ci.get(idx, (0, idx))
        # Read abort_reason before clearing the chunk namespace
        abort_reason = self._chunk_ctx.get("abort_reason", "chunk failure")

        # Mark all f{fi}_c{ci}_* tasks that are not already done as failed
        prefix = f"f{fi}_c{ci}_"
        for task in self.progress.tasks:
            if task.id.startswith(prefix) and task.status not in ("done",):
                try:
                    self.progress.mark_failed(task.id, error=abort_reason[:200])
                except Exception:
                    pass
        try:
            self.progress.save()
        except Exception:
            pass

        # Clear the per-chunk namespace atomically (prevents state leakage to next chunk).
        # idx-keyed context keys (chunk_{idx}_*) are already safe — each chunk uses
        # its own idx — so only the chunk namespace dict needs explicit teardown.
        self.context.pop("chunk", None)
        self._txn = None
        self._pre_graph = None
        # WRITE appends this chunk's op inverses to _run_inverses; CLEANUP clears it.
        # A rolled-back chunk never reaches CLEANUP, so drop its now-stale inverses
        # here or the next chunk's CLEANUP journals them (corrupting /revert replay).
        self._run_inverses.clear()

        # Record that at least one chunk failed (used by cleanup to set "partial").
        # failed_chunks is the per-chunk ledger: context["error"] is last-write-wins,
        # which once collapsed 6 batch failures into "batch 5 failed" and fed a
        # false "5/6 ok" success report downstream.
        self.context["has_partial_failure"] = True
        # `phase` is recorded structurally, not left to be parsed back out of the
        # error prose: a frontend replaying a stored run (no live phase stream)
        # can then still say WHERE each chunk died, not just that it did.
        self.context.setdefault("failed_chunks", []).append(
            {"chunk": f"f{fi}_c{ci}", "phase": self._failed_phase_id(),
             "error": abort_reason[:200]}
        )

        # Per-file accounting is CLEANUP-anchored, but a failed last chunk never
        # reaches CLEANUP — so this file (whose earlier chunks may have committed
        # real notes) would otherwise get no log.md line / files_summary entry.
        # Emit it here on the file boundary; _log_nucleate_completion is guarded
        # against double-recording the success path.
        file_group = self._file_chunks.get(fi, {})
        n_chunks_in_file = len(file_group.get("chunks", []))
        if ci + 1 >= n_chunks_in_file:
            try:
                from silica.router.states.finalize import _log_nucleate_completion
                _log_nucleate_completion(
                    self, fi, file_group.get("source_file", self.inbox_file)
                )
            except Exception as _le:
                logger.debug("containment: per-file log skipped (non-fatal): %s", _le)

        # Advance to next uncommitted chunk, or conclude the run as partial
        self._get_chunks_from_context_if_empty()
        next_idx = self._next_uncommitted_chunk_idx(self._current_chunk_idx + 1)
        # Same warm-attach guard as _on_pipeline_end: a next chunk belonging
        # to a later file must advance through RECON, not jump straight in.
        if next_idx < len(self._chunks) and (
            self._chunk_flat_to_fi_ci.get(next_idx, (self._current_file_idx, 0))[0]
            == self._current_file_idx
        ):
            self._current_chunk_idx = next_idx
            logger.info(
                "Chunk f%d_c%d failed — advancing to chunk %d of %d.",
                fi, ci, self._current_chunk_idx + 1, len(self._chunks),
            )
            self.state = InjectorState.COLLISION if self._has_collision_phase else InjectorState.DELEGATE
        elif self._advance_file_or_done():
            logger.info("Chunk f%d_c%d failed (last chunk of file) — advancing to next file.", fi, ci)
        else:
            logger.info(
                "Chunk f%d_c%d failed (last uncommitted chunk). Run concludes with partial success.", fi, ci
            )
            # Same flush as _on_pipeline_end's DONE: completed files' parked
            # declarations must survive a failing last file.
            try:
                from silica.router.states import finalize as _fz
                _fz.flush_residue_pending(self, wait=True)
            except Exception as _fe:
                logger.debug("residue flush failed (non-fatal): %s", _fe)
            # "partial" implies something committed; with zero commits the honest
            # verdict is "failed" (the old unconditional "partial" helped sell a
            # fully-failed run as a mostly-successful one).
            self.context["final_status"] = (
                "partial" if self.context.get("committed_chunks") else "failed"
            )
            self.state = InjectorState.DONE

    def _next_uncommitted_chunk_idx(self, start: int) -> int:
        """Return the first chunk index >= start whose file is not already committed."""
        idx = start
        committed = self._committed_file_indices  # always set in __init__
        while idx < len(self._chunks):
            fi, _ = self._chunk_flat_to_fi_ci.get(idx, (0, 0))
            if fi not in committed:
                return idx
            # defensive. Committed files are pruned before PAYLOAD (the sole
            # writer of _chunks), so their chunks never enter this list and this skip
            # rarely fires — kept as a guard against that invariant drifting.
            logger.info("Skipping already-committed file %d chunk %d", fi, idx)
            idx += 1
        return idx

    @property
    def _current_source_file(self) -> str:
        """Vault-relative path of the inbox file for the current chunk."""
        fi, _ = self._chunk_flat_to_fi_ci.get(self._current_chunk_idx, (0, 0))
        if fi in self._file_chunks:
            return self._file_chunks[fi]["source_file"]
        return self.inbox_file

    @property
    def _current_content_hash(self) -> str:
        """Content hash for the inbox file of the current chunk."""
        fi, _ = self._chunk_flat_to_fi_ci.get(self._current_chunk_idx, (0, 0))
        if self._file_content_hashes and fi < len(self._file_content_hashes):
            return self._file_content_hashes[fi]
        return self.context.get("source_content_hash", "")

    # ------------------------------------------------------------------
    # Ledger helpers (C5)
    # ------------------------------------------------------------------

    def _write_ledger_for_file(self, fi: int, status: str) -> None:
        """Record this chunk's ops into the ledger, attributed to file fi."""
        try:
            from silica.kernel.write.ledger import get_ledger
            ledger = get_ledger()
            txn_id = self._chunk_ctx.get("txn_id", "unknown")

            # Use per-file canonical/hash when available; fall back to context
            if fi < len(self._file_canonicals):
                source_canonical = self._file_canonicals[fi]
                content_hash = self._file_content_hashes[fi] if fi < len(self._file_content_hashes) else None
            else:
                source_canonical = self.context.get("source_canonical", "")
                content_hash = self.context.get("source_content_hash")

            ops = load_ops(self._chunk_ctx["ops_path"])
            for op in ops:
                if op.op == OpType.skip:
                    continue
                ledger.record(
                    txn_id=txn_id,
                    source_canonical=source_canonical,
                    path=op.touched_ref(),
                    op=op.op.value if op.op else "",
                    status=status,
                    content_hash=content_hash,
                )
        except Exception as e:
            logger.warning("Failed to write ledger for file %d: %s", fi, e)

    def _write_ledger_rollback(self, txn_id: str) -> None:
        try:
            from silica.kernel.write.ledger import get_ledger
            get_ledger().mark_rolled_back(txn_id)
        except Exception as e:
            logger.warning("Failed to mark rollback in ledger: %s", e)
