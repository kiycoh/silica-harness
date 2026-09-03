# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""L3 Router / Orchestrator for Silica — Organizer FSM.

Deterministic state machine for the /organize pipeline.

States:
    SCAN       — list vault notes in scope
    CLASSIFY   — L1 co-occurrence matching (zero LLM cost)
    ARBITRATE  — L2 LLM arbiter for ambiguous notes (best-effort)
    PLAN       — generate MoveOp list from Classifications
    SNAPSHOT   — snapshot pre-move state for rollback
    MOVE       — execute DRIVER.move() calls (graph-safe)
    LINT       — graph regression gate (orphan check)
    CLEANUP    — ledger commit
    ROLLBACK   — restore pre-move state on gate failure
    DONE / ERROR — terminal states
"""
from __future__ import annotations

import logging
import os
from enum import Enum, auto
from typing import Any

from silica.driver import DRIVER
from silica.driver.base import NoteRef, GraphSnapshot
from silica.kernel.write.ops import Op, OpType
from silica.kernel.organize.taxonomy import Taxonomy
from silica.router.base_fsm import BaseFSM

logger = logging.getLogger(__name__)


class OrganizerState(Enum):
    INIT      = auto()
    SCAN      = auto()
    CLASSIFY  = auto()
    ARBITRATE = auto()
    PLAN      = auto()
    SNAPSHOT  = auto()
    MOVE      = auto()
    LINT      = auto()
    CLEANUP   = auto()
    ROLLBACK  = auto()
    DONE      = auto()
    ERROR     = auto()


class OrganizerFSM(BaseFSM[OrganizerState]):
    """Deterministic state machine for the /organize pipeline."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        scope: str = "",
        dry_run: bool = True,
        llm_arbiter: bool = True,
        move_uncategorized: bool = False,
    ) -> None:
        self.taxonomy = taxonomy
        self.scope = scope
        self.dry_run = dry_run
        self.llm_arbiter = llm_arbiter
        self.move_uncategorized = move_uncategorized

        self.state = OrganizerState.INIT
        self.context: dict[str, Any] = {
            "note_paths": [],
            "classifications": [],
            "move_ops": [],
            "move_results": [],
        }
        self._tmp_files: list[str] = []
        self._txn = None
        self._pre_graph: GraphSnapshot | None = None
        self._undo_run_id: str | None = None

        # Bundled package data — if it's missing the install is broken; fail fast.
        from silica.router.recipe_parser import load_recipe
        self._recipe = load_recipe("organizer")

        # BaseFSM contract
        self._phase_label = "Organizer"
        self._done_state = OrganizerState.DONE
        self._error_state = OrganizerState.ERROR
        self._rollback_state = OrganizerState.ROLLBACK

        self._phase_to_state: dict[str, OrganizerState] = {
            "scan":      OrganizerState.SCAN,
            "classify":  OrganizerState.CLASSIFY,
            "arbitrate": OrganizerState.ARBITRATE,
            "plan":      OrganizerState.PLAN,
            "snapshot":  OrganizerState.SNAPSHOT,
            "move":      OrganizerState.MOVE,
            "lint":      OrganizerState.LINT,
            "cleanup":   OrganizerState.CLEANUP,
            "rollback":  OrganizerState.ROLLBACK,
        }

        self._HANDLERS = {
            OrganizerState.SCAN:      self._handle_scan,
            OrganizerState.CLASSIFY:  self._handle_classify,
            OrganizerState.ARBITRATE: self._handle_arbitrate,
            OrganizerState.PLAN:      self._handle_plan,
            OrganizerState.SNAPSHOT:  self._handle_snapshot,
            OrganizerState.MOVE:      self._handle_move,
            OrganizerState.LINT:      self._handle_lint,
            OrganizerState.CLEANUP:   self._handle_cleanup,
            OrganizerState.ROLLBACK:  self._handle_rollback,
        }

        self._ON_ERROR = {
            OrganizerState.SCAN:     OrganizerState.ERROR,
            OrganizerState.CLASSIFY: OrganizerState.ERROR,
            OrganizerState.PLAN:     OrganizerState.ERROR,
            OrganizerState.SNAPSHOT: OrganizerState.ERROR,
            OrganizerState.MOVE:     OrganizerState.ROLLBACK,
            OrganizerState.LINT:     OrganizerState.ROLLBACK,
        }

    def run(self) -> dict[str, Any]:
        self.state = OrganizerState.SCAN
        self._run_loop()

        if self.state == self._done_state:
            if "final_status" not in self.context:
                self.context["final_status"] = "Success"
        elif self.state == self._error_state:
            if "final_status" not in self.context:
                self.context["final_status"] = (
                    f"Failed: {self.context.get('error', 'unknown error')}"
                )

        return self.context

    # ------------------------------------------------------------------
    # State Handlers
    # ------------------------------------------------------------------

    def _handle_scan(self) -> None:
        """List all vault notes in the taxonomy scope."""
        try:
            refs = DRIVER.list_files(self.scope or "")
        except Exception as e:
            raise RuntimeError(f"SCAN: list_files failed: {e}") from e

        note_paths = [
            ref.path or ref.name
            for ref in refs
            if (ref.path or ref.name).endswith(".md")
        ]

        # If taxonomy has its own scope, additionally filter by it
        if self.taxonomy.scope:
            scope_prefix = self.taxonomy.scope.replace("\\", "/").rstrip("/") + "/"
            note_paths = [
                p for p in note_paths
                if p.replace("\\", "/").startswith(scope_prefix)
            ]

        logger.info("SCAN: found %d notes in scope '%s'", len(note_paths), self.scope or "(vault-wide)")
        self.context["note_paths"] = note_paths
        self._transition_success()

    def _handle_classify(self) -> None:
        """L1 deterministic classification via co-occurrence stems."""
        from silica.kernel.organize.classify import classify_notes

        note_paths = self.context["note_paths"]
        if not note_paths:
            logger.info("CLASSIFY: no notes to classify — skipping")
            self.context["classifications"] = []
            self._transition_success()
            return

        # Load co-occurrence store once and pass it in to avoid repeated disk reads
        cooccur_store = None
        try:
            from silica.kernel.recall.cooccurrence import get_cooccur_store
            from silica.config import CONFIG
            cooccur_store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
            if len(cooccur_store) == 0:
                cooccur_store = None
        except Exception as exc:
            logger.debug("CLASSIFY: CooccurStore unavailable (%s) — continuing with keyword-only", exc)

        # Run L1 only (no LLM); ARBITRATE phase handles the LLM leg
        classifications = classify_notes(
            note_paths,
            self.taxonomy,
            cooccur_store=cooccur_store,
            llm_arbiter=False,   # L2 handled separately in ARBITRATE
            move_uncategorized=self.move_uncategorized,
        )

        logger.info(
            "CLASSIFY: %d notes classified (%d need move)",
            len(classifications),
            sum(1 for c in classifications if c.needs_move),
        )
        self.context["classifications"] = classifications
        self.context["_cooccur_store"] = cooccur_store   # pass through for ARBITRATE
        self._transition_success()

    def _handle_arbitrate(self) -> None:
        """L2 LLM arbiter for notes in the ambiguous confidence band."""
        from silica.kernel.organize.classify import (
            Classification,
            _DEFAULT_TAU_HIGH,
            _DEFAULT_TAU_LOW,
            _llm_arbitrate,
        )

        classifications: list[Classification] = self.context["classifications"]

        if not self.llm_arbiter:
            logger.info("ARBITRATE: llm_arbiter=False — skipping")
            self._transition_success()
            return

        ambiguous: list[tuple[str, str, list]] = [
            (c.note_path, "", [])
            for c in classifications
            if c.evidence not in ("keyword", "llm") and _DEFAULT_TAU_LOW <= c.confidence < _DEFAULT_TAU_HIGH
        ]

        if not ambiguous:
            logger.info("ARBITRATE: no ambiguous notes — skipping LLM call")
            self._transition_success()
            return

        logger.info("ARBITRATE: sending %d ambiguous notes to LLM", len(ambiguous))

        # Re-fetch snippets for the LLM context
        enriched_ambiguous = []
        for note_path, _s, _r in ambiguous:
            snippet = ""
            try:
                nc = DRIVER.read_note(note_path)
                from silica.kernel.write import frontmatter
                _data, _fm, body = frontmatter.split(nc.content)
                snippet = body[:300]
            except Exception:
                pass
            enriched_ambiguous.append((note_path, snippet, self.taxonomy.rules))

        try:
            llm_choices = _llm_arbitrate(enriched_ambiguous, self.taxonomy)
        except Exception as exc:
            # ARBITRATE is best-effort — log and continue with L1 results
            logger.warning("ARBITRATE: LLM arbiter failed (%s) — using L1 results", exc)
            self._transition_success()
            return

        # Patch the classifications in-place
        path_to_idx = {c.note_path: i for i, c in enumerate(classifications)}
        for note_path, _s, _r in enriched_ambiguous:
            chosen = llm_choices.get(note_path, self.taxonomy.uncategorized)
            idx = path_to_idx.get(note_path)
            if idx is None:
                continue
            c = classifications[idx]
            classifications[idx] = Classification(
                note_path=c.note_path,
                current_folder=c.current_folder,
                target_folder=chosen,
                confidence=1.0,
                evidence="llm",
                needs_move=(
                    c.current_folder != chosen
                    and (self.move_uncategorized or chosen != self.taxonomy.uncategorized)
                ),
                title=c.title,
                rule_themes=c.rule_themes,
            )

        logger.info("ARBITRATE: %d notes reclassified by LLM", len(llm_choices))
        self.context["classifications"] = classifications
        self._transition_success()

    def _handle_plan(self) -> None:
        """Generate MoveOp list from classifications."""
        from silica.kernel.organize.classify import Classification

        classifications: list[Classification] = self.context["classifications"]
        move_ops: list[Op] = []

        for c in classifications:
            if not c.needs_move:
                continue
            basename = os.path.basename(c.note_path)
            to_path = os.path.join(
                c.target_folder.replace("\\", "/"),
                basename,
            ).replace("\\", "/")

            # Skip if the computed destination is identical to the source
            if c.note_path.replace("\\", "/") == to_path:
                continue

            move_ops.append(Op(
                op=OpType.move,
                heading=c.title,
                source_basename=basename,
                from_path=c.note_path,
                to_path=to_path,
            ))

        logger.info(
            "PLAN: %d moves planned out of %d classified notes",
            len(move_ops), len(classifications),
        )

        self.context["move_ops"] = move_ops
        self.context["plan_summary"] = {
            "total_notes": len(classifications),
            "moves_planned": len(move_ops),
            "plan": [
                {
                    "from": op.from_path,
                    "to": op.to_path,
                    "confidence": next(
                        (c.confidence for c in classifications if c.note_path == op.from_path),
                        0.0,
                    ),
                    "evidence": next(
                        (c.evidence for c in classifications if c.note_path == op.from_path),
                        "?",
                    ),
                }
                for op in move_ops
            ],
        }

        # In dry_run mode, stop here — do not execute moves
        if self.dry_run:
            logger.info("PLAN: dry_run=True — returning plan without executing moves")
            self.context["final_status"] = "DryRun"
            self.state = OrganizerState.DONE
            return

        self._transition_success()

    def _handle_snapshot(self) -> None:
        """Snapshot pre-move state for rollback (C3 strategy)."""
        move_ops = self.context["move_ops"]
        if not move_ops:
            # Nothing to snapshot
            self.context["snapshot"] = {}
            self._transition_success()
            return

        from silica.driver.base import Txn
        from silica.kernel.write.ops import InverseOp, InverseOpKind
        import uuid

        # Open an undo-journal run so `/revert` can move these notes back later.
        # Only reached in apply mode with real moves (PLAN returns early on dry_run).
        from silica.config import CONFIG
        from silica.kernel.write.undo_journal import get_undo_journal
        self._undo_run_id = get_undo_journal().start_run(
            source=f"organize:{self.scope or 'vault'}",
            vault=getattr(CONFIG, "vault_path", None) or None,
        )

        # For move ops, the inverse is moving back (from_path → to_path becomes to_path → from_path)
        # We also pre-capture the pre-move graph snapshot for the lint gate.
        inverses = [
            InverseOp(
                kind=InverseOpKind.move_back,   # new kind added below
                path=op.from_path,
                to_path=op.to_path,
            )
            for op in move_ops
        ]

        txn_id = uuid.uuid4().hex
        refs = [
            NoteRef(name=os.path.splitext(os.path.basename(op.from_path))[0], path=op.from_path)
            for op in move_ops
        ]
        self._txn = Txn(id=txn_id, refs=refs, inverses=inverses)
        self.context["snapshot"] = {
            "txn_id": txn_id,
            "inverses": [inv.model_dump() for inv in inverses],
        }
        self.context["txn_id"] = txn_id

        # Pre-move graph snapshot for lint gate
        try:
            self._pre_graph = DRIVER.graph_snapshot(refs)
        except Exception as exc:
            logger.warning("SNAPSHOT: pre-graph snapshot failed (%s) — lint gate will skip", exc)

        self._transition_success()

    def _handle_move(self) -> None:
        """Execute moves via DRIVER.move() — graph-safe (wikilinks updated automatically)."""
        move_ops = self.context["move_ops"]
        if not move_ops:
            self.context["move_results"] = []
            self._transition_success()
            return

        results = []
        failures = []
        for op in move_ops:
            try:
                DRIVER.move(op.from_path, op.to_path)
                results.append({"from": op.from_path, "to": op.to_path, "success": True})
                logger.debug("MOVE: %s → %s", op.from_path, op.to_path)
            except Exception as exc:
                failures.append({"from": op.from_path, "to": op.to_path, "error": str(exc)})
                logger.error("MOVE: failed %s → %s: %s", op.from_path, op.to_path, exc)

        self.context["move_results"] = results
        self.context["move_failures"] = failures

        max_failure_rate = self._get_recipe_gate("move_failure_max", 0.10)
        failure_rate = len(failures) / len(move_ops) if move_ops else 0.0
        if failure_rate > max_failure_rate:
            raise RuntimeError(
                f"MOVE: failure rate {failure_rate:.1%} > {max_failure_rate:.1%} "
                f"({len(failures)}/{len(move_ops)} failed)"
            )

        logger.info(
            "MOVE: completed — %d moved, %d failed",
            len(results), len(failures),
        )
        self._transition_success()

    def _handle_lint(self) -> None:
        """Graph regression gate: check for unplanned orphans after moves."""
        if self._pre_graph is None:
            logger.info("LINT: no pre-graph snapshot — skipping graph regression")
            self._transition_success()
            return

        move_ops = self.context["move_ops"]
        refs_after = [
            NoteRef(name=os.path.splitext(os.path.basename(op.to_path))[0], path=op.to_path)
            for op in move_ops
        ]

        try:
            post_graph = DRIVER.graph_snapshot(refs_after)
        except Exception as exc:
            logger.warning("LINT: post-graph snapshot failed (%s) — skipping gate", exc)
            self._transition_success()
            return

        try:
            from silica.kernel.graph_diff import check_graph_regression
            created_paths = [op.to_path for op in move_ops]
            success, errors = check_graph_regression(
                self._pre_graph, post_graph, created_paths, frozenset()
            )
            if not success:
                nonblocking = ("Unplanned orphans", "Backlink drift")
                orphan_errors = [e for e in errors if e.startswith(nonblocking)]
                blocking_errors = [e for e in errors if not e.startswith(nonblocking)]
                if orphan_errors:
                    logger.warning("LINT: orphan/drift warning (non-blocking): %s", "; ".join(orphan_errors))
                if blocking_errors:
                    self.context["abort_reason"] = (
                        f"Graph regression after move: {'; '.join(blocking_errors)}"
                    )
                    self.state = OrganizerState.ROLLBACK
                    return
        except Exception as exc:
            logger.error("LINT: graph_diff check failed (%s) — treating as non-blocking", exc)

        self._transition_success()

    def _handle_cleanup(self) -> None:
        """Mark run as committed in the ledger and journal move inverses for /revert."""
        self.context["final_status"] = "Success"
        self._write_ledger("committed")
        self._record_undo_inverses()
        self._transition_success()

    def _record_undo_inverses(self) -> None:
        """Persist each committed move's inverse so `/revert <run>` can move it back.

        Only successful moves are journalled — a move that failed (or was rolled
        back before reaching CLEANUP) never lands here.
        """
        if not self._undo_run_id:
            return
        import hashlib
        from silica.kernel.write.ops import InverseOp, InverseOpKind
        from silica.kernel.write.undo_journal import get_undo_journal

        journal = get_undo_journal()
        for res in self.context.get("move_results", []):
            if not res.get("success"):
                continue
            from_path, to_path = res["from"], res["to"]
            try:
                post = DRIVER.read_note(to_path).content
                post_hash = hashlib.sha256((post or "").encode("utf-8")).hexdigest()
            except Exception:
                post_hash = None
            journal.record(
                self._undo_run_id,
                InverseOp(kind=InverseOpKind.move_back, path=from_path, to_path=to_path),
                post_hash,
            )

    def _handle_rollback(self) -> None:
        """Undo moves by calling DRIVER.move() in reverse."""
        snapshot = self.context.get("snapshot", {})
        inverses = snapshot.get("inverses", [])

        if not inverses:
            logger.info("ROLLBACK: no inverses to apply")
        else:
            successful_dests = {res["to"] for res in self.context.get("move_results", [])}
            for inv_dict in inverses:
                from_path = inv_dict.get("path")       # original source
                to_path = inv_dict.get("to_path")      # where we moved it
                if not from_path or not to_path:
                    continue
                if to_path not in successful_dests:
                    logger.debug("ROLLBACK: skipping restore for %s → %s (move did not succeed)", to_path, from_path)
                    continue
                # Reverse: move back from to_path → from_path
                try:
                    DRIVER.move(to_path, from_path)
                    logger.info("ROLLBACK: restored %s → %s", to_path, from_path)
                except Exception as exc:
                    logger.error("ROLLBACK: failed to restore %s → %s: %s", to_path, from_path, exc)

        self.context["final_status"] = (
            f"Rolled Back: {self.context.get('abort_reason', 'unknown reason')}"
        )
        self._transition_success()

    # ------------------------------------------------------------------
    # Ledger helpers
    # ------------------------------------------------------------------

    def _write_ledger(self, status: str) -> None:
        try:
            from silica.kernel.write.ledger import get_ledger
            txn_id = self.context.get("txn_id", "unknown")
            for op in self.context.get("move_ops", []):
                canonical = (op.to_path or op.from_path or "").removesuffix(".md").lower()
                get_ledger().record(
                    txn_id=txn_id,
                    source_canonical=canonical,
                    path=op.to_path or op.from_path or "",
                    op="move",
                    status=status,
                )
        except Exception as exc:
            logger.warning("ORGANIZER: ledger write failed: %s", exc)


