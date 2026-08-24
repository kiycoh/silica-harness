# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Composed tools — L2/L3 logic promoted to system tools.

From SILICA.md §4.2:
  Composed tools encode mechanical workflows that span multiple atomic operations.

This module is a re-export facade: the implementations live in the domain
modules silica.tools.pipeline (injector stages), silica.tools.notes
(single-note fast path), silica.tools.graph (indexes/search/linking/audit),
and silica.tools.runners (full FSM runs and sub-agent batches). Importing
this module registers every composed tool in the TOOLS registry. Tests and
call sites import (and monkeypatch) tool functions through this namespace.
"""
from __future__ import annotations

from silica.tools.pipeline import (  # noqa: F401
    BulkWriteArgs,
    AnnealArgs,
    DeferredRetryArgs,
    LintArgs,
    PayloadArgs,
    ReconArgs,
    SanitizeArgs,
    SubmitRepairedOpsArgs,
    ValidateOpsArgs,
    _same_note,
    silica_anneal,
    silica_bulk_write,
    silica_deferred_retry,
    silica_lint,
    silica_payload,
    silica_recon,
    silica_sanitize,
    silica_validate_ops,
    submit_repaired_ops,
)
from silica.tools.notes import (  # noqa: F401
    PatchNoteArgs,
    WriteNoteArgs,
    silica_patch_note,
    silica_write_note,
)
from silica.tools.events import (  # noqa: F401
    AgendaArgs,
    EventCreateArgs,
    EventUpdateArgs,
    silica_agenda,
    silica_event_create,
    silica_event_update,
)
from silica.tools.graph import (  # noqa: F401
    AutolinkArgs,
    BacklinkArgs,
    CooccurrenceRefreshArgs,
    EmbedRefreshArgs,
    GraphExportArgs,
    LexicalRefreshArgs,
    SemanticSearchArgs,
    VaultReportArgs,
    _in_folder,
    silica_autolink,
    silica_backlink,
    silica_cooccurrence_refresh,
    silica_embed_refresh,
    silica_graph_export,
    silica_health,
    silica_lexical_refresh,
    silica_related,
    silica_semantic_search,
    silica_vault_report,
)
from silica.tools.aliases import (  # noqa: F401
    AliasesArgs,
    silica_aliases,
)
from silica.tools.curate import (  # noqa: F401
    CurateArgs,
    silica_curate,
)
from silica.tools.runners import (  # noqa: F401
    DedupFolderArgs,
    DedupPairsArgs,
    EnrichBatchArgs,
    GenerateTaxonomyArgs,
    LedgerDigestArgs,
    RefineBatchArgs,
    RunInjectorArgs,
    RunOrganizerArgs,
    silica_dedup,
    silica_dedup_pairs,
    silica_enrich_batch,
    silica_generate_taxonomy,
    silica_ledger_digest,
    silica_refine_batch,
    silica_run_injector,
    silica_run_organizer,
)
