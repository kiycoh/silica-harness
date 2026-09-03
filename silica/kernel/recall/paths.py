# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Path canonicalization for vault notes and Silica runtime directories.

The CLI backend resolves notes by their vault-relative POSIX path. Any code
path that accepts a user- or agent-supplied note path MUST canonicalize it
through ``to_vault_relative`` before handing it to the driver — otherwise
absolute filesystem paths reach the Obsidian CLI verbatim and surface as a
misleading "No matches found" / "File not found", because the CLI indexes
by vault-relative path.

This module is the single source of truth for that normalization.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterator
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

from silica.config import CONFIG

try:  # POSIX only; see index_lock for what Windows gives up.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def in_folder(path: str, folder: str | None) -> bool:
    """True if vault-rel `path` is inside `folder` (empty folder ⇒ whole vault).

    Single source of truth for folder-scoping: index reconciliation
    (embed/cooccur build_index), the /embed, /cooccur, /dedup tools, the
    graph-report signals, and the cooccur/relatedness scope filters. The `.md`
    strip is a no-op on keyspaces that are already suffix-free (cooccur_key).
    """
    if not folder:
        return True
    f = folder.replace("\\", "/").strip("/").lower()
    p = path.replace("\\", "/").strip("/").lower().removesuffix(".md")
    return p == f or p.startswith(f + "/")


# The mode an ordinary `open(path, "w")` would have produced, snapshotted once:
# mkstemp opens 0600, so a file this helper CREATES would otherwise be private
# where every other tool's would not. Read at import (before the pools start)
# because os.umask is process-global and read-modify-write — probing it later
# would hand a concurrent creator a 0666 window.
_UMASK = os.umask(0o022)  # os.umask only reads by writing; put it straight back
os.umask(_UMASK)
_NEW_FILE_MODE = 0o666 & ~_UMASK


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Torn-write-proof write: tmp file in the same dir, fsync, os.replace.

    For derived indexes and bundles rewritten in place — a crash or full
    disk mid-write must leave the previous file intact, not a truncated one.

    Four edges beyond the naive tmp+replace (each one a fielded failure in
    graphify's sibling helper): resolve symlinks first so the tmp lands on the
    destination's filesystem and the write goes THROUGH the link instead of
    replacing it; preserve an existing destination's mode, and give a brand-new
    one the umask default, instead of leaving mkstemp's 0600 — notes are the
    user's own files and an external syncer or a second account has to keep the
    access an ordinary create would have granted; and when os.replace hits a
    locked destination (Windows: antivirus or an open reader) retry briefly —
    the hold is transient — and only then degrade to copy2, which is NOT atomic
    (it truncates before it copies), so the degradation is logged instead of
    silently voiding the guarantee this function is named after.
    """
    real = Path(os.path.realpath(path))
    real.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=real.parent, prefix=real.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            mode = os.stat(real).st_mode & 0o7777
        except OSError:
            mode = _NEW_FILE_MODE  # the destination does not exist yet
        with contextlib.suppress(OSError):
            os.chmod(tmp, mode)
        try:
            os.replace(tmp, real)
        except PermissionError:
            import time
            for _ in range(3):
                time.sleep(0.05)
                try:
                    os.replace(tmp, real)
                    break
                except PermissionError:
                    continue
            else:
                logger.warning(
                    "atomic_write_bytes: %s still locked after retries — "
                    "falling back to a non-atomic copy (torn-write window "
                    "while the copy runs)", real)
                shutil.copy2(tmp, real)
                os.unlink(tmp)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    # ponytail: no directory fsync — post-power-loss rename durability is
    # filesystem-dependent, and every caller's file is rebuildable or
    # re-produceable; upgrade if a real loss is ever traced here.


def resolve_vault_path(vault_path: str | None = None) -> str | None:
    """An explicit vault path, else the live CONFIG's, else None.

    The `vault_path=None means "the active vault"` convention shared by the
    kernel's on-disk ledgers (provenance, run_log). CONFIG is read lazily and
    defensively: a caller running before config import must get None, not an
    ImportError.
    """
    if vault_path:
        return vault_path
    try:
        from silica.config import CONFIG

        return getattr(CONFIG, "vault_path", None) or None
    except Exception:
        return None


def vault_epoch(vault: str | None = None) -> str:
    """Cheap validity signature of a vault's current file state, or "".

    One stat walk — sorted (path, mtime_ns, size) folded into a hash: the memo
    key for vault-wide derivatives (report, graph payload, timeline). Any
    create, edit, delete or move changes it, out-of-band ones included, since
    this observes the disk and not an index. Size rides along so an
    mtime-preserving restore (rsync -a, Syncthing) still bumps it.

    Deliberately NOT debounced: a cached epoch would vouch for a vault it has
    not looked at, and an edit inside the window would serve a pre-edit
    derivative as fresh. The stat walk is the whole price (~10 ms per 1k
    notes). Returns "" when no signature can be taken (unbound vault): ""
    means "do not memoize", never "nothing changed".

    Deliberately driver-free: the graph viewer split forbids this chain from
    reaching the agent layer, and a stat walk needs nothing above stdlib.
    """
    import hashlib

    raw = vault or getattr(CONFIG, "vault_path", "") or ""
    if not str(raw).strip():
        return ""  # unbound — Path("") would silently mean the CWD
    root = Path(raw)
    if not root.is_dir():
        return ""
    try:
        ignored = ignore_matcher(root)
        h = hashlib.sha1(str(root).encode("utf-8"))
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames if not d.startswith(".") and not ignored(d)
            )
            for fn in sorted(filenames):
                if not fn.endswith(".md"):
                    continue
                p = Path(dirpath) / fn
                try:
                    st = p.stat()
                except OSError:
                    continue  # deleted mid-walk: absent, like the sweep treats it
                h.update(
                    f"{p.relative_to(root)}\x00{st.st_mtime_ns}\x00{st.st_size}\x00"
                    .encode("utf-8", errors="replace")
                )
        return h.hexdigest()
    except Exception as e:
        logger.debug("vault_epoch unavailable (%s)", e)
        return ""


def quarantine(path: Path) -> Path | None:
    """Rename a corrupt state file aside — never clobbered, never deleted.

    Derived stores rebuild from empty afterwards; authoritative stores keep
    the bytes here for manual inspection. `silica doctor` surfaces any
    `*.corrupt.*` file it finds. Returns the quarantine path, or None if the
    rename itself failed (callers treat that as "proceed anyway": read paths
    must not raise).
    """
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = path.with_name(f"{path.name}.corrupt.{stamp}")
    n = 0
    while dest.exists():  # same-second collision: bump, never overwrite
        n += 1
        dest = path.with_name(f"{path.name}.corrupt.{stamp}.{n}")
    try:
        path.rename(dest)
        return dest
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Verbatim source leaves (spec-harness-promotion 2026-07-24 §2)
# ---------------------------------------------------------------------------

# Vault folder holding verbatim source leaves. Leaves are retrieval-invisible:
# excluded from search, embeddings, co-occurrence, and the autolink title
# index (one rule, all indexes — partial exclusion reintroduces the dilution
# the LoCoMo hybrid arm measured). A leaf is reachable only through an
# explicit `## Sources` wikilink and silica_read_note.
SOURCES_DIR = "sources"

# The H2 under which a note links back to its verbatim leaf. Its presence is
# also the note-side signal that the source is still retrievable, which is what
# separates a grounded claim from a purely distilled one (reliability_tier).
SOURCES_MARKER = "## Sources"


def is_source_leaf(path: str) -> bool:
    """True when `path` (vault-relative, any separator) lives under sources/.

    A vault whose writes are confined to `silica/` stores its leaves at
    `silica/sources/`, and those were answering every search: the bare-root
    check missed them, so the verbatim lectures competed with the notes
    distilled from them and won, being longer. Both roots answer, like
    `run_log` composes its journal path — legacy root leaves stay invisible
    forever.
    """
    norm = (path or "").replace("\\", "/").lstrip("/")
    if norm.startswith(SOURCES_DIR + "/"):
        return True
    try:
        from silica.kernel.vault_manifest import in_write_dir

        composed = in_write_dir(SOURCES_DIR).replace("\\", "/").strip("/")
    except Exception:  # config not resolvable — the bare root is the honest guess
        return False
    return bool(composed) and norm.startswith(composed + "/")


# ---------------------------------------------------------------------------
# Silica runtime directory helpers
# ---------------------------------------------------------------------------

_SILICA_HOME = Path.home() / ".silica"


def silica_tmp_dir() -> Path:
    """Return the pipeline staging directory (~/.silica/tmp/), creating it if needed.

    All FSM temporary files (ops JSON, payload chunks, distiller output) live
    here instead of the system temp directory so they survive the pipeline run
    and are inspectable for debugging.  The FSM removes them on successful
    completion via _cleanup_tmp().
    """
    d = _SILICA_HOME / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def index_file(name: str) -> Path:
    """Resolve one derived-index file (``<name>.json``) under the current vault's
    index dir. The per-store ``_index_path`` shims delegate here (kept as module
    functions so tests can still monkeypatch them per store)."""
    return index_dir() / f"{name}.json"


def build_postings(docs: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Invert ``{doc: {term: count}}`` into the postings index ``{term: {doc: count}}``.
    The shared inner loop of every term/stem postings build (co-occurrence, lexical)."""
    idx: dict[str, dict[str, int]] = {}
    for doc, terms in docs.items():
        for term, count in terms.items():
            idx.setdefault(term, {})[doc] = count
    return idx


def path_keyed_singleton(cache: dict, key: str, factory):
    """Return cache[key], building it via factory() on first access.

    The shared shape behind every per-index-path store singleton (embed,
    co-occurrence, lexical): keying by resolved index path follows a /vault
    switch automatically. Callers own the cache dict (and its clear()), and
    call `sync_from_disk()` on what they get back (see DiskSynced)."""
    inst = cache.get(key)
    if inst is None:
        inst = factory()
        cache[key] = inst
    return inst


def disk_stamp(path: Path) -> tuple[int, int] | None:
    """(mtime_ns, size) of an index file, None when it does not exist.

    The change signal behind DiskSynced. Any difference counts, not just a
    newer mtime: a restored backup moves it backwards, the same rule the
    index sweep's stamps and the driver's roster already follow. Size is the
    tie-break for a rewrite landing on the same timestamp (coarse
    filesystems round mtime to a second or worse).
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


@contextlib.contextmanager
def index_lock(path: Path) -> Iterator[None]:
    """Advisory flock held across one read-merge-write of an index file.

    The lock file is derived from the index path alone (`<dir>/locks/<name>`),
    NOT from `workqueue.path_lease`, whose lock dir comes from the CURRENT
    vault: the save that most needs the lock is the atexit flush of a store
    whose vault was already switched away from, and two processes have to
    agree on the file without consulting config. Without fcntl (Windows) the
    write proceeds unlocked: the cross-process race then narrows to the
    serialize window instead of closing. A lock-file failure is logged and
    degrades the same way rather than failing the save.
    """
    if fcntl is None:
        yield
        return
    real = Path(os.path.realpath(path))
    fd: int | None = None
    try:
        try:
            lock_dir = real.parent / "locks"
            lock_dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(lock_dir / (real.name + ".lock")), os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as e:
            logger.warning("index_lock: cross-process lock unavailable for %s (%s)", real.name, e)
            if fd is not None:
                os.close(fd)
                fd = None
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


class DiskSynced:
    """Keeps a process-singleton index store honest against other processes.

    `path_keyed_singleton` hands every caller in a process ONE store, and that
    store read its file exactly once. So a second Silica process writing the
    same index (`silica nucleate` in a terminal while the GUI or the MCP
    server is up) was invisible to the first until restart, and the first's
    next save() then wrote its stale memory over the second's work. Last
    writer won, and nothing logged it.

    Two rules, no daemon, no database:
      * a `disk_stamp` of the file is taken at every load and save; a
        different stamp on the next access means someone else wrote, and the
        store re-reads the file and lays its own unsaved entries back on top
        (`_take_disk`), so memory = disk + mine, mine winning per key;
      * save() is read-merge-write under `index_lock`, never a blind
        overwrite.

    Subclasses own the dirty bookkeeping: which keys they upserted or dropped
    since the last sync, as live sets returned by `_dirty_sets()`. The mixin
    empties them once a save lands and refills them if it does not. Hooks:
    `_read_disk()` (file -> state, empty on absent/corrupt, never raises),
    `_take_disk(state)` (replace own state with it, overlay dirty),
    `_snapshot()` (under `_lock`: a consistent view), `_serialize(snapshot)`
    (outside it: the slow part, hundreds of ms on a big index).
    """
    _path: Path
    _lock: "threading.RLock"
    _disk_stamp: tuple[int, int] | None = None

    def _read_disk(self):  # pragma: no cover - contract
        raise NotImplementedError

    def _take_disk(self, state) -> None:  # pragma: no cover - contract
        raise NotImplementedError

    def _snapshot(self):  # pragma: no cover - contract
        raise NotImplementedError

    def _serialize(self, snapshot) -> bytes:  # pragma: no cover - contract
        raise NotImplementedError

    def _dirty_sets(self) -> tuple[set, ...]:  # pragma: no cover - contract
        raise NotImplementedError

    def _load(self) -> None:
        """First read, from __init__. Unconditional: `sync_from_disk` would
        skip an absent file, and the embed store still has a legacy path to
        fall back to on exactly that case."""
        self._disk_stamp = disk_stamp(self._path)
        self._take_disk(self._read_disk())

    def sync_from_disk(self) -> bool:
        """Re-read the file if another process wrote it. True when it did.

        One stat when nothing changed; the accessors call this on every
        lookup, so that is the price of the guarantee.
        """
        seen = disk_stamp(self._path)
        if seen == self._disk_stamp:
            return False
        with self._lock:
            if seen == self._disk_stamp:  # a sibling thread took it first
                return False
            # Stamp BEFORE reading. A write landing between the two leaves
            # the stamp older than the file, so the next access re-reads (one
            # redundant load). The reverse order would record a file we never
            # read and hide that write until the one after it.
            self._disk_stamp = seen
            self._take_disk(self._read_disk())
        return True

    def is_dirty(self) -> bool:
        """True when this store holds something the file does not.

        A read of exactly what save() would write, from the bookkeeping
        subclasses already keep. It exists because a no-op save is not free and
        not silent: rewriting an unchanged index still moves the file's mtime,
        and `vault_version()` digests those mtimes to tell the GUI that the
        vault moved under its cached views. The co-occurrence refresh that every
        graph export runs made the strip announce "vault changed" after every
        build of the graph it had just drawn, and rewrote 9.5 MB to say it.
        """
        return any(self._dirty_sets())

    def save(self) -> Path:
        with index_lock(self._path):
            with self._lock:
                self.sync_from_disk()
                snapshot = self._snapshot()
                sets = self._dirty_sets()
                saved = [set(s) for s in sets]
                # Emptied IN PLACE, not rebound: a mutation racing the write
                # below adds to this same set and stays dirty, because it is
                # not in the snapshot being written.
                for s in sets:
                    s.clear()
            try:
                atomic_write_bytes(self._path, self._serialize(snapshot))
            except BaseException:
                with self._lock:
                    for s, back in zip(sets, saved):
                        s |= back
                raise
            with self._lock:
                # Under the file lock, so the stamp is of OUR file: a reader
                # thread that synced between the write and here recorded the
                # same one and overlaid the live dirty set, which is exactly
                # the state this process should hold.
                self._disk_stamp = disk_stamp(self._path)
        return self._path


def is_obsidian_vault(path) -> bool:
    """True when `path` is an Obsidian vault (carries a `.obsidian/` dir).

    Never decides *which* folder is the vault (that is always the one you named
    or launched in), only whether writing notes into its root is at home there:
    an Obsidian vault writes in place, a source tree gets a `write_dir`
    (`onboarding.adopt`). Non-existent paths are False.
    """
    return (Path(path) / ".obsidian").is_dir()


# Directories never walked when sampling or indexing a vault: vendored trees and
# build output, never someone's notes. Hidden dirs are pruned separately (that
# covers .git/.venv/.obsidian). Deliberately NOT `.gitignore`: a gitignored
# folder can be exactly where private notes live (this repo's own `docs/`), so
# honouring it would hide notes. Per-vault additions go in `.silicaignore`.
# Directories a shell or a GUI client can start in that are never a vault: the
# FHS top level, macOS's, and the temp dir. Exact match only, /tmp/<x> is
# somebody's folder while /tmp is nobody's. A root process with cwd /tmp
# adopted it and wrote /tmp/vault.yaml (2026-09-02), which the session hook
# then found above every pytest tmp dir and greeted with the wrong vault.
SYSTEM_DIRS: frozenset[str] = frozenset({
    "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib64", "/media",
    "/mnt", "/opt", "/proc", "/run", "/sbin", "/srv", "/sys", "/tmp", "/usr",
    "/var", "/var/tmp", "/private/tmp", "/Users", "/Volumes", "/Applications",
    "/System", "/Library", tempfile.gettempdir(),
})


def is_system_dir(path) -> bool:
    return str(Path(path).resolve()) in SYSTEM_DIRS


NOISE_DIRS: frozenset[str] = frozenset({
    "node_modules", "vendor", "build", "dist", "target", "__pycache__",
    "site-packages", "coverage", "htmlcov",
})

# Per-vault extension of NOISE_DIRS, because "grows when it bites" should not
# mean "edit the source". Seeded by `onboarding.adopt.seed_silicaignore`.
SILICAIGNORE_REL = ".silicaignore"


def _ignore_patterns(vault) -> tuple[frozenset[str], tuple[str, ...]]:
    """(exact names, globs) to prune: NOISE_DIRS plus `<vault>/.silicaignore`.

    Split so the common case stays a set lookup; a missing or unreadable file
    means the built-ins alone (an ignore file is a convenience, never a gate).
    """
    names, globs = set(NOISE_DIRS), []
    try:
        text = (Path(vault) / SILICAIGNORE_REL).read_text(encoding="utf-8")
    except OSError:
        return frozenset(names), ()
    for line in text.splitlines():
        pat = line.split("#", 1)[0].strip().strip("/")
        if not pat:
            continue
        (globs.append(pat) if any(c in pat for c in "*?[") else names.add(pat))
    return frozenset(names), tuple(globs)


def ignore_matcher(vault) -> Callable[[str], bool]:
    """Predicate over a directory NAME: True ⇒ do not walk it.

    Matches by name at any depth, like NOISE_DIRS itself — not by path, so
    `docs/private` does not work but `private` and `*.egg-info` do.
    ponytail: name-only matching, add path anchoring when someone needs it.

    Reads the file once: build the matcher before a walk, never inside it.
    """
    names, globs = _ignore_patterns(vault)
    if not globs:
        return names.__contains__
    return lambda d: d in names or any(fnmatch(d, g) for g in globs)


# Root-level files that mark a source tree regardless of file counts (a repo
# whose sources all sit under src/ still has one at the top).
_CODE_MARKERS: frozenset[str] = frozenset({
    "pyproject.toml", "setup.py", "package.json", "Cargo.toml", "go.mod",
    "CMakeLists.txt", "Makefile", "pom.xml", "build.gradle", "Gemfile",
    "composer.json", "mix.exs", "build.sbt", "Package.swift",
})


def looks_like_code(root, sample_max: int = 400) -> bool:
    """True when `root` reads as a source tree rather than a folder of prose.

    Two signals, cheapest first: a root-level build/manifest marker, else a
    bounded walk comparing code files to prose files. Never git: an Obsidian
    vault under git is ordinary, and `.git/` says nothing about content.

    Only picks the *default* write root at adoption (`onboarding.adopt`), which
    the vault then declares in `vault.yaml` — a wrong guess costs one edit, not
    a wrong vault. Ratio rather than mere presence so a stray snippet.py in a
    notes folder does not read as a codebase.
    """
    from silica.kernel.code.codeast.base import BARE_LANGUAGES, EXTENSION_MAP

    root = Path(root)
    if not root.is_dir():
        return False
    if any((root / marker).is_file() for marker in _CODE_MARKERS):
        return True

    code_exts = {e for e, lang in EXTENSION_MAP.items() if lang not in BARE_LANGUAGES}
    code = prose = seen = 0
    skip = ignore_matcher(root)
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and not skip(d)]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in code_exts:
                code += 1
            elif ext in (".md", ".txt", ".docx", ".pdf"):
                prose += 1
            seen += 1
            if seen >= sample_max:
                return code > prose
    return code > prose


def resolve_repo_root(vault: str | Path) -> tuple[Path | None, str | None]:
    """Code-lane repo root for `vault`, validating the vault⊂repo invariant (ADR-0019).

    Returns (root, warning). Valid layouts: a repo-mode vault (`<root>/docs/silica`
    or any plain dir — git discovers the target repo above it) and an Obsidian
    vault that is itself the repo root. An Obsidian vault nested inside a
    FOREIGN git repo yields (None, warning): code lane disabled, never grounded
    on the wrong repo. (None, None) when git is absent or no repo contains the
    vault. Pure resolution, no caching — see `repo_root_for`.
    """
    from silica.kernel.code import gitstate

    v = Path(vault).resolve()
    root = gitstate.find_repo_root(v)
    if root is None:
        return None, None
    if is_obsidian_vault(v) and root != v:
        return None, (
            f"code lane disabled: Obsidian vault {v} is nested inside "
            f"foreign git repo {root}"
        )
    return root, None


# Resolved-once storage for the code-lane root (ADR-0019), keyed by resolved
# vault path so it follows /vault switches and stays correct for entry points
# that never run the CLI startup (GUI, MCP). "No repo" results are NOT cached,
# so a `git init` after first resolution is still picked up.
_REPO_ROOT_CACHE: dict[str, tuple[Path | None, str | None]] = {}


def _repo_root_resolved(vault: str | Path) -> tuple[Path | None, str | None]:
    raw = str(vault or "").strip()
    if not raw:
        return None, None
    key = str(Path(raw).resolve())
    hit = _REPO_ROOT_CACHE.get(key)
    if hit is not None:
        return hit
    root, warn = resolve_repo_root(raw)
    if warn:
        logger.warning(warn)
    if root is not None or warn is not None:
        _REPO_ROOT_CACHE[key] = (root, warn)
    return root, warn


def repo_root_for(vault: str | Path) -> Path | None:
    """The single choke point every code-lane consumer derives its repo root
    through (ADR-0019): resolved once per vault, invariant-validated, warning
    logged once. None ⇒ code lane disabled for this vault."""
    return _repo_root_resolved(vault)[0]


def repo_root_warning(vault: str | Path) -> str | None:
    """The invariant-violation message for `vault`, if any — for the CLI to
    surface loudly at startup and /vault switch."""
    return _repo_root_resolved(vault)[1]


def clear_repo_root_cache() -> None:
    """Drop cached repo-root resolutions (test isolation)."""
    _REPO_ROOT_CACHE.clear()


def index_dir_for(vault: str) -> Path:
    """Per-vault index namespace for an explicit `vault` path, independent of
    the global CONFIG singleton. Same digest scheme as `index_dir()` — the
    two agree whenever `vault == CONFIG.vault_path`.

    Callers that need to resolve a *specific* vault's on-disk index (e.g. a
    diagnostic comparing a passed-in config's vault against whatever vault
    the live global CONFIG currently points at) MUST use this rather than
    `index_dir()`, which only ever resolves the global singleton and would
    silently compare the wrong vault's state.
    """
    base = _SILICA_HOME / "index"
    vault = (vault or "").strip()
    if not vault:
        return base
    return base / vault_digest(vault)


def vault_digest(vault: str) -> str:
    """The per-vault namespace key: sha1 of the resolved vault path, 12 chars.

    One scheme shared by every per-vault runtime directory (index, capture
    WAL), so a namespace computed by one seam is readable by the others.
    """
    return hashlib.sha1(str(Path(vault).resolve()).encode("utf-8")).hexdigest()[:12]


def inbox_dir_for(vault: str) -> Path:
    """Per-vault capture WAL: ~/.silica/inbox/<digest12>/.

    Out of the vault on purpose: raw transcripts are private conversation data
    and must never sit inside a committable repo, out of the note scanner, and
    out of Obsidian's index.
    """
    return _SILICA_HOME / "inbox" / vault_digest(vault)


def index_dir() -> Path:
    """Per-vault index namespace: ~/.silica/index/<digest12>/ keyed by the
    resolved vault path; legacy global ~/.silica/index/ when no vault is
    configured. Per-vault state follows the vault (ADR-0014), so /vault
    switch no longer serves another vault's entries."""
    return index_dir_for(getattr(CONFIG, "vault_path", "") or "")


def _is_under(base: Path, target: Path) -> bool:
    """True when `target` is strictly inside `base` (both already normalized)."""
    return target != base and base in target.parents


def contain_in_vault(path: str, vault: Path) -> str:
    """The vault containment choke point: `path` as a safe POSIX vault-relative
    path, or ``ValueError``.

    Every driver write MUST pass its caller-supplied path through here before
    joining it onto the vault root. ``Path(vault) / rel`` silently DISCARDS the
    vault root when `rel` is absolute, and joins a `../..` verbatim, so a path
    that merely *looks* vault-relative is not a boundary — this is.

    Accepted: ordinary relative paths, and absolute paths under the vault
    (relativized). Rejected: absolute paths outside the vault, any ``..`` that
    escapes, a symlink whose target leaves the vault, and the vault root itself
    (a note is never the directory).

    Containment is decided on the fully resolved paths — a note that is a
    symlink out of the vault escapes on write even though its own path reads as
    vault-relative. Existence is NOT required: resolving a not-yet-created file
    only normalizes, which is what `create()` needs.
    """
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("Empty path is not a valid vault reference")

    root = Path(vault)
    candidate = Path(raw.replace("\\", "/"))
    if not candidate.is_absolute():
        candidate = root / candidate

    real_root = Path(os.path.realpath(root))
    real = Path(os.path.realpath(candidate))
    if not _is_under(real_root, real):
        raise ValueError(
            f"Path {path!r} escapes the vault {root.as_posix()!r}"
        )

    # Prefer the lexical form so an intra-vault symlinked folder keeps the key
    # the index already uses; fall back to the resolved one (an unresolved or
    # symlinked vault root), which is contained by construction.
    #
    # The lexical form is only usable when it still names the file containment
    # was decided on: `normpath` cancels `a/..` textually, which is a LIE as
    # soon as `a` is a symlink. `out/back/../x.md`, with `out` leaving the vault
    # and `back` pointing into it, resolves inside (so it passes above) yet
    # collapses to `out/x.md` — and the caller rejoins that onto the vault root,
    # writing through `out` to the outside. Comparing the two resolutions costs
    # one realpath and closes that gap.
    root_norm = Path(os.path.normpath(str(root)))
    cand_norm = Path(os.path.normpath(str(candidate)))
    if _is_under(root_norm, cand_norm) and Path(os.path.realpath(cand_norm)) == real:
        return cand_norm.relative_to(root_norm).as_posix()
    return real.relative_to(real_root).as_posix()


def to_vault_relative(path: str, *, ensure_md: bool = True) -> str:
    """Normalize an arbitrary note path to POSIX vault-relative form.

    Rules:
      - already-relative paths pass through (POSIX-normalized, leading
        slashes stripped);
      - absolute paths *under* the configured vault root are relativized;
      - absolute paths *outside* the vault, and relative paths whose ``..``
        segments walk out of it, raise ``ValueError`` with a clear diagnostic —
        they would otherwise become a silent "File not found" when the CLI
        fails to resolve them, or worse, a write outside the vault;
      - if ``ensure_md`` is True (default) and the result does not end in
        ``.md``, the extension is appended.

    The vault root is read at call time from ``CONFIG.vault_path`` so test
    fixtures that mutate the singleton continue to work.
    """
    if not path:
        raise ValueError("Empty path is not a valid vault reference")

    p = Path(path)
    vault_str = getattr(CONFIG, "vault_path", None) or ""
    if not vault_str:
        if p.is_absolute():
            raise ValueError(
                f"Absolute path {path!r} provided but SILICA_VAULT is not configured"
            )
        # No vault to contain against: the historical pass-through, unchanged.
        rel = p.as_posix().strip("/")
    else:
        try:
            rel = contain_in_vault(path, Path(vault_str))
        except ValueError as exc:
            # Wording kept verbatim: this diagnostic is what the CLI surfaces to
            # the user for a bad /organize --scope, and it is asserted on.
            raise ValueError(
                f"Path {path!r} is outside the configured vault "
                f"{Path(vault_str).as_posix()!r}"
            ) from exc

    if ensure_md and not rel.endswith(".md"):
        rel += ".md"
    return rel


def is_inbox_path(path: str) -> bool:
    """True when a vault-relative path sits anywhere under an inbox root
    (case-insensitive). The inbox is staging, never a write or merge target —
    callers use this to filter candidates and reject ops.

    EVERY root answers, composed and bare, whichever one `active_inbox_dir`
    happens to resolve to for this vault. The question here is "may an op target
    this?", and the answer is no for both the inbox Silica stages into and the
    one the user keeps — deriving the set from the resolved inbox alone left
    whichever lost the resolution unguarded, and patch ops aimed at it stopped
    being rejected.
    """
    from silica.config import CONFIG
    from silica.kernel.vault_manifest import active_inbox_dir, in_write_dir

    p = path.replace("\\", "/").lstrip("/").casefold()
    inbox = (getattr(CONFIG, "inbox_dir", "") or "").strip("/")
    roots = {active_inbox_dir(), in_write_dir(inbox), inbox, "Inbox"}
    return any(p.startswith(r.casefold() + "/") for r in roots if r)


def resolve_target_dir(target_dir: str) -> str:
    """Fold a user-typed vault folder onto the existing tree, case-insensitively.

    'Informatica/Intelligenza Artificiale' typed against a vault holding
    'Informatica/Intelligenza artificiale' silently forks the tree on a
    case-sensitive filesystem: new-note writes ENOENT through the Obsidian
    bridge and patch paths mismatch their expected collisions. Each segment
    adopts the casing of an existing folder when one matches case-insensitively;
    unmatched segments keep the typed casing (a genuinely new folder).
    Absolute paths and unconfigured vaults pass through untouched.
    """
    vault_str = getattr(CONFIG, "vault_path", None) or ""
    if not target_dir or not vault_str or Path(target_dir).is_absolute():
        return target_dir
    base = Path(vault_str)
    resolved: list[str] = []
    for seg in Path(target_dir.strip("/")).parts:
        cur = base.joinpath(*resolved)
        if not (cur / seg).is_dir() and cur.is_dir():
            seg = next(
                (e.name for e in cur.iterdir()
                 if e.is_dir() and e.name.casefold() == seg.casefold()),
                seg,
            )
        resolved.append(seg)
    return "/".join(resolved)
