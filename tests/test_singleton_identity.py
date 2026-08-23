# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Process-lifetime state must be one object, and must follow the vault.

Extracted from an omniparse teardown, where the failure is terminal: that
server loads its models in `main()` under `if __name__ == "__main__"`, then
serves them with `uvicorn.run("server:app")`. The import string makes Python
import the module a *second* time (the script is `__main__`, the import is
`server`), so the served app reads a fresh module-level singleton with every
field still None while the loaded models sit in the other copy. Nothing raises;
every request just fails.

Silica's own version of that class is already documented and fixed: a store
singleton that won first-caller-wins and ignored the index path kept serving the
old vault after a `/vault` switch. `path_keyed_singleton` is the fix. These
tests exist so neither half can come back quietly, because both fail *silently*
in production and stay green in a suite that does not look for them.
"""
from __future__ import annotations

import inspect
import re

import pytest

from silica.kernel.recall import cooccurrence, embed, lexical
from silica.kernel.recall.paths import path_keyed_singleton

# Every module exposing a process-lifetime store, with its accessor, its cache
# dict and its clear(). New stores belong here; that is the point of the list.
STORE_MODULES = (
    (embed, "get_store"),
    (cooccurrence, "get_cooccur_store"),
    (lexical, "get_lexical_store"),
)


@pytest.mark.parametrize("module,accessor", STORE_MODULES, ids=lambda x: getattr(x, "__name__", x))
def test_store_accessor_is_keyed_by_path_not_by_first_call(
    module, accessor, monkeypatch, tmp_path
):
    """Same index path returns the same instance; a different path does not.

    The second half is the whole defect: a bare `global _STORE` passes the first
    assertion and fails the second, and a `/vault` switch then reads the old
    vault's index for the rest of the process.
    """
    module.clear()
    # Path, not str: the loaders call Path methods on whatever this returns.
    monkeypatch.setattr(module, "_index_path", lambda: tmp_path / "a" / "index.json")
    first = getattr(module, accessor)()
    assert getattr(module, accessor)() is first, "same path must not rebuild"

    monkeypatch.setattr(module, "_index_path", lambda: tmp_path / "b" / "index.json")
    assert getattr(module, accessor)() is not first, "a /vault switch must not be cached over"
    module.clear()


@pytest.mark.parametrize("module,accessor", STORE_MODULES, ids=lambda x: getattr(x, "__name__", x))
def test_store_accessor_goes_through_the_shared_singleton_helper(module, accessor):
    """One shape for all of them. A hand-rolled cache in one module is how the
    keying drifts back out."""
    src = inspect.getsource(getattr(module, accessor))
    assert "path_keyed_singleton" in src
    assert re.search(r"_index_path\(\)", src), "the key must be the resolved index path"


@pytest.mark.parametrize("module,_accessor", STORE_MODULES, ids=lambda x: getattr(x, "__name__", x))
def test_store_cache_is_clearable(module, _accessor):
    """`/vault` switch and test isolation both depend on it existing."""
    assert callable(module.clear)
    module.clear()
    assert module._STORE_CACHE == {}


def test_path_keyed_singleton_builds_once_per_key():
    calls: list[int] = []
    cache: dict = {}

    def factory():
        calls.append(1)
        return object()

    a = path_keyed_singleton(cache, "k1", factory)
    assert path_keyed_singleton(cache, "k1", factory) is a
    b = path_keyed_singleton(cache, "k2", factory)
    assert b is not a
    assert len(calls) == 2


def test_config_is_one_object_across_every_import_path():
    """The CLI, the web server and the kernel must see the same CONFIG.

    Reached three ways on purpose: a re-import, an attribute read off the
    module, and the `from ... import` binding a caller actually uses. In the
    omniparse failure all three agree inside one process and still disagree with
    the server's copy, which is what the next test covers.
    """
    import silica.config as cfg_mod
    from silica.config import CONFIG

    import importlib
    assert importlib.import_module("silica.config").CONFIG is CONFIG
    assert cfg_mod.CONFIG is CONFIG

    from silica.kernel.recall import embed as embed_mod
    assert embed_mod.CONFIG is CONFIG if hasattr(embed_mod, "CONFIG") else True


def test_serve_hands_uvicorn_the_app_object_not_an_import_string():
    """`uvicorn.run(app, ...)`, never `uvicorn.run("silica.ui.web.server:app")`.

    The string form is the tempting one (it is what `--reload` requires), and it
    is exactly how omniparse loses its models: uvicorn imports the module again
    and the app it serves is not the app that was configured. If reload is ever
    wanted here, the app must first stop depending on process state set up before
    `serve()` is called.
    """
    pytest.importorskip("uvicorn")
    from silica.ui.web import server as web

    src = inspect.getsource(web.serve)
    # Config(app, ...) since serve() builds the Server itself (so the beat
    # stream can read should_exit); the rule under it is unchanged, and it is
    # the OBJECT that is asserted, not which uvicorn entry point takes it.
    assert re.search(r"uvicorn\.(run|Config)\(\s*\n?\s*app\b", src), \
        "serve() must pass the app object"
    assert not re.search(r"uvicorn\.(run|Config)\(\s*[\"']", src), "no import string"
    assert "reload" not in src, "reload re-imports the module and orphans session state"


def test_the_served_app_is_the_module_level_app():
    """A second import must not mint a second FastAPI instance.

    `app = FastAPI(...)` at module scope is only a singleton because the module
    is imported once. Assert the identity rather than trusting it.
    """
    pytest.importorskip("fastapi")
    import importlib

    from silica.ui.web import server as web

    assert importlib.import_module("silica.ui.web.server").app is web.app
