# tests/test_codegraph.py
"""kernel/codegraph — derived structural code index (spec-code-lane §1)."""
from pathlib import Path

from silica.kernel.code.codegraph import classify_import, is_first_party

PY_FILES = {
    "silica/__init__.py",
    "silica/kernel/__init__.py",
    "silica/kernel/embed.py",
    "silica/kernel/paths.py",
    "silica/cli.py",
}
TS_FILES = {
    "src/app.ts",
    "src/local/helper.ts",
    "src/lib/index.ts",
}


def test_python_absolute_module():
    kind, val = classify_import("silica.kernel.paths", "silica/cli.py", PY_FILES, "python", Path("."))
    assert (kind, val) == ("resolved", "silica/kernel/paths.py")


def test_python_from_import_name_backs_off_to_module():
    # from silica.kernel import paths → "silica.kernel.paths" resolves to the module file
    kind, val = classify_import("silica.kernel.embed", "silica/cli.py", PY_FILES, "python", Path("."))
    assert (kind, val) == ("resolved", "silica/kernel/embed.py")
    # from silica.kernel import SOMETHING_IN_INIT → falls back to the package __init__
    kind, val = classify_import("silica.kernel.CONFIG", "silica/cli.py", PY_FILES, "python", Path("."))
    assert (kind, val) == ("resolved", "silica/kernel/__init__.py")


def test_python_relative():
    kind, val = classify_import(".paths.atomic_write_bytes", "silica/kernel/embed.py", PY_FILES, "python", Path("."))
    assert (kind, val) == ("resolved", "silica/kernel/paths.py")
    kind, val = classify_import("..cli", "silica/kernel/embed.py", PY_FILES, "python", Path("."))
    assert (kind, val) == ("resolved", "silica/cli.py")


def test_python_external_and_unresolved(tmp_path):
    (tmp_path / "silica").mkdir()
    kind, val = classify_import("numpy.linalg", "silica/cli.py", PY_FILES, "python", tmp_path)
    assert (kind, val) == ("external", "numpy")
    # first-party (silica/ dir exists on disk) but no matching file → unresolved, counted.
    # 3-segment so back-off stops at silica/ghost/__init__.py (absent), never the silica
    # package __init__ — a genuinely unresolvable first-party import (cf. Task 4 pkg.ghost.nope).
    kind, val = classify_import("silica.ghost.deep", "silica/cli.py", PY_FILES, "python", tmp_path)
    assert (kind, val) == ("unresolved", "silica.ghost.deep")


def test_ts_relative_with_extension_inference():
    kind, val = classify_import("./local/helper", "src/app.ts", TS_FILES, "typescript", Path("."))
    assert (kind, val) == ("resolved", "src/local/helper.ts")
    kind, val = classify_import("./lib", "src/app.ts", TS_FILES, "typescript", Path("."))
    assert (kind, val) == ("resolved", "src/lib/index.ts")


def test_ts_bare_external_and_alias_unresolved():
    kind, val = classify_import("react", "src/app.ts", TS_FILES, "typescript", Path("."))
    assert (kind, val) == ("external", "react")
    kind, val = classify_import("@/lib/x", "src/app.ts", TS_FILES, "typescript", Path("."))
    assert (kind, val) == ("unresolved", "@/lib/x")


def test_moved_helpers_still_work(tmp_path):
    (tmp_path / "silica" / "kernel").mkdir(parents=True)
    assert is_first_party("silica.kernel.embed", tmp_path)
    assert not is_first_party("numpy", tmp_path)


import subprocess


from silica.kernel.code import codegraph


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


def _seed_mini_repo(root: Path) -> None:
    """3-file py repo with cross imports (spec §8 fixture)."""
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "paths.py").write_text(
        "import os\n\ndef norm(p: str) -> str:\n    return p\n", encoding="utf-8")
    (root / "pkg" / "embed.py").write_text(
        "from .paths import norm\nimport numpy\n\nclass Embedder:\n    pass\n", encoding="utf-8")
    (root / "main.py").write_text(
        "from pkg import embed\nfrom pkg.ghost import nope\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=root, check=True)


def test_build_resolves_edges_external_unresolved(tmp_path):
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    g = codegraph.build_codegraph(tmp_path)
    assert g.files["pkg/embed.py"]["imports"] == ["pkg/paths.py"]
    assert g.files["pkg/embed.py"]["external"] == ["numpy"]
    assert g.files["main.py"]["imports"] == ["pkg/embed.py"]
    assert g.files["main.py"]["unresolved"] == ["pkg.ghost.nope"]
    assert g.fan_in("pkg/paths.py") == 1
    assert g.importers("pkg/embed.py") == ["main.py"]
    syms = {s["name"] for s in g.files["pkg/embed.py"]["symbols"]}
    assert "Embedder" in syms


def test_build_is_deterministic_byte_for_byte(tmp_path):
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    a = codegraph._serialize(codegraph.build_codegraph(tmp_path))
    b = codegraph._serialize(codegraph.build_codegraph(tmp_path))
    assert a == b


def test_load_codegraph_none_outside_repo(tmp_path):
    assert codegraph.load_codegraph(tmp_path) is None


def test_load_rebuilds_on_head_move_and_mtime(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    store = tmp_path / "cg.json"
    monkeypatch.setattr(codegraph, "store_path", lambda: store)
    g1 = codegraph.load_codegraph(tmp_path)
    assert store.exists() and g1 is not None
    # valid store → served from disk (marker: mutate the file set → invalid)
    (tmp_path / "new.py").write_text("x = 1\n", encoding="utf-8")  # untracked, supported
    g2 = codegraph.load_codegraph(tmp_path)
    assert "new.py" in g2.files  # file-set mismatch forced a full rebuild


def test_parse_error_file_present_never_aborts(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    (tmp_path / "bad.py").write_text("def x(: pass", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "c"], cwd=tmp_path, check=True)
    g = codegraph.build_codegraph(tmp_path)
    assert "bad.py" in g.files  # tree-sitter is error-tolerant: entry exists either way


def test_notebook_is_a_file_node(tmp_path):
    import json as _json
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    nb = _json.dumps({"nbformat": 4, "metadata": {"kernelspec": {"language": "python"}},
                      "cells": [{"cell_type": "code",
                                 "source": "from pkg.paths import norm\nimport pandas\n"}]})
    (tmp_path / "analysis.ipynb").write_text(nb, encoding="utf-8")
    g = codegraph.build_codegraph(tmp_path)
    assert g.files["analysis.ipynb"]["imports"] == ["pkg/paths.py"]
    assert g.files["analysis.ipynb"]["external"] == ["pandas"]
    assert g.fan_in("pkg/paths.py") == 2  # embed.py + the notebook


def test_malformed_notebook_gets_parse_error_entry(tmp_path):
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    (tmp_path / "bad.ipynb").write_text("{not json", encoding="utf-8")
    g = codegraph.build_codegraph(tmp_path)  # build never aborts (spec §1)
    assert g.files["bad.ipynb"]["parse_error"] is True
    assert g.files["bad.ipynb"]["symbols"] == []


def test_code_vocabulary_top_fan_in(tmp_path):
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    g = codegraph.build_codegraph(tmp_path)
    vocab = codegraph.code_vocabulary(g, cap=2)
    # top-2 by fan-in: pkg/paths.py and pkg/embed.py (1 importer each, path tiebreak)
    assert "paths" in vocab and "norm" in vocab        # module stem + symbol
    assert "embed" in vocab and "Embedder" in vocab
    assert "__init__" not in vocab                      # noise stems excluded
    assert vocab == list(dict.fromkeys(vocab))          # deduped, stable order


# ---------------------------------------------------------------------------
# Task 4: store v2 — import-scoped call edges
# ---------------------------------------------------------------------------

from silica.kernel.code.codegraph import build_codegraph


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_call_edges_resolved_bare_dotted_alias(tmp_path):
    _init_repo(tmp_path)
    root = tmp_path
    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/util.py", "def helper():\n    pass\n")
    _write(root, "pkg/alias_target.py", "def go():\n    pass\n")
    _write(root, "pkg/app.py", (
        "import os\n"
        "from pkg.util import helper\n"
        "from pkg import util\n"
        "import pkg.alias_target as at\n\n"
        "def main():\n"
        "    helper()\n"
        "    util.helper()\n"
        "    at.go()\n"
        "    os.path.join('a')\n"
        "    local()\n\n"
        "def local():\n    pass\n"
    ))
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=root, check=True)
    graph = build_codegraph(root)
    edges = graph.files["pkg/app.py"]["calls"]
    assert {"target": "pkg/util.py", "callee": "helper", "caller": "main"} in edges
    assert {"target": "pkg/alias_target.py", "callee": "go", "caller": "main"} in edges
    # bare helper() and util.helper() dedupe to one edge
    assert len([e for e in edges if e["target"] == "pkg/util.py"]) == 1
    # external (os.path.join) never becomes an edge
    assert all(e["target"].startswith("pkg/") for e in edges)
    assert not any(e["callee"] == "join" for e in edges)
    # local() IS an edge: defined in this same file, so it is first-party
    assert {"target": "pkg/app.py", "callee": "local", "caller": "main"} in edges
    assert ("pkg/app.py", "pkg/util.py", "helper", "main") in graph.call_edges()


# ---------------------------------------------------------------------------
# Languages spec §4: Java import resolution + call edges
# ---------------------------------------------------------------------------

JAVA_FILES = {
    "src/main/java/com/example/util/Helper.java",
    "src/main/java/com/example/app/App.java",
}


def test_java_suffix_resolution_absorbs_src_main_java(tmp_path):
    kind, val = classify_import(
        "com.example.util.Helper", "src/main/java/com/example/app/App.java",
        JAVA_FILES, "java", tmp_path)
    assert (kind, val) == ("resolved", "src/main/java/com/example/util/Helper.java")


def test_java_multiple_matches_shortest_path_wins(tmp_path):
    files = {"a/com/foo/Bar.java", "vendored/deep/com/foo/Bar.java"}
    kind, val = classify_import("com.foo.Bar", "a/com/foo/Main.java", files, "java", tmp_path)
    assert (kind, val) == ("resolved", "a/com/foo/Bar.java")


def test_java_wildcard_import_unresolved(tmp_path):
    kind, val = classify_import(
        "com.example.io.*", "src/main/java/com/example/app/App.java",
        JAVA_FILES, "java", tmp_path)
    assert (kind, val) == ("unresolved", "com.example.io.*")


def test_java_zero_match_first_segment_dir_is_unresolved(tmp_path):
    (tmp_path / "com").mkdir()
    kind, val = classify_import("com.ghost.Thing", "App.java", JAVA_FILES, "java", tmp_path)
    assert (kind, val) == ("unresolved", "com.ghost.Thing")


def test_java_stdlib_is_external_despite_the_src_main_java_root(tmp_path):
    """`src/main/java/` is the standard Maven/Gradle source root, so the
    first-party directory probe sees `/java/` and would claim the whole JDK."""
    for mod, top in (("java.util.List", "java.util"), ("javax.swing.JFrame", "javax.swing")):
        kind, val = classify_import(
            mod, "src/main/java/com/example/app/App.java", JAVA_FILES, "java", tmp_path)
        assert (kind, val) == ("external", top)


def test_java_external_labeled_with_two_segments(tmp_path):
    kind, val = classify_import(
        "org.springframework.boot.SpringApplication", "App.java", JAVA_FILES, "java", tmp_path)
    assert (kind, val) == ("external", "org.springframework")


def test_java_build_import_and_call_edges(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "src/main/java/com/ex/util/Helper.java",
           "package com.ex.util;\npublic class Helper {\n"
           "    public static void assist() {}\n}\n")
    _write(tmp_path, "src/main/java/com/ex/app/App.java",
           "package com.ex.app;\n\nimport com.ex.util.Helper;\n"
           "import org.springframework.boot.SpringApplication;\n\n"
           "public class App {\n    public static void main(String[] args) {\n"
           "        Helper.assist();\n        SpringApplication.run();\n    }\n}\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)
    g = build_codegraph(tmp_path)
    app = g.files["src/main/java/com/ex/app/App.java"]
    assert app["imports"] == ["src/main/java/com/ex/util/Helper.java"]
    assert app["external"] == ["org.springframework"]
    assert app["has_main_guard"] is True
    assert {"target": "src/main/java/com/ex/util/Helper.java",
            "callee": "assist", "caller": "App"} in app["calls"]


# ---------------------------------------------------------------------------
# Languages spec §4: C/C++ include resolution + graph-level call join
# ---------------------------------------------------------------------------

C_FILES = {"src/app/main.c", "src/util/helper.h", "src/util/helper.c", "vendor/x/src/util/helper.h"}


def test_c_quoted_include_importer_dir_relative(tmp_path):
    kind, val = classify_import('"util/helper.h"', "src/main.c",
                                {"src/util/helper.h", "src/main.c"}, "c", tmp_path)
    assert (kind, val) == ("resolved", "src/util/helper.h")


def test_c_quoted_include_root_relative_then_suffix(tmp_path):
    kind, val = classify_import('"src/util/helper.h"', "src/app/main.c", C_FILES, "c", tmp_path)
    assert (kind, val) == ("resolved", "src/util/helper.h")
    # suffix match (shortest path wins over the vendored copy)
    kind, val = classify_import('"util/helper.h"', "other/place.c", C_FILES, "c", tmp_path)
    assert (kind, val) == ("resolved", "src/util/helper.h")


def test_c_angled_include_external_with_text_label(tmp_path):
    kind, val = classify_import("<stdio.h>", "src/app/main.c", C_FILES, "c", tmp_path)
    assert (kind, val) == ("external", "stdio.h")


def test_c_quoted_unresolvable_is_unresolved(tmp_path):
    kind, val = classify_import('"ghost/nope.h"', "src/app/main.c", C_FILES, "c", tmp_path)
    assert (kind, val) == ("unresolved", "ghost/nope.h")


def test_c_call_edge_through_direct_resolved_include(tmp_path):
    _init_repo(tmp_path)
    _write(tmp_path, "util/helper.h", "int assist(int x);\n")
    _write(tmp_path, "util/helper.c",
           '#include "helper.h"\n\nint assist(int x) { return x; }\n')
    _write(tmp_path, "main.c",
           '#include "util/helper.h"\n#include <stdio.h>\n\n'
           "int main(void) {\n    assist(1);\n    printf(\"x\");\n    return 0;\n}\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)
    g = build_codegraph(tmp_path)
    main = g.files["main.c"]
    assert main["imports"] == ["util/helper.h"]
    assert main["external"] == ["stdio.h"]
    # graph-level join: assist is a symbol of the directly included helper.h
    assert {"target": "util/helper.h", "callee": "assist", "caller": "main"} in main["calls"]
    # printf never becomes an edge (no resolved include carries it)
    assert all(e["callee"] != "printf" for e in main["calls"])


def test_bare_file_enters_graph_without_edges(tmp_path):
    _init_repo(tmp_path)
    _seed_mini_repo(tmp_path)
    _write(tmp_path, "site/index.html", "<html><body>hi</body></html>\n")
    _write(tmp_path, "config.toml", "[tool]\nname = 'x'\n")
    g = build_codegraph(tmp_path)
    for rel in ("site/index.html", "config.toml"):
        entry = g.files[rel]
        assert entry["parse_error"] is False
        assert entry["imports"] == [] and entry["calls"] == []
        assert entry["symbols"] == []


def test_call_edge_survives_external_import_shadowing(tmp_path):
    # `import yamlmod` (unresolvable) prefix-matches yamlmod.load() first; the
    # resolver must fall through to the first-party `from pkg import yamlmod`
    # instead of silently dropping the edge
    _init_repo(tmp_path)
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/yamlmod.py", "def load():\n    pass\n")
    _write(tmp_path, "app.py", (
        "import yamlmod\n"
        "from pkg import yamlmod\n\n"
        "def main():\n"
        "    yamlmod.load()\n"
    ))
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)
    graph = build_codegraph(tmp_path)
    edges = graph.files["app.py"]["calls"]
    assert {"target": "pkg/yamlmod.py", "callee": "load", "caller": "main"} in edges


def test_ts_reexport_resolves_to_its_source(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "codec.ts").write_text(
        "export function parse() {}\nexport function format() {}\n", encoding="utf-8")
    (tmp_path / "index.ts").write_text(
        "export { parse, format as fmt } from './codec';\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)
    g = codegraph.build_codegraph(tmp_path)
    entry = g.files["index.ts"]
    assert entry["imports"] == ["codec.ts"]
    assert {(r["name"], r["from"]) for r in entry["reexports"]} == {
        ("parse", "codec.ts"), ("fmt", "codec.ts")}
