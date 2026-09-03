"""kernel/codewiki — partition + digest for the behavioral code wiki."""
from silica.kernel.code.codegraph import CodeGraph
from silica.kernel.code.codewiki import (
    _spend_doc_budget, module_roots, partition, source_root)


def _graph(paths, entries=None):
    files = {p: {"language": "python", "imports": [], "external": [],
                 "unresolved": [], "symbols": [], "calls": [],
                 "module_doc": "", "module_comments": [], "dunder_all": None,
                 "has_main_guard": False, "parse_error": False}
             for p in paths}
    for p, extra in (entries or {}).items():
        files[p].update(extra)
    return CodeGraph(head_ref="abc123", files=files)


def test_source_root_is_densest_top_dir():
    g = _graph(["silica/cli.py", "silica/kernel/a.py", "silica/kernel/b.py",
                "tools/x.py"])
    assert source_root(g) == "silica"


def test_source_root_ignores_tests_and_docs_when_counting():
    # test-heavy repo: tests/ outnumbers the real source package, but the
    # source root must still be the code package, never the test suite.
    g = _graph(["silica/cli.py", "silica/kernel/a.py",
                "tests/test_a.py", "tests/test_b.py", "tests/test_c.py",
                "docs/conf.py"])
    assert source_root(g) == "silica"


def test_partition_subdirs_and_core():
    g = _graph(["silica/cli.py", "silica/config.py",
                "silica/kernel/a.py", "silica/kernel/graph/deep.py",
                "silica/router/r.py"])
    subs = {s.key: s for s in partition(g)}
    assert set(subs) == {"(root)", "kernel", "router"}
    assert subs["(root)"].members == ["silica/cli.py", "silica/config.py"]
    # deep files roll up to the immediate subdir
    assert "silica/kernel/graph/deep.py" in subs["kernel"].members


def test_root_key_never_collides_with_a_real_core_dir():
    # a repo with <root>/core/ AND loose files: two distinct subsystems, never
    # silently merged under one key
    g = _graph(["pkg/main.py", "pkg/core/engine.py"])
    subs = {s.key: s for s in partition(g)}
    assert subs["(root)"].members == ["pkg/main.py"]
    assert subs["core"].members == ["pkg/core/engine.py"]


def test_source_root_density_ignores_bare_files():
    # a site/ directory full of HTML must not win source-root over the real
    # code package; bare files under the source root still enter the partition
    g = _graph(["src/a.py", "src/b.py",
                "site/x.html", "site/y.html", "site/z.html",
                "src/config.toml"])
    assert source_root(g) == "src"
    subs = {s.key: s for s in partition(g)}
    assert "src/config.toml" in subs["(root)"].members


def test_source_root_loose_bare_files_do_not_count():
    # loose HTML at the repo root must not drag the source root to ""
    g = _graph(["a.html", "b.html", "c.html", "src/a.py", "src/b.py"])
    assert source_root(g) == "src"


def test_maven_scaffolding_and_root_package_are_not_subsystems():
    # src/main/java says nothing about the design, and neither does the
    # io/github/app root package: the subsystems are the leaf packages.
    g = _graph(["src/main/java/io/github/app/App.java",
                "src/main/java/io/github/app/svc/A.java",
                "src/main/java/io/github/app/svc/B.java",
                "src/main/java/io/github/app/ui/W.java",
                "src/test/java/io/github/app/svc/ATest.java"])
    subs = {s.key: s for s in partition(g)}
    assert set(subs) == {"(root)", "svc", "ui"}
    assert subs["svc"].path == "src/main/java/io/github/app/svc"
    assert subs["(root)"].members == ["src/main/java/io/github/app/App.java"]
    # test sources are not the code, at any depth
    assert all("src/test/" not in m for s in subs.values() for m in s.members)


def test_corridor_descent_stops_at_a_real_fork_and_at_loose_code():
    # a dir holding code of its own is never a corridor, however few files it
    # has: descending past it would hide them
    g = _graph(["src/main/java/io/github/app/App.java",
                "src/main/java/io/github/app/svc/A.java"])
    subs = {s.key: s for s in partition(g)}
    assert set(subs) == {"(root)", "svc"}
    # a bare (non symbol-bearing) sibling must not fork the corridor
    g = _graph(["src/main/java/io/app/svc/A.java", "src/main/java/io/app/ui/W.java",
                "src/main/resources/app.toml"])
    subs = {s.key: s for s in partition(g)}
    assert set(subs) == {"svc", "ui"}


def test_multi_module_repo_keeps_every_module():
    # source_root() elects the densest module; the others must not vanish
    g = _graph(["core/src/main/java/io/gh/g/manager/M.java",
                "core/src/main/java/io/gh/g/entity/P.java",
                "core/src/main/java/io/gh/g/Main.java",
                "desktop/src/main/java/io/gh/g/Launcher.java"])
    assert module_roots(g) == ["core", "desktop"]
    subs = {s.key: s for s in partition(g)}
    assert set(subs) == {"core", "core.entity", "core.manager", "desktop"}
    # keys are module-qualified: one module's "manager" is not another's
    assert subs["core.manager"].path == "core/src/main/java/io/gh/g/manager"
    assert subs["desktop"].members == ["desktop/src/main/java/io/gh/g/Launcher.java"]


def test_single_module_repo_is_not_treated_as_multi_module():
    g = _graph(["src/main/java/io/gh/g/svc/A.java", "src/main/java/io/gh/g/App.java"])
    assert module_roots(g) == []
    assert {s.key for s in partition(g)} == {"(root)", "svc"}


def test_flat_repo_root_excludes_tests_and_docs():
    g = _graph(["app.py", "lib/x.py", "tests/test_x.py", "docs/conf.py"])
    subs = {s.key: s for s in partition(g)}
    assert "tests" not in subs and "docs" not in subs
    assert "(root)" in subs and "lib" in subs


# ---------------------------------------------------------------------------
# Task 6: SubsystemDigest
# ---------------------------------------------------------------------------

from silica.kernel.code.codewiki import build_digests, cross_edges, edges_ref


def _rich_graph():
    return _graph(
        ["silica/cli.py",
         "silica/kernel/util.py", "silica/kernel/other.py",
         "silica/router/r.py"],
        entries={
            "silica/cli.py": {
                "imports": ["silica/kernel/util.py"],
                "calls": [{"target": "silica/kernel/util.py",
                           "callee": "helper", "caller": "main"}],
                "has_main_guard": True,
                "symbols": [{"kind": "function", "name": "main", "parent": "",
                             "signature": "def main()", "doc": "", "doc_full": "",
                             "decorators": []}],
            },
            "silica/router/r.py": {
                "imports": ["silica/kernel/util.py"],
                "symbols": [{"kind": "function", "name": "dispatch", "parent": "",
                             "signature": "def dispatch()", "doc": "", "doc_full": "",
                             "decorators": ["app.command"]}],
            },
            "silica/kernel/util.py": {
                "dunder_all": ["helper"],
                "external": ["orjson"],
                "symbols": [
                    {"kind": "function", "name": "helper", "parent": "",
                     "signature": "def helper()", "doc": "", "doc_full": "", "decorators": []},
                    {"kind": "function", "name": "not_exported", "parent": "",
                     "signature": "def not_exported()", "doc": "", "doc_full": "", "decorators": []},
                ],
            },
        },
    )


def test_digest_collaborators_two_weights_and_publics(tmp_path):
    g = _rich_graph()
    digests = {d.key: d for d in build_digests(g, partition(g), tmp_path)}
    core = digests["(root)"]
    assert ("kernel", 1, 1) in core.collaborators_out       # 1 import edge, 1 call edge
    kernel = digests["kernel"]
    assert ("(root)", 1, 1) in kernel.collaborators_in
    assert ("router", 1, 0) in kernel.collaborators_in      # import-only, zero calls
    syms = kernel.public_symbols["silica/kernel/util.py"]
    # __all__ is the authority on the doc budget, not on inclusion: the
    # unexported symbol still shows up, marked brief
    assert [s["name"] for s in syms] == ["helper", "not_exported"]
    assert {s["name"]: s.get("brief", False) for s in syms} == \
        {"helper": False, "not_exported": True}
    assert "orjson" in kernel.external_deps


def test_digest_entry_points_labeled(tmp_path):
    g = _rich_graph()
    digests = {d.key: d for d in build_digests(g, partition(g), tmp_path)}
    labels = dict(digests["(root)"].entry_points)
    assert "__main__ guard" in labels["silica/cli.py"]
    labels_r = dict(digests["router"].entry_points)
    assert "registration decorator" in labels_r["silica/router/r.py"]


def test_struct_sig_changes_on_body_only_call_change(tmp_path):
    g1 = _rich_graph()
    d1 = {d.key: d for d in build_digests(g1, partition(g1), tmp_path)}
    g2 = _rich_graph()
    g2.files["silica/cli.py"]["calls"] = []   # body-only edit: call removed
    d2 = {d.key: d for d in build_digests(g2, partition(g2), tmp_path)}
    assert d1["(root)"].struct_sig != d2["(root)"].struct_sig
    assert d1["(root)"].members == d2["(root)"].members


def test_struct_sig_changes_on_docstring_only_change(tmp_path):
    # the digest's main prose payload is docs, not structure: a docstring
    # rewrite must flip the sig or the wiki narrates the old behavior forever
    g1 = _rich_graph()
    d1 = {d.key: d for d in build_digests(g1, partition(g1), tmp_path)}
    g2 = _rich_graph()
    g2.files["silica/kernel/util.py"]["module_doc"] = "Now caches results."
    d2 = {d.key: d for d in build_digests(g2, partition(g2), tmp_path)}
    assert d1["kernel"].struct_sig != d2["kernel"].struct_sig
    assert d1["(root)"].struct_sig == d2["(root)"].struct_sig  # untouched subsystem stable


def test_cross_edges_and_ref(tmp_path):
    g = _rich_graph()
    edges = cross_edges(g, partition(g))
    assert ("(root)", "kernel", 1, 1) in edges
    assert ("router", "kernel", 1, 0) in edges
    ref = edges_ref(edges)
    # weight-only change must NOT move the ref
    bumped = [(a, b, iw + 5, cw) for (a, b, iw, cw) in edges]
    assert edges_ref(bumped) == ref


# ---------------------------------------------------------------------------
# Task 7: flow sketches + deterministic mermaid
# ---------------------------------------------------------------------------

from silica.kernel.code.codewiki import flow_sketches, render_mermaid


def test_flow_sketches_deterministic_and_capped():
    adj = {"a.py": ["b.py", "c.py"], "b.py": ["d.py"], "c.py": [], "d.py": []}
    flows = flow_sketches(adj, ["a.py"])
    assert flows == flow_sketches(adj, ["a.py"])          # stable
    assert ["a.py", "b.py", "d.py"] in flows
    assert all(len(f) <= 6 for f in flows)
    assert len(flows) <= 3


def test_flow_sketches_cycle_safe():
    adj = {"a.py": ["b.py"], "b.py": ["a.py"]}
    flows = flow_sketches(adj, ["a.py"])
    assert flows == [["a.py", "b.py"]]


def test_render_mermaid_byte_stable():
    edges = [("router", "kernel", 2, 1), ("core", "kernel", 1, 1)]
    block = render_mermaid(edges)
    assert block.startswith("```mermaid\ngraph LR")
    assert block == render_mermaid(list(reversed(edges)))  # order-insensitive
    # nodes sorted: core=n0, kernel=n1, router=n2
    assert 'n0["core"] --> n1["kernel"]' in block
    assert 'n2["router"] --> n1["kernel"]' in block


def test_wiki_over_mixed_language_fixture(tmp_path):
    # end-to-end: build_codegraph → partition → build_digests on a repo mixing
    # java, c, and bare html; the html file rides along, the code drives
    import subprocess
    from silica.kernel.code.codegraph import build_codegraph
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    def w(rel, text):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    w("src/util/Helper.java",
      "package util;\npublic class Helper {\n    public static void assist() {}\n}\n")
    w("src/app/App.java",
      "package app;\nimport util.Helper;\npublic class App {\n"
      "    public static void main(String[] args) { Helper.assist(); }\n}\n")
    w("src/core/lib.c", "int go(void) { return 0; }\n")
    w("src/app/page.html", "<html></html>\n")
    g = build_codegraph(tmp_path)
    subs = partition(g)
    digests = {d.key: d for d in build_digests(g, subs, tmp_path)}
    assert source_root(g) == "src"
    assert "src/app/page.html" in digests["app"].members   # bare file rides along
    assert ("util", 1, 1) in digests["app"].collaborators_out  # import + call edge
    labels = dict(digests["app"].entry_points)
    assert "__main__ guard" in labels["src/app/App.java"]


def test_render_mermaid_survives_reserved_and_symbol_keys():
    # `end` is a Mermaid reserved word; `(root)` carries non-word characters —
    # both must render as labels on enumerated ids, never as bare node ids
    block = render_mermaid([("(root)", "end", 1, 0)])
    assert '["(root)"] --> ' in block and '["end"]' in block
    for line in block.splitlines():
        assert not line.strip().startswith("end")


def test_library_subsystem_has_no_entry_points_but_keeps_flows(tmp_path):
    """kernel/ has no main guard, no script, no registration decorator: the
    honest answer is an empty entry-point list. The flow section still shows
    how it is reached, seeded from the caller that enters it."""
    g = _rich_graph()
    digests = {d.key: d for d in build_digests(g, partition(g), tmp_path)}
    kernel = digests["kernel"]
    assert kernel.entry_points == []
    assert ["silica/cli.py", "silica/kernel/util.py"] in kernel.flow_sketches


def test_flow_prefixes_extend_instead_of_restarting():
    adj = {"a.py": ["b.py"], "b.py": ["c.py"]}
    assert flow_sketches(adj, [], prefixes=[["z.py", "a.py"]]) == \
        [["z.py", "a.py", "b.py", "c.py"]]


def test_doc_budget_trims_the_tail_and_spares_the_privileged():
    class _G:
        def fan_in(self, p):
            return {"hub.py": 9, "mid.py": 5, "tail.py": 0, "entry.py": 0}[p]

    def syms(n):
        return [{"kind": "function", "name": f"f{i}", "signature": "f()",
                 "doc": "one line", "doc_full": "x" * 5000} for i in range(n)]

    publics = {p: syms(4) for p in ("hub.py", "mid.py", "tail.py", "entry.py")}
    trimmed = _spend_doc_budget(publics, _G(), privileged={"entry.py"})

    def full(p):
        return [s for s in publics[p] if not s.get("brief")]

    assert full("entry.py") == publics["entry.py"]   # privileged is spent first
    assert full("hub.py")                            # highest fan-in keeps prose
    assert not full("tail.py")                       # budget exhausted
    assert "tail.py" in trimmed and "entry.py" not in trimmed
    # nothing vanishes: every symbol still carries its signature
    assert all(s["signature"] for s in publics["tail.py"])


def test_doc_budget_leaves_a_small_subsystem_untouched():
    class _G:
        def fan_in(self, p):
            return 1

    publics = {"a.py": [{"name": "f", "signature": "f()", "doc": "d",
                         "doc_full": "long"}]}
    assert _spend_doc_budget(publics, _G(), privileged=set()) == set()
    assert not publics["a.py"][0].get("brief")


def _files(*paths):
    return _graph(list(paths))


def test_oversized_subsystem_descends_one_level():
    """93 files under one key is a catalogue; its own subdirs name the split."""
    big = [f"silica/kernel/{lane}/m{i}.py"
           for lane in ("write", "recall", "code") for i in range(15)]
    loose = ["silica/kernel/workqueue.py", "silica/kernel/progress.py"]
    g = _graph([*big, *loose, "silica/cli.py"])
    keys = {s.key: len(s.members) for s in partition(g)}
    assert keys == {"kernel.write": 15, "kernel.recall": 15,
                    "kernel.code": 15, "kernel": 2, "(root)": 1}


def test_small_subsystem_is_left_whole():
    g = _graph([f"silica/kernel/{d}/m.py" for d in "abc"] + ["silica/cli.py"])
    assert {s.key for s in partition(g)} == {"kernel", "(root)"}


def test_big_but_flat_package_does_not_descend():
    """No subdirectories below means nothing to descend into."""
    g = _graph([f"silica/kernel/m{i}.py" for i in range(45)] + ["silica/cli.py"])
    assert {s.key: len(s.members) for s in partition(g)} == {"kernel": 45, "(root)": 1}
