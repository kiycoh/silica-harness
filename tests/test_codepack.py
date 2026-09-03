# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""kernel/codepack - deterministic context pack for one file (spec-code-recall)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from silica.kernel.code import codegraph, codepack

GAME_MODEL = """package game;

import java.util.List;
import util.Vec2;

public class GameModel extends Entity {
    private Vec2 pos;

    public void tick() {
        pos = new Vec2(1, 2);
    }
}
"""

ENTITY = """package game;

public class Entity {
    public void update() {
    }
}
"""

HUD = """package game;

public class Hud {
    public void draw() {
    }
}
"""

VEC2 = """package util;

public class Vec2 {
    public Vec2(int x, int y) {
    }

    public int len() {
        return 0;
    }
}
"""

LAUNCHER = """package app;

import game.GameModel;

public class Launcher {
    public static void main(String[] args) {
        GameModel m = new GameModel();
    }
}
"""

TARGET = "src/main/java/game/GameModel.java"


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A four-package Java repo, git-committed, with an isolated graph store."""
    from silica.kernel.recall import paths

    paths.clear_repo_root_cache()
    monkeypatch.setattr(codegraph, "store_path", lambda: tmp_path / "cg.json")
    _init_repo(tmp_path)
    _write(tmp_path, TARGET, GAME_MODEL)
    _write(tmp_path, "src/main/java/game/Entity.java", ENTITY)
    _write(tmp_path, "src/main/java/game/Hud.java", HUD)
    _write(tmp_path, "src/main/java/util/Vec2.java", VEC2)
    _write(tmp_path, "src/main/java/app/Launcher.java", LAUNCHER)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)
    yield tmp_path
    paths.clear_repo_root_cache()


def test_target_is_verbatim_when_it_fits(repo):
    pack = codepack.code_pack(repo, TARGET)
    assert pack["target_mode"] == "verbatim"
    assert pack["truncated"] is False
    assert GAME_MODEL.rstrip("\n") in pack["text"]
    assert pack["text"].startswith(f"## target {TARGET} @ {pack['head_ref']} mode: verbatim\n")
    assert pack["sections"]["target"] == [TARGET]
    assert pack["head_ref"]


def test_degrades_outside_a_git_repo(tmp_path, monkeypatch):
    from silica.kernel.recall import paths

    paths.clear_repo_root_cache()
    monkeypatch.setattr(codegraph, "store_path", lambda: tmp_path / "cg.json")
    _write(tmp_path, "lonely.py", "def f():\n    return 1\n")
    try:
        pack = codepack.code_pack(tmp_path, "lonely.py")
    finally:
        paths.clear_repo_root_cache()
    assert pack["target_mode"] == "verbatim"
    assert "def f():" in pack["text"]
    assert pack["head_ref"] == ""
    assert any("no code graph" in d for d in pack["dropped"])


def test_unreadable_target_raises(repo):
    with pytest.raises(ValueError):
        codepack.code_pack(repo, "src/main/java/game/Ghost.java")


def test_target_over_budget_falls_back_to_outline(repo):
    pack = codepack.code_pack(repo, TARGET, budget_chars=120)
    assert pack["target_mode"] == "outline"
    assert pack["truncated"] is True
    assert "public class GameModel extends Entity" in pack["text"]
    assert "public void tick()" in pack["text"]
    assert "pos = new Vec2(1, 2);" not in pack["text"]  # bodies are gone


def test_outline_is_served_even_when_it_busts_the_budget(repo):
    pack = codepack.code_pack(repo, TARGET, budget_chars=10)
    assert pack["target_mode"] == "outline"
    assert "public class GameModel extends Entity" in pack["text"]  # never less than the target


def test_no_symbols_means_no_outline_to_fall_back_to(repo):
    _write(repo, "notes.txt", "x" * 500)
    pack = codepack.code_pack(repo, "notes.txt", budget_chars=50)
    assert pack["target_mode"] == "verbatim"  # an empty outline is worse than a long file
    assert "xxx" in pack["text"]


def test_outline_skip_bare_name_only_drops_the_addressed_symbol():
    # skip="Foo" must drop the top-level symbol named Foo and its own
    # members, but an unrelated method that merely shares the name Foo
    # (here, a member of a different class, Bar.Foo) must survive.
    entry = {
        "symbols": [
            {"kind": "class", "name": "Foo", "parent": "", "signature": "class Foo", "doc": ""},
            {"kind": "method", "name": "bar", "parent": "Foo", "signature": "void bar()", "doc": ""},
            {"kind": "class", "name": "Bar", "parent": "", "signature": "class Bar", "doc": ""},
            {"kind": "method", "name": "Foo", "parent": "Bar", "signature": "void Foo()", "doc": ""},
        ]
    }
    outline = codepack._outline(entry, skip="Foo")
    assert "class Foo" not in outline
    assert "void bar()" not in outline
    assert "class Bar" in outline
    assert "void Foo()" in outline  # unrelated Bar.Foo(), not a member of Foo


def test_outline_drops_the_members_of_a_filtered_out_inner_class():
    # Review finding 2: `_outline`'s skip was one level deep while
    # `_signatures` was already transitive. With a two-level entry the members
    # of a skipped inner class used to be re-parented onto the outer class
    # (a false fact, and a duplicate of text already served verbatim above),
    # or left as orphan indented lines with no owner at all.
    entry = {
        "symbols": [
            {"kind": "class", "name": "Outer", "parent": "",
             "signature": "public class Outer", "doc": ""},
            {"kind": "class", "name": "Builder", "parent": "Outer",
             "signature": "static class Builder", "doc": ""},
            {"kind": "method", "name": "withA", "parent": "Builder",
             "signature": "public Builder withA()", "doc": ""},
            {"kind": "method", "name": "build", "parent": "Builder",
             "signature": "public Outer build()", "doc": ""},
        ]
    }
    assert codepack._outline(entry, skip="Outer.Builder") == "public class Outer"
    assert codepack._outline(entry, skip="Outer") == ""


def test_outline_skip_does_not_poison_a_same_named_class_elsewhere():
    # The closure is keyed by bare name, the only key `parent` ever carries,
    # so a surviving symbol must re-open its own name: two unrelated classes
    # can share a simple name, and skipping `Outer.Builder` must not silence
    # the unrelated `Other.Builder`'s members.
    entry = {
        "symbols": [
            {"kind": "class", "name": "Outer", "parent": "",
             "signature": "public class Outer", "doc": ""},
            {"kind": "class", "name": "Builder", "parent": "Outer",
             "signature": "static class Builder", "doc": ""},
            {"kind": "method", "name": "withA", "parent": "Builder",
             "signature": "public Builder withA()", "doc": ""},
            {"kind": "class", "name": "Other", "parent": "",
             "signature": "public class Other", "doc": ""},
            {"kind": "class", "name": "Builder", "parent": "Other",
             "signature": "static class Builder", "doc": ""},
            {"kind": "method", "name": "step2", "parent": "Builder",
             "signature": "public Other step2()", "doc": ""},
        ]
    }
    outline = codepack._outline(entry, skip="Outer.Builder")
    assert "withA" not in outline
    assert "public Other step2()" in outline


def test_selector_serves_one_method_verbatim_and_the_rest_as_outline(repo):
    pack = codepack.code_pack(repo, f"{TARGET}#GameModel.tick")
    assert pack["target_mode"] == "symbol"
    assert pack["truncated"] is True
    assert "pos = new Vec2(1, 2);" in pack["text"]          # the selected body, verbatim
    assert "-- rest of file, outline --" in pack["text"]
    assert "public class GameModel extends Entity" in pack["text"]
    assert pack["text"].count("public void tick()") == 1    # not repeated in the outline


def test_selector_on_a_class_serves_the_whole_class(repo):
    pack = codepack.code_pack(repo, "src/main/java/util/Vec2.java#Vec2")
    assert pack["target_mode"] == "symbol"
    assert "public int len()" in pack["text"]
    assert "return 0;" in pack["text"]


def test_unknown_selector_degrades_to_the_whole_file(repo):
    pack = codepack.code_pack(repo, f"{TARGET}#GameModel.nosuch")
    assert pack["target_mode"] == "verbatim"
    assert any("nosuch" in d for d in pack["dropped"])


def test_python_top_level_function_selector(repo):
    _write(repo, "tool.py", "import os\n\n\ndef alpha():\n    return os.sep\n\n\ndef beta():\n    return 2\n")
    pack = codepack.code_pack(repo, "tool.py#alpha")
    assert pack["target_mode"] == "symbol"
    assert "return os.sep" in pack["text"]
    assert "return 2" not in pack["text"]


def test_hierarchy_lists_declared_bases_and_repo_subtypes(repo):
    pack = codepack.code_pack(repo, TARGET)
    assert "GameModel extends Entity" in pack["text"]
    assert pack["sections"]["hierarchy"] == ["GameModel"]

    sub = codepack.code_pack(repo, "src/main/java/game/Entity.java")
    assert f"Entity <- {TARGET}#GameModel" in sub["text"]
    assert sub["sections"]["hierarchy"] == [f"{TARGET}#GameModel"]


# One pure function, one table: each comment heads the regression its rows pin.
@pytest.mark.parametrize("signature, expected", [
    # the four families, as declared
    ("public class A extends B implements C, D<E>", ["B", "C", "D"]),
    ("class A(B, C, metaclass=M)", ["B", "C"]),
    ("class A : public B, private C", ["B", "C"]),
    ("public class A", []),
    ("public class A implements C, D", ["C", "D"]),
    ("public class A extends B", ["B"]),
    ("public class Foo extends Base<T>", ["Base"]),
    # a multi-parameter generic base's inner commas are not base separators:
    # `HashMap<K, V>` is one base, not two
    ("public class Foo extends HashMap<K, V>", ["HashMap"]),
    ("class A extends B<C, D> implements E", ["B", "E"]),
    ("class A(B[C, D])", ["B"]),
    ("struct A : public B<C, D>", ["B"]),
    # a qualified base names the class, not its package or enclosing type
    ("public class Foo extends com.example.Base", ["Base"]),
    ("public class A extends Map.Entry", ["Entry"]),
    # Review finding 1, two independent fabrications in one function.
    # (a) the C++ base regex ran on every family with no discriminator and
    #     `[^{]+` ran to end of line, so a Python declaration line with a
    #     trailing comment read the comment as its base list.
    # (b) `_IDENT` had no `:`, so `std::runtime_error` tokenized as two
    #     identifiers and the first-identifier rule picked the NAMESPACE,
    #     losing every real base and inventing one that does not exist.
    ("class Config:  # noqa: D101", []),
    ("class Alone:  # a plain marker class", []),
    ("class Plain:", []),
    ("class MyError : public std::runtime_error", ["runtime_error"]),
    ("class Two : public ns::A, public ns::B", ["A", "B"]),
    # a bare `/` is division, not a comment. Excluding the character rather
    # than the `//` and `/*` tokens cut the closing `>` off the template span,
    # so the generic strip could not fire and the surviving comma split one
    # base into two: the fabrication this function exists to prevent.
    ("class A : public Matrix<int, N/2>", ["Matrix"]),
    ("class A : public Matrix<W/2, H/2>", ["Matrix"]),
    ("class A : public Array<T, SIZE/2>, public Base", ["Array", "Base"]),
    # a trailing comment is cut in every family; without it the Java clause
    # reads `D` and `E` as declared bases
    ("class A : public B // note", ["B"]),
    ("class A : public B /* note */", ["B"]),
    ("class A extends B implements C // see D, E", ["B", "C"]),
    # a declaration wrapped after the separator leaves it trailing. The
    # last-segment rule reduced that to "" and dropped a base right there in
    # the text.
    ("public class A extends B.", ["B"]),
    ("class A : public ns::", ["ns"]),
])
def test_supertypes(signature, expected):
    assert codepack._supertypes(signature) == expected


def test_hierarchy_section_is_absent_when_empty(repo):
    pack = codepack.code_pack(repo, "src/main/java/game/Hud.java")
    assert "## hierarchy" not in pack["text"]
    assert "hierarchy" not in pack["sections"]


def test_selector_prefers_the_shallowest_match_over_a_nested_shadow():
    # A nested class named GameModel (inside Wrapper) shadows the real,
    # top-level GameModel in document order. The selector must resolve to
    # the shallowest (top-level) declaration, never the nested one.
    src = (
        "class Wrapper {\n"
        "    class GameModel {\n"
        "        void tick() { int decoy = 999; }\n"
        "    }\n"
        "}\n"
        "\n"
        "public class GameModel extends Entity {\n"
        "    public void tick() { int real = 1; }\n"
        "}\n"
    )
    picked = codepack._symbol_source(src, "java", "GameModel.tick")
    assert picked is not None
    assert "real = 1" in picked
    assert "decoy" not in picked


def test_neighborhood_has_imports_first_then_mentioned_package_siblings(repo):
    pack = codepack.code_pack(repo, TARGET)
    assert pack["sections"]["neighborhood"] == [
        "src/main/java/util/Vec2.java",     # resolved import, crosses a package
        "src/main/java/game/Entity.java",   # package sibling, no import needed
    ]
    assert "src/main/java/game/Hud.java" not in pack["sections"]["neighborhood"]
    assert "public int len()" in pack["text"]      # neighbour signatures are there
    assert "return 0;" not in pack["text"]         # neighbour bodies are not


def test_python_siblings_never_enter_even_when_mentioned(repo):
    # Minor A: pkg/b.py needs a real import too (pkg/c.py), so the
    # neighbourhood section actually exists and is non-empty. Without that,
    # `pack["sections"].get("neighborhood", [])` returns `[]` whether the
    # feature works or does not exist at all, and the assertion below cannot
    # discriminate between the two.
    _write(repo, "pkg/__init__.py", "")
    _write(repo, "pkg/a.py", "def alpha():\n    return 1\n")
    _write(repo, "pkg/c.py", "def gamma():\n    return 3\n")
    _write(repo, "pkg/b.py",
           "from pkg.c import gamma\n"
           "# alpha is named here but never imported\n"
           "def beta():\n    return gamma() + 2\n")
    pack = codepack.code_pack(repo, "pkg/b.py")
    assert pack["sections"]["neighborhood"] == ["pkg/c.py"]  # the real import enters
    assert "pkg/a.py" not in pack["sections"]["neighborhood"]


def test_private_members_stay_out_of_neighbour_signatures(repo):
    _write(repo, "src/main/java/game/Secret.java",
           "package game;\n\npublic class Secret {\n"
           "    private void hidden() {\n    }\n\n    public void shown() {\n    }\n}\n")
    _write(repo, TARGET, GAME_MODEL.replace("private Vec2 pos;", "private Vec2 pos;\n    private Secret s;"))
    pack = codepack.code_pack(repo, TARGET)
    assert "public void shown()" in pack["text"]
    assert "hidden" not in pack["text"]


def test_import_never_used_in_body_is_not_a_neighbor(repo):
    # Finding 1: an import line always names the file it imports, so the
    # mention filter must not count the import line itself as a mention —
    # otherwise every resolved import survives regardless of real use.
    src = GAME_MODEL.replace("import util.Vec2;\n", "import util.Vec2;\nimport app.Launcher;\n")
    _write(repo, TARGET, src)
    _write(repo, "src/main/java/app/Launcher.java", LAUNCHER)
    pack = codepack.code_pack(repo, TARGET)
    # Vec2 is used in the body (not just imported): still a neighbour.
    assert "src/main/java/util/Vec2.java" in pack["sections"]["neighborhood"]
    # Launcher is imported but never named anywhere outside the import line.
    assert "src/main/java/app/Launcher.java" not in pack["sections"]["neighborhood"]


def test_multi_line_typescript_import_is_masked_whole():
    # Review finding 4: the mask anchored on the first physical line only, so
    # the braced multi-line form (prettier's default past 80 columns) left the
    # imported names AND the closing `} from "./mod";` line unmasked, and the
    # mention filter was effectively off for idiomatic TS.
    src = 'import {\n  Foo,\n  Bar,\n} from "./mod";\n\nconst x = 1;\n'
    body = codepack._mask_imports(src)
    assert len(body) == len(src)                       # offsets stay valid
    assert body.index("const x = 1;") == src.index("const x = 1;")
    entry = {"symbols": [{"name": "Foo", "parent": ""}, {"name": "Bar", "parent": ""}]}
    assert codepack._first_mention(body, "src/mod.ts", entry) is None


def test_multi_line_python_import_is_masked_whole():
    src = "from pkg.mod import (\n    alpha,\n    beta,\n)\n\n\ndef use():\n    return 1\n"
    body = codepack._mask_imports(src)
    assert len(body) == len(src)
    assert body.index("def use():") == src.index("def use():")
    entry = {"symbols": [{"name": "alpha", "parent": ""}, {"name": "beta", "parent": ""}]}
    assert codepack._first_mention(body, "pkg/mod.py", entry) is None


def test_an_unclosable_bracket_does_not_mask_the_rest_of_the_file():
    # The bracket counter cannot close a `(` opened inside a comment, so the
    # run used to reach end of file: every candidate then had no mention, the
    # whole neighbourhood vanished, and nothing landed in `dropped` to say why,
    # because an entry that never enters the fill loop is never recorded.
    # A blank line ends the run, since no import statement spans one.
    src = "import os  # TODO(alice\n\n\ndef real():\n    return Helper()\n"
    body = codepack._mask_imports(src)
    assert len(body) == len(src)
    assert body.strip() != ""                          # not the whole file
    assert body.index("def real():") == src.index("def real():")
    entry = {"symbols": [{"name": "Helper", "parent": ""}]}
    assert codepack._first_mention(body, "pkg/helper.py", entry) is not None


def test_private_modifier_order_does_not_leak_the_member():
    # Finding 2: `private` is a legal Java modifier in any order relative to
    # `static`/`final`; a bare `sig.startswith("private ")` check misses it.
    entry = {
        "symbols": [
            {"kind": "class", "name": "Widget", "parent": "", "signature": "public class Widget", "doc": ""},
            {"kind": "method", "name": "shown", "parent": "Widget", "signature": "public void shown()", "doc": ""},
            {"kind": "method", "name": "reordered", "parent": "Widget",
             "signature": "static private void reordered()", "doc": ""},
        ]
    }
    sigs = codepack._signatures(entry)
    assert "public void shown()" in sigs
    assert "reordered" not in sigs


def test_private_inner_class_hides_its_own_public_members():
    # Minor C: a public method of a class that was itself filtered out (here,
    # a private inner class) must not be reparented onto the outer class —
    # it must be dropped along with its declaring class.
    entry = {
        "symbols": [
            {"kind": "class", "name": "Outer", "parent": "", "signature": "public class Outer", "doc": ""},
            {"kind": "class", "name": "Inner", "parent": "Outer", "signature": "private class Inner", "doc": ""},
            {"kind": "method", "name": "leaked", "parent": "Inner", "signature": "public void leaked()", "doc": ""},
            {"kind": "method", "name": "shown", "parent": "Outer", "signature": "public void shown()", "doc": ""},
        ]
    }
    sigs = codepack._signatures(entry)
    assert "public class Outer" in sigs
    assert "public void shown()" in sigs
    assert "leaked" not in sigs


def test_same_named_class_in_a_different_scope_is_not_poisoned():
    # Round-2 review finding: `hidden` is keyed by bare name (the only key
    # `parent` ever carries), so a filtered `Outer.Builder` must not also
    # hide the unrelated, public `Other.Builder`'s own members. Document
    # order (a class always precedes its own members) is what makes this
    # safe: `Builder` is re-opened the moment the second, surviving `Builder`
    # is itself emitted, before its own children are read.
    entry = {
        "symbols": [
            {"kind": "class", "name": "Outer", "parent": "", "signature": "public class Outer", "doc": ""},
            {"kind": "class", "name": "Builder", "parent": "Outer", "signature": "private class Builder", "doc": ""},
            {"kind": "method", "name": "step1", "parent": "Builder", "signature": "public void step1()", "doc": ""},
            {"kind": "method", "name": "topMethod", "parent": "Outer", "signature": "public void topMethod()", "doc": ""},
            {"kind": "class", "name": "Other", "parent": "", "signature": "public class Other", "doc": ""},
            {"kind": "class", "name": "Builder", "parent": "Other", "signature": "public class Builder", "doc": ""},
            {"kind": "method", "name": "step2", "parent": "Builder", "signature": "public void step2()", "doc": ""},
        ]
    }
    sigs = codepack._signatures(entry)
    assert "step1" not in sigs               # Outer.Builder is private: dropped with its child
    assert "public void step2()" in sigs     # Other.Builder is public: its child must survive


def test_external_and_importers_sections(repo):
    pack = codepack.code_pack(repo, TARGET)
    assert "## external\njava.util" in pack["text"]
    assert pack["sections"]["external"] == ["java.util"]
    assert "## importers (fan-in 1)\nsrc/main/java/app/Launcher.java" in pack["text"]
    assert pack["sections"]["importers"] == ["src/main/java/app/Launcher.java"]


def test_section_order_is_fixed(repo):
    text = codepack.code_pack(repo, TARGET)["text"]
    headers = ["## target", "## hierarchy", "## neighborhood", "## external", "## importers"]
    # .find, not .index: a section that legitimately went empty should fail as
    # a missing header, not as an opaque ValueError from inside the fixture.
    order = [(h, text.find(h)) for h in headers]
    assert [h for h, at in order if at < 0] == []
    assert [at for _, at in order] == sorted(at for _, at in order)


def test_fill_stops_at_the_first_entry_that_does_not_fit(repo):
    full = codepack.code_pack(repo, TARGET)
    # -30 is less than the cost of the last entry (the importers line), so
    # exactly one entry — the last one in section order — fails to fit.
    budget = len(full["text"]) - 30
    tight = codepack.code_pack(repo, TARGET, budget_chars=budget)

    assert tight["target_mode"] == "verbatim"   # the target still fits whole
    assert tight["truncated"] is False          # truncated is about the target, not the sections
    assert len(tight["text"]) <= budget
    assert tight["dropped"] == ["importers: src/main/java/app/Launcher.java"]
    assert "importers" not in tight["sections"]
    assert tight["sections"]["neighborhood"] == full["sections"]["neighborhood"]


def test_dropped_is_a_suffix_of_the_entry_order(repo):
    # Review finding 3: `dropped` carries two kinds of entry. Degrade notes
    # are prefixed "note: " and come first; budget drops are "<section>:
    # <label>" and form a suffix of the entry order. An unresolved selector
    # puts both kinds in the same list, the case the old fixture never hit.
    target = f"{TARGET}#GameModel.nosuch"
    full = codepack.code_pack(repo, target)
    note = "note: selector 'GameModel.nosuch' not found, degraded to a file-level pack"
    assert full["dropped"] == [note]
    order = [f"{sec}: {label}"
             for sec in ("hierarchy", "neighborhood", "external", "importers")
             for label in full["sections"].get(sec, [])]
    # -100 lands mid-fill: three of the five entries drop and `neighborhood`
    # itself is only partially filled (one of its two entries survives), the
    # case most likely to break under a future refactor. A budget that drops
    # every entry (e.g. len(GAME_MODEL) + 60) would make `dropped == order`
    # look like a suffix by accident — the guard below rules that out.
    tight = codepack.code_pack(repo, target, budget_chars=len(full["text"]) - 100)
    assert tight["dropped"][0] == note               # notes first, then the budget tail
    tail = tight["dropped"][1:]
    assert not any(d.startswith("note: ") for d in tail)
    assert 0 < len(tail) < len(order)                # a PROPER suffix, not the whole list
    assert tail == order[len(order) - len(tail):]


def test_target_overhead_is_reconciled_with_the_pack_budget(repo):
    # Before this fix, `_target_block` compared the raw source length against
    # the FULL budget while the fill loop measured `used` against `len(text)`:
    # at budget_chars == len(GAME_MODEL) the target used to read as verbatim
    # even though the header/newline overhead around it meant the resulting
    # pack overshot its own budget by over 50%, with `truncated` misreporting
    # `False`. The header + trailing newline must come out of the target's
    # own budget so a larger budget never selects a worse (bigger) mode than
    # a smaller one would have.
    budget = len(GAME_MODEL)
    pack = codepack.code_pack(repo, TARGET, budget_chars=budget)
    assert len(pack["text"]) <= budget
    assert pack["target_mode"] == "outline"
    assert pack["truncated"] is True


def test_a_pathologically_small_budget_still_serves_the_target(repo):
    pack = codepack.code_pack(repo, TARGET, budget_chars=1)
    assert pack["text"]
    assert "GameModel" in pack["text"]


def test_golden_pack_byte_for_byte(repo):
    pack = codepack.code_pack(repo, TARGET)
    expected = f"""## target {TARGET} @ {pack['head_ref']} mode: verbatim
package game;

import java.util.List;
import util.Vec2;

public class GameModel extends Entity {{
    private Vec2 pos;

    public void tick() {{
        pos = new Vec2(1, 2);
    }}
}}

## hierarchy
GameModel extends Entity

## neighborhood
src/main/java/util/Vec2.java
public class Vec2
  public Vec2(int x, int y)
  public int len()
src/main/java/game/Entity.java
public class Entity
  public void update()

## external
java.util

## importers (fan-in 1)
src/main/java/app/Launcher.java
"""
    assert pack["text"] == expected


def test_two_calls_are_identical(repo):
    a = codepack.code_pack(repo, TARGET)
    b = codepack.code_pack(repo, TARGET)
    assert a == b


def test_tool_returns_a_pack(repo, monkeypatch):
    from silica.tools.codedocs_tool import silica_code_pack

    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(repo))
    res = silica_code_pack(target=TARGET)
    assert res["status"] == "ok"
    assert res["target_mode"] == "verbatim"
    assert "## neighborhood" in res["text"]


def test_tool_reports_a_bad_path_instead_of_raising(repo, monkeypatch):
    from silica.tools.codedocs_tool import silica_code_pack

    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(repo))
    res = silica_code_pack(target="nope/missing.java")
    assert res["status"] == "error"
    assert "missing.java" in res["message"]


def test_tool_serves_on_the_default_mcp_surface():
    # Flipped 2026-08-29 (was: outside CORE): the datapolis field probe showed
    # the default surface is where a coding client needs the pack — kept out,
    # the tool existed but nothing could reach it (ADR-0033 amendment).
    from silica.ui.mcp import CORE_TOOLS, exposed_tools

    assert "silica_code_pack" in CORE_TOOLS
    assert "silica_code_pack" in exposed_tools()


def test_tool_reports_missing_vault_instead_of_serving_the_cwd(monkeypatch):
    # An unset vault_path must not silently fall through to code_pack's own
    # `root = repo_root_for(vault) or Path(vault)` CWD fallback and report
    # "ok" from whatever repo the process happens to be running in. The
    # wrapper must catch this before calling code_pack, and the message must
    # name the vault, not the target, so a caller can tell this apart from a
    # genuinely bad target path.
    from silica.tools.codedocs_tool import silica_code_pack

    monkeypatch.setattr("silica.config.CONFIG.vault_path", "")
    res = silica_code_pack(target=TARGET)
    assert res["status"] == "error"
    assert "vault" in res["message"]
    assert TARGET not in res["message"]


def test_a_cpp_type_alias_is_not_masked_as_an_import():
    # `using Alias = Neighbor;` is a real use of Neighbor and often the only
    # place it is named, so masking it hid that neighbour completely. A plain
    # using-declaration still names its target the way an import does.
    src = "using Vec = geom::Vector;\nusing std::string;\nint n;\n"
    body = codepack._mask_imports(src)
    assert body.splitlines()[0] == "using Vec = geom::Vector;"
    assert body.splitlines()[1].strip() == ""
    assert len(body) == len(src)  # offsets preserved


def test_python_symbol_named_private_is_not_read_as_a_modifier():
    # `private` is a modifier token in Java, not in Python, where it can only
    # be the symbol's own name. A public function must not vanish over it.
    py = {"language": "python", "symbols": [
        {"kind": "function", "name": "private", "parent": "",
         "signature": "def private()", "doc": ""}]}
    assert codepack._signatures(py) == "def private()"
    java = {"language": "java", "symbols": [
        {"kind": "method", "name": "f", "parent": "",
         "signature": "private void f()", "doc": ""}]}
    assert codepack._signatures(java) == ""


def test_symbol_source_has_no_selector_table_for_c_and_cpp():
    # C and C++ names sit inside `declarator`, so there is no selector table
    # and a whole-file pack is the honest degrade (D4).
    src = "struct Foo {\n    int x;\n};\n"
    assert codepack._symbol_source(src, "c", "Foo") is None
    assert codepack._symbol_source(src, "cpp", "Foo") is None
    assert codepack._symbol_source(src, "", "Foo") is None


def test_symbol_source_resolves_javascript_through_the_typescript_table():
    src = "function alpha() {\n  return 1;\n}\n\nfunction beta() {\n  return 2;\n}\n"
    picked = codepack._symbol_source(src, "javascript", "beta")
    assert picked == "function beta() {\n  return 2;\n}"


class _FakeGraph:
    """Stand-in for a CodeGraph: `_neighborhood` only ever reads `.files`."""

    def __init__(self, files: dict) -> None:
        self.files = files


def _pyfile(*names: str) -> dict:
    return {"language": "python", "imports": [],
            "symbols": [{"kind": "function", "name": n, "parent": "",
                         "signature": f"def {n}()", "doc": ""} for n in names]}


def test_neighborhood_breaks_an_equal_offset_tie_by_path():
    # Both neighbours are first named by the SAME token, so the offset cannot
    # order them and the path sort has to; otherwise the order is whatever the
    # import list happened to be, and the pack stops being deterministic.
    graph = _FakeGraph({"z/late.py": _pyfile("shared"), "a/early.py": _pyfile("shared")})
    entry = {"language": "python", "imports": ["z/late.py", "a/early.py"], "symbols": []}
    out = codepack._neighborhood(graph, "main.py", entry, "shared()\n")
    assert [label for label, _ in out] == ["a/early.py", "z/late.py"]


def test_neighborhood_drops_a_neighbour_with_nothing_public_to_show():
    graph = _FakeGraph({"helper.py": _pyfile("_internal")})
    entry = {"language": "python", "imports": ["helper.py"], "symbols": []}
    source = "helper.run()\n"
    # mentioned (by its stem) but every symbol it declares is private, so
    # there is no block to emit and the entry is skipped, not emitted empty.
    assert codepack._neighborhood(graph, "main.py", entry, source) == []
    graph.files["helper.py"]["symbols"].append(
        {"kind": "function", "name": "run", "parent": "", "signature": "def run()", "doc": ""})
    assert [label for label, _ in
            codepack._neighborhood(graph, "main.py", entry, source)] == ["helper.py"]


def test_empty_sections_are_absent_rather_than_emitted_empty(repo):
    # Hud imports nothing and nothing imports it, so neither header may appear:
    # an empty section is budget spent to say nothing.
    pack = codepack.code_pack(repo, "src/main/java/game/Hud.java")
    assert "## external" not in pack["text"]
    assert "## importers" not in pack["text"]
    assert "external" not in pack["sections"]
    assert "importers" not in pack["sections"]


def test_tool_reports_a_whitespace_only_vault_as_missing(monkeypatch):
    from silica.tools.codedocs_tool import silica_code_pack

    monkeypatch.setattr("silica.config.CONFIG.vault_path", "   ")
    res = silica_code_pack(target=TARGET)
    assert res["status"] == "error"
    assert "vault" in res["message"]
    assert TARGET not in res["message"]  # not misreported as a bad target


def test_tool_reports_an_unreadable_code_graph_instead_of_raising(repo, monkeypatch):
    # load_codegraph's store-write path raises OSError, not ValueError. The
    # wrapper caught only ValueError, so an unwritable store crashed the tool.
    from silica.tools.codedocs_tool import silica_code_pack

    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(repo))
    monkeypatch.setattr(codegraph, "load_codegraph",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("store is read-only")))
    res = silica_code_pack(target=TARGET)
    assert res["status"] == "error"
    assert "read-only" in res["message"]


# --- reply ergonomics, measured 2026-09-03 on this repo ---------------------
# provenance.py (40k chars) at budget 8k came back as an outline with `dropped`
# listing `external: hashlib, json, re...` as fetchable sections, no hint of
# what budget would have served it verbatim, and the same neighbourhood
# outline repeated in every pack under kernel/write.

PY_A = "import json\nimport os\nfrom pkg import b\n\n\ndef f():\n    return json.dumps(b.g())\n"
PY_B = "def g():\n    return 1\n"


@pytest.fixture
def pyrepo(tmp_path, monkeypatch):
    from silica.kernel.recall import paths

    paths.clear_repo_root_cache()
    monkeypatch.setattr(codegraph, "store_path", lambda: tmp_path / "cg.json")
    _init_repo(tmp_path)
    _write(tmp_path, "pkg/a.py", PY_A)
    _write(tmp_path, "pkg/b.py", PY_B)
    _write(tmp_path, "pkg/__init__.py", "")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)
    yield tmp_path
    paths.clear_repo_root_cache()


def test_python_stdlib_imports_are_not_external_deps(pyrepo):
    pack = codepack.code_pack(pyrepo, "pkg/a.py")
    assert "external" not in pack["sections"]
    assert not any(d.startswith("external:") for d in pack["dropped"])


def test_pack_says_how_big_the_target_is_and_what_budget_serves_it_verbatim(pyrepo):
    src = (pyrepo / "pkg/a.py").read_text()
    small = codepack.code_pack(pyrepo, "pkg/a.py", budget_chars=40)
    assert small["target_mode"] == "outline"
    assert small["target_chars"] == len(src)
    again = codepack.code_pack(pyrepo, "pkg/a.py", budget_chars=small["verbatim_at"])
    assert again["target_mode"] == "verbatim"
    assert "verbatim_at" not in again


def test_sections_filter_skips_the_neighbourhood(repo):
    pack = codepack.code_pack(repo, TARGET, sections=["target", "importers"])
    assert set(pack["sections"]) == {"target", "importers"}
    assert not any(d.startswith("neighborhood:") or d.startswith("hierarchy:")
                   for d in pack["dropped"])
    assert "## neighborhood" not in pack["text"]
