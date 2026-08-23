import datetime
import logging

from silica.kernel.write.templates import (
    BUILTIN_TEMPLATE,
    ensure_ai_flag,
    prepare_fields,
    render_note,
    stamp_sources,
    template_spoke,
)


def test_spoke_does_not_double_wrap_bracketed_parent_and_related():
    """Distiller often emits parent/related already as [[X]] (hence write.py's
    .strip('[]')). template_spoke must not re-wrap them into [[[[X]]]], which
    Obsidian reads as an unresolved link and trips the graph regression gate."""
    out = template_spoke(
        heading="Modelli Linguistici Generativi (GPT)",
        snippet="testo",
        hub="[[IA Generativa]]",
        related=["[[Reti Neurali Profonde (Deep Learning)]]", "calcolo parallelo"],
        parent="[[Rinascita dell'IA]]",
    )
    assert "[[[[" not in out and "]]]]" not in out, out
    assert 'parent note: "[[Rinascita dell\'IA]]"' in out
    assert '"[[Reti Neurali Profonde (Deep Learning)]]"' in out
    assert '"[[calcolo parallelo]]"' in out  # bare name still wrapped exactly once
    assert '"[[IA Generativa]]"' in out


def test_ensure_ai_flag_stamps_missing_field_on_legacy_note():
    """Root cause of the 'all patches reverted' bug: user notes predating the
    `AI` convention lack the field, and the OFM lint fails the whole note on a
    patch. ensure_ai_flag stamps `AI: true` (honest provenance) so the lint passes.
    """
    from silica.kernel.link import ofm
    legacy = "---\ntags:\n  - statistica\n---\n# Varianza\nLa varianza misura la dispersione."
    stamped = ensure_ai_flag(legacy)
    assert "AI: true" in stamped.split("---")[1]
    assert not any("AI" in v for v in ofm.ofm_lint(stamped, stem="Varianza")["violations"])


def test_ensure_ai_flag_is_conservative():
    """Idempotent; never overwrites the user's own AI value; no-op without frontmatter."""
    legacy = "---\ntags:\n  - x\n---\n# n\nbody"
    once = ensure_ai_flag(legacy)
    assert ensure_ai_flag(once) == once            # idempotent
    assert once.count("AI: true") == 1
    user_false = "---\nAI: false\ntags:\n  - x\n---\n# n\nbody"
    assert ensure_ai_flag(user_false) == user_false  # keeps user's explicit value
    assert ensure_ai_flag("# no frontmatter\nbody") == "# no frontmatter\nbody"


def test_builtin_template_parity_with_template_spoke():
    """The guarantee that existing vaults do not change behavior: the built-in
    template through render_note is byte-identical to template_spoke."""
    cases = [
        dict(heading="Varianza", snippet="La varianza misura la dispersione.", hub="Statistica"),
        dict(heading="GPT", snippet="testo", hub="[[IA Generativa]]",
             parent="[[Rinascita dell'IA]]",
             related=["[[Reti Neurali]]", "calcolo parallelo"],
             tags=["Tag One", "tag-one"], title="Custom Title"),
        dict(heading="Empty", snippet="", hub="AI"),
    ]
    for c in cases:
        legacy = template_spoke(**c)
        fields = prepare_fields(
            title=c.get("title") or c["heading"], body=c["snippet"], hub=c["hub"],
            tags=c.get("tags"), related=c.get("related"), parent=c.get("parent"),
        )
        assert render_note(BUILTIN_TEMPLATE, fields) == legacy, c


def test_render_drops_empty_lines_and_expands_lists():
    tpl = ("---\nparent note: {{parent}}\nrelated: {{related}}\n"
           "tags: {{tags}}\nAI: true\n---\n\n# {{title}}\n\n{{body}}\n")
    out = render_note(tpl, {"parent": "", "related": ['"[[A]]"', '"[[B]]"'],
                            "tags": [], "title": "T", "body": "B"})
    assert "parent note" not in out          # empty scalar: line dropped
    assert "tags" not in out                 # empty list: key line dropped
    assert 'related:\n  - "[[A]]"\n  - "[[B]]"\n' in out
    assert out.endswith("# T\n\nB\n")
    assert "AI: true" in out                 # literal lines pass through


def test_render_removes_unknown_placeholder_with_warning(caplog):
    tpl = "---\nfoo: {{mystery}}\nAI: true\n---\n\nBody {{alsounknown}}end\n"
    with caplog.at_level(logging.WARNING):
        out = render_note(tpl, {})
    assert "mystery" not in out and "foo" not in out   # fm line dropped
    assert "Body end" in out                           # body: removed in place
    assert "unknown placeholder" in caplog.text


def test_prepare_fields_strips_leading_yaml_from_body(caplog):
    with caplog.at_level(logging.WARNING):
        f = prepare_fields(title="T", body="---\ntags: [x]\n---\n\nReal body.")
    assert f["body"] == "Real body."
    assert "leading YAML" in caplog.text


def test_prepare_fields_fallbacks_without_hub():
    """Write-tool shape: no hub -> no hub merge, no default tag, parent optional."""
    f = prepare_fields(title="T", body="B")
    assert f["parent"] == "" and f["related"] == [] and f["tags"] == []
    assert f["date"] == datetime.date.today().isoformat()
    f2 = prepare_fields(title="T", body="B", parent="[[P]]", related=["P", "Q"])
    assert f2["parent"] == '"[[P]]"'
    assert f2["related"] == ['"[[P]]"', '"[[Q]]"']   # deduplicated


def test_render_preserves_template_body_spacing():
    """The renderer never reformats the author's body spacing: zero blank
    lines after the closing fence stays zero, two stays two."""
    out0 = render_note("---\nAI: true\n---\n# {{title}}\n", {"title": "T"})
    assert out0 == "---\nAI: true\n---\n# T\n"
    out2 = render_note("---\nAI: true\n---\n\n\n# {{title}}\n", {"title": "T"})
    assert out2 == "---\nAI: true\n---\n\n\n# T\n"


def test_floor_preserves_prior_frontmatter_on_bare_rewrite():
    """A model that omits frontmatter while rewriting means 'keep it', not
    'delete it': the prior block is re-injected verbatim, AI ensured,
    last modified refreshed — no silent metadata loss, lint green."""
    from silica.kernel.link import ofm
    from silica.kernel.write.templates import ensure_system_floor
    prior = ("---\ntags:\n  - keep\naliases:\n  - K\ncustom: v\n"
             "last modified: 2020-01-01\n---\n\n# Old\n\nold body\n")
    out = ensure_system_floor("# New\n\nnew body\n", prior=prior)
    head, tail = out.split("\n---\n", 1)
    assert "tags:\n  - keep" in head and "custom: v" in head and "aliases:" in head
    assert "AI: true" in head
    today = datetime.date.today().isoformat()
    assert f"last modified: {today}" in head
    assert "2020-01-01" not in head
    assert tail == "\n# New\n\nnew body\n"
    # feeding the output back as prior must not grow blank lines
    out2 = ensure_system_floor("# Again\n\nbody2\n", prior=out)
    assert "\n---\n\n# Again" in out2 and "\n\n\n" not in out2
    assert "old body" not in out
    assert not any("AI" in v for v in ofm.ofm_lint(out, stem="New")["violations"])


def test_floor_creates_minimal_block_when_no_prior():
    from silica.kernel.write.templates import ensure_system_floor
    today = datetime.date.today().isoformat()
    out = ensure_system_floor("# N\n\nbody\n")
    assert out.startswith(f"---\nAI: true\nlast modified: {today}\n---\n\n# N")
    # prior without a block is the same case
    assert ensure_system_floor("# N\n\nbody\n", prior="# bare prior\n") == out


def test_stamp_sources_unions_and_is_idempotent():
    """A note written by one source and patched by another lists both; re-stamping
    an already-listed source must be byte-identical (re-ingest safety)."""
    from silica.kernel.write import frontmatter

    base = "---\nAI: true\n---\n\n# N\nbody\n"
    two = stamp_sources(stamp_sources(base, "lec-01.md"), "lec-02.md")
    data, _, body = frontmatter.split(two)
    assert data["sources"] == ["lec-01.md", "lec-02.md"]
    assert body == "# N\nbody\n"
    assert stamp_sources(two, "lec-01.md") == two
    # Plain strings, never wikilinks: a link would enter the graph.
    assert "[[" not in two.split("\n---\n")[0]


def test_stamp_sources_passthrough_without_frontmatter_or_source():
    assert stamp_sources("# N\nbody\n", "s.md") == "# N\nbody\n"
    fm = "---\nAI: true\n---\n\nx"
    assert stamp_sources(fm, "") == fm


def test_floor_with_existing_block_is_ensure_ai_flag():
    """Content that carries its own frontmatter: today's behavior, unchanged —
    prior is ignored, last modified NOT touched."""
    from silica.kernel.write.templates import ensure_system_floor
    c = "---\ntags:\n  - x\nlast modified: 2020-01-01\n---\n# n\nbody"
    assert ensure_system_floor(c) == ensure_ai_flag(c)
    assert ensure_system_floor(c, prior="---\nother: y\n---\nold") == ensure_ai_flag(c)
    assert "2020-01-01" in ensure_system_floor(c)


def test_floor_adds_last_modified_when_prior_lacks_it():
    from silica.kernel.write.templates import ensure_system_floor
    out = ensure_system_floor("# N\nb\n", prior="---\ntags:\n  - k\n---\n\nold\n")
    assert f"last modified: {datetime.date.today().isoformat()}" in out.split("\n---\n")[0]


def test_floor_preserves_crlf_prior_metadata():
    """CRLF notes (Windows-synced vaults): prior metadata still preserved,
    fences canonicalized — not silently replaced by the minimal block."""
    from silica.kernel.write.templates import ensure_system_floor
    prior = "---\r\ntags:\r\n  - keep\r\n---\r\n\r\nold\r\n"
    out = ensure_system_floor("# New\n\nbody\n", prior=prior)
    head = out.split("\n---\n")[0]
    assert "keep" in head and "AI: true" in head


def test_floor_canonicalizes_fence_whitespace():
    """Trailing space on the prior's closing fence must not mangle the splice:
    last modified lands inside the block, never after the body."""
    from silica.kernel.write.templates import ensure_system_floor
    prior = "---\ntags:\n  - keep\n--- \n\nold\n"
    out = ensure_system_floor("# New\n\nbody\n", prior=prior)
    head, tail = out.split("\n---\n", 1)
    assert "keep" in head and "AI: true" in head
    assert f"last modified: {datetime.date.today().isoformat()}" in head
    assert "last modified" not in tail


def test_crlf_template_renders_lists_not_reprs(tpl_vault):
    """A CRLF template file must render list placeholders as YAML sequences,
    not Python reprs (read path normalizes line endings)."""
    from silica.kernel.write.templates import render_note, resolve_template
    (tpl_vault / "templates").mkdir()
    (tpl_vault / "templates" / "win.md").write_bytes(
        b"---\r\ntags: {{tags}}\r\nAI: true\r\n---\r\n\r\n{{body}}\r\n")
    out = render_note(resolve_template("win"), {"tags": ["a", "b"], "body": "B"})
    head = out.split("\n---\n")[0]
    assert "tags:\n  - a\n  - b" in out and "[" not in head


import pytest

from silica.kernel.vault_manifest import reset_manifest_cache


@pytest.fixture
def tpl_vault(tmp_path, monkeypatch):
    """Vault root for template resolution; manifest cache reset around it."""
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(tmp_path))
    reset_manifest_cache()
    yield tmp_path
    reset_manifest_cache()


def _write_tpl(vault, name, body="{{body}}\n"):
    d = vault / "templates"
    d.mkdir(exist_ok=True)
    (d / f"{name}.md").write_text(f"---\nx-tpl: {name}\nAI: true\n---\n\n{body}",
                                  encoding="utf-8")


def test_resolution_explicit_beats_vault_default(tpl_vault):
    from silica.kernel.write.templates import resolve_template
    _write_tpl(tpl_vault, "paper")
    _write_tpl(tpl_vault, "other")
    (tpl_vault / "vault.yaml").write_text(
        "conventions:\n  default_template: other\n", encoding="utf-8")
    reset_manifest_cache()
    assert "x-tpl: paper" in resolve_template("paper")
    assert "x-tpl: other" in resolve_template()


def test_resolution_builtin_when_unconfigured(tpl_vault):
    from silica.kernel.write.templates import BUILTIN_TEMPLATE, resolve_template
    assert resolve_template() == BUILTIN_TEMPLATE


def test_resolution_broken_default_falls_back_to_builtin(tpl_vault, caplog):
    """The pipeline never stops for a broken template (soft, like vault.yaml)."""
    from silica.kernel.write.templates import BUILTIN_TEMPLATE, resolve_template
    (tpl_vault / "templates").mkdir()
    (tpl_vault / "templates" / "broken.md").write_text("---\nunterminated",
                                                       encoding="utf-8")
    (tpl_vault / "vault.yaml").write_text(
        "conventions:\n  default_template: broken\n", encoding="utf-8")
    reset_manifest_cache()
    with caplog.at_level(logging.WARNING):
        assert resolve_template() == BUILTIN_TEMPLATE
    assert "broken" in caplog.text


def test_unknown_explicit_name_raises_listing_available(tpl_vault):
    from silica.kernel.write.templates import TemplateNotFoundError, resolve_template
    _write_tpl(tpl_vault, "paper")
    with pytest.raises(TemplateNotFoundError, match="paper"):
        resolve_template("nope")


def test_path_shaped_template_names_rejected(tpl_vault, caplog):
    """Template names are user-authored and reach a file path — separators
    and traversal are rejected, mirroring the templates_dir trust boundary."""
    from silica.kernel.write.templates import (BUILTIN_TEMPLATE, TemplateNotFoundError,
                                         resolve_template)
    with pytest.raises(TemplateNotFoundError):
        resolve_template("../evil")
    with pytest.raises(TemplateNotFoundError):
        resolve_template("D:evil")
    (tpl_vault / "vault.yaml").write_text(
        "conventions:\n  default_template: ../evil\n", encoding="utf-8")
    reset_manifest_cache()
    with caplog.at_level(logging.WARNING):
        assert resolve_template() == BUILTIN_TEMPLATE
    assert "invalid vault default template name" in caplog.text


# --- stamp_citation (2026-08-15: DOI/authors reach the note) -----------------

def test_stamp_citation_adds_prefixed_keys():
    from silica.kernel.write.templates import stamp_citation
    content = "---\ntitle: A\nAI: true\n---\n\n# A\n\nbody\n"
    out = stamp_citation(content, {
        "doi": "10.20944/preprints202603.0359.v1",
        "authors": "Tang, He, Zhao",
        "source_title": "LLM Agent Memory",
    })
    assert 'source_doi: "10.20944/preprints202603.0359.v1"' in out
    assert 'source_authors: "Tang, He, Zhao"' in out
    assert 'source_title: "LLM Agent Memory"' in out
    assert out.endswith("# A\n\nbody\n")


def test_stamp_citation_first_writer_wins():
    from silica.kernel.write.templates import stamp_citation
    content = '---\ntitle: A\nsource_doi: "10.1/first"\n---\n\nbody\n'
    out = stamp_citation(content, {"doi": "10.2/second"})
    assert out.count("source_doi") == 1
    assert "10.1/first" in out and "10.2/second" not in out


def test_stamp_citation_empty_or_no_frontmatter_is_noop():
    from silica.kernel.write.templates import stamp_citation
    assert stamp_citation("no frontmatter", {"doi": "10.1/x"}) == "no frontmatter"
    c = "---\ntitle: A\n---\n\nbody\n"
    assert stamp_citation(c, {}) == c


def test_hub_note_never_parents_itself():
    """The auto-created hub op carries hub=<its own name>, so the hub note
    shipped with `parent note: [[appunti]]` ON appunti itself — a self-link
    that self-attests. A note is never its own parent or its own relative."""
    from silica.kernel.write.templates import prepare_fields

    f = prepare_fields(title="appunti", body="x", hub="appunti")
    assert f["parent"] == ""
    assert '"[[appunti]]"' not in f["related"]


def test_self_reference_guard_leaves_real_hubs_alone():
    from silica.kernel.write.templates import prepare_fields

    f = prepare_fields(title="Slow start", body="x", hub="appunti")
    assert f["parent"] == '"[[appunti]]"'
    assert '"[[appunti]]"' in f["related"]


def test_stamp_sources_after_a_frontmatter_dump_round_trip():
    """The list `yaml.safe_dump` writes is UNINDENTED, and the splice must find
    it. A note whose frontmatter has been re-emitted by `frontmatter.dump` (any
    property edit does that) reaches the second source with `sources:` followed
    by flush-left items; when the removal regex misses them the key line goes
    and the items stay, which is not YAML at all. Measured 2026-08-23 on a real
    nucleate batch: 2 of 52 notes unparseable, both patched by a second source.
    """
    import yaml

    from silica.kernel.write import frontmatter

    note = frontmatter.dump({"type": "Note", "sources": ["a.md"]}, "body\n")
    assert "\nsources:\n- a.md\n" in note, "safe_dump stopped writing flush-left"

    out = stamp_sources(note, "b.md")
    data, raw, _ = frontmatter.split(out)
    yaml.safe_load(raw)
    assert data["sources"] == ["a.md", "b.md"]
    assert raw.count("sources:") == 1


def test_stamp_documents_after_a_frontmatter_dump_round_trip():
    """`documents:` splices with the same regex, so it fails the same way."""
    import yaml

    from silica.kernel.write import frontmatter
    from silica.kernel.write.templates import stamp_documents

    note = frontmatter.dump({"type": "Note", "documents": ["a.pdf"]}, "body\n")
    out = stamp_documents(note, ["b.pdf"])
    data, raw, _ = frontmatter.split(out)
    yaml.safe_load(raw)
    assert data["documents"] == ["a.pdf", "b.pdf"]
    assert raw.count("documents:") == 1


def test_stamp_citation_does_not_duplicate_keys_after_a_sources_splice():
    """The cascade the corruption caused: `stamp_citation` skips keys already
    present, but reads them through `frontmatter.split`, which returns None on
    broken YAML. A note the sources splice had just broken therefore had every
    citation key appended a second time (`source_title` twice, 2026-08-23)."""
    from silica.kernel.write import frontmatter
    from silica.kernel.write.templates import stamp_citation

    note = frontmatter.dump(
        {"type": "Note", "sources": ["a.md"], "source_title": "Paper"}, "body\n")
    out = stamp_citation(stamp_sources(note, "b.md"), {"title": "Paper"})
    _, raw, _ = frontmatter.split(out)
    assert raw.count("source_title:") == 1
