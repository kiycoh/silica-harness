# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""JSON-schema compaction for the toolset the model receives every iteration.

pydantic emits `anyOf: [X, {"type": "null"}]` plus `"default": null` for every
Optional field — pure noise to the model, since omitting an optional field
already means null. The compactor collapses the anyOf to X and drops null
defaults; informative defaults (5, "", false) survive.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from silica.tools import Tool


class _Params(BaseModel):
    path: str = Field(description="where")
    k: int = Field(default=5, description="how many")
    template: str | None = Field(default=None, description="named template")
    props: dict[str, str] | None = None
    tags: list[str] | None = None


def _schema() -> dict:
    t = Tool(lambda **kw: "", "t", "desc", _Params, "atomic")
    return t.json_schema()["function"]["parameters"]


def test_optional_fields_lose_the_anyof_null_wrapper():
    props = _schema()["properties"]
    assert props["template"] == {"type": "string", "description": "named template"}
    assert props["props"] == {"type": "object",
                              "additionalProperties": {"type": "string"}}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}


def test_informative_defaults_survive_null_defaults_do_not():
    props = _schema()["properties"]
    assert props["k"]["default"] == 5
    assert "default" not in props["template"]


def test_required_and_titles_behave_as_before():
    schema = _schema()
    assert schema["required"] == ["path"]
    assert "title" not in json.dumps(schema)


def test_a_real_anyof_union_is_left_alone():
    class P(BaseModel):
        ref: str | int = Field(description="name or id")

    t = Tool(lambda **kw: "", "t", "d", P, "atomic")
    ref = t.json_schema()["function"]["parameters"]["properties"]["ref"]
    assert "anyOf" in ref  # two live branches: not the Optional pattern


def test_a_parameter_actually_named_title_survives_compaction():
    """`title` is dropped as an ANNOTATION, never as a field.

    The pass keyed on the string alone, at any depth — so it deleted the real
    `title` parameter out of `properties` while `required` went on naming it.
    Live tools shipped a required field with no definition (silica_event_create,
    silica_write_note, silica_graph_export): a strict validator rejects the call
    outright, a lenient one drops the value.
    """
    class P(BaseModel):
        title: str = Field(description="the event title")
        start: str = Field(description="when")
        body: str = Field(default="", description="what")

    schema = Tool(lambda **kw: "", "t", "d", P, "atomic").json_schema()["function"]["parameters"]
    assert schema["properties"]["title"] == {"type": "string", "description": "the event title"}
    assert set(schema["required"]) <= set(schema["properties"])


def test_every_registered_tool_defines_every_field_it_requires():
    """The invariant, over the real toolset — the compactor is the only thing
    between the params model and the wire."""
    import silica.tools.atomic          # noqa: F401  (registration side effect)
    import silica.tools.composed        # noqa: F401
    import silica.tools.graph           # noqa: F401
    import silica.tools.notes           # noqa: F401
    import silica.tools.runners         # noqa: F401
    import silica.tools.wrapped         # noqa: F401
    from silica.tools import TOOLS

    assert TOOLS, "no tools registered"
    for name, t in TOOLS.items():
        params = t.json_schema()["function"]["parameters"]
        undefined = set(params.get("required", [])) - set(params.get("properties", {}))
        assert not undefined, f"{name} requires undefined field(s): {sorted(undefined)}"
