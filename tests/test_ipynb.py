# tests/test_ipynb.py
"""kernel/ipynb — defensive nbformat-v4 cell extraction (spec-code-lane §3)."""
import json

import pytest

from silica.kernel.code.ipynb import CODEAST_LANGUAGE, parse_cells


def _nb(cells, language="python"):
    return json.dumps({
        "nbformat": 4,
        "metadata": {"kernelspec": {"language": language}},
        "cells": cells,
    })


def test_markdown_code_split_and_output_ignored():
    nb = _nb([
        {"cell_type": "markdown", "source": ["# Title\n", "Intro text.\n"]},
        {"cell_type": "code", "source": ["import numpy\n", "x = 1\n"],
         "outputs": [{"data": {"image/png": "AAAA_base64_AAAA"}}]},
        {"cell_type": "code", "source": "from pkg.paths import norm\n"},
    ])
    cells = parse_cells(nb)
    assert cells.markdown == ["# Title\nIntro text.\n"]
    assert "import numpy" in cells.code
    assert "from pkg.paths import norm" in cells.code
    assert "base64" not in cells.code          # outputs never enter
    assert cells.language == "python"


def test_ipython_magic_lines_stripped():
    nb = _nb([{"cell_type": "code",
               "source": "%matplotlib inline\n!pip install x\n?help\nimport os\n"}])
    cells = parse_cells(nb)
    assert cells.code.strip() == "import os"


def test_kernel_language_default_and_mapping():
    assert parse_cells(_nb([], language="R")).language == "R"
    assert "R" not in CODEAST_LANGUAGE
    plain = json.dumps({"cells": []})           # no metadata at all
    assert parse_cells(plain).language == "python"


def test_malformed_raises_valueerror():
    with pytest.raises(ValueError):
        parse_cells("{not json")
    with pytest.raises(ValueError):
        parse_cells(json.dumps({"nbformat": 4}))  # cells absent
    with pytest.raises(ValueError):
        parse_cells(json.dumps([1, 2]))           # not an object
