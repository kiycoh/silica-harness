import os
from unittest.mock import patch

from silica.config import SilicaConfig as Config


def test_residue_check_defaults_to_auto():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SILICA_RESIDUE_CHECK", None)
        assert Config().residue_check == "auto"


def test_residue_check_reads_env():
    with patch.dict(os.environ, {"SILICA_RESIDUE_CHECK": "off"}):
        assert Config().residue_check == "off"
