from pathlib import Path

import pytest

from sat_rs_vlm.plugins.errors import PluginValidationError
from sat_rs_vlm.plugins.manifest import resolve_inside


def test_parent_escape_is_rejected(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    with pytest.raises(PluginValidationError, match="escapes"):
        resolve_inside(plugin, "../outside.py", label="entrypoint")


def test_absolute_path_is_rejected(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    with pytest.raises(PluginValidationError, match="must be relative"):
        resolve_inside(plugin, str((tmp_path / "outside.py").resolve()), label="entrypoint")
