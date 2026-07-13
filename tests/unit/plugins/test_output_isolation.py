from pathlib import Path

import pytest

from sat_rs_vlm.plugins.discovery import discover_plugins
from sat_rs_vlm.plugins.errors import PluginExecutionError
from sat_rs_vlm.plugins.runtime import _safe_output_dir


def test_output_cannot_target_another_plugin_directory(fake_plugin_root: Path) -> None:
    plugin = discover_plugins([fake_plugin_root])["fake_strategy"]
    other_plugin = fake_plugin_root / "plugins" / "other_strategy" / "checkpoints"

    with pytest.raises(PluginExecutionError, match="outside the current plugin"):
        _safe_output_dir(plugin, {"experiment": {"name": "run"}}, str(other_plugin))
