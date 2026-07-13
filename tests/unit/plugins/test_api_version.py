from pathlib import Path

import pytest

from sat_rs_vlm.plugins.errors import PluginCompatibilityError
from sat_rs_vlm.plugins.manifest import load_plugin_manifest


def test_unknown_api_version_is_rejected(fake_plugin_root: Path) -> None:
    plugin = fake_plugin_root / "plugins" / "fake_strategy"
    text = (plugin / "plugin.yaml").read_text(encoding="utf-8")
    (plugin / "plugin.yaml").write_text(
        text.replace('api_version: "1"', 'api_version: "99"'),
        encoding="utf-8",
    )
    with pytest.raises(PluginCompatibilityError, match="api_version"):
        load_plugin_manifest(plugin)
