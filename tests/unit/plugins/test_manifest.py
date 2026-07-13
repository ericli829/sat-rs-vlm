from pathlib import Path

import pytest

from sat_rs_vlm.plugins.errors import PluginValidationError
from sat_rs_vlm.plugins.manifest import load_plugin_manifest


def test_manifest_is_parsed(fake_plugin_root: Path) -> None:
    manifest = load_plugin_manifest(fake_plugin_root / "plugins" / "fake_strategy")
    assert manifest.plugin.name == "fake_strategy"
    assert manifest.schema_version == "1"


def test_manifest_name_must_match_directory(fake_plugin_root: Path) -> None:
    plugin = fake_plugin_root / "plugins" / "fake_strategy"
    text = (plugin / "plugin.yaml").read_text(encoding="utf-8")
    (plugin / "plugin.yaml").write_text(
        text.replace("name: fake_strategy", "name: another_name"),
        encoding="utf-8",
    )
    with pytest.raises(PluginValidationError, match="must match"):
        load_plugin_manifest(plugin)
