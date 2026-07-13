import sys
from pathlib import Path

from sat_rs_vlm.plugins.discovery import discover_plugins
from sat_rs_vlm.plugins.loader import load_external_plugin


def test_plain_folder_plugin_loads_without_sys_path_change(fake_plugin_root: Path) -> None:
    before = list(sys.path)
    discovered = discover_plugins([fake_plugin_root])["fake_strategy"]
    plugin = load_external_plugin(discovered)
    assert plugin.name == "fake_strategy"
    assert sys.path == before
