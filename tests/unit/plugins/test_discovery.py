import shutil
from pathlib import Path

import pytest

from sat_rs_vlm.plugins.discovery import discover_plugins, resolve_plugin_roots
from sat_rs_vlm.plugins.errors import PluginValidationError


def test_unconfigured_roots_are_empty(tmp_path: Path) -> None:
    assert resolve_plugin_roots(project_root=tmp_path, environ={}) == []
    assert discover_plugins([]) == {}


def test_valid_plugin_is_discovered(fake_plugin_root: Path) -> None:
    plugins = discover_plugins([fake_plugin_root])
    assert list(plugins) == ["fake_strategy"]


def test_duplicate_names_are_rejected(fake_plugin_root: Path, tmp_path: Path) -> None:
    second = tmp_path / "second"
    (second / "plugins").mkdir(parents=True)
    shutil.copytree(
        fake_plugin_root / "plugins" / "fake_strategy",
        second / "plugins" / "fake_strategy",
    )
    with pytest.raises(PluginValidationError, match="duplicate"):
        discover_plugins([fake_plugin_root, second])
