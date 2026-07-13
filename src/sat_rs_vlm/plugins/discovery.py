"""只在显式根目录中发现本地外部插件。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.plugins.errors import PluginValidationError
from sat_rs_vlm.plugins.manifest import ExternalPluginManifest, load_plugin_manifest, resolve_inside

PLUGIN_ROOT_ENV = "SAT_RS_VLM_PLUGIN_ROOT"


@dataclass(frozen=True)
class DiscoveredPlugin:
    """已完成 manifest 验证但尚未加载 Python 入口的插件。"""

    root: Path
    directory: Path
    manifest: ExternalPluginManifest


def _config_roots(config_path: Path, project_root: Path) -> list[Path]:
    if not config_path.is_file():
        return []
    payload: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not bool(payload.get("enabled", False)):
        return []
    roots = []
    for item in payload.get("plugin_roots", []):
        path = Path(str(item)).expanduser()
        roots.append(path if path.is_absolute() else project_root / path)
    return roots


def resolve_plugin_roots(
    *,
    project_root: str | Path,
    cli_roots: list[str | Path] | None = None,
    config_path: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> list[Path]:
    """按 CLI > 环境变量 > YAML > 空列表解析插件根目录。"""

    project = Path(project_root).resolve()
    values: list[str | Path]
    if cli_roots:
        values = list(cli_roots)
    else:
        environment = environ if environ is not None else os.environ
        env_value = environment.get(PLUGIN_ROOT_ENV, "").strip()
        if env_value:
            values = [item for item in env_value.split(os.pathsep) if item]
        else:
            external_config = (
                Path(config_path)
                if config_path is not None
                else project / "configs" / "external_plugins.yaml"
            )
            return [path.resolve() for path in _config_roots(external_config, project)]
    resolved = []
    for value in values:
        path = Path(value).expanduser()
        resolved.append((path if path.is_absolute() else project / path).resolve())
    return resolved


def _discovery_directories(root: Path) -> list[Path]:
    """读取 pack discovery_paths；没有 pack 时仅检查 root/plugins 和 root。"""

    pack_path = root / "plugin_pack.yaml"
    if not pack_path.is_file():
        candidates = [root / "plugins", root]
        return [path for path in candidates if path.is_dir()]
    payload: dict[str, Any] = yaml.safe_load(pack_path.read_text(encoding="utf-8")) or {}
    if str(payload.get("schema_version")) != "1":
        raise PluginValidationError(
            plugin_name="<plugin-pack>",
            stage="discovery",
            reason=f"unsupported plugin_pack schema in {pack_path}",
            suggested_action="Set plugin_pack.yaml schema_version to '1'.",
        )
    paths = payload.get("plugins", {}).get("discovery_paths", ["plugins"])
    return [
        resolve_inside(root, str(relative), label="plugin pack discovery path")
        for relative in paths
    ]


def discover_plugins(roots: list[str | Path]) -> dict[str, DiscoveredPlugin]:
    """扫描显式根目录；同名插件冲突时列出两个路径并失败。"""

    discovered: dict[str, DiscoveredPlugin] = {}
    for raw_root in roots:
        root = Path(raw_root).resolve()
        if not root.is_dir():
            raise PluginValidationError(
                plugin_name="<plugin-root>",
                stage="discovery",
                reason=f"plugin root does not exist: {root}",
                suggested_action="Pass an existing --plugin-root directory.",
            )
        candidates: set[Path] = set()
        for discovery_dir in _discovery_directories(root):
            if (discovery_dir / "plugin.yaml").is_file():
                candidates.add(discovery_dir)
            candidates.update(
                child for child in discovery_dir.iterdir() if (child / "plugin.yaml").is_file()
            )
        for plugin_dir in sorted(candidates, key=str):
            manifest = load_plugin_manifest(plugin_dir)
            name = manifest.plugin.name
            if name in discovered:
                previous = discovered[name].directory
                raise PluginValidationError(
                    plugin_name=name,
                    stage="discovery",
                    reason=f"duplicate plugin name at {previous} and {plugin_dir}",
                    suggested_action="Remove one root or rename one plugin.",
                )
            discovered[name] = DiscoveredPlugin(root, plugin_dir.resolve(), manifest)
    return discovered
