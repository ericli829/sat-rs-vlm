"""YAML 分层加载器，实现本地与云端一致的优先级规则。"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.configuration.environment import expand_environment
from sat_rs_vlm.configuration.merge import deep_merge, set_dotted_value


@dataclass(frozen=True)
class LayeredConfigRequest:
    """一次配置解析所需的各层文件和覆盖值。"""

    base_configs: Sequence[Path] = field(default_factory=tuple)
    environment_config: Path | None = None
    experiment_config: Path | None = None
    cli_overrides: Mapping[str, Any] = field(default_factory=dict)
    project_root: Path = field(default_factory=Path.cwd)
    allow_unresolved: bool = False


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return dict(loaded)


def _environment_layer(environ: Mapping[str, str]) -> dict[str, Any]:
    from sat_rs_vlm.configuration.paths import PathConfig

    paths: dict[str, str] = {}
    for env_name, field_name in PathConfig.ENV_TO_FIELD.items():
        if env_name in environ:
            paths[field_name] = environ[env_name]
    return {"paths": paths} if paths else {}


def load_layered_config(
    request: LayeredConfigRequest,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """按 defaults < base < env < experiment < environment variables < CLI 合并。"""

    env = dict(os.environ if environ is None else environ)
    layers = [_read_yaml(path) for path in request.base_configs]
    if request.environment_config is not None:
        layers.append(_read_yaml(request.environment_config))
    if request.experiment_config is not None:
        layers.append(_read_yaml(request.experiment_config))
    layers.append(_environment_layer(env))
    merged = deep_merge(*layers)
    for key, value in request.cli_overrides.items():
        if value is not None:
            set_dotted_value(merged, key, value)
    return dict(
        expand_environment(
            merged,
            environ=env,
            allow_unresolved=request.allow_unresolved,
        )
    )


def write_resolved_config(config: Mapping[str, Any], path: Path) -> None:
    """把最终配置快照写为可读 YAML。"""

    def yaml_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): yaml_safe(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [yaml_safe(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(yaml_safe(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
