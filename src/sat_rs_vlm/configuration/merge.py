"""配置字典合并与命令行覆盖工具。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def deep_merge(*layers: Mapping[str, Any]) -> dict[str, Any]:
    """按传入顺序递归合并配置，后面的层拥有更高优先级。

    列表和标量整体替换，只有字典会递归合并。该规则可预测，适合 YAML
    配置；同时避免 target_modules 等列表被意外拼接。
    """

    result: dict[str, Any] = {}
    for layer in layers:
        for key, value in layer.items():
            current = result.get(key)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                result[key] = deep_merge(current, value)
            else:
                result[key] = deepcopy(value)
    return result


def set_dotted_value(config: dict[str, Any], key: str, value: Any) -> None:
    """把 `training.max_steps` 形式的键写入嵌套配置。"""

    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("Override key must not be empty.")
    cursor = config
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set '{key}': '{part}' is not a mapping.")
        cursor = child
    cursor[parts[-1]] = value
