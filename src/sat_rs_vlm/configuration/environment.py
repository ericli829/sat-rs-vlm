"""环境变量展开工具。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_environment(
    value: Any,
    *,
    environ: Mapping[str, str],
    allow_unresolved: bool = False,
) -> Any:
    """递归展开字符串中的 `${NAME}`。

    Args:
        value: 字符串、列表、字典或其他配置值。
        environ: 用于解析变量的环境映射，测试时可注入。
        allow_unresolved: 为真时保留缺失变量；否则抛出明确异常。

    Returns:
        与输入结构相同、字符串变量已展开的新对象。
    """

    if isinstance(value, str):
        missing: set[str] = set()

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in environ:
                missing.add(name)
                return match.group(0)
            return environ[name]

        expanded = ENV_PATTERN.sub(replace, value)
        if missing and not allow_unresolved:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Missing environment variable(s): {names}")
        return expanded
    if isinstance(value, Mapping):
        return {
            key: expand_environment(
                item,
                environ=environ,
                allow_unresolved=allow_unresolved,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            expand_environment(
                item,
                environ=environ,
                allow_unresolved=allow_unresolved,
            )
            for item in value
        ]
    return value
