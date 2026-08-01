"""量化实验 JSON 序列化、文件大小和 manifest 工具。"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def to_json_safe(value: Any) -> Any:
    """递归转换 Path、Pydantic、numpy/torch scalar 为标准 JSON 类型。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return to_json_safe(value.value)
    if isinstance(value, BaseModel):
        return to_json_safe(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_json_safe(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def directory_size_bytes(path: str | Path) -> int | None:
    """统计序列化产物字节数；目录不存在时返回 None。"""

    root = Path(path)
    if not root.exists():
        return None
    if root.is_file():
        return root.stat().st_size
    return sum(file.stat().st_size for file in root.rglob("*") if file.is_file())


def write_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """先写入 report_file 字段，再执行 JSON 序列化。"""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    final_payload = dict(payload)
    final_payload["report_file"] = str(report_path)
    report_path.write_text(
        json.dumps(to_json_safe(final_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path
