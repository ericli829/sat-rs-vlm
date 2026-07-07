"""JSON Lines 读写工具。

算法：
    每行解析为一个独立 JSON 对象，空行会被跳过。该格式适合保存 prompt 列表、
    遥感样本索引和轻量评测结果。
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """逐行读取 JSONL 文件。

    参数：
        path：JSONL 文件路径。

    返回值：
        Iterator[dict[str, Any]]：惰性迭代的 JSON 对象字典。
    """

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """写入 JSONL 文件。

    参数：
        path：输出文件路径。
        rows：可迭代的字典对象，每个对象写成一行 JSON。

    返回值：
        None。
    """

    with Path(path).open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
