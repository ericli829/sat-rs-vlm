"""Qwen3-VL 训练数据集。

支持两种 JSONL 行格式：
1. 已转换好的 Qwen3-VL messages 格式。
2. 项目内部 instruction/images/answer 格式。

Dataset 只做轻量 JSON 读取和格式归一化，不加载 processor，也不做图像 tensor 编码。
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sat_rs_vlm.utils.jsonl import read_jsonl


def sample_to_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """将内部样本格式转换为 Qwen3-VL messages。

    如果 row 已经包含 messages，则直接返回该字段。
    """

    if "messages" in row:
        return list(row["messages"])
    required = ("images", "instruction", "answer")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Sample {row.get('id', '<unknown>')} missing fields: {missing}")
    content = [{"type": "image", "image": str(image)} for image in list(row["images"])]
    content.append({"type": "text", "text": str(row["instruction"])})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": str(row["answer"])},
    ]


class Qwen3VLDataset:
    """Qwen3-VL 指令微调数据集。

    参数：
        jsonl_path：训练/验证 JSONL 文件。
        max_samples：可选最大读取条数，用于 smoke test。
        skip_bad_samples：是否跳过坏样本；默认 False，遇到坏样本直接报错。
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        max_samples: int | None = None,
        *,
        skip_bad_samples: bool = False,
    ) -> None:
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {self.path}")
        rows: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for row in read_jsonl(self.path):
            try:
                rows.append(self._normalize_row(row))
            except (KeyError, TypeError, ValueError) as exc:
                if not skip_bad_samples:
                    raise
                skipped.append(
                    {
                        "id": str(row.get("id", "<unknown>")),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        self._rows = rows[:max_samples] if max_samples is not None else rows
        self.skipped_samples = skipped

    def __len__(self) -> int:
        """返回样本数量。"""

        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """读取单条归一化样本。"""

        return self._rows[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """按输入顺序迭代归一化样本。"""

        return iter(self._rows)

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        """归一化单条 JSONL 样本。"""

        sample_id = str(row.get("id", "unknown"))
        return {
            "id": sample_id,
            "messages": sample_to_messages(row),
            "task_type": str(row.get("task_type", "unknown")),
            "metadata": dict(row.get("metadata", {})),
        }
