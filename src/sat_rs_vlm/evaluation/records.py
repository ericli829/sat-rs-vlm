"""输入 Prediction JSONL 的轻量 Schema 与校验。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EvaluationError(RuntimeError):
    """评测输入、配置或输出安全检查失败。"""


class InputValidationError(EvaluationError):
    """输入 JSONL 不符合当前仓库的 Prediction 基础字段。"""


RESERVED_EVALUATION_FIELDS = {
    "metric_profile",
    "metric_label",
    "protocol_status",
    "protocol_provenance",
    "parsed_prediction",
    "parse_ok",
    "parse_error",
    "sample_metrics",
    "semantic_profile",
    "semantic_reference_source",
    "semantic_prediction",
    "semantic_reference",
    "semantic_metrics",
}


@dataclass(frozen=True)
class PredictionRecord:
    """保持原始扩展字段的 Prediction 记录。"""

    raw: dict[str, Any]
    line_number: int
    id: str
    task_type: str
    prediction: str
    reference: str
    metadata: dict[str, Any]
    inference_latency_ms: float | None

    @classmethod
    def from_mapping(cls, row: Any, line_number: int) -> PredictionRecord:
        if not isinstance(row, dict):
            raise InputValidationError(f"line {line_number}: JSON value must be an object")
        missing = [
            name for name in ("id", "task_type", "prediction", "reference") if name not in row
        ]
        if missing:
            raise InputValidationError(f"line {line_number}: missing fields: {missing}")
        collisions = sorted(RESERVED_EVALUATION_FIELDS.intersection(row))
        if collisions:
            raise InputValidationError(
                f"line {line_number}: input already contains evaluation fields: {collisions}"
            )
        sample_id = row["id"]
        task_type = row["task_type"]
        prediction = row["prediction"]
        reference = row["reference"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise InputValidationError(f"line {line_number}: id must be a non-empty string")
        if not isinstance(task_type, str) or not task_type.strip():
            raise InputValidationError(f"line {line_number}: task_type must be a non-empty string")
        if not isinstance(prediction, str):
            raise InputValidationError(f"line {line_number}: prediction must be a string")
        if not isinstance(reference, str):
            raise InputValidationError(f"line {line_number}: reference must be a string")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise InputValidationError(f"line {line_number}: metadata must be an object")
        latency = row.get("inference_latency_ms")
        normalized_latency: float | None = None
        if latency is not None:
            if isinstance(latency, bool) or not isinstance(latency, int | float):
                raise InputValidationError(
                    f"line {line_number}: inference_latency_ms must be numeric or null"
                )
            normalized_latency = float(latency)
            if not math.isfinite(normalized_latency) or normalized_latency < 0:
                raise InputValidationError(
                    f"line {line_number}: inference_latency_ms must be finite and non-negative"
                )
        return cls(
            raw=dict(row),
            line_number=line_number,
            id=sample_id.strip(),
            task_type=task_type.strip().lower(),
            prediction=prediction,
            reference=reference,
            metadata=dict(metadata),
            inference_latency_ms=normalized_latency,
        )


def read_prediction_jsonl(
    path: Path,
    *,
    strict: bool,
) -> tuple[list[PredictionRecord], list[dict[str, Any]]]:
    """读取 JSONL；非 strict 模式跳过坏行并返回错误清单。"""

    if not path.is_file():
        raise InputValidationError(f"predictions file does not exist: {path}")
    records: list[PredictionRecord] = []
    errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    # Accept both ordinary UTF-8 and BOM-prefixed JSONL produced by common
    # Windows tooling without changing the bytes used for provenance hashes.
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record = PredictionRecord.from_mapping(payload, line_number)
                if record.id in seen_ids:
                    raise InputValidationError(
                        f"line {line_number}: duplicate prediction id: {record.id}"
                    )
            except (json.JSONDecodeError, InputValidationError) as exc:
                if strict:
                    raise InputValidationError(str(exc)) from exc
                errors.append(
                    {
                        "line_number": line_number,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                continue
            seen_ids.add(record.id)
            records.append(record)
    if not records:
        raise InputValidationError("predictions file contains no valid records")
    return records, errors
