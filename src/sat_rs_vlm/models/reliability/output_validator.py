"""遥感 VLM 输出的结构、数值与任务约束验证。

验证器接受模型原始文本、字典或项目的 Pydantic 推理结果。错误和警告使用稳定代码，
`normalized_output` 给保护策略和指标模块提供统一输入，不把自然语言错误信息作为协议。
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from sat_rs_vlm.data.task_protocol import parse_count
from sat_rs_vlm.domain.tasks import TaskType
from sat_rs_vlm.models.reliability.schemas import ValidationResult

Number = int | float
VqaQuestionType = Literal["yes_no", "number", "direction", "short_answer"]
NON_FINITE_PATTERN = re.compile(r"(?<!\w)(?:nan|[-+]?inf(?:inity)?)(?!\w)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
REPEATED_CHARACTER_PATTERN = re.compile(r"(.)\1{15,}", re.DOTALL)
REPEATED_TOKEN_PATTERN = re.compile(r"\b(\w+)(?:\s+\1){5,}\b", re.IGNORECASE)


def extract_first_number(text: str) -> int | float | None:
    """提取文本中的第一个十进制数；整数保持 int，其他值返回 float。"""

    match = NUMBER_PATTERN.search(text)
    if match is None:
        return None
    value = float(match.group(0))
    return int(value) if value.is_integer() else value


def _append_once(values: list[str], code: str) -> None:
    if code not in values:
        values.append(code)


def _contains_non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, str):
        return NON_FINITE_PATTERN.search(value) is not None
    if isinstance(value, Mapping):
        return any(_contains_non_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_non_finite(item) for item in value)
    return False


def _contains_letter_or_number(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _detect_degenerate_text(text: str, errors: list[str]) -> None:
    """Detect obvious generation collapse before confidence voting.

    These checks intentionally target only high-confidence degeneration patterns:
    repeated symbols such as ``!!!!!!!!`` or ``???????``, one-token loops, and
    very low-diversity long strings. Short valid answers such as ``yes`` or ``3``
    should not be rejected by this generic guard.
    """

    stripped = text.strip()
    if not stripped:
        return
    if REPEATED_CHARACTER_PATTERN.search(stripped):
        _append_once(errors, "degenerate_repeated_character")
    if REPEATED_TOKEN_PATTERN.search(stripped):
        _append_once(errors, "degenerate_repeated_token")
    if len(stripped) >= 8 and not _contains_letter_or_number(stripped):
        _append_once(errors, "degenerate_symbol_only")
    compact = "".join(stripped.split())
    if len(compact) >= 32:
        unique_count = len(set(compact))
        most_common = max((compact.count(character) for character in set(compact)), default=0)
        if unique_count <= 3 or most_common / len(compact) >= 0.85:
            _append_once(errors, "degenerate_low_diversity")


def _as_payload(output: Any) -> Any:
    if isinstance(output, BaseModel):
        return output.model_dump(mode="json")
    if isinstance(output, Mapping):
        return dict(output)
    return output


def _parse_json_text(text: str, errors: list[str]) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        _append_once(errors, "invalid_json")
        return text


def _coordinates(box: Any) -> tuple[list[Any] | None, str]:
    if isinstance(box, Mapping):
        label = str(box.get("label", "")).strip()
        nested = box.get("bbox")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
            return list(nested), label
        keys = ("x_min", "y_min", "x_max", "y_max")
        if all(key in box for key in keys):
            return [box[key] for key in keys], label
        alternate = ("x1", "y1", "x2", "y2")
        if all(key in box for key in alternate):
            return [box[key] for key in alternate], label
        return None, label
    if isinstance(box, Sequence) and not isinstance(box, (str, bytes, bytearray)):
        return list(box), ""
    return None, ""


def _validate_detection(
    payload: Any,
    errors: list[str],
    *,
    coordinate_range: tuple[float, float],
) -> dict[str, Any] | None:
    default_label = ""
    if isinstance(payload, Mapping):
        default_label = str(payload.get("label", "")).strip()
        if "boxes" in payload:
            boxes = payload["boxes"]
        elif "bbox" in payload:
            boxes = [{"label": default_label, "bbox": payload["bbox"]}]
        else:
            boxes = None
    elif isinstance(payload, list):
        boxes = payload
    else:
        boxes = None
    if not isinstance(boxes, list) or not boxes:
        _append_once(errors, "detection_bbox_missing")
        return None

    normalized_boxes: list[dict[str, Any]] = []
    low, high = coordinate_range
    for box in boxes:
        coordinates, label = _coordinates(box)
        label = label or default_label
        if not label:
            _append_once(errors, "detection_label_empty")
        if coordinates is None or len(coordinates) != 4:
            _append_once(errors, "detection_bbox_format")
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in coordinates)
        except (TypeError, ValueError):
            _append_once(errors, "detection_bbox_non_numeric")
            continue
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            _append_once(errors, "non_finite_value")
            continue
        if not (x1 < x2 and y1 < y2):
            _append_once(errors, "detection_bbox_invalid_order")
        if not all(low <= value <= high for value in (x1, y1, x2, y2)):
            _append_once(errors, "detection_bbox_out_of_range")
            # 兼容旧测试和已生成报告中的稳定错误码。
            _append_once(errors, "bbox_out_of_unit_range")
        normalized_boxes.append({"label": label, "bbox": [x1, y1, x2, y2]})
    return {"boxes": normalized_boxes}


def _validate_counting(
    payload: Any, errors: list[str], warnings: list[str]
) -> dict[str, int] | None:
    candidate: Any = payload
    if isinstance(payload, Mapping):
        if "count" in payload:
            candidate = payload["count"]
        elif "answer" in payload:
            candidate = payload["answer"]
            _append_once(warnings, "counting_unstructured")
        else:
            _append_once(errors, "counting_count_missing")
            _append_once(errors, "counting_number_missing")
            return None
    elif isinstance(payload, str):
        _append_once(warnings, "counting_unstructured")

    parsed = parse_count(candidate)
    if parsed.value is None:
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            numeric = float(candidate)
            if numeric < 0:
                _append_once(errors, "counting_negative")
            if math.isfinite(numeric) and not numeric.is_integer():
                _append_once(errors, "counting_not_integer")
        _append_once(errors, "counting_number_missing")
        return None
    return {"count": parsed.value}


def _validate_vqa(
    payload: Any,
    errors: list[str],
    *,
    question_type: VqaQuestionType | None,
) -> str | int | float | None:
    answer = payload.get("answer") if isinstance(payload, Mapping) else payload
    text = str(answer).strip()
    if not text:
        _append_once(errors, "vqa_answer_empty")
        return None
    if len(text) > 128:
        _append_once(errors, "vqa_answer_too_long")
    normalized = " ".join(text.split())
    lowered = normalized.lower()
    if question_type == "yes_no":
        mapping = {"yes": "yes", "no": "no", "是": "yes", "否": "no"}
        if lowered not in mapping:
            _append_once(errors, "vqa_yes_no_invalid")
        else:
            normalized = mapping[lowered]
    elif question_type == "number":
        number = extract_first_number(normalized)
        if number is None:
            _append_once(errors, "vqa_number_missing")
        else:
            return number
    elif question_type == "direction":
        allowed = {
            "north",
            "south",
            "east",
            "west",
            "northeast",
            "northwest",
            "southeast",
            "southwest",
            "北",
            "南",
            "东",
            "西",
            "东北",
            "西北",
            "东南",
            "西南",
        }
        if lowered not in allowed:
            _append_once(errors, "vqa_direction_invalid")
    return normalized


def validate_prediction(
    task_type: str | TaskType,
    output: Any,
    *,
    max_length: int = 4096,
    coordinate_range: tuple[float, float] = (0.0, 1.0),
    vqa_question_type: VqaQuestionType | None = None,
) -> ValidationResult:
    """验证单条模型输出并返回规范化结果。

    detection 默认要求 `[0, 1]` 范围的 `xyxy`；counting 要求非负整数；VQA 可通过
    `vqa_question_type` 启用 Yes/No、数字或方位约束。其他任务执行通用空值、JSON、
    非有限值和长度检查。
    """

    task = task_type.value if isinstance(task_type, TaskType) else str(task_type)
    errors: list[str] = []
    warnings: list[str] = []
    payload = _as_payload(output)
    if payload is None:
        return ValidationResult(valid=False, errors=["output_empty"])
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return ValidationResult(valid=False, errors=["output_empty"])
        if len(stripped) > max_length:
            _append_once(errors, "output_too_long")
        _detect_degenerate_text(stripped, errors)
        should_parse = task == TaskType.DETECTION.value or stripped.startswith(("{", "["))
        payload = _parse_json_text(stripped, errors) if should_parse else stripped
    if isinstance(payload, (Mapping, list)) and not payload:
        _append_once(errors, "empty_json")
    if _contains_non_finite(payload):
        _append_once(errors, "non_finite_value")

    normalized: Any = payload
    if task == TaskType.DETECTION.value and "invalid_json" not in errors:
        normalized = _validate_detection(payload, errors, coordinate_range=coordinate_range)
    elif task == TaskType.COUNTING.value:
        normalized = _validate_counting(payload, errors, warnings)
    elif task == TaskType.VQA.value:
        normalized = _validate_vqa(
            payload,
            errors,
            question_type=vqa_question_type,
        )
    return ValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        normalized_output=normalized,
    )
