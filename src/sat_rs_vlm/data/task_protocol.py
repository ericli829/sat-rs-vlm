"""遥感结构化任务的基础协议和解析器。

本模块只处理任务数据语义，不负责评测聚合或可靠性判定。训练数据转换、普通评测、
量化 benchmark 和可靠性输出验证均可复用这里的 counting、detection 与坐标解析逻辑。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class BBoxFormat(str, Enum):
    """输入框坐标格式；禁止根据数值范围静默猜测。"""

    NORMALIZED_0_1 = "normalized_0_1"
    PERCENT_0_100 = "percent_0_100"
    SCALED_0_1000 = "scaled_0_1000"
    PIXEL_XYXY = "pixel_xyxy"


@dataclass(frozen=True)
class CountingParseResult:
    """计数解析结果；`value=None` 表示无法可靠归一化。"""

    value: int | None
    reason: str | None = None


@dataclass(frozen=True)
class DetectionParseResult:
    """Detection JSON 的规范化结果。"""

    label: str
    bbox: tuple[float, float, float, float]
    valid_coordinate_range: bool


_INTEGER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?![\w.])")
_WORD_TOKEN_PATTERN = re.compile(r"[a-z]+")
_SMALL_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_ZERO_PHRASES = re.compile(r"\b(?:no|none|nothing|zero)\b", re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """从纯 JSON、Markdown code fence 或混合文本中提取第一个 JSON object。"""

    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        candidates.insert(0, fenced.group(1))
    decoder = json.JSONDecoder()
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(json.dumps(payload))
            break
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _parse_number_words(text: str) -> list[int]:
    tokens = _WORD_TOKEN_PATTERN.findall(text.lower().replace("-", " "))
    results: list[int] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in _SMALL_NUMBERS and token not in _TENS:
            index += 1
            continue
        value = 0
        consumed = False
        while index < len(tokens):
            token = tokens[index]
            if token in _SMALL_NUMBERS:
                value += _SMALL_NUMBERS[token]
            elif token in _TENS:
                value += _TENS[token]
            elif token == "hundred" and consumed:
                value *= 100
            elif token == "and" and consumed:
                index += 1
                continue
            else:
                break
            consumed = True
            index += 1
        if consumed:
            results.append(value)
    return results


def parse_count(value: Any) -> CountingParseResult:
    """把结构化或自然语言计数转换为非负整数。

    支持整数、`{"count": 2}`、数字文本、英文数字以及 no/none。出现多个互相矛盾的
    数字、负数或小数时返回 unresolved，不会凭空选择一个数字。
    """

    if isinstance(value, bool):
        return CountingParseResult(None, "boolean_is_not_count")
    if isinstance(value, int):
        return (
            CountingParseResult(value)
            if value >= 0
            else CountingParseResult(None, "negative_count")
        )
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer() or value < 0:
            return CountingParseResult(None, "count_must_be_non_negative_integer")
        return CountingParseResult(int(value))
    if isinstance(value, dict):
        if "count" not in value:
            return CountingParseResult(None, "count_field_missing")
        return parse_count(value["count"])

    text = str(value).strip()
    if not text:
        return CountingParseResult(None, "empty_count")
    payload = extract_json_object(text)
    if payload is not None and "count" in payload:
        return parse_count(payload["count"])

    digit_values = [int(match.group(0)) for match in _INTEGER_PATTERN.finditer(text)]
    if digit_values:
        unique = set(digit_values)
        if len(unique) != 1:
            return CountingParseResult(None, "ambiguous_multiple_counts")
        return parse_count(digit_values[0])
    word_values = _parse_number_words(text)
    if word_values:
        unique = set(word_values)
        if len(unique) != 1:
            return CountingParseResult(None, "ambiguous_multiple_counts")
        return parse_count(word_values[0])
    if _ZERO_PHRASES.search(text):
        return CountingParseResult(0)
    return CountingParseResult(None, "count_unresolved")


def counting_json(value: Any) -> str | None:
    """返回紧凑 `{"count":n}`；无法可靠解析时返回 None。"""

    parsed = parse_count(value)
    if parsed.value is None:
        return None
    return json.dumps({"count": parsed.value}, ensure_ascii=False, separators=(",", ":"))


def normalize_bbox(
    values: Any,
    *,
    source_format: str | BBoxFormat,
    target_format: str | BBoxFormat = BBoxFormat.NORMALIZED_0_1,
    image_size: tuple[int, int] | None = None,
    clip: bool = True,
) -> tuple[list[float], bool]:
    """按显式格式把 xyxy 坐标转换到 normalized 0-1。

    `pixel_xyxy` 必须提供 `(width, height)`。函数不根据坐标最大值猜测来源格式。
    返回 `(normalized_bbox, changed)`，其中 changed 表示发生缩放、裁剪或端点重排。
    """

    source = BBoxFormat(source_format)
    target = BBoxFormat(target_format)
    if target is not BBoxFormat.NORMALIZED_0_1:
        raise ValueError(f"Unsupported bbox target_format: {target}")
    raw = [float(item) for item in list(values or [])]
    if len(raw) != 4:
        raise ValueError(f"bbox must contain 4 values, got: {raw}")
    if not all(math.isfinite(item) for item in raw):
        raise ValueError(f"bbox values must be finite, got: {raw}")
    if source is BBoxFormat.NORMALIZED_0_1:
        scaled = list(raw)
    elif source is BBoxFormat.PERCENT_0_100:
        scaled = [item / 100.0 for item in raw]
    elif source is BBoxFormat.SCALED_0_1000:
        scaled = [item / 1000.0 for item in raw]
    else:
        if image_size is None or image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("pixel_xyxy bbox requires positive image_size=(width, height)")
        width, height = image_size
        scaled = [raw[0] / width, raw[1] / height, raw[2] / width, raw[3] / height]
    if clip:
        scaled = [min(1.0, max(0.0, item)) for item in scaled]
    elif any(item < 0.0 or item > 1.0 for item in scaled):
        raise ValueError(f"normalized bbox is outside [0,1]: {scaled}")
    x_min, x_max = sorted((scaled[0], scaled[2]))
    y_min, y_max = sorted((scaled[1], scaled[3]))
    normalized = [x_min, y_min, x_max, y_max]
    return normalized, raw != normalized


def parse_detection(
    value: Any,
    *,
    coordinate_format: str | BBoxFormat = BBoxFormat.NORMALIZED_0_1,
    image_size: tuple[int, int] | None = None,
) -> DetectionParseResult | None:
    """解析单目标 detection JSON，并分别报告坐标范围是否合法。

    新协议使用 ``label``/``bbox``。为兼容合并前已经生成的单目标 fixture，也接受
    仅含一个元素的 ``labels``/``boxes``；多目标旧结构不会被静默截断。
    """

    payload = value if isinstance(value, dict) else extract_json_object(str(value))
    if not isinstance(payload, dict):
        return None
    label_value = payload.get("label")
    bbox_value = payload.get("bbox")
    if label_value is None or bbox_value is None:
        labels = payload.get("labels")
        boxes = payload.get("boxes")
        if not isinstance(labels, list) or not isinstance(boxes, list):
            return None
        if len(labels) != 1 or len(boxes) != 1:
            return None
        label_value = labels[0]
        bbox_value = boxes[0]
    label = str(label_value).strip().lower()
    try:
        raw = [float(item) for item in list(bbox_value)]
    except (TypeError, ValueError):
        return None
    if len(raw) != 4 or not label or not all(math.isfinite(item) for item in raw):
        return None
    coordinate_valid = True
    try:
        normalized, _ = normalize_bbox(
            raw,
            source_format=coordinate_format,
            image_size=image_size,
            clip=False,
        )
    except ValueError:
        normalized = raw
        coordinate_valid = False
    if not (normalized[0] < normalized[2] and normalized[1] < normalized[3]):
        coordinate_valid = False
    x_min, y_min, x_max, y_max = normalized
    return DetectionParseResult(
        label,
        (x_min, y_min, x_max, y_max),
        coordinate_valid,
    )
