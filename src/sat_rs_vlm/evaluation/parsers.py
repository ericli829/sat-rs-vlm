"""与 sat-rs-vlm@449bc857 字段约定兼容的独立解析器。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JsonObjectResult:
    payload: dict[str, Any] | None
    strict_json: bool
    reason: str | None


@dataclass(frozen=True)
class GroundingParseResult:
    label: str | None
    bbox: tuple[float, float, float, float] | None
    valid_json: bool
    parse_ok: bool
    coordinate_valid: bool
    parse_error: str | None
    coordinate_error: str | None


@dataclass(frozen=True)
class CountParseResult:
    value: int | None
    reason: str | None = None


@dataclass(frozen=True)
class ChangePredictionResult:
    value: int | None
    normalized_text: str
    reason: str | None = None


_INTEGER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?![\w.])")
_WORD_TOKEN_PATTERN = re.compile(r"[a-z]+")
_ZERO_PHRASES = re.compile(r"\b(?:no|none|nothing|zero)\b", re.IGNORECASE)
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

_NO_CHANGE_EXPRESSIONS = {
    "no change has occurred",
    "no change occurred",
    "no changes have occurred",
    "there is no change",
    "there are no changes",
    "there is no difference",
    "no difference",
    "the two scenes seem identical",
    "the two scenes are identical",
    "the scene is the same as before",
    "the scene remains the same as before",
    "almost nothing has changed",
    "nothing has changed",
    "unchanged",
}


def extract_json_object(text: str) -> JsonObjectResult:
    """解析纯 JSON、Markdown code fence 或混合文本中的首个 JSON object。"""

    stripped = text.strip()
    if not stripped:
        return JsonObjectResult(None, False, "empty_text")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return JsonObjectResult(payload, True, None)

    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return JsonObjectResult(payload, False, None)

    decoder = json.JSONDecoder()
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return JsonObjectResult(payload, False, None)
    return JsonObjectResult(None, False, "json_object_not_found")


def _normalize_bbox(
    values: list[float],
    coordinate_format: str | None,
    image_size: tuple[int, int] | None,
) -> tuple[tuple[float, float, float, float], bool, str | None]:
    def bbox_tuple(items: list[float]) -> tuple[float, float, float, float]:
        return (float(items[0]), float(items[1]), float(items[2]), float(items[3]))

    if coordinate_format is None:
        return bbox_tuple(values), False, "coordinate_format_unresolved"
    if coordinate_format == "normalized_0_1":
        scaled = values
    elif coordinate_format == "percent_0_100":
        scaled = [value / 100.0 for value in values]
    elif coordinate_format == "scaled_0_1000":
        scaled = [value / 1000.0 for value in values]
    elif coordinate_format == "pixel_xyxy":
        if image_size is None or image_size[0] <= 0 or image_size[1] <= 0:
            return bbox_tuple(values), False, "pixel_coordinates_require_image_size"
        width, height = image_size
        scaled = [
            values[0] / width,
            values[1] / height,
            values[2] / width,
            values[3] / height,
        ]
    else:
        return bbox_tuple(values), False, f"unsupported_coordinate_format:{coordinate_format}"

    bbox = bbox_tuple(scaled)
    x_min, y_min, x_max, y_max = bbox
    if not all(0.0 <= value <= 1.0 for value in bbox):
        return bbox, False, "bbox_out_of_range"
    if not (x_min < x_max and y_min < y_max):
        return bbox, False, "bbox_not_strict_xyxy"
    return bbox, True, None


def parse_grounding(
    value: Any,
    *,
    coordinate_format: str | None,
    image_size: tuple[int, int] | None = None,
) -> GroundingParseResult:
    """解析单目标 label+bbox，并将显式坐标格式转换到 normalized_0_1。"""

    extracted = (
        JsonObjectResult(value, True, None)
        if isinstance(value, dict)
        else extract_json_object(str(value))
    )
    payload = extracted.payload
    if payload is None:
        return GroundingParseResult(
            None,
            None,
            False,
            False,
            False,
            extracted.reason,
            None,
        )

    label_value = payload.get("label")
    bbox_value = payload.get("bbox")
    if label_value is None or bbox_value is None:
        labels = payload.get("labels")
        boxes = payload.get("boxes")
        if not isinstance(labels, list) or not isinstance(boxes, list):
            return GroundingParseResult(
                None, None, True, False, False, "label_or_bbox_missing", None
            )
        if len(labels) != 1 or len(boxes) != 1:
            return GroundingParseResult(
                None, None, True, False, False, "legacy_schema_must_have_one_target", None
            )
        label_value = labels[0]
        bbox_value = boxes[0]

    label = str(label_value).strip().lower()
    if not label:
        return GroundingParseResult(None, None, True, False, False, "empty_label", None)
    if not isinstance(bbox_value, (list, tuple)) or len(bbox_value) != 4:
        return GroundingParseResult(
            label, None, True, False, False, "bbox_must_have_four_values", None
        )
    try:
        raw_bbox = [float(item) for item in bbox_value]
    except (TypeError, ValueError):
        return GroundingParseResult(
            label, None, True, False, False, "bbox_contains_non_numeric_value", None
        )
    if not all(math.isfinite(item) for item in raw_bbox):
        return GroundingParseResult(
            label, None, True, False, False, "bbox_contains_non_finite_value", None
        )
    bbox, coordinate_valid, coordinate_error = _normalize_bbox(
        raw_bbox,
        coordinate_format,
        image_size,
    )
    return GroundingParseResult(
        label,
        bbox,
        True,
        True,
        coordinate_valid,
        None,
        coordinate_error,
    )


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


def parse_count(value: Any) -> CountParseResult:
    """把结构化或自然语言计数转换为唯一非负整数。"""

    if isinstance(value, bool):
        return CountParseResult(None, "boolean_is_not_count")
    if isinstance(value, int):
        return CountParseResult(value) if value >= 0 else CountParseResult(None, "negative_count")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer() or value < 0:
            return CountParseResult(None, "count_must_be_non_negative_integer")
        return CountParseResult(int(value))
    if isinstance(value, dict):
        if "count" not in value:
            return CountParseResult(None, "count_field_missing")
        return parse_count(value["count"])

    text = str(value).strip()
    if not text:
        return CountParseResult(None, "empty_count")
    payload = extract_json_object(text).payload
    if payload is not None and "count" in payload:
        return parse_count(payload["count"])

    digit_values = [int(match.group(0)) for match in _INTEGER_PATTERN.finditer(text)]
    if digit_values:
        unique = set(digit_values)
        if len(unique) != 1:
            return CountParseResult(None, "ambiguous_multiple_counts")
        return parse_count(digit_values[0])
    word_values = _parse_number_words(text)
    if word_values:
        unique = set(word_values)
        if len(unique) != 1:
            return CountParseResult(None, "ambiguous_multiple_counts")
        return parse_count(word_values[0])
    if _ZERO_PHRASES.search(text):
        return CountParseResult(0)
    return CountParseResult(None, "count_unresolved")


def normalize_text(text: str) -> str:
    """与当前仓库一致：小写、去引号和标点、压缩空白。"""

    value = re.sub(r"[\"'`]", "", text.strip().lower())
    value = re.sub(r"[^\w\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def parse_change_prediction(text: str) -> ChangePredictionResult:
    """把 LEVIR-CC 自然语言输出解析为 0=无变化、1=发生变化。

    无变化只接受完整标准化表达，避免把包含否定词的复合变化描述误判为无变化。
    """

    normalized = normalize_text(text)
    if not normalized:
        return ChangePredictionResult(None, normalized, "empty_prediction")
    return ChangePredictionResult(
        0 if normalized in _NO_CHANGE_EXPRESSIONS else 1,
        normalized,
        None,
    )


def text_tokens(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())
