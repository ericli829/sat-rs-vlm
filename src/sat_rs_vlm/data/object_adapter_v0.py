"""RS Object Adapter v0 的确定性数据审计、泄漏保护和监督构建工具。

本模块只处理 VRSBench detection/counting 数据，不加载模型，也不执行训练。
核心算法是：先按 image identity 删除所有评测图片，再复用项目已有的
``parse_detection``/``parse_count`` 解析器，把 annotation row 聚合为
``(image, target_class)`` pair，最后按 image 做稳定 train/validation 划分。
任何无法可靠解析的内容都会进入 audit 并被排除，不会通过猜测修复。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.task_protocol import parse_count, parse_detection
from sat_rs_vlm.evaluation.parsers import normalize_text
from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl

SCHEMA_VERSION = "1.0"
BUILDER_VERSION = "rs-object-adapter-v0-1.1"
SUPPORTED_TASKS = frozenset({"detection", "counting"})
COUNT_BINS = ("0-2", "3-5", "6-10", "11+")
NEUTRAL_QUANTITY_MODIFIERS = frozenset({"unique", "distinct", "individual", "separate", "total"})

# 仅用于 VRSBench counting QA 与 detection taxonomy 的已确认等价命名。
# 这里故意不包含 car/vehicle、boat/ship 等上位或下位概念映射，因为 v0 的
# class-conditioned adapter 无法表达这些问题中的属性和子集限定。
VRSBENCH_COUNTING_EQUIVALENT_ALIASES = {
    "airplane": ["plane", "planes"],
    "baseball diamond": ["baseball field", "baseball fields"],
    "trainstation": ["train station", "train stations"],
    "soccer ball field": ["soccer field", "soccer fields"],
    "golffield": ["golf course", "golf courses", "golf field", "golf fields"],
    "ground track field": [
        "ground track and field area",
        "ground track and field areas",
    ],
}


class DataAuditBlocked(ValueError):
    """表示数据审计 hard blocker，调用方不得继续进入训练。"""

    def __init__(self, blockers: Sequence[str], manifest_path: Path | None = None) -> None:
        self.blockers = tuple(str(item) for item in blockers)
        suffix = f"; manifest={manifest_path}" if manifest_path is not None else ""
        super().__init__("Object Adapter v0 data audit blocked: " + "; ".join(self.blockers) + suffix)


@dataclass(frozen=True)
class ClassResolution:
    """确定性类别解析结果。"""

    class_name: str | None
    status: str
    source: str | None = None
    matches: tuple[str, ...] = ()


@dataclass(frozen=True)
class DetectionBox:
    """一条已经通过坐标审计的 detection 框。"""

    sample_id: str
    image: str
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]


def canonical_image_identity(row_or_messages: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> str:
    """从 messages/image 字段提取完整、可移植的图片身份。

    算法只统一斜杠并移除开头的 ``./``，不会退化为 basename；这样同名图片位于
    不同目录时仍被视为不同图片。返回空字符串表示输入没有可靠图片项。
    """

    messages: Any
    if isinstance(row_or_messages, Mapping):
        messages = row_or_messages.get("messages")
        if messages is None:
            values = row_or_messages.get("images", [])
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                messages = [{"content": [{"type": "image", "image": value}]} for value in values]
    else:
        messages = row_or_messages
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ""
    identities: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content", [])
        if isinstance(content, Mapping):
            content = [content]
        if isinstance(content, str):
            continue
        if not isinstance(content, Sequence):
            continue
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "image":
                continue
            raw = item.get("image", item.get("path", ""))
            value = str(raw).strip().replace("\\", "/")
            while value.startswith("./"):
                value = value[2:]
            if value:
                identities.append(value)
    unique = sorted(set(identities))
    if len(unique) != 1:
        return "" if not unique else "|".join(unique)
    return unique[0]


def extract_prompt(row: Mapping[str, Any]) -> str:
    """提取 user 消息中的完整文本，供 counting 类别确定性匹配使用。"""

    messages = row.get("messages", [])
    if not isinstance(messages, Sequence):
        return str(row.get("instruction", ""))
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping) or str(message.get("role", "")).lower() != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, Sequence):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
    if parts:
        return " ".join(parts).strip()
    return str(row.get("instruction", "")).strip()


def extract_answer(row: Mapping[str, Any]) -> Any:
    """提取 assistant answer；兼容 messages 和旧 instruction/answer 格式。"""

    messages = row.get("messages", [])
    if isinstance(messages, Sequence):
        for message in reversed(messages):
            if isinstance(message, Mapping) and str(message.get("role", "")).lower() == "assistant":
                return message.get("content", "")
    return row.get("answer", "")


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metadata", {})
    return value if isinstance(value, Mapping) else {}


def dataset_name(row: Mapping[str, Any]) -> str:
    """读取 dataset 名称；不对未知来源进行语义猜测。"""

    return str(_metadata(row).get("dataset", row.get("dataset", ""))).strip()


def normalize_class(value: Any) -> str:
    """把标签标准化为类别词，不生成语义同义词。"""

    return normalize_text(str(value)).replace("_", " ").replace("-", " ").strip()


def _regular_plural(label: str) -> str:
    """按小型英语规则生成类别标签复数，不引入词形库。"""

    normalized = normalize_class(label)
    if not normalized:
        return ""
    prefix, separator, last_word = normalized.rpartition(" ")
    if last_word.endswith(("s", "x", "z", "ch", "sh")):
        plural_last_word = f"{last_word}es"
    elif len(last_word) > 1 and last_word.endswith("y") and last_word[-2] not in "aeiou":
        plural_last_word = f"{last_word[:-1]}ies"
    else:
        plural_last_word = f"{last_word}s"
    return f"{prefix}{separator}{plural_last_word}" if separator else plural_last_word


def _class_aliases(class_name: str) -> list[str]:
    """生成 canonical 标签和确定性规则复数。"""

    normalized = normalize_class(class_name)
    aliases = [normalized, _regular_plural(normalized)]
    return list(dict.fromkeys(alias for alias in aliases if alias))


def build_class_vocab(class_names: Iterable[str]) -> dict[str, Any]:
    """从合法 detection 标签构造固定 class vocabulary。"""

    classes = sorted({normalize_class(name) for name in class_names if normalize_class(name)})
    return {
        "schema_version": SCHEMA_VERSION,
        "classes": classes,
        "class_to_id": {name: index for index, name in enumerate(classes)},
        "aliases": {name: _class_aliases(name) for name in classes},
    }


def _vocab_alias_index(class_vocab: Mapping[str, Any]) -> dict[str, set[str]]:
    classes = {normalize_class(value) for value in class_vocab.get("classes", [])}
    index: dict[str, set[str]] = defaultdict(set)
    aliases = class_vocab.get("aliases", {})
    if isinstance(aliases, Mapping):
        for class_name, values in aliases.items():
            canonical = normalize_class(class_name)
            if canonical not in classes:
                continue
            values_iter = values if isinstance(values, Sequence) and not isinstance(values, str) else [values]
            for alias in values_iter:
                normalized = normalize_class(alias)
                if normalized:
                    index[normalized].add(canonical)
    for class_name in classes:
        for alias in _class_aliases(class_name):
            index[alias].add(class_name)
    return index


def _counting_alias_index(class_vocab: Mapping[str, Any]) -> dict[str, set[str]]:
    """构建 counting 专用 alias 索引，不继承旧 vocab 中的宽松 prompt alias。"""

    classes = {normalize_class(value) for value in class_vocab.get("classes", [])}
    index: dict[str, set[str]] = defaultdict(set)
    for class_name in classes:
        aliases = [*_class_aliases(class_name), *VRSBENCH_COUNTING_EQUIVALENT_ALIASES.get(class_name, [])]
        for alias in aliases:
            normalized = normalize_class(alias)
            if normalized:
                index[normalized].add(class_name)
    return index


def _longest_prompt_matches(prompt: str, alias_index: Mapping[str, set[str]]) -> tuple[str, ...]:
    normalized_prompt = normalize_text(prompt).replace("_", " ").replace("-", " ")
    candidates: list[tuple[int, str, str]] = []
    for alias, classes in alias_index.items():
        if not alias:
            continue
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        if re.search(pattern, normalized_prompt, flags=re.IGNORECASE):
            for class_name in classes:
                candidates.append((len(alias.split()), alias, class_name))
    if not candidates:
        return ()
    longest = max(item[0] for item in candidates)
    return tuple(sorted({item[2] for item in candidates if item[0] == longest}))


def resolve_prompt_class(prompt: str, class_vocab: Mapping[str, Any]) -> ClassResolution:
    """Resolve a class from prompt aliases without consulting a reference answer."""

    matches = _longest_prompt_matches(prompt, _vocab_alias_index(class_vocab))
    if len(matches) == 1:
        return ClassResolution(matches[0], "resolved", "prompt", matches)
    if len(matches) > 1:
        return ClassResolution(None, "ambiguous", "prompt", matches)
    return ClassResolution(None, "unresolved", None, ())


def _strip_counting_output_protocol(prompt: str) -> str:
    """移除训练 prompt 附加的输出协议，避免协议文本参与目标类别解析。"""

    match = re.search(
        r"\b(?:return\s+only\s+(?:a\s+json\s+object|the\s+integer)|"
        r"do\s+not\s+include\s+any\s+other\s+text)\b",
        prompt,
        flags=re.IGNORECASE,
    )
    return prompt[: match.start()].strip() if match else prompt.strip()


def _cardinality_prompt_target(prompt: str) -> str | None:
    """提取 exact-cardinality 问题开头的 target remainder；其他形式返回 ``None``。"""

    normalized = normalize_text(_strip_counting_output_protocol(prompt)).replace("_", " ").replace("-", " ")
    normalized = normalized.strip()
    match = re.match(r"^how\s+many\s+(?P<target>.+)$", normalized, flags=re.IGNORECASE)
    if match is None:
        match = re.match(
            r"^what\s+is\s+(?:the\s+)?(?:total\s+)?number\s+of\s+(?P<target>.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
    if match is None:
        return None
    target = match.group("target").strip()
    while target:
        modifier = re.match(r"^(unique|distinct|individual|separate|total)\b\s*", target, flags=re.IGNORECASE)
        if modifier is None or modifier.group(1).lower() not in NEUTRAL_QUANTITY_MODIFIERS:
            break
        target = target[modifier.end() :].strip()
    return target or None


def resolve_cardinality_prompt_class(prompt: str, class_vocab: Mapping[str, Any]) -> ClassResolution:
    """只从 exact-cardinality 问题的 target 开头解析 counting 类别。"""

    target = _cardinality_prompt_target(prompt)
    if target is None:
        return ClassResolution(None, "unsupported_form", "cardinality_prompt", ())
    candidates: list[tuple[int, int, str]] = []
    for alias, classes in _counting_alias_index(class_vocab).items():
        if re.match(rf"^{re.escape(alias)}(?!\w)", target, flags=re.IGNORECASE):
            for class_name in classes:
                candidates.append((len(alias.split()), len(alias), class_name))
    if not candidates:
        return ClassResolution(None, "unresolved", "cardinality_prompt", ())
    best_tokens = max(item[0] for item in candidates)
    best_characters = max(item[1] for item in candidates if item[0] == best_tokens)
    matches = tuple(
        sorted(
            {
                item[2]
                for item in candidates
                if item[0] == best_tokens and item[1] == best_characters
            }
        )
    )
    if len(matches) == 1:
        return ClassResolution(matches[0], "resolved", "cardinality_prompt", matches)
    return ClassResolution(None, "ambiguous", "cardinality_prompt", matches)


def resolve_counting_class(row: Mapping[str, Any], class_vocab: Mapping[str, Any]) -> ClassResolution:
    """按 metadata 优先、cardinality target-prefix 其次解析 counting 目标类别。

    不读取 counting answer 推断类别，也不引入 WordNet 或人工语义映射。
    """

    aliases = _counting_alias_index(class_vocab)
    metadata = _metadata(row)
    for field in ("target_class", "object_class", "category", "label"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = normalize_class(value)
        matches = tuple(sorted(aliases.get(normalized, set())))
        if len(matches) == 1:
            return ClassResolution(matches[0], "resolved", f"metadata.{field}", matches)
        if len(matches) > 1:
            return ClassResolution(None, "ambiguous", f"metadata.{field}", matches)
        return ClassResolution(None, "unresolved", f"metadata.{field}", ())
    return resolve_cardinality_prompt_class(extract_prompt(row), class_vocab)


def bbox_iou_xyxy(first: Sequence[float], second: Sequence[float]) -> float:
    """计算 normalized xyxy IoU。"""

    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    area_b = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_detection_boxes(
    boxes: Sequence[DetectionBox], *, iou_threshold: float = 0.95
) -> tuple[list[DetectionBox], int]:
    """按 sample id 排序，移除同 image/class 下 IoU>=阈值的重复框。"""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0,1]")
    retained: list[DetectionBox] = []
    removed = 0
    for box in sorted(boxes, key=lambda item: (item.sample_id, item.bbox_xyxy)):
        if any(bbox_iou_xyxy(box.bbox_xyxy, old.bbox_xyxy) >= iou_threshold for old in retained):
            removed += 1
        else:
            retained.append(box)
    return retained, removed


def count_bin(value: int) -> str:
    """把 count 映射到固定审计分桶。"""

    if value <= 2:
        return "0-2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    return "11+"


def _pair_id(image: str, class_name: str) -> str:
    digest = hashlib.sha256(f"{image}\x1f{class_name}".encode("utf-8")).hexdigest()[:16]
    return f"rs_object_v0_{digest}"


def _parse_training_rows(
    rows: Sequence[Mapping[str, Any]],
    protected_images: set[str],
    class_vocab: Mapping[str, Any] | None,
    audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[str], set[str], dict[str, list[str]]]:
    """解析 detection/counting 行并返回中间 annotation records。"""

    records: list[dict[str, Any]] = []
    before_images: set[str] = set()
    removed_images: set[str] = set()
    unresolved_examples: dict[str, list[str]] = {
        "non_cardinality": [],
        "unresolved": [],
        "ambiguous": [],
    }
    for row in rows:
        if dataset_name(row) != "VRSBench":
            audit["non_vrs_rows_excluded"] = int(audit.get("non_vrs_rows_excluded", 0)) + 1
            continue
        image = canonical_image_identity(row)
        if not image:
            audit["invalid_image_rows"] = int(audit.get("invalid_image_rows", 0)) + 1
            continue
        before_images.add(image)
        if image in protected_images:
            removed_images.add(image)
            continue
        task = str(row.get("task_type", "")).strip().lower()
        if task not in SUPPORTED_TASKS:
            audit["unsupported_task_rows_excluded"] = int(audit.get("unsupported_task_rows_excluded", 0)) + 1
            continue
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            audit["invalid_id_rows"] = int(audit.get("invalid_id_rows", 0)) + 1
            continue
        answer = extract_answer(row)
        if task == "detection":
            audit["detection_total"] += 1
            parsed = parse_detection(
                answer,
                coordinate_format=str(_metadata(row).get("bbox_target_format", "normalized_0_1")),
            )
            if parsed is None or not parsed.valid_coordinate_range:
                audit["detection_invalid"] += 1
                continue
            label = normalize_class(parsed.label)
            if not label:
                audit["detection_invalid"] += 1
                continue
            records.append(
                {
                    "kind": "detection",
                    "sample_id": sample_id,
                    "image": image,
                    "class_name": label,
                    "bbox_xyxy": tuple(float(value) for value in parsed.bbox),
                }
            )
        else:
            # 第一轮只收集 detection 标签来构造 vocabulary；counting 必须等
            # vocabulary 确定后再做一次权威解析，不能使用空 vocabulary 记审计。
            if class_vocab is None:
                continue
            audit["counting_total"] += 1
            parsed_count = parse_count(answer)
            prompt = extract_prompt(row)
            if _cardinality_prompt_target(prompt) is None:
                resolution = ClassResolution(None, "unsupported_form", "cardinality_prompt", ())
            else:
                resolution = resolve_counting_class(row, class_vocab)
            status_distribution = audit["counting_resolution_status_distribution"]
            status_distribution[resolution.status] = int(status_distribution.get(resolution.status, 0)) + 1
            if resolution.status == "unsupported_form":
                audit["counting_non_cardinality_excluded"] += 1
                if len(unresolved_examples["non_cardinality"]) < 30:
                    unresolved_examples["non_cardinality"].append(str(row.get("id", "")) + ": " + prompt)
            else:
                audit["counting_cardinality_eligible"] += 1
            if resolution.status == "resolved":
                audit["counting_class_resolved"] += 1
            elif resolution.status == "ambiguous":
                audit["counting_class_ambiguous"] += 1
                if len(unresolved_examples["ambiguous"]) < 30:
                    unresolved_examples["ambiguous"].append(str(row.get("id", "")) + ": " + extract_prompt(row))
            elif resolution.status == "unresolved":
                audit["counting_class_unresolved"] += 1
                if len(unresolved_examples["unresolved"]) < 30:
                    unresolved_examples["unresolved"].append(str(row.get("id", "")) + ": " + prompt)
            if parsed_count.value is None or resolution.class_name is None:
                continue
            records.append(
                {
                    "kind": "counting",
                    "sample_id": sample_id,
                    "image": image,
                    "class_name": resolution.class_name,
                    "count": int(parsed_count.value),
                }
            )
    audit["detection_valid"] = int(audit["detection_total"] - audit["detection_invalid"])
    audit["detection_parse_rate"] = (
        audit["detection_valid"] / audit["detection_total"] if audit["detection_total"] else 0.0
    )
    audit["counting_class_resolution_rate"] = (
        audit["counting_class_resolved"] / audit["counting_cardinality_eligible"]
        if audit["counting_cardinality_eligible"]
        else 0.0
    )
    audit["train_images_before_exclusion"] = len(before_images)
    audit["train_images_removed_for_eval_overlap"] = len(removed_images)
    audit["train_images_after_exclusion"] = len(before_images - removed_images)
    return records, before_images, removed_images, unresolved_examples


def construct_object_pairs(
    records: Sequence[Mapping[str, Any]],
    class_vocab: Mapping[str, Any],
    *,
    dedup_iou: float = 0.95,
    audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """把 annotation records 聚合为 image/class pair，并严格区分监督类型。"""

    target_audit = audit if audit is not None else _empty_audit()
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"detection": [], "counting": []})
    for record in records:
        key = (str(record["image"]), str(record["class_name"]))
        grouped[key][str(record["kind"])].append(record)
    class_to_id = {str(key): int(value) for key, value in class_vocab.get("class_to_id", {}).items()}
    pairs: list[dict[str, Any]] = []
    conflict_count = 0
    raw_boxes = 0
    dedup_boxes = 0
    duplicate_removed = 0
    for (image, class_name), values in sorted(grouped.items()):
        boxes = [
            DetectionBox(
                sample_id=str(item["sample_id"]),
                image=image,
                class_name=class_name,
                bbox_xyxy=tuple(float(value) for value in item["bbox_xyxy"]),
            )
            for item in values["detection"]
        ]
        raw_boxes += len(boxes)
        unique_boxes, removed = deduplicate_detection_boxes(boxes, iou_threshold=dedup_iou)
        dedup_boxes += len(unique_boxes)
        duplicate_removed += removed
        counts = sorted({int(item["count"]) for item in values["counting"]})
        if len(counts) > 1:
            supervision_type = "conflict"
            conflict_count += 1
        else:
            count_value = counts[0] if counts else None
            box_count = len(unique_boxes)
            if count_value is not None and box_count > count_value:
                supervision_type = "conflict"
                conflict_count += 1
            elif count_value is not None and box_count == count_value:
                supervision_type = "full_set"
            elif count_value is not None and box_count > 0:
                supervision_type = "partial_set"
            elif count_value is not None:
                supervision_type = "count_only"
            elif box_count > 0:
                supervision_type = "detection_only"
            else:
                continue
        if supervision_type == "conflict":
            continue
        source_ids = sorted(
            {str(item["sample_id"]) for item in values["detection"]}
            | {str(item["sample_id"]) for item in values["counting"]}
        )
        pair = {
            "id": _pair_id(image, class_name),
            "image": image,
            "class_name": class_name,
            "class_id": class_to_id[class_name],
            "boxes_xyxy": [list(item.bbox_xyxy) for item in unique_boxes],
            "count": counts[0] if counts else None,
            "supervision_type": supervision_type,
            "source_sample_ids": source_ids,
            "detection_sample_ids": sorted(str(item["sample_id"]) for item in values["detection"]),
            "counting_sample_ids": sorted(str(item["sample_id"]) for item in values["counting"]),
        }
        pairs.append(pair)
    target_audit.update(
        {
            "raw_detection_boxes": raw_boxes,
            "deduplicated_detection_boxes": dedup_boxes,
            "duplicate_boxes_removed": duplicate_removed,
            "conflict_pairs_excluded": conflict_count,
        }
    )
    return pairs


def stable_image_split(
    pairs: Sequence[Mapping[str, Any]], *, seed: int = 42, val_fraction: float = 0.05
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """使用 SHA256(seed+image) 的稳定顺序进行 image-level train/val 划分。"""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    images = sorted({str(pair["image"]) for pair in pairs})
    ranked = sorted(
        images,
        key=lambda image: hashlib.sha256(f"{seed}\x1f{image}".encode("utf-8")).hexdigest(),
    )
    if len(ranked) <= 1:
        val_images: set[str] = set()
    else:
        val_count = min(len(ranked) - 1, max(1, int(round(len(ranked) * val_fraction))))
        val_images = set(ranked[:val_count])
    train = [dict(pair) for pair in pairs if str(pair["image"]) not in val_images]
    validation = [dict(pair) for pair in pairs if str(pair["image"]) in val_images]
    train.sort(key=lambda pair: str(pair["id"]))
    validation.sort(key=lambda pair: str(pair["id"]))
    train_images = {str(pair["image"]) for pair in train}
    val_image_set = {str(pair["image"]) for pair in validation}
    proof = {
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "train_images": len(train_images),
        "val_images": len(val_image_set),
        "train_val_image_overlap": len(train_images.intersection(val_image_set)),
        "train_image_ids": sorted(train_images),
        "val_image_ids": sorted(val_image_set),
    }
    return train, validation, proof


def _empty_audit() -> dict[str, Any]:
    return {
        "detection_total": 0,
        "detection_invalid": 0,
        "counting_total": 0,
        "counting_cardinality_eligible": 0,
        "counting_non_cardinality_excluded": 0,
        "counting_class_resolved": 0,
        "counting_class_unresolved": 0,
        "counting_class_ambiguous": 0,
        "counting_resolution_status_distribution": {},
        "non_vrs_rows_excluded": 0,
        "invalid_image_rows": 0,
        "unsupported_task_rows_excluded": 0,
        "invalid_id_rows": 0,
    }


def _distribution(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "unknown")) for row in rows).items()))


def _count_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(count_bin(int(row["count"])) for row in rows if row.get("count") is not None).items()))


def _count_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(int(row["count"]) for row in rows if row.get("count") is not None)
    if not values:
        return {"count": 0, "max": None, "p50": None, "p90": None, "p95": None, "p99": None}
    def percentile(percent: float) -> float:
        index = min(len(values) - 1, int(math.ceil(percent * len(values))) - 1)
        return float(values[index])
    return {
        "count": len(values),
        "max": max(values),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def build_object_adapter_dataset_from_rows(
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    train_source: str = "<in-memory>",
    protected_eval_source: str = "<in-memory>",
    train_source_sha256: str | None = None,
    protected_eval_source_sha256: str | None = None,
    seed: int = 42,
    val_fraction: float = 0.05,
    dedup_iou: float = 0.95,
    enforce_blockers: bool = True,
) -> dict[str, Any]:
    """从已读取 rows 构建 v0 五个冻结资产，并返回 manifest。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    protected_images = {
        canonical_image_identity(row)
        for row in eval_rows
        if dataset_name(row) == "VRSBench" and canonical_image_identity(row)
    }
    audit = _empty_audit()
    audit["protected_eval_image_count"] = len(protected_images)
    records, before_images, removed_images, unresolved_examples = _parse_training_rows(
        train_rows, protected_images, None, audit
    )
    class_names = {str(item["class_name"]) for item in records if item["kind"] == "detection"}
    class_vocab = build_class_vocab(class_names)
    # Counting class resolution depends on the vocabulary learned from detection labels;
    # reparse counting rows only, keeping the same image-level exclusion boundary.
    audit["counting_total"] = 0
    audit["counting_cardinality_eligible"] = 0
    audit["counting_non_cardinality_excluded"] = 0
    audit["counting_class_resolved"] = 0
    audit["counting_class_unresolved"] = 0
    audit["counting_class_ambiguous"] = 0
    audit["counting_resolution_status_distribution"] = {}
    # The first pass cannot resolve counting labels before the detection
    # vocabulary exists.  Do not leak those provisional examples into the
    # final audit after the authoritative second pass.
    unresolved_examples = {"non_cardinality": [], "unresolved": [], "ambiguous": []}
    records = [item for item in records if item["kind"] == "detection"]
    for row in train_rows:
        if dataset_name(row) != "VRSBench" or canonical_image_identity(row) in protected_images:
            continue
        if str(row.get("task_type", "")).strip().lower() != "counting":
            continue
        audit["counting_total"] += 1
        prompt = extract_prompt(row)
        if _cardinality_prompt_target(prompt) is None:
            resolution = ClassResolution(None, "unsupported_form", "cardinality_prompt", ())
        else:
            audit["counting_cardinality_eligible"] += 1
            resolution = resolve_counting_class(row, class_vocab)
        status_distribution = audit["counting_resolution_status_distribution"]
        status_distribution[resolution.status] = int(status_distribution.get(resolution.status, 0)) + 1
        if resolution.status == "resolved":
            audit["counting_class_resolved"] += 1
            parsed = parse_count(extract_answer(row))
            if parsed.value is not None:
                records.append(
                    {
                        "kind": "counting",
                        "sample_id": str(row.get("id", "")),
                        "image": canonical_image_identity(row),
                        "class_name": resolution.class_name,
                        "count": int(parsed.value),
                    }
                )
        elif resolution.status == "unsupported_form":
            audit["counting_non_cardinality_excluded"] += 1
            if len(unresolved_examples["non_cardinality"]) < 30:
                unresolved_examples["non_cardinality"].append(str(row.get("id", "")) + ": " + prompt)
        elif resolution.status == "ambiguous":
            audit["counting_class_ambiguous"] += 1
            if len(unresolved_examples["ambiguous"]) < 30:
                unresolved_examples["ambiguous"].append(str(row.get("id", "")) + ": " + prompt)
        elif resolution.status == "unresolved":
            audit["counting_class_unresolved"] += 1
            if len(unresolved_examples["unresolved"]) < 30:
                unresolved_examples["unresolved"].append(str(row.get("id", "")) + ": " + prompt)
    audit["counting_class_resolution_rate"] = (
        audit["counting_class_resolved"] / audit["counting_cardinality_eligible"]
        if audit["counting_cardinality_eligible"]
        else 0.0
    )
    if audit["counting_cardinality_eligible"] != (
        audit["counting_class_resolved"]
        + audit["counting_class_unresolved"]
        + audit["counting_class_ambiguous"]
    ):
        raise RuntimeError("Counting cardinality audit accounting is inconsistent")
    pairs = construct_object_pairs(records, class_vocab, dedup_iou=dedup_iou, audit=audit)
    train_pairs, val_pairs, split_proof = stable_image_split(
        pairs, seed=seed, val_fraction=val_fraction
    )
    audit["train_images_before_exclusion"] = len(before_images)
    audit["train_images_removed_for_eval_overlap"] = len(removed_images)
    audit["train_images_after_exclusion"] = len(before_images - removed_images)
    audit["final_image_overlap_count"] = len((before_images - removed_images).intersection(protected_images))
    audit["train_pair_count"] = len(train_pairs)
    audit["val_pair_count"] = len(val_pairs)
    audit["pair_supervision_distribution"] = _distribution(pairs, "supervision_type")
    audit["train_supervision_distribution"] = _distribution(train_pairs, "supervision_type")
    audit["val_supervision_distribution"] = _distribution(val_pairs, "supervision_type")
    audit["count_bin_distribution"] = _count_distribution(pairs)
    audit["count_stats"] = _count_stats(pairs)
    audit["class_distribution"] = _distribution(pairs, "class_name")
    audit["counting_resolution_status_distribution"] = dict(
        sorted(audit["counting_resolution_status_distribution"].items())
    )
    audit["non_cardinality_examples"] = unresolved_examples["non_cardinality"]
    audit["unresolved_prompt_examples"] = unresolved_examples["unresolved"]
    audit["ambiguous_prompt_examples"] = unresolved_examples["ambiguous"]
    audit["class_vocab_size"] = len(class_vocab["classes"])
    blockers: list[str] = []
    if audit["final_image_overlap_count"] != 0:
        blockers.append("final_image_overlap_count must be 0")
    if split_proof["train_val_image_overlap"] != 0:
        blockers.append("train_val_image_overlap must be 0")
    if float(audit["detection_parse_rate"]) < 0.99:
        blockers.append(f"detection_parse_rate={audit['detection_parse_rate']:.4f} < 0.99")
    if float(audit["counting_class_resolution_rate"]) < 0.90:
        blockers.append(
            f"counting_class_resolution_rate={audit['counting_class_resolution_rate']:.4f} < 0.90"
        )
    max_count = audit["count_stats"]["max"]
    if max_count is not None and int(max_count) > 64:
        blockers.append(f"max_count={max_count} > num_queries=64")
    full_positive = sum(
        1 for row in pairs if row["supervision_type"] == "full_set" and int(row.get("count") or 0) > 0
    )
    audit["full_set_positive_pair_count"] = full_positive
    if full_positive < 100:
        blockers.append(f"full_set_positive_pair_count={full_positive} < 100")
    if not train_pairs:
        blockers.append("train data is empty")
    if not val_pairs:
        blockers.append("internal validation data is empty")
    audit["hard_blockers"] = blockers
    audit["status"] = "blocked" if blockers else "passed"

    train_path = destination / "train.jsonl"
    val_path = destination / "val.jsonl"
    vocab_path = destination / "class_vocab.json"
    audit_path = destination / "audit.json"
    manifest_path = destination / "manifest.json"
    write_jsonl(train_path, train_pairs)
    write_jsonl(val_path, val_pairs)
    vocab_path.write_text(json.dumps(class_vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_payload = {"schema_version": SCHEMA_VERSION, "builder_version": BUILDER_VERSION, **audit}
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "seed": int(seed),
        "source_file": str(train_source),
        "source_sha256": train_source_sha256,
        "protected_eval_source": str(protected_eval_source),
        "protected_eval_source_sha256": protected_eval_source_sha256,
        "protected_eval_image_count": len(protected_images),
        "train_images_before_exclusion": len(before_images),
        "train_images_removed_for_eval_overlap": len(removed_images),
        "train_images_after_exclusion": len(before_images - removed_images),
        "final_image_overlap_count": audit["final_image_overlap_count"],
        "train_val_image_overlap": split_proof["train_val_image_overlap"],
        "split": split_proof,
        "selected_blocks": [5, 11, 17, 23],
        "num_queries": 64,
        "class_vocab_size": len(class_vocab["classes"]),
        "output_files": {},
        "audit_status": audit["status"],
        "hard_blockers": blockers,
        "statistics": {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "pair_supervision_distribution": audit["pair_supervision_distribution"],
            "count_bin_distribution": audit["count_bin_distribution"],
            "max_count": max_count,
            "class_distribution": audit["class_distribution"],
        },
    }
    for path in (train_path, val_path, vocab_path, audit_path):
        manifest["output_files"][path.name] = {"path": path.name, "sha256": file_sha256(path)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    if blockers and enforce_blockers:
        raise DataAuditBlocked(blockers, manifest_path)
    return manifest


def build_object_adapter_dataset(
    train_source: str | Path,
    protected_eval_source: str | Path,
    *,
    output_dir: str | Path,
    seed: int = 42,
    val_fraction: float = 0.05,
    dedup_iou: float = 0.95,
    enforce_blockers: bool = True,
) -> dict[str, Any]:
    """读取两份正式 JSONL 并执行 v0 数据审计/构建。"""

    train_path = Path(train_source)
    eval_path = Path(protected_eval_source)
    if not train_path.is_file():
        raise FileNotFoundError(f"Object Adapter training source does not exist: {train_path}")
    if not eval_path.is_file():
        raise FileNotFoundError(f"Evaluation protection source does not exist: {eval_path}")
    return build_object_adapter_dataset_from_rows(
        list(read_jsonl(train_path)),
        list(read_jsonl(eval_path)),
        output_dir=output_dir,
        train_source=str(train_path),
        protected_eval_source=str(eval_path),
        train_source_sha256=file_sha256(train_path),
        protected_eval_source_sha256=file_sha256(eval_path),
        seed=seed,
        val_fraction=val_fraction,
        dedup_iou=dedup_iou,
        enforce_blockers=enforce_blockers,
    )


def validate_data_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """训练启动前校验 manifest 声明的 train/val/vocab/audit SHA。"""

    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("builder_version") != BUILDER_VERSION:
        raise ValueError(f"Unsupported Object Adapter manifest builder: {path}")
    if payload.get("audit_status") != "passed":
        raise DataAuditBlocked(payload.get("hard_blockers", ["audit_status is not passed"]), path)
    if int(payload.get("final_image_overlap_count", -1)) != 0:
        raise ValueError("Object Adapter manifest has non-zero eval image overlap")
    if int(payload.get("train_val_image_overlap", -1)) != 0:
        raise ValueError("Object Adapter manifest has train/val image overlap")
    for name, record in dict(payload.get("output_files", {})).items():
        file_path = Path(str(record.get("path", "")))
        if not file_path.is_file():
            candidate = path.parent / file_path
            file_path = candidate if candidate.is_file() else file_path
        if not file_path.is_file():
            raise FileNotFoundError(f"Object Adapter asset missing: {name}")
        if file_sha256(file_path) != str(record.get("sha256", "")):
            raise ValueError(f"Object Adapter asset SHA mismatch: {name}")
    return payload
