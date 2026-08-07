"""从当前 DatasetManifest 分片构建固定、均衡的可靠性评测样本。"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from sat_rs_vlm.data.manifest import (
    DatasetManifest,
    load_dataset_manifest,
    load_manifest_split,
)
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl

DEFAULT_RELIABILITY_TASKS = (
    "captioning",
    "vqa",
    "counting",
    "detection",
    "scene_classification",
)


def _message_text(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") != role:
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(
                str(item.get("text", "")).strip()
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
    return ""


def _images(row: dict[str, Any]) -> list[str]:
    raw = row.get("images")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(value) for value in raw]
    images: list[str] = []
    messages = row.get("messages", [])
    if isinstance(messages, list):
        for message in messages:
            content = message.get("content", []) if isinstance(message, dict) else []
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image" and item.get("image"):
                    images.append(str(item["image"]))
    return images


def _safe_image_path(dataset_root: Path, value: str) -> str:
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or (pure.parts and ":" in pure.parts[0]):
        raise ValueError(f"Reliability sample image path must be relative: {value}")
    candidate = (dataset_root / Path(*pure.parts)).resolve()
    if candidate != dataset_root and dataset_root not in candidate.parents:
        raise ValueError(f"Reliability sample image escapes dataset root: {value}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Reliability sample image does not exist: {candidate}")
    return pure.as_posix()


def _resolve_source_image(source_root: Path, value: str) -> Path:
    normalized = value.replace("\\", "/")
    parts = tuple(part for part in normalized.split("/") if part)
    candidates: list[Path] = []
    direct = Path(value).expanduser()
    if direct.is_absolute():
        candidates.append(direct)
    candidates.append(source_root.joinpath(*parts))
    for index, part in enumerate(parts):
        if part.lower() == "images":
            candidates.append(source_root.joinpath(*parts[index:]))
            break

    attempted: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        attempted.append(str(resolved))
        if resolved != source_root and source_root not in resolved.parents:
            continue
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Reliability sample image does not exist: {value}; attempted: {', '.join(attempted)}"
    )


def _portable_source_image_path(
    common_image_root: Path,
    source_root: Path,
    value: str,
) -> str:
    image = _resolve_source_image(source_root, value)
    try:
        return image.relative_to(common_image_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Reliability sample image is outside common image root: {image}"
        ) from exc


def _normalized_messages(
    row: dict[str, Any],
    *,
    question: str,
    reference: str,
    raw_images: list[str],
    portable_images: list[str],
) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in portable_images),
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": reference},
        ]

    image_mapping = dict(zip(raw_images, portable_images, strict=True))
    fallback_images = iter(portable_images)
    normalized: list[dict[str, Any]] = []
    for message in messages:
        message_copy = dict(message)
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            normalized.append(message_copy)
            continue
        normalized_content: list[Any] = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                normalized_content.append(item)
                continue
            item_copy = dict(item)
            raw_image = str(item_copy.get("image", ""))
            portable_image = image_mapping.get(raw_image)
            if portable_image is None:
                portable_image = next(fallback_images, raw_image)
            item_copy["image"] = portable_image
            normalized_content.append(item_copy)
        message_copy["content"] = normalized_content
        normalized.append(message_copy)
    return normalized


def _source_rows(
    source: Mapping[str, Any],
) -> tuple[Path, str, str, list[dict[str, Any]], dict[str, str] | None]:
    name = str(source.get("name", "")).strip()
    if not name:
        raise ValueError("Reliability source is missing name")
    source_root_value = source.get("dataset_root")
    if not source_root_value:
        raise ValueError(f"Reliability source {name} is missing dataset_root")
    source_root = Path(str(source_root_value)).expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"Reliability source root does not exist: {source_root}")
    source_split = str(source.get("source_split", "validation"))
    manifest_value = source.get("dataset_manifest")
    if manifest_value:
        manifest = load_dataset_manifest(Path(str(manifest_value)).expanduser().resolve())
        owners = _detect_split_leakage(source_root, manifest)
        return (
            source_root,
            name,
            source_split,
            load_manifest_split(source_root, manifest, source_split),
            owners,
        )

    eval_value = source.get("eval_file")
    if not eval_value:
        raise ValueError(
            f"Reliability source {name} requires dataset_manifest or eval_file"
        )
    eval_file = Path(str(eval_value)).expanduser().resolve()
    if not eval_file.is_file():
        raise FileNotFoundError(f"Reliability source JSONL does not exist: {eval_file}")
    rows = list(read_jsonl(eval_file))
    ids = [str(row.get("id", "")).strip() for row in rows]
    if any(not sample_id for sample_id in ids):
        raise ValueError(f"Reliability source {name} contains a sample without id")
    duplicate_ids = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(
            f"Reliability source {name} contains duplicate ids: {', '.join(duplicate_ids[:5])}"
        )
    return source_root, name, source_split, rows, None


def _task_sample_counts(source: Mapping[str, Any], default_count: int) -> dict[str, int]:
    configured = source.get("task_samples")
    if isinstance(configured, Mapping):
        counts = {str(task): int(count) for task, count in configured.items()}
    else:
        tasks = tuple(str(task) for task in source.get("tasks", DEFAULT_RELIABILITY_TASKS))
        count = int(source.get("samples_per_task", default_count))
        counts = {task: count for task in tasks}
    if not counts or any(count <= 0 for count in counts.values()):
        raise ValueError("Reliability source task sample counts must be positive")
    return counts


def build_multisource_reliability_eval_manifest(
    common_image_root: str | Path,
    sources: list[dict[str, Any]],
    *,
    output_path: str | Path,
    samples_per_task: int,
    seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one balanced reliability manifest from multiple dataset sources."""

    if samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    common_root = Path(common_image_root).expanduser().resolve()
    if not common_root.is_dir():
        raise NotADirectoryError(f"Common image root does not exist: {common_root}")
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Reliability eval manifest already exists: {output}")
    if not sources:
        raise ValueError("At least one reliability source is required")

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    source_statistics: dict[str, Any] = {}
    for source_index, source in enumerate(sources):
        source_root, name, source_split, rows, owners = _source_rows(source)
        task_counts = _task_sample_counts(source, samples_per_task)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("task_type", "unknown"))].append(row)

        source_selected = 0
        for task_index, (task, requested) in enumerate(task_counts.items()):
            candidates = list(grouped[task])
            random.Random(seed + source_index * 10_000 + task_index).shuffle(candidates)
            if bool(source.get("group_by_images", False)):
                unique_candidates: list[dict[str, Any]] = []
                seen_images: set[tuple[str, ...]] = set()
                for row in candidates:
                    image_key = tuple(_images(row))
                    if image_key in seen_images:
                        continue
                    seen_images.add(image_key)
                    unique_candidates.append(row)
                candidates = unique_candidates
            if len(candidates) < requested:
                raise ValueError(
                    f"Not enough {task} samples in {name}: required={requested}, "
                    f"available={len(candidates)}"
                )

            for row in candidates[:requested]:
                sample_id = str(row.get("id", "")).strip()
                if owners is not None and owners.get(sample_id) != source_split:
                    raise ValueError(f"Sample split ownership mismatch: {sample_id}")
                if sample_id in selected_ids:
                    raise ValueError(f"Reliability sources contain duplicate id: {sample_id}")
                messages = row.get("messages", [])
                question = str(row.get("instruction", "")).strip()
                reference = str(row.get("answer", "")).strip()
                if isinstance(messages, list):
                    question = question or _message_text(messages, "user")
                    reference = reference or _message_text(messages, "assistant")
                raw_images = _images(row)
                portable_images = [
                    _portable_source_image_path(common_root, source_root, image)
                    for image in raw_images
                ]
                if not portable_images:
                    raise ValueError(f"Reliability sample has no image: {sample_id}")
                if task == "change_detection" and len(portable_images) != 2:
                    raise ValueError(
                        f"Change-detection reliability sample must have two images: {sample_id}"
                    )
                metadata = dict(row.get("metadata", {}))
                metadata.setdefault("dataset", name)
                metadata["reliability_source"] = name
                selected.append(
                    {
                        "id": sample_id,
                        "task_type": task,
                        "question": question,
                        "reference": reference,
                        "images": portable_images,
                        "source_split": source_split,
                        "metadata": metadata,
                        "messages": _normalized_messages(
                            row,
                            question=question,
                            reference=reference,
                            raw_images=raw_images,
                            portable_images=portable_images,
                        ),
                    }
                )
                selected_ids.add(sample_id)
                source_selected += 1
        source_statistics[name] = {
            "source_split": source_split,
            "num_samples": source_selected,
            "task_distribution": dict(task_counts),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    statistics = {
        "schema_version": "1.0",
        "dataset_name": "multisource",
        "dataset_names": list(source_statistics),
        "seed": seed,
        "samples_per_task": samples_per_task,
        "num_samples": len(selected),
        "task_distribution": dict(Counter(row["task_type"] for row in selected)),
        "source_distribution": dict(
            Counter(str(row["metadata"]["reliability_source"]) for row in selected)
        ),
        "sources": source_statistics,
        "output": output.name,
    }
    output.with_suffix(".stats.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return statistics


def _detect_split_leakage(
    dataset_root: Path,
    manifest: DatasetManifest,
) -> dict[str, str]:
    owners: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for row in load_manifest_split(dataset_root, manifest, split):
            sample_id = str(row.get("id", "")).strip()
            if not sample_id:
                continue
            previous = owners.get(sample_id)
            if previous is not None and previous != split:
                raise ValueError(
                    f"Dataset split leakage detected for sample id {sample_id}: {previous}, {split}"
                )
            owners[sample_id] = split
    return owners


def build_reliability_eval_manifest(
    dataset_root: str | Path,
    manifest_path: str | Path,
    *,
    source_split: str,
    output_path: str | Path,
    samples_per_task: int,
    seed: int,
    tasks: tuple[str, ...] = DEFAULT_RELIABILITY_TASKS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """均衡抽样并写出 JSONL 与统计文件。

    每条输出包含 ID、任务、问题、参考答案、相对图片路径、源 split 和元数据。
    在抽样前检查 train/validation/test ID 泄漏，并验证每张图片存在。
    """

    if samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    root = Path(dataset_root).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Reliability eval manifest already exists: {output}")
    manifest = load_dataset_manifest(manifest_file)
    owners = _detect_split_leakage(root, manifest)
    rows = load_manifest_split(root, manifest, source_split)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task_type", "unknown"))
        if task in tasks:
            grouped[task].append(row)
    missing = {task: len(grouped[task]) for task in tasks if len(grouped[task]) < samples_per_task}
    if missing:
        raise ValueError(
            f"Not enough samples for balanced reliability manifest: required={samples_per_task}, "
            f"available={missing}"
        )

    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for task in tasks:
        candidates = list(grouped[task])
        rng.shuffle(candidates)
        for row in candidates[:samples_per_task]:
            sample_id = str(row.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"Reliability sample in {source_split} is missing id")
            if owners.get(sample_id) != source_split:
                raise ValueError(f"Sample split ownership mismatch: {sample_id}")
            messages = row.get("messages", [])
            question = str(row.get("instruction", "")).strip()
            reference = str(row.get("answer", "")).strip()
            if isinstance(messages, list):
                question = question or _message_text(messages, "user")
                reference = reference or _message_text(messages, "assistant")
            images = [_safe_image_path(root, image) for image in _images(row)]
            if not images:
                raise ValueError(f"Reliability sample has no image: {sample_id}")
            selected.append(
                {
                    "id": sample_id,
                    "task_type": task,
                    "question": question,
                    "reference": reference,
                    "images": images,
                    "source_split": source_split,
                    "metadata": dict(row.get("metadata", {})),
                    "messages": messages,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    statistics = {
        "schema_version": "1.0",
        "dataset_name": manifest.dataset_name,
        "dataset_version": manifest.dataset_version,
        "source_split": source_split,
        "seed": seed,
        "samples_per_task": samples_per_task,
        "num_samples": len(selected),
        "task_distribution": dict(Counter(row["task_type"] for row in selected)),
        "output": output.name,
    }
    statistics_path = output.with_suffix(".stats.json")
    statistics_path.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return statistics
