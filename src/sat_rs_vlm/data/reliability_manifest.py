"""从当前 DatasetManifest 分片构建固定、均衡的可靠性评测样本。"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from sat_rs_vlm.data.manifest import (
    DatasetManifest,
    load_dataset_manifest,
    load_manifest_split,
)
from sat_rs_vlm.utils.jsonl import write_jsonl

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
