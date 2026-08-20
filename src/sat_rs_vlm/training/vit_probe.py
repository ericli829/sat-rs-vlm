"""Qwen3-VL-4B visual probe 的确定性数据组织与 checkpoint 辅助工具。

本模块不加载模型，也不执行训练。它只完成两件事：

1. 从合法训练 population 中按 source/task 配额、固定 seed、无放回抽样，
   并排除 Unified Evaluation Tiers v2 的所有保护 ID；
2. 给 Trainer 产生的中间 checkpoint 补齐 processor 和 strategy manifest，
   使中间结果可以直接进入现有 Evaluation v1.5 加载器。

采样结果写成冻结 JSONL，后续训练不依赖运行时随机抽样。
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """计算文件 SHA256，用于数据和实验 manifest。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取并检查 JSONL 行对象和 sample ID。"""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("id"):
                raise ValueError(f"Invalid sample at {path}:{line_number}; id is required")
            rows.append(value)
    return rows


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metadata", {})
    return value if isinstance(value, Mapping) else {}


def canonical_source(row: Mapping[str, Any]) -> str:
    """将 metadata 中的 dataset/training_source 归一为两个 probe source。"""

    value = str(_metadata(row).get("dataset", _metadata(row).get("training_source", "")))
    lowered = value.lower().replace("_", "-")
    if "levir" in lowered:
        return "LEVIR-CC"
    if "vrs" in lowered:
        return "VRSBench"
    return value or "unknown"


def _protected_ids(manifest_path: str | Path) -> set[str]:
    """从 tiers v2 manifest 收集 E1/E2/E3 的全部 sample ID。"""

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    protected: set[str] = set()
    tiers = payload.get("tiers", {})
    if not isinstance(tiers, Mapping):
        raise ValueError(f"Tier manifest has no tiers mapping: {manifest_path}")
    for record in tiers.values():
        if not isinstance(record, Mapping):
            continue
        values = record.get("sample_ids", [])
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            protected.update(str(value) for value in values)
    if not protected:
        raise ValueError(f"Protected tier manifest contains no sample IDs: {manifest_path}")
    return protected


def _stable_shuffled(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    values = sorted(rows, key=lambda row: str(row["id"]))
    random.Random(seed).shuffle(values)
    return values


def _take_quota(
    rows: list[dict[str, Any]],
    quota: int,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = _stable_shuffled(rows, seed)
    return shuffled[: max(0, quota)], shuffled[max(0, quota) :]


def _sample_source(
    rows: list[dict[str, Any]],
    quota: int,
    task_targets: Mapping[str, int],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """先按 task 配额取样，短缺额度再在同 source 剩余样本中确定性补齐。"""

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_type", "unknown"))].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    actual: Counter[str] = Counter()
    for offset, task in enumerate(sorted(task_targets)):
        task_rows = by_task.get(task, [])
        take, _ = _take_quota(task_rows, int(task_targets[task]), seed=seed + offset)
        selected.extend(take)
        selected_ids.update(str(row["id"]) for row in take)
        actual[task] += len(take)

    remaining_quota = max(0, quota - len(selected))
    if remaining_quota:
        remaining = [row for row in rows if str(row["id"]) not in selected_ids]
        extra, _ = _take_quota(remaining, remaining_quota, seed=seed + 10000)
        selected.extend(extra)
        actual.update(str(row.get("task_type", "unknown")) for row in extra)

    if len(selected) > quota:
        selected = _stable_shuffled(selected, seed + 20000)[:quota]
        actual = Counter(str(row.get("task_type", "unknown")) for row in selected)
    return selected, dict(sorted(actual.items()))


def build_probe_dataset(
    source_files: Sequence[str | Path],
    *,
    output_dir: str | Path,
    protected_evaluation_manifest: str | Path,
    target_samples: int = 6000,
    source_targets: Mapping[str, int] | None = None,
    task_targets: Mapping[str, int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """构建 4B last-2 probe 数据集并返回 manifest。

    参数：
        source_files: 合法训练 JSONL 路径列表。
        output_dir: 输出目录，生成 ``train.jsonl`` 和 ``manifest.json``。
        protected_evaluation_manifest: tiers v2 manifest 路径。
        target_samples: 目标样本数，实际不足时使用全部可用样本。
        source_targets/task_targets: source/task 的首选配额。
        seed: 抽样随机种子。
    返回：
        可 JSON 序列化的采样 manifest。
    """

    if not source_files:
        raise ValueError("At least one source training JSONL is required")
    protected = _protected_ids(protected_evaluation_manifest)
    source_hashes = {str(Path(path)): sha256_file(path) for path in source_files}
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for source_file in source_files:
        for row in read_jsonl(source_file):
            sample_id = str(row["id"])
            if sample_id in by_id:
                duplicate_count += 1
                continue
            by_id[sample_id] = row

    overlap = sorted(set(by_id).intersection(protected))
    for sample_id in overlap:
        del by_id[sample_id]

    population = list(by_id.values())
    requested_sources = dict(source_targets or {"VRSBench": 4500, "LEVIR-CC": 1500})
    requested_tasks = dict(
        task_targets
        or {
            "captioning": 900,
            "detection": 900,
            "counting": 900,
            "scene_classification": 900,
            "vqa": 900,
            "change_detection": 1500,
        }
    )
    available_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in population:
        available_by_source[canonical_source(row)].append(row)

    selected: list[dict[str, Any]] = []
    source_distribution: dict[str, int] = {}
    task_distribution: Counter[str] = Counter()
    for offset, (source, quota) in enumerate(sorted(requested_sources.items())):
        rows = available_by_source.get(source, [])
        source_task_targets = {
            task: value
            for task, value in requested_tasks.items()
            if (source == "VRSBench" and task != "change_detection")
            or (source == "LEVIR-CC" and task == "change_detection")
        }
        take, actual_tasks = _sample_source(
            rows,
            min(int(quota), len(rows)),
            source_task_targets,
            seed=seed + offset * 1000,
        )
        selected.extend(take)
        source_distribution[source] = len(take)
        task_distribution.update(actual_tasks)

    requested_total = min(int(target_samples), len(population))
    if len(selected) < requested_total:
        selected_ids = {str(row["id"]) for row in selected}
        remaining = [row for row in population if str(row["id"]) not in selected_ids]
        extra, _ = _take_quota(
            remaining,
            requested_total - len(selected),
            seed=seed + 90000,
        )
        selected.extend(extra)
        for row in extra:
            source_distribution[canonical_source(row)] = source_distribution.get(
                canonical_source(row), 0
            ) + 1
            task_distribution[str(row.get("task_type", "unknown"))] += 1

    selected = _stable_shuffled(selected[:requested_total], seed + 99000)
    selected_ids = [str(row["id"]) for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Probe sampler produced duplicate sample IDs")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "train.jsonl"
    with train_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": "1.0",
        "experiment": "qwen3vl_4b_vit_probe_last2",
        "seed": seed,
        "target_samples": target_samples,
        "total_samples": len(selected),
        "dataset_distribution": dict(sorted(source_distribution.items())),
        "task_distribution": dict(sorted(task_distribution.items())),
        "unique_count": len(selected_ids),
        "duplicate_count": duplicate_count,
        "protected_eval_ids_count": len(protected),
        "protected_source_overlap_count": len(overlap),
        "protected_source_overlap_ids_preview": overlap[:20],
        "protected_eval_overlap_count": 0,
        "protected_evaluation_manifest": str(Path(protected_evaluation_manifest)),
        "source_files": [str(Path(path)) for path in source_files],
        "source_file_sha256": source_hashes,
        "output_file": str(train_path),
        "output_sha256": sha256_file(train_path),
        "sample_ids": selected_ids,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def make_checkpoint_evaluable(
    experiment_dir: str | Path,
    checkpoint_dir: str | Path,
    *,
    checkpoint_step: int | None = None,
    require_visual_sidecar: bool = True,
) -> Path:
    """补齐中间 checkpoint 的 processor/manifest，保持 sidecar 原样不动。

    ``require_visual_sidecar`` 默认保持 ``True``，兼容旧 ViT probe 和 visual-tuned
    checkpoint 的完整性约束。LoRA-only 阶段可以显式传入 ``False``，因为这类
    checkpoint 没有需要保存的视觉 sidecar。
    """

    source = Path(experiment_dir)
    destination = Path(checkpoint_dir)
    required_manifest = source / "strategy_manifest.json"
    source_processor = source / "processor"
    if not required_manifest.is_file() or not source_processor.is_dir():
        raise FileNotFoundError(
            "Final experiment must contain strategy_manifest.json and processor before "
            f"checkpoint promotion: {source}"
        )
    missing_adapter_files = [
        name
        for name in ("adapter_model.safetensors", "adapter_config.json")
        if not (destination / name).is_file()
    ]
    if missing_adapter_files:
        raise FileNotFoundError(
            f"Checkpoint adapter files are missing from {destination}: {missing_adapter_files}"
        )
    if require_visual_sidecar:
        sidecar = destination / "visual_trainable_weights.safetensors"
        if not sidecar.is_file():
            raise FileNotFoundError(f"Checkpoint visual sidecar is missing: {sidecar}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(required_manifest, destination / "strategy_manifest.json")
    shutil.copytree(source_processor, destination / "processor", dirs_exist_ok=True)
    manifest_path = destination / "strategy_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if checkpoint_step is not None:
        payload["probe_checkpoint_step"] = checkpoint_step
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination
