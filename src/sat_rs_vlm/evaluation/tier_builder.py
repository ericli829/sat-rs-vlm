"""Unified Evaluation Tiers v2 的确定性分层构建实现。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.configuration.environment import expand_environment
from sat_rs_vlm.evaluation.tiers import file_sha256
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl


def row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return dict(value) if isinstance(value, Mapping) else {}


def row_images(row: Mapping[str, Any]) -> list[str]:
    """读取 Qwen messages 中的全部图像路径。"""

    images: list[str] = []
    for message in list(row.get("messages", [])):
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "image":
                images.append(str(item.get("image", "")))
    return images


def _bbox_area(row: Mapping[str, Any]) -> float | None:
    metadata = row_metadata(row)
    value = metadata.get("bbox_clipped", metadata.get("bbox_raw"))
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _count_value(row: Mapping[str, Any]) -> int | None:
    import re

    reference: Any = row.get("reference")
    if reference is None:
        for message in reversed(list(row.get("messages", []))):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                reference = message.get("content")
                break
    match = re.search(r"(?<![\d.])(\d+)(?![\d.])", str(reference or ""))
    return int(match.group(1)) if match else None


def task_subtype(
    row: Mapping[str, Any],
    *,
    small_max: float,
    medium_max: float,
) -> str:
    """返回真实 metadata/GT 驱动的任务子类型，不从问题文本猜测。"""

    metadata = row_metadata(row)
    task = str(row.get("task_type", "unknown"))
    if task == "detection":
        area = _bbox_area(row)
        if area is None:
            return "unknown"
        if area <= small_max:
            return "small"
        if area <= medium_max:
            return "medium"
        return "large"
    if task == "counting":
        count = _count_value(row)
        if count is None:
            return "unknown"
        if count >= 10:
            return "10+"
        if count >= 5:
            return "5-9"
        return str(count)
    if task == "vqa":
        return str(metadata.get("qa_type", "unknown"))
    if task == "change_detection":
        return str(metadata.get("changeflag", "unknown"))
    return str(metadata.get("source_task", "default"))


def row_cell(
    row: Mapping[str, Any],
    *,
    small_max: float,
    medium_max: float,
) -> tuple[str, str, str]:
    metadata = row_metadata(row)
    return (
        str(metadata.get("dataset", "unknown")),
        str(row.get("task_type", "unknown")),
        task_subtype(row, small_max=small_max, medium_max=medium_max),
    )


def distribution(
    rows: Sequence[Mapping[str, Any]],
    *,
    small_max: float,
    medium_max: float,
) -> dict[str, Any]:
    datasets: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    subtypes: Counter[str] = Counter()
    for row in rows:
        dataset, task, subtype = row_cell(
            row,
            small_max=small_max,
            medium_max=medium_max,
        )
        datasets[dataset] += 1
        tasks[task] += 1
        subtypes[f"{dataset}/{task}/{subtype}"] += 1
    return {
        "sample_count": len(rows),
        "dataset": dict(sorted(datasets.items())),
        "task": dict(sorted(tasks.items())),
        "subtype": dict(sorted(subtypes.items())),
    }


def allocate_counts(
    capacities: Mapping[str, int],
    target: int,
    *,
    weights: Mapping[str, float],
    shortage_policy: str = "redistribute",
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """按权重分配整数配额，并显式记录容量不足后的重分配。"""

    normalized_capacities = {str(key): int(value) for key, value in capacities.items()}
    if target < 0 or target > sum(normalized_capacities.values()):
        raise ValueError(
            f"Target {target} is outside available population {sum(normalized_capacities.values())}"
        )
    active = [key for key, capacity in normalized_capacities.items() if capacity > 0]
    normalized_weights = {key: float(weights.get(key, 0.0)) for key in active}
    if any(value < 0.0 for value in normalized_weights.values()):
        raise ValueError("Allocation weights must be non-negative")
    weight_sum = sum(normalized_weights.values())
    if not active or weight_sum <= 0.0:
        raise ValueError("At least one available allocation cell must have positive weight")
    normalized_weights = {key: value / weight_sum for key, value in normalized_weights.items()}

    raw = {key: target * normalized_weights[key] for key in active}
    requested = {key: math.floor(raw[key]) for key in active}
    for key in sorted(active, key=lambda value: (-(raw[value] % 1.0), value))[
        : target - sum(requested.values())
    ]:
        requested[key] += 1

    if shortage_policy == "fail":
        shortage = {
            key: requested[key] - normalized_capacities[key]
            for key in active
            if requested[key] > normalized_capacities[key]
        }
        if shortage:
            raise ValueError(f"Allocation capacity shortage: {shortage}")
    elif shortage_policy != "redistribute":
        raise ValueError("shortage_policy must be 'redistribute' or 'fail'")

    actual = {
        key: min(requested[key], normalized_capacities[key])
        for key in active
    }
    warnings: list[str] = []
    for key in active:
        if requested[key] > normalized_capacities[key]:
            warnings.append(
                f"{key} requested {requested[key]} but capacity is "
                f"{normalized_capacities[key]}; shortfall redistributed"
            )
    while sum(actual.values()) < target:
        available = [key for key in active if actual[key] < normalized_capacities[key]]
        if not available:
            raise ValueError("Allocation exhausted all capacities before reaching target")
        next_total = sum(actual.values()) + 1
        key = max(
            available,
            key=lambda value: (
                next_total * normalized_weights[value] - actual[value],
                normalized_capacities[value] - actual[value],
                value,
            ),
        )
        actual[key] += 1
    return dict(sorted(requested.items())), dict(sorted(actual.items())), warnings


def _stable_rows(rows: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[str, str]:
        sample_id = str(row.get("id", ""))
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
        return digest, sample_id

    return sorted(rows, key=key)


def _select_subtypes(
    rows: Sequence[dict[str, Any]],
    target: int,
    *,
    seed: int,
    small_max: float,
    medium_max: float,
    shortage_policy: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            task_subtype(row, small_max=small_max, medium_max=medium_max)
        ].append(row)
    capacities = {key: len(value) for key, value in grouped.items()}
    _, allocation, _ = allocate_counts(
        capacities,
        target,
        weights={key: 1.0 for key in capacities},
        shortage_policy=shortage_policy,
    )
    selected: list[dict[str, Any]] = []
    for offset, subtype in enumerate(sorted(grouped)):
        selected.extend(_stable_rows(grouped[subtype], seed + offset)[: allocation[subtype]])
    return selected


def select_hierarchical_tier(
    rows: Sequence[dict[str, Any]],
    target: int,
    *,
    seed: int,
    dataset_weights: Mapping[str, float],
    task_balance: str,
    small_max: float,
    medium_max: float,
    shortage_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按 dataset -> task -> subtype 进行确定性层次化选择。"""

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row_metadata(row).get("dataset", "unknown"))].append(row)
    dataset_capacities = {key: len(value) for key, value in by_dataset.items()}
    requested_dataset, actual_dataset, warnings = allocate_counts(
        dataset_capacities,
        target,
        weights=dataset_weights,
        shortage_policy=shortage_policy,
    )
    selected: list[dict[str, Any]] = []
    task_report: dict[str, Any] = {}
    for dataset_index, dataset in enumerate(sorted(by_dataset)):
        dataset_rows = by_dataset[dataset]
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in dataset_rows:
            by_task[str(row.get("task_type", "unknown"))].append(row)
        task_capacities = {key: len(value) for key, value in by_task.items()}
        if task_balance == "sqrt_population":
            task_weights = {key: math.sqrt(value) for key, value in task_capacities.items()}
        elif task_balance == "natural":
            task_weights = {key: float(value) for key, value in task_capacities.items()}
        else:
            raise ValueError("task_balance must be 'sqrt_population' or 'natural'")
        requested_task, actual_task, task_warnings = allocate_counts(
            task_capacities,
            actual_dataset[dataset],
            weights=task_weights,
            shortage_policy=shortage_policy,
        )
        warnings.extend(f"{dataset}/{message}" for message in task_warnings)
        task_report[dataset] = {
            "population": dict(sorted(task_capacities.items())),
            "weights": dict(sorted(task_weights.items())),
            "requested": requested_task,
            "actual": actual_task,
        }
        for task_index, task in enumerate(sorted(by_task)):
            selected.extend(
                _select_subtypes(
                    by_task[task],
                    actual_task[task],
                    seed=seed + dataset_index * 10_000 + task_index * 100,
                    small_max=small_max,
                    medium_max=medium_max,
                    shortage_policy=shortage_policy,
                )
            )
    selected = _stable_rows(selected, seed + 900_000)
    if len(selected) != target or len({str(row["id"]) for row in selected}) != target:
        raise ValueError("Hierarchical tier selection did not produce unique target count")
    return selected, {
        "dataset": {
            "population": dict(sorted(dataset_capacities.items())),
            "weights": dict(sorted((str(k), float(v)) for k, v in dataset_weights.items())),
            "requested": requested_dataset,
            "actual": actual_dataset,
        },
        "tasks": task_report,
        "warnings": warnings,
    }


def validate_portable_images(rows: Sequence[Mapping[str, Any]], common_root: Path) -> None:
    """确保每条样本的 portable image path 可在 common root 下解析。"""

    root = common_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Common image root does not exist: {root}")
    for row in rows:
        sample_id = str(row.get("id", ""))
        dataset = str(row_metadata(row).get("dataset", "unknown"))
        images = row_images(row)
        if not images:
            raise ValueError(f"Sample {sample_id} dataset={dataset} contains no image path")
        for image in images:
            relative = Path(image)
            candidate = Path(os.path.abspath(root / relative))
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Image path escapes common root: sample={sample_id}, "
                    f"dataset={dataset}, image={image}"
                ) from exc
            if relative.is_absolute() or not candidate.is_file():
                raise FileNotFoundError(
                    f"Image path cannot be resolved: sample={sample_id}, "
                    f"dataset={dataset}, image={image}, resolved={candidate}"
                )


def _sampling_fraction(
    tier_distribution: Mapping[str, Any],
    population_distribution: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dimension in ("dataset", "task", "subtype"):
        population = dict(population_distribution.get(dimension, {}))
        tier = dict(tier_distribution.get(dimension, {}))
        output[dimension] = {
            key: (int(tier.get(key, 0)) / int(count) if int(count) else None)
            for key, count in sorted(population.items())
        }
    return output


def build_unified_evaluation_tiers(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """从完整多源评测 population 生成冻结 Unified E1/E2/E3 v2。"""

    config_file = Path(config_path)
    payload = dict(yaml.safe_load(config_file.read_text(encoding="utf-8")) or {})
    payload = dict(
        expand_environment(payload, environ=os.environ, allow_unresolved=False)
    )
    if str(payload.get("schema_version")) != "2.0":
        raise ValueError("Unified evaluation tier config requires schema_version='2.0'")
    root = Path(project_root).resolve()

    def project_path(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path

    data_config = dict(payload.get("data", {}))
    source_files = [project_path(value) for value in data_config.get("source_files", [])]
    train_files = [project_path(value) for value in data_config.get("train_files", [])]
    if not source_files:
        raise ValueError("Unified tiers require data.source_files")
    rows: list[dict[str, Any]] = []
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation population source does not exist: {path}")
        rows.extend(dict(row) for row in read_jsonl(path))
    ids = [str(row.get("id", "")) for row in rows]
    if any(not sample_id for sample_id in ids):
        raise ValueError("Every evaluation row must have a non-empty id")
    if len(set(ids)) != len(ids):
        raise ValueError("Evaluation population contains duplicate sample IDs")
    train_ids = {
        str(row.get("id", ""))
        for path in train_files
        for row in read_jsonl(path)
    }
    leakage = sorted(set(ids).intersection(train_ids))
    if leakage:
        raise ValueError(f"Evaluation/training ID leakage detected: {leakage[:5]}")
    common_root = project_path(data_config["common_image_root"])
    validate_portable_images(rows, common_root)

    stratification = dict(payload.get("stratification", {}))
    thresholds = dict(
        stratification.get(
            "detection_area_thresholds",
            {"small_max": 0.01, "medium_max": 0.10},
        )
    )
    small_max = float(thresholds["small_max"])
    medium_max = float(thresholds["medium_max"])
    dataset_weights = {
        str(key): float(value)
        for key, value in dict(stratification.get("dataset_weights", {})).items()
    }
    datasets = {str(row_metadata(row).get("dataset", "unknown")) for row in rows}
    expected_datasets = set(payload.get("evaluation_scope", {}).get("datasets", []))
    if datasets != expected_datasets:
        raise ValueError(
            f"Evaluation scope mismatch: expected={sorted(expected_datasets)}, "
            f"actual={sorted(datasets)}"
        )
    missing_weights = sorted(datasets.difference(dataset_weights))
    if missing_weights:
        raise ValueError(f"dataset_weights missing sources: {missing_weights}")
    seed = int(payload.get("seed", 42))
    tiers = dict(payload.get("tiers", {}))
    e2_target = min(int(tiers["E2"]["target_samples"]), len(rows))
    e1_target = min(int(tiers["E1"]["target_samples"]), e2_target)
    task_balance = str(stratification.get("task_balance", "sqrt_population"))
    shortage_policy = str(stratification.get("source_shortage_policy", "redistribute"))
    e2_rows, e2_allocation = select_hierarchical_tier(
        rows,
        e2_target,
        seed=seed + 1,
        dataset_weights=dataset_weights,
        task_balance=task_balance,
        small_max=small_max,
        medium_max=medium_max,
        shortage_policy=shortage_policy,
    )
    e1_rows, e1_allocation = select_hierarchical_tier(
        e2_rows,
        e1_target,
        seed=seed,
        dataset_weights=dataset_weights,
        task_balance=task_balance,
        small_max=small_max,
        medium_max=medium_max,
        shortage_policy=shortage_policy,
    )
    e3_rows = sorted(rows, key=lambda row: str(row["id"]))
    e1_ids = {str(row["id"]) for row in e1_rows}
    e2_ids = {str(row["id"]) for row in e2_rows}
    e3_ids = {str(row["id"]) for row in e3_rows}
    if not e1_ids <= e2_ids or not e2_ids <= e3_ids:
        raise ValueError("Unified tier subset invariant failed")

    required_tasks = set(payload.get("evaluation_scope", {}).get("tasks", []))
    e2_tasks = {str(row.get("task_type", "unknown")) for row in e2_rows}
    if not required_tasks <= e2_tasks:
        raise ValueError(f"E2 is missing required tasks: {sorted(required_tasks - e2_tasks)}")

    output_dir = project_path(payload.get("output_dir", "data/evaluation/tiers_v2"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tier_rows = {"E1": e1_rows, "E2": e2_rows, "E3": e3_rows}
    tier_paths = {
        "E1": output_dir / "e1_quick.jsonl",
        "E2": output_dir / "e2_standard.jsonl",
        "E3": output_dir / "e3_full.jsonl",
    }
    for tier, tier_data in tier_rows.items():
        write_jsonl(tier_paths[tier], tier_data)
    population_distribution = distribution(
        e3_rows,
        small_max=small_max,
        medium_max=medium_max,
    )
    manifest: dict[str, Any] = {
        "schema_version": "2.0",
        "tier_version": str(payload.get("tier_version", "unified-v2")),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "evaluation_scope": {
            "datasets": sorted(datasets),
            "tasks": sorted(required_tasks),
            "description": "Unified VRSBench + LEVIR-CC diagnostic evaluation tiers",
        },
        "evaluation_unit": dict(payload.get("evaluation_scope", {}).get("evaluation_unit", {})),
        "common_image_root": str(data_config["common_image_root"]),
        "source_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in source_files
        ],
        "train_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in train_files
        ],
        "stratification": {
            "dataset_weights": dataset_weights,
            "task_balance": task_balance,
            "subtype_balance": "equal_with_capacity_redistribution",
            "source_shortage_policy": shortage_policy,
            "detection_area_thresholds": thresholds,
        },
        "population_distribution": population_distribution,
        "train_evaluation_overlap": 0,
        "image_path_validation": {"valid": True, "checked_samples": len(rows)},
        "E1_subset_of_E2": True,
        "E2_subset_of_E3": True,
        "tiers": {},
    }
    allocations = {"E1": e1_allocation, "E2": e2_allocation, "E3": None}
    for tier, tier_data in tier_rows.items():
        tier_distribution = distribution(
            tier_data,
            small_max=small_max,
            medium_max=medium_max,
        )
        manifest["tiers"][tier] = {
            "path": str(tier_paths[tier]),
            "sample_count": len(tier_data),
            "sha256": file_sha256(tier_paths[tier]),
            "distribution": tier_distribution,
            "sampling_fraction": _sampling_fraction(
                tier_distribution,
                population_distribution,
            ),
            "allocation": allocations[tier],
            "sample_ids": [str(row["id"]) for row in tier_data],
        }
    manifest_path = output_dir / "evaluation_tiers_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
