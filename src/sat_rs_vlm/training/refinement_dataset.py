"""H2 Global Multitask Refinement 的可复现数据构建协议。"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sat_rs_vlm.evaluation.tier_builder import (
    allocate_counts,
    distribution,
    select_hierarchical_tier,
)
from sat_rs_vlm.evaluation.tiers import UNIFIED_TIER_VERSION, file_sha256
from sat_rs_vlm.training.config import (
    H2RefinementConfig,
    HardAdaptationConfig,
)
from sat_rs_vlm.training.hard_example_mining import score_training_samples
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl


def _source(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    values = metadata if isinstance(metadata, Mapping) else {}
    return str(values.get("dataset", values.get("training_source", "unknown")))


def _task(row: Mapping[str, Any]) -> str:
    return str(row.get("task_type", "unknown")).strip().lower() or "unknown"


def _cell(row: Mapping[str, Any]) -> tuple[str, str]:
    return _source(row), _task(row)


def _rows_by_id(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id:
            raise ValueError(f"Every {label} row must have a non-empty id")
        if sample_id in output:
            raise ValueError(f"Duplicate {label} id: {sample_id}")
        output[sample_id] = row
    return output


def load_protected_e3(manifest_path: str | Path) -> dict[str, Any]:
    """读取并验证 Unified v2 E3 身份及全部受保护 ID。"""

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Unified evaluation manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    tier_version = str(payload.get("tier_version", ""))
    if tier_version != UNIFIED_TIER_VERSION:
        raise ValueError(
            f"H2 requires {UNIFIED_TIER_VERSION}, got {tier_version or 'missing'}"
        )
    e3 = dict(payload.get("tiers", {}).get("E3", {}))
    sample_ids = [str(value) for value in e3.get("sample_ids", [])]
    if len(sample_ids) != int(e3.get("sample_count", -1)):
        raise ValueError("Unified E3 manifest sample IDs/count are inconsistent")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Unified E3 manifest contains duplicate sample IDs")
    sha256 = str(e3.get("sha256", ""))
    if not sha256:
        raise ValueError("Unified E3 manifest is missing SHA256")
    return {
        "version": tier_version,
        "sha256": sha256,
        "sample_ids": set(sample_ids),
        "sample_count": len(sample_ids),
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
    }


def _annotate(row: Mapping[str, Any], **metadata_values: Any) -> dict[str, Any]:
    output = dict(row)
    metadata = dict(row.get("metadata", {}))
    metadata.update(metadata_values)
    output["metadata"] = metadata
    return output


def _distribution_by_source_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source": dict(sorted(Counter(_source(row) for row in rows).items())),
        "task": dict(sorted(Counter(_task(row) for row in rows).items())),
        "source_task": dict(
            sorted(Counter(f"{_source(row)}/{_task(row)}" for row in rows).items())
        ),
    }


def build_h2_mining_candidates(
    training_rows: Sequence[Mapping[str, Any]],
    protected_e3: Mapping[str, Any],
    config: H2RefinementConfig,
    *,
    output_file: str | Path,
    manifest_file: str | Path,
    source_training_file: str | Path,
    bbox_small_max: float = 0.01,
    bbox_medium_max: float = 0.10,
) -> dict[str, Any]:
    """构建不加载模型的 H2 mining candidates，并 fail-closed 排除 E3。"""

    by_id = _rows_by_id(training_rows, "training")
    protected_ids = set(protected_e3["sample_ids"])
    legal_rows = [row for sample_id, row in by_id.items() if sample_id not in protected_ids]
    if config.mining_target_samples > len(legal_rows):
        raise ValueError(
            f"H2 mining target {config.mining_target_samples} exceeds legal training "
            f"population {len(legal_rows)}"
        )
    selected, allocation = select_hierarchical_tier(
        [dict(row) for row in legal_rows],
        config.mining_target_samples,
        seed=config.seed,
        dataset_weights=config.source_weights,
        task_balance=config.task_balance,
        small_max=bbox_small_max,
        medium_max=bbox_medium_max,
        shortage_policy="redistribute",
    )
    selected = [
        _annotate(row, h2_mining_candidate=True, h2_candidate_seed=config.seed)
        for row in selected
    ]
    selected_ids = {str(row["id"]) for row in selected}
    leakage = sorted(selected_ids.intersection(protected_ids))
    if leakage:
        raise ValueError(f"H2 mining candidate evaluation leakage: {leakage[:5]}")
    destination = Path(output_file)
    manifest_path = Path(manifest_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination, selected)
    source_path = Path(source_training_file)
    manifest = {
        "schema_version": "2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "source_checkpoint": config.source_checkpoint,
        "source_checkpoint_role": "replay_generalist",
        "config_snapshot": config.model_dump(mode="json"),
        "target_samples": config.mining_target_samples,
        "actual_samples": len(selected),
        "requested_source_weights": config.source_weights,
        "task_balance": config.task_balance,
        "allocation": allocation,
        "distribution": _distribution_by_source_task(selected),
        "tier_distribution": distribution(
            selected,
            small_max=bbox_small_max,
            medium_max=bbox_medium_max,
        ),
        "protected_evaluation_tier": {
            "version": protected_e3["version"],
            "E3_sha256": protected_e3["sha256"],
            "excluded_id_count": protected_e3["sample_count"],
        },
        "source_training_file": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        },
        "sample_ids": [str(row["id"]) for row in selected],
        "duplicate_check": {"passed": len(selected_ids) == len(selected)},
        "evaluation_leakage_check": {"passed": not leakage, "overlap_count": 0},
        "output": {"path": str(destination), "sha256": file_sha256(destination)},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _task_weights(rows: Sequence[Mapping[str, Any]], mode: str) -> dict[str, float]:
    counts = Counter(_task(row) for row in rows)
    if mode == "sqrt_population":
        return {task: math.sqrt(count) for task, count in counts.items()}
    if mode == "natural":
        return {task: float(count) for task, count in counts.items()}
    raise ValueError(f"Unsupported H2 task balance mode: {mode}")


def _source_role_targets(config: H2RefinementConfig) -> dict[str, dict[str, int]]:
    roles = {
        "regular_representative": config.difficulty_mix.regular_representative,
        "medium_hard": config.difficulty_mix.medium_hard,
        "core_hard": config.difficulty_mix.core_hard,
    }
    role_capacities = {role: config.target_samples for role in roles}
    _, role_counts, _ = allocate_counts(
        role_capacities,
        config.target_samples,
        weights=roles,
        shortage_policy="fail",
    )
    _, source_totals, _ = allocate_counts(
        {source: config.target_samples for source in config.source_weights},
        config.target_samples,
        weights=config.source_weights,
        shortage_policy="fail",
    )
    output: dict[str, dict[str, int]] = {}
    source_capacities = {source: config.target_samples for source in config.source_weights}
    for role, count in role_counts.items():
        _, source_counts, _ = allocate_counts(
            source_capacities,
            count,
            weights=config.source_weights,
            shortage_policy="fail",
        )
        output[role] = source_counts
    current_totals = {
        source: sum(output[role][source] for role in output)
        for source in source_totals
    }
    while current_totals != source_totals:
        under = next(
            source
            for source in sorted(source_totals)
            if current_totals[source] < source_totals[source]
        )
        over = next(
            source
            for source in sorted(source_totals)
            if current_totals[source] > source_totals[source]
        )
        role = next(
            role
            for role in sorted(output)
            if output[role][over] > 0
        )
        output[role][over] -= 1
        output[role][under] += 1
        current_totals[over] -= 1
        current_totals[under] += 1
    return output


def _task_targets(
    rows: Sequence[Mapping[str, Any]],
    target: int,
    *,
    mode: str,
) -> dict[str, int]:
    capacities = Counter(_task(row) for row in rows)
    _, actual, _ = allocate_counts(
        capacities,
        target,
        weights=_task_weights(rows, mode),
        shortage_policy="fail",
    )
    return actual


def _ranked_cell_rows(
    scored: Sequence[Mapping[str, Any]],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for score in scored:
        sample_id = str(score["id"])
        candidate = candidates_by_id[sample_id]
        combined = dict(score)
        combined["source"] = _source(candidate)
        combined["task_type"] = _task(candidate)
        cells[(combined["source"], combined["task_type"])].append(combined)
    for values in cells.values():
        values.sort(key=lambda row: (-float(row["hard_score"]), str(row["id"])))
    return cells


def _score_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["metadata"]["hard_score"]) for row in rows]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _annotate_ranked(
    training_row: Mapping[str, Any],
    score: Mapping[str, Any],
    *,
    role: str,
    rank: int,
    cell_size: int,
) -> dict[str, Any]:
    return _annotate(
        training_row,
        h2_data_role=role,
        hard_score=float(score["hard_score"]),
        hard_reason=list(score.get("hard_reason", [])),
        hard_diagnostics=dict(score.get("hard_diagnostics", {})),
        difficulty_rank=rank,
        difficulty_cell=f"{_source(training_row)}/{_task(training_row)}",
        difficulty_percentile=(1.0 - ((rank - 1) / cell_size) if cell_size else None),
    )


def build_h2_refinement_dataset(
    training_rows: Sequence[Mapping[str, Any]],
    mining_candidates: Sequence[Mapping[str, Any]],
    evaluated_rows: Sequence[Mapping[str, Any]],
    protected_e3: Mapping[str, Any],
    config: H2RefinementConfig,
    hard_config: HardAdaptationConfig,
    *,
    source_training_file: str | Path,
    mining_candidates_file: str | Path,
    prediction_source: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """按 source/task cell-local ranking 构建 H2 regular/medium/core 数据集。"""

    if config.difficulty_mode != "cell_rank":
        raise ValueError(
            "The formal H2 builder requires difficulty_mode='cell_rank'; threshold mode "
            "is experimental and must use a separate experiment implementation"
        )
    if not config.source_checkpoint:
        raise ValueError("H2 final dataset requires a Replay generalist source checkpoint")
    training_by_id = _rows_by_id(training_rows, "training")
    candidates_by_id = _rows_by_id(mining_candidates, "mining candidate")
    evaluated_by_id = _rows_by_id(evaluated_rows, "evaluated prediction")
    candidate_ids = set(candidates_by_id)
    if set(evaluated_by_id) != candidate_ids:
        raise ValueError(
            "Mining evaluation must contain exactly the candidate IDs: "
            f"missing={sorted(candidate_ids - set(evaluated_by_id))[:5]}, "
            f"unexpected={sorted(set(evaluated_by_id) - candidate_ids)[:5]}"
        )
    missing_training = sorted(candidate_ids.difference(training_by_id))
    if missing_training:
        raise ValueError(
            f"Mining candidates absent from training population: {missing_training[:5]}"
        )
    protected_ids = set(protected_e3["sample_ids"])
    illegal_training = sorted(set(training_by_id).intersection(protected_ids))
    if illegal_training:
        raise ValueError(
            f"Source training population contains protected E3 IDs: {illegal_training[:5]}"
        )

    scored = score_training_samples(list(evaluated_by_id.values()), hard_config)
    cells = _ranked_cell_rows(scored, candidates_by_id)
    role_source_targets = _source_role_targets(config)
    selected_core: list[dict[str, Any]] = []
    selected_medium: list[dict[str, Any]] = []
    selected_hard_ids: set[str] = set()
    task_allocation_manifest: dict[str, Any] = {}
    for source in sorted(config.source_weights):
        source_training = [row for row in training_rows if _source(row) == source]
        source_candidates = [row for row in mining_candidates if _source(row) == source]
        if not source_training or not source_candidates:
            raise ValueError(f"H2 source is missing from training/candidates: {source}")
        core_targets = _task_targets(
            source_training,
            role_source_targets["core_hard"][source],
            mode=config.task_balance,
        )
        medium_targets = _task_targets(
            source_training,
            role_source_targets["medium_hard"][source],
            mode=config.task_balance,
        )
        population_counts = Counter(_task(row) for row in source_training)
        task_allocation_manifest[source] = {
            "population": dict(sorted(population_counts.items())),
            "weights": dict(sorted(_task_weights(source_training, config.task_balance).items())),
            "core_requested": core_targets,
            "medium_requested": medium_targets,
        }
        for task in sorted(population_counts):
            ranked = cells.get((source, task), [])
            core_count = core_targets.get(task, 0)
            medium_count = medium_targets.get(task, 0)
            required = core_count + medium_count
            if len(ranked) < required:
                raise ValueError(
                    f"Insufficient evaluated candidates for cell {source}/{task}: "
                    f"required={required}, available={len(ranked)}"
                )
            for index, score in enumerate(ranked[:core_count], 1):
                sample_id = str(score["id"])
                selected_core.append(
                    _annotate_ranked(
                        training_by_id[sample_id],
                        score,
                        role="core_hard",
                        rank=index,
                        cell_size=len(ranked),
                    )
                )
                selected_hard_ids.add(sample_id)
            for offset, score in enumerate(
                ranked[core_count : core_count + medium_count],
                core_count + 1,
            ):
                sample_id = str(score["id"])
                selected_medium.append(
                    _annotate_ranked(
                        training_by_id[sample_id],
                        score,
                        role="medium_hard",
                        rank=offset,
                        cell_size=len(ranked),
                    )
                )
                selected_hard_ids.add(sample_id)

    regular_target = sum(role_source_targets["regular_representative"].values())
    regular_pool = [
        dict(row)
        for sample_id, row in training_by_id.items()
        if sample_id not in selected_hard_ids and sample_id not in protected_ids
    ]
    selected_regular, regular_allocation = select_hierarchical_tier(
        regular_pool,
        regular_target,
        seed=config.seed + 1,
        dataset_weights=config.source_weights,
        task_balance=config.task_balance,
        small_max=hard_config.bbox_area_thresholds.small_max,
        medium_max=hard_config.bbox_area_thresholds.medium_max,
        shortage_policy="redistribute",
    )
    selected_regular = [
        _annotate(row, h2_data_role="regular_representative")
        for row in selected_regular
    ]
    all_rows = [*selected_regular, *selected_medium, *selected_core]
    random.Random(config.seed).shuffle(all_rows)
    all_ids = [str(row["id"]) for row in all_rows]
    if len(all_rows) != config.target_samples or len(set(all_ids)) != len(all_rows):
        raise ValueError("H2 final selection count/duplicate invariant failed")
    leakage = sorted(set(all_ids).intersection(protected_ids))
    if leakage:
        raise ValueError(f"H2 final dataset evaluation leakage: {leakage[:5]}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "core": destination / "core_hard_train.jsonl",
        "medium": destination / "medium_hard_train.jsonl",
        "regular": destination / "regular_representative_train.jsonl",
        "combined": destination / "h2_train.jsonl",
        "manifest": destination / "h2_manifest.json",
    }
    write_jsonl(paths["core"], selected_core)
    write_jsonl(paths["medium"], selected_medium)
    write_jsonl(paths["regular"], selected_regular)
    write_jsonl(paths["combined"], all_rows)
    role_rows = {
        "regular_representative": selected_regular,
        "medium_hard": selected_medium,
        "core_hard": selected_core,
    }
    source_path = Path(source_training_file)
    candidates_path = Path(mining_candidates_file)
    predictions_path = Path(prediction_source)
    role_counts = {role: len(rows) for role, rows in role_rows.items()}
    source_counts = Counter(_source(row) for row in all_rows)
    hard_score_summary: dict[str, Any] = {}
    for role in ("medium_hard", "core_hard"):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in role_rows[role]:
            grouped[f"{_source(row)}/{_task(row)}"].append(row)
        hard_score_summary[role] = {
            cell: _score_summary(rows) for cell, rows in sorted(grouped.items())
        }
    manifest = {
        "schema_version": "2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "config_snapshot": config.model_dump(mode="json"),
        "source_checkpoint": config.source_checkpoint,
        "source_checkpoint_role": "replay_generalist",
        "prediction_source": str(predictions_path),
        "prediction_source_sha256": file_sha256(predictions_path),
        "evaluation_contract_version": config.evaluation_contract_version,
        "protected_evaluation_tier": {
            "version": protected_e3["version"],
            "E3_sha256": protected_e3["sha256"],
            "excluded_id_count": protected_e3["sample_count"],
        },
        "mining_candidates": {
            "path": str(candidates_path),
            "sha256": file_sha256(candidates_path),
            "count": len(mining_candidates),
        },
        "source_training_file": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
            "count": len(training_rows),
        },
        "target_samples": config.target_samples,
        "difficulty_mode": "source_task_cell_rank",
        "stable_ranking": "hard_score DESC, id ASC",
        "requested_difficulty_mix": config.difficulty_mix.model_dump(mode="json"),
        "actual_difficulty_mix": {
            role: count / len(all_rows) for role, count in role_counts.items()
        },
        "requested_source_weights": config.source_weights,
        "actual_source_distribution": {
            source: count / len(all_rows) for source, count in sorted(source_counts.items())
        },
        "role_source_targets": role_source_targets,
        "task_allocation": task_allocation_manifest,
        "regular_allocation": regular_allocation,
        "role_distributions": {
            role: _distribution_by_source_task(rows) for role, rows in role_rows.items()
        },
        "combined_distribution": _distribution_by_source_task(all_rows),
        "hard_score_summary": hard_score_summary,
        "all_selected_sample_ids": all_ids,
        "duplicate_check": {"passed": len(set(all_ids)) == len(all_ids)},
        "evaluation_leakage_check": {"passed": not leakage, "overlap_count": 0},
        "output_sha256": {
            "core": file_sha256(paths["core"]),
            "medium": file_sha256(paths["medium"]),
            "regular": file_sha256(paths["regular"]),
            "h2_train": file_sha256(paths["combined"]),
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    return [dict(row) for row in read_jsonl(path)]
