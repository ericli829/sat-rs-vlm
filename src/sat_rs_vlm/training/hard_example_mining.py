"""Evaluation v1.5-backed hard-example mining and deterministic replay mixing."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.task_protocol import parse_detection
from sat_rs_vlm.training.config import HardAdaptationConfig
from sat_rs_vlm.training.data_statistics import bbox_area_bucket
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _float_metric(metrics: Mapping[str, Any], key: str, default: float) -> float:
    value = metrics.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def score_hard_example(
    row: Mapping[str, Any],
    config: HardAdaptationConfig,
) -> dict[str, Any]:
    """Score one Evaluation v1.5 row while preserving task-specific reasons.

    The function delegates parsing and base metrics to fields already produced by
    Evaluation v1.5. It never computes a competing accuracy contract.
    """

    task = str(row.get("task_type", "unknown")).strip().lower()
    metrics_value = row.get("sample_metrics", {})
    metrics = metrics_value if isinstance(metrics_value, Mapping) else {}
    weights = config.score_weights
    reasons: list[str] = []
    diagnostics: dict[str, Any] = {}

    if task == "detection":
        parse_ok = bool(row.get("parse_ok", metrics.get("parse_success", False)))
        valid_coordinate = bool(metrics.get("valid_coordinate", False))
        label_match = bool(metrics.get("label_match", False))
        iou = _float_metric(metrics, "iou", 0.0)
        center_distance = _float_metric(metrics, "normalized_center_distance", 1.0)
        if not parse_ok:
            reasons.append("parse_failure")
        elif not valid_coordinate:
            reasons.append("invalid_coordinate")
        if not label_match:
            reasons.append("label_error")
        if iou < 0.5:
            reasons.append("low_iou")
        if center_distance >= 0.25:
            reasons.append("large_center_offset")
        reference = parse_detection(str(row.get("reference", "")))
        area_bucket = "unavailable"
        bbox_area: float | None = None
        if reference is not None and reference.valid_coordinate_range:
            x_min, y_min, x_max, y_max = reference.bbox
            bbox_area = (x_max - x_min) * (y_max - y_min)
            area_bucket = bbox_area_bucket(bbox_area, config.bbox_area_thresholds)
            if area_bucket == "small":
                reasons.append("small_object")
        score = (
            weights.detection_iou * (1.0 - _clamp01(iou))
            + weights.detection_label_error * float(not label_match)
            + weights.detection_parse_failure * float(not parse_ok or not valid_coordinate)
            + weights.detection_center_distance * _clamp01(center_distance)
            + weights.detection_small_object_bonus * float(area_bucket == "small")
        )
        diagnostics.update(
            iou=iou,
            generalized_iou=metrics.get("generalized_iou"),
            normalized_center_distance=center_distance,
            label_match=label_match,
            valid_coordinate=valid_coordinate,
            bbox_area=bbox_area,
            bbox_area_bucket=area_bucket,
        )
    elif task == "counting":
        parse_ok = bool(row.get("parse_ok", metrics.get("number_parse_success", False)))
        absolute_error_value = metrics.get("absolute_error")
        absolute_error = (
            float(absolute_error_value) if isinstance(absolute_error_value, (int, float)) else None
        )
        if not parse_ok:
            reasons.append("parse_failure")
        if absolute_error is not None and absolute_error > 0:
            reasons.append("count_error")
        if absolute_error is not None and absolute_error >= 2:
            reasons.append("absolute_error_ge_2")
        error_severity = 1.0 if absolute_error is None else _clamp01(absolute_error / 2.0)
        score = (
            weights.counting_absolute_error * error_severity
            + weights.counting_parse_failure * float(not parse_ok)
        )
        diagnostics.update(
            absolute_error=absolute_error,
            exact_correct=metrics.get("exact_count_correct"),
            within_1_correct=metrics.get("within_1_correct"),
        )
    elif task in {"vqa", "scene_classification"}:
        normalized_correct = bool(metrics.get("normalized_exact_match", False))
        score = weights.text_error * float(not normalized_correct)
        if not bool(row.get("parse_ok", True)):
            reasons.append("parse_failure")
        if not normalized_correct:
            reasons.append("normalized_answer_error")
        metadata_value = row.get("metadata", {})
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        diagnostics["qa_type"] = metadata.get("qa_type") or metadata.get("question_type")
    elif task in {"captioning", "change_detection"}:
        rouge = _float_metric(metrics, "rouge_l_f1_approx", 0.0)
        chrf = _float_metric(metrics, "chrf_approx", 0.0)
        cider = _float_metric(metrics, "cider_d_single_reference_approx", 0.0)
        score = (
            weights.caption_rouge_l * (1.0 - _clamp01(rouge))
            + weights.caption_chrf * (1.0 - _clamp01(chrf))
            + weights.caption_cider * (1.0 - _clamp01(cider / 10.0))
        )
        if not bool(row.get("parse_ok", False)):
            reasons.append("parse_failure")
        reasons.append("low_caption_quality" if task == "captioning" else "change_caption_error")
        diagnostics.update(rouge_l=rouge, chrf=chrf, cider_d_approx=cider)
    else:
        normalized_correct = bool(metrics.get("normalized_exact_match", False))
        score = weights.text_error * float(not normalized_correct)
        if not normalized_correct:
            reasons.append("task_error")

    return {
        "id": str(row.get("id", "")),
        "task_type": task,
        "hard_score": float(score),
        "hard_reason": sorted(set(reasons)),
        "hard_diagnostics": diagnostics,
    }


def load_evaluation_ids(path: str | Path) -> set[str]:
    """Read protected evaluation IDs from JSON, JSONL, or one-ID-per-line text."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Evaluation ID file does not exist: {source}")
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
        values = payload.get("ids", payload) if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise ValueError("Evaluation ID JSON must be a list or an object with an ids list")
        return {str(value).strip() for value in values if str(value).strip()}
    ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            payload = json.loads(stripped)
            if not isinstance(payload, dict) or "id" not in payload:
                raise ValueError(f"line {line_number}: JSONL evaluation ID row needs id")
            ids.add(str(payload["id"]).strip())
        else:
            ids.add(stripped)
    return ids


def _source_and_task(row: Mapping[str, Any]) -> tuple[str, str]:
    metadata_value = row.get("metadata", {})
    metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
    source = str(metadata.get("training_source") or metadata.get("dataset") or "unknown")
    task = str(row.get("task_type", "unknown")).strip().lower() or "unknown"
    return source, task


def _annotate_training_row(
    row: Mapping[str, Any],
    *,
    role: str,
    score: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(row)
    metadata_value = result.get("metadata", {})
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
    metadata["h1_data_role"] = role
    if score is not None:
        metadata["hard_score"] = score["hard_score"]
        metadata["hard_reason"] = score["hard_reason"]
        metadata["hard_diagnostics"] = score["hard_diagnostics"]
    result["metadata"] = metadata
    return result


def _select_replay(
    candidates: Sequence[Mapping[str, Any]],
    target_count: int,
    *,
    seed: int,
) -> list[Mapping[str, Any]]:
    """Select deterministic stratified replay with source/task coverage first."""

    if target_count <= 0 or not candidates:
        return []
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[_source_and_task(row)].append(row)
    rng = random.Random(seed)
    for rows in groups.values():
        rows.sort(key=lambda value: str(value.get("id", "")))
        rng.shuffle(rows)
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    # Preserve one sample from every available source/task cell. Tiny fixtures may
    # therefore exceed the requested ratio; the actual ratio is recorded.
    target_count = min(len(candidates), max(target_count, len(groups)))
    for key in sorted(groups):
        row = groups[key][0]
        selected.append(row)
        selected_ids.add(str(row.get("id", "")))
    remaining = [row for row in candidates if str(row.get("id", "")) not in selected_ids]
    remaining.sort(key=lambda value: str(value.get("id", "")))
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, target_count - len(selected))])
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(rows: Sequence[Mapping[str, Any]], index: int) -> dict[str, int]:
    return dict(sorted(Counter(_source_and_task(row)[index] for row in rows).items()))


def build_hard_example_dataset(
    training_rows: Sequence[Mapping[str, Any]],
    evaluated_rows: Sequence[Mapping[str, Any]],
    excluded_evaluation_ids: set[str],
    config: HardAdaptationConfig,
    *,
    seed: int,
    output_dir: str | Path,
    prediction_source: str,
    source_checkpoint: str,
) -> dict[str, Any]:
    """Build hard, replay, and mixed H1 JSONL files with fail-closed leakage checks."""

    if config.require_evaluation_exclusions:
        if not excluded_evaluation_ids:
            raise ValueError("Evaluation exclusions are required but no IDs were supplied")
        if len(excluded_evaluation_ids) < config.fixed_evaluation_sample_count:
            raise ValueError(
                "Expected at least "
                f"{config.fixed_evaluation_sample_count} protected evaluation IDs, got "
                f"{len(excluded_evaluation_ids)}"
            )
    training_by_id = {str(row.get("id", "")): row for row in training_rows}
    if "" in training_by_id:
        raise ValueError("Every training row must have a non-empty id")
    duplicate_count = len(training_rows) - len(training_by_id)
    if duplicate_count:
        raise ValueError(f"Training data contains {duplicate_count} duplicate sample IDs")

    scored: list[dict[str, Any]] = []
    unmatched_prediction_ids: list[str] = []
    excluded_prediction_ids: list[str] = []
    for evaluated in evaluated_rows:
        sample_id = str(evaluated.get("id", ""))
        if sample_id in excluded_evaluation_ids:
            excluded_prediction_ids.append(sample_id)
            continue
        if sample_id not in training_by_id:
            unmatched_prediction_ids.append(sample_id)
            continue
        score = score_hard_example(evaluated, config)
        if float(score["hard_score"]) >= config.hard_score_threshold:
            scored.append(score)
    scored.sort(key=lambda value: (-float(value["hard_score"]), str(value["id"])))
    if config.max_hard_samples is not None:
        scored = scored[: config.max_hard_samples]
    hard_ids = {str(value["id"]) for value in scored}
    hard_rows = [
        _annotate_training_row(training_by_id[str(value["id"])], role="hard", score=value)
        for value in scored
    ]
    replay_candidates = [
        row
        for sample_id, row in training_by_id.items()
        if sample_id not in hard_ids and sample_id not in excluded_evaluation_ids
    ]
    desired_replay = (
        round(len(hard_rows) * config.replay_ratio / config.hard_ratio) if hard_rows else 0
    )
    selected_replay = _select_replay(replay_candidates, desired_replay, seed=seed)
    replay_rows = [_annotate_training_row(row, role="regular_replay") for row in selected_replay]
    replay_sources = {_source_and_task(row)[0] for row in replay_rows}
    replay_tasks = {_source_and_task(row)[1] for row in replay_rows}
    missing_replay_sources = sorted(set(config.required_replay_sources).difference(replay_sources))
    missing_replay_tasks = sorted(set(config.required_replay_tasks).difference(replay_tasks))
    if config.enforce_replay_coverage and (missing_replay_sources or missing_replay_tasks):
        raise ValueError(
            "Regular replay does not satisfy required coverage: "
            f"missing_sources={missing_replay_sources}, missing_tasks={missing_replay_tasks}. "
            "Add regular training samples or explicitly revise the H1 coverage contract."
        )
    mixed_rows = [*hard_rows, *replay_rows]
    random.Random(seed).shuffle(mixed_rows)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    hard_path = destination / "hard_train.jsonl"
    replay_path = destination / "replay_train.jsonl"
    mixed_path = destination / "h1_train.jsonl"
    manifest_path = destination / "hard_manifest.json"
    write_jsonl(hard_path, hard_rows)
    write_jsonl(replay_path, replay_rows)
    write_jsonl(mixed_path, mixed_rows)
    total = len(mixed_rows)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": source_checkpoint,
        "prediction_source": prediction_source,
        "evaluation_contract_version": config.evaluation_contract_version,
        "mining_config": config.model_dump(mode="json"),
        "hard_thresholds": {
            "hard_score_threshold": config.hard_score_threshold,
            "bbox_area_thresholds": config.bbox_area_thresholds.model_dump(),
        },
        "hard_sample_count": len(hard_rows),
        "regular_replay_count": len(replay_rows),
        "requested_mix": {"hard": config.hard_ratio, "replay": config.replay_ratio},
        "actual_mix": {
            "hard": len(hard_rows) / total if total else None,
            "replay": len(replay_rows) / total if total else None,
        },
        "task_distribution": {
            "hard": _distribution(hard_rows, 1),
            "replay": _distribution(replay_rows, 1),
            "combined": _distribution(mixed_rows, 1),
        },
        "source_distribution": {
            "hard": _distribution(hard_rows, 0),
            "replay": _distribution(replay_rows, 0),
            "combined": _distribution(mixed_rows, 0),
        },
        "hard_sample_ids": [str(row.get("id")) for row in hard_rows],
        "regular_replay_ids": [str(row.get("id")) for row in replay_rows],
        "replay_coverage": {
            "required_sources": config.required_replay_sources,
            "required_tasks": config.required_replay_tasks,
            "missing_sources": missing_replay_sources,
            "missing_tasks": missing_replay_tasks,
            "satisfied": not missing_replay_sources and not missing_replay_tasks,
        },
        "excluded_evaluation_ids": sorted(excluded_evaluation_ids),
        "excluded_evaluation_id_count": len(excluded_evaluation_ids),
        "evaluation_exclusion_statement": (
            f"The fixed {config.fixed_evaluation_sample_count}-sample evaluation set IDs "
            "are excluded from H1 training."
        ),
        "excluded_prediction_ids": sorted(set(excluded_prediction_ids)),
        "unmatched_prediction_ids": sorted(set(unmatched_prediction_ids)),
        "seed": seed,
        "checksums": {
            "hard_train.jsonl": _sha256(hard_path),
            "replay_train.jsonl": _sha256(replay_path),
            "h1_train.jsonl": _sha256(mixed_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def resolve_evaluated_predictions(path: str | Path) -> Path:
    """Resolve either a v1.5 output directory or evaluated_predictions JSONL."""

    source = Path(path)
    candidate = source / "evaluated_predictions.jsonl" if source.is_dir() else source
    if not candidate.is_file():
        raise FileNotFoundError(f"Evaluation v1.5 evaluated predictions not found: {candidate}")
    return candidate


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Materialize JSONL rows for deterministic joining and output."""

    return list(read_jsonl(path))
