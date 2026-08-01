"""clean、fault 与 recovery 输出的统一可靠性指标。"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from sat_rs_vlm.domain.tasks import TaskType
from sat_rs_vlm.evaluation.metrics import score_counting, score_detection
from sat_rs_vlm.models.reliability.output_validator import (
    validate_prediction,
)


def _index_rows(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise ValueError(f"{label} prediction row is missing id")
        if sample_id in indexed:
            raise ValueError(f"{label} predictions contain duplicate id: {sample_id}")
        indexed[sample_id] = row
    return indexed


def build_prediction_pairs(
    clean_rows: Iterable[dict[str, Any]],
    fault_rows: Iterable[dict[str, Any]],
    recovered_rows: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """按样本 ID 严格配对预测，并预计算输出合法性和精确匹配字段。"""

    clean = _index_rows(clean_rows, "clean")
    fault = _index_rows(fault_rows, "fault")
    if set(clean) != set(fault):
        missing_fault = sorted(set(clean).difference(fault))
        missing_clean = sorted(set(fault).difference(clean))
        raise ValueError(
            f"Clean/fault sample IDs differ; missing_fault={missing_fault}, "
            f"missing_clean={missing_clean}"
        )
    recovered = _index_rows(recovered_rows, "recovered") if recovered_rows is not None else None
    if recovered is not None and set(recovered) != set(clean):
        raise ValueError("Recovered prediction IDs differ from clean prediction IDs")

    pairs: list[dict[str, Any]] = []
    for sample_id, clean_row in clean.items():
        fault_row = fault[sample_id]
        task = str(clean_row.get("task_type", fault_row.get("task_type", TaskType.UNKNOWN.value)))
        reference = str(clean_row.get("reference", fault_row.get("reference", "")))
        clean_prediction = str(clean_row.get("prediction", ""))
        fault_prediction = str(fault_row.get("prediction", ""))
        clean_validation = validate_prediction(task, clean_prediction)
        fault_validation = validate_prediction(task, fault_prediction)
        pair: dict[str, Any] = {
            "id": sample_id,
            "task_type": task,
            "reference": reference,
            "clean_prediction": clean_prediction,
            "fault_prediction": fault_prediction,
            "changed": clean_prediction.strip() != fault_prediction.strip(),
            "clean_valid": clean_validation.valid,
            "fault_valid": fault_validation.valid,
            "clean_validation_errors": clean_validation.errors,
            "fault_validation_errors": fault_validation.errors,
            "clean_exact_match": clean_prediction.strip() == reference.strip(),
            "fault_exact_match": fault_prediction.strip() == reference.strip(),
            "metadata": clean_row.get("metadata", {}),
        }
        if recovered is not None:
            recovered_prediction = str(recovered[sample_id].get("prediction", ""))
            recovered_validation = validate_prediction(task, recovered_prediction)
            pair.update(
                {
                    "recovered_prediction": recovered_prediction,
                    "recovered_valid": recovered_validation.valid,
                    "recovery_success": recovered_prediction.strip() == clean_prediction.strip(),
                    "post_recovery_exact_match": recovered_prediction.strip() == reference.strip(),
                }
            )
        pairs.append(pair)
    return pairs


def _rate(rows: list[dict[str, Any]], field: str) -> float | None:
    return sum(bool(row.get(field)) for row in rows) / len(rows) if rows else None


def _task_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    empty_rate = (
        sum(not str(row.get("fault_prediction", "")).strip() for row in rows) / len(rows)
        if rows
        else None
    )
    clean_exact = _rate(rows, "clean_exact_match")
    fault_exact = _rate(rows, "fault_exact_match")
    metrics: dict[str, Any] = {
        "num_samples": len(rows),
        "changed_rate": _rate(rows, "changed"),
        "invalid_rate": (
            sum(not bool(row.get("fault_valid")) for row in rows) / len(rows) if rows else None
        ),
        "empty_rate": empty_rate,
        "clean_exact_match": clean_exact,
        "fault_exact_match": fault_exact,
        "exact_match_drop": (
            clean_exact - fault_exact
            if clean_exact is not None and fault_exact is not None
            else None
        ),
        "recovery_success_rate": (
            _rate(rows, "recovery_success")
            if any("recovery_success" in row for row in rows)
            else None
        ),
        "post_recovery_exact_match": (
            _rate(rows, "post_recovery_exact_match")
            if any("post_recovery_exact_match" in row for row in rows)
            else None
        ),
    }
    task = str(rows[0].get("task_type", "")) if rows else ""
    if task == TaskType.COUNTING.value:
        errors: list[float] = []
        correct = 0
        for row in rows:
            score = score_counting(
                str(row.get("fault_prediction", "")),
                str(row.get("reference", "")),
            )
            if score["mae"] is None:
                continue
            error = float(score["mae"])
            errors.append(error)
            correct += int(float(score["acc_exact"]) == 1.0)
        metrics["counting_mae"] = sum(errors) / len(errors) if errors else None
        metrics["counting_accuracy"] = correct / len(errors) if errors else None
    if task == TaskType.DETECTION.value:
        ious: list[float] = []
        for row in rows:
            score = score_detection(
                str(row.get("fault_prediction", "")),
                str(row.get("reference", "")),
            )
            if score["iou"] is not None:
                ious.append(float(score["iou"]))
        metrics["detection_mean_iou"] = sum(ious) / len(ious) if ious else None
        metrics["detection_acc_at_0_5"] = (
            sum(iou >= 0.5 for iou in ious) / len(ious) if ious else None
        )
    return metrics


def summarize_reliability(
    pairs: Iterable[dict[str, Any]],
    *,
    execution_mode: str,
    experiment_name: str,
    run_id: str,
) -> dict[str, Any]:
    """生成带版本、运行标识、总体与按任务指标的稳定报告 schema。"""

    rows = list(pairs)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task_type", TaskType.UNKNOWN.value))].append(row)
    overall = _task_metrics(rows)
    by_task = {task: _task_metrics(task_rows) for task, task_rows in sorted(grouped.items())}
    for section in (overall, *by_task.values()):
        for key, value in section.items():
            if isinstance(value, float) and not math.isfinite(value):
                section[key] = None
    return {
        "schema_version": "1.0",
        "execution_mode": execution_mode,
        "experiment_name": experiment_name,
        "run_id": run_id,
        "overall": overall,
        "by_task": by_task,
    }
