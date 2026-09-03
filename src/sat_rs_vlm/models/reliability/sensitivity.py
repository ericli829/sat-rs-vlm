"""Aggregate repeated SEU conditions into versioned sensitivity groups.

The experiment runner stores one paired Evaluation v1.5 comparison per repeat.
This module is the canonical adapter between those raw conditions and deployment
risk policy. It deliberately consumes task metrics instead of reducing every
remote-sensing task to exact-match accuracy.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

SENSITIVITY_SCHEMA_VERSION = "1.0"

# Ordered preferences let newer Evaluation contracts add a stronger metric while
# old v1.5 reports remain readable.
TASK_METRICS: dict[str, tuple[str, ...]] = {
    "detection": ("iou", "acc_at_0_5"),
    "counting": ("exact_count_accuracy", "absolute_error"),
    "scene_classification": ("normalized_accuracy",),
    "vqa": ("normalized_accuracy",),
    "captioning": ("rouge_l_f1_approx", "cider_d_single_reference_approx"),
    "change_detection": ("balanced_accuracy", "binary_accuracy"),
}


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean_ci95(values: Iterable[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {"mean": None, "std": None, "ci95": None, "samples": 0}
    mean = statistics.fmean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(samples)) if len(samples) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95": [mean - margin, mean + margin],
        "samples": len(samples),
    }


def task_degradations(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract task-aware degradation from paired comparison output.

    Evaluation reports express ``improvement_mean`` so positive is always better,
    including lower-is-better metrics such as MAE. Degradation is therefore the
    negated improvement. Negative values are retained as measured improvements;
    policy code clamps them only when determining risk.
    """

    rows: list[dict[str, Any]] = []
    by_task = comparison.get("by_task", {})
    if not isinstance(by_task, Mapping):
        return rows
    for task, preferred_metrics in TASK_METRICS.items():
        task_payload = by_task.get(task, {})
        metrics = task_payload.get("metrics", {}) if isinstance(task_payload, Mapping) else {}
        if not isinstance(metrics, Mapping):
            continue
        for metric_name in preferred_metrics:
            metric = metrics.get(metric_name)
            if not isinstance(metric, Mapping) or metric.get("status") != "ok":
                continue
            improvement = _numeric(metric.get("improvement_mean"))
            if improvement is None:
                continue
            improvement_ci = metric.get("improvement_ci95_paired_bootstrap")
            degradation_ci = None
            if (
                isinstance(improvement_ci, list)
                and len(improvement_ci) == 2
                and _numeric(improvement_ci[0]) is not None
                and _numeric(improvement_ci[1]) is not None
            ):
                degradation_ci = [-float(improvement_ci[1]), -float(improvement_ci[0])]
            rows.append(
                {
                    "task": task,
                    "metric": metric_name,
                    "degradation": -improvement,
                    "paired_ci95": degradation_ci,
                    "num_samples": metric.get("num_samples"),
                }
            )
    return rows


def _condition_diagnostics(condition: Mapping[str, Any]) -> dict[str, float | None]:
    comparison = condition.get("comparison", {})
    overall = comparison.get("overall", {}) if isinstance(comparison, Mapping) else {}
    injection = condition.get("injection", {})
    evaluation = injection.get("evaluation", {}) if isinstance(injection, Mapping) else {}
    return {
        "changed_rate": _numeric(overall.get("prediction_changed_rate")),
        "invalid_rate": (
            _numeric(evaluation.get("invalid_rate")) if isinstance(evaluation, Mapping) else None
        ),
    }


def aggregate_sensitivity_conditions(
    conditions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate repeats by target, layer, bit plane and fault intensity."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for condition in conditions:
        key = (
            str(condition.get("target")),
            tuple(int(value) for value in condition.get("layers", [])),
            str(condition.get("bit_plane", "all")),
            int(condition.get("num_bits", condition.get("intensity", 0))),
        )
        grouped[key].append(condition)

    groups: list[dict[str, Any]] = []
    for (target, layers, bit_plane, intensity), repeats in sorted(grouped.items()):
        diagnostics = [_condition_diagnostics(item) for item in repeats]
        task_values: dict[tuple[str, str], list[float]] = defaultdict(list)
        task_metadata: dict[tuple[str, str], dict[str, Any]] = {}
        for condition in repeats:
            comparison = condition.get("comparison", {})
            for row in task_degradations(comparison if isinstance(comparison, Mapping) else {}):
                metric_key = (str(row["task"]), str(row["metric"]))
                task_values[metric_key].append(float(row["degradation"]))
                task_metadata[metric_key] = row
        task_rows: list[dict[str, Any]] = []
        for metric_key in sorted(task_values):
            stats = _mean_ci95(task_values[metric_key])
            task_rows.append(
                {
                    "task": metric_key[0],
                    "metric": metric_key[1],
                    "degradation_mean": stats["mean"],
                    "degradation_std": stats["std"],
                    "degradation_ci95_across_repeats": stats["ci95"],
                    "repeats": stats["samples"],
                    "paired_num_samples": task_metadata[metric_key].get("num_samples"),
                }
            )
        changed = _mean_ci95(
            value for row in diagnostics if (value := row["changed_rate"]) is not None
        )
        invalid = _mean_ci95(
            value for row in diagnostics if (value := row["invalid_rate"]) is not None
        )
        groups.append(
            {
                "target": target,
                "layers": list(layers),
                "bit_plane": bit_plane,
                "intensity": intensity,
                "repeats": len(repeats),
                "condition_ids": [str(item.get("id", "")) for item in repeats],
                "changed_rate_mean": changed["mean"],
                "changed_rate_std": changed["std"],
                "changed_rate_ci95": changed["ci95"],
                "invalid_rate_mean": invalid["mean"],
                "invalid_rate_std": invalid["std"],
                "invalid_rate_ci95": invalid["ci95"],
                "task_degradations": task_rows,
            }
        )
    return {
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        "aggregation": "target+layers+bit_plane+intensity across repeats",
        "groups": groups,
    }
