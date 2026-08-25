"""Aggregate fault conditions into task-aware sensitivity groups."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

SENSITIVITY_SCHEMA_VERSION = "1.0"
TASK_METRICS = {
    "detection": ("iou", "acc_at_0_5"),
    "counting": ("exact_count_accuracy", "absolute_error"),
    "scene_classification": ("normalized_accuracy",),
    "vqa": ("normalized_accuracy",),
    "captioning": ("rouge_l_f1_approx", "cider_d_single_reference_approx"),
    "change_detection": ("balanced_accuracy", "binary_accuracy"),
}

def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) else None

def _stats(values: Iterable[float]) -> dict[str, Any]:
    vals = list(values)
    if not vals:
        return {"mean": None, "std": None, "ci95": None, "samples": 0}
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": [mean - margin, mean + margin], "samples": len(vals)}

def task_degradations(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    by_task = comparison.get("by_task", {})
    if not isinstance(by_task, Mapping):
        return rows
    for task, metric_names in TASK_METRICS.items():
        payload = by_task.get(task, {})
        metrics = payload.get("metrics", {}) if isinstance(payload, Mapping) else {}
        for name in metric_names:
            metric = metrics.get(name, {}) if isinstance(metrics, Mapping) else {}
            if not isinstance(metric, Mapping) or metric.get("status") != "ok":
                continue
            improvement = _num(metric.get("improvement_mean"))
            if improvement is None:
                continue
            rows.append({"task": task, "metric": name, "degradation": -improvement, "num_samples": metric.get("num_samples")})
    return rows

def aggregate_sensitivity_conditions(conditions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for condition in conditions:
        key = (str(condition.get("target")), tuple(condition.get("layers", [])), str(condition.get("bit_plane", "all")), int(condition.get("num_bits", condition.get("intensity", 0))))
        grouped[key].append(condition)
    groups = []
    for (target, layers, bit_plane, intensity), repeats in sorted(grouped.items()):
        changed = _stats(v for c in repeats for v in [_num(c.get("comparison", {}).get("overall", {}).get("prediction_changed_rate"))] if v is not None)
        invalid = _stats(v for c in repeats for v in [_num(c.get("injection", {}).get("evaluation", {}).get("invalid_rate"))] if v is not None)
        task_values = defaultdict(list)
        for c in repeats:
            for row in task_degradations(c.get("comparison", {}) if isinstance(c.get("comparison"), Mapping) else {}):
                task_values[(row["task"], row["metric"])].append(row["degradation"])
        task_rows = [{"task": t, "metric": m, **{f"degradation_{k}": v for k, v in _stats(vals).items() if k != "samples"}, "repeats": len(vals)} for (t, m), vals in sorted(task_values.items())]
        groups.append({"target": target, "layers": list(layers), "bit_plane": bit_plane, "intensity": intensity, "repeats": len(repeats), "condition_ids": [str(c.get("id", "")) for c in repeats], "changed_rate_mean": changed["mean"], "changed_rate_ci95": changed["ci95"], "invalid_rate_mean": invalid["mean"], "invalid_rate_ci95": invalid["ci95"], "task_degradations": task_rows})
    return {"schema_version": SENSITIVITY_SCHEMA_VERSION, "aggregation": "target+layers+bit_plane+intensity", "groups": groups}
