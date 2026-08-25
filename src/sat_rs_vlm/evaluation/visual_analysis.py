"""Small-object and visual-budget diagnostics for Evaluation v1.5 outputs."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from sat_rs_vlm.data.task_protocol import parse_detection
from sat_rs_vlm.training.config import BBoxAreaThresholdConfig


def bbox_area_bucket(area: float, thresholds: BBoxAreaThresholdConfig) -> str:
    """Classify normalized bounding-box area without the training package."""
    if area < thresholds.small_max:
        return "small"
    if area < thresholds.medium_max:
        return "medium"
    return "large"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def pearson_correlation(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Compute Pearson r, returning None for missing or constant populations."""

    if len(pairs) < 2:
        return None
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean) for left_value, right_value in pairs
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    denominator = left_scale * right_scale
    return numerator / denominator if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def analyze_detection_visual_factors(
    rows: Sequence[Mapping[str, Any]],
    thresholds: BBoxAreaThresholdConfig,
) -> dict[str, Any]:
    """Aggregate detection quality by bbox size and correlate visual diagnostics."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    area_iou_pairs: list[tuple[float, float]] = []
    token_iou_pairs: list[tuple[float, float]] = []
    resolution_iou_pairs: list[tuple[float, float]] = []
    unavailable = {"visual_token_count": 0, "image_resolution": 0}
    for row in rows:
        if str(row.get("task_type", "")).lower() != "detection":
            continue
        reference = parse_detection(str(row.get("reference", "")))
        if reference is None or not reference.valid_coordinate_range:
            continue
        x_min, y_min, x_max, y_max = reference.bbox
        area = (x_max - x_min) * (y_max - y_min)
        metrics_value = row.get("sample_metrics", {})
        metrics = metrics_value if isinstance(metrics_value, Mapping) else {}
        iou = _number(metrics.get("iou"))
        if iou is None:
            continue
        bucket = bbox_area_bucket(area, thresholds)
        sample = {
            "iou": iou,
            "generalized_iou": _number(metrics.get("generalized_iou")),
            "center_distance": _number(metrics.get("normalized_center_distance")),
            "label_match": float(bool(metrics.get("label_match", False))),
            "parse_success": float(bool(metrics.get("parse_success", row.get("parse_ok", False)))),
        }
        buckets[bucket].append(sample)
        area_iou_pairs.append((area, iou))
        metadata_value = row.get("metadata", {})
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        visual_tokens = _number(
            metadata.get("approximate_visual_tokens", metadata.get("visual_token_count"))
        )
        if visual_tokens is None:
            unavailable["visual_token_count"] += 1
        else:
            token_iou_pairs.append((visual_tokens, iou))
        width = _number(metadata.get("image_width", metadata.get("width")))
        height = _number(metadata.get("image_height", metadata.get("height")))
        if width is None or height is None:
            unavailable["image_resolution"] += 1
        else:
            resolution_iou_pairs.append((width * height, iou))

    by_size: dict[str, Any] = {}
    for bucket in ("small", "medium", "large"):
        values = buckets.get(bucket, [])
        by_size[bucket] = {
            "sample_count": len(values),
            "mean_iou": _mean([float(value["iou"]) for value in values]),
            "mean_generalized_iou": _mean(
                [
                    float(value["generalized_iou"])
                    for value in values
                    if value["generalized_iou"] is not None
                ]
            ),
            "mean_center_distance": _mean(
                [
                    float(value["center_distance"])
                    for value in values
                    if value["center_distance"] is not None
                ]
            ),
            "label_match_rate": _mean([float(value["label_match"]) for value in values]),
            "parse_success_rate": _mean([float(value["parse_success"]) for value in values]),
        }
    return {
        "bbox_area_thresholds": thresholds.model_dump(),
        "by_bbox_size": by_size,
        "correlations": {
            "iou_vs_bbox_area": pearson_correlation(area_iou_pairs),
            "iou_vs_visual_token_count": pearson_correlation(token_iou_pairs),
            "iou_vs_image_pixel_count": pearson_correlation(resolution_iou_pairs),
        },
        "correlation_sample_counts": {
            "bbox_area": len(area_iou_pairs),
            "visual_token_count": len(token_iou_pairs),
            "image_resolution": len(resolution_iou_pairs),
        },
        "unavailable": unavailable,
        "interpretation_guardrail": (
            "Correlation is diagnostic, not causal; unavailable processor/image metadata "
            "is not imputed."
        ),
    }


def compare_visual_adaptation(
    before_rows: Sequence[Mapping[str, Any]],
    after_rows: Sequence[Mapping[str, Any]],
    thresholds: BBoxAreaThresholdConfig,
) -> dict[str, Any]:
    """Compare paired Evaluation v1.5 detection outputs before and after H1."""

    before_by_id = {str(row.get("id")): row for row in before_rows}
    after_by_id = {str(row.get("id")): row for row in after_rows}
    paired_ids = sorted(set(before_by_id).intersection(after_by_id))
    paired_before = [before_by_id[sample_id] for sample_id in paired_ids]
    paired_after = [after_by_id[sample_id] for sample_id in paired_ids]
    before = analyze_detection_visual_factors(paired_before, thresholds)
    after = analyze_detection_visual_factors(paired_after, thresholds)
    deltas: dict[str, Any] = {}
    for bucket in ("small", "medium", "large"):
        before_iou = before["by_bbox_size"][bucket]["mean_iou"]
        after_iou = after["by_bbox_size"][bucket]["mean_iou"]
        deltas[bucket] = {
            "mean_iou_delta": (
                float(after_iou) - float(before_iou)
                if before_iou is not None and after_iou is not None
                else None
            )
        }
    return {
        "schema_version": "1.0",
        "evaluation_contract": "Evaluation v1.5 evaluated_predictions",
        "paired_sample_count": len(paired_ids),
        "before": before,
        "after": after,
        "bbox_size_deltas": deltas,
    }
