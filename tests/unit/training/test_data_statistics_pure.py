from __future__ import annotations

import pytest

from sat_rs_vlm.training.config import BBoxAreaThresholdConfig
from sat_rs_vlm.training.data_statistics import (
    bbox_area_bucket,
    numeric_summary,
    percentile,
    stratified_sample_by_task,
    task_counts,
)


def test_numeric_statistics_do_not_require_torch() -> None:
    assert percentile([1, 2, 3, 4], 0.90) == pytest.approx(3.7)
    assert numeric_summary([1, 2, 3, 4]) == {
        "sample_count": 4,
        "mean": 2.5,
        "median": 2.5,
        "p90": pytest.approx(3.7),
        "p95": pytest.approx(3.85),
        "max": 4.0,
    }
    assert numeric_summary([])["mean"] is None


def test_task_aggregation_and_sampling_are_deterministic() -> None:
    samples = [
        {"id": "vqa-1", "task_type": "vqa"},
        {"id": "vqa-2", "task_type": "vqa"},
        {"id": "caption-1", "task_type": "captioning"},
        {"id": "caption-2", "task_type": "captioning"},
    ]

    assert task_counts(samples) == {"vqa": 2, "captioning": 2}
    first = stratified_sample_by_task(samples, 1, seed=42)
    second = stratified_sample_by_task(samples, 1, seed=42)
    assert [sample["id"] for sample in first] == [sample["id"] for sample in second]
    assert {sample["task_type"] for sample in first} == {"vqa", "captioning"}


def test_bbox_area_buckets_use_explicit_configuration() -> None:
    thresholds = BBoxAreaThresholdConfig(small_max=0.02, medium_max=0.10)

    assert bbox_area_bucket(0.01, thresholds) == "small"
    assert bbox_area_bucket(0.05, thresholds) == "medium"
    assert bbox_area_bucket(0.20, thresholds) == "large"
