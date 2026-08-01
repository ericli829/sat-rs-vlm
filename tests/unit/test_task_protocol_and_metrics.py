from __future__ import annotations

import json

import pytest

from sat_rs_vlm.data.prompt_templates import strengthen_answer
from sat_rs_vlm.data.task_protocol import (
    BBoxFormat,
    counting_json,
    normalize_bbox,
    parse_count,
)
from sat_rs_vlm.evaluation.metrics import score_detection, summarize_predictions


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2", 2),
        ("2 vehicles", 2),
        ("There are 2 vehicles", 2),
        ("Two", 2),
        ("two ships", 2),
        ("No vehicles", 0),
        ("none", 0),
        ('{"count":2}', 2),
        ("twenty-one buildings", 21),
    ],
)
def test_counting_normalization(text: str, expected: int) -> None:
    assert parse_count(text).value == expected
    assert counting_json(text) == json.dumps({"count": expected}, separators=(",", ":"))


def test_counting_unresolved_is_not_fabricated() -> None:
    assert parse_count("several buildings").value is None
    assert parse_count("one or two ships").reason == "ambiguous_multiple_counts"
    assert strengthen_answer("counting", "several buildings") == "several buildings"


def test_bbox_formats_are_explicit() -> None:
    percent, _ = normalize_bbox(
        [10, 20, 80, 90],
        source_format=BBoxFormat.PERCENT_0_100,
    )
    pixels, _ = normalize_bbox(
        [10, 20, 80, 90],
        source_format=BBoxFormat.PIXEL_XYXY,
        image_size=(100, 100),
    )
    assert percent == pixels == [0.1, 0.2, 0.8, 0.9]
    with pytest.raises(ValueError, match="image_size"):
        normalize_bbox([10, 20, 80, 90], source_format=BBoxFormat.PIXEL_XYXY)


def test_detection_metrics_separate_json_range_label_and_iou() -> None:
    reference = '{"label":"ship","bbox":[0.1,0.2,0.4,0.5]}'
    invalid_range = '{"label":"ship","bbox":[100,200,400,500]}'
    score = score_detection(invalid_range, reference)
    assert score["valid_json"] is True
    assert score["valid_coordinate_range"] is False
    assert score["iou"] is None

    summary = summarize_predictions(
        [
            {
                "task_type": "detection",
                "prediction": reference,
                "reference": reference,
            }
        ]
    )
    metrics = summary["by_task"]["detection"]
    assert metrics["valid_json_rate"] == 1.0
    assert metrics["valid_coordinate_range"] == 1.0
    assert metrics["mean_iou"] == 1.0
    assert metrics["detection_exact_at_0_5"] == 1.0
