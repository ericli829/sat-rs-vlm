from __future__ import annotations

import json

import pytest

from sat_rs_vlm.data.prompt_templates import strengthen_answer
from sat_rs_vlm.data.task_protocol import (
    BBoxFormat,
    counting_json,
    normalize_bbox,
    parse_count,
    parse_detection,
)
from sat_rs_vlm.evaluation.metrics import score_detection, score_sample, summarize_predictions


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

    parsed_percent = parse_detection(
        '{"label":"ship","bbox":[10,20,80,90]}',
        coordinate_format=BBoxFormat.PERCENT_0_100,
    )
    parsed_pixel = parse_detection(
        '{"label":"ship","bbox":[10,20,80,90]}',
        coordinate_format=BBoxFormat.PIXEL_XYXY,
        image_size=(100, 100),
    )
    assert parsed_percent is not None and parsed_percent.valid_coordinate_range
    assert parsed_pixel is not None and parsed_pixel.bbox == parsed_percent.bbox


def test_detection_parser_accepts_only_single_target_legacy_schema() -> None:
    legacy = parse_detection('{"boxes":[[0.1,0.2,0.4,0.5]],"labels":["ship"]}')
    assert legacy is not None
    assert legacy.label == "ship"
    assert legacy.bbox == (0.1, 0.2, 0.4, 0.5)
    assert (
        parse_detection('{"boxes":[[0.1,0.2,0.4,0.5],[0.2,0.3,0.5,0.6]],"labels":["ship","boat"]}')
        is None
    )


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


def test_change_detection_uses_caption_metrics() -> None:
    score = score_sample(
        "change_detection",
        "two buildings were added",
        "two buildings were added near the road",
    )

    assert score["bleu_1"] > 0
    assert score["rouge_l"] > 0
    assert "exact_match" not in score
