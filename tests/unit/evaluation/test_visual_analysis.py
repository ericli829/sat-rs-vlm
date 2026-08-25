from __future__ import annotations

from sat_rs_vlm.evaluation.visual_analysis import compare_visual_adaptation
from sat_rs_vlm.training.config import BBoxAreaThresholdConfig


def _row(sample_id: str, bbox: list[float], iou: float) -> dict[str, object]:
    return {
        "id": sample_id,
        "task_type": "detection",
        "reference": f'{{"label":"car","bbox":{bbox}}}',
        "parse_ok": True,
        "sample_metrics": {
            "parse_success": True,
            "label_match": True,
            "iou": iou,
            "generalized_iou": iou - 0.1,
            "normalized_center_distance": 1 - iou,
        },
        "metadata": {
            "approximate_visual_tokens": 256,
            "image_width": 1024,
            "image_height": 1024,
        },
    }


def test_small_object_paired_delta_and_correlations() -> None:
    before = [_row("small", [0.1, 0.1, 0.15, 0.15], 0.2)]
    after = [_row("small", [0.1, 0.1, 0.15, 0.15], 0.5)]

    report = compare_visual_adaptation(
        before,
        after,
        BBoxAreaThresholdConfig(small_max=0.01, medium_max=0.1),
    )

    assert report["paired_sample_count"] == 1
    assert report["before"]["by_bbox_size"]["small"]["mean_iou"] == 0.2
    assert report["bbox_size_deltas"]["small"]["mean_iou_delta"] == 0.3
    assert report["after"]["by_bbox_size"]["medium"]["mean_iou"] is None
