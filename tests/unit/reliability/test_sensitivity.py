from __future__ import annotations

from sat_rs_vlm.models.reliability.sensitivity import aggregate_sensitivity_conditions


def test_aggregate_groups_by_target_layer_plane_and_intensity() -> None:
    conditions = [
        {
            "id": "a",
            "target": "visual_blocks",
            "layers": [2],
            "bit_plane": "exponent",
            "num_bits": 1,
            "comparison": {
                "overall": {"prediction_changed_rate": 0.4},
                "by_task": {"vqa": {"metrics": {"normalized_accuracy": {"status": "ok", "improvement_mean": -0.2}}}},
            },
        },
        {
            "id": "b",
            "target": "visual_blocks",
            "layers": [2],
            "bit_plane": "exponent",
            "num_bits": 1,
            "comparison": {"overall": {"prediction_changed_rate": 0.6}},
        },
    ]
    report = aggregate_sensitivity_conditions(conditions)
    assert report["schema_version"] == "1.0"
    group = report["groups"][0]
    assert group["target"] == "visual_blocks"
    assert group["layers"] == [2]
    assert group["changed_rate_mean"] == 0.5
    assert group["repeats"] == 2
    assert group["task_degradations"][0]["degradation_mean"] == 0.2
