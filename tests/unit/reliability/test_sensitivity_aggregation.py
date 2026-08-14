from sat_rs_vlm.models.reliability.sensitivity import aggregate_sensitivity_conditions


def _condition(repeat: int, degradation: float) -> dict[str, object]:
    return {
        "id": f"condition-{repeat}",
        "target": "visual_blocks",
        "layers": [26],
        "bit_plane": "exponent",
        "num_bits": 10,
        "comparison": {
            "overall": {"prediction_changed_rate": 0.2 + repeat * 0.1},
            "by_task": {
                "detection": {
                    "metrics": {
                        "iou": {
                            "status": "ok",
                            "improvement_mean": -degradation,
                            "improvement_ci95_paired_bootstrap": [
                                -degradation - 0.01,
                                -degradation + 0.01,
                            ],
                            "num_samples": 50,
                        }
                    }
                }
            },
        },
        "injection": {"evaluation": {"invalid_rate": 0.01}},
    }


def test_conditions_aggregate_by_fault_surface_across_repeats() -> None:
    report = aggregate_sensitivity_conditions([_condition(0, 0.1), _condition(1, 0.2)])
    assert report["schema_version"] == "1.0"
    group = report["groups"][0]
    assert group["repeats"] == 2
    assert group["layers"] == [26]
    assert group["changed_rate_mean"] == 0.25
    degradation = group["task_degradations"][0]
    assert degradation["task"] == "detection"
    assert degradation["metric"] == "iou"
    assert degradation["degradation_mean"] == 0.15000000000000002
