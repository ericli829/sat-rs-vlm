from sat_rs_vlm.evaluation.reliability.metrics import (
    build_prediction_pairs,
    summarize_reliability,
)


def test_metrics_are_reported_overall_and_by_task() -> None:
    clean = [
        {"id": "v", "task_type": "vqa", "prediction": "yes", "reference": "yes"},
        {"id": "c", "task_type": "counting", "prediction": "2", "reference": "2"},
    ]
    fault = [
        {"id": "v", "task_type": "vqa", "prediction": "no", "reference": "yes"},
        {"id": "c", "task_type": "counting", "prediction": "4", "reference": "2"},
    ]
    recovered = clean

    pairs = build_prediction_pairs(clean, fault, recovered)
    summary = summarize_reliability(
        pairs,
        execution_mode="smoke_mock",
        experiment_name="test",
        run_id="run",
    )

    assert summary["schema_version"] == "1.0"
    assert summary["execution_mode"] == "smoke_mock"
    assert summary["overall"]["changed_rate"] == 1.0
    assert summary["overall"]["fault_exact_match"] == 0.0
    assert summary["overall"]["recovery_success_rate"] == 1.0
    assert summary["by_task"]["counting"]["counting_mae"] == 2.0
