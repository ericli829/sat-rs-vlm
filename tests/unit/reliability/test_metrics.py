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


def test_change_detection_reports_caption_drop_and_dataset_summary() -> None:
    clean = [
        {
            "id": "change-1",
            "task_type": "change_detection",
            "prediction": "a building was added",
            "reference": "a building was added",
            "metadata": {"reliability_source": "LEVIR-CC"},
        }
    ]
    fault = [
        {
            **clean[0],
            "prediction": "no change",
        }
    ]

    pairs = build_prediction_pairs(clean, fault)
    summary = summarize_reliability(
        pairs,
        execution_mode="real_inference",
        experiment_name="test",
        run_id="run",
    )

    change_metrics = summary["by_task"]["change_detection"]
    assert change_metrics["clean_bleu_1"] == 1.0
    assert change_metrics["fault_bleu_1"] == 0.0
    assert change_metrics["bleu_1_drop"] == 1.0
    assert summary["by_dataset"]["LEVIR-CC"]["num_samples"] == 1
