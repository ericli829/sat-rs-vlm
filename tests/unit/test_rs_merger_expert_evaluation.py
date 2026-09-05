from sat_rs_vlm.evaluation.rs_merger_expert import (
    render_summary,
    summarize_counting_focused_predictions,
    summarize_counting_predictions,
    update_experiment_matrix,
)


def test_count_metrics_include_dense_bin_rmse_and_bias():
    rows = [
        {
            "task_type": "counting",
            "parsed_reference": 6,
            "parsed_prediction": 8,
            "count_bin": "6-10",
        },
        {
            "task_type": "counting",
            "parsed_reference": 10,
            "parsed_prediction": 9,
            "count_bin": "6-10",
        },
        {
            "task_type": "counting",
            "parsed_reference": 2,
            "parsed_prediction": None,
            "count_bin": "0-2",
        },
    ]
    result = summarize_counting_predictions(rows)
    assert result["overall"]["n"] == 3
    assert result["overall"]["parse_rate"] == 2 / 3
    dense = result["count_bins"]["6-10"]
    assert dense["n"] == 2
    assert dense["mae"] == 1.5
    assert dense["bias"] == 0.5
    assert dense["within_1"] == 0.5


def test_experiment_matrix_is_machine_backed(tmp_path):
    metrics = {
        "overall": {"exact": 0.5, "within_1": 0.75, "mae": 1.0, "bias": -0.25},
        "count_bins": {"6-10": {"exact": 0.4, "within_1": 0.6, "mae": 1.5}},
    }
    matrix = tmp_path / "experiment_matrix.md"
    update_experiment_matrix(
        matrix,
        experiment="C2",
        architecture="detail",
        training_summary={
            "trainable_params": 24,
            "peak_allocated_vram_gb": 10.0,
            "elapsed_seconds": 20.0,
        },
        metrics=metrics,
    )
    assert "| C2 | detail | 24 |" in matrix.read_text(encoding="utf-8")
    assert matrix.with_suffix(".json").is_file()


def test_counting_focused_summary_keeps_guard_metrics_and_baseline_delta():
    rows = [
        {
            "task_type": "counting",
            "parsed_reference": 4,
            "parsed_prediction": 4,
            "count_bin": "3-5",
            "prediction": "4",
            "reference": "4",
        },
        {"task_type": "scene", "prediction": "urban", "reference": "urban"},
    ]
    baseline = summarize_counting_focused_predictions(rows)
    result = summarize_counting_focused_predictions(
        [dict(row, parsed_prediction=3) if row["task_type"] == "counting" else row for row in rows],
        baseline,
    )
    assert result["counting_focused"]["overall"]["within_1"] == 1.0
    assert result["non_counting_e1_guard"]["by_task"]["scene"]["normalized_exact_match"] == 1.0
    assert "baseline_delta" in result
    summary = render_summary(result, baseline)
    assert "Counting-focused results" in summary
    assert "Non-counting E1 guard results" in summary
    assert "Delta versus R1/C0 baseline" in summary
