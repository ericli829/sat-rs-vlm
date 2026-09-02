from counting_system.eval.export_v15 import export_predictions_v15, row_to_v15_prediction
from counting_system.tiling import resolve_overlap


def test_resolve_overlap_ratio():
    assert resolve_overlap(1333, {"overlap_ratio": 0.15}) == 200
    assert resolve_overlap(1333, {"overlap": 180, "overlap_ratio": 0.15}) == 180


def test_row_to_v15_prediction():
    row = {
        "sample_id": "Counting_Overall_counting_0",
        "question": "How many ships are there in the entire picture?",
        "pred": 12,
        "ref": 153,
        "latency_sec": 1.25,
        "detector": "grounding_dino",
        "category": "Counting/Overall counting",
        "provenance": {"tiles_run": 16, "tiles_planned": 16},
    }
    out = row_to_v15_prediction(row)
    assert out["id"] == row["sample_id"]
    assert out["task_type"] == "counting"
    assert out["prediction"] == "12"
    assert out["reference"] == "153"
    assert out["metadata"]["metrics_protocol"].startswith("formal_e2_parse")
    assert out["metadata"]["upstream_branch"] == "feature/vlm-semantic-alignment"
    assert out["inference_latency_ms"] == 1250.0
    assert out["telemetry"]["vision_input"]["tile_count"] == 16


def test_export_predictions_v15_batch():
    rows = export_predictions_v15([{"sample_id": "a", "question": "How many cars?", "pred": 1, "ref": 2}])
    assert len(rows) == 1
    assert rows[0]["question"] == "How many cars?"
