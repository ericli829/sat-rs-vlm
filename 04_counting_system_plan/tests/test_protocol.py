from counting_system.eval.metrics import summarize_counts
from counting_system.eval.protocol import (
    build_benchmark_report,
    build_protocol_manifest,
    inventory_system_models,
    summarize_by_category,
)


def test_summarize_by_category():
    rows = [
        {
            "category": "Counting/Overall counting",
            "l2_category": "default",
            "pred": 3,
            "ref": 3,
            "choice_match": True,
        },
        {
            "category": "Counting/Regional counting",
            "l2_category": "default",
            "pred": 4,
            "ref": 3,
            "choice_match": False,
        },
    ]
    by_cat = summarize_by_category(rows)
    assert "overall_counting" in by_cat
    assert by_cat["overall_counting"]["exact_match"] == 1.0
    assert by_cat["regional_counting"]["exact_match"] == 0.0


def test_inventory_within_32b():
    inv = inventory_system_models(backends_used=["grounding_dino"])
    assert inv["within_32b_limit"] is True
    assert inv["total_system_params_b"] < 1.0
    assert inv["offline_only"] is True


def test_build_benchmark_report_smoke():
    rows = [
        {
            "category": "Counting/Overall counting",
            "l2_category": "default",
            "pred": 2,
            "ref": 2,
            "choice_match": True,
            "latency_sec": 1.5,
            "options": ["2", "3"],
            "answer_letter": "A",
        }
    ]
    metrics = summarize_counts([(2, 2)])
    metrics["elapsed_sec"] = 1.5
    report = build_benchmark_report(
        rows=rows,
        pairs=[(2, 2)],
        metrics=metrics,
        backends_used=["grounding_dino"],
        cold_start_sec=0.8,
    )
    assert report["protocol"]["primary_metrics"] == ["exact_match", "rmse"]
    assert report["metrics"]["exact_match"] == 1.0
    assert report["resources"]["within_32b_limit"] is True
    assert report["timing"]["cold_start_sec"] == 0.8


def test_protocol_manifest_discloses_pipeline():
    manifest = build_protocol_manifest(language="en", official_aligned=False)
    assert manifest["language"] == "en"
    assert manifest["official_aligned"] is False
    assert manifest["upstream_branch"] == "feature/vlm-semantic-alignment"
    assert manifest["dedup_policy"] == "global_coordinate_core_ownership_nms"
    assert "gpt_judge" in manifest["excluded_primary_metrics"]
    assert manifest["input_pipeline"]["native_tile_size"] == 1333
