"""Regression tests for reusing the established LEVIR binary caption judge."""

from __future__ import annotations

from scripts.evaluation.evaluate_levir_visual_semantics import evaluate


def test_stored_binary_judge_overrides_caption_keyword_only_for_binary_metric() -> None:
    gold_rows = {
        "sample": {
            "sample_id": "sample",
            "image_t1_path": "before.png",
            "image_t2_path": "after.png",
            "gold_change_label": "0",
            "gold_changed_objects": "none",
            "gold_change_directions": "none",
            "gold_change_events": "",
            "annotation_confidence": "high",
            "label_source": "dual_annotator_agreement",
        }
    }
    predictions = {
        "sample": {
            "id": "sample",
            "prediction": "A new building appears near the road.",
            "prediction_changeflag": 0,
            "binary_prediction_source": "local_llm_judge",
        }
    }

    scored, audit_only = evaluate(gold_rows, predictions)

    assert audit_only == []
    assert scored[0]["caption_semantics"]["change_label"] == 1
    assert scored[0]["prediction"]["change_label"] == 0
    assert scored[0]["binary_decision"] == {
        "value": 0,
        "source": "local_llm_judge",
        "used_stored_judge": True,
        "status": "resolved",
    }
    assert scored[0]["sample_metrics"]["binary_correct"] is True
    assert scored[0]["prediction"]["events"] == [
        {"object": "building", "direction": "appearance_construction"}
    ]


def test_missing_binary_judge_is_unresolved_not_caption_keyword_fallback() -> None:
    gold_rows = {
        "sample": {
            "sample_id": "sample",
            "image_t1_path": "before.png",
            "image_t2_path": "after.png",
            "gold_change_label": "1",
            "gold_changed_objects": "building",
            "gold_change_directions": "appearance_construction",
            "gold_change_events": "building:appearance_construction",
            "annotation_confidence": "high",
            "label_source": "dual_annotator_agreement",
        }
    }
    predictions = {"sample": {"id": "sample", "prediction": "A new building appears."}}

    scored, audit_only = evaluate(gold_rows, predictions)

    assert audit_only == []
    assert scored[0]["caption_semantics"]["change_label"] == 1
    assert scored[0]["prediction"]["change_label"] is None
    assert scored[0]["binary_decision"] == {
        "value": None,
        "source": "unresolved_missing_binary_judge",
        "used_stored_judge": False,
        "status": "unresolved",
    }
    assert scored[0]["sample_metrics"]["binary_parse_success"] is False
    assert scored[0]["prediction"]["events"] == [
        {"object": "building", "direction": "appearance_construction"}
    ]
