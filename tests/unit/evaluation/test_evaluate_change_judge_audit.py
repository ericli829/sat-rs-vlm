from __future__ import annotations

import pytest
from scripts.evaluation.evaluate_change_judge_audit import (
    evaluate_audit,
    evaluate_single_annotator,
)


def test_audit_summary_uses_human_caption_labels_not_image_truth() -> None:
    annotator_a = {
        "a": {"caption": "No change.", "human_caption_semantic_label": "0"},
        "b": {"caption": "A house was built.", "human_caption_semantic_label": "1"},
        "c": {"caption": "Unclear.", "human_caption_semantic_label": "U"},
    }
    annotator_b = {
        "a": {"caption": "No change.", "human_caption_semantic_label": "0"},
        "b": {"caption": "A house was built.", "human_caption_semantic_label": "1"},
        "c": {"caption": "Unclear.", "human_caption_semantic_label": "1"},
    }
    answer_key = {
        "a": {"old_parser_decision": 0, "local_judge_decision": 0},
        "b": {"old_parser_decision": 1, "local_judge_decision": 1},
        "c": {"old_parser_decision": 1, "local_judge_decision": None},
    }

    summary, disagreements = evaluate_audit(annotator_a, annotator_b, answer_key)

    assert summary["raw_agreement"] == pytest.approx(2 / 3)
    assert summary["num_unadjudicated_disagreements"] == 1
    assert summary["old_contextual_parser"]["accuracy_on_resolved"] == 1.0
    assert summary["local_small_llm_judge"]["accuracy_on_resolved"] == 1.0
    assert len(disagreements) == 1


def test_audit_rejects_mismatched_id_sets() -> None:
    with pytest.raises(ValueError, match="must match"):
        evaluate_audit(
            {"a": {"human_caption_semantic_label": "0"}},
            {"b": {"human_caption_semantic_label": "0"}},
            {"a": {"old_parser_decision": 0}},
        )


def test_single_annotator_report_is_explicitly_preliminary() -> None:
    annotator = {
        "a": {"human_caption_semantic_label": "0"},
        "b": {"human_caption_semantic_label": "1"},
    }
    answer_key = {
        "a": {"old_parser_decision": 1, "local_judge_decision": 0},
        "b": {"old_parser_decision": 1, "local_judge_decision": 1},
    }

    summary, disagreements = evaluate_single_annotator(annotator, answer_key)

    assert summary["audit_mode"] == "single_annotator_preliminary"
    assert summary["cohen_kappa"] is None
    assert summary["old_contextual_parser"]["accuracy_on_resolved"] == 0.5
    assert summary["local_small_llm_judge"]["accuracy_on_resolved"] == 1.0
    assert summary["paired_judge_comparison"]["local_only_correct"] == 1
    assert disagreements == []
