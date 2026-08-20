from __future__ import annotations

from pathlib import Path

from scripts.evaluation.evaluate_local_judge_semantic_slices import _read_rows, build_slice_report
from scripts.evaluation.prepare_local_judge_sft_dataset import split_supervised_rows

ROOT = Path(__file__).resolve().parents[3]


def test_sft_split_is_stratified_and_excludes_single_uncertain_example() -> None:
    rows = [
        {
            "audit_id": f"zero-{index}",
            "caption": f"no change {index}",
            "human_change_label": "0",
            "human_changed_objects": "none",
            "human_change_directions": "none",
            "human_annotation_confidence": "high",
            "label_source": "annotator_agreement",
        }
        for index in range(5)
    ] + [
        {
            "audit_id": f"one-{index}",
            "caption": f"building appeared {index}",
            "human_change_label": "1",
            "human_changed_objects": "building",
            "human_change_directions": "appearance_construction",
            "human_annotation_confidence": "high",
            "label_source": "annotator_agreement",
        }
        for index in range(5)
    ] + [
        {
            "audit_id": "uncertain",
            "caption": "Maybe changed.",
            "human_change_label": "U",
            "human_changed_objects": "unknown",
            "human_change_directions": "unknown",
            "human_annotation_confidence": "low",
            "label_source": "annotator_agreement",
        }
    ]

    train, validation, uncertain = split_supervised_rows(rows, validation_ratio=0.2, seed=7)

    assert {row["target_label"] for row in validation} == {"0", "1"}
    assert len(train) == 8
    assert len(validation) == 2
    assert len(uncertain) == 1


def test_semantic_slices_report_change_recall_and_hard_cases() -> None:
    gold = [
        {
            "audit_id": "zero",
            "caption": "No change.",
            "human_change_label": "0",
            "human_changed_objects": "none",
            "human_change_directions": "none",
        },
        {
            "audit_id": "one",
            "caption": "A road appeared.",
            "human_change_label": "1",
            "human_changed_objects": "road",
            "human_change_directions": "appearance_construction",
        },
    ]
    judged = [
        {"audit_id": "zero", "local_judge_decision": "0"},
        {"audit_id": "one", "local_judge_decision": "0"},
    ]

    report, hard_cases = build_slice_report(gold, judged)

    assert report["overall"]["fn"] == 1
    assert report["by_changed_object"]["road"]["change_recall"] == 0.0
    assert hard_cases[0]["error_type"] == "false_negative"


def test_slice_reader_accepts_answer_key_json() -> None:
    path = ROOT / "tests" / "fixtures" / "evaluation" / "local_judge_answer_key.json"

    assert _read_rows(path) == [{"audit_id": "one", "local_judge_decision": "1"}]
