from __future__ import annotations

from scripts.evaluation.prepare_levir_fine_semantic_adjudication import build_adjudication_rows


def test_adjudication_ignores_multi_label_order_and_keeps_only_changed_disagreements() -> None:
    annotator_a = {
        "same": {
            "caption": "A building and road appeared.",
            "human_change_label": "1",
            "human_changed_objects": "building|road",
            "human_change_directions": "appearance_construction",
        },
        "different": {
            "caption": "A building changed.",
            "human_change_label": "1",
            "human_changed_objects": "building",
            "human_change_directions": "state_change_unspecified",
        },
        "no_change": {
            "caption": "No change.",
            "human_change_label": "0",
            "human_changed_objects": "none",
            "human_change_directions": "none",
        },
    }
    annotator_b = {
        "same": {
            "caption": "A building and road appeared.",
            "human_change_label": "1",
            "human_changed_objects": "road|building",
            "human_change_directions": "appearance_construction",
        },
        "different": {
            "caption": "A building changed.",
            "human_change_label": "1",
            "human_changed_objects": "building|road",
            "human_change_directions": "appearance_construction",
        },
        "no_change": {
            "caption": "No change.",
            "human_change_label": "0",
            "human_changed_objects": "none",
            "human_change_directions": "none",
        },
    }

    rows, summary = build_adjudication_rows(annotator_a, annotator_b)

    assert [row["audit_id"] for row in rows] == ["different"]
    assert summary["num_changed_rows"] == 2
    assert summary["num_object_exact_agreements"] == 1
    assert summary["num_direction_exact_agreements"] == 1
