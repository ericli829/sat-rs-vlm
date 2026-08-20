from __future__ import annotations

from scripts.evaluation.finalize_levir_fine_semantic_gold import build_gold_rows


def test_finalizer_uses_adjudication_for_disagreement_and_conservative_confidence() -> None:
    annotator_a = {
        "agreed": {
            "caption": "No change.",
            "human_change_label": "0",
            "human_changed_objects": "none",
            "human_change_directions": "none",
            "human_annotation_confidence": "high",
        },
        "different": {
            "caption": "A building appeared.",
            "human_change_label": "1",
            "human_changed_objects": "building",
            "human_change_directions": "appearance_construction",
            "human_annotation_confidence": "high",
        },
    }
    annotator_b = {
        "agreed": {**annotator_a["agreed"], "human_annotation_confidence": "medium"},
        "different": {
            **annotator_a["different"],
            "human_changed_objects": "building|road",
        },
    }
    adjudications = {
        "different": {
            "adjudicated_objects": "building",
            "adjudicated_directions": "appearance_construction",
            "adjudicated_confidence": "high",
            "adjudication_note": "road was not explicit",
        }
    }

    rows, counts = build_gold_rows(annotator_a, annotator_b, adjudications)

    assert counts == {"num_agreement_rows": 1, "num_adjudicated_rows": 1}
    assert rows[0]["human_annotation_confidence"] == "medium"
    assert rows[1]["label_source"] == "adjudicated"
