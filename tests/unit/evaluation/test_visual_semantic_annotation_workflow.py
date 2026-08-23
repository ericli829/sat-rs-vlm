from __future__ import annotations

import pytest
from scripts.evaluation.finalize_levir_visual_semantic_gold import build_gold_rows
from scripts.evaluation.prepare_levir_visual_semantic_adjudication import (
    build_adjudication_rows,
)
from scripts.evaluation.prepare_levir_visual_semantic_audit import build_annotation_rows
from scripts.evaluation.validate_levir_visual_semantic_annotations import validate_rows

CONTRACT = {
    "labels": {
        "objects": ["building", "road", "none", "unknown"],
        "directions": [
            "appearance_construction",
            "disappearance_demolition",
            "none",
            "unknown",
        ],
    }
}


def _changed_row(audit_id: str = "visual_0001") -> dict[str, str]:
    return {
        "audit_id": audit_id,
        "sample_id": "sample-1",
        "image_t1_path": "before.png",
        "image_t2_path": "after.png",
        "gold_change_label": "1",
        "gold_changed_objects": "building",
        "gold_change_directions": "appearance_construction",
        "gold_change_events": "building:appearance_construction",
        "annotation_confidence": "high",
        "annotation_note": "",
    }


def test_prepared_visual_sheet_is_blinded_and_has_empty_labels() -> None:
    prepared = build_annotation_rows(
        [
            {
                "sample_id": "sample-1",
                "image_t1_path": "before.png",
                "image_t2_path": "after.png",
            }
        ],
        sample_size=None,
        seed=1,
    )

    assert prepared[0]["audit_id"] == "visual_0001"
    assert prepared[0]["gold_change_label"] == ""
    assert "prediction" not in prepared[0]
    assert "changeflag" not in prepared[0]


def test_visual_validator_requires_explicit_and_consistent_events() -> None:
    normalized = validate_rows([_changed_row()], CONTRACT)

    assert normalized[0]["gold_change_events"] == ["building:appearance_construction"]
    invalid = _changed_row()
    invalid["gold_change_events"] = "road:appearance_construction"
    with pytest.raises(ValueError, match="event object/direction"):
        validate_rows([invalid], CONTRACT)


def test_adjudication_sheet_contains_only_disagreements() -> None:
    left = validate_rows([_changed_row()], CONTRACT)[0]
    right_source = _changed_row()
    right_source["gold_change_directions"] = "disappearance_demolition"
    right_source["gold_change_events"] = "building:disappearance_demolition"
    right = validate_rows([right_source], CONTRACT)[0]

    disagreements = build_adjudication_rows({"visual_0001": left}, {"visual_0001": right})

    assert len(disagreements) == 1
    assert disagreements[0]["annotator_a_events"] == "building:appearance_construction"
    assert disagreements[0]["annotator_b_events"] == "building:disappearance_demolition"


def test_finalizer_requires_adjudication_and_uses_adjudicated_event() -> None:
    left = validate_rows([_changed_row()], CONTRACT)[0]
    right_source = _changed_row()
    right_source["gold_change_directions"] = "disappearance_demolition"
    right_source["gold_change_events"] = "building:disappearance_demolition"
    right = validate_rows([right_source], CONTRACT)[0]
    adjudication = {
        "visual_0001": {
            **_changed_row(),
            "adjudication_note": "building was added after inspecting both images",
        }
    }

    with pytest.raises(ValueError, match="missing adjudication"):
        build_gold_rows({"visual_0001": left}, {"visual_0001": right}, {})
    gold, counts = build_gold_rows({"visual_0001": left}, {"visual_0001": right}, adjudication)

    assert counts == {"num_agreement_rows": 0, "num_adjudicated_rows": 1}
    assert gold[0]["gold_change_events"] == "building:appearance_construction"
    assert gold[0]["label_source"] == "third_annotator_adjudication"


def test_finalizer_does_not_require_adjudication_when_all_rows_agree() -> None:
    agreed = validate_rows([_changed_row()], CONTRACT)[0]

    gold, counts = build_gold_rows({"visual_0001": agreed}, {"visual_0001": agreed}, {})

    assert counts == {"num_agreement_rows": 1, "num_adjudicated_rows": 0}
    assert gold[0]["label_source"] == "dual_annotator_agreement"
