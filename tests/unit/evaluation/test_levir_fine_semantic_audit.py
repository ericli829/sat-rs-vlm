from __future__ import annotations

import pytest
from scripts.evaluation.prepare_levir_fine_semantic_audit import build_annotation_rows
from scripts.evaluation.validate_levir_fine_semantic_annotations import validate_rows

SCHEMA = {
    "fields": {
        "human_changed_objects": {"values": ["building", "road", "none", "unknown"]},
        "human_change_directions": {
            "values": ["appearance_construction", "none", "unknown"]
        },
        "human_annotation_confidence": {"values": ["high", "medium", "low"]},
    }
}


def test_preparation_locks_existing_binary_gold_and_only_leaves_change_rows_open() -> None:
    rows = build_annotation_rows(
        [
            {"audit_id": "no", "caption": "No change.", "human_gold_label": "0"},
            {"audit_id": "yes", "caption": "A house was built.", "human_gold_label": "1"},
            {"audit_id": "unclear", "caption": "Maybe changed.", "human_gold_label": "U"},
        ]
    )
    assert rows[0]["human_changed_objects"] == "none"
    assert rows[0]["human_change_directions"] == "none"
    assert rows[1]["human_changed_objects"] == ""
    assert rows[2]["human_changed_objects"] == "unknown"


def test_validator_accepts_consistent_fine_semantic_rows() -> None:
    result = validate_rows(
        [
            {
                "audit_id": "no",
                "caption": "No change.",
                "human_change_label": "0",
                "human_changed_objects": "none",
                "human_change_directions": "none",
                "human_annotation_confidence": "high",
            },
            {
                "audit_id": "yes",
                "caption": "A building and road appeared.",
                "human_change_label": "1",
                "human_changed_objects": "building|road",
                "human_change_directions": "appearance_construction",
                "human_annotation_confidence": "high",
            },
        ],
        SCHEMA,
    )
    assert result[1]["human_changed_objects"] == ["building", "road"]


def test_validator_rejects_binary_semantic_contradictions() -> None:
    with pytest.raises(ValueError, match="label 0 requires"):
        validate_rows(
            [
                {
                    "audit_id": "bad",
                    "human_change_label": "0",
                    "human_changed_objects": "building",
                    "human_change_directions": "appearance_construction",
                    "human_annotation_confidence": "high",
                }
            ],
            SCHEMA,
        )
