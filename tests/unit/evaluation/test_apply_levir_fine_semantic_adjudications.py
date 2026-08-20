from __future__ import annotations

import pytest
from scripts.evaluation.apply_levir_fine_semantic_adjudications import apply_decisions


def test_apply_decisions_covers_all_rows_and_validates_final_schema() -> None:
    rows = [
        {
            "audit_id": "one",
            "caption": "A building was built.",
            "human_change_label": "1",
        }
    ]
    decisions = {
        "one": {
            "adjudicated_objects": "building",
            "adjudicated_directions": "appearance_construction",
            "adjudicated_confidence": "high",
            "adjudication_note": "explicit construction",
        }
    }

    completed = apply_decisions(rows, decisions)

    assert completed[0]["adjudicated_objects"] == "building"


def test_apply_decisions_rejects_incomplete_or_invalid_values() -> None:
    rows = [{"audit_id": "one", "caption": "Changed.", "human_change_label": "1"}]
    with pytest.raises(ValueError, match="cover every row"):
        apply_decisions(rows, {})
    with pytest.raises(ValueError, match="requires object"):
        apply_decisions(
            rows,
            {
                "one": {
                    "adjudicated_objects": "none",
                    "adjudicated_directions": "appearance_construction",
                    "adjudicated_confidence": "low",
                    "adjudication_note": "",
                }
            },
        )
