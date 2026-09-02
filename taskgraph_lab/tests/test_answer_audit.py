from __future__ import annotations

from pathlib import Path

from taskgraph_lab.quality.answer_audit import audit_choice_answer, load_answer_index

FIXTURES = Path(__file__).parent / "fixtures"


def test_answer_cardinality_audit_is_separate_from_runtime_validation() -> None:
    multi = audit_choice_answer(
        answer="ABD",
        choices=["(A) lake", "(B) farm", "(C) mall", "(D) residential"],
        final_answer_type="CHOICE_MULTI",
    )
    assert multi["valid"] is True
    assert multi["selected_choice_labels"] == ["A", "B", "D"]

    mismatch = audit_choice_answer(
        answer="ABD",
        choices=["(A) lake", "(B) farm", "(C) mall", "(D) residential"],
        final_answer_type="CHOICE_SINGLE",
    )
    assert mismatch["valid"] is False
    assert mismatch["expected_answer_type"] == "CHOICE_MULTI"


def test_single_choice_answer_audit() -> None:
    for answer_type in ("CHOICE_SINGLE", "CHOICE_MULTI"):
        result = audit_choice_answer(
            answer="B",
            choices=["(A) one", "(B) two"],
            final_answer_type=answer_type,
        )
        assert result["valid"] is True
        assert result["expected_answer_type"] == "CHOICE_SINGLE_OR_CHOICE_MULTI"


def test_answer_index_uses_post_generation_source_fields() -> None:
    index = load_answer_index(
        xlrs_json=FIXTURES / "xlrs_fixture.json",
        mme_json=FIXTURES / "mme_fixture.json",
    )
    assert index["xlrs_bbox_001"]["answer"] == "A"
    assert index["mme_rs_perception_remote_sensing_count_0001"]["answer"] == "B"
