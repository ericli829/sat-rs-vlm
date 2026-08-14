import pytest

from sat_rs_vlm.models.reliability.output_validator import validate_prediction


@pytest.mark.parametrize("output", ["", "   ", None])
def test_empty_output_is_invalid(output: object) -> None:
    result = validate_prediction("captioning", output)
    assert not result.valid
    assert "output_empty" in result.errors


@pytest.mark.parametrize("output", ["NaN", "Inf", '{"answer": Infinity}'])
def test_non_finite_output_is_invalid(output: str) -> None:
    result = validate_prediction("vqa", output)
    assert not result.valid
    assert "non_finite_value" in result.errors


def test_counting_requires_non_negative_integer() -> None:
    missing = validate_prediction("counting", "many objects")
    negative = validate_prediction("counting", {"count": -1})
    decimal = validate_prediction("counting", {"count": 1.5})

    assert "counting_number_missing" in missing.errors
    assert "counting_negative" in negative.errors
    assert "counting_not_integer" in decimal.errors
    assert validate_prediction("counting", {"count": 3}).normalized_output == {"count": 3}


def test_detection_validates_bbox_label_order_and_range() -> None:
    valid = validate_prediction("detection", '{"label":"ship","bbox":[0.1,0.2,0.3,0.4]}')
    invalid = validate_prediction("detection", '{"label":"","bbox":[0.8,0.2,0.1,1.4]}')

    assert valid.valid
    assert "detection_label_empty" in invalid.errors
    assert "detection_bbox_invalid_order" in invalid.errors
    assert "detection_bbox_out_of_range" in invalid.errors


def test_vqa_type_constraints() -> None:
    assert validate_prediction("vqa", "是", vqa_question_type="yes_no").normalized_output == "yes"
    assert (
        "vqa_yes_no_invalid"
        in validate_prediction("vqa", "maybe", vqa_question_type="yes_no").errors
    )
    assert validate_prediction("vqa", "north", vqa_question_type="direction").valid


@pytest.mark.parametrize(
    ("output", "error"),
    [
        ("!!!!!!!!!!!!!!!!!!!!", "degenerate_repeated_character"),
        ("loop loop loop loop loop loop", "degenerate_repeated_token"),
        ("!@#$%^&*", "degenerate_symbol_only"),
        ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "degenerate_low_diversity"),
    ],
)
def test_generation_collapse_patterns_are_rejected(output: str, error: str) -> None:
    assert error in validate_prediction("captioning", output).errors


@pytest.mark.parametrize("output", ["yes", "no", "3", "north"])
def test_valid_short_answers_are_not_rejected_as_degenerate(output: str) -> None:
    assert validate_prediction("vqa", output).valid
