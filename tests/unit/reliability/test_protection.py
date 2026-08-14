import pytest

from sat_rs_vlm.models.reliability.protection import (
    clamp_state_dict,
    majority_vote_text,
    output_guard_vote,
)


def test_majority_vote_normalizes_whitespace_and_case() -> None:
    result = majority_vote_text([" Ship ", "ship", "harbor"])
    assert result.selected.strip().lower() == "ship"
    assert result.has_majority


def test_output_guard_discards_invalid_and_uses_fallback() -> None:
    valid = '{"label":"ship","bbox":[0.1,0.1,0.2,0.2]}'
    invalid = '{"label":"ship","bbox":[1.2,0.1,0.2,0.2]}'
    guarded = output_guard_vote("detection", [invalid, valid, valid], fallback="fallback")
    fallback = output_guard_vote("detection", [invalid], fallback=valid)

    assert guarded.selected == valid
    assert guarded.num_valid_inputs == 2
    assert fallback.selected == valid
    assert fallback.used_fallback


def test_numeric_counting_vote() -> None:
    result = output_guard_vote("counting", ["2", "count: 2", "3"], fallback="0")
    assert result.selected in {"2", "count: 2"}
    assert result.votes["number:2"] == 2


def test_weight_clamp_returns_new_state_and_statistics() -> None:
    torch = pytest.importorskip("torch")
    clean = {"weight": torch.tensor([-1.0, 0.0, 1.0])}
    fault = {"weight": torch.tensor([-10.0, float("nan"), 10.0])}

    protected, report = clamp_state_dict(clean, fault)

    assert fault["weight"][0].item() == -10.0
    assert protected["weight"].tolist() == [-1.0, 0.0, 1.0]
    assert report.experimental
    assert report.clipped_elements == 3


def test_output_guard_vote_rejects_degenerate_generation() -> None:
    valid = "A remote sensing image shows an airport apron."
    degenerate = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    guarded = output_guard_vote("captioning", [degenerate, valid, valid], fallback="fallback")

    assert guarded.selected == valid
    assert guarded.num_valid_inputs == 2
    assert guarded.rejected[0]["index"] == 0
    assert "degenerate_repeated_character" in guarded.rejected[0]["errors"]
