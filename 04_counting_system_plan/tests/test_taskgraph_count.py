import pytest

from counting_system.contracts import validate_count_inputs
from counting_system.executor import CountExecutor, CountParams
from counting_system.runtime import (
    EntitySet,
    ImageRef,
    Region,
    SelectResult,
    SelectStatus,
    SelectResultConsumptionError,
)
from counting_system.target import build_target


def test_count_requires_exactly_one_role():
    with pytest.raises(ValueError, match="exactly one"):
        validate_count_inputs({})
    with pytest.raises(ValueError, match="exactly one"):
        image = ImageRef(uri_or_key="memory://a", width=8, height=8)
        validate_count_inputs({"image": image, "entities": EntitySet(())})


def test_count_params_match_taskgraph_schema():
    params = CountParams(target=build_target("ship"), entire=True)
    dumped = params.to_dict()
    assert dumped["entire"] is True
    assert dumped["target"]["category"] == "ship"
    assert "attributes" in dumped["target"]


def test_region_input_forces_entire_false():
    image = ImageRef(uri_or_key="memory://a", width=100, height=100)
    region = Region(image, (0.0, 0.0, 50.0, 50.0))
    executor = CountExecutor({"detector": {"backend": "fake"}})
    parsed = executor._parse_params(
        {"target": {"category": "ship"}, "entire": True},
        {"image": region},
    )
    assert parsed.entire is False


def test_ambiguous_select_is_rejected():
    selected = SelectResult(
        selected=EntitySet(()),
        status=SelectStatus.AMBIGUOUS,
        method="geometry",
        reason="tie",
    )
    with pytest.raises(SelectResultConsumptionError):
        CountExecutor({"detector": {"backend": "fake"}}).execute({"entities": selected}, {"target": {"category": "ship"}, "entire": False})


def test_target_phrase_matches_schema():
    spec = build_target("ship")
    spec.attributes = {"color": "white"}
    assert spec.category == "ship"
    assert spec.phrase() == "white ship"
    assert spec.name == "ship"
