from counting_system.executor import CountExecutor
from counting_system.runtime import Entity, EntitySet, ImageRef, Region, SelectResult, SelectStatus
from counting_system.target import build_target


def _ships(n: int = 3) -> EntitySet:
    image = ImageRef(uri_or_key="memory://ships", width=100, height=100)
    entities = tuple(
        Entity(
            Region(image, (i * 20.0, i * 20.0, i * 20.0 + 10.0, i * 20.0 + 10.0)),
            "ship",
            1.0,
        )
        for i in range(n)
    )
    return EntitySet(entities)


def test_entityset_count_does_not_redetect():
    entities = _ships(3)
    result = CountExecutor({"detector": {"backend": "fake"}})(entities, build_target("ship"))
    assert result.count == 3
    assert result.to_scalar().value == 3
    assert result.provenance["mode"] == "entityset"
    assert result.provenance["redetect"] is False
    assert result.provenance["detector_calls"] == 0
    assert result.provenance["provider"] == "cardinality"


def test_count_execute_entities_role_returns_scalar():
    scalar = CountExecutor({"detector": {"backend": "fake"}}).execute(
        {"entities": _ships(2)},
        {"target": {"category": "ship"}, "entire": False},
    )
    assert scalar.value == 2
    assert scalar.provenance["redetect"] is False


def test_count_select_empty_is_zero():
    empty = SelectResult(
        selected=EntitySet(()),
        status=SelectStatus.EMPTY,
        method="geometry",
        reason="no_match",
    )
    result = CountExecutor({"detector": {"backend": "fake"}})({"entities": empty}, "ship")
    assert result.count == 0
