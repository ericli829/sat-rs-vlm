from counting_system.executor import CountExecutor
from counting_system.runtime import Entity, EntitySet
from counting_system.target import build_target


def test_entityset_count_does_not_redetect():
    entities = EntitySet(
        [
            Entity((0, 0, 10, 10), label="ship"),
            Entity((20, 20, 30, 30), label="ship"),
            Entity((40, 40, 50, 50), label="ship"),
        ]
    )
    result = CountExecutor({"detector": {"backend": "fake"}})(entities, build_target("ship"))
    assert result.count == 3
    assert result.to_scalar().value == 3
    assert result.provenance["mode"] == "entityset"
    assert result.provenance["redetect"] is False
    assert result.provenance["detector_calls"] == 0
