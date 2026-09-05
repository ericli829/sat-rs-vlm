"""RegionSet support: COUNT union-scope counting + CLASSIFY source contract."""

from __future__ import annotations

from pathlib import Path

from sat_rs_vlm.taskgraph.contracts import validate_runtime_inputs
from sat_rs_vlm.taskgraph.operators import CountExecutor
from sat_rs_vlm.taskgraph.providers import FakeCountingProvider
from sat_rs_vlm.taskgraph.runtime_types import ImageRef, Region, RegionSet
from sat_rs_vlm.taskgraph.schema import GraphNode, OperatorName

IMG = Path("tests/fixtures/miniature_dataset/images/vqa.ppm")


def _image() -> ImageRef:
    return ImageRef(str(IMG), width=400, height=300)


def _region_set() -> RegionSet:
    image = _image()
    return RegionSet(
        (
            Region(image, (0.0, 0.0, 100.0, 100.0), {"id": "r0"}),
            Region(image, (200.0, 150.0, 300.0, 240.0), {"id": "r1"}),
        ),
        {"provider": "retriever"},
    )


def _count_node() -> GraphNode:
    return GraphNode(
        id="n2",
        op=OperatorName.COUNT,
        inputs={"image": "$n1"},
        params={"target": {"category": "building"}, "entire": False},
    )


def test_count_contract_accepts_region_set() -> None:
    validate_runtime_inputs("COUNT", {"image": _region_set()})  # must not raise


def test_classify_contract_accepts_region_set() -> None:
    validate_runtime_inputs("CLASSIFY", {"source": _region_set()})  # must not raise
    validate_runtime_inputs("MULTILABEL_CLASSIFY", {"source": _region_set()})


def test_count_executor_unions_region_set_scope() -> None:
    provider = FakeCountingProvider(boxes=[(10, 10, 40, 40), (220, 170, 260, 210), (900, 900, 990, 990)])
    executor = CountExecutor(provider)
    outcome = executor.execute(_count_node(), {"image": _region_set()}, context=None)  # type: ignore[arg-type]

    assert outcome.value.value == 2  # third box is outside the union scope
    assert outcome.value.provenance["region_set"] == "union"
    assert outcome.value.provenance["region_set_component_count"] == 2
    provider.calls[0].scope is not None
    req = provider.calls[0]
    assert isinstance(req.scope, Region)
    assert list(req.scope.bbox_xyxy_global) == [0.0, 0.0, 300.0, 240.0]
    assert req.scope.provenance.get("region_set_union") is True


def test_count_executor_empty_region_set_returns_zero() -> None:
    executor = CountExecutor(FakeCountingProvider())
    outcome = executor.execute(_count_node(), {"image": RegionSet((), {})}, context=None)  # type: ignore[arg-type]
    assert outcome.value.value == 0
    assert outcome.value.provenance["source"] == "RegionSet"
