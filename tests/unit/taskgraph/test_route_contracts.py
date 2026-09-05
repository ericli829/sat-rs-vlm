"""BUILD_ROUTE_CONTEXT consumes SELECT results (rank-tied endpoint evidence).

Regression coverage for XLRS Route planning failures:
- Complex_reasoning_Route_planning_*: BUILD_ROUTE_CONTEXT.start/goal rejected
  SelectResult in the input contract and the endpoint resolver raised on
  rank-tied EntitySets; routes are endpoint-requiring so a tied highest-score
  candidate is pinned deterministically instead of hard-failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sat_rs_vlm.taskgraph.contracts import validate_runtime_inputs
from sat_rs_vlm.taskgraph.operators import GeometryExecutor
from sat_rs_vlm.taskgraph.runtime_types import (
    Entity,
    EntitySet,
    ImageRef,
    Region,
    SelectResult,
    SelectStatus,
)

IMG = Path("tests/fixtures/miniature_dataset/images/vqa.ppm")


def _image() -> ImageRef:
    return ImageRef(str(IMG), width=400, height=300)


def _ambiguous_select() -> SelectResult:
    image = _image()
    tied = EntitySet(
        (
            Entity(Region(image, (10.0, 10.0, 60.0, 60.0)), "forest", 0.8, {"candidate_id": "a"}),
            Entity(Region(image, (200.0, 150.0, 260.0, 210.0)), "forest", 0.8, {"candidate_id": "b"}),
        ),
        {"select": "RANK", "mode": "RANK"},
    )
    return SelectResult(
        selected=tied,
        status=SelectStatus.AMBIGUOUS,
        method="geometry",
        reason="rank_tie",
    )


def test_route_context_contract_accepts_select_result() -> None:
    image = _image()
    validate_runtime_inputs(
        "BUILD_ROUTE_CONTEXT",
        {"image": ImageRef(str(IMG), width=400, height=300), "start": _ambiguous_select(), "goal": _ambiguous_select()},
    )  # must not raise


def test_route_resolver_pins_tied_highest() -> None:
    selected, metadata = GeometryExecutor._resolve_route_endpoint(_ambiguous_select(), "start")
    assert isinstance(selected, Entity)
    assert metadata["policy"] == "highest_score_from_AMBIGUOUS"
    assert metadata["candidate_count"] == 2


def test_route_resolver_pins_tied_entityset() -> None:
    image = _image()
    tied = EntitySet(
        (
            Entity(Region(image, (10.0, 10.0, 60.0, 60.0)), "forest", 0.8, {}),
            Entity(Region(image, (200.0, 150.0, 260.0, 210.0)), "forest", 0.8, {}),
        ),
        {"select": "RANK"},
    )
    selected, metadata = GeometryExecutor._resolve_route_endpoint(tied, "goal")
    assert isinstance(selected, Entity)
    assert metadata["policy"] == "highest_tied_pinned"
