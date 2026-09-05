"""RELATION consumes FIND_MARKER RegionSet + rank-tied AMBIGUOUS selections.

Regression coverage for two XLRS sample failures:
- Object_spatial_relationship_Object_spatial_relationship_45:
  FIND_MARKER output (RegionSet) fed into RELATION.subject was rejected by the
  input contract even though the composer already renders RegionSet crops.
- Object_spatial_relationship_Object_spatial_relationship_137:
  SELECT_RANK produced an AMBIGUOUS (rank-tie) SelectResult that RELATION.reference
  refused even though the policy comment claimed AMBIGUOUS was tolerated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sat_rs_vlm.taskgraph.contracts import validate_runtime_inputs
from sat_rs_vlm.taskgraph.runtime_types import (
    Entity,
    EntitySet,
    ImageRef,
    Region,
    RegionSet,
    SelectResult,
    SelectStatus,
    unwrap_select_result,
)

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
        {"provider": "find_marker"},
    )


def test_relation_contract_accepts_region_set() -> None:
    validate_runtime_inputs(
        "RELATION",
        {"subject": _region_set(), "reference": _region_set()},
    )  # must not raise


def test_relation_contract_accepts_region_set_and_entityset_mix() -> None:
    image = _image()
    entities = EntitySet(
        (Entity("roundabout", 0.9, Region(image, (10.0, 10.0, 60.0, 60.0), {})),),
        {"provider": "lae"},
    )
    validate_runtime_inputs(
        "RELATION",
        {"subject": _region_set(), "reference": entities},
    )  # must not raise


def _ambiguous_select() -> SelectResult:
    image = _image()
    tied = EntitySet(
        (
            Entity("forest", 0.8, Region(image, (10.0, 10.0, 60.0, 60.0), {})),
            Entity("forest", 0.8, Region(image, (200.0, 150.0, 260.0, 210.0), {})),
        ),
        {"select": "RANK"},
    )
    return SelectResult(
        selected=tied,
        status=SelectStatus.AMBIGUOUS,
        method="geometry",
        reason="rank_tie",
    )


def test_unwrap_ambiguous_rejected_by_default() -> None:
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        unwrap_select_result(
            _ambiguous_select(),
            allow_empty=True,
            require_single=False,
            consumer="RELATION.reference",
        )


def test_unwrap_ambiguous_allowed_for_relation() -> None:
    result = unwrap_select_result(
        _ambiguous_select(),
        allow_empty=True,
        require_single=False,
        allow_ambiguous=True,
        consumer="RELATION.reference",
    )
    assert isinstance(result, EntitySet)
    assert len(result.entities) == 2


def test_unwrap_ambiguous_still_enforces_single_when_required() -> None:
    with pytest.raises(ValueError, match="requires one selected object"):
        unwrap_select_result(
            _ambiguous_select(),
            allow_empty=True,
            require_single=True,
            allow_ambiguous=True,
            consumer="ATTRIBUTE.entity",
        )
