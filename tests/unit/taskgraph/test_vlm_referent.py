"""Unit tests for ReferentRefiner.vlm_referent (VLM primary singleton locate)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.referent_refinement import ReferentRefiner
from sat_rs_vlm.taskgraph.runtime_types import EntitySet, ImageRef, Region
from sat_rs_vlm.taskgraph.schema import TargetSpec

WIDTH, HEIGHT = 400, 300


class _PointerProvider:
    def __init__(self, message: str) -> None:
        self.message = message
        self.requests: list[object] = []

    def infer(self, request: object):  # noqa: ANN001 - duck typing for the refiner
        self.requests.append(request)
        return SimpleNamespace(text=self.message)


def _make_image(tmp_path: Path) -> ImageRef:
    path = tmp_path / "scene.png"
    Image.new("RGB", (WIDTH, HEIGHT), (90, 120, 100)).save(path)
    return ImageRef(str(path), width=WIDTH, height=HEIGHT)


def _refiner(tmp_path: Path, message: str) -> tuple[ReferentRefiner, _PointerProvider]:
    provider = _PointerProvider(message)
    refiner = ReferentRefiner(provider, InputComposer(tmp_path / "refiner"))
    return refiner, provider


def _target() -> TargetSpec:
    return TargetSpec.model_validate({"category": "building"})


def test_vlm_referent_parses_bbox(tmp_path: Path) -> None:
    image = _make_image(tmp_path)
    refiner, provider = _refiner(tmp_path, "The building is at 120 80 340 260")
    result = refiner.vlm_referent(image, question="Where is the building?", target=_target())

    assert result is not None
    entities = result.entities
    assert isinstance(entities, EntitySet)
    assert len(entities.entities) == 1
    entity = entities.entities[0]
    assert list(entity.region.bbox_xyxy_global) == [120.0, 80.0, 340.0, 260.0]
    assert entity.label == "building"
    assert entity.provenance.get("vlm_referent") is True
    assert entities.provenance["resolution_status"] == "VLM_REFERENT_RESOLVED"
    assert entities.provenance["provider"] == "semantic_2b"
    assert len(provider.requests) == 1


def test_vlm_referent_rejects_garbage(tmp_path: Path) -> None:
    image = _make_image(tmp_path)
    refiner, provider = _refiner(tmp_path, "I see buildings everywhere, cannot count them.")
    result = refiner.vlm_referent(image, question="Where is the building?", target=_target())
    assert result is None
    assert len(provider.requests) == 1


def test_vlm_referent_rejects_degenerate_boxes(tmp_path: Path) -> None:
    image = _make_image(tmp_path)
    refiner, _ = _refiner(tmp_path, "0 0 0 0")
    assert refiner.vlm_referent(image, question="Where?", target=_target()) is None


def test_vlm_referent_skips_non_full_extent_region(tmp_path: Path) -> None:
    image = _make_image(tmp_path)
    region = Region(image, (10.0, 10.0, 200.0, 150.0), {})
    refiner, provider = _refiner(tmp_path, "120 80 340 260")
    assert refiner.vlm_referent(region, question="Where?", target=_target()) is None
    assert provider.requests == []
