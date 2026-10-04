from __future__ import annotations

import json

import pytest

from sat_rs_vlm.competition.protocol_v2.answers import AnswerResolver
from sat_rs_vlm.competition.protocol_v2.coordinates import declared_space, to_pixel_region
from sat_rs_vlm.competition.protocol_v2.errors import BBoxValidationError
from sat_rs_vlm.competition.protocol_v2.models import AnswerContract
from sat_rs_vlm.taskgraph.choice import ChoiceResolver
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.providers import FakeSemanticVLMProvider
from sat_rs_vlm.taskgraph.runtime_types import Answer, ImageRef, Region
from sat_rs_vlm.taskgraph.semantic_decision import SemanticDecisionLayer


@pytest.fixture
def image(tmp_path):
    return ImageRef(str(tmp_path / "original"), 1000, 800)


@pytest.fixture
def resolver(tmp_path):
    provider = FakeSemanticVLMProvider()
    composer = InputComposer(tmp_path / "composer")
    result = AnswerResolver(
        provider, composer, ChoiceResolver(provider, composer), SemanticDecisionLayer(provider)
    )
    yield result
    composer.close()


@pytest.mark.parametrize(
    "space,box",
    [
        ("normalized_0_1", [0.1, 0.1, 0.5, 0.5]),
        ("scaled_0_1000", [100, 100, 500, 500]),
        ("normalized_1000", [100, 100, 500, 500]),
        ("normalized_0_1000", [100, 100, 500, 500]),
        ("percent_0_100", [10, 10, 50, 50]),
        ("pixel_xyxy", [100, 80, 500, 400]),
    ],
)
def test_original_frame_conversion(image, space, box):
    result = to_pixel_region(box, image, coordinate_format=space)
    assert result.region.bbox_xyxy_global == pytest.approx((100, 80, 500, 400))
    assert result.metadata["source_coordinate_format"] in {
        "normalized_0_1",
        "scaled_0_1000",
        "percent_0_100",
        "pixel_xyxy",
    }
    assert result.region.provenance["coordinate_space"] == "absolute_xyxy_global"


def test_preserves_small_pixel_box_and_normalized_full_frame(image):
    # Numeric ranges alone do not identify coordinate systems.
    result = to_pixel_region([0.1, 0.1, 0.5, 0.5], image, coordinate_format="pixel_xyxy")
    assert result.region.bbox_xyxy_global == (0.1, 0.1, 0.5, 0.5)
    result = to_pixel_region([0, 0, 1, 1], image, coordinate_format="normalized_0_1")
    assert result.region.bbox_xyxy_global == (0, 0, 1000, 800)


@pytest.mark.parametrize(
    "space,box",
    [
        ("normalized_0_1", [0.1, 0.1, 0.5, 0.5]),
        ("scaled_0_1000", [100, 100, 500, 500]),
        ("crop_local", [40, 30, 200, 150]),
    ],
)
def test_crop_scale_then_offset(image, space, box):
    result = to_pixel_region(
        box,
        image,
        coordinate_format=space,
        provenance={"coordinate_frame": "crop", "reference_bbox_xyxy_global": [200, 100, 600, 400]},
    )
    assert result.region.bbox_xyxy_global == pytest.approx((240, 130, 400, 250))
    assert result.metadata["source_coordinate_frame"] == "crop"


def test_cxcywh_and_no_double_conversion(image):
    result = to_pixel_region(
        [0.3, 0.3, 0.4, 0.4],
        image,
        coordinate_format="normalized_0_1",
        provenance={"bbox_format": "cxcywh"},
    )
    assert result.region.bbox_xyxy_global == pytest.approx((100, 80, 500, 400))
    repeat = to_pixel_region(
        result.region.bbox_xyxy_global,
        image,
        coordinate_format=declared_space(result.region.provenance),
        provenance=result.region.provenance,
    )
    assert repeat.region.bbox_xyxy_global == result.region.bbox_xyxy_global
    assert repeat.metadata["converted"] is False


@pytest.mark.parametrize(
    "space,box,metadata",
    [
        ("normalized_0_1", [0.1, 0.1, 1.01, 0.5], {}),
        ("scaled_0_1000", [100, -1, 500, 500], {}),
        ("normalized_0_1", [0.5, 0.1, 0.1, 0.5], {}),
        ("normalized_0_1", [0.1, 0.1, 0.5, 0.5], {"coordinate_frame": "crop"}),
        ("crop_local", [1, 1, 10, 10], {}),
        ("pixel_xyxy", [1, 1, 1200, 20], {}),
        ("pixel_xyxy", [1, True, 10, 20], {}),
        ("normalized_0_1", [0, 0, float("nan"), 1], {}),
        ("normalized_0_1", ["0", 0, 1, 1], {}),
        ("unknown", [1, 1, 2, 2], {}),
        (
            "normalized_0_1",
            [0.1, 0.1, 0.5, 0.5],
            {"coordinate_frame": "crop", "reference_bbox_xyxy_global": [200, 100, 1200, 400]},
        ),
    ],
)
def test_rejects_invalid_or_ambiguous_boxes(image, space, box, metadata):
    with pytest.raises(BBoxValidationError):
        to_pixel_region(box, image, coordinate_format=space, provenance=metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        {"normalized": True},
        {"coordinate_space": "normalized_0_1", "coordinate_mode": "absolute_xyxy_global"},
        {"coordinate_space": "pixel_xyxy", "normalized": True},
        {"coordinate_space": "normalized_0_1", "normalized": False},
        {"coordinate_space": "normalized_0_1", "normalized": "True"},
    ],
)
def test_declaration_is_explicit_and_consistent(metadata):
    with pytest.raises(BBoxValidationError):
        declared_space(metadata)


def test_conversion_rejects_conflicting_source_declaration(image):
    with pytest.raises(BBoxValidationError):
        to_pixel_region(
            [0.1, 0.1, 0.5, 0.5],
            image,
            coordinate_format="normalized_0_1",
            provenance={"coordinate_space": "pixel_xyxy"},
        )
    result = to_pixel_region(
        [0.1, 0.1, 0.5, 0.5],
        image,
        coordinate_format="normalized_0_1",
        provenance={"normalized": True},
    )
    assert result.region.bbox_xyxy_global == pytest.approx((100, 80, 500, 400))


def test_resolver_model_json_declared_and_configured_spaces(resolver, image):
    contract = AnswerContract("bbox", image_width=1000, image_height=800)
    source = Answer(
        json.dumps({"label": "ship", "bbox": [0.1, 0.1, 0.5, 0.5]}),
        provenance={"coordinate_format": "normalized_0_1"},
    )
    assert resolver.resolve(source, "Locate", contract, (image,)) == pytest.approx(
        [100, 80, 500, 400]
    )
    assert resolver.last_bbox_conversion["converted"] is True
    with pytest.raises(BBoxValidationError):
        resolver.resolve(Answer(source.text), "Locate", contract, (image,))
    resolver.bbox_model_coordinate_format = "normalized_0_1"
    assert resolver.resolve(Answer(source.text), "Locate", contract, (image,)) == pytest.approx(
        [100, 80, 500, 400]
    )
    # A trained-model default never rescales an existing global Region.
    pixels = Region(image, (100, 80, 500, 400))
    assert resolver.resolve(pixels, "Locate", contract, (image,)) == [100, 80, 500, 400]
    assert resolver.last_bbox_conversion["converted"] is False


def test_model_json_requires_unambiguous_image_and_single_box(resolver, image, tmp_path):
    other = ImageRef(str(tmp_path / "second"), 1000, 800)
    contract = AnswerContract("bbox", image_width=1000, image_height=800)
    text = json.dumps({"bbox": [100, 80, 500, 400]})
    metadata = {"coordinate_format": "pixel_xyxy"}
    with pytest.raises(BBoxValidationError):
        resolver.resolve(Answer(text, provenance=metadata), "Locate", contract, (image, other))
    metadata["image_index"] = 1
    assert resolver.resolve(
        Answer(text, provenance=metadata), "Locate", contract, (image, other)
    ) == [100, 80, 500, 400]
    for invalid in ("Reasoning: " + text, '{"boxes":[[1,2,3,4],[5,6,7,8]]}', "[1,2,3]"):
        with pytest.raises(BBoxValidationError):
            resolver.resolve(
                Answer(invalid, provenance=metadata), "Locate", contract, (image, other)
            )
