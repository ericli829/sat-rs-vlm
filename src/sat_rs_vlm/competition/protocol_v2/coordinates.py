"""Convert explicitly declared model coordinates to original-image pixel Regions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from sat_rs_vlm.taskgraph.runtime_types import ImageRef, Region

from .errors import BBoxValidationError

PIXEL_SPACES = {
    "global",
    "global_pixel",
    "pixel",
    "pixel_xyxy",
    "original_pixel",
    "absolute_original_pixel_xyxy",
    "absolute_xyxy_global",
}
SCALED_SPACES = {
    "normalized_0_1": 1.0,
    "normalized_1000": 1000.0,
    "normalized_0_1000": 1000.0,
    "scaled_0_1000": 1000.0,
    "percent_0_100": 100.0,
}


def declared_space(provenance: Mapping, *, default: str | None = None) -> str:
    normalized = provenance.get("normalized")
    if normalized is not None and type(normalized) is not bool:
        raise BBoxValidationError("Invalid normalization declaration")
    values = [
        provenance[key]
        for key in ("coordinate_space", "coordinate_format", "coordinate_mode", "coordinates")
        if key in provenance and provenance[key] is not None
    ]
    canonical = set()
    for value in values:
        if not isinstance(value, str):
            raise BBoxValidationError("Invalid coordinate-space declaration")
        if value in PIXEL_SPACES:
            canonical.add("pixel_xyxy")
        elif value in SCALED_SPACES:
            canonical.add(
                {1.0: "normalized_0_1", 1000.0: "scaled_0_1000", 100.0: "percent_0_100"}[
                    SCALED_SPACES[value]
                ]
            )
        elif value == "crop_local":
            canonical.add("crop_local")
        else:
            raise BBoxValidationError("Unsupported coordinate space")
    if len(canonical) > 1:
        raise BBoxValidationError("Conflicting coordinate spaces")
    if canonical:
        result = canonical.pop()
        is_normalized = result in SCALED_SPACES
        if normalized is not None and normalized is not is_normalized:
            raise BBoxValidationError("Conflicting normalization declaration")
        return result
    if provenance.get("normalized"):
        raise BBoxValidationError("Normalization scale must be explicit")
    if default is None:
        raise BBoxValidationError("Model output coordinate space is undeclared")
    return declared_space({"coordinate_format": default})


def numeric_box(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise BBoxValidationError("Bbox must have four numeric values")
    if any(type(item) not in (float, int) for item in value):
        raise BBoxValidationError("Bbox must have four numeric values")
    try:
        box = tuple(float(item) for item in value)
    except (ValueError, OverflowError) as exc:
        raise BBoxValidationError("Non-finite coordinates") from exc
    if not all(math.isfinite(item) for item in box):
        raise BBoxValidationError("Non-finite coordinates")
    return box


@dataclass(frozen=True)
class PixelBoxConversion:
    region: Region
    metadata: dict


def to_pixel_region(
    value: object,
    image: ImageRef,
    *,
    coordinate_format: str,
    provenance: Mapping | None = None,
) -> PixelBoxConversion:
    metadata = dict(provenance or {})
    space = declared_space({"coordinate_format": coordinate_format})
    if any(
        metadata.get(key) is not None
        for key in ("coordinate_space", "coordinate_format", "coordinate_mode", "coordinates")
    ):
        if declared_space(metadata) != space:
            raise BBoxValidationError("Conflicting coordinate spaces")
    else:
        declared_space({"coordinate_format": space, "normalized": metadata.get("normalized")})
    box = numeric_box(value)
    original = box
    shape = metadata.get("bbox_format", "xyxy")
    if shape == "cxcywh":
        cx, cy, width, height = box
        if width <= 0 or height <= 0:
            raise BBoxValidationError("Bbox must have positive size")
        box = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
    elif shape != "xyxy":
        raise BBoxValidationError("Unsupported bbox layout")
    if image.width is None or image.height is None:
        raise BBoxValidationError("Original dimensions are required")
    frame = metadata.get("coordinate_frame", "crop" if space == "crop_local" else "original_image")
    if frame == "crop":
        scope = numeric_box(metadata.get("reference_bbox_xyxy_global"))
        if not (
            0 <= scope[0] < scope[2] <= image.width and 0 <= scope[1] < scope[3] <= image.height
        ):
            raise BBoxValidationError("Invalid crop coordinate frame")
    elif frame == "original_image" and space != "crop_local":
        scope = (0.0, 0.0, float(image.width), float(image.height))
    else:
        raise BBoxValidationError("Unknown or missing coordinate frame")
    divisor = SCALED_SPACES.get(space)
    if divisor is not None:
        if not all(0 <= item <= divisor for item in box):
            raise BBoxValidationError("Normalized bbox is outside its declared range")
        width, height = scope[2] - scope[0], scope[3] - scope[1]
        box = (
            box[0] * width / divisor,
            box[1] * height / divisor,
            box[2] * width / divisor,
            box[3] * height / divisor,
        )
    if not (
        0 <= box[0] < box[2] <= scope[2] - scope[0] and 0 <= box[1] < box[3] <= scope[3] - scope[1]
    ):
        raise BBoxValidationError("Bbox is outside the declared frame")
    pixels = (box[0] + scope[0], box[1] + scope[1], box[2] + scope[0], box[3] + scope[1])
    trace = {
        "source_bbox": list(original),
        "source_coordinate_format": space,
        "source_bbox_format": shape,
        "source_coordinate_frame": frame,
        "reference_bbox_xyxy_global": list(scope),
        "output_bbox_xyxy_global": list(pixels),
        "converted": space != "pixel_xyxy" or frame != "original_image" or shape != "xyxy",
    }
    # Output coordinates now satisfy the existing Region global-pixel contract.
    result_provenance = {
        **metadata,
        "coordinate_space": "absolute_xyxy_global",
        "normalized": False,
        "coordinate_frame": "original_image",
        "bbox_format": "xyxy",
        "bbox_conversion": trace,
    }
    for key in ("coordinate_format", "coordinate_mode", "coordinates"):
        result_provenance.pop(key, None)
    return PixelBoxConversion(Region(image, pixels, result_provenance), trace)
