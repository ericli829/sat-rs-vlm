"""Convert TaskGraph runtime objects to counting_system objects and back."""

from __future__ import annotations

from typing import Any

from sat_rs_vlm.taskgraph.runtime_types import Entity, EntitySet, ImageRef, Region
from sat_rs_vlm.taskgraph.schema import TargetSpec as GraphTargetSpec

from .bootstrap import ensure_counting_system_importable

ensure_counting_system_importable()

from counting_system.runtime import (  # noqa: E402
    Detection as CountDetection,
)
from counting_system.runtime import ImageRef as CountImageRef
from counting_system.runtime import Region as CountRegion
from counting_system.target import TargetSpec as CountTargetSpec
from counting_system.target import build_target  # noqa: E402


def to_counting_image(image: ImageRef) -> CountImageRef:
    return CountImageRef(
        uri_or_key=image.uri_or_key,
        width=image.width,
        height=image.height,
        provenance=dict(image.provenance),
    )


def to_counting_region(region: Region) -> CountRegion:
    return CountRegion(
        image=to_counting_image(region.image),
        bbox_xyxy_global=region.bbox_xyxy_global,
        provenance=dict(region.provenance),
    )


def to_counting_scope(scope: ImageRef | Region) -> CountImageRef | CountRegion:
    if isinstance(scope, Region):
        return to_counting_region(scope)
    return to_counting_image(scope)


def to_counting_target(target: GraphTargetSpec) -> CountTargetSpec:
    spec = build_target(target.category)
    spec.attributes = dict(target.attributes)
    return spec


def to_taskgraph_entity(detection: CountDetection, image: ImageRef) -> Entity:
    provenance = {
        "tile_id": detection.tile_id,
        "scale_id": detection.scale_id,
        "coordinate_mode": "absolute_original_pixel_xyxy",
        **dict(detection.provenance),
    }
    return Entity(
        Region(
            image,
            detection.bbox_xyxy_global,
            {
                "provider": "counting_system",
                "coordinate_mode": "absolute_original_pixel_xyxy",
            },
        ),
        detection.label,
        float(detection.score),
        provenance,
    )


def to_taskgraph_entity_set(
    detections: Any,
    image: ImageRef,
    *,
    extra_provenance: dict[str, Any] | None = None,
) -> EntitySet:
    entities = tuple(to_taskgraph_entity(item, image) for item in detections)
    provenance = {"provider": "counting_system", **dict(extra_provenance or {})}
    return EntitySet(entities, provenance)
