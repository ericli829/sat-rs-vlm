"""Typed values exchanged between production TaskGraph nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

BBox = tuple[float, float, float, float]


def _bbox(value: BBox) -> BBox:
    box = tuple(float(item) for item in value)
    if len(box) != 4 or not all(math.isfinite(item) for item in box):
        raise ValueError("bbox must contain four finite coordinates")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("bbox must have positive area")
    return box


@dataclass(frozen=True)
class ImageRef:
    uri_or_key: str
    width: int | None = None
    height: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.uri_or_key).expanduser()


@dataclass(frozen=True)
class Region:
    image: ImageRef
    bbox_xyxy_global: BBox
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox_xyxy_global", _bbox(self.bbox_xyxy_global))


@dataclass(frozen=True)
class RegionSet:
    regions: tuple[Region, ...]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Entity:
    region: Region
    label: str
    score: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntitySet:
    entities: tuple[Entity, ...]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScalarInt:
    value: int
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScalarFloat:
    value: float
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Boolean:
    value: bool
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Label:
    value: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelSet:
    values: tuple[str, ...]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteContext:
    image: ImageRef
    start: Entity | EntitySet | Region
    goal: Entity | EntitySet | Region
    context_region: Region
    marker_visual_path: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    value: RuntimeObject
    description: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceSet:
    evidence: tuple[Evidence, ...]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Answer:
    text: str
    confidence: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChoiceResult:
    choice_id: str
    raw_response: str
    confidence: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


RuntimeObject: TypeAlias = (
    ImageRef
    | Region
    | RegionSet
    | Entity
    | EntitySet
    | ScalarInt
    | ScalarFloat
    | Boolean
    | Label
    | LabelSet
    | RouteContext
    | Evidence
    | EvidenceSet
    | Answer
)

STRUCTURED_AUTHORITATIVE_TYPES = (ScalarInt, ScalarFloat, Boolean, Label, LabelSet)
VISUAL_TYPES = (ImageRef, Region, RegionSet, Entity, EntitySet, RouteContext)


def runtime_type_name(value: RuntimeObject) -> str:
    return type(value).__name__


def runtime_summary(value: RuntimeObject) -> dict[str, Any]:
    """Return trace-safe metadata, never image bytes or tensors."""

    summary: dict[str, Any] = {"type": runtime_type_name(value)}
    provenance = getattr(value, "provenance", None)
    if isinstance(provenance, dict):
        trace_provenance = {
            key: provenance[key]
            for key in ("provider", "model_id")
            if provenance.get(key) is not None
        }
        if trace_provenance:
            summary["provenance"] = trace_provenance
    if isinstance(value, ScalarInt | ScalarFloat | Boolean | Label):
        summary["value"] = value.value
    elif isinstance(value, LabelSet):
        summary["values"] = list(value.values)
    elif isinstance(value, Region):
        summary["bbox_xyxy_global"] = list(value.bbox_xyxy_global)
    elif isinstance(value, RegionSet):
        summary["count"] = len(value.regions)
        summary["boxes"] = [list(item.bbox_xyxy_global) for item in value.regions]
    elif isinstance(value, Entity):
        summary.update(
            label=value.label, score=value.score, bbox=list(value.region.bbox_xyxy_global)
        )
    elif isinstance(value, EntitySet):
        summary["count"] = len(value.entities)
        summary["detections"] = [
            {"label": item.label, "score": item.score, "bbox": list(item.region.bbox_xyxy_global)}
            for item in value.entities
        ]
    elif isinstance(value, Answer):
        summary["text"] = value.text[:256]
    elif isinstance(value, RouteContext):
        summary["bbox_xyxy_global"] = list(value.context_region.bbox_xyxy_global)
    return summary
