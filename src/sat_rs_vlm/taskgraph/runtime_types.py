"""Typed values exchanged between production TaskGraph nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
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


class SelectStatus(str, Enum):
    """Explicit terminal state for a SELECT operator.

    SELECT is allowed to return an empty set, but must never conceal an
    unresolved or ambiguous decision as an ordinary empty result.
    """

    OK = "OK"
    EMPTY = "EMPTY"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SelectResult:
    """A selection together with the method and decision state used to obtain it."""

    selected: EntitySet | Region | RegionSet
    status: SelectStatus
    method: str
    reason: str | None = None
    confidence: float | None = None
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
class ChoiceScoreResult:
    selected_ids: tuple[str, ...]
    scores: dict[str, float]
    answer_type: str
    reasoning_text: str | None
    provider: str
    model_id: str
    method: str
    cache_reused: bool
    latency_ms: dict[str, float | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.answer_type not in {"CHOICE_SINGLE", "CHOICE_MULTI"}:
            raise ValueError("choice score answer_type must be CHOICE_SINGLE or CHOICE_MULTI")
        if self.answer_type == "CHOICE_SINGLE" and len(self.selected_ids) != 1:
            raise ValueError("CHOICE_SINGLE requires exactly one selected id")
        if len(self.selected_ids) != len(set(self.selected_ids)):
            raise ValueError("selected choice ids must be unique")
        if any(choice_id not in self.scores for choice_id in self.selected_ids):
            raise ValueError("every selected choice id must have a score")


@dataclass(frozen=True)
class ChoiceResult:
    selected_ids: tuple[str, ...]
    answer_type: str
    raw_response: str
    confidence: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.answer_type not in {"CHOICE_SINGLE", "CHOICE_MULTI"}:
            raise ValueError("choice result answer_type must be CHOICE_SINGLE or CHOICE_MULTI")
        if self.answer_type == "CHOICE_SINGLE" and len(self.selected_ids) != 1:
            raise ValueError("CHOICE_SINGLE requires exactly one selected id")
        if len(self.selected_ids) != len(set(self.selected_ids)):
            raise ValueError("selected choice ids must be unique")

    @property
    def choice_id(self) -> str | None:
        """Legacy single-choice view; multi-choice is never flattened to a string."""

        if self.answer_type != "CHOICE_SINGLE":
            return None
        return self.selected_ids[0] if len(self.selected_ids) == 1 else None

    @property
    def choice_ids(self) -> tuple[str, ...]:
        """Compatibility alias for callers migrated before ``selected_ids`` was frozen."""

        return self.selected_ids


RuntimeObject: TypeAlias = (
    ImageRef
    | Region
    | RegionSet
    | Entity
    | EntitySet
    | SelectResult
    | ScalarInt
    | ScalarFloat
    | Boolean
    | Label
    | LabelSet
    | RouteContext
    | Evidence
    | EvidenceSet
    | Answer
    | ChoiceScoreResult
)


class SelectResultConsumptionError(ValueError):
    """A downstream consumer attempted to use an unsafe SELECT state."""


def unwrap_select_result(
    value: RuntimeObject,
    *,
    allow_empty: bool,
    require_single: bool = False,
    consumer: str = "downstream",
) -> RuntimeObject:
    """Materialize a SELECT result according to one explicit downstream policy.

    ``OK`` always exposes ``selected``. ``EMPTY`` is exposed only to set-aware
    consumers that opt in. Unresolved, ambiguous, and error states never flow
    into another operator or VLM implicitly.
    """

    if not isinstance(value, SelectResult):
        return value
    if value.status is SelectStatus.EMPTY:
        if not allow_empty:
            raise SelectResultConsumptionError(
                f"{consumer} refuses SELECT status {value.status.value}"
            )
    elif value.status is not SelectStatus.OK:
        raise SelectResultConsumptionError(
            f"{consumer} refuses SELECT status {value.status.value}"
        )

    selected: RuntimeObject = value.selected
    if not require_single:
        return selected
    if isinstance(selected, EntitySet):
        if len(selected.entities) == 1:
            return selected.entities[0]
        cardinality = len(selected.entities)
    elif isinstance(selected, RegionSet):
        if len(selected.regions) == 1:
            return selected.regions[0]
        cardinality = len(selected.regions)
    elif isinstance(selected, Region):
        return selected
    else:  # pragma: no cover - SelectResult.selected is a closed union.
        raise SelectResultConsumptionError(
            f"{consumer} cannot consume SELECT payload {type(selected).__name__}"
        )
    raise SelectResultConsumptionError(
        f"{consumer} requires one selected object, got {cardinality}"
    )

STRUCTURED_AUTHORITATIVE_TYPES = (ScalarInt, ScalarFloat, Boolean, Label, LabelSet)
VISUAL_TYPES = (ImageRef, Region, RegionSet, Entity, EntitySet, SelectResult, RouteContext)


def runtime_type_name(value: RuntimeObject | ChoiceResult) -> str:
    return type(value).__name__


def runtime_summary(value: RuntimeObject | ChoiceResult) -> dict[str, Any]:
    """Return trace-safe metadata, never image bytes or tensors."""

    summary: dict[str, Any] = {"type": runtime_type_name(value)}
    if isinstance(value, (ScalarInt, ScalarFloat, Boolean, Label)):
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
    elif isinstance(value, SelectResult):
        summary.update(
            status=value.status.value,
            method=value.method,
            reason=value.reason,
            confidence=value.confidence,
            selected=runtime_summary(value.selected),
            provenance=dict(value.provenance),
        )
    elif isinstance(value, Answer):
        summary["text"] = value.text[:256]
    elif isinstance(value, RouteContext):
        summary["bbox_xyxy_global"] = list(value.context_region.bbox_xyxy_global)
    elif isinstance(value, ChoiceScoreResult):
        summary.update(
            selected_ids=list(value.selected_ids),
            scores=dict(value.scores),
            answer_type=value.answer_type,
            reasoning_text=(value.reasoning_text or "")[:256],
            provider=value.provider,
            model_id=value.model_id,
            method=value.method,
            cache_reused=value.cache_reused,
            latency_ms=dict(value.latency_ms),
            metadata=dict(value.metadata),
        )
    elif isinstance(value, ChoiceResult):
        summary.update(
            answer_type=value.answer_type,
            selected_ids=list(value.selected_ids),
            choice_id=value.choice_id,
            raw_response=value.raw_response,
            confidence=value.confidence,
            provenance=dict(value.provenance),
        )
    return summary
