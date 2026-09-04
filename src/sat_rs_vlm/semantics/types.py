"""Dependency-light semantic types shared by evaluation and runtime routing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

OPERATIONS = frozenset(
    {
        "count",
        "existence",
        "attribute",
        "position",
        "grounding",
        "relation",
        "category",
        "global_scene",
        "open_reasoning",
        "unknown",
    }
)
ATTRIBUTES = frozenset({"color", "size", "shape", "orientation", "category"})
SPATIAL_SCOPES = frozenset(
    {
        "global",
        "left",
        "right",
        "upper",
        "lower",
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
        "center_left",
        "center_right",
        "center",
        "north",
        "south",
        "east",
        "west",
    }
)


@dataclass(frozen=True)
class TermMention:
    canonical: str
    alias: str
    start: int
    end: int


@dataclass(frozen=True)
class SemanticFacts:
    objects: tuple[str, ...]
    counts: tuple[tuple[str, int], ...]
    relations: tuple[tuple[str, str, str], ...]
    changes: tuple[tuple[str | None, str], ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": list(self.objects),
            "counts": [
                {"object": object_name, "count": count}
                for object_name, count in self.counts
            ],
            "relations": [
                {"subject": subject, "predicate": predicate, "object": object_name}
                for subject, predicate, object_name in self.relations
            ],
            "changes": [
                {"object": object_name, "change_type": change_type}
                for object_name, change_type in self.changes
            ],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RelationSpec:
    subject: str
    predicate: str
    object: str

    def to_dict(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }


@dataclass(frozen=True)
class TaskSpec:
    raw_question: str
    operation: str
    targets: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    relations: tuple[RelationSpec, ...] = ()
    spatial_scope: str = "global"
    scope: str = "image"
    multi_instance: bool = False
    given_bbox: tuple[float, float, float, float] | None = None
    confidence: float = 0.0
    parser_source: str = "unknown"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.raw_question.strip():
            raise ValueError("TaskSpec.raw_question must not be empty")
        if self.operation not in OPERATIONS:
            raise ValueError(f"unsupported TaskSpec operation: {self.operation!r}")
        if self.spatial_scope not in SPATIAL_SCOPES:
            raise ValueError(f"unsupported TaskSpec spatial_scope: {self.spatial_scope!r}")
        invalid_attributes = set(self.attributes).difference(ATTRIBUTES)
        if invalid_attributes:
            raise ValueError(f"unsupported TaskSpec attributes: {sorted(invalid_attributes)}")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("TaskSpec.confidence must be finite and between 0 and 1")
        if self.given_bbox is not None:
            if len(self.given_bbox) != 4 or not all(math.isfinite(v) for v in self.given_bbox):
                raise ValueError("TaskSpec.given_bbox must contain four finite values")
            x1, y1, x2, y2 = self.given_bbox
            if x2 <= x1 or y2 <= y1:
                raise ValueError("TaskSpec.given_bbox must be a non-degenerate xyxy box")

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_question": self.raw_question,
            "operation": self.operation,
            "targets": list(self.targets),
            "attributes": list(self.attributes),
            "relations": [relation.to_dict() for relation in self.relations],
            "spatial_scope": self.spatial_scope,
            "scope": self.scope,
            "multi_instance": self.multi_instance,
            "given_bbox": list(self.given_bbox) if self.given_bbox is not None else None,
            "confidence": self.confidence,
            "parser_source": self.parser_source,
            "warnings": list(self.warnings),
        }
