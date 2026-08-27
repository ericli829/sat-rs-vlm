"""Canonical data structures for query-aware UHR localization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from sat_rs_vlm.semantics import TaskSpec

BBox = tuple[float, float, float, float]


class LocatorError(RuntimeError):
    """The locator configuration, provider, or result was invalid."""


@dataclass
class SearchRegion:
    region_id: str
    parent_id: str | None
    depth: int
    core_xyxy: BBox
    view_xyxy: BBox
    score: float = 0.0
    score_components: dict[str, Any] = field(default_factory=dict)
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise LocatorError("SearchRegion.depth must be non-negative")
        if not math.isfinite(float(self.score)):
            raise LocatorError("SearchRegion.score must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "core_xyxy": list(self.core_xyxy),
            "view_xyxy": list(self.view_xyxy),
            "score": float(self.score),
            "score_components": dict(self.score_components),
            "children": list(self.children),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SearchPlan:
    use_detector: bool
    use_retrieval: bool
    use_spatial: bool
    bypass_locator: bool
    desired_multi_region: bool
    target_phrases: tuple[str, ...] = ()
    route: str = "default"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_detector": self.use_detector,
            "use_retrieval": self.use_retrieval,
            "use_spatial": self.use_spatial,
            "bypass_locator": self.bypass_locator,
            "desired_multi_region": self.desired_multi_region,
            "target_phrases": list(self.target_phrases),
            "route": self.route,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class LocatorResult:
    regions_xyxy: tuple[BBox, ...]
    scores: tuple[float, ...]
    task_spec: TaskSpec
    search_plan: SearchPlan
    search_trace: tuple[dict[str, Any], ...]
    processed_area_ratio: float
    selected_union_area_ratio: float
    processed_union_area_ratio: float
    depth_reached: int
    latency_ms: dict[str, float]
    provider_provenance: dict[str, Any]
    warnings: tuple[str, ...] = ()
    region_details: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if len(self.regions_xyxy) != len(self.scores):
            raise LocatorError("locator regions and scores must have equal lengths")
        if not math.isfinite(self.processed_area_ratio) or self.processed_area_ratio < 0.0:
            raise LocatorError("processed_area_ratio must be finite and non-negative")
        for label, value in (
            ("selected_union_area_ratio", self.selected_union_area_ratio),
            ("processed_union_area_ratio", self.processed_union_area_ratio),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise LocatorError(f"{label} must be finite and between 0 and 1")
        if self.depth_reached < 0:
            raise LocatorError("depth_reached must be non-negative")

    @property
    def inspected_area_ratio(self) -> float:
        """Deprecated compatibility alias for cumulative processed crop work."""

        return self.processed_area_ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            "regions_xyxy": [list(box) for box in self.regions_xyxy],
            "scores": [float(score) for score in self.scores],
            "task_spec": self.task_spec.to_dict(),
            "search_plan": self.search_plan.to_dict(),
            "search_trace": [dict(item) for item in self.search_trace],
            "processed_area_ratio": self.processed_area_ratio,
            "selected_union_area_ratio": self.selected_union_area_ratio,
            "processed_union_area_ratio": self.processed_union_area_ratio,
            "inspected_area_ratio": self.processed_area_ratio,
            "depth_reached": self.depth_reached,
            "latency_ms": dict(self.latency_ms),
            "provider_provenance": dict(self.provider_provenance),
            "warnings": list(self.warnings),
            "region_details": [dict(item) for item in self.region_details],
        }
