"""Deterministic multi-ROI validation, suppression, merging, and context expansion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .geometry import bbox_iou, canonical_bbox, clamp_bbox, expand_with_halo
from .types import LocatorError, SearchRegion


def _axis_overlap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _are_adjacent(left: SearchRegion, right: SearchRegion, gap: float) -> bool:
    a, b = left.core_xyxy, right.core_xyxy
    horizontal_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    vertical_gap = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    vertical_overlap = _axis_overlap(a[1], a[3], b[1], b[3])
    horizontal_overlap = _axis_overlap(a[0], a[2], b[0], b[2])
    return (horizontal_gap <= gap and vertical_overlap > 0.0) or (
        vertical_gap <= gap and horizontal_overlap > 0.0
    )


class RegionFusion:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        values = dict(config or {})
        self.overlap_iou_threshold = float(values.get("overlap_iou_threshold", 0.65))
        self.max_regions = int(values.get("max_regions", 6))
        self.min_regions = int(values.get("min_regions", 1))
        self.score_threshold = (
            float(values["score_threshold"])
            if values.get("score_threshold") is not None
            else None
        )
        self.merge_adjacent = bool(values.get("merge_adjacent", False))
        self.adjacent_gap = float(values.get("adjacent_gap", 4.0))
        self.context_margin = float(values.get("context_margin", 0.08))
        if not 0.0 <= self.overlap_iou_threshold <= 1.0:
            raise LocatorError("fusion overlap_iou_threshold must be between 0 and 1")
        if (
            self.max_regions < 1
            or self.min_regions < 1
            or self.min_regions > self.max_regions
            or self.adjacent_gap < 0.0
            or self.context_margin < 0.0
        ):
            raise LocatorError("fusion region/gap/margin values are invalid")
        if self.score_threshold is not None and not 0.0 <= self.score_threshold <= 1.0:
            raise LocatorError("fusion score_threshold must be between 0 and 1")

    def fuse(
        self,
        regions: Sequence[SearchRegion],
        image_width: int,
        image_height: int,
    ) -> tuple[SearchRegion, ...]:
        valid: list[SearchRegion] = []
        for region in regions:
            try:
                core = clamp_bbox(region.core_xyxy, image_width, image_height)
            except LocatorError:
                continue
            valid.append(replace(region, core_xyxy=core))
        valid.sort(key=lambda item: (-item.score, item.depth, item.region_id))
        selected: list[SearchRegion] = []
        eligible = (
            [region for region in valid if region.score >= self.score_threshold]
            if self.score_threshold is not None
            else valid
        )
        fallback = [region for region in valid if region not in eligible]

        def select_from(candidates: Sequence[SearchRegion], target: int) -> None:
            for region in candidates:
                if any(
                    bbox_iou(region.core_xyxy, kept.core_xyxy) > self.overlap_iou_threshold
                    for kept in selected
                ):
                    continue
                selected.append(region)
                if len(selected) >= target:
                    break

        select_from(eligible, self.max_regions)
        if self.score_threshold is not None and len(selected) < self.min_regions:
            select_from(fallback, self.min_regions)

        if self.merge_adjacent:
            merged: list[SearchRegion] = []
            for region in selected:
                target = next(
                    (
                        index
                        for index, existing in enumerate(merged)
                        if _are_adjacent(existing, region, self.adjacent_gap)
                    ),
                    None,
                )
                if target is None:
                    merged.append(region)
                    continue
                existing = merged[target]
                union = canonical_bbox(
                    (
                        min(existing.core_xyxy[0], region.core_xyxy[0]),
                        min(existing.core_xyxy[1], region.core_xyxy[1]),
                        max(existing.core_xyxy[2], region.core_xyxy[2]),
                        max(existing.core_xyxy[3], region.core_xyxy[3]),
                    )
                )
                metadata = dict(existing.metadata)
                metadata["merged_region_ids"] = [
                    *metadata.get("merged_region_ids", [existing.region_id]),
                    region.region_id,
                ]
                merged[target] = replace(
                    existing,
                    core_xyxy=union,
                    score=max(existing.score, region.score),
                    metadata=metadata,
                )
            selected = merged

        return tuple(
            replace(
                region,
                view_xyxy=expand_with_halo(
                    region.core_xyxy,
                    self.context_margin,
                    image_width,
                    image_height,
                ),
                metadata={
                    **region.metadata,
                    "coordinate_mode": "absolute_original_pixel_xyxy",
                    "context_margin": self.context_margin,
                    "selection_policy": (
                        "score_threshold_then_min_fill"
                        if self.score_threshold is not None
                        else "top_score"
                    ),
                    "score_threshold": self.score_threshold,
                    "min_regions": self.min_regions,
                    "max_regions": self.max_regions,
                },
            )
            for region in selected[: self.max_regions]
        )
