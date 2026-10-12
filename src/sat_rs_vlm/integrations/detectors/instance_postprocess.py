"""Shared conservative instance post-processing for detector proposals.

The module deliberately depends only on the Python standard library so it can
also be imported by the isolated LAE-DINO worker. It keeps ordinary IoU NMS,
then uses containment and box scale to find collection-like proposals. A
collection is replaced only when one local detector pass returns clearly
separate object proposals; otherwise its region is reported as unresolved.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

Box = Sequence[float]
Refine = Callable[[list[float]], tuple[list[list[float]], list[float]]]


def box_area(box: Box) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def box_iou(left: Box, right: Box) -> float:
    intersection = intersection_area(left, right)
    union = box_area(left) + box_area(right) - intersection
    return intersection / union if union > 0.0 else 0.0


def intersection_area(left: Box, right: Box) -> float:
    width = max(0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0])))
    height = max(0.0, min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1])))
    return width * height


def intersection_over_smaller(left: Box, right: Box) -> float:
    smaller = min(box_area(left), box_area(right))
    return intersection_area(left, right) / smaller if smaller > 0.0 else 0.0


def _center(box: Box) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def _diagonal(box: Box) -> float:
    return math.hypot(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))


def _same_instance(left: Box, right: Box, *, cross_tile: bool = False) -> bool:
    """Match close duplicate proposals without merging merely touching boxes."""

    overlap = box_iou(left, right)
    if overlap >= 0.55:
        return True
    left_area, right_area = box_area(left), box_area(right)
    if min(left_area, right_area) <= 0.0:
        return False
    area_ratio = max(left_area, right_area) / min(left_area, right_area)
    if area_ratio > 3.5:
        return False
    left_center, right_center = _center(left), _center(right)
    distance = math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])
    smaller_diagonal = min(_diagonal(left), _diagonal(right))
    if smaller_diagonal <= 0.0 or distance > 0.42 * smaller_diagonal:
        return False
    nested_repeat = intersection_over_smaller(left, right) >= 0.95
    # Tile-edge boxes can be clipped differently, so their IoU may be low even
    # when most of the smaller proposal covers the same ship. Requiring a
    # substantial shared area and a close center keeps this conservative in
    # dense harbors while catching the common cross-tile size/edge variation.
    cross_tile_repeat = cross_tile and (
        overlap >= 0.18 or intersection_over_smaller(left, right) >= 0.72
    )
    return nested_repeat or cross_tile_repeat


def conservative_nms_indices(
    boxes: Sequence[Box], scores: Sequence[float], threshold: float = 0.4
) -> list[int]:
    """Return score-ordered boxes after the configured ordinary IoU NMS."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("nms threshold must be between 0 and 1")
    if len(boxes) != len(scores):
        raise ValueError("proposal boxes and scores must have equal lengths")
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    kept: list[int] = []
    while order:
        current = order.pop(0)
        kept.append(current)
        order = [index for index in order if box_iou(boxes[current], boxes[index]) <= threshold]
    return kept


def same_instance_dedup_indices(
    boxes: Sequence[Box],
    scores: Sequence[float],
    indices: Sequence[int],
    source_groups: Sequence[int | str] | None = None,
) -> list[int]:
    kept: list[int] = []
    for index in indices:
        if not any(
            _same_instance(
                boxes[index],
                boxes[other],
                cross_tile=(
                    source_groups is not None
                    and source_groups[index] != source_groups[other]
                ),
            )
            for other in kept
        ):
            kept.append(index)
    return kept


def _contained_children(
    boxes: Sequence[Box],
    parent_index: int,
    indices: Sequence[int],
    source_groups: Sequence[int | str] | None,
) -> list[int]:
    parent = boxes[parent_index]
    parent_area = box_area(parent)
    if parent_area <= 0.0:
        return []
    children = []
    for index in indices:
        if index == parent_index:
            continue
        child_area = box_area(boxes[index])
        if child_area <= 0.0 or parent_area / child_area < 2.0:
            continue
        if intersection_over_smaller(parent, boxes[index]) >= 0.82:
            children.append(index)
    return same_instance_dedup_indices(boxes, [0.0] * len(boxes), children, source_groups)


def _independent(boxes: Sequence[Box], indices: Sequence[int]) -> bool:
    """Require spatial separation as well as low overlap for a proposed split."""

    for offset, left_index in enumerate(indices):
        left = boxes[left_index]
        left_center = _center(left)
        for right_index in indices[offset + 1 :]:
            right = boxes[right_index]
            right_center = _center(right)
            distance = math.hypot(
                left_center[0] - right_center[0], left_center[1] - right_center[1]
            )
            if box_iou(left, right) > 0.45 or distance < 0.75 * max(
                _diagonal(left), _diagonal(right)
            ):
                return False
    return True


def postprocess_instances(
    boxes: Sequence[Box],
    scores: Sequence[float],
    *,
    nms_iou: float = 0.6,
    source_ids: Sequence[int | str] | None = None,
    source_groups: Sequence[int | str] | None = None,
    reference_areas: Sequence[float] | None = None,
    refine: Refine | None = None,
    max_refinement_regions: int = 4,
    large_box_area_fraction: float = 0.20,
    relative_score_for_large_box: float = 0.6,
) -> tuple[list[list[float]], list[float], dict[str, Any]]:
    """Deduplicate, analyse nested proposals, and refine collection regions.

    ``refine`` receives an expanded-in-the-caller search bbox and must return
    boxes in the same global coordinate frame. It is called at most once per
    selected region; output is never recursively refined.
    """

    if len(boxes) != len(scores):
        raise ValueError("proposal boxes and scores must have equal lengths")
    if source_ids is not None and len(source_ids) != len(boxes):
        raise ValueError("source_ids must match proposal count")
    if source_groups is not None and len(source_groups) != len(boxes):
        raise ValueError("source_groups must match proposal count")
    if reference_areas is not None and len(reference_areas) != len(boxes):
        raise ValueError("reference_areas must match proposal count")
    if max_refinement_regions < 0:
        raise ValueError("max_refinement_regions must be non-negative")

    started = time.perf_counter()
    initial = conservative_nms_indices(boxes, scores, nms_iou)
    initial_count = len(initial)
    kept = same_instance_dedup_indices(boxes, scores, initial, source_groups)
    duplicate_count = initial_count - len(kept)
    max_score = max((float(scores[index]) for index in kept), default=0.0)

    suspects: list[tuple[float, int, list[int]]] = []
    for index in kept:
        children = _contained_children(boxes, index, kept, source_groups)
        reference_area = (
            float(reference_areas[index])
            if reference_areas is not None
            else max((float(box[2]) for box in boxes), default=0.0)
            * max((float(box[3]) for box in boxes), default=0.0)
        )
        area_fraction = box_area(boxes[index]) / reference_area if reference_area > 0.0 else 0.0
        high_support = len(children) >= 2
        nested_region_review = len(children) == 1 and area_fraction >= 0.15
        strong_large = (
            area_fraction >= large_box_area_fraction
            and float(scores[index]) >= relative_score_for_large_box * max_score
        )
        if high_support or nested_region_review or strong_large:
            # Prefer regions with multiple spatially separate children, then
            # larger regions; score only breaks ties to retain likely ships.
            priority = len(children) * 10.0 + area_fraction + float(scores[index]) * 0.01
            suspects.append((priority, index, children))
    suspects.sort(reverse=True)

    unresolved: list[dict[str, Any]] = []
    aggregate_regions: list[dict[str, Any]] = []
    processed_regions: list[list[float]] = []
    attempted = 0
    confirmed_splits = 0
    confirmed_singletons = 0
    refinement_latency_ms = 0.0

    # Work against original proposal indices, so each crop has stable evidence.
    replacement_indices: set[int] = set()
    replacements: list[
        tuple[list[float], float, int | str | None, int | str | None]
    ] = []
    for _, parent_index, children in suspects:
        parent = list(map(float, boxes[parent_index]))
        if any(intersection_over_smaller(parent, region) >= 0.8 for region in processed_regions):
            continue
        affected = [
            index
            for index in kept
            if index == parent_index or intersection_over_smaller(parent, boxes[index]) >= 0.82
        ]
        if attempted >= max_refinement_regions or refine is None:
            unresolved.append(
                {
                    "bbox_xyxy": parent,
                    "score": float(scores[parent_index]),
                    "source_id": source_ids[parent_index] if source_ids is not None else None,
                    "contained_candidate_count": len(children),
                    "status": "UNRESOLVED",
                    "reason": "refinement_unavailable_or_budget_exhausted",
                }
            )
            replacement_indices.update(affected)
            processed_regions.append(parent)
            continue

        attempted += 1
        refine_started = time.perf_counter()
        refinement_error = None
        try:
            refined_boxes, refined_scores = refine(parent)
        except Exception as exc:
            refined_boxes, refined_scores = [], []
            refinement_error = f"{type(exc).__name__}: {exc}"
        finally:
            refinement_latency_ms += (time.perf_counter() - refine_started) * 1000.0
        if len(refined_boxes) != len(refined_scores):
            raise ValueError("refinement boxes and scores must have equal lengths")
        refined_indices = conservative_nms_indices(refined_boxes, refined_scores, nms_iou)
        refined_indices = same_instance_dedup_indices(
            refined_boxes, refined_scores, refined_indices
        )
        # Keep only detections that touch the search region; a crop-padding
        # area may expose nearby objects, but they are outside this replacement.
        refined_indices = [
            index
            for index in refined_indices
            if intersection_over_smaller(parent, refined_boxes[index]) > 0.0
        ]
        split_indices = [
            index
            for index in refined_indices
            if box_area(refined_boxes[index]) < 0.8 * box_area(parent)
        ]
        if len(split_indices) >= 2 and _independent(refined_boxes, split_indices):
            replacement_indices.update(affected)
            for index in split_indices:
                replacements.append(
                    (
                        list(map(float, refined_boxes[index])),
                        float(refined_scores[index]),
                        source_ids[parent_index] if source_ids is not None else None,
                        source_groups[parent_index] if source_groups is not None else None,
                    )
                )
            confirmed_splits += 1
            aggregate_regions.append(
                {
                    "bbox_xyxy": parent,
                    "status": "SPLIT",
                    "instance_count": len(split_indices),
                    "source_id": source_ids[parent_index] if source_ids is not None else None,
                }
            )
            processed_regions.append(parent)
        elif len(refined_indices) == 1 and len(children) <= 1:
            replacement_indices.update(affected)
            index = refined_indices[0]
            replacements.append(
                (
                    list(map(float, refined_boxes[index])),
                    float(refined_scores[index]),
                    source_ids[parent_index] if source_ids is not None else None,
                    source_groups[parent_index] if source_groups is not None else None,
                )
            )
            confirmed_singletons += 1
            aggregate_regions.append(
                {
                    "bbox_xyxy": parent,
                    "status": "SINGLE_INSTANCE",
                    "instance_count": 1,
                    "source_id": source_ids[parent_index] if source_ids is not None else None,
                }
            )
            processed_regions.append(parent)
        else:
            replacement_indices.update(affected)
            unresolved.append(
                {
                    "bbox_xyxy": parent,
                    "score": float(scores[parent_index]),
                    "source_id": source_ids[parent_index] if source_ids is not None else None,
                    "contained_candidate_count": len(children),
                    "refined_candidate_count": len(refined_indices),
                    "status": "UNRESOLVED",
                    "reason": (
                        refinement_error
                        or "refinement_did_not_confirm_a_single_instance_or_clear_split"
                    ),
                }
            )
            processed_regions.append(parent)

    final_boxes: list[list[float]] = []
    final_scores: list[float] = []
    final_sources: list[int | str | None] = []
    final_groups: list[int | str | None] = []
    for index in kept:
        if index in replacement_indices:
            continue
        final_boxes.append(list(map(float, boxes[index])))
        final_scores.append(float(scores[index]))
        final_sources.append(source_ids[index] if source_ids is not None else None)
        final_groups.append(source_groups[index] if source_groups is not None else None)
    for box, score, source, group in replacements:
        final_boxes.append(box)
        final_scores.append(score)
        final_sources.append(source)
        final_groups.append(group)

    final_indices = conservative_nms_indices(final_boxes, final_scores, nms_iou)
    final_indices = same_instance_dedup_indices(
        final_boxes, final_scores, final_indices, final_groups
    )
    final_boxes = [final_boxes[index] for index in final_indices]
    final_scores = [final_scores[index] for index in final_indices]
    final_sources = [final_sources[index] for index in final_indices]
    final_groups = [final_groups[index] for index in final_indices]
    metadata = {
        "schema_version": "detector-instance-postprocess-v1",
        "nms_iou_threshold": float(nms_iou),
        "raw_candidate_count": len(boxes),
        "after_conservative_nms_count": initial_count,
        "same_instance_duplicates_removed": duplicate_count,
        "confirmed_instance_count": len(final_boxes),
        "local_refinement_attempted": attempted,
        "local_refinement_confirmed_splits": confirmed_splits,
        "local_refinement_confirmed_singletons": confirmed_singletons,
        "local_refinement_latency_ms": refinement_latency_ms,
        "aggregate_regions": aggregate_regions,
        "unresolved_regions": unresolved,
        "candidate_pool_complete": not unresolved,
        "final_source_ids": final_sources,
        "total_latency_ms": (time.perf_counter() - started) * 1000.0,
    }
    return final_boxes, final_scores, metadata
