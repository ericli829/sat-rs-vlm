"""Pure global-coordinate geometry for core/halo search and scoring."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .types import BBox, LocatorError


def canonical_bbox(box: Sequence[float], *, label: str = "bbox") -> BBox:
    try:
        values = tuple(float(value) for value in box)
    except (TypeError, ValueError) as exc:
        raise LocatorError(f"{label} must contain four numeric values") from exc
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise LocatorError(f"{label} must contain four finite values")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise LocatorError(f"{label} must be a non-degenerate xyxy box")
    return values


def clamp_bbox(box: Sequence[float], image_width: int, image_height: int) -> BBox:
    if image_width < 1 or image_height < 1:
        raise LocatorError("image dimensions must be positive")
    values = canonical_bbox(box)
    clamped = (
        min(max(values[0], 0.0), float(image_width)),
        min(max(values[1], 0.0), float(image_height)),
        min(max(values[2], 0.0), float(image_width)),
        min(max(values[3], 0.0), float(image_height)),
    )
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        raise LocatorError("bbox is outside image bounds after clamping")
    return clamped


def bbox_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = canonical_bbox(box)
    return (x2 - x1) * (y2 - y1)


def intersection_area(left: Sequence[float], right: Sequence[float]) -> float:
    a = canonical_bbox(left, label="left bbox")
    b = canonical_bbox(right, label="right bbox")
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = intersection_area(left, right)
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0.0 else 0.0


def bbox_coverage(box: Sequence[float], region: Sequence[float]) -> float:
    """Fraction of an evidence box covered by a candidate region."""

    return intersection_area(box, region) / bbox_area(box)


def expand_with_halo(
    core_xyxy: Sequence[float],
    halo_ratio: float,
    image_width: int,
    image_height: int,
) -> BBox:
    if halo_ratio < 0.0:
        raise LocatorError("halo_ratio must be non-negative")
    x1, y1, x2, y2 = canonical_bbox(core_xyxy, label="core bbox")
    halo_x = (x2 - x1) * halo_ratio
    halo_y = (y2 - y1) * halo_ratio
    return clamp_bbox(
        (x1 - halo_x, y1 - halo_y, x2 + halo_x, y2 + halo_y),
        image_width,
        image_height,
    )


def subdivide_core(core_xyxy: Sequence[float], grid_size: int) -> tuple[BBox, ...]:
    if grid_size < 2:
        raise LocatorError("grid_size must be at least 2")
    x1, y1, x2, y2 = canonical_bbox(core_xyxy, label="parent core bbox")
    x_edges = [x1 + (x2 - x1) * index / grid_size for index in range(grid_size + 1)]
    y_edges = [y1 + (y2 - y1) * index / grid_size for index in range(grid_size + 1)]
    return tuple(
        (x_edges[column], y_edges[row], x_edges[column + 1], y_edges[row + 1])
        for row in range(grid_size)
        for column in range(grid_size)
    )


def spatial_prior(
    region_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    scope: str,
) -> float:
    x1, y1, x2, y2 = clamp_bbox(region_xyxy, image_width, image_height)
    x = ((x1 + x2) / 2.0) / image_width
    y = ((y1 + y2) / 2.0) / image_height
    if scope in {"left", "west"}:
        return 1.0 - x
    if scope in {"right", "east"}:
        return x
    if scope in {"upper", "north"}:
        return 1.0 - y
    if scope in {"lower", "south"}:
        return y
    if scope == "upper_left":
        return (1.0 - x) * (1.0 - y)
    if scope == "upper_right":
        return x * (1.0 - y)
    if scope == "lower_left":
        return (1.0 - x) * y
    if scope == "lower_right":
        return x * y
    if scope in {"center_left", "center_right"}:
        vertical_center = max(0.0, 1.0 - abs(y - 0.5) / 0.5)
        horizontal = 1.0 - x if scope == "center_left" else x
        return horizontal * vertical_center
    if scope == "center":
        distance = math.hypot(x - 0.5, y - 0.5) / math.hypot(0.5, 0.5)
        return max(0.0, 1.0 - distance)
    if scope == "global":
        return 0.5
    raise LocatorError(f"unsupported spatial scope: {scope!r}")


def rectangle_union_area(boxes: Iterable[Sequence[float]]) -> float:
    canonical = [canonical_bbox(box) for box in boxes]
    if not canonical:
        return 0.0
    x_edges = sorted({value for box in canonical for value in (box[0], box[2])})
    total = 0.0
    for left, right in zip(x_edges, x_edges[1:], strict=False):
        if right <= left:
            continue
        intervals = sorted(
            (box[1], box[3]) for box in canonical if box[0] < right and box[2] > left
        )
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for interval_start, interval_end in intervals[1:]:
            if interval_start > end:
                covered += end - start
                start, end = interval_start, interval_end
            else:
                end = max(end, interval_end)
        covered += end - start
        total += (right - left) * covered
    return total
