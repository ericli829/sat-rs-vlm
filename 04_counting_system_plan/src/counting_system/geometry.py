"""bbox 几何：IoU、NMS、中心归属、坐标裁剪。"""

from __future__ import annotations

from typing import Iterable, Sequence

from .runtime import BBox, Detection

BBoxLike = Sequence[float]


def clip_bbox(bbox: BBoxLike, width: float, height: float) -> BBox:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0 = min(max(x0, 0.0), width)
    y0 = min(max(y0, 0.0), height)
    x1 = min(max(x1, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def bbox_area(bbox: BBoxLike) -> float:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_center(bbox: BBoxLike) -> tuple[float, float]:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def point_in_bbox(point: tuple[float, float], bbox: BBoxLike, *, inclusive: bool = True) -> bool:
    x, y = point
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if inclusive:
        return x0 <= x <= x1 and y0 <= y <= y1
    return x0 <= x < x1 and y0 <= y < y1


def iou(a: BBoxLike, b: BBoxLike) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def local_to_global(
    local_xyxy: BBoxLike,
    crop_xyxy: BBoxLike,
    *,
    local_size: tuple[float, float] | None = None,
) -> BBox:
    """把 tile 内检测框映射回原图。local_size 为送进 detector 的宽高，可与 crop 不同。"""
    lx0, ly0, lx1, ly1 = (float(v) for v in local_xyxy)
    cx0, cy0, cx1, cy1 = (float(v) for v in crop_xyxy)
    crop_w = max(cx1 - cx0, 1e-6)
    crop_h = max(cy1 - cy0, 1e-6)
    if local_size is None:
        sx = sy = 1.0
    else:
        lw, lh = (float(v) for v in local_size)
        sx = crop_w / max(lw, 1e-6)
        sy = crop_h / max(lh, 1e-6)
    return (cx0 + lx0 * sx, cy0 + ly0 * sy, cx0 + lx1 * sx, cy0 + ly1 * sy)


def nms(detections: Sequence[Detection], iou_thr: float = 0.5) -> list[Detection]:
    ordered = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for det in ordered:
        if all(iou(det.bbox_xyxy_global, other.bbox_xyxy_global) < iou_thr for other in kept):
            kept.append(det)
    return kept


def greedy_match(
    coarse: Sequence[Detection],
    fine: Sequence[Detection],
    iou_thr: float,
) -> tuple[dict[int, list[int]], set[int]]:
    """coarse index -> matched fine indices。每个 fine 最多匹配一个 IoU 最大的 coarse。"""
    used_fine: set[int] = set()
    mapping: dict[int, list[int]] = {i: [] for i in range(len(coarse))}
    pairs: list[tuple[float, int, int]] = []
    for ci, cd in enumerate(coarse):
        for fi, fd in enumerate(fine):
            score = iou(cd.bbox_xyxy_global, fd.bbox_xyxy_global)
            if score >= iou_thr:
                pairs.append((score, ci, fi))
    pairs.sort(reverse=True)
    for _, ci, fi in pairs:
        if fi in used_fine:
            continue
        mapping[ci].append(fi)
        used_fine.add(fi)
    return mapping, used_fine


def unique_labels(items: Iterable[Detection]) -> list[str]:
    seen: list[str] = []
    for det in items:
        if det.label not in seen:
            seen.append(det.label)
    return seen
