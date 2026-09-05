"""同尺度 Core Ownership + NMS，以及跨尺度融合。"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from .geometry import greedy_match, iou, nms, point_in_bbox
from .runtime import Detection, DetectionSet
from .tiling import Tile


def apply_core_ownership(detections: Sequence[Detection], tiles: Sequence[Tile]) -> list[Detection]:
    by_id = {tile.tile_id: tile for tile in tiles}
    kept: list[Detection] = []
    for det in detections:
        tile = by_id.get(det.tile_id)
        if tile is None:
            kept.append(det)
            continue
        if point_in_bbox(det.center, tile.core_xyxy, inclusive=True):
            kept.append(det)
    return kept


def same_scale_dedup(
    detections: Sequence[Detection],
    tiles: Sequence[Tile],
    *,
    nms_iou: float = 0.5,
) -> list[Detection]:
    owned = apply_core_ownership(detections, tiles)
    by_scale: dict[str, list[Detection]] = defaultdict(list)
    for det in owned:
        by_scale[det.scale_id].append(det)
    out: list[Detection] = []
    for group in by_scale.values():
        out.extend(nms(group, iou_thr=nms_iou))
    return out


def cross_scale_fusion(
    detections: Sequence[Detection],
    *,
    iou_thr: float = 0.5,
    prefer: Sequence[str] = ("fine", "native", "global"),
) -> list[Detection]:
    """
    保守规则：
    - coarse↔fine 一一对应 → 保留更细检测
    - 一个 coarse 内有多个 fine → coarse 可能是 aggregate，丢 coarse
    - coarse 无 fine match → 保留 coarse
    """
    by_scale: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        by_scale[det.scale_id].append(det)
    present = [s for s in prefer if by_scale.get(s)]
    extra = [s for s in by_scale if s not in present]
    order = present + extra
    if len(order) <= 1:
        return list(detections)

    current = list(by_scale[order[0]])
    for coarse_name in order[1:]:
        coarse = list(by_scale[coarse_name])
        mapping, matched_fine = greedy_match(coarse, current, iou_thr)
        fused: list[Detection] = []
        for ci, cd in enumerate(coarse):
            fine_ids = mapping.get(ci) or []
            if len(fine_ids) == 1:
                # 1-1：保留更细
                continue
            if len(fine_ids) >= 2:
                # 1-N：丢 coarse，保留 fine
                continue
            fused.append(cd)
        for fi, fd in enumerate(current):
            fused.append(fd)
            _ = matched_fine  # fine 全部保留
        current = fused
    return current


def fuse_detections(
    raw: Iterable[Detection],
    tiles: Sequence[Tile],
    *,
    score_threshold: float,
    nms_iou: float,
    cross_iou: float,
) -> tuple[list[Detection], dict]:
    raw_list = list(raw)
    dropped_by_score = [d for d in raw_list if d.score < score_threshold]
    survivors = [d for d in raw_list if d.score >= score_threshold]
    after_own = apply_core_ownership(survivors, tiles)
    after_same = same_scale_dedup(survivors, tiles, nms_iou=nms_iou)
    fused = cross_scale_fusion(after_same, iou_thr=cross_iou)
    stats = {
        "raw": len(raw_list),
        "below_threshold": len(dropped_by_score),
        "after_threshold": len(survivors),
        "after_ownership": len(after_own),
        "after_same_scale": len(after_same),
        "after_fusion": len(fused),
        "score_threshold": score_threshold,
        "nms_iou": nms_iou,
        "cross_scale_iou": cross_iou,
    }
    return fused, stats


def duplicate_rate(detections: Sequence[Detection], iou_thr: float = 0.5) -> float:
    if len(detections) < 2:
        return 0.0
    dup = 0
    for i, a in enumerate(detections):
        for b in detections[i + 1 :]:
            if iou(a.bbox_xyxy_global, b.bbox_xyxy_global) >= iou_thr:
                dup += 1
                break
    return dup / len(detections)


def empty_detection_set() -> DetectionSet:
    return DetectionSet([])
