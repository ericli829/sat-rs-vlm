"""Generic overlapping-tile wrapper for tiny-object proposal providers."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from .protocol import (
    ProposalError,
    ProposalProvider,
    ProposalResult,
    canonicalize_proposals,
)


def tile_starts(length: int, tile_size: int, overlap_ratio: float) -> tuple[int, ...]:
    """Return deterministic starts that cover an axis without a tiny edge tile."""

    if length < 1 or tile_size < 1:
        raise ProposalError("image length and tile_size must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ProposalError("tile overlap_ratio must be in [0, 1)")
    if length <= tile_size:
        return (0,)
    step = max(1, round(tile_size * (1.0 - overlap_ratio)))
    starts = list(range(0, length - tile_size + 1, step))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(dict.fromkeys(starts))


def generate_tiles(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap_ratio: float,
) -> tuple[tuple[int, int, int, int], ...]:
    x_starts = tile_starts(image_width, tile_size, overlap_ratio)
    y_starts = tile_starts(image_height, tile_size, overlap_ratio)
    return tuple(
        (
            x,
            y,
            min(x + tile_size, image_width),
            min(y + tile_size, image_height),
        )
        for y in y_starts
        for x in x_starts
    )


def _iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def global_nms(
    boxes: list[list[float]],
    scores: list[float],
    threshold: float,
    *,
    top_k: int | None = None,
) -> tuple[list[list[float]], list[float], list[int]]:
    if not 0.0 <= threshold <= 1.0:
        raise ProposalError("global_nms_iou must be between 0 and 1")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    keep: list[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        if top_k is not None and len(keep) >= top_k:
            break
        order = [
            index for index in order if _iou(boxes[current], boxes[index]) <= threshold
        ]
    return (
        [boxes[index] for index in keep],
        [scores[index] for index in keep],
        keep,
    )


class TiledProposalProvider:
    """Run an unaware base provider on overlapping crops and deduplicate globally."""

    provider_name = "tiled"

    def __init__(
        self,
        base_provider: ProposalProvider,
        config: Mapping[str, Any],
        *,
        base_provider_name: str,
    ) -> None:
        self.base_provider = base_provider
        self.base_provider_name = base_provider_name
        self.config = dict(config)
        self.tile_size = int(self.config.get("tile_size", 1333))
        self.overlap_ratio = float(self.config.get("overlap_ratio", 0.15))
        self.global_nms_iou = float(self.config.get("global_nms_iou", 0.4))
        top_k_value = self.config.get("global_top_k")
        self.global_top_k = int(top_k_value) if top_k_value is not None else None
        if self.tile_size < 1:
            raise ProposalError("tiled detector tile_size must be positive")
        if not 0.0 <= self.overlap_ratio < 1.0:
            raise ProposalError("tiled detector overlap_ratio must be in [0, 1)")
        if not 0.0 <= self.global_nms_iou <= 1.0:
            raise ProposalError("tiled detector global_nms_iou must be in [0, 1]")
        if self.global_top_k is not None and self.global_top_k < 1:
            raise ProposalError("tiled detector global_top_k must be positive")
        tiling_identity = {
            "tile_size": self.tile_size,
            "overlap_ratio": self.overlap_ratio,
            "global_nms_iou": self.global_nms_iou,
            "global_top_k": self.global_top_k,
        }
        encoded = json.dumps(tiling_identity, sort_keys=True, separators=(",", ":"))
        self.tiling_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self.model_id = (
            f"tiled:{getattr(base_provider, 'model_id', base_provider_name)}:"
            f"{self.tiling_sha256[:12]}"
        )
        self.model_identity = {
            "provider": self.provider_name,
            "base_provider": self.base_provider_name,
            "base_model_identity": getattr(base_provider, "model_identity", None),
            "base_model_id": str(
                getattr(base_provider, "model_id", self.base_provider_name)
            ),
            "tiling": tiling_identity,
            "tiling_sha256": self.tiling_sha256,
        }

    def predict(self, image_path: Path, target_phrase: str) -> ProposalResult:
        resolved_image = Path(image_path).expanduser().resolve()
        if not resolved_image.is_file():
            raise ProposalError(f"tiled detector image does not exist: {resolved_image}")
        started = time.perf_counter()
        raw_boxes: list[list[float]] = []
        raw_scores: list[float] = []
        raw_records: list[dict[str, Any]] = []
        tile_records: list[dict[str, Any]] = []
        base_latency_ms = 0.0
        with Image.open(resolved_image) as source:
            image = source.convert("RGB")
            tiles = generate_tiles(
                image.width,
                image.height,
                self.tile_size,
                self.overlap_ratio,
            )
            with tempfile.TemporaryDirectory(prefix="uhr_tiled_detector_") as temporary:
                temporary_root = Path(temporary)
                for tile_index, (x1, y1, x2, y2) in enumerate(tiles):
                    tile_path = temporary_root / f"tile_{tile_index:05d}.png"
                    image.crop((x1, y1, x2, y2)).save(tile_path)
                    result = self.base_provider.predict(tile_path, target_phrase)
                    base_latency_ms += result.latency_ms
                    local_boxes, local_scores, validation = canonicalize_proposals(
                        result.boxes_xyxy,
                        result.scores,
                        image_width=x2 - x1,
                        image_height=y2 - y1,
                    )
                    tile_records.append(
                        {
                            "tile_id": tile_index,
                            "tile_xyxy": [x1, y1, x2, y2],
                            "raw_proposal_count": len(result.boxes_xyxy),
                            "valid_proposal_count": len(local_boxes),
                            "base_latency_ms": result.latency_ms,
                            "base_provider": result.provider,
                            "base_model_id": result.model_id,
                            "validation": validation,
                        }
                    )
                    for local_box, score in zip(local_boxes, local_scores, strict=True):
                        global_box = [
                            local_box[0] + x1,
                            local_box[1] + y1,
                            local_box[2] + x1,
                            local_box[3] + y1,
                        ]
                        raw_index = len(raw_boxes)
                        raw_boxes.append(global_box)
                        raw_scores.append(score)
                        raw_records.append(
                            {
                                "raw_index": raw_index,
                                "tile_id": tile_index,
                                "local_box_xyxy": local_box,
                                "global_box_xyxy": global_box,
                                "score": score,
                            }
                        )
        boxes, scores, keep = global_nms(
            raw_boxes,
            raw_scores,
            self.global_nms_iou,
            top_k=self.global_top_k,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ProposalResult(
            boxes_xyxy=boxes,
            scores=scores,
            latency_ms=latency_ms,
            provider=self.provider_name,
            model_id=self.model_id,
            metadata={
                "schema_version": "uhr-tiled-proposals-v1",
                "coordinate_mode": "absolute_original_pixel_xyxy",
                "target_phrase": target_phrase,
                "base_provider": self.base_provider_name,
                "base_model_id": str(
                    getattr(self.base_provider, "model_id", self.base_provider_name)
                ),
                "model_identity": self.model_identity,
                "tile_size": self.tile_size,
                "overlap_ratio": self.overlap_ratio,
                "global_nms_iou": self.global_nms_iou,
                "tile_count": len(tile_records),
                "tiles": tile_records,
                "raw_proposal_count": len(raw_boxes),
                "deduplicated_proposal_count": len(boxes),
                "raw_proposals": raw_records,
                "kept_raw_indices": keep,
                "base_latency_ms": base_latency_ms,
                "wrapper_latency_ms": latency_ms,
            },
        )

    def close(self) -> None:
        self.base_provider.close()
