"""合成检测器：按像素颜色/已知框返回检测，用于 Fake E2E。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..geometry import local_to_global
from ..runtime import Detection
from .base import DetectionRequest, DetectionResponse


class FakeDetector:
    """不调用网络。图像中非背景色连通域，或 request.extra 注入的 boxes。"""

    name = "fake"

    def __init__(self, *, background: tuple[int, int, int] = (0, 0, 0), min_area: int = 4):
        self.background = background
        self.min_area = min_area

    def detect(self, request: DetectionRequest) -> DetectionResponse:
        image = request.image.convert("RGB")
        arr = np.asarray(image)
        bg = np.array(self.background, dtype=np.int16)
        mask = np.any(np.abs(arr.astype(np.int16) - bg) > 12, axis=2)
        boxes = _connected_boxes(mask, min_area=self.min_area)
        label = request.target.name
        local_w, local_h = image.size
        dets: list[Detection] = []
        for i, box in enumerate(boxes):
            global_box = local_to_global(box, request.tile.crop_xyxy, local_size=(local_w, local_h))
            dets.append(
                Detection(
                    bbox_xyxy_global=global_box,
                    score=0.99,
                    label=label,
                    tile_id=request.tile.tile_id,
                    scale_id=request.tile.scale_id,
                    provenance={
                        "backend": self.name,
                        "local_xyxy": list(box),
                        "crop_xyxy": list(request.tile.crop_xyxy),
                        "local_size": [local_w, local_h],
                    },
                )
            )
        return DetectionResponse(detections=dets, raw_count=len(dets), backend=self.name)

    def close(self) -> None:
        return None


def _connected_boxes(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            minx = maxx = x
            miny = maxy = y
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                minx = min(minx, cx)
                maxx = max(maxx, cx)
                miny = min(miny, cy)
                maxy = max(maxy, cy)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area >= min_area:
                boxes.append((minx, miny, maxx + 1, maxy + 1))
    return boxes
