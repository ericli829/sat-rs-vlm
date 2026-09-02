"""合成 UHR 图：已知框，供 Fake E2E / 去重 / 映射测试。"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

from .runtime import BBox, ImageRef


@dataclass(slots=True)
class Blob:
    bbox: BBox
    color: tuple[int, int, int] = (255, 32, 32)


def write_blob_image(
    path: str,
    *,
    width: int,
    height: int,
    blobs: list[Blob],
    background: tuple[int, int, int] = (0, 0, 0),
) -> ImageRef:
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    for blob in blobs:
        x0, y0, x1, y1 = (int(v) for v in blob.bbox)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=blob.color)
    image.save(path)
    return ImageRef(path=path, image_id="synth", width=width, height=height)
