"""图像加载、Region 裁剪、送入 detector 前的 resize。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .runtime import ImageRef, Region
from .tiling import Tile


def load_image(image: ImageRef) -> Image.Image:
    path = Path(image.path)
    if not path.exists():
        raise FileNotFoundError(path)
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(path).convert("RGB")


def ensure_size(image: ImageRef, pil: Image.Image | None = None) -> ImageRef:
    if image.width and image.height:
        return image
    if pil is None:
        pil = load_image(image)
    w, h = pil.size
    return image.with_size(w, h)


def crop_tile(pil: Image.Image, tile: Tile) -> Image.Image:
    x0, y0, x1, y1 = (int(round(v)) for v in tile.crop_xyxy)
    x0 = max(0, min(x0, pil.width))
    x1 = max(0, min(x1, pil.width))
    y0 = max(0, min(y0, pil.height))
    y1 = max(0, min(y1, pil.height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty crop for tile {tile.tile_id}: {tile.crop_xyxy}")
    crop = pil.crop((x0, y0, x1, y1))
    return resize_for_detector(crop, tile)


def resize_for_detector(crop: Image.Image, tile: Tile) -> Image.Image:
    """把 tile 最长边缩放到 detector_input，Fine 尺度会对小 crop 上采样。"""
    target = int(tile.detector_input or 0)
    if target <= 0:
        return crop
    w, h = crop.size
    long_side = max(w, h)
    if long_side == target:
        return crop
    scale = target / float(long_side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    return crop.resize((nw, nh), resample)


def region_from_named(image: ImageRef, name: str) -> Region:
    """TOP / BOTTOM_LEFT / CENTER_RIGHT 等绝对位置区域。"""
    image = ensure_size(image)
    w = float(image.width or 0)
    h = float(image.height or 0)
    key = name.strip().upper().replace("-", "_").replace(" ", "_")
    thirds_x = (0.0, w / 3.0, 2.0 * w / 3.0, w)
    thirds_y = (0.0, h / 3.0, 2.0 * h / 3.0, h)
    halves_x = (0.0, w / 2.0, w)
    halves_y = (0.0, h / 2.0, h)
    table = {
        "TOP": (0.0, 0.0, w, h / 2.0),
        "BOTTOM": (0.0, h / 2.0, w, h),
        "LEFT": (0.0, 0.0, w / 2.0, h),
        "RIGHT": (w / 2.0, 0.0, w, h),
        "CENTER": (w / 4.0, h / 4.0, 3.0 * w / 4.0, 3.0 * h / 4.0),
        "TOP_LEFT": (halves_x[0], halves_y[0], halves_x[1], halves_y[1]),
        "TOP_RIGHT": (halves_x[1], halves_y[0], halves_x[2], halves_y[1]),
        "BOTTOM_LEFT": (halves_x[0], halves_y[1], halves_x[1], halves_y[2]),
        "BOTTOM_RIGHT": (halves_x[1], halves_y[1], halves_x[2], halves_y[2]),
        "TOP_CENTER": (thirds_x[1], 0.0, thirds_x[2], h / 2.0),
        "BOTTOM_CENTER": (thirds_x[1], h / 2.0, thirds_x[2], h),
        "CENTER_LEFT": (0.0, thirds_y[1], w / 2.0, thirds_y[2]),
        "CENTER_RIGHT": (w / 2.0, thirds_y[1], w, thirds_y[2]),
    }
    if key not in table:
        raise KeyError(f"unknown region name: {name}")
    return Region(
        image=image,
        bbox_xyxy_global=table[key],
        provenance={"region_id": key, "label": key, "position": key},
    )
