"""检测框 overlay。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .runtime import CountResult, Detection, ImageRef
from .image_ops import load_image


def draw_overlay(
    image: ImageRef | Image.Image,
    detections: list[Detection],
    *,
    title: str = "",
    count: int | None = None,
) -> Image.Image:
    pil = image if isinstance(image, Image.Image) else load_image(image)
    canvas = pil.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    colors = [(255, 64, 64), (64, 220, 120), (64, 140, 255), (255, 200, 40)]
    scale_index = {"global": 0, "native": 1, "fine": 2}
    for i, det in enumerate(detections):
        color = colors[scale_index.get(det.scale_id, i % len(colors))]
        x0, y0, x1, y1 = det.bbox_xyxy_global
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, canvas.width // 800))
        label = f"{det.label} {det.score:.2f} [{det.scale_id}]"
        ty = max(0, y0 - 12)
        draw.text((x0 + 2, ty), label, fill=color, font=font)
    header = title or ""
    if count is not None:
        header = f"{header} count={count}".strip()
    if header:
        draw.rectangle([0, 0, canvas.width, 22], fill=(0, 0, 0))
        draw.text((6, 4), header, fill=(255, 255, 255), font=font)
    return canvas


def save_overlay(
    image: ImageRef | Image.Image,
    result: CountResult,
    path: str | Path,
    *,
    title: str = "",
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas = draw_overlay(image, list(result.detections), title=title, count=result.count)
    canvas.save(dest)
    return dest
