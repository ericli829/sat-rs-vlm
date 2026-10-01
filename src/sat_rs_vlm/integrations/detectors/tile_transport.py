"""Lossless, inexpensive file transport for synchronous detector sidecars."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .protocol import ProposalError


def tile_image_format(value: str) -> str:
    image_format = str(value).strip().lower()
    if image_format not in {"bmp", "png"}:
        raise ProposalError("tile_image_format must be 'bmp' or 'png'")
    return image_format


def save_tile_image(image: Image.Image, path: Path, image_format: str) -> None:
    """BMP avoids compression work; PNG remains available for constrained storage."""

    if image.mode == "RGB":
        image.save(path, format=image_format.upper())
    else:
        converted = image.convert("RGB")
        try:
            converted.save(path, format=image_format.upper())
        finally:
            converted.close()
