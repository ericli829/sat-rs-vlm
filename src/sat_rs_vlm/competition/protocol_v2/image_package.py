"""Immutable manifest lookup and identity-cached asset verification."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pydantic import Field

from .errors import ImageHashMismatchError, ImageNotFoundError, ProtocolValidationError
from .models import ImageDescriptor, WireModel


class ManifestAsset(ImageDescriptor):
    size: int = Field(ge=0)


class Manifest(WireModel):
    format: str
    dataset_id: str = Field(min_length=1)
    assets: list[ManifestAsset]


@dataclass(frozen=True)
class ResolvedImage:
    path: Path
    width: int
    height: int
    descriptor: ImageDescriptor


class ImagePackage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        try:
            manifest = Manifest.model_validate(
                json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
            )
        except (OSError, ValueError) as exc:
            raise ProtocolValidationError("Invalid sealed image manifest") from exc
        if manifest.format != "rsu-sealed-images-v1":
            raise ProtocolValidationError("Unsupported image package format")
        self.dataset_id = manifest.dataset_id
        self.assets = {asset.asset_id: asset for asset in manifest.assets}
        if len(self.assets) != len(manifest.assets) or not self.assets:
            raise ProtocolValidationError("Duplicate or empty asset catalog")
        self._cache: dict[str, tuple[tuple[int, ...], int, int]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _identity(stat: os.stat_result) -> tuple[int, ...]:
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def resolve(self, descriptor: ImageDescriptor) -> ResolvedImage:
        asset = self.assets.get(descriptor.asset_id)
        if asset is None:
            raise ImageNotFoundError()
        if (asset.sha256, asset.mime_type) != (descriptor.sha256, descriptor.mime_type):
            raise ImageHashMismatchError()
        path = (self.root / "assets" / descriptor.asset_id).resolve()
        if path.parent != self.root / "assets":
            raise ImageNotFoundError()
        try:
            with self._lock, path.open("rb") as stream:
                identity = self._identity(os.fstat(stream.fileno()))
                if identity[2] != asset.size:
                    raise ImageHashMismatchError()
                cached = self._cache.get(asset.asset_id)
                if cached is None or cached[0] != identity:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                    if digest.hexdigest() != asset.sha256:
                        raise ImageHashMismatchError()
                    stream.seek(0)
                    with Image.open(stream) as image:
                        expected = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
                        if image.format != expected[asset.mime_type]:
                            raise ImageHashMismatchError()
                        width, height = image.size
                        image.verify()
                    if identity != self._identity(os.fstat(stream.fileno())):
                        raise ImageHashMismatchError()
                    cached = (identity, width, height)
                    self._cache[asset.asset_id] = cached
                return ResolvedImage(path, cached[1], cached[2], descriptor)
        except FileNotFoundError as exc:
            raise ImageNotFoundError() from exc
        except (OSError, SyntaxError) as exc:
            raise ImageHashMismatchError() from exc

    def resolve_all(self, descriptors: list[ImageDescriptor]) -> tuple[ResolvedImage, ...]:
        # Array order is temporal order. Manifest order and IDs do not reorder it.
        return tuple(self.resolve(item) for item in descriptors)
