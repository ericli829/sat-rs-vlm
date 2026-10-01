"""Old counting_system Detector interface backed by the current ProposalProvider."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from sat_rs_vlm.integrations.detectors.protocol import ProposalError, ProposalProvider
from sat_rs_vlm.integrations.detectors.tile_transport import (
    cpu_prepared_items,
    save_tile_image,
    tile_image_format,
)

from .bootstrap import ensure_counting_system_importable

ensure_counting_system_importable()

from counting_system.detector.base import DetectionRequest, DetectionResponse  # noqa: E402
from counting_system.geometry import local_to_global  # noqa: E402
from counting_system.runtime import Detection  # noqa: E402


@dataclass(frozen=True)
class _PreparedTile:
    path: Path
    phrase: str
    local_size: tuple[int, int]
    crop_xyxy: tuple[float, float, float, float]
    label: str
    tile_id: str
    scale_id: str


class CountingProposalDetectorBridge:
    """Tile crop → ProposalProvider.predict → original-image XYXY Detection.

    Counting System owns Global/Native/Fine tiling. The wrapped provider must
    be a non-tiled sidecar such as ``lae_dino_lae1m``.
    """

    name = "proposal_bridge"

    def __init__(
        self, provider: ProposalProvider, *, image_format: str = "bmp", prefetch: bool = True
    ) -> None:
        self._provider = provider
        self.provider_name = getattr(provider, "provider_name", "proposal")
        self.name = self.provider_name
        self.impl_name = self.provider_name
        self.tile_image_format = tile_image_format(image_format)
        self.tile_prefetch = prefetch
        self.call_count = 0

    def detect(self, request: DetectionRequest) -> DetectionResponse:
        with tempfile.TemporaryDirectory(prefix="counting_tile_") as temp_dir:
            tile_path = Path(temp_dir) / f"tile.{self.tile_image_format}"
            return self._predict_prepared(self._prepare(request, tile_path))

    def _prepare(self, request: DetectionRequest, tile_path: Path) -> _PreparedTile:
        phrase = request.texts or request.target.detection_phrase()
        save_tile_image(request.image, tile_path, self.tile_image_format)
        return _PreparedTile(
            tile_path,
            phrase,
            request.image.size,
            request.tile.crop_xyxy,
            request.target.name,
            request.tile.tile_id,
            request.tile.scale_id,
        )

    def detect_many(self, requests: Iterable[DetectionRequest]) -> Iterator[DetectionResponse]:
        """Prepare one CPU/file tile ahead; run sidecar inference strictly serially."""

        def indexed_requests():
            source = iter(requests)
            try:
                yield from enumerate(source)
            finally:
                close = getattr(source, "close", None)
                if callable(close):
                    close()

        with tempfile.TemporaryDirectory(prefix="counting_tiles_") as temp_dir:
            root = Path(temp_dir)

            def prepare(item: tuple[int, DetectionRequest]) -> _PreparedTile:
                index, request = item
                return self._prepare(request, root / f"tile_{index:05d}.{self.tile_image_format}")

            def release(tile: _PreparedTile) -> None:
                tile.path.unlink(missing_ok=True)

            with cpu_prepared_items(
                indexed_requests(), prepare, release, prefetch=self.tile_prefetch
            ) as prepared:
                for tile in prepared:
                    yield self._predict_prepared(tile)

    def _predict_prepared(self, tile: _PreparedTile) -> DetectionResponse:
        # Retain a scalar diagnostic, never requests that own full tile images.
        self.call_count += 1
        try:
            result = self._provider.predict(tile.path, tile.phrase)
        except ProposalError as exc:
            raise RuntimeError(f"LAE sidecar failure: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(
                f"Counting provider unavailable: LAE sidecar failure: {exc}"
            ) from exc
        local_w, local_h = tile.local_size
        detections: list[Detection] = []
        for box, score in zip(result.boxes_xyxy, result.scores, strict=True):
            global_box = local_to_global(box, tile.crop_xyxy, local_size=(local_w, local_h))
            detections.append(
                Detection(
                    bbox_xyxy_global=global_box,
                    score=float(score),
                    label=tile.label,
                    tile_id=tile.tile_id,
                    scale_id=tile.scale_id,
                    provenance={
                        "backend": self.impl_name,
                        "local_xyxy": [float(value) for value in box],
                        "crop_xyxy": list(tile.crop_xyxy),
                        "local_size": [local_w, local_h],
                        "coordinate_mode": "absolute_original_pixel_xyxy",
                        "model_id": result.model_id,
                    },
                )
            )
        return DetectionResponse(
            detections=detections,
            raw_count=len(detections),
            backend=self.impl_name,
            extra={"proposal_metadata": dict(result.metadata)},
        )

    def close(self) -> None:
        self._provider.close()
