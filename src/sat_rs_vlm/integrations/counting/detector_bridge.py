"""Old counting_system Detector interface backed by the current ProposalProvider."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from sat_rs_vlm.integrations.detectors.protocol import ProposalError, ProposalProvider

from .bootstrap import ensure_counting_system_importable

ensure_counting_system_importable()

from counting_system.detector.base import DetectionRequest, DetectionResponse  # noqa: E402
from counting_system.geometry import local_to_global  # noqa: E402
from counting_system.runtime import Detection  # noqa: E402


class CountingProposalDetectorBridge:
    """Tile crop → ProposalProvider.predict → original-image XYXY Detection.

    Counting System owns Global/Native/Fine tiling. The wrapped provider must
    be a non-tiled sidecar such as ``lae_dino_lae1m``.
    """

    name = "proposal_bridge"

    def __init__(self, provider: ProposalProvider) -> None:
        self._provider = provider
        self.provider_name = getattr(provider, "provider_name", "proposal")
        self.name = self.provider_name
        self.impl_name = self.provider_name
        self.calls: list[Any] = []

    def detect(self, request: DetectionRequest) -> DetectionResponse:
        self.calls.append(request)
        phrase = request.texts or request.target.phrase()
        local_w, local_h = request.image.size
        with tempfile.TemporaryDirectory(prefix="counting_tile_") as temp_dir:
            tile_path = Path(temp_dir) / "tile.png"
            request.image.convert("RGB").save(tile_path)
            try:
                result = self._provider.predict(tile_path, phrase)
            except ProposalError as exc:
                raise RuntimeError(f"LAE sidecar failure: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Counting provider unavailable: LAE sidecar failure: {exc}"
                ) from exc
        detections: list[Detection] = []
        for box, score in zip(result.boxes_xyxy, result.scores, strict=True):
            global_box = local_to_global(
                box, request.tile.crop_xyxy, local_size=(local_w, local_h)
            )
            detections.append(
                Detection(
                    bbox_xyxy_global=global_box,
                    score=float(score),
                    label=request.target.name,
                    tile_id=request.tile.tile_id,
                    scale_id=request.tile.scale_id,
                    provenance={
                        "backend": self.impl_name,
                        "local_xyxy": [float(value) for value in box],
                        "crop_xyxy": list(request.tile.crop_xyxy),
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
