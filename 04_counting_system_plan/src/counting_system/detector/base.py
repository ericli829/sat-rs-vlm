from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from ..runtime import Detection
from ..target import TargetSpec
from ..tiling import Tile


@dataclass(slots=True)
class DetectionRequest:
    image: Image.Image
    target: TargetSpec
    tile: Tile
    score_threshold: float = 0.0
    texts: str = ""


@dataclass(slots=True)
class DetectionResponse:
    detections: list[Detection] = field(default_factory=list)
    raw_count: int = 0
    backend: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    name: str

    def detect(self, request: DetectionRequest) -> DetectionResponse: ...

    def close(self) -> None: ...
