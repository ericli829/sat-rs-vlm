"""Optional downstream answer-model boundary for Locator Multi-ROI output."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .types import BBox, LocatorError


@dataclass(frozen=True)
class MultiROIRequest:
    image_path: Path
    question: str
    regions_xyxy: tuple[BBox, ...]
    region_provenance: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise LocatorError("MultiROIRequest.question must not be empty")
        if not self.regions_xyxy:
            raise LocatorError("MultiROIRequest requires at least one region")
        if self.region_provenance and len(self.region_provenance) != len(
            self.regions_xyxy
        ):
            raise LocatorError("MultiROIRequest region/provenance length mismatch")


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    latency_ms: float
    provider: str
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise LocatorError("AnswerResult.latency_ms must be finite and non-negative")


class AnswerModel(Protocol):
    """Consume final global-coordinate Multi-ROI crops without owning localization."""

    provider_name: str

    def answer(self, request: MultiROIRequest) -> AnswerResult:
        """Answer one question using the selected original-image regions."""

    def close(self) -> None:
        """Release provider resources."""
