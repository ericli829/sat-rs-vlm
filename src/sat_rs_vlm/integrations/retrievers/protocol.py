"""Dependency-light region retriever protocol and result validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

RegionXYXY = Sequence[float]


class RetrievalError(RuntimeError):
    """A retriever or retrieval payload violated the shared protocol."""


class RetrieverProvider(Protocol):
    provider_name: str

    def score_regions(
        self,
        image_path: Path,
        query: str,
        regions_xyxy: Sequence[RegionXYXY],
    ) -> RetrievalResult:
        """Return one finite relevance score per input region, preserving order."""

    def close(self) -> None:
        """Release long-lived model resources."""


@dataclass(frozen=True)
class RetrievalResult:
    scores: list[float]
    latency_ms: float
    provider: str
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical_scores = [float(score) for score in self.scores]
        if not all(math.isfinite(score) for score in canonical_scores):
            raise RetrievalError("retrieval scores must all be finite")
        latency = float(self.latency_ms)
        if not math.isfinite(latency) or latency < 0.0:
            raise RetrievalError("retrieval latency_ms must be finite and non-negative")
        if not str(self.provider).strip() or not str(self.model_id).strip():
            raise RetrievalError("retrieval provider and model_id must not be empty")
        object.__setattr__(self, "scores", canonical_scores)
        object.__setattr__(self, "latency_ms", latency)

    def validate_length(self, expected: int) -> RetrievalResult:
        if len(self.scores) != expected:
            raise RetrievalError(
                f"retrieval score length mismatch: {len(self.scores)} != {expected}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": list(self.scores),
            "latency_ms": self.latency_ms,
            "provider": self.provider,
            "model_id": self.model_id,
            "metadata": dict(self.metadata),
        }
