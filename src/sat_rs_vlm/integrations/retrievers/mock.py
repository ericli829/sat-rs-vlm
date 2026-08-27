"""Deterministic dependency-free retriever for locator tests and CLI smoke."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .protocol import RegionXYXY, RetrievalError, RetrievalResult


class MockRetrieverProvider:
    provider_name = "mock"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.model_id = str(self.config.get("model_id", "mock-retriever-v1"))
        self.calls: list[tuple[Path, str, int]] = []

    def score_regions(
        self,
        image_path: Path,
        query: str,
        regions_xyxy: Sequence[RegionXYXY],
    ) -> RetrievalResult:
        started = time.perf_counter()
        regions = list(regions_xyxy)
        self.calls.append((Path(image_path), query, len(regions)))
        configured = self.config.get("scores")
        if configured is not None:
            scores = [float(score) for score in configured]
            if len(scores) != len(regions):
                raise RetrievalError(
                    f"mock score length mismatch: {len(scores)} != {len(regions)}"
                )
        else:
            digest = hashlib.sha256(query.strip().lower().encode("utf-8")).digest()
            query_bias = int.from_bytes(digest[:2], "big") / 65535.0
            scores = []
            for index, box in enumerate(regions):
                values = [float(value) for value in box]
                if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
                    raise RetrievalError(f"invalid mock retrieval region at index {index}")
                center_signal = (values[0] + values[1] + values[2] + values[3]) % 997.0
                scores.append((0.7 * query_bias) + (0.3 * center_signal / 997.0))
        return RetrievalResult(
            scores=scores,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=self.provider_name,
            model_id=self.model_id,
            metadata={"deterministic": True, "batch_size": len(regions)},
        ).validate_length(len(regions))

    def close(self) -> None:
        return None
