"""Deterministic provider used by tests and protocol smoke runs."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .protocol import ProposalResult


class MockProposalProvider:
    provider_name = "mock"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.calls: list[tuple[Path, str]] = []

    def predict(self, image_path: Path, target_phrase: str) -> ProposalResult:
        started = time.perf_counter()
        self.calls.append((Path(image_path), target_phrase))
        result = ProposalResult(
            boxes_xyxy=[[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]],
            scores=[0.9, 0.8],
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=self.provider_name,
            model_id="mock-v1",
            metadata={"target_phrase": target_phrase, "deterministic": True},
        )
        return result

    def close(self) -> None:
        return None

