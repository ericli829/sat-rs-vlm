"""Shared scorer batch result."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..types import LocatorError


@dataclass(frozen=True)
class ScoreBatch:
    name: str
    scores: tuple[float, ...]
    available: bool
    metadata: tuple[dict[str, Any], ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        if len(self.scores) != len(self.metadata):
            raise LocatorError(f"{self.name} scorer scores/metadata length mismatch")
        if not all(math.isfinite(float(score)) for score in self.scores):
            raise LocatorError(f"{self.name} scorer returned a non-finite score")

    @classmethod
    def unavailable(cls, name: str, count: int, reason: str) -> ScoreBatch:
        return cls(
            name=name,
            scores=(0.0,) * count,
            available=False,
            metadata=tuple({"available": False, "reason": reason} for _ in range(count)),
            reason=reason,
        )
