"""Depth-pool-normalized, auditable multi-source score composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..types import LocatorError, SearchRegion
from .protocol import ScoreBatch


def depth_pool_normalize(scores: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(score) for score in scores)
    if not values:
        return ()
    if len(values) == 1:
        return (1.0,)
    low, high = min(values), max(values)
    if high == low:
        return (0.5,) * len(values)
    return tuple((value - low) / (high - low) for value in values)


@dataclass(frozen=True)
class CompositeScoreResult:
    scores: tuple[float, ...]
    components: tuple[dict[str, Any], ...]
    active_scorers: tuple[str, ...]


class CompositeRegionScorer:
    def __init__(self, weights: Mapping[str, Any] | None = None) -> None:
        configured = dict(weights or {})
        self.weights = {
            "detector": float(configured.get("detector", 1.0)),
            "retrieval": float(configured.get("retrieval", 1.0)),
            "spatial": float(configured.get("spatial", 0.5)),
            "parent": float(configured.get("parent", 0.25)),
            "redundancy": float(configured.get("redundancy", 0.2)),
        }
        if any(weight < 0.0 for weight in self.weights.values()):
            raise LocatorError("composite scorer weights must be non-negative")

    def score(
        self,
        regions: Sequence[SearchRegion],
        batches: Sequence[ScoreBatch],
        *,
        parent_scores: Sequence[float],
    ) -> CompositeScoreResult:
        count = len(regions)
        if len(parent_scores) != count:
            raise LocatorError("parent score length mismatch")
        for batch in batches:
            if len(batch.scores) != count:
                raise LocatorError(f"{batch.name} scorer output length mismatch")
        available = {
            batch.name: batch
            for batch in batches
            if batch.available and self.weights.get(batch.name, 0.0) > 0.0
        }
        normalized = {
            name: depth_pool_normalize(batch.scores) for name, batch in available.items()
        }
        active_weights = sum(self.weights[name] for name in available)
        if self.weights["parent"] > 0.0:
            active_weights += self.weights["parent"]
        if active_weights <= 0.0:
            active_weights = 1.0
        normalized_parent_scores = depth_pool_normalize(parent_scores)

        scores: list[float] = []
        details: list[dict[str, Any]] = []
        for index in range(count):
            parent_score = float(parent_scores[index])
            normalized_parent = normalized_parent_scores[index]
            weighted_sum = self.weights["parent"] * normalized_parent
            component_detail: dict[str, Any] = {
                "parent": {
                    "available": self.weights["parent"] > 0.0,
                    "raw": parent_score,
                    "normalized": normalized_parent,
                    "weight": self.weights["parent"],
                }
            }
            for batch in batches:
                is_available = batch.name in available
                normalized_score = normalized[batch.name][index] if is_available else None
                weight = self.weights.get(batch.name, 0.0)
                if is_available and normalized_score is not None:
                    weighted_sum += weight * normalized_score
                component_detail[batch.name] = {
                    "available": is_available,
                    "raw": float(batch.scores[index]),
                    "normalized": normalized_score,
                    "weight": weight,
                    "metadata": dict(batch.metadata[index]),
                    "reason": batch.reason,
                }
            fused = weighted_sum / active_weights
            component_detail["fused"] = {
                "score": fused,
                "active_weight_sum": active_weights,
                "redundancy_weight": self.weights["redundancy"],
            }
            scores.append(fused)
            details.append(component_detail)
        return CompositeScoreResult(
            scores=tuple(scores),
            components=tuple(details),
            active_scorers=tuple(sorted(available)),
        )


# Compatibility for external callers from the original per-parent implementation.
sibling_normalize = depth_pool_normalize
