"""Adaptive cumulative-mass beam selection and configurable stop policy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .geometry import bbox_iou
from .types import LocatorError, SearchRegion


def standardized_logits(scores: Sequence[float], *, epsilon: float = 1e-8) -> tuple[float, ...]:
    """Map one depth's fused scores to a stable softmax logit scale."""

    values = tuple(float(score) for score in scores)
    if not all(math.isfinite(value) for value in values):
        raise LocatorError("beam scores must all be finite")
    if not values:
        return ()
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    standard_deviation = math.sqrt(max(variance, 0.0))
    if standard_deviation <= epsilon:
        return (0.0,) * len(values)
    return tuple((value - mean) / standard_deviation for value in values)


def softmax_probabilities(logits: Sequence[float], temperature: float) -> tuple[float, ...]:
    if temperature <= 0.0:
        raise LocatorError("beam temperature must be positive")
    if not logits:
        return ()
    scaled = [float(logit) / temperature for logit in logits]
    if not all(math.isfinite(value) for value in scaled):
        raise LocatorError("beam logits must all be finite")
    maximum = max(scaled)
    exponentials = [math.exp(score - maximum) for score in scaled]
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


@dataclass(frozen=True)
class BeamSelection:
    selected_indices: tuple[int, ...]
    standardized_logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    cumulative_probability: float
    redundancy_penalties: tuple[float, ...]
    effective_scores: tuple[float, ...]
    entropy: float


def adaptive_beam_select(
    regions: Sequence[SearchRegion],
    scores: Sequence[float],
    *,
    temperature: float,
    cumulative_mass: float,
    max_beam: int,
    redundancy_weight: float,
) -> BeamSelection:
    if len(regions) != len(scores) or not regions:
        raise LocatorError("adaptive beam requires equally sized non-empty regions and scores")
    if not 0.0 < cumulative_mass <= 1.0:
        raise LocatorError("cumulative_mass must be in (0, 1]")
    if max_beam < 1:
        raise LocatorError("max_beam must be positive")
    if redundancy_weight < 0.0:
        raise LocatorError("redundancy_weight must be non-negative")
    logits = standardized_logits(scores)
    probabilities = softmax_probabilities(logits, temperature)
    remaining = set(range(len(regions)))
    selected: list[int] = []
    penalties = [0.0] * len(regions)
    effective = list(logits)
    logit_span = max(max(logits) - min(logits), 1.0)
    cumulative = 0.0
    while remaining and len(selected) < min(max_beam, len(regions)):
        for index in remaining:
            overlap = max(
                (
                    bbox_iou(regions[index].core_xyxy, regions[chosen].core_xyxy)
                    for chosen in selected
                ),
                default=0.0,
            )
            penalties[index] = redundancy_weight * overlap * logit_span
            effective[index] = logits[index] - penalties[index]
        chosen = max(remaining, key=lambda index: (effective[index], logits[index], -index))
        selected.append(chosen)
        remaining.remove(chosen)
        cumulative += probabilities[chosen]
        if cumulative >= cumulative_mass:
            break
    return BeamSelection(
        selected_indices=tuple(selected),
        standardized_logits=logits,
        probabilities=probabilities,
        cumulative_probability=cumulative,
        redundancy_penalties=tuple(penalties),
        effective_scores=tuple(effective),
        entropy=-sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        ),
    )


@dataclass(frozen=True)
class StopPolicyConfig:
    target_view_size: int = 1280
    max_depth: int = 3
    max_regions: int = 64
    max_processed_area_ratio: float = 3.0

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> StopPolicyConfig:
        values = dict(config or {})
        result = cls(
            target_view_size=int(values.get("target_view_size", cls.target_view_size)),
            max_depth=int(values.get("max_depth", cls.max_depth)),
            max_regions=int(values.get("max_regions", cls.max_regions)),
            max_processed_area_ratio=float(
                values.get(
                    "max_processed_area_ratio",
                    values.get("max_area_ratio", cls.max_processed_area_ratio),
                )
            ),
        )
        if result.target_view_size < 1 or result.max_depth < 1 or result.max_regions < 1:
            raise LocatorError("stop-policy size, depth, and region limits must be positive")
        if result.max_processed_area_ratio <= 0.0:
            raise LocatorError("stop-policy processed-area limit must be positive")
        return result


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reasons: tuple[str, ...]


def evaluate_stop(
    region: SearchRegion,
    config: StopPolicyConfig,
    *,
    evaluated_regions: int,
    processed_area_ratio: float,
) -> StopDecision:
    width = region.core_xyxy[2] - region.core_xyxy[0]
    height = region.core_xyxy[3] - region.core_xyxy[1]
    reasons: list[str] = []
    if max(width, height) <= config.target_view_size:
        reasons.append("target_view_size")
    if region.depth >= config.max_depth:
        reasons.append("max_depth")
    if evaluated_regions >= config.max_regions:
        reasons.append("max_regions")
    if processed_area_ratio >= config.max_processed_area_ratio:
        reasons.append("max_processed_area_ratio")
    return StopDecision(stop=bool(reasons), reasons=tuple(reasons))
