"""Adaptive cumulative-mass beam selection and configurable stop policy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .geometry import bbox_iou
from .types import LocatorError, SearchRegion


def softmax_probabilities(scores: Sequence[float], temperature: float) -> tuple[float, ...]:
    if temperature <= 0.0:
        raise LocatorError("beam temperature must be positive")
    if not scores:
        return ()
    scaled = [float(score) / temperature for score in scores]
    maximum = max(scaled)
    exponentials = [math.exp(score - maximum) for score in scaled]
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


@dataclass(frozen=True)
class BeamSelection:
    selected_indices: tuple[int, ...]
    probabilities: tuple[float, ...]
    cumulative_probability: float
    redundancy_penalties: tuple[float, ...]
    effective_scores: tuple[float, ...]


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
    probabilities = softmax_probabilities(scores, temperature)
    remaining = set(range(len(regions)))
    selected: list[int] = []
    penalties = [0.0] * len(regions)
    effective = [float(score) for score in scores]
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
            penalties[index] = redundancy_weight * overlap
            effective[index] = float(scores[index]) - penalties[index]
        chosen = max(remaining, key=lambda index: (effective[index], scores[index], -index))
        selected.append(chosen)
        remaining.remove(chosen)
        cumulative += probabilities[chosen]
        if cumulative >= cumulative_mass:
            break
    return BeamSelection(
        selected_indices=tuple(selected),
        probabilities=probabilities,
        cumulative_probability=cumulative,
        redundancy_penalties=tuple(penalties),
        effective_scores=tuple(effective),
    )


@dataclass(frozen=True)
class StopPolicyConfig:
    target_view_size: int = 1280
    max_depth: int = 3
    min_score_gain: float = 0.01
    max_regions: int = 64
    max_area_ratio: float = 3.0
    posterior_stop_threshold: float = 0.985

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> StopPolicyConfig:
        values = dict(config or {})
        result = cls(
            target_view_size=int(values.get("target_view_size", cls.target_view_size)),
            max_depth=int(values.get("max_depth", cls.max_depth)),
            min_score_gain=float(values.get("min_score_gain", cls.min_score_gain)),
            max_regions=int(values.get("max_regions", cls.max_regions)),
            max_area_ratio=float(values.get("max_area_ratio", cls.max_area_ratio)),
            posterior_stop_threshold=float(
                values.get("posterior_stop_threshold", cls.posterior_stop_threshold)
            ),
        )
        if result.target_view_size < 1 or result.max_depth < 1 or result.max_regions < 1:
            raise LocatorError("stop-policy size, depth, and region limits must be positive")
        if result.min_score_gain < 0.0 or result.max_area_ratio <= 0.0:
            raise LocatorError("stop-policy gain/area limits are invalid")
        if not 0.0 < result.posterior_stop_threshold <= 1.0:
            raise LocatorError("posterior_stop_threshold must be in (0, 1]")
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
    inspected_area_ratio: float,
    score_gain: float | None,
    posterior_max: float | None,
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
    if inspected_area_ratio >= config.max_area_ratio:
        reasons.append("max_area_ratio")
    if score_gain is not None and score_gain < config.min_score_gain:
        reasons.append("min_score_gain")
    if posterior_max is not None and posterior_max >= config.posterior_stop_threshold:
        reasons.append("posterior_concentrated")
    return StopDecision(stop=bool(reasons), reasons=tuple(reasons))
