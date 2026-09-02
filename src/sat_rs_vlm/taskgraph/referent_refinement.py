"""Conservative semantic refinement for detector-backed LOCATE candidates."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .choice_config import ChoiceSystemConfig
from .input_composer import InputComposer
from .providers import ChoiceScoringRequest, SemanticVLMProvider
from .runtime_types import Entity, EntitySet, ImageRef, Region
from .schema import TargetSpec


@dataclass(frozen=True)
class ReferentRefinementConfig:
    enabled: bool = False
    max_candidates: int = 8
    semantic_weight: float = 0.75
    geometry_weight: float = 0.25
    minimum_margin: float = 0.02
    candidate_halo_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ValueError("locate_refinement.max_candidates must be positive")
        if self.semantic_weight < 0.0 or self.geometry_weight < 0.0:
            raise ValueError("locate_refinement weights must be non-negative")
        if self.semantic_weight + self.geometry_weight <= 0.0:
            raise ValueError("locate_refinement weights must have a positive total")
        if self.minimum_margin < 0.0:
            raise ValueError("locate_refinement.minimum_margin must be non-negative")
        if not 0.0 <= self.candidate_halo_ratio <= 1.0:
            raise ValueError("locate_refinement.candidate_halo_ratio must be in [0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ReferentRefinementConfig:
        if value is None:
            return cls()
        return cls(
            enabled=bool(value.get("enabled", False)),
            max_candidates=int(value.get("max_candidates", 8)),
            semantic_weight=float(value.get("semantic_weight", 0.75)),
            geometry_weight=float(value.get("geometry_weight", 0.25)),
            minimum_margin=float(value.get("minimum_margin", 0.02)),
            candidate_halo_ratio=float(value.get("candidate_halo_ratio", 0.2)),
        )


@dataclass(frozen=True)
class ReferentRefinementResult:
    entities: EntitySet
    metadata: dict[str, Any]


class ReferentRefiner:
    """Use the existing semantic model only to verify bounded candidates."""

    _SPATIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("upper_right", re.compile(r"upper[- ]right|top[- ]right|north[- ]east|upper right")),
        ("upper_left", re.compile(r"upper[- ]left|top[- ]left|north[- ]west|upper left")),
        ("lower_right", re.compile(r"lower[- ]right|bottom[- ]right|south[- ]east|lower right")),
        ("lower_left", re.compile(r"lower[- ]left|bottom[- ]left|south[- ]west|lower left")),
        ("top", re.compile(r"\btop\b|\bupper\b|\bnorth\b")),
        ("bottom", re.compile(r"\bbottom\b|\blower\b|\bsouth\b")),
        ("left", re.compile(r"\bleft\b|\bwest\b")),
        ("right", re.compile(r"\bright\b|\beast\b")),
        ("center", re.compile(r"\bcenter\b|\bcentre\b|\bmiddle\b")),
    )

    def __init__(
        self,
        semantic: SemanticVLMProvider,
        composer: InputComposer,
        config: ReferentRefinementConfig | None = None,
        *,
        choice_config: ChoiceSystemConfig | None = None,
    ) -> None:
        self.semantic = semantic
        self.composer = composer
        self.config = config or ReferentRefinementConfig()
        self.choice_config = choice_config or ChoiceSystemConfig()
        self.provider_name = "referent_refiner"

    @staticmethod
    def _candidate_id(entity: Entity, index: int) -> str:
        value = entity.provenance.get("candidate_id")
        return str(value) if value is not None else f"candidate_{index + 1:04d}"

    @staticmethod
    def _normalize(values: Sequence[float]) -> list[float]:
        if not values:
            return []
        low = min(values)
        high = max(values)
        if high - low <= 1e-12:
            return [1.0] * len(values)
        return [(value - low) / (high - low) for value in values]

    @classmethod
    def _spatial_hint(cls, question: str) -> str | None:
        lowered = question.casefold()
        if "corner" in lowered:
            for name in ("upper_right", "upper_left", "lower_right", "lower_left"):
                if name in lowered.replace("-", "_"):
                    return name
            for name, pattern in cls._SPATIAL_PATTERNS[:4]:
                if pattern.search(lowered):
                    return name
        for name, pattern in cls._SPATIAL_PATTERNS:
            if pattern.search(lowered):
                return name
        return None

    @staticmethod
    def _geometry_prior(entity: Entity, question: str) -> float:
        hint = ReferentRefiner._spatial_hint(question)
        if hint is None:
            return 0.5
        image = entity.region.image
        width, height = ReferentRefiner._image_size(image)
        x1, y1, x2, y2 = entity.region.bbox_xyxy_global
        x = ((x1 + x2) / 2.0) / width
        y = ((y1 + y2) / 2.0) / height
        targets = {
            "upper_left": (0.0, 0.0),
            "upper_right": (1.0, 0.0),
            "lower_left": (0.0, 1.0),
            "lower_right": (1.0, 1.0),
            "top": (0.5, 0.0),
            "bottom": (0.5, 1.0),
            "left": (0.0, 0.5),
            "right": (1.0, 0.5),
            "center": (0.5, 0.5),
        }
        target_x, target_y = targets[hint]
        distance = math.hypot(x - target_x, y - target_y) / math.sqrt(2.0)
        return max(0.0, min(1.0, 1.0 - distance))

    @staticmethod
    def _image_size(image: ImageRef) -> tuple[float, float]:
        if image.width and image.height:
            return float(image.width), float(image.height)
        from PIL import Image

        with Image.open(image.path.resolve()) as source:
            return float(source.width), float(source.height)

    def _rank_for_budget(self, entities: tuple[Entity, ...], question: str) -> tuple[int, ...]:
        detector_scores = [
            float(entity.score) if entity.score is not None and math.isfinite(entity.score) else 0.0
            for entity in entities
        ]
        geometry_scores = [self._geometry_prior(entity, question) for entity in entities]
        detector_norm = self._normalize(detector_scores)
        weight_total = self.config.semantic_weight + self.config.geometry_weight
        ranking = [
            (
                (
                    self.config.semantic_weight * detector_norm[index]
                    + self.config.geometry_weight * geometry_scores[index]
                )
                / weight_total,
                index,
            )
            for index in range(len(entities))
        ]
        ranking.sort(key=lambda item: (-item[0], item[1]))
        return tuple(index for _, index in ranking[: self.config.max_candidates])

    def refine(
        self,
        candidates: EntitySet,
        *,
        question: str,
        target: TargetSpec,
        trigger_reason: str,
    ) -> ReferentRefinementResult:
        started = time.perf_counter()
        entities = tuple(candidates.entities)
        candidate_ids = [self._candidate_id(entity, index) for index, entity in enumerate(entities)]
        common = {
            "applied": False,
            "method": "none",
            "input_candidate_count": len(entities),
            "output_candidate_count": len(entities),
            "selected_candidate_ids": candidate_ids,
            "semantic_scores": {},
            "geometry_prior_scores": {
                candidate_id: self._geometry_prior(entity, question)
                for candidate_id, entity in zip(candidate_ids, entities, strict=True)
            },
            "detector_scores": {
                candidate_id: entity.score
                for candidate_id, entity in zip(candidate_ids, entities, strict=True)
            },
            "trigger_reason": trigger_reason,
            "resolution_status": "MULTIPLE_VALID" if len(entities) > 1 else "PRIMARY_RESOLVED",
            "latency_ms": 0.0,
        }
        if len(entities) <= 1:
            return ReferentRefinementResult(candidates, common)
        if not self.config.enabled:
            return ReferentRefinementResult(candidates, common)

        selected_indices = self._rank_for_budget(entities, question)
        selected_entities = tuple(entities[index] for index in selected_indices)
        selected_ids = [candidate_ids[index] for index in selected_indices]
        semantic_candidates = EntitySet(
            selected_entities,
            {
                **candidates.provenance,
                "referent_refinement_candidate_subset": True,
                "candidate_ids": selected_ids,
            },
        )
        model_input = self.composer.compose_named(
            {"candidates": semantic_candidates},
            question=(
                "You are verifying which supplied candidate matches the referring expression. "
                "Do not answer the original multiple-choice question. Do not search outside the "
                "supplied candidates. Select exactly one candidate only if the evidence supports "
                "it.\n\n"
                f"Original referring question: {question}\n"
                f"Target specification: category={target.category}; "
                f"attributes={dict(target.attributes)}"
            ),
            options=tuple(
                f"Candidate {chr(ord('A') + index)}: {candidate_id}"
                for index, candidate_id in enumerate(selected_ids)
            ),
        )
        mapping = model_input.metadata.get("candidate_mapping")
        if not isinstance(mapping, Mapping) or len(mapping) != len(selected_entities):
            common["failure_reason"] = "candidate_mapping_unavailable"
            common["resolution_status"] = "UNRESOLVED"
            common["latency_ms"] = (time.perf_counter() - started) * 1000.0
            output = EntitySet(
                selected_entities,
                {
                    **candidates.provenance,
                    "referent_refinement": common,
                    "fallback_required": True,
                    "fallback_kind": "semantic_candidate",
                },
            )
            return ReferentRefinementResult(output, common)
        choice_ids = tuple(str(choice_id) for choice_id in mapping)
        scorer = getattr(self.semantic, "reason_and_choose", None)
        try:
            if not callable(scorer):
                raise RuntimeError("semantic provider lacks reason_and_choose")
            scored = scorer(
                ChoiceScoringRequest(
                    model_input=model_input,
                    answer_type="CHOICE_SINGLE",
                    choice_ids=choice_ids,
                    option_texts=tuple(
                        str(item) for item in model_input.options
                    ),
                    single_choice_suffix=self.choice_config.single_choice_suffix,
                    multi_verify_template=self.choice_config.multi_verify_template,
                    multi_select_threshold=0.0,
                    purpose="referent_refinement",
                )
            )
            semantic_scores = {
                selected_ids[index]: float(scored.scores.get(choice_id, 0.0))
                for index, choice_id in enumerate(choice_ids)
            }
            semantic_norm = self._normalize(list(semantic_scores.values()))
            geometry_scores = [
                self._geometry_prior(entity, question) for entity in selected_entities
            ]
            weight_total = self.config.semantic_weight + self.config.geometry_weight
            fused_scores = [
                (
                    self.config.semantic_weight * semantic_norm[index]
                    + self.config.geometry_weight * geometry_scores[index]
                )
                / weight_total
                for index in range(len(selected_entities))
            ]
            order = sorted(
                range(len(selected_entities)),
                key=lambda index: (-fused_scores[index], index),
            )
            best, second = order[0], fused_scores[order[1]] if len(order) > 1 else -1.0
            margin = fused_scores[best] - second
            common.update(
                {
                    "applied": True,
                    "method": "semantic_2b_candidate_verification",
                    "input_candidate_count": len(entities),
                    "semantic_candidate_ids": selected_ids,
                    "semantic_scores": semantic_scores,
                    "geometry_prior_scores": {
                        candidate_id: geometry_scores[index]
                        for index, candidate_id in enumerate(selected_ids)
                    },
                    "fused_scores": {
                        candidate_id: fused_scores[index]
                        for index, candidate_id in enumerate(selected_ids)
                    },
                    "candidate_budget_indices": list(selected_indices),
                    "semantic_selected_ids": [
                        selected_ids[index]
                        for index, choice_id in enumerate(choice_ids)
                        if choice_id in scored.selected_ids
                    ],
                    "selected_candidate_ids": [selected_ids[best]],
                    "output_candidate_count": 1,
                    "margin": margin,
                    "resolution_status": (
                        "REFINED_RESOLVED"
                        if margin > self.config.minimum_margin
                        else "UNRESOLVED"
                    ),
                    "provider": scored.provider,
                    "model_id": scored.model_id,
                    "latency": dict(scored.latency_ms),
                }
            )
            if margin <= self.config.minimum_margin:
                common["failure_reason"] = "semantic_margin_below_threshold"
                output = EntitySet(
                    selected_entities,
                    {
                        **candidates.provenance,
                        "referent_refinement": common,
                        "fallback_required": True,
                        "fallback_kind": "semantic_candidate",
                    },
                )
            else:
                selected = selected_entities[best]
                output = EntitySet(
                    (Entity(
                        selected.region,
                        selected.label,
                        float(fused_scores[best]),
                        {
                            **selected.provenance,
                            "referent_refinement": {
                                "selected": True,
                                "candidate_id": selected_ids[best],
                                "semantic_score": semantic_scores[selected_ids[best]],
                                "geometry_prior_score": geometry_scores[best],
                                "fused_score": fused_scores[best],
                            },
                        },
                    ),),
                    {**candidates.provenance, "referent_refinement": common},
                )
        except Exception as exc:
            common.update(
                {
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "resolution_status": "UNRESOLVED",
                    "method": "semantic_2b_candidate_verification",
                }
            )
            output = EntitySet(
                selected_entities,
                {
                    **candidates.provenance,
                    "referent_refinement": common,
                    "fallback_required": True,
                    "fallback_kind": "semantic_candidate",
                },
            )
        common["latency_ms"] = (time.perf_counter() - started) * 1000.0
        output.provenance["referent_refinement"] = common
        return ReferentRefinementResult(output, common)

    def visual_fallback(
        self,
        scope: ImageRef | Region,
        *,
        question: str,
        target: TargetSpec,
        reason: str,
    ) -> ReferentRefinementResult:
        image = scope if isinstance(scope, ImageRef) else scope.image
        if isinstance(scope, Region):
            region = scope
        else:
            width, height = self._image_size(image)
            region = Region(
                image,
                (0.0, 0.0, width, height),
                {"semantic_fallback_scope": "image"},
            )
        metadata = {
            "applied": False,
            "method": "semantic_visual_fallback",
            "input_candidate_count": 0,
            "output_candidate_count": 1,
            "selected_candidate_ids": ["semantic_fallback_0001"],
            "semantic_scores": {},
            "geometry_prior_scores": {},
            "detector_scores": {},
            "trigger_reason": reason,
            "resolution_status": "SEMANTIC_FALLBACK_RESOLVED",
            "fallback_triggered": True,
            "fallback_reason": reason,
            "fallback_provider": "semantic_2b",
            "fallback_scope": "scoped_region" if isinstance(scope, Region) else "image",
            "question": question,
            "target_spec": {
                "category": target.category,
                "attributes": dict(target.attributes),
            },
            "latency_ms": 0.0,
        }
        output = EntitySet(
            (
                Entity(
                    region,
                    target.category,
                    0.0,
                    {
                        "candidate_id": "semantic_fallback_0001",
                        "fallback_required": True,
                        "fallback_kind": "semantic_visual",
                        "referent_refinement": metadata,
                    },
                ),
            ),
            {
                "provider": "semantic_2b",
                "capability": "semantic_visual_fallback",
                "fallback_required": True,
                "fallback_kind": "semantic_visual",
                "referent_refinement": metadata,
                "resolution_status": "SEMANTIC_FALLBACK_RESOLVED",
            },
        )
        return ReferentRefinementResult(output, metadata)
