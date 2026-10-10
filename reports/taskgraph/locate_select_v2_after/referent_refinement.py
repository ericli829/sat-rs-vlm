"""Conservative semantic refinement for detector-backed LOCATE candidates."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .choice_config import ChoiceSystemConfig
from .input_composer import InputComposer
from .providers import ChoiceScoringRequest, SemanticVLMProvider, VLMRequest
from .runtime_types import Entity, EntitySet, ImageRef, Region, RegionSet
from .schema import TargetSpec


@dataclass(frozen=True)
class ReferentRefinementConfig:
    enabled: bool = False
    max_candidates: int = 8
    semantic_weight: float = 0.75
    geometry_weight: float = 0.25
    minimum_margin: float = 0.02
    candidate_halo_ratio: float = 0.2
    # When the downstream graph consumes a SINGLE target, ask the semantic VLM
    # to point at it directly (bbox) instead of relying on the detector +
    # reranker chain; the detector path remains the fallback.
    singleton_vlm_primary: bool = False
    single_detector_fallback: bool = True
    single_max_pixels: int = 409600
    single_min_box_area: float = 4.0
    attribute_policy: str = "unresolved"

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
        if self.single_max_pixels < 65536 or self.single_min_box_area <= 0:
            raise ValueError("invalid LOCATE_SINGLE visual/bbox budget")
        if self.attribute_policy not in {"unresolved", "vlm"}:
            raise ValueError("attribute_policy must be unresolved or vlm")

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
            singleton_vlm_primary=bool(value.get("singleton_vlm_primary", False)),
            single_detector_fallback=bool(value.get("single_detector_fallback", True)),
            single_max_pixels=int(value.get("single_max_pixels", 409600)),
            single_min_box_area=float(value.get("single_min_box_area", 4.0)),
            attribute_policy=str(value.get("attribute_policy", "unresolved")),
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

    @classmethod
    def _fallback_grid_indices(cls, question: str, count: int) -> tuple[int, ...]:
        if count < 1:
            return ()
        spatial_indices = {
            "upper_left": 0,
            "top": 1,
            "upper_right": 2,
            "left": 3,
            "center": 4,
            "right": 5,
            "lower_left": 6,
            "bottom": 7,
            "lower_right": 8,
        }
        index = spatial_indices.get(cls._spatial_hint(question), min(4, count - 1))
        return (index,) if index < count else (count - 1,)

    def select_grid_ids(
        self,
        grids: Sequence[Region],
        *,
        question: str,
        target: TargetSpec,
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        started = time.perf_counter()
        regions = tuple(grids)
        choice_ids = tuple(chr(ord("A") + index) for index in range(len(regions)))
        grid_ids = [
            str(region.provenance.get("grid_id", f"grid_{index + 1:02d}"))
            for index, region in enumerate(regions)
        ]
        metadata: dict[str, Any] = {
            "method": "coarse_grid_semantic_selection",
            "grid_count": len(regions),
            "grid_ids": grid_ids,
            "selected_grid_ids": [],
            "selected_grid_indices": [],
            "grid_scores": {},
            "selection_status": "UNRESOLVED",
            "question": question,
            "target_spec": {
                "category": target.category,
                "attributes": dict(target.attributes),
            },
            "provider": None,
            "model_id": None,
            "reasoning_text": None,
            "latency_ms": 0.0,
        }
        if not regions:
            metadata["failure_reason"] = "no_grid_regions"
            metadata["latency_ms"] = (time.perf_counter() - started) * 1000.0
            return (), metadata

        def geometry_fallback(reason: str | None = None) -> tuple[tuple[int, ...], dict[str, Any]]:
            indices = self._fallback_grid_indices(question, len(regions))
            metadata.update(
                {
                    "method": "coarse_grid_geometry_fallback",
                    "selected_grid_ids": [grid_ids[index] for index in indices],
                    "selected_grid_indices": list(indices),
                    "selection_status": "GEOMETRY_FALLBACK",
                    "failure_reason": reason,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
            return indices, metadata

        if not self.config.enabled:
            return geometry_fallback("semantic_refinement_disabled")

        option_positions = (
            "top-left",
            "top-center",
            "top-right",
            "middle-left",
            "center",
            "middle-right",
            "bottom-left",
            "bottom-center",
            "bottom-right",
        )
        option_texts = tuple(
            f"Grid {choice_id}: "
            f"{option_positions[index] if index < len(option_positions) else grid_id}"
            for index, (choice_id, grid_id) in enumerate(zip(choice_ids, grid_ids, strict=True))
        )
        try:
            model_input = self.composer.compose_named(
                {"candidates": RegionSet(regions, {"coarse_grid": True, "grid_ids": grid_ids})},
                question=(
                    "Select one to three grid cells that most likely contain the target. "
                    "Use only the marked 3x3 overview and do not answer the original question.\n\n"
                    f"Original referring expression: {question}\n"
                    f"Target specification: category={target.category}; "
                    f"attributes={dict(target.attributes)}"
                ),
                options=option_texts,
            )
            mapping = model_input.metadata.get("candidate_mapping")
            if not isinstance(mapping, Mapping) or len(mapping) != len(regions):
                raise RuntimeError("grid candidate mapping unavailable")
            scorer = getattr(self.semantic, "reason_and_choose", None)
            if not callable(scorer):
                raise RuntimeError("semantic provider lacks reason_and_choose")
            scored = scorer(
                ChoiceScoringRequest(
                    model_input=model_input,
                    answer_type="CHOICE_MULTI",
                    choice_ids=choice_ids,
                    option_texts=option_texts,
                    single_choice_suffix=self.choice_config.single_choice_suffix,
                    multi_verify_template=self.choice_config.multi_verify_template,
                    multi_select_threshold=0.0,
                    purpose="locate_grid_fallback",
                )
            )
            scores = {
                choice_id: float(scored.scores.get(choice_id, 0.0)) for choice_id in choice_ids
            }
            ranked = sorted(
                range(len(choice_ids)),
                key=lambda index: (-scores[choice_ids[index]], index),
            )
            selected = {choice_id for choice_id in scored.selected_ids if choice_id in scores}
            selected_indices = tuple(index for index in ranked if choice_ids[index] in selected)[:3]
            if not selected_indices:
                selected_indices = (ranked[0],)
            selected_indices = tuple(sorted(selected_indices))
            metadata.update(
                {
                    "selected_grid_ids": [grid_ids[index] for index in selected_indices],
                    "selected_grid_indices": list(selected_indices),
                    "grid_scores": {
                        grid_ids[index]: scores[choice_ids[index]] for index in range(len(regions))
                    },
                    "selection_status": "SEMANTIC_RESOLVED",
                    "provider": scored.provider,
                    "model_id": scored.model_id,
                    "reasoning_text": scored.reasoning_text,
                    "selected_model_choice_ids": [choice_ids[index] for index in selected_indices],
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
            return selected_indices, metadata
        except Exception as exc:
            return geometry_fallback(f"{type(exc).__name__}: {exc}")

    def _rank_for_budget(self, entities: tuple[Entity, ...], question: str) -> tuple[int, ...]:
        detector_scores = [
            float(entity.score) if entity.score is not None and math.isfinite(entity.score) else 0.0
            for entity in entities
        ]
        detector_norm = self._normalize(detector_scores)
        if self.config.semantic_weight <= 0.0:
            # No-VLM mode: rank strictly by detector score (highest first).
            ranking = [(-detector_norm[index], index) for index in range(len(entities))]
            ranking.sort(key=lambda item: (item[0], item[1]))
            return tuple(index for _, index in ranking[: self.config.max_candidates])
        geometry_scores = [self._geometry_prior(entity, question) for entity in entities]
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

        if self.config.semantic_weight <= 0.0:
            # Singleton-requiring downstream: no semantic verification — pin
            # the highest-scoring candidate (detector score descending).
            # NOTE: refine() is only invoked for single-target consumers
            # (should_refine), so SELECT/GROUP/COUNT/RELATION paths are
            # untouched and keep their full candidate lists.
            top_index = selected_indices[0]
            top_entity = entities[top_index]
            top_id = candidate_ids[top_index]
            resolution = "PRIMARY_RESOLVED"
            common.update(
                {
                    "applied": True,
                    "method": "singleton_highest_score",
                    "semantic_scores": {},
                    "selected_candidate_ids": [top_id],
                    "resolution_status": resolution,
                    "vlm_verification": "skipped_by_weight",
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
            output = EntitySet(
                (top_entity,),
                {
                    **candidates.provenance,
                    "referent_refinement": common,
                    "resolution_status": resolution,
                    "selected_candidate_ids": [top_id],
                },
            )
            return ReferentRefinementResult(output, common)

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
                    option_texts=tuple(str(item) for item in model_input.options),
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
                        "REFINED_RESOLVED" if margin > self.config.minimum_margin else "UNRESOLVED"
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
                    (
                        Entity(
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
                        ),
                    ),
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

    def vlm_referent(
        self,
        scope: ImageRef | Region,
        *,
        question: str = "",
        target: TargetSpec,
    ) -> ReferentRefinementResult:
        """LOCATE_SINGLE: current TargetSpec only; question is a compatibility argument."""
        from dataclasses import replace

        started = time.perf_counter()
        metadata: dict[str, Any] = {
            "method": "vlm_single",
            "target_spec": target.model_dump(mode="json"),
            "resolution_status": "UNRESOLVED",
            "candidate_pool_complete": False,
            "coordinate_system": "render_pixel_xyxy",
            "question_used": False,
        }
        entities: tuple[Entity, ...] = ()
        try:
            provider_pixels = getattr(
                getattr(self.semantic, "model_config", None), "max_pixels", None
            )
            pixel_budget = (
                min(self.config.single_max_pixels, provider_pixels)
                if provider_pixels
                else self.config.single_max_pixels
            )
            model_input = self.composer.compose_locate(scope, pixel_budget=pixel_budget)
            width, height = model_input.metadata["render_size"]
            attributes = json.dumps(target.attributes, ensure_ascii=False, sort_keys=True)
            prompt = (
                "Task: LOCATE_SINGLE. Localize one target object in this remote-sensing visual.\n"
                f"Target category: {target.category}\n"
                f"Intrinsic attributes: {attributes}\n"
                f"Actual supplied visual size: {width} x {height} pixels.\n"
                "Search scope: the supplied image or crop, and nothing outside it.\n"
                "Use ONLY the target definition supplied to this LOCATE node. "
                "Do not solve the overall "
                "question, use downstream SELECT conditions, or invent additional landmarks.\n"
                "If one well-supported matching object is identifiable, return exactly "
                '{"status":"OK","bbox":[x1,y1,x2,y2]}.\n'
                'If none is visible, return exactly {"status":"NONE"}.\n'
                "If several plausible targets cannot be distinguished, return exactly "
                '{"status":"AMBIGUOUS"}.\n'
                'If evidence is insufficient, return exactly {"status":"UNRESOLVED"}.\n'
                "The bbox uses pixel XYXY of the ACTUAL supplied visual, top-left origin. "
                "No Markdown, prose, or extra fields."
            )
            model_input = replace(model_input, question=prompt)
            result = self.semantic.infer(VLMRequest(model_input, output_contract="locate_single"))
            metadata.update(
                {
                    "visual_transform": model_input.metadata,
                    "provider": result.provider,
                    "raw_response": result.text,
                    "provider_metadata": result.metadata,
                }
            )
            payload = json.loads(result.text)
            if not isinstance(payload, dict) or payload.get("status") not in {
                "OK",
                "NONE",
                "AMBIGUOUS",
                "UNRESOLVED",
            }:
                raise ValueError("invalid_locate_status")
            status = payload["status"]
            if status != "OK":
                if set(payload) != {"status"}:
                    raise ValueError("unexpected_locate_fields")
                metadata["resolution_status"] = "EMPTY" if status == "NONE" else status
                metadata["candidate_pool_complete"] = status == "NONE"
            else:
                box = payload.get("bbox")
                if set(payload) != {"status", "bbox"} or not isinstance(box, list) or len(box) != 4:
                    raise ValueError("invalid_bbox_shape")
                if any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in box
                ):
                    raise ValueError("bbox_must_be_finite_numbers")
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError("bbox_order_or_bounds_invalid")
                if (x2 - x1) * (y2 - y1) < self.config.single_min_box_area:
                    raise ValueError("bbox_area_too_small")
                transform = model_input.metadata
                sx, sy = transform["scale_xy"]
                ox, oy = transform["crop_origin_global"]
                global_box = (x1 / sx + ox, y1 / sy + oy, x2 / sx + ox, y2 / sy + oy)
                original_width, original_height = transform["original_image_size"]
                scope_box = transform["scope_bbox_xyxy_global"]
                if not (
                    0 <= global_box[0] < global_box[2] <= original_width
                    and 0 <= global_box[1] < global_box[3] <= original_height
                    and scope_box[0] <= global_box[0]
                    and scope_box[1] <= global_box[1]
                    and global_box[2] <= scope_box[2]
                    and global_box[3] <= scope_box[3]
                ):
                    raise ValueError("mapped_bbox_outside_scope")
                image = scope.image if isinstance(scope, Region) else scope
                metadata.update(
                    {
                        "resolution_status": "VLM_REFERENT_RESOLVED",
                        "candidate_pool_complete": True,
                        "bbox_xyxy_render": box,
                        "bbox_xyxy_global": list(global_box),
                    }
                )
                entities = (
                    Entity(
                        Region(image, global_box, {"vlm_referent": True}),
                        target.category,
                        None,
                        {"candidate_id": "vlm_referent_0001", "vlm_referent": True},
                    ),
                )
        except Exception as exc:
            metadata.update(
                {
                    "resolution_status": "UNRESOLVED",
                    "candidate_pool_complete": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
        metadata["latency_ms"] = (time.perf_counter() - started) * 1000.0
        return ReferentRefinementResult(EntitySet(entities, dict(metadata)), metadata)

    def visual_fallback(
        self, scope: ImageRef | Region, *, question: str, target: TargetSpec, reason: str
    ) -> ReferentRefinementResult:
        """Compatibility entry point; an unlocalized scope is never an Entity."""
        metadata = {
            "method": "semantic_visual_fallback",
            "resolution_status": "UNRESOLVED",
            "candidate_pool_complete": False,
            "failure_reason": reason,
            "output_candidate_count": 0,
        }
        return ReferentRefinementResult(EntitySet((), dict(metadata)), metadata)
