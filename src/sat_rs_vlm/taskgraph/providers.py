"""Replaceable capability contracts and adapters for production providers."""

from __future__ import annotations

import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image

from sat_rs_vlm.infrastructure.config import ModelConfig
from sat_rs_vlm.integrations.detectors.protocol import ProposalProvider
from sat_rs_vlm.integrations.locators.protocol import LocatorProvider
from sat_rs_vlm.integrations.retrievers.protocol import RetrieverProvider
from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine

from .runtime_types import (
    BBox,
    ChoiceScoreResult,
    Entity,
    EntitySet,
    ImageRef,
    Region,
    RuntimeObject,
)
from .schema import TargetSpec, TaskGraph


def _bbox_contains(
    outer: Sequence[float], inner: Sequence[float], *, tolerance: float = 1e-6
) -> bool:
    return (
        float(inner[0]) >= float(outer[0]) - tolerance
        and float(inner[1]) >= float(outer[1]) - tolerance
        and float(inner[2]) <= float(outer[2]) + tolerance
        and float(inner[3]) <= float(outer[3]) + tolerance
    )


def _bbox_intersection(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float] | None:
    box = (
        max(float(left[0]), float(right[0])),
        max(float(left[1]), float(right[1])),
        min(float(left[2]), float(right[2])),
        min(float(left[3]), float(right[3])),
    )
    return box if box[0] < box[2] and box[1] < box[3] else None


@dataclass(frozen=True)
class DetectionRequest:
    scope: ImageRef | Region
    target: TargetSpec
    task_hint: str | None = None


@dataclass(frozen=True)
class DetectionSet:
    detections: EntitySet
    latency_ms: float
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DetectionProvider(Protocol):
    provider_name: str

    def detect(self, request: DetectionRequest) -> DetectionSet: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RegionRetrievalRequest:
    image: ImageRef | Region
    query: str
    search_scope: Region | None = None
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("region retrieval query must not be empty")
        if self.max_candidates is not None and self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.search_scope is None:
            return
        image = self.image if isinstance(self.image, ImageRef) else self.image.image
        if self.search_scope.image.uri_or_key != image.uri_or_key:
            raise ValueError("search_scope must reference the same image")
        if isinstance(self.image, Region) and not _bbox_contains(
            self.image.bbox_xyxy_global, self.search_scope.bbox_xyxy_global
        ):
            raise ValueError("nested search_scope must be contained by input Region")

    def effective_scope(self) -> Region | None:
        if self.search_scope is not None:
            return self.search_scope
        return self.image if isinstance(self.image, Region) else None


@dataclass(frozen=True)
class RegionCandidate:
    region: Region
    relevance_score: float
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionCandidates:
    candidates: tuple[RegionCandidate, ...]
    provider: str
    latency_ms: float = 0.0


class RegionRetrieverProvider(Protocol):
    provider_name: str

    def retrieve(self, request: RegionRetrievalRequest) -> RegionCandidates: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ModelSource:
    role: str
    value: RuntimeObject


@dataclass(frozen=True)
class ModelInput:
    visual_inputs: tuple[str, ...]
    structured_context: str
    question: str
    options: tuple[str, ...] = ()
    visual_roles: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLMRequest:
    model_input: ModelInput
    output_contract: str = "text"


@dataclass(frozen=True)
class VLMResult:
    text: str
    provider: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CachedChoiceUnavailableError(RuntimeError):
    """The configured backend explicitly lacks cached choice capability."""


@dataclass(frozen=True)
class ChoiceScoringRequest:
    model_input: ModelInput
    answer_type: str
    choice_ids: tuple[str, ...]
    option_texts: tuple[str, ...]
    single_choice_suffix: str
    multi_verify_template: str
    multi_select_threshold: float = 0.0
    purpose: str = "final_choice"

    def __post_init__(self) -> None:
        if self.answer_type not in {"CHOICE_SINGLE", "CHOICE_MULTI"}:
            raise ValueError("choice scoring answer_type is invalid")
        if not self.choice_ids or len(self.choice_ids) != len(self.option_texts):
            raise ValueError("choice ids and option texts must be non-empty and aligned")
        if not self.single_choice_suffix:
            raise ValueError("single choice suffix must not be empty")
        if (
            "{choice_id}" not in self.multi_verify_template
            or "{option_text}" not in self.multi_verify_template
        ):
            raise ValueError("multi verify template must include choice_id and option_text")


class SemanticVLMProvider(Protocol):
    provider_name: str

    def infer(self, request: VLMRequest) -> VLMResult: ...

    def reason_and_choose(self, request: ChoiceScoringRequest) -> ChoiceScoreResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PlannerRequest:
    question: str
    question_type: str
    choices: tuple[str, ...]
    inputs: Mapping[str, ImageRef]
    sample_id: str | None = None


class PlannerProvider(Protocol):
    provider_name: str

    def plan(self, request: PlannerRequest) -> TaskGraph: ...


@dataclass(frozen=True)
class EvidenceSufficiencyRequest:
    question: str
    region: Region
    task_hint: str | None = None


@dataclass(frozen=True)
class EvidenceSufficiencyResult:
    status: str
    score: float | None = None


class EvidenceSufficiencyProvider(Protocol):
    provider_name: str

    def assess(self, request: EvidenceSufficiencyRequest) -> EvidenceSufficiencyResult: ...


class ProposalDetectionAdapter:
    """Adapt the existing LAE/other ProposalProvider without changing it."""

    def __init__(self, provider: ProposalProvider) -> None:
        self._provider = provider
        self.provider_name = provider.provider_name

    @staticmethod
    def _image_scope(scope: ImageRef | Region) -> tuple[ImageRef, tuple[float, float]]:
        if isinstance(scope, ImageRef):
            return scope, (0.0, 0.0)
        return scope.image, (scope.bbox_xyxy_global[0], scope.bbox_xyxy_global[1])

    def detect(self, request: DetectionRequest) -> DetectionSet:
        image_ref, offset = self._image_scope(request.scope)
        source_path = image_ref.path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"detection image does not exist: {source_path}")
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="taskgraph_detection_") as temp_dir:
            detector_path = source_path
            if isinstance(request.scope, Region):
                with Image.open(source_path) as source:
                    crop = source.convert("RGB").crop(request.scope.bbox_xyxy_global)
                    detector_path = Path(temp_dir) / "scope.png"
                    crop.save(detector_path)
            result = self._provider.predict(detector_path, request.target.phrase())
        entities = []
        for box, score in zip(result.boxes_xyxy, result.scores, strict=True):
            global_box = (
                float(box[0]) + offset[0],
                float(box[1]) + offset[1],
                float(box[2]) + offset[0],
                float(box[3]) + offset[1],
            )
            entities.append(
                Entity(
                    region=Region(
                        image=image_ref,
                        bbox_xyxy_global=global_box,
                        provenance={
                            "provider": result.provider,
                            "coordinate_mode": "absolute_original_pixel_xyxy",
                        },
                    ),
                    label=request.target.category,
                    score=float(score),
                    provenance={
                        "provider": result.provider,
                        "model_id": result.model_id,
                        "scale_tile_metadata": dict(result.metadata),
                    },
                )
            )
        return DetectionSet(
            detections=EntitySet(
                tuple(entities),
                provenance={
                    "provider": result.provider,
                    "model_id": result.model_id,
                    "proposal_metadata": dict(result.metadata),
                },
            ),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=result.provider,
            metadata=dict(result.metadata),
        )

    def close(self) -> None:
        self._provider.close()


class FakeDetectionProvider:
    provider_name = "fake_lae"

    def __init__(self, boxes: Sequence[Sequence[float]] | None = None) -> None:
        self.boxes = [cast(BBox, tuple(float(item) for item in box)) for box in (boxes or [])]
        self.calls: list[DetectionRequest] = []

    def detect(self, request: DetectionRequest) -> DetectionSet:
        self.calls.append(request)
        image = request.scope if isinstance(request.scope, ImageRef) else request.scope.image
        entities = tuple(
            Entity(
                Region(image, box, {"provider": self.provider_name}),
                request.target.category,
                max(0.01, 0.99 - index * 0.01),
                {"provider": self.provider_name},
            )
            for index, box in enumerate(self.boxes)
        )
        return DetectionSet(
            EntitySet(entities, {"provider": self.provider_name}),
            0.0,
            self.provider_name,
            {"deterministic": True},
        )

    def close(self) -> None:
        return None


class LocatorRegionRetrieverAdapter:
    """Expose the existing UHR Locator as the generic candidate capability."""

    def __init__(self, locator: LocatorProvider) -> None:
        self._locator = locator
        self.provider_name = locator.provider_name

    def retrieve(self, request: RegionRetrievalRequest) -> RegionCandidates:
        image = request.image if isinstance(request.image, ImageRef) else request.image.image
        source_path = image.path.resolve()
        scope = request.effective_scope()
        offset = (0.0, 0.0)
        if scope is None:
            result = self._locator.locate(source_path, request.query)
        else:
            offset = scope.bbox_xyxy_global[:2]
            with tempfile.TemporaryDirectory(prefix="taskgraph_retrieval_") as temp_dir:
                crop_path = Path(temp_dir) / "scope.png"
                with Image.open(source_path) as source:
                    source.convert("RGB").crop(scope.bbox_xyxy_global).save(crop_path)
                result = self._locator.locate(crop_path, request.query)
        candidates_list = []
        for index, (box, score) in enumerate(zip(result.regions_xyxy, result.scores, strict=True)):
            global_box = (
                float(box[0]) + offset[0],
                float(box[1]) + offset[1],
                float(box[2]) + offset[0],
                float(box[3]) + offset[1],
            )
            if scope is not None:
                clipped = _bbox_intersection(global_box, scope.bbox_xyxy_global)
                if clipped is None:
                    continue
                global_box = clipped
            details = (
                dict(result.region_details[index]) if index < len(result.region_details) else {}
            )
            candidates_list.append(
                RegionCandidate(
                    Region(
                        image,
                        global_box,
                        {
                            "locator": self.provider_name,
                            "coordinate_mode": "absolute_original_pixel_xyxy",
                            "search_scope": (
                                list(scope.bbox_xyxy_global) if scope is not None else None
                            ),
                        },
                    ),
                    float(score),
                    {
                        "locator": self.provider_name,
                        "local_bbox_xyxy": list(box),
                        "global_bbox_xyxy": list(global_box),
                        "details": details,
                    },
                )
            )
            if (
                request.max_candidates is not None
                and len(candidates_list) >= request.max_candidates
            ):
                break
        candidates = tuple(candidates_list)
        return RegionCandidates(candidates, self.provider_name, result.latency_ms.get("total", 0.0))

    def close(self) -> None:
        self._locator.close()


class ScoredGridRegionRetrieverAdapter:
    """Adapt score-only RetrieverProvider by supplying explicit grid candidates."""

    def __init__(self, provider: RetrieverProvider, *, grid_size: int = 3) -> None:
        self._provider = provider
        self.provider_name = provider.provider_name
        self.grid_size = grid_size

    def retrieve(self, request: RegionRetrievalRequest) -> RegionCandidates:
        image = request.image if isinstance(request.image, ImageRef) else request.image.image
        with Image.open(image.path) as source:
            width, height = source.size
        effective_scope = request.effective_scope()
        scope = (
            effective_scope.bbox_xyxy_global
            if effective_scope is not None
            else (0.0, 0.0, float(width), float(height))
        )
        cell_width = (scope[2] - scope[0]) / self.grid_size
        cell_height = (scope[3] - scope[1]) / self.grid_size
        boxes = [
            (
                scope[0] + x * cell_width,
                scope[1] + y * cell_height,
                scope[0] + (x + 1) * cell_width,
                scope[1] + (y + 1) * cell_height,
            )
            for y in range(self.grid_size)
            for x in range(self.grid_size)
        ]
        scored = self._provider.score_regions(image.path, request.query, boxes)
        order = sorted(range(len(boxes)), key=lambda index: (-scored.scores[index], index))
        order = order[: request.max_candidates or len(order)]
        candidates = tuple(
            RegionCandidate(
                Region(image, boxes[index], {"retriever": self.provider_name}),
                scored.scores[index],
                {"retriever": self.provider_name, "model_id": scored.model_id},
            )
            for index in order
        )
        return RegionCandidates(candidates, self.provider_name, scored.latency_ms)

    def close(self) -> None:
        self._provider.close()


class FakeRegionRetriever:
    provider_name = "fake_region_retriever"

    def __init__(self, candidates: Sequence[tuple[Sequence[float], float]] | None = None) -> None:
        self._candidates = list(candidates or [])

    def retrieve(self, request: RegionRetrievalRequest) -> RegionCandidates:
        image = request.image if isinstance(request.image, ImageRef) else request.image.image
        scope = request.effective_scope()
        candidates = []
        for box, score in self._candidates:
            global_box = cast(BBox, tuple(float(value) for value in box))
            if scope is not None:
                clipped = _bbox_intersection(global_box, scope.bbox_xyxy_global)
                if clipped is None:
                    continue
                global_box = clipped
            candidates.append(
                RegionCandidate(
                    Region(
                        image,
                        global_box,
                        {
                            "fake": True,
                            "search_scope": (
                                list(scope.bbox_xyxy_global) if scope is not None else None
                            ),
                        },
                    ),
                    float(score),
                    {"provider": self.provider_name},
                )
            )
            if request.max_candidates is not None and len(candidates) >= request.max_candidates:
                break
        return RegionCandidates(
            tuple(candidates),
            self.provider_name,
        )

    def close(self) -> None:
        return None


class LazyQwenSemanticProvider:
    """Lazy semantic adapter reusing the repository HuggingFace VLM engine."""

    provider_name = "qwen3_vl"

    def __init__(self, model_config: ModelConfig, *, role: str = "semantic") -> None:
        self.model_config = model_config
        self.role = role
        self._engine: HuggingFaceVLMEngine | None = None

    def _load(self) -> HuggingFaceVLMEngine:
        if self._engine is None:
            self._engine = HuggingFaceVLMEngine(
                model_id=self.model_config.model_id,
                device=self.model_config.device,
                dtype=self.model_config.dtype,
                max_new_tokens=self.model_config.max_new_tokens,
                trust_remote_code=self.model_config.trust_remote_code,
                local_files_only=self.model_config.local_files_only,
            )
        return self._engine

    @property
    def engine_identity(self) -> str | None:
        return self._engine.model_identity if self._engine is not None else None

    @staticmethod
    def _prompt(model_input: ModelInput, *, reasoning_instruction: str | None = None) -> str:
        prompt_parts = []
        if model_input.visual_inputs:
            roles = model_input.visual_roles or tuple(
                f"VISUAL_{index}" for index in range(1, len(model_input.visual_inputs) + 1)
            )
            manifest = "\n".join(
                f"[image_{index}] role: {role}" for index, role in enumerate(roles, start=1)
            )
            prompt_parts.append("Visual inputs:\n" + manifest)
        if model_input.structured_context:
            prompt_parts.append("Structured results:\n" + model_input.structured_context)
        prompt_parts.append("Question:\n" + model_input.question)
        if model_input.options:
            prompt_parts.append("Options:\n" + "\n".join(model_input.options))
        if reasoning_instruction:
            prompt_parts.append(reasoning_instruction)
        return "\n\n".join(prompt_parts)

    def infer(self, request: VLMRequest) -> VLMResult:
        model_input = request.model_input
        prompt = self._prompt(model_input)
        engine = self._load()
        allowed_outputs: tuple[str, ...] | None = None
        if (
            request.output_contract == "selection"
            and self.model_config.selection_constrained_decoding
        ):
            candidate_mapping = model_input.metadata.get("candidate_mapping")
            if isinstance(candidate_mapping, Mapping):
                labels = tuple(str(label) for label in candidate_mapping)
                # SELECT v1 intentionally keeps the candidate canvas small.
                # 2^8 produces only 256 valid complete strings and is practical
                # for a trie-like token mask; larger sets fall back to parsing.
                if 0 < len(labels) <= 8 and all(
                    len(label) == 1 and label.isalpha() for label in labels
                ):
                    allowed_outputs = ("NONE",) + tuple(
                        ",".join(group)
                        for size in range(1, len(labels) + 1)
                        for group in combinations(labels, size)
                    )
        if not model_input.visual_inputs:
            # Qwen is a VLM, but structured authoritative choice must remain text-only.
            generated = engine.generate_text(
                prompt=prompt, image_paths=[], allowed_outputs=allowed_outputs
            )
        else:
            generated = engine.generate_text(
                prompt=prompt,
                image_paths=list(model_input.visual_inputs),
                allowed_outputs=allowed_outputs,
            )
        return VLMResult(
            generated,
            f"{self.provider_name}:{self.role}",
            metadata={
                "output_contract": request.output_contract,
                "constrained_decoding": allowed_outputs is not None,
                "allowed_output_count": len(allowed_outputs or ()),
            },
        )

    def reason_and_choose(self, request: ChoiceScoringRequest) -> ChoiceScoreResult:
        if request.purpose == "route_choice":
            instruction = (
                "Analyze the route-planning problem using the marked start, goal, obstacles, "
                "spatial layout, and supplied route options. Compare feasible routes carefully. "
                "The final option will be selected in a separate constrained step."
            )
        elif request.purpose == "select_relation":
            instruction = (
                "Analyze which candidate object or objects satisfy the requested relation. "
                "Use candidate labels only as visual references during reasoning. A separate "
                "constrained step will determine the final selection."
            )
        else:
            instruction = (
                "Analyze the visual evidence, question, and all candidate options carefully. "
                "Reason through the problem before making the final decision. The final option "
                "will be selected in a separate constrained step."
            )
        engine = self._load()
        scorer = getattr(engine, "reason_and_choose", None)
        if not callable(scorer):
            raise CachedChoiceUnavailableError(
                "Qwen engine does not expose cached reasoning-to-choice scoring"
            )
        result = scorer(
            self._prompt(request.model_input, reasoning_instruction=instruction),
            list(request.model_input.visual_inputs),
            choice_ids=request.choice_ids,
            option_texts=request.option_texts,
            answer_type=request.answer_type,
            single_choice_suffix=request.single_choice_suffix,
            multi_verify_template=request.multi_verify_template,
            multi_select_threshold=request.multi_select_threshold,
            reasoning_max_new_tokens=self.model_config.max_new_tokens,
        )
        return ChoiceScoreResult(
            selected_ids=result.selected_ids,
            scores=result.scores,
            answer_type=result.answer_type,
            reasoning_text=result.reasoning_text,
            provider=f"{self.provider_name}:{self.role}",
            model_id=engine.model_id,
            method=result.method,
            cache_reused=result.cache_reused,
            latency_ms=result.latency_ms,
            metadata={**result.metadata, "purpose": request.purpose},
        )

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
        self._engine = None


class FakeSemanticVLMProvider:
    provider_name = "fake_vlm"

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        default: str = "A",
        *,
        choice_scores: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.responses = dict(responses or {})
        self.default = default
        self.choice_scores = {
            purpose: {choice_id: float(score) for choice_id, score in scores.items()}
            for purpose, scores in (choice_scores or {}).items()
        }
        self.calls: list[VLMRequest] = []
        self.choice_calls: list[ChoiceScoringRequest] = []

    def infer(self, request: VLMRequest) -> VLMResult:
        self.calls.append(request)
        response = self.responses.get(request.output_contract)
        if response is None and request.output_contract in {
            "choice",
            "choice_single",
            "choice_multi",
        }:
            response = self.responses.get("choice")
        if response is None and request.output_contract in {
            "choice",
            "choice_single",
            "choice_multi",
        }:
            match = re.search(r"^value:\s*(.+)$", request.model_input.structured_context, re.M)
            if match:
                value = match.group(1).strip().casefold()
                for index, option in enumerate(request.model_input.options):
                    normalized = re.sub(r"^\s*[\(\[]?[A-Z][\)\].:]?\s*", "", option).strip()
                    if normalized.casefold() == value:
                        response = chr(ord("A") + index)
                        break
        response = response or self.default
        return VLMResult(response, self.provider_name, 1.0, {"deterministic": True})

    def _fixture_selected_ids(self, request: ChoiceScoringRequest) -> tuple[str, ...]:
        response = self.responses.get(request.purpose)
        if response is None and request.purpose == "select_relation":
            response = self.responses.get("selection")
        if response is None:
            response = self.responses.get(request.answer_type.casefold())
        if response is None:
            response = self.responses.get("choice")
        response = (response or self.default).strip()
        try:
            import json

            payload = json.loads(response)
        except (ValueError, TypeError):
            payload = None
        values: list[str] = []
        if isinstance(payload, dict):
            raw = payload.get("selected_ids", payload.get("choice_ids"))
            if isinstance(raw, list):
                values = [str(item).strip().upper() for item in raw]
        if not values and response.upper() in request.choice_ids:
            values = [response.upper()]
        if not values and request.purpose == "select_relation":
            raw_items = [item.strip().upper() for item in response.split(",")]
            if raw_items and all(item in request.choice_ids for item in raw_items):
                values = raw_items
            elif raw_items and all(item.isdigit() for item in raw_items):
                indices = [int(item) for item in raw_items]
                if 0 not in indices and all(
                    1 <= item <= len(request.choice_ids) for item in indices
                ):
                    indices = [item - 1 for item in indices]
                if all(0 <= item < len(request.choice_ids) for item in indices):
                    values = [request.choice_ids[item] for item in indices]
        return tuple(dict.fromkeys(item for item in values if item in request.choice_ids))

    def reason_and_choose(self, request: ChoiceScoringRequest) -> ChoiceScoreResult:
        self.choice_calls.append(request)
        fixture_scores = self.choice_scores.get(request.purpose)
        if fixture_scores is None:
            fixture_scores = self.choice_scores.get(request.answer_type.casefold())
        selected_fixture = self._fixture_selected_ids(request)
        scores = {
            choice_id: (
                float(fixture_scores[choice_id])
                if fixture_scores is not None and choice_id in fixture_scores
                else (1.0 if choice_id in selected_fixture else -1.0)
            )
            for choice_id in request.choice_ids
        }
        selected_ids: tuple[str, ...]
        if request.answer_type == "CHOICE_SINGLE":
            selected_ids = (max(request.choice_ids, key=lambda item: scores[item]),)
            method = "fake_kv_cached_logits"
        else:
            selected_ids = tuple(
                choice_id
                for choice_id in request.choice_ids
                if scores[choice_id] > request.multi_select_threshold
            )
            method = "fake_kv_cached_binary_verification"
        reasoning = self.responses.get(
            f"{request.purpose}_reasoning",
            self.responses.get("reasoning", "Fake free reasoning; letters A/B/C are not parsed."),
        )
        return ChoiceScoreResult(
            selected_ids=selected_ids,
            scores=scores,
            answer_type=request.answer_type,
            reasoning_text=reasoning,
            provider=self.provider_name,
            model_id="fake-model",
            method=method,
            cache_reused=True,
            latency_ms={
                "vision_prefill_ms": 0.0,
                "text_prefill_ms": 0.0,
                "total_prefill_ms": 0.0,
                "reasoning_decode_ms": 0.0,
                "reasoning_total_ms": 0.0,
                "cache_clone_ms": 0.0,
                "suffix_tokenize_ms": 0.0,
                "choice_suffix_prefill_ms": 0.0,
                "choice_scoring_ms": 0.0,
                "choice_total_ms": 0.0,
                "total_ms": 0.0,
            },
            metadata={
                "initial_prefill_tokens": 1,
                "reasoning_tokens": 1,
                "choice_suffix_tokens": 1,
                "choice_scored_tokens": len(request.choice_ids),
                "visual_prefill_count": 1 if request.model_input.visual_inputs else 0,
                "reasoning_pass_count": 1,
                "session_released": True,
                "reasoning_cache_mode": (
                    "consume_in_place"
                    if request.answer_type == "CHOICE_SINGLE"
                    else "fork_per_option"
                ),
                "peak_vram_mb": None,
                "purpose": request.purpose,
            },
        )

    def close(self) -> None:
        return None


class FixturePlannerProvider:
    provider_name = "fixture_planner"

    def __init__(self, fixtures: Mapping[str, TaskGraph | Mapping[str, Any]]) -> None:
        self.fixtures = {
            key: value if isinstance(value, TaskGraph) else TaskGraph.model_validate(value)
            for key, value in fixtures.items()
        }

    def plan(self, request: PlannerRequest) -> TaskGraph:
        key = request.sample_id or request.question
        try:
            graph = self.fixtures[key]
        except KeyError as exc:
            raise KeyError(f"no fixture TaskGraph for {key!r}") from exc
        if tuple(graph.choices or ()) != request.choices:
            raise ValueError("fixture graph choices differ from original dataset options")
        return graph


class FakeEvidenceSufficiencyProvider:
    provider_name = "fake_evidence_sufficiency"

    def __init__(self, status: str = "SUFFICIENT", score: float = 1.0) -> None:
        self.status = status
        self.score = score

    def assess(self, request: EvidenceSufficiencyRequest) -> EvidenceSufficiencyResult:
        return EvidenceSufficiencyResult(self.status, self.score)


def parse_selection_indices(text: str, count: int) -> tuple[int, ...]:
    """Parse a candidate selection without mistaking prose numbers for ids.

    Constrained decoding normally yields canonical ``A,C`` or ``NONE``.  This
    parser deliberately remains tolerant for model/back-end compatibility, but
    numeric ids are accepted only when the entire reply is a numeric list or
    they are explicitly introduced as candidate/option identifiers.
    """

    normalized = text.strip()
    if normalized.casefold() in {"none", "no", "empty", "null"}:
        return ()

    letters = re.findall(r"(?<![A-Z0-9])([A-Z])(?![A-Z0-9])", normalized.upper())
    if letters:
        values = [ord(item) - ord("A") for item in letters]
        if any(item < 0 or item >= count for item in values):
            raise ValueError(f"selection provider returned out-of-range candidate ids: {letters}")
        return tuple(dict.fromkeys(values))
    numeric_list = re.fullmatch(r"\s*\[?\s*\d+(?:\s*[,，]\s*\d+)*\s*\]?\s*", normalized)
    explicit_numeric = re.findall(
        r"(?:candidate|candidates|option|options|候选|选项)\s*#?\s*(\d+)",
        normalized,
        flags=re.IGNORECASE,
    )
    values = (
        [int(item) for item in re.findall(r"\d+", normalized)]
        if numeric_list
        else [int(item) for item in explicit_numeric]
    )
    if not values:
        raise ValueError(f"selection provider returned no candidate ids: {text!r}")
    if 0 not in values and all(1 <= item <= count for item in values):
        values = [item - 1 for item in values]
    if any(item < 0 or item >= count for item in values):
        raise ValueError(f"selection provider returned out-of-range candidate ids: {values}")
    return tuple(dict.fromkeys(values))
