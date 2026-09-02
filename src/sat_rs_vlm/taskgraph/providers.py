"""Replaceable capability contracts and adapters for production providers."""

from __future__ import annotations

import math
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
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

    def __post_init__(self) -> None:
        score = float(self.relevance_score)
        if not math.isfinite(score):
            raise ValueError("region candidate relevance_score must be finite")
        object.__setattr__(self, "relevance_score", score)


@dataclass(frozen=True)
class RegionCandidates:
    candidates: tuple[RegionCandidate, ...]
    provider: str
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        latency = float(self.latency_ms)
        if not math.isfinite(latency) or latency < 0.0:
            raise ValueError("region candidate latency_ms must be finite and non-negative")
        if not str(self.provider).strip():
            raise ValueError("region candidate provider must not be empty")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "latency_ms", latency)


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


@dataclass(frozen=True)
class FiniteDecisionRequest:
    """Generic cached reasoning-to-finite-decision request.

    Benchmark choice is one caller of this primitive. Intermediate semantic
    alignment uses canonical values directly and never parses the free
    reasoning text.
    """

    model_input: ModelInput
    decision_mode: str
    candidate_ids: tuple[str, ...]
    candidate_texts: tuple[str, ...]
    single_decision_suffix: str
    multi_verify_template: str
    select_threshold: float = 0.0
    purpose: str = "semantic_decision"
    reasoning_instruction: str | None = None

    def __post_init__(self) -> None:
        if self.decision_mode not in {"SINGLE", "MULTI", "BINARY"}:
            raise ValueError("finite decision mode must be SINGLE, MULTI, or BINARY")
        if not self.candidate_ids or len(self.candidate_ids) != len(self.candidate_texts):
            raise ValueError("finite decision candidates must be non-empty and aligned")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("finite decision candidate ids must be unique")
        if self.decision_mode == "BINARY" and len(self.candidate_ids) != 2:
            raise ValueError("binary finite decision requires exactly two candidates")
        if not self.single_decision_suffix:
            raise ValueError("single decision suffix must not be empty")
        if (
            "{choice_id}" not in self.multi_verify_template
            or "{option_text}" not in self.multi_verify_template
        ):
            raise ValueError("multi verify template must include choice_id and option_text")


@dataclass(frozen=True)
class FiniteDecisionResult:
    selected_ids: tuple[str, ...]
    scores: dict[str, float]
    decision_mode: str
    reasoning_text: str | None
    provider: str
    model_id: str
    method: str
    cache_reused: bool
    latency_ms: dict[str, float | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision_mode not in {"SINGLE", "MULTI", "BINARY"}:
            raise ValueError("finite decision result mode is invalid")
        if self.decision_mode in {"SINGLE", "BINARY"} and len(self.selected_ids) != 1:
            raise ValueError("single and binary decisions require exactly one selected id")
        if len(self.selected_ids) != len(set(self.selected_ids)):
            raise ValueError("selected finite decision ids must be unique")
        if any(candidate not in self.scores for candidate in self.selected_ids):
            raise ValueError("every selected finite decision id must have a score")


class SemanticVLMProvider(Protocol):
    provider_name: str

    def infer(self, request: VLMRequest) -> VLMResult: ...

    def reason_and_decide(self, request: FiniteDecisionRequest) -> FiniteDecisionResult: ...

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
    region: Region | None = None
    task_hint: str | None = None
    evidence: tuple[RuntimeObject, ...] = ()
    sample_id: str | None = None
    evidence_version: str = "v1"

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("evidence sufficiency question must not be empty")
        if self.region is None and not self.evidence:
            raise ValueError("evidence sufficiency requires region or evidence")
        if not self.evidence_version.strip():
            raise ValueError("evidence_version must not be empty")

    @property
    def sources(self) -> tuple[RuntimeObject, ...]:
        region = (self.region,) if self.region is not None else ()
        return region + self.evidence


class EvidenceSufficiencyStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class EvidenceSufficiencyResult:
    status: EvidenceSufficiencyStatus | str
    score: float | None = None
    reason_code: str | None = None
    provider: str = "unknown"
    model_id: str = "unknown"
    method: str = "unknown"
    cache_reused: bool = False
    latency_ms: dict[str, float | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", EvidenceSufficiencyStatus(self.status))
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("evidence sufficiency score must be between 0 and 1")


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
        entities = []
        for index, box in enumerate(self.boxes):
            global_box = box
            if isinstance(request.scope, Region):
                clipped = _bbox_intersection(box, request.scope.bbox_xyxy_global)
                if clipped is None:
                    continue
                global_box = clipped
            entities.append(
                Entity(
                    Region(
                        image,
                        global_box,
                        {
                            "provider": self.provider_name,
                            "search_scope": (
                                list(request.scope.bbox_xyxy_global)
                                if isinstance(request.scope, Region)
                                else None
                            ),
                        },
                    ),
                    request.target.category,
                    max(0.01, 0.99 - index * 0.01),
                    {"provider": self.provider_name},
                )
            )
        return DetectionSet(
            EntitySet(tuple(entities), {"provider": self.provider_name}),
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

    def __init__(
        self,
        provider: RetrieverProvider,
        *,
        grid_size: int = 3,
        default_max_candidates: int = 5,
        candidate_window_ratio: float | None = None,
    ) -> None:
        if grid_size < 1:
            raise ValueError("retriever grid_size must be positive")
        if default_max_candidates < 1:
            raise ValueError("retriever default_max_candidates must be positive")
        if candidate_window_ratio is not None and not 0.0 < candidate_window_ratio <= 1.0:
            raise ValueError("retriever candidate_window_ratio must be in (0, 1]")
        self._provider = provider
        self.provider_name = provider.provider_name
        self.grid_size = grid_size
        self.default_max_candidates = default_max_candidates
        self.candidate_window_ratio = candidate_window_ratio

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
        scope_width = scope[2] - scope[0]
        scope_height = scope[3] - scope[1]
        window_ratio = self.candidate_window_ratio or 1.0 / self.grid_size
        cell_width = scope_width * window_ratio
        cell_height = scope_height * window_ratio
        if self.grid_size == 1:
            x_starts = [scope[0] + (scope_width - cell_width) / 2.0]
            y_starts = [scope[1] + (scope_height - cell_height) / 2.0]
        else:
            x_stride = (scope_width - cell_width) / (self.grid_size - 1)
            y_stride = (scope_height - cell_height) / (self.grid_size - 1)
            x_starts = [scope[0] + x * x_stride for x in range(self.grid_size)]
            y_starts = [scope[1] + y * y_stride for y in range(self.grid_size)]
        boxes = [
            (x_start, y_start, x_start + cell_width, y_start + cell_height)
            for y_start in y_starts
            for x_start in x_starts
        ]
        scored = self._provider.score_regions(image.path, request.query, boxes)
        order = sorted(range(len(boxes)), key=lambda index: (-scored.scores[index], index))
        order = order[: request.max_candidates or self.default_max_candidates]
        candidates = tuple(
            RegionCandidate(
                Region(
                    image,
                    boxes[index],
                    {
                        "retriever": self.provider_name,
                        "coordinate_mode": "absolute_original_pixel_xyxy",
                        "search_scope": list(scope),
                    },
                ),
                scored.scores[index],
                {
                    "provider": self.provider_name,
                    "model_id": scored.model_id,
                    "bbox_xyxy_global": list(boxes[index]),
                    "search_scope": list(scope),
                    "tile": {
                        "level": 1,
                        "index": index,
                        "row": index // self.grid_size,
                        "column": index % self.grid_size,
                        "grid_size": self.grid_size,
                    },
                    "candidate_geometry": {
                        "layout": "uniform_sliding_grid",
                        "window_ratio": window_ratio,
                        "overlapping": window_ratio > 1.0 / self.grid_size,
                    },
                    "provider_metadata": dict(getattr(scored, "metadata", {})),
                },
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
                "model_id": str(getattr(engine, "model_id", "unknown")),
                "output_contract": request.output_contract,
                "constrained_decoding": allowed_outputs is not None,
                "allowed_output_count": len(allowed_outputs or ()),
            },
        )

    @staticmethod
    def _choice_instruction(purpose: str) -> str:
        if purpose == "route_choice":
            return (
                "Analyze the route-planning problem using the marked start, goal, obstacles, "
                "spatial layout, and supplied route options. Compare feasible routes carefully. "
                "The final option will be selected in a separate constrained step."
            )
        if purpose == "select_relation":
            return (
                "Analyze which candidate object or objects satisfy the requested relation. "
                "Use candidate labels only as visual references during reasoning. A separate "
                "constrained step will determine the final selection."
            )
        return (
            "Analyze the visual evidence, question, and all candidate options carefully. "
            "Reason through the problem before making the final decision. The final option "
            "will be selected in a separate constrained step."
        )

    def reason_and_decide(self, request: FiniteDecisionRequest) -> FiniteDecisionResult:
        engine = self._load()
        scorer = getattr(engine, "reason_and_choose", None)
        if not callable(scorer):
            raise CachedChoiceUnavailableError(
                "Qwen engine does not expose cached reasoning-to-decision scoring"
            )
        answer_type = "CHOICE_MULTI" if request.decision_mode == "MULTI" else "CHOICE_SINGLE"
        result = scorer(
            self._prompt(
                request.model_input,
                reasoning_instruction=request.reasoning_instruction,
            ),
            list(request.model_input.visual_inputs),
            choice_ids=request.candidate_ids,
            option_texts=request.candidate_texts,
            answer_type=answer_type,
            single_choice_suffix=request.single_decision_suffix,
            multi_verify_template=request.multi_verify_template,
            multi_select_threshold=request.select_threshold,
            reasoning_max_new_tokens=self.model_config.max_new_tokens,
        )
        return FiniteDecisionResult(
            selected_ids=result.selected_ids,
            scores=result.scores,
            decision_mode=request.decision_mode,
            reasoning_text=result.reasoning_text,
            provider=f"{self.provider_name}:{self.role}",
            model_id=engine.model_id,
            method=result.method,
            cache_reused=result.cache_reused,
            latency_ms=result.latency_ms,
            metadata={**result.metadata, "purpose": request.purpose},
        )

    def reason_and_choose(self, request: ChoiceScoringRequest) -> ChoiceScoreResult:
        decided = self.reason_and_decide(
            FiniteDecisionRequest(
                model_input=request.model_input,
                decision_mode=("MULTI" if request.answer_type == "CHOICE_MULTI" else "SINGLE"),
                candidate_ids=request.choice_ids,
                candidate_texts=request.option_texts,
                single_decision_suffix=request.single_choice_suffix,
                multi_verify_template=request.multi_verify_template,
                select_threshold=request.multi_select_threshold,
                purpose=request.purpose,
                reasoning_instruction=self._choice_instruction(request.purpose),
            )
        )
        return ChoiceScoreResult(
            selected_ids=decided.selected_ids,
            scores=decided.scores,
            answer_type=request.answer_type,
            reasoning_text=decided.reasoning_text,
            provider=decided.provider,
            model_id=decided.model_id,
            method=decided.method,
            cache_reused=decided.cache_reused,
            latency_ms=decided.latency_ms,
            metadata=decided.metadata,
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
        self.semantic_calls: list[FiniteDecisionRequest] = []
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

    def _finite_fixture_selected_ids(self, request: FiniteDecisionRequest) -> tuple[str, ...]:
        response = self.responses.get(request.purpose)
        if response is None and request.purpose.startswith("semantic_"):
            response = self.responses.get(request.purpose.removeprefix("semantic_"))
        if response is None:
            response = self.responses.get(request.decision_mode.casefold())
        response = (response or self.default).strip()
        try:
            import json

            payload = json.loads(response)
        except (ValueError, TypeError):
            payload = None
        values: list[str] = []
        if isinstance(payload, dict):
            raw = payload.get("selected_ids", payload.get("candidate_ids"))
            if isinstance(raw, list):
                values = [str(item).strip() for item in raw]
        if not values:
            canonical = {item.casefold(): item for item in request.candidate_ids}
            normalized = response.casefold()
            if normalized in {"true", "1"}:
                normalized = "yes"
            elif normalized in {"false", "0"}:
                normalized = "no"
            if normalized in canonical:
                values = [canonical[normalized]]
        if not values and request.decision_mode == "MULTI":
            raw_items = [item.strip() for item in response.split(",")]
            if raw_items and all(item in request.candidate_ids for item in raw_items):
                values = raw_items
        return tuple(dict.fromkeys(item for item in values if item in request.candidate_ids))

    def reason_and_decide(self, request: FiniteDecisionRequest) -> FiniteDecisionResult:
        self.semantic_calls.append(request)
        fixture_scores = self.choice_scores.get(request.purpose)
        if fixture_scores is None:
            fixture_scores = self.choice_scores.get(request.decision_mode.casefold())
        selected_fixture = self._finite_fixture_selected_ids(request)
        scores = {
            candidate_id: (
                float(fixture_scores[candidate_id])
                if fixture_scores is not None and candidate_id in fixture_scores
                else (1.0 if candidate_id in selected_fixture else -1.0)
            )
            for candidate_id in request.candidate_ids
        }
        selected_ids: tuple[str, ...]
        if request.decision_mode in {"SINGLE", "BINARY"}:
            selected_ids = (max(request.candidate_ids, key=lambda item: scores[item]),)
            method = "fake_kv_cached_logits"
            cache_mode = "consume_in_place"
        else:
            selected_ids = tuple(
                candidate_id
                for candidate_id in request.candidate_ids
                if scores[candidate_id] > request.select_threshold
            )
            method = "fake_kv_cached_binary_verification"
            cache_mode = "fork_per_option"
        reasoning = self.responses.get(
            f"{request.purpose}_reasoning",
            self.responses.get("reasoning", "Fake free reasoning is never parsed."),
        )
        return FiniteDecisionResult(
            selected_ids=selected_ids,
            scores=scores,
            decision_mode=request.decision_mode,
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
                "choice_scored_tokens": len(request.candidate_ids),
                "visual_prefill_count": 1 if request.model_input.visual_inputs else 0,
                "reasoning_pass_count": 1,
                "session_released": True,
                "reasoning_cache_mode": cache_mode,
                "peak_vram_mb": None,
                "purpose": request.purpose,
            },
        )

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
        self.status = EvidenceSufficiencyStatus(status)
        self.score = score
        self.calls: list[EvidenceSufficiencyRequest] = []

    def assess(self, request: EvidenceSufficiencyRequest) -> EvidenceSufficiencyResult:
        self.calls.append(request)
        return EvidenceSufficiencyResult(
            self.status,
            self.score,
            reason_code="fake_fixture",
            provider=self.provider_name,
            model_id="fake",
            method="fake_structured_status",
        )


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
