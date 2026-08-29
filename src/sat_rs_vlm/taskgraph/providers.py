"""Replaceable capability contracts and adapters for production providers."""

from __future__ import annotations

import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from sat_rs_vlm.infrastructure.config import ModelConfig
from sat_rs_vlm.integrations.detectors.protocol import ProposalProvider
from sat_rs_vlm.integrations.locators.protocol import LocatorProvider
from sat_rs_vlm.integrations.retrievers.protocol import RetrieverProvider
from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine

from .runtime_types import (
    Entity,
    EntitySet,
    ImageRef,
    Region,
)
from .schema import TargetSpec, TaskGraph


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
class ModelInput:
    visual_inputs: tuple[str, ...]
    structured_context: str
    question: str
    options: tuple[str, ...] = ()
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


class SemanticVLMProvider(Protocol):
    provider_name: str

    def infer(self, request: VLMRequest) -> VLMResult: ...

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
        self.boxes = [tuple(float(item) for item in box) for box in (boxes or [])]
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
        result = self._locator.locate(image.path, request.query)
        limit = request.max_candidates or len(result.regions_xyxy)
        candidates = tuple(
            RegionCandidate(
                Region(image, tuple(box), {"locator": self.provider_name}),
                float(score),
                {"locator": self.provider_name, "details": dict(result.region_details[index])},
            )
            for index, (box, score) in enumerate(
                zip(result.regions_xyxy[:limit], result.scores[:limit], strict=True)
            )
        )
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
        scope = (
            request.search_scope.bbox_xyxy_global
            if request.search_scope is not None
            else request.image.bbox_xyxy_global
            if isinstance(request.image, Region)
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
        items = self._candidates[: request.max_candidates or len(self._candidates)]
        return RegionCandidates(
            tuple(
                RegionCandidate(
                    Region(image, tuple(float(value) for value in box), {"fake": True}),
                    float(score),
                    {"provider": self.provider_name},
                )
                for box, score in items
            ),
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

    def infer(self, request: VLMRequest) -> VLMResult:
        model_input = request.model_input
        prompt_parts = []
        if model_input.structured_context:
            prompt_parts.append("Structured results:\n" + model_input.structured_context)
        prompt_parts.append("Question:\n" + model_input.question)
        if model_input.options:
            prompt_parts.append("Options:\n" + "\n".join(model_input.options))
            prompt_parts.append("Return only the option id.")
        prompt = "\n\n".join(prompt_parts)
        engine = self._load()
        if not model_input.visual_inputs:
            # Qwen is a VLM, but structured authoritative choice must remain text-only.
            generated = engine.generate_text(prompt=prompt, image_paths=[])
        else:
            generated = engine.generate_text(
                prompt=prompt, image_paths=list(model_input.visual_inputs)
            )
        return VLMResult(
            generated,
            f"{self.provider_name}:{self.role}",
            metadata={"output_contract": request.output_contract},
        )

    def close(self) -> None:
        self._engine = None


class FakeSemanticVLMProvider:
    provider_name = "fake_vlm"

    def __init__(self, responses: Mapping[str, str] | None = None, default: str = "A") -> None:
        self.responses = dict(responses or {})
        self.default = default
        self.calls: list[VLMRequest] = []

    def infer(self, request: VLMRequest) -> VLMResult:
        self.calls.append(request)
        response = self.responses.get(request.output_contract)
        if response is None and request.output_contract == "choice":
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
    """Parse 0/1-based candidate ids from a provider response."""

    values = [int(item) for item in re.findall(r"\d+", text)]
    if not values:
        raise ValueError(f"selection provider returned no candidate ids: {text!r}")
    if 0 not in values and all(1 <= item <= count for item in values):
        values = [item - 1 for item in values]
    if any(item < 0 or item >= count for item in values):
        raise ValueError(f"selection provider returned out-of-range candidate ids: {values}")
    return tuple(dict.fromkeys(values))
