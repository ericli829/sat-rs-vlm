"""Replaceable capability contracts and adapters for production providers."""

from __future__ import annotations

import math
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
        limit = request.max_candidates or self.default_max_candidates
        order = order[:limit]
        candidates = tuple(
            RegionCandidate(
                Region(
                    image,
                    boxes[index],
                    {
                        "retriever": self.provider_name,
                        "coordinate_mode": "absolute_original_pixel_xyxy",
                        "tile": {
                            "level": 1,
                            "index": index,
                            "row": index // self.grid_size,
                            "column": index % self.grid_size,
                            "grid_size": self.grid_size,
                        },
                    },
                ),
                scored.scores[index],
                {
                    "provider": self.provider_name,
                    "model_id": scored.model_id,
                    "bbox_xyxy_global": list(boxes[index]),
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

    def infer(self, request: VLMRequest) -> VLMResult:
        model_input = request.model_input
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
    """Parse stable letter ids, with numeric ids retained for compatibility."""

    letters = re.findall(r"(?<![A-Z0-9])([A-Z])(?![A-Z0-9])", text.upper())
    if letters:
        values = [ord(item) - ord("A") for item in letters]
        if any(item < 0 or item >= count for item in values):
            raise ValueError(f"selection provider returned out-of-range candidate ids: {letters}")
        return tuple(dict.fromkeys(values))
    values = [int(item) for item in re.findall(r"\d+", text)]
    if not values:
        raise ValueError(f"selection provider returned no candidate ids: {text!r}")
    if 0 not in values and all(1 <= item <= count for item in values):
        values = [item - 1 for item in values]
    if any(item < 0 or item >= count for item in values):
        raise ValueError(f"selection provider returned out-of-range candidate ids: {values}")
    return tuple(dict.fromkeys(values))
