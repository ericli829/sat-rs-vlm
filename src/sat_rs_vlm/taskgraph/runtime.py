"""Composition root for DIRECT and TASKGRAPH_UHR execution."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sat_rs_vlm.infrastructure.config import ModelConfig
from sat_rs_vlm.infrastructure.telemetry import SystemTelemetry, collect_provider_inventory

from .choice import ChoiceRequest, ChoiceResolver
from .executor import CapabilityRouter, ExecutorBinding, GraphExecutor
from .input_composer import InputComposer
from .operators import (
    CountExecutor,
    GeometryExecutor,
    LocateExecutor,
    OperatorContext,
    SelectExecutor,
    SemanticExecutor,
)
from .providers import (
    DetectionProvider,
    DetectionRequest,
    FakeDetectionProvider,
    FakeRegionRetriever,
    FakeSemanticVLMProvider,
    FixturePlannerProvider,
    LazyQwenSemanticProvider,
    LocatorRegionRetrieverAdapter,
    PlannerProvider,
    PlannerRequest,
    ProposalDetectionAdapter,
    RegionRetrieverProvider,
    ScoredGridRegionRetrieverAdapter,
    SemanticVLMProvider,
    VLMRequest,
)
from .routing import DatasetExecutionPolicy, ExecutionMode, ExecutionModeRouter
from .runtime_types import (
    Answer,
    ChoiceResult,
    ImageRef,
    RuntimeObject,
    ScalarInt,
    runtime_summary,
)
from .schema import AnswerType, OperatorName, QuestionType, TargetSpec, TaskGraph, parse_taskgraph
from .store import RuntimeStore
from .tracing import ExecutionTrace


@dataclass(frozen=True)
class RuntimeRequest:
    sample_id: str
    dataset: str
    task_category: str
    question: str
    image_paths: tuple[str, ...]
    options: tuple[str, ...] = ()
    question_type: QuestionType = QuestionType.FREE_FORM
    target_category: str | None = None
    graph: TaskGraph | dict[str, Any] | str | None = None


@dataclass(frozen=True)
class RuntimeResult:
    execution_mode: ExecutionMode
    output: RuntimeObject | ChoiceResult | tuple[RuntimeObject, ...]
    trace: ExecutionTrace
    store: RuntimeStore | None = None


@dataclass
class RuntimeProviders:
    detection: DetectionProvider
    semantic_2b: SemanticVLMProvider
    route_4b: SemanticVLMProvider
    retriever: RegionRetrieverProvider
    choice: SemanticVLMProvider
    planner: PlannerProvider | None = None

    def close(self) -> None:
        seen: set[int] = set()
        for provider in (
            self.detection,
            self.semantic_2b,
            self.route_4b,
            self.retriever,
            self.choice,
        ):
            if id(provider) not in seen:
                provider.close()
                seen.add(id(provider))


class TaskGraphRuntime:
    def __init__(
        self,
        providers: RuntimeProviders,
        *,
        policy: DatasetExecutionPolicy | None = None,
        composer: InputComposer | None = None,
        semantic_categories: set[str] | None = None,
        retrieval_max_candidates: int = 5,
        count_gate_config: dict[str, Any] | None = None,
    ) -> None:
        self.providers = providers
        self.composer = composer or InputComposer()
        self.mode_router = ExecutionModeRouter(policy)
        geometry = GeometryExecutor()
        locate = LocateExecutor(
            providers.detection,
            providers.retriever,
            semantic_categories=semantic_categories,
            max_candidates=retrieval_max_candidates,
        )
        gate = dict(count_gate_config or {})
        count = CountExecutor(
            providers.detection,
            providers.retriever,
            gate_enabled=bool(gate.get("enabled", False)),
            gate_threshold=float(gate.get("threshold", 0.0)),
            gate_max_regions=int(gate.get("max_regions", 9)),
            gate_nms_iou=float(gate.get("nms_iou", 0.5)),
        )
        select = SelectExecutor(providers.semantic_2b)
        semantic = SemanticExecutor(providers.semantic_2b)
        route = SemanticExecutor(providers.route_4b, provider_name="route_vlm")
        bindings = {
            OperatorName.REGION: ExecutorBinding(geometry),
            OperatorName.REGION_FROM_BBOX: ExecutorBinding(geometry),
            OperatorName.FIND_MARKER: ExecutorBinding(geometry),
            OperatorName.LOCATE: ExecutorBinding(locate),
            OperatorName.SELECT: ExecutorBinding(select),
            OperatorName.GROUP: ExecutorBinding(geometry),
            OperatorName.COUNT: ExecutorBinding(count),
            OperatorName.ABS_DIFF: ExecutorBinding(geometry),
            OperatorName.BUILD_ROUTE_CONTEXT: ExecutorBinding(geometry),
            OperatorName.ATTRIBUTE: ExecutorBinding(semantic),
            OperatorName.CLASSIFY: ExecutorBinding(semantic),
            OperatorName.MULTILABEL_CLASSIFY: ExecutorBinding(semantic),
            OperatorName.MOTION: ExecutorBinding(semantic),
            OperatorName.RELATION: ExecutorBinding(semantic),
            OperatorName.VLM_REASON: ExecutorBinding(semantic),
            OperatorName.ROUTE_REASON: ExecutorBinding(route),
            OperatorName.MATCH_CHOICE: ExecutorBinding(semantic),
        }
        self.graph_executor = GraphExecutor(CapabilityRouter(bindings))
        self.choice_resolver = ChoiceResolver(providers.choice, self.composer)

    @staticmethod
    def _images(request: RuntimeRequest) -> dict[str, ImageRef]:
        return {
            f"image{index}": ImageRef(path, provenance={"dataset": request.dataset})
            for index, path in enumerate(request.image_paths)
        }

    def _choice_or_answer(
        self,
        sources: tuple[RuntimeObject, ...],
        question: str,
        options: tuple[str, ...],
        answer_type: AnswerType,
    ) -> RuntimeObject | ChoiceResult | tuple[RuntimeObject, ...]:
        if answer_type in {AnswerType.CHOICE_SINGLE, AnswerType.CHOICE_MULTI}:
            return self.choice_resolver.resolve(ChoiceRequest(sources, question, options))
        return sources[0] if len(sources) == 1 else sources

    def _taskgraph(self, request: RuntimeRequest, images: dict[str, ImageRef]) -> RuntimeResult:
        phase_timing: dict[str, float | None] = {
            "planner": None,
            "executor": None,
            "postprocess": None,
        }
        if request.graph is not None:
            graph = (
                request.graph
                if isinstance(request.graph, TaskGraph)
                else parse_taskgraph(request.graph)
            )
        elif self.providers.planner is not None:
            planner_started = time.perf_counter()
            graph = self.providers.planner.plan(
                PlannerRequest(
                    request.question,
                    request.question_type.value,
                    request.options,
                    {f"${key}": value for key, value in images.items()},
                    request.sample_id,
                )
            )
            phase_timing["planner"] = (time.perf_counter() - planner_started) * 1000.0
        else:
            raise ValueError("TASKGRAPH_UHR requires graph input or a configured PlannerProvider")
        if graph.question != request.question:
            raise ValueError("TaskGraph question differs from the dataset question")
        if tuple(graph.choices or ()) != request.options:
            raise ValueError("TaskGraph choices differ from the original dataset options")
        store = RuntimeStore({f"${key}": value for key, value in images.items()})
        context = OperatorContext(request.question, request.options, self.composer)
        trace = self.graph_executor.execute(
            graph,
            store,
            sample_id=request.sample_id,
            execution_mode=ExecutionMode.TASKGRAPH_UHR.value,
            context=context,
        )
        executor_timing = trace.telemetry.get("executor", {}).get("timing_ms", {})
        if isinstance(executor_timing, dict):
            phase_timing["executor"] = executor_timing.get("e2e")
        postprocess_started = time.perf_counter()
        sources = tuple(store.get(ref) for ref in graph.final.sources)
        output = self._choice_or_answer(
            sources, graph.final.question, request.options, graph.final.answer_type
        )
        if isinstance(output, ChoiceResult):
            trace.choice_provider = str(output.provenance.get("provider", "unknown"))
            trace.choice_result = {
                "choice_id": output.choice_id,
                "raw_response": output.raw_response,
                "confidence": output.confidence,
                "provenance": output.provenance,
            }
        else:
            trace.result = (
                runtime_summary(output)
                if not isinstance(output, tuple)
                else {"sources": [runtime_summary(item) for item in output]}
            )
        phase_timing["postprocess"] = (time.perf_counter() - postprocess_started) * 1000.0
        trace.telemetry["phase_timing_ms"] = phase_timing
        trace.telemetry["generation_events"] = [
            dict(node.telemetry["generation"])
            for node in trace.nodes
            if isinstance(node.telemetry.get("generation"), dict)
        ]
        if trace.choice_result is not None:
            choice_generation = trace.choice_result.get("provenance", {}).get("generation")
            if isinstance(choice_generation, dict):
                trace.telemetry["generation_events"].append(dict(choice_generation))
        return RuntimeResult(ExecutionMode.TASKGRAPH_UHR, output, trace, store)

    def _direct_vlm(self, request: RuntimeRequest, images: dict[str, ImageRef]) -> RuntimeResult:
        sources = tuple(images.values())
        trace = ExecutionTrace(request.sample_id, ExecutionMode.DIRECT_VLM.value)
        if request.options:
            output: RuntimeObject | ChoiceResult = self.choice_resolver.resolve(
                ChoiceRequest(sources, request.question, request.options)
            )
            trace.choice_provider = str(output.provenance.get("provider", "unknown"))
            trace.choice_result = {
                "choice_id": output.choice_id,
                "raw_response": output.raw_response,
            }
        else:
            model_input = self.composer.compose(list(sources), question=request.question)
            result = self.providers.semantic_2b.infer(VLMRequest(model_input, "direct_vlm"))
            output = Answer(
                result.text,
                result.confidence,
                {"provider": result.provider, **result.metadata},
            )
            trace.result = runtime_summary(output)
            generation = result.metadata.get("generation")
            if isinstance(generation, dict):
                trace.telemetry["generation_events"] = [dict(generation)]
        return RuntimeResult(ExecutionMode.DIRECT_VLM, output, trace)

    def _direct_detection(
        self, request: RuntimeRequest, images: dict[str, ImageRef]
    ) -> RuntimeResult:
        if len(images) != 1:
            raise ValueError("DIRECT_DETECTION requires exactly one image")
        target = TargetSpec(category=request.target_category or "object")
        detected = self.providers.detection.detect(
            DetectionRequest(next(iter(images.values())), target, request.task_category)
        )
        is_count = request.task_category.casefold() in {"count", "counting"}
        source: RuntimeObject = (
            ScalarInt(
                len(detected.detections.entities),
                {"provider": detected.provider, "detection": detected.detections.provenance},
            )
            if is_count
            else detected.detections
        )
        trace = ExecutionTrace(request.sample_id, ExecutionMode.DIRECT_DETECTION.value)
        if request.options:
            output: RuntimeObject | ChoiceResult = self.choice_resolver.resolve(
                ChoiceRequest((source,), request.question, request.options)
            )
            trace.choice_provider = str(output.provenance.get("provider", "unknown"))
            trace.choice_result = {
                "choice_id": output.choice_id,
                "raw_response": output.raw_response,
            }
        else:
            output = source
            trace.result = runtime_summary(source)
        return RuntimeResult(ExecutionMode.DIRECT_DETECTION, output, trace)

    def run(self, request: RuntimeRequest) -> RuntimeResult:
        monitor = SystemTelemetry("taskgraph_runtime_request", reset_cuda_peaks=True)
        preprocess_started = time.perf_counter()
        with monitor:
            if not request.image_paths:
                raise ValueError("runtime request requires at least one image")
            images = self._images(request)
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
            mode = self.mode_router.route(request.dataset, request.task_category)
            if mode is ExecutionMode.DIRECT_VLM:
                result = self._direct_vlm(request, images)
            elif mode is ExecutionMode.DIRECT_DETECTION:
                result = self._direct_detection(request, images)
            else:
                result = self._taskgraph(request, images)
        system_telemetry = monitor.to_dict()
        system_telemetry.update(
            {
                "sample_id": request.sample_id,
                "dataset": request.dataset,
                "task_category": request.task_category,
                "execution_mode": result.execution_mode.value,
                "provider_inventory": collect_provider_inventory(
                    [
                        self.providers.detection,
                        self.providers.semantic_2b,
                        self.providers.route_4b,
                        self.providers.retriever,
                        self.providers.choice,
                        self.providers.planner,
                    ]
                ),
            }
        )
        result.trace.telemetry["system"] = system_telemetry
        phase_timing = result.trace.telemetry.setdefault("phase_timing_ms", {})
        if isinstance(phase_timing, dict):
            phase_timing.setdefault("preprocess", preprocess_ms)
            phase_timing.setdefault("e2e", system_telemetry["timing_ms"]["e2e"])
            phase_timing.setdefault("ttft", None)
        generation_events = result.trace.telemetry.setdefault("generation_events", [])
        if not generation_events:
            candidates: list[dict[str, Any]] = []
            if result.trace.result and isinstance(result.trace.result.get("provenance"), dict):
                generation = result.trace.result["provenance"].get("generation")
                if isinstance(generation, dict):
                    candidates.append(generation)
            if result.trace.choice_result:
                generation = result.trace.choice_result.get("provenance", {}).get("generation")
                if isinstance(generation, dict):
                    candidates.append(generation)
            generation_events.extend(candidates)
        return result

    def close(self) -> None:
        self.providers.close()
        self.composer.close()


def fake_runtime(
    *,
    detection_boxes: list[list[float]] | None = None,
    semantic_responses: dict[str, str] | None = None,
    route_responses: dict[str, str] | None = None,
    choice_responses: dict[str, str] | None = None,
    retrieval_candidates: list[tuple[list[float], float]] | None = None,
    planner_fixtures: dict[str, TaskGraph | dict[str, Any]] | None = None,
    policy: DatasetExecutionPolicy | None = None,
) -> TaskGraphRuntime:
    return TaskGraphRuntime(
        RuntimeProviders(
            detection=FakeDetectionProvider(detection_boxes),
            semantic_2b=FakeSemanticVLMProvider(semantic_responses),
            route_4b=FakeSemanticVLMProvider(route_responses),
            retriever=FakeRegionRetriever(retrieval_candidates),
            choice=FakeSemanticVLMProvider(choice_responses),
            planner=FixturePlannerProvider(planner_fixtures or {}) if planner_fixtures else None,
        ),
        policy=policy,
    )


def runtime_from_config(config: dict[str, Any]) -> TaskGraphRuntime:
    """Build providers lazily from an explicit YAML/JSON-style mapping."""

    import os

    from sat_rs_vlm.configuration.environment import expand_environment

    config = expand_environment(config, environ=os.environ)
    providers = config.get("providers", {})
    if not isinstance(providers, dict):
        raise TypeError("providers config must be a mapping")

    detection_cfg = dict(providers.get("detection", {"kind": "fake"}))
    detection_kind = str(detection_cfg.pop("kind", "fake"))
    if detection_kind == "fake":
        detection: DetectionProvider = FakeDetectionProvider(detection_cfg.get("boxes"))
    else:
        from sat_rs_vlm.integrations.detectors.registry import create_proposal_provider

        detection = ProposalDetectionAdapter(
            create_proposal_provider(detection_kind, detection_cfg)
        )

    def semantic_provider(name: str, role: str) -> SemanticVLMProvider:
        section = dict(providers.get(name, {"kind": "fake"}))
        kind = str(section.pop("kind", "fake"))
        if kind == "fake":
            return FakeSemanticVLMProvider(
                section.get("responses"), str(section.get("default", "A"))
            )
        if kind == "qwen3_vl":
            return LazyQwenSemanticProvider(ModelConfig.model_validate(section), role=role)
        raise ValueError(f"unsupported semantic provider kind: {kind}")

    semantic_2b = semantic_provider("semantic_2b", "general_2b")
    route_4b = semantic_provider("route_4b", "route_4b")
    choice = semantic_provider("choice", "choice_2b")

    retriever_cfg = dict(providers.get("region_retriever", {"kind": "fake"}))
    retriever_kind = str(retriever_cfg.pop("kind", "fake"))
    if retriever_kind == "fake":
        candidates = [(item["bbox"], item["score"]) for item in retriever_cfg.get("candidates", [])]
        retriever: RegionRetrieverProvider = FakeRegionRetriever(candidates)
    elif retriever_kind == "uhr_locator":
        from sat_rs_vlm.integrations.locators.registry import create_locator

        retriever = LocatorRegionRetrieverAdapter(create_locator("hierarchical", retriever_cfg))
    else:
        from sat_rs_vlm.integrations.retrievers.registry import create_retriever_provider

        scorer = create_retriever_provider(retriever_kind, retriever_cfg)
        retriever = ScoredGridRegionRetrieverAdapter(
            scorer,
            grid_size=int(retriever_cfg.get("grid_size", 3)),
            default_max_candidates=int(retriever_cfg.get("max_candidates", 5)),
            candidate_window_ratio=(
                float(retriever_cfg["candidate_window_ratio"])
                if retriever_cfg.get("candidate_window_ratio") is not None
                else None
            ),
        )

    planner = None
    planner_cfg = dict(providers.get("planner", {}))
    if planner_cfg:
        kind = str(planner_cfg.get("kind", "fixture"))
        if kind != "fixture":
            raise ValueError("only fixture planner is available until a checkpoint is selected")
        fixture_path = Path(str(planner_cfg["fixture_file"]))
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        planner = FixturePlannerProvider(payload)

    policy = DatasetExecutionPolicy.from_mapping(config.get("dataset_policy"))
    composer_cfg = config.get("input_composer", {})
    if not isinstance(composer_cfg, dict):
        raise TypeError("input_composer config must be a mapping")
    composer = InputComposer(
        candidate_halo_ratio=float(composer_cfg.get("candidate_halo_ratio", 0.2))
    )
    return TaskGraphRuntime(
        RuntimeProviders(detection, semantic_2b, route_4b, retriever, choice, planner),
        policy=policy,
        composer=composer,
        semantic_categories=set(config.get("semantic_region_categories", [])),
        retrieval_max_candidates=int(retriever_cfg.get("max_candidates", 5)),
        count_gate_config=dict(config.get("count_gate", {})),
    )
