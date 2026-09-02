"""Composition root for DIRECT and TASKGRAPH_UHR execution."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sat_rs_vlm.infrastructure.config import ModelConfig

from .answerability import AnswerabilityConfig, EvidenceSufficiencyExecutor
from .capabilities import TargetCapabilityClassifier
from .choice import ChoiceRequest, ChoiceResolver
from .choice_config import ChoiceSystemConfig
from .execution_plan import FinalChoiceFusionConfig
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
    EvidenceSufficiencyRequest,
    EvidenceSufficiencyResult,
    FakeDetectionProvider,
    FakeRegionRetriever,
    FakeSemanticVLMProvider,
    FixturePlannerProvider,
    LazyQwenSemanticProvider,
    LocatorRegionRetrieverAdapter,
    PlannerProvider,
    PlannerRequest,
    ProposalDetectionAdapter,
    Qwen3VLPlannerProvider,
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
    unwrap_select_result,
)
from .schema import (
    AnswerType,
    OperatorName,
    QuestionType,
    TargetSpec,
    TaskGraph,
    parse_taskgraph,
)
from .semantic_decision import SemanticDecisionConfig
from .store import RuntimeStore
from .tracing import (
    SYSTEM_TELEMETRY_STAGES,
    ExecutionTrace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RuntimeRequest:
    sample_id: str
    dataset: str
    task_category: str
    question: str
    image_paths: tuple[str, ...]
    options: tuple[str, ...] = ()
    question_type: QuestionType = QuestionType.FREE_FORM
    choice_answer_type: AnswerType | None = None
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
            self.planner,
        ):
            if provider is None:
                continue
            if id(provider) not in seen:
                provider.close()
                seen.add(id(provider))


def _provider_children(provider: Any) -> tuple[Any, ...]:
    children: list[Any] = []
    for name in (
        "_provider",
        "_locator",
        "_retriever_provider",
        "_detector_provider",
        "retriever_provider",
        "detector_provider",
        "_engine",
        "base_provider",
    ):
        child = getattr(provider, name, None)
        if child is not None and child is not provider:
            children.append(child)
    return tuple(children)


def _provider_parameter_count(provider: Any) -> int | str:
    seen: set[int] = set()
    pending = [provider]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        model = current
        if hasattr(current, "_model"):
            model = current._model
        if model is not None:
            parameters = getattr(model, "parameters", None)
            if callable(parameters):
                try:
                    return sum(int(parameter.numel()) for parameter in parameters())
                except Exception:
                    pass
        try:
            pending.extend(_provider_children(current))
        except Exception:
            pass
    return "NOT_AVAILABLE"


def _provider_names(provider: Any) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[int] = set()
    pending = [provider]
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        try:
            name = getattr(current, "provider_name", None)
        except Exception:
            name = None
        if name is not None and str(name).strip() and str(name) not in names:
            names.append(str(name))
        try:
            pending.extend(_provider_children(current))
        except Exception:
            pass
    return tuple(names)


def _provider_torch(providers: RuntimeProviders) -> Any | None:
    seen: set[int] = set()
    pending: list[Any] = [
        providers.detection,
        providers.semantic_2b,
        providers.route_4b,
        providers.retriever,
        providers.choice,
        providers.planner,
    ]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        try:
            torch = getattr(current, "_torch", None)
        except Exception:
            torch = None
        if torch is not None:
            return torch
        try:
            pending.extend(_provider_children(current))
        except Exception:
            pass
    return None


def _reset_runtime_resources(providers: RuntimeProviders) -> Any | None:
    try:
        torch = _provider_torch(providers)
        cuda = getattr(torch, "cuda", None)
        available = bool(getattr(cuda, "is_available", lambda: False)())
    except Exception:
        return torch if "torch" in locals() else None
    if cuda is None or not available:
        return torch
    try:
        cuda.synchronize()
        cuda.reset_peak_memory_stats()
    except Exception:
        pass
    return torch


def _capture_runtime_resources(torch: Any | None) -> dict[str, Any]:
    try:
        from sat_rs_vlm.evaluation.performance import process_memory_snapshot_mb
    except Exception:
        process_memory_snapshot_mb = None

    try:
        cuda = getattr(torch, "cuda", None)
        gpu_available = bool(
            cuda is not None and bool(getattr(cuda, "is_available", lambda: False)())
        )
    except Exception:
        cuda = None
        gpu_available = False
    resource_errors: list[str] = []
    allocated: float | None = None
    reserved: float | None = None
    if gpu_available:
        try:
            cuda.synchronize()
            allocated = float(cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            reserved = float(cuda.max_memory_reserved()) / (1024.0 * 1024.0)
        except Exception as exc:
            allocated = None
            reserved = None
            resource_errors.append(f"gpu:{type(exc).__name__}")
    try:
        memory = process_memory_snapshot_mb() if process_memory_snapshot_mb else {}
    except Exception as exc:
        memory = {}
        resource_errors.append(f"rss:{type(exc).__name__}")
    return {
        "peak_gpu_allocated_mb": allocated,
        "peak_gpu_reserved_mb": reserved,
        "process_rss_mb": memory.get("rss_mb"),
        "process_peak_rss_mb": memory.get("os_peak_rss_mb"),
        "gpu_status": "available" if gpu_available else "NOT_AVAILABLE_FROM_BACKEND",
        "rss_status": "available" if memory.get("rss_mb") is not None else "NOT_AVAILABLE",
        "telemetry_errors": resource_errors,
        "measurement_window": (
            "TaskGraphRuntime.run: before provider execution through final output; "
            "GPU peak counters reset at run start"
        ),
    }


class TaskGraphRuntime:
    def __init__(
        self,
        providers: RuntimeProviders,
        *,
        policy: DatasetExecutionPolicy | None = None,
        composer: InputComposer | None = None,
        semantic_categories: set[str] | None = None,
        capability_classifier: TargetCapabilityClassifier | None = None,
        choice_config: ChoiceSystemConfig | None = None,
        semantic_decision_config: SemanticDecisionConfig | None = None,
        final_choice_fusion_config: FinalChoiceFusionConfig | None = None,
        answerability_config: AnswerabilityConfig | None = None,
    ) -> None:
        self.providers = providers
        self.composer = composer or InputComposer()
        self.choice_config = choice_config or ChoiceSystemConfig()
        self.semantic_decision_config = semantic_decision_config or SemanticDecisionConfig()
        self.final_choice_fusion_config = final_choice_fusion_config or FinalChoiceFusionConfig()
        self.answerability_config = answerability_config or AnswerabilityConfig()
        self.mode_router = ExecutionModeRouter(policy)
        geometry = GeometryExecutor()
        locate = LocateExecutor(
            providers.detection,
            providers.retriever,
            semantic_categories=semantic_categories,
            capability_classifier=capability_classifier,
        )
        count = CountExecutor(providers.detection)
        select = SelectExecutor(providers.semantic_2b, self.choice_config)
        semantic = SemanticExecutor(
            providers.semantic_2b,
            model_role="semantic_2b",
            choice_config=self.choice_config,
            semantic_config=self.semantic_decision_config,
        )
        route = SemanticExecutor(
            providers.route_4b,
            provider_name="route_vlm",
            model_role="route_4b",
            choice_config=self.choice_config,
            semantic_config=self.semantic_decision_config,
        )
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
        self.graph_executor = GraphExecutor(
            CapabilityRouter(bindings),
            self.final_choice_fusion_config,
        )
        self.choice_resolver = ChoiceResolver(providers.choice, self.composer, self.choice_config)
        self.answerability = EvidenceSufficiencyExecutor(
            providers.semantic_2b,
            self.composer,
            self.answerability_config,
        )

    def assess_answerability(
        self, request: EvidenceSufficiencyRequest
    ) -> EvidenceSufficiencyResult:
        """Run the optional sufficiency service; it never changes graph execution control flow."""

        return self.answerability.assess(request)

    @staticmethod
    def _images(request: RuntimeRequest) -> dict[str, ImageRef]:
        return {
            f"image{index}": ImageRef(path, provenance={"dataset": request.dataset})
            for index, path in enumerate(request.image_paths)
        }

    def _choice_or_answer(
        self,
        sources: tuple[RuntimeObject, ...],
        question: str | None,
        options: tuple[str, ...],
        answer_type: AnswerType,
    ) -> RuntimeObject | ChoiceResult | tuple[RuntimeObject, ...]:
        if answer_type in {AnswerType.CHOICE_SINGLE, AnswerType.CHOICE_MULTI}:
            return self.choice_resolver.resolve(
                ChoiceRequest(sources, question, options, answer_type)
            )
        return sources[0] if len(sources) == 1 else sources

    @staticmethod
    def _direct_choice_answer_type(request: RuntimeRequest) -> AnswerType:
        if request.choice_answer_type is not None:
            return request.choice_answer_type
        if request.question_type is QuestionType.MULTIPLE_CHOICE_MULTI:
            return AnswerType.CHOICE_MULTI
        return AnswerType.CHOICE_SINGLE

    def _taskgraph(self, request: RuntimeRequest, images: dict[str, ImageRef]) -> RuntimeResult:
        planner_ms = 0.0
        planner_status = "deferred"
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
            planner_ms = (time.perf_counter() - planner_started) * 1000.0
            planner_status = "executed"
        else:
            raise ValueError("TASKGRAPH_UHR requires graph input or a configured PlannerProvider")
        if graph.question != request.question:
            raise ValueError("TaskGraph question differs from the dataset question")
        if tuple(graph.choices or ()) != request.options:
            raise ValueError("TaskGraph choices differ from the original dataset options")
        store = RuntimeStore({f"${key}": value for key, value in images.items()})
        context = OperatorContext(request.question, request.options, self.composer)
        graph_started = time.perf_counter()
        trace = self.graph_executor.execute(
            graph,
            store,
            sample_id=request.sample_id,
            execution_mode=ExecutionMode.TASKGRAPH_UHR.value,
            context=context,
        )
        trace.telemetry["graph_execution_ms"] = (time.perf_counter() - graph_started) * 1000.0
        trace.telemetry["planner_status"] = planner_status
        trace.telemetry["planner_ms"] = planner_ms
        planner_metadata = getattr(self.providers.planner, "last_metadata", None)
        if isinstance(planner_metadata, Mapping):
            trace.telemetry["planner_metadata"] = dict(planner_metadata)
        postprocess_started = time.perf_counter()
        choice_requested = graph.final.answer_type in {
            AnswerType.CHOICE_SINGLE,
            AnswerType.CHOICE_MULTI,
        }
        sources = tuple(
            unwrap_select_result(
                store.get(ref),
                allow_empty=False,
                consumer=f"final.sources[{ref}]",
            )
            for ref in graph.final.sources
        )
        choice_started = time.perf_counter()
        output = self._choice_or_answer(
            sources,
            graph.final.question or None,
            request.options,
            graph.final.answer_type,
        )
        choice_ms = (time.perf_counter() - choice_started) * 1000.0 if choice_requested else 0.0
        choice_model_called = bool(
            choice_requested
            and self.choice_resolver.last_score_result is not None
            and self.choice_resolver.last_score_result.provider != "structured_deterministic"
            and not any(type(source).__name__ == "ChoiceScoreResult" for source in sources)
        )
        trace.telemetry["choice_ms"] = choice_ms
        trace.telemetry["choice_model_called"] = choice_model_called
        trace.telemetry["choice_status"] = (
            "executed"
            if choice_model_called
            else "deterministic_or_precomputed"
            if choice_requested
            else "not_used"
        )
        if isinstance(output, ChoiceResult):
            trace.choice_provider = str(output.provenance.get("provider", "unknown"))
            trace.choice_result = runtime_summary(output)
        else:
            trace.result = (
                runtime_summary(output)
                if not isinstance(output, tuple)
                else {"sources": [runtime_summary(item) for item in output]}
            )
        trace.telemetry["postprocess_ms"] = max(
            0.0,
            (time.perf_counter() - postprocess_started) * 1000.0 - choice_ms,
        )
        return RuntimeResult(ExecutionMode.TASKGRAPH_UHR, output, trace, store)

    def _direct_vlm(self, request: RuntimeRequest, images: dict[str, ImageRef]) -> RuntimeResult:
        sources = tuple(images.values())
        trace = ExecutionTrace(request.sample_id, ExecutionMode.DIRECT_VLM.value)
        if request.options:
            choice_started = time.perf_counter()
            choice_output = self.choice_resolver.resolve(
                ChoiceRequest(
                    sources,
                    request.question,
                    request.options,
                    self._direct_choice_answer_type(request),
                )
            )
            trace.telemetry["choice_ms"] = (time.perf_counter() - choice_started) * 1000.0
            trace.telemetry["choice_model_called"] = bool(
                self.choice_resolver.last_score_result is not None
                and self.choice_resolver.last_score_result.provider != "structured_deterministic"
            )
            trace.telemetry["choice_status"] = (
                "executed"
                if trace.telemetry["choice_model_called"]
                else "deterministic_or_precomputed"
            )
            output: RuntimeObject | ChoiceResult = choice_output
            trace.choice_provider = str(choice_output.provenance.get("provider", "unknown"))
            trace.choice_result = runtime_summary(choice_output)
        else:
            model_input = self.composer.compose(list(sources), question=request.question)
            semantic_started = time.perf_counter()
            result = self.providers.semantic_2b.infer(VLMRequest(model_input, "direct_vlm"))
            trace.telemetry["semantic_vlm_ms"] = (
                time.perf_counter() - semantic_started
            ) * 1000.0
            trace.telemetry["choice_model_called"] = False
            trace.telemetry["choice_status"] = "not_used"
            trace.telemetry["semantic_metadata"] = dict(result.metadata)
            output = Answer(result.text, result.confidence, {"provider": result.provider})
            trace.result = runtime_summary(output)
        return RuntimeResult(ExecutionMode.DIRECT_VLM, output, trace)

    def _direct_detection(
        self, request: RuntimeRequest, images: dict[str, ImageRef]
    ) -> RuntimeResult:
        if len(images) != 1:
            raise ValueError("DIRECT_DETECTION requires exactly one image")
        target = TargetSpec(category=request.target_category or "object")
        detector_started = time.perf_counter()
        detected = self.providers.detection.detect(
            DetectionRequest(next(iter(images.values())), target, request.task_category)
        )
        detector_ms = (time.perf_counter() - detector_started) * 1000.0
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
        trace.telemetry["detector_ms"] = detector_ms
        trace.telemetry["detector_metadata"] = dict(detected.metadata)
        if request.options:
            choice_started = time.perf_counter()
            choice_output = self.choice_resolver.resolve(
                ChoiceRequest(
                    (source,),
                    request.question,
                    request.options,
                    self._direct_choice_answer_type(request),
                )
            )
            trace.telemetry["choice_ms"] = (time.perf_counter() - choice_started) * 1000.0
            trace.telemetry["choice_model_called"] = bool(
                self.choice_resolver.last_score_result is not None
                and self.choice_resolver.last_score_result.provider != "structured_deterministic"
            )
            trace.telemetry["choice_status"] = (
                "executed"
                if trace.telemetry["choice_model_called"]
                else "deterministic_or_precomputed"
            )
            output: RuntimeObject | ChoiceResult = choice_output
            trace.choice_provider = str(choice_output.provenance.get("provider", "unknown"))
            trace.choice_result = runtime_summary(choice_output)
        else:
            output = source
            trace.result = runtime_summary(source)
            trace.telemetry["choice_model_called"] = False
            trace.telemetry["choice_status"] = "not_used"
        return RuntimeResult(ExecutionMode.DIRECT_DETECTION, output, trace)

    @staticmethod
    def _latency_value(metadata: Mapping[str, Any]) -> float | None:
        raw = metadata.get("latency_ms")
        if isinstance(raw, (int, float)):
            return max(0.0, float(raw))
        if not isinstance(raw, Mapping):
            return None
        for key in ("total_ms", "total", "choice_total_ms", "reasoning_total_ms"):
            value = raw.get(key)
            if isinstance(value, (int, float)):
                return max(0.0, float(value))
        return None

    @staticmethod
    def _node_stage(node: Any) -> str | None:
        metadata = node.trace_metadata
        stage = metadata.get("stage")
        if stage in {"detector", "retriever", "semantic_vlm", "route_vlm", "choice"}:
            return str(stage)
        capability = metadata.get("capability")
        if capability == "DETECTOR":
            return "detector"
        if capability == "RETRIEVER":
            return "retriever"
        if metadata.get("model_role") == "route_4b" or node.operator == "ROUTE_REASON":
            return "route_vlm"
        if node.operator in {
            "ATTRIBUTE",
            "CLASSIFY",
            "MULTILABEL_CLASSIFY",
            "MOTION",
            "RELATION",
            "VLM_REASON",
            "MATCH_CHOICE",
        }:
            return "semantic_vlm"
        if node.operator == "SELECT" and str(node.provider).startswith("qwen3_vl"):
            return "semantic_vlm"
        return None

    @staticmethod
    def _add_unique(values: list[str], candidate: object) -> None:
        if candidate is None or not isinstance(candidate, (str, int, float)):
            return
        text = str(candidate).strip()
        if text and text.casefold() not in {"unknown", "none"} and text not in values:
            values.append(text)

    @classmethod
    def _collect_named_values(
        cls, value: object, keys: set[str], output: list[str], *, depth: int = 0
    ) -> None:
        if depth > 5:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in keys:
                    cls._add_unique(output, item)
                cls._collect_named_values(item, keys, output, depth=depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                cls._collect_named_values(item, keys, output, depth=depth + 1)

    def _finalize_trace(
        self,
        trace: ExecutionTrace,
        *,
        mode: ExecutionMode,
        routing_ms: float,
        started: float,
        torch: Any | None,
    ) -> None:
        stage_values: dict[str, float] = {}
        stage_status: dict[str, str] = {}
        for node in trace.nodes:
            stage = self._node_stage(node)
            if stage is None:
                continue
            value = self._latency_value(node.trace_metadata)
            if value is None:
                value = max(0.0, float(node.latency_ms))
            stage_values[stage] = stage_values.get(stage, 0.0) + value
            stage_status[stage + "_ms"] = "executed"
        for stage in ("semantic_vlm", "route_vlm", "retriever", "detector"):
            value = trace.telemetry.get(stage + "_ms")
            if isinstance(value, (int, float)):
                stage_values[stage] = max(0.0, float(value))
                stage_status[stage + "_ms"] = "executed"

        choice_value = trace.telemetry.get("choice_ms")
        if isinstance(choice_value, (int, float)):
            trace.choice_ms = max(0.0, float(choice_value))
            choice_called = trace.telemetry.get("choice_model_called") is True
            stage_values["choice"] = trace.choice_ms
            stage_status["choice_ms"] = (
                str(trace.telemetry.get("choice_status"))
                if trace.telemetry.get("choice_status") is not None
                else "executed"
                if choice_called
                else "deterministic_or_precomputed"
            )
        trace.routing_ms = max(0.0, routing_ms)
        stage_values["routing"] = trace.routing_ms
        stage_status["routing_ms"] = "executed"

        planner_value = trace.telemetry.get("planner_ms")
        trace.planner_ms = (
            max(0.0, float(planner_value)) if isinstance(planner_value, (int, float)) else 0.0
        )
        stage_status["planner_ms"] = str(trace.telemetry.get("planner_status", "not_used"))

        postprocess_value = trace.telemetry.get("postprocess_ms")
        trace.postprocess_ms = (
            max(0.0, float(postprocess_value))
            if isinstance(postprocess_value, (int, float))
            else 0.0
        )
        stage_status["postprocess_ms"] = (
            "executed" if postprocess_value is not None else "not_used"
        )
        for stage_name in ("retriever", "detector", "semantic_vlm", "route_vlm"):
            field_name = stage_name + "_ms"
            setattr(trace, field_name, stage_values.get(stage_name, 0.0))
            stage_status.setdefault(field_name, "not_used")
        if trace.choice_ms is None:
            trace.choice_ms = stage_values.get("choice", 0.0)
            stage_status.setdefault("choice_ms", "not_used")
        trace.e2e_ms = (time.perf_counter() - started) * 1000.0
        stage_status["e2e_ms"] = "executed"
        for field_name in SYSTEM_TELEMETRY_STAGES:
            stage_status.setdefault(field_name, "not_used")
        trace.stage_status = stage_status

        activated_providers: list[str] = []
        activated_roles: list[str] = []
        activated_counts: dict[str, int | str | None] = {}
        stage_providers = {
            "detector": ("detector", self.providers.detection),
            "retriever": ("retriever", self.providers.retriever),
            "semantic_vlm": ("semantic_2b", self.providers.semantic_2b),
            "route_vlm": ("route_4b", self.providers.route_4b),
        }
        used_stages = {
            stage
            for stage, value in stage_values.items()
            if stage in stage_providers and value >= 0.0
        }
        for stage, (role, provider) in stage_providers.items():
            if stage not in used_stages or stage_status.get(stage + "_ms") == "not_used":
                continue
            for name in _provider_names(provider):
                self._add_unique(activated_providers, name)
            activated_counts[role] = _provider_parameter_count(provider)
            self._add_unique(activated_roles, role)
        if trace.telemetry.get("choice_model_called") is True:
            choice_provider = self.providers.choice
            for name in _provider_names(choice_provider):
                self._add_unique(activated_providers, name)
            choice_role = getattr(choice_provider, "role", None) or "choice_2b"
            self._add_unique(activated_roles, choice_role)
            if str(choice_role) not in activated_counts:
                activated_counts[str(choice_role)] = _provider_parameter_count(choice_provider)
        for node in trace.nodes:
            self._add_unique(activated_providers, node.provider)
            self._collect_named_values(
                node.trace_metadata,
                {"provider", "provider_name", "activated_provider"},
                activated_providers,
            )
            self._collect_named_values(
                node.trace_metadata,
                {"model_role", "role"},
                activated_roles,
            )
        self._add_unique(activated_providers, trace.choice_provider)
        trace.activated_providers = activated_providers
        trace.activated_model_roles = activated_roles
        trace.activated_parameter_counts = activated_counts
        trace.resource_metrics = _capture_runtime_resources(torch)
        trace.telemetry.update(
            {
                "runtime_telemetry_version": "taskgraph_runtime_v1",
                "execution_mode": mode.value,
                "timing_source": "wall_clock_with_provider_metadata_when_available",
                "stage_values_ms": {
                    name: getattr(trace, name) for name in SYSTEM_TELEMETRY_STAGES
                },
                "measurement_window": trace.resource_metrics.get("measurement_window"),
            }
        )

    def run(self, request: RuntimeRequest) -> RuntimeResult:
        if not request.image_paths:
            raise ValueError("runtime request requires at least one image")
        started = time.perf_counter()
        torch = _reset_runtime_resources(self.providers)
        images = self._images(request)
        routing_started = time.perf_counter()
        mode = self.mode_router.route(request.dataset, request.task_category)
        routing_ms = (time.perf_counter() - routing_started) * 1000.0
        if mode is ExecutionMode.DIRECT_VLM:
            result = self._direct_vlm(request, images)
        elif mode is ExecutionMode.DIRECT_DETECTION:
            result = self._direct_detection(request, images)
        else:
            result = self._taskgraph(request, images)
        self._finalize_trace(
            result.trace,
            mode=mode,
            routing_ms=routing_ms,
            started=started,
            torch=torch,
        )
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
    semantic_choice_scores: dict[str, dict[str, float]] | None = None,
    route_choice_scores: dict[str, dict[str, float]] | None = None,
    retrieval_candidates: list[tuple[list[float], float]] | None = None,
    planner_fixtures: dict[str, TaskGraph | dict[str, Any]] | None = None,
    policy: DatasetExecutionPolicy | None = None,
    choice_config: ChoiceSystemConfig | None = None,
    semantic_decision_config: SemanticDecisionConfig | None = None,
    final_choice_fusion_config: FinalChoiceFusionConfig | None = None,
    answerability_config: AnswerabilityConfig | None = None,
    semantic_categories: set[str] | None = None,
) -> TaskGraphRuntime:
    shared_2b = FakeSemanticVLMProvider(
        {**(semantic_responses or {}), **(choice_responses or {})},
        choice_scores=semantic_choice_scores,
    )
    return TaskGraphRuntime(
        RuntimeProviders(
            detection=FakeDetectionProvider(detection_boxes),
            semantic_2b=shared_2b,
            route_4b=FakeSemanticVLMProvider(route_responses, choice_scores=route_choice_scores),
            retriever=FakeRegionRetriever(retrieval_candidates),
            choice=shared_2b,
            planner=FixturePlannerProvider(planner_fixtures or {}) if planner_fixtures else None,
        ),
        policy=policy,
        semantic_categories=semantic_categories,
        choice_config=choice_config,
        semantic_decision_config=semantic_decision_config,
        final_choice_fusion_config=final_choice_fusion_config,
        answerability_config=answerability_config,
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
                section.get("responses"),
                str(section.get("default", "A")),
                choice_scores=section.get("choice_scores"),
            )
        if kind == "qwen3_vl":
            return LazyQwenSemanticProvider(ModelConfig.model_validate(section), role=role)
        raise ValueError(f"unsupported semantic provider kind: {kind}")

    semantic_2b = semantic_provider("semantic_2b", "semantic_2b")
    route_4b = semantic_provider("route_4b", "route_4b")
    choice_section = providers.get("choice", {"reuse": "semantic_2b"})
    if not isinstance(choice_section, dict):
        raise TypeError("providers.choice config must be a mapping")
    reuse = choice_section.get("reuse")
    if reuse is not None:
        reusable = {"semantic_2b": semantic_2b, "route_4b": route_4b}
        try:
            choice = reusable[str(reuse)]
        except KeyError as exc:
            raise ValueError("providers.choice.reuse must name semantic_2b or route_4b") from exc
    else:
        choice = semantic_provider("choice", "choice_2b")

    retriever_cfg = dict(providers.get("region_retriever", {"kind": "fake"}))
    retriever_kind = str(retriever_cfg.pop("kind", "fake"))
    if retriever_kind == "fake":
        candidates = [(item["bbox"], item["score"]) for item in retriever_cfg.get("candidates", [])]
        retriever: RegionRetrieverProvider = FakeRegionRetriever(candidates)
    elif retriever_kind == "uhr_locator":
        from sat_rs_vlm.integrations.locators.registry import create_locator

        locator_config_path = retriever_cfg.pop(
            "config_path", retriever_cfg.pop("config_file", None)
        )
        inline_locator_config = retriever_cfg.pop("config", {})
        if not isinstance(inline_locator_config, Mapping):
            raise TypeError("providers.region_retriever.config must be a mapping")

        def merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
            merged = dict(base)
            for key, value in override.items():
                if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                    merged[key] = merge_mapping(merged[key], value)
                else:
                    merged[key] = value
            return merged

        locator_config: dict[str, Any]
        if locator_config_path is not None:
            locator_path = Path(str(locator_config_path)).expanduser()
            if not locator_path.is_absolute():
                candidate = Path.cwd() / locator_path
                repository_candidate = PROJECT_ROOT / locator_path
                locator_path = candidate if candidate.is_file() else repository_candidate
            from sat_rs_vlm.integrations.locators.config import load_locator_config

            locator_config = load_locator_config(locator_path)
        else:
            locator_config = {}
        locator_config = merge_mapping(locator_config, retriever_cfg)
        locator_config = merge_mapping(locator_config, inline_locator_config)
        if not isinstance(locator_config.get("retriever", {}), Mapping):
            raise TypeError("uhr_locator retriever config must be a mapping")
        retriever_section = dict(locator_config.get("retriever", {}))
        if retriever_section.get("provider") and not locator_config.get("provider_configs"):
            provider_name = str(retriever_section["provider"])
            provider_config = retriever_section.get("config", {})
            if not isinstance(provider_config, Mapping):
                raise TypeError("uhr_locator retriever.config must be a mapping")
            locator_config["provider_configs"] = {
                "retriever": {provider_name: dict(provider_config)}
            }
        retriever = LocatorRegionRetrieverAdapter(
            create_locator("hierarchical", locator_config)
        )
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
        if kind == "fixture":
            fixture_path = Path(str(planner_cfg["fixture_file"]))
            if not fixture_path.is_absolute():
                candidate = Path.cwd() / fixture_path
                repository_candidate = PROJECT_ROOT / fixture_path
                fixture_path = candidate if candidate.is_file() else repository_candidate
            payload = json.loads(fixture_path.read_text(encoding="utf-8"))
            planner = FixturePlannerProvider(payload)
        elif kind == "qwen3vl_lora":
            planner = Qwen3VLPlannerProvider(planner_cfg, role="planner_4b")
        else:
            raise ValueError(f"unsupported planner provider kind: {kind}")

    policy = DatasetExecutionPolicy.from_mapping(config.get("dataset_policy"))
    choice_config = ChoiceSystemConfig.from_mapping(config.get("choice"))
    semantic_decision_config = SemanticDecisionConfig.from_mapping(config.get("semantic_decision"))
    final_choice_fusion_config = FinalChoiceFusionConfig.from_mapping(
        config.get("final_vlm_choice_fusion")
    )
    answerability_config = AnswerabilityConfig.from_mapping(config.get("answerability"))
    capability_cfg = config.get("capability_routing", {})
    if not isinstance(capability_cfg, Mapping):
        raise TypeError("capability_routing config must be a mapping")
    ontology_value = capability_cfg.get(
        "ontology_path", "configs/eval/semantic/remote_sensing_ontology.json"
    )
    ontology_path = Path(str(ontology_value)).expanduser()
    if not ontology_path.is_absolute():
        candidate = Path.cwd() / ontology_path
        repository_candidate = PROJECT_ROOT / ontology_path
        ontology_path = candidate if candidate.is_file() else repository_candidate
    legacy_categories = config.get("semantic_region_categories", [])
    if not isinstance(legacy_categories, (list, tuple, set)):
        raise TypeError("semantic_region_categories must be a sequence")
    detector_overrides = capability_cfg.get("detector_overrides", [])
    retriever_overrides = capability_cfg.get("retriever_overrides", [])
    if not isinstance(detector_overrides, (list, tuple, set)) or not isinstance(
        retriever_overrides, (list, tuple, set)
    ):
        raise TypeError("capability routing overrides must be sequences")
    capability_classifier = TargetCapabilityClassifier.from_ontology_path(
        ontology_path,
        unresolved_policy=str(capability_cfg.get("unresolved_policy", "detector_fallback")),
        detector_overrides=detector_overrides,
        retriever_overrides=retriever_overrides,
        legacy_region_overrides=legacy_categories,
    )
    composer_cfg = config.get("input_composer", {})
    if not isinstance(composer_cfg, dict):
        raise TypeError("input_composer config must be a mapping")
    composer = InputComposer(
        candidate_halo_ratio=float(composer_cfg.get("candidate_halo_ratio", 0.2)),
        entity_set_union_area_threshold=float(
            composer_cfg.get("entity_set_union_area_threshold", 0.55)
        ),
        entity_set_max_side=int(composer_cfg.get("entity_set_max_side", 1536)),
        entity_set_max_crops=int(composer_cfg.get("entity_set_max_crops", 16)),
        route_max_side=int(composer_cfg.get("route_max_side", 1536)),
    )
    return TaskGraphRuntime(
        RuntimeProviders(detection, semantic_2b, route_4b, retriever, choice, planner),
        policy=policy,
        composer=composer,
        semantic_categories=set(config.get("semantic_region_categories", [])),
        capability_classifier=capability_classifier,
        choice_config=choice_config,
        semantic_decision_config=semantic_decision_config,
        final_choice_fusion_config=final_choice_fusion_config,
        answerability_config=answerability_config,
    )
