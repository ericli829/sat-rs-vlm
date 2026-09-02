"""Composition root for DIRECT and TASKGRAPH_UHR execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sat_rs_vlm.infrastructure.config import ModelConfig

from .answerability import AnswerabilityConfig, EvidenceSufficiencyExecutor
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
    CountingProvider,
    CountingRequest,
    DetectionProvider,
    DetectionRequest,
    EvidenceSufficiencyRequest,
    EvidenceSufficiencyResult,
    FakeCountingProvider,
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
    unwrap_select_result,
)
from .schema import AnswerType, OperatorName, QuestionType, TargetSpec, TaskGraph, parse_taskgraph
from .semantic_decision import SemanticDecisionConfig
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
    counting: CountingProvider | None = None

    def __post_init__(self) -> None:
        if self.counting is None:
            self.counting = FakeCountingProvider()

    def close(self) -> None:
        seen: set[int] = set()
        for provider in (
            self.detection,
            self.counting,
            self.semantic_2b,
            self.route_4b,
            self.retriever,
            self.choice,
        ):
            if provider is None:
                continue
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
        )
        counting = providers.counting or FakeCountingProvider()
        providers.counting = counting
        count = CountExecutor(counting)
        select = SelectExecutor(providers.semantic_2b, self.choice_config)
        semantic = SemanticExecutor(
            providers.semantic_2b,
            choice_config=self.choice_config,
            semantic_config=self.semantic_decision_config,
        )
        route = SemanticExecutor(
            providers.route_4b,
            provider_name="route_vlm",
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
        if request.graph is not None:
            graph = (
                request.graph
                if isinstance(request.graph, TaskGraph)
                else parse_taskgraph(request.graph)
            )
        elif self.providers.planner is not None:
            graph = self.providers.planner.plan(
                PlannerRequest(
                    request.question,
                    request.question_type.value,
                    request.options,
                    {f"${key}": value for key, value in images.items()},
                    request.sample_id,
                )
            )
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
        sources = tuple(
            unwrap_select_result(
                store.get(ref),
                allow_empty=False,
                consumer=f"final.sources[{ref}]",
            )
            for ref in graph.final.sources
        )
        output = self._choice_or_answer(
            sources,
            graph.final.question or None,
            request.options,
            graph.final.answer_type,
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
        return RuntimeResult(ExecutionMode.TASKGRAPH_UHR, output, trace, store)

    def _direct_vlm(self, request: RuntimeRequest, images: dict[str, ImageRef]) -> RuntimeResult:
        sources = tuple(images.values())
        trace = ExecutionTrace(request.sample_id, ExecutionMode.DIRECT_VLM.value)
        if request.options:
            choice_output = self.choice_resolver.resolve(
                ChoiceRequest(
                    sources,
                    request.question,
                    request.options,
                    self._direct_choice_answer_type(request),
                )
            )
            output: RuntimeObject | ChoiceResult = choice_output
            trace.choice_provider = str(choice_output.provenance.get("provider", "unknown"))
            trace.choice_result = runtime_summary(choice_output)
        else:
            model_input = self.composer.compose(list(sources), question=request.question)
            result = self.providers.semantic_2b.infer(VLMRequest(model_input, "direct_vlm"))
            output = Answer(result.text, result.confidence, {"provider": result.provider})
            trace.result = runtime_summary(output)
        return RuntimeResult(ExecutionMode.DIRECT_VLM, output, trace)

    def _direct_detection(
        self, request: RuntimeRequest, images: dict[str, ImageRef]
    ) -> RuntimeResult:
        if len(images) != 1:
            raise ValueError("DIRECT_DETECTION requires exactly one image")
        target = TargetSpec(category=request.target_category or "object")
        scope = next(iter(images.values()))
        is_count = request.task_category.casefold() in {"count", "counting"}
        if is_count:
            counting = self.providers.counting
            if counting is None:
                raise RuntimeError("COUNT requires RuntimeProviders.counting")
            counted = counting.count(CountingRequest(scope, target, entire=True))
            source: RuntimeObject = ScalarInt(
                counted.count,
                {"provider": counted.provider, "detection": counted.detections.provenance},
            )
        else:
            detected = self.providers.detection.detect(
                DetectionRequest(scope, target, request.task_category)
            )
            source = detected.detections
        trace = ExecutionTrace(request.sample_id, ExecutionMode.DIRECT_DETECTION.value)
        if request.options:
            choice_output = self.choice_resolver.resolve(
                ChoiceRequest(
                    (source,),
                    request.question,
                    request.options,
                    self._direct_choice_answer_type(request),
                )
            )
            output: RuntimeObject | ChoiceResult = choice_output
            trace.choice_provider = str(choice_output.provenance.get("provider", "unknown"))
            trace.choice_result = runtime_summary(choice_output)
        else:
            output = source
            trace.result = runtime_summary(source)
        return RuntimeResult(ExecutionMode.DIRECT_DETECTION, output, trace)

    def run(self, request: RuntimeRequest) -> RuntimeResult:
        if not request.image_paths:
            raise ValueError("runtime request requires at least one image")
        images = self._images(request)
        mode = self.mode_router.route(request.dataset, request.task_category)
        if mode is ExecutionMode.DIRECT_VLM:
            return self._direct_vlm(request, images)
        if mode is ExecutionMode.DIRECT_DETECTION:
            return self._direct_detection(request, images)
        return self._taskgraph(request, images)

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
            counting=FakeCountingProvider(detection_boxes),
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
    if detection_kind == "counting_system":
        raise ValueError(
            "providers.detection.kind='counting_system' is no longer supported. "
            "Configure providers.counting.kind='counting_system' instead; "
            "providers.detection remains the LOCATE DetectionProvider."
        )
    if detection_kind == "fake":
        detection: DetectionProvider = FakeDetectionProvider(detection_cfg.get("boxes"))
    else:
        from sat_rs_vlm.integrations.detectors.registry import create_proposal_provider

        detection = ProposalDetectionAdapter(
            create_proposal_provider(detection_kind, detection_cfg)
        )

    counting_cfg = dict(providers.get("counting", {"kind": "fake"}))
    counting_kind = str(counting_cfg.pop("kind", "fake"))
    if counting_kind == "fake":
        counting: CountingProvider = FakeCountingProvider(counting_cfg.get("boxes"))
    elif counting_kind == "counting_system":
        from sat_rs_vlm.integrations.counting import CountingSystemProvider

        counting = CountingSystemProvider.from_config(counting_cfg)
    else:
        raise ValueError(
            f"unsupported counting provider kind: {counting_kind}; "
            "use fake or counting_system"
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

    semantic_2b = semantic_provider("semantic_2b", "general_2b")
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
    choice_config = ChoiceSystemConfig.from_mapping(config.get("choice"))
    semantic_decision_config = SemanticDecisionConfig.from_mapping(config.get("semantic_decision"))
    final_choice_fusion_config = FinalChoiceFusionConfig.from_mapping(
        config.get("final_vlm_choice_fusion")
    )
    answerability_config = AnswerabilityConfig.from_mapping(config.get("answerability"))
    composer_cfg = config.get("input_composer", {})
    if not isinstance(composer_cfg, dict):
        raise TypeError("input_composer config must be a mapping")
    composer = InputComposer(
        candidate_halo_ratio=float(composer_cfg.get("candidate_halo_ratio", 0.2)),
        entity_set_union_area_threshold=float(
            composer_cfg.get("entity_set_union_area_threshold", 0.55)
        ),
        entity_set_max_side=int(composer_cfg.get("entity_set_max_side", 1536)),
        route_max_side=int(composer_cfg.get("route_max_side", 1536)),
    )
    return TaskGraphRuntime(
        RuntimeProviders(
            detection, semantic_2b, route_4b, retriever, choice, planner, counting
        ),
        policy=policy,
        composer=composer,
        semantic_categories=set(config.get("semantic_region_categories", [])),
        choice_config=choice_config,
        semantic_decision_config=semantic_decision_config,
        final_choice_fusion_config=final_choice_fusion_config,
        answerability_config=answerability_config,
    )
