"""Topological graph execution and capability routing."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import cast

from sat_rs_vlm.infrastructure.telemetry import SystemTelemetry

from .contracts import validate_runtime_inputs, validate_runtime_output
from .execution_plan import ExecutionPlan, FinalChoiceFusionConfig
from .operators import OperatorContext, OperatorExecutor, OperatorOutcome
from .runtime_types import (
    RuntimeObject,
    runtime_summary,
    runtime_type_name,
    unwrap_select_result,
)
from .schema import GraphNode, OperatorName, TaskGraph
from .store import RuntimeStore
from .tracing import ExecutionTrace, NodeTrace


class TaskGraphExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        node_id: str,
        operator: str,
        provider: str,
        input_refs: dict[str, str | list[str]],
        exception: Exception,
        retryable: bool = False,
    ) -> None:
        self.details = {
            "node_id": node_id,
            "operator": operator,
            "provider": provider,
            "input_refs": input_refs,
            "exception": f"{type(exception).__name__}: {exception}",
            "retryable": retryable,
        }
        super().__init__(str(self.details))


@dataclass(frozen=True)
class ExecutorBinding:
    primary: OperatorExecutor
    fallback: OperatorExecutor | None = None


class CapabilityRouter:
    """Explicit operator-to-capability registry; graphs never contain model names."""

    _SELECT_INPUT_POLICIES: dict[OperatorName, dict[str, tuple[bool, bool, bool]]] = {
        OperatorName.SELECT: {
            "candidates": (True, False, False),
            # A rank-tied reference is valid geometry evidence: the executor's
            # _single_reference picks the highest-scoring region.
            "reference": (False, False, True),
            "scope": (False, True, False),
        },
        OperatorName.GROUP: {"entities": (True, False, False)},
        OperatorName.COUNT: {
            # COUNT.image accepts a SUBREGION SelectResult (always one Region).
            "image": (False, True, False),
            "entities": (True, False, False),
        },
        # Semantic consumers tolerate an empty selection: the empty EntitySet
        # reaches the operator, which falls back to question-grounded VLM
        # answering instead of hard-failing.  require_single keeps multi-select
        # rejection intact.
        OperatorName.ATTRIBUTE: {"entity": (True, True, False)},
        OperatorName.CLASSIFY: {"source": (True, True, False)},
        OperatorName.MULTILABEL_CLASSIFY: {"source": (True, True, False)},
        OperatorName.MOTION: {
            "source": (False, True, False),
            "before": (False, True, False),
            "after": (False, True, False),
        },
        OperatorName.RELATION: {
            # Tolerate AMBIGUOUS/EMPTY selections: the semantic RELATION step
            # pins each side to its top-scoring entity instead of hard-failing
            # on a multi-candidate or rank-tied SELECT result.
            "subject": (True, False, True),
            "reference": (True, False, True),
        },
        OperatorName.VLM_REASON: {
            "image": (False, True, False),
            "evidence": (False, False, False),
        },
        # SUBREGION SELECT results (always exactly one Region) are valid visual
        # scopes; materialize them before the runtime contract check.
        OperatorName.REGION: {"image": (False, True, False)},
        OperatorName.REGION_FROM_BBOX: {"image": (False, True, False)},
        OperatorName.FIND_MARKER: {"image": (False, True, False)},
        OperatorName.LOCATE: {"image": (False, True, False)},
        OperatorName.BUILD_ROUTE_CONTEXT: {
            "image": (False, True, False),
            # START/GOAL are endpoint evidence: materialize AMBIGUOUS selections
            # to their selected set; GeometryExecutor pins the highest score.
            "start": (False, False, True),
            "goal": (False, False, True),
        },
    }

    def __init__(self, bindings: dict[OperatorName, ExecutorBinding]) -> None:
        self.bindings = dict(bindings)

    @classmethod
    def _materialize_select_inputs(
        cls,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
    ) -> dict[str, RuntimeObject | list[RuntimeObject]]:
        policies = cls._SELECT_INPUT_POLICIES.get(node.op, {})
        materialized: dict[str, RuntimeObject | list[RuntimeObject]] = {}
        for role, value in inputs.items():
            policy = policies.get(role)
            if policy is None:
                materialized[role] = value
                continue
            allow_empty, require_single, allow_ambiguous = policy

            def unwrap(
                item: RuntimeObject,
                *,
                _allow_empty: bool = allow_empty,
                _require_single: bool = require_single,
                _allow_ambiguous: bool = allow_ambiguous,
                _role: str = role,
            ) -> RuntimeObject:
                return unwrap_select_result(
                    item,
                    allow_empty=_allow_empty,
                    require_single=_require_single,
                    allow_ambiguous=_allow_ambiguous,
                    consumer=f"{node.op.value}.{_role}",
                )

            materialized[role] = (
                [unwrap(item) for item in value] if isinstance(value, list) else unwrap(value)
            )
        return materialized

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        context: OperatorContext,
    ) -> tuple[OperatorOutcome, str | None]:
        try:
            binding = self.bindings[node.op]
        except KeyError as exc:
            raise KeyError(f"no capability binding for operator {node.op.value}") from exc
        try:
            # Materialize select-aware inputs FIRST: a SUBREGION SelectResult
            # (always exactly one Region) is a valid visual scope for image
            # roles but is not itself an ImageRef/Region.  The runtime contract
            # validates what the operator actually receives.
            inputs = self._materialize_select_inputs(node, inputs)
            validate_runtime_inputs(node.op.value, inputs)
        except Exception as contract_error:
            raise TaskGraphExecutionError(
                node_id=node.id,
                operator=node.op.value,
                provider="input_contract",
                input_refs=node.inputs,
                exception=contract_error,
            ) from contract_error
        try:
            outcome = binding.primary.execute(node, inputs, context)
            validate_runtime_output(
                node.op.value,
                outcome.value,
                final_choice_fusion=bool(
                    context.execution_hint and context.execution_hint.final_choice_fusion
                ),
            )
            return outcome, None
        except Exception as primary_error:
            if binding.fallback is None:
                raise TaskGraphExecutionError(
                    node_id=node.id,
                    operator=node.op.value,
                    provider=binding.primary.provider_name,
                    input_refs=node.inputs,
                    exception=primary_error,
                ) from primary_error
            try:
                outcome = binding.fallback.execute(node, inputs, context)
                validate_runtime_output(
                    node.op.value,
                    outcome.value,
                    final_choice_fusion=bool(
                        context.execution_hint and context.execution_hint.final_choice_fusion
                    ),
                )
                return outcome, binding.primary.provider_name
            except Exception as fallback_error:
                raise TaskGraphExecutionError(
                    node_id=node.id,
                    operator=node.op.value,
                    provider=binding.fallback.provider_name,
                    input_refs=node.inputs,
                    exception=fallback_error,
                ) from fallback_error


class GraphExecutor:
    def __init__(
        self,
        router: CapabilityRouter,
        fusion_config: FinalChoiceFusionConfig | None = None,
    ) -> None:
        self.router = router
        self.fusion_config = fusion_config or FinalChoiceFusionConfig()

    @staticmethod
    def _input_types(value: RuntimeObject | list[RuntimeObject]) -> str | list[str]:
        if isinstance(value, list):
            return [runtime_type_name(item) for item in value]
        return runtime_type_name(value)

    def execute(
        self,
        graph: TaskGraph,
        store: RuntimeStore,
        *,
        sample_id: str,
        execution_mode: str,
        context: OperatorContext,
    ) -> ExecutionTrace:
        monitor = SystemTelemetry("taskgraph_executor", reset_cuda_peaks=False)
        try:
            with monitor:
                trace = self._execute_graph(
                    graph,
                    store,
                    sample_id=sample_id,
                    execution_mode=execution_mode,
                    context=context,
                )
        except TaskGraphExecutionError as exc:
            exc.details["executor_telemetry"] = monitor.to_dict()
            raise
        executor_telemetry = monitor.to_dict()
        executor_telemetry["activated_providers"] = sorted(
            {
                provider
                for node in trace.nodes
                for provider in (node.provider, node.fallback)
                if provider
            }
        )
        executor_telemetry["fallback_count"] = sum(
            node.fallback is not None for node in trace.nodes
        )
        executor_telemetry["node_count"] = len(trace.nodes)
        trace.telemetry["executor"] = executor_telemetry
        return trace

    def _execute_graph(
        self,
        graph: TaskGraph,
        store: RuntimeStore,
        *,
        sample_id: str,
        execution_mode: str,
        context: OperatorContext,
    ) -> ExecutionTrace:
        trace = ExecutionTrace(
            sample_id=sample_id,
            execution_mode=execution_mode,
            taskgraph=graph.model_dump(mode="json"),
            final_sources=list(graph.final.sources),
            final_question=graph.final.question,
        )
        plan = ExecutionPlan.build(
            graph,
            options=context.choices,
            config=self.fusion_config,
        )
        for node in graph.nodes:
            hint = plan.hint_for(node.id)
            node_context = replace(
                context,
                execution_hint=hint,
                final_sources=tuple(graph.final.sources),
                final_question=graph.final.question,
                graph_nodes=tuple(graph.nodes),
            )
            resolved = {name: store.resolve(refs) for name, refs in node.inputs.items()}
            started = time.perf_counter()
            fallback = None
            try:
                outcome, fallback = self.router.execute(node, resolved, node_context)
                store.put(node.id, outcome.value)
            except TaskGraphExecutionError as exc:
                producers = {
                    f"${producer.id}": producer.op.value for producer in graph.nodes
                }
                exc.details["input_producers"] = {
                    str(role): (
                        [producers.get(str(ref)) for ref in refs]
                        if isinstance(refs, list)
                        else producers.get(str(refs))
                    )
                    for role, refs in node.inputs.items()
                }
                trace.nodes.append(
                    NodeTrace(
                        node_id=node.id,
                        operator=node.op.value,
                        input_refs=dict(node.inputs),
                        resolved_input_types={
                            name: self._input_types(value) for name, value in resolved.items()
                        },
                        provider=str(exc.details["provider"]),
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        final_choice_fusion=hint.final_choice_fusion,
                        fusion_reason=hint.fusion_reason,
                        error=dict(exc.details),
                    )
                )
                exc.execution_trace = trace
                raise
            trace.nodes.append(
                NodeTrace(
                    node_id=node.id,
                    operator=node.op.value,
                    input_refs=dict(node.inputs),
                    resolved_input_types={
                        name: self._input_types(value) for name, value in resolved.items()
                    },
                    provider=outcome.provider,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    execution_mode=cast(str | None, outcome.trace_metadata.get("execution_mode")),
                    semantic_method=cast(str | None, outcome.trace_metadata.get("semantic_method")),
                    cache_reused=cast(bool | None, outcome.trace_metadata.get("cache_reused")),
                    final_choice_fusion=cast(
                        bool | None,
                        outcome.trace_metadata.get("final_choice_fusion", hint.final_choice_fusion),
                    ),
                    fusion_reason=cast(
                        str | None,
                        outcome.trace_metadata.get("fusion_reason", hint.fusion_reason),
                    ),
                    output_runtime_type=runtime_type_name(outcome.value),
                    output_summary=runtime_summary(outcome.value),
                    telemetry=dict(getattr(outcome.value, "provenance", {})),
                    fallback=fallback,
                    trace_metadata=dict(outcome.trace_metadata),
                )
            )
        return trace
