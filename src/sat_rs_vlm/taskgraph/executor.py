"""Topological graph execution and capability routing."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import cast

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

    _SELECT_INPUT_POLICIES: dict[OperatorName, dict[str, tuple[bool, bool]]] = {
        OperatorName.SELECT: {
            "candidates": (True, False),
            "reference": (False, False),
            "scope": (False, True),
        },
        OperatorName.GROUP: {"entities": (True, False)},
        OperatorName.COUNT: {
            # COUNT.image accepts a SUBREGION SelectResult (always one Region).
            "image": (False, True),
            "entities": (True, False),
        },
        OperatorName.ATTRIBUTE: {"entity": (False, True)},
        OperatorName.CLASSIFY: {"source": (False, True)},
        OperatorName.MULTILABEL_CLASSIFY: {"source": (False, True)},
        OperatorName.MOTION: {
            "source": (False, True),
            "before": (False, True),
            "after": (False, True),
        },
        OperatorName.RELATION: {
            "subject": (False, True),
            "reference": (False, True),
        },
        OperatorName.VLM_REASON: {
            "image": (False, True),
            "evidence": (False, False),
        },
        # SUBREGION SELECT results (always exactly one Region) are valid visual
        # scopes; materialize them before the runtime contract check.
        OperatorName.REGION: {"image": (False, True)},
        OperatorName.REGION_FROM_BBOX: {"image": (False, True)},
        OperatorName.FIND_MARKER: {"image": (False, True)},
        OperatorName.LOCATE: {"image": (False, True)},
        OperatorName.BUILD_ROUTE_CONTEXT: {"image": (False, True)},
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
            allow_empty, require_single = policy

            def unwrap(
                item: RuntimeObject,
                *,
                _allow_empty: bool = allow_empty,
                _require_single: bool = require_single,
                _role: str = role,
            ) -> RuntimeObject:
                return unwrap_select_result(
                    item,
                    allow_empty=_allow_empty,
                    require_single=_require_single,
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
                    fallback=fallback,
                    trace_metadata=dict(outcome.trace_metadata),
                )
            )
        return trace
