"""Topological graph execution and capability routing."""

from __future__ import annotations

import time
from dataclasses import dataclass

from sat_rs_vlm.infrastructure.telemetry import SystemTelemetry

from .contracts import validate_runtime_inputs
from .operators import OperatorContext, OperatorExecutor, OperatorOutcome
from .runtime_types import RuntimeObject, runtime_summary, runtime_type_name
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

    def __init__(self, bindings: dict[OperatorName, ExecutorBinding]) -> None:
        self.bindings = dict(bindings)

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
            return binding.primary.execute(node, inputs, context), None
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
    def __init__(self, router: CapabilityRouter) -> None:
        self.router = router

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
        for node in graph.nodes:
            resolved = {name: store.resolve(refs) for name, refs in node.inputs.items()}
            started = time.perf_counter()
            fallback = None
            try:
                outcome, fallback = self.router.execute(node, resolved, context)
                store.put(node.id, outcome.value)
            except TaskGraphExecutionError as exc:
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
                        telemetry={},
                        error=dict(exc.details),
                    )
                )
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
                    output_runtime_type=runtime_type_name(outcome.value),
                    output_summary=runtime_summary(outcome.value),
                    telemetry=dict(getattr(outcome.value, "provenance", {})),
                    fallback=fallback,
                )
            )
        return trace
