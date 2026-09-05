"""Physical execution analysis for final-only VLM choice fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import AnswerType, OperatorName, TaskGraph

DEFAULT_FUSION_OPERATORS = frozenset(
    {
        OperatorName.VLM_REASON,
        OperatorName.MATCH_CHOICE,
        OperatorName.ROUTE_REASON,
        OperatorName.ATTRIBUTE,
        OperatorName.CLASSIFY,
        OperatorName.MULTILABEL_CLASSIFY,
        OperatorName.MOTION,
        OperatorName.RELATION,
    }
)


@dataclass(frozen=True)
class FinalChoiceFusionConfig:
    mode: str = "auto"
    operators: frozenset[OperatorName] = DEFAULT_FUSION_OPERATORS

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "off"}:
            raise ValueError("final_vlm_choice_fusion.mode must be auto or off")

    @classmethod
    def from_mapping(cls, value: Any) -> FinalChoiceFusionConfig:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("final_vlm_choice_fusion config must be a mapping")
        raw_operators = value.get("operators")
        operators = (
            DEFAULT_FUSION_OPERATORS
            if raw_operators is None
            else frozenset(OperatorName(str(item)) for item in raw_operators)
        )
        return cls(mode=str(value.get("mode", "auto")), operators=operators)


@dataclass(frozen=True)
class NodeExecutionHint:
    node_id: str
    final_choice_fusion: bool
    answer_type: AnswerType | None
    options: tuple[str, ...]
    question: str
    fusion_reason: str


@dataclass(frozen=True)
class ExecutionPlan:
    hints: dict[str, NodeExecutionHint]

    @classmethod
    def build(
        cls,
        graph: TaskGraph,
        *,
        options: tuple[str, ...],
        config: FinalChoiceFusionConfig | None = None,
    ) -> ExecutionPlan:
        config = config or FinalChoiceFusionConfig()
        final_refs = tuple(graph.final.sources)
        consumer_counts = {node.id: 0 for node in graph.nodes}
        for consumer in graph.nodes:
            for raw in consumer.inputs.values():
                refs = raw if isinstance(raw, list) else [raw]
                for ref in refs:
                    if ref.startswith("$n") and ref[1:] in consumer_counts:
                        consumer_counts[ref[1:]] += 1

        hints: dict[str, NodeExecutionHint] = {}
        for node in graph.nodes:
            ref = f"${node.id}"
            enabled = False
            if node.op not in config.operators:
                reason = "operator_not_eligible"
            elif config.mode == "off":
                reason = "disabled_by_config"
            elif graph.final.answer_type not in {
                AnswerType.CHOICE_SINGLE,
                AnswerType.CHOICE_MULTI,
            }:
                reason = "free_form_final"
            elif len(final_refs) != 1:
                reason = "multiple_final_sources"
            elif ref != final_refs[0]:
                reason = "not_final_source"
            elif consumer_counts[node.id] != 0:
                reason = "has_downstream_consumer"
            elif not options:
                reason = "no_original_options"
            else:
                enabled = True
                reason = "eligible"
            hints[node.id] = NodeExecutionHint(
                node_id=node.id,
                final_choice_fusion=enabled,
                answer_type=graph.final.answer_type if enabled else None,
                options=options if enabled else (),
                question=graph.final.question,
                fusion_reason=reason,
            )
        return cls(hints)

    def hint_for(self, node_id: str) -> NodeExecutionHint:
        return self.hints[node_id]
