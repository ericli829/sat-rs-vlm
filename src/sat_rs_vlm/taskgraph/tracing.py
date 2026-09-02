"""Compact JSON-serializable execution tracing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

NOT_AVAILABLE_FROM_BACKEND = "NOT_AVAILABLE_FROM_BACKEND"
SYSTEM_TELEMETRY_STAGES = (
    "routing_ms",
    "planner_ms",
    "retriever_ms",
    "detector_ms",
    "semantic_vlm_ms",
    "route_vlm_ms",
    "choice_ms",
    "postprocess_ms",
    "e2e_ms",
)


@dataclass
class NodeTrace:
    node_id: str
    operator: str
    input_refs: dict[str, str | list[str]]
    resolved_input_types: dict[str, str | list[str]]
    provider: str
    latency_ms: float
    execution_mode: str | None = None
    semantic_method: str | None = None
    cache_reused: bool | None = None
    final_choice_fusion: bool | None = None
    fusion_reason: str | None = None
    output_runtime_type: str | None = None
    output_summary: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    fallback: str | None = None
    trace_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    sample_id: str
    execution_mode: str
    taskgraph: dict[str, Any] | None = None
    nodes: list[NodeTrace] = field(default_factory=list)
    final_sources: list[str] = field(default_factory=list)
    final_question: str | None = None
    choice_provider: str | None = None
    choice_result: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    routing_ms: float | None = None
    planner_ms: float | None = None
    retriever_ms: float | None = None
    detector_ms: float | None = None
    semantic_vlm_ms: float | None = None
    route_vlm_ms: float | None = None
    choice_ms: float | None = None
    postprocess_ms: float | None = None
    e2e_ms: float | None = None
    stage_status: dict[str, str] = field(default_factory=dict)
    activated_providers: list[str] = field(default_factory=list)
    activated_model_roles: list[str] = field(default_factory=list)
    activated_parameter_counts: dict[str, int | str | None] = field(default_factory=dict)
    resource_metrics: dict[str, Any] = field(default_factory=dict)
    ttft_ms: float | str | None = NOT_AVAILABLE_FROM_BACKEND
    visual_tokens: int | str | None = NOT_AVAILABLE_FROM_BACKEND
    output_tokens: int | str | None = NOT_AVAILABLE_FROM_BACKEND
    decode_tokens_per_s: float | str | None = NOT_AVAILABLE_FROM_BACKEND
    telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output
