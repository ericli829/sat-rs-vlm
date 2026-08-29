"""Compact JSON-serializable execution tracing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NodeTrace:
    node_id: str
    operator: str
    input_refs: dict[str, str | list[str]]
    resolved_input_types: dict[str, str | list[str]]
    provider: str
    latency_ms: float
    output_runtime_type: str | None = None
    output_summary: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    fallback: str | None = None


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output
