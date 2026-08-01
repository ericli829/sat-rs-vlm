"""普通、量化和可靠性实验共用的 prediction JSONL 基础 schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictionRecord(BaseModel):
    """单条预测记录；扩展字段不能改变六个基础字段的语义。"""

    id: str
    task_type: str
    prediction: str
    reference: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    inference_latency_ms: float | None = None
    variant: Literal["baseline", "quantized", "fault", "recovered"] | None = None
    backend: Literal["none", "bnb_int8", "torch_dynamic_int8"] | None = None
    compression: dict[str, Any] = Field(default_factory=dict)
    fault_case: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    recovery: dict[str, Any] = Field(default_factory=dict)
