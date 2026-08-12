"""统一量化实验配置及环境变量解析。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from sat_rs_vlm.configuration.environment import expand_environment


class QuantModelConfig(BaseModel):
    base_model: str
    processor_id: str | None = None
    adapter_path: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str | None = "sdpa"


class QuantBackendConfig(BaseModel):
    backend: Literal["baseline", "torch_dynamic_int8", "bnb_int8"]
    device: Literal["cpu", "cuda"]
    save_artifact: bool = False

    @model_validator(mode="after")
    def validate_backend_device(self) -> QuantBackendConfig:
        if self.backend == "torch_dynamic_int8" and self.device != "cpu":
            raise ValueError("torch_dynamic_int8 requires device='cpu'")
        if self.backend == "bnb_int8" and self.device != "cuda":
            raise ValueError("bnb_int8 requires device='cuda'")
        return self


class QuantDataConfig(BaseModel):
    eval_file: str
    image_root: str
    max_eval_samples: int = Field(default=20, gt=0)
    max_seq_length: int = Field(default=1024, gt=0)


class QuantGenerationConfig(BaseModel):
    do_sample: bool = False
    num_beams: int = Field(default=1, gt=0)
    max_new_tokens: int = Field(default=128, gt=0)
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    task_max_new_tokens: dict[str, int] = Field(default_factory=dict)
    change_binary_enabled: bool = True
    change_binary_max_new_tokens: int = Field(default=8, gt=0)


class QuantBenchmarkConfig(BaseModel):
    warmup_samples: int = Field(default=2, ge=0)
    repeats: int = Field(default=2, gt=0)
    seed: int = 42
    latency_scope: Literal["single_sample_end_to_end"] = "single_sample_end_to_end"


class QuantOutputConfig(BaseModel):
    output_dir: str


class QuantizationExperimentConfig(BaseModel):
    """完整量化 benchmark 配置。"""

    model: QuantModelConfig
    quantization: QuantBackendConfig
    data: QuantDataConfig
    generation: QuantGenerationConfig = Field(default_factory=QuantGenerationConfig)
    benchmark: QuantBenchmarkConfig = Field(default_factory=QuantBenchmarkConfig)
    output: QuantOutputConfig

    @model_validator(mode="after")
    def fill_processor(self) -> QuantizationExperimentConfig:
        if self.model.processor_id is None:
            self.model.processor_id = self.model.base_model
        return self


def load_quantization_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> QuantizationExperimentConfig:
    """读取 YAML、展开现有环境变量并应用少量 CLI 覆盖。"""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Quantization config must be a mapping: {config_path}")
    expanded = dict(
        expand_environment(
            payload,
            environ=dict(os.environ if environ is None else environ),
            allow_unresolved=False,
        )
    )
    for dotted_key, value in dict(overrides or {}).items():
        if value is None:
            continue
        target: dict[str, Any] = expanded
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override non-mapping key: {dotted_key}")
            target = child
        target[parts[-1]] = value
    return QuantizationExperimentConfig.model_validate(expanded)
