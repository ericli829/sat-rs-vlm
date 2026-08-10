"""统一量化、Evaluation v1.5 和敏感度实验配置及环境变量解析。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from sat_rs_vlm.configuration.environment import expand_environment


class QuantModelConfig(BaseModel):
    """模型来源；dynamic INT8 优先使用已经合并 LoRA 的 ``merged_model``。"""

    base_model: str
    merged_model: str | None = None
    processor_id: str | None = None
    adapter_path: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str | None = "sdpa"

    @property
    def model_source(self) -> str:
        """返回本次量化实际加载的权重目录。"""

        return self.merged_model or self.base_model


class QuantBackendConfig(BaseModel):
    """量化方法及设备。

    ``method=dynamic_int8`` 是新配置名称；``backend=torch_dynamic_int8`` 继续兼容旧配置。
    未实现的 GPTQ/AWQ/INT4/QAT 名称会在后端注册表阶段清晰失败，不会静默降级。
    """

    backend: str = "torch_dynamic_int8"
    method: str | None = None
    device: Literal["cpu", "cuda"] = "cpu"
    save_artifact: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_method(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        method = payload.get("method")
        backend = payload.get("backend")
        aliases = {
            "none": "baseline",
            "baseline": "baseline",
            "dynamic_int8": "torch_dynamic_int8",
            "pytorch_dynamic_int8": "torch_dynamic_int8",
            "torch_dynamic_int8": "torch_dynamic_int8",
            "bnb_int8": "bnb_int8",
        }
        if backend is None and method is not None:
            payload["backend"] = aliases.get(str(method), str(method))
        if method is None and payload.get("backend") is not None:
            reverse = {
                "baseline": "none",
                "torch_dynamic_int8": "dynamic_int8",
                "bnb_int8": "bnb_int8",
            }
            payload["method"] = reverse.get(str(payload["backend"]), str(payload["backend"]))
        return payload

    @model_validator(mode="after")
    def validate_backend_device(self) -> QuantBackendConfig:
        aliases = {
            "none": "baseline",
            "baseline": "baseline",
            "dynamic_int8": "torch_dynamic_int8",
            "pytorch_dynamic_int8": "torch_dynamic_int8",
            "torch_dynamic_int8": "torch_dynamic_int8",
            "bnb_int8": "bnb_int8",
        }
        if self.method is not None and aliases.get(self.method, self.method) != self.backend:
            raise ValueError(
                f"quantization method={self.method!r} conflicts with backend={self.backend!r}"
            )
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


class QuantBenchmarkConfig(BaseModel):
    warmup_samples: int = Field(default=2, ge=0)
    repeats: int = Field(default=2, gt=0)
    seed: int = 42
    latency_scope: Literal["single_sample_end_to_end"] = "single_sample_end_to_end"


class QuantEvaluationConfig(BaseModel):
    """量化前后共用的 Evaluation v1.5 契约，保证指标可直接比较。"""

    contract: str = "configs/eval/evaluation_contract_v1.5.yaml"
    manifest: str | None = None
    strict: bool = True
    semantic: bool = True
    semantic_contract: str = "configs/eval/semantic/semantic_contract.json"
    semantic_ontology: str = "configs/eval/semantic/remote_sensing_ontology.json"
    dataset: str | None = None
    tasks: list[str] = Field(default_factory=list)
    sample_num: int | None = Field(default=None, gt=0)
    bootstrap_resamples: int = Field(default=1000, gt=0)


class QuantSensitivityConfig(BaseModel):
    """逐组件或逐层组量化的扫描范围与安全上限。"""

    method: Literal["component_wise", "layer_wise"] = "component_wise"
    layer_group_size: int = Field(default=6, gt=0)
    skip_modules: list[str] = Field(default_factory=lambda: ["vision_encoder"])
    max_groups: int | None = Field(default=None, gt=0)


class QuantOutputConfig(BaseModel):
    output_dir: str = "reports/evaluation/quantization"
    figures_dir: str | None = None


class QuantizationExperimentConfig(BaseModel):
    """完整量化 benchmark 配置。"""

    model: QuantModelConfig
    quantization: QuantBackendConfig
    data: QuantDataConfig
    generation: QuantGenerationConfig = Field(default_factory=QuantGenerationConfig)
    benchmark: QuantBenchmarkConfig = Field(default_factory=QuantBenchmarkConfig)
    evaluation: QuantEvaluationConfig = Field(default_factory=QuantEvaluationConfig)
    sensitivity: QuantSensitivityConfig = Field(default_factory=QuantSensitivityConfig)
    output: QuantOutputConfig = Field(default_factory=QuantOutputConfig)

    @model_validator(mode="after")
    def fill_processor(self) -> QuantizationExperimentConfig:
        if self.model.processor_id is None:
            self.model.processor_id = self.model.base_model
        if self.evaluation.sample_num is not None:
            self.data.max_eval_samples = self.evaluation.sample_num
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
