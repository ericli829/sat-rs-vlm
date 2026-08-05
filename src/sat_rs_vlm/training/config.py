"""Qwen3-VL 本地/远程训练配置模型与路径解析工具。

本模块只负责配置解析，不加载 torch/transformers 等重依赖。路径可以来自 YAML、
环境变量或 CLI 覆盖；训练脚本拿到解析后的 Path 后再决定是否加载真实模型。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class TrainModelConfig(BaseModel):
    """基座模型与 processor 配置。

    `model_dir/processor_dir` 用于本地模型目录，`model_id/processor_id` 保留远程或
    Hugging Face 缓存模型 ID。优先使用本地目录。
    """

    model_id: str | None = None
    processor_id: str | None = None
    model_dir: str | None = None
    processor_dir: str | None = None
    trust_remote_code: bool = True
    local_files_only: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device_map: str = "auto"

    @model_validator(mode="after")
    def fill_processor_source(self) -> TrainModelConfig:
        """如果 processor 未显式配置，则复用模型来源。"""

        if self.processor_dir is None and self.model_dir is not None:
            self.processor_dir = self.model_dir
        if self.processor_id is None and self.model_id is not None:
            self.processor_id = self.model_id
        return self


class TrainDataConfig(BaseModel):
    """训练数据配置。"""

    train_file: str
    val_file: str
    image_root: str = "."
    max_seq_length: int = 4096
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    skip_bad_samples: bool = False
    data_composition: Literal["full", "balanced_quota", "detection_quota"] = "full"
    sampling_mode: Literal["uniform", "weighted", "alternating_source"] = "uniform"
    task_sampling_weights: dict[str, float] = Field(default_factory=dict)
    source_batch_pattern: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sampling(self) -> TrainDataConfig:
        """加权采样必须显式启用并提供正权重。"""

        if self.sampling_mode == "weighted" and not self.task_sampling_weights:
            raise ValueError("sampling_mode='weighted' requires task_sampling_weights")
        if self.sampling_mode == "alternating_source" and not self.source_batch_pattern:
            raise ValueError(
                "sampling_mode='alternating_source' requires source_batch_pattern"
            )
        if any(value <= 0 for value in self.task_sampling_weights.values()):
            raise ValueError("task_sampling_weights must contain only positive values")
        return self


class TrainConfig(BaseModel):
    """Trainer 训练超参数。"""

    output_dir: str
    method: str = "qlora"
    freeze_vision_encoder: bool = True
    freeze_projector: bool = False
    num_train_epochs: float = 3
    max_steps: int | None = None
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = Field(default=0, ge=0)
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False
    max_grad_norm: float = 1.0
    seed: int = 42
    resume_from_checkpoint: str | None = None


class LoRAConfig(BaseModel):
    """LoRA adapter 配置。"""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=list)
    initial_adapter_dir: str | None = None


class QLoRAConfig(BaseModel):
    """QLoRA 4bit 量化配置。"""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


class EvalConfig(BaseModel):
    """训练期间评测配置。"""

    do_eval: bool = True
    predict_with_generate: bool = True
    max_new_tokens: int = 256
    num_beams: int = 1


class LoggingConfig(BaseModel):
    """实验日志配置。"""

    report_to: str = "none"
    experiment_name: str = "qwen3vl-rs-lora-baseline"


class Qwen3VLTrainingConfig(BaseModel):
    """完整 Qwen3-VL 微调配置。"""

    model: TrainModelConfig
    data: TrainDataConfig
    training: TrainConfig
    lora: LoRAConfig
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


@dataclass(frozen=True)
class TrainingPathOverrides:
    """CLI 路径和样本数覆盖项。"""

    model_dir: str | None = None
    processor_dir: str | None = None
    train_file: str | None = None
    val_file: str | None = None
    image_root: str | None = None
    output_dir: str | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_steps: int | None = None
    local_files_only: bool | None = None
    method: str | None = None
    max_seq_length: int | None = None
    initial_adapter_dir: str | None = None
    learning_rate: float | None = None
    num_train_epochs: float | None = None


@dataclass(frozen=True)
class ResolvedTrainingPaths:
    """训练脚本使用的已解析路径。"""

    model_source: str
    processor_source: str
    model_dir: Path | None
    processor_dir: Path | None
    train_file: Path
    val_file: Path
    image_root: Path
    output_dir: Path
    initial_adapter_dir: Path | None


def expand_env_vars(value: Any, *, allow_unresolved: bool = False) -> Any:
    """递归展开 `${VAR}` 环境变量。

    缺少环境变量时抛出 ValueError，并明确指出变量名。
    """

    if isinstance(value, str):
        missing: list[str] = []

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                missing.append(name)
                return match.group(0)
            return os.environ[name]

        expanded = ENV_PATTERN.sub(replace, value)
        if missing and not allow_unresolved:
            names = ", ".join(sorted(set(missing)))
            raise ValueError(f"Missing environment variable(s): {names}")
        return expanded
    if isinstance(value, dict):
        return {
            key: expand_env_vars(item, allow_unresolved=allow_unresolved)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [expand_env_vars(item, allow_unresolved=allow_unresolved) for item in value]
    return value


def resolve_path(value: str | Path, base_dir: str | Path | None = None) -> Path:
    """解析相对/绝对路径并返回 Path。

    相对路径默认相对当前工作目录，也可通过 base_dir 指定。
    """

    value_text = str(value)
    if ENV_PATTERN.search(value_text):
        raise ValueError(f"Unresolved environment variable in path: {value_text}")
    path = Path(value_text).expanduser()
    if path.is_absolute():
        return path
    return (Path(base_dir) if base_dir is not None else Path.cwd()) / path


def load_training_config(
    path: str | Path,
    *,
    allow_unresolved_env: bool = False,
) -> Qwen3VLTrainingConfig:
    """加载训练 YAML 配置，支持 `${ENV}` 变量展开。"""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Training config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        data: dict[str, Any] = yaml.safe_load(file) or {}
    expanded = expand_env_vars(data, allow_unresolved=allow_unresolved_env)
    return Qwen3VLTrainingConfig.model_validate(expanded)


def apply_training_overrides(
    config: Qwen3VLTrainingConfig,
    overrides: TrainingPathOverrides,
) -> Qwen3VLTrainingConfig:
    """应用 CLI 覆盖项并返回新的配置对象。"""

    model_updates: dict[str, Any] = {}
    data_updates: dict[str, Any] = {}
    train_updates: dict[str, Any] = {}
    lora_updates: dict[str, Any] = {}
    if overrides.model_dir is not None:
        model_updates["model_dir"] = overrides.model_dir
        if overrides.processor_dir is None:
            model_updates["processor_dir"] = overrides.model_dir
    if overrides.processor_dir is not None:
        model_updates["processor_dir"] = overrides.processor_dir
    if overrides.local_files_only is not None:
        model_updates["local_files_only"] = overrides.local_files_only
    if overrides.train_file is not None:
        data_updates["train_file"] = overrides.train_file
    if overrides.val_file is not None:
        data_updates["val_file"] = overrides.val_file
    if overrides.image_root is not None:
        data_updates["image_root"] = overrides.image_root
    if overrides.max_train_samples is not None:
        data_updates["max_train_samples"] = overrides.max_train_samples
    if overrides.max_eval_samples is not None:
        data_updates["max_eval_samples"] = overrides.max_eval_samples
    if overrides.max_seq_length is not None:
        data_updates["max_seq_length"] = overrides.max_seq_length
    if overrides.output_dir is not None:
        train_updates["output_dir"] = overrides.output_dir
    if overrides.max_steps is not None:
        train_updates["max_steps"] = overrides.max_steps
    if overrides.method is not None:
        train_updates["method"] = overrides.method
    if overrides.learning_rate is not None:
        train_updates["learning_rate"] = overrides.learning_rate
    if overrides.num_train_epochs is not None:
        train_updates["num_train_epochs"] = overrides.num_train_epochs
    if overrides.initial_adapter_dir is not None:
        lora_updates["initial_adapter_dir"] = overrides.initial_adapter_dir

    return config.model_copy(
        update={
            "model": config.model.model_copy(update=model_updates),
            "data": config.data.model_copy(update=data_updates),
            "training": config.training.model_copy(update=train_updates),
            "lora": config.lora.model_copy(update=lora_updates),
        }
    )


def resolve_training_paths(
    config: Qwen3VLTrainingConfig,
    base_dir: str | Path | None = None,
) -> ResolvedTrainingPaths:
    """将配置中的路径统一解析为 Path。"""

    model_dir = resolve_path(config.model.model_dir, base_dir) if config.model.model_dir else None
    processor_dir = (
        resolve_path(config.model.processor_dir, base_dir) if config.model.processor_dir else None
    )
    model_source = str(model_dir) if model_dir is not None else str(config.model.model_id or "")
    processor_source = (
        str(processor_dir) if processor_dir is not None else str(config.model.processor_id or "")
    )
    initial_adapter_dir = (
        resolve_path(config.lora.initial_adapter_dir, base_dir)
        if config.lora.initial_adapter_dir
        else None
    )
    if not model_source:
        raise ValueError("Model source is empty. Set model_dir or model_id.")
    if not processor_source:
        raise ValueError("Processor source is empty. Set processor_dir or processor_id.")
    return ResolvedTrainingPaths(
        model_source=model_source,
        processor_source=processor_source,
        model_dir=model_dir,
        processor_dir=processor_dir,
        train_file=resolve_path(config.data.train_file, base_dir),
        val_file=resolve_path(config.data.val_file, base_dir),
        image_root=resolve_path(config.data.image_root, base_dir),
        output_dir=resolve_path(config.training.output_dir, base_dir),
        initial_adapter_dir=initial_adapter_dir,
    )
