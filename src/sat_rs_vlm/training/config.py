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
            raise ValueError("sampling_mode='alternating_source' requires source_batch_pattern")
        if any(value <= 0 for value in self.task_sampling_weights.values()):
            raise ValueError("task_sampling_weights must contain only positive values")
        return self


class TrainConfig(BaseModel):
    """Trainer 训练超参数。"""

    output_dir: str
    method: str = "qlora"
    freeze_vision_encoder: bool = True
    freeze_projector: bool = False
    num_train_epochs: float | None = 3
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

    @model_validator(mode="after")
    def validate_training_length(self) -> TrainConfig:
        """Require either an epoch budget or an explicit positive step budget."""

        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("training.max_steps must be positive when configured")
        if self.num_train_epochs is None and self.max_steps is None:
            raise ValueError("Set training.max_steps when training.num_train_epochs is null")
        if self.num_train_epochs is not None and self.num_train_epochs <= 0:
            raise ValueError("training.num_train_epochs must be positive when configured")
        return self


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


DEFAULT_MULTITASK_LOSS_WEIGHTS: dict[str, float] = {
    "captioning": 1.0,
    "detection": 1.0,
    "counting": 1.0,
    "scene_classification": 1.0,
    "vqa": 1.0,
    "change_detection": 1.0,
}


class MultitaskLossConfig(BaseModel):
    """独立的多任务 loss 配置；不得与数据采样权重混用。"""

    mode: Literal["token_mean", "task_weighted"] = "task_weighted"
    task_weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_MULTITASK_LOSS_WEIGHTS)
    )
    unknown_task_weight: float = Field(default=1.0, gt=0.0)
    strict_task_metadata: bool = True

    @model_validator(mode="after")
    def validate_task_weights(self) -> MultitaskLossConfig:
        """拒绝非正权重，并归一化任务键，避免配置中的大小写产生隐藏分支。"""

        normalized = {
            str(task).strip().lower(): float(weight) for task, weight in self.task_weights.items()
        }
        if any(not task for task in normalized):
            raise ValueError("loss.task_weights keys must not be empty")
        invalid = {task: weight for task, weight in normalized.items() if weight <= 0.0}
        if invalid:
            raise ValueError(f"loss.task_weights must contain only positive values: {invalid}")
        self.task_weights = normalized
        return self


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


class BBoxAreaThresholdConfig(BaseModel):
    """Normalized bbox-area boundaries shared by statistics, mining, and evaluation."""

    small_max: float = Field(default=0.01, gt=0.0, lt=1.0)
    medium_max: float = Field(default=0.10, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_order(self) -> BBoxAreaThresholdConfig:
        if self.small_max >= self.medium_max:
            raise ValueError("bbox_area_thresholds.small_max must be less than medium_max")
        return self


class TrainingStatisticsConfig(BaseModel):
    """Controls the lightweight pre-training dataset and supervision audit."""

    enabled: bool = False
    output_dir: str = "reports/training_statistics"
    inspect_images: bool = True
    max_samples: int | None = Field(default=None, ge=1)
    bbox_area_thresholds: BBoxAreaThresholdConfig = Field(default_factory=BBoxAreaThresholdConfig)


class VisionTuningConfig(BaseModel):
    """Explicit partial Qwen3-VL visual-tower adaptation settings."""

    enabled: bool = False
    unfreeze_last_n_blocks: int = Field(default=2, ge=1)
    train_main_merger: bool = True
    train_deepstack_mergers: bool = False
    train_patch_embed: bool = False


class OptimizationGroupConfig(BaseModel):
    """Learning rates for the non-overlapping H1 trainable parameter groups."""

    lora_lr: float = Field(default=1.0e-5, gt=0.0)
    visual_merger_lr: float = Field(default=5.0e-6, gt=0.0)
    vision_lr: float = Field(default=1.0e-6, gt=0.0)

    @model_validator(mode="after")
    def validate_learning_rate_order(self) -> OptimizationGroupConfig:
        if not self.vision_lr < self.visual_merger_lr < self.lora_lr:
            raise ValueError(
                "H1 learning rates must satisfy vision_lr < visual_merger_lr < lora_lr"
            )
        return self


class TrainableAuditConfig(BaseModel):
    """Controls validation of the model's final trainable parameter surface."""

    fail_on_unexpected_trainable: bool = True
    report_dir: str = "reports/training"


class HardScoreWeightsConfig(BaseModel):
    """Configurable weights used to preserve hard-example severity."""

    detection_iou: float = Field(default=0.45, ge=0.0)
    detection_label_error: float = Field(default=0.20, ge=0.0)
    detection_parse_failure: float = Field(default=0.25, ge=0.0)
    detection_center_distance: float = Field(default=0.10, ge=0.0)
    detection_small_object_bonus: float = Field(default=0.05, ge=0.0)
    counting_absolute_error: float = Field(default=0.60, ge=0.0)
    counting_parse_failure: float = Field(default=0.40, ge=0.0)
    text_error: float = Field(default=1.0, ge=0.0)
    caption_rouge_l: float = Field(default=0.45, ge=0.0)
    caption_chrf: float = Field(default=0.30, ge=0.0)
    caption_cider: float = Field(default=0.25, ge=0.0)


class HardAdaptationConfig(BaseModel):
    """Hard mining, replay composition, and leakage-protection configuration."""

    enabled: bool = False
    source_checkpoint: str | None = None
    prediction_source: str | None = None
    source_train_file: str | None = None
    evaluation_ids_file: str | None = None
    output_dir: str = "data/processed/hard_examples"
    evaluation_contract_version: str = "1.5"
    hard_ratio: float = Field(default=0.70, gt=0.0, lt=1.0)
    replay_ratio: float = Field(default=0.30, gt=0.0, lt=1.0)
    hard_score_threshold: float = Field(default=0.35, ge=0.0)
    max_hard_samples: int | None = Field(default=None, ge=1)
    fixed_evaluation_sample_count: int = Field(default=593, ge=1)
    require_evaluation_exclusions: bool = True
    enforce_replay_coverage: bool = True
    required_replay_sources: list[str] = Field(default_factory=lambda: ["VRSBench", "LEVIR-CC"])
    required_replay_tasks: list[str] = Field(
        default_factory=lambda: [
            "detection",
            "counting",
            "vqa",
            "captioning",
            "scene_classification",
            "change_detection",
        ]
    )
    score_weights: HardScoreWeightsConfig = Field(default_factory=HardScoreWeightsConfig)
    bbox_area_thresholds: BBoxAreaThresholdConfig = Field(default_factory=BBoxAreaThresholdConfig)

    @model_validator(mode="after")
    def validate_mix(self) -> HardAdaptationConfig:
        if abs(self.hard_ratio + self.replay_ratio - 1.0) > 1.0e-9:
            raise ValueError("hard_ratio and replay_ratio must sum to 1.0")
        return self


class Qwen3VLTrainingConfig(BaseModel):
    """完整 Qwen3-VL 微调配置。"""

    model: TrainModelConfig
    data: TrainDataConfig
    training: TrainConfig
    lora: LoRAConfig
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)
    loss: MultitaskLossConfig = Field(default_factory=MultitaskLossConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    statistics: TrainingStatisticsConfig = Field(default_factory=TrainingStatisticsConfig)
    vision_tuning: VisionTuningConfig = Field(default_factory=VisionTuningConfig)
    optimization: OptimizationGroupConfig = Field(default_factory=OptimizationGroupConfig)
    trainable_audit: TrainableAuditConfig = Field(default_factory=TrainableAuditConfig)
    hard_adaptation: HardAdaptationConfig = Field(default_factory=HardAdaptationConfig)


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
