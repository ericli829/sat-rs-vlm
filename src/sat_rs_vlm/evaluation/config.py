"""统一评估、配对比较和绘图脚本使用的 YAML 配置模型。

配置类只描述离线评估工作流，不负责加载真实模型。模型生成阶段仍读取原有
``configs/eval/qwen3vl_*.yaml``，生成的 predictions JSONL 再交给本模块描述的统一
Evaluation v1.5 流程。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from sat_rs_vlm.configuration.environment import expand_environment
from sat_rs_vlm.training.config import BBoxAreaThresholdConfig


class EvaluationInputConfig(BaseModel):
    """评估及比较阶段的输入路径；所有字段均可由 CLI 覆盖。"""

    predictions: str | None = None
    baseline_dir: str | None = None
    candidate_dir: str | None = None


class EvaluationProtocolConfig(BaseModel):
    """Evaluation v1.5 协议、语义评估和延迟语义配置。"""

    contract: str = "configs/eval/evaluation_contract_v1.5.yaml"
    manifest: str | None = None
    strict: bool = True
    semantic: bool = True
    semantic_contract: str = "configs/eval/semantic/semantic_contract.json"
    semantic_ontology: str = "configs/eval/semantic/remote_sensing_ontology.json"
    latency_semantics: Literal["unresolved", "single_sample", "batch_amortized_per_sample"] = (
        "unresolved"
    )
    eval_batch_size: int | None = Field(default=None, gt=0)
    group_by_task: bool | None = None
    dataset: str | None = None
    tasks: list[str] = Field(default_factory=list)
    sample_num: int | None = Field(default=None, gt=0)
    tier: Literal["E1", "E2", "E3"] | None = None
    tiers_manifest: str | None = None


class EvaluationComparisonConfig(BaseModel):
    """同一批样本的 baseline/candidate 配对比较参数。"""

    bootstrap_resamples: int = Field(default=1000, gt=0)
    seed: int = 20260806


def _default_plot_formats() -> list[Literal["png", "svg"]]:
    return ["png", "svg"]


class EvaluationPlotConfig(BaseModel):
    """绘图输入以 ``标签=结果目录`` 形式保存，便于跨平台 YAML 表达。"""

    evaluations: list[str] = Field(default_factory=list)
    comparisons: list[str] = Field(default_factory=list)
    formats: list[Literal["png", "svg"]] = Field(default_factory=_default_plot_formats)
    overwrite: bool = False


class EvaluationOutputConfig(BaseModel):
    """统一实验目录；图表默认写入其 ``figures`` 子目录。"""

    output_dir: str = "reports/evaluation/default"
    figures_dir: str | None = None


class EvaluationWorkflowConfig(BaseModel):
    """三个 Evaluation v1.5 脚本共用的完整配置。"""

    input: EvaluationInputConfig = Field(default_factory=EvaluationInputConfig)
    evaluation: EvaluationProtocolConfig = Field(default_factory=EvaluationProtocolConfig)
    comparison: EvaluationComparisonConfig = Field(default_factory=EvaluationComparisonConfig)
    plotting: EvaluationPlotConfig = Field(default_factory=EvaluationPlotConfig)
    output: EvaluationOutputConfig = Field(default_factory=EvaluationOutputConfig)


class EvaluationTierSourceConfig(BaseModel):
    """A validation population source used to build frozen evaluation tiers.

    ``image_prefix`` makes image paths portable beneath a shared ``DATA_ROOT``.
    ``manifest_path`` is a stable logical path recorded instead of a machine-local
    absolute path expanded from an environment variable.
    """

    name: str
    eval_file: str
    train_file: str | None = None
    image_prefix: str = ""
    manifest_path: str | None = None


class EvaluationTierSizeConfig(BaseModel):
    """Requested tier size; E3 uses ``mode=full`` and ignores target_samples."""

    target_samples: int | None = Field(default=None, gt=0)
    mode: Literal["sampled", "full"] = "sampled"


class ExistingE1Config(BaseModel):
    """Optional immutable ID list recovered from a historical fixed evaluation set."""

    ids_file: str | None = None
    samples_file: str | None = None
    required: bool = False
    origin: str = "generated"


class EvaluationTierOutputConfig(BaseModel):
    """Project-relative output paths for the frozen JSONL assets and manifest."""

    directory: str = "data/evaluation/tiers"
    e1_file: str = "e1_quick.jsonl"
    e2_file: str = "e2_standard.jsonl"
    e3_file: str = "e3_full.jsonl"
    manifest_file: str = "evaluation_tiers_manifest.json"


class EvaluationTierBuildConfig(BaseModel):
    """Typed configuration for deterministic, nested evaluation-tier construction."""

    schema_version: str = "1.0"
    seed: int = 42
    sources: list[EvaluationTierSourceConfig]
    tiers: dict[Literal["E1", "E2", "E3"], EvaluationTierSizeConfig]
    existing_e1: ExistingE1Config = Field(default_factory=ExistingE1Config)
    bbox_area_thresholds: BBoxAreaThresholdConfig = Field(
        default_factory=BBoxAreaThresholdConfig
    )
    output: EvaluationTierOutputConfig = Field(default_factory=EvaluationTierOutputConfig)

    def validate_semantics(self) -> None:
        """Validate tier ordering and source uniqueness after Pydantic type checks."""

        missing = {"E1", "E2", "E3"} - set(self.tiers)
        if missing:
            raise ValueError(f"Missing evaluation tier configuration: {sorted(missing)}")
        e1 = self.tiers["E1"]
        e2 = self.tiers["E2"]
        e3 = self.tiers["E3"]
        if e1.mode != "sampled" or e2.mode != "sampled" or e3.mode != "full":
            raise ValueError("E1/E2 must be sampled and E3 must use mode='full'")
        if e1.target_samples is None or e2.target_samples is None:
            raise ValueError("E1 and E2 require target_samples")
        if e1.target_samples > e2.target_samples:
            raise ValueError("E1 target_samples must not exceed E2 target_samples")
        source_names = [source.name.strip().lower() for source in self.sources]
        if len(source_names) != len(set(source_names)):
            raise ValueError("Evaluation tier source names must be unique")


def load_evaluation_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> EvaluationWorkflowConfig:
    """读取 YAML、展开 ``${ENV_VAR}``，并执行 Pydantic 类型和范围校验。"""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation config must be a mapping: {config_path}")
    expanded = expand_environment(
        payload,
        environ=dict(os.environ if environ is None else environ),
        allow_unresolved=False,
    )
    return EvaluationWorkflowConfig.model_validate(expanded)


def load_evaluation_tier_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> EvaluationTierBuildConfig:
    """Load the tier-builder YAML through the project's environment resolver."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation tier config must be a mapping: {config_path}")
    expanded = expand_environment(
        payload,
        environ=dict(os.environ if environ is None else environ),
        allow_unresolved=False,
    )
    config = EvaluationTierBuildConfig.model_validate(expanded)
    config.validate_semantics()
    return config
