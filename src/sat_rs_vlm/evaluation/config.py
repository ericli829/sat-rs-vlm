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
