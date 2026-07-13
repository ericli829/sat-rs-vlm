"""sat-rs-vlm 外部微调插件 API v1。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sat_rs_vlm.plugins.context import PluginContext

EXTERNAL_PLUGIN_API_VERSION = "1"


class ExternalFineTuningPlugin(ABC):
    """普通本地文件夹插件必须实现的最小接口。"""

    api_version: str = EXTERNAL_PLUGIN_API_VERSION
    name: str
    version: str

    def model_load_kwargs(
        self,
        context: PluginContext,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """返回策略特有模型加载参数，例如 QLoRA 的 4-bit 配置。"""

        return {}

    @abstractmethod
    def validate(
        self,
        context: PluginContext,
        config: Mapping[str, Any],
    ) -> None:
        """验证策略配置和当前环境。"""

    @abstractmethod
    def prepare_model(
        self,
        context: PluginContext,
        model: Any,
        processor: Any,
        config: Mapping[str, Any],
    ) -> Any:
        """注入 Adapter 或选择性冻结/解冻模型。"""

    @abstractmethod
    def build_training_arguments(
        self,
        context: PluginContext,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """返回公共 Trainer 接受的训练参数字典。"""

    @abstractmethod
    def save_artifacts(
        self,
        context: PluginContext,
        model: Any,
        processor: Any,
        output_dir: Path,
    ) -> None:
        """保存 Adapter 或完整模型工件。"""

    def evaluate(
        self,
        context: PluginContext,
        checkpoint_dir: Path,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """可选策略评估；返回 None 时使用公共 Trainer 指标。"""

        return None

    def optimizer_parameter_groups(
        self,
        context: PluginContext,
        model: Any,
        config: Mapping[str, Any],
    ) -> list[dict[str, Any]] | None:
        """可选自定义优化器参数组。"""

        return None

    def trainer_callbacks(
        self,
        context: PluginContext,
        config: Mapping[str, Any],
    ) -> list[Any]:
        """可选 Trainer callback，例如 AdaLoRA 动态秩更新。"""

        return []

    def report_details(
        self,
        context: PluginContext,
        model: Any,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """可选策略特有报告字段。"""

        return {}
