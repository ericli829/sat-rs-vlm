"""集中配置加载模块。

算法/流程：
    1. 从显式 path、环境变量 SAT_RS_VLM_CONFIG 或默认 configs/default.yaml 定位配置文件。
    2. 使用 PyYAML 读取 YAML。
    3. 使用 Pydantic 模型补默认值、做类型校验，并兼容第一阶段旧字段。

接口：
    load_config(path=None) -> AppSettings。
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from sat_rs_vlm.domain.exceptions import ConfigurationError


class AppConfig(BaseModel):
    """应用级配置。

    参数：
        name：应用名称。
        environment：运行环境名称，例如 local/dev/prod。
        log_level：日志级别字符串。
    """

    name: str = "sat-rs-vlm"
    environment: str = "local"
    log_level: str = "INFO"


class ModelConfig(BaseModel):
    """模型后端配置。

    参数：
        backend：模型后端，支持 mock 或 huggingface。
        model_id：HuggingFace 模型 ID；mock 后端可为空。
        device：运行设备，auto/cpu/cuda 等。
        dtype：模型权重 dtype，auto/float16/bfloat16/float32 等。
        max_new_tokens：生成式推理最大新 token 数。
        trust_remote_code：是否允许 transformers 加载远程自定义代码。
        local_files_only：是否只从本地缓存加载模型。

    兼容性：
        accept_phase_one_keys 会把旧字段 engine/model_name 映射到 backend/model_id。
    """

    backend: str = "mock"
    model_id: str = ""
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 256
    trust_remote_code: bool = True
    local_files_only: bool = False
    # Applies only to TaskGraph SELECT.  Other VLM tasks remain open-ended.
    selection_constrained_decoding: bool = True

    @model_validator(mode="before")
    @classmethod
    def accept_phase_one_keys(cls, data: Any) -> Any:
        """兼容第一阶段配置字段。

        参数：
            data：Pydantic 传入的原始 model 配置对象。

        返回值：
            Any：映射后的配置字典或原始值。
        """

        if isinstance(data, dict):
            updated = dict(data)
            if "backend" not in updated and "engine" in updated:
                updated["backend"] = updated["engine"]
            if "model_id" not in updated and "model_name" in updated:
                updated["model_id"] = updated["model_name"]
            return updated
        return data


class RuntimeConfig(BaseModel):
    """运行时配置。

    参数：
        seed：随机种子。
        log_level：运行时日志级别。
        enable_profiler：是否把推理耗时和设备信息写入 raw_output.profile。
    """

    seed: int = 42
    log_level: str = "INFO"
    enable_profiler: bool = True

    @model_validator(mode="before")
    @classmethod
    def accept_phase_one_keys(cls, data: Any) -> Any:
        """丢弃第一阶段已废弃但可能仍存在的 runtime 字段。

        参数：
            data：原始 runtime 配置。

        返回值：
            Any：移除 device/batch_size 后的配置字典或原始值。
        """

        if isinstance(data, dict):
            updated = dict(data)
            updated.pop("device", None)
            updated.pop("batch_size", None)
            return updated
        return data


class ReliabilityConfig(BaseModel):
    """可靠性相关配置。

    参数：
        enable_bitflip_simulation：是否启用 bit flip 注入。
        bitflip_probability：bit flip 概率，占位给后续批量模拟使用。
        enable_checksum：是否启用文件 checksum 校验。
        enable_fault_recovery：是否启用故障恢复流程。
    """

    enable_bitflip_simulation: bool = False
    bitflip_probability: float = 0.0
    enable_checksum: bool = False
    enable_fault_recovery: bool = False


class AppSettings(BaseModel):
    """完整应用配置聚合对象。

    作用：
        作为 CLI、HTTP 和 InferenceService.from_config 的统一配置入口。
    """

    app: AppConfig = Field(default_factory=AppConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)


def _default_config_path() -> Path:
    """返回默认配置文件路径。

    返回值：
        Path：项目根目录下 configs/default.yaml。
    """

    return Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> AppSettings:
    """加载 YAML 配置并转为 Pydantic 对象。

    参数：
        path：可选配置路径；为空时使用 SAT_RS_VLM_CONFIG 或默认配置文件。

    返回值：
        AppSettings：包含 app/model/runtime/reliability 的强类型配置。

    异常：
        ConfigurationError：配置文件不存在时抛出，并提示用户如何指定配置。
    """

    configured_path = path or os.getenv("SAT_RS_VLM_CONFIG") or _default_config_path()
    config_path = Path(configured_path)
    if not config_path.exists():
        raise ConfigurationError(
            f"Config file does not exist: {config_path}. "
            "Pass --config configs/default.yaml or set SAT_RS_VLM_CONFIG."
        )
    with config_path.open("r", encoding="utf-8") as file:
        data: dict[str, Any] = yaml.safe_load(file) or {}
    return AppSettings.model_validate(data)
