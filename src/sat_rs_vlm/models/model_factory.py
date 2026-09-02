"""模型工厂。

作用：
    将配置中的 backend 映射为具体模型引擎，实现应用层与模型实现层解耦。

算法：
    mock 后端直接创建 MockVLMEngine；huggingface 后端延迟导入 HuggingFaceVLMEngine，
    避免基础启动依赖 torch/transformers。
"""

from sat_rs_vlm.infrastructure.config import ModelConfig
from sat_rs_vlm.models.base import BaseVLMEngine
from sat_rs_vlm.models.mock_model import MockVLMEngine


def create_vlm_engine(model_config: ModelConfig) -> BaseVLMEngine:
    """创建 VLM 推理引擎。

    参数：
        model_config：模型配置，包含 backend、model_id、device、dtype 等。

    返回值：
        BaseVLMEngine：具体模型实例，但类型上只暴露统一协议。

    异常：
        ValueError：backend 不是 mock 或 huggingface 时抛出。
        ImportError：huggingface 后端缺少可选模型依赖时由 HF 引擎抛出。
    """

    backend = model_config.backend.lower()
    if backend == "mock":
        return MockVLMEngine()
    if backend == "huggingface":
        from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine

        return HuggingFaceVLMEngine(
            model_id=model_config.model_id,
            adapter_path=model_config.adapter_path,
            device=model_config.device,
            dtype=model_config.dtype,
            max_new_tokens=model_config.max_new_tokens,
            trust_remote_code=model_config.trust_remote_code,
            local_files_only=model_config.local_files_only,
        )
    raise ValueError(
        f"Unsupported model backend: {model_config.backend}. "
        "Use backend='mock' or backend='huggingface'."
    )
