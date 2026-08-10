"""统一量化、benchmark 和敏感度分析公共 API。

顶层包是新代码的唯一实现位置；``sat_rs_vlm.compression.quantization`` 仅用于兼容旧版
导入路径。当前稳定后端包括 CPU dynamic INT8、CUDA bitsandbytes INT8 和未量化基线，
注册表为后续 INT4、GPTQ、AWQ、QAT 与 mixed precision 保留扩展点。
"""

from sat_rs_vlm.quantization.config import (
    QuantizationExperimentConfig,
    load_quantization_config,
)
from sat_rs_vlm.quantization.quantizer import (
    create_backend,
    list_backends,
    quantize_dynamic_linear,
    register_backend,
)

__all__ = [
    "QuantizationExperimentConfig",
    "create_backend",
    "list_backends",
    "load_quantization_config",
    "quantize_dynamic_linear",
    "register_backend",
]
