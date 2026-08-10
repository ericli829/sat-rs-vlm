"""兼容旧导入路径；新代码应使用 :mod:`sat_rs_vlm.quantization`。"""

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
