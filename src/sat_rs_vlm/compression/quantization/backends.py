"""已弃用的后端导入兼容层；实现位于 ``sat_rs_vlm.quantization.quantizer``。"""

from sat_rs_vlm.quantization.quantizer import (
    BaselineBackend,
    BitsAndBytesInt8Backend,
    QuantizationBackend,
    TorchDynamicInt8Backend,
    UnsupportedQuantizationError,
    create_backend,
    list_backends,
    quantize_dynamic_linear,
    register_backend,
)

__all__ = [
    "BaselineBackend",
    "BitsAndBytesInt8Backend",
    "QuantizationBackend",
    "TorchDynamicInt8Backend",
    "UnsupportedQuantizationError",
    "create_backend",
    "list_backends",
    "quantize_dynamic_linear",
    "register_backend",
]
