"""根据显式配置和设备能力选择训练精度。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrecisionDecision:
    """最终精度开关及其判定原因。"""

    bf16: bool
    fp16: bool
    mode: str
    reason: str


def select_precision(
    *,
    device: str,
    bf16: bool | None,
    fp16: bool | None,
    cuda_available: bool,
    bf16_supported: bool,
) -> PrecisionDecision:
    """选择 BF16、FP16 或 FP32。

    显式布尔值优先。CPU 上即使显式请求半精度也会失败，防止 Trainer
    进入不受支持的混合精度路径。
    """

    if bf16 is True and fp16 is True:
        raise ValueError("bf16 and fp16 cannot both be enabled.")
    requested_cuda = device == "cuda" or (device == "auto" and cuda_available)
    if not requested_cuda:
        if bf16 is True or fp16 is True:
            raise ValueError("FP16/BF16 training requires CUDA; use mock mode on CPU.")
        return PrecisionDecision(False, False, "fp32", "CPU execution disables mixed precision.")
    if bf16 is True:
        if not bf16_supported:
            raise ValueError(
                "BF16 was explicitly requested but the CUDA device does not support it."
            )
        return PrecisionDecision(True, False, "bf16", "BF16 explicitly enabled.")
    if fp16 is True:
        return PrecisionDecision(False, True, "fp16", "FP16 explicitly enabled.")
    if bf16 is False and fp16 is False:
        return PrecisionDecision(False, False, "fp32", "Mixed precision explicitly disabled.")
    if bf16_supported:
        return PrecisionDecision(True, False, "bf16", "Auto-selected BF16 on supported CUDA.")
    return PrecisionDecision(False, True, "fp16", "Auto-selected FP16 because BF16 is unavailable.")
