"""单粒子翻转故障注入、检测、保护和恢复公共 API。"""

from sat_rs_vlm.models.reliability.bitflip import (
    flip_bit_at,
    flip_random_bit,
    flip_random_tensor_bits,
    flip_random_value_bits,
    flip_tensor_bit,
    flip_value_bit,
)
from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.models.reliability.schemas import BitFlipRecord, ValidationResult

__all__ = [
    "BitFlipRecord",
    "ValidationResult",
    "file_sha256",
    "flip_bit_at",
    "flip_random_bit",
    "flip_random_tensor_bits",
    "flip_random_value_bits",
    "flip_tensor_bit",
    "flip_value_bit",
]
