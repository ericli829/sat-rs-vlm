"""旧 bit flip API 的兼容包装。

算法已统一到 :mod:`sat_rs_vlm.models.reliability.bitflip`。保留本模块是为了不破坏
第一阶段代码和外部调用方；本文件不再维护第二份实现。
"""

from sat_rs_vlm.models.reliability.bitflip import flip_bit_at, flip_random_bit

__all__ = ["flip_bit_at", "flip_random_bit"]
