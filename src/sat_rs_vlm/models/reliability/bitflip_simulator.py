"""bit flip 故障注入模拟器。

算法：
    对 bytes/bytearray 或非负整数执行异或操作。随机翻转时先选择字节和 bit 位；
    定点翻转时直接根据 bit_index 定位目标 bit。该模块用于模拟空间辐射导致的
    单粒子翻转，为后续容错和 checksum 流程提供测试入口。
"""

import random


def flip_random_bit(value: bytes | bytearray | int, seed: int | None = None) -> bytes | int:
    """随机翻转一个 bit。

    参数：
        value：bytes、bytearray 或非负整数。
        seed：可选随机种子；传入后结果可复现。

    返回值：
        bytes | int：翻转后的新值。bytes-like 输入统一返回 bytes。

    异常：
        ValueError：整数为负数时抛出。
    """

    rng = random.Random(seed)
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Only non-negative integers are supported.")
        bit_length = max(value.bit_length(), 1)
        return value ^ (1 << rng.randrange(bit_length))

    data = bytearray(value)
    if not data:
        return bytes(data)
    byte_index = rng.randrange(len(data))
    bit_index = rng.randrange(8)
    data[byte_index] ^= 1 << bit_index
    return bytes(data)


def flip_bit_at(value: bytes | bytearray | int, bit_index: int) -> bytes | int:
    """翻转指定位置的 bit。

    参数：
        value：bytes、bytearray 或非负整数。
        bit_index：从 0 开始的 bit 下标；bytes 输入按字节序从前往后计数。

    返回值：
        bytes | int：翻转后的新值。

    异常：
        ValueError：bit_index 为负、整数为负或 bit_index 超出 bytes 长度时抛出。
    """

    if bit_index < 0:
        raise ValueError("bit_index must be non-negative.")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Only non-negative integers are supported.")
        return value ^ (1 << bit_index)

    data = bytearray(value)
    total_bits = len(data) * 8
    if bit_index >= total_bits:
        raise ValueError("bit_index exceeds payload size.")
    byte_index, inner_bit = divmod(bit_index, 8)
    data[byte_index] ^= 1 << inner_bit
    return bytes(data)
