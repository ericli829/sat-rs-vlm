"""bytes、整数和 PyTorch tensor 的统一 bit flip 算法。

算法使用 XOR 翻转目标位。随机接口在完整 bit 地址空间中无放回抽样，固定 seed 会得到
相同的目标和记录。所有接口默认复制输入，tensor 也不会被原地修改。
"""

from __future__ import annotations

import math
import random
from typing import Any, TypeAlias

from sat_rs_vlm.models.reliability.schemas import BitFlipRecord

BitValue: TypeAlias = bytes | bytearray | int


def _display_scalar(value: Any) -> int | float | str | None:
    """把 tensor 标量转换为可安全写入 JSON 的值。"""

    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, bool):
        return int(item)
    if isinstance(item, int):
        return item
    if isinstance(item, float):
        return item if math.isfinite(item) else str(item)
    return str(item) if item is not None else None


def _restore_bytes_type(data: bytearray, original: bytes | bytearray) -> bytes | bytearray:
    return bytearray(data) if isinstance(original, bytearray) else bytes(data)


def flip_value_bit(
    value: BitValue,
    *,
    bit_index: int,
    target_name: str = "value",
    seed: int | None = None,
) -> tuple[BitValue, BitFlipRecord]:
    """翻转 bytes/bytearray 的全局 bit，或非负整数的指定 bit。

    `bit_index` 从最低有效位 0 开始。字节序列按字节从前向后计数。返回新值和
    `BitFlipRecord`，不会修改输入。
    """

    if bit_index < 0:
        raise ValueError("bit_index must be non-negative")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Only non-negative integers are supported")
        byte_index, local_bit = divmod(bit_index, 8)
        before_byte = (value >> (byte_index * 8)) & 0xFF
        updated = value ^ (1 << bit_index)
        after_byte = (updated >> (byte_index * 8)) & 0xFF
        record = BitFlipRecord(
            target_name=target_name,
            flat_index=0,
            byte_index=byte_index,
            bit_index=local_bit,
            dtype="int",
            before_value=value,
            after_value=updated,
            before_bytes=f"{before_byte:02x}",
            after_bytes=f"{after_byte:02x}",
            seed=seed,
        )
        return updated, record

    data = bytearray(value)
    if bit_index >= len(data) * 8:
        raise ValueError("bit_index exceeds payload size")
    byte_index, local_bit = divmod(bit_index, 8)
    before_byte = data[byte_index]
    data[byte_index] ^= 1 << local_bit
    updated_bytes = _restore_bytes_type(data, value)
    record = BitFlipRecord(
        target_name=target_name,
        flat_index=byte_index,
        byte_index=byte_index,
        bit_index=local_bit,
        dtype="bytearray" if isinstance(value, bytearray) else "bytes",
        shape=[len(data)],
        before_value=before_byte,
        after_value=data[byte_index],
        before_bytes=f"{before_byte:02x}",
        after_bytes=f"{data[byte_index]:02x}",
        seed=seed,
    )
    return updated_bytes, record


def flip_random_value_bits(
    value: BitValue,
    *,
    num_bits: int = 1,
    seed: int | None = None,
    target_name: str = "value",
) -> tuple[BitValue, list[BitFlipRecord]]:
    """在 bytes/bytearray/int 的 bit 地址空间中无放回随机翻转。"""

    if num_bits < 0:
        raise ValueError("num_bits must be non-negative")
    bit_count = max(value.bit_length(), 1) if isinstance(value, int) else len(value) * 8
    if num_bits > bit_count:
        raise ValueError(f"num_bits={num_bits} exceeds available bits={bit_count}")
    updated: BitValue = value
    records: list[BitFlipRecord] = []
    for global_bit in random.Random(seed).sample(range(bit_count), num_bits):
        updated, record = flip_value_bit(
            updated,
            bit_index=global_bit,
            target_name=target_name,
            seed=seed,
        )
        records.append(record)
    return updated, records


def flip_bit_at(value: BitValue, bit_index: int) -> bytes | int:
    """兼容旧 API：定点翻转并只返回新值；bytearray 统一返回 bytes。"""

    updated, _ = flip_value_bit(value, bit_index=bit_index)
    return bytes(updated) if isinstance(updated, bytearray) else updated


def flip_random_bit(value: BitValue, seed: int | None = None) -> bytes | int:
    """兼容旧 API：随机翻转一个 bit 并只返回新值。"""

    if not isinstance(value, int) and not value:
        return b""
    updated, _ = flip_random_value_bits(value, num_bits=1, seed=seed)
    return bytes(updated) if isinstance(updated, bytearray) else updated


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Tensor bit flip requires the optional 'model' dependencies") from exc
    return torch


def _tensor_layout(tensor: Any, torch: Any) -> tuple[Any, int]:
    layouts = {
        torch.float32: (torch.int32, 32),
        torch.float16: (torch.int16, 16),
        torch.bfloat16: (torch.int16, 16),
        torch.int8: (torch.int8, 8),
        torch.uint8: (torch.uint8, 8),
    }
    if tensor.dtype not in layouts:
        supported = "float32, float16, bfloat16, int8, uint8"
        raise TypeError(f"Unsupported tensor dtype {tensor.dtype}; supported dtypes: {supported}")
    return layouts[tensor.dtype]


def tensor_bit_width(tensor: Any) -> int:
    """返回受支持 tensor dtype 的每元素 bit 数；不支持时明确报错。"""

    torch = _torch_module()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    return int(_tensor_layout(tensor, torch)[1])


def flip_tensor_bit(
    tensor: Any,
    *,
    flat_index: int,
    bit_index: int,
    target_name: str = "tensor",
    seed: int | None = None,
) -> tuple[Any, BitFlipRecord]:
    """翻转 tensor 指定元素中的指定 bit，并返回复制张量与记录。"""

    torch = _torch_module()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    integer_dtype, bits_per_element = _tensor_layout(tensor, torch)
    if flat_index < 0 or flat_index >= tensor.numel():
        raise ValueError("flat_index is outside the tensor")
    if bit_index < 0 or bit_index >= bits_per_element:
        raise ValueError(f"bit_index must be in [0, {bits_per_element - 1}]")

    updated = tensor.detach().clone().contiguous()
    before_scalar = updated.reshape(-1)[flat_index].clone()
    integer_view = updated.view(integer_dtype).reshape(-1)
    before_storage = int(integer_view[flat_index].item())
    storage_mask = (1 << bits_per_element) - 1
    before_unsigned = before_storage & storage_mask
    after_unsigned = before_unsigned ^ (1 << bit_index)
    sign_bit = 1 << (bits_per_element - 1)
    after_storage = (
        after_unsigned
        if integer_dtype == torch.uint8 or after_unsigned < sign_bit
        else after_unsigned - (1 << bits_per_element)
    )
    integer_view[flat_index] = after_storage
    after_scalar = updated.reshape(-1)[flat_index].clone()
    element_bytes = bits_per_element // 8
    record = BitFlipRecord(
        target_name=target_name,
        flat_index=flat_index,
        byte_index=flat_index * element_bytes + bit_index // 8,
        bit_index=bit_index,
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=list(tensor.shape),
        before_value=_display_scalar(before_scalar),
        after_value=_display_scalar(after_scalar),
        before_bytes=before_unsigned.to_bytes(element_bytes, "little").hex(),
        after_bytes=after_unsigned.to_bytes(element_bytes, "little").hex(),
        seed=seed,
    )
    return updated, record


def flip_random_tensor_bits(
    tensor: Any,
    *,
    num_bits: int = 1,
    seed: int | None = None,
    target_name: str = "tensor",
) -> tuple[Any, list[BitFlipRecord]]:
    """在 tensor 的完整元素-bit 空间中无放回随机翻转多个 bit。"""

    torch = _torch_module()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    _, bits_per_element = _tensor_layout(tensor, torch)
    total_bits = tensor.numel() * bits_per_element
    if num_bits < 0:
        raise ValueError("num_bits must be non-negative")
    if num_bits > total_bits:
        raise ValueError(f"num_bits={num_bits} exceeds available bits={total_bits}")
    updated = tensor.detach().clone().contiguous()
    records: list[BitFlipRecord] = []
    for address in random.Random(seed).sample(range(total_bits), num_bits):
        flat_index, bit_index = divmod(address, bits_per_element)
        updated, record = flip_tensor_bit(
            updated,
            flat_index=flat_index,
            bit_index=bit_index,
            target_name=target_name,
            seed=seed,
        )
        records.append(record)
    return updated, records
