from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sat_rs_vlm.training.utils import model_input_device, move_to_device


class FakeTensor:
    """记录 `.to(device)` 调用的最小 tensor 替身。"""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def to(self, device: object) -> FakeTensor:
        self.device = str(device)
        return self


def fake_torch(*, cuda_available: bool = True) -> SimpleNamespace:
    """构造设备工具函数所需的最小 torch 替身。"""

    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        device=lambda value: value,
        is_tensor=lambda value: isinstance(value, FakeTensor),
    )


def test_model_input_device_prefers_embedding_weight_device() -> None:
    embedding = SimpleNamespace(weight=FakeTensor("cuda:1"))
    model = SimpleNamespace(
        get_input_embeddings=lambda: embedding,
        parameters=lambda: iter([FakeTensor("cuda:0")]),
    )

    device = model_input_device(model, fake_torch())

    assert str(device) == "cuda:1"


def test_model_input_device_falls_back_to_first_parameter() -> None:
    model = SimpleNamespace(parameters=lambda: iter([FakeTensor("cuda:0")]))

    device = model_input_device(model, fake_torch())

    assert str(device) == "cuda:0"


def test_move_to_device_preserves_nested_batch_structure() -> None:
    batch: dict[str, Any] = {
        "input_ids": FakeTensor(),
        "vision": [FakeTensor(), (FakeTensor(),)],
        "metadata": "sample",
    }

    moved = move_to_device(batch, "cuda:0", fake_torch())

    assert moved["input_ids"].device == "cuda:0"
    assert moved["vision"][0].device == "cuda:0"
    assert moved["vision"][1][0].device == "cuda:0"
    assert moved["metadata"] == "sample"
