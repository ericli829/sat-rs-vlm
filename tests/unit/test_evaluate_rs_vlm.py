from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts.evaluate_rs_vlm import (
    build_generation_kwargs,
    generate_prediction,
    iter_evaluation_batches,
    summarize,
    validate_local_adapter,
)


class FakeTensor:
    """提供 shape 和设备移动能力的最小 tensor 替身。"""

    def __init__(self, shape: tuple[int, ...], device: str = "cpu") -> None:
        self.shape = shape
        self.device = device

    def to(self, device: object) -> FakeTensor:
        self.device = str(device)
        return self


class FakeOutputIds:
    """模拟 generate 返回的二维 token tensor。"""

    def __getitem__(self, key: object) -> list[list[int]]:
        assert key == (slice(None), slice(3, None))
        return [[9, 10]]


def test_greedy_generation_omits_temperature() -> None:
    kwargs = build_generation_kwargs(
        {"max_new_tokens": 64, "do_sample": False, "temperature": 0.0, "num_beams": 1}
    )

    assert kwargs == {"max_new_tokens": 64, "do_sample": False, "num_beams": 1}


def test_sampling_generation_includes_sampling_parameters() -> None:
    kwargs = build_generation_kwargs(
        {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
    )

    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.8
    assert kwargs["top_k"] == 20


def test_validate_local_adapter_requires_config_and_weights(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        validate_local_adapter(str(adapter), local_files_only=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="weights"):
        validate_local_adapter(str(adapter), local_files_only=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    validate_local_adapter(str(adapter), local_files_only=True)


def test_generate_prediction_moves_batch_to_model_input_device() -> None:
    input_ids = FakeTensor((1, 3))
    pixel_values = FakeTensor((4, 8))

    class Model:
        def get_input_embeddings(self) -> Any:
            return SimpleNamespace(weight=FakeTensor((1,), "cuda:0"))

        def generate(self, **kwargs: Any) -> FakeOutputIds:
            assert kwargs["input_ids"].device == "cuda:0"
            assert kwargs["pixel_values"].device == "cuda:0"
            assert "temperature" not in kwargs
            return FakeOutputIds()

    class Processor:
        def batch_decode(self, token_ids: Any, **kwargs: Any) -> list[str]:
            assert token_ids == [[9, 10]]
            assert kwargs["clean_up_tokenization_spaces"] is False
            return ["answer"]

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        device=lambda value: value,
        is_tensor=lambda value: isinstance(value, FakeTensor),
        inference_mode=lambda: nullcontext(),
    )
    collator = lambda batch: {  # noqa: E731
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    prediction = generate_prediction(
        Model(),
        Processor(),
        collator,  # type: ignore[arg-type]
        {"id": "sample"},
        {"do_sample": False},
        torch,
    )

    assert prediction == "answer"


def test_summary_reports_empty_prediction_rate() -> None:
    summary = summarize(
        [
            {"task_type": "vqa", "prediction": "", "reference": "yes"},
            {"task_type": "vqa", "prediction": "yes", "reference": "yes"},
        ]
    )

    assert summary["overall"]["empty_predictions"] == 1
    assert summary["overall"]["empty_prediction_rate"] == 0.5
    assert summary["by_task"]["vqa"]["empty_prediction_rate"] == 0.5


def test_evaluation_batches_group_by_task_and_preserve_original_indexes() -> None:
    dataset = [
        {"id": "caption-1", "task_type": "captioning"},
        {"id": "vqa-1", "task_type": "vqa"},
        {"id": "caption-2", "task_type": "captioning"},
    ]

    batches = list(iter_evaluation_batches(dataset, 2, group_by_task=True))

    assert batches == [
        ("captioning", [(0, dataset[0]), (2, dataset[2])]),
        ("vqa", [(1, dataset[1])]),
    ]


def test_evaluation_batches_reject_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        list(iter_evaluation_batches([], 0, group_by_task=True))
