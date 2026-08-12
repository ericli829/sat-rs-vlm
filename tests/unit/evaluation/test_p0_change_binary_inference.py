from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

from sat_rs_vlm.evaluation.inference import (
    CHANGE_BINARY_PROMPT,
    build_change_binary_sample,
    change_binary_inference_enabled,
    timed_change_binary_prediction,
)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], device: str = "cpu") -> None:
        self.shape = shape
        self.device = device

    def to(self, device: object) -> FakeTensor:
        self.device = str(device)
        return self


class FakeOutputIds:
    def __getitem__(self, key: object) -> list[list[int]]:
        assert key == (slice(None), slice(3, None))
        return [[9]]


def change_sample() -> dict[str, Any]:
    return {
        "id": "levir-1",
        "task_type": "change_detection",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "before.png"},
                    {"type": "image", "image": "after.png"},
                    {"type": "text", "text": "Describe the changes."},
                ],
            },
            {"role": "assistant", "content": "No change has occurred."},
        ],
    }


def test_build_change_binary_sample_keeps_images_and_replaces_text() -> None:
    sample = change_sample()

    binary = build_change_binary_sample(sample)

    assert binary["task_type"] == "change_binary"
    assert binary["messages"][0]["content"][-1] == {
        "type": "text",
        "text": CHANGE_BINARY_PROMPT,
    }
    assert [item["image"] for item in binary["messages"][0]["content"][:-1]] == [
        "before.png",
        "after.png",
    ]
    assert sample["task_type"] == "change_detection"
    assert change_binary_inference_enabled(sample, {})
    assert not change_binary_inference_enabled(sample, {"change_binary_enabled": False})


def test_timed_change_binary_prediction_uses_short_independent_generation() -> None:
    input_ids = FakeTensor((1, 3))

    class Model:
        def get_input_embeddings(self) -> Any:
            return SimpleNamespace(weight=FakeTensor((1,), "cuda:0"))

        def generate(self, **kwargs: Any) -> FakeOutputIds:
            assert kwargs["max_new_tokens"] == 5
            assert kwargs["do_sample"] is False
            return FakeOutputIds()

    class Processor:
        def batch_decode(self, token_ids: Any, **kwargs: Any) -> list[str]:
            assert token_ids == [[9]]
            return ["Answer: 0"]

    def collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
        assert batch[0]["task_type"] == "change_binary"
        assert batch[0]["messages"][0]["content"][-1]["text"] == CHANGE_BINARY_PROMPT
        return {"input_ids": input_ids}

    torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda device: None),
        is_tensor=lambda value: isinstance(value, FakeTensor),
        inference_mode=lambda: nullcontext(),
    )

    raw, flag, latency_ms = timed_change_binary_prediction(
        Model(),
        Processor(),
        collator,  # type: ignore[arg-type]
        change_sample(),
        {"do_sample": False, "change_binary_max_new_tokens": 5},
        torch,
    )

    assert raw == "Answer: 0"
    assert flag == 0
    assert latency_ms >= 0
