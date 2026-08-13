from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator

torch = pytest.importorskip("torch")


class FakeProcessor:
    def __init__(self, padding_side: str) -> None:
        self.tokenizer = SimpleNamespace(padding_side=padding_side, pad_token_id=0)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        text = str(messages[0].get("content", ""))
        prompt_length = 3 if "long" in text else 2
        has_assistant = any(message.get("role") == "assistant" for message in messages)
        length = prompt_length + 2 if has_assistant and not add_generation_prompt else prompt_length
        return f"length={length}"

    def __call__(self, *, text: list[str], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        lengths = [int(item.split("=")[1]) for item in text]
        sequence_length = max(lengths)
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        for length in lengths:
            values = list(range(1, length + 1))
            padding = [0] * (sequence_length - length)
            if self.tokenizer.padding_side == "left":
                ids.append(padding + values)
                masks.append(padding + [1] * length)
            else:
                ids.append(values + padding)
                masks.append([1] * length + padding)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }


def _sample(sample_id: str, text: str, *, assistant: bool = True) -> dict[str, Any]:
    messages = [{"role": "user", "content": text}]
    if assistant:
        messages.append({"role": "assistant", "content": "answer"})
    return {"id": sample_id, "messages": messages, "task_type": "vqa"}


@pytest.mark.parametrize(
    ("padding_side", "expected"),
    [
        ("right", [[-100, -100, -100, 4, 5], [-100, -100, 3, 4, -100]]),
        ("left", [[-100, -100, -100, 4, 5], [-100, -100, -100, 3, 4]]),
    ],
)
def test_assistant_only_mask_handles_padding(
    monkeypatch: pytest.MonkeyPatch,
    padding_side: str,
    expected: list[list[int]],
) -> None:
    monkeypatch.setattr(
        Qwen3VLDataCollator,
        "_process_vision_info",
        staticmethod(lambda messages: (None, None)),
    )
    collator = Qwen3VLDataCollator(FakeProcessor(padding_side), 32, ".")

    batch = collator([_sample("long", "long question"), _sample("short", "short")])

    assert batch["labels"].tolist() == expected


def test_generation_mode_has_no_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Qwen3VLDataCollator,
        "_process_vision_info",
        staticmethod(lambda messages: (None, None)),
    )
    collator = Qwen3VLDataCollator(FakeProcessor("right"), 32, ".", for_generation=True)

    assert "labels" not in collator([_sample("one", "long question")])


def test_training_collator_optionally_carries_task_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Qwen3VLDataCollator,
        "_process_vision_info",
        staticmethod(lambda messages: (None, None)),
    )
    collator = Qwen3VLDataCollator(
        FakeProcessor("right"),
        32,
        ".",
        include_task_metadata=True,
    )

    batch = collator([_sample("one", "long question")])

    assert batch["task_types"] == ["vqa"]


def test_missing_assistant_tokens_reports_sample_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Qwen3VLDataCollator,
        "_process_vision_info",
        staticmethod(lambda messages: (None, None)),
    )
    collator = Qwen3VLDataCollator(FakeProcessor("right"), 32, ".")

    with pytest.raises(ValueError, match="sample missing-answer"):
        collator([_sample("missing-answer", "long question", assistant=False)])
