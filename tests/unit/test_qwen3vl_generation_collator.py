from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator


class FakeProcessor:
    """记录 generation collator 传入的消息和模板参数。"""

    def __init__(self) -> None:
        self.messages: list[list[dict[str, Any]]] = []
        self.generation_flags: list[bool] = []
        self.processor_calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        self.messages.append(messages)
        self.generation_flags.append(add_generation_prompt)
        return "templated prompt"

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.processor_calls.append(dict(kwargs))
        return {"input_ids": object(), "pixel_values": object()}


def test_generation_collator_removes_answer_and_adds_generation_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    processor = FakeProcessor()
    collator = Qwen3VLDataCollator(
        processor,
        max_seq_length=1024,
        image_root=tmp_path,
        for_generation=True,
    )
    monkeypatch.setattr(
        collator,
        "_process_vision_info",
        lambda messages: ([["image"]], None),
    )
    sample = {
        "id": "sample",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.name},
                    {"type": "text", "text": "Describe the image."},
                ],
            },
            {"role": "assistant", "content": "Ground-truth answer."},
        ],
    }

    batch = collator([sample])

    assert [message["role"] for message in processor.messages[0]] == ["user"]
    assert processor.generation_flags == [True]
    assert "labels" not in batch
    assert processor.processor_calls[0]["truncation"] is True
    assert processor.processor_calls[0]["max_length"] == 1024


def test_generation_collator_can_disable_text_truncation_for_visual_only_workflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    processor = FakeProcessor()
    collator = Qwen3VLDataCollator(
        processor,
        max_seq_length=128,
        image_root=tmp_path,
        for_generation=True,
        truncation=False,
    )
    monkeypatch.setattr(collator, "_process_vision_info", lambda messages: ([["image"]], None))

    collator(
        [
            {
                "id": "visual-only",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image.name},
                            {"type": "text", "text": "Find all ships."},
                        ],
                    }
                ],
            }
        ]
    )

    assert processor.processor_calls[0]["truncation"] is False
    assert "max_length" not in processor.processor_calls[0]
