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
        del kwargs
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
