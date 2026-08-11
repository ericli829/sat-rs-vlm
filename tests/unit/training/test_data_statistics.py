from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.training.data_statistics import analyze_training_data, numeric_summary, percentile

torch = pytest.importorskip("torch")


class LengthProcessor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(padding_side="right")
        self.image_processor = SimpleNamespace(merge_size=2)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        del tokenize
        has_assistant = any(message.get("role") == "assistant" for message in messages)
        length = 8 if has_assistant and not add_generation_prompt else 4
        return f"length={length}"

    def __call__(self, *, text: list[str], truncation: bool, **kwargs: Any) -> dict[str, Any]:
        max_length = int(kwargs.get("max_length", 10_000))
        lengths = [int(value.split("=")[1]) for value in text]
        lengths = [min(length, max_length) if truncation else length for length in lengths]
        width = max(lengths)
        ids = [list(range(1, length + 1)) + [0] * (width - length) for length in lengths]
        masks = [[1] * length + [0] * (width - length) for length in lengths]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }


def _sample(sample_id: str, task: str = "vqa") -> dict[str, Any]:
    return {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "task_type": task,
        "metadata": {"dataset": "VRSBench", "training_source": "VRSBench"},
    }


def test_token_diagnostics_reuse_assistant_mask_and_detect_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Qwen3VLDataCollator,
        "_process_vision_info",
        staticmethod(lambda messages: (None, None)),
    )
    collator = Qwen3VLDataCollator(LengthProcessor(), 6, ".")

    diagnostics = collator.tokenization_diagnostics(_sample("one"))

    assert diagnostics["prompt_tokens"] == 4
    assert diagnostics["assistant_tokens"] == 2
    assert diagnostics["total_tokens"] == 6
    assert diagnostics["truncated"] is True
    assert diagnostics["assistant_truncated"] is True
    labels = diagnostics["encoded"]["labels"]
    assert labels.tolist() == [[-100, -100, -100, -100, 5, 6]]
    assert int((labels != -100).sum().item()) == 2


def test_task_aggregation_and_percentiles_use_exact_supervised_counts() -> None:
    class FakeCollator:
        max_seq_length = 1024
        processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))

        def tokenization_diagnostics(self, sample: dict[str, Any]) -> dict[str, Any]:
            supervised = int(sample["metadata"]["supervised"])
            return {
                "encoded": {},
                "prompt_tokens": 5,
                "assistant_tokens": supervised,
                "total_tokens": 5 + supervised,
                "uncapped_total_tokens": 5 + supervised,
                "truncated": False,
                "assistant_truncated": False,
            }

    samples = [_sample("a", "vqa"), _sample("b", "vqa"), _sample("c", "captioning")]
    samples[0]["metadata"]["supervised"] = 1
    samples[1]["metadata"]["supervised"] = 3
    samples[2]["metadata"]["supervised"] = 6
    report = analyze_training_data(
        samples,
        FakeCollator(),  # type: ignore[arg-type]
        image_root=".",
        inspect_images=False,
    )

    assert report["supervised_token_statistics"]["total_supervised_tokens"] == 10
    assert report["supervised_token_statistics"]["by_task"]["vqa"]["mean"] == 2
    assert (
        report["supervised_token_statistics"]["by_task"]["captioning"]["supervised_token_share"]
        == 0.6
    )
    assert report["task_statistics"]["vqa"]["sample_count"] == 2
    assert report["truncation_statistics"]["truncation_rate"] == 0
    assert percentile([1, 2, 3, 4], 0.90) == pytest.approx(3.7)
    assert numeric_summary([1, 2, 3, 4])["p95"] == pytest.approx(3.85)
