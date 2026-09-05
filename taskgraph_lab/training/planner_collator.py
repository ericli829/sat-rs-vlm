"""Text-only Qwen3-VL collator for Planner language-model training."""

from __future__ import annotations

from typing import Any

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator


class PlannerTextDataCollator(Qwen3VLDataCollator):
    """Reuse assistant masking while omitting absent visual processor inputs."""

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = super().__call__(batch)
        # The shared MultitaskTrainer consumes task_types but deliberately does
        # not pass Planner-only bookkeeping fields into the model forward.
        encoded.pop("sample_ids", None)
        encoded.pop("auxiliary_counts", None)
        return encoded

    @staticmethod
    def _process_vision_info(
        messages: list[list[dict[str, Any]]],
    ) -> tuple[Any, Any]:
        has_visual = any(
            isinstance(message.get("content"), list)
            and any(
                isinstance(item, dict) and item.get("type") in {"image", "video"}
                for item in message["content"]
            )
            for sample in messages
            for message in sample
        )
        if not has_visual:
            return None, None
        return Qwen3VLDataCollator._process_vision_info(messages)
