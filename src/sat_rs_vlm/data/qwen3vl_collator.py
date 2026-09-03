"""Qwen3-VL 数据整理器。

Collator 在 batch 级别调用 AutoProcessor，把 messages 中的文本和单图/多图输入
编码为模型需要的张量。图像路径解析规则：绝对路径直接使用；相对路径相对
image_root；找不到文件时抛出包含 sample id 的错误。训练模式只对 assistant 答案
token 计算损失，user、图像占位 token 和 padding 均设置为 -100。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

from sat_rs_vlm.data.object_adapter_v0 import (
    _cardinality_prompt_target,
    extract_answer,
    extract_prompt,
)
from sat_rs_vlm.data.task_protocol import parse_count
from sat_rs_vlm.infrastructure.telemetry import visual_input_telemetry

QWEN_UTILS_ERROR = 'qwen-vl-utils is required. Install with: pip install -e ".[model]"'


class Qwen3VLDataCollator:
    """Qwen3-VL 训练 collator。"""

    def __init__(
        self,
        processor: Any,
        max_seq_length: int,
        image_root: str | Path,
        *,
        debug_shapes: bool = False,
        for_generation: bool = False,
        include_task_metadata: bool = False,
        truncation: bool = True,
    ) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.image_root = Path(image_root)
        self.debug_shapes = debug_shapes
        self.for_generation = for_generation
        self.include_task_metadata = include_task_metadata
        # Qwen3-VL 的图像占位 token 数必须与 processor 生成的视觉特征数一致。
        # 正式语言模型训练保持默认截断；仅视觉特征提取可显式关闭它，避免把
        # 图像占位 token 截断后再与完整 pixel_values 配对。
        self.truncation = truncation
        self.last_batch_telemetry: dict[str, Any] = {}

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """编码一个 batch 并生成 labels。"""

        if self.processor is None:
            raise ValueError("processor is required when collating a real batch.")
        normalized_messages = [self._messages_with_resolved_images(sample) for sample in batch]
        prompt_messages = [
            [message for message in messages if message.get("role") != "assistant"]
            for messages in normalized_messages
        ]
        if self.for_generation:
            texts = [
                str(
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                for messages in prompt_messages
            ]
            image_inputs, video_inputs = self._process_vision_info(prompt_messages)
            encoded = self._encode(
                texts,
                image_inputs,
                video_inputs,
                truncation=self.truncation,
            )
        else:
            full_texts = [
                str(
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
                for messages in normalized_messages
            ]
            prompt_texts = [
                str(
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                for messages in prompt_messages
            ]
            image_inputs, video_inputs = self._process_vision_info(normalized_messages)
            encoded = self._encode(
                full_texts,
                image_inputs,
                video_inputs,
                truncation=self.truncation,
            )
            prompt_encoded = self._encode(
                prompt_texts,
                image_inputs,
                video_inputs,
                truncation=self.truncation,
            )
            sample_ids = [str(sample.get("id", "<unknown>")) for sample in batch]
            encoded["labels"] = self._build_assistant_labels(
                encoded,
                prompt_encoded,
                sample_ids,
            )
            if self.include_task_metadata:
                encoded["task_types"] = [
                    str(sample.get("task_type", "unknown")).strip().lower() or "unknown"
                    for sample in batch
                ]
                encoded["sample_ids"] = [str(sample.get("id", "<unknown>")) for sample in batch]
                encoded["auxiliary_counts"] = [
                    self._auxiliary_count_target(sample) for sample in batch
                ]
        image_groups = [
            list(images) if isinstance(images, list | tuple) else []
            for images in image_inputs
        ]
        self.last_batch_telemetry = visual_input_telemetry(
            image_groups,
            encoded.get("image_grid_thw"),
        )
        if self.debug_shapes:
            shapes = {
                key: tuple(value.shape) for key, value in encoded.items() if hasattr(value, "shape")
            }
            print(f"Batch tensor shapes: {shapes}")
        return dict(encoded)

    @staticmethod
    def _auxiliary_count_target(sample: dict[str, Any]) -> int:
        """Return a strict exact-cardinality target, or -100 for aux masking."""

        if str(sample.get("task_type", "")).strip().lower() != "counting":
            return -100
        if _cardinality_prompt_target(extract_prompt(sample)) is None:
            return -100
        parsed = parse_count(extract_answer(sample))
        return int(parsed.value) if parsed.value is not None else -100

    def _encode(
        self,
        texts: list[str],
        image_inputs: Any,
        video_inputs: Any,
        *,
        truncation: bool = True,
        padding: bool = True,
    ) -> dict[str, Any]:
        """调用 processor 编码文本和视觉输入。"""

        kwargs: dict[str, Any] = {
            "text": texts,
            "images": image_inputs,
            "videos": video_inputs,
            "padding": padding,
            "truncation": truncation,
            "return_tensors": "pt",
        }
        if truncation:
            kwargs["max_length"] = self.max_seq_length
        return dict(self.processor(**kwargs))

    def tokenization_diagnostics(self, sample: dict[str, Any]) -> dict[str, Any]:
        """返回与真实训练 mask 同源的监督和截断诊断。

        capped labels 仍由 :meth:`_build_assistant_labels` 构造；是否截断则通过同一
        processor 的无上限编码判定，不用字符串长度近似。
        """

        if self.processor is None:
            raise ValueError("processor is required for tokenization diagnostics")
        normalized = self._messages_with_resolved_images(sample)
        prompt_messages = [message for message in normalized if message.get("role") != "assistant"]
        full_text = str(
            self.processor.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
        prompt_text = str(
            self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        image_inputs, video_inputs = self._process_vision_info([normalized])
        capped = self._encode([full_text], image_inputs, video_inputs, truncation=True)
        capped_prompt = self._encode(
            [prompt_text], image_inputs, video_inputs, truncation=True
        )
        labels = self._build_assistant_labels(
            capped,
            capped_prompt,
            [str(sample.get("id", "<unknown>"))],
        )
        uncapped = self._encode(
            [full_text],
            image_inputs,
            video_inputs,
            truncation=False,
            padding=False,
        )
        uncapped_prompt = self._encode(
            [prompt_text],
            image_inputs,
            video_inputs,
            truncation=False,
            padding=False,
        )
        capped_total = self._active_length(capped)
        uncapped_total = self._active_length(uncapped)
        uncapped_prompt_length = self._active_length(uncapped_prompt)
        uncapped_assistant = max(0, uncapped_total - uncapped_prompt_length)
        supervised = int((labels[0] != -100).sum().item())
        encoded = dict(capped)
        encoded["labels"] = labels
        return {
            "encoded": encoded,
            "prompt_tokens": capped_total - supervised,
            "assistant_tokens": supervised,
            "total_tokens": capped_total,
            "uncapped_prompt_tokens": uncapped_prompt_length,
            "uncapped_assistant_tokens": uncapped_assistant,
            "uncapped_total_tokens": uncapped_total,
            "truncated": uncapped_total > capped_total,
            "assistant_truncated": uncapped_assistant > supervised,
        }

    @staticmethod
    def _active_length(encoded: dict[str, Any]) -> int:
        """统计单样本 processor 输出中的非 padding token。"""

        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            return int(attention_mask[0].sum().item())
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise ValueError("processor output must contain input_ids or attention_mask")
        return int(input_ids.shape[-1])

    def _build_assistant_labels(
        self,
        encoded: dict[str, Any],
        prompt_encoded: dict[str, Any],
        sample_ids: list[str],
    ) -> Any:
        """根据完整序列和 generation prompt 长度构造 assistant-only labels。"""

        labels = encoded["input_ids"].clone()
        attention_mask = encoded.get("attention_mask")
        prompt_mask = prompt_encoded.get("attention_mask")
        if attention_mask is None or prompt_mask is None:
            raise ValueError("assistant-only mask requires processor attention_mask output")
        padding_side = self._padding_side()
        _, sequence_length = labels.shape
        for index, sample_id in enumerate(sample_ids):
            full_length = int(attention_mask[index].sum().item())
            prompt_length = int(prompt_mask[index].sum().item())
            answer_length = full_length - prompt_length
            if answer_length <= 0:
                raise ValueError(
                    "No assistant tokens remain after tokenization/truncation for sample "
                    f"{sample_id}; full_length={full_length}, prompt_length={prompt_length}"
                )
            if padding_side == "left":
                answer_start = sequence_length - answer_length
                labels[index, :answer_start] = -100
            else:
                labels[index, :prompt_length] = -100
                labels[index, full_length:] = -100
            labels[index, attention_mask[index] == 0] = -100
            if int((labels[index] != -100).sum().item()) == 0:
                raise ValueError(f"No supervised assistant tokens for sample {sample_id}")
        return labels

    def _padding_side(self) -> str:
        """读取 tokenizer padding 方向并拒绝未知值。"""

        tokenizer = getattr(self.processor, "tokenizer", None)
        side = getattr(tokenizer, "padding_side", None) if tokenizer is not None else None
        side = side or getattr(self.processor, "padding_side", "right")
        normalized = str(side).lower()
        if normalized not in {"left", "right"}:
            raise ValueError(f"Unsupported tokenizer padding_side: {side}")
        return normalized

    def _messages_with_resolved_images(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """返回图像路径已解析为绝对路径的 messages。"""

        resolved_messages: list[dict[str, Any]] = []
        for message in list(sample["messages"]):
            content = message.get("content")
            if not isinstance(content, list):
                resolved_messages.append(dict(message))
                continue
            resolved_content: list[dict[str, Any]] = []
            for item in content:
                item_copy = dict(item)
                if item_copy.get("type") == "image":
                    raw_path = str(item_copy["image"])
                    path = self._resolve_image_path(raw_path)
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Image path does not exist for sample {sample.get('id')}: {raw_path} "
                            f"(resolved: {path})"
                        )
                    item_copy["image"] = str(path)
                resolved_content.append(item_copy)
            message_copy = dict(message)
            message_copy["content"] = resolved_content
            resolved_messages.append(message_copy)
        return resolved_messages

    def _resolve_image_path(self, image_path: str) -> Path:
        """解析图像路径。"""

        path = Path(image_path).expanduser()
        if path.is_absolute():
            return path
        return self.image_root / path

    @staticmethod
    def _process_vision_info(messages: list[list[dict[str, Any]]]) -> tuple[Any, Any]:
        """逐样本调用 qwen_vl_utils.process_vision_info。"""

        try:
            qwen_utils = importlib.import_module("qwen_vl_utils")
        except ModuleNotFoundError as exc:
            raise ImportError(QWEN_UTILS_ERROR) from exc
        process_vision_info = qwen_utils.process_vision_info
        image_inputs: list[Any] = []
        video_inputs: list[Any] = []
        has_video = False
        for sample_messages in messages:
            sample_images, sample_videos = cast(
                tuple[Any, Any], process_vision_info(sample_messages)
            )
            image_inputs.append(sample_images)
            video_inputs.append(sample_videos)
            has_video = has_video or bool(sample_videos)
        return image_inputs, video_inputs if has_video else None
