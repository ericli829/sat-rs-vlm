"""Qwen3-VL 数据整理器。

Collator 在 batch 级别调用 AutoProcessor，把 messages 中的文本和单图/多图输入
编码为模型需要的张量。图像路径解析规则：绝对路径直接使用；相对路径相对
image_root；找不到文件时抛出包含 sample id 的错误。
"""

import importlib
from pathlib import Path
from typing import Any, cast

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
    ) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.image_root = Path(image_root)
        self.debug_shapes = debug_shapes
        self.for_generation = for_generation

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """编码一个 batch 并生成 labels。"""

        if self.processor is None:
            raise ValueError("processor is required when collating a real batch.")
        normalized_messages = [self._messages_with_resolved_images(sample) for sample in batch]
        if self.for_generation:
            normalized_messages = [
                [message for message in messages if message.get("role") != "assistant"]
                for messages in normalized_messages
            ]
        texts = [
            str(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=self.for_generation,
                )
            )
            for messages in normalized_messages
        ]
        image_inputs, video_inputs = self._process_vision_info(normalized_messages)
        encoded = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        if not self.for_generation:
            labels = encoded["input_ids"].clone()
            pad_token_id = getattr(self.processor, "pad_token_id", None)
            if pad_token_id is None and getattr(self.processor, "tokenizer", None) is not None:
                pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100
            # TODO: 精确定位 assistant answer token span，将 user prompt token 也 mask 为 -100。
            encoded["labels"] = labels
        if self.debug_shapes:
            shapes = {
                key: tuple(value.shape) for key, value in encoded.items() if hasattr(value, "shape")
            }
            print(f"Batch tensor shapes: {shapes}")
        return dict(encoded)

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
