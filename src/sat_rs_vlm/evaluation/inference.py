"""Qwen3-VL 普通评测、量化和可靠性评测共用的生成辅助函数。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.evaluation.parsers import parse_explicit_change_prediction
from sat_rs_vlm.training.utils import model_input_device, move_to_device

CHANGE_BINARY_PROMPT_VERSION = "levir_semantic_change_binary_v1"
CHANGE_BINARY_PROMPT = (
    "The first remote-sensing image is before and the second is after. Determine whether "
    "a meaningful building or permanent structural change occurred.\n"
    "Building construction, demolition, expansion, replacement, or removal counts as a "
    "semantic change.\n"
    "Ignore illumination, season, resolution, viewpoint, registration noise, and temporary "
    "objects.\n"
    "Answer exactly one digit:\n"
    "0 = no semantic change\n"
    "1 = semantic change"
)


@dataclass(frozen=True)
class PredictionTiming:
    """单样本生成的可复现性能记录。"""

    end_to_end_latency_ms: float
    generation_latency_ms: float
    ttft_ms: float | None
    decode_latency_ms: float
    output_token_count: int
    generation_tokens_per_second: float | None
    decode_tokens_per_second: float | None
    input_profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_to_end_latency_ms": self.end_to_end_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "ttft_ms": self.ttft_ms,
            "decode_latency_ms": self.decode_latency_ms,
            "output_token_count": self.output_token_count,
            "generation_tokens_per_second": self.generation_tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "input_profile": self.input_profile,
        }


def extract_reference(messages: list[dict[str, Any]]) -> str:
    """提取首条 assistant 标准答案。"""

    for message in messages:
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def extract_message_inputs(
    sample: dict[str, Any], image_root: str | Path
) -> tuple[list[Path], str, str]:
    """从 messages 解析单图/多图路径、用户问题和 reference。"""

    root = Path(image_root)
    images: list[Path] = []
    questions: list[str] = []
    messages = list(sample.get("messages", []))
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            questions.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image":
                path = Path(str(item.get("image", ""))).expanduser()
                images.append(path if path.is_absolute() else root / path)
            elif item.get("type") == "text":
                questions.append(str(item.get("text", "")))
    question = "\n".join(part.strip() for part in questions if part.strip())
    return images, question, extract_reference(messages)


def build_generation_kwargs(
    generation_config: dict[str, Any], task_type: str | None = None
) -> dict[str, Any]:
    """构造 generate 参数；贪心生成不传 temperature 等无效采样参数。"""

    task_limits = generation_config.get("task_max_new_tokens", {})
    max_tokens = generation_config.get("max_new_tokens", 256)
    if task_type and isinstance(task_limits, dict) and task_type in task_limits:
        max_tokens = task_limits[task_type]
    do_sample = bool(generation_config.get("do_sample", False))
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_tokens),
        "do_sample": do_sample,
        "num_beams": int(generation_config.get("num_beams", 1)),
    }
    if do_sample:
        kwargs["temperature"] = float(generation_config.get("temperature", 1.0))
        if "top_p" in generation_config:
            kwargs["top_p"] = float(generation_config["top_p"])
        if "top_k" in generation_config:
            kwargs["top_k"] = int(generation_config["top_k"])
    return kwargs


def _synchronize(torch: Any, device: Any) -> None:
    synchronize = getattr(getattr(torch, "cuda", None), "synchronize", None)
    if callable(synchronize) and str(device).startswith("cuda"):
        synchronize(device)


def generate_prediction(
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    sample: dict[str, Any],
    generation_config: dict[str, Any],
    torch: Any,
) -> str:
    """只解码 input_length 之后的新 token。"""

    batch = collator([sample])
    input_length = int(batch["input_ids"].shape[-1])
    input_device = model_input_device(model, torch)
    batch = move_to_device(batch, input_device, torch)
    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            **build_generation_kwargs(generation_config, str(sample.get("task_type", ""))),
        )
    generated_ids = output_ids[:, input_length:]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return str(decoded[0]).strip() if decoded else ""


def _first_token_timer() -> tuple[Any | None, Any | None]:
    """创建 Transformers 首Token回调；缺少运行依赖时保持TTFT为空。"""

    try:
        from transformers.generation.stopping_criteria import (  # type: ignore[import-not-found]
            StoppingCriteria,
            StoppingCriteriaList,
        )
    except ImportError:
        return None, None

    class FirstTokenTimer(StoppingCriteria):
        timestamp: float | None = None

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            del input_ids, scores, kwargs
            if self.timestamp is None:
                self.timestamp = time.perf_counter()
            return False

    timer = FirstTokenTimer()
    return timer, StoppingCriteriaList([timer])


def _token_count(token_ids: Any) -> int:
    """读取生成Token数量，并兼容最小化单元测试替身。"""

    shape = getattr(token_ids, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])
    try:
        return len(token_ids[0])
    except (IndexError, KeyError, TypeError):
        return 0


def _tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return [int(dimension) for dimension in shape] if shape is not None else None


def _to_nested_list(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _input_profile(
    sample: dict[str, Any],
    collator: Qwen3VLDataCollator,
    batch: dict[str, Any],
) -> dict[str, Any]:
    """记录实际送入 Processor 的视觉输入规格，无法解析时明确标注。"""

    image_paths: list[Path] = []
    for message in list(sample.get("messages", [])):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            raw_path = Path(str(item.get("image", ""))).expanduser()
            root = getattr(collator, "image_root", Path())
            image_paths.append(raw_path if raw_path.is_absolute() else root / raw_path)
    original_sizes: list[list[int] | None] = []
    try:
        from PIL import Image

        for path in image_paths:
            with Image.open(path) as image:
                original_sizes.append([int(image.width), int(image.height)])
    except (ImportError, OSError):
        original_sizes = [None] * len(image_paths)
    tensor_shapes = {
        key: shape
        for key, value in batch.items()
        if (shape := _tensor_shape(value)) is not None
        and key in {"pixel_values", "pixel_values_videos"}
    }
    image_grid = _to_nested_list(batch.get("image_grid_thw"))
    visual_token_count: int | None = None
    token_status = "unresolved_no_image_grid"
    if isinstance(image_grid, list):
        try:
            raw_tokens = sum(int(grid[0]) * int(grid[1]) * int(grid[2]) for grid in image_grid)
            image_processor = getattr(getattr(collator, "processor", None), "image_processor", None)
            merge_size = getattr(image_processor, "merge_size", None)
            if isinstance(merge_size, int) and merge_size > 0:
                visual_token_count = raw_tokens // (merge_size * merge_size)
                token_status = "derived_from_image_grid_thw_and_merge_size"
            else:
                token_status = "unresolved_missing_processor_merge_size"
        except (IndexError, TypeError, ValueError):
            token_status = "unresolved_invalid_image_grid_thw"
    metadata = sample.get("metadata", {})
    tile_count = (
        metadata.get("tile_count", metadata.get("num_tiles"))
        if isinstance(metadata, dict)
        else None
    )
    tile_status = "metadata_reported" if isinstance(tile_count, int) else "unresolved"
    return {
        "image_count": len(image_paths),
        "tile_count": tile_count if isinstance(tile_count, int) else None,
        "tile_count_status": tile_status,
        "original_image_sizes": original_sizes,
        "processed_tensor_shapes": tensor_shapes,
        "image_grid_thw": image_grid,
        "visual_token_count": visual_token_count,
        "visual_token_count_status": token_status,
    }


def generate_prediction_with_timing(
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    sample: dict[str, Any],
    generation_config: dict[str, Any],
    torch: Any,
) -> tuple[str, PredictionTiming]:
    """生成预测并记录端到端时延、TTFT、Token数和解码速度。"""

    device = model_input_device(model, torch)
    _synchronize(torch, device)
    end_to_end_started = time.perf_counter()
    batch = collator([sample])
    input_profile = _input_profile(sample, collator, batch)
    input_length = int(batch["input_ids"].shape[-1])
    batch = move_to_device(batch, device, torch)
    first_token_timer, stopping_criteria = _first_token_timer()
    generation_kwargs = build_generation_kwargs(
        generation_config, str(sample.get("task_type", ""))
    )
    if stopping_criteria is not None:
        generation_kwargs["stopping_criteria"] = stopping_criteria
    generation_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**batch, **generation_kwargs)
    _synchronize(torch, device)
    generation_finished = time.perf_counter()
    generated_ids = output_ids[:, input_length:]
    decode_started = time.perf_counter()
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    finished = time.perf_counter()
    token_count = _token_count(generated_ids)
    first_token_timestamp = getattr(first_token_timer, "timestamp", None)
    ttft_ms = (
        (float(first_token_timestamp) - end_to_end_started) * 1000.0
        if isinstance(first_token_timestamp, (int, float))
        else None
    )
    generation_seconds = generation_finished - generation_started
    generation_tokens_per_second = (
        token_count / generation_seconds if token_count > 0 and generation_seconds > 0 else None
    )
    decode_seconds = (
        generation_finished - float(first_token_timestamp)
        if isinstance(first_token_timestamp, (int, float))
        else 0.0
    )
    decode_tokens_per_second = (
        (token_count - 1) / decode_seconds
        if token_count > 1 and decode_seconds > 0
        else None
    )
    timing = PredictionTiming(
        end_to_end_latency_ms=(finished - end_to_end_started) * 1000.0,
        generation_latency_ms=generation_seconds * 1000.0,
        ttft_ms=ttft_ms,
        decode_latency_ms=(finished - decode_started) * 1000.0,
        output_token_count=token_count,
        generation_tokens_per_second=generation_tokens_per_second,
        decode_tokens_per_second=decode_tokens_per_second,
        input_profile=input_profile,
    )
    return (str(decoded[0]).strip() if decoded else ""), timing


def timed_prediction(
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    sample: dict[str, Any],
    generation_config: dict[str, Any],
    torch: Any,
) -> tuple[str, float]:
    """测量单样本端到端延迟；CUDA 前后显式同步。"""

    prediction, timing = generate_prediction_with_timing(
        model,
        processor,
        collator,
        sample,
        generation_config,
        torch,
    )
    return prediction, timing.end_to_end_latency_ms


def timed_prediction_with_telemetry(
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    sample: dict[str, Any],
    generation_config: dict[str, Any],
    torch: Any,
) -> tuple[str, PredictionTiming]:
    """供评测入口使用的性能监测生成函数。"""

    return generate_prediction_with_timing(
        model,
        processor,
        collator,
        sample,
        generation_config,
        torch,
    )


def is_change_detection_sample(sample: dict[str, Any]) -> bool:
    """Return whether a sample uses the repository change-detection task type."""

    task_type = str(sample.get("task_type", "")).strip().lower().replace("-", "_")
    return task_type == "change_detection"


def change_binary_inference_enabled(
    sample: dict[str, Any], generation_config: dict[str, Any]
) -> bool:
    """Enable the legacy auxiliary binary pass only when explicitly requested."""

    return is_change_detection_sample(sample) and bool(
        generation_config.get("change_binary_enabled", False)
    )


def build_change_binary_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Reuse the original images while replacing the caption question with P0 binary QA."""

    image_items: list[dict[str, Any]] = []
    for message in list(sample.get("messages", [])):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                image_items.append(dict(item))
    if not image_items:
        raise ValueError(
            f"Change binary inference requires image items for sample {sample.get('id')}"
        )
    binary_sample = dict(sample)
    binary_sample["task_type"] = "change_binary"
    binary_sample["messages"] = [
        {
            "role": "user",
            "content": [
                *image_items,
                {"type": "text", "text": CHANGE_BINARY_PROMPT},
            ],
        }
    ]
    return binary_sample


def timed_change_binary_prediction(
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    sample: dict[str, Any],
    generation_config: dict[str, Any],
    torch: Any,
) -> tuple[str, int | None, float]:
    """Run the independent P0 binary question and parse only its explicit answer."""

    binary_config = dict(generation_config)
    binary_config["max_new_tokens"] = int(
        generation_config.get("change_binary_max_new_tokens", 8)
    )
    binary_config["task_max_new_tokens"] = {}
    raw, latency_ms = timed_prediction(
        model,
        processor,
        collator,
        build_change_binary_sample(sample),
        binary_config,
        torch,
    )
    parsed = parse_explicit_change_prediction(raw)
    return raw, parsed.value, latency_ms
