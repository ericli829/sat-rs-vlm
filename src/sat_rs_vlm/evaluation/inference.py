"""Qwen3-VL 普通评测、量化和可靠性评测共用的生成辅助函数。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.training.utils import model_input_device, move_to_device


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

    return generate_predictions(
        model,
        processor,
        collator,
        [sample],
        generation_config,
        torch,
        task_type=str(sample.get("task_type", "")),
    )[0]


def generate_predictions(
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    samples: Sequence[dict[str, Any]],
    generation_config: dict[str, Any],
    torch: Any,
    *,
    task_type: str | None = None,
) -> list[str]:
    """Generate and decode one homogeneous evaluation batch."""

    if not samples:
        return []
    batch = collator(list(samples))
    input_length = int(batch["input_ids"].shape[-1])
    input_device = model_input_device(model, torch)
    batch = move_to_device(batch, input_device, torch)
    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            **build_generation_kwargs(generation_config, task_type),
        )
    generated_ids = output_ids[:, input_length:]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if len(decoded) != len(samples):
        raise RuntimeError(
            f"Decoded prediction count mismatch: expected {len(samples)}, got {len(decoded)}"
        )
    return [str(value).strip() for value in decoded]


def timed_prediction(
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    sample: dict[str, Any],
    generation_config: dict[str, Any],
    torch: Any,
) -> tuple[str, float]:
    """测量单样本端到端延迟；CUDA 前后显式同步。"""

    device = model_input_device(model, torch)
    _synchronize(torch, device)
    started = time.perf_counter()
    prediction = generate_prediction(model, processor, collator, sample, generation_config, torch)
    _synchronize(torch, device)
    return prediction, (time.perf_counter() - started) * 1000.0


def timed_predictions(
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    samples: Sequence[dict[str, Any]],
    generation_config: dict[str, Any],
    torch: Any,
    *,
    task_type: str | None = None,
) -> tuple[list[str], float]:
    """Return batched predictions and amortized latency per sample."""

    if not samples:
        return [], 0.0
    device = model_input_device(model, torch)
    _synchronize(torch, device)
    started = time.perf_counter()
    predictions = generate_predictions(
        model,
        processor,
        collator,
        samples,
        generation_config,
        torch,
        task_type=task_type,
    )
    _synchronize(torch, device)
    latency_ms = (time.perf_counter() - started) * 1000.0 / len(samples)
    return predictions, latency_ms
