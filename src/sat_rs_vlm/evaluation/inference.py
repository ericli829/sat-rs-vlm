"""Qwen3-VL 普通评测、量化和可靠性评测共用的生成辅助函数。"""

from __future__ import annotations

import time
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
