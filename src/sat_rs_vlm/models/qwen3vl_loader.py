"""Qwen3-VL 基座、Processor 与可选 LoRA adapter 的共享加载边界。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def compatible_model_class(transformers: Any) -> Any:
    """选择当前 Transformers 版本实际提供的多模态模型类。"""

    for name in (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            return model_class
    raise ImportError("Transformers does not provide a Qwen3-VL compatible model class")


def validate_local_adapter(adapter_source: str | Path, *, local_files_only: bool) -> None:
    """在 PEFT 加载前验证本地 adapter，避免误当成远程仓库 ID。"""

    path = Path(adapter_source)
    if not path.is_dir():
        if local_files_only:
            raise FileNotFoundError(f"Local LoRA adapter directory does not exist: {path}")
        return
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"LoRA adapter_config.json does not exist: {path}")
    weights = (path / "adapter_model.safetensors", path / "adapter_model.bin")
    if not any(candidate.is_file() for candidate in weights):
        raise FileNotFoundError(f"LoRA adapter weights do not exist in: {path}")


def load_qwen3vl_processor(
    modules: dict[str, Any],
    processor_source: str,
    processor_kwargs: dict[str, Any] | None = None,
) -> Any:
    """加载一次共享 Processor。"""

    return modules["transformers"].AutoProcessor.from_pretrained(
        processor_source,
        **dict(processor_kwargs or {}),
    )


def load_qwen3vl_model(
    *,
    modules: dict[str, Any],
    base_model: str,
    model_kwargs: dict[str, Any] | None = None,
    adapter_path: str | None = None,
) -> Any:
    """加载 Qwen3-VL，并按需挂载同一 LoRA adapter。"""

    transformers = modules["transformers"]
    model = compatible_model_class(transformers).from_pretrained(
        base_model,
        **dict(model_kwargs or {}),
    )
    if adapter_path is not None:
        local_files_only = bool((model_kwargs or {}).get("local_files_only", True))
        validate_local_adapter(adapter_path, local_files_only=local_files_only)
        peft = modules.get("peft")
        if peft is None:
            raise ImportError("PEFT is required when adapter_path is configured")
        model = peft.PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=local_files_only,
        )
    if hasattr(model, "eval"):
        model.eval()
    return model


def load_qwen3vl(
    *,
    modules: dict[str, Any],
    base_model: str,
    processor_source: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    processor_kwargs: dict[str, Any] | None = None,
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    """加载共享 Processor 和 Qwen3-VL，可选挂载同一 LoRA adapter。"""

    processor = load_qwen3vl_processor(
        modules,
        processor_source or base_model,
        processor_kwargs,
    )
    model = load_qwen3vl_model(
        modules=modules,
        base_model=base_model,
        model_kwargs=model_kwargs,
        adapter_path=adapter_path,
    )
    return model, processor
