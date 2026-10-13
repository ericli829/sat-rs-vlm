"""One Qwen3-VL base with serialized, cache-isolated PEFT adapter selection."""

from __future__ import annotations

import gc
import json
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _resolved_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"shared Qwen3-VL {label} directory does not exist: {path}")
    return path


def _adapter_contract(path: Path) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"shared Qwen3-VL adapter must use PEFT LoRA: {config_path}")
    if not any(
        (path / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(f"shared Qwen3-VL adapter weights are missing: {path}")
    if config.get("modules_to_save") or config.get("target_parameters"):
        raise ValueError("shared Qwen3-VL adapters cannot replace or target non-LoRA modules")
    if config.get("use_dora") or config.get("lora_bias"):
        raise ValueError("shared Qwen3-VL adapters require standard bias-free LoRA")
    return config


class SharedQwen3VLModelManager:
    """Own one model/processor and swap compatible adapters under one lock.

    Adapter changes flush each registered role's external KV sessions before
    changing PEFT state, so cached prefixes never cross model roles.
    """

    def __init__(
        self,
        *,
        base_model: str,
        processor_source: str,
        adapters: Mapping[str, str],
        model_kwargs: Mapping[str, Any],
        processor_kwargs: Mapping[str, Any],
        default_adapter: str,
    ) -> None:
        self.base_model = str(_resolved_directory(base_model, "base model"))
        self.processor_source = str(_resolved_directory(processor_source, "processor"))
        if Path(self.base_model) != Path(self.processor_source):
            raise ValueError("shared Qwen3-VL requires the same base model and processor directory")
        self.adapters = {
            str(name): str(_resolved_directory(path, f"{name} adapter"))
            for name, path in adapters.items()
        }
        if len(self.adapters) < 2 or default_adapter not in self.adapters:
            raise ValueError("shared Qwen3-VL requires at least two adapters and a valid default")
        self.default_adapter = default_adapter
        self.model_kwargs = dict(model_kwargs)
        self.processor_kwargs = dict(processor_kwargs)
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._active_adapter: str | None = None
        self._cache_releasers: dict[str, Any] = {}
        self._load_ms: float | None = None
        self._switch_count = 0
        self._switch_ms: list[float] = []
        self.adapter_configs = {
            name: _adapter_contract(Path(path)) for name, path in self.adapters.items()
        }
        self._validate_adapter_compatibility()

    def _validate_adapter_compatibility(self) -> None:
        configs = list(self.adapter_configs.values())
        reference = configs[0]
        base_stem = Path(self.base_model).name.casefold()
        for config in configs:
            declared = Path(str(config.get("base_model_name_or_path", ""))).name.casefold()
            if not declared or declared != base_stem:
                raise ValueError(
                    "shared Qwen3-VL adapters must declare the configured base model"
                )
            for field in (
                "peft_type",
                "r",
                "lora_alpha",
                "lora_dropout",
                "target_modules",
                "bias",
                "fan_in_fan_out",
            ):
                current, expected = config.get(field), reference.get(field)
                if field == "target_modules":
                    current = tuple(sorted(current or ()))
                    expected = tuple(sorted(expected or ()))
                if current != expected:
                    raise ValueError(f"shared Qwen3-VL adapter mismatch in PEFT field {field!r}")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def active_adapter(self) -> str | None:
        return self._active_adapter

    @property
    def model(self) -> Any | None:
        return self._model

    @property
    def processor(self) -> Any | None:
        return self._processor

    @property
    def torch(self) -> Any | None:
        return self._torch

    @property
    def metrics(self) -> dict[str, Any]:
        with self._lock:
            output: dict[str, Any] = {
                "enabled": True,
                "loaded": self.loaded,
                "base_model": self.base_model,
                "processor_source": self.processor_source,
                "adapter_names": list(self.adapters),
                "active_adapter": self._active_adapter,
                "load_ms": self._load_ms,
                "adapter_switch_count": self._switch_count,
                "adapter_switch_ms": list(self._switch_ms),
                "shared_model_object_id": id(self._model) if self._model is not None else None,
                "gpu_allocated_mb": None,
                "gpu_reserved_mb": None,
            }
            cuda = getattr(self._torch, "cuda", None)
            if cuda is not None and cuda.is_available():
                try:
                    output["gpu_allocated_mb"] = round(
                        float(cuda.memory_allocated()) / (1024**2), 2
                    )
                    output["gpu_reserved_mb"] = round(
                        float(cuda.memory_reserved()) / (1024**2), 2
                    )
                except Exception:
                    pass
            return output

    def register_cache_releaser(self, adapter_name: str, callback: Any) -> None:
        if adapter_name not in self.adapters:
            raise ValueError(f"unknown shared adapter role: {adapter_name}")
        with self._lock:
            self._cache_releasers[adapter_name] = callback

    def load(self) -> tuple[Any, Any, Any, Any]:
        with self._lock:
            if self._model is not None and self._processor is not None and self._torch is not None:
                tokenizer = getattr(self._processor, "tokenizer", self._processor)
                return self._model, self._processor, tokenizer, self._torch
            started = time.perf_counter()
            import peft
            import torch
            import transformers

            from .qwen3vl_loader import compatible_model_class, load_qwen3vl_processor

            processor = load_qwen3vl_processor(
                {"transformers": transformers},
                self.processor_source,
                self.processor_kwargs,
            )
            base = compatible_model_class(transformers).from_pretrained(
                self.base_model,
                **self.model_kwargs,
            )
            planner_name = self.default_adapter
            model = peft.PeftModel.from_pretrained(
                base,
                self.adapters[planner_name],
                adapter_name=planner_name,
                is_trainable=False,
                local_files_only=True,
            )
            for name, path in self.adapters.items():
                if name == planner_name:
                    continue
                model.load_adapter(
                    path,
                    adapter_name=name,
                    is_trainable=False,
                    local_files_only=True,
                )
            model.set_adapter(planner_name)
            if hasattr(model, "eval"):
                model.eval()
            self._model, self._processor, self._torch = model, processor, torch
            self._active_adapter = planner_name
            self._load_ms = (time.perf_counter() - started) * 1000.0
            tokenizer = getattr(processor, "tokenizer", processor)
            if hasattr(tokenizer, "padding_side"):
                tokenizer.padding_side = "left"
            return model, processor, tokenizer, torch

    @contextmanager
    def activate(self, adapter_name: str) -> Iterator[tuple[Any, Any, Any, Any]]:
        if adapter_name not in self.adapters:
            raise ValueError(f"unknown shared adapter role: {adapter_name}")
        with self._lock:
            model, processor, tokenizer, torch = self.load()
            if self._active_adapter != adapter_name:
                started = time.perf_counter()
                for release in tuple(self._cache_releasers.values()):
                    try:
                        release()
                    except Exception:
                        pass
                model.set_adapter(adapter_name)
                self._active_adapter = adapter_name
                self._switch_count += 1
                self._switch_ms.append((time.perf_counter() - started) * 1000.0)
            yield model, processor, tokenizer, torch

    def close(self) -> None:
        with self._lock:
            for release in tuple(self._cache_releasers.values()):
                try:
                    release()
                except Exception:
                    pass
            self._cache_releasers.clear()
            model, torch = self._model, self._torch
            close = getattr(model, "close", None)
            if callable(close):
                close()
            self._model = None
            self._processor = None
            self._torch = None
            self._active_adapter = None
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


__all__ = ["SharedQwen3VLModelManager"]
