"""统一量化后端注册与模型准备逻辑。"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from typing import Any

from sat_rs_vlm.models.qwen3vl_loader import load_qwen3vl_model
from sat_rs_vlm.quantization.config import QuantizationExperimentConfig
from sat_rs_vlm.training.utils import resolve_torch_dtype


class UnsupportedQuantizationError(RuntimeError):
    """当前设备、依赖或模型组合不支持所选后端。"""


def quantize_dynamic_linear(model: Any, torch: Any) -> Any:
    """对 CPU `torch.nn.Linear` 执行动态 qint8 量化。"""

    quantization = getattr(torch, "ao", None)
    quantization = getattr(quantization, "quantization", None)
    quantize_dynamic = getattr(quantization, "quantize_dynamic", None)
    if quantize_dynamic is None:
        quantize_dynamic = torch.quantization.quantize_dynamic
    model = model.to("cpu")
    model.eval()
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)


class QuantizationBackend(ABC):
    name: str
    device: str
    requires_bitsandbytes: bool = False

    def validate(self, config: QuantizationExperimentConfig, torch: Any | None = None) -> None:
        """检查依赖、设备和 adapter 组合，不进行静默降级。"""

        if self.requires_bitsandbytes and importlib.util.find_spec("bitsandbytes") is None:
            raise UnsupportedQuantizationError(
                "bnb_int8 requires optional bitsandbytes; install the qlora extra "
                "in a compatible CUDA environment"
            )
        if self.device == "cuda" and torch is not None and not bool(torch.cuda.is_available()):
            raise UnsupportedQuantizationError("bnb_int8 requires an available CUDA device")
        if self.name == "torch_dynamic_int8" and config.model.adapter_path:
            raise UnsupportedQuantizationError(
                "torch_dynamic_int8 + unmerged LoRA adapter is not verified; "
                "merge the adapter first"
            )

    def _common_kwargs(
        self,
        config: QuantizationExperimentConfig,
        modules: dict[str, Any],
        *,
        cpu: bool,
    ) -> dict[str, Any]:
        torch = modules["torch"]
        kwargs: dict[str, Any] = {
            "local_files_only": config.model.local_files_only,
            "trust_remote_code": config.model.trust_remote_code,
        }
        if not cpu:
            kwargs["device_map"] = config.model.device_map
        dtype = torch.float32 if cpu else resolve_torch_dtype(torch, config.model.torch_dtype)
        kwargs["dtype"] = dtype if dtype is not None else "auto"
        if config.model.attn_implementation:
            kwargs["attn_implementation"] = config.model.attn_implementation
        return kwargs

    @abstractmethod
    def load_model(
        self,
        config: QuantizationExperimentConfig,
        modules: dict[str, Any],
        *,
        quantized: bool,
    ) -> Any:
        """加载 baseline 或量化模型。"""

    @abstractmethod
    def compression_metadata(self, model: Any, torch: Any, *, quantized: bool) -> dict[str, Any]:
        """返回实际后端和 dtype 信息。"""


class BaselineBackend(QuantizationBackend):
    name = "baseline"
    device = "cpu"

    def load_model(
        self,
        config: QuantizationExperimentConfig,
        modules: dict[str, Any],
        *,
        quantized: bool,
    ) -> Any:
        del quantized
        cpu = config.quantization.device == "cpu"
        model = load_qwen3vl_model(
            modules=modules,
            base_model=config.model.model_source,
            model_kwargs=self._common_kwargs(config, modules, cpu=cpu),
            adapter_path=config.model.adapter_path,
        )
        return model.to("cpu") if cpu else model

    def compression_metadata(self, model: Any, torch: Any, *, quantized: bool) -> dict[str, Any]:
        del torch, quantized
        parameter = next(iter(model.parameters()), None)
        dtype = str(getattr(parameter, "dtype", "unknown")).removeprefix("torch.")
        device = str(getattr(parameter, "device", "unknown"))
        return {
            "backend": "none",
            "device": device,
            "weight_dtype": dtype,
            "compute_dtype": dtype,
            "benchmark_only": False,
            "reload_supported": True,
        }


class TorchDynamicInt8Backend(QuantizationBackend):
    name = "torch_dynamic_int8"
    device = "cpu"

    def load_model(
        self,
        config: QuantizationExperimentConfig,
        modules: dict[str, Any],
        *,
        quantized: bool,
    ) -> Any:
        model = load_qwen3vl_model(
            modules=modules,
            base_model=config.model.model_source,
            model_kwargs=self._common_kwargs(config, modules, cpu=True),
            adapter_path=None,
        )
        return quantize_dynamic_linear(model, modules["torch"]) if quantized else model.to("cpu")

    def compression_metadata(self, model: Any, torch: Any, *, quantized: bool) -> dict[str, Any]:
        del model, torch
        return {
            "backend": self.name if quantized else "none",
            "device": "cpu",
            "weight_dtype": "qint8" if quantized else "float32",
            "compute_dtype": "float32",
            "benchmark_only": bool(quantized),
            "reload_supported": False if quantized else True,
        }


class BitsAndBytesInt8Backend(QuantizationBackend):
    name = "bnb_int8"
    device = "cuda"
    requires_bitsandbytes = True

    def load_model(
        self,
        config: QuantizationExperimentConfig,
        modules: dict[str, Any],
        *,
        quantized: bool,
    ) -> Any:
        kwargs = self._common_kwargs(config, modules, cpu=False)
        if quantized:
            kwargs["quantization_config"] = modules["transformers"].BitsAndBytesConfig(
                load_in_8bit=True
            )
        return load_qwen3vl_model(
            modules=modules,
            base_model=config.model.model_source,
            model_kwargs=kwargs,
            adapter_path=config.model.adapter_path,
        )

    def compression_metadata(self, model: Any, torch: Any, *, quantized: bool) -> dict[str, Any]:
        parameter = next(iter(model.parameters()), None)
        dtype = str(getattr(parameter, "dtype", "unknown")).removeprefix("torch.")
        return {
            "backend": self.name if quantized else "none",
            "device": "cuda",
            "weight_dtype": "int8" if quantized else dtype,
            "compute_dtype": dtype,
            "benchmark_only": False,
            "reload_supported": None if quantized else True,
            "reload_verified": False if quantized else True,
            "cuda_version": getattr(torch.version, "cuda", None),
        }


_BACKENDS: dict[str, type[QuantizationBackend]] = {
    "baseline": BaselineBackend,
    "torch_dynamic_int8": TorchDynamicInt8Backend,
    "bnb_int8": BitsAndBytesInt8Backend,
}


def list_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def register_backend(
    name: str,
    backend: type[QuantizationBackend],
    *,
    replace: bool = False,
) -> None:
    """注册未来 INT4/GPTQ/AWQ/QAT 后端；默认拒绝覆盖已有稳定实现。"""

    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Quantization backend name must not be empty")
    if normalized in _BACKENDS and not replace:
        raise ValueError(f"Quantization backend is already registered: {normalized}")
    _BACKENDS[normalized] = backend


def create_backend(name: str) -> QuantizationBackend:
    """按注册名创建后端，未知后端立即失败。"""

    backend = _BACKENDS.get(name.strip().lower())
    if backend is None:
        raise ValueError(f"Unknown quantization backend '{name}'; available={list_backends()}")
    return backend()
