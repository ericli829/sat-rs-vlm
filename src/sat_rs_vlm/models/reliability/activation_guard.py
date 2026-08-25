"""Forward-hook activation anomaly detection for inference safety.

The guard observes selected module outputs without changing them.  It detects
NaN/Inf and excessive absolute values, then lets the caller fail closed after a
generation batch.  It is a detector, not a recovery or KV-cache rollback method.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field


class ActivationAnomaly(BaseModel):
    module: str
    tensor_shape: list[int]
    dtype: str
    reason: str
    max_abs: float | None = None


GuardMode = Literal["research", "deployment"]


class ActivationGuard:
    def __init__(self, model: Any, *, module_patterns: list[str], max_abs: float, mode: GuardMode = "research") -> None:
        if not module_patterns:
            raise ValueError("module_patterns must not be empty")
        if max_abs <= 0:
            raise ValueError("max_abs must be positive")
        if mode not in {"research", "deployment"}:
            raise ValueError("mode must be 'research' or 'deployment'")
        self.model = model
        self.module_patterns = tuple(module_patterns)
        self.max_abs = float(max_abs)
        self.mode = mode
        self.anomalies: list[ActivationAnomaly] = []
        self._handles: list[Any] = []
        self._checked_tensors = 0

    def _tensors(self, value: Any) -> list[Any]:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Activation guard requires torch") from exc
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, Mapping):
            return [tensor for item in value.values() for tensor in self._tensors(item)]
        if isinstance(value, (tuple, list)):
            return [tensor for item in value for tensor in self._tensors(item)]
        if hasattr(value, "to_tuple"):
            return self._tensors(value.to_tuple())
        return []

    def _hook(self, name: str):
        def observe(_: Any, __: tuple[Any, ...], output: Any) -> None:
            import torch
            for tensor in self._tensors(output):
                if not tensor.is_floating_point():
                    continue
                self._checked_tensors += 1
                finite = torch.isfinite(tensor)
                if not bool(finite.all().item()):
                    self.anomalies.append(ActivationAnomaly(
                        module=name, tensor_shape=list(tensor.shape),
                        dtype=str(tensor.dtype).removeprefix("torch."),
                        reason="non_finite",
                    ))
                    continue
                current_max = float(tensor.detach().abs().max().item()) if tensor.numel() else 0.0
                if current_max > self.max_abs:
                    self.anomalies.append(ActivationAnomaly(
                        module=name, tensor_shape=list(tensor.shape),
                        dtype=str(tensor.dtype).removeprefix("torch."),
                        reason="max_abs_exceeded", max_abs=current_max,
                    ))
        return observe

    def install(self) -> int:
        """Register hooks on selected modules and return the matched count."""

        matched = 0
        for name, module in self.model.named_modules():
            if name and any(pattern in name for pattern in self.module_patterns):
                self._handles.append(module.register_forward_hook(self._hook(name)))
                matched += 1
        if not matched:
            raise ValueError(f"No modules matched activation patterns: {self.module_patterns}")
        return matched

    def assert_healthy(self) -> None:
        if self.anomalies:
            raise RuntimeError(
                "Activation guard blocked inference due to anomalies: "
                + ", ".join(f"{item.module}:{item.reason}" for item in self.anomalies[:5])
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "scope": "selected_module_forward_outputs",
            "does_not_protect": ["parameter_memory", "kv_cache_recovery", "automatic_recompute"],
            "module_patterns": list(self.module_patterns),
            "max_abs_threshold": self.max_abs,
            "mode": self.mode,
            "checked_tensors": self._checked_tensors,
            "anomalies": [item.model_dump(mode="json") for item in self.anomalies],
        }
