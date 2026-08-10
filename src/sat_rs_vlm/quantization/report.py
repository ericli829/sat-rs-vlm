"""公平量化 benchmark 的延迟、环境和对比报告。"""

from __future__ import annotations

import importlib.metadata
import platform
import statistics
from collections.abc import Sequence
from typing import Any


def latency_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    """返回 mean/median/p50/p95/min/max/samples，空列表保留 None。"""

    if not values:
        return {
            "mean": None,
            "median": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
            "samples": 0,
        }
    ordered = sorted(float(value) for value in values)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))
    median = statistics.median(ordered)
    return {
        "mean": statistics.fmean(ordered),
        "median": median,
        "p50": median,
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
        "samples": len(ordered),
    }


def environment_metadata(torch: Any | None = None) -> dict[str, Any]:
    """记录公平比较所需的软件、操作系统和设备信息。"""

    packages: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    device: dict[str, Any] = {"cuda_available": False, "name": platform.processor() or None}
    if torch is not None and bool(torch.cuda.is_available()):
        device = {
            "cuda_available": True,
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "cuda_version": getattr(torch.version, "cuda", None),
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": device,
    }


def comparison_summary(
    baseline: dict[str, Any] | None,
    quantized: dict[str, Any] | None,
) -> dict[str, Any]:
    """跳过 baseline 时 speedup 等不可计算字段保持 None。"""

    baseline_mean = (
        baseline.get("latency_ms", {}).get("mean") if isinstance(baseline, dict) else None
    )
    quantized_mean = (
        quantized.get("latency_ms", {}).get("mean") if isinstance(quantized, dict) else None
    )
    speedup = None
    if isinstance(baseline_mean, (int, float)) and isinstance(quantized_mean, (int, float)):
        speedup = float(baseline_mean) / float(quantized_mean) if quantized_mean else None
    return {
        "speedup": speedup,
        "accuracy_retention": None,
        "note": "Compare per-task metrics directly; no universal accuracy scalar is fabricated.",
    }
