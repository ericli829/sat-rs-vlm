"""Small helpers for safely sizing detector parallelism."""

from __future__ import annotations

import math
from typing import Any


def available_cuda_memory_gb() -> float | None:
    """Return currently free CUDA memory, when the optional backend is usable."""

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
    except Exception:
        return None
    return float(free_bytes) / float(1024**3)


def resolve_parallel_workers(
    requested: Any,
    *,
    max_workers: int = 3,
    worker_vram_gb: float = 4.0,
    vram_reserve_gb: float = 6.0,
) -> int:
    """Resolve a serial, explicit, or VRAM-aware worker count.

    ``auto`` uses the memory currently free in the parent process.  The reserve
    leaves room for transient model activations and unrelated CUDA allocations;
    callers still get a serial fallback when CUDA telemetry is unavailable.
    """

    bounded_max = max(1, int(max_workers))
    per_worker = float(worker_vram_gb)
    reserve = float(vram_reserve_gb)
    if not math.isfinite(per_worker) or per_worker <= 0.0:
        raise ValueError("parallel worker_vram_gb must be finite and positive")
    if not math.isfinite(reserve) or reserve < 0.0:
        raise ValueError("parallel vram_reserve_gb must be finite and non-negative")

    if requested is None:
        return 1
    if isinstance(requested, bool):
        return 1
    normalized = str(requested).strip().casefold()
    if normalized in {"", "serial", "disabled", "off"}:
        return 1
    if normalized in {"auto", "adaptive"}:
        free_memory_gb = available_cuda_memory_gb()
        if free_memory_gb is None:
            return 1
        available_budget_gb = max(0.0, free_memory_gb - reserve)
        return max(1, min(bounded_max, math.floor(available_budget_gb / per_worker)))

    explicit_workers = int(requested)
    if explicit_workers < 1:
        raise ValueError("parallel_workers must be positive")
    return min(bounded_max, explicit_workers)
