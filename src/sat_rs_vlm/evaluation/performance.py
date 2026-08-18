"""可复现的本地推理性能监测与报告辅助函数。"""

from __future__ import annotations

import ctypes
import os
import platform
import statistics
import subprocess
import tracemalloc
from collections import defaultdict
from collections.abc import Iterable, Mapping
from ctypes import wintypes
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

PERFORMANCE_REPORT_VERSION = "inference_performance_v1"


def _number_statistics(values: Iterable[float]) -> dict[str, float | int | None]:
    """汇总时延类数值，P95 采用 nearest-rank 定义。"""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "mean": None,
            "median": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
            "samples": 0,
        }
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


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _nvidia_smi_metadata() -> dict[str, Any]:
    """查询 NVIDIA 驱动；不可用时显式记录而非中断评测。"""

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}", "gpus": []}
    if result.returncode != 0:
        return {
            "status": "unavailable",
            "reason": (result.stderr or "nvidia-smi failed").strip(),
            "gpus": [],
        }
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "driver_version": fields[2],
                    "total_memory_mb": float(fields[3]),
                }
            )
        except ValueError:
            continue
    return {"status": "ok" if gpus else "unavailable", "reason": None, "gpus": gpus}


def _windows_memory_mb() -> dict[str, float | None]:
    """读取 Windows 当前进程工作集；不依赖 psutil。"""

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    try:
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            ctypes.sizeof(counters),
        )
        if not ok:
            return {"rss_mb": None, "os_peak_rss_mb": None}
        mib = 1024 * 1024
        return {
            "rss_mb": float(counters.WorkingSetSize) / mib,
            "os_peak_rss_mb": float(counters.PeakWorkingSetSize) / mib,
        }
    except (AttributeError, OSError):
        return {"rss_mb": None, "os_peak_rss_mb": None}


def process_memory_snapshot_mb() -> dict[str, float | None]:
    """返回当前与操作系统可见的进程内存峰值。"""

    if os.name == "nt":
        return _windows_memory_mb()
    try:
        status = ("/proc/self/status")
        values: dict[str, int] = {}
        with open(status, encoding="utf-8") as handle:
            for line in handle:
                key, _, raw_value = line.partition(":")
                if key in {"VmRSS", "VmHWM"}:
                    values[key] = int(raw_value.split()[0])
        return {
            "rss_mb": values.get("VmRSS", 0) / 1024 if "VmRSS" in values else None,
            "os_peak_rss_mb": values.get("VmHWM", 0) / 1024 if "VmHWM" in values else None,
        }
    except (OSError, ValueError, IndexError):
        return {"rss_mb": None, "os_peak_rss_mb": None}


def environment_metadata(torch: Any | None, *, model_config: Mapping[str, Any]) -> dict[str, Any]:
    """记录复核资源测量所需的软硬件与模型执行配置。"""

    cuda_available = bool(
        torch is not None
        and getattr(torch, "cuda", None) is not None
        and bool(torch.cuda.is_available())
    )
    accelerator: dict[str, Any] = {
        "cuda_available": cuda_available,
        "nvidia_smi": _nvidia_smi_metadata(),
    }
    if cuda_available:
        try:
            device_index = int(torch.cuda.current_device())
            properties = torch.cuda.get_device_properties(device_index)
            accelerator.update(
                {
                    "device_index": device_index,
                    "name": torch.cuda.get_device_name(device_index),
                    "total_memory_mb": float(properties.total_memory) / (1024 * 1024),
                    "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                }
            )
        except (AttributeError, RuntimeError):
            accelerator["name"] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": {"logical_cores": os.cpu_count(), "processor": platform.processor() or None},
        "packages": _package_versions(),
        "accelerator": accelerator,
        "model_execution_config": {
            key: model_config.get(key)
            for key in ("torch_dtype", "device_map", "attn_implementation")
            if key in model_config
        },
    }


def _directory_size_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def model_resource_metadata(model: Any, *, model_config: Mapping[str, Any]) -> dict[str, Any]:
    """统计已加载模型的参数量和可访问本地模型文件。"""

    try:
        parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    except (AttributeError, RuntimeError, TypeError):
        parameter_count = None
    sources: list[dict[str, Any]] = []
    total_storage_bytes = 0
    visited: set[Path] = set()
    for config_key in ("base_model", "adapter_path"):
        source = model_config.get(config_key)
        if not isinstance(source, str):
            continue
        path = Path(source).expanduser()
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in visited:
            continue
        visited.add(resolved)
        try:
            storage_bytes = _directory_size_bytes(resolved)
        except OSError:
            continue
        total_storage_bytes += storage_bytes
        sources.append(
            {
                "role": config_key,
                "path": str(resolved),
                "storage_bytes": storage_bytes,
            }
        )
    return {
        "loaded_model_logical_parameter_count": parameter_count,
        "local_model_storage_bytes": total_storage_bytes if sources else None,
        "local_sources": sources,
        "storage_status": "resolved" if sources else "unresolved",
        "note": (
            "The parameter count is for the loaded model object. Local storage only includes "
            "configured model/adapter directories that are available on this host."
        ),
    }


class PerformanceMonitor:
    """收集正式推理阶段的资源与时延，不将预热计入性能结果。"""

    def __init__(self, torch: Any | None, *, device: Any | None = None) -> None:
        self._torch = torch
        self._device = device
        self._enabled_cuda = bool(
            torch is not None
            and getattr(torch, "cuda", None) is not None
            and bool(torch.cuda.is_available())
        )
        self._started = False
        self._latencies_ms: list[float] = []
        self._ttft_ms: list[float] = []
        self._generation_tokens_per_second: list[float] = []
        self._decode_tokens_per_second: list[float] = []
        self._output_tokens: list[int] = []
        self._image_counts: list[int] = []
        self._visual_token_counts: list[int] = []
        self._visual_token_statuses: dict[str, int] = defaultdict(int)
        self._rss_samples_mb: list[float] = []
        self._by_task: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def start(self) -> None:
        """重置可重置的峰值计数器并启动正式阶段内存采样。"""

        if self._started:
            raise RuntimeError("Performance monitor has already started")
        self._started = True
        if self._enabled_cuda:
            try:
                self._torch.cuda.synchronize(self._device)
                self._torch.cuda.reset_peak_memory_stats(self._device)
            except (AttributeError, RuntimeError, TypeError):
                pass
        tracemalloc.start()
        self._observe_memory()

    def _observe_memory(self) -> None:
        snapshot = process_memory_snapshot_mb()
        if isinstance(snapshot["rss_mb"], float):
            self._rss_samples_mb.append(snapshot["rss_mb"])

    def record(
        self,
        task_type: str,
        timing: Mapping[str, Any],
        *,
        system_latency_ms: float,
        input_profile: Mapping[str, Any] | None = None,
    ) -> None:
        """记录一个样本；system_latency_ms 包括任何启用的辅助模型路径。"""

        if not self._started:
            raise RuntimeError("Performance monitor must be started before recording")
        task = str(task_type)
        latency = float(system_latency_ms)
        self._latencies_ms.append(latency)
        self._by_task[task]["system_end_to_end_latency_ms"].append(latency)
        for key, values, task_key in (
            ("ttft_ms", self._ttft_ms, "ttft_ms"),
            (
                "generation_tokens_per_second",
                self._generation_tokens_per_second,
                "generation_tokens_per_second",
            ),
            (
                "decode_tokens_per_second",
                self._decode_tokens_per_second,
                "decode_tokens_per_second",
            ),
        ):
            value = timing.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
                self._by_task[task][task_key].append(float(value))
        tokens = timing.get("output_token_count")
        if isinstance(tokens, int):
            self._output_tokens.append(tokens)
            self._by_task[task]["output_token_count"].append(float(tokens))
        if input_profile is not None:
            image_count = input_profile.get("image_count")
            if isinstance(image_count, int):
                self._image_counts.append(image_count)
            visual_tokens = input_profile.get("visual_token_count")
            if isinstance(visual_tokens, int):
                self._visual_token_counts.append(visual_tokens)
            status = input_profile.get("visual_token_count_status")
            if isinstance(status, str):
                self._visual_token_statuses[status] += 1
        self._observe_memory()

    def finish(
        self,
        *,
        requested_samples: int,
        completed_samples: int,
        failed_samples: int,
        warmup_samples: int,
        startup_and_model_load_ms: float,
        model_load_ms: float,
        config: Mapping[str, Any],
        environment: Mapping[str, Any],
        model_resources: Mapping[str, Any],
        batch_size: int,
        repeats: int,
    ) -> dict[str, Any]:
        """生成可独立归档的性能报告。"""

        if not self._started:
            raise RuntimeError("Performance monitor has not started")
        current_python_bytes, peak_python_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        final_memory = process_memory_snapshot_mb()
        if isinstance(final_memory["rss_mb"], float):
            self._rss_samples_mb.append(final_memory["rss_mb"])
        gpu: dict[str, float | None] = {
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
        }
        if self._enabled_cuda:
            try:
                mib = 1024 * 1024
                gpu = {
                    "peak_allocated_mb": float(
                        self._torch.cuda.max_memory_allocated(self._device)
                    )
                    / mib,
                    "peak_reserved_mb": float(self._torch.cuda.max_memory_reserved(self._device))
                    / mib,
                }
            except (AttributeError, RuntimeError, TypeError):
                pass
        by_task = {
            task: {
                "system_end_to_end_latency_ms": _number_statistics(
                    values["system_end_to_end_latency_ms"]
                ),
                "ttft_ms": _number_statistics(values["ttft_ms"]),
                "generation_tokens_per_second": _number_statistics(
                    values["generation_tokens_per_second"]
                ),
                "decode_tokens_per_second": _number_statistics(
                    values["decode_tokens_per_second"]
                ),
                "output_token_count": _number_statistics(values["output_token_count"]),
            }
            for task, values in sorted(self._by_task.items())
        }
        return {
            "schema_version": PERFORMANCE_REPORT_VERSION,
            "measurement_scope": {
                "latency": "single_sample_end_to_end",
                "latency_start": "before image preprocessing/collation",
                "latency_end": "after generated text decoding and CUDA synchronization",
                "ttft": "from the same start point to the first generated token callback",
                "decode_speed": "(generated_tokens - 1) / (generation_end - first_token)",
                "warmup_excluded": True,
                "peak_gpu_memory": "CUDA max allocated/reserved after warmup reset",
                "cpu_memory": "current process working-set samples; OS peak is process lifetime",
            },
            "run": {
                "requested_samples": requested_samples,
                "completed_samples": completed_samples,
                "failed_samples": failed_samples,
                "warmup_samples": warmup_samples,
                "batch_size": batch_size,
                "repeats": repeats,
                "startup_and_model_load_ms": startup_and_model_load_ms,
                "model_load_ms": model_load_ms,
            },
            "latency_ms": _number_statistics(self._latencies_ms),
            "ttft_ms": _number_statistics(self._ttft_ms),
            "generation_tokens_per_second": _number_statistics(
                self._generation_tokens_per_second
            ),
            "decode_tokens_per_second": _number_statistics(self._decode_tokens_per_second),
            "output_token_count": _number_statistics(float(value) for value in self._output_tokens),
            "input_profile": {
                "image_count": _number_statistics(float(value) for value in self._image_counts),
                "visual_token_count": _number_statistics(
                    float(value) for value in self._visual_token_counts
                ),
                "visual_token_count_statuses": dict(sorted(self._visual_token_statuses.items())),
            },
            "memory_mb": {
                "inference_peak_process_rss_mb": max(self._rss_samples_mb, default=None),
                "os_process_peak_rss_mb": final_memory["os_peak_rss_mb"],
                "python_tracemalloc_current_mb": current_python_bytes / (1024 * 1024),
                "python_tracemalloc_peak_mb": peak_python_bytes / (1024 * 1024),
                "gpu": gpu,
            },
            "by_task": by_task,
            "environment": dict(environment),
            "model_resources": dict(model_resources),
            "config": dict(config),
        }
