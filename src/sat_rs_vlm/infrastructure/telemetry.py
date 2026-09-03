"""Shared full-system timing, resource sampling, and runtime provenance."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

_MIB = 1024 * 1024


class GenerationTelemetry:
    """Record generation boundaries exposed by Transformers generation hooks."""

    def __init__(self) -> None:
        self.started_at: float | None = None
        self.first_token_at: float | None = None
        self.finished_at: float | None = None
        self.decode_started_at: float | None = None
        self.decode_finished_at: float | None = None
        self.preprocess_started_at: float | None = None
        self.preprocess_finished_at: float | None = None
        self.generation_started_at: float | None = None
        self.generation_finished_at: float | None = None
        self.generated_tokens: int | None = None
        self.output_token_counts: list[int | None] = []
        self.vision_input: dict[str, Any] = {}

    def start(self) -> None:
        self.started_at = time.perf_counter()

    def start_preprocess(self) -> None:
        self.preprocess_started_at = time.perf_counter()

    def finish_preprocess(self) -> None:
        self.preprocess_finished_at = time.perf_counter()

    def start_generation(self) -> None:
        self.generation_started_at = time.perf_counter()

    def finish_generation(self, generated_tokens: int | None) -> None:
        self.generation_finished_at = time.perf_counter()
        self.finish(generated_tokens)

    def mark_first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def finish(self, generated_tokens: int | None) -> None:
        self.finished_at = time.perf_counter()
        self.generated_tokens = generated_tokens

    def start_decode(self) -> None:
        self.decode_started_at = time.perf_counter()

    def finish_decode(self) -> None:
        self.decode_finished_at = time.perf_counter()

    def to_dict(self) -> dict[str, Any]:
        def elapsed(start: float | None, end: float | None) -> float | None:
            if start is None or end is None:
                return None
            return (end - start) * 1000.0

        ttft_ms = elapsed(self.started_at, self.first_token_at)
        decode_generation_ms = elapsed(self.first_token_at, self.finished_at)
        generated = self.generated_tokens
        return {
            "timing_ms": {
                "preprocess": elapsed(self.preprocess_started_at, self.preprocess_finished_at),
                "model_generate": elapsed(
                    self.generation_started_at, self.generation_finished_at
                ),
                "ttft": ttft_ms,
                "decode_generation": decode_generation_ms,
                "decode": elapsed(self.decode_started_at, self.decode_finished_at),
                "e2e": elapsed(self.started_at, self.decode_finished_at or self.finished_at),
            },
            "tokens": {
                "generated": generated,
                "output": self.output_token_counts,
                "decode_tokens_per_second": (
                    generated / (decode_generation_ms / 1000.0)
                    if generated is not None and decode_generation_ms and decode_generation_ms > 0
                    else None
                ),
                "decode_tokens_per_second_status": (
                    "ok" if generated is not None and decode_generation_ms else "unavailable"
                ),
            },
            "vision_input": dict(self.vision_input),
        }


class _FirstTokenLogitsProcessor:
    """Transformers-compatible callback used only for timing observation."""

    def __init__(self, telemetry: GenerationTelemetry) -> None:
        self.telemetry = telemetry

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        del input_ids
        self.telemetry.mark_first_token()
        return scores


def first_token_logits_processor(telemetry: GenerationTelemetry) -> Any | None:
    """Return a Transformers processor list, or None if Transformers is unavailable."""

    try:
        from transformers.generation.logits_process import (
            LogitsProcessorList,
        )
    except ImportError:
        return None
    return LogitsProcessorList([_FirstTokenLogitsProcessor(telemetry)])


def visual_input_telemetry(
    image_groups: list[list[Any]],
    image_grid_thw: Any = None,
    *,
    patch_size: int = 16,
    merge_size: int = 2,
    resize_policy: str = "processor_image_grid",
) -> dict[str, Any]:
    """Summarize original and processor image geometry without image tensors."""

    grid_value = image_grid_thw
    if hasattr(grid_value, "detach"):
        grid_value = grid_value.detach().cpu()
    if hasattr(grid_value, "tolist"):
        grid_value = grid_value.tolist()
    if grid_value is None:
        grids: list[list[int]] = []
    elif isinstance(grid_value, list) and grid_value and isinstance(grid_value[0], int | float):
        grids = [[int(item) for item in grid_value]]
    else:
        grids = [
            [int(item) for item in row]
            for row in list(grid_value or [])
            if isinstance(row, list | tuple) and len(row) >= 3
        ]

    original_sizes: list[list[list[int]]] = []
    processed_sizes: list[list[list[int] | None]] = []
    visual_tokens = 0
    grid_index = 0
    for group in image_groups:
        group_original: list[list[int]] = []
        group_processed: list[list[int] | None] = []
        for image in group:
            size = getattr(image, "size", None)
            if isinstance(size, tuple) and len(size) >= 2:
                group_original.append([int(size[0]), int(size[1])])
            else:
                group_original.append([])
            grid = grids[grid_index] if grid_index < len(grids) else None
            grid_index += 1
            if grid is None:
                group_processed.append(None)
                continue
            temporal, height, width = grid[:3]
            group_processed.append([width * patch_size, height * patch_size])
            visual_tokens += (temporal * height * width) // max(1, merge_size**2)
        original_sizes.append(group_original)
        processed_sizes.append(group_processed)
    image_count = sum(len(group) for group in image_groups)
    return {
        "original_size": original_sizes,
        "processed_size": processed_sizes,
        "resize_policy": resize_policy,
        "tile_count": 0,
        "crop_count": 0,
        "image_count": image_count,
        "image_grid_thw": grids,
        "visual_token_count": visual_tokens or None,
        "visual_token_count_method": "image_grid_thw_divided_by_merge_area",
    }


def _optional_psutil() -> Any | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def process_rss_bytes() -> int | None:
    """Return current process RSS without making psutil a hard dependency."""

    psutil = _optional_psutil()
    if psutil is not None:
        return int(psutil.Process().memory_info().rss)
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("page_fault_count", wintypes.DWORD),
                    ("peak_working_set_size", ctypes.c_size_t),
                    ("working_set_size", ctypes.c_size_t),
                    ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                    ("quota_paged_pool_usage", ctypes.c_size_t),
                    ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                    ("quota_non_paged_pool_usage", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            return int(counters.working_set_size) if ok else None
        except (AttributeError, OSError):
            return None
    try:
        statm = Path("/proc/self/statm")
        if statm.is_file():
            resident_pages = int(statm.read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, ValueError):
        return None
    return None


def _total_memory_bytes() -> int | None:
    psutil = _optional_psutil()
    if psutil is not None:
        return int(psutil.virtual_memory().total)
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("avail_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.length = ctypes.sizeof(status)
            return int(status.total_phys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)
            ) else None
        except (AttributeError, OSError):
            return None
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return None


def _mb(value: int | None) -> float | None:
    return round(value / _MIB, 3) if value is not None else None


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON semantics rather than platform-specific whitespace or ordering."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def collect_prompt_provenance(
    *,
    dataset: str,
    task_type: str,
    question: str,
    options: list[str] | tuple[str, ...],
    metadata: dict[str, Any] | None = None,
    graph: Any = None,
) -> dict[str, Any]:
    """Return a stable identity for the prompt material sent to a runtime."""

    source_metadata = metadata or {}
    profile = str(source_metadata.get("prompt_profile") or "taskgraph_runtime_v1")
    version = str(source_metadata.get("prompt_version") or "taskgraph_runtime_v1")
    payload = {
        "dataset": dataset,
        "task_type": task_type,
        "prompt_profile": profile,
        "prompt_version": version,
        "question": question,
        "options": list(options),
        "graph": graph,
    }
    return {
        "profile": profile,
        "version": version,
        "sha256": canonical_json_sha256(payload),
    }


class SystemTelemetry:
    """Measure one declared execution scope using explicit, reproducible semantics."""

    def __init__(
        self,
        scope: str,
        *,
        torch_module: Any | None = None,
        reset_cuda_peaks: bool = False,
        cpu_sample_interval_s: float = 0.05,
    ) -> None:
        self.scope = scope
        self.torch = torch_module if torch_module is not None else sys.modules.get("torch")
        self.reset_cuda_peaks = reset_cuda_peaks
        self.cpu_sample_interval_s = max(0.0, float(cpu_sample_interval_s))
        self.started_at_utc: str | None = None
        self.ended_at_utc: str | None = None
        self.elapsed_ms: float | None = None
        self.success: bool | None = None
        self.error_type: str | None = None
        self.cpu_rss_start_bytes: int | None = None
        self.cpu_rss_end_bytes: int | None = None
        self.peak_cpu_rss_bytes: int | None = None
        self.peak_gpu_allocated_bytes: int | None = None
        self.peak_gpu_reserved_bytes: int | None = None
        self._started_perf: float | None = None
        self._cuda_available = False
        self._cuda_peaks_reset = False
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None

    def _sample_cpu(self) -> None:
        rss = process_rss_bytes()
        if rss is not None and (
            self.peak_cpu_rss_bytes is None or rss > self.peak_cpu_rss_bytes
        ):
            self.peak_cpu_rss_bytes = rss

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.cpu_sample_interval_s):
            self._sample_cpu()

    def _synchronize_cuda(self) -> None:
        synchronize = getattr(getattr(self.torch, "cuda", None), "synchronize", None)
        if self._cuda_available and callable(synchronize):
            synchronize()

    def _refresh_cuda_state(self) -> None:
        if self.torch is None:
            self.torch = sys.modules.get("torch")
        cuda = getattr(self.torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        self._cuda_available = bool(callable(is_available) and is_available())

    def __enter__(self) -> SystemTelemetry:
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self.cpu_rss_start_bytes = process_rss_bytes()
        self.peak_cpu_rss_bytes = self.cpu_rss_start_bytes
        self._refresh_cuda_state()
        cuda = getattr(self.torch, "cuda", None)
        self._synchronize_cuda()
        if self._cuda_available and self.reset_cuda_peaks:
            reset = getattr(cuda, "reset_peak_memory_stats", None)
            if callable(reset):
                reset()
                self._cuda_peaks_reset = True
        self._started_perf = time.perf_counter()
        if self.cpu_sample_interval_s > 0:
            self._sampler = threading.Thread(
                target=self._sample_until_stopped,
                name=f"{self.scope}-rss-sampler",
                daemon=True,
            )
            self._sampler.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc, traceback
        self._refresh_cuda_state()
        self._synchronize_cuda()
        if self._started_perf is not None:
            self.elapsed_ms = (time.perf_counter() - self._started_perf) * 1000.0
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=max(0.1, self.cpu_sample_interval_s * 2))
        self.cpu_rss_end_bytes = process_rss_bytes()
        self._sample_cpu()
        cuda = getattr(self.torch, "cuda", None)
        if self._cuda_available:
            allocated = getattr(cuda, "max_memory_allocated", None)
            reserved = getattr(cuda, "max_memory_reserved", None)
            if callable(allocated):
                self.peak_gpu_allocated_bytes = int(allocated())
            if callable(reserved):
                self.peak_gpu_reserved_bytes = int(reserved())
        self.success = exc_type is None
        self.error_type = exc_type.__name__ if exc_type is not None else None
        self.ended_at_utc = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "scope": self.scope,
            "success": self.success,
            "error_type": self.error_type,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "timing_ms": {"e2e": self.elapsed_ms},
            "resources": {
                "cpu_rss_start_mb": _mb(self.cpu_rss_start_bytes),
                "cpu_rss_end_mb": _mb(self.cpu_rss_end_bytes),
                "peak_cpu_rss_mb": _mb(self.peak_cpu_rss_bytes),
                "peak_gpu_allocated_mb": _mb(self.peak_gpu_allocated_bytes),
                "peak_gpu_reserved_mb": _mb(self.peak_gpu_reserved_bytes),
            },
            "measurement": {
                "clock": "time.perf_counter",
                "cpu_rss": "process RSS sampled during scope",
                "cpu_sample_interval_ms": self.cpu_sample_interval_s * 1000.0,
                "cuda_synchronized": self._cuda_available,
                "cuda_peak_stats_reset": self._cuda_peaks_reset,
            },
        }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_driver_version() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    versions = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    return ",".join(versions) or None


def collect_runtime_environment(torch_module: Any | None = None) -> dict[str, Any]:
    """Collect the software and hardware fields required for reproducible reports."""

    torch = torch_module if torch_module is not None else sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    cuda_available = bool(callable(is_available) and is_available())
    gpu_count = int(cuda.device_count()) if cuda_available and cuda is not None else 0
    gpus: list[dict[str, Any]] = []
    if cuda is None:
        cuda_available = False
    else:
        for index in range(gpu_count):
            properties = cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": str(cuda.get_device_name(index)),
                    "total_memory_mb": _mb(int(properties.total_memory)),
                }
            )
    torch_version = str(getattr(torch, "__version__", "")) or _package_version("torch")
    torch_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    total_memory = _total_memory_bytes()
    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {
            "model": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or None,
            "logical_cores": os.cpu_count(),
            "memory_total_mb": _mb(total_memory),
        },
        "gpu": {
            "cuda_available": cuda_available,
            "count": gpu_count,
            "devices": gpus,
            "driver_version": _nvidia_driver_version() if cuda_available else None,
            "cuda_runtime": str(torch_cuda) if torch_cuda is not None else None,
        },
        "software": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "torch": torch_version,
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "peft": _package_version("peft"),
        },
    }


def collect_model_inventory(model: Any, model_paths: list[str | Path]) -> dict[str, Any]:
    """Summarize loaded parameters and actual local model file storage."""

    parameter_count: int | None = None
    parameter_bytes: int | None = None
    dtype_parameters: dict[str, int] = {}
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        parameter_count = 0
        parameter_bytes = 0
        for parameter in parameters():
            count = int(parameter.numel())
            parameter_count += count
            element_size = getattr(parameter, "element_size", None)
            if callable(element_size):
                parameter_bytes += count * int(element_size())
            dtype = str(getattr(parameter, "dtype", "unknown")).removeprefix("torch.")
            dtype_parameters[dtype] = dtype_parameters.get(dtype, 0) + count

    roots: list[dict[str, Any]] = []
    seen_files: set[Path] = set()
    storage_bytes = 0
    for raw_path in model_paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            roots.append({"path": str(path), "available": False, "bytes": None, "files": 0})
            continue
        files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
        root_bytes = 0
        root_files = 0
        for file in files:
            resolved = file.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            size = file.stat().st_size
            root_bytes += size
            storage_bytes += size
            root_files += 1
        roots.append(
            {
                "path": str(path.resolve()),
                "available": True,
                "bytes": root_bytes,
                "files": root_files,
            }
        )
    storage_complete = bool(roots) and all(root.get("available") is True for root in roots)
    return {
        "parameter_count": parameter_count,
        "loaded_parameter_bytes": parameter_bytes,
        "parameters_by_dtype": dict(sorted(dtype_parameters.items())),
        "local_model_storage_bytes": storage_bytes if storage_complete else None,
        "storage_roots": roots,
    }


_PROVIDER_CHILD_ATTRIBUTES = (
    "_provider",
    "provider",
    "base_provider",
    "_locator",
    "detector_provider",
    "retriever_provider",
    "_delegate",
)


def _provider_children(provider: Any) -> list[Any]:
    children: list[Any] = []
    for name in _PROVIDER_CHILD_ATTRIBUTES:
        child = getattr(provider, name, None)
        if child is None or isinstance(child, str | bytes | int | float | bool):
            continue
        if child is provider or not hasattr(child, "provider_name"):
            continue
        children.append(child)
    return children


def _provider_model(provider: Any) -> Any | None:
    model = getattr(provider, "_model", None)
    if model is not None:
        return model
    engine = getattr(provider, "_engine", None)
    return getattr(engine, "_model", None) if engine is not None else None


def _provider_model_paths(provider: Any) -> list[str | Path]:
    paths: list[str | Path] = []
    for name in ("checkpoint", "model_path"):
        value = getattr(provider, name, None)
        if value:
            paths.append(value)
    config = getattr(provider, "model_config", None)
    model_id = getattr(config, "model_id", None) if config is not None else None
    if model_id and Path(str(model_id)).expanduser().exists():
        paths.append(str(model_id))
    extra = getattr(provider, "telemetry_model_paths", None)
    if callable(extra):
        paths.extend(extra())
    return list(dict.fromkeys(str(Path(item).expanduser()) for item in paths))


def _walk_provider_leaves(providers: list[Any]) -> tuple[list[Any], list[Any]]:
    leaves: list[Any] = []
    non_models: list[Any] = []
    seen: set[int] = set()

    def visit(provider: Any) -> None:
        if provider is None or id(provider) in seen:
            return
        seen.add(id(provider))
        children = _provider_children(provider)
        for child in children:
            visit(child)
        model = _provider_model(provider)
        paths = _provider_model_paths(provider)
        declared = getattr(provider, "telemetry_parameter_count", None)
        if (
            model is not None
            or paths
            or declared is not None
            or getattr(provider, "model_config", None)
        ):
            leaves.append(provider)
        elif not children:
            non_models.append(provider)

    for provider in providers:
        visit(provider)
    return leaves, non_models


def preload_provider_models(providers: list[Any]) -> dict[str, Any]:
    """Explicitly initialize lazy model leaves and report isolated load time."""

    leaves, _ = _walk_provider_leaves(providers)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for provider in leaves:
        preload = getattr(provider, "preload", None)
        provider_started = time.perf_counter()
        status = "not_supported"
        error: str | None = None
        if callable(preload):
            try:
                preload()
                status = "loaded"
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "provider": str(getattr(provider, "provider_name", type(provider).__name__)),
                "role": str(getattr(provider, "role", "")) or None,
                "status": status,
                "latency_ms": (time.perf_counter() - provider_started) * 1000.0,
                "error": error,
            }
        )
        if error is not None:
            raise RuntimeError(f"provider preload failed for {records[-1]['provider']}: {error}")
    return {
        "scope": "all_configured_model_providers",
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "providers": records,
        "all_supported": bool(records) and all(item["status"] == "loaded" for item in records),
    }


def collect_provider_inventory(providers: list[Any]) -> dict[str, Any]:
    """Describe runtime providers without forcing lazy model initialization.

    Loaded providers are counted from their actual model parameters. Lazy or
    non-model providers remain explicit in the manifest with ``declared_only``
    or ``not_a_model`` status instead of receiving guessed parameter counts.
    """

    models: list[dict[str, Any]] = []
    total_parameters = 0
    total_storage = 0
    counted_parameters = True
    counted_storage = True
    counted_storage_roots: set[Path] = set()
    leaves, non_models = _walk_provider_leaves(providers)
    for provider in leaves:
        provider_name = str(getattr(provider, "provider_name", type(provider).__name__))
        config = getattr(provider, "model_config", None)
        model_id = str(
            getattr(config, "model_id", "")
            or getattr(provider, "model_id", "")
            or ""
        )
        model = _provider_model(provider)
        role = str(getattr(provider, "role", "")) or None
        inventory = collect_model_inventory(model, _provider_model_paths(provider))
        declared_parameter_count = getattr(provider, "telemetry_parameter_count", None)
        if inventory["parameter_count"] is None and declared_parameter_count is not None:
            inventory["parameter_count"] = int(declared_parameter_count)
        model_loaded = model is not None or bool(getattr(provider, "telemetry_model_loaded", False))
        parameter_count = inventory["parameter_count"]
        storage_bytes = inventory["local_model_storage_bytes"]
        if parameter_count is None:
            counted_parameters = False
        else:
            total_parameters += int(parameter_count)
        if storage_bytes is None:
            counted_storage = False
        else:
            for root in inventory["storage_roots"]:
                if not root.get("available") or root.get("bytes") is None:
                    continue
                root_path = Path(str(root["path"])).resolve()
                if root_path in counted_storage_roots:
                    continue
                counted_storage_roots.add(root_path)
                total_storage += int(root["bytes"])
        models.append(
            {
                "provider": provider_name,
                "identity": f"{provider_name}:{role}" if role else provider_name,
                "role": role,
                "model_id": model_id or None,
                "status": "loaded" if model_loaded else "declared_only",
                "model_load_ms": getattr(provider, "telemetry_model_load_ms", None),
                **inventory,
            }
        )
    for provider in non_models:
        counted_parameters = False
        counted_storage = False
        models.append(
            {
                "provider": str(getattr(provider, "provider_name", type(provider).__name__)),
                "identity": str(getattr(provider, "provider_name", type(provider).__name__)),
                "role": None,
                "model_id": None,
                "status": "not_a_model",
                "parameter_count": None,
                "local_model_storage_bytes": None,
            }
        )
    return {
        "models": models,
        "total_parameter_count": total_parameters if counted_parameters else None,
        "known_parameter_count": total_parameters,
        "total_model_storage_bytes": total_storage if counted_storage else None,
        "known_model_storage_bytes": total_storage,
        "parameter_accounting_status": "complete" if counted_parameters else "partial",
        "storage_accounting_status": "complete" if counted_storage else "partial",
    }


def collect_repository_provenance(project_root: str | Path) -> dict[str, Any]:
    """Return commit and dirty state without modifying repository metadata."""

    root = Path(project_root).resolve()
    git = shutil.which("git")
    if git is None:
        return {"root": str(root), "commit": None, "dirty": None}

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                [git, "-C", str(root), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain")
    return {
        "root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }
