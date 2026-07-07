"""轻量级推理性能监控模块。

算法/流程：
    使用上下文管理器记录进入和退出时间，计算 latency_ms。torch 通过动态导入
    探测；如果不存在，profiler 仍返回 CPU/无 CUDA 的基础统计。
"""

from __future__ import annotations

import importlib
import importlib.util
import time
from types import TracebackType
from typing import Any


class InferenceProfiler:
    """推理性能上下文管理器。

    参数：
        backend：当前模型后端名称，例如 mock 或 huggingface。
        device：配置中的运行设备，例如 auto/cpu/cuda。

    返回值：
        进入上下文时返回自身；退出后可调用 to_dict() 获取统计结果。
    """

    def __init__(self, backend: str, device: str) -> None:
        """初始化 profiler 状态。

        参数：
            backend：模型后端名称。
            device：运行设备名称。
        """

        self.backend = backend
        self.device = device
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.latency_ms: float | None = None
        self.cuda_available = False
        self.peak_memory_mb: float | None = None
        self._perf_start: float | None = None
        self._torch: Any | None = None

    def __enter__(self) -> InferenceProfiler:
        """开始计时并探测 CUDA 状态。

        返回值：
            InferenceProfiler：当前上下文对象。
        """

        self.start_time = time.time()
        self._perf_start = time.perf_counter()
        self._torch = self._load_torch()
        if self._torch is not None:
            self.cuda_available = bool(self._torch.cuda.is_available())
            if self.cuda_available:
                self._torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """结束计时并记录峰值显存。

        参数：
            exc_type：上下文内部异常类型，未使用。
            exc：上下文内部异常对象，未使用。
            traceback：异常栈，未使用。

        返回值：
            None。
        """

        del exc_type, exc, traceback
        self.end_time = time.time()
        if self._perf_start is not None:
            self.latency_ms = (time.perf_counter() - self._perf_start) * 1000
        if self._torch is not None and self.cuda_available:
            self.peak_memory_mb = float(self._torch.cuda.max_memory_allocated() / (1024 * 1024))

    def to_dict(self) -> dict[str, Any]:
        """导出性能统计。

        返回值：
            dict[str, Any]：包含 start_time、end_time、latency_ms、backend、
            device、cuda_available、peak_memory_mb。
        """

        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "latency_ms": self.latency_ms,
            "backend": self.backend,
            "device": self.device,
            "cuda_available": self.cuda_available,
            "peak_memory_mb": self.peak_memory_mb,
        }

    @staticmethod
    def _load_torch() -> Any | None:
        """动态加载 torch。

        返回值：
            Any | None：torch 模块对象；未安装时返回 None。
        """

        if importlib.util.find_spec("torch") is None:
            return None
        return importlib.import_module("torch")
