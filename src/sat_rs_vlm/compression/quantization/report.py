"""已弃用的量化报告工具导入兼容层。"""

from sat_rs_vlm.quantization.report import (
    comparison_summary,
    environment_metadata,
    latency_statistics,
)

__all__ = ["comparison_summary", "environment_metadata", "latency_statistics"]
