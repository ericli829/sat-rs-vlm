"""已弃用的 benchmark 导入兼容层。"""

from sat_rs_vlm.quantization.benchmark import (
    assert_comparable_sample_ids,
    planned_variants,
    run_benchmark,
    run_variant_evaluation,
    validate_assets,
)

__all__ = [
    "assert_comparable_sample_ids",
    "planned_variants",
    "run_benchmark",
    "run_variant_evaluation",
    "validate_assets",
]
