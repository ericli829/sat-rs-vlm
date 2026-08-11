"""从层敏感度报告生成可复现的 bitsandbytes 混合 INT8 配置。"""

from __future__ import annotations

from typing import Any


def build_mixed_precision_config(
    base_config: dict[str, Any],
    sensitivity_report: dict[str, Any],
    *,
    keep_top_groups: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """保留高敏感组和共享权重层的原精度，量化其余 Linear 层。"""

    sensitive_groups = {str(name) for name in sensitivity_report.get("sensitive_groups", [])}
    if keep_top_groups < 0:
        raise ValueError("keep_top_groups must be non-negative")

    results = sensitivity_report.get("results")
    if not isinstance(results, list):
        raise ValueError("Sensitivity report results must be a list")

    if keep_top_groups:
        ranked_results = sorted(
            (result for result in results if isinstance(result, dict)),
            key=lambda result: float(result.get("sensitivity_score", 0.0)),
            reverse=True,
        )
        sensitive_groups.update(
            str(result["name"])
            for result in ranked_results[:keep_top_groups]
            if str(result.get("name", "")).strip()
        )
    if not sensitive_groups:
        raise ValueError("Sensitivity report contains no groups to preserve in a mixed model")

    skipped_modules: set[str] = set()
    matched_groups: list[str] = []
    for result in results:
        if not isinstance(result, dict) or str(result.get("name")) not in sensitive_groups:
            continue
        module_names = result.get("module_names")
        if not isinstance(module_names, list):
            raise ValueError(f"Sensitive group {result.get('name')!r} has no module_names list")
        skipped_modules.update(str(name) for name in module_names if str(name).strip())
        matched_groups.append(str(result["name"]))

    missing_groups = sorted(sensitive_groups.difference(matched_groups))
    if missing_groups:
        raise ValueError(f"Sensitive groups missing from results: {missing_groups}")

    grouping = sensitivity_report.get("grouping", {})
    if isinstance(grouping, dict):
        tied_modules = grouping.get("automatically_skipped_tied_linear_modules", [])
        if isinstance(tied_modules, list):
            skipped_modules.update(str(name) for name in tied_modules if str(name).strip())

    if not skipped_modules:
        raise ValueError("Mixed model resolved no modules to preserve in original precision")

    payload = dict(base_config)
    quantization = dict(payload.get("quantization", {}))
    quantization.update(
        {
            "backend": "bnb_int8",
            "device": "cuda",
            "llm_int8_skip_modules": sorted(skipped_modules),
        }
    )
    payload["quantization"] = quantization
    summary = {
        "sensitive_groups": sorted(sensitive_groups),
        "keep_top_groups": keep_top_groups,
        "preserved_module_count": len(skipped_modules),
        "preserved_modules": sorted(skipped_modules),
    }
    return payload, summary
