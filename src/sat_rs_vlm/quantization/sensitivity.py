"""Dynamic INT8 层级与组件级敏感度分析算法。

模块只处理三件事：扫描 Linear 层、按组件或固定大小分组、根据 Evaluation v1.5 的
主任务指标计算退化分数。真实模型加载和 predictions 生成由脚本与 benchmark 模块负责，
因此本模块可用小型 fake model 在 CPU 单元测试中验证。
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class SensitivityGroup:
    """一次局部量化实验对应的模块集合。"""

    name: str
    kind: Literal["component", "layer_group"]
    module_names: tuple[str, ...]
    parameter_count: int


@dataclass(frozen=True)
class SensitivityResult:
    """一个分组相对 FP32 baseline 的统一指标变化。"""

    name: str
    kind: str
    module_names: tuple[str, ...]
    parameter_count: int
    sensitivity_score: float
    metric_deltas: dict[str, dict[str, Any]]
    evaluation_dir: str
    task_scores: dict[str, float] = field(default_factory=dict)
    completed_samples: int = 0
    failed_samples: int = 0
    parameter_fraction: float | None = None
    latency_mean_ms: float | None = None
    speedup_vs_baseline: float | None = None


PREFERRED_METRICS: dict[str, bool] = {
    "continuous_mean_iou": True,
    "exact_count_accuracy": True,
    "normalized_accuracy": True,
    "micro_normalized_accuracy": True,
    "token_f1": True,
    "rouge_l_f1_approx": True,
    "chrf_approx": True,
    "change_event_f1": True,
    "changeflag_valid_rate": True,
    "binary_parse_success_rate": True,
    "binary_accuracy": True,
    "balanced_accuracy": True,
    "change_precision": True,
    "change_recall": True,
    "change_f1": True,
    "matthews_correlation_coefficient": True,
    "cohen_kappa": True,
    "false_positive_rate": False,
    "false_negative_rate": False,
}


PROTOCOL_TASKS = {
    "vrsbench_visual_grounding": "detection",
    "generic_single_target_grounding_internal": "detection",
    "vrsbench_counting": "counting",
    "generic_counting_internal": "counting",
    "vrsbench_open_vqa": "vqa",
    "generic_text_internal": "vqa",
    "vrsbench_detailed_caption": "captioning",
    "generic_captioning_internal": "captioning",
    "levir_cc_change_caption": "change_detection",
    "generic_change_captioning_internal": "change_detection",
}


def classify_component(module_name: str) -> str:
    """根据 Qwen3-VL 常见命名候选判断视觉、投影器、语言模型或其他组件。"""

    lowered = module_name.lower()
    if any(
        token in lowered
        for token in ("projector", "connector", "mm_projector", "multi_modal", "merger")
    ):
        return "multimodal_projector"
    if any(token in lowered for token in ("visual", "vision", "patch_embed")):
        return "vision_encoder"
    if any(
        token in lowered for token in ("language_model", "model.layers", "lm_head", "embed_tokens")
    ):
        return "language_model"
    return "other"


def discover_linear_modules(model: Any, torch: Any) -> dict[str, Any]:
    """返回模型中全部 ``torch.nn.Linear``，键为可用于模块替换的完整名称。"""

    return {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, torch.nn.Linear)
    }


def discover_tied_linear_modules(model: Any, torch: Any) -> tuple[str, ...]:
    """返回权重与其他参数共享存储的 Linear 层。

    bitsandbytes 的 ``Linear8bitLt`` 不能安全地替换这类层。Qwen3-VL 的 ``lm_head``
    通常与词嵌入共享权重，因此 GPU INT8 敏感度实验应始终保留它的原始精度。
    """

    try:
        named_parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_parameters = model.named_parameters()

    parameter_names: dict[tuple[str, int, int], list[str]] = {}
    for name, parameter in named_parameters:
        if not isinstance(parameter, torch.Tensor):
            continue
        identity = (str(parameter.device), int(parameter.data_ptr()), int(parameter.numel()))
        parameter_names.setdefault(identity, []).append(name)

    tied: list[str] = []
    for name, module in discover_linear_modules(model, torch).items():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        identity = (str(weight.device), int(weight.data_ptr()), int(weight.numel()))
        if len(parameter_names.get(identity, ())) > 1:
            tied.append(name)
    return tuple(sorted(tied))


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def _transformer_block_key(module_name: str) -> str:
    match = re.match(r"^(.*?(?:layers|blocks)\.\d+)(?:\.|$)", module_name)
    if match:
        return match.group(1)
    return module_name.rsplit(".", 1)[0] if "." in module_name else module_name


def build_sensitivity_groups(
    model: Any,
    torch: Any,
    *,
    method: Literal["component_wise", "layer_wise"],
    layer_group_size: int = 6,
    layer_grouping: Literal["fixed_linear", "transformer_block"] = "fixed_linear",
    include_modules: tuple[str, ...] = (),
    skip_modules: tuple[str, ...] = ("visual",),
    max_groups: int | None = None,
) -> list[SensitivityGroup]:
    """扫描并分组 Linear 层。

    ``skip_modules`` 可填写组件名，也可填写完整模块名的大小写不敏感片段；未匹配到任何
    层时立即失败，避免把空实验报告成成功。
    """

    if layer_group_size < 1:
        raise ValueError("layer_group_size must be positive")
    modules = discover_linear_modules(model, torch)
    include_tokens = tuple(token.lower() for token in include_modules)
    skip_tokens = tuple(token.lower() for token in skip_modules)
    selected = {
        name: module
        for name, module in modules.items()
        if (
            not include_tokens
            or any(
                token == classify_component(name) or token in name.lower()
                for token in include_tokens
            )
        )
        and not any(
            token == classify_component(name) or token in name.lower() for token in skip_tokens
        )
    }
    if not selected:
        raise ValueError("No Linear modules matched the sensitivity configuration")

    raw_groups: list[tuple[str, Literal["component", "layer_group"], list[str]]] = []
    if method == "component_wise":
        grouped: dict[str, list[str]] = {}
        for name in sorted(selected):
            grouped.setdefault(classify_component(name), []).append(name)
        raw_groups = [
            (component, "component", names) for component, names in sorted(grouped.items()) if names
        ]
    elif method == "layer_wise" and layer_grouping == "fixed_linear":
        names = sorted(selected, key=_natural_key)
        raw_groups = [
            (
                f"linear_group_{index // layer_group_size + 1:03d}",
                "layer_group",
                names[index : index + layer_group_size],
            )
            for index in range(0, len(names), layer_group_size)
        ]
    elif method == "layer_wise" and layer_grouping == "transformer_block":
        component_blocks: dict[str, dict[str, list[str]]] = {}
        for name in selected:
            component = classify_component(name)
            block = _transformer_block_key(name)
            component_blocks.setdefault(component, {}).setdefault(block, []).append(name)
        for component in sorted(component_blocks):
            blocks = component_blocks[component]
            block_names = sorted(blocks, key=_natural_key)
            for index in range(0, len(block_names), layer_group_size):
                chunk = block_names[index : index + layer_group_size]
                names = [
                    module_name
                    for block_name in chunk
                    for module_name in sorted(blocks[block_name], key=_natural_key)
                ]
                raw_groups.append(
                    (
                        f"{component}_blocks_{index // layer_group_size + 1:03d}",
                        "layer_group",
                        names,
                    )
                )
    else:
        raise ValueError(
            f"Unsupported sensitivity grouping: method={method}, layer_grouping={layer_grouping}"
        )

    if max_groups is not None:
        raw_groups = raw_groups[:max_groups]
    return [
        SensitivityGroup(
            name=name,
            kind=kind,
            module_names=tuple(names),
            parameter_count=sum(
                int(parameter.numel())
                for module_name in names
                for parameter in selected[module_name].parameters(recurse=False)
            ),
        )
        for name, kind, names in raw_groups
    ]


def quantize_named_linear_modules(
    model: Any,
    torch: Any,
    module_names: tuple[str, ...],
    *,
    inplace: bool = False,
) -> Any:
    """只对指定完整名称的 Linear 层执行 PyTorch dynamic qint8 量化。

    参数 ``inplace=False`` 时先深拷贝模型，适合小模型测试；真实 2B 模型应传入
    ``inplace=True`` 并在每个实验组重新加载基座，以降低峰值内存。
    """

    if not module_names:
        raise ValueError("module_names must not be empty")
    available = discover_linear_modules(model, torch)
    missing = sorted(set(module_names).difference(available))
    if missing:
        raise ValueError(f"Sensitivity modules do not exist: {missing}")
    target = model if inplace else copy.deepcopy(model)
    quantization = getattr(getattr(torch, "ao", None), "quantization", None)
    quantize_dynamic = getattr(quantization, "quantize_dynamic", None)
    if quantize_dynamic is None:
        quantize_dynamic = torch.quantization.quantize_dynamic
    quantized = quantize_dynamic(
        target,
        qconfig_spec=set(module_names),
        dtype=torch.qint8,
        inplace=True,
    )
    quantized_modules = dict(quantized.named_modules())
    not_quantized = [
        name
        for name in module_names
        if "quantized.dynamic" not in quantized_modules[name].__class__.__module__
    ]
    if not_quantized:
        raise RuntimeError(f"Requested modules were not dynamically quantized: {not_quantized}")
    return quantized


def _metric_values(payload: Any, prefix: tuple[str, ...] = ()) -> dict[str, float]:
    values: dict[str, float] = {}
    if isinstance(payload, dict):
        value = payload.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and prefix:
            values[".".join(prefix)] = float(value)
        for key, child in payload.items():
            if key != "value":
                values.update(_metric_values(child, (*prefix, str(key))))
    elif isinstance(payload, list):
        for index, child in enumerate(payload):
            values.update(_metric_values(child, (*prefix, str(index))))
    return values


def _metric_task(path: str) -> str:
    parts = path.split(".")
    if "by_task" in parts:
        index = parts.index("by_task")
        if index + 1 < len(parts):
            return parts[index + 1]
    if "by_protocol" in parts:
        index = parts.index("by_protocol")
        if index + 1 < len(parts):
            return PROTOCOL_TASKS.get(parts[index + 1], parts[index + 1])
    if "semantic" in parts:
        return "semantic"
    return "overall"


def calculate_sensitivity_breakdown(
    baseline_metrics: dict[str, Any],
    quantized_metrics: dict[str, Any],
    *,
    preferred_metrics: dict[str, bool] | None = None,
    task_weights: dict[str, float] | None = None,
    metric_weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, dict[str, Any]], dict[str, float]]:
    """Calculate metric deltas, equalized per-task scores, and their weighted total."""

    directions = preferred_metrics or PREFERRED_METRICS
    configured_task_weights = task_weights or {}
    configured_metric_weights = metric_weights or {}
    baseline_values = _metric_values(baseline_metrics)
    quantized_values = _metric_values(quantized_metrics)
    deltas: dict[str, dict[str, Any]] = {}
    by_task: dict[str, list[tuple[float, float]]] = {}
    comparable_paths = set(baseline_values).intersection(quantized_values)
    tasks_with_by_task_metrics = {
        _metric_task(path) for path in comparable_paths if "by_task" in path.split(".")
    }
    for path in sorted(comparable_paths):
        task = _metric_task(path)
        if "by_protocol" in path.split(".") and task in tasks_with_by_task_metrics:
            continue
        metric_name = path.rsplit(".", 1)[-1]
        if metric_name not in directions:
            continue
        baseline = baseline_values[path]
        candidate = quantized_values[path]
        higher_is_better = directions[metric_name]
        raw_delta = candidate - baseline
        harmful_delta = -raw_delta if higher_is_better else raw_delta
        normalized_degradation = max(0.0, harmful_delta) / max(abs(baseline), 1.0)
        metric_weight = float(configured_metric_weights.get(metric_name, 1.0))
        by_task.setdefault(task, []).append((normalized_degradation, metric_weight))
        deltas[path] = {
            "baseline": baseline,
            "quantized": candidate,
            "delta": raw_delta,
            "higher_is_better": higher_is_better,
            "normalized_degradation": normalized_degradation,
            "task": task,
            "metric_weight": metric_weight,
        }
    if not deltas:
        raise ValueError("No comparable Evaluation v1.5 primary metrics were found")
    task_scores = {
        task: sum(value * weight for value, weight in values)
        / sum(weight for _, weight in values)
        for task, values in by_task.items()
    }
    total_weight = sum(float(configured_task_weights.get(task, 1.0)) for task in task_scores)
    score = sum(
        value * float(configured_task_weights.get(task, 1.0))
        for task, value in task_scores.items()
    ) / total_weight
    return score, deltas, task_scores


def calculate_sensitivity(
    baseline_metrics: dict[str, Any],
    quantized_metrics: dict[str, Any],
    *,
    preferred_metrics: dict[str, bool] | None = None,
    task_weights: dict[str, float] | None = None,
    metric_weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, dict[str, Any]]]:
    """从同一 Evaluation v1.5 契约结果计算平均归一化退化分数。

    ``preferred_metrics`` 的布尔值表示指标是否越大越好。分数只使用任务主指标，
    不使用 Keyword Hit Rate；缺失值保持缺失并从分母中排除。
    """

    score, deltas, _ = calculate_sensitivity_breakdown(
        baseline_metrics,
        quantized_metrics,
        preferred_metrics=preferred_metrics,
        task_weights=task_weights,
        metric_weights=metric_weights,
    )
    return score, deltas


def validate_variant_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    require_same_samples: bool,
    max_failure_rate: float,
) -> None:
    """Reject partial or mismatched evaluations before attributing damage to a layer group."""

    for label, report in (("baseline", baseline), ("candidate", candidate)):
        requested = int(report.get("requested_samples", 0))
        failed = int(report.get("failed_samples", 0))
        failure_rate = failed / requested if requested else 1.0
        if failure_rate > max_failure_rate:
            raise RuntimeError(
                f"{label} failure rate {failure_rate:.4f} exceeds {max_failure_rate:.4f}"
            )
    if require_same_samples and baseline.get("sample_ids") != candidate.get("sample_ids"):
        raise RuntimeError("Baseline and sensitivity variant completed different sample IDs")


def build_sensitivity_report(
    *,
    model_source: str,
    method: str,
    baseline_evaluation_dir: str,
    results: list[SensitivityResult],
    sensitive_threshold: float = 0.05,
    insensitive_threshold: float = 0.01,
    baseline_performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成稳定 JSON 报告，并按敏感度从高到低给出混合精度建议。"""

    ranked = sorted(results, key=lambda item: item.sensitivity_score, reverse=True)
    sensitive = [
        item.name for item in ranked if item.sensitivity_score >= sensitive_threshold
    ]
    insensitive = [
        item.name for item in ranked if item.sensitivity_score < insensitive_threshold
    ]
    recommendations: list[str] = []
    if sensitive:
        recommendations.append(f"Keep high-sensitivity groups in FP16/FP32: {sensitive}")
    if insensitive:
        recommendations.append(f"INT8 candidates with low measured degradation: {insensitive}")
    if not recommendations:
        recommendations.append("Review per-task metric deltas before choosing mixed precision.")
    return {
        "schema_version": "1.0",
        "metric_contract": "evaluation-v1.5",
        "model_source": model_source,
        "method": method,
        "baseline_evaluation_dir": baseline_evaluation_dir,
        "baseline_performance": baseline_performance or {},
        "thresholds": {
            "sensitive": sensitive_threshold,
            "insensitive": insensitive_threshold,
        },
        "results": [asdict(item) for item in ranked],
        "sensitive_groups": sensitive,
        "recommendations": recommendations,
    }


def write_sensitivity_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """保存机器可读 JSON 与便于审阅的 Markdown 报告。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "sensitivity_report.json"
    markdown_path = destination / "sensitivity_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Quantization Sensitivity Report",
        "",
        f"- Model: `{report['model_source']}`",
        f"- Method: `{report['method']}`",
        f"- Metric contract: `{report['metric_contract']}`",
        "",
        "| Group | Kind | Parameters | Fraction | Sensitivity | Latency ms | Speedup | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        lines.append(
            f"| {result['name']} | {result['kind']} | {result['parameter_count']} | "
            f"{float(result.get('parameter_fraction') or 0.0):.4f} | "
            f"{result['sensitivity_score']:.6f} | "
            f"{float(result.get('latency_mean_ms') or 0.0):.2f} | "
            f"{float(result.get('speedup_vs_baseline') or 0.0):.3f} | "
            f"{result.get('failed_samples', 0)} |"
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in report["recommendations"])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def plot_sensitivity_report(report: dict[str, Any], output_dir: str | Path) -> list[Path]:
    """生成敏感度和参数量柱状图；matplotlib 仅在调用绘图时才是必需依赖。"""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Sensitivity plotting requires matplotlib") from exc

    results = list(report.get("results", []))
    if not results:
        raise ValueError("Sensitivity report contains no results")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    names = [str(item["name"]) for item in results]
    scores = [float(item["sensitivity_score"]) for item in results]
    parameters = [int(item["parameter_count"]) for item in results]

    generated: list[Path] = []
    for filename, values, ylabel in (
        ("sensitivity_scores.png", scores, "Normalized metric degradation"),
        ("quantized_parameter_counts.png", parameters, "Parameters"),
    ):
        figure, axis = plt.subplots(figsize=(max(8, len(names) * 0.7), 5))
        axis.bar(range(len(names)), values, color="#0072B2")
        axis.set_xticks(range(len(names)), names, rotation=35, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title("Quantization sensitivity by group")
        figure.tight_layout()
        path = destination / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        generated.append(path)
    return generated
