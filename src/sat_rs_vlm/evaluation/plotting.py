"""Generate reproducible static figures from v1.5 or v1.6 evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PlottingError(ValueError):
    """Raised when plotting inputs or destinations are invalid."""


@dataclass(frozen=True)
class NamedEvaluation:
    """One named supported evaluation directory and its loaded summary."""

    label: str
    directory: Path
    summary_path: Path
    summary: dict[str, Any]
    rows_path: Path | None
    hashes: dict[str, str]


@dataclass(frozen=True)
class NamedComparison:
    """One named paired-comparison directory and its loaded summary."""

    label: str
    directory: Path
    summary_path: Path
    summary: dict[str, Any]
    hashes: dict[str, str]


@dataclass(frozen=True)
class MetricDisplay:
    """Human-readable plotting metadata for one evaluation metric."""

    zh_name: str
    abbreviation: str
    description: str
    direction: str
    value_format: str
    category: str
    tasks: tuple[str, ...]


COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
RATE_LIMITS = (0.0, 1.0)
SUPPORTED_CONTRACT_VERSIONS = frozenset({"1.5", "1.6"})
PRESENTATION_VERSION = "zh-scientific-v1"
METRIC_REGISTRY_VERSION = "1.0"
CJK_FONT_CANDIDATES = (
    "Noto Sans SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "SimHei",
)

TASK_DISPLAY = {
    "captioning": "图像描述（Caption）",
    "counting": "目标计数（Counting）",
    "detection": "目标定位（Grounding）",
    "scene_classification": "场景分类",
    "vqa": "视觉问答（VQA）",
    "change_detection": "变化检测（LEVIR-CC）",
}

QA_TYPE_DISPLAY = {
    "image": "图像整体（image）",
    "object category": "目标类别（object category）",
    "object color": "目标颜色（object color）",
    "object direction": "目标方向（object direction）",
    "object existence": "目标存在性（object existence）",
    "object position": "目标位置（object position）",
    "object quantity": "目标数量（object quantity）",
    "object shape": "目标形状（object shape）",
    "object size": "目标大小（object size）",
    "position": "空间位置（position）",
    "color": "颜色属性（color）",
    "reasoning": "综合推理（reasoning）",
    "rural or urban": "城乡类型（rural or urban）",
    "scene type": "场景类型（scene type）",
}


def _display(
    zh_name: str,
    abbreviation: str,
    description: str,
    direction: str,
    value_format: str,
    category: str,
    *tasks: str,
) -> MetricDisplay:
    return MetricDisplay(
        zh_name,
        abbreviation,
        description,
        direction,
        value_format,
        category,
        tuple(tasks),
    )


METRIC_DISPLAY: dict[str, MetricDisplay] = {
    "iou": _display(
        "交并比",
        "IoU",
        "单个预测框与参考框的重叠程度",
        "higher",
        "score3",
        "statistical",
        "detection",
    ),
    "generalized_iou": _display(
        "广义交并比",
        "GIoU",
        "单样本定位重叠及空间接近程度",
        "higher",
        "score3",
        "statistical",
        "detection",
    ),
    "acc_at_0_5": _display(
        "定位准确率",
        "Acc@0.5",
        "单样本IoU是否达到0.5",
        "higher",
        "percent",
        "statistical",
        "detection",
    ),
    "absolute_error": _display(
        "绝对误差",
        "Absolute Error",
        "预测数量与参考数量之差的绝对值",
        "lower",
        "error2",
        "statistical",
        "counting",
    ),
    "normalized_accuracy": _display(
        "归一化准确率",
        "Normalized Accuracy",
        "标准化文本后是否与参考答案一致",
        "higher",
        "percent",
        "statistical",
        "vqa",
        "scene_classification",
    ),
    "continuous_mean_iou": _display(
        "平均交并比",
        "Mean IoU",
        "预测框与参考框的平均重叠程度",
        "higher",
        "score3",
        "internal",
        "detection",
    ),
    "continuous_mean_generalized_iou": _display(
        "平均广义交并比",
        "Mean GIoU",
        "同时考虑重叠和框间距离的平均定位质量",
        "higher",
        "score3",
        "internal",
        "detection",
    ),
    "continuous_acc_at_0_5": _display(
        "定位准确率",
        "Acc@0.5",
        "IoU不低于0.5的样本比例",
        "higher",
        "percent",
        "internal",
        "detection",
    ),
    "continuous_acc_at_0_7": _display(
        "高质量定位准确率",
        "Acc@0.7",
        "IoU不低于0.7的样本比例",
        "higher",
        "percent",
        "internal",
        "detection",
    ),
    "exact_count_accuracy": _display(
        "精确计数准确率",
        "Exact Accuracy",
        "预测数量与参考数量完全一致的比例",
        "higher",
        "percent",
        "internal",
        "counting",
    ),
    "accuracy_within_1": _display(
        "允许误差1准确率",
        "Accuracy within ±1",
        "计数绝对误差不超过1的比例",
        "higher",
        "percent",
        "internal",
        "counting",
    ),
    "mae_on_parsed": _display(
        "平均绝对误差",
        "MAE",
        "解析成功样本的平均计数绝对误差",
        "lower",
        "error2",
        "internal",
        "counting",
    ),
    "rmse_on_parsed": _display(
        "均方根误差",
        "RMSE",
        "对较大计数误差更敏感的误差指标",
        "lower",
        "error2",
        "internal",
        "counting",
    ),
    "micro_normalized_accuracy": _display(
        "归一化准确率",
        "Normalized Accuracy",
        "文本标准化后与参考答案完全一致的比例",
        "higher",
        "percent",
        "internal",
        "vqa",
        "scene_classification",
    ),
    "token_f1": _display(
        "词元重合综合分",
        "Token F1",
        "预测与参考答案词元精确率和召回率的调和平均",
        "higher",
        "score3",
        "internal",
        "vqa",
        "scene_classification",
    ),
    "bleu_1_approx": _display(
        "一元词重合度",
        "BLEU-1",
        "预测与参考文本的一元词组重合程度",
        "higher",
        "score3",
        "approx",
        "captioning",
        "change_detection",
    ),
    "bleu_4_approx": _display(
        "四元词组重合度",
        "BLEU-4",
        "预测与参考文本的四元词组重合程度",
        "higher",
        "score3",
        "approx",
        "captioning",
        "change_detection",
    ),
    "rouge_l_f1_approx": _display(
        "最长公共子序列得分",
        "ROUGE-L",
        "基于最长公共子序列衡量文本内容覆盖",
        "higher",
        "score3",
        "approx",
        "captioning",
        "change_detection",
    ),
    "meteor_exact_approx": _display(
        "词语对齐综合分",
        "METEOR",
        "综合词语匹配精确率、召回率与顺序",
        "higher",
        "score3",
        "approx",
        "captioning",
        "change_detection",
    ),
    "chrf_approx": _display(
        "字符级综合分",
        "chrF",
        "基于字符n元组衡量文本相似度",
        "higher",
        "score3",
        "approx",
        "captioning",
        "change_detection",
    ),
    "cider_d_single_reference_approx": _display(
        "描述共识得分",
        "CIDEr-D",
        "按信息量加权的描述文本相似度",
        "higher",
        "score3",
        "approx",
        "captioning",
        "change_detection",
    ),
    "length_ratio": _display(
        "生成/参考长度比",
        "Length Ratio",
        "预测与参考答案的平均词元长度比",
        "target1",
        "score3",
        "diagnostic",
        "captioning",
    ),
    "object_precision": _display(
        "目标提及准确率",
        "Object Precision",
        "预测提及目标中参考文本支持的比例",
        "higher",
        "percent",
        "diagnostic",
        "captioning",
    ),
    "object_recall": _display(
        "目标提及召回率",
        "Object Recall",
        "参考文本目标中被预测覆盖的比例",
        "higher",
        "percent",
        "diagnostic",
        "captioning",
    ),
    "object_f1": _display(
        "目标提及综合分",
        "Object F1",
        "目标提及准确率和召回率的调和平均",
        "higher",
        "percent",
        "diagnostic",
        "captioning",
    ),
    "object_omission_rate": _display(
        "目标遗漏率",
        "Object Omission",
        "参考文本目标未被预测提及的比例",
        "lower",
        "percent",
        "diagnostic",
        "captioning",
    ),
    "reference_unsupported_object_rate": _display(
        "参考文本不支持提及率",
        "Unsupported Mention",
        "预测目标提及中未获参考文本支持的比例",
        "lower",
        "percent",
        "diagnostic",
        "captioning",
    ),
    "count_consistency_accuracy": _display(
        "数量一致率",
        "Count Consistency",
        "预测与参考文本数量陈述一致的比例",
        "higher",
        "percent",
        "diagnostic",
        "captioning",
    ),
    "spatial_relation_f1": _display(
        "空间关系综合分",
        "Spatial Relation F1",
        "空间关系提及的词典规则F1",
        "higher",
        "percent",
        "diagnostic",
        "captioning",
    ),
    "binary_accuracy": _display(
        "准确率",
        "Accuracy",
        "全部样本中变化状态判断正确的比例",
        "higher",
        "percent",
        "internal",
        "change_detection",
    ),
    "balanced_accuracy": _display(
        "平衡准确率",
        "Balanced Accuracy",
        "有变化和无变化召回率的平均值",
        "higher",
        "percent",
        "internal",
        "change_detection",
    ),
    "change_precision": _display(
        "变化查准率",
        "Precision",
        "判为有变化的样本中真正发生变化的比例",
        "higher",
        "percent",
        "internal",
        "change_detection",
    ),
    "change_recall": _display(
        "变化召回率",
        "Recall",
        "真实发生变化的样本中被成功识别的比例",
        "higher",
        "percent",
        "internal",
        "change_detection",
    ),
    "change_f1": _display(
        "变化综合分",
        "F1",
        "变化查准率与变化召回率的调和平均",
        "higher",
        "percent",
        "internal",
        "change_detection",
    ),
    "matthews_correlation_coefficient": _display(
        "马修斯相关系数",
        "MCC",
        "综合四类混淆情况的平衡相关性指标",
        "higher",
        "score3",
        "internal",
        "change_detection",
    ),
    "cohen_kappa": _display(
        "一致性系数",
        "Kappa",
        "扣除随机一致后的分类一致性",
        "higher",
        "score3",
        "internal",
        "change_detection",
    ),
    "false_positive_rate": _display(
        "误报率",
        "FPR",
        "真实无变化却被判为有变化的比例",
        "lower",
        "percent",
        "internal",
        "change_detection",
    ),
    "false_negative_rate": _display(
        "漏报率",
        "FNR",
        "真实有变化却被判为无变化的比例",
        "lower",
        "percent",
        "internal",
        "change_detection",
    ),
    "true_negatives": _display(
        "真负例",
        "TN",
        "真实无变化且预测无变化的样本数",
        "higher",
        "count",
        "statistical",
        "change_detection",
    ),
    "false_positives": _display(
        "假正例",
        "FP",
        "真实无变化但预测有变化的样本数",
        "lower",
        "count",
        "statistical",
        "change_detection",
    ),
    "false_negatives": _display(
        "假负例",
        "FN",
        "真实有变化但预测无变化的样本数",
        "lower",
        "count",
        "statistical",
        "change_detection",
    ),
    "true_positives": _display(
        "真正例",
        "TP",
        "真实有变化且预测有变化的样本数",
        "higher",
        "count",
        "statistical",
        "change_detection",
    ),
    "explicit_binary_decision_rate": _display(
        "独立二分类判定占比",
        "Explicit Binary",
        "由独立0/1推理提供变化结论的比例",
        "higher",
        "percent",
        "diagnostic",
        "change_detection",
    ),
    "caption_fallback_decision_rate": _display(
        "描述文本兼容回退占比",
        "Caption Fallback",
        "因缺少独立二分类字段而从描述文本兼容解析的比例",
        "lower",
        "percent",
        "diagnostic",
        "change_detection",
    ),
    "latency_ms_mean": _display(
        "平均延迟",
        "Mean",
        "全部有效样本的平均推理延迟",
        "lower",
        "latency1",
        "efficiency",
        "overall",
    ),
    "latency_ms_p50": _display(
        "中位延迟", "P50", "50%样本不超过的推理延迟", "lower", "latency1", "efficiency", "overall"
    ),
    "latency_ms_p95": _display(
        "长尾延迟", "P95", "95%样本不超过的推理延迟", "lower", "latency1", "efficiency", "overall"
    ),
}


def _metric_label(metric: str) -> str:
    display = METRIC_DISPLAY[metric]
    return f"{display.zh_name}\n（{display.abbreviation}）"


def _format_value(value: float, value_format: str) -> str:
    if value_format == "percent":
        return f"{value:.1%}"
    if value_format == "error2":
        return f"{value:.2f}"
    if value_format == "latency1":
        return f"{value:.1f} ms"
    if value_format == "count":
        return f"{int(round(value)):,}"
    return f"{value:.3f}"


def _evaluation_sample_count(evaluation: NamedEvaluation) -> int | None:
    distribution = evaluation.summary.get("overall", {}).get("task_distribution", {})
    if not isinstance(distribution, dict):
        return None
    counts = [
        value
        for value in distribution.values()
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    return sum(counts) if counts else None


def _evaluation_context(evaluations: Iterable[NamedEvaluation], attribute: str) -> str:
    items = list(evaluations)
    counts = sorted(
        {
            count
            for evaluation in items
            if (count := _evaluation_sample_count(evaluation)) is not None
        }
    )
    versions = sorted({str(item.summary.get("contract_version")) for item in items})
    count_text = f"n={counts[0]:,}" if len(counts) == 1 else "样本量见图例"
    version_text = "/".join(versions)
    return f"{count_text}｜评测契约 v{version_text}｜{attribute}"


def _add_figure_note(figure: Any, note: str) -> None:
    figure.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, color="#4D4D4D")


def _comparison_context(comparison: NamedComparison) -> str:
    paired = comparison.summary.get("overall", {}).get("num_paired_samples")
    count_text = (
        f"n={paired:,}"
        if isinstance(paired, int) and not isinstance(paired, bool)
        else "配对样本量见比较清单"
    )
    version = comparison.summary.get("required_contract_version")
    return f"{count_text}｜评测契约 v{version}｜配对统计指标"


CORE_METRICS: dict[str, tuple[tuple[str, str], ...]] = {
    "detection": (
        ("continuous_mean_iou", _metric_label("continuous_mean_iou")),
        ("continuous_mean_generalized_iou", _metric_label("continuous_mean_generalized_iou")),
        ("continuous_acc_at_0_5", _metric_label("continuous_acc_at_0_5")),
        ("continuous_acc_at_0_7", _metric_label("continuous_acc_at_0_7")),
    ),
    "counting_accuracy": (
        ("exact_count_accuracy", _metric_label("exact_count_accuracy")),
        ("accuracy_within_1", _metric_label("accuracy_within_1")),
    ),
    "counting_error": (
        ("mae_on_parsed", _metric_label("mae_on_parsed")),
        ("rmse_on_parsed", _metric_label("rmse_on_parsed")),
    ),
    "text": (
        ("vqa.micro_normalized_accuracy", "VQA归一化准确率\n（Normalized Accuracy）"),
        ("vqa.token_f1", "VQA词元重合综合分\n（Token F1）"),
        (
            "scene_classification.micro_normalized_accuracy",
            "场景归一化准确率\n（Normalized Accuracy）",
        ),
        ("scene_classification.token_f1", "场景词元重合综合分\n（Token F1）"),
    ),
    "captioning": (
        ("bleu_1_approx", _metric_label("bleu_1_approx")),
        ("bleu_4_approx", _metric_label("bleu_4_approx")),
        ("rouge_l_f1_approx", _metric_label("rouge_l_f1_approx")),
        ("meteor_exact_approx", _metric_label("meteor_exact_approx")),
        ("chrf_approx", _metric_label("chrf_approx")),
        ("cider_d_single_reference_approx", _metric_label("cider_d_single_reference_approx")),
    ),
}

COMPARISON_METRICS: tuple[tuple[str, str, str], ...] = (
    ("detection", "iou", "目标定位 · 交并比（IoU）"),
    ("detection", "generalized_iou", "目标定位 · 广义交并比（GIoU）"),
    ("detection", "acc_at_0_5", "目标定位 · 准确率（Acc@0.5）"),
    ("counting", "absolute_error", "目标计数 · 绝对误差"),
    ("counting", "exact_count_accuracy", "目标计数 · 精确准确率"),
    ("vqa", "normalized_accuracy", "视觉问答 · 归一化准确率"),
    ("scene_classification", "normalized_accuracy", "场景分类 · 归一化准确率"),
    ("captioning", "rouge_l_f1_approx", "图像描述 · ROUGE-L"),
    ("captioning", "chrf_approx", "图像描述 · chrF"),
    ("captioning", "cider_d_single_reference_approx", "图像描述 · CIDEr-D"),
    ("change_detection", "binary_accuracy", "变化检测 · 准确率"),
    ("change_detection", "change_f1", "变化检测 · F1"),
)

REPRESENTATIVE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("detection", "iou", "目标定位 · IoU"),
    ("counting", "absolute_error", "目标计数 · 绝对误差"),
    ("vqa", "normalized_accuracy", "视觉问答 · 归一化准确率"),
    (
        "scene_classification",
        "normalized_accuracy",
        "场景分类 · 归一化准确率",
    ),
    ("captioning", "rouge_l_f1_approx", "图像描述 · ROUGE-L"),
    ("change_detection", "binary_accuracy", "LEVIR-CC · 变化准确率"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlottingError(f"Unable to read JSON object: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlottingError(f"Expected a JSON object: {path}")
    return payload


def _validate_label(label: str) -> str:
    normalized = label.strip()
    if not normalized or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", normalized):
        raise PlottingError(
            f"Invalid label {label!r}; use letters, numbers, dot, underscore, or hyphen."
        )
    return normalized


def parse_named_path(specification: str) -> tuple[str, Path]:
    """Parse a CLI ``label=path`` value."""

    if "=" not in specification:
        raise PlottingError(f"Expected LABEL=PATH, received: {specification!r}")
    label, raw_path = specification.split("=", 1)
    label = _validate_label(label)
    if not raw_path.strip():
        raise PlottingError(f"Missing path for label {label!r}")
    return label, Path(raw_path).expanduser().resolve()


def _validate_unique_labels(items: Iterable[tuple[str, Path]], kind: str) -> None:
    seen: set[str] = set()
    for label, _ in items:
        if label in seen:
            raise PlottingError(f"Duplicate {kind} label: {label}")
        seen.add(label)


def load_evaluations(specifications: Iterable[str]) -> list[NamedEvaluation]:
    """Load and validate named v1.5/v1.6 evaluation directories."""

    parsed = [parse_named_path(specification) for specification in specifications]
    _validate_unique_labels(parsed, "evaluation")
    evaluations: list[NamedEvaluation] = []
    for label, directory in parsed:
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            raise PlottingError(f"Missing summary.json for evaluation {label}: {directory}")
        summary = _load_json(summary_path)
        contract_version = str(summary.get("contract_version"))
        if contract_version not in SUPPORTED_CONTRACT_VERSIONS:
            raise PlottingError(
                f"Evaluation {label} uses contract {summary.get('contract_version')!r}; "
                f"supported versions are {sorted(SUPPORTED_CONTRACT_VERSIONS)}."
            )
        candidate_rows_path = directory / "evaluated_predictions.jsonl"
        rows_path: Path | None
        hashes = {"summary.json": _sha256(summary_path)}
        if candidate_rows_path.is_file():
            rows_path = candidate_rows_path
            hashes["evaluated_predictions.jsonl"] = _sha256(candidate_rows_path)
        else:
            rows_path = None
        evaluations.append(
            NamedEvaluation(label, directory, summary_path, summary, rows_path, hashes)
        )
    if not evaluations:
        raise PlottingError("At least one --evaluation LABEL=PATH input is required.")
    return evaluations


def load_comparisons(specifications: Iterable[str]) -> list[NamedComparison]:
    """Load and validate named v1.5/v1.6 paired-comparison directories."""

    parsed = [parse_named_path(specification) for specification in specifications]
    _validate_unique_labels(parsed, "comparison")
    comparisons: list[NamedComparison] = []
    for label, directory in parsed:
        summary_path = directory / "comparison_summary.json"
        if not summary_path.is_file():
            raise PlottingError(
                f"Missing comparison_summary.json for comparison {label}: {directory}"
            )
        summary = _load_json(summary_path)
        contract_version = str(summary.get("required_contract_version"))
        if contract_version not in SUPPORTED_CONTRACT_VERSIONS:
            raise PlottingError(
                f"Comparison {label} requires contract "
                f"{summary.get('required_contract_version')!r}; supported versions are "
                f"{sorted(SUPPORTED_CONTRACT_VERSIONS)}."
            )
        comparisons.append(
            NamedComparison(
                label,
                directory,
                summary_path,
                summary,
                {"comparison_summary.json": _sha256(summary_path)},
            )
        )
    return comparisons


def _metric_record(summary: dict[str, Any], task: str, metric: str) -> dict[str, Any] | None:
    task_payload = summary.get("by_task", {}).get(task, {})
    record = task_payload.get("metrics", {}).get(metric)
    return record if isinstance(record, dict) else None


def _metric_value(summary: dict[str, Any], task: str, metric: str) -> float | None:
    record = _metric_record(summary, task, metric)
    if not record or record.get("status") != "ok":
        return None
    value = record.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _overall_metric(summary: dict[str, Any], metric: str) -> float | None:
    record = summary.get("overall", {}).get("metrics", {}).get(metric)
    if not isinstance(record, dict) or record.get("status") != "ok":
        return None
    value = record.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _semantic_metric(summary: dict[str, Any], metric: str) -> float | None:
    record = summary.get("semantic", {}).get("overall", {}).get("metrics", {}).get(metric)
    if not isinstance(record, dict) or record.get("status") != "ok":
        return None
    value = record.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _qa_type_metric(summary: dict[str, Any], qa_type: str) -> tuple[float, int] | None:
    payload = summary.get("by_qa_type", {}).get(qa_type, {})
    record = payload.get("metrics", {}).get("micro_normalized_accuracy")
    if not isinstance(record, dict) or record.get("status") != "ok":
        return None
    value = record.get("value")
    count = record.get("num_samples")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or isinstance(count, bool)
        or not isinstance(count, int)
    ):
        return None
    return float(value), count


def _diagnostic_rows(evaluation: NamedEvaluation) -> dict[str, list[float]]:
    diagnostics: dict[str, list[float]] = {
        "grounding_iou": [],
        "count_absolute_error": [],
        "count_signed_error": [],
    }
    if evaluation.rows_path is None:
        return diagnostics
    try:
        with evaluation.rows_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PlottingError(
                        f"Invalid JSONL in {evaluation.rows_path} line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise PlottingError(
                        f"Expected JSON object in {evaluation.rows_path} line {line_number}."
                    )
                metrics = row.get("sample_metrics", {})
                if not isinstance(metrics, dict):
                    continue
                task = str(row.get("task_type", ""))
                if task == "detection":
                    value = metrics.get("iou")
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        diagnostics["grounding_iou"].append(float(value))
                elif task == "counting":
                    absolute = metrics.get("absolute_error")
                    signed = metrics.get("signed_error")
                    if isinstance(absolute, (int, float)) and not isinstance(absolute, bool):
                        diagnostics["count_absolute_error"].append(float(absolute))
                    if isinstance(signed, (int, float)) and not isinstance(signed, bool):
                        diagnostics["count_signed_error"].append(float(signed))
    except OSError as exc:
        raise PlottingError(f"Unable to read {evaluation.rows_path}: {exc}") from exc
    return diagnostics


def _select_cjk_font(matplotlib: Any) -> str:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in CJK_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    raise PlottingError(
        "未找到可用中文字体；请安装 Noto Sans SC、Source Han Sans SC、"
        "Microsoft YaHei 或 SimHei 后重试。"
    )


def _prepare_matplotlib() -> tuple[Any, Any, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlottingError(
            "Plotting requires the optional dependency: pip install -e '.[reliability-plot]'"
        ) from exc
    cjk_font = _select_cjk_font(matplotlib)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [cjk_font, "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "axes.grid.axis": "y",
            "axes.unicode_minus": False,
        }
    )
    return matplotlib, plt, cjk_font


def _grouped_bars(
    axis: Any,
    labels: list[str],
    series: list[tuple[str, list[float | None]]],
    *,
    rate_axis: bool = True,
    annotate: bool = False,
    value_formats: list[str] | None = None,
) -> bool:
    available = any(value is not None for _, values in series for value in values)
    if not available:
        return False
    width = min(0.8 / max(len(series), 1), 0.32)
    centers = list(range(len(labels)))
    for index, (name, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        positions = [center + offset for center in centers]
        plotted = [float(value) if value is not None else 0.0 for value in values]
        bars = axis.bar(
            positions,
            plotted,
            width=width,
            label=name,
            color=COLORS[index % len(COLORS)],
            alpha=0.9,
        )
        for metric_index, (bar, value) in enumerate(zip(bars, values, strict=True)):
            if value is None:
                bar.set_alpha(0.12)
                continue
            if annotate:
                value_format = (
                    value_formats[metric_index]
                    if value_formats and metric_index < len(value_formats)
                    else "score3"
                )
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    _format_value(value, value_format),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90 if len(labels) > 5 else 0,
                )
    axis.set_xticks(centers, labels, rotation=18, ha="right")
    if rate_axis:
        axis.set_ylim(*RATE_LIMITS)
    axis.legend(frameon=False, fontsize=8)
    return True


def _save_figure(
    figure: Any,
    destination: Path,
    stem: str,
    formats: tuple[str, ...],
) -> list[Path]:
    paths: list[Path] = []
    for image_format in formats:
        path = destination / f"{stem}.{image_format}"
        metadata: dict[str, Any] = {"Creator": "sat-rs-vlm 中文科研绘图 v1.6"}
        if image_format == "svg":
            metadata["Date"] = None
        figure.savefig(path, bbox_inches="tight", metadata=metadata)
        if image_format == "svg":
            normalized_svg = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized_svg + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _task_distribution(plt: Any, evaluations: list[NamedEvaluation]) -> Any | None:
    task_order = ("captioning", "counting", "detection", "scene_classification", "vqa")
    applicable = [
        evaluation
        for evaluation in evaluations
        if any(
            task in evaluation.summary.get("overall", {}).get("task_distribution", {})
            for task in task_order
        )
    ]
    if not applicable:
        return None
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    series: list[tuple[str, list[float | None]]] = []
    for evaluation in applicable:
        distribution = evaluation.summary.get("overall", {}).get("task_distribution", {})
        series.append(
            (
                evaluation.label,
                [
                    float(distribution[task]) if isinstance(distribution.get(task), int) else None
                    for task in task_order
                ],
            )
        )
    _grouped_bars(
        axis,
        [TASK_DISPLAY[task] for task in task_order],
        series,
        rate_axis=False,
        annotate=True,
        value_formats=["count"] * len(task_order),
    )
    axis.set_ylabel("样本数（条）")
    axis.set_title("VRSBench五类评测任务的样本分布")
    _add_figure_note(figure, _evaluation_context(applicable, "样本统计"))
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def _core_metrics(plt: Any, evaluations: list[NamedEvaluation]) -> Any | None:
    applicable = [
        evaluation
        for evaluation in evaluations
        if "detection" in evaluation.summary.get("by_task", {})
    ]
    if not applicable:
        return None
    figure, axes = plt.subplots(3, 2, figsize=(16, 12))
    panels = (
        (axes[0, 0], "detection", "目标定位质量｜越高越好", True),
        (axes[0, 1], "counting_accuracy", "目标计数准确性｜越高越好", True),
        (axes[1, 0], "counting_error", "目标计数误差｜越低越好", False),
        (axes[1, 1], "text", "视觉问答与场景分类｜越高越好", True),
        (axes[2, 0], "captioning", "图像描述质量｜越高越好", True),
    )
    any_panel = False
    for axis, profile, title, rate_axis in panels:
        definitions = CORE_METRICS[profile]
        labels = [label for _, label in definitions]
        series: list[tuple[str, list[float | None]]] = []
        for evaluation in applicable:
            values: list[float | None] = []
            for metric_path, _ in definitions:
                if profile == "text":
                    task, metric = metric_path.split(".", 1)
                elif profile.startswith("counting"):
                    task, metric = "counting", metric_path
                else:
                    task, metric = profile, metric_path
                values.append(_metric_value(evaluation.summary, task, metric))
            series.append((evaluation.label, values))
        formats = []
        for metric_path, _ in definitions:
            metric_key = metric_path.split(".", 1)[-1]
            formats.append(METRIC_DISPLAY[metric_key].value_format)
        plotted = _grouped_bars(
            axis,
            labels,
            series,
            rate_axis=rate_axis,
            annotate=True,
            value_formats=formats,
        )
        any_panel = any_panel or plotted
        axis.set_title(title)
        axis.set_ylabel("得分" if rate_axis else "计数误差")
    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.02,
        0.92,
        "读图说明\n\n准确率与重叠度越高越好。\nMAE/RMSE越低越好。\n图像描述指标为内部近似指标。",
        va="top",
        fontsize=11,
        linespacing=1.55,
    )
    figure.suptitle("VRSBench多任务核心评测指标", fontsize=15)
    _add_figure_note(figure, _evaluation_context(applicable, "内部评测指标"))
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    return figure if any_panel else None


def _grounding_cdf(
    plt: Any,
    evaluations: list[NamedEvaluation],
    diagnostics: dict[str, dict[str, list[float]]],
) -> Any | None:
    available = [
        (evaluation.label, diagnostics[evaluation.label]["grounding_iou"])
        for evaluation in evaluations
        if diagnostics[evaluation.label]["grounding_iou"]
    ]
    if not available:
        return None
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    for index, (label, values) in enumerate(available):
        ordered = sorted(values)
        cumulative = [(position + 1) / len(ordered) for position in range(len(ordered))]
        axis.plot(ordered, cumulative, label=f"{label} (n={len(ordered):,})", color=COLORS[index])
    for threshold in (0.5, 0.7):
        axis.axvline(threshold, color="#555555", linestyle="--", linewidth=1)
        axis.text(threshold + 0.01, 0.04, f"IoU={threshold}", rotation=90, va="bottom")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("逐样本交并比（IoU）")
    axis.set_ylabel("累计样本比例")
    axis.set_title("目标定位IoU累积分布｜曲线整体越靠右越好")
    axis.legend(frameon=False)
    _add_figure_note(
        figure,
        "IoU越大表示预测框与参考框越重合；虚线标出IoU=0.5和0.7两个定位阈值。｜"
        + _evaluation_context(evaluations, "内部逐样本诊断"),
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return figure


def _counting_errors(
    plt: Any,
    evaluations: list[NamedEvaluation],
    diagnostics: dict[str, dict[str, list[float]]],
) -> Any | None:
    available = [
        evaluation
        for evaluation in evaluations
        if diagnostics[evaluation.label]["count_absolute_error"]
    ]
    if not available:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    error_labels = ("完全正确", "误差为1", "误差2–3", "误差大于3")
    direction_labels = ("低估", "完全正确", "高估")
    error_series: list[tuple[str, list[float | None]]] = []
    direction_series: list[tuple[str, list[float | None]]] = []
    for evaluation in available:
        absolute = diagnostics[evaluation.label]["count_absolute_error"]
        signed = diagnostics[evaluation.label]["count_signed_error"]
        total = len(absolute)
        error_series.append(
            (
                evaluation.label,
                [
                    sum(value == 0 for value in absolute) / total,
                    sum(0 < value <= 1 for value in absolute) / total,
                    sum(1 < value <= 3 for value in absolute) / total,
                    sum(value > 3 for value in absolute) / total,
                ],
            )
        )
        if signed:
            signed_total = len(signed)
            direction_series.append(
                (
                    evaluation.label,
                    [
                        sum(value < 0 for value in signed) / signed_total,
                        sum(value == 0 for value in signed) / signed_total,
                        sum(value > 0 for value in signed) / signed_total,
                    ],
                )
            )
    _grouped_bars(
        axes[0], list(error_labels), error_series, annotate=True, value_formats=["percent"] * 4
    )
    axes[0].set_ylabel("样本比例")
    axes[0].set_title("计数绝对误差分布｜错误区间越小越好")
    _grouped_bars(
        axes[1],
        list(direction_labels),
        direction_series,
        annotate=True,
        value_formats=["percent"] * 3,
    )
    axes[1].set_ylabel("样本比例")
    axes[1].set_title("计数偏差方向｜区分低估与高估")
    _add_figure_note(
        figure,
        "绝对误差表示预测数量与参考数量之差的绝对值。｜"
        + _evaluation_context(available, "内部逐样本诊断"),
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return figure


def _qa_type_accuracy(plt: Any, evaluations: list[NamedEvaluation]) -> Any | None:
    applicable = [evaluation for evaluation in evaluations if evaluation.summary.get("by_qa_type")]
    qa_types = sorted(
        {
            qa_type
            for evaluation in applicable
            for qa_type in evaluation.summary.get("by_qa_type", {})
        }
    )
    if not qa_types:
        return None
    series: list[tuple[str, list[float | None]]] = []
    sample_counts: list[int] = []
    for qa_type in qa_types:
        counts = [
            result[1]
            for evaluation in applicable
            if (result := _qa_type_metric(evaluation.summary, qa_type)) is not None
        ]
        sample_counts.append(max(counts) if counts else 0)
    for evaluation in applicable:
        series.append(
            (
                evaluation.label,
                [
                    result[0]
                    if (result := _qa_type_metric(evaluation.summary, qa_type)) is not None
                    else None
                    for qa_type in qa_types
                ],
            )
        )
    figure, axis = plt.subplots(figsize=(11.5, max(7.0, len(qa_types) * 0.58)))
    display_labels = [
        f"{QA_TYPE_DISPLAY.get(name.lower(), name)}（n={count:,}）"
        for name, count in zip(qa_types, sample_counts, strict=True)
    ]
    height = min(0.8 / max(len(series), 1), 0.32)
    centers = list(range(len(display_labels)))
    for index, (name, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * height
        positions = [center + offset for center in centers]
        plotted = [float(value) if value is not None else 0.0 for value in values]
        bars = axis.barh(
            positions,
            plotted,
            height=height,
            label=name,
            color=COLORS[index % len(COLORS)],
            alpha=0.9,
        )
        for bar, value in zip(bars, values, strict=True):
            if value is None:
                bar.set_alpha(0.12)
                continue
            axis.text(
                min(float(value) + 0.008, 0.985),
                bar.get_y() + bar.get_height() / 2,
                _format_value(float(value), "percent"),
                ha="left" if float(value) < 0.93 else "right",
                va="center",
                fontsize=8,
            )
    axis.set_yticks(centers, display_labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("归一化准确率")
    axis.set_title("视觉问答（VQA）不同问题类型的准确率｜越高越好")
    axis.legend(frameon=False, loc="lower right")
    _add_figure_note(
        figure,
        "归一化准确率：对大小写、标点和空格标准化后，与参考答案完全一致的比例。｜"
        + _evaluation_context(applicable, "内部评测指标"),
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def _caption_quality(plt: Any, evaluations: list[NamedEvaluation]) -> Any | None:
    applicable = [
        evaluation
        for evaluation in evaluations
        if "captioning" in evaluation.summary.get("by_task", {})
    ]
    if not applicable:
        return None
    quality = CORE_METRICS["captioning"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    series = [
        (
            evaluation.label,
            [_metric_value(evaluation.summary, "captioning", metric) for metric, _ in quality],
        )
        for evaluation in applicable
    ]
    _grouped_bars(
        axes[0],
        [label for _, label in quality],
        series,
        annotate=True,
        value_formats=[METRIC_DISPLAY[metric].value_format for metric, _ in quality],
    )
    axes[0].set_ylabel("文本质量得分")
    axes[0].set_title("图像描述文本质量｜内部近似指标，越高越好")
    length_series = [
        (
            evaluation.label,
            [_metric_value(evaluation.summary, "captioning", "length_ratio")],
        )
        for evaluation in applicable
    ]
    _grouped_bars(
        axes[1],
        [_metric_label("length_ratio")],
        length_series,
        rate_axis=False,
        annotate=True,
        value_formats=["score3"],
    )
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="与参考文本等长")
    axes[1].set_ylabel("长度比")
    axes[1].set_title("图像描述长度诊断｜越接近1越好")
    _add_figure_note(
        figure,
        "BLEU、ROUGE-L、METEOR、chrF和CIDEr-D均为内部近似实现；长度比用于判断输出是否偏长或偏短。｜"
        + _evaluation_context(applicable, "内部近似指标"),
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def _semantic_diagnostics(plt: Any, evaluations: list[NamedEvaluation]) -> Any | None:
    ability_definitions = (
        "object_precision",
        "object_recall",
        "object_f1",
        "count_consistency_accuracy",
        "spatial_relation_f1",
    )
    error_definitions = (
        "object_omission_rate",
        "reference_unsupported_object_rate",
    )
    applicable = [
        evaluation
        for evaluation in evaluations
        if "captioning" in evaluation.summary.get("by_task", {})
    ]
    ability_series = [
        (
            evaluation.label,
            [_semantic_metric(evaluation.summary, metric) for metric in ability_definitions],
        )
        for evaluation in applicable
    ]
    error_series = [
        (
            evaluation.label,
            [_semantic_metric(evaluation.summary, metric) for metric in error_definitions],
        )
        for evaluation in applicable
    ]
    if not any(
        value is not None for _, values in ability_series + error_series for value in values
    ):
        return None
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8), width_ratios=(2.2, 1.0))
    _grouped_bars(
        axes[0],
        [_metric_label(metric) for metric in ability_definitions],
        ability_series,
        annotate=True,
        value_formats=["percent"] * len(ability_definitions),
    )
    axes[0].set_ylabel("诊断得分/比例")
    axes[0].set_title("参考文本语义能力诊断｜越高越好")
    _grouped_bars(
        axes[1],
        [_metric_label(metric) for metric in error_definitions],
        error_series,
        annotate=True,
        value_formats=["percent"] * len(error_definitions),
    )
    axes[1].set_ylabel("错误比例")
    axes[1].set_title("参考文本语义错误诊断｜越低越好")
    figure.suptitle("图像描述的参考文本语义诊断", fontsize=14)
    _add_figure_note(
        figure,
        "重要限制：这些规则只比较预测文本与参考文本，不读取图像，不代表图像级事实正确率或幻觉率。｜"
        + _evaluation_context(applicable, "内部规则诊断"),
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    return figure


def _comparison_record(
    comparison: NamedComparison, task: str, metric: str
) -> dict[str, Any] | None:
    record = comparison.summary.get("by_task", {}).get(task, {}).get("metrics", {}).get(metric)
    return record if isinstance(record, dict) and record.get("status") == "ok" else None


def _paired_improvement(plt: Any, comparison: NamedComparison) -> Any | None:
    labels: list[str] = []
    means: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    colors: list[str] = []
    uncertain: list[bool] = []
    for task, metric, label in COMPARISON_METRICS:
        record = _comparison_record(comparison, task, metric)
        if record is None:
            continue
        mean_value = record.get("improvement_mean")
        interval = record.get("improvement_ci95_paired_bootstrap")
        if (
            isinstance(mean_value, bool)
            or not isinstance(mean_value, (int, float))
            or not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(value, (int, float)) for value in interval)
        ):
            continue
        mean_float = float(mean_value)
        low, high = float(interval[0]), float(interval[1])
        labels.append(label)
        means.append(mean_float)
        lower_errors.append(max(0.0, mean_float - low))
        upper_errors.append(max(0.0, high - mean_float))
        colors.append(COLORS[2] if low > 0 else COLORS[5] if high < 0 else "#999999")
        uncertain.append(low <= 0 <= high)
    if not labels:
        return None
    figure, axis = plt.subplots(figsize=(10, max(5.5, len(labels) * 0.48)))
    positions = list(range(len(labels)))
    axis.barh(positions, means, color=colors, alpha=0.9)
    axis.errorbar(
        means,
        positions,
        xerr=[lower_errors, upper_errors],
        fmt="none",
        ecolor="#222222",
        capsize=3,
        linewidth=1,
    )
    axis.axvline(0.0, color="#222222", linewidth=1)
    for position, mean, is_uncertain in zip(positions, means, uncertain, strict=True):
        if is_uncertain:
            axis.text(
                mean,
                position - 0.28,
                "差异不确定",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#666666",
            )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("方向统一后的改善量（正值表示改善，负值表示退化）")
    axis.set_title(f"逐样本配对改善与Bootstrap 95%置信区间 · {comparison.label}")
    _add_figure_note(
        figure,
        "绿色表示置信区间完全高于0，橙色表示完全低于0；灰色且跨越0表示当前证据不足以确认稳定差异。",
    )
    figure.text(
        0.99,
        0.01,
        _comparison_context(comparison),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4D4D4D",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    return figure


def _win_tie_loss(plt: Any, comparison: NamedComparison) -> Any | None:
    labels: list[str] = []
    wins: list[float] = []
    ties: list[float] = []
    losses: list[float] = []
    for task, metric, label in REPRESENTATIVE_METRICS:
        record = _comparison_record(comparison, task, metric)
        if record is None:
            continue
        raw_win = record.get("wins")
        raw_tie = record.get("ties")
        raw_loss = record.get("losses")
        if (
            not isinstance(raw_win, int)
            or isinstance(raw_win, bool)
            or not isinstance(raw_tie, int)
            or isinstance(raw_tie, bool)
            or not isinstance(raw_loss, int)
            or isinstance(raw_loss, bool)
        ):
            continue
        win_count = raw_win
        tie_count = raw_tie
        loss_count = raw_loss
        total = win_count + tie_count + loss_count
        if total <= 0:
            continue
        labels.append(label)
        wins.append(win_count / total)
        ties.append(tie_count / total)
        losses.append(loss_count / total)
    if not labels:
        return None
    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    positions = list(range(len(labels)))
    axis.barh(positions, wins, color=COLORS[2], label="改善")
    axis.barh(positions, ties, left=wins, color="#BDBDBD", label="持平")
    left = [win + tie for win, tie in zip(wins, ties, strict=True)]
    axis.barh(positions, losses, left=left, color=COLORS[5], label="退化")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("配对样本比例")
    axis.set_title(f"同一样本上的改善、持平与退化比例 · {comparison.label}")
    axis.legend(frameon=False, ncol=3, loc="lower right")
    _add_figure_note(
        figure,
        "每个任务选取一个代表指标，对相同ID样本逐项比较。｜" + _comparison_context(comparison),
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def _levir_evaluation(evaluations: list[NamedEvaluation]) -> NamedEvaluation | None:
    for evaluation in evaluations:
        if "change_detection" in evaluation.summary.get("by_task", {}):
            return evaluation
    return None


def _levir_confusion(plt: Any, evaluation: NamedEvaluation) -> Any | None:
    metric_names = ("true_negatives", "false_positives", "false_negatives", "true_positives")
    values = [_metric_value(evaluation.summary, "change_detection", name) for name in metric_names]
    if any(value is None for value in values):
        return None
    tn, fp, fn, tp = (int(value) for value in values if value is not None)
    matrix = [[tn, fp], [fn, tp]]
    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    image = axis.imshow(matrix, cmap="Blues")
    rows = [tn + fp, fn + tp]
    for row_index in range(2):
        for column_index in range(2):
            count = matrix[row_index][column_index]
            proportion = count / rows[row_index] if rows[row_index] else 0.0
            axis.text(
                column_index,
                row_index,
                f"{count:,}\n{proportion:.1%}",
                ha="center",
                va="center",
                color="white" if count > max(tn, fp, fn, tp) / 2 else "#222222",
                fontsize=12,
            )
    axis.set_xticks((0, 1), ("预测无变化", "预测有变化"))
    axis.set_yticks((0, 1), ("真实无变化", "真实有变化"))
    axis.set_xlabel("预测类别")
    axis.set_ylabel("真实类别")
    axis.set_title(f"LEVIR-CC变化判定混淆矩阵 · {evaluation.label}")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="样本数")
    _add_figure_note(
        figure,
        "每个单元格依次显示样本数和按真实类别归一化的比例。｜"
        + _evaluation_context([evaluation], "内部评测指标"),
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def _levir_binary_metrics(plt: Any, evaluation: NamedEvaluation) -> Any | None:
    core_metrics = (
        "binary_accuracy",
        "balanced_accuracy",
        "change_precision",
        "change_recall",
        "change_f1",
        "matthews_correlation_coefficient",
        "cohen_kappa",
    )
    error_metrics = ("false_positive_rate", "false_negative_rate")
    source_metrics = ("explicit_binary_decision_rate", "caption_fallback_decision_rate")
    core_values = [
        _metric_value(evaluation.summary, "change_detection", name) for name in core_metrics
    ]
    error_values = [
        _metric_value(evaluation.summary, "change_detection", name) for name in error_metrics
    ]
    source_values = [
        _metric_value(evaluation.summary, "change_detection", name) for name in source_metrics
    ]
    if not any(value is not None for value in core_values + error_values + source_values):
        return None
    has_source = any(value is not None for value in source_values)
    figure, axes = plt.subplots(
        1,
        3 if has_source else 2,
        figsize=(17 if has_source else 14, 5.8),
        width_ratios=(2.8, 1.0, 1.1) if has_source else (2.8, 1.0),
    )
    core_axis, error_axis = axes[0], axes[1]
    _grouped_bars(
        core_axis,
        [_metric_label(metric) for metric in core_metrics],
        [(evaluation.label, core_values)],
        annotate=True,
        value_formats=[METRIC_DISPLAY[metric].value_format for metric in core_metrics],
    )
    core_axis.set_ylabel("性能得分/比例")
    core_axis.set_title("核心变化判定能力｜越高越好")
    _grouped_bars(
        error_axis,
        [_metric_label(metric) for metric in error_metrics],
        [(evaluation.label, error_values)],
        annotate=True,
        value_formats=["percent", "percent"],
    )
    error_axis.set_ylabel("错误比例")
    error_axis.set_title("错误诊断｜越低越好")
    if has_source:
        source_axis = axes[2]
        _grouped_bars(
            source_axis,
            [_metric_label(metric) for metric in source_metrics],
            [(evaluation.label, source_values)],
            annotate=True,
            value_formats=["percent", "percent"],
        )
        source_axis.set_ylabel("判定样本比例")
        source_axis.set_title("变化结论来源｜P0应以独立二分类为主")
    figure.suptitle(f"LEVIR-CC变化判定性能 · {evaluation.label}", fontsize=14)
    sample_count = _evaluation_sample_count(evaluation)
    count_note = f"n={sample_count:,}｜" if sample_count is not None else ""
    _add_figure_note(
        figure,
        f"{count_note}内部指标｜Recall表示真实变化被发现的比例；FNR表示真实变化被漏掉的比例。",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    return figure


def _levir_caption_metrics(plt: Any, evaluation: NamedEvaluation) -> Any | None:
    definitions = (
        "bleu_1_approx",
        "bleu_4_approx",
        "rouge_l_f1_approx",
        "meteor_exact_approx",
        "chrf_approx",
        "cider_d_single_reference_approx",
    )
    all_values = [
        _metric_value(evaluation.summary, "change_detection", name) for name in definitions
    ]
    positive_values = [
        _metric_value(evaluation.summary, "change_detection", f"positive_change_{name}")
        for name in definitions
    ]
    if not any(value is not None for value in all_values + positive_values):
        return None
    figure, axis = plt.subplots(figsize=(10, 5.3))
    _grouped_bars(
        axis,
        [_metric_label(metric) for metric in definitions],
        [("全部样本", all_values), ("真实有变化子集", positive_values)],
        annotate=True,
        value_formats=[METRIC_DISPLAY[metric].value_format for metric in definitions],
    )
    axis.set_ylabel("文本质量得分")
    axis.set_title(f"LEVIR-CC变化描述质量 · {evaluation.label}｜越高越好")
    _add_figure_note(
        figure,
        "所有文本指标均为内部近似实现；有变化子集用于排除大量无变化模板回答对结果的影响。｜"
        + _evaluation_context([evaluation], "内部近似指标"),
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def _latency_context(summary: dict[str, Any]) -> tuple[Any, ...] | None:
    context = summary.get("overall", {}).get("latency_context", {})
    if not isinstance(context, dict) or context.get("status") != "resolved":
        return None
    semantics = context.get("semantics")
    batch_size = context.get("eval_batch_size")
    grouped = context.get("group_by_task")
    if not semantics:
        return None
    return semantics, batch_size, grouped


def _latency_semantics_display(semantics: Any) -> str:
    labels = {
        "batch_amortized_per_sample": "批处理摊销后的单样本延迟",
        "per_sample": "逐样本延迟",
    }
    return labels.get(str(semantics), str(semantics))


def _latency(plt: Any, evaluations: list[NamedEvaluation]) -> tuple[Any | None, str | None]:
    groups: dict[tuple[Any, ...], list[NamedEvaluation]] = {}
    for evaluation in evaluations:
        context = _latency_context(evaluation.summary)
        values = [
            _overall_metric(evaluation.summary, metric)
            for metric in ("latency_ms_mean", "latency_ms_p50", "latency_ms_p95")
        ]
        if context is not None and all(value is not None for value in values):
            groups.setdefault(context, []).append(evaluation)
    compatible = [(context, group) for context, group in groups.items() if len(group) >= 2]
    if not compatible:
        return None, "Latency chart skipped: fewer than two evaluations share one resolved context."
    context, group = max(compatible, key=lambda item: len(item[1]))
    definitions = (
        ("latency_ms_mean", _metric_label("latency_ms_mean")),
        ("latency_ms_p50", _metric_label("latency_ms_p50")),
        ("latency_ms_p95", _metric_label("latency_ms_p95")),
    )
    series = [
        (
            evaluation.label,
            [_overall_metric(evaluation.summary, metric) for metric, _ in definitions],
        )
        for evaluation in group
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    _grouped_bars(
        axis,
        [label for _, label in definitions],
        series,
        rate_axis=False,
        annotate=True,
        value_formats=["latency1"] * 3,
    )
    axis.set_ylabel("单样本延迟（毫秒）")
    grouped_text = "按任务分组" if context[2] else "未按任务分组"
    axis.set_title(
        f"推理延迟对比｜越低越好｜{_latency_semantics_display(context[0])}，"
        f"批大小={context[1]}，{grouped_text}"
    )
    _add_figure_note(
        figure,
        "仅比较测量语义、批大小和任务分组方式完全一致的结果；P95反映长尾延迟。｜"
        + _evaluation_context(group, "效率指标"),
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    omitted = len(evaluations) - len(group)
    note = (
        f"Latency chart omitted {omitted} evaluation(s) with incompatible contexts."
        if omitted
        else None
    )
    return figure, note


def _validate_formats(formats: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for image_format in formats:
        value = image_format.strip().lower()
        if value not in {"png", "svg"}:
            raise PlottingError(f"Unsupported image format: {image_format!r}; use png or svg.")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise PlottingError("At least one output format is required.")
    return tuple(normalized)


def _validate_output_directory(output_dir: Path, *, overwrite: bool) -> Path:
    destination = output_dir.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise PlottingError(f"Output path exists and is not a directory: {destination}")
    if destination.is_dir() and any(destination.iterdir()) and not overwrite:
        raise PlottingError(
            f"Output directory is not empty: {destination}; pass --overwrite to replace "
            "generated files explicitly."
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def plot_evaluation_results(
    evaluation_specs: Iterable[str],
    comparison_specs: Iterable[str],
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("png", "svg"),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate every applicable v1.5/v1.6 evaluation figure and a trace manifest."""

    normalized_formats = _validate_formats(formats)
    evaluations = load_evaluations(evaluation_specs)
    comparisons = load_comparisons(comparison_specs)
    evaluation_versions = {
        str(evaluation.summary.get("contract_version")) for evaluation in evaluations
    }
    if len(evaluation_versions) != 1:
        raise PlottingError(
            "All evaluation inputs in one plot run must use the same contract_version; "
            f"received {sorted(evaluation_versions)}."
        )
    evaluation_contract = next(iter(evaluation_versions))
    for comparison in comparisons:
        required = str(comparison.summary.get("required_contract_version"))
        if required != evaluation_contract:
            raise PlottingError(
                f"Comparison {comparison.label} requires contract {required!r}, but the "
                f"evaluation inputs use {evaluation_contract!r}."
            )
    destination = _validate_output_directory(Path(output_dir), overwrite=overwrite)
    matplotlib, plt, cjk_font = _prepare_matplotlib()
    diagnostics = {evaluation.label: _diagnostic_rows(evaluation) for evaluation in evaluations}
    generated: dict[str, list[str]] = {}
    skipped: list[dict[str, str]] = []

    def render(stem: str, figure: Any | None, reason: str) -> None:
        if figure is None:
            skipped.append({"figure": stem, "reason": reason})
            return
        paths = _save_figure(figure, destination, stem, normalized_formats)
        plt.close(figure)
        generated[stem] = [path.name for path in paths]

    render(
        "task_sample_distribution",
        _task_distribution(plt, evaluations),
        "No VRSBench task distribution is available.",
    )
    render(
        "vrsbench_core_metrics",
        _core_metrics(plt, evaluations),
        "No VRSBench core task metrics are available.",
    )
    render(
        "grounding_iou_cdf",
        _grounding_cdf(plt, evaluations, diagnostics),
        "No evaluated_predictions.jsonl with per-sample grounding IoU is available.",
    )
    render(
        "counting_error_distribution",
        _counting_errors(plt, evaluations, diagnostics),
        "No evaluated_predictions.jsonl with parsed counting errors is available.",
    )
    render(
        "vqa_accuracy_by_type",
        _qa_type_accuracy(plt, evaluations),
        "No QA-type summaries are available.",
    )
    render(
        "caption_quality_and_length",
        _caption_quality(plt, evaluations),
        "No VRSBench caption summaries are available.",
    )
    render(
        "semantic_reference_text_diagnostics",
        _semantic_diagnostics(plt, evaluations),
        "No implemented reference-text semantic metrics are available.",
    )
    for comparison in comparisons:
        render(
            f"paired_improvement_ci_{comparison.label}",
            _paired_improvement(plt, comparison),
            f"Comparison {comparison.label} has no supported confidence intervals.",
        )
        render(
            f"win_tie_loss_{comparison.label}",
            _win_tie_loss(plt, comparison),
            f"Comparison {comparison.label} has no supported win/tie/loss metrics.",
        )
    levir = _levir_evaluation(evaluations)
    render(
        "levir_cc_confusion_matrix",
        _levir_confusion(plt, levir) if levir is not None else None,
        "No LEVIR-CC change-detection summary is available.",
    )
    render(
        "levir_cc_binary_metrics",
        _levir_binary_metrics(plt, levir) if levir is not None else None,
        "No LEVIR-CC binary metrics are available.",
    )
    render(
        "levir_cc_caption_metrics",
        _levir_caption_metrics(plt, levir) if levir is not None else None,
        "No LEVIR-CC change-caption metrics are available.",
    )
    latency_figure, latency_note = _latency(plt, evaluations)
    render(
        "inference_latency",
        latency_figure,
        latency_note or "No comparable resolved latency contexts are available.",
    )
    if latency_figure is not None and latency_note:
        skipped.append({"figure": "inference_latency_partial_input", "reason": latency_note})

    if not generated:
        raise PlottingError("No applicable figures could be generated from the supplied inputs.")

    manifest = {
        "schema_version": "1.0",
        "implementation_version": "sat-rs-vlm-evaluation-plotting-v1.6",
        "language": "zh-CN",
        "presentation_version": PRESENTATION_VERSION,
        "metric_registry_version": METRIC_REGISTRY_VERSION,
        "font_family": cjk_font,
        "contract_versions": sorted(evaluation_versions),
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "matplotlib_version": matplotlib.__version__,
        "formats": list(normalized_formats),
        "evaluations": [
            {
                "label": evaluation.label,
                "source_directory_name": evaluation.directory.name,
                "contract_version": evaluation.summary.get("contract_version"),
                "hashes": evaluation.hashes,
            }
            for evaluation in evaluations
        ],
        "comparisons": [
            {
                "label": comparison.label,
                "source_directory_name": comparison.directory.name,
                "required_contract_version": comparison.summary.get("required_contract_version"),
                "hashes": comparison.hashes,
            }
            for comparison in comparisons
        ],
        "generated": generated,
        "skipped": skipped,
        "remote_write_performed": False,
    }
    manifest_path = destination / "plot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"output_dir": destination, "manifest": manifest_path, "generated": generated}
