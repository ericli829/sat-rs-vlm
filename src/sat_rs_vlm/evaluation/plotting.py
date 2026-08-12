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


COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
RATE_LIMITS = (0.0, 1.0)
SUPPORTED_CONTRACT_VERSIONS = frozenset({"1.5", "1.6"})

TASK_DISPLAY = {
    "captioning": "Caption",
    "counting": "Counting",
    "detection": "Grounding",
    "scene_classification": "Scene classification",
    "vqa": "VQA",
    "change_detection": "LEVIR-CC",
}

CORE_METRICS: dict[str, tuple[tuple[str, str], ...]] = {
    "detection": (
        ("continuous_mean_iou", "Mean IoU"),
        ("continuous_mean_generalized_iou", "Mean GIoU"),
        ("continuous_acc_at_0_5", "Acc@0.5"),
        ("continuous_acc_at_0_7", "Acc@0.7"),
    ),
    "counting_accuracy": (
        ("exact_count_accuracy", "Exact accuracy"),
        ("accuracy_within_1", "Accuracy within ±1"),
    ),
    "counting_error": (
        ("mae_on_parsed", "MAE ↓"),
        ("rmse_on_parsed", "RMSE ↓"),
    ),
    "text": (
        ("vqa.micro_normalized_accuracy", "VQA normalized accuracy"),
        ("vqa.token_f1", "VQA token F1"),
        ("scene_classification.micro_normalized_accuracy", "Scene normalized accuracy"),
        ("scene_classification.token_f1", "Scene token F1"),
    ),
    "captioning": (
        ("bleu_1_approx", "BLEU-1 Approx"),
        ("bleu_4_approx", "BLEU-4 Approx"),
        ("rouge_l_f1_approx", "ROUGE-L Approx"),
        ("chrf_approx", "chrF Approx"),
        ("cider_d_single_reference_approx", "CIDEr-D Approx"),
    ),
}

COMPARISON_METRICS: tuple[tuple[str, str, str], ...] = (
    ("detection", "iou", "Grounding IoU"),
    ("detection", "generalized_iou", "Grounding GIoU"),
    ("detection", "acc_at_0_5", "Grounding Acc@0.5"),
    ("counting", "absolute_error", "Counting abs. error"),
    ("counting", "exact_count_accuracy", "Counting exact acc."),
    ("vqa", "normalized_accuracy", "VQA normalized acc."),
    ("scene_classification", "normalized_accuracy", "Scene normalized acc."),
    ("captioning", "rouge_l_f1_approx", "Caption ROUGE-L"),
    ("captioning", "chrf_approx", "Caption chrF"),
    ("captioning", "cider_d_single_reference_approx", "Caption CIDEr-D"),
    ("change_detection", "binary_accuracy", "LEVIR binary accuracy"),
    ("change_detection", "change_f1", "LEVIR change F1"),
)

REPRESENTATIVE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("detection", "iou", "Grounding · IoU"),
    ("counting", "absolute_error", "Counting · Absolute error"),
    ("vqa", "normalized_accuracy", "VQA · Normalized accuracy"),
    (
        "scene_classification",
        "normalized_accuracy",
        "Scene · Normalized accuracy",
    ),
    ("captioning", "rouge_l_f1_approx", "Caption · ROUGE-L"),
    ("change_detection", "binary_accuracy", "LEVIR-CC · Binary accuracy"),
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


def _prepare_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlottingError(
            "Plotting requires the optional dependency: pip install -e '.[reliability-plot]'"
        ) from exc
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
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
        }
    )
    return matplotlib, plt


def _grouped_bars(
    axis: Any,
    labels: list[str],
    series: list[tuple[str, list[float | None]]],
    *,
    rate_axis: bool = True,
    annotate: bool = False,
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
        for bar, value in zip(bars, values, strict=True):
            if value is None:
                bar.set_alpha(0.12)
                continue
            if annotate:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90 if len(labels) > 5 else 0,
                )
    axis.set_xticks(centers, labels, rotation=24, ha="right")
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
        metadata: dict[str, Any] = {"Creator": "sat-rs-vlm evaluation plotting v1.5"}
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
    )
    axis.set_ylabel("Samples")
    axis.set_title("Evaluation task distribution")
    figure.tight_layout()
    return figure


def _core_metrics(plt: Any, evaluations: list[NamedEvaluation]) -> Any | None:
    applicable = [
        evaluation
        for evaluation in evaluations
        if "detection" in evaluation.summary.get("by_task", {})
    ]
    if not applicable:
        return None
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = (
        (axes[0, 0], "detection", "Grounding quality", True),
        (axes[0, 1], "counting_accuracy", "Counting accuracy", True),
        (axes[0, 2], "counting_error", "Counting error (lower is better)", False),
        (axes[1, 0], "text", "VQA and scene classification", True),
        (axes[1, 1], "captioning", "Caption quality (internal Approx)", True),
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
        plotted = _grouped_bars(axis, labels, series, rate_axis=rate_axis)
        any_panel = any_panel or plotted
        axis.set_title(title)
        axis.set_ylabel("Score" if rate_axis else "Count error")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.92,
        "Metric note\n\nRates and overlap scores: higher is better.\n"
        "MAE/RMSE: lower is better.\nCaption metrics are internal approximations.",
        va="top",
        fontsize=11,
        linespacing=1.55,
    )
    figure.suptitle("VRSBench core metrics", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
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
    axis.set_xlabel("Per-sample IoU")
    axis.set_ylabel("Cumulative proportion")
    axis.set_title("Grounding IoU empirical CDF")
    axis.legend(frameon=False)
    figure.tight_layout()
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
    error_labels = ("Exact", "±1 (not exact)", "2–3", ">3")
    direction_labels = ("Under-count", "Exact", "Over-count")
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
    _grouped_bars(axes[0], list(error_labels), error_series, annotate=True)
    axes[0].set_ylabel("Sample proportion")
    axes[0].set_title("Absolute count error bands")
    _grouped_bars(axes[1], list(direction_labels), direction_series, annotate=True)
    axes[1].set_ylabel("Sample proportion")
    axes[1].set_title("Count bias direction")
    figure.tight_layout()
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
    figure, axis = plt.subplots(figsize=(max(12, len(qa_types) * 0.85), 6.3))
    display_labels = [
        f"{name}\n(n={count:,})" for name, count in zip(qa_types, sample_counts, strict=True)
    ]
    _grouped_bars(axis, display_labels, series)
    axis.set_ylabel("Normalized accuracy")
    axis.set_title("VQA accuracy by QA type")
    figure.tight_layout()
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
    _grouped_bars(axes[0], [label for _, label in quality], series, annotate=True)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Caption quality (internal Approx)")
    length_series = [
        (
            evaluation.label,
            [_metric_value(evaluation.summary, "captioning", "length_ratio")],
        )
        for evaluation in applicable
    ]
    _grouped_bars(
        axes[1],
        ["Prediction/reference token ratio"],
        length_series,
        rate_axis=False,
        annotate=True,
    )
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="Reference parity")
    axes[1].set_ylabel("Length ratio")
    axes[1].set_title("Caption length diagnostic")
    figure.tight_layout()
    return figure


def _semantic_diagnostics(plt: Any, evaluations: list[NamedEvaluation]) -> Any | None:
    definitions = (
        ("object_precision", "Object precision"),
        ("object_recall", "Object recall"),
        ("object_f1", "Object F1"),
        ("object_omission_rate", "Object omission ↓"),
        ("reference_unsupported_object_rate", "Unsupported mention ↓"),
        ("count_consistency_accuracy", "Count consistency"),
        ("spatial_relation_f1", "Spatial relation F1"),
    )
    applicable = [
        evaluation
        for evaluation in evaluations
        if "captioning" in evaluation.summary.get("by_task", {})
    ]
    series = [
        (
            evaluation.label,
            [_semantic_metric(evaluation.summary, metric) for metric, _ in definitions],
        )
        for evaluation in applicable
    ]
    if not any(value is not None for _, values in series for value in values):
        return None
    figure, axis = plt.subplots(figsize=(12, 5.8))
    _grouped_bars(axis, [label for _, label in definitions], series)
    axis.set_ylabel("Internal diagnostic score/rate")
    axis.set_title("Reference-text semantic diagnostics (not image-grounded factuality)")
    figure.tight_layout()
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
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Direction-normalized improvement (positive is better)")
    axis.set_title(f"Paired improvement with 95% bootstrap CI · {comparison.label}")
    figure.tight_layout()
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
    axis.barh(positions, wins, color=COLORS[2], label="Win")
    axis.barh(positions, ties, left=wins, color="#BDBDBD", label="Tie")
    left = [win + tie for win, tie in zip(wins, ties, strict=True)]
    axis.barh(positions, losses, left=left, color=COLORS[5], label="Loss")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Paired sample proportion")
    axis.set_title(f"Win / tie / loss by representative metric · {comparison.label}")
    axis.legend(frameon=False, ncol=3, loc="lower right")
    figure.tight_layout()
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
    axis.set_xticks((0, 1), ("Predicted no change", "Predicted change"))
    axis.set_yticks((0, 1), ("Actual no change", "Actual change"))
    axis.set_title(f"LEVIR-CC confusion matrix · {evaluation.label}")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Samples")
    figure.tight_layout()
    return figure


def _levir_binary_metrics(plt: Any, evaluation: NamedEvaluation) -> Any | None:
    definitions = (
        ("binary_accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("change_precision", "Precision"),
        ("change_recall", "Recall"),
        ("change_f1", "F1"),
        ("matthews_correlation_coefficient", "MCC"),
        ("cohen_kappa", "Kappa"),
        ("false_positive_rate", "FPR ↓"),
        ("false_negative_rate", "FNR ↓"),
    )
    values = [
        _metric_value(evaluation.summary, "change_detection", name) for name, _ in definitions
    ]
    if not any(value is not None for value in values):
        return None
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    bars = axis.bar(
        [label for _, label in definitions],
        [float(value) if value is not None else 0.0 for value in values],
        color=[COLORS[0]] * 7 + [COLORS[5]] * 2,
    )
    for bar, value in zip(bars, values, strict=True):
        if value is not None:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score / rate")
    axis.set_title(f"LEVIR-CC binary change metrics · {evaluation.label}")
    axis.tick_params(axis="x", rotation=24)
    figure.tight_layout()
    return figure


def _levir_caption_metrics(plt: Any, evaluation: NamedEvaluation) -> Any | None:
    definitions = (
        ("bleu_1_approx", "BLEU-1"),
        ("bleu_4_approx", "BLEU-4"),
        ("rouge_l_f1_approx", "ROUGE-L"),
        ("meteor_exact_approx", "METEOR"),
        ("chrf_approx", "chrF"),
        ("cider_d_single_reference_approx", "CIDEr-D"),
    )
    all_values = [
        _metric_value(evaluation.summary, "change_detection", name) for name, _ in definitions
    ]
    positive_values = [
        _metric_value(evaluation.summary, "change_detection", f"positive_change_{name}")
        for name, _ in definitions
    ]
    if not any(value is not None for value in all_values + positive_values):
        return None
    figure, axis = plt.subplots(figsize=(10, 5.3))
    _grouped_bars(
        axis,
        [label for _, label in definitions],
        [("All samples", all_values), ("Actual-change subset", positive_values)],
        annotate=True,
    )
    axis.set_ylabel("Internal Approx score")
    axis.set_title(f"LEVIR-CC change-caption quality · {evaluation.label}")
    figure.tight_layout()
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
        ("latency_ms_mean", "Mean"),
        ("latency_ms_p50", "P50"),
        ("latency_ms_p95", "P95"),
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
    )
    axis.set_ylabel("Latency (ms per sample)")
    axis.set_title(f"Inference latency · {context[0]}, batch={context[1]}, grouped={context[2]}")
    figure.tight_layout()
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
    matplotlib, plt = _prepare_matplotlib()
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
