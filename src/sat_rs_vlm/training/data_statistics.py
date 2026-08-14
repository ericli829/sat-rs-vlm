"""Training-data and assistant-supervision statistics for Qwen3-VL.

The module intentionally does not load a model or execute a forward pass. Token
counts come from :class:`Qwen3VLDataCollator` diagnostics, so ``labels != -100`` is
the single source of truth for supervised assistant tokens.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.task_protocol import parse_count, parse_detection
from sat_rs_vlm.training.config import BBoxAreaThresholdConfig


def percentile(values: Sequence[float | int], probability: float) -> float | None:
    """Return a deterministic linearly interpolated percentile.

    ``probability`` is in ``[0, 1]``. Linear interpolation over ``(n - 1) * p``
    matches the common default used by NumPy while keeping this module dependency
    free.
    """

    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def numeric_summary(values: Iterable[float | int]) -> dict[str, float | int | None]:
    """Summarize one numeric population without replacing missing data with zero."""

    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "sample_count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "sample_count": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "p90": percentile(numbers, 0.90),
        "p95": percentile(numbers, 0.95),
        "max": max(numbers),
    }


def bbox_area_bucket(area: float, thresholds: BBoxAreaThresholdConfig) -> str:
    """Classify normalized bbox area using report-visible configuration."""

    if area <= thresholds.small_max:
        return "small"
    if area <= thresholds.medium_max:
        return "medium"
    return "large"


def task_counts(samples: Sequence[Mapping[str, Any]]) -> Counter[str]:
    """Count normalized task types without tokenizing or opening any image."""

    return Counter(
        str(sample.get("task_type", "unknown")).strip().lower() or "unknown" for sample in samples
    )


def stratified_sample_by_task(
    samples: Sequence[Mapping[str, Any]],
    samples_per_task: int,
    *,
    seed: int,
) -> list[Mapping[str, Any]]:
    """Draw a deterministic, bounded random sample from every observed task.

    This is used for fast token-length diagnostics. It deliberately preserves no
    original task proportion: per-task means can then be estimated with comparable
    sample support, while population task counts remain in the final report.
    """

    if samples_per_task < 1:
        raise ValueError("samples_per_task must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        task = str(sample.get("task_type", "unknown")).strip().lower() or "unknown"
        grouped[task].append(sample)
    rng = random.Random(seed)
    selected: list[Mapping[str, Any]] = []
    for task in sorted(grouped):
        candidates = list(grouped[task])
        rng.shuffle(candidates)
        selected.extend(candidates[:samples_per_task])
    rng.shuffle(selected)
    return selected


def _distribution(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {
        key: {"sample_count": count, "proportion": count / total if total else 0.0}
        for key, count in sorted(counter.items())
    }


def _assistant_text(sample: Mapping[str, Any]) -> str:
    for message in reversed(list(sample.get("messages", []))):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            )
    return ""


def _image_paths(sample: Mapping[str, Any], image_root: Path) -> list[Path]:
    paths: list[Path] = []
    for message in list(sample.get("messages", [])):
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "image":
                continue
            path = Path(str(item.get("image", ""))).expanduser()
            paths.append(path if path.is_absolute() else image_root / path)
    return paths


def _visual_grid_statistics(encoded: Mapping[str, Any], merge_size: int) -> tuple[list[int], int]:
    grid = encoded.get("image_grid_thw")
    if grid is None:
        return [], 0
    raw_rows = grid.detach().cpu().tolist() if hasattr(grid, "detach") else list(grid)
    visual_tokens: list[int] = []
    divisor = max(1, int(merge_size)) ** 2
    for row in raw_rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            continue
        product = int(row[0]) * int(row[1]) * int(row[2])
        visual_tokens.append(product // divisor)
    return visual_tokens, sum(visual_tokens)


def _processor_merge_size(collator: Qwen3VLDataCollator) -> int:
    image_processor = getattr(collator.processor, "image_processor", None)
    value = getattr(image_processor, "merge_size", None)
    if value is None:
        value = getattr(image_processor, "spatial_merge_size", 2)
    try:
        return int(value if value is not None else 2)
    except (TypeError, ValueError):
        return 2


def analyze_training_data(
    samples: Sequence[Mapping[str, Any]],
    collator: Qwen3VLDataCollator,
    *,
    image_root: str | Path,
    bbox_thresholds: BBoxAreaThresholdConfig | None = None,
    inspect_images: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_every: int = 100,
    population_task_counts: Mapping[str, int] | None = None,
    analysis_selection: Mapping[str, Any] | None = None,
    training_sampling_mode: str = "uniform",
    task_sampling_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Analyze source/task composition, exact supervision, and visual properties.

    Parameters:
        samples: Normalized Qwen3-VL samples containing ``messages`` and metadata.
        collator: The same configured Collator used by training.
        image_root: Root used to resolve relative image paths.
        bbox_thresholds: Normalized area thresholds for small/medium/large boxes.
        inspect_images: When false, image dimensions are reported as unavailable.
        progress_callback: Optional callback receiving ``(processed, total)``.
        progress_every: Callback interval in samples. The final sample is always
            reported when a callback is configured.
        population_task_counts: Optional full-dataset task counts when a smaller
            analysis sample was selected before tokenization.
        analysis_selection: JSON-safe description of the selected analysis sample.
        training_sampling_mode: Existing training sampler mode for interpretation.
        task_sampling_weights: Existing sampler weights; these are not loss weights.

    Returns:
        A JSON-serializable statistics report. Missing low-cost observations remain
        ``None``/``unavailable`` rather than being fabricated.
    """

    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    thresholds = bbox_thresholds or BBoxAreaThresholdConfig()
    root = Path(image_root)
    dataset_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    analyzed_task_counts: Counter[str] = Counter()
    token_rows: list[dict[str, Any]] = []
    per_task_tokens: dict[str, list[dict[str, Any]]] = defaultdict(list)
    class_counts: Counter[str] = Counter()
    area_buckets: Counter[str] = Counter()
    bbox_areas: list[float] = []
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    bbox_aspects: list[float] = []
    counts: list[int] = []
    count_frequency: Counter[str] = Counter()
    image_widths: list[int] = []
    image_heights: list[int] = []
    image_aspects: list[float] = []
    image_errors: list[dict[str, str]] = []
    visual_grid_tokens: list[int] = []
    visual_grid_rows: list[list[int]] = []
    merge_size = _processor_merge_size(collator)

    total_samples = len(samples)
    for index, sample in enumerate(samples, start=1):
        metadata_value = sample.get("metadata", {})
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        dataset = str(metadata.get("dataset") or metadata.get("training_source") or "unknown")
        source = str(metadata.get("training_source") or dataset)
        task = str(sample.get("task_type", "unknown")).strip().lower() or "unknown"
        dataset_counts[dataset] += 1
        source_counts[source] += 1
        analyzed_task_counts[task] += 1

        diagnostics = collator.tokenization_diagnostics(dict(sample))
        row = {
            "id": str(sample.get("id", "unknown")),
            "prompt_tokens": int(diagnostics["prompt_tokens"]),
            "assistant_tokens": int(diagnostics["assistant_tokens"]),
            "total_tokens": int(diagnostics["total_tokens"]),
            "uncapped_total_tokens": int(diagnostics["uncapped_total_tokens"]),
            "truncated": bool(diagnostics["truncated"]),
            "assistant_truncated": bool(diagnostics["assistant_truncated"]),
        }
        token_rows.append(row)
        per_task_tokens[task].append(row)

        encoded = diagnostics["encoded"]
        grid_values, visual_count = _visual_grid_statistics(encoded, merge_size)
        if grid_values:
            visual_grid_tokens.append(visual_count)
            grid = encoded.get("image_grid_thw")
            raw_rows = grid.detach().cpu().tolist() if hasattr(grid, "detach") else list(grid)
            visual_grid_rows.extend([list(map(int, item)) for item in raw_rows])

        answer = _assistant_text(sample)
        if task == "detection":
            parsed = parse_detection(answer)
            if parsed is not None and parsed.valid_coordinate_range:
                x_min, y_min, x_max, y_max = parsed.bbox
                width = x_max - x_min
                height = y_max - y_min
                area = width * height
                class_counts[parsed.label] += 1
                bbox_widths.append(width)
                bbox_heights.append(height)
                bbox_areas.append(area)
                bbox_aspects.append(width / height)
                area_buckets[bbox_area_bucket(area, thresholds)] += 1
        elif task == "counting":
            parsed_count = parse_count(answer).value
            if parsed_count is not None:
                counts.append(parsed_count)
                if parsed_count <= 4:
                    bucket = str(parsed_count)
                elif parsed_count < 10:
                    bucket = "5+"
                else:
                    bucket = "10+"
                count_frequency[bucket] += 1

        if inspect_images:
            for image_path in _image_paths(sample, root):
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                    image_widths.append(width)
                    image_heights.append(height)
                    if height > 0:
                        image_aspects.append(width / height)
                except (OSError, ValueError) as exc:
                    image_errors.append({"path": str(image_path), "error": str(exc)})

        if progress_callback is not None and (
            index % progress_every == 0 or index == total_samples
        ):
            progress_callback(index, total_samples)

    total = total_samples
    total_supervised = sum(int(row["assistant_tokens"]) for row in token_rows)
    task_token_summary: dict[str, Any] = {}
    for task, rows in sorted(per_task_tokens.items()):
        supervised = [int(row["assistant_tokens"]) for row in rows]
        task_total = sum(supervised)
        task_token_summary[task] = {
            "sample_count": len(rows),
            "total_supervised_tokens": task_total,
            "supervised_token_share": (task_total / total_supervised if total_supervised else None),
            **{
                key: value
                for key, value in numeric_summary(supervised).items()
                if key != "sample_count"
            },
        }

    population_counts = Counter(
        {
            str(task): int(count)
            for task, count in (population_task_counts or task_counts(samples)).items()
        }
    )
    population_total = sum(population_counts.values())
    sampler_weights = {
        str(task): float(weight) for task, weight in (task_sampling_weights or {}).items()
    }
    sampling_mass = {
        task: count * sampler_weights.get(task, 1.0) for task, count in population_counts.items()
    }
    total_sampling_mass = sum(sampling_mass.values())
    estimated_contribution: dict[str, Any] = {}
    estimates: dict[str, float] = {}
    for task, values in task_token_summary.items():
        mean_tokens = values["mean"]
        population_count = population_counts.get(task, 0)
        estimated = float(mean_tokens) * population_count if mean_tokens is not None else None
        if estimated is not None:
            estimates[task] = estimated
        estimated_contribution[task] = {
            "population_sample_count": population_count,
            "population_sample_share": (
                population_count / population_total if population_total else None
            ),
            "sampler_weight": sampler_weights.get(task, 1.0),
            "estimated_sample_draw_share": (
                sampling_mass.get(task, 0.0) / total_sampling_mass if total_sampling_mass else None
            ),
            "analyzed_sample_count": values["sample_count"],
            "mean_supervised_tokens": mean_tokens,
            "estimated_supervised_tokens": estimated,
        }
    estimated_total = sum(estimates.values())
    for _task, values in estimated_contribution.items():
        token_share = (
            float(values["estimated_supervised_tokens"]) / estimated_total
            if values["estimated_supervised_tokens"] is not None and estimated_total
            else None
        )
        sample_share = values["population_sample_share"]
        values["estimated_supervised_token_share"] = token_share
        values["token_share_minus_sample_share"] = (
            token_share - float(sample_share)
            if token_share is not None and sample_share is not None
            else None
        )

    truncated = sum(bool(row["truncated"]) for row in token_rows)
    assistant_truncated = sum(bool(row["assistant_truncated"]) for row in token_rows)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "sample_count": total,
        "dataset_statistics": _distribution(dataset_counts, total),
        "training_source_statistics": _distribution(source_counts, total),
        "task_statistics": _distribution(analyzed_task_counts, total),
        "analysis_selection": dict(analysis_selection or {"mode": "full_dataset"}),
        "population_task_statistics": _distribution(population_counts, population_total),
        "supervised_token_statistics": {
            "definition": "labels != -100",
            "total_supervised_tokens": total_supervised,
            "by_task": task_token_summary,
            "estimated_supervised_token_exposure": {
                "estimation_method": (
                    "population task count multiplied by analyzed mean supervised tokens"
                ),
                "purpose": "token budget, truncation, and supervision-density diagnostic",
                "not_a_task_loss_weight": True,
                "training_sampling_mode": training_sampling_mode,
                "task_sampling_weights": sampler_weights,
                "task_sampling_weights_are_loss_weights": False,
                "by_task": estimated_contribution,
            },
            "loss_weighting_interpretation": {
                "model_loss_reduction": "mean over labels != -100 within each model call",
                "task_level_control": "sampler draw frequency, not token-count totals",
                "task_sampling_weights_are_loss_weights": False,
                "batch_size_caveat": (
                    "With one sample per micro-batch, each selected sample contributes one "
                    "mean-reduced loss. With mixed multi-sample micro-batches, supervised "
                    "tokens are mean-reduced across that micro-batch."
                ),
            },
        },
        "sequence_statistics": {
            "prompt_tokens": numeric_summary(int(row["prompt_tokens"]) for row in token_rows),
            "assistant_tokens": numeric_summary(int(row["assistant_tokens"]) for row in token_rows),
            "total_tokens": numeric_summary(int(row["total_tokens"]) for row in token_rows),
        },
        "truncation_statistics": {
            "max_seq_length": collator.max_seq_length,
            "method": "same processor/chat template; capped vs truncation=False encoding",
            "truncated_samples": truncated,
            "truncation_rate": truncated / total if total else None,
            "assistant_truncated_samples": assistant_truncated,
            "assistant_truncation_rate": assistant_truncated / total if total else None,
        },
        "detection_statistics": {
            "bbox_protocol": "label+bbox normalized_0_1",
            "bbox_area_thresholds": thresholds.model_dump(),
            "class_frequency": dict(sorted(class_counts.items())),
            "bbox_area_bucket_frequency": dict(sorted(area_buckets.items())),
            "bbox_area": numeric_summary(bbox_areas),
            "bbox_width": numeric_summary(bbox_widths),
            "bbox_height": numeric_summary(bbox_heights),
            "bbox_aspect_ratio": numeric_summary(bbox_aspects),
        },
        "counting_statistics": {
            "bucket_definition": {"5+": "5-9", "10+": ">=10"},
            "frequency": {
                key: count_frequency.get(key, 0) for key in ("0", "1", "2", "3", "4", "5+", "10+")
            },
            **numeric_summary(counts),
        },
        "image_statistics": {
            "status": "ok" if inspect_images else "unavailable",
            "unavailable_reason": None if inspect_images else "inspect_images=false",
            "width": numeric_summary(image_widths),
            "height": numeric_summary(image_heights),
            "aspect_ratio": numeric_summary(image_aspects),
            "read_error_count": len(image_errors),
            "read_errors": image_errors,
        },
        "visual_token_statistics": {
            "status": "ok" if visual_grid_tokens else "unavailable",
            "unavailable_reason": (
                None if visual_grid_tokens else "processor did not return image_grid_thw"
            ),
            "processor_spatial_merge_size": merge_size,
            "approximation": "product(grid_thw) / spatial_merge_size^2",
            "image_grid_thw": visual_grid_rows,
            "approximate_visual_tokens": numeric_summary(visual_grid_tokens),
        },
    }
    return report


def statistics_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise human-readable companion to ``summary.json``."""

    lines = [
        "# Training Data Statistics",
        "",
        f"- Samples: {report.get('sample_count', 0)}",
        "- Supervision definition: `labels != -100`",
        "",
        "## Task Distribution",
        "",
        "| Task | Samples | Proportion |",
        "|---|---:|---:|",
    ]
    for task, values in dict(report.get("task_statistics", {})).items():
        lines.append(f"| {task} | {values['sample_count']} | {float(values['proportion']):.4f} |")
    supervision = dict(report.get("supervised_token_statistics", {}))
    exposure = dict(supervision.get("estimated_supervised_token_exposure", {}))
    interpretation = dict(supervision.get("loss_weighting_interpretation", {}))
    truncation = dict(report.get("truncation_statistics", {}))
    lines.extend(
        [
            "",
            "## Supervision Weighting Interpretation",
            "",
            "- Task-level training weight is controlled by sampler draw frequency, not by "
            "supervised-token totals.",
            "- `task_sampling_weights` are sampling weights, not loss weights.",
            "- Model loss reduction: "
            f"`{interpretation.get('model_loss_reduction', 'unavailable')}`.",
            "- Supervised-token exposure below is a token-budget/truncation diagnostic, "
            "not an effective task-loss weighting table.",
            "",
            "| Task | Population share | Estimated sampler draw share | Token exposure share |",
            "|---|---:|---:|---:|",
        ]
    )
    for task, values in dict(exposure.get("by_task", {})).items():
        population_share = values.get("population_sample_share")
        draw_share = values.get("estimated_sample_draw_share")
        token_share = values.get("estimated_supervised_token_share")
        lines.append(
            "| "
            f"{task} | {float(population_share):.4f} | {float(draw_share):.4f} | "
            f"{float(token_share):.4f} |"
            if population_share is not None and draw_share is not None and token_share is not None
            else f"| {task} | unavailable | unavailable | unavailable |"
        )
    lines.extend(
        [
            "",
            "## Truncation",
            "",
            f"- Max sequence length: {truncation.get('max_seq_length')}",
            f"- Truncated samples: {truncation.get('truncated_samples')}",
            f"- Assistant-truncated samples: {truncation.get('assistant_truncated_samples')}",
            "",
            "## Detection Area Buckets",
            "",
            "Thresholds: `"
            + json.dumps(
                dict(report.get("detection_statistics", {})).get("bbox_area_thresholds", {}),
                ensure_ascii=False,
            )
            + "`",
            "",
            "## Notes",
            "",
            "Visual token counts are processor-grid approximations. "
            "Unavailable fields are kept explicit.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_statistics_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write JSON and Markdown reports and return their paths."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_json = destination / "summary.json"
    summary_md = destination / "summary.md"
    summary_json.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_md.write_text(statistics_markdown(report), encoding="utf-8")
    return {"summary_json": summary_json, "summary_md": summary_md}
