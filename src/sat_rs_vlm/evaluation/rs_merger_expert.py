"""Counting Expert composite loading, hard routing, and frozen-protocol metrics."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from sat_rs_vlm.data.object_adapter_v0 import count_bin, extract_answer
from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.data.task_protocol import parse_count
from sat_rs_vlm.models.rs_merger_expert import RSMergerExpertController, route_for_task


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _count_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[float] = []
    signed: list[float] = []
    exact = 0
    within_one = 0
    for row in rows:
        reference = row.get("parsed_reference")
        prediction = row.get("parsed_prediction")
        if not isinstance(reference, int) or not isinstance(prediction, int):
            continue
        difference = float(prediction - reference)
        error = abs(difference)
        errors.append(error)
        signed.append(difference)
        exact += int(error == 0)
        within_one += int(error <= 1)
    n = len(rows)
    parsed = len(errors)
    return {
        "n": n,
        "parse_rate": parsed / n if n else None,
        "exact": exact / n if n else None,
        "within_1": within_one / n if n else None,
        "mae": _mean(errors),
        "rmse": math.sqrt(sum(value * value for value in errors) / parsed) if parsed else None,
        "bias": _mean(signed),
    }


def summarize_counting_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counting = [row for row in rows if str(row.get("task_type", "")).lower() == "counting"]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in counting:
        grouped[str(row.get("count_bin", "unknown"))].append(row)
    return {
        "schema_version": "1.0",
        "metrics_protocol": "existing_parse_count_with_fixed_reference_bins",
        "overall": _count_metrics(counting),
        "count_bins": {
            name: _count_metrics(grouped.get(name, [])) for name in ("0-2", "3-5", "6-10", "11+")
        },
    }


def _delta(value: Any, baseline: Any) -> float | None:
    return (
        float(value) - float(baseline)
        if isinstance(value, (int, float)) and isinstance(baseline, (int, float))
        else None
    )


def render_summary(metrics: Mapping[str, Any], baseline: Mapping[str, Any] | None = None) -> str:
    overall = dict(metrics["overall"])
    dense = dict(metrics["count_bins"]["6-10"])
    lines = [
        "# RS Counting Merger Expert Evaluation",
        "",
        f"- n: {overall.get('n')}",
        f"- parse rate: {overall.get('parse_rate')}",
        f"- exact: {overall.get('exact')}",
        f"- within 1: {overall.get('within_1')}",
        f"- MAE: {overall.get('mae')}",
        f"- RMSE: {overall.get('rmse')}",
        f"- bias: {overall.get('bias')}",
        "",
        "## Count bins",
        "",
        "| bin | n | exact | within 1 | MAE | RMSE | bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics["count_bins"].items():
        lines.append(
            f"| {name} | {row.get('n')} | {row.get('exact')} | {row.get('within_1')} | "
            f"{row.get('mae')} | {row.get('rmse')} | {row.get('bias')} |"
        )
    if baseline is not None:
        base_overall = dict(baseline["overall"])
        base_dense = dict(baseline["count_bins"]["6-10"])
        lines.extend(
            [
                "",
                "## Delta versus C0/R1",
                "",
                f"- delta exact: {_delta(overall.get('exact'), base_overall.get('exact'))}",
                "- delta within 1: "
                f"{_delta(overall.get('within_1'), base_overall.get('within_1'))}",
                f"- delta MAE: {_delta(overall.get('mae'), base_overall.get('mae'))}",
                f"- delta bias: {_delta(overall.get('bias'), base_overall.get('bias'))}",
                f"- delta 6-10 MAE: {_delta(dense.get('mae'), base_dense.get('mae'))}",
            ]
        )
    return "\n".join(lines) + "\n"


def update_experiment_matrix(
    matrix_path: str | Path,
    *,
    experiment: str,
    architecture: str,
    training_summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    """Update a machine-backed C0/C1/C2/C3 comparison table."""

    destination = Path(matrix_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records_path = destination.with_suffix(".json")
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.is_file() else {}
    overall = dict(metrics["overall"])
    dense = dict(metrics["count_bins"]["6-10"])
    records[experiment] = {
        "architecture": architecture,
        "trainable_params": training_summary.get("trainable_params", 0),
        "peak_vram_gb": training_summary.get("peak_allocated_vram_gb"),
        "runtime_seconds": training_summary.get("elapsed_seconds"),
        "exact": overall.get("exact"),
        "within_1": overall.get("within_1"),
        "mae": overall.get("mae"),
        "bias": overall.get("bias"),
        "dense_exact": dense.get("exact"),
        "dense_within_1": dense.get("within_1"),
        "dense_mae": dense.get("mae"),
    }
    records_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# RS Merger Expert experiment matrix",
        "",
        "| experiment | architecture | trainable params | VRAM | runtime | exact | "
        "within 1 | MAE | bias | 6-10 exact | 6-10 within 1 | 6-10 MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("C0", "C1", "C2", "C3"):
        row = records.get(name, {})
        values = [
            name,
            row.get("architecture", "pending"),
            row.get("trainable_params", "pending"),
            row.get("peak_vram_gb", "pending"),
            row.get("runtime_seconds", "pending"),
            row.get("exact", "pending"),
            row.get("within_1", "pending"),
            row.get("mae", "pending"),
            row.get("bias", "pending"),
            row.get("dense_exact", "pending"),
            row.get("dense_within_1", "pending"),
            row.get("dense_mae", "pending"),
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_expert_weights(
    controller: RSMergerExpertController, checkpoint: str | Path
) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("safetensors is required to load merger experts") from exc
    root = Path(checkpoint)
    manifest_path = root / "expert_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expert manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = root / str(manifest.get("expert_weights", "expert_model.safetensors"))
    if not weights.is_file():
        raise FileNotFoundError(f"Expert weights are missing: {weights}")
    controller.load_expert_state_dict(load_file(str(weights), device="cpu"))
    return manifest


def evaluate_rows(
    *,
    model: Any,
    processor: Any,
    controller: RSMergerExpertController,
    tier_file: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    expert_variant: str,
    max_eval_samples: int | None = None,
    max_new_tokens: int = 64,
    baseline_metrics: str | Path | None = None,
    force_base: bool = False,
) -> dict[str, Any]:
    """Evaluate one sample at a time so mixed canonical tasks can hard-route safely."""

    dataset = Qwen3VLDataset(tier_file, max_samples=max_eval_samples)
    collator = Qwen3VLDataCollator(
        processor,
        max_seq_length=4096,
        image_root=image_root,
        for_generation=True,
        include_task_metadata=False,
    )
    device = next(model.parameters()).device
    model.eval()
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in dataset:
            task = sample["task_type"].strip().lower()
            active = "base" if force_base else route_for_task(task)
            controller.set_active_expert(active)
            batch = collator([sample])
            inputs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            prompt_length = int(inputs["input_ids"].shape[-1])
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            reference = str(extract_answer(sample))
            parsed_reference = parse_count(reference).value if task == "counting" else None
            parsed_prediction = parse_count(text).value if task == "counting" else None
            error = (
                abs(parsed_prediction - parsed_reference)
                if isinstance(parsed_reference, int) and isinstance(parsed_prediction, int)
                else None
            )
            predictions.append(
                {
                    "id": sample["id"],
                    "image": next(
                        (
                            str(item.get("image", ""))
                            for message in sample["messages"]
                            for item in (
                                message.get("content", [])
                                if isinstance(message.get("content"), list)
                                else []
                            )
                            if isinstance(item, Mapping) and item.get("type") == "image"
                        ),
                        "",
                    ),
                    "task_type": task,
                    "reference": reference,
                    "prediction": text,
                    "parsed_reference": parsed_reference,
                    "parsed_prediction": parsed_prediction,
                    "abs_error": error,
                    "count_bin": count_bin(parsed_reference)
                    if isinstance(parsed_reference, int)
                    else None,
                    "active_expert": active,
                    "expert_variant": expert_variant,
                }
            )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = summarize_counting_predictions(predictions)
    baseline = None
    if baseline_metrics is not None:
        baseline = json.loads(Path(baseline_metrics).read_text(encoding="utf-8"))
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(render_summary(metrics, baseline), encoding="utf-8")
    return metrics
