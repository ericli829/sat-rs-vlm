"""Counting Expert composite loading, hard routing, and frozen-protocol metrics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from sat_rs_vlm.data.object_adapter_v0 import count_bin, extract_answer, extract_prompt
from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.data.task_protocol import parse_count
from sat_rs_vlm.evaluation.counting_protocol import summarize_exact_cardinality_counting
from sat_rs_vlm.evaluation.metrics import summarize_predictions
from sat_rs_vlm.evaluation.tiers import canonical_jsonl_sha256, file_sha256
from sat_rs_vlm.models.rs_merger_expert import RSMergerExpertController, route_for_task


def summarize_counting_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score only exact-cardinality rows with the formal R1 counting protocol."""

    compatible_rows: list[Mapping[str, Any]] = []
    for source in rows:
        if (
            str(source.get("task_type", "")).lower() == "counting"
            and not str(source.get("question", "")).strip()
            and "parsed_reference" in source
        ):
            row = dict(source)
            row["question"] = "How many objects are visible?"
            row["reference"] = str(source.get("parsed_reference", ""))
            parsed_prediction = source.get("parsed_prediction")
            row["prediction"] = "" if parsed_prediction is None else str(parsed_prediction)
            compatible_rows.append(row)
        else:
            compatible_rows.append(source)
    return summarize_exact_cardinality_counting(compatible_rows)


def _numeric_delta(candidate: Any, baseline: Any) -> Any:
    if isinstance(candidate, Mapping) and isinstance(baseline, Mapping):
        return {
            key: _numeric_delta(value, baseline[key])
            for key, value in candidate.items()
            if key in baseline
        }
    return _delta(candidate, baseline)


def summarize_counting_focused_predictions(
    rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize E_COUNT_V1/V2 while retaining the merger metric schema."""

    counting = summarize_counting_predictions(rows)
    guard = summarize_predictions(
        [row for row in rows if str(row.get("task_type", "")).lower() != "counting"]
    )
    metrics: dict[str, Any] = {
        "schema_version": "1.1",
        "overall": counting["overall"],
        "count_bins": counting["count_bins"],
        "counting_focused": counting,
        "non_counting_e1_guard": guard,
    }
    if baseline is not None:
        baseline_counting = baseline.get("counting_focused", baseline)
        baseline_guard = baseline.get("non_counting_e1_guard")
        delta: dict[str, Any] = {
            "counting_focused": _numeric_delta(counting, baseline_counting),
        }
        if isinstance(baseline_guard, Mapping):
            delta["non_counting_e1_guard"] = _numeric_delta(guard, baseline_guard)
        metrics["baseline_delta"] = delta
    return metrics


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
        "# RS Counting-focused Merger Expert Evaluation",
        "",
        "## Counting-focused results",
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
    guard = metrics.get("non_counting_e1_guard", {})
    lines.extend(["", "## Non-counting E1 guard results", ""])
    if isinstance(guard, Mapping):
        guard_overall = dict(guard.get("overall", {}))
        lines.append(f"- n: {guard_overall.get('num_samples', 0)}")
        for task, task_metrics in dict(guard.get("by_task", {})).items():
            rendered = json.dumps(task_metrics, ensure_ascii=False, sort_keys=True)
            lines.append(f"- {task}: {rendered}")
    if baseline is not None:
        base_overall = dict(baseline["overall"])
        base_dense = dict(baseline["count_bins"]["6-10"])
        lines.extend(
            [
                "",
                "## Delta versus R1/C0 baseline",
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
    """Update a machine-backed C0/C1/C2/C3/C4-Wide comparison table."""

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
    for name in ("C0", "C1", "C2", "C3", "C4-Wide"):
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
    tier_manifest: str | Path | None = None,
    image_root: str | Path,
    output_dir: str | Path,
    expert_variant: str,
    max_eval_samples: int | None = None,
    max_new_tokens: int = 64,
    baseline_metrics: str | Path | None = None,
    force_base: bool = False,
    eval_batch_size: int = 1,
) -> dict[str, Any]:
    """Evaluate task-homogeneous batches so every batch has one hard route."""

    tier_provenance: dict[str, Any] | None = None
    if tier_manifest is not None:
        manifest_path = Path(tier_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tier_path = Path(tier_file)
        expected_canonical = manifest.get("canonical_jsonl_sha256")
        expected_raw = manifest.get("raw_sha256", manifest.get("final_tier_sha256"))
        actual_hash = file_sha256(tier_path)
        provenance_warnings: list[str] = []
        if expected_canonical:
            actual_canonical = canonical_jsonl_sha256(tier_path)
            if str(expected_canonical) != actual_canonical:
                raise ValueError(
                    "Counting-focused tier canonical JSONL SHA256 mismatch: "
                    f"expected {expected_canonical}, got {actual_canonical}"
                )
            if expected_raw and str(expected_raw) != actual_hash:
                provenance_warnings.append(
                    "raw SHA drift accepted because canonical JSONL SHA matches; "
                    "semantic benchmark identity is unchanged"
                )
        elif expected_raw and str(expected_raw) != actual_hash:
            raise ValueError(
                "Counting-focused tier SHA256 mismatch: "
                f"expected {expected_raw}, got {actual_hash}"
            )
        tier_provenance = {
            "manifest": manifest_path.as_posix(),
            "tier_name": manifest.get("tier_name"),
            "sha256": actual_hash,
            "raw_sha256": actual_hash,
            "canonical_jsonl_sha256": (
                canonical_jsonl_sha256(tier_path) if expected_canonical else None
            ),
            "total_rows": manifest.get("total_rows"),
            "provenance_warnings": provenance_warnings,
        }

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
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    indexed_predictions: list[tuple[int, dict[str, Any]]] = []
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, sample in enumerate(dataset):
        grouped.setdefault(sample["task_type"].strip().lower(), []).append((index, sample))
    with torch.no_grad():
        for task, task_rows in grouped.items():
            active = "base" if force_base else route_for_task(task)
            controller.set_active_expert(active)
            for start in range(0, len(task_rows), eval_batch_size):
                chunk = task_rows[start : start + eval_batch_size]
                batch = collator([sample for _index, sample in chunk])
                inputs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                prompt_width = int(inputs["input_ids"].shape[-1])
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                texts = processor.batch_decode(
                    generated[:, prompt_width:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for (source_index, sample), text in zip(chunk, texts, strict=True):
                    reference = str(extract_answer(sample))
                    parsed_reference = parse_count(reference).value if task == "counting" else None
                    parsed_prediction = parse_count(text).value if task == "counting" else None
                    error = (
                        abs(parsed_prediction - parsed_reference)
                        if isinstance(parsed_reference, int) and isinstance(parsed_prediction, int)
                        else None
                    )
                    indexed_predictions.append(
                        (
                            source_index,
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
                                "question": extract_prompt(sample),
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
                            },
                        )
                    )
    predictions = [row for _index, row in sorted(indexed_predictions, key=lambda item: item[0])]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    baseline = None
    if baseline_metrics is not None:
        baseline = json.loads(Path(baseline_metrics).read_text(encoding="utf-8"))
    metrics = summarize_counting_focused_predictions(predictions, baseline)
    if tier_provenance is not None:
        metrics["tier_provenance"] = tier_provenance
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(render_summary(metrics, baseline), encoding="utf-8")
    return metrics
