"""同一评测集上两个同版本结果目录的逐样本配对比较。"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sat_rs_vlm.evaluation.records import EvaluationError


class ComparisonError(EvaluationError):
    """配对评测输入不兼容或输出位置不安全。"""


@dataclass(frozen=True)
class MetricSpec:
    name: str
    sample_key: str
    higher_is_better: bool


METRIC_SPECS: dict[str, tuple[MetricSpec, ...]] = {
    "detection": (
        MetricSpec("iou", "iou", True),
        MetricSpec("generalized_iou", "generalized_iou", True),
        MetricSpec("normalized_center_distance", "normalized_center_distance", False),
        MetricSpec("acc_at_0_5", "correct_at_0_5", True),
        MetricSpec("acc_at_0_7", "correct_at_0_7", True),
        MetricSpec(
            "label_and_iou_accuracy_at_0_5",
            "label_and_iou_correct_at_0_5",
            True,
        ),
    ),
    "counting": (
        MetricSpec("absolute_error", "absolute_error", False),
        MetricSpec("absolute_percentage_error", "absolute_percentage_error", False),
        MetricSpec("exact_count_accuracy", "exact_count_correct", True),
        MetricSpec("accuracy_within_1", "within_1_correct", True),
    ),
    "vqa": (
        MetricSpec("exact_match", "exact_match", True),
        MetricSpec("normalized_accuracy", "normalized_exact_match", True),
        MetricSpec("token_f1", "token_f1", True),
        MetricSpec("normalized_edit_similarity", "normalized_edit_similarity", True),
    ),
    "scene_classification": (
        MetricSpec("exact_match", "exact_match", True),
        MetricSpec("normalized_accuracy", "normalized_exact_match", True),
        MetricSpec("token_f1", "token_f1", True),
        MetricSpec("normalized_edit_similarity", "normalized_edit_similarity", True),
    ),
    "captioning": (
        MetricSpec("bleu_1_approx", "bleu_1_approx", True),
        MetricSpec("bleu_2_approx", "bleu_2_approx", True),
        MetricSpec("bleu_3_approx", "bleu_3_approx", True),
        MetricSpec("bleu_4_approx", "bleu_4_approx", True),
        MetricSpec("rouge_l_f1_approx", "rouge_l_f1_approx", True),
        MetricSpec("meteor_exact_approx", "meteor_exact_approx", True),
        MetricSpec("chrf_approx", "chrf_approx", True),
        MetricSpec(
            "cider_d_single_reference_approx",
            "cider_d_single_reference_approx",
            True,
        ),
    ),
}

PRIMARY_METRIC = {
    "detection": "iou",
    "counting": "absolute_error",
    "vqa": "normalized_accuracy",
    "scene_classification": "normalized_accuracy",
    "captioning": "rouge_l_f1_approx",
}

TOLERANCE = 1e-12
REQUIRED_FILES = (
    "evaluated_predictions.jsonl",
    "summary.json",
    "evaluation_manifest.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComparisonError(f"JSON root must be an object: {path}")
    return payload


def _validate_evaluation_dir(path: Path, role: str) -> dict[str, Path]:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise ComparisonError(f"{role} evaluation directory does not exist: {root}")
    files = {name: root / name for name in REQUIRED_FILES}
    missing = [name for name, file in files.items() if not file.is_file()]
    if missing:
        raise ComparisonError(f"{role} evaluation directory is missing files: {missing}")
    return files


def _validate_output_dir(
    output_dir: Path,
    protected_repository: Path | None = None,
) -> Path:
    output = output_dir.expanduser().resolve()
    protected = protected_repository.expanduser().resolve() if protected_repository else None
    if protected is not None and (output == protected or protected in output.parents):
        raise ComparisonError(
            f"comparison output must not be inside protected repository: {protected}"
        )
    if output.exists():
        if not output.is_dir():
            raise ComparisonError(f"comparison output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise ComparisonError(f"comparison output already exists and is not empty: {output}")
    return output


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ComparisonError(f"{path} line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ComparisonError(f"{path} line {line_number}: row must be an object")
            yield row


def _compact_row(row: dict[str, Any], source: Path) -> dict[str, Any]:
    for key in ("id", "task_type", "prediction", "reference", "sample_metrics"):
        if key not in row:
            raise ComparisonError(f"{source}: evaluated row is missing {key}")
    if not isinstance(row["id"], str) or not row["id"].strip():
        raise ComparisonError(f"{source}: evaluated row id must be a non-empty string")
    if not isinstance(row["sample_metrics"], dict):
        raise ComparisonError(f"{source}: sample_metrics must be an object for {row['id']}")
    return {
        "id": row["id"],
        "task_type": str(row["task_type"]).strip().lower(),
        "prediction": str(row["prediction"]),
        "reference": str(row["reference"]),
        "metadata": dict(row.get("metadata", {})),
        "sample_metrics": dict(row["sample_metrics"]),
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _outcome(improvement_delta: float) -> str:
    if abs(improvement_delta) <= TOLERANCE:
        return "tie"
    return "win" if improvement_delta > 0 else "loss"


def _bootstrap_ci(
    values: list[float],
    *,
    resamples: int,
    seed: int,
) -> list[float] | None:
    if not values:
        return None
    if len(set(values)) == 1:
        return [values[0], values[0]]
    generator = random.Random(seed)
    sample_size = len(values)
    estimates: list[float] = []
    for _ in range(resamples):
        estimates.append(
            statistics.fmean(values[generator.randrange(sample_size)] for _ in range(sample_size))
        )
    estimates.sort()
    lower_index = int(0.025 * (resamples - 1))
    upper_index = int(0.975 * (resamples - 1) + 0.999999999)
    upper_index = min(upper_index, resamples - 1)
    return [estimates[lower_index], estimates[upper_index]]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def compare_evaluations(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
    *,
    protected_repository: str | Path | None = None,
    bootstrap_resamples: int = 1000,
    seed: int = 20260806,
) -> dict[str, Path]:
    if bootstrap_resamples < 1:
        raise ComparisonError("bootstrap resamples must be a positive integer")
    baseline_files = _validate_evaluation_dir(Path(baseline_dir), "baseline")
    candidate_files = _validate_evaluation_dir(Path(candidate_dir), "candidate")
    protected = Path(protected_repository) if protected_repository is not None else None
    destination = _validate_output_dir(Path(output_dir), protected)

    baseline_manifest = _load_json(baseline_files["evaluation_manifest.json"])
    candidate_manifest = _load_json(candidate_files["evaluation_manifest.json"])
    baseline_summary = _load_json(baseline_files["summary.json"])
    candidate_summary = _load_json(candidate_files["summary.json"])
    for role, manifest, summary in (
        ("baseline", baseline_manifest, baseline_summary),
        ("candidate", candidate_manifest, candidate_summary),
    ):
        if str(manifest.get("contract_version")) != "1.5":
            raise ComparisonError(f"{role} manifest must use contract_version 1.5")
        if str(summary.get("contract_version")) != "1.5":
            raise ComparisonError(f"{role} summary must use contract_version 1.5")

    baseline_rows: dict[str, dict[str, Any]] = {}
    for raw in _iter_jsonl(baseline_files["evaluated_predictions.jsonl"]):
        row = _compact_row(raw, baseline_files["evaluated_predictions.jsonl"])
        if row["id"] in baseline_rows:
            raise ComparisonError(f"duplicate baseline id: {row['id']}")
        baseline_rows[row["id"]] = row

    seen_candidate: set[str] = set()
    compatibility_errors: list[dict[str, str]] = []
    unexpected_candidate = 0
    task_mismatches = 0
    reference_mismatches = 0
    paired_inputs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw in _iter_jsonl(candidate_files["evaluated_predictions.jsonl"]):
        candidate = _compact_row(raw, candidate_files["evaluated_predictions.jsonl"])
        sample_id = candidate["id"]
        if sample_id in seen_candidate:
            raise ComparisonError(f"duplicate candidate id: {sample_id}")
        seen_candidate.add(sample_id)
        baseline = baseline_rows.get(sample_id)
        if baseline is None:
            unexpected_candidate += 1
            if len(compatibility_errors) < 20:
                compatibility_errors.append({"id": sample_id, "error": "candidate_only_id"})
            continue
        incompatible = False
        if baseline["task_type"] != candidate["task_type"]:
            task_mismatches += 1
            incompatible = True
            if len(compatibility_errors) < 20:
                compatibility_errors.append({"id": sample_id, "error": "task_type_mismatch"})
        if baseline["reference"] != candidate["reference"]:
            reference_mismatches += 1
            incompatible = True
            if len(compatibility_errors) < 20:
                compatibility_errors.append({"id": sample_id, "error": "reference_mismatch"})
        if not incompatible:
            paired_inputs.append((baseline, candidate))

    missing_candidate_ids = set(baseline_rows) - seen_candidate
    for sample_id in sorted(missing_candidate_ids)[: max(0, 20 - len(compatibility_errors))]:
        compatibility_errors.append({"id": sample_id, "error": "baseline_only_id"})
    if missing_candidate_ids or unexpected_candidate or task_mismatches or reference_mismatches:
        raise ComparisonError(
            "evaluation inputs are not pair-compatible: "
            f"baseline_only={len(missing_candidate_ids)}, "
            f"candidate_only={unexpected_candidate}, task_mismatches={task_mismatches}, "
            f"reference_mismatches={reference_mismatches}, examples={compatibility_errors}"
        )

    values: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "baseline": [],
            "candidate": [],
            "raw_delta": [],
            "improvement_delta": [],
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }
    )
    task_counts: dict[str, int] = defaultdict(int)
    paired_rows: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    prediction_changed = 0

    for baseline, candidate in paired_inputs:
        task = baseline["task_type"]
        task_counts[task] += 1
        changed = baseline["prediction"] != candidate["prediction"]
        prediction_changed += int(changed)
        comparisons: dict[str, Any] = {}
        for spec in METRIC_SPECS.get(task, ()):
            baseline_value = _numeric(baseline["sample_metrics"].get(spec.sample_key))
            candidate_value = _numeric(candidate["sample_metrics"].get(spec.sample_key))
            if baseline_value is None or candidate_value is None:
                comparisons[spec.name] = {
                    "status": "not_comparable",
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                }
                continue
            raw_delta = candidate_value - baseline_value
            improvement_delta = raw_delta if spec.higher_is_better else -raw_delta
            outcome = _outcome(improvement_delta)
            bucket = values[(task, spec.name)]
            bucket["baseline"].append(baseline_value)
            bucket["candidate"].append(candidate_value)
            bucket["raw_delta"].append(raw_delta)
            bucket["improvement_delta"].append(improvement_delta)
            bucket[{"win": "wins", "tie": "ties", "loss": "losses"}[outcome]] += 1
            comparisons[spec.name] = {
                "status": "ok",
                "baseline": baseline_value,
                "candidate": candidate_value,
                "candidate_minus_baseline": raw_delta,
                "improvement_delta": improvement_delta,
                "higher_is_better": spec.higher_is_better,
                "outcome": outcome,
            }

        primary_name = PRIMARY_METRIC.get(task)
        primary = comparisons.get(primary_name or "", {})
        primary_outcome = primary.get("outcome", "not_comparable")
        paired = {
            "id": baseline["id"],
            "task_type": task,
            "reference": baseline["reference"],
            "metadata": baseline["metadata"],
            "baseline_prediction": baseline["prediction"],
            "candidate_prediction": candidate["prediction"],
            "prediction_changed": changed,
            "primary_metric": primary_name,
            "primary_outcome": primary_outcome,
            "metric_comparisons": comparisons,
        }
        paired_rows.append(paired)
        if primary_outcome == "win":
            improvements.append(paired)
        elif primary_outcome == "loss":
            regressions.append(paired)

    by_task: dict[str, Any] = {}
    for task in sorted(task_counts):
        metrics: dict[str, Any] = {}
        for spec in METRIC_SPECS.get(task, ()):
            summary_bucket: dict[str, Any] | None = values.get((task, spec.name))
            if not summary_bucket or not summary_bucket["baseline"]:
                metrics[spec.name] = {"status": "not_available", "num_samples": 0}
                continue
            baseline_mean = statistics.fmean(summary_bucket["baseline"])
            candidate_mean = statistics.fmean(summary_bucket["candidate"])
            raw_delta_mean = candidate_mean - baseline_mean
            relative = (
                100.0 * raw_delta_mean / abs(baseline_mean)
                if abs(baseline_mean) > TOLERANCE
                else None
            )
            metric_seed = seed ^ int.from_bytes(
                hashlib.sha256(f"{task}:{spec.name}".encode()).digest()[:8],
                "big",
            )
            metrics[spec.name] = {
                "status": "ok",
                "num_samples": len(summary_bucket["baseline"]),
                "higher_is_better": spec.higher_is_better,
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "candidate_minus_baseline": raw_delta_mean,
                "relative_change_percent": relative,
                "improvement_mean": statistics.fmean(summary_bucket["improvement_delta"]),
                "improvement_ci95_paired_bootstrap": _bootstrap_ci(
                    summary_bucket["improvement_delta"],
                    resamples=bootstrap_resamples,
                    seed=metric_seed,
                ),
                "wins": summary_bucket["wins"],
                "ties": summary_bucket["ties"],
                "losses": summary_bucket["losses"],
            }
        by_task[task] = {
            "num_samples": task_counts[task],
            "primary_metric": PRIMARY_METRIC.get(task),
            "metrics": metrics,
        }

    total = len(paired_rows)
    summary = {
        "schema_version": "1.0",
        "implementation_version": "standalone-paired-comparison-v1.0",
        "required_contract_version": "1.5",
        "overall": {
            "num_paired_samples": total,
            "prediction_changed": prediction_changed,
            "prediction_changed_rate": prediction_changed / total if total else None,
            "improvements": len(improvements),
            "regressions": len(regressions),
        },
        "bootstrap": {
            "method": "paired percentile bootstrap",
            "confidence_level": 0.95,
            "resamples": bootstrap_resamples,
            "seed": seed,
        },
        "by_task": by_task,
    }
    outputs = {
        "comparison_summary": destination / "comparison_summary.json",
        "paired_comparison": destination / "paired_comparison.jsonl",
        "improvements": destination / "improvements.jsonl",
        "regressions": destination / "regressions.jsonl",
        "comparison_manifest": destination / "comparison_manifest.json",
    }
    manifest = {
        "schema_version": "1.0",
        "implementation_version": "standalone-paired-comparison-v1.0",
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "baseline_directory": str(Path(baseline_dir).expanduser().resolve()),
        "candidate_directory": str(Path(candidate_dir).expanduser().resolve()),
        "baseline_contract_version": baseline_manifest.get("contract_version"),
        "candidate_contract_version": candidate_manifest.get("contract_version"),
        "baseline_hashes": {name: _sha256(path) for name, path in baseline_files.items()},
        "candidate_hashes": {name: _sha256(path) for name, path in candidate_files.items()},
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "tolerance": TOLERANCE,
        "output_files": {name: str(path) for name, path in outputs.items()},
        "remote_write_performed": False,
        "protected_repository": (
            str(Path(protected_repository).expanduser().resolve())
            if protected_repository is not None
            else None
        ),
    }

    destination.mkdir(parents=True, exist_ok=True)
    _write_json(outputs["comparison_summary"], summary)
    _write_jsonl(outputs["paired_comparison"], paired_rows)
    _write_jsonl(outputs["improvements"], improvements)
    _write_jsonl(outputs["regressions"], regressions)
    _write_json(outputs["comparison_manifest"], manifest)
    return outputs
