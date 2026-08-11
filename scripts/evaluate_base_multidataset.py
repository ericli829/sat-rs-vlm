"""Load one base Qwen3-VL model and evaluate multiple datasets sequentially."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

if __package__:
    from scripts.evaluate_rs_vlm import evaluate, load_model, load_yaml, resolve_project_path
else:
    from evaluate_rs_vlm import evaluate, load_model, load_yaml, resolve_project_path

from sat_rs_vlm.configuration.environment import expand_environment
from sat_rs_vlm.training.utils import safe_import_model_dependencies

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_METRICS = {
    "continuous_mean_iou",
    "acc_at_0_5",
    "exact_count_accuracy",
    "accuracy_within_1",
    "normalized_accuracy",
    "token_f1",
    "rouge_l_f1_approx",
    "chrf_approx",
    "binary_accuracy",
    "balanced_accuracy",
    "change_precision",
    "change_recall",
    "change_f1",
    "matthews_correlation_coefficient",
    "cohen_kappa",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _load_workflow(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Multi-dataset config must be a mapping: {path}")
    return dict(expand_environment(payload, environ=os.environ, allow_unresolved=False))


def _metric_values(payload: Any, prefix: tuple[str, ...] = ()) -> dict[str, float]:
    values: dict[str, float] = {}
    if isinstance(payload, dict):
        value = payload.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and prefix:
            if prefix[-1] in SUMMARY_METRICS:
                values[".".join(prefix)] = float(value)
        for key, child in payload.items():
            if key != "value":
                values.update(_metric_values(child, (*prefix, str(key))))
    elif isinstance(payload, list):
        for index, child in enumerate(payload):
            values.update(_metric_values(child, (*prefix, str(index))))
    return values


def _write_summary(report: dict[str, Any], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "base_multidataset_summary.json"
    markdown_path = destination / "base_multidataset_summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Base Model Multi-dataset Evaluation",
        "",
        f"- Model: `{report['model_source']}`",
        f"- Datasets: {', '.join(report['runs'])}",
        "",
    ]
    for name, run in report["runs"].items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Samples: {run['sample_count']}",
                f"- Output: `{run['output_dir']}`",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )
        lines.extend(f"| {metric} | {value:.6f} |" for metric, value in run["metrics"].items())
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    workflow = _load_workflow(config_path)
    evaluations = list(workflow.get("evaluations", []))
    if len(evaluations) < 2:
        raise ValueError("Multi-dataset evaluation requires at least two evaluations")
    output_root = (
        output_override.expanduser().resolve()
        if output_override is not None
        else resolve_project_path(str(workflow["output_dir"]))
    )

    loaded: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    for entry in evaluations:
        name = str(entry["name"])
        child_path = resolve_project_path(str(entry["config"]))
        child_config = load_yaml(child_path)
        loaded.append((dict(entry), child_path, child_config))
        if not name.strip():
            raise ValueError("Evaluation names must not be empty")

    first_model_config = loaded[0][2].get("model", {})
    for entry, child_path, child_config in loaded[1:]:
        if child_config.get("model", {}) != first_model_config:
            raise ValueError(
                f"All evaluations must use the same base model configuration: {child_path}"
            )

    modules = safe_import_model_dependencies(require_bitsandbytes=False)
    model, processor = load_model(loaded[0][2], modules)
    runs: dict[str, Any] = {}
    for entry, child_path, child_config in loaded:
        name = str(entry["name"])
        max_samples = entry.get("max_eval_samples")
        if max_samples is not None:
            child_config.setdefault("data", {})["max_eval_samples"] = int(max_samples)
        result = evaluate(
            child_path,
            output_dir=output_root / name,
            batch_size_override=(
                int(entry["batch_size"]) if entry.get("batch_size") is not None else None
            ),
            loaded_model=model,
            loaded_processor=processor,
            loaded_modules=modules,
            config_override=child_config,
        )
        metrics_payload = json.loads(Path(result["summary_file"]).read_text(encoding="utf-8"))
        runs[name] = {
            "config": str(child_path),
            "output_dir": str(output_root / name),
            "sample_count": result["sample_count"],
            "batch_size": result["batch_size"],
            "metrics": _metric_values(metrics_payload),
            "summary_file": result["summary_file"],
            "evaluation_dir": result["evaluation_dir"],
        }
    report = {
        "schema_version": "1.0",
        "model_source": str(first_model_config.get("base_model")),
        "runs": runs,
    }
    _write_summary(report, output_root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args.config, args.output_dir)
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
