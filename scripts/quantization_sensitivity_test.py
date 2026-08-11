"""使用统一 Evaluation v1.5 指标运行 GPU INT8 层或组件敏感度实验。"""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sat_rs_vlm.models.qwen3vl_loader import load_qwen3vl_processor
from sat_rs_vlm.quantization.benchmark import (
    run_benchmark,
    run_variant_evaluation,
    validate_assets,
)
from sat_rs_vlm.quantization.config import load_quantization_config
from sat_rs_vlm.quantization.quantizer import create_backend
from sat_rs_vlm.quantization.sensitivity import (
    SensitivityResult,
    build_sensitivity_groups,
    build_sensitivity_report,
    calculate_sensitivity_breakdown,
    discover_linear_modules,
    discover_tied_linear_modules,
    plot_sensitivity_report,
    validate_variant_comparison,
    write_sensitivity_report,
)
from sat_rs_vlm.training.utils import safe_import_model_dependencies, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析统一 YAML、少量安全覆盖和 dry-run 参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method", choices=("component_wise", "layer_wise"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(argv)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _clear_allocator(torch: Any) -> None:
    """调用方删除最后一个模型引用后，回收 Python 与 CUDA allocator 缓存。"""

    gc.collect()
    if bool(torch.cuda.is_available()):
        torch.cuda.empty_cache()


def _safe_group_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "group"


def _write_progress(
    destination: Path,
    *,
    status: str,
    completed: int,
    total: int,
    current_group: str | None,
    results: list[SensitivityResult],
    error: str | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "completed_groups": completed,
        "total_groups": total,
        "current_group": current_group,
        "completed_group_names": [result.name for result in results],
        "error": error,
    }
    temporary = destination / "sensitivity_progress.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination / "sensitivity_progress.json")


def run_sensitivity(args: argparse.Namespace) -> dict[str, Any]:
    """执行资产检查、baseline、逐组量化评估和报告保存。"""

    overrides = {"sensitivity.method": args.method}
    config = load_quantization_config(args.config, overrides=overrides)
    backend = create_backend(config.quantization.backend)
    destination = _project_path(args.output_dir or (Path(config.output.output_dir) / "sensitivity"))
    if args.dry_run:
        config.output.output_dir = str(destination)
        return run_benchmark(
            config,
            backend,
            project_root=PROJECT_ROOT,
            dry_run=True,
        )
    if backend.name != "bnb_int8" or config.quantization.device != "cuda":
        raise ValueError("Sensitivity analysis requires backend=bnb_int8 and device=cuda")

    modules = safe_import_model_dependencies(
        require_bitsandbytes=backend.requires_bitsandbytes
    )
    torch = modules["torch"]
    dataset, _, _, image_root = validate_assets(
        config,
        backend,
        project_root=PROJECT_ROOT,
        torch=torch,
    )
    set_seed(config.benchmark.seed)
    processor = load_qwen3vl_processor(
        modules,
        str(config.model.processor_id),
        {
            "local_files_only": config.model.local_files_only,
            "trust_remote_code": config.model.trust_remote_code,
        },
    )
    config.quantization.save_artifact = False

    baseline_model = backend.load_model(config, modules, quantized=False)
    tied_linear_modules = discover_tied_linear_modules(baseline_model, torch)
    configured_skip_modules = tuple(
        dict.fromkeys((*config.sensitivity.skip_modules, *tied_linear_modules))
    )
    if tied_linear_modules:
        print(
            "[sensitivity] keeping tied Linear modules in original precision: "
            + ", ".join(tied_linear_modules),
            flush=True,
        )
    groups = build_sensitivity_groups(
        baseline_model,
        torch,
        method=config.sensitivity.method,
        layer_group_size=config.sensitivity.layer_group_size,
        layer_grouping=config.sensitivity.layer_grouping,
        include_modules=tuple(config.sensitivity.include_modules),
        skip_modules=configured_skip_modules,
        max_groups=config.sensitivity.max_groups,
    )
    all_linear_module_names = tuple(sorted(discover_linear_modules(baseline_model, torch)))
    print(
        f"[sensitivity] discovered {len(groups)} groups with "
        f"method={config.sensitivity.method}, grouping={config.sensitivity.layer_grouping}",
        flush=True,
    )
    baseline_report = run_variant_evaluation(
        variant="baseline",
        quantized=False,
        model=baseline_model,
        processor=processor,
        dataset=dataset,
        image_root=image_root,
        config=config,
        backend=backend,
        torch=torch,
        output_dir=destination,
        project_root=PROJECT_ROOT,
    )
    validate_variant_comparison(
        baseline_report,
        baseline_report,
        require_same_samples=config.sensitivity.require_same_samples,
        max_failure_rate=config.sensitivity.max_failure_rate,
    )
    del baseline_model
    _clear_allocator(torch)

    results: list[SensitivityResult] = []
    _write_progress(
        destination,
        status="running",
        completed=0,
        total=len(groups),
        current_group=None,
        results=results,
    )
    baseline_latency = baseline_report.get("latency_ms", {}).get("mean")
    baseline_parameters = int(baseline_report.get("logical_parameter_count", 0))
    try:
        for group_index, group in enumerate(groups, start=1):
            print(
                f"[sensitivity] group {group_index}/{len(groups)}: {group.name} "
                f"({len(group.module_names)} Linear modules)",
                flush=True,
            )
            _write_progress(
                destination,
                status="running",
                completed=len(results),
                total=len(groups),
                current_group=group.name,
                results=results,
            )
            target_names = set(group.module_names)
            skipped_names = tuple(
                name for name in all_linear_module_names if name not in target_names
            )
            quantized_model = backend.load_selective_model(
                config,
                modules,
                target_module_names=group.module_names,
                skipped_module_names=skipped_names,
            )
            variant_report = run_variant_evaluation(
                variant=_safe_group_name(group.name),
                quantized=True,
                model=quantized_model,
                processor=processor,
                dataset=dataset,
                image_root=image_root,
                config=config,
                backend=backend,
                torch=torch,
                output_dir=destination / "groups",
                project_root=PROJECT_ROOT,
            )
            validate_variant_comparison(
                baseline_report,
                variant_report,
                require_same_samples=config.sensitivity.require_same_samples,
                max_failure_rate=config.sensitivity.max_failure_rate,
            )
            score, deltas, task_scores = calculate_sensitivity_breakdown(
                baseline_report["metrics"],
                variant_report["metrics"],
                task_weights=dict(config.sensitivity.task_weights),
                metric_weights=dict(config.sensitivity.metric_weights),
            )
            metrics_path = Path(variant_report["evaluation_outputs"]["metrics"])
            variant_latency = variant_report.get("latency_ms", {}).get("mean")
            speedup = None
            if isinstance(baseline_latency, (int, float)) and isinstance(
                variant_latency, (int, float)
            ):
                speedup = (
                    float(baseline_latency) / float(variant_latency)
                    if variant_latency
                    else None
                )
            results.append(
                SensitivityResult(
                    name=group.name,
                    kind=group.kind,
                    module_names=group.module_names,
                    parameter_count=group.parameter_count,
                    sensitivity_score=score,
                    metric_deltas=deltas,
                    evaluation_dir=str(metrics_path.parent),
                    task_scores=task_scores,
                    completed_samples=int(variant_report.get("completed_samples", 0)),
                    failed_samples=int(variant_report.get("failed_samples", 0)),
                    parameter_fraction=(
                        group.parameter_count / baseline_parameters if baseline_parameters else None
                    ),
                    latency_mean_ms=(
                        float(variant_latency)
                        if isinstance(variant_latency, (int, float))
                        else None
                    ),
                    speedup_vs_baseline=speedup,
                )
            )
            del quantized_model
            _clear_allocator(torch)
            _write_progress(
                destination,
                status="running",
                completed=len(results),
                total=len(groups),
                current_group=None,
                results=results,
            )
    except Exception as exc:
        _write_progress(
            destination,
            status="failed",
            completed=len(results),
            total=len(groups),
            current_group=groups[len(results)].name if len(results) < len(groups) else None,
            results=results,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    baseline_metrics_path = Path(baseline_report["evaluation_outputs"]["metrics"])
    report = build_sensitivity_report(
        model_source=config.model.model_source,
        method=config.sensitivity.method,
        baseline_evaluation_dir=str(baseline_metrics_path.parent),
        results=results,
        sensitive_threshold=config.sensitivity.sensitive_threshold,
        insensitive_threshold=config.sensitivity.insensitive_threshold,
        baseline_performance={
            "latency_ms": baseline_report.get("latency_ms"),
            "completed_samples": baseline_report.get("completed_samples"),
            "failed_samples": baseline_report.get("failed_samples"),
            "logical_parameter_count": baseline_report.get("logical_parameter_count"),
        },
    )
    report["sampling"] = {
        "strategy": config.data.sampling_strategy,
        "seed": config.data.sampling_seed,
        "samples_per_task": dict(config.data.samples_per_task),
        "sample_ids": baseline_report.get("sample_ids", []),
    }
    report["grouping"] = {
        "layer_grouping": config.sensitivity.layer_grouping,
        "layer_group_size": config.sensitivity.layer_group_size,
        "include_modules": list(config.sensitivity.include_modules),
        "skip_modules": list(configured_skip_modules),
        "automatically_skipped_tied_linear_modules": list(tied_linear_modules),
    }
    report["quantization"] = {
        "backend": backend.name,
        "device": config.quantization.device,
        "llm_int8_threshold": config.quantization.llm_int8_threshold,
        "selective_gpu_quantization": backend.name == "bnb_int8",
    }
    outputs = write_sensitivity_report(report, destination)
    report["outputs"] = {name: str(path) for name, path in outputs.items()}
    if args.plot:
        figures = plot_sensitivity_report(report, destination / "figures")
        report["figures"] = [str(path) for path in figures]
        write_sensitivity_report(report, destination)
    _write_progress(
        destination,
        status="completed",
        completed=len(results),
        total=len(groups),
        current_group=None,
        results=results,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_sensitivity(args)
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if bool(report.get("success", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
