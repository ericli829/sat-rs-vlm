"""Qwen3-VL + LoRA 遥感任务评测脚本。"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml
from sat_rs_vlm.models.qwen3vl_loader import (
    load_qwen3vl,
)
from sat_rs_vlm.models.qwen3vl_loader import (
    validate_local_adapter as _validate_local_adapter,
)

from sat_rs_vlm.configuration.environment import expand_environment
from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.evaluation.checkpoint_loader import (
    load_finetuned_checkpoint,
    read_strategy_manifest,
)
from sat_rs_vlm.evaluation.inference import (
    CHANGE_BINARY_PROMPT_VERSION,
    change_binary_inference_enabled,
    extract_reference,
    timed_change_binary_prediction,
    timed_prediction,
    timed_prediction_with_telemetry,
)
from sat_rs_vlm.evaluation.inference import (
    build_generation_kwargs as _build_generation_kwargs,
)
from sat_rs_vlm.evaluation.inference import (
    generate_prediction as _generate_prediction,
)
from sat_rs_vlm.evaluation.metrics import summarize_predictions
from sat_rs_vlm.evaluation.performance import (
    PerformanceMonitor,
    environment_metadata,
    model_resource_metadata,
)
from sat_rs_vlm.training.utils import (
    MODEL_DEPS_ERROR,
    model_input_device,
    resolve_torch_dtype,
    safe_import_model_dependencies,
)
from sat_rs_vlm.utils.jsonl import write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_generation_kwargs(generation_cfg: dict[str, Any]) -> dict[str, Any]:
    """兼容旧脚本导入，委托统一生成参数实现。"""

    return _build_generation_kwargs(generation_cfg)


def generate_prediction(
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    sample: dict[str, Any],
    generation_cfg: dict[str, Any],
    torch: Any,
) -> str:
    """兼容旧脚本导入，委托统一新增 token 解码实现。"""

    return _generate_prediction(model, processor, collator, sample, generation_cfg, torch)


def validate_local_adapter(adapter_source: str, *, local_files_only: bool) -> None:
    """兼容旧评测测试和调用方。"""

    _validate_local_adapter(adapter_source, local_files_only=local_files_only)


def parse_args() -> argparse.Namespace:
    """解析评测参数。"""

    parser = argparse.ArgumentParser(description="Evaluate remote-sensing VLM predictions.")
    parser.add_argument(
        "--config",
        default="configs/eval/qwen3vl_eval.yaml",
        help="Path to eval YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Experiment directory containing strategy_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory; keeps the evaluated checkpoint read-only.",
    )
    parser.add_argument(
        "--performance-monitor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable automatic inference performance report (default: config value, enabled).",
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=None,
        help="Override the number of unmeasured warmup samples.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置。"""

    with path.open("r", encoding="utf-8") as file:
        loaded = dict(yaml.safe_load(file) or {})
    return dict(expand_environment(loaded, environ=os.environ, allow_unresolved=False))


def resolve_project_path(value: str | Path) -> Path:
    """把相对路径解析到项目根目录。"""

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_model_source(value: str) -> str:
    """优先把存在的相对路径解析为本地路径，否则保留 HuggingFace 模型 ID。"""

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    project_path = PROJECT_ROOT / path
    return str(project_path) if project_path.exists() else value


def load_model(config: dict[str, Any], modules: dict[str, Any]) -> tuple[Any, Any]:
    """加载 base model、LoRA adapter 和 processor。"""

    torch = modules["torch"]
    model_cfg = dict(config["model"])
    local_files_only = bool(model_cfg.get("local_files_only", True))
    base_model = resolve_model_source(str(model_cfg["base_model"]))
    processor_id = resolve_model_source(str(model_cfg.get("processor_id", base_model)))
    raw_adapter = model_cfg.get("adapter_path")
    adapter_source = resolve_model_source(str(raw_adapter)) if raw_adapter else None
    model_kwargs: dict[str, Any] = {
        "device_map": model_cfg.get("device_map", "auto"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "local_files_only": local_files_only,
    }
    dtype = resolve_torch_dtype(torch, str(model_cfg.get("torch_dtype", "auto")))
    model_kwargs["dtype"] = dtype if dtype is not None else "auto"
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    return load_qwen3vl(
        modules=modules,
        base_model=base_model,
        processor_source=processor_id,
        model_kwargs=model_kwargs,
        processor_kwargs={
            "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
            "local_files_only": local_files_only,
        },
        adapter_path=adapter_source,
    )


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """兼容旧调用，内部使用统一任务指标。"""

    return summarize_predictions(predictions)


def evaluate(
    config_path: Path,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
    *,
    performance_monitor: bool | None = None,
    warmup_samples: int | None = None,
) -> None:
    """执行评测。"""

    startup_started = time.perf_counter()
    config = load_yaml(config_path)
    data_cfg = dict(config["data"])
    eval_file = resolve_project_path(str(data_cfg["eval_file"]))
    if not eval_file.is_file():
        raise FileNotFoundError(f"Evaluation JSONL file does not exist: {eval_file}")
    require_bitsandbytes = False
    if checkpoint is not None:
        require_bitsandbytes = bool(read_strategy_manifest(checkpoint).get("quantized_base", False))
    modules = safe_import_model_dependencies(require_bitsandbytes=require_bitsandbytes)
    model_load_started = time.perf_counter()
    if checkpoint is None:
        model, processor = load_model(config, modules)
    else:
        model, processor, _ = load_finetuned_checkpoint(
            checkpoint,
            dict(config.get("model", {})),
            modules,
        )
    model_load_ms = (time.perf_counter() - model_load_started) * 1000.0
    torch = modules["torch"]
    generation_cfg = dict(config.get("generation", {}))
    dataset = Qwen3VLDataset(eval_file, data_cfg.get("max_eval_samples"))
    collator = Qwen3VLDataCollator(
        processor,
        max_seq_length=int(data_cfg.get("max_seq_length", 4096)),
        image_root=resolve_project_path(str(data_cfg["image_root"])),
        for_generation=True,
    )

    performance_cfg = dict(config.get("performance", {}))
    monitor_enabled = (
        bool(performance_cfg.get("enabled", True))
        if performance_monitor is None
        else performance_monitor
    )
    configured_warmups = int(performance_cfg.get("warmup_samples", 2))
    effective_warmups = configured_warmups if warmup_samples is None else warmup_samples
    if effective_warmups < 0:
        raise ValueError("warmup samples must be non-negative")
    effective_warmups = min(effective_warmups, len(dataset))
    continue_on_error = bool(performance_cfg.get("continue_on_error", True))
    batch_size = int(performance_cfg.get("batch_size", 1))
    repeats = int(performance_cfg.get("repeats", 1))
    if batch_size != 1:
        raise ValueError("evaluate_rs_vlm.py currently measures only batch_size=1")
    if repeats != 1:
        raise ValueError("evaluate_rs_vlm.py currently measures only repeats=1")
    monitor: PerformanceMonitor | None = None
    if monitor_enabled:
        for index in range(effective_warmups):
            warmup_sample = dataset[index]
            timed_prediction(
                model,
                processor,
                collator,
                warmup_sample,
                generation_cfg,
                torch,
            )
            if change_binary_inference_enabled(warmup_sample, generation_cfg):
                timed_change_binary_prediction(
                    model,
                    processor,
                    collator,
                    warmup_sample,
                    generation_cfg,
                    torch,
                )
        monitor = PerformanceMonitor(torch, device=model_input_device(model, torch))
        monitor.start()

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, sample in enumerate(dataset, start=1):
        try:
            if monitor is None:
                prediction, latency_ms = timed_prediction(
                    model,
                    processor,
                    collator,
                    sample,
                    generation_cfg,
                    torch,
                )
                timing_fields: dict[str, Any] = {
                    "end_to_end_latency_ms": latency_ms,
                    "generation_latency_ms": None,
                    "ttft_ms": None,
                    "decode_latency_ms": None,
                    "output_token_count": None,
                    "generation_tokens_per_second": None,
                    "decode_tokens_per_second": None,
                }
            else:
                prediction, timing = timed_prediction_with_telemetry(
                    model,
                    processor,
                    collator,
                    sample,
                    generation_cfg,
                    torch,
                )
                latency_ms = timing.end_to_end_latency_ms
                timing_fields = timing.to_dict()
        except (RuntimeError, ValueError, OSError) as exc:
            failures.append(
                {
                    "id": str(sample.get("id", "")),
                    "task_type": str(sample.get("task_type", "")),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            if continue_on_error:
                print(f"Inference failed for sample {sample.get('id')}: {exc}")
                continue
            raise RuntimeError(f"Inference failed for sample {sample.get('id')}: {exc}") from exc
        row: dict[str, Any] = {
            "id": sample["id"],
            "task_type": sample["task_type"],
            "prediction": prediction,
            "reference": extract_reference(sample["messages"]),
            "metadata": sample.get("metadata", {}),
            "inference_latency_ms": latency_ms,
        }
        system_latency_ms = latency_ms
        if monitor is not None:
            row["performance"] = dict(timing_fields)
        if change_binary_inference_enabled(sample, generation_cfg):
            try:
                binary_raw, binary_flag, binary_latency_ms = timed_change_binary_prediction(
                    model,
                    processor,
                    collator,
                    sample,
                    generation_cfg,
                    torch,
                )
                row.update(
                    {
                        "prediction_changeflag": binary_flag,
                        "binary_prediction": binary_raw,
                        "binary_prediction_parse_ok": binary_flag is not None,
                        "binary_prompt_version": CHANGE_BINARY_PROMPT_VERSION,
                        "binary_inference_latency_ms": binary_latency_ms,
                        "total_inference_latency_ms": latency_ms + binary_latency_ms,
                    }
                )
                system_latency_ms += binary_latency_ms
                if monitor is not None:
                    row["performance"].update(
                        {
                            "auxiliary_binary_latency_ms": binary_latency_ms,
                            "system_end_to_end_latency_ms": system_latency_ms,
                        }
                    )
            except (RuntimeError, ValueError, OSError) as exc:
                failures.append(
                    {
                        "id": str(sample.get("id", "")),
                        "task_type": str(sample.get("task_type", "")),
                        "error_type": f"auxiliary_{type(exc).__name__}",
                        "message": str(exc),
                    }
                )
                if not continue_on_error:
                    raise RuntimeError(
                        f"Auxiliary change inference failed for sample {sample.get('id')}: {exc}"
                    ) from exc
                row.update(
                    {
                        "prediction_changeflag": None,
                        "binary_prediction": "",
                        "binary_prediction_parse_ok": False,
                        "binary_prompt_version": CHANGE_BINARY_PROMPT_VERSION,
                        "binary_inference_error": f"{type(exc).__name__}: {exc}",
                        "total_inference_latency_ms": latency_ms,
                    }
                )
                if monitor is not None:
                    row["performance"]["auxiliary_binary_error"] = f"{type(exc).__name__}: {exc}"
        if monitor is not None:
            monitor.record(
                str(sample["task_type"]),
                timing_fields,
                system_latency_ms=system_latency_ms,
                input_profile=timing_fields.get("input_profile"),
            )
        predictions.append(row)
        if index == 1 or index % 10 == 0 or index == len(dataset):
            print(f"Evaluated {index}/{len(dataset)} samples")

    output_cfg = dict(config["output"])
    if output_dir is not None:
        summary_file = output_dir.resolve() / "summary.json"
        predictions_file = output_dir.resolve() / "predictions.jsonl"
    elif checkpoint is None:
        summary_file = resolve_project_path(str(output_cfg["summary_file"]))
        predictions_file = resolve_project_path(str(output_cfg["predictions_file"]))
    else:
        eval_output = checkpoint.resolve() / "evaluation"
        summary_file = eval_output / "summary.json"
        predictions_file = eval_output / "predictions.jsonl"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(predictions)
    performance_file = summary_file.parent / "performance_report.json"
    if monitor is not None:
        report = monitor.finish(
            requested_samples=len(dataset),
            completed_samples=len(predictions),
            failed_samples=len(failures),
            warmup_samples=effective_warmups,
            startup_and_model_load_ms=(time.perf_counter() - startup_started) * 1000.0,
            model_load_ms=model_load_ms,
            config={
                "generation": generation_cfg,
                "max_seq_length": int(data_cfg.get("max_seq_length", 4096)),
                "max_eval_samples": data_cfg.get("max_eval_samples"),
                "change_binary_enabled": bool(
                    generation_cfg.get("change_binary_enabled", False)
                ),
                "continue_on_error": continue_on_error,
                "batch_size": batch_size,
                "repeats": repeats,
            },
            environment=environment_metadata(torch, model_config=dict(config.get("model", {}))),
            model_resources=model_resource_metadata(
                model,
                model_config=dict(config.get("model", {})),
            ),
            batch_size=batch_size,
            repeats=repeats,
        )
        report["failures"] = failures
        performance_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["performance"] = {
            "report_file": performance_file.name,
            "latency_ms": report["latency_ms"],
            "ttft_ms": report["ttft_ms"],
            "generation_tokens_per_second": report["generation_tokens_per_second"],
            "memory_mb": report["memory_mb"],
        }
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(predictions_file, predictions)
    print(f"Saved summary to {summary_file}")
    print(f"Saved predictions to {predictions_file}")
    if monitor is not None:
        print(f"Saved performance report to {performance_file}")


def main() -> int:
    """脚本入口。"""

    args = parse_args()
    try:
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        output_dir = Path(args.output_dir) if args.output_dir else None
        evaluate(
            Path(args.config),
            checkpoint,
            output_dir,
            performance_monitor=args.performance_monitor,
            warmup_samples=args.warmup_samples,
        )
    except ImportError as exc:
        raise SystemExit(str(exc) or MODEL_DEPS_ERROR) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
