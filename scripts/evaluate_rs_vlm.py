"""Qwen3-VL + LoRA 遥感任务评测脚本。"""

# The source path is inserted below before importing the local package.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.configuration.environment import expand_environment
from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.evaluation.checkpoint_loader import (
    load_finetuned_checkpoint,
    read_strategy_manifest,
)
from sat_rs_vlm.evaluation.inference import (
    build_generation_kwargs as _build_generation_kwargs,
)
from sat_rs_vlm.evaluation.inference import (
    count_decoded_output_tokens,
    extract_reference,
    timed_predictions,
)
from sat_rs_vlm.evaluation.inference import (
    generate_prediction as _generate_prediction,
)
from sat_rs_vlm.evaluation.metrics import summarize_predictions
from sat_rs_vlm.evaluation.runner import run_evaluation, validate_output_directory
from sat_rs_vlm.evaluation.tiers import resolve_tier_identity, validate_tier_asset
from sat_rs_vlm.infrastructure.telemetry import (
    GenerationTelemetry,
    SystemTelemetry,
    collect_model_inventory,
    collect_repository_provenance,
    collect_runtime_environment,
)
from sat_rs_vlm.models.qwen3vl_loader import (
    load_qwen3vl,
)
from sat_rs_vlm.models.qwen3vl_loader import (
    validate_local_adapter as _validate_local_adapter,
)
from sat_rs_vlm.training.utils import (
    MODEL_DEPS_ERROR,
    resolve_torch_dtype,
    safe_import_model_dependencies,
)
from sat_rs_vlm.utils.jsonl import write_jsonl

DEFAULT_EVALUATION_CONTRACT = (
    PROJECT_ROOT / "configs/eval/evaluation_contract_v1.5.yaml"
)


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

    return _generate_prediction(
        model, processor, collator, sample, generation_cfg, torch
    )


def validate_local_adapter(adapter_source: str, *, local_files_only: bool) -> None:
    """兼容旧评测测试和调用方。"""

    _validate_local_adapter(adapter_source, local_files_only=local_files_only)


def parse_args() -> argparse.Namespace:
    """解析评测参数。"""

    parser = argparse.ArgumentParser(
        description="Evaluate remote-sensing VLM predictions."
    )
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
        "--batch-size",
        type=int,
        default=None,
        help="Override data.eval_batch_size from the YAML config.",
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


def resolve_evaluation_outputs(
    config_path: Path,
    config: dict[str, Any],
    checkpoint: Path | None,
    output_dir: Path | None,
) -> tuple[Path, Path, Path]:
    """统一解析兼容 summary、原始 predictions 和 v1.5 评估目录。"""

    output_cfg = dict(config["output"])
    if output_dir is not None:
        root = output_dir.resolve()
        return (
            root / "summary.json",
            root / "predictions.jsonl",
            root / "evaluation_v1_5",
        )
    if checkpoint is not None:
        root = checkpoint.resolve() / "evaluation"
        return (
            root / "summary.json",
            root / "predictions.jsonl",
            root / "evaluation_v1_5",
        )
    summary_file = resolve_project_path(str(output_cfg["summary_file"]))
    predictions_file = resolve_project_path(str(output_cfg["predictions_file"]))
    configured_dir = output_cfg.get("evaluation_dir")
    evaluation_dir = (
        resolve_project_path(str(configured_dir))
        if configured_dir
        else PROJECT_ROOT / "reports" / "evaluation" / config_path.stem
    )
    return summary_file, predictions_file, evaluation_dir


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
    """兼容旧调用的诊断摘要；正式评估产物由 Evaluation v1.5 runner 生成。"""

    return summarize_predictions(predictions)


def iter_evaluation_batches(
    dataset: Sequence[dict[str, Any]],
    batch_size: int,
    *,
    group_by_task: bool,
) -> Iterator[tuple[str | None, list[tuple[int, dict[str, Any]]]]]:
    """Yield indexed batches, optionally grouping samples by task type."""

    if batch_size < 1:
        raise ValueError(f"Evaluation batch size must be positive, got {batch_size}")
    if group_by_task:
        groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, sample in enumerate(dataset):
            groups[str(sample.get("task_type", "unknown"))].append((index, sample))
        for task_type, indexed_samples in groups.items():
            for start in range(0, len(indexed_samples), batch_size):
                yield task_type, indexed_samples[start : start + batch_size]
        return
    indexed_samples = list(enumerate(dataset))
    for start in range(0, len(indexed_samples), batch_size):
        yield None, indexed_samples[start : start + batch_size]


def evaluate(
    config_path: Path,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
    batch_size_override: int | None = None,
    *,
    loaded_model: Any | None = None,
    loaded_processor: Any | None = None,
    loaded_modules: dict[str, Any] | None = None,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行评测。"""

    config = config_override if config_override is not None else load_yaml(config_path)
    evaluation_cfg = dict(config.get("evaluation", {}))
    tier_identity: dict[str, Any] | None = None
    if evaluation_cfg.get("tier") is not None:
        configured_tier = resolve_tier_identity(config, project_root=PROJECT_ROOT)
        tier_identity = validate_tier_asset(
            tier=configured_tier["tier"],
            eval_file=resolve_project_path(configured_tier["eval_file"]),
            manifest_path=resolve_project_path(configured_tier["tiers_manifest"]),
        )
    summary_file, predictions_file, evaluation_dir = resolve_evaluation_outputs(
        config_path,
        config,
        checkpoint,
        output_dir,
    )
    validate_output_directory(evaluation_dir)
    data_cfg = dict(config["data"])
    eval_file = resolve_project_path(str(data_cfg["eval_file"]))
    if not eval_file.is_file():
        raise FileNotFoundError(f"Evaluation JSONL file does not exist: {eval_file}")
    supplied = (loaded_model, loaded_processor, loaded_modules)
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        raise ValueError(
            "loaded_model, loaded_processor, and loaded_modules must be supplied together"
        )
    model_load_started = time.perf_counter()
    if loaded_modules is None:
        require_bitsandbytes = False
        if checkpoint is not None:
            require_bitsandbytes = bool(
                read_strategy_manifest(checkpoint).get("quantized_base", False)
            )
        modules = safe_import_model_dependencies(
            require_bitsandbytes=require_bitsandbytes
        )
        if checkpoint is None:
            model, processor = load_model(config, modules)
        else:
            model, processor, _ = load_finetuned_checkpoint(
                checkpoint,
                dict(config.get("model", {})),
                modules,
            )
    else:
        if checkpoint is not None:
            raise ValueError("A checkpoint cannot be combined with a preloaded model")
        modules = loaded_modules
        model = loaded_model
        processor = loaded_processor
    model_load_time_ms = (
        (time.perf_counter() - model_load_started) * 1000.0
        if loaded_modules is None
        else None
    )
    model.eval()
    torch = modules["torch"]
    evaluation_started = time.perf_counter()
    generation_cfg = dict(config.get("generation", {}))
    dataset = Qwen3VLDataset(eval_file, data_cfg.get("max_eval_samples"))
    batch_size = int(batch_size_override or data_cfg.get("eval_batch_size", 1))
    group_by_task = bool(data_cfg.get("group_by_task", True))
    warmup_runs = int(evaluation_cfg.get("warmup_runs", 0))
    repeat_runs = int(evaluation_cfg.get("repeat_runs", 1))
    log_every = max(1, int(data_cfg.get("log_every_samples", 100)))
    if batch_size < 1:
        raise ValueError(f"Evaluation batch size must be positive, got {batch_size}")
    if warmup_runs < 0:
        raise ValueError(f"Evaluation warmup_runs cannot be negative, got {warmup_runs}")
    if repeat_runs < 1:
        raise ValueError(f"Evaluation repeat_runs must be positive, got {repeat_runs}")
    tokenizer = getattr(processor, "tokenizer", None)
    if batch_size > 1 and tokenizer is not None:
        tokenizer.padding_side = "left"
    collator = Qwen3VLDataCollator(
        processor,
        max_seq_length=int(data_cfg.get("max_seq_length", 4096)),
        image_root=resolve_project_path(str(data_cfg["image_root"])),
        for_generation=True,
    )
    model_config = dict(config.get("model", {}))
    model_name = str(model_config.get("base_model", type(model).__name__))

    predictions_by_index: list[dict[str, Any] | None] = [None] * len(dataset)
    measured_output_tokens_total = 0
    measured_output_token_samples = 0
    evaluated = 0
    failed_samples = 0
    next_log = 1
    print(
        f"Evaluating {len(dataset)} samples with batch_size={batch_size}, "
        f"group_by_task={group_by_task}, warmup_runs={warmup_runs}, "
        f"repeat_runs={repeat_runs}"
    )
    evaluation_batches = list(
        iter_evaluation_batches(dataset, batch_size, group_by_task=group_by_task)
    )
    if evaluation_batches and warmup_runs:
        warmup_task_type, warmup_batch = evaluation_batches[0]
        warmup_samples = [sample for _, sample in warmup_batch]
        for _ in range(warmup_runs):
            timed_predictions(
                model,
                processor,
                collator,
                warmup_samples,
                generation_cfg,
                torch,
                task_type=warmup_task_type,
            )
    with SystemTelemetry(
        "main_evaluation_prediction_loop",
        torch_module=torch,
        reset_cuda_peaks=True,
    ) as prediction_monitor:
        for task_type, indexed_batch in evaluation_batches:
            samples = [sample for _, sample in indexed_batch]
            measured_latencies: list[float] = []
            batch_predictions: list[str] = []
            batch_output_token_counts: list[int | None] = []
            batch_output_tokens_total = 0
            batch_output_token_samples = 0
            batch_generation_telemetry: GenerationTelemetry | None = None
            try:
                for _ in range(repeat_runs):
                    generation_telemetry = GenerationTelemetry()
                    repeated_predictions, repeated_latency_ms = timed_predictions(
                        model,
                        processor,
                        collator,
                        samples,
                        generation_cfg,
                        torch,
                        task_type=task_type,
                        telemetry=generation_telemetry,
                    )
                    repeated_token_counts = count_decoded_output_tokens(
                        processor, repeated_predictions
                    )
                    available_repeated_counts = [
                        count for count in repeated_token_counts if count is not None
                    ]
                    batch_output_tokens_total += sum(available_repeated_counts)
                    batch_output_token_samples += len(available_repeated_counts)
                    if not batch_predictions:
                        batch_predictions = repeated_predictions
                        batch_output_token_counts = repeated_token_counts
                        batch_generation_telemetry = generation_telemetry
                    measured_latencies.append(repeated_latency_ms)
                latency_ms = sum(measured_latencies) / len(measured_latencies)
                generation_payload = (
                    batch_generation_telemetry.to_dict()
                    if batch_generation_telemetry is not None
                    else {}
                )
                if isinstance(generation_payload.get("timing_ms"), dict):
                    generation_payload["timing_ms"]["batch_e2e"] = generation_payload[
                        "timing_ms"
                    ].get("e2e")
                    generation_payload["timing_ms"]["e2e"] = latency_ms
                for (original_index, sample), prediction, output_token_count in zip(
                    indexed_batch, batch_predictions, batch_output_token_counts, strict=True
                ):
                    predictions_by_index[original_index] = {
                        "id": sample["id"],
                        "task_type": sample["task_type"],
                        "prediction": prediction,
                        "reference": extract_reference(sample["messages"]),
                        "metadata": sample.get("metadata", {}),
                        "inference_latency_ms": latency_ms,
                        "latency_semantics": "batch_amortized_model_path",
                        "output_token_count": output_token_count,
                        "output_token_count_method": "retokenized_decoded_output",
                        "telemetry": {
                            "schema_version": "1.0",
                            "success": True,
                            "scope": "standalone_vlm_batch",
                            "timing_ms": {
                                "e2e": latency_ms,
                                "preprocess": None,
                                "model_generate": None,
                                "decode": None,
                                "ttft": None,
                            },
                            "tokens": {"output": output_token_count},
                            "resources": {
                                "scope": "main_evaluation_prediction_loop",
                                "available_after_run": True,
                            },
                            "activated_models": [model_name],
                            **generation_payload,
                        },
                    }
                measured_output_tokens_total += batch_output_tokens_total
                measured_output_token_samples += batch_output_token_samples
            except Exception as batch_error:
                # A single malformed image or processor input must not discard the
                # rest of the evaluation. Retry each item separately so the bad
                # sample can be represented explicitly in the prediction artifact.
                for original_index, sample in indexed_batch:
                    try:
                        individual_predictions: list[str] = []
                        individual_latencies: list[float] = []
                        individual_counts: list[int | None] = []
                        for _ in range(repeat_runs):
                            generation_telemetry = GenerationTelemetry()
                            prediction_values, individual_latency = timed_predictions(
                                model,
                                processor,
                                collator,
                                [sample],
                                generation_cfg,
                                torch,
                                task_type=task_type,
                                telemetry=generation_telemetry,
                            )
                            individual_prediction = prediction_values[0]
                            individual_predictions.append(individual_prediction)
                            individual_latencies.append(individual_latency)
                            individual_counts.extend(
                                count_decoded_output_tokens(processor, [individual_prediction])
                            )
                        prediction = individual_predictions[0]
                        output_token_count = individual_counts[0]
                        latency_ms = sum(individual_latencies) / len(individual_latencies)
                        generation_payload = generation_telemetry.to_dict()
                        measured_output_tokens_total += sum(
                            count for count in individual_counts if count is not None
                        )
                        measured_output_token_samples += sum(
                            count is not None for count in individual_counts
                        )
                        predictions_by_index[original_index] = {
                            "id": sample["id"],
                            "task_type": sample["task_type"],
                            "prediction": prediction,
                            "reference": extract_reference(sample["messages"]),
                            "metadata": sample.get("metadata", {}),
                            "inference_latency_ms": latency_ms,
                            "latency_semantics": "single_sample_retry_after_batch_failure",
                            "output_token_count": output_token_count,
                            "output_token_count_method": "retokenized_decoded_output",
                            "telemetry": {
                                "schema_version": "1.0",
                                "success": True,
                                "scope": "standalone_vlm_single_sample_retry",
                                "timing_ms": {"e2e": latency_ms, "ttft": None},
                                "tokens": {"output": output_token_count},
                                "batch_error": type(batch_error).__name__,
                                **generation_payload,
                            },
                        }
                    except Exception as sample_error:
                        failed_samples += 1
                        predictions_by_index[original_index] = {
                            "id": sample["id"],
                            "task_type": sample["task_type"],
                            "prediction": "",
                            "reference": extract_reference(sample["messages"]),
                            "metadata": sample.get("metadata", {}),
                            "inference_latency_ms": None,
                            "latency_semantics": "failed_inference",
                            "output_token_count": None,
                            "output_token_count_method": None,
                            "telemetry": {
                                "schema_version": "1.0",
                                "success": False,
                                "scope": "standalone_vlm",
                                "timing_ms": {"e2e": None, "ttft": None},
                                "tokens": {"output": None},
                                "error_type": type(sample_error).__name__,
                                "error_message": str(sample_error),
                                "batch_error_type": type(batch_error).__name__,
                            },
                        }
            evaluated += len(samples)
            if evaluated >= next_log or evaluated == len(dataset):
                print(f"Evaluated {evaluated}/{len(dataset)} samples")
                next_log = ((evaluated // log_every) + 1) * log_every

    if any(prediction is None for prediction in predictions_by_index):
        raise RuntimeError("Evaluation finished with missing predictions")
    predictions = [
        prediction for prediction in predictions_by_index if prediction is not None
    ]
    prediction_telemetry = prediction_monitor.to_dict()
    resources = dict(prediction_telemetry["resources"])
    for prediction in predictions:
        telemetry = prediction.get("telemetry")
        if isinstance(telemetry, dict):
            telemetry_resources = telemetry.setdefault("resources", {})
            if isinstance(telemetry_resources, dict):
                telemetry_resources.update(resources)
                telemetry_resources["scope"] = "main_evaluation_prediction_loop"
    resource_benchmark = {
        "scope": prediction_telemetry["scope"],
        "timing_ms": prediction_telemetry["timing_ms"],
        "resources": resources,
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "repeat_runs": repeat_runs,
        "latency_semantics": "batch_amortized_model_path",
    }

    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions_file, predictions)
    print(f"Saved predictions to {predictions_file}")

    contract_path = resolve_project_path(
        str(evaluation_cfg.get("contract", DEFAULT_EVALUATION_CONTRACT))
    )
    manifest_value = evaluation_cfg.get("manifest") or data_cfg.get("manifest")
    evaluation_outputs = run_evaluation(
        predictions_file,
        evaluation_dir,
        contract_path=contract_path,
        manifest_path=(
            resolve_project_path(str(manifest_value)) if manifest_value else None
        ),
        strict=bool(evaluation_cfg.get("strict", True)),
        semantic_enabled=bool(evaluation_cfg.get("semantic", True)),
        semantic_contract_path=resolve_project_path(
            str(
                evaluation_cfg.get(
                    "semantic_contract",
                    "configs/eval/semantic/semantic_contract.json",
                )
            )
        ),
        semantic_ontology_path=resolve_project_path(
            str(
                evaluation_cfg.get(
                    "semantic_ontology",
                    "configs/eval/semantic/remote_sensing_ontology.json",
                )
            )
        ),
        latency_semantics=str(
            evaluation_cfg.get("latency_semantics", "batch_amortized_per_sample")
        ),
        eval_batch_size=batch_size,
        group_by_task=group_by_task,
        evaluation_tier=tier_identity["tier"] if tier_identity else None,
        evaluation_tier_version=(
            tier_identity["tier_version"] if tier_identity else None
        ),
        evaluation_tier_sha256=tier_identity["sha256"] if tier_identity else None,
        resource_benchmark=resource_benchmark,
    )
    evaluation_runtime_seconds = time.perf_counter() - evaluation_started
    peak_vram_mb = resources["peak_gpu_allocated_mb"]
    output_token_counts = [
        int(row["output_token_count"])
        for row in predictions
        if row.get("output_token_count") is not None
    ]
    generation_records = [
        row.get("telemetry", {})
        for row in predictions
        if isinstance(row.get("telemetry", {}), dict)
    ]
    generation_timing = [
        record.get("timing_ms", {})
        for record in generation_records
        if isinstance(record.get("timing_ms", {}), dict)
    ]
    ttft_values = [
        float(timing["ttft"])
        for timing in generation_timing
        if timing.get("ttft") is not None
    ]
    decode_generation_ms = sum(
        float(timing["decode_generation"])
        for timing in generation_timing
        if timing.get("decode_generation") is not None
    )
    generated_tokens = sum(
        int(record["tokens"]["generated"])
        for record in generation_records
        if isinstance(record.get("tokens"), dict)
        and record["tokens"].get("generated") is not None
    )
    visual_token_values = [
        int(record["vision_input"]["visual_token_count"])
        for record in generation_records
        if isinstance(record.get("vision_input"), dict)
        and record["vision_input"].get("visual_token_count") is not None
    ]
    prediction_loop_ms = prediction_telemetry["timing_ms"]["e2e"]
    metadata_path = (
        output_dir / "evaluation_metadata.json"
        if output_dir is not None
        else evaluation_dir.parent / "evaluation_metadata.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_summary_path = metadata_path.parent / "telemetry_summary.json"
    system_manifest_path = metadata_path.parent / "system_manifest.json"
    telemetry_summary = {
        "schema_version": "1.0",
        "prediction_loop": prediction_telemetry,
        "model_load": {
            "latency_ms": model_load_time_ms,
            "semantics": (
                "dependencies_and_model_load"
                if model_load_time_ms is not None
                else "not_measured_for_preloaded_model"
            ),
        },
        "evaluation_runtime_ms": evaluation_runtime_seconds * 1000.0,
        "evaluation_runtime_semantics": (
            "prediction_generation_plus_metric_evaluation_and_report_writes"
        ),
        "sample_count": len(predictions),
        "failed_samples": failed_samples,
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "repeat_runs": repeat_runs,
        "latency_semantics": "batch_amortized_model_path",
        "single_sample_full_system_e2e_available": False,
        "single_sample_full_system_e2e_note": (
            "This entry point measures the standalone VLM model path. "
            "Full-system routing telemetry is emitted by TaskGraphRuntime."
        ),
        "tokens": {
            "output_token_count": sum(output_token_counts) if output_token_counts else None,
            "samples_with_output_token_count": len(output_token_counts),
            "count_method": "retokenized_decoded_output",
            "output_tokens_per_prediction_loop_second": (
                measured_output_tokens_total / (prediction_loop_ms / 1000.0)
                if measured_output_token_samples and prediction_loop_ms
                else None
            ),
            "decode_tokens_per_second": (
                generated_tokens / (decode_generation_ms / 1000.0)
                if generated_tokens and decode_generation_ms > 0
                else None
            ),
            "decode_tokens_per_second_status": (
                "ok" if generated_tokens and decode_generation_ms > 0 else "unavailable"
            ),
            "ttft_ms": sum(ttft_values) / len(ttft_values) if ttft_values else None,
            "ttft_status": "ok" if ttft_values else "unavailable",
            "visual_token_count": (
                sum(visual_token_values) if visual_token_values else None
            ),
            "visual_token_count_status": "ok" if visual_token_values else "unavailable",
        },
    }
    model_cfg = model_config
    model_paths: list[str | Path] = []
    for key in ("base_model", "adapter_path"):
        value = model_cfg.get(key)
        if value:
            model_paths.append(resolve_model_source(str(value)))
    if checkpoint is not None:
        model_paths.append(checkpoint)
    model_inventory = collect_model_inventory(model, model_paths)
    system_manifest = {
        "schema_version": "1.0",
        "system": {
            "scope": "standalone_vlm_evaluation",
            "total_parameter_count": model_inventory["parameter_count"],
            "total_model_storage_bytes": model_inventory["local_model_storage_bytes"],
        },
        "models": [
            {
                "name": str(model_cfg.get("base_model", type(model).__name__)),
                "role": "answer_vlm",
                **model_inventory,
            }
        ],
        "runtime": collect_runtime_environment(torch),
        "benchmark": {
            "sample_count": len(predictions),
            "failed_samples": failed_samples,
            "batch_size": batch_size,
            "warmup_runs": warmup_runs,
            "repeat_runs": repeat_runs,
            "repeat_output_policy": "first_repeat_used_for_scoring_all_repeats_profiled",
            "warmup_scope": "first_evaluation_batch",
            "group_by_task": group_by_task,
            "cache_policy": str(evaluation_cfg.get("cache_policy", "unspecified")),
            "generation": generation_cfg,
            "precision": model_cfg.get("torch_dtype", model_cfg.get("dtype", "auto")),
            "timing_boundaries": {
                "prediction_loop": "collation through device transfer, generate, and decode",
                "model_load": "dependency import and model/checkpoint construction",
            },
        },
        "repository": collect_repository_provenance(PROJECT_ROOT),
    }
    telemetry_summary_path.write_text(
        json.dumps(telemetry_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    system_manifest_path.write_text(
        json.dumps(system_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "eval_batch_size": batch_size,
                "sample_count": len(predictions),
                "evaluation_runtime_seconds": evaluation_runtime_seconds,
                "samples_per_second": (
                    len(predictions) / evaluation_runtime_seconds
                    if evaluation_runtime_seconds > 0
                    else None
                ),
                "peak_vram_mb": peak_vram_mb,
                "peak_gpu_reserved_mb": resources["peak_gpu_reserved_mb"],
                "peak_cpu_rss_mb": resources["peak_cpu_rss_mb"],
                "model_load_time_ms": model_load_time_ms,
                "failed_samples": failed_samples,
                "warmup_runs": warmup_runs,
                "repeat_runs": repeat_runs,
                "latency_semantics": "batch_amortized_model_path",
                "telemetry_summary_file": str(telemetry_summary_path),
                "system_manifest_file": str(system_manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_bytes(evaluation_outputs["metrics"].read_bytes())
    print(f"Saved Evaluation v1.5 metrics to {evaluation_outputs['metrics']}")
    print(f"Saved compatibility summary to {summary_file}")
    return {
        "config": str(config_path),
        "summary_file": str(summary_file),
        "predictions_file": str(predictions_file),
        "evaluation_dir": str(evaluation_dir),
        "evaluation_outputs": {
            name: str(path) for name, path in evaluation_outputs.items()
        },
        "evaluation_metadata": str(metadata_path),
        "telemetry_summary": str(telemetry_summary_path),
        "system_manifest": str(system_manifest_path),
        "sample_count": len(predictions),
        "batch_size": batch_size,
    }


def main() -> int:
    """脚本入口。"""

    args = parse_args()
    try:
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        output_dir = Path(args.output_dir) if args.output_dir else None
        evaluate(Path(args.config), checkpoint, output_dir, args.batch_size)
    except ImportError as exc:
        raise SystemExit(str(exc) or MODEL_DEPS_ERROR) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
