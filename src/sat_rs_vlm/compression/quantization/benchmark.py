"""统一量化 benchmark 编排：固定样本、公平生成、失败统计和报告。"""

from __future__ import annotations

import gc
import importlib
import shutil
import time
import tracemalloc
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.compression.quantization.backends import QuantizationBackend
from sat_rs_vlm.compression.quantization.config import QuantizationExperimentConfig
from sat_rs_vlm.compression.quantization.manifest import (
    directory_size_bytes,
    to_json_safe,
    write_json_report,
)
from sat_rs_vlm.compression.quantization.report import (
    comparison_summary,
    environment_metadata,
    latency_statistics,
)
from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.evaluation.inference import (
    extract_message_inputs,
    extract_reference,
    timed_prediction,
)
from sat_rs_vlm.evaluation.metrics import summarize_predictions
from sat_rs_vlm.models.qwen3vl_loader import (
    load_qwen3vl_processor,
    validate_local_adapter,
)
from sat_rs_vlm.training.utils import safe_import_model_dependencies, set_seed
from sat_rs_vlm.utils.jsonl import write_jsonl


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def validate_assets(
    config: QuantizationExperimentConfig,
    backend: QuantizationBackend,
    *,
    project_root: Path,
    torch: Any | None = None,
) -> tuple[Qwen3VLDataset, list[dict[str, Any]], Path, Path]:
    """不加载大模型地验证配置、模型/adapter、数据、图片和样本 manifest。"""

    base_model = _project_path(config.model.base_model, project_root)
    if config.model.local_files_only and not base_model.is_dir():
        raise FileNotFoundError(f"Local base model directory does not exist: {base_model}")
    if config.model.local_files_only:
        original_base = config.model.base_model
        config.model.base_model = str(base_model)
        if config.model.processor_id == original_base:
            config.model.processor_id = str(base_model)
    if config.model.adapter_path:
        adapter_path = _project_path(config.model.adapter_path, project_root)
        config.model.adapter_path = str(adapter_path)
        validate_local_adapter(
            adapter_path,
            local_files_only=config.model.local_files_only,
        )
    backend.validate(config, torch)
    eval_file = _project_path(config.data.eval_file, project_root)
    image_root = _project_path(config.data.image_root, project_root)
    if not eval_file.is_file():
        raise FileNotFoundError(f"Evaluation JSONL does not exist: {eval_file}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    dataset = Qwen3VLDataset(eval_file, config.data.max_eval_samples)
    if len(dataset) == 0:
        raise ValueError(f"Evaluation dataset contains no samples: {eval_file}")
    manifest: list[dict[str, Any]] = []
    missing: list[str] = []
    for sample in dataset:
        images, question, reference = extract_message_inputs(sample, image_root)
        if not images:
            missing.append(f"{sample['id']}: no image in messages")
        for image in images:
            if not image.is_file():
                missing.append(f"{sample['id']}: {image}")
        manifest.append(
            {
                "id": sample["id"],
                "task_type": sample["task_type"],
                "images": [str(image) for image in images],
                "question": question,
                "reference": reference,
            }
        )
    if missing:
        raise FileNotFoundError("Evaluation sample images are missing: " + "; ".join(missing[:20]))
    return dataset, manifest, eval_file, image_root


def _logical_parameters(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _save_artifact(
    model: Any,
    processor: Any,
    backend: QuantizationBackend,
    output_dir: Path,
    metadata: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    """保存实验产物，但只有 reload 验证通过后才能标记为可部署。"""

    artifact = output_dir / "artifact"
    if artifact.exists():
        shutil.rmtree(artifact)
    artifact.mkdir(parents=True)
    manifest = dict(metadata)
    manifest.update({"deployable": False, "reload_verified": False})
    if backend.name == "torch_dynamic_int8":
        torch.save(model.state_dict(), artifact / "model_state_dict.pt")
        quantized_layers = [
            name
            for name, module in model.named_modules()
            if "quantized.dynamic" in module.__class__.__module__
        ]
        manifest.update(
            {
                "artifact_type": "benchmark_only_state_dict",
                "benchmark_only": True,
                "quantized_layers": quantized_layers,
                "reload_supported": False,
                "reason": "Generic Qwen3-VL dynamic-quantized reconstruction is not implemented.",
            }
        )
    else:
        model.save_pretrained(artifact)
        processor.save_pretrained(artifact / "processor")
        manifest.update(
            {
                "artifact_type": "pretrained_directory_unverified",
                "reload_supported": None,
                "reason": "save_pretrained completed but reload smoke has not run.",
            }
        )
    write_json_report(artifact / "quantization_manifest.json", manifest)
    manifest["serialized_artifact_bytes"] = directory_size_bytes(artifact)
    return manifest


def _run_variant(
    *,
    variant: str,
    quantized: bool,
    model: Any,
    processor: Any,
    dataset: Qwen3VLDataset,
    image_root: Path,
    config: QuantizationExperimentConfig,
    backend: QuantizationBackend,
    torch: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """运行一个固定样本变体；每个失败样本保留 ID、异常类型和消息。"""

    collator = Qwen3VLDataCollator(
        processor,
        config.data.max_seq_length,
        image_root,
        for_generation=True,
    )
    warmup_count = min(config.benchmark.warmup_samples, len(dataset))
    for index in range(warmup_count):
        try:
            timed_prediction(
                model,
                processor,
                collator,
                dataset[index],
                config.generation.model_dump(mode="python"),
                torch,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise RuntimeError(
                f"Warmup failed for sample {dataset[index]['id']}: {exc}"
            ) from exc

    if bool(torch.cuda.is_available()):
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    latency_values: list[float] = []
    for sample in dataset:
        prediction: str | None = None
        sample_latencies: list[float] = []
        for repeat in range(config.benchmark.repeats):
            try:
                generated, latency = timed_prediction(
                    model,
                    processor,
                    collator,
                    sample,
                    config.generation.model_dump(mode="python"),
                    torch,
                )
                prediction = generated if prediction is None else prediction
                sample_latencies.append(latency)
            except (RuntimeError, ValueError, OSError) as exc:
                failures.append(
                    {
                        "id": str(sample["id"]),
                        "repeat": str(repeat),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        if prediction is None:
            continue
        latency_values.extend(sample_latencies)
        rows.append(
            {
                "id": sample["id"],
                "task_type": sample["task_type"],
                "prediction": prediction,
                "reference": extract_reference(sample["messages"]),
                "metadata": sample.get("metadata", {}),
                "inference_latency_ms": sum(sample_latencies) / len(sample_latencies),
                "variant": variant,
                "backend": backend.name if quantized else "none",
            }
        )
    _, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    duration = time.perf_counter() - started
    compression = backend.compression_metadata(model, torch, quantized=quantized)
    artifact: dict[str, Any] | None = None
    if quantized and config.quantization.save_artifact:
        artifact = _save_artifact(model, processor, backend, output_dir, compression, torch)
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(variant_dir / "predictions.jsonl", rows)
    report: dict[str, Any] = {
        "variant": variant,
        "backend": backend.name if quantized else "none",
        "success": len(rows) > 0,
        "requested_samples": len(dataset),
        "completed_samples": len(rows),
        "failed_samples": len({failure["id"] for failure in failures}),
        "failures": failures,
        "latency_scope": config.benchmark.latency_scope,
        "latency_ms": latency_statistics(latency_values),
        "duration_seconds": duration,
        "logical_parameter_count": _logical_parameters(model),
        "serialized_artifact_bytes": (
            artifact.get("serialized_artifact_bytes") if artifact else None
        ),
        "peak_cpu_memory_mb": None,
        "peak_python_memory_mb": peak_python_bytes / (1024 * 1024),
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            if bool(torch.cuda.is_available())
            else None
        ),
        "compression": compression,
        "artifact": artifact,
        "metrics": summarize_predictions(rows),
        "sample_ids": [row["id"] for row in rows],
    }
    write_json_report(variant_dir / "report.json", report)
    return report


def _release(model: Any, torch: Any) -> None:
    del model
    gc.collect()
    if bool(torch.cuda.is_available()):
        torch.cuda.empty_cache()


def planned_variants(backend_name: str, *, skip_baseline: bool) -> tuple[str, ...]:
    """返回实际执行变体；跳过 baseline 时不生成虚假对比字段。"""

    if backend_name == "baseline":
        if skip_baseline:
            raise ValueError("--skip-baseline leaves no variant when backend='baseline'")
        return ("baseline",)
    return ("quantized",) if skip_baseline else ("baseline", "quantized")


def assert_comparable_sample_ids(
    baseline: dict[str, Any] | None,
    quantized: dict[str, Any] | None,
) -> None:
    """确保公平比较使用完全相同且顺序一致的成功样本。"""

    if baseline is None or quantized is None:
        return
    if baseline.get("sample_ids") != quantized.get("sample_ids"):
        raise RuntimeError("Baseline and quantized variants completed different sample IDs")


def run_benchmark(
    config: QuantizationExperimentConfig,
    backend: QuantizationBackend,
    *,
    project_root: Path,
    skip_baseline: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """执行资产检查、共享 Processor 加载和 baseline/quantized 公平比较。"""

    torch = None
    modules: dict[str, Any] | None = None
    if dry_run:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Quantization dry-run requires torch for backend capability checks"
            ) from exc
    else:
        modules = safe_import_model_dependencies(
            require_bitsandbytes=backend.requires_bitsandbytes
        )
        torch = modules["torch"]
    dataset, sample_manifest, eval_file, image_root = validate_assets(
        config,
        backend,
        project_root=project_root,
        torch=torch,
    )
    output_dir = _project_path(config.output.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "sample_manifest.jsonl", sample_manifest)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            to_json_safe(config.model_dump(mode="python")),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if dry_run:
        report = {
            "success": True,
            "dry_run": True,
            "backend": backend.name,
            "eval_file": str(eval_file),
            "image_root": str(image_root),
            "requested_samples": len(dataset),
            "sample_ids": [row["id"] for row in sample_manifest],
            "output_dir": str(output_dir),
        }
        write_json_report(output_dir / "dry_run_report.json", report)
        return report

    assert torch is not None and modules is not None
    set_seed(config.benchmark.seed)
    processor = load_qwen3vl_processor(
        modules,
        str(config.model.processor_id),
        {
            "local_files_only": config.model.local_files_only,
            "trust_remote_code": config.model.trust_remote_code,
        },
    )
    baseline_report = None
    quantized_report = None
    variants = planned_variants(backend.name, skip_baseline=skip_baseline)
    if "baseline" in variants:
        baseline_model = backend.load_model(config, modules, quantized=False)
        baseline_report = _run_variant(
            variant="baseline",
            quantized=False,
            model=baseline_model,
            processor=processor,
            dataset=dataset,
            image_root=image_root,
            config=config,
            backend=backend,
            torch=torch,
            output_dir=output_dir,
        )
        del baseline_model
        gc.collect()
        if bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()
    if "quantized" in variants:
        quantized_model = backend.load_model(config, modules, quantized=True)
        quantized_report = _run_variant(
            variant="quantized",
            quantized=True,
            model=quantized_model,
            processor=processor,
            dataset=dataset,
            image_root=image_root,
            config=config,
            backend=backend,
            torch=torch,
            output_dir=output_dir,
        )
        del quantized_model
        gc.collect()
        if bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()
    assert_comparable_sample_ids(baseline_report, quantized_report)
    report = {
        "schema_version": "1.0",
        "success": all(
            result is None or bool(result.get("success"))
            for result in (baseline_report, quantized_report)
        ),
        "dry_run": False,
        "backend": backend.name,
        "seed": config.benchmark.seed,
        "generation": config.generation.model_dump(mode="python"),
        "sample_manifest": str(output_dir / "sample_manifest.jsonl"),
        "environment": environment_metadata(torch),
        "baseline": baseline_report,
        "quantized": quantized_report,
        "comparison": comparison_summary(baseline_report, quantized_report),
    }
    write_json_report(output_dir / "benchmark_report.json", report)
    return report
