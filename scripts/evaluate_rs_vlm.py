"""Qwen3-VL + LoRA 遥感任务评测脚本。"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml

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
    extract_reference,
    timed_predictions,
)
from sat_rs_vlm.evaluation.inference import (
    generate_prediction as _generate_prediction,
)
from sat_rs_vlm.evaluation.metrics import summarize_predictions
from sat_rs_vlm.evaluation.runner import run_evaluation, validate_output_directory
from sat_rs_vlm.evaluation.tiers import resolve_tier_identity, validate_tier_asset
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_CONTRACT = PROJECT_ROOT / "configs/eval/evaluation_contract_v1.5.yaml"


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
        return root / "summary.json", root / "predictions.jsonl", root / "evaluation_v1_5"
    if checkpoint is not None:
        root = checkpoint.resolve() / "evaluation"
        return root / "summary.json", root / "predictions.jsonl", root / "evaluation_v1_5"
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
    if loaded_modules is None:
        require_bitsandbytes = False
        if checkpoint is not None:
            require_bitsandbytes = bool(
                read_strategy_manifest(checkpoint).get("quantized_base", False)
            )
        modules = safe_import_model_dependencies(require_bitsandbytes=require_bitsandbytes)
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
    model.eval()
    torch = modules["torch"]
    generation_cfg = dict(config.get("generation", {}))
    dataset = Qwen3VLDataset(eval_file, data_cfg.get("max_eval_samples"))
    batch_size = int(batch_size_override or data_cfg.get("eval_batch_size", 1))
    group_by_task = bool(data_cfg.get("group_by_task", True))
    log_every = max(1, int(data_cfg.get("log_every_samples", 100)))
    if batch_size < 1:
        raise ValueError(f"Evaluation batch size must be positive, got {batch_size}")
    tokenizer = getattr(processor, "tokenizer", None)
    if batch_size > 1 and tokenizer is not None:
        tokenizer.padding_side = "left"
    collator = Qwen3VLDataCollator(
        processor,
        max_seq_length=int(data_cfg.get("max_seq_length", 4096)),
        image_root=resolve_project_path(str(data_cfg["image_root"])),
        for_generation=True,
    )

    predictions_by_index: list[dict[str, Any] | None] = [None] * len(dataset)
    evaluated = 0
    next_log = 1
    print(
        f"Evaluating {len(dataset)} samples with batch_size={batch_size}, "
        f"group_by_task={group_by_task}"
    )
    for task_type, indexed_batch in iter_evaluation_batches(
        dataset,
        batch_size,
        group_by_task=group_by_task,
    ):
        samples = [sample for _, sample in indexed_batch]
        batch_predictions, latency_ms = timed_predictions(
            model,
            processor,
            collator,
            samples,
            generation_cfg,
            torch,
            task_type=task_type,
        )
        for (original_index, sample), prediction in zip(
            indexed_batch, batch_predictions, strict=True
        ):
            predictions_by_index[original_index] = {
                "id": sample["id"],
                "task_type": sample["task_type"],
                "prediction": prediction,
                "reference": extract_reference(sample["messages"]),
                "metadata": sample.get("metadata", {}),
                "inference_latency_ms": latency_ms,
            }
        evaluated += len(samples)
        if evaluated >= next_log or evaluated == len(dataset):
            print(f"Evaluated {evaluated}/{len(dataset)} samples")
            next_log = ((evaluated // log_every) + 1) * log_every

    if any(prediction is None for prediction in predictions_by_index):
        raise RuntimeError("Evaluation finished with missing predictions")
    predictions = [prediction for prediction in predictions_by_index if prediction is not None]

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
        manifest_path=(resolve_project_path(str(manifest_value)) if manifest_value else None),
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
        evaluation_tier_sha256=tier_identity["sha256"] if tier_identity else None,
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
        "evaluation_outputs": {name: str(path) for name, path in evaluation_outputs.items()},
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
