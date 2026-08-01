"""Qwen3-VL 本地 LoRA/QLoRA 遥感指令微调脚本。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.data.task_sampler import build_weighted_sampler, create_trainer
from sat_rs_vlm.models.qwen3vl_loader import compatible_model_class
from sat_rs_vlm.training.config import (
    Qwen3VLTrainingConfig,
    ResolvedTrainingPaths,
    TrainingPathOverrides,
    apply_training_overrides,
    load_training_config,
    resolve_training_paths,
)
from sat_rs_vlm.training.freeze import freeze_projector, freeze_vision_encoder
from sat_rs_vlm.training.utils import (
    MODEL_DEPS_ERROR,
    count_trainable_parameters,
    model_input_device,
    move_to_device,
    resolve_torch_dtype,
    safe_import_model_dependencies,
    set_seed,
    torch_device_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析训练脚本参数。"""

    parser = argparse.ArgumentParser(description="Train local Qwen3-VL with LoRA/QLoRA.")
    parser.add_argument("--config", default="configs/train/qwen3vl_local_smoke.yaml")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--processor-dir", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--method", choices=("lora", "qlora"), default=None)
    local_group = parser.add_mutually_exclusive_group()
    local_group.add_argument("--local-files-only", dest="local_files_only", action="store_true")
    local_group.add_argument("--no-local-files-only", dest="local_files_only", action="store_false")
    parser.set_defaults(local_files_only=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> TrainingPathOverrides:
    """从 CLI 参数构造训练覆盖项。"""

    return TrainingPathOverrides(
        model_dir=args.model_dir,
        processor_dir=args.processor_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        image_root=args.image_root,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_steps=args.max_steps,
        local_files_only=args.local_files_only,
        method=args.method,
        max_seq_length=args.max_seq_length,
    )


def load_config_with_overrides(args: argparse.Namespace) -> Qwen3VLTrainingConfig:
    """加载配置并应用 CLI 覆盖。"""

    config = load_training_config(args.config, allow_unresolved_env=True)
    config = apply_training_overrides(config, build_overrides(args))
    if args.skip_eval:
        config = config.model_copy(
            update={"evaluation": config.evaluation.model_copy(update={"do_eval": False})}
        )
    return config


def precision_flags(torch: Any, config: Qwen3VLTrainingConfig) -> tuple[bool, bool]:
    """根据设备能力返回 Trainer bf16/fp16 开关。"""

    if not bool(torch.cuda.is_available()):
        return False, False
    if config.training.bf16:
        supported = not hasattr(torch.cuda, "is_bf16_supported") or bool(
            torch.cuda.is_bf16_supported()
        )
        if supported:
            return True, False
        print("WARNING: bfloat16 is not supported; using fp16 Trainer precision.")
        return False, True
    return False, bool(config.training.fp16)


def build_model_kwargs(config: Qwen3VLTrainingConfig, modules: dict[str, Any]) -> dict[str, Any]:
    """构造 from_pretrained 参数。"""

    torch = modules["torch"]
    transformers = modules["transformers"]
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
        "local_files_only": config.model.local_files_only,
        "device_map": config.model.device_map,
        "attn_implementation": config.model.attn_implementation,
    }
    torch_dtype = resolve_torch_dtype(torch, config.model.torch_dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if config.training.method == "qlora":
        bnb_dtype = resolve_torch_dtype(torch, config.qlora.bnb_4bit_compute_dtype)
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=config.qlora.load_in_4bit,
            bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_dtype,
            bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
        )
    return kwargs


def load_qwen3vl_model(
    config: Qwen3VLTrainingConfig,
    paths: ResolvedTrainingPaths,
    modules: dict[str, Any],
) -> Any:
    """加载 Qwen3-VL 模型。"""

    model_cls = compatible_model_class(modules["transformers"])
    return model_cls.from_pretrained(paths.model_source, **build_model_kwargs(config, modules))


def apply_lora(model: Any, config: Qwen3VLTrainingConfig, modules: dict[str, Any]) -> Any:
    """注入 LoRA adapter。"""

    if config.training.method not in {"lora", "qlora"}:
        raise ValueError("Only LoRA/QLoRA are supported by default; full fine-tuning is disabled.")
    peft = modules["peft"]
    lora_config = peft.LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    return peft.get_peft_model(model, lora_config)


def build_training_arguments(
    config: Qwen3VLTrainingConfig,
    paths: ResolvedTrainingPaths,
    transformers: Any,
    torch: Any,
) -> Any:
    """构造 Trainer 参数。"""

    bf16, fp16 = precision_flags(torch, config)
    kwargs: dict[str, Any] = {
        "output_dir": str(paths.output_dir),
        "num_train_epochs": config.training.num_train_epochs,
        "max_steps": config.training.max_steps if config.training.max_steps is not None else -1,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "warmup_ratio": config.training.warmup_ratio,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "logging_steps": config.training.logging_steps,
        "save_steps": config.training.save_steps,
        "eval_steps": config.training.eval_steps,
        "save_strategy": "steps",
        "save_total_limit": config.training.save_total_limit,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "max_grad_norm": config.training.max_grad_norm,
        "report_to": config.logging.report_to,
        "remove_unused_columns": False,
    }
    if config.evaluation.do_eval:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["eval_strategy"] = "no"
    try:
        return transformers.TrainingArguments(**kwargs)
    except TypeError:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return transformers.TrainingArguments(**kwargs)


def load_datasets(
    config: Qwen3VLTrainingConfig,
    paths: ResolvedTrainingPaths,
) -> tuple[Qwen3VLDataset, Qwen3VLDataset | None]:
    """加载训练和验证数据集。"""

    train_dataset = Qwen3VLDataset(
        paths.train_file,
        config.data.max_train_samples,
        skip_bad_samples=config.data.skip_bad_samples,
    )
    eval_dataset = None
    if config.evaluation.do_eval:
        eval_dataset = Qwen3VLDataset(
            paths.val_file,
            config.data.max_eval_samples,
            skip_bad_samples=config.data.skip_bad_samples,
        )
    return train_dataset, eval_dataset


def print_resolved_summary(
    config: Qwen3VLTrainingConfig,
    paths: ResolvedTrainingPaths,
    modules: dict[str, Any] | None,
    train_samples: int,
    eval_samples: int,
    param_stats: tuple[int, int, float] | None = None,
) -> None:
    """打印训练前摘要。"""

    print("Resolved configuration")
    print(f"Model directory: {paths.model_source}")
    print(f"Processor directory: {paths.processor_source}")
    print(f"Train file: {paths.train_file}")
    print(f"Val file: {paths.val_file}")
    print(f"Image root: {paths.image_root}")
    print(f"Output directory: {paths.output_dir}")
    print(f"Local files only: {config.model.local_files_only}")
    if modules is not None:
        torch = modules["torch"]
        transformers = modules["transformers"]
        peft = modules["peft"]
        device = torch_device_summary(torch)
        print(f"Torch version: {getattr(torch, '__version__', 'unknown')}")
        print(f"Transformers version: {getattr(transformers, '__version__', 'unknown')}")
        print(f"PEFT version: {getattr(peft, '__version__', 'unknown')}")
        print(f"CUDA available: {device['cuda_available']}")
        print(f"CUDA device name: {device.get('device_name')}")
    print(f"Training method: {config.training.method}")
    print(f"Data composition: {config.data.data_composition}")
    print(f"Sampling mode: {config.data.sampling_mode}")
    print(f"LoRA rank: {config.lora.r}")
    print(f"Train samples: {train_samples}")
    print(f"Eval samples: {eval_samples}")
    print(f"Max steps: {config.training.max_steps}")
    if param_stats is not None:
        trainable, total, ratio = param_stats
        print(f"Trainable parameters: {trainable}")
        print(f"Total parameters: {total}")
        print(f"Trainable ratio: {ratio:.6f}")


def save_report(report: dict[str, Any], output_dir: Path) -> None:
    """保存 smoke_train_report.json。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    """读取实验关键依赖版本，不额外导入大型模块。"""

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def build_strategy_manifest(
    model: Any,
    config: Qwen3VLTrainingConfig,
    paths: ResolvedTrainingPaths,
    param_stats: tuple[int, int, float],
    device: dict[str, Any],
) -> dict[str, Any]:
    """构造统一 checkpoint 自描述文件，供评估与可靠性流程加载。"""

    parameter = next(iter(model.parameters()), None)
    actual_dtype = str(getattr(parameter, "dtype", "unknown")).removeprefix("torch.")
    targets = tuple(config.lora.target_modules)
    matched_modules = sorted(
        {
            name
            for name, _ in model.named_modules()
            if any(name == target or name.endswith(f".{target}") for target in targets)
        }
    )
    trainable, total, ratio = param_stats
    quantization = None
    if config.training.method == "qlora":
        quantization = {
            "load_in_4bit": config.qlora.load_in_4bit,
            "quant_type": config.qlora.bnb_4bit_quant_type,
            "use_double_quant": config.qlora.bnb_4bit_use_double_quant,
            "compute_dtype": config.qlora.bnb_4bit_compute_dtype,
        }
    return {
        "schema_version": "1.0",
        "strategy": config.training.method,
        "adapter_based": True,
        "quantized_base": config.training.method == "qlora",
        "supports_merge": True,
        "checkpoint_type": "adapter",
        "model_dir": paths.model_source,
        "processor_dir": str(paths.output_dir / "processor"),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": ratio,
        "matched_modules": matched_modules,
        "target_modules": list(targets),
        "actual_dtype": actual_dtype,
        "quantization": quantization,
        "library_versions": _package_versions(
            ("torch", "transformers", "peft", "accelerate", "bitsandbytes")
        ),
        "device": device,
    }


def save_strategy_manifest(manifest: dict[str, Any], output_dir: Path) -> None:
    """把统一策略 manifest 写到 adapter 根目录。"""

    (output_dir / "strategy_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def dry_run(config: Qwen3VLTrainingConfig, paths: ResolvedTrainingPaths) -> None:
    """只检查配置、路径、Dataset 和 Collator 初始化。"""

    if paths.model_dir is not None and not paths.model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {paths.model_dir}")
    if paths.processor_dir is not None and not paths.processor_dir.exists():
        raise FileNotFoundError(f"Processor directory does not exist: {paths.processor_dir}")
    if not paths.train_file.exists():
        raise FileNotFoundError(f"Train file does not exist: {paths.train_file}")
    if not paths.val_file.exists():
        raise FileNotFoundError(f"Val file does not exist: {paths.val_file}")
    train_dataset, eval_dataset = load_datasets(config, paths)
    Qwen3VLDataCollator(None, config.data.max_seq_length, paths.image_root)
    print_resolved_summary(
        config,
        paths,
        modules=None,
        train_samples=len(train_dataset),
        eval_samples=len(eval_dataset) if eval_dataset is not None else 0,
    )
    print("Dry run passed. No model was loaded.")


def forward_only(config: Qwen3VLTrainingConfig, paths: ResolvedTrainingPaths) -> None:
    """执行单 batch 前向传播并打印 loss。"""

    modules = safe_import_model_dependencies(require_bitsandbytes=config.training.method == "qlora")
    torch = modules["torch"]
    transformers = modules["transformers"]
    processor = transformers.AutoProcessor.from_pretrained(
        paths.processor_source,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
    )
    model = load_qwen3vl_model(config, paths, modules)
    train_dataset, _ = load_datasets(config, paths)
    collator = Qwen3VLDataCollator(
        processor,
        config.data.max_seq_length,
        paths.image_root,
        debug_shapes=True,
    )
    batch = collator([train_dataset[0]])
    input_device = model_input_device(model, torch)
    batch = move_to_device(batch, input_device, torch)
    print(f"Forward-only input device: {input_device}")
    if hasattr(model, "eval"):
        model.eval()
    with torch.inference_mode():
        output = model(**batch)
    loss = getattr(output, "loss", None)
    print(f"Forward-only loss: {float(loss.detach().cpu()) if loss is not None else None}")


def train(
    config: Qwen3VLTrainingConfig,
    paths: ResolvedTrainingPaths,
    config_path: Path,
) -> dict[str, Any]:
    """执行 LoRA/QLoRA 训练。"""

    started = time.perf_counter()
    modules = safe_import_model_dependencies(require_bitsandbytes=config.training.method == "qlora")
    torch = modules["torch"]
    transformers = modules["transformers"]
    set_seed(config.training.seed)

    processor = transformers.AutoProcessor.from_pretrained(
        paths.processor_source,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
    )
    model = load_qwen3vl_model(config, paths, modules)
    if config.training.freeze_vision_encoder:
        freeze_vision_encoder(model)
    if config.training.freeze_projector:
        freeze_projector(model)
    if config.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model = apply_lora(model, config, modules)

    train_dataset, eval_dataset = load_datasets(config, paths)
    param_stats = count_trainable_parameters(model)
    print_resolved_summary(
        config,
        paths,
        modules,
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
        param_stats,
    )
    collator = Qwen3VLDataCollator(
        processor,
        config.data.max_seq_length,
        paths.image_root,
        debug_shapes=True,
    )
    train_sampler = None
    if config.data.sampling_mode == "weighted":
        train_sampler = build_weighted_sampler(
            train_dataset,
            config.data.task_sampling_weights,
            seed=config.training.seed,
        )
        print(f"Task-weighted sampling enabled: {config.data.task_sampling_weights}")
    trainer = create_trainer(
        transformers,
        train_sampler=train_sampler,
        trainer_kwargs={
            "model": model,
            "args": build_training_arguments(config, paths, transformers, torch),
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": collator,
        },
    )
    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(paths.output_dir))
    processor_dir = paths.output_dir / "processor"
    processor.save_pretrained(processor_dir)
    trainer.save_state()
    shutil.copyfile(config_path, paths.output_dir / "training_config.yaml")

    final_loss = None
    if trainer.state.log_history:
        for row in reversed(trainer.state.log_history):
            if "loss" in row:
                final_loss = float(row["loss"])
                break
    device = torch_device_summary(torch)
    save_strategy_manifest(
        build_strategy_manifest(model, config, paths, param_stats, device),
        paths.output_dir,
    )
    peak_memory = 0.0
    if bool(torch.cuda.is_available()):
        peak_memory = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return {
        "success": True,
        "model_dir": paths.model_source,
        "train_file": str(paths.train_file),
        "val_file": str(paths.val_file),
        "output_dir": str(paths.output_dir),
        "max_steps": config.training.max_steps,
        "data_composition": config.data.data_composition,
        "sampling_mode": config.data.sampling_mode,
        "task_sampling_weights": config.data.task_sampling_weights,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "final_loss": final_loss,
        "duration_seconds": time.perf_counter() - started,
        "cuda_available": device["cuda_available"],
        "device_name": device.get("device_name"),
        "peak_memory_mb": peak_memory,
    }


def main() -> int:
    """脚本入口。"""

    args = parse_args()
    config_path = Path(args.config)
    config = load_config_with_overrides(args)
    paths = resolve_training_paths(config)
    try:
        if args.dry_run:
            dry_run(config, paths)
            return 0
        if args.forward_only:
            forward_only(config, paths)
            return 0
        report = train(config, paths, config_path)
        save_report(report, paths.output_dir)
        return 0
    except Exception as exc:
        error_report = {"success": False, "error": str(exc)}
        try:
            save_report(error_report, paths.output_dir)
        except (OSError, TypeError) as report_error:
            print(f"WARNING: could not save failure report: {report_error}")
        if isinstance(exc, ImportError):
            raise SystemExit(str(exc) or MODEL_DEPS_ERROR) from exc
        raise


if __name__ == "__main__":
    raise SystemExit(main())
