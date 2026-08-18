"""本地与云端共用的 LoRA 训练包装入口。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.configuration.layered import (  # noqa: E402
    LayeredConfigRequest,
    load_layered_config,
    write_resolved_config,
)
from sat_rs_vlm.configuration.paths import PathConfig, resolve_path_value  # noqa: E402
from sat_rs_vlm.configuration.precision import select_precision  # noqa: E402
from sat_rs_vlm.data.manifest import (  # noqa: E402
    load_dataset_manifest,
    resolve_split_path,
    validate_dataset,
)
from sat_rs_vlm.training.experiment import (  # noqa: E402
    create_experiment_layout,
    disk_report,
    environment_snapshot,
    git_commit,
    resolve_resume_checkpoint,
    write_json,
)

BASE_CONFIGS = (
    PROJECT_ROOT / "configs/base/default.yaml",
    PROJECT_ROOT / "configs/base/model/qwen3_vl_2b.yaml",
    PROJECT_ROOT / "configs/base/dataset/vrsbench.yaml",
    PROJECT_ROOT / "configs/base/training/lora.yaml",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析统一训练参数，并保留旧 LoRA 常用路径覆盖。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="实验或环境训练配置。")
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--base-config", type=Path, action="append")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--environment", choices=("local", "autodl"), default="local")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--processor-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--mock", action="store_true", help="不加载真实模型的控制流 smoke。")
    return parser.parse_args(argv)


def _environment_config(name: str) -> Path:
    return (
        PROJECT_ROOT / "configs/cloud/autodl.yaml"
        if name == "autodl"
        else PROJECT_ROOT / "configs/local/paths.yaml"
    )


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "paths.dataset_root": str(args.dataset_root) if args.dataset_root else None,
        "model.model_path": str(args.model_dir) if args.model_dir else None,
        "model.processor_path": str(args.processor_dir) if args.processor_dir else None,
        "data.manifest_path": str(args.manifest) if args.manifest else None,
        "data.max_train_samples": args.max_train_samples,
        "data.max_validation_samples": args.max_eval_samples,
        "training.max_steps": args.max_steps,
    }


def _effective_environment(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    mock = args.mock
    if args.config and args.config.name == "train_lora_smoke.yaml":
        mock = True
    if mock:
        env.setdefault("DATA_ROOT", str(PROJECT_ROOT / "tests/fixtures/miniature_dataset"))
        env.setdefault("MODEL_ROOT", str(PROJECT_ROOT / ".models"))
    return env


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    """加载基础、环境、实验、环境变量与 CLI 五层配置。"""

    experiment = args.experiment_config or args.config
    base_configs = (*BASE_CONFIGS, *(args.base_config or ()))
    request = LayeredConfigRequest(
        base_configs=base_configs,
        environment_config=args.env_config or _environment_config(args.environment),
        experiment_config=experiment,
        cli_overrides=_cli_overrides(args),
        project_root=PROJECT_ROOT,
    )
    return load_layered_config(request, environ=_effective_environment(args))


def _manifest_root(manifest_path: Path) -> Path:
    if manifest_path.parent.name == "project_metadata":
        return manifest_path.parent.parent
    return manifest_path.parent


def _cuda_capabilities() -> tuple[bool, bool]:
    try:
        import torch
    except ImportError:
        return False, False
    available = bool(torch.cuda.is_available())
    supported = bool(available and torch.cuda.is_bf16_supported())
    return available, supported


def _resolve_precision(config: dict[str, Any], *, mock: bool) -> dict[str, Any]:
    training = dict(config.get("training", {}))
    runtime = dict(config.get("runtime", {}))
    cuda_available, bf16_supported = (False, False) if mock else _cuda_capabilities()
    decision = select_precision(
        device="cpu" if mock else str(runtime.get("device", "auto")),
        bf16=training.get("bf16"),
        fp16=training.get("fp16"),
        cuda_available=cuda_available,
        bf16_supported=bf16_supported,
    )
    training["bf16"] = decision.bf16
    training["fp16"] = decision.fp16
    config["training"] = training
    return {
        "mode": decision.mode,
        "reason": decision.reason,
        "cuda_available": cuda_available,
        "bf16_supported": bf16_supported,
    }


def _resolve_assets(
    config: dict[str, Any],
    paths: PathConfig,
    *,
    mock: bool,
) -> tuple[Path, Path, Path]:
    data = dict(config["data"])
    manifest_path = resolve_path_value(str(data["manifest_path"]), base_dir=PROJECT_ROOT)
    dataset_root = _manifest_root(manifest_path)
    manifest = load_dataset_manifest(manifest_path)
    train_file = resolve_split_path(dataset_root, manifest, str(data.get("split", "train")))
    val_file = resolve_split_path(
        dataset_root,
        manifest,
        str(data.get("validation_split", "validation")),
    )
    report = validate_dataset(
        dataset_root,
        manifest_name=str(manifest_path.relative_to(dataset_root)),
        sample_images=4,
    )
    if not report.valid:
        raise ValueError("Dataset validation failed: " + "; ".join(report.errors[:5]))
    if not mock:
        model_path = resolve_path_value(str(config["model"]["model_path"]), base_dir=PROJECT_ROOT)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local model directory does not exist: {model_path}")
        paths.validate_inputs(require_dataset=True, require_model=True)
    return dataset_root, train_file, val_file


def _legacy_config(
    config: dict[str, Any],
    *,
    experiment_dir: Path,
    dataset_root: Path,
    train_file: Path,
    val_file: Path,
    resume: Path | None,
) -> dict[str, Any]:
    model = dict(config["model"])
    training = dict(config["training"])
    lora = dict(config["lora"])
    data = dict(config["data"])
    runtime = dict(config.get("runtime", {}))
    model_path = resolve_path_value(str(model["model_path"]), base_dir=PROJECT_ROOT)
    processor_path = resolve_path_value(
        str(model.get("processor_path", model_path)),
        base_dir=PROJECT_ROOT,
    )
    return {
        "model": {
            "model_dir": str(model_path),
            "processor_dir": str(processor_path),
            "local_files_only": bool(model.get("local_files_only", True)),
            "trust_remote_code": bool(model.get("trust_remote_code", True)),
            "torch_dtype": str(model.get("torch_dtype", "auto")),
            "device_map": str(model.get("device_map", "auto")),
            "attn_implementation": str(model.get("attn_implementation", "sdpa")),
        },
        "data": {
            "train_file": str(train_file),
            "val_file": str(val_file),
            "image_root": str(dataset_root),
            "max_seq_length": int(data.get("max_seq_length", 1024)),
            "max_train_samples": data.get("max_train_samples"),
            "max_eval_samples": data.get("max_validation_samples"),
            "skip_bad_samples": bool(data.get("skip_bad_samples", False)),
            "data_composition": str(data.get("data_composition", "full")),
            "sampling_mode": str(data.get("sampling_mode", "uniform")),
            "task_sampling_weights": dict(data.get("task_sampling_weights", {})),
        },
        "training": {
            "output_dir": str(experiment_dir / "checkpoints"),
            "method": "lora",
            "freeze_vision_encoder": bool(training.get("freeze_vision_encoder", True)),
            "freeze_projector": bool(training.get("freeze_projector", False)),
            "num_train_epochs": int(training.get("num_train_epochs", 2)),
            "max_steps": training.get("max_steps"),
            "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 1)),
            "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 1)),
            "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 16)),
            "learning_rate": float(training.get("learning_rate", 1e-4)),
            "weight_decay": float(training.get("weight_decay", 0.01)),
            "warmup_ratio": float(training.get("warmup_ratio", 0.03)),
            "lr_scheduler_type": str(training.get("lr_scheduler_type", "cosine")),
            "logging_steps": int(training.get("logging_steps", 10)),
            "save_steps": int(training.get("save_steps", 200)),
            "eval_steps": int(training.get("eval_steps", 200)),
            "save_total_limit": int(training.get("save_total_limit", 2)),
            "bf16": bool(training["bf16"]),
            "fp16": bool(training["fp16"]),
            "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
            "dataloader_num_workers": int(
                training.get("dataloader_num_workers", runtime.get("num_workers", 0))
            ),
            "dataloader_pin_memory": bool(
                training.get("dataloader_pin_memory", runtime.get("pin_memory", True))
            ),
            "dataloader_persistent_workers": bool(
                training.get(
                    "dataloader_persistent_workers",
                    runtime.get("persistent_workers", False),
                )
            ),
            "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
            "seed": int(training.get("seed", config.get("experiment", {}).get("seed", 42))),
            "resume_from_checkpoint": str(resume) if resume else None,
        },
        "lora": {
            "r": int(lora.get("rank", lora.get("r", 16))),
            "alpha": int(lora.get("alpha", 32)),
            "dropout": float(lora.get("dropout", 0.05)),
            "target_modules": list(lora.get("target_modules", [])),
        },
        "qlora": {
            "load_in_4bit": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        },
        "evaluation": dict(config.get("evaluation", {})),
        "logging": {
            "report_to": "none",
            "experiment_name": str(config.get("experiment", {}).get("name", "lora")),
        },
    }


def _run_mock(
    experiment_dir: Path,
    config: dict[str, Any],
    train_file: Path,
    resume: Path | None,
) -> None:
    from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset

    started = time.perf_counter()
    maximum = config["data"].get("max_train_samples")
    dataset = Qwen3VLDataset(train_file, maximum)
    if not dataset:
        raise ValueError("Mock training requires at least one training sample.")
    steps = int(config["training"].get("max_steps") or 1)
    checkpoint = experiment_dir / "checkpoints" / f"checkpoint-{steps}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    write_json(
        checkpoint / "trainer_state.json",
        {"global_step": steps, "mock": True, "resumed_from": str(resume) if resume else None},
    )
    write_json(
        experiment_dir / "metrics/train.json",
        {"loss": 0.0, "steps": steps, "samples": len(dataset), "mock": True},
    )
    write_json(
        experiment_dir / "artifacts/mock_adapter.json",
        {"format": "mock", "note": "No model weights were created."},
    )
    write_json(
        experiment_dir / "train_report.json",
        {
            "success": True,
            "mode": "mock",
            "samples": len(dataset),
            "steps": steps,
            "duration_seconds": time.perf_counter() - started,
            "checkpoint": str(checkpoint),
        },
    )
    (experiment_dir / "logs/train.log").write_text(
        f"mock training completed: samples={len(dataset)} steps={steps}\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> Path:
    """执行预检，然后运行 Mock 或委托稳定 LoRA 脚本。"""

    config = load_config(args)
    mock = bool(args.mock or config.get("runtime", {}).get("mock", False))
    precision = _resolve_precision(config, mock=mock)
    path_config = PathConfig.from_mapping(
        config.get("paths", {}),
        project_root=PROJECT_ROOT,
        environ=_effective_environment(args),
    )
    path_config.create_output_directories()
    dataset_root, train_file, val_file = _resolve_assets(config, path_config, mock=mock)
    experiment = dict(config.get("experiment", {}))
    training = dict(config.get("training", {}))
    output = dict(config.get("output", {}))
    seed = int(experiment.get("seed", training.get("seed", 42)))
    requested_resume = "latest" if args.resume_latest else args.resume_from_checkpoint
    explicit_output = args.output_dir
    if args.resume_latest and explicit_output is None:
        raise ValueError("--resume-latest requires --output-dir to identify the experiment.")
    if args.resume_from_checkpoint and explicit_output is None:
        checkpoint_path = Path(args.resume_from_checkpoint).expanduser().resolve()
        if checkpoint_path.parent.name != "checkpoints":
            raise ValueError(
                "Without --output-dir, a resume checkpoint must be "
                "<experiment>/checkpoints/checkpoint-<step>."
            )
        explicit_output = checkpoint_path.parent.parent
    experiment_dir = create_experiment_layout(
        path_config.output_root,
        group=str(output.get("experiment_group", training.get("output_name", "lora"))),
        experiment_name=str(experiment.get("name", "lora")),
        seed=seed,
        explicit_output=explicit_output,
    )
    resume = resolve_resume_checkpoint(requested_resume, experiment_dir)
    config.setdefault("training", {})["resume_from_checkpoint"] = str(resume) if resume else None
    write_resolved_config(config, experiment_dir / "config_resolved.yaml")
    write_json(experiment_dir / "environment.json", environment_snapshot())
    write_json(
        experiment_dir / "preflight.json",
        {
            "dataset_root": str(dataset_root),
            "train_file": str(train_file),
            "validation_file": str(val_file),
            "precision": precision,
            "disk": disk_report(experiment_dir),
            "mock": mock,
        },
    )
    (experiment_dir / "git_commit.txt").write_text(
        git_commit(PROJECT_ROOT) + "\n",
        encoding="utf-8",
    )
    legacy = _legacy_config(
        config,
        experiment_dir=experiment_dir,
        dataset_root=dataset_root,
        train_file=train_file,
        val_file=val_file,
        resume=resume,
    )
    legacy_path = experiment_dir / "artifacts/legacy_training_config.yaml"
    legacy_path.write_text(
        yaml.safe_dump(legacy, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    command = [sys.executable, str(PROJECT_ROOT / "scripts/train_qwen3vl_lora.py")]
    command.extend(["--config", str(legacy_path)])
    if args.dry_run:
        command.append("--dry-run")
    if args.forward_only:
        command.append("--forward-only")
    if args.skip_eval:
        command.append("--skip-eval")
    (experiment_dir / "command.txt").write_text(
        subprocess.list2cmdline(command) + "\n",
        encoding="utf-8",
    )
    if mock:
        _run_mock(experiment_dir, config, train_file, resume)
    else:
        environment = dict(os.environ)
        environment.update(path_config.as_environment())
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            write_json(
                experiment_dir / "train_report.json",
                {"success": False, "returncode": completed.returncode},
            )
            raise RuntimeError(
                f"Stable LoRA training entry failed with exit code {completed.returncode}."
            )
    print(json.dumps({"success": True, "experiment_dir": str(experiment_dir)}, indent=2))
    return experiment_dir


def main() -> int:
    """命令行入口。"""

    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
