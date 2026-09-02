"""Run text-only Planner LoRA training through the repository training stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import scripts.train_qwen3vl_lora as shared_training
import yaml

from sat_rs_vlm.training.config import (
    TrainingPathOverrides,
    apply_training_overrides,
    load_training_config,
    resolve_training_paths,
)
from taskgraph_lab.training.planner_collator import PlannerTextDataCollator
from taskgraph_lab.training.planner_dataset import file_sha256

LAB_ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_SCOPE = ".language_model."


def audit_language_only_trainables(model: Any) -> dict[str, Any]:
    trainable = [
        (name, int(parameter.numel()))
        for name, parameter in model.named_parameters()
        if bool(parameter.requires_grad)
    ]
    if not trainable:
        raise ValueError("Planner training has no trainable parameters")
    invalid = [
        name for name, _ in trainable if LANGUAGE_SCOPE not in f".{name}" or "lora_" not in name
    ]
    if invalid:
        raise ValueError(
            "Planner training exposed parameters outside language-model LoRA: "
            + ", ".join(invalid[:50])
        )
    return {
        "scope": "qwen3_vl.language_model.lora_only",
        "parameter_count": sum(count for _, count in trainable),
        "tensor_count": len(trainable),
        "sample_names": [name for name, _ in trainable[:50]],
        "invalid_names": [],
    }


def install_planner_training_boundaries() -> None:
    """Inject the lab collator and add a strict trainable-scope gate."""

    original_apply_lora = shared_training.apply_lora

    def planner_apply_lora(
        model: Any,
        config: Any,
        paths: Any,
        modules: dict[str, Any],
    ) -> Any:
        prepared = original_apply_lora(model, config, paths, modules)
        audit = audit_language_only_trainables(prepared)
        prepared._taskgraph_planner_trainable_audit = audit
        shared_training._taskgraph_planner_trainable_audit = audit
        print("Planner trainable audit: " + json.dumps(audit, sort_keys=True), flush=True)
        return prepared

    shared_training.Qwen3VLDataCollator = PlannerTextDataCollator
    shared_training.apply_lora = planner_apply_lora


def validate_config(config: Any) -> None:
    if config.training.method != "lora":
        raise ValueError("Local Planner prototype requires training.method=lora")
    if not config.training.freeze_vision_encoder:
        raise ValueError("Planner training must freeze the vision encoder")
    if not config.training.freeze_projector:
        raise ValueError("Planner training must freeze the visual projector")
    if config.vision_tuning.enabled:
        raise ValueError("Planner training must not enable vision_tuning")
    expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if set(config.lora.target_modules) != expected:
        raise ValueError("Planner LoRA target_modules must be the seven language projections")


def add_planner_provenance(output_dir: Path, paths: Any, report: dict[str, Any]) -> None:
    dataset_manifest_path = paths.train_file.parent / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    planner = {
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "population_count": dataset_manifest["population_count"],
        "target_format": dataset_manifest["target_format"],
        "planner_dsl_version": dataset_manifest["planner_dsl_version"],
        "prompt_version": dataset_manifest["prompt_version"],
        "trainable_scope": "qwen3_vl.language_model.lora_only",
    }
    report["taskgraph_planner"] = planner
    shared_training.save_report(report, output_dir)
    strategy_path = output_dir / "strategy_manifest.json"
    if strategy_path.is_file():
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
        strategy["taskgraph_planner"] = planner
        strategy_path.write_text(
            json.dumps(strategy, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL-2B language LoRA as Planner")
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "qwen3vl_2b_planner_lora_local.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_training_config(args.config, allow_unresolved_env=True)
    config = apply_training_overrides(
        config,
        TrainingPathOverrides(
            output_dir=str(args.output_dir) if args.output_dir is not None else None,
            max_steps=args.max_steps,
        ),
    )
    validate_config(config)
    paths = resolve_training_paths(config)
    manifest_path = paths.train_file.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Planner dataset manifest is missing: {manifest_path}")
    install_planner_training_boundaries()
    if args.dry_run:
        shared_training.dry_run(config, paths)
        return 0
    if args.forward_only:
        shared_training.forward_only(config, paths)
        return 0
    existing_artifacts = (
        [
            path
            for path in paths.output_dir.iterdir()
            if path.name not in {"train.stdout.log", "train.stderr.log"}
        ]
        if paths.output_dir.exists()
        else []
    )
    if existing_artifacts:
        raise FileExistsError(
            f"Planner output directory is not empty; choose a new run directory: {paths.output_dir}"
        )
    try:
        report = shared_training.train(config, paths, args.config)
        report["planner_trainable_audit"] = getattr(
            shared_training, "_taskgraph_planner_trainable_audit", None
        )
        add_planner_provenance(paths.output_dir, paths, report)
    except Exception as exc:
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        shared_training.save_report(
            {"success": False, "error_type": type(exc).__name__, "error": str(exc)},
            paths.output_dir,
        )
        raise
    print(yaml.safe_dump(report, allow_unicode=True, sort_keys=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
