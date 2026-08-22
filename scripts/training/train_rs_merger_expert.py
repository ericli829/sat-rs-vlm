"""Train C1/C2/C3 from the frozen, provenance-checked R1 foundation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.models.rs_merger_expert import (
    RSMergerExpertController,
    source_architecture_audit,
    validate_expected_qwen4b_contract,
)
from sat_rs_vlm.training.rs_merger_expert import (
    capture_parity_snapshot,
    compare_parity_snapshots,
    expert_step0_parity,
    load_r1_foundation,
    resolve_effective_epoch_plan,
    run_training,
    save_composite_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--output-root")
    return parser.parse_args()


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _probe_batch(processor: Any, config: dict[str, Any], max_samples: int = 2) -> dict[str, Any]:
    dataset = Qwen3VLDataset(config["data"]["train_file"], max_samples=max_samples)
    if len(dataset) < 1:
        raise ValueError("At least one fixed training sample is required for parity")
    collator = Qwen3VLDataCollator(
        processor,
        int(config["training"].get("max_seq_length", 2048)),
        config["data"]["image_root"],
    )
    return collator([dataset[index] for index in range(len(dataset))])


def main() -> int:
    args = parse_args()
    if args.dry_run and args.forward_only:
        raise ValueError("--dry-run and --forward-only are mutually exclusive")
    config = _expand(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    modules = {"torch": torch, "transformers": transformers, "peft": peft}
    processor = transformers.AutoProcessor.from_pretrained(
        config["model"]["base_model"], local_files_only=True
    )
    probe = _probe_batch(processor, config)
    model, processor, r1_report = load_r1_foundation(
        modules=modules,
        base_model=config["model"]["base_model"],
        r1_checkpoint=config["model"]["r1_checkpoint"],
        visual_sidecar=config["model"]["visual_sidecar"],
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "local_files_only": True,
        },
        integration=config["model"].get("r1_integration", "merge"),
        probe_batch=probe,
    )
    architecture = source_architecture_audit(model)
    validate_expected_qwen4b_contract(architecture)
    foundation_probe = capture_parity_snapshot(model, probe)
    expert = config["expert"]
    controller = RSMergerExpertController(
        model,
        variant=expert["expert_variant"],
        interface_lora_enabled=bool(expert["interface_lora"]["enabled"]),
        detail_hidden_size=int(expert.get("detail_hidden_size", 512)),
        lora_rank=int(expert["interface_lora"].get("r", 16)),
        lora_alpha=float(expert["interface_lora"].get("alpha", 32)),
        lora_dropout=float(expert["interface_lora"].get("dropout", 0.05)),
    )
    trainable_audit = controller.freeze_base_and_enable_expert()
    controller.set_active_expert("base")
    base_route_probe = capture_parity_snapshot(model, probe)
    base_route_parity = compare_parity_snapshots(foundation_probe, base_route_probe)
    if not base_route_parity["passed"]:
        raise ValueError(f"Foundation/base route parity failed: {base_route_parity}")
    step0 = expert_step0_parity(model, controller, probe)
    if args.resume_from_checkpoint:
        from safetensors.torch import load_file

        resume_root = Path(args.resume_from_checkpoint)
        resume_weights = resume_root / "expert_model.safetensors"
        if not resume_weights.is_file():
            raise FileNotFoundError(f"Resume expert weights are missing: {resume_weights}")
        controller.load_expert_state_dict(load_file(str(resume_weights), device="cpu"))
    dataset = Qwen3VLDataset(config["data"]["train_file"], max_samples=args.max_train_samples)
    plan = resolve_effective_epoch_plan(
        len(dataset),
        per_device_batch_size=int(config["training"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        target_effective_epochs=float(config["training"].get("target_effective_epochs", 1.0)),
        max_steps=args.max_steps,
    )
    preflight = {
        "experiment": config["experiment"],
        "architecture": architecture,
        "r1_integration": r1_report,
        "resume": {
            "source": args.resume_from_checkpoint,
            "expert_weights_restored": bool(args.resume_from_checkpoint),
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
        },
        "foundation_base_route_parity": base_route_parity,
        "step0_parity": step0,
        "trainable_audit": trainable_audit,
        "training_plan": plan.__dict__,
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_root or config["output"]["root"]) / f"{config['experiment']}_{timestamp}"
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = run_training(
        model=model,
        processor=processor,
        controller=controller,
        train_file=config["data"]["train_file"],
        image_root=config["data"]["image_root"],
        output_dir=root,
        training_config=config["training"],
        max_train_samples=args.max_train_samples,
        max_steps=args.max_steps,
        forward_only=args.forward_only,
    )
    if args.forward_only:
        (root / "forward_only.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return 0
    audit_path = Path(config["provenance"]["architecture_audit"])
    r1_manifest = Path(config["model"]["r1_checkpoint"]) / "strategy_manifest.json"
    data_manifest = Path(config["data"]["manifest"])
    manifest = {
        "schema_version": "1.0",
        "experiment": config["experiment"],
        "variant": expert["variant"],
        "task": "counting",
        "source_r1_checkpoint": config["model"]["r1_checkpoint"],
        "source_r1_manifest_sha256": file_sha256(r1_manifest),
        "source_visual_sidecar_sha256": file_sha256(config["model"]["visual_sidecar"]),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "peft_version": peft.__version__,
        "architecture_audit_sha256": file_sha256(audit_path),
        "train_data_sha256": file_sha256(config["data"]["train_file"]),
        "train_data_manifest_sha256": file_sha256(data_manifest),
        "selected_vit_blocks": architecture["deepstack_visual_indexes"]
        + [architecture["vision_block_count"] - 1],
        "spatial_merge_size": architecture["spatial_merge_size"],
        "visual_hidden_size": architecture["vision_hidden_size"],
        "llm_hidden_size": architecture["llm_hidden_size"],
        "output_visual_token_policy": "same_as_qwen",
        "merger_parameter_count": trainable_audit["expert_parameter_count"],
        "interface_lora_parameter_count": trainable_audit["interface_lora_parameter_count"],
        "total_trainable_parameter_count": trainable_audit["total_trainable_parameter_count"],
        "interface_lora": {
            "layers": [0, 1, 2, 3],
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            **expert["interface_lora"],
        },
        "initialization": {"step0_functionally_equal_to_r1": True},
        "route": {"counting": "counting_expert", "fallback": "base_r1"},
        "r1_integration": r1_report["implementation"],
    }
    save_composite_checkpoint(
        controller,
        root / "checkpoint",
        manifest=manifest,
        training_summary=summary,
        resolved_config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
