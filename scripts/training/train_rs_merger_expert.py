"""Train C1/C2/C3 from the frozen, provenance-checked R1 foundation."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml
from safetensors.torch import load_file

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.evaluation.rs_merger_expert import evaluate_rows
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
    inspect_expert_checkpoint,
    load_expert_checkpoint,
    load_r1_foundation,
    resolve_effective_epoch_plan,
    run_training,
    save_composite_checkpoint,
    validate_expert_checkpoint_compatibility,
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
    parser.add_argument(
        "--resume-expert-checkpoint",
        help="Expert sidecar directory containing expert_model.safetensors/manifest.",
    )
    parser.add_argument("--output-root")
    parser.add_argument(
        "--target-total-effective-epochs",
        type=float,
        choices=(2.0, 3.0, 4.0),
        help="Continuation target; resolves extra epochs from parent checkpoint metadata.",
    )
    parser.add_argument(
        "--finalize-existing-run",
        metavar="RUN_DIR",
        help="Recover fixed evaluation and final checkpoint artifacts without retraining.",
    )
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


def _build_expert_manifest(
    *,
    config: dict[str, Any],
    expert: dict[str, Any],
    architecture: dict[str, Any],
    r1_report: dict[str, Any],
    audit_sha: str,
    r1_manifest_sha: str,
    sidecar_sha: str,
    data_manifest: Path,
    trainable_audit: dict[str, Any],
    resume_report: dict[str, Any],
    transformers: Any,
    torch: Any,
    peft: Any,
    finalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "experiment": config["experiment"],
        "variant": expert["variant"],
        "expert_variant": expert["expert_variant"],
        "detail_hidden_size": int(expert.get("detail_hidden_size", 512)),
        "local_depth": int(expert.get("local_depth", 1)),
        "task": "counting",
        "source_r1_checkpoint": config["model"]["r1_checkpoint"],
        "source_r1_manifest_sha256": r1_manifest_sha,
        "source_visual_sidecar_sha256": sidecar_sha,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "peft_version": peft.__version__,
        "foundation_precision": (
            "real_4bit_nf4_bf16_compute"
            if bool(config["model"].get("load_in_4bit", False))
            else "real_bf16"
        ),
        "architecture_audit_sha256": audit_sha,
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
        "count_head_parameter_count": trainable_audit["count_head_parameter_count"],
        "count_loss": dict(config["training"].get("count_loss", {})),
        "interface_lora": {
            "layers": [0, 1, 2, 3],
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            **expert["interface_lora"],
        },
        "initialization": {"step0_functionally_equal_to_r1_before_continuation_load": True},
        "route": {"counting": "counting_expert", "fallback": "base_r1"},
        "r1_integration": r1_report["implementation"],
        "continuation": resume_report,
        "parent_checkpoint": None,
    }
    if finalization is not None:
        payload["finalization"] = finalization
    return payload


def _write_learning_curve(root: Path, learning_curve: list[dict[str, Any]]) -> None:
    (root / "learning_curve.json").write_text(
        json.dumps(learning_curve, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Fixed E_COUNT_V2 learning curve",
        "",
        "| checkpoint | exact | within-1 | MAE |",
        "|---|---:|---:|---:|",
    ]
    for point in learning_curve:
        overall = point["metrics"]["counting_focused"]["overall"]
        lines.append(
            f"| {point['checkpoint']} | {overall.get('exact')} | "
            f"{overall.get('within_1')} | {overall.get('mae')} |"
        )
    (root / "learning_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_fixed_curve(
    *,
    root: Path,
    config: dict[str, Any],
    expert: dict[str, Any],
    model: Any,
    processor: Any,
    controller: Any,
    load_weights: bool = True,
    require_epoch_checkpoints: bool = False,
    max_eval_samples: int | None = None,
) -> list[dict[str, Any]]:
    fixed_eval = config["data"].get("fixed_eval")
    fixed_manifest = config["data"].get("fixed_eval_manifest")
    if not fixed_eval:
        return []
    if not fixed_manifest:
        raise ValueError("fixed_eval_manifest must be explicitly configured with fixed_eval")
    if "${" in str(fixed_eval) or "${" in str(fixed_manifest):
        raise ValueError("Fixed evaluation tier paths are unresolved")
    eval_image_root = config["data"].get("eval_image_root", config["data"]["image_root"])
    if "${" in str(eval_image_root):
        raise ValueError("Fixed evaluation image root is unresolved; set EVAL_DATA_ROOT")
    epoch_weights = sorted(
        (root / "epoch_checkpoints").glob("epoch_*/expert_model.safetensors"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    if require_epoch_checkpoints and not epoch_weights:
        raise FileNotFoundError("No completed epoch checkpoints are available for recovery")
    evaluation_sources: list[tuple[str, Path | None]] = [
        (path.parent.name, path) for path in epoch_weights
    ]
    if not evaluation_sources:
        evaluation_sources = [("final_partial_epoch", None)]
    final_state = {
        name: value.detach().cpu().clone()
        for name, value in controller.expert_state_dict().items()
    }
    learning_curve: list[dict[str, Any]] = []
    try:
        for label, weights_path in evaluation_sources:
            if load_weights and weights_path is not None:
                controller.load_expert_state_dict(load_file(str(weights_path), device="cpu"))
            metrics = evaluate_rows(
                model=model,
                processor=processor,
                controller=controller,
                tier_file=fixed_eval,
                tier_manifest=fixed_manifest,
                image_root=eval_image_root,
                output_dir=root / "fixed_eval" / label,
                expert_variant=str(expert["variant"]),
                max_eval_samples=max_eval_samples,
                max_new_tokens=int(config["training"].get("eval_max_new_tokens", 64)),
                eval_batch_size=int(config["training"].get("eval_batch_size", 4)),
            )
            learning_curve.append({"checkpoint": label, "metrics": metrics})
    finally:
        controller.load_expert_state_dict(final_state)
    _write_learning_curve(root, learning_curve)
    return learning_curve


def _load_recovered_training_summary(root: Path, torch: Any) -> dict[str, Any]:
    state_path = root / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Completed run is missing training_state.pt: {state_path}")
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("training_state.pt must contain a mapping")
    preflight_path = root / "preflight.json"
    preflight = (
        json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight_path.is_file()
        else {}
    )
    plan = dict(preflight.get("training_plan", {}))
    log_path = root / "train_log.jsonl"
    last_log: dict[str, Any] = {}
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last_log = json.loads(line)
    summary: dict[str, Any] = {
        "mode": "recovered_post_training",
        "recovered_from_existing_run": True,
        "training_reexecuted": False,
        "completed_effective_epochs": float(state.get("completed_effective_epochs", 0.0)),
        "global_optimizer_steps": int(
            state.get("global_optimizer_step", last_log.get("global_optimizer_step", 0))
        ),
        "optimizer_steps": int(
            last_log.get("optimizer_step", state.get("global_optimizer_step", 0))
        ),
        "loss_total": last_log.get("loss_total"),
        "elapsed_seconds": last_log.get("elapsed_seconds"),
        "trainable_params": dict(preflight.get("trainable_audit", {})).get(
            "total_trainable_parameter_count"
        ),
        "trainable_audit": preflight.get("trainable_audit"),
        "plan": plan,
        "training_state_path": state_path.as_posix(),
        "recovery_sources": [
            name
            for name in (
                "training_state.pt",
                "train_log.jsonl",
                "memory_log.json",
                "preflight.json",
            )
            if (root / name).is_file()
        ],
    }
    memory_path = root / "memory_log.json"
    if memory_path.is_file():
        summary["memory"] = json.loads(memory_path.read_text(encoding="utf-8"))
    return summary


def finalize_existing_run(
    *,
    root: Path,
    config: dict[str, Any],
    expert: dict[str, Any],
    model: Any,
    processor: Any,
    controller: Any,
    manifest: dict[str, Any],
    torch: Any,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    """Finalize a completed training run without constructing an optimizer."""

    state_path = root / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Finalize run is missing training_state.pt: {state_path}")
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")
    completed = float(state.get("completed_effective_epochs", 0.0))
    target = float(config["training"].get("target_effective_epochs", 1.0))
    if completed + 1e-9 < target:
        raise ValueError(
            "Cannot finalize incomplete run: "
            f"completed_effective_epochs={completed}, target={target}"
        )
    epoch_weights = sorted(
        (root / "epoch_checkpoints").glob("epoch_*/expert_model.safetensors"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    expected_epoch = int(target + 0.999999)
    available_epochs = [int(path.parent.name.split("_")[-1]) for path in epoch_weights]
    if any(epoch not in available_epochs for epoch in range(1, expected_epoch + 1)):
        raise FileNotFoundError(
            "Finalize run requires "
            f"epoch_01..epoch_{expected_epoch:02d}; available={available_epochs}"
        )
    if any(epoch > expected_epoch for epoch in available_epochs):
        raise ValueError(
            f"Finalize run has epoch checkpoints beyond target epoch {expected_epoch}: "
            f"available={available_epochs}"
        )
    final_epoch_path = root / "epoch_checkpoints" / f"epoch_{expected_epoch:02d}"
    epoch_manifest_path = final_epoch_path / "epoch_manifest.json"
    if not epoch_manifest_path.is_file():
        raise FileNotFoundError(f"Final epoch manifest is missing: {epoch_manifest_path}")
    epoch_manifest = json.loads(epoch_manifest_path.read_text(encoding="utf-8"))
    if float(epoch_manifest.get("completed_effective_epochs", 0.0)) + 1e-9 < target:
        raise ValueError("Final epoch checkpoint is not marked as a complete target epoch")
    final_weights_path = final_epoch_path / "expert_model.safetensors"
    controller.load_expert_state_dict(load_file(str(final_weights_path), device="cpu"))
    final_state = {
        name: value.detach().cpu().clone()
        for name, value in controller.expert_state_dict().items()
    }
    summary = _load_recovered_training_summary(root, torch)
    learning_curve = _evaluate_fixed_curve(
        root=root,
        config=config,
        expert=expert,
        model=model,
        processor=processor,
        controller=controller,
        require_epoch_checkpoints=True,
        max_eval_samples=max_eval_samples,
    )
    summary["fixed_eval_learning_curve"] = learning_curve
    summary["finalization"] = {
        "mode": "recovered_post_training",
        "source_run": root.as_posix(),
        "source_epoch": expected_epoch,
        "training_reexecuted": False,
    }
    manifest["finalization"] = summary["finalization"]
    checkpoint_root = root / "checkpoint"
    if checkpoint_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing final checkpoint during recovery: {checkpoint_root}"
        )
    save_composite_checkpoint(
        controller,
        checkpoint_root,
        manifest=manifest,
        training_summary=summary,
        resolved_config=config,
        training_state_path=state_path,
    )
    saved_state = load_file(str(checkpoint_root / "expert_model.safetensors"), device="cpu")
    if set(saved_state) != set(final_state) or any(
        not torch.equal(saved_state[name], final_state[name]) for name in final_state
    ):
        raise AssertionError("Recovered final checkpoint differs from final epoch expert state")
    (root / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    if args.dry_run and args.forward_only:
        raise ValueError("--dry-run and --forward-only are mutually exclusive")
    config = _expand(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    torch = transformers = peft = None
    model = processor = controller = probe = dataset = summary = None
    foundation_probe = base_route_probe = continuation_base_probe = None
    restored_state = None
    root: Path | None = None
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        modules = {"torch": torch, "transformers": transformers, "peft": peft}
        processor = transformers.AutoProcessor.from_pretrained(
            config["model"]["base_model"], local_files_only=True
        )
        probe = _probe_batch(processor, config)
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "local_files_only": True,
        }
        if bool(config["model"].get("load_in_4bit", False)):
            quantization_skip_modules = list(
                config["model"].get("quantization_skip_modules", [])
            )
            model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=quantization_skip_modules or None,
            )
        model, processor, r1_report = load_r1_foundation(
            modules=modules,
            base_model=config["model"]["base_model"],
            r1_checkpoint=config["model"]["r1_checkpoint"],
            visual_sidecar=config["model"]["visual_sidecar"],
            model_kwargs=model_kwargs,
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
            local_depth=int(expert.get("local_depth", 1)),
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

        audit_path = Path(config["provenance"]["architecture_audit"])
        r1_manifest = Path(config["model"]["r1_checkpoint"]) / "strategy_manifest.json"
        data_manifest = Path(config["data"]["manifest"])
        audit_sha = file_sha256(audit_path)
        r1_manifest_sha = file_sha256(r1_manifest)
        sidecar_sha = file_sha256(config["model"]["visual_sidecar"])
        configured_resume = dict(config.get("continuation", {})).get("source_checkpoint")
        resume_path = args.resume_expert_checkpoint or configured_resume
        if isinstance(resume_path, str) and "${" in resume_path:
            raise ValueError(
                "Continuation checkpoint environment variable is unresolved; "
                "pass --resume-expert-checkpoint explicitly"
            )
        resume = None
        resume_report: dict[str, Any] = {
            "source": None,
            "expert_weights_restored": False,
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "continuation_mode": "new expert initialized from R1",
        }
        continuation_base_parity = None
        if resume_path:
            resume = inspect_expert_checkpoint(resume_path)
            resume_report = validate_expert_checkpoint_compatibility(
                resume,
                expected_variant=expert["variant"],
                expected_expert_variant=expert["expert_variant"],
                expected_detail_hidden_size=int(expert.get("detail_hidden_size", 512)),
                expected_local_depth=int(expert.get("local_depth", 1)),
                expected_interface_lora=expert["interface_lora"],
                architecture=architecture,
                architecture_audit_sha256=audit_sha,
                source_r1_manifest_sha256=r1_manifest_sha,
                source_visual_sidecar_sha256=sidecar_sha,
                source_r1_checkpoint=config["model"]["r1_checkpoint"],
                r1_integration=r1_report["implementation"],
            )
            load_expert_checkpoint(controller, resume)
            controller.set_active_expert("base")
            continuation_base_probe = capture_parity_snapshot(model, probe)
            continuation_base_parity = compare_parity_snapshots(
                foundation_probe, continuation_base_probe
            )
            if not continuation_base_parity["passed"]:
                raise ValueError(
                    f"Loaded expert changed the frozen base route: {continuation_base_parity}"
                )
            restored_state = controller.expert_state_dict()
            resume_report["restored_tensor_count"] = len(restored_state)
            resume_report["exact_state_key_parity"] = True

        count_loss_config = dict(config["training"].get("count_loss", {}))
        if bool(count_loss_config.get("enabled", False)):
            controller.configure_count_head(
                max_count=int(count_loss_config.get("max_count", 15)),
                head_hidden_size=int(count_loss_config.get("head_hidden_size", 512)),
                distribution=str(count_loss_config.get("distribution", "categorical")),
            )
        trainable_audit = controller.freeze_base_and_enable_expert()

        dataset = Qwen3VLDataset(config["data"]["train_file"], max_samples=args.max_train_samples)
        if args.target_total_effective_epochs is not None:
            if resume is None:
                raise ValueError("--target-total-effective-epochs requires continuation")
            extra_epochs = args.target_total_effective_epochs - resume.completed_effective_epochs
            if extra_epochs <= 0:
                raise ValueError(
                    "Target total effective epochs must exceed the parent checkpoint progress"
                )
            config["training"]["target_effective_epochs"] = extra_epochs
            resume_report["target_total_effective_epochs"] = args.target_total_effective_epochs
            resume_report["resolved_extra_effective_epochs"] = extra_epochs
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
            "resume": resume_report,
            "foundation_base_route_parity": base_route_parity,
            "continuation_base_route_parity": continuation_base_parity,
            "step0_r1_parity_before_continuation_load": step0,
            "trainable_audit": trainable_audit,
            "training_plan": plan.__dict__,
        }
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        if args.dry_run:
            return 0
        if args.finalize_existing_run:
            if args.skip_eval:
                raise ValueError("--finalize-existing-run always performs fixed evaluation")
            if (
                args.resume_expert_checkpoint
                or args.forward_only
                or args.max_steps is not None
                or args.target_total_effective_epochs is not None
            ):
                raise ValueError(
                    "--finalize-existing-run cannot be combined with training/resume controls"
                )
            root = Path(args.finalize_existing_run).expanduser().resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"Existing run directory does not exist: {root}")
            manifest = _build_expert_manifest(
                config=config,
                expert=expert,
                architecture=architecture,
                r1_report=r1_report,
                audit_sha=audit_sha,
                r1_manifest_sha=r1_manifest_sha,
                sidecar_sha=sidecar_sha,
                data_manifest=data_manifest,
                trainable_audit=trainable_audit,
                resume_report=resume_report,
                transformers=transformers,
                torch=torch,
                peft=peft,
            )
            summary = finalize_existing_run(
                root=root,
                config=config,
                expert=expert,
                model=model,
                processor=processor,
                controller=controller,
                manifest=manifest,
                torch=torch,
                max_eval_samples=args.max_eval_samples,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        root = Path(args.output_root or config["output"]["root"]) / (
            f"{config['experiment']}_{timestamp}"
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
            resume_training_state_path=(resume.training_state_path if resume else None),
            resume_completed_effective_epochs=(
                resume.completed_effective_epochs if resume else 0.0
            ),
        )
        if args.forward_only:
            (root / "forward_only.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return 0
        manifest = _build_expert_manifest(
            config=config,
            expert=expert,
            architecture=architecture,
            r1_report=r1_report,
            audit_sha=audit_sha,
            r1_manifest_sha=r1_manifest_sha,
            sidecar_sha=sidecar_sha,
            data_manifest=data_manifest,
            trainable_audit=trainable_audit,
            resume_report=resume_report,
            transformers=transformers,
            torch=torch,
            peft=peft,
        )
        if config["data"].get("fixed_eval") and not args.skip_eval:
            summary["fixed_eval_learning_curve"] = _evaluate_fixed_curve(
                root=root,
                config=config,
                expert=expert,
                model=model,
                processor=processor,
                controller=controller,
                max_eval_samples=args.max_eval_samples,
            )
        save_composite_checkpoint(
            controller,
            root / "checkpoint",
            manifest=manifest,
            training_summary=summary,
            resolved_config=config,
            training_state_path=summary.get("training_state_path"),
        )
        return 0
    finally:
        cleanup: dict[str, Any] | None = None
        if torch is not None and torch.cuda.is_available():
            cleanup = {
                "before_cleanup": {
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "reserved_bytes": int(torch.cuda.memory_reserved()),
                }
            }
        if controller is not None:
            controller.close(restore_modules=True)
        restored_state = None
        foundation_probe = base_route_probe = continuation_base_probe = None
        summary = dataset = probe = controller = processor = model = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            assert cleanup is not None
            cleanup["after_cleanup"] = {
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
            }
            cleanup["reserved_memory_note"] = (
                "reserved_bytes is CUDA allocator cache and is not reported as a live leak"
            )
        if root is not None and cleanup is not None and root.is_dir():
            (root / "process_cleanup_memory.json").write_text(
                json.dumps(cleanup, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    raise SystemExit(main())
