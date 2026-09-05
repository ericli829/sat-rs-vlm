"""Evaluate one composite RS merger checkpoint with canonical hard routing."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import yaml

from sat_rs_vlm.evaluation.rs_merger_expert import (
    evaluate_rows,
    load_expert_weights,
    update_experiment_matrix,
)
from sat_rs_vlm.models.qwen3vl_loader import load_qwen3vl
from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.models.rs_merger_expert import RSMergerExpertController, source_architecture_audit
from sat_rs_vlm.training.rs_merger_expert import validate_checkpoint_provenance
from sat_rs_vlm.training.vision_tuning import load_visual_sidecar


def resolve_controller_spec(manifest: dict) -> dict[str, object]:
    variant_map = {
        "c1_clone": "clone",
        "c2_rs_detail": "rs_detail",
        "c3_rs_detail_lora": "rs_detail",
        "c4_wide": "rs_detail",
    }
    variant_name = str(manifest.get("variant"))
    variant = variant_map.get(variant_name)
    if variant is None:
        raise ValueError(f"Unsupported composite variant: {variant_name!r}")
    return {
        "variant": variant,
        "detail_hidden_size": int(manifest.get("detail_hidden_size", 512)),
        "interface_lora_enabled": int(manifest.get("interface_lora_parameter_count", 0)) > 0,
    }


def resolve_real4bit_skip_modules(checkpoint: str | Path, manifest: dict) -> list[str] | None:
    if manifest.get("foundation_precision") != "real_4bit_nf4_bf16_compute":
        return None
    config_path = Path(checkpoint) / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Real-4bit checkpoint is missing config_resolved.yaml: {checkpoint}"
        )
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = dict(resolved.get("model", {}))
    if not bool(model_config.get("load_in_4bit", False)):
        raise ValueError("Real-4bit manifest disagrees with resolved checkpoint configuration")
    skip_modules = model_config.get("quantization_skip_modules")
    if not isinstance(skip_modules, list) or not all(
        isinstance(item, str) and item for item in skip_modules
    ):
        raise ValueError("Real-4bit checkpoint must record quantization_skip_modules")
    return list(skip_modules)


def configure_decoder_only_generation_padding(processor: object) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "padding_side"):
        raise TypeError("Evaluation processor must expose tokenizer.padding_side")
    tokenizer.padding_side = "left"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--r1-checkpoint", required=True)
    parser.add_argument("--visual-sidecar", required=True)
    parser.add_argument("--expert-checkpoint", required=True)
    parser.add_argument("--architecture-audit", required=True)
    parser.add_argument("--tier-file", required=True)
    parser.add_argument("--tier-manifest")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-metrics")
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--verify-batch1-parity", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--force-base", action="store_true")
    parser.add_argument(
        "--experiment-matrix",
        default="reports/rs_merger_expert/experiment_matrix.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_manifest = json.loads(
        (Path(args.expert_checkpoint) / "expert_manifest.json").read_text(encoding="utf-8")
    )
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "local_files_only": True,
    }
    real4bit_skip_modules = resolve_real4bit_skip_modules(args.expert_checkpoint, raw_manifest)
    if real4bit_skip_modules is not None:
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=real4bit_skip_modules,
        )
    model, processor = load_qwen3vl(
        modules={"torch": torch, "transformers": transformers, "peft": peft},
        base_model=args.base_model,
        model_kwargs=model_kwargs,
        adapter_path=args.r1_checkpoint,
    )
    configure_decoder_only_generation_padding(processor)
    load_visual_sidecar(model, args.visual_sidecar)
    integration = str(raw_manifest.get("r1_integration", ""))
    if integration == "merge_and_unload":
        merge = getattr(model, "merge_and_unload", None)
        if not callable(merge):
            raise TypeError("Checkpoint requires merge_and_unload but loaded R1 is not PEFT")
        model = merge(safe_merge=True)
    elif integration != "frozen_peft_additive":
        raise ValueError(f"Unsupported checkpoint R1 integration: {integration!r}")
    controller_spec = resolve_controller_spec(raw_manifest)
    architecture = source_architecture_audit(model)
    controller = RSMergerExpertController(
        model,
        variant=str(controller_spec["variant"]),
        detail_hidden_size=int(controller_spec["detail_hidden_size"]),
        local_depth=int(raw_manifest.get("local_depth", 1)),
        interface_lora_enabled=bool(controller_spec["interface_lora_enabled"]),
        lora_rank=int(raw_manifest.get("interface_lora", {}).get("r", 16)),
        lora_alpha=float(raw_manifest.get("interface_lora", {}).get("alpha", 32)),
        lora_dropout=float(raw_manifest.get("interface_lora", {}).get("dropout", 0.05)),
    )
    count_loss = raw_manifest.get("count_loss", {})
    if isinstance(count_loss, dict) and bool(count_loss.get("enabled", False)):
        controller.configure_count_head(
            max_count=int(count_loss.get("max_count", 15)),
            head_hidden_size=int(count_loss.get("head_hidden_size", 512)),
            distribution=str(count_loss.get("distribution", "categorical")),
        )
    manifest = load_expert_weights(controller, args.expert_checkpoint)
    provenance_report = validate_checkpoint_provenance(
        manifest,
        architecture_audit_sha256=file_sha256(args.architecture_audit),
        source_r1_manifest_sha256=file_sha256(Path(args.r1_checkpoint) / "strategy_manifest.json"),
        source_visual_sidecar_sha256=file_sha256(args.visual_sidecar),
    )
    expected = {
        "spatial_merge_size": architecture["spatial_merge_size"],
        "visual_hidden_size": architecture["vision_hidden_size"],
        "llm_hidden_size": architecture["llm_hidden_size"],
        "output_visual_token_policy": "same_as_qwen",
    }
    mismatch = [
        f"{key}: checkpoint={manifest.get(key)!r}, runtime={value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatch:
        raise ValueError("Composite architecture mismatch: " + "; ".join(mismatch))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = Path(args.output_root) / f"{Path(args.tier_file).stem}_{timestamp}"
    metrics = evaluate_rows(
        model=model,
        processor=processor,
        controller=controller,
        tier_file=args.tier_file,
        tier_manifest=args.tier_manifest,
        image_root=args.image_root,
        output_dir=output,
        expert_variant=str(manifest["variant"]),
        max_eval_samples=args.max_eval_samples,
        max_new_tokens=args.max_new_tokens,
        baseline_metrics=args.baseline_metrics,
        force_base=args.force_base,
        eval_batch_size=args.eval_batch_size,
    )
    metrics["checkpoint_provenance"] = provenance_report
    if args.verify_batch1_parity:
        if args.eval_batch_size == 1:
            raise ValueError("--verify-batch1-parity requires --eval-batch-size > 1")
        batch1_output = Path(str(output) + "_batch1")
        batch1_metrics = evaluate_rows(
            model=model,
            processor=processor,
            controller=controller,
            tier_file=args.tier_file,
            tier_manifest=args.tier_manifest,
            image_root=args.image_root,
            output_dir=batch1_output,
            expert_variant=str(manifest["variant"]),
            max_eval_samples=args.max_eval_samples,
            max_new_tokens=args.max_new_tokens,
            baseline_metrics=args.baseline_metrics,
            force_base=args.force_base,
            eval_batch_size=1,
        )
        batch1_metrics["checkpoint_provenance"] = provenance_report
        batched_predictions = (output / "predictions.jsonl").read_text(encoding="utf-8")
        batch1_predictions = (batch1_output / "predictions.jsonl").read_text(encoding="utf-8")
        parity = {
            "schema_version": "1.0",
            "eval_batch_size": args.eval_batch_size,
            "prediction_file_exact": batched_predictions == batch1_predictions,
            "metrics_exact": metrics == batch1_metrics,
        }
        parity["passed"] = bool(parity["prediction_file_exact"] and parity["metrics_exact"])
        (output / "batch1_parity.json").write_text(
            json.dumps(parity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not parity["passed"]:
            raise AssertionError(f"Batched evaluation differs from batch=1: {parity}")
    experiment_map = {
        "c1_clone": "C1",
        "c2_rs_detail": "C2",
        "c3_rs_detail_lora": "C3",
        "c4_wide": "C4-Wide",
    }
    experiment = "C0" if args.force_base else experiment_map[str(manifest["variant"])]
    architecture_names = {
        "C0": "formal R1 base route",
        "C1": "four cloned R1 mergers",
        "C2": "four independent RS detail branches",
        "C3": "C2 plus shallow q/k/v/o interface LoRA",
        "C4-Wide": "four RS detail branches with hidden size 1024",
    }
    summary_path = Path(args.expert_checkpoint) / "training_summary.json"
    training_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file() and not args.force_base
        else {"trainable_params": 0}
    )
    update_experiment_matrix(
        args.experiment_matrix,
        experiment=experiment,
        architecture=architecture_names[experiment],
        training_summary=training_summary,
        metrics=metrics,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
