"""Evaluate one composite RS merger checkpoint with canonical hard routing."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

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
    model, processor = load_qwen3vl(
        modules={"torch": torch, "transformers": transformers, "peft": peft},
        base_model=args.base_model,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "local_files_only": True,
        },
        adapter_path=args.r1_checkpoint,
    )
    load_visual_sidecar(model, args.visual_sidecar)
    integration = str(raw_manifest.get("r1_integration", ""))
    if integration == "merge_and_unload":
        merge = getattr(model, "merge_and_unload", None)
        if not callable(merge):
            raise TypeError("Checkpoint requires merge_and_unload but loaded R1 is not PEFT")
        model = merge(safe_merge=True)
    elif integration != "frozen_peft_additive":
        raise ValueError(f"Unsupported checkpoint R1 integration: {integration!r}")
    variant_map = {
        "c1_clone": "clone",
        "c2_rs_detail": "rs_detail",
        "c3_rs_detail_lora": "rs_detail",
    }
    variant = variant_map.get(str(raw_manifest.get("variant")))
    if variant is None:
        raise ValueError(f"Unsupported composite variant: {raw_manifest.get('variant')!r}")
    interface = int(raw_manifest.get("interface_lora_parameter_count", 0)) > 0
    architecture = source_architecture_audit(model)
    controller = RSMergerExpertController(
        model,
        variant=variant,
        interface_lora_enabled=interface,
        lora_rank=int(raw_manifest.get("interface_lora", {}).get("r", 16)),
        lora_alpha=float(raw_manifest.get("interface_lora", {}).get("alpha", 32)),
        lora_dropout=float(raw_manifest.get("interface_lora", {}).get("dropout", 0.05)),
    )
    manifest = load_expert_weights(controller, args.expert_checkpoint)
    validate_checkpoint_provenance(
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
    )
    experiment_map = {
        "c1_clone": "C1",
        "c2_rs_detail": "C2",
        "c3_rs_detail_lora": "C3",
    }
    experiment = "C0" if args.force_base else experiment_map[str(manifest["variant"])]
    architecture_names = {
        "C0": "formal R1 base route",
        "C1": "four cloned R1 mergers",
        "C2": "four independent RS detail branches",
        "C3": "C2 plus shallow q/k/v/o interface LoRA",
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
