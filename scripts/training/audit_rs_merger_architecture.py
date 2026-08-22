"""Runtime architecture audit for the exact Qwen3-VL/R1 foundation."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from sat_rs_vlm.models.qwen3vl_loader import load_qwen3vl
from sat_rs_vlm.models.rs_merger_expert import (
    source_architecture_audit,
    validate_expected_qwen4b_contract,
)
from sat_rs_vlm.training.vision_tuning import load_visual_sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--r1-checkpoint", required=True)
    parser.add_argument("--visual-sidecar", required=True)
    parser.add_argument("--output-dir", default="reports/rs_merger_expert")
    parser.add_argument("--allow-architecture-mismatch", action="store_true")
    return parser.parse_args()


def _markdown(audit: dict[str, object]) -> str:
    lines = [
        "# Qwen3-VL R1 source architecture audit",
        "",
        f"- status: {audit.get('status')}",
        f"- Transformers: {audit.get('transformers_version')}",
        f"- PyTorch: {audit.get('torch_version')}",
        f"- PEFT: {audit.get('peft_version')}",
        f"- vision blocks: {audit.get('vision_block_count')}",
        f"- vision hidden: {audit.get('vision_hidden_size')}",
        f"- LLM hidden: {audit.get('llm_hidden_size')}",
        f"- spatial merge: {audit.get('spatial_merge_size')}",
        f"- DeepStack taps: {audit.get('deepstack_visual_indexes')}",
        "",
        "## DeepStack first-consumption order",
        "",
    ]
    for row in audit.get("deepstack_injection_order", []):
        lines.append(f"- {row}")
    lines.extend(["", "## Merger modules", ""])
    for merger in audit.get("mergers", []):
        lines.append(f"- {merger}")
    lines.extend(["", "## LLM layers 0-3 q/k/v/o", "", "```json"])
    lines.append(
        json.dumps(audit.get("language_layer_attention_paths"), ensure_ascii=False, indent=2)
    )
    lines.extend(["```", ""])
    if audit.get("blockers"):
        lines.extend(["## Blockers", "", *[f"- {item}" for item in audit["blockers"]], ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    modules = {"torch": torch, "transformers": transformers, "peft": peft}
    model, _processor = load_qwen3vl(
        modules=modules,
        base_model=args.base_model,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "local_files_only": True,
        },
        adapter_path=args.r1_checkpoint,
    )
    sidecar_names = load_visual_sidecar(model, args.visual_sidecar)
    audit = source_architecture_audit(model)
    audit.update(
        {
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "peft_version": peft.__version__,
            "r1_checkpoint": str(Path(args.r1_checkpoint).resolve()),
            "visual_sidecar": str(Path(args.visual_sidecar).resolve()),
            "visual_sidecar_loaded_parameter_count": len(sidecar_names),
            "blockers": [],
        }
    )
    try:
        validate_expected_qwen4b_contract(audit)
        audit["status"] = "compatible"
    except ValueError as exc:
        audit["status"] = "blocked_architecture_mismatch"
        audit["blockers"] = [str(exc)]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_architecture_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "source_architecture_audit.md").write_text(_markdown(audit), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["blockers"] and not args.allow_architecture_mismatch:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
