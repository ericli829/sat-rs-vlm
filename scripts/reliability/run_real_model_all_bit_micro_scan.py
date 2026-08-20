"""Exhaustively verify every real model tensor's bit positions on CPU."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText
    from sat_rs_vlm.models.reliability.fault_injector import ParameterSelector, inject_model_parameter_bitflips, tensor_bit_width

    model = AutoModelForImageTextToText.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True, device_map="cpu", dtype=torch.bfloat16, low_cpu_mem_usage=True)
    parameters = dict(model.named_parameters())
    rows = []
    for name, parameter in parameters.items():
        try:
            width = tensor_bit_width(parameter)
        except (TypeError, ValueError):
            continue
        if parameter.numel() == 0:
            continue
        category = "vision_encoder" if "visual" in name else "embeddings" if "embed_tokens" in name or "lm_head" in name else "attention" if "self_attn" in name else "mlp" if ".mlp." in name else "language_model"
        layer_match = re.search(r"(?:layers?|blocks?)\.(\d+)", name)
        layer = int(layer_match.group(1)) if layer_match else None
        parameter_flat = parameter.detach().reshape(-1)
        before = parameter_flat[0].clone()
        before_bits = before.view(torch.int16 if parameter.dtype in (torch.float16, torch.bfloat16) else torch.int32)
        for bit_index in range(width):
            records = inject_model_parameter_bitflips(model, num_bits=1, seed=20260821 + bit_index, selector=ParameterSelector(parameter_names=(name,)), bit_index=bit_index, flat_index=0)
            after = parameter_flat[0].clone()
            after_bits = after.view(before_bits.dtype)
            xor_mask = int((before_bits ^ after_bits).item()) & ((1 << width) - 1)
            rows.append({"parameter": name, "category": category, "layer": layer, "dtype": str(parameter.dtype), "bit_index": bit_index, "expected_mask": 1 << bit_index, "actual_mask": xor_mask, "passed": len(records) == 1 and xor_mask == (1 << bit_index)})
            with torch.no_grad():
                parameter_flat[0].copy_(before)
    destination = args.output.resolve(); destination.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "1.0", "model": str(args.model), "parameters": len(parameters), "conditions": len(rows), "rows": rows, "all_conditions_passed": all(row["passed"] for row in rows)}
    (destination / "all_bit_micro_scan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = Counter((row["category"], row["bit_index"]) for row in rows if row["passed"])
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    categories = sorted({row["category"] for row in rows}); bits = sorted({row["bit_index"] for row in rows})
    matrix = [[sum(summary[(category, bit)] for _ in [0]) for bit in bits] for category in categories]
    fig, ax = plt.subplots(figsize=(11, max(4, len(categories) * 0.6)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis"); ax.set_xticks(range(len(bits)), bits); ax.set_yticks(range(len(categories)), categories); ax.set_xlabel("Exact bit index"); ax.set_ylabel("Model region"); ax.set_title("Real Qwen3-VL exhaustive bit-injection coverage"); fig.colorbar(image, ax=ax, label="Successful injections"); fig.tight_layout(); fig.savefig(destination / "all_bit_injection_coverage.png", dpi=180); plt.close(fig)
    print(json.dumps({"output": str(destination), "parameters": len(parameters), "conditions": len(rows), "all_conditions_passed": report["all_conditions_passed"]}, ensure_ascii=False, indent=2))
    return 0 if report["all_conditions_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
