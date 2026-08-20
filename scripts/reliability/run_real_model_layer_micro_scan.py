"""Run a small all-layer real-model injection scan and plot coverage."""

from __future__ import annotations

import argparse
import json
import sys
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
    from sat_rs_vlm.models.reliability.fault_injector import ParameterSelector, inject_model_parameter_bitflips

    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=True,
        device_map="cpu", dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    parameters = dict(model.named_parameters())
    rows = []
    for target, suffix in (("attention", "self_attn.q_proj.weight"), ("mlp", "mlp.down_proj.weight")):
        for layer in range(28):
            matches = [name for name in parameters if name.endswith(f"language_model.layers.{layer}.{suffix}")]
            if len(matches) != 1:
                raise RuntimeError(f"expected one {target} layer {layer}, got {matches}")
            name = matches[0]
            parameter = parameters[name]
            before = parameter.detach().reshape(-1)[0].clone()
            records = inject_model_parameter_bitflips(
                model, num_bits=1, seed=20260821 + layer,
                selector=ParameterSelector(parameter_names=(name,)),
                bit_index=7, flat_index=0,
            )
            after = parameter.detach().reshape(-1)[0].clone()
            changed = bool(not torch.equal(before, after))
            rows.append({
                "target": target, "layer": layer, "parameter": name,
                "dtype": str(parameter.dtype), "bit_plane": "exponent", "bit_index": 7,
                "flat_index": 0, "actual_bit_flips": len(records), "parameter_changed": changed,
                "before_value": float(before), "after_value": float(after),
            })
            parameter.data.reshape(-1)[0].copy_(before)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "1.0", "model": str(args.model), "conditions": len(rows), "rows": rows,
              "all_conditions_passed": all(row["actual_bit_flips"] == 1 and row["parameter_changed"] for row in rows)}
    (destination / "micro_scan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for target, color in (("attention", "#2563eb"), ("mlp", "#dc2626")):
        subset = [row for row in rows if row["target"] == target]
        ax.plot([row["layer"] for row in subset], [int(row["parameter_changed"]) for row in subset], "o-", label=target, color=color)
    ax.set_xlabel("Language-model layer")
    ax.set_ylabel("Real parameter changed (1=yes)")
    ax.set_title("Qwen3-VL real-model all-layer injection coverage")
    ax.set_ylim(-0.05, 1.05); ax.set_xticks(range(28)); ax.legend(); fig.tight_layout()
    fig.savefig(destination / "all_layer_injection_coverage.png", dpi=180); plt.close(fig)
    print(json.dumps({"output": str(destination), "conditions": len(rows), "all_conditions_passed": report["all_conditions_passed"]}, ensure_ascii=False, indent=2))
    return 0 if report["all_conditions_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
