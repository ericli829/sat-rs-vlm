"""Load a real Qwen3-VL checkpoint and verify one exact CPU injection."""

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
    from sat_rs_vlm.models.reliability.fault_injector import (
        ParameterSelector,
        inject_model_parameter_bitflips,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=True,
        device_map="cpu", dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    parameters = dict(model.named_parameters())
    target_suffixes = (
        "language_model.layers.0.self_attn.q_proj.weight",
        "visual.blocks.0.attn.proj.weight",
        "visual.merger.linear_fc1.weight",
    )
    flat_index = 0
    bit_index = 7
    checks = []
    for suffix in target_suffixes:
        candidates = [name for name in parameters if name.endswith(suffix)]
        if len(candidates) != 1:
            raise RuntimeError(f"could not resolve exact target {suffix}: {candidates[:5]}")
        target = candidates[0]
        parameter = parameters[target]
        before = parameter.detach().clone()
        records = inject_model_parameter_bitflips(
            model, num_bits=1, seed=20260821,
            selector=ParameterSelector(parameter_names=(target,)),
            bit_index=bit_index, flat_index=flat_index,
        )
        after = parameter.detach()
        xor_mask = int((before.reshape(-1).view(torch.int16)[flat_index] ^ after.reshape(-1).view(torch.int16)[flat_index]).item()) & 0xFFFF
        with torch.no_grad():
            parameter.copy_(before)
        checks.append({
            "passed": xor_mask == (1 << bit_index) and torch.equal(parameter.detach(), before),
            "restored": torch.equal(parameter.detach(), before), "target": target,
            "dtype": str(before.dtype), "shape": list(before.shape), "flat_index": flat_index,
            "bit_index": bit_index, "xor_mask_hex": f"0x{xor_mask:04x}",
            "record": records[0].model_dump(mode="json"),
        })
    payload = {
        "schema_version": "2.0", "model": str(args.model),
        "passed": all(item["passed"] for item in checks),
        "parameter_count": len(parameters), "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
