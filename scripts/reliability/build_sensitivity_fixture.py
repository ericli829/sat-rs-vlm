"""Create deterministic, inference-free artifacts for sensitivity plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    conditions = []
    projections = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    planes = (("sign", 31, 0.72), ("exponent", 27, 0.91), ("mantissa", 8, 0.18))
    targets = (("attention", 0), ("attention", 14), ("mlp", 0), ("mlp", 14))
    for index, (target, layer) in enumerate(targets):
        for plane, bit_index, base_rate in planes:
            condition_id = f"fixture-{target}-l{layer}-{plane}"
            num_bits = 1
            conditions.append({
                "id": condition_id,
                "target": target,
                "layers": [layer],
                "bit_plane": plane,
                "num_bits": num_bits,
                "repeat": 0,
                "seed": 1000 + len(conditions),
            })
            projection = projections[(index + bit_index) % len(projections)]
            changed_rate = min(1.0, base_rate + layer * 0.002 + index * 0.01)
            directory = root / "conditions" / condition_id
            write_json(directory / "fault_injection_summary.json", {
                "schema_version": "2.0",
                "condition_id": condition_id,
                "planned_bit_flips": num_bits,
                "actual_bit_flips": num_bits,
                "records": [{
                    "target_name": f"model.layers.{layer}.{target}.{projection}.weight",
                    "flat_index": 3,
                    "bit_index": bit_index,
                    "before_hex": "00000000",
                    "after_hex": f"{1 << bit_index:08x}",
                }],
            })
            write_json(directory / "comparison" / "comparison.json", {
                "schema_version": "1.0",
                "overall": {"changed_rate": changed_rate, "invalid_rate": changed_rate / 10},
                "by_task": {
                    "caption": {"exact_match_drop": changed_rate * 0.7},
                    "vqa": {"exact_match_drop": changed_rate * 0.5},
                },
            })
    write_json(root / "condition_plan.json", {"schema_version": "2.0", "conditions": conditions})
    write_json(root / "protection" / "strategy_results.json", {"strategies": [
        {"strategy": "monitor", "recovery_rate": 0.15, "latency_overhead": 0.1},
        {"strategy": "selective_retry", "recovery_rate": 0.76, "latency_overhead": 1.8},
        {"strategy": "trusted_copy", "recovery_rate": 0.98, "latency_overhead": 4.6},
    ]})
    print(json.dumps({"output": str(root), "conditions": len(conditions)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
