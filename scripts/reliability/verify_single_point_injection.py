"""Run an auditable CPU-only exact single-bit injection check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    from sat_rs_vlm.models.reliability.fault_injector import (
        ParameterSelector,
        inject_model_parameter_bitflips,
    )

    class TinyAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.k_proj = torch.nn.Linear(4, 4, bias=False)

    model = TinyAttention()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    target_name = "q_proj.weight"
    flat_index = 5
    bit_index = 30
    records = inject_model_parameter_bitflips(
        model,
        num_bits=1,
        seed=20260821,
        selector=ParameterSelector(parameter_names=(target_name,)),
        bit_index=bit_index,
        flat_index=flat_index,
    )
    after = dict(model.named_parameters())
    changed_parameters = [
        name for name in before if not torch.equal(before[name], after[name].detach())
    ]
    before_bits = before[target_name].reshape(-1).view(torch.int32)
    after_bits = after[target_name].detach().reshape(-1).view(torch.int32)
    changed_elements = torch.nonzero(before_bits != after_bits).reshape(-1).tolist()
    xor_value = int((before_bits[flat_index] ^ after_bits[flat_index]).item()) & 0xFFFFFFFF
    passed = (
        changed_parameters == [target_name]
        and changed_elements == [flat_index]
        and xor_value == 1 << bit_index
        and len(records) == 1
    )
    payload = {
        "schema_version": "1.0",
        "passed": passed,
        "target_parameter": target_name,
        "flat_index": flat_index,
        "bit_index": bit_index,
        "changed_parameters": changed_parameters,
        "changed_elements": changed_elements,
        "xor_mask_hex": f"0x{xor_value:08x}",
        "unchanged_parameter": "k_proj.weight",
        "record": records[0].model_dump(mode="json") if records else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
