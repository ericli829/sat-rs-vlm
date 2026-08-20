"""Build a CPU-readable parameter/bit inventory from a model or Adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, help="Adapter directory or safetensors file")
    parser.add_argument("--model-config", type=Path, help="Resolved evaluation YAML for CPU model inventory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bit-planes", nargs="+", default=["sign", "exponent", "mantissa"])
    args = parser.parse_args()
    if not args.adapter and not args.model_config:
        raise SystemExit("provide --adapter or --model-config")
    from sat_rs_vlm.models.reliability.fault_injector import (
        load_safetensors_state,
        model_fault_inventory,
        selectable_parameters,
        selector_for_fault_target,
        summarize_fault_inventory,
    )

    if args.model_config:
        import yaml
        from scripts.evaluate_rs_vlm import load_model, load_yaml
        from sat_rs_vlm.training.utils import safe_import_model_dependencies
        config = load_yaml(args.model_config)
        config.setdefault("model", {})["device_map"] = "cpu"
        modules = safe_import_model_dependencies(require_bitsandbytes=False)
        model, _ = load_model(config, modules)
        state_model = model
        source = args.model_config
    else:
        source = args.adapter / "adapter_model.safetensors" if args.adapter.is_dir() else args.adapter
        state, _ = load_safetensors_state(source)
        state_model = type("StateModel", (), {"named_parameters": lambda self: iter(state.items())})()
    rows = []
    for plane in args.bit_planes:
        inventory = model_fault_inventory(state_model, selector=None if args.model_config else selector_for_fault_target("lora_adapter"), bit_plane=plane)
        rows.append({"bit_plane": plane, "inventory": inventory, "groups": summarize_fault_inventory(inventory)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": "1.0", "source": str(source), "planes": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "planes": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
