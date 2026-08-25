"""Run one clean or in-memory SEU-faulted Evaluation v1.5 inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fault-target")
    parser.add_argument("--fault-num-bits", type=int)
    parser.add_argument("--fault-seed", type=int)
    parser.add_argument(
        "--fault-bit-plane", choices=("all", "sign", "exponent", "mantissa"), default="all"
    )
    parser.add_argument("--fault-layers", nargs="*", type=int, default=[])
    parser.add_argument("--fault-parameter-name")
    parser.add_argument("--fault-bit-index", type=int)
    parser.add_argument("--fault-flat-index", type=int)
    parser.add_argument("--activation-guard", action="store_true")
    parser.add_argument("--activation-guard-mode", choices=("research", "deployment"), default="research")
    parser.add_argument("--activation-patterns", nargs="+", default=["self_attn", "mlp"])
    parser.add_argument("--activation-max-abs", type=float, default=10000.0)
    return parser.parse_args()


def main() -> int:
    from scripts.evaluate_rs_vlm import evaluate, load_model, load_yaml

    from sat_rs_vlm.models.reliability.activation_guard import ActivationGuard
    from sat_rs_vlm.models.reliability.fault_injector import (
        inject_model_parameter_bitflips,
        model_fault_inventory,
        selector_for_fault_target,
    )
    from sat_rs_vlm.training.experiment import write_json
    from sat_rs_vlm.training.utils import safe_import_model_dependencies

    args = parse_args()
    is_fault = args.fault_target is not None
    if is_fault and (args.fault_num_bits is None or args.fault_seed is None):
        raise SystemExit("fault target requires --fault-num-bits and --fault-seed")
    config = load_yaml(args.config)
    modules = safe_import_model_dependencies(require_bitsandbytes=False)
    model, processor = load_model(config, modules)
    records: list[Any] = []
    inventory: dict[str, Any] | None = None
    if is_fault:
        selector = selector_for_fault_target(args.fault_target, layer_indices=tuple(args.fault_layers))
        if args.fault_parameter_name:
            selector = selector.__class__(
                name_contains=selector.name_contains,
                name_regex=selector.name_regex,
                module_names=selector.module_names,
                layer_indices=selector.layer_indices,
                parameter_names=(args.fault_parameter_name,),
                lora_scope=selector.lora_scope,
            )
        inventory = model_fault_inventory(model, selector=selector, bit_plane=args.fault_bit_plane)
        if args.fault_bit_index is not None:
            allowed = {
                "all": set(range(64)),
                "sign": {15, 31},
                "exponent": set(range(7, 15)) | set(range(23, 31)),
                "mantissa": set(range(0, 7)) | set(range(0, 23)),
            }[args.fault_bit_plane]
            if args.fault_bit_index not in allowed:
                raise SystemExit(
                    f"fault bit index {args.fault_bit_index} is incompatible with "
                    f"bit plane {args.fault_bit_plane}"
                )
        records = inject_model_parameter_bitflips(
            model,
            num_bits=args.fault_num_bits,
            seed=args.fault_seed,
            selector=selector,
            bit_plane=args.fault_bit_plane,
            bit_index=args.fault_bit_index,
            flat_index=args.fault_flat_index,
        )
    guard = None
    guard_report = None
    if args.activation_guard:
        guard = ActivationGuard(
            model, module_patterns=list(args.activation_patterns), max_abs=args.activation_max_abs,
            mode=args.activation_guard_mode,
        )
        guard.install()
    try:
        result = evaluate(
            args.config,
            output_dir=args.output_dir,
            loaded_model=model,
            loaded_processor=processor,
            loaded_modules=modules,
            config_override=config,
        )
    finally:
        if guard is not None:
            guard.close()
            guard_report = guard.report()
            write_json(args.output_dir / "activation_guard_report.json", guard_report)
    guard_triggered = bool(guard_report and guard_report.get("anomalies"))
    write_json(
        args.output_dir / "fault_injection_summary.json",
        {
            "schema_version": "2.0",
            "condition_id": args.output_dir.name,
            "mode": "fault" if is_fault else "baseline",
            "target": args.fault_target,
            "layers": args.fault_layers,
            "bit_plane": args.fault_bit_plane if is_fault else None,
            "parameter_name": args.fault_parameter_name,
            "bit_index": args.fault_bit_index,
            "flat_index": args.fault_flat_index,
            "requested_bit_flips": args.fault_num_bits if is_fault else 0,
            "actual_bit_flips": len(records),
            "planned_bit_flips": args.fault_num_bits if is_fault else 0,
            "execution_status": "completed_guarded" if guard_triggered else "completed",
            "guard_triggered": guard_triggered,
            "seed": args.fault_seed,
            "candidate_bits": inventory["candidate_bits"] if inventory else 0,
            "inventory": inventory,
            "records": [record.model_dump(mode="json") for record in records],
            "evaluation": result,
            "activation_guard": guard_report,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
