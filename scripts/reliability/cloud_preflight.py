"""One-command cloud preflight, model alignment, and scan-plan validation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/reliability/experiments/v15_sensitivity.yaml")
    parser.add_argument("--environment", choices=("local", "autodl"), default="autodl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from scripts.reliability.run_v15_sensitivity import build_conditions, load_config, preflight_report

    namespace = argparse.Namespace(config=args.config, environment=args.environment, output_root=None)
    config, _ = load_config(namespace)
    paths = preflight_report(config)
    model_path = Path(str(config["model"]["base_model"])).expanduser()
    selectors = {}
    keys: list[str] = []
    dtype_counts: dict[str, int] = {}
    try:
        from safetensors import safe_open
        for file_path in sorted(model_path.glob("*.safetensors")):
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    keys.append(name)
                    dtype = str(handle.get_slice(name).get_dtype())
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    except (ImportError, OSError) as exc:
        paths["model_alignment_error"] = str(exc)
    from sat_rs_vlm.models.reliability.fault_injector import selector_for_fault_target
    for target in ("language_model", "attention", "mlp", "vision_encoder", "embeddings", "lora_adapter"):
        selector = selector_for_fault_target(target)
        matches = [name for name in keys if selector.matches(name)]
        layers = sorted({int(value) for name in matches for value in re.findall(r"(?:layers?|blocks?)\.(\d+)", name)})
        selectors[target] = {"matched_tensors": len(matches), "layers": layers, "sample_names": matches[:5]}
    conditions = build_conditions(config)
    report = {
        "schema_version": "1.0", "config": str(args.config.resolve()),
        "path_and_cuda_preflight": paths, "model": str(model_path),
        "model_tensor_count": len(keys), "dtype_counts": dtype_counts,
        "selector_alignment": selectors, "condition_count": len(conditions),
        "checks": {
            "paths_and_cuda": bool(paths.get("success")) and bool(paths.get("cuda", {}).get("available")),
            "model_keys_read": bool(keys),
            "attention_or_mlp_matched": bool(selectors["attention"]["matched_tensors"] and selectors["mlp"]["matched_tensors"]),
            "condition_plan_valid": len(conditions) > 0,
        },
    }
    report["passed"] = all(report["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "condition_count": len(conditions), "checks": report["checks"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
