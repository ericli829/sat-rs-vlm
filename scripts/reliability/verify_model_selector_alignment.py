"""Check fault selectors against real safetensors keys without loading weights."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def bit_positions(dtype: str, plane: str) -> list[int]:
    width = {"F16": 16, "BF16": 16, "F32": 32, "I8": 8, "U8": 8}[dtype]
    if plane == "all":
        return list(range(width))
    if dtype in {"F16", "BF16"}:
        return {"sign": [15], "exponent": list(range(7 if dtype == "BF16" else 10, 15)), "mantissa": list(range(7 if dtype == "BF16" else 10))}[plane]
    if dtype == "F32":
        return {"sign": [31], "exponent": list(range(23, 31)), "mantissa": list(range(23))}[plane]
    return [7] if plane == "sign" else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from safetensors import safe_open
    from sat_rs_vlm.models.reliability.fault_injector import selector_for_fault_target

    files = sorted(args.model.glob("*.safetensors")) if args.model.is_dir() else [args.model]
    rows = []
    for model_file in files:
        with safe_open(str(model_file), framework="pt", device="cpu") as handle:
            rows.extend({"name": name, "shape": handle.get_slice(name).get_shape(), "dtype": str(handle.get_slice(name).get_dtype())} for name in handle.keys())
    selectors = ("language_model", "attention", "mlp", "vision_encoder", "visual_blocks", "visual_merger", "embeddings")
    report = {"schema_version": "1.0", "model": str(args.model), "total_tensors": len(rows), "targets": {}}
    for target in selectors:
        selector = selector_for_fault_target(target)
        matched = [row for row in rows if selector.matches(row["name"])]
        layer_counts = Counter()
        projections = Counter()
        for row in matched:
            found = re.search(r"(?:layers?|blocks?)\.(\d+)", row["name"])
            if found:
                layer_counts[int(found.group(1))] += 1
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                if projection in row["name"]:
                    projections[projection] += 1
        dtype = Counter(row["dtype"] for row in matched)
        report["targets"][target] = {
            "matched_tensors": len(matched),
            "dtype_counts": dict(dtype),
            "layers": sorted(layer_counts),
            "layer_counts": {str(k): v for k, v in sorted(layer_counts.items())},
            "projection_counts": dict(sorted(projections.items())),
            "bit_positions": {plane: bit_positions(next(iter(dtype), "BF16"), plane) for plane in ("sign", "exponent", "mantissa")},
            "sample_names": [row["name"] for row in matched[:8]],
        }
    language_layers = report["targets"]["language_model"]["layers"]
    visual_layers = report["targets"]["visual_blocks"]["layers"]
    report["discovered"] = {"language_layers": language_layers, "visual_layers": visual_layers}
    report["checks"] = {
        "language_layers_contiguous": bool(language_layers) and language_layers == list(range(max(language_layers) + 1)),
        "attention_matches_language_layers": report["targets"]["attention"]["layers"] == language_layers,
        "mlp_matches_language_layers": report["targets"]["mlp"]["layers"] == language_layers,
        "vision_matched": report["targets"]["vision_encoder"]["matched_tensors"] > 0,
        "visual_blocks_contiguous": bool(visual_layers) and visual_layers == list(range(max(visual_layers) + 1)),
        "visual_merger_matched": report["targets"]["visual_merger"]["matched_tensors"] > 0,
        "embeddings_matched": report["targets"]["embeddings"]["matched_tensors"] > 0,
    }
    report["passed"] = all(report["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "checks": report["checks"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
