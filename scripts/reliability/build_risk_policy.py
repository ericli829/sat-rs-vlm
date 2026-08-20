"""Build an auditable low/medium/high/critical policy from a sensitivity run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _metric(payload: dict, *names: str) -> float | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    for key in ("overall", "metrics"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            found = _metric(nested, *names)
            if found is not None:
                return found
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from sat_rs_vlm.models.reliability.risk_policy import build_protection_policy

    root = args.sensitivity_root.resolve()
    plan = json.loads((root / "condition_plan.json").read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    completed = 0
    for condition in plan.get("conditions", []):
        directory = root / "conditions" / condition["id"]
        comparison_path = directory / "comparison" / "comparison.json"
        if not comparison_path.is_file():
            continue
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        grouped[(str(condition["target"]), str(condition["bit_plane"]), str(condition.get("layers", [])))].append({
            "changed": _metric(comparison, "prediction_changed_rate", "changed_rate") or 0.0,
            "invalid": _metric(comparison, "invalid_rate") or 0.0,
            "drop": _metric(comparison, "exact_match_drop") or 0.0,
        })
        completed += 1
    groups = []
    for (target, bit_plane, layer_text), values in grouped.items():
        layers = json.loads(layer_text.replace("'", '"')) if layer_text != "[]" else []
        groups.append({
            "target": target, "layers": layers, "bit_plane": bit_plane,
            "changed_rate_mean": statistics.fmean(item["changed"] for item in values),
            "invalid_rate_mean": statistics.fmean(item["invalid"] for item in values),
            "exact_match_drop_mean": statistics.fmean(item["drop"] for item in values),
            "sample_count": len(values),
        })
    policy = build_protection_policy(groups)
    report = {**policy, "source": str(root), "completed_conditions": completed, "groups": len(groups)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "completed_conditions": completed, "groups": len(groups)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
