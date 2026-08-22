"""Validate the strict RTX 5090 and real-4B two-step smoke artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-report", required=True)
    parser.add_argument("--real-smoke-report", required=True)
    parser.add_argument(
        "--output", default="reports/rs_merger_expert/counting_cloud_gate.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment = json.loads(Path(args.environment_report).read_text(encoding="utf-8"))
    smoke = json.loads(Path(args.real_smoke_report).read_text(encoding="utf-8"))
    blockers = []
    if environment.get("status") != "pass" or not environment.get("strict_5090"):
        blockers.append("strict RTX 5090 environment gate did not pass")
    variants = smoke.get("variants", [])
    names = {str(item.get("variant")) for item in variants}
    required_fragments = ("c2_count", "c3_count", "c4_wide_count")
    for fragment in required_fragments:
        if not any(fragment in name for name in names):
            blockers.append(f"missing real-4B smoke variant containing {fragment}")
    for item in variants:
        if int(item.get("optimizer_steps", -1)) != 2:
            blockers.append(f"{item.get('variant')} did not complete exactly two steps")
        if item.get("save_reload_generate_parser_metric") != "pass":
            blockers.append(f"{item.get('variant')} save/reload/generate/metric gate failed")
    report = {
        "schema_version": "1.0",
        "environment_report": str(Path(args.environment_report).resolve()),
        "real_smoke_report": str(Path(args.real_smoke_report).resolve()),
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "formal_training_authorized": not blockers,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
