"""Generate a tiered protection recommendation from a sensitivity summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sat_rs_vlm.models.reliability.risk_policy import build_protection_policy
from sat_rs_vlm.models.reliability.sensitivity import aggregate_sensitivity_conditions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.sensitivity_summary.read_text(encoding="utf-8-sig"))
    if isinstance(payload.get("groups"), list):
        groups_payload = payload
    elif isinstance(payload.get("conditions"), list):
        groups_payload = aggregate_sensitivity_conditions(payload["conditions"])
    else:
        raise ValueError("sensitivity input must contain groups or raw conditions")
    groups = list(groups_payload["groups"])
    policy = build_protection_policy(groups)
    groups_output = args.output.with_name("sensitivity_groups.json")
    groups_output.write_text(
        json.dumps(groups_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "success": True,
                "groups": len(groups),
                "decisions": len(policy["decisions"]),
                "groups_output": str(groups_output),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
