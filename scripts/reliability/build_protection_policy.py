"""Generate a tiered protection recommendation from a sensitivity summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sat_rs_vlm.models.reliability.risk_policy import build_protection_policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.sensitivity_summary.read_text(encoding="utf-8-sig"))
    policy = build_protection_policy(list(payload.get("groups", [])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"success": True, "decisions": len(policy["decisions"]), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
