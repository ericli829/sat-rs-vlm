"""Estimate H1 max-step candidates from a completed Stage-B training budget."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-stage-steps", type=int, required=True)
    parser.add_argument(
        "--output",
        default="reports/training_statistics/h1_step_recommendations.json",
    )
    return parser.parse_args()


def estimate(reference_steps: int) -> dict[str, int]:
    if reference_steps <= 0:
        raise ValueError("reference-stage-steps must be positive")
    return {
        f"{percentage}%": max(1, math.ceil(reference_steps * percentage / 100))
        for percentage in (10, 15, 20, 25)
    }


def main() -> int:
    args = parse_args()
    payload = {
        "reference_stage_steps": args.reference_stage_steps,
        "candidate_max_steps": estimate(args.reference_stage_steps),
        "selection_policy": (
            "Choose in configuration after reviewing hard-data size, truncation, and "
            "resource reports; no candidate is selected automatically."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
