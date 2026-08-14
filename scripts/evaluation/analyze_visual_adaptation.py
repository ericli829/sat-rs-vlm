"""Compare H1 small-object and visual-budget diagnostics from Evaluation v1.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sat_rs_vlm.evaluation.visual_analysis import compare_visual_adaptation
from sat_rs_vlm.training.config import BBoxAreaThresholdConfig
from sat_rs_vlm.training.hard_example_mining import load_rows, resolve_evaluated_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--small-max", type=float, default=0.01)
    parser.add_argument("--medium-max", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = BBoxAreaThresholdConfig(
        small_max=args.small_max,
        medium_max=args.medium_max,
    )
    report = compare_visual_adaptation(
        load_rows(resolve_evaluated_predictions(args.before)),
        load_rows(resolve_evaluated_predictions(args.after)),
        thresholds,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved visual adaptation analysis: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
