"""从标准可靠性 metrics 生成静态图表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sat_rs_vlm.evaluation.reliability.plotting import plot_reliability_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated = plot_reliability_results(args.input, args.output)
    print(json.dumps({"generated": [str(path) for path in generated]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
