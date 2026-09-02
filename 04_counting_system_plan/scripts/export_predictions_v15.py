#!/usr/bin/env python3
"""将 COUNT benchmark 产物转为 sat-rs-vlm Evaluation v1.5 predictions JSONL。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from counting_system.eval.export_v15 import load_benchmark_rows, write_predictions_v15


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROOT / "outputs" / "xlrs_benchmark" / "predictions.jsonl"),
        help="run_benchmark 输出的 predictions.jsonl 或 predictions.json",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "xlrs_benchmark" / "predictions_v15.jsonl"),
    )
    parser.add_argument("--dataset", default="XLRS-Bench-lite")
    parser.add_argument("--language", default="en")
    parser.add_argument("--official-protocol", action="store_true")
    args = parser.parse_args()

    rows = load_benchmark_rows(args.input)
    if not rows:
        print(f"no rows in {args.input}", file=sys.stderr)
        return 2
    out = write_predictions_v15(
        args.output,
        rows,
        dataset=args.dataset,
        language=args.language,
        official_protocol=bool(args.official_protocol),
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
