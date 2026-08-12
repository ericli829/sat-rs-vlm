"""比较同一评测集上的两个同契约v1.5/v1.6结果并生成配对置信区间。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.comparison import (  # noqa: E402
    ComparisonError,
    compare_evaluations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--protect-repository",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = compare_evaluations(
            args.baseline_dir,
            args.candidate_dir,
            args.output_dir,
            protected_repository=PROJECT_ROOT if args.protect_repository else None,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
    except (ComparisonError, OSError, ValueError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
