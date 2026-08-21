"""Run dependency-free LEVIR Caption rules and queue ambiguous rows locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.change_judge import run_server_rule_router  # noqa: E402
from sat_rs_vlm.evaluation.records import EvaluationError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_server_rule_router(
            args.predictions,
            args.output_dir,
            strict=args.strict,
        )
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"Server rule routing failed: {exc}", file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
