"""Audit complete-system performance artifacts without loading model weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.performance_audit import audit_taskgraph_performance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--submission",
        action="store_true",
        help="Treat missing real-GPU and reproducibility fields as blockers.",
    )
    parser.add_argument(
        "--require-official",
        action="store_true",
        help="Require full-split sample provenance and a certified input manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/performance_audit.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_taskgraph_performance(
        args.run_dir,
        submission=args.submission,
        require_official=args.require_official,
    )
    output = args.output or args.run_dir / "performance_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
