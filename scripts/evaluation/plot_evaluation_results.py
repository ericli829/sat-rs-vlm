"""Generate unified v1.5/v1.6 evaluation figures from result directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.plotting import PlottingError, plot_evaluation_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="Named v1.5/v1.6 evaluation directory; repeat for multiple models/datasets.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Named paired-comparison directory; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "svg"),
        metavar="FORMAT",
        help="One or more output formats: png svg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow generated files and the manifest to be replaced in a non-empty directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = plot_evaluation_results(
            args.evaluation,
            args.comparison,
            args.output_dir,
            formats=args.formats,
            overwrite=args.overwrite,
        )
    except (PlottingError, OSError, ValueError) as exc:
        print(f"Plotting failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(outputs["output_dir"]),
                "manifest": str(outputs["manifest"]),
                "generated": outputs["generated"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
