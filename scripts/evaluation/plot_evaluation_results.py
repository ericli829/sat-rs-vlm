"""Generate unified v1.5 evaluation figures from result directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.config import (  # noqa: E402
    EvaluationWorkflowConfig,
    load_evaluation_config,
)
from sat_rs_vlm.evaluation.plotting import PlottingError, plot_evaluation_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Evaluation workflow YAML config.")
    parser.add_argument(
        "--evaluation",
        action="append",
        default=None,
        metavar="LABEL=DIR",
        help="Named v1.5 evaluation directory; repeat for multiple models/datasets.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Named paired-comparison directory; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        metavar="FORMAT",
        help="One or more output formats: png svg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow generated files and the manifest to be replaced in a non-empty directory.",
    )
    return parser.parse_args()


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_named_paths(values: list[str]) -> list[str]:
    """只解析 ``LABEL=DIR`` 的目录部分，保留标签供绘图图例使用。"""

    resolved: list[str] = []
    for value in values:
        if "=" not in value:
            raise PlottingError(f"named input must use LABEL=DIR: {value}")
        label, directory = value.split("=", 1)
        resolved.append(f"{label}={_project_path(directory)}")
    return resolved


def main() -> int:
    args = parse_args()
    try:
        config = (
            load_evaluation_config(args.config)
            if args.config is not None
            else EvaluationWorkflowConfig()
        )
        evaluations = args.evaluation or config.plotting.evaluations
        if not evaluations:
            raise PlottingError("at least one evaluation is required via CLI or plotting config")
        comparisons = args.comparison or config.plotting.comparisons
        figures_dir = (
            args.output_dir
            or config.output.figures_dir
            or str(Path(config.output.output_dir) / "figures")
        )
        outputs = plot_evaluation_results(
            _resolve_named_paths(evaluations),
            _resolve_named_paths(comparisons),
            _project_path(figures_dir),
            formats=args.formats or config.plotting.formats,
            overwrite=args.overwrite or config.plotting.overwrite,
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
