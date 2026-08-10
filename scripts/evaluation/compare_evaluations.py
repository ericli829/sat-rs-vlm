"""比较同一评测集上的两个 v1.5 结果目录并生成配对置信区间。"""

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
from sat_rs_vlm.evaluation.config import (  # noqa: E402
    EvaluationWorkflowConfig,
    load_evaluation_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Evaluation workflow YAML config.")
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--protect-repository",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        config = (
            load_evaluation_config(args.config)
            if args.config is not None
            else EvaluationWorkflowConfig()
        )
        baseline_value = args.baseline_dir or config.input.baseline_dir
        candidate_value = args.candidate_dir or config.input.candidate_dir
        if baseline_value is None or candidate_value is None:
            raise ComparisonError(
                "baseline and candidate directories are required via CLI or input config"
            )
        outputs = compare_evaluations(
            _project_path(baseline_value),
            _project_path(candidate_value),
            _project_path(args.output_dir or config.output.output_dir),
            protected_repository=PROJECT_ROOT if args.protect_repository else None,
            bootstrap_resamples=(args.bootstrap_resamples or config.comparison.bootstrap_resamples),
            seed=args.seed if args.seed is not None else config.comparison.seed,
        )
    except (ComparisonError, OSError, ValueError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
