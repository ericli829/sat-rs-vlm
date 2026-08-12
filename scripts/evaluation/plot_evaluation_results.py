"""从v1.5/v1.6评测结果目录生成统一中文科研图。"""

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
        help="命名的v1.5/v1.6评测目录；可重复传入多个模型或数据集。",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="命名的逐样本配对比较目录；可重复传入。",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "svg"),
        metavar="FORMAT",
        help="一种或多种输出格式：png svg。两种格式使用相同的中文图表内容。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许在非空目录中替换同名生成图和清单文件。",
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
