"""生成 4B ViT probe 的 baseline/checkpoint-100/checkpoint-200 配对报告。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.vit_probe_comparison import (  # noqa: E402
    compare_vit_probe_evaluations,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--checkpoint100-dir", type=Path, required=True)
    parser.add_argument("--checkpoint200-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    outputs = compare_vit_probe_evaluations(
        args.baseline_dir,
        args.checkpoint100_dir,
        args.checkpoint200_dir,
        args.output_dir,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

