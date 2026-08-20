"""统一评估包装入口，复用已经跑通的 evaluate_rs_vlm.py。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """解析评估参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Required new empty directory for this exact model/checkpoint evaluation run.",
    )
    return parser.parse_args()


def main() -> int:
    """调用稳定评估脚本并透传退出码。"""

    args = parse_args()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_rs_vlm.py"),
        "--config",
        str(args.config),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.checkpoint:
        command.extend(["--checkpoint", str(args.checkpoint)])
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
