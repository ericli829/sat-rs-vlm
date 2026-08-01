"""运行不加载真实模型的本地训练控制流 smoke。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """解析本地 smoke 参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/local/train_lora_smoke.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    """调用统一训练入口并强制 Mock 模式。"""

    args = parse_args()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/training/run_train.py"),
        "--config",
        str(args.config),
        "--environment",
        "local",
        "--mock",
    ]
    if args.output_dir:
        command.extend(["--output-dir", str(args.output_dir)])
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
