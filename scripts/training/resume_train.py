"""从显式或最新 checkpoint 恢复统一 LoRA 训练。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """解析恢复训练参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", choices=("local", "autodl"), default="local")
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--resume-from-checkpoint")
    group.add_argument("--latest", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def main() -> int:
    """把恢复参数交给统一训练入口。"""

    args = parse_args()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/training/run_train.py"),
        "--config",
        str(args.config),
        "--environment",
        args.environment,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.latest:
        command.append("--resume-latest")
    else:
        command.extend(["--resume-from-checkpoint", args.resume_from_checkpoint])
    if args.mock:
        command.append("--mock")
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
