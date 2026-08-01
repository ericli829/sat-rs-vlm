"""导出一次实验可复现所需的环境、pip、GPU 和命令快照。"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析环境导出参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command", default="")
    return parser.parse_args()


def _run(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def main() -> int:
    """导出环境；没有 nvidia-smi 只记录状态，不视为失败。"""

    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pip_code, pip_text = _run([sys.executable, "-m", "pip", "freeze"])
    (output / "pip-freeze.txt").write_text(pip_text + "\n", encoding="utf-8")
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        nvidia_code, nvidia_text = _run([nvidia])
    else:
        nvidia_code, nvidia_text = 127, "nvidia-smi is not available."
    (output / "nvidia-smi.txt").write_text(nvidia_text + "\n", encoding="utf-8")
    (output / "command.txt").write_text(args.command + "\n", encoding="utf-8")
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pip_freeze_returncode": pip_code,
        "nvidia_smi_returncode": nvidia_code,
    }
    (output / "environment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if pip_code == 0 else pip_code


if __name__ == "__main__":
    raise SystemExit(main())
