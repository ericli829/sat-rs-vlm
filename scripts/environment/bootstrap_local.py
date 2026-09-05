"""跨平台创建或复用 `.venv`，并安装显式选择的依赖组。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """解析本地环境初始化参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=PROJECT_ROOT / ".venv")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--with-dev", action="store_true")
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument(
        "--with-retriever",
        action="store_true",
        help="Install the pyproject retriever extra (open_clip_torch and timm).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main() -> int:
    """创建环境并用当前解释器的 venv 模块安装项目。"""

    args = parse_args()
    venv = args.venv.resolve()
    extras = ["dev"] if args.with_dev else []
    if args.with_model:
        extras.append("model")
    if args.with_retriever:
        extras.append("retriever")
    requirement = f".[{','.join(extras)}]" if extras else "."
    commands: list[list[str]] = []
    if args.clean and venv.exists():
        if args.dry_run:
            print(f"Would remove: {venv}")
        else:
            shutil.rmtree(venv)
    if not _venv_python(venv).is_file():
        commands.append([sys.executable, "-m", "venv", str(venv)])
    python = _venv_python(venv)
    commands.append([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    commands.append([str(python), "-m", "pip", "install", "-e", requirement])
    for command in commands:
        print(subprocess.list2cmdline(command))
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    print(f"Environment ready: {venv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
