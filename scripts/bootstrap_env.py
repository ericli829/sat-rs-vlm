"""自动创建本地开发环境。

算法/流程：
    1. 自动向上查找包含 pyproject.toml 的项目根目录。
    2. 校验 Python >= 3.10。
    3. 创建或重建 `.venv`。
    4. 使用 `.venv` 内的 Python 执行 pip 升级和 editable 安装。
    5. 可选安装 `[model]` 大模型依赖。

接口：
    python scripts/bootstrap_env.py [--with-model] [--clean]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)


def find_project_root() -> Path:
    """定位项目根目录。

    返回值：
        Path：包含 pyproject.toml 的目录。
    """

    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise SystemExit("Could not locate project root containing pyproject.toml.")


def check_python_version() -> None:
    """检查当前 Python 版本。

    返回值：
        None。

    异常：
        SystemExit：版本低于 3.10 时退出。
    """

    if sys.version_info < MIN_PYTHON:
        version = ".".join(str(part) for part in MIN_PYTHON)
        raise SystemExit(f"Python >= {version} is required. Current: {sys.version}")


def venv_python_path(venv_dir: Path) -> Path:
    """返回虚拟环境中的 Python 路径。

    参数：
        venv_dir：虚拟环境目录。

    返回值：
        Path：Windows 下为 Scripts/python.exe，类 Unix 下为 bin/python。
    """

    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_command(command: list[str], cwd: Path) -> None:
    """执行子进程命令并检查返回码。

    参数：
        command：命令参数列表。
        cwd：命令工作目录。

    返回值：
        None。

    异常：
        SystemExit：命令失败时给出清晰错误和下一步排查方向。
    """

    printable = " ".join(command)
    print(f"+ {printable}")
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Command failed with exit code {exc.returncode}: {printable}\n"
            "Please check network access and package index availability, then retry."
        ) from exc


def create_venv(root: Path, clean: bool) -> Path:
    """创建 `.venv`。

    参数：
        root：项目根目录。
        clean：是否先删除已有 `.venv`。

    返回值：
        Path：虚拟环境目录。
    """

    venv_dir = root / ".venv"
    if clean and venv_dir.exists():
        print(f"Removing existing virtual environment: {venv_dir}")
        shutil.rmtree(venv_dir)
    if not venv_dir.exists():
        run_command([sys.executable, "-m", "venv", str(venv_dir)], root)
    return venv_dir


def print_activation_hint(root: Path) -> None:
    """打印虚拟环境激活命令。

    参数：
        root：项目根目录。

    返回值：
        None。
    """

    if sys.platform == "win32":
        activate = ".venv\\Scripts\\activate"
    else:
        activate = "source .venv/bin/activate"
    print("\nEnvironment ready.")
    print(f"Project root: {root}")
    print(f"Activate with: {activate}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    返回值：
        argparse.Namespace：包含 with_model 和 clean 两个布尔参数。
    """

    parser = argparse.ArgumentParser(description="Bootstrap sat-rs-vlm development environment.")
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Install optional model dependencies.",
    )
    parser.add_argument("--clean", action="store_true", help="Recreate .venv from scratch.")
    parser.add_argument(
        "--torch-index-url",
        default=None,
        help=(
            "Optional official PyTorch wheel index, for example "
            "https://download.pytorch.org/whl/cu130."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """脚本主入口。

    返回值：
        int：成功返回 0；失败通过 SystemExit 退出。
    """

    args = parse_args()
    check_python_version()
    root = find_project_root()
    venv_dir = create_venv(root, clean=args.clean)
    python = venv_python_path(venv_dir)
    if not python.exists():
        raise SystemExit(f"Could not find virtualenv Python at {python}")

    run_command(
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        root,
    )
    run_command([str(python), "-m", "pip", "install", "-e", ".[dev]"], root)
    if args.with_model:
        if args.torch_index_url:
            run_command(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--force-reinstall",
                    "--timeout",
                    "1000",
                    "--retries",
                    "10",
                    "torch",
                    "torchvision",
                    "--index-url",
                    args.torch_index_url,
                ],
                root,
            )
        run_command([str(python), "-m", "pip", "install", "-e", ".[model]"], root)
        run_command([str(python), "scripts/check_env.py", "--require-model"], root)

    print_activation_hint(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
