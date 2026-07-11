"""检查当前 Python 环境。

算法/流程：
    1. 打印 Python 路径、版本和是否位于项目 `.venv`。
    2. 检查基础依赖、开发依赖和可选模型依赖是否可 import。
    3. 如果 torch 可用，额外打印 CUDA 信息。
    4. 基础依赖缺失时返回非 0；模型依赖缺失只提示 optional。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

BASE_DEPENDENCIES = ("pydantic", "yaml", "typer", "fastapi", "uvicorn")
DEV_DEPENDENCIES = ("pytest", "httpx")
MODEL_DEPENDENCIES = ("torch", "transformers", "PIL", "peft", "bitsandbytes", "qwen_vl_utils")


def parse_args() -> argparse.Namespace:
    """解析环境检查参数。

    返回值：
        argparse.Namespace：require_model 表示模型运行时异常是否导致非零退出。
    """

    parser = argparse.ArgumentParser(description="Check the sat-rs-vlm Python environment.")
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="Fail when model dependencies or the PyTorch runtime are unavailable.",
    )
    return parser.parse_args()


def find_project_root() -> Path:
    """定位项目根目录。

    返回值：
        Path：包含 pyproject.toml 的目录；找不到时返回当前工作目录。
    """

    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def is_project_venv(root: Path) -> bool:
    """判断当前解释器是否来自项目 `.venv`。

    参数：
        root：项目根目录。

    返回值：
        bool：sys.prefix 是否等于 root/.venv。
    """

    return Path(sys.prefix).resolve() == (root / ".venv").resolve()


def check_imports(names: tuple[str, ...], *, optional: bool) -> bool:
    """检查模块是否可导入。

    参数：
        names：模块名列表。
        optional：是否为可选依赖；可选依赖缺失不导致失败。

    返回值：
        bool：全部必需依赖存在时为 True。
    """

    all_found = True
    for name in names:
        if importlib.util.find_spec(name) is None:
            status = "optional dependency missing" if optional else "missing"
            print(f"[{status}] {name}")
            all_found = False
        else:
            print(f"[ok] {name}")
    return all_found


def print_torch_info() -> bool:
    """打印 torch 和 CUDA 信息。

    返回值：
        bool：torch 可正常导入时为 True；缺失或 DLL 加载失败时为 False。
    """

    if importlib.util.find_spec("torch") is None:
        return False
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        print(f"[model runtime broken] torch import failed: {exc}")
        print("Reinstall a matching PyTorch build from https://pytorch.org/get-started/locally/.")
        return False
    print(f"torch version: {getattr(torch, '__version__', 'unknown')}")
    cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
    print(f"PyTorch CUDA build: {cuda_build}")
    cuda_available = bool(torch.cuda.is_available())
    print(f"CUDA available: {cuda_available}")
    if cuda_build is None:
        print("[warning] A CPU-only PyTorch wheel is installed; GPU inference is unavailable.")
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    print(f"CUDA device count: {device_count}")
    if cuda_available and device_count > 0:
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"CUDA current device name: {device_name}")
    return True


def main() -> int:
    """脚本主入口。

    返回值：
        int：基础依赖完整返回 0，否则返回 1。
    """

    args = parse_args()
    root = find_project_root()
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Project root: {root}")
    print(f"Inside project .venv: {is_project_venv(root)}")

    print("\nBase dependencies:")
    base_ok = check_imports(BASE_DEPENDENCIES, optional=False)
    print("\nDevelopment dependencies:")
    check_imports(DEV_DEPENDENCIES, optional=True)
    print("\nModel dependencies:")
    model_dependencies_ok = check_imports(MODEL_DEPENDENCIES, optional=True)
    torch_runtime_ok = print_torch_info()

    if not base_ok:
        print("\nBase dependencies are missing. Run: python scripts/bootstrap_env.py")
        return 1
    if args.require_model and not (model_dependencies_ok and torch_runtime_ok):
        print("\nModel runtime is not ready. Run: python scripts/bootstrap_env.py --with-model")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
