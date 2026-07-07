"""检查当前 Python 环境。

算法/流程：
    1. 打印 Python 路径、版本和是否位于项目 `.venv`。
    2. 检查基础依赖、开发依赖和可选模型依赖是否可 import。
    3. 如果 torch 可用，额外打印 CUDA 信息。
    4. 基础依赖缺失时返回非 0；模型依赖缺失只提示 optional。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

BASE_DEPENDENCIES = ("pydantic", "yaml", "typer", "fastapi", "uvicorn")
DEV_DEPENDENCIES = ("pytest", "httpx")
MODEL_DEPENDENCIES = ("torch", "transformers", "PIL", "peft", "bitsandbytes", "qwen_vl_utils")


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

    ok = True
    for name in names:
        if importlib.util.find_spec(name) is None:
            status = "optional dependency missing" if optional else "missing"
            print(f"[{status}] {name}")
            if not optional:
                ok = False
        else:
            print(f"[ok] {name}")
    return ok


def print_torch_info() -> None:
    """打印 torch 和 CUDA 信息。

    返回值：
        None。torch 未安装时静默返回。
    """

    if importlib.util.find_spec("torch") is None:
        return
    torch = importlib.import_module("torch")
    print(f"torch version: {getattr(torch, '__version__', 'unknown')}")
    cuda_available = bool(torch.cuda.is_available())
    print(f"CUDA available: {cuda_available}")
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    print(f"CUDA device count: {device_count}")
    if cuda_available and device_count > 0:
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"CUDA current device name: {device_name}")


def main() -> int:
    """脚本主入口。

    返回值：
        int：基础依赖完整返回 0，否则返回 1。
    """

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
    check_imports(MODEL_DEPENDENCIES, optional=True)
    print_torch_info()

    if not base_ok:
        print("\nBase dependencies are missing. Run: python scripts/bootstrap_env.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
