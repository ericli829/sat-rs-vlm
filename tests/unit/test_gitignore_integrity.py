"""验证 Git 忽略规则不会意外排除项目源码。

该测试使用 ``git check-ignore --no-index`` 检查源码路径。``--no-index`` 会同时
检查已跟踪文件，因此即使某个文件曾经被提交，过宽的新增规则也能被及时发现。
测试不需要模型、GPU 或网络；当源码以不含 ``.git`` 的发布包形式运行时会跳过。
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = ("src", "scripts", "tests")


def _python_source_paths() -> list[str]:
    """返回需要受版本控制保护的 Python 源码相对路径列表。"""

    paths: list[str] = []
    for root_name in SOURCE_ROOTS:
        root = REPOSITORY_ROOT / root_name
        if root.is_dir():
            paths.extend(
                path.relative_to(REPOSITORY_ROOT).as_posix() for path in root.rglob("*.py")
            )
    return sorted(paths)


def test_python_sources_are_not_ignored_by_git() -> None:
    """确保任何 Python 源码都不会命中 ``.gitignore`` 规则。"""

    if not (REPOSITORY_ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("Git metadata or executable is unavailable")

    source_paths = _python_source_paths()
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=REPOSITORY_ROOT,
        input="\n".join(source_paths),
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode not in (0, 1):
        pytest.fail(
            f"git check-ignore failed with exit code {result.returncode}: {result.stderr.strip()}"
        )

    ignored_paths = [line for line in result.stdout.splitlines() if line]

    assert not ignored_paths, "Python source files must not be ignored by Git:\n" + "\n".join(
        ignored_paths
    )
