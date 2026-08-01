"""统一训练入口使用的实验目录、恢复点和环境快照工具。"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    """按 Trainer 的 `checkpoint-<step>` 步数返回最新恢复点。"""

    candidates: list[tuple[int, Path]] = []
    if not checkpoints_dir.is_dir():
        return None
    for path in checkpoints_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def resolve_resume_checkpoint(value: str | None, experiment_dir: Path) -> Path | None:
    """解析显式恢复点或 `latest`，并验证目录存在。"""

    if value is None:
        return None
    if value == "latest":
        checkpoint = latest_checkpoint(experiment_dir / "checkpoints")
        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoint-* directory found in {experiment_dir / 'checkpoints'}"
            )
        return checkpoint
    checkpoint = Path(value).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    return checkpoint


def create_experiment_layout(
    output_root: Path,
    *,
    group: str,
    experiment_name: str,
    seed: int,
    explicit_output: Path | None = None,
) -> Path:
    """创建一次实验的标准目录结构并返回实验根目录。"""

    if explicit_output is not None:
        experiment_dir = explicit_output.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = output_root / group / f"{timestamp}_{experiment_name}_seed{seed}"
    for child in ("logs", "checkpoints", "predictions", "metrics", "artifacts"):
        (experiment_dir / child).mkdir(parents=True, exist_ok=True)
    return experiment_dir


def git_commit(project_root: Path) -> str:
    """读取当前 Git commit；非 Git 环境返回 `unknown`。"""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def environment_snapshot() -> dict[str, Any]:
    """收集不触发模型加载的基础运行环境信息。"""

    packages = (
        "pydantic",
        "PyYAML",
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "path_environment": {
            name: os.environ.get(name)
            for name in (
                "PROJECT_ROOT",
                "DATA_ROOT",
                "MODEL_ROOT",
                "OUTPUT_ROOT",
                "CACHE_ROOT",
                "TMPDIR",
                "HF_HOME",
                "HF_HUB_CACHE",
                "TORCH_HOME",
                "PIP_CACHE_DIR",
            )
        },
    }


def disk_report(path: Path) -> dict[str, float]:
    """返回目标磁盘的总量、已用量和剩余量（GiB）。"""

    usage = shutil.disk_usage(path)
    gib = 1024**3
    return {
        "total_gib": round(usage.total / gib, 2),
        "used_gib": round(usage.used / gib, 2),
        "free_gib": round(usage.free / gib, 2),
    }


def write_json(path: Path, payload: Any) -> None:
    """以 UTF-8 和缩进格式写 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
