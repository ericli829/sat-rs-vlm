"""可靠性实验标准目录和报告写入工具。"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sat_rs_vlm.configuration.layered import write_resolved_config
from sat_rs_vlm.training.experiment import write_json


@dataclass(frozen=True)
class ReliabilityRunLayout:
    """一次可靠性实验的全部标准目录。"""

    root: Path
    logs: Path
    clean: Path
    faults: Path
    fault_adapters: Path
    predictions: Path
    metrics: Path
    protection: Path
    figures: Path
    artifacts: Path


def make_run_id() -> str:
    """生成 UTC 时间与短 UUID 组合的无冲突运行标识。"""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def create_reliability_run_layout(
    output_root: str | Path,
    *,
    experiment_name: str,
    run_id: str | None = None,
    overwrite: bool = False,
) -> ReliabilityRunLayout:
    """创建 `${OUTPUT_ROOT}/reliability/<experiment>/<run_id>`，默认拒绝覆盖。"""

    if not experiment_name.strip() or any(token in experiment_name for token in ("/", "\\", "..")):
        raise ValueError("experiment_name must be a safe directory name")
    chosen_run_id = run_id or make_run_id()
    if not chosen_run_id.strip() or any(token in chosen_run_id for token in ("/", "\\", "..")):
        raise ValueError("run_id must be a safe directory name")
    root = (
        Path(output_root).expanduser().resolve() / "reliability" / experiment_name / chosen_run_id
    )
    if root.exists() and not overwrite:
        raise FileExistsError(f"Reliability run already exists: {root}")
    if root.exists():
        shutil.rmtree(root)
    layout = ReliabilityRunLayout(
        root=root,
        logs=root / "logs",
        clean=root / "clean",
        faults=root / "faults",
        fault_adapters=root / "faults" / "adapters",
        predictions=root / "predictions",
        metrics=root / "metrics",
        protection=root / "protection",
        figures=root / "figures",
        artifacts=root / "artifacts",
    )
    for path in (
        layout.logs,
        layout.clean,
        layout.faults,
        layout.fault_adapters,
        layout.predictions,
        layout.metrics,
        layout.protection,
        layout.figures,
        layout.artifacts,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return layout


def open_reliability_run_layout(
    output_root: str | Path,
    *,
    experiment_name: str,
    run_id: str,
) -> ReliabilityRunLayout:
    """打开已存在的标准运行目录，用于显式 `--resume`。"""

    root = Path(output_root).expanduser().resolve() / "reliability" / experiment_name / run_id
    if not root.is_dir():
        raise FileNotFoundError(f"Reliability run does not exist for resume: {root}")
    layout = ReliabilityRunLayout(
        root=root,
        logs=root / "logs",
        clean=root / "clean",
        faults=root / "faults",
        fault_adapters=root / "faults" / "adapters",
        predictions=root / "predictions",
        metrics=root / "metrics",
        protection=root / "protection",
        figures=root / "figures",
        artifacts=root / "artifacts",
    )
    for path in (
        layout.logs,
        layout.clean,
        layout.faults,
        layout.fault_adapters,
        layout.predictions,
        layout.metrics,
        layout.protection,
        layout.figures,
        layout.artifacts,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return layout


def write_run_metadata(
    layout: ReliabilityRunLayout,
    *,
    resolved_config: dict[str, Any],
    command: str,
    environment: dict[str, Any],
    git_commit: str,
) -> None:
    """写出 resolved config、命令、环境和 Git 版本。"""

    write_resolved_config(resolved_config, layout.root / "config_resolved.yaml")
    (layout.root / "command.txt").write_text(command.rstrip() + "\n", encoding="utf-8")
    write_json(layout.root / "environment.json", environment)
    (layout.root / "git_commit.txt").write_text(git_commit.rstrip() + "\n", encoding="utf-8")


def write_metric_reports(layout: ReliabilityRunLayout, summary: dict[str, Any]) -> None:
    """将标准报告拆分为总体和按任务两个稳定文件。"""

    write_json(layout.metrics / "summary.json", summary)
    write_json(
        layout.metrics / "by_task.json",
        {
            "schema_version": summary.get("schema_version", "1.0"),
            "execution_mode": summary.get("execution_mode"),
            "experiment_name": summary.get("experiment_name"),
            "run_id": summary.get("run_id"),
            "by_task": summary.get("by_task", {}),
        },
    )
