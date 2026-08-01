"""从标准 metrics 文件生成静态图表，不执行推理或故障注入。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json" if path.is_dir() else path
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "overall" not in payload:
        raise ValueError(f"Not a standardized reliability summary: {summary_path}")
    return payload


def plot_reliability_results(input_path: str | Path, output_dir: str | Path) -> list[Path]:
    """绘制总体故障率和按任务 changed/invalid rate，返回生成文件列表。"""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires the optional 'reliability-plot' extra") from exc

    summary = _load_summary(Path(input_path).expanduser().resolve())
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    overall = dict(summary.get("overall", {}))
    rate_names = ("changed_rate", "invalid_rate", "empty_rate", "exact_match_drop")
    available: list[tuple[str, float]] = []
    for name in rate_names:
        value = overall.get(name)
        if isinstance(value, (int, float)):
            available.append((name, float(value)))
    if available:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar([name for name, _ in available], [value for _, value in available])
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Rate")
        axis.set_title("Reliability summary")
        figure.tight_layout()
        output = destination / "reliability_rates.png"
        figure.savefig(output, dpi=160)
        plt.close(figure)
        generated.append(output)

    by_task = dict(summary.get("by_task", {}))
    if by_task:
        tasks = list(by_task)
        changed = [float(by_task[task].get("changed_rate") or 0.0) for task in tasks]
        invalid = [float(by_task[task].get("invalid_rate") or 0.0) for task in tasks]
        positions = list(range(len(tasks)))
        figure, axis = plt.subplots(figsize=(max(8, len(tasks) * 1.5), 4.5))
        axis.bar([position - 0.2 for position in positions], changed, width=0.4, label="changed")
        axis.bar([position + 0.2 for position in positions], invalid, width=0.4, label="invalid")
        axis.set_xticks(positions, tasks, rotation=20, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Rate")
        axis.legend()
        figure.tight_layout()
        output = destination / "reliability_by_task.png"
        figure.savefig(output, dpi=160)
        plt.close(figure)
        generated.append(output)
    return generated
