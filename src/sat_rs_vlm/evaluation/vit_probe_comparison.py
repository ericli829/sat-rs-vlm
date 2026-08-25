"""4B visual probe 的三点 E1 配对比较封装。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sat_rs_vlm.evaluation.comparison import compare_evaluations


def _summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "comparison_summary.json").read_text(encoding="utf-8"))


def compare_vit_probe_evaluations(
    baseline_dir: str | Path,
    checkpoint100_dir: str | Path,
    checkpoint200_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    bootstrap_resamples: int = 1000,
) -> dict[str, Path]:
    """比较 baseline-vs-100 和 baseline-vs-200，并生成汇总 JSON/Markdown。"""

    root = Path(output_dir)
    pair100 = root / "baseline_vs_checkpoint100"
    pair200 = root / "baseline_vs_checkpoint200"
    first = compare_evaluations(
        baseline_dir,
        checkpoint100_dir,
        pair100,
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    second = compare_evaluations(
        baseline_dir,
        checkpoint200_dir,
        pair200,
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    first_summary = _summary(pair100)
    second_summary = _summary(pair200)
    first_sha = first_summary.get("evaluation_tier_sha256")
    second_sha = second_summary.get("evaluation_tier_sha256")
    if first_sha != second_sha:
        raise ValueError(
            "Probe E1 SHA mismatch: "
            f"checkpoint100={first_sha}, checkpoint200={second_sha}"
        )

    payload = {
        "schema_version": "1.0",
        "experiment": "qwen3vl_4b_vit_probe_last2",
        "evaluation_tier": first_summary.get("evaluation_tier"),
        "evaluation_tier_version": first_summary.get("evaluation_tier_version"),
        "evaluation_tier_sha256": first_sha,
        "quick_directional_warning": (
            "E1 用于快速方向判断，不用于最终统计结论；明显正向结果必须在 E2 上复核。"
        ),
        "baseline_vs_checkpoint100": first_summary,
        "baseline_vs_checkpoint200": second_summary,
        "pairwise_outputs": {
            "checkpoint100": {key: str(value) for key, value in first.items()},
            "checkpoint200": {key: str(value) for key, value in second.items()},
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    comparison_json = root / "comparison.json"
    comparison_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    comparison_md = root / "comparison.md"
    comparison_md.write_text(_markdown(payload), encoding="utf-8")
    return {"comparison": comparison_json, "markdown": comparison_md}


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-VL-4B ViT Last-2 Probe E1 Comparison",
        "",
        f"- Evaluation tier: `{payload.get('evaluation_tier')}` / `{payload.get('evaluation_tier_version')}`",
        f"- Evaluation tier SHA256: `{payload.get('evaluation_tier_sha256')}`",
        "- E1 用于快速方向判断，不用于最终统计结论；明显正向结果必须在 E2 上复核。",
        "",
        "## Paired Results",
        "",
        "| Comparison | Task | Primary metric | Baseline | Candidate | Delta | CI95 | Wins/Ties/Losses |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for label in ("baseline_vs_checkpoint100", "baseline_vs_checkpoint200"):
        summary = payload[label]
        for task, task_payload in sorted(summary.get("by_task", {}).items()):
            primary = task_payload.get("primary_metric")
            metric = task_payload.get("metrics", {}).get(primary, {})
            if metric.get("status") != "ok":
                continue
            ci = metric.get("improvement_ci95_paired_bootstrap") or {}
            wins = f"{metric.get('wins', 0)}/{metric.get('ties', 0)}/{metric.get('losses', 0)}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        label.replace("baseline_vs_", "baseline vs "),
                        task,
                        str(primary),
                        f"{metric['baseline_mean']:.6f}",
                        f"{metric['candidate_mean']:.6f}",
                        f"{metric['candidate_minus_baseline']:.6f}",
                        f"[{ci.get('low')}, {ci.get('high')}]",
                        wins,
                    ]
                )
                + "|"
            )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "本实验只改变已训练 Round1 adapter 上的 ViT last-2 trainable surface；LoRA rank、target modules、loss、prompt、E1 数据和 generation config 均保持不变。",
            "",
            "若 Counting/Detection/Scene 中至少两个方向改善且 broad tasks 无系统性退化，可考虑 last-4、main merger 或 E2 复核；否则不要自动扩大解冻范围。",
            "",
        ]
    )
    return "\n".join(lines)

