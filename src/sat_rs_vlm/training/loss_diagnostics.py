"""多任务 Causal LM 损失长度偏置的只读诊断工具。

本模块不加载模型、不反向传播，也不修改 Trainer。它接收一次真实前向得到的
``logits`` 与 assistant-only ``labels``，将标准 Causal LM 的 token 级交叉熵拆分到
样本和任务两个层级，用于区分“任务采样比例”和“token 在 batch loss 分子中的比例”。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def analyze_causal_lm_batch_loss(
    *,
    logits: Any,
    labels: Any,
    sample_ids: Sequence[str],
    task_types: Sequence[str],
    torch: Any,
    model_batch_loss: float,
) -> dict[str, Any]:
    """拆分一个混合任务 batch 的 Causal LM 交叉熵。

    算法：与 Transformers ``ForCausalLMLoss`` 保持一致，将 labels 左移一位，
    忽略 ``-100``，再计算每个有效 assistant token 的 unreduced cross entropy。
    每个样本的 token CE 求均值就是 per-sample normalized loss；所有有效 token CE
    的均值就是当前模型 batch loss 的可独立复算值。

    参数：
        logits: 模型输出，形状为 ``[batch, sequence, vocabulary]``。
        labels: Collator 生成的 assistant-only 标签，padding/user/image token 为 -100。
        sample_ids: 与 batch 顺序一致的样本 ID。
        task_types: 与 batch 顺序一致的标准任务类型。
        torch: 动态导入的 torch 模块，避免导入本模块时要求 GPU/torch。
        model_batch_loss: ``outputs.loss`` 转换得到的标量，用于与手工复算交叉校验。

    返回：
        JSON 可序列化字典，含每个样本、每个任务与 batch 级两种 loss 的统计。
    """

    if len(sample_ids) != len(task_types):
        raise ValueError("sample_ids and task_types must have the same length")
    if int(logits.shape[0]) != len(sample_ids) or tuple(labels.shape[:2]) != tuple(
        logits.shape[:2]
    ):
        raise ValueError("logits, labels, sample_ids, and task_types must describe one batch")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    rows: list[dict[str, Any]] = []
    total_cross_entropy = 0.0
    total_supervised_tokens = 0
    for index, (sample_id, task_type) in enumerate(zip(sample_ids, task_types, strict=True)):
        valid = shifted_labels[index] != -100
        causal_supervised_tokens = int(valid.sum().item())
        label_supervised_tokens = int((labels[index] != -100).sum().item())
        if causal_supervised_tokens < 1:
            raise ValueError(f"Sample has no causal-LM supervised tokens: {sample_id}")
        token_logits = shifted_logits[index][valid].float()
        token_labels = shifted_labels[index][valid].to(token_logits.device)
        token_cross_entropy = torch.nn.functional.cross_entropy(
            token_logits,
            token_labels,
            reduction="none",
        )
        sum_cross_entropy = float(token_cross_entropy.sum().item())
        mean_token_ce = float(token_cross_entropy.mean().item())
        rows.append(
            {
                "sample_id": str(sample_id),
                "task_type": str(task_type),
                "supervised_token_count": label_supervised_tokens,
                "causal_supervised_token_count": causal_supervised_tokens,
                "sum_cross_entropy": sum_cross_entropy,
                "mean_token_ce": mean_token_ce,
                "sample_loss": mean_token_ce,
            }
        )
        total_cross_entropy += sum_cross_entropy
        total_supervised_tokens += causal_supervised_tokens

    manual_batch_loss = total_cross_entropy / total_supervised_tokens
    per_sample_mean_loss = sum(float(row["sample_loss"]) for row in rows) / len(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_type"])].append(row)
    task_statistics: dict[str, dict[str, Any]] = {}
    total_samples = len(rows)
    for task, task_rows in sorted(grouped.items()):
        task_tokens = sum(int(row["supervised_token_count"]) for row in task_rows)
        task_cross_entropy = sum(float(row["sum_cross_entropy"]) for row in task_rows)
        task_statistics[task] = {
            "sample_count": len(task_rows),
            "sample_share": len(task_rows) / total_samples,
            "supervised_tokens": task_tokens,
            "supervised_token_share": task_tokens / total_supervised_tokens,
            "sum_cross_entropy": task_cross_entropy,
            "loss_numerator_share": task_cross_entropy / total_cross_entropy,
            "mean_token_ce": task_cross_entropy / task_tokens,
            "mean_sample_loss": sum(float(row["sample_loss"]) for row in task_rows)
            / len(task_rows),
        }
    return {
        "sample_rows": rows,
        "task_statistics": task_statistics,
        "batch_statistics": {
            "sample_count": total_samples,
            "supervised_tokens": total_supervised_tokens,
            "sum_cross_entropy": total_cross_entropy,
            "current_batch_loss": float(model_batch_loss),
            "manual_batch_token_mean_loss": manual_batch_loss,
            "model_manual_loss_absolute_difference": abs(
                float(model_batch_loss) - manual_batch_loss
            ),
            "per_sample_normalized_loss_mean": per_sample_mean_loss,
        },
    }


def summarize_multitask_loss_bias(
    batch_reports: Sequence[Mapping[str, Any]],
    *,
    material_share_gap: float,
) -> dict[str, Any]:
    """汇总多个混合 batch，并给出可追溯的长度偏置判断。

    ``loss_numerator_share`` 与 ``sample_share`` 的正差大于阈值，且该份额更接近
    ``supervised_token_share``，才认定为“与答案长度一致的 over-representation”。
    这避免把模型本身某任务 token CE 较高误判为纯长度偏置。

    参数：
        batch_reports: :func:`analyze_causal_lm_batch_loss` 的输出列表。
        material_share_gap: 判定明显份额偏移的绝对阈值，例如 0.10 表示十个百分点。

    返回：
        跨 batch 聚合的任务表、两种 loss 均值和结构化结论。
    """

    if not batch_reports:
        raise ValueError("At least one batch report is required")
    if not 0.0 < material_share_gap < 1.0:
        raise ValueError("material_share_gap must be between 0 and 1")

    task_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"samples": 0.0, "tokens": 0.0, "cross_entropy": 0.0, "sample_loss_sum": 0.0}
    )
    batch_losses: list[float] = []
    per_sample_losses: list[float] = []
    peak_memory_allocated: list[float] = []
    peak_memory_reserved: list[float] = []
    for report in batch_reports:
        stats = dict(report["batch_statistics"])
        batch_losses.append(float(stats["current_batch_loss"]))
        per_sample_losses.append(float(stats["per_sample_normalized_loss_mean"]))
        for task, values in dict(report["task_statistics"]).items():
            total = task_totals[str(task)]
            total["samples"] += float(values["sample_count"])
            total["tokens"] += float(values["supervised_tokens"])
            total["cross_entropy"] += float(values["sum_cross_entropy"])
            total["sample_loss_sum"] += float(values["mean_sample_loss"]) * float(
                values["sample_count"]
            )
        memory = report.get("memory")
        if isinstance(memory, Mapping):
            allocated = memory.get("peak_memory_allocated_mb")
            reserved = memory.get("peak_memory_reserved_mb")
            if allocated is not None:
                peak_memory_allocated.append(float(allocated))
            if reserved is not None:
                peak_memory_reserved.append(float(reserved))

    total_samples = sum(values["samples"] for values in task_totals.values())
    total_tokens = sum(values["tokens"] for values in task_totals.values())
    total_cross_entropy = sum(values["cross_entropy"] for values in task_totals.values())
    tasks: dict[str, dict[str, Any]] = {}
    length_associated_tasks: list[str] = []
    for task, values in sorted(task_totals.items()):
        sample_share = values["samples"] / total_samples
        token_share = values["tokens"] / total_tokens
        numerator_share = values["cross_entropy"] / total_cross_entropy
        share_gap = numerator_share - sample_share
        closer_to_token_share = abs(numerator_share - token_share) <= abs(
            numerator_share - sample_share
        )
        materially_overrepresented = share_gap >= material_share_gap
        length_associated = materially_overrepresented and closer_to_token_share
        if length_associated:
            length_associated_tasks.append(task)
        tasks[task] = {
            "sample_count": int(values["samples"]),
            "sample_share": sample_share,
            "supervised_tokens": int(values["tokens"]),
            "supervised_token_share": token_share,
            "sum_cross_entropy": values["cross_entropy"],
            "loss_numerator_share": numerator_share,
            "loss_numerator_share_minus_sample_share": share_gap,
            "mean_token_ce": values["cross_entropy"] / values["tokens"],
            "mean_sample_loss": values["sample_loss_sum"] / values["samples"],
            "materially_overrepresented": materially_overrepresented,
            "length_associated_overrepresentation": length_associated,
        }

    focus_tasks = [task for task in ("captioning", "detection") if task in tasks]
    weaker_tasks = [task for task in ("vqa", "counting") if task in tasks]
    focus_overrepresented = [
        task for task in focus_tasks if bool(tasks[task]["length_associated_overrepresentation"])
    ]
    weaker_underrepresented = [
        task
        for task in weaker_tasks
        if float(tasks[task]["loss_numerator_share_minus_sample_share"]) <= -material_share_gap
    ]
    supports_per_sample = bool(focus_overrepresented and weaker_underrepresented)
    return {
        "batch_count": len(batch_reports),
        "sample_count": int(total_samples),
        "supervised_tokens": int(total_tokens),
        "current_batch_loss_mean": sum(batch_losses) / len(batch_losses),
        "per_sample_normalized_loss_mean": sum(per_sample_losses) / len(per_sample_losses),
        "task_statistics": tasks,
        "memory": {
            "peak_memory_allocated_mb": max(peak_memory_allocated, default=None),
            "peak_memory_reserved_mb": max(peak_memory_reserved, default=None),
        },
        "judgement": {
            "material_share_gap": material_share_gap,
            "length_associated_overrepresented_tasks": length_associated_tasks,
            "captioning_or_detection_overrepresented": focus_overrepresented,
            "vqa_or_counting_underrepresented": weaker_underrepresented,
            "supports_per_sample_normalized_loss_experiment": supports_per_sample,
            "conclusion": (
                "Evidence supports a per-sample normalized-loss experiment: long-answer "
                "tasks are materially overrepresented in the batch loss numerator while "
                "VQA/counting are underrepresented."
                if supports_per_sample
                else "This diagnostic does not yet provide sufficient evidence to change the "
                "formal loss. Inspect task shares and mean token CE before proposing H1 changes."
            ),
        },
    }


def multitask_loss_bias_markdown(summary: Mapping[str, Any]) -> str:
    """将聚合诊断结果渲染为面向实验决策的 Markdown 报告。

    参数：
        summary: :func:`summarize_multitask_loss_bias` 产出的汇总字典。

    返回：
        UTF-8 Markdown 文本；不会写文件，方便调用方选择输出目录。
    """

    judgement = dict(summary["judgement"])
    memory = dict(summary["memory"])
    length_associated = ", ".join(judgement["length_associated_overrepresented_tasks"]) or "none"
    focus_overrepresented = (
        ", ".join(judgement["captioning_or_detection_overrepresented"]) or "none"
    )
    weaker_underrepresented = ", ".join(judgement["vqa_or_counting_underrepresented"]) or "none"
    lines = [
        "# Multitask Loss Length-Bias Diagnostic",
        "",
        f"- Mixed batches: {summary['batch_count']}",
        f"- Executed samples: {summary['sample_count']}",
        f"- Supervised assistant tokens: {summary['supervised_tokens']}",
        f"- Current batch loss mean: {float(summary['current_batch_loss_mean']):.6f}",
        "- Per-sample normalized loss mean: "
        f"{float(summary['per_sample_normalized_loss_mean']):.6f}",
        f"- Peak CUDA allocated memory (MB): {memory.get('peak_memory_allocated_mb')}",
        f"- Peak CUDA reserved memory (MB): {memory.get('peak_memory_reserved_mb')}",
        "",
        "## Task Shares",
        "",
        "| Task | Sample share | Supervised-token share | Loss-numerator share | "
        "Mean token CE | Mean sample loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, values in dict(summary["task_statistics"]).items():
        lines.append(
            f"| {task} | {float(values['sample_share']):.4f} | "
            f"{float(values['supervised_token_share']):.4f} | "
            f"{float(values['loss_numerator_share']):.4f} | "
            f"{float(values['mean_token_ce']):.6f} | "
            f"{float(values['mean_sample_loss']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Judgement",
            "",
            f"- Material share-gap threshold: {float(judgement['material_share_gap']):.4f}",
            f"- Length-associated overrepresented tasks: {length_associated}",
            f"- Captioning/detection overrepresented: {focus_overrepresented}",
            f"- VQA/counting underrepresented: {weaker_underrepresented}",
            "- Per-sample normalized-loss experiment supported: "
            f"{judgement['supports_per_sample_normalized_loss_experiment']}",
            f"- Conclusion: {judgement['conclusion']}",
            "",
            "## Interpretation",
            "",
            "The normal Causal LM batch loss is the mean CE across all assistant tokens "
            "in the mixed batch. `loss_numerator_share` therefore tests token-level "
            "influence before the common batch denominator is applied. This is a read-only "
            "diagnostic; it does not change the formal H1 loss or Trainer.",
        ]
    )
    return "\n".join(lines) + "\n"
