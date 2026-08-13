"""Qwen3-VL 多任务训练的可替换 Loss Strategy 接口。

该模块只负责 logits、assistant-only labels 与 task metadata 之间的数学计算，不依赖
Trainer、Dataset 或具体模型类。所有策略都执行标准 causal shift，并在 float32 中计算
交叉熵，以兼容 fp16/bf16 前向并避免低精度 softmax 的数值风险。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sat_rs_vlm.training.config import MultitaskLossConfig


@dataclass(frozen=True)
class MultitaskLossOutput:
    """一次多任务 loss 计算的标量结果与无梯度诊断数据。"""

    loss: Any
    diagnostics: dict[str, Any]


class LossStrategy(ABC):
    """多任务 loss 策略抽象；新增算法只需实现 ``compute``。"""

    name: str

    @abstractmethod
    def compute(
        self,
        *,
        token_losses: Any,
        valid_mask: Any,
        task_types: Sequence[str],
        config: MultitaskLossConfig,
        torch: Any,
    ) -> MultitaskLossOutput:
        """根据 unreduced token CE、有效 mask 与 task 类型返回最终 loss。"""


def _diagnostics(
    *,
    token_losses: Any,
    valid_mask: Any,
    task_types: Sequence[str],
    sample_losses: Any,
    sample_weights: Any,
    contribution_basis: str,
    loss: Any,
) -> dict[str, Any]:
    """聚合每个任务的样本数、监督 token 数和 mean CE，且不保留计算图。"""

    task_rows: dict[str, list[int]] = defaultdict(list)
    for index, task in enumerate(task_types):
        task_rows[str(task)].append(index)
    by_task: dict[str, dict[str, float | int]] = {}
    detached_losses = token_losses.detach()
    detached_mask = valid_mask.detach()
    detached_sample_losses = sample_losses.detach()
    detached_sample_weights = sample_weights.detach()
    total_token_loss = detached_losses.sum()
    total_effective_numerator = (detached_sample_losses * detached_sample_weights).sum()
    for task, indices in sorted(task_rows.items()):
        task_token_sum = detached_losses[indices].sum()
        task_token_count = detached_mask[indices].sum()
        task_effective_numerator = (
            detached_sample_losses[indices] * detached_sample_weights[indices]
        ).sum()
        by_task[task] = {
            "samples": len(indices),
            "supervised_tokens": int(task_token_count.item()),
            "mean_token_ce": float((task_token_sum / task_token_count).item()),
            "mean_sample_loss": float(detached_sample_losses[indices].mean().item()),
            "token_loss_numerator_share": float((task_token_sum / total_token_loss).item()),
            "effective_loss_numerator_share": float(
                (task_effective_numerator / total_effective_numerator).item()
            ),
        }
    return {
        "loss/total": float(loss.detach().item()),
        "effective_contribution_basis": contribution_basis,
        "by_task": by_task,
    }


class TokenMeanLossStrategy(LossStrategy):
    """历史兼容策略：对 batch 中所有有效 assistant token 统一求均值。"""

    name = "token_mean"

    def compute(
        self,
        *,
        token_losses: Any,
        valid_mask: Any,
        task_types: Sequence[str],
        config: MultitaskLossConfig,
        torch: Any,
    ) -> MultitaskLossOutput:
        del config, torch
        token_counts = valid_mask.sum(dim=1)
        sample_losses = token_losses.sum(dim=1) / token_counts
        loss = token_losses.sum() / token_counts.sum()
        token_equivalent_weights = token_counts.to(dtype=sample_losses.dtype)
        return MultitaskLossOutput(
            loss=loss,
            diagnostics=_diagnostics(
                token_losses=token_losses,
                valid_mask=valid_mask,
                task_types=task_types,
                sample_losses=sample_losses,
                sample_weights=token_equivalent_weights,
                contribution_basis="supervised_token_count",
                loss=loss,
            ),
        )


class TaskWeightedLossStrategy(LossStrategy):
    """默认策略：先对每条样本的 token CE 求均值，再按任务权重聚合样本 loss。"""

    name = "task_weighted"

    def compute(
        self,
        *,
        token_losses: Any,
        valid_mask: Any,
        task_types: Sequence[str],
        config: MultitaskLossConfig,
        torch: Any,
    ) -> MultitaskLossOutput:
        token_counts = valid_mask.sum(dim=1)
        sample_losses = token_losses.sum(dim=1) / token_counts
        weights = torch.tensor(
            [config.task_weights.get(task, config.unknown_task_weight) for task in task_types],
            dtype=sample_losses.dtype,
            device=sample_losses.device,
        )
        loss = (sample_losses * weights).sum() / weights.sum()
        return MultitaskLossOutput(
            loss=loss,
            diagnostics=_diagnostics(
                token_losses=token_losses,
                valid_mask=valid_mask,
                task_types=task_types,
                sample_losses=sample_losses,
                sample_weights=weights,
                contribution_basis="configured_task_weight_per_sample",
                loss=loss,
            ),
        )


LOSS_STRATEGIES: Mapping[str, LossStrategy] = {
    "token_mean": TokenMeanLossStrategy(),
    "task_weighted": TaskWeightedLossStrategy(),
}


def compute_multitask_loss(
    logits: Any,
    labels: Any,
    task_types: Sequence[str] | None,
    config: MultitaskLossConfig,
    *,
    torch: Any,
) -> MultitaskLossOutput:
    """计算配置指定的多任务 causal-LM loss。

    参数：
        logits: Qwen3-VL 输出，形状 ``[batch, sequence, vocabulary]``。
        labels: Collator 的 assistant-only labels；非监督位置必须为 ``-100``。
        task_types: batch 内每条样本的任务类型；缺失时按 strict 配置报错或使用 unknown。
        config: 独立 loss 配置，决定策略、任务权重和 metadata 严格性。
        torch: 动态导入的 torch 模块，避免模块导入时强制重依赖。

    返回：
        :class:`MultitaskLossOutput`，其中 ``loss`` 保留梯度，diagnostics 已 detach。

    异常：
        ValueError: batch 维度不一致、task metadata 缺失、或 causal shift 后某样本无监督。
    """

    if logits.ndim != 3 or labels.ndim != 2 or tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError(
            "Expected logits [batch, sequence, vocab] matching labels [batch, sequence]"
        )
    batch_size = int(labels.shape[0])
    if task_types is None:
        if config.strict_task_metadata:
            raise ValueError(
                "Batch is missing task_types required by loss.strict_task_metadata=true"
            )
        normalized_tasks = ["unknown"] * batch_size
    else:
        normalized_tasks = [str(task).strip().lower() or "unknown" for task in task_types]
        if len(normalized_tasks) != batch_size:
            raise ValueError("task_types length must match the logits batch dimension")

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    valid_mask = shift_labels != -100
    token_counts = valid_mask.sum(dim=1)
    empty_indices = (token_counts == 0).nonzero(as_tuple=False).flatten().tolist()
    if empty_indices:
        raise ValueError(
            "No supervised assistant tokens remain after causal shift for batch indices: "
            + ", ".join(str(index) for index in empty_indices)
        )
    valid_logits = shift_logits[valid_mask].float()
    valid_labels = shift_labels[valid_mask]
    valid_losses = torch.nn.functional.cross_entropy(
        valid_logits,
        valid_labels,
        reduction="none",
    )
    token_losses = torch.zeros_like(shift_labels, dtype=valid_losses.dtype)
    token_losses[valid_mask] = valid_losses
    strategy = LOSS_STRATEGIES.get(config.mode)
    if strategy is None:  # Defensive protection for programmatically constructed configs.
        choices = ", ".join(sorted(LOSS_STRATEGIES))
        raise ValueError(f"Unknown loss mode: {config.mode}. Available modes: {choices}")
    return strategy.compute(
        token_losses=token_losses,
        valid_mask=valid_mask,
        task_types=normalized_tasks,
        config=config,
        torch=torch,
    )
