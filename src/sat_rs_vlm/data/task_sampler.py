"""按 task_type 构造可复现的训练采样权重与 Trainer。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_task_sample_weights(
    dataset: Sequence[Mapping[str, Any]],
    task_weights: Mapping[str, float] | None,
) -> list[float] | None:
    """为每条样本生成正权重；空配置表示均匀采样。"""

    if not task_weights:
        return None
    normalized = {str(key): float(value) for key, value in task_weights.items()}
    invalid = {key: value for key, value in normalized.items() if value <= 0}
    if invalid:
        raise ValueError(f"Task sampling weights must be positive: {invalid}")
    return [normalized.get(str(row.get("task_type", "unknown")), 1.0) for row in dataset]


def build_weighted_sampler(
    dataset: Sequence[Mapping[str, Any]],
    task_weights: Mapping[str, float] | None,
    *,
    seed: int = 42,
) -> Any | None:
    """构造 torch WeightedRandomSampler，不在模块导入时强制依赖 torch。"""

    weights = build_task_sample_weights(dataset, task_weights)
    if weights is None:
        return None
    try:
        import torch
        from torch.utils.data import WeightedRandomSampler
    except ImportError as exc:  # pragma: no cover - 真实训练环境分支
        raise ImportError("torch is required for task-weighted sampling") from exc
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def create_trainer(
    transformers: Any,
    *,
    train_sampler: Any | None,
    trainer_kwargs: dict[str, Any],
) -> Any:
    """创建标准 Trainer 或显式覆写采样扩展点的加权 Trainer。

    Transformers 暂无公开 sampler 构造参数，因此集中在此处覆写 `_get_train_sampler`，
    避免在训练脚本运行时 monkey patch 私有方法。
    """

    if train_sampler is None:
        return transformers.Trainer(**trainer_kwargs)

    class TaskWeightedTrainer(transformers.Trainer):  # type: ignore[misc]
        def _get_train_sampler(self, train_dataset: Any | None = None) -> Any:
            del train_dataset
            return train_sampler

    return TaskWeightedTrainer(**trainer_kwargs)
