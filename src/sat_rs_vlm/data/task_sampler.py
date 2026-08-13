"""按任务或数据来源构造可复现的训练采样器。

本模块只控制“某类样本出现多少次”，不参与 loss 权重计算。Loss 权重由
``training.losses`` 和 ``loss.task_weights`` 独立负责。
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from typing import Any


def build_task_sample_weights(
    dataset: Sequence[Mapping[str, Any]],
    task_weights: Mapping[str, float] | None,
) -> list[float] | None:
    """为每条样本生成正采样权重；空配置表示均匀采样。"""

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
    """构造 ``WeightedRandomSampler``，不在模块导入时强制依赖 torch。"""

    weights = build_task_sample_weights(dataset, task_weights)
    if weights is None:
        return None
    try:
        import torch
        from torch.utils.data import WeightedRandomSampler
    except ImportError as exc:  # pragma: no cover - 仅真实训练环境会触发
        raise ImportError("torch is required for task-weighted sampling") from exc
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def build_alternating_source_sampler(
    dataset: Sequence[Mapping[str, Any]],
    source_batch_pattern: Sequence[str],
    *,
    batch_size: int,
    seed: int = 42,
) -> Any:
    """按固定来源顺序构造同来源 batch 的可复现 sampler。"""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pattern = [str(source) for source in source_batch_pattern]
    if not pattern:
        raise ValueError("source_batch_pattern must not be empty")

    source_indices: dict[str, list[int]] = {}
    for index, row in enumerate(dataset):
        metadata = row.get("metadata", {})
        source = str(metadata.get("training_source", "")) if isinstance(metadata, Mapping) else ""
        source_indices.setdefault(source, []).append(index)
    missing = sorted(set(pattern).difference(source_indices))
    if missing:
        raise ValueError(f"Batch-pattern sources are missing from dataset: {missing}")

    batches_per_cycle = Counter(pattern)
    cycle_capacity = min(
        (len(source_indices[source]) // batch_size) // required_batches
        for source, required_batches in batches_per_cycle.items()
    )
    if cycle_capacity < 1:
        raise ValueError("Not enough samples to build one complete source batch cycle")
    sample_count = cycle_capacity * len(pattern) * batch_size

    class AlternatingSourceSampler:
        def __init__(self) -> None:
            self.iteration = 0

        def __len__(self) -> int:
            return sample_count

        def __iter__(self) -> Iterator[int]:
            rng = random.Random(seed + self.iteration)
            self.iteration += 1
            shuffled = {source: list(indices) for source, indices in source_indices.items()}
            for indices in shuffled.values():
                rng.shuffle(indices)
            positions = {source: 0 for source in shuffled}
            for _ in range(cycle_capacity):
                for source in pattern:
                    start = positions[source]
                    end = start + batch_size
                    yield from shuffled[source][start:end]
                    positions[source] = end

    return AlternatingSourceSampler()
