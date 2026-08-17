"""按 task/source 构造可复现的训练 sampler；Trainer 由 training.trainer 统一创建。"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
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


def build_alternating_source_sampler(
    dataset: Sequence[Mapping[str, Any]],
    source_batch_pattern: Sequence[str],
    *,
    batch_size: int,
    seed: int = 42,
    exhaustion_policy: str = "truncate",
) -> Any:
    """按 source batch pattern 排序索引。

    ``truncate`` 精确保留历史行为；``coverage_first`` 在完整 pattern 无法继续后，
    将所有 source 的剩余样本各使用一次。尾部允许混合/不满 batch，以换取零遗漏。
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pattern = [str(source) for source in source_batch_pattern]
    if not pattern:
        raise ValueError("source_batch_pattern must not be empty")
    if exhaustion_policy not in {"truncate", "coverage_first"}:
        raise ValueError("exhaustion_policy must be 'truncate' or 'coverage_first'")

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
    if cycle_capacity < 1 and exhaustion_policy == "truncate":
        raise ValueError("Not enough samples to build one complete source batch cycle")
    sample_count = (
        sum(len(indices) for indices in source_indices.values())
        if exhaustion_policy == "coverage_first"
        else cycle_capacity * len(pattern) * batch_size
    )

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
            if exhaustion_policy == "coverage_first":
                tail_sources: list[str] = []
                seen: set[str] = set()
                for source in pattern:
                    if source not in seen:
                        tail_sources.append(source)
                        seen.add(source)
                tail_sources.extend(source for source in sorted(shuffled) if source not in seen)
                for source in tail_sources:
                    yield from shuffled[source][positions[source] :]

    return AlternatingSourceSampler()
