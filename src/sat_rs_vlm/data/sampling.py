"""E1 系列实验使用的分层、配额和可选有放回采样。"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def group_by_task(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按 task_type 分组并复制为普通字典。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task_type", "unknown"))].append(dict(row))
    return dict(grouped)


def allocate_quotas(
    grouped: Mapping[str, list[dict[str, Any]]],
    *,
    total: int | None = None,
    per_task: int | None = None,
    explicit: Mapping[str, int] | None = None,
    with_replacement: bool = False,
) -> dict[str, int]:
    """计算每任务配额；无放回时不会分配超过可用样本数。"""

    if sum(value is not None for value in (total, per_task, explicit)) != 1:
        raise ValueError("Exactly one of total, per_task, or explicit quotas is required")
    if explicit is not None:
        missing = sorted(set(explicit).difference(grouped))
        if missing:
            raise ValueError(f"Quota tasks are missing from input: {missing}")
        quotas = {task: int(count) for task, count in explicit.items() if int(count) > 0}
    elif per_task is not None:
        quotas = {task: int(per_task) for task in sorted(grouped)}
    else:
        assert total is not None
        if total <= 0 or not grouped:
            return {}
        tasks = sorted(grouped)
        base, remainder = divmod(total, len(tasks))
        quotas = {task: base + int(index < remainder) for index, task in enumerate(tasks)}
    if not with_replacement:
        quotas = {task: min(count, len(grouped[task])) for task, count in quotas.items()}
        if total is not None:
            remaining = total - sum(quotas.values())
            ordered_tasks = sorted(
                grouped,
                key=lambda item: len(grouped[item]) - quotas[item],
                reverse=True,
            )
            for task in ordered_tasks:
                capacity = len(grouped[task]) - quotas[task]
                take = min(remaining, capacity)
                quotas[task] += take
                remaining -= take
                if remaining <= 0:
                    break
    return quotas


def sample_by_task(
    rows: Iterable[Mapping[str, Any]],
    quotas: Mapping[str, int],
    *,
    seed: int = 42,
    with_replacement: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """按配额选择样本并返回实际任务统计。"""

    grouped = group_by_task(rows)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    stats: dict[str, int] = {}
    for task, count in quotas.items():
        pool = list(grouped.get(task, []))
        if count > len(pool) and not with_replacement:
            chosen = pool
        elif count > len(pool):
            if not pool:
                raise ValueError(f"Cannot sample missing task with replacement: {task}")
            chosen = [rng.choice(pool) for _ in range(count)]
        else:
            rng.shuffle(pool)
            chosen = pool[:count]
        selected.extend(chosen)
        stats[task] = len(chosen)
    rng.shuffle(selected)
    return selected, stats
