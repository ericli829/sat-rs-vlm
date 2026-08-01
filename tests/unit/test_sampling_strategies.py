from __future__ import annotations

import pytest

from sat_rs_vlm.data.sampling import allocate_quotas, group_by_task, sample_by_task
from sat_rs_vlm.data.task_sampler import build_task_sample_weights


def _rows() -> list[dict[str, str]]:
    return [
        {"id": "d1", "task_type": "detection"},
        {"id": "d2", "task_type": "detection"},
        {"id": "c1", "task_type": "counting"},
        {"id": "v1", "task_type": "vqa"},
    ]


def test_balanced_and_explicit_quotas_are_reproducible() -> None:
    rows = _rows()
    grouped = group_by_task(rows)
    quotas = allocate_quotas(grouped, total=3)
    first, stats = sample_by_task(rows, quotas, seed=42)
    second, _ = sample_by_task(rows, quotas, seed=42)
    assert stats == {"counting": 1, "detection": 1, "vqa": 1}
    assert [row["id"] for row in first] == [row["id"] for row in second]


def test_task_weighted_sampler_values_and_validation() -> None:
    assert build_task_sample_weights(_rows(), {"detection": 2.0, "counting": 1.5}) == [
        2.0,
        2.0,
        1.5,
        1.0,
    ]
    with pytest.raises(ValueError, match="positive"):
        build_task_sample_weights(_rows(), {"detection": 0.0})
