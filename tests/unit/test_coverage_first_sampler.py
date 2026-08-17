from __future__ import annotations

from typing import Any

from sat_rs_vlm.data.task_sampler import build_alternating_source_sampler


def _rows(source: str, count: int) -> list[dict[str, Any]]:
    return [
        {"id": f"{source}-{index}", "metadata": {"training_source": source}}
        for index in range(count)
    ]


def test_historical_alternating_sampler_still_truncates_to_complete_pattern() -> None:
    dataset = _rows("VRSBench", 10) + _rows("LEVIR-CC", 3)
    sampler = build_alternating_source_sampler(
        dataset,
        ["VRSBench", "VRSBench", "VRSBench", "LEVIR-CC"],
        batch_size=1,
        seed=42,
    )
    assert len(list(sampler)) == 12


def test_coverage_first_alternating_sampler_exposes_every_index_once() -> None:
    dataset = _rows("VRSBench", 10) + _rows("LEVIR-CC", 3)
    sampler = build_alternating_source_sampler(
        dataset,
        ["VRSBench", "VRSBench", "VRSBench", "LEVIR-CC"],
        batch_size=2,
        seed=42,
        exhaustion_policy="coverage_first",
    )
    indices = list(sampler)
    assert len(indices) == len(dataset)
    assert set(indices) == set(range(len(dataset)))
    assert len(indices) == len(set(indices))
