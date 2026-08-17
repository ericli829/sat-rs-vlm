from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sat_rs_vlm.data.cyclic_training import (
    assert_no_evaluation_leakage,
    combine_source_rounds,
    load_protected_e3_ids,
    partition_group_variants,
    partition_task_population,
    validate_cycle_coverage,
)


def row(sample_id: str, task: str, source: str = "VRSBench") -> dict[str, Any]:
    return {
        "id": sample_id,
        "task_type": task,
        "metadata": {"training_source": source},
        "messages": [],
    }


def test_task_buckets_are_deterministic_complete_and_keep_partial_bucket() -> None:
    population = [row(f"d-{index}", "detection") for index in range(5)] + [
        row(f"v-{index}", "vqa") for index in range(3)
    ]
    first = partition_task_population(
        population, {"detection": 2, "vqa": 2}, seed=42, cycle_index=0
    )
    second = partition_task_population(
        population, {"detection": 2, "vqa": 2}, seed=42, cycle_index=0
    )

    assert [[item["id"] for item in bucket] for bucket in first] == [
        [item["id"] for item in bucket] for bucket in second
    ]
    assert [len(bucket) for bucket in first] == [4, 3, 1]
    assert validate_cycle_coverage(population, first)["valid"] is True


def test_different_cycle_changes_order_without_changing_coverage() -> None:
    population = [row(f"d-{index}", "detection") for index in range(20)]
    first = partition_task_population(
        population, {"detection": 4}, seed=42, cycle_index=0
    )
    second = partition_task_population(
        population, {"detection": 4}, seed=42, cycle_index=1
    )

    assert [[item["id"] for item in bucket] for bucket in first] != [
        [item["id"] for item in bucket] for bucket in second
    ]
    assert validate_cycle_coverage(population, first)["valid"] is True
    assert validate_cycle_coverage(population, second)["valid"] is True


def test_levir_variants_rotate_without_repeat_before_full_coverage() -> None:
    population = [row(f"caption-{index}", "change_detection", "LEVIR-CC") for index in range(5)]
    rounds, report = partition_group_variants(
        population,
        variants_per_round=2,
        seed=42,
        cycle_index=0,
        image_key=lambda _: ("A.png", "B.png"),
    )

    assert [len(bucket) for bucket in rounds] == [2, 2, 1]
    assert validate_cycle_coverage(population, rounds)["valid"] is True
    assert report["caption_variant_count"] == 5
    assert report["per_round_variant_count"] == [2, 2, 1]


def test_combined_sources_keep_every_sample() -> None:
    source_rounds = {
        "VRSBench": [[row("v0", "vqa")], [row("v1", "vqa")]],
        "LEVIR-CC": [[row("l0", "change_detection", "LEVIR-CC")]],
    }
    rounds = combine_source_rounds(source_rounds, seed=42, cycle_index=0)
    population = [
        item for buckets in source_rounds.values() for bucket in buckets for item in bucket
    ]
    assert validate_cycle_coverage(population, rounds)["valid"] is True


def test_protected_e3_ids_are_loaded_and_leakage_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"tiers":{"E3":{"sample_ids":["eval-1","eval-2"]}}}', "utf-8")
    protected = load_protected_e3_ids(manifest)

    assert protected == {"eval-1", "eval-2"}
    with pytest.raises(ValueError, match="protected Unified Evaluation E3"):
        assert_no_evaluation_leakage([row("eval-1", "vqa")], protected)
