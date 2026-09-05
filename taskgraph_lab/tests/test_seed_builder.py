from __future__ import annotations

from pathlib import Path

from taskgraph_lab.datasets.base import NormalizedSample
from taskgraph_lab.tools.build_seed_set import build_seed_set

FIXTURE = Path(__file__).parent / "fixtures/normalized_samples.jsonl"


def samples() -> list[NormalizedSample]:
    return [
        NormalizedSample.model_validate_json(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_seed_builder_is_deterministic_and_structure_aware() -> None:
    config = {
        "max_total": 8,
        "seed": 42,
        "per_dataset": {"XLRS_Bench": 8, "MME_RealWorld_RS": 8},
        "per_category": {
            "entire_image_count": 1,
            "absolute_region_count": 1,
            "two_image_abs_diff": 1,
            "route_planning": 1,
            "other": 4,
        },
    }
    left, left_manifest = build_seed_set(samples(), config)
    right, right_manifest = build_seed_set(samples(), config)
    assert [item.sample_id for item in left] == [item.sample_id for item in right]
    assert left_manifest == right_manifest
    assert left_manifest["category_distribution"]["two_image_abs_diff"] == 1
    assert left_manifest["category_distribution"]["route_planning"] == 1


def test_seed_builder_excludes_previously_used_ids() -> None:
    population = samples()
    excluded = {population[0].sample_id, population[1].sample_id}
    config = {
        "max_total": len(population),
        "seed": 42,
        "per_dataset": {"XLRS_Bench": len(population), "MME_RealWorld_RS": len(population)},
        "per_category": {"other": len(population)},
    }
    selected, manifest = build_seed_set(population, config, excluded_ids=excluded)
    selected_ids = {item.sample_id for item in selected}
    assert selected_ids.isdisjoint(excluded)
    assert manifest["excluded_id_count"] == 2
    assert manifest["excluded_candidate_count"] == 2


def test_seed_builder_can_include_only_requested_hard_categories() -> None:
    population = samples()
    config = {
        "max_total": len(population),
        "seed": 42,
        "include_categories": ["route_planning", "two_image_abs_diff"],
        "per_dataset": {
            "XLRS_Bench": len(population),
            "MME_RealWorld_RS": len(population),
        },
        "per_category": {
            "route_planning": len(population),
            "two_image_abs_diff": len(population),
        },
    }

    selected, manifest = build_seed_set(population, config)

    assert selected
    assert set(manifest["category_distribution"]) == {
        "route_planning",
        "two_image_abs_diff",
    }
    assert manifest["include_categories"] == [
        "route_planning",
        "two_image_abs_diff",
    ]
    assert manifest["candidate_count"] == len(selected)
    assert manifest["category_filtered_count"] == len(population) - len(selected)
